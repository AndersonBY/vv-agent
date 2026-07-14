from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

import pytest

from vv_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.state import Checkpoint, InMemoryStateStore, StateStoreSpec
from vv_agent.types import AgentResult, AgentStatus, AgentTask, CompletionReason, Message

pytest.importorskip("celery")

DispatchHandler = Callable[[dict[str, Any]], dict[str, Any]]


class _SchedulerStore(InMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_error: Exception | None = None
        self.finalize_mode: Literal["normal", "false", "error"] = "normal"
        self.ack_mode: Literal["normal", "false", "false_after_delete", "error"] = "normal"
        self.finalized_checkpoints: list[Checkpoint] = []

    def state_store_spec(self) -> StateStoreSpec:
        return StateStoreSpec(kind="sqlite", location="/tmp/vv-agent-scheduler-contract.sqlite3")

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        if self.load_error is not None:
            raise self.load_error
        return super().load_checkpoint(task_id)

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        if self.finalize_mode == "false":
            return False
        if self.finalize_mode == "error":
            raise RuntimeError("injected finalize failure")
        self.finalized_checkpoints.append(deepcopy(checkpoint))
        return super().finalize_checkpoint(checkpoint, expected_revision=expected_revision)

    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool:
        if self.ack_mode == "false":
            return False
        if self.ack_mode == "error":
            raise RuntimeError("injected acknowledge failure")
        acknowledged = super().acknowledge_terminal(task_id, expected_revision=expected_revision)
        if self.ack_mode == "false_after_delete":
            return False
        return acknowledged


class _ImmediateResult:
    def __init__(self, handler: Callable[[], dict[str, Any]]) -> None:
        self._handler = handler
        self.revoked = False

    def get(self, *, timeout: float) -> dict[str, Any]:
        assert timeout > 0
        return self._handler()

    def revoke(self, *, terminate: bool) -> None:
        assert terminate is False
        self.revoked = True


class _RecordingCeleryApp:
    def __init__(self, handler: DispatchHandler) -> None:
        self._handler = handler
        self.envelopes: list[dict[str, Any]] = []
        self.results: list[_ImmediateResult] = []

    def send_task(self, _name: str, *, kwargs: dict[str, Any], serializer: str) -> _ImmediateResult:
        assert serializer == "json"
        envelope = kwargs["envelope_dict"]
        self.envelopes.append(envelope)
        result = _ImmediateResult(lambda: self._handler(envelope))
        self.results.append(result)
        return result


def _task(task_id: str) -> AgentTask:
    return AgentTask(task_id=task_id, model="model", system_prompt="system", user_prompt="prompt")


def _checkpoint(task_id: str, *, cycle_index: int = 0) -> Checkpoint:
    return Checkpoint(
        task_id=task_id,
        cycle_index=cycle_index,
        status=AgentStatus.RUNNING,
        messages=[Message(role="user", content="primary context")],
        cycles=[],
        shared_state={"tenant": "contract"},
    )


def _backend(store: _SchedulerStore, handler: DispatchHandler) -> tuple[CeleryBackend, _RecordingCeleryApp]:
    app = _RecordingCeleryApp(handler)
    backend = CeleryBackend(
        celery_app=app,
        state_store=store,
        runtime_recipe=RuntimeRecipe(
            settings_file="/tmp/unused-settings.py",
            backend="test",
            model="model",
            workspace="/tmp/vv-agent-scheduler-contract",
        ),
        dispatch_timeout_seconds=1,
    )
    return backend, app


def _execute(
    backend: CeleryBackend,
    task: AgentTask,
    *,
    max_cycles: int = 2,
    ctx: ExecutionContext | None = None,
) -> AgentResult:
    def unexpected_cycle(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("distributed scheduler must not execute a local cycle")

    return backend.execute(
        task=task,
        initial_messages=[Message(role="user", content="primary context")],
        shared_state={"tenant": "contract"},
        cycle_executor=unexpected_cycle,
        ctx=ctx,
        max_cycles=max_cycles,
    )


def _commit_terminal(
    store: _SchedulerStore,
    task_id: str,
    cycle_index: int,
    *,
    final_answer: str = "durable terminal",
) -> tuple[AgentResult, int]:
    claimed = store.claim_checkpoint(
        task_id,
        cycle_index,
        claim_token=f"worker-{cycle_index}",
        lease_expires_at_ms=200,
        now_ms=100,
    )
    assert claimed is not None
    result = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        final_answer=final_answer,
        shared_state=claimed.shared_state,
    )
    claimed.cycle_index = cycle_index
    claimed.status = result.status
    claimed.terminal_result = result
    expected_revision = claimed.revision
    assert store.commit_checkpoint(
        claimed,
        claim_token=f"worker-{cycle_index}",
        expected_revision=expected_revision,
    )
    return result, expected_revision + 1


def _commit_unfinished(store: _SchedulerStore, task_id: str, cycle_index: int) -> None:
    claimed = store.claim_checkpoint(
        task_id,
        cycle_index,
        claim_token=f"worker-{cycle_index}",
        lease_expires_at_ms=200,
        now_ms=100,
    )
    assert claimed is not None
    claimed.cycle_index = cycle_index
    claimed.status = AgentStatus.RUNNING
    expected_revision = claimed.revision
    assert store.commit_checkpoint(
        claimed,
        claim_token=f"worker-{cycle_index}",
        expected_revision=expected_revision,
    )


def test_create_conflict_replays_and_acknowledges_terminal_without_dispatch() -> None:
    store = _SchedulerStore()
    task = _task("create-terminal")
    checkpoint = _checkpoint(task.task_id, cycle_index=1)
    terminal = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=checkpoint.messages,
        cycles=checkpoint.cycles,
        final_answer="replayed",
        shared_state=checkpoint.shared_state,
    )
    checkpoint.status = terminal.status
    checkpoint.revision = 4
    checkpoint.terminal_result = terminal
    assert store.create_checkpoint(checkpoint)
    backend, app = _backend(store, lambda _envelope: pytest.fail("terminal replay must not dispatch"))

    result = _execute(backend, task)

    assert result.final_answer == "replayed"
    assert app.envelopes == []
    assert store.load_checkpoint(task.task_id) is None


def test_create_conflict_rejects_inconsistent_durable_terminal() -> None:
    store = _SchedulerStore()
    task = _task("create-terminal-inconsistent")
    checkpoint = _checkpoint(task.task_id, cycle_index=1)
    terminal = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[Message(role="assistant", content="terminal history")],
        cycles=[],
        final_answer="terminal",
        shared_state={"tenant": "contract"},
    )
    checkpoint.status = terminal.status
    checkpoint.revision = 4
    checkpoint.terminal_result = terminal
    assert store.create_checkpoint(checkpoint)
    backend, app = _backend(store, lambda _envelope: pytest.fail("terminal replay must not dispatch"))

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "checkpoint fields do not match its terminal result" in (result.error or "")
    assert app.envelopes == []
    assert store.load_checkpoint(task.task_id) is not None


def test_create_conflict_resumes_unclaimed_running_checkpoint_from_next_cycle() -> None:
    store = _SchedulerStore()
    task = _task("create-resume")
    assert store.create_checkpoint(_checkpoint(task.task_id, cycle_index=1))

    def finish(envelope: dict[str, Any]) -> dict[str, Any]:
        assert envelope["cycle_index"] == 2
        terminal, revision = _commit_terminal(store, task.task_id, 2, final_answer="resumed")
        return {"finished": True, "result": terminal.to_dict(), "checkpoint_revision": revision}

    backend, app = _backend(store, finish)

    result = _execute(backend, task, max_cycles=3)

    assert result.final_answer == "resumed"
    assert [envelope["cycle_index"] for envelope in app.envelopes] == [2]
    assert store.load_checkpoint(task.task_id) is None


def test_create_conflict_with_claim_returns_in_progress_coordination_failure() -> None:
    store = _SchedulerStore()
    task = _task("create-claimed")
    assert store.create_checkpoint(_checkpoint(task.task_id))
    claimed = store.claim_checkpoint(
        task.task_id,
        1,
        claim_token="active-worker",
        lease_expires_at_ms=(1 << 64) - 1,
        now_ms=100,
    )
    assert claimed is not None
    backend, app = _backend(store, lambda _envelope: pytest.fail("claimed conflict must not dispatch"))

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "coordination failure" in (result.error or "").lower()
    assert "in progress" in (result.error or "").lower()
    assert app.envelopes == []
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None and persisted.claim_token == "active-worker"


def test_create_conflict_with_expired_claim_reclaims_the_same_cycle() -> None:
    store = _SchedulerStore()
    task = _task("create-expired-claim")
    assert store.create_checkpoint(_checkpoint(task.task_id))
    assert store.claim_checkpoint(
        task.task_id,
        1,
        claim_token="expired-worker",
        lease_expires_at_ms=1,
        now_ms=0,
    ) is not None

    def finish(envelope: dict[str, Any]) -> dict[str, Any]:
        assert envelope["cycle_index"] == 1
        terminal, revision = _commit_terminal(store, task.task_id, 1, final_answer="reclaimed")
        return {"finished": True, "result": terminal.to_dict(), "checkpoint_revision": revision}

    backend, app = _backend(store, finish)

    result = _execute(backend, task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "reclaimed"
    assert [envelope["cycle_index"] for envelope in app.envelopes] == [1]
    assert store.load_checkpoint(task.task_id) is None


def test_unfinished_load_error_preserves_primary_snapshot_and_context() -> None:
    store = _SchedulerStore()
    task = _task("load-error")

    def fail_load(_envelope: dict[str, Any]) -> dict[str, Any]:
        store.load_error = RuntimeError("injected load failure")
        return {"finished": False}

    backend, _app = _backend(store, fail_load)

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert result.messages[0].content == "primary context"
    assert result.shared_state == {"tenant": "contract"}
    assert "returned unfinished" in (result.error or "")
    assert "injected load failure" in (result.error or "")
    store.load_error = None
    assert store.load_checkpoint(task.task_id) is not None


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [("false", "CAS returned false"), ("error", "finalization raised an error")],
)
def test_compatibility_finished_payload_does_not_hide_finalize_failure(
    mode: Literal["false", "error"],
    expected_error: str,
) -> None:
    store = _SchedulerStore()
    store.finalize_mode = mode
    task = _task(f"finalize-{mode}")
    terminal = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[Message(role="assistant", content="dispatcher only")],
        cycles=[],
        final_answer="must not escape without persistence",
        shared_state={"source": "dispatcher"},
    )
    backend, _app = _backend(store, lambda _envelope: {"finished": True, "result": terminal.to_dict()})

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert expected_error in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None and persisted.terminal_result is None


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [("false", "acknowledgement returned false"), ("error", "acknowledgement raised an error")],
)
def test_worker_terminal_does_not_hide_ack_failure(
    mode: Literal["false", "error"],
    expected_error: str,
) -> None:
    store = _SchedulerStore()
    store.ack_mode = mode
    task = _task(f"ack-{mode}")

    def finish(_envelope: dict[str, Any]) -> dict[str, Any]:
        terminal, revision = _commit_terminal(store, task.task_id, 1)
        return {"finished": True, "result": terminal.to_dict(), "checkpoint_revision": revision}

    backend, _app = _backend(store, finish)

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert expected_error in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None and persisted.terminal_result is not None
    assert persisted.cycle_index == 1


def test_ack_false_after_concurrent_delete_is_treated_as_acknowledged() -> None:
    store = _SchedulerStore()
    store.ack_mode = "false_after_delete"
    task = _task("ack-concurrent-delete")

    def finish(_envelope: dict[str, Any]) -> dict[str, Any]:
        terminal, revision = _commit_terminal(store, task.task_id, 1)
        return {"finished": True, "result": terminal.to_dict(), "checkpoint_revision": revision}

    backend, _app = _backend(store, finish)

    result = _execute(backend, task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "durable terminal"
    assert store.load_checkpoint(task.task_id) is None


def test_finished_revision_requires_dispatcher_result_to_match_durable_terminal() -> None:
    store = _SchedulerStore()
    task = _task("terminal-mismatch")

    def mismatched_finish(_envelope: dict[str, Any]) -> dict[str, Any]:
        terminal, revision = _commit_terminal(store, task.task_id, 1)
        dispatcher_result = AgentResult.from_dict(terminal.to_dict())
        dispatcher_result.final_answer = "not durable"
        return {
            "finished": True,
            "result": dispatcher_result.to_dict(),
            "checkpoint_revision": revision,
        }

    backend, _app = _backend(store, mismatched_finish)

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "does not match the durable terminal" in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None and persisted.terminal_result is not None


def test_unfinished_without_exact_durable_progress_is_coordination_failure() -> None:
    store = _SchedulerStore()
    task = _task("unfinished-no-progress")
    backend, _app = _backend(store, lambda _envelope: {"finished": False})

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "without durable progress" in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None
    assert persisted.cycle_index == 0
    assert persisted.terminal_result is None


def test_unfinished_payload_cannot_include_a_result() -> None:
    store = _SchedulerStore()
    task = _task("unfinished-with-result")
    payload = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[],
        cycles=[],
        final_answer="invalid",
    )
    backend, _app = _backend(
        store,
        lambda _envelope: {"finished": False, "result": payload.to_dict()},
    )

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "unfinished payload with a result" in (result.error or "")
    assert store.load_checkpoint(task.task_id) is not None


def test_explicit_null_revision_uses_compatibility_finalization() -> None:
    store = _SchedulerStore()
    task = _task("null-revision")
    terminal = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[Message(role="assistant", content="compatibility")],
        cycles=[],
        final_answer="compatibility",
        shared_state={"source": "dispatcher"},
    )
    backend, _app = _backend(
        store,
        lambda _envelope: {
            "finished": True,
            "result": terminal.to_dict(),
            "checkpoint_revision": None,
        },
    )

    result = _execute(backend, task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "compatibility"
    assert store.load_checkpoint(task.task_id) is None


def test_non_cancellation_context_error_is_not_reclassified_as_cancellation() -> None:
    class BrokenContext(ExecutionContext):
        def check_cancelled(self) -> None:
            raise RuntimeError("context checker failed")

    store = _SchedulerStore()
    task = _task("context-error")
    backend, _app = _backend(store, lambda _envelope: pytest.fail("must not dispatch"))

    with pytest.raises(RuntimeError, match="context checker failed"):
        _execute(backend, task, ctx=BrokenContext())

    checkpoint = store.load_checkpoint(task.task_id)
    assert checkpoint is not None and checkpoint.terminal_result is None


def test_dispatch_error_replays_worker_terminal_as_authoritative() -> None:
    store = _SchedulerStore()
    task = _task("dispatch-error-terminal")

    def commit_then_fail(_envelope: dict[str, Any]) -> dict[str, Any]:
        _commit_terminal(store, task.task_id, 1, final_answer="worker won")
        raise RuntimeError("result backend disconnected")

    backend, _app = _backend(store, commit_then_fail)

    result = _execute(backend, task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "worker won"
    assert store.load_checkpoint(task.task_id) is None


def test_dispatch_error_with_worker_claim_is_uncertain_and_not_overwritten() -> None:
    store = _SchedulerStore()
    task = _task("dispatch-error-claimed")

    def claim_then_fail(_envelope: dict[str, Any]) -> dict[str, Any]:
        claimed = store.claim_checkpoint(
            task.task_id,
            1,
            claim_token="worker-still-running",
            lease_expires_at_ms=200,
            now_ms=100,
        )
        assert claimed is not None
        raise RuntimeError("transport lost")

    backend, _app = _backend(store, claim_then_fail)

    result = _execute(backend, task)

    assert result.status == AgentStatus.FAILED
    assert "durable outcome is uncertain" in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None
    assert persisted.claim_token == "worker-still-running"
    assert persisted.terminal_result is None


def test_max_cycles_finalize_false_is_not_reported_as_normal_max_cycles() -> None:
    store = _SchedulerStore()
    store.finalize_mode = "false"
    task = _task("max-cycles-finalize-false")

    def advance(_envelope: dict[str, Any]) -> dict[str, Any]:
        _commit_unfinished(store, task.task_id, 1)
        return {"finished": False}

    backend, _app = _backend(store, advance)

    result = _execute(backend, task, max_cycles=1)

    assert result.status == AgentStatus.FAILED
    assert "CAS returned false" in (result.error or "")
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None and persisted.terminal_result is None


def test_cancellation_observed_while_waiting_does_not_overwrite_worker_claim() -> None:
    from celery.exceptions import TimeoutError as CeleryTimeoutError

    store = _SchedulerStore()
    task = _task("cancel-waiting-claimed")
    token = CancellationToken()

    class ClaimingResult(_ImmediateResult):
        def get(self, *, timeout: float) -> dict[str, Any]:
            claimed = store.claim_checkpoint(
                task.task_id,
                1,
                claim_token="worker-during-cancel",
                lease_expires_at_ms=200,
                now_ms=100,
            )
            assert claimed is not None
            token.cancel("cancel while waiting")
            raise CeleryTimeoutError()

    class ClaimingCeleryApp(_RecordingCeleryApp):
        def send_task(self, _name: str, *, kwargs: dict[str, Any], serializer: str) -> _ImmediateResult:
            assert serializer == "json"
            self.envelopes.append(kwargs["envelope_dict"])
            result = ClaimingResult(lambda: pytest.fail("unused handler"))
            self.results.append(result)
            return result

    app = ClaimingCeleryApp(lambda _envelope: pytest.fail("custom result handles dispatch"))
    backend = CeleryBackend(
        celery_app=app,
        state_store=store,
        runtime_recipe=RuntimeRecipe(
            settings_file="/tmp/unused-settings.py",
            backend="test",
            model="model",
            workspace="/tmp/vv-agent-scheduler-contract",
        ),
        dispatch_timeout_seconds=1,
    )

    result = _execute(backend, task, ctx=ExecutionContext(cancellation_token=token))

    assert result.status == AgentStatus.FAILED
    assert "durable outcome is uncertain" in (result.error or "")
    assert app.results[0].revoked is True
    persisted = store.load_checkpoint(task.task_id)
    assert persisted is not None
    assert persisted.claim_token == "worker-during-cancel"
    assert persisted.terminal_result is None


def test_distributed_scheduler_preserves_cancellation_reason() -> None:
    store = _SchedulerStore()
    task = _task("cancel-reason")
    token = CancellationToken()
    token.cancel("host shutdown")
    backend, _app = _backend(store, lambda _envelope: pytest.fail("must not dispatch"))

    result = _execute(backend, task, ctx=ExecutionContext(cancellation_token=token))

    assert result.status == AgentStatus.FAILED
    assert result.error == "host shutdown"
    assert result.completion_reason == CompletionReason.CANCELLED
    assert len(store.finalized_checkpoints) == 1
    terminal = store.finalized_checkpoints[0].terminal_result
    assert terminal is not None
    assert terminal.completion_reason == CompletionReason.CANCELLED
    assert store.load_checkpoint(task.task_id) is None
