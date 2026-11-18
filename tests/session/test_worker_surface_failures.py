"""Worker-surface failure propagation regressions over temporary SQLite stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event, current_thread
from time import monotonic, sleep
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid
from vv_agent.kernel._sql import LedgerIntegrityError
from vv_agent.llm.base import LlmRequest
from vv_agent.model import ScriptedModelProvider, resolve_model_selection
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.providers import Accepted
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import RuntimeNotReady, static_model_selection_requirements
from vv_agent.session.supervisor import _tick
from vv_agent.session.surfaces import SessionDriver, _SessionFailure
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentStatus, CompletionReason, LLMResponse, ToolCall, ToolExecutionResult

from .conftest import open_store

pytestmark = pytest.mark.sqlite_file

MASTER_KEY = b"w" * 32


@contextmanager
def _temporary_store(database: Path, *, write_level: int) -> Iterator[Any]:
    with open_store(database) as store:
        store.install_schema()
        if write_level > 1:
            store._rows("UPDATE sk_settings SET write_level=%s WHERE singleton=1", (write_level,))
        yield store


@contextmanager
def _temporary_driver(tmp_path: Path, name: str, *, worker_level: int = 1) -> Iterator[SessionDriver]:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{name}-keys.sqlite3")
    try:
        with _temporary_store(tmp_path / f"{name}.sqlite", write_level=worker_level) as store:
            driver = SessionDriver(
                store=store,
                content_key_provider=provider,
                worker_level=worker_level,
                handler_versions=("1",),
            )
            try:
                yield driver
            finally:
                driver.close()
    finally:
        provider.close()


def _config(responses: list[LLMResponse]) -> RunConfig:
    return RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", responses))


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def _force_due(driver: SessionDriver, sid: str) -> None:
    with driver.store.atomic():
        cast(Any, driver.store)._rows(
            "UPDATE sk_session SET next_drive_ms=0, drive_retry_at_ms=0 WHERE tenant_id=%s AND session_id=%s",
            (driver.tenant_id, sid),
        )
    driver._worker.wake(sid)


@function_tool
def external_wait() -> str:
    raise AssertionError("external_wait must use its provider")


@function_tool
def healthy_external_wait() -> str:
    raise AssertionError("healthy_external_wait must use its provider")


@function_tool
def replacement_external_wait() -> str:
    raise AssertionError("replacement_external_wait must use its provider")


def _runtime_factory_for(owner: SessionDriver, providers: dict[str, Any]):
    def runtime_factory(store, agent, config, task, session_id):
        runtime = owner.runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
        )
        runtime.providers.update(providers)
        return runtime

    return runtime_factory


class _InvalidQueryProvider:
    def __init__(self) -> None:
        self.handle: dict[str, Any] = {}
        self.result: dict[str, Any] = {}
        self.queries = 0

    def submit(self, plan, *, context):
        del context
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "invalid-query",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "invalid-query/evidence",
            "query_ref": None,
            "cancel_ref": None,
        }
        self.result = ToolExecutionResult(
            plan.payload["request"]["id"],
            "late provider result",
        ).to_dict()
        return Accepted(self.handle)

    def query(self, handle):
        assert handle == self.handle
        self.queries += 1
        return Accepted(handle | {"job_id": "wrong-handle"})

    def cancel(self, handle):
        raise AssertionError(f"invalid-query provider must never cancel {handle}")

    def authenticate(self, item, plan):
        return (
            item.kind == "provider_result"
            and item.payload["operation_id"] == plan.operation_id
            and item.payload["attempt"] == plan.attempt
            and item.payload["request_digest"] == plan.payload["request_digest"]
            and item.payload["provider_binding"] == plan.payload["provider_binding"]
            and item.payload["result"] == self.result
            and item.payload["evidence"] == [self.handle["evidence"]]
        )


class _PendingProvider(_InvalidQueryProvider):
    def query(self, handle):
        assert handle == self.handle
        self.queries += 1
        return Accepted(handle)


class _PrefilledInvalidQueryProvider(_InvalidQueryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.error = ProviderResultInvalid()
        self.error._query_operation_key = ("stale/operation", 99)

    def query(self, handle):
        assert handle == self.handle
        self.queries += 1
        return Accepted(handle)


def _induce_provider_invalid(
    driver: SessionDriver,
    sid: str,
    provider: _InvalidQueryProvider,
    *,
    tools=(external_wait,),
):
    def runtime_factory(store, agent, config, task, session_id):
        runtime = driver.runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
        )
        runtime.providers["external_wait"] = provider
        return runtime

    driver.runtime_factory = runtime_factory
    driver.create(sid, "")
    model = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
            LLMResponse("recovered"),
        ],
    )
    handle = driver.start(
        sid,
        Agent("provider", "Wait externally.", model="m", tools=list(tools)),
        RunConfig(model_provider=model),
        "go",
    )
    _wait_for(lambda: bool(provider.handle))
    driver._worker.wait_idle(timeout=3)
    _force_due(driver, sid)
    _wait_for(
        lambda: (
            (failure := driver._errors.get(sid)) is not None
            and isinstance(failure.error, ProviderResultInvalid)
            and failure.terminal
        )
    )
    driver._worker.wait_idle(timeout=3)
    return handle, model, driver._errors[sid]


def _push_in_transaction_and_wake(driver: SessionDriver, sid: str, item: InboxItem):
    with driver.store.atomic() as tx:
        receipt = driver._push_in_transaction(
            tx,
            sid,
            item,
            actor=driver._session_principal(sid),
            actor_subject_id="regression",
        )
    driver._worker.wake(sid)
    return receipt


def test_runtime_not_ready_retries_real_factory_without_duplicate_model_call(tmp_path: Path) -> None:
    with _temporary_driver(tmp_path, "not-ready") as driver:
        factory_calls = 0
        model_calls: list[LlmRequest] = []

        def runtime_factory(store, agent, config, task, session_id):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 1:
                raise RuntimeNotReady(10)
            return driver.runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=store,
            )

        def response(request: LlmRequest) -> LLMResponse:
            model_calls.append(request)
            return LLMResponse("ready")

        provider = ScriptedModelProvider.from_callback("scripted", "m", response)
        driver.runtime_factory = runtime_factory
        driver.create("not-ready", str(tmp_path))
        handle = driver.start(
            "not-ready",
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "go",
        )

        assert handle.result(timeout=5).final_output == "ready"
        assert factory_calls == 2
        assert len(model_calls) == 1


def test_external_wait_held_steer_backs_off_and_does_not_starve_another_session(tmp_path: Path) -> None:
    provider = _PendingProvider()
    held = "a-held"

    with _temporary_driver(tmp_path, "rotation", worker_level=2) as driver:

        def runtime_factory(store, agent, config, task, session_id):
            runtime = driver.runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=store,
            )
            if session_id == held:
                runtime.providers["external_wait"] = provider
                runtime.poll_ms = 60_000
            return runtime

        driver.runtime_factory = runtime_factory
        driver.create(held, str(tmp_path))
        driver.start(
            held,
            Agent("provider", "Wait externally.", model="m", tools=[external_wait]),
            _config(
                [
                    LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
                    LLMResponse("unreachable"),
                ]
            ),
            "go",
        )
        _wait_for(lambda: bool(provider.handle))
        driver._worker.wait_idle(timeout=3)
        driver._worker.scan_interval = 1.0

        state, _, _ = driver.store.read_state(held)
        assert state.active_turn_id is not None
        turn = state.turns[state.active_turn_id]
        driver.push(
            held,
            InboxItem(
                "held-steer",
                "steer",
                {"content": "wait"},
                state.active_turn_id,
                turn.start._payload["generation"],
            ),
        )

        driver.create("z-other", str(tmp_path))
        other = driver.start(
            "z-other",
            Agent("other", "Finish.", model="m"),
            _config([LLMResponse("other done")]),
            "go",
        )
        assert other.result(timeout=5).final_output == "other done"

        row = cast(Any, driver.store)._rows(
            "SELECT drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (driver.tenant_id, held),
        )
        assert row[0][0] > 0
        assert provider.queries <= 1


def test_factory_error_identity_and_backoff_cause_reach_subscriber(tmp_path: Path) -> None:
    with _temporary_driver(tmp_path, "factory-error") as driver:
        original_error = ValueError("factory-original")
        backoff_error = OSError("backoff-write-failed")
        factory_calls: list[str] = []
        backoff_calls: list[int] = []

        def broken_backoff(_lease, *, retry_after_ms: int) -> None:
            backoff_calls.append(retry_after_ms)
            raise backoff_error

        def runtime_factory(store, agent, config, task, session_id):
            del agent, config, task
            factory_calls.append(session_id)
            store.defer_drive = broken_backoff
            raise original_error

        driver.runtime_factory = runtime_factory
        driver.create("factory-error", str(tmp_path))
        handle = driver.start(
            "factory-error",
            Agent("agent", "Answer.", model="m"),
            _config([]),
            "go",
        )

        with pytest.raises(ValueError) as caught:
            handle.result(timeout=5)
        assert caught.value is original_error
        assert caught.value.__cause__ is backoff_error
        assert factory_calls == ["factory-error"]
        assert len(backoff_calls) == 1


def test_pre_core_sqlite_lock_defers_before_release_and_recovers_without_model_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "pre-core-sqlite-lock"
    lock_error = sqlite3.OperationalError("database is locked")
    model_calls: list[LlmRequest] = []

    def response(request: LlmRequest) -> LLMResponse:
        model_calls.append(request)
        return LLMResponse("lock recovered")

    with _temporary_driver(tmp_path, sid) as driver:
        driver.create(sid, str(tmp_path))
        handle = driver.start(
            sid,
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=ScriptedModelProvider.from_callback("scripted", "m", response)),
            "go",
            autostart=False,
        )
        assert handle._pending_submission is not None
        handle._pending_submission()
        handle._pending_submission = None
        handle._runtime_reservation = None

        with driver._open_drive_store() as store:
            bound = store.for_tenant(driver.tenant_id)
            lease = bound.acquire(sid, owner="pre-core-lock", ttl_ms=driver._worker.support.lease_ttl_ms)
            assert lease is not None
            original_read = bound.read
            locked = True

            def locked_read(*args, **kwargs):
                nonlocal locked
                if locked:
                    locked = False
                    raise lock_error
                return original_read(*args, **kwargs)

            monkeypatch.setattr(bound, "read", locked_read)
            driver._drive_worker_session(bound, sid, lease=lease)

            failure = driver._errors[sid]
            assert failure.error is lock_error
            assert not failure.terminal
            assert failure.attempts == 1
            retry_at_ms = cast(Any, bound)._one(
                "SELECT drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (driver.tenant_id, sid),
            )[0]
            assert retry_at_ms > 0

        _force_due(driver, sid)
        with driver._open_drive_store() as claim_store:
            claimed = _tick(
                claim_store,
                project=lambda *_args: None,
                dispatch_claimed=lambda session_id, claimed_lease: driver._drive_worker_session(
                    claim_store,
                    session_id,
                    lease=claimed_lease,
                ),
                work_kinds=("drive",),
                stop_after_claimed_drive=True,
            )

        assert claimed == 1
        state, records, _ = driver.store.read_state(sid)
        recovered_failure = driver._errors.get(sid)
        assert state.active_turn_id is None
        assert any(stored.record.kind == "turn_ended" and stored.record.payload["status"] == "completed" for stored in records)
        assert len(model_calls) == 1
        assert recovered_failure is None


@pytest.mark.parametrize(
    "lock_message",
    ["database is locked", "database table is locked: sk_session"],
)
def test_transient_sqlite_lock_retries_without_terminal_failure_or_model_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_message: str,
) -> None:
    from vv_agent.session.kernel import _Driver

    original_step = _Driver.step
    original_drive = SessionDriver._drive_worker_session
    lock_error = sqlite3.OperationalError(lock_message)
    step_calls = 0
    observed_terminal: list[bool] = []
    observed_attempts: list[int] = []
    observed_retry_after_ms: list[int] = []
    model_calls: list[LlmRequest] = []

    def locked_once(driver):
        nonlocal step_calls
        step_calls += 1
        if step_calls == 1:
            raise lock_error
        return original_step(driver)

    def track_failure(driver, store, session_id, *, lease):
        try:
            return original_drive(driver, store, session_id, lease=lease)
        finally:
            failure = driver._errors.get(session_id)
            if failure is not None and failure.error is lock_error:
                observed_terminal.append(failure.terminal)
                observed_attempts.append(failure.attempts)
                retry_at = cast(Any, store)._one(
                    "SELECT drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                    (driver.tenant_id, session_id),
                )[0]
                observed_retry_after_ms.append(retry_at)

    def response(request: LlmRequest) -> LLMResponse:
        model_calls.append(request)
        return LLMResponse("lock recovered")

    monkeypatch.setattr(_Driver, "step", locked_once)
    monkeypatch.setattr(SessionDriver, "_drive_worker_session", track_failure)

    sid = "sqlite-lock-retry"
    with _temporary_driver(tmp_path, sid) as driver:
        driver.create(sid, str(tmp_path))
        handle = driver.start(
            sid,
            Agent("agent", "Answer.", model="m"),
            RunConfig(model_provider=ScriptedModelProvider.from_callback("scripted", "m", response)),
            "go",
        )

        assert handle.result(timeout=5).final_output == "lock recovered"
        assert observed_terminal == [False]
        assert observed_attempts == [1]
        assert observed_retry_after_ms and observed_retry_after_ms[0] > 0
        assert step_calls >= 2
        assert len(model_calls) == 1
        state, records, _ = driver.store.read_state(sid)
        assert state.active_turn_id is None
        assert all(stored.record.kind != "turn_failed" for stored in records)


def test_ledger_read_failure_reaches_subscriber_once_without_turn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_error = LedgerIntegrityError("corrupt worker snapshot")
    worker_reads = 0
    worker_drive_calls = 0
    second_drive_finished = Event()

    from vv_agent.session.kernel import _Driver

    original_step = _Driver.step
    original_drive = SessionDriver._drive_worker_session

    def corrupt_step(selected):
        nonlocal worker_reads
        if current_thread().name.startswith("session-local-worker-drive"):
            worker_reads += 1
            raise read_error
        return original_step(selected)

    def tracked_worker_drive(self, store, session_id, *, lease):
        nonlocal worker_drive_calls
        call_number = 0
        if current_thread().name.startswith("session-local-worker-drive"):
            worker_drive_calls += 1
            call_number = worker_drive_calls
        try:
            return original_drive(self, store, session_id, lease=lease)
        finally:
            if call_number >= 2:
                second_drive_finished.set()

    monkeypatch.setattr(_Driver, "step", corrupt_step)
    monkeypatch.setattr(SessionDriver, "_drive_worker_session", tracked_worker_drive)

    with _temporary_driver(tmp_path, "corrupt-read") as driver:
        driver.create("corrupt-read", str(tmp_path))
        handle = driver.start(
            "corrupt-read",
            Agent("agent", "Answer.", model="m"),
            _config([]),
            "go",
        )

        with pytest.raises(LedgerIntegrityError) as caught:
            handle.result(timeout=5)
        assert caught.value is read_error

        assert second_drive_finished.wait(5)
        assert worker_drive_calls >= 2
        assert worker_reads == 1

        state, records, _ = driver.store.read_state("corrupt-read")
        assert state.turns == {}
        assert all(stored.record.kind not in {"turn_failed", "turn_ended"} for stored in records)
        assert len(driver.store.peek_inbox("corrupt-read")) == 1


@pytest.mark.parametrize("recovery", ["provider_result", "cancel"])
def test_provider_invalid_fence_survives_unrelated_inputs_until_real_recovery(
    tmp_path: Path,
    recovery: str,
) -> None:
    provider = _InvalidQueryProvider()
    sid = f"provider-fence-{recovery}"

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:

        def runtime_factory(store, agent, config, task, session_id):
            runtime = driver.runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=store,
            )
            runtime.providers["external_wait"] = provider
            return runtime

        driver.runtime_factory = runtime_factory
        driver.create(sid, str(tmp_path))
        model = ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
                LLMResponse("recovered"),
            ],
        )
        handle = driver.start(
            sid,
            Agent("provider", "Wait externally.", model="m", tools=[external_wait]),
            RunConfig(model_provider=model),
            "go",
        )

        driver._worker.wait_idle(timeout=3)
        _force_due(driver, sid)
        _wait_for(
            lambda: (
                (failure := driver._errors.get(sid)) is not None
                and isinstance(failure.error, ProviderResultInvalid)
                and failure.terminal
            )
        )
        driver._worker.wait_idle(timeout=3)
        failure = driver._errors[sid]
        assert provider.queries == 1

        selected_model = {"model": "m", "route": {}}
        selection = InboxItem(
            "select",
            "model_selection",
            {"selection": selected_model},
            feature_level=2,
            requirements=static_model_selection_requirements(
                selected_model,
                resolve_model=lambda selection: resolve_model_selection(model, selection),
                handler_version="1",
            ),
        )
        driver.push(sid, selection)
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        replay = driver.push(sid, selection)
        assert replay.replayed
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        driver.push(sid, InboxItem("stale-control", "control", {"action": "cancel"}, "stale-turn"))
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        state, _, _ = driver.store.read_state(sid)
        assert state.active_turn_id is not None
        turn = state.turns[state.active_turn_id]
        driver.push(
            sid,
            InboxItem(
                "held-steer",
                "steer",
                {"content": "must wait for the provider"},
                state.active_turn_id,
                turn.start._payload["generation"],
            ),
        )
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        driver.push(
            sid,
            InboxItem(
                "stale-generation-steer",
                "steer",
                {"content": "stale generation must stay pending"},
                state.active_turn_id,
                turn.start._payload["generation"] + 1,
            ),
        )
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        _force_due(driver, sid)
        driver._worker.wait_idle(timeout=3)
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1

        _force_due(driver, sid)
        if recovery == "provider_result":
            payload = {
                "operation_id": provider.handle["operation_id"],
                "attempt": provider.handle["attempt"],
                "request_digest": provider.handle["request_digest"],
                "provider_binding": provider.handle["provider"],
                "result": provider.result,
                "usage": {},
                "evidence": [provider.handle["evidence"]],
            }
            driver.push(
                sid,
                InboxItem(
                    "provider-result",
                    "provider_result",
                    payload,
                    state.active_turn_id,
                    turn.start._payload["generation"],
                ),
            )
            result = handle.result(timeout=5)
            assert result.status is AgentStatus.COMPLETED
            assert result.final_output == "recovered"
        else:
            driver.control(sid, "cancel", "cancel-provider")
            result = handle.result(timeout=5)
            assert result.status is AgentStatus.FAILED
            assert result.completion_reason is CompletionReason.CANCELLED
        assert provider.queries == 1


def test_tx_push_provider_result_advances_the_core_without_requery(tmp_path: Path) -> None:
    provider = _InvalidQueryProvider()
    sid = "tx-provider-result"

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:
        handle, _, _failure = _induce_provider_invalid(driver, sid, provider)
        state, _, _ = driver.store.read_state(sid)
        assert state.active_turn_id is not None
        turn = state.turns[state.active_turn_id]
        payload = {
            "operation_id": provider.handle["operation_id"],
            "attempt": provider.handle["attempt"],
            "request_digest": provider.handle["request_digest"],
            "provider_binding": provider.handle["provider"],
            "result": provider.result,
            "usage": {},
            "evidence": [provider.handle["evidence"]],
        }

        _push_in_transaction_and_wake(
            driver,
            sid,
            InboxItem(
                "provider-result",
                "provider_result",
                payload,
                state.active_turn_id,
                turn.start._payload["generation"],
            ),
        )

        assert handle.result(timeout=5).final_output == "recovered"
        assert provider.queries == 1
        assert sid not in driver._errors


def test_tx_push_archive_preserves_provider_failure_and_ack(tmp_path: Path) -> None:
    provider = _InvalidQueryProvider()
    sid = "tx-provider-archive"

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:
        _, _, failure = _induce_provider_invalid(driver, sid, provider)
        failure.warning_sent = True

        _push_in_transaction_and_wake(driver, sid, InboxItem("archive", "control", {"action": "archive"}))
        _wait_for(
            lambda: any(
                stored.record.kind == "input_applied" and stored.record.payload["input"]["input_id"] == "archive"
                for stored in driver.store.read(sid, limit=100).records
            )
        )
        driver._worker.wait_idle(timeout=3)

        assert driver._errors.get(sid) is failure
        assert failure.warning_sent is True
        assert provider.queries == 1
        assert driver.store.read_state(sid)[0].archived is True


def test_tx_push_forged_provider_result_keeps_failure_and_advances_progress_head(tmp_path: Path) -> None:
    provider = _InvalidQueryProvider()
    sid = "tx-provider-forged"

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:
        _, _, failure = _induce_provider_invalid(driver, sid, provider)
        failure.warning_sent = True
        state, _, _ = driver.store.read_state(sid)
        assert state.active_turn_id is not None
        turn = state.turns[state.active_turn_id]
        payload = {
            "operation_id": provider.handle["operation_id"],
            "attempt": provider.handle["attempt"],
            "request_digest": provider.handle["request_digest"],
            "provider_binding": provider.handle["provider"],
            "result": provider.result,
            "usage": {},
            "evidence": ["forged"],
        }

        _push_in_transaction_and_wake(
            driver,
            sid,
            InboxItem(
                "forged-provider-result",
                "provider_result",
                payload,
                state.active_turn_id,
                turn.start._payload["generation"],
            ),
        )
        _wait_for(
            lambda: any(
                stored.record.kind == "input_applied" and stored.record.payload["input"]["input_id"] == "forged-provider-result"
                for stored in driver.store.read(sid, limit=100).records
            )
        )
        driver._worker.wait_idle(timeout=3)

        assert driver._errors.get(sid) is failure
        assert failure.warning_sent is True
        assert failure.progress_head == driver.store.read(sid, limit=1).head_seq
        assert provider.queries == 1


def test_provider_invalid_observer_keeps_failure_before_worker_head_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _InvalidQueryProvider()
    sid = "tx-provider-observer-race"

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:
        _, _, failure = _induce_provider_invalid(driver, sid, provider)
        failure.warning_sent = True
        observed: list[bool] = []
        from vv_agent.session import kernel as kernel_module

        original_drive = kernel_module.drive

        def drive_then_observe(store, session_id, **kwargs):
            result = original_drive(store, session_id, **kwargs)
            if session_id == sid:
                driver._snapshot_status(store, sid, None, failure.turn_id or "")
                observed.append(
                    driver._errors.get(sid) is failure
                    and failure.warning_sent
                    and driver.store.read(sid, limit=1).head_seq > failure.progress_head
                )
            return result

        monkeypatch.setattr(kernel_module, "drive", drive_then_observe)
        _push_in_transaction_and_wake(driver, sid, InboxItem("archive", "control", {"action": "archive"}))
        _wait_for(lambda: bool(observed))
        driver._worker.wait_idle(timeout=3)

        assert observed == [True]
        assert driver._errors.get(sid) is failure
        assert provider.queries == 1


def test_provider_failure_identity_does_not_rebind_to_new_wait_across_workers(tmp_path: Path) -> None:
    sid = "cross-worker-provider-identity"
    database = tmp_path / f"{sid}.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{sid}-keys.sqlite3")
    bad_provider = _InvalidQueryProvider()
    healthy_provider = _PendingProvider()
    model_calls: list[LlmRequest] = []
    responses = iter(
        [
            LLMResponse("", [ToolCall("bad-call", "external_wait", {})]),
            LLMResponse("", [ToolCall("healthy-call", "healthy_external_wait", {})]),
        ]
    )

    def response(request: LlmRequest) -> LLMResponse:
        model_calls.append(request)
        return next(responses)

    model = ScriptedModelProvider.from_callback("scripted", "m", response)
    agent = Agent(
        "provider",
        "Wait externally.",
        model="m",
        tools=[external_wait, healthy_external_wait],
    )
    config = RunConfig(model_provider=model)

    providers = {
        "external_wait": bad_provider,
        "healthy_external_wait": healthy_provider,
    }

    stores = ExitStack()
    first = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    second = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    try:
        first.runtime_factory = _runtime_factory_for(first, providers)
        first.create(sid, "")
        first.start(sid, agent, config, "go")

        _wait_for(lambda: bool(bad_provider.handle))
        first._worker.wait_idle(timeout=3)
        _force_due(first, sid)
        _wait_for(
            lambda: (
                (failure := first._errors.get(sid)) is not None
                and isinstance(failure.error, ProviderResultInvalid)
                and failure.terminal
            )
        )
        first._worker.wait_idle(timeout=3)
        failure = first._errors[sid]
        failure.warning_sent = True

        original_state, _, _ = first.store.read_state(sid)
        original_key = (bad_provider.handle["operation_id"], bad_provider.handle["attempt"])
        assert isinstance(failure.error, ProviderResultInvalid)
        assert failure.error._query_operation_key == original_key
        assert failure.fenced_operations == frozenset({original_key})

        first._worker.stop()
        assert first._worker.join(5)

        second.runtime_factory = _runtime_factory_for(second, providers)
        second._runtimes[sid] = second._static_registration(
            sid,
            agent,
            config,
            None,
            handler_version="1",
            armed=True,
        )
        second._worker.start()
        turn = original_state.turns[original_state.active_turn_id or ""]
        second.push(
            sid,
            InboxItem(
                "recover-op1",
                "provider_result",
                {
                    "operation_id": bad_provider.handle["operation_id"],
                    "attempt": bad_provider.handle["attempt"],
                    "request_digest": bad_provider.handle["request_digest"],
                    "provider_binding": bad_provider.handle["provider"],
                    "result": bad_provider.result,
                    "usage": {},
                    "evidence": [bad_provider.handle["evidence"]],
                },
                failure.turn_id,
                turn.start._payload["generation"],
            ),
        )

        _wait_for(lambda: healthy_provider.queries == 1)
        second._worker.wait_idle(timeout=3)
        second._worker.stop()
        assert second._worker.join(5)

        recovered_state, _, _ = first.store.read_state(sid)
        healthy_key = (
            healthy_provider.handle["operation_id"],
            healthy_provider.handle["attempt"],
        )
        assert healthy_provider.handle != bad_provider.handle
        assert recovered_state.active_turn_id == failure.turn_id
        assert failure.fenced_operations == frozenset({original_key})
        assert healthy_key not in failure.fenced_operations
        assert len(model_calls) == 2

        _force_due(first, sid)
        with first._open_drive_store() as claim_store:
            claimed = _tick(
                claim_store,
                project=lambda *_args: None,
                dispatch_claimed=lambda session_id, lease: first._drive_worker_session(
                    claim_store,
                    session_id,
                    lease=lease,
                ),
                work_kinds=("drive",),
                stop_after_claimed_drive=True,
            )
        assert claimed == 1

        assert first._errors.get(sid) is None
        assert healthy_provider.queries == 2
        assert bad_provider.queries == 1
        assert len(model_calls) == 2
    finally:
        second.close()
        first.close()
        stores.close()
        key_provider.close()


def test_provider_query_error_fences_only_the_failing_same_batch_operation(tmp_path: Path) -> None:
    sid = "provider-same-batch-identity"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{sid}-keys.sqlite3")
    bad_provider = _InvalidQueryProvider()
    healthy_provider = _PendingProvider()
    model = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse(
                "",
                [
                    ToolCall("bad-call", "external_wait", {}),
                    ToolCall("healthy-call", "healthy_external_wait", {}),
                ],
            )
        ],
    )
    stores = ExitStack()
    driver = SessionDriver(
        store=stores.enter_context(_temporary_store(tmp_path / f"{sid}.sqlite", write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    try:
        driver.runtime_factory = _runtime_factory_for(
            driver,
            {
                "external_wait": bad_provider,
                "healthy_external_wait": healthy_provider,
            },
        )
        driver.create(sid, "")
        driver.start(
            sid,
            Agent(
                "provider",
                "Wait externally.",
                model="m",
                tools=[external_wait, healthy_external_wait],
            ),
            RunConfig(model_provider=model),
            "go",
        )

        _wait_for(lambda: bool(bad_provider.handle))
        driver._worker.wait_idle(timeout=3)
        _force_due(driver, sid)
        _wait_for(
            lambda: (
                (failure := driver._errors.get(sid)) is not None
                and isinstance(failure.error, ProviderResultInvalid)
                and failure.fenced_operations
            )
        )

        state, _, _ = driver.store.read_state(sid)
        bad_key = (bad_provider.handle["operation_id"], bad_provider.handle["attempt"])
        failure = driver._errors[sid]
        healthy_operations = [
            (operation_id, max(operation.attempts))
            for operation_id, operation in state.operations.items()
            if operation.turn_id == failure.turn_id
            and (operation_id, max(operation.attempts)) != bad_key
            and operation.kind == "tool"
        ]
        assert len(healthy_operations) == 1
        healthy_key = healthy_operations[0]
        failure = driver._errors[sid]
        assert isinstance(failure.error, ProviderResultInvalid)
        assert failure.error._query_operation_key == bad_key
        assert failure.fenced_operations == frozenset({bad_key})
        assert healthy_key not in failure.fenced_operations

        turn = state.turns[state.active_turn_id or ""]
        _push_in_transaction_and_wake(
            driver,
            sid,
            InboxItem(
                "complete-bad",
                "provider_result",
                {
                    "operation_id": bad_provider.handle["operation_id"],
                    "attempt": bad_provider.handle["attempt"],
                    "request_digest": bad_provider.handle["request_digest"],
                    "provider_binding": bad_provider.handle["provider"],
                    "result": bad_provider.result,
                    "usage": {},
                    "evidence": [bad_provider.handle["evidence"]],
                },
                failure.turn_id,
                turn.start._payload["generation"],
            ),
        )
        _wait_for(lambda: healthy_provider.queries == 1 and sid not in driver._errors)
        driver._worker.wait_idle(timeout=3)
        healthy_state, _, _ = driver.store.read_state(sid)
        submitted_healthy_key = (
            healthy_provider.handle["operation_id"],
            healthy_provider.handle["attempt"],
        )
        assert submitted_healthy_key == healthy_key
        assert healthy_state.waits[healthy_key]["handle"]["kind"] == "provider"
    finally:
        driver.close()
        stores.close()
        key_provider.close()


def test_provider_query_boundary_replaces_preexisting_private_operation_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "provider-prefilled-error-identity"
    provider = _PrefilledInvalidQueryProvider()
    from vv_agent.kernel import _driver_recovery as recovery_module

    def reject_acceptance(plan, outcome):
        del plan, outcome
        raise provider.error

    monkeypatch.setattr(recovery_module, "validate_acceptance", reject_acceptance)
    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:
        driver.runtime_factory = _runtime_factory_for(driver, {"external_wait": provider})
        driver.create(sid, "")
        driver.start(
            sid,
            Agent("provider", "Wait externally.", model="m", tools=[external_wait]),
            _config([LLMResponse("", [ToolCall("external-call", "external_wait", {})])]),
            "go",
        )
        _wait_for(lambda: bool(provider.handle))
        driver._worker.wait_idle(timeout=3)
        _force_due(driver, sid)
        _wait_for(
            lambda: (failure := driver._errors.get(sid)) is not None and failure.error is provider.error and failure.terminal
        )

        failure = driver._errors[sid]
        current_key = (provider.handle["operation_id"], provider.handle["attempt"])
        assert failure.error is provider.error
        assert failure.error.code == ProviderResultInvalid.code
        assert str(failure.error) == ProviderResultInvalid().code
        assert failure.error._query_operation_key == current_key
        assert failure.fenced_operations == frozenset({current_key})
        assert provider.queries == 1


def test_provider_failure_observer_reconciles_remote_terminal_without_local_claim(tmp_path: Path) -> None:
    sid = "provider-observer-remote-terminal"
    database = tmp_path / f"{sid}.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{sid}-keys.sqlite3")
    provider = _InvalidQueryProvider()
    model = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("external-call", "external_wait", {})]),
            LLMResponse("remote done"),
        ],
    )
    agent = Agent("provider", "Wait externally.", model="m", tools=[external_wait])
    config = RunConfig(model_provider=model)
    providers = {"external_wait": provider}
    stores = ExitStack()
    first = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    second = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    try:
        first.runtime_factory = _runtime_factory_for(first, providers)
        first.create(sid, "")
        first.start(sid, agent, config, "go")
        _wait_for(lambda: bool(provider.handle))
        first._worker.wait_idle(timeout=3)
        _force_due(first, sid)
        _wait_for(
            lambda: (
                (failure := first._errors.get(sid)) is not None
                and isinstance(failure.error, ProviderResultInvalid)
                and failure.terminal
            )
        )
        first._worker.wait_idle(timeout=3)
        failure = first._errors[sid]
        failure.warning_sent = True
        old_key = (provider.handle["operation_id"], provider.handle["attempt"])
        assert failure.fenced_operations == frozenset({old_key})
        first._worker.stop()
        assert first._worker.join(5)

        state, _, _ = first.store.read_state(sid)
        turn = state.turns[failure.turn_id or ""]
        second.runtime_factory = _runtime_factory_for(second, providers)
        second._runtimes[sid] = second._static_registration(
            sid,
            agent,
            config,
            None,
            handler_version="1",
            armed=True,
        )
        second._worker.start()
        second.push(
            sid,
            InboxItem(
                "remote-result",
                "provider_result",
                {
                    "operation_id": provider.handle["operation_id"],
                    "attempt": provider.handle["attempt"],
                    "request_digest": provider.handle["request_digest"],
                    "provider_binding": provider.handle["provider"],
                    "result": provider.result,
                    "usage": {},
                    "evidence": [provider.handle["evidence"]],
                },
                failure.turn_id,
                turn.start._payload["generation"],
            ),
        )

        def remote_terminal() -> bool:
            current_state, _, _ = first.store.read_state(sid)
            turn = current_state.turns.get(failure.turn_id or "")
            return turn is not None and turn.ended

        _wait_for(remote_terminal)
        second._worker.wait_idle(timeout=3)
        second._worker.stop()
        assert second._worker.join(5)

        observed_state, _, _ = first._read_observed_state(first.store, sid)
        _, _, error = first._snapshot_status(
            first.store,
            sid,
            observed_state,
            failure.turn_id or "",
        )
        assert first._errors.get(sid) is None
        assert error is None
    finally:
        second.close()
        first.close()
        stores.close()
        key_provider.close()


def test_recovered_operation_new_query_failure_gets_independent_identity_and_ack(tmp_path: Path) -> None:
    sid = "provider-new-failure-identity"
    database = tmp_path / f"{sid}.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{sid}-keys.sqlite3")
    old_provider = _InvalidQueryProvider()
    new_provider = _InvalidQueryProvider()
    model_calls: list[LlmRequest] = []
    responses = iter(
        [
            LLMResponse("", [ToolCall("old-call", "external_wait", {})]),
            LLMResponse("", [ToolCall("new-call", "replacement_external_wait", {})]),
        ]
    )

    def response(request: LlmRequest) -> LLMResponse:
        model_calls.append(request)
        return next(responses)

    model = ScriptedModelProvider.from_callback("scripted", "m", response)
    agent = Agent(
        "provider",
        "Wait externally.",
        model="m",
        tools=[external_wait, replacement_external_wait],
    )
    config = RunConfig(model_provider=model)
    providers = {
        "external_wait": old_provider,
        "replacement_external_wait": new_provider,
    }
    stores = ExitStack()
    first = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    writer = SessionDriver(
        store=stores.enter_context(_temporary_store(database, write_level=2)),
        content_key_provider=key_provider,
        worker_level=2,
        handler_versions=("1",),
    )
    try:
        first.runtime_factory = _runtime_factory_for(first, providers)
        first.create(sid, "")
        first.start(sid, agent, config, "go")
        _wait_for(lambda: bool(old_provider.handle))
        first._worker.wait_idle(timeout=3)
        _force_due(first, sid)
        _wait_for(
            lambda: (
                (old_failure := first._errors.get(sid)) is not None
                and isinstance(old_failure.error, ProviderResultInvalid)
                and old_failure.terminal
            )
        )
        first._worker.wait_idle(timeout=3)
        old_failure = first._errors[sid]
        old_failure.warning_sent = True
        old_key = (old_provider.handle["operation_id"], old_provider.handle["attempt"])
        assert old_failure.fenced_operations == frozenset({old_key})
        first._worker.stop()
        assert first._worker.join(5)

        state, _, _ = first.store.read_state(sid)
        turn = state.turns[old_failure.turn_id or ""]
        writer.push(
            sid,
            InboxItem(
                "recover-old",
                "provider_result",
                {
                    "operation_id": old_provider.handle["operation_id"],
                    "attempt": old_provider.handle["attempt"],
                    "request_digest": old_provider.handle["request_digest"],
                    "provider_binding": old_provider.handle["provider"],
                    "result": old_provider.result,
                    "usage": {},
                    "evidence": [old_provider.handle["evidence"]],
                },
                old_failure.turn_id,
                turn.start._payload["generation"],
            ),
        )

        _force_due(first, sid)
        with first._open_drive_store() as claim_store:
            claimed = _tick(
                claim_store,
                project=lambda *_args: None,
                dispatch_claimed=lambda session_id, lease: first._drive_worker_session(
                    claim_store,
                    session_id,
                    lease=lease,
                ),
                work_kinds=("drive",),
                stop_after_claimed_drive=True,
            )
        assert claimed == 1

        if sid not in first._errors:
            assert new_provider.handle
            _force_due(first, sid)
            with first._open_drive_store() as claim_store:
                claimed = _tick(
                    claim_store,
                    project=lambda *_args: None,
                    dispatch_claimed=lambda session_id, lease: first._drive_worker_session(
                        claim_store,
                        session_id,
                        lease=lease,
                    ),
                    work_kinds=("drive",),
                    stop_after_claimed_drive=True,
                )
            assert claimed == 1

        new_failure = first._errors[sid]
        new_key = (new_provider.handle["operation_id"], new_provider.handle["attempt"])
        assert new_failure is not old_failure
        assert new_failure.error is not old_failure.error
        assert isinstance(new_failure.error, ProviderResultInvalid)
        assert new_failure.error._query_operation_key == new_key
        assert new_failure.fenced_operations == frozenset({new_key})
        assert new_failure.warning_sent is False
        assert old_provider.queries == 1
        assert new_provider.queries == 1
        assert len(model_calls) == 2
    finally:
        writer.close()
        first.close()
        stores.close()
        key_provider.close()


def test_provider_batch_approval_scope_is_hydrated_by_the_core(tmp_path: Path) -> None:
    provider = _InvalidQueryProvider()
    sid = "tx-provider-approval"
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def approval_effect() -> str:
        calls.append("approved")
        return "approved"

    class AskApproval:
        def should_request(self, request):
            del request
            return True

        def decide(self, request):
            del request
            return None

    with _temporary_driver(tmp_path, sid, worker_level=2) as driver:

        def runtime_factory(store, agent, config, task, session_id):
            runtime = driver.runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=store,
            )
            runtime.providers["external_wait"] = provider
            return runtime

        driver.runtime_factory = runtime_factory
        driver.create(sid, str(tmp_path))
        model = ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse(
                    "",
                    [
                        ToolCall("approval-call", "approval_effect", {}),
                        ToolCall("external-call", "external_wait", {}),
                    ],
                ),
                LLMResponse("recovered"),
            ],
        )
        driver.start(
            sid,
            Agent("provider", "Run a batch.", model="m", tools=[external_wait, approval_effect]),
            RunConfig(model_provider=model, approval_provider=AskApproval()),
            "go",
        )
        driver._worker.wait_idle(timeout=3)

        def approval_wait():
            try:
                return next(
                    wait for _target, _runtime, wait in driver.waits(sid, driver._runtime_view(sid)) if wait["kind"] == "approval"
                )
            except (KeyError, StopIteration):
                return None

        _wait_for(lambda: approval_wait() is not None)
        approval = approval_wait()
        assert approval is not None
        state, _, _ = driver.store.read_state(sid)
        failure = _SessionFailure(
            state.active_turn_id,
            ProviderResultInvalid(),
            1,
            driver.store.read(sid, limit=1).head_seq,
            True,
            5,
            0.01,
            0.5,
            lambda low, high: low,
        )
        failure.warning_sent = True
        driver._errors[sid] = failure

        driver.approve(
            sid,
            driver._runtime_view(sid),
            approval["request_id"],
            "approve",
            "approval-answer",
        )
        _wait_for(lambda: calls == ["approved"])
        driver._worker.wait_idle(timeout=3)

        applied = next(
            stored.record
            for stored in driver.store.read(sid, limit=200).records
            if stored.record.kind == "input_applied" and stored.record.payload["input"]["input_id"] == "approval-answer"
        )
        assert applied.payload["disposition"] == "applied"
