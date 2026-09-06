from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from support import FactoryModelProvider

from vv_agent import (
    Agent,
    CheckpointConfig,
    RunBudgetLimits,
    RunConfig,
    Runner,
    ToolContext,
    function_tool,
)
from vv_agent.checkpoint import AmbiguousToolPolicy, CheckpointError, OperationState, ResumePolicy
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import BaseRuntimeHook, BeforeLLMEvent, CheckpointStore
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.state import Checkpoint, CheckpointRenewal, RenewOutcome
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.types import AgentStatus, CompletionReason, LLMResponse, ToolCall


class FaultStore(InMemoryCheckpointStore):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault
        self.tripped = False

    def _trip(self) -> None:
        if self.tripped:
            return
        self.tripped = True
        raise SystemExit(f"fault:{self.fault}")

    def progress_checkpoint(
        self,
        checkpoint: Any,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        model_state = checkpoint.model_call_journal[-1].state if checkpoint.model_call_journal else None
        tool_state = checkpoint.tool_journal[-1].state if checkpoint.tool_journal else None
        if self.fault == "before_model_receipt" and model_state is OperationState.SUCCEEDED:
            self._trip()
        result = super().progress_checkpoint(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if self.fault == "after_model_started" and model_state is OperationState.STARTED:
            self._trip()
        if self.fault == "after_tool_planned" and tool_state is OperationState.PLANNED:
            self._trip()
        return result

    def commit_checkpoint(
        self,
        checkpoint: Any,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if self.fault == "before_cycle_commit":
            self._trip()
        result = super().commit_checkpoint(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if self.fault == "after_cycle_commit":
            self._trip()
        return result

    def acknowledge_terminal(
        self,
        checkpoint_key: str,
        *,
        expected_revision: int,
    ) -> bool:
        if self.fault == "before_terminal_ack":
            self._trip()
        return super().acknowledge_terminal(
            checkpoint_key,
            expected_revision=expected_revision,
        )


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        function_call_available=True,
    )


def _provider(
    factory: Callable[[], ScriptedLLM],
) -> FactoryModelProvider:
    return FactoryModelProvider(factory=factory, resolved=_resolved())


def _config(
    store: CheckpointStore,
    *,
    key: str,
    provider: FactoryModelProvider,
    max_cycles: int = 2,
    capability_refs: dict[str, dict[str, str]] | None = None,
) -> RunConfig:
    return RunConfig(
        model_provider=provider,
        max_cycles=max_cycles,
        no_tool_policy="continue",
        checkpoint_config=CheckpointConfig(
            key=key,
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
            capability_refs=capability_refs or {},
        ),
    )


def _expire_claim(store: InMemoryCheckpointStore, key: str) -> None:
    with store._lock:
        store._store[key].lease_expires_at_ms = 1


class _HeartbeatStore(InMemoryCheckpointStore):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.renew_calls = 0

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal:
        self.renew_calls += 1
        if self.mode == "transient" and self.renew_calls == 1:
            raise RuntimeError("store unavailable")
        if self.mode == "false":
            return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=0)
        if self.mode == "cancel":
            return CheckpointRenewal(
                outcome=RenewOutcome.CANCEL_REQUESTED,
                lease_expires_at_ms=lease_expires_at_ms,
            )
        return CheckpointRenewal(
            outcome=RenewOutcome.RENEWED,
            lease_expires_at_ms=lease_expires_at_ms,
        )


def _heartbeat_controller(store: InMemoryCheckpointStore, key: str) -> tuple[CheckpointResumeController, Checkpoint]:
    claimed = cast(
        Checkpoint,
        SimpleNamespace(
            checkpoint_key=key,
            claim_token="heartbeat-claim",
            claimed_cycle=1,
            lease_expires_at_ms=5_000,
        ),
    )
    controller = CheckpointResumeController.__new__(CheckpointResumeController)
    controller.checkpoint = claimed
    controller.store = store
    controller.lease_duration_ms = 1_000
    controller._owned_claim_token = "heartbeat-claim"
    controller._heartbeat_error = None
    controller._heartbeat_stop = SimpleNamespace(set=lambda: None)
    controller._now_ms = lambda: 1_000
    return controller, claimed


def test_heartbeat_retries_transient_store_error_only_while_local_lease_is_live() -> None:
    store = _HeartbeatStore("transient")
    controller, claimed = _heartbeat_controller(store, "heartbeat-transient")
    assert CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert store.renew_calls == 1
    assert CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert store.renew_calls == 2
    assert claimed.lease_expires_at_ms == 2_000


def test_heartbeat_keeps_live_claim_while_propagating_cancel_request() -> None:
    store = _HeartbeatStore("cancel")
    controller, claimed = _heartbeat_controller(store, "heartbeat-cancel")

    assert CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert claimed.cancel_requested
    assert claimed.claim_token == "heartbeat-claim"
    assert controller._owned_claim_token == "heartbeat-claim"
    assert controller._heartbeat_error is None

    assert CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert store.renew_calls == 2


@pytest.mark.parametrize("mode", ["false", "expired"])
def test_heartbeat_false_or_expired_local_lease_drops_claim_without_retry(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _HeartbeatStore("false" if mode == "false" else "success")
    controller, claimed = _heartbeat_controller(store, f"heartbeat-{mode}")
    monkeypatch.setattr(controller, "_now_ms", lambda: 6_000 if mode == "expired" else 1_000)
    assert not CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert (store.renew_calls, claimed.claim_token, controller._owned_claim_token) == (0 if mode == "expired" else 1, None, None)
    with pytest.raises(CheckpointError, match="checkpoint lease") as error:
        CheckpointResumeController._assert_heartbeat(controller)
    assert error.value.code == "checkpoint_lease_lost"


@pytest.mark.parametrize("method", ["_progress", "_renew_claim_before_dispatch"])
def test_expired_local_claim_rejects_progress_and_dispatch(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _HeartbeatStore("success")
    controller, _claimed = _heartbeat_controller(store, f"heartbeat-{method}")
    monkeypatch.setattr(controller, "_now_ms", lambda: 6_000)
    with pytest.raises(CheckpointError, match="checkpoint lease") as error:
        getattr(CheckpointResumeController, method)(controller)
    assert (error.value.code, store.renew_calls) == ("checkpoint_lease_lost", 0)


def test_dispatch_transient_renewal_failure_preserves_claim_for_retry() -> None:
    store = _HeartbeatStore("transient")
    controller, claimed = _heartbeat_controller(store, "heartbeat-dispatch-transient")

    with pytest.raises(CheckpointError) as error:
        controller._renew_claim_before_dispatch()

    assert error.value.code == "checkpoint_store_conflict"
    assert claimed.claim_token == "heartbeat-claim"
    assert controller._owned_claim_token == "heartbeat-claim"
    controller._renew_claim_before_dispatch()
    assert store.renew_calls == 2


def test_renewal_return_after_known_expiry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _HeartbeatStore("success")
    controller, claimed = _heartbeat_controller(store, "heartbeat-late")
    monkeypatch.setattr(controller, "_now_ms", lambda: (1_000, 6_000)[store.renew_calls])
    assert not CheckpointResumeController._heartbeat_tick(controller, claimed, "heartbeat-claim")
    assert store.renew_calls == 1


def _finish_response(message: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id=f"finish-{message}",
                name="task_finish",
                arguments={"message": message},
            )
        ],
    )


def test_f1_crash_before_model_intent_resumes_with_one_model_call() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    class CrashBeforeIntent(BaseRuntimeHook):
        def __init__(self) -> None:
            self.crash = True

        def before_llm(self, event: BeforeLLMEvent) -> None:
            del event
            if self.crash:
                self.crash = False
                raise SystemExit("fault:before_model_intent")

    hook = CrashBeforeIntent()

    def model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return _finish_response("done")

        return ScriptedLLM(steps=[complete])

    agent = Agent(
        name="fault-f1-agent",
        instructions="Finish once.",
        model="test-model",
        hooks=[hook],
    )
    config = _config(
        store,
        key="fault-f1",
        provider=_provider(model),
        capability_refs={"runtime_hook:0": {"id": "hook.f1", "version": "1"}},
    )

    with pytest.raises(SystemExit, match="before_model_intent"):
        Runner.run_sync(agent, "run F1", run_config=config)
    crashed = store.load_checkpoint("fault-f1")
    assert crashed is not None
    assert crashed.model_call_journal == []

    resumed = Runner.run_sync(agent, "run F1", run_config=config)

    assert resumed.status is AgentStatus.COMPLETED
    assert model_calls == 1


@pytest.mark.parametrize(
    ("fault", "provider_calls"),
    [
        ("after_model_started", 0),
        ("before_model_receipt", 1),
    ],
)
def test_f2_f3_started_model_recovers_as_explicit_ambiguity(
    fault: str,
    provider_calls: int,
) -> None:
    store = FaultStore(fault)
    model_calls = 0

    def model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return _finish_response("unretained")

        return ScriptedLLM(steps=[complete])

    agent = Agent(name=f"fault-{fault}-agent", instructions="Finish once.", model="test-model")
    config = _config(store, key=f"fault-{fault}", provider=_provider(model))

    with pytest.raises(SystemExit, match=f"fault:{fault}"):
        Runner.run_sync(agent, f"run {fault}", run_config=config)
    assert model_calls == provider_calls
    _expire_claim(store, f"fault-{fault}")

    resumed = Runner.run_sync(
        agent,
        f"run {fault}",
        run_config=_config(
            store,
            key=f"fault-{fault}",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.completion_reason is None
    assert model_calls == provider_calls


def test_f4_planned_tool_is_invoked_after_resume() -> None:
    store = FaultStore("after_tool_planned")
    tool_calls = 0

    @function_tool(name="fault_f4_write", tool_metadata={"idempotency": "supported"})
    def write(context: ToolContext) -> str:
        nonlocal tool_calls
        assert context.idempotency_key is not None
        tool_calls += 1
        return "written"

    response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call-f4", name="fault_f4_write", arguments={})],
    )
    agent = Agent(
        name="fault-f4-agent",
        instructions="Write once.",
        model="test-model",
        tools=[write],
        tool_use_behavior="stop_on_first_tool",
    )
    config = _config(
        store,
        key="fault-f4",
        provider=_provider(lambda: ScriptedLLM(steps=[response])),
    )

    with pytest.raises(SystemExit, match="after_tool_planned"):
        Runner.run_sync(agent, "run F4", run_config=config)
    assert tool_calls == 0
    _expire_claim(store, "fault-f4")

    resumed = Runner.run_sync(
        agent,
        "run F4",
        run_config=_config(
            store,
            key="fault-f4",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert tool_calls == 1


def test_f6_durable_tool_receipt_replays_without_external_calls() -> None:
    store = FaultStore("before_cycle_commit")
    model_calls = 0
    tool_calls = 0

    @function_tool(name="fault_f6_write", tool_metadata={"idempotency": "supported"})
    def write(context: ToolContext) -> str:
        nonlocal tool_calls
        assert context.idempotency_key is not None
        tool_calls += 1
        return "written"

    def model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call-f6", name="fault_f6_write", arguments={})],
            )

        return ScriptedLLM(steps=[complete])

    agent = Agent(
        name="fault-f6-agent",
        instructions="Write once.",
        model="test-model",
        tools=[write],
    )
    config = _config(store, key="fault-f6", provider=_provider(model), max_cycles=2)

    with pytest.raises(SystemExit, match="before_cycle_commit"):
        Runner.run_sync(agent, "run F6", run_config=config)
    assert (model_calls, tool_calls) == (1, 1)
    _expire_claim(store, "fault-f6")

    resumed = Runner.run_sync(
        agent,
        "run F6",
        run_config=_config(
            store,
            key="fault-f6",
            provider=_provider(lambda: ScriptedLLM(steps=[_finish_response("done")])),
            max_cycles=2,
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert (model_calls, tool_calls) == (1, 1)


def test_f7_committed_cycle_resumes_at_next_cycle() -> None:
    store = FaultStore("after_cycle_commit")
    model_calls = 0

    def first_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="continue")

        return ScriptedLLM(steps=[complete])

    agent = Agent(name="fault-f7-agent", instructions="Continue, then finish.", model="test-model")
    config = _config(store, key="fault-f7", provider=_provider(first_model), max_cycles=2)

    with pytest.raises(SystemExit, match="after_cycle_commit"):
        Runner.run_sync(agent, "run F7", run_config=config)
    committed = store.load_checkpoint("fault-f7")
    assert committed is not None
    assert committed.cycle_index == 1
    assert committed.claim_token is None

    def second_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return _finish_response("done")

        return ScriptedLLM(steps=[complete])

    resumed = Runner.run_sync(
        agent,
        "run F7",
        run_config=_config(
            store,
            key="fault-f7",
            provider=_provider(second_model),
            max_cycles=2,
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert len(resumed.raw_result.cycles) == 2
    assert model_calls == 2


def test_committed_budget_usage_is_cumulative_after_resume() -> None:
    store = FaultStore("after_cycle_commit")
    model_calls = 0
    limits = RunBudgetLimits(max_total_tokens=25)

    def first_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(
                content="continue",
                raw={
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 5,
                        "total_tokens": 20,
                    }
                },
            )

        return ScriptedLLM(steps=[complete])

    agent = Agent(
        name="budget-resume-agent",
        instructions="Continue once, then answer.",
        model="test-model",
    )
    first_config = _config(
        store,
        key="budget-resume",
        provider=_provider(first_model),
        max_cycles=2,
    )
    first_config.budget_limits = limits

    with pytest.raises(SystemExit, match="after_cycle_commit"):
        Runner.run_sync(agent, "run budget resume", run_config=first_config)
    committed = store.load_checkpoint("budget-resume")
    assert committed is not None
    assert committed.budget_usage is not None
    assert committed.budget_usage.total_tokens == 20

    def second_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(
                content="would exceed",
                raw={
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "total_tokens": 10,
                    }
                },
            )

        return ScriptedLLM(steps=[complete])

    second_config = _config(
        store,
        key="budget-resume",
        provider=_provider(second_model),
        max_cycles=2,
    )
    second_config.budget_limits = limits
    resumed = Runner.run_sync(
        agent,
        "run budget resume",
        run_config=second_config,
    )

    assert resumed.status is AgentStatus.FAILED
    assert resumed.completion_reason is CompletionReason.BUDGET_EXHAUSTED
    assert resumed.budget_usage is not None
    assert resumed.budget_usage.total_tokens == 30
    assert model_calls == 2


def test_f8_terminal_commit_replays_before_ack_without_external_calls() -> None:
    store = FaultStore("before_terminal_ack")
    model_calls = 0

    def model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return _finish_response("durable terminal")

        return ScriptedLLM(steps=[complete])

    agent = Agent(name="fault-f8-agent", instructions="Finish once.", model="test-model")
    config = _config(store, key="fault-f8", provider=_provider(model))

    with pytest.raises(SystemExit, match="before_terminal_ack"):
        Runner.run_sync(agent, "run F8", run_config=config)
    terminal = store.load_checkpoint("fault-f8")
    assert terminal is not None
    assert terminal.terminal_result is not None
    assert terminal.terminal_acknowledged is False

    replay = Runner.run_sync(
        agent,
        "run F8",
        run_config=_config(
            store,
            key="fault-f8",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )

    assert replay.status is AgentStatus.COMPLETED
    assert replay.final_output == "durable terminal"
    assert model_calls == 1
    acknowledged = store.load_checkpoint("fault-f8")
    assert acknowledged is not None
    assert acknowledged.terminal_acknowledged is True


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="SIGKILL is unavailable")
def test_sigkill_after_tool_side_effect_requires_reconciliation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sigkill-checkpoint.sqlite3"
    side_effect = tmp_path / "side-effect.txt"
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        from pathlib import Path

        from vv_agent import Agent, CheckpointConfig, RunConfig, Runner, ToolIdempotency, function_tool
        from vv_agent.checkpoint import AmbiguousToolPolicy, ResumePolicy
        from vv_agent.llm import ScriptedLLM
        from vv_agent.model import ScriptedModelProvider
        from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
        from vv_agent.types import LLMResponse, ToolCall

        database = Path(sys.argv[1])
        side_effect = Path(sys.argv[2])

        response = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="call-sigkill", name="sigkill_write", arguments={"value": "once"})],
        )
        provider = ScriptedModelProvider(
            backend="test",
            default_model="test-model",
            llm=ScriptedLLM(steps=[response]),
        )

        @function_tool(name="sigkill_write", tool_metadata={"idempotency": "unknown"})
        def sigkill_write(value: str) -> str:
            side_effect.write_text(value, encoding="utf-8")
            os.kill(os.getpid(), signal.SIGKILL)
            return "unreachable"

        Runner.run_sync(
            Agent(
                name="sigkill-agent",
                instructions="Write exactly once.",
                model="test-model",
                tools=[sigkill_write],
            ),
            "run SIGKILL canary",
            run_config=RunConfig(
                model_provider=provider,
                max_cycles=1,
                no_tool_policy="continue",
                checkpoint_config=CheckpointConfig(
                    key="sigkill-case",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
                    store=SqliteCheckpointStore(database),
                ),
            ),
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(database), str(side_effect)],
        check=False,
        timeout=30,
    )

    assert completed.returncode == -signal.SIGKILL
    assert side_effect.read_text(encoding="utf-8") == "once"
    store = SqliteCheckpointStore(database)
    crashed = store.load_checkpoint("sigkill-case")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.STARTED
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE checkpoints SET lease_expires_at_ms = 1 WHERE checkpoint_key = ?",
            ("sigkill-case",),
        )

    model_calls = 0
    tool_calls = 0

    def empty_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            raise AssertionError("model must not run before reconciliation")

        return ScriptedLLM(steps=[complete])

    @function_tool(name="sigkill_write", tool_metadata={"idempotency": "unknown"})
    def sigkill_write(value: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return value

    resumed = Runner.run_sync(
        Agent(
            name="sigkill-agent",
            instructions="Write exactly once.",
            model="test-model",
            tools=[sigkill_write],
        ),
        "run SIGKILL canary",
        run_config=_config(
            store,
            key="sigkill-case",
            provider=_provider(empty_model),
            max_cycles=1,
        ),
    )

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.completion_reason is None
    assert (model_calls, tool_calls) == (0, 0)
    assert side_effect.read_text(encoding="utf-8") == "once"
