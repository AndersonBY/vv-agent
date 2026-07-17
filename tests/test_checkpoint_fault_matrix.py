from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    CheckpointConfig,
    RunBudgetLimits,
    RunConfig,
    Runner,
    ToolContext,
    ToolIdempotency,
    function_tool,
)
from vv_agent.checkpoint import OperationState, ResumePolicy
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import BaseRuntimeHook, BeforeLLMEvent, CheckpointStoreV2
from vv_agent.runtime.state import InMemoryStateStore
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.types import AgentStatus, CompletionReason, LLMResponse, ToolCall


class FaultStore(InMemoryStateStore):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault
        self.tripped = False

    def _trip(self) -> None:
        if self.tripped:
            return
        self.tripped = True
        raise SystemExit(f"fault:{self.fault}")

    def progress_checkpoint_v2(
        self,
        checkpoint: Any,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        model_state = (
            checkpoint.model_call_journal[-1].state
            if checkpoint.model_call_journal
            else None
        )
        tool_state = (
            checkpoint.tool_journal[-1].state
            if checkpoint.tool_journal
            else None
        )
        if self.fault == "before_model_receipt" and model_state is OperationState.SUCCEEDED:
            self._trip()
        result = super().progress_checkpoint_v2(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if self.fault == "after_model_started" and model_state is OperationState.STARTED:
            self._trip()
        if self.fault == "after_tool_planned" and tool_state is OperationState.PLANNED:
            self._trip()
        return result

    def commit_checkpoint_v2(
        self,
        checkpoint: Any,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if self.fault == "before_cycle_commit":
            self._trip()
        result = super().commit_checkpoint_v2(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if self.fault == "after_cycle_commit":
            self._trip()
        return result

    def acknowledge_terminal_v2(
        self,
        checkpoint_key: str,
        *,
        expected_revision: int,
    ) -> bool:
        if self.fault == "before_terminal_ack":
            self._trip()
        return super().acknowledge_terminal_v2(
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
) -> Callable[[Agent, RunConfig], tuple[ScriptedLLM, ResolvedModelConfig]]:
    def resolve(
        agent: Agent,
        run_config: RunConfig,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del agent, run_config
        return factory(), _resolved()

    return resolve


def _config(
    store: CheckpointStoreV2,
    *,
    key: str,
    provider: Callable[[Agent, RunConfig], tuple[ScriptedLLM, ResolvedModelConfig]],
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
            capability_refs=capability_refs or {},
        ),
    )


def _expire_claim(store: InMemoryStateStore, key: str) -> None:
    with store._lock:
        store._store_v2[key].lease_expires_at_ms = 1


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
    store = InMemoryStateStore()
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
    crashed = store.load_checkpoint_v2("fault-f1")
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

    @function_tool(name="fault_f4_write", idempotency=ToolIdempotency.SUPPORTED)
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

    @function_tool(name="fault_f6_write", idempotency=ToolIdempotency.SUPPORTED)
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
    committed = store.load_checkpoint_v2("fault-f7")
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
    committed = store.load_checkpoint_v2("budget-resume")
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
    terminal = store.load_checkpoint_v2("fault-f8")
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
    acknowledged = store.load_checkpoint_v2("fault-f8")
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
        from vv_agent.checkpoint import ResumePolicy
        from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
        from vv_agent.llm import ScriptedLLM
        from vv_agent.runtime.stores.sqlite import SqliteStateStore
        from vv_agent.types import LLMResponse, ToolCall

        database = Path(sys.argv[1])
        side_effect = Path(sys.argv[2])

        def provider(agent, run_config):
            del agent, run_config
            endpoint = EndpointConfig(
                endpoint_id="test",
                api_key="test-key",
                api_base="https://example.invalid/v1",
            )
            resolved = ResolvedModelConfig(
                backend="test",
                requested_model="test-model",
                selected_model="test-model",
                model_id="test-model",
                endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
                function_call_available=True,
            )
            response = LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call-sigkill", name="sigkill_write", arguments={"value": "once"})],
            )
            return ScriptedLLM(steps=[response]), resolved

        @function_tool(name="sigkill_write", idempotency=ToolIdempotency.UNKNOWN)
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
                    store=SqliteStateStore(database),
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
    store = SqliteStateStore(database)
    crashed = store.load_checkpoint_v2("sigkill-case")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.STARTED
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE checkpoints_v2 SET lease_expires_at_ms = 1 WHERE checkpoint_key = ?",
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

    @function_tool(name="sigkill_write", idempotency=ToolIdempotency.UNKNOWN)
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
