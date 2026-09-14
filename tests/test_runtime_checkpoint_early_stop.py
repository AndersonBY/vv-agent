from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from test_checkpoint import _minimal_checkpoint
from test_checkpoint_runner import _config, _provider

from vv_agent import Agent, Runner, ToolCallOutcome, ToolContext, function_tool
from vv_agent.budget import BudgetEnforcementBoundary, HostCost, RunBudgetLimits
from vv_agent.checkpoint import AmbiguousToolPolicy, CheckpointConfig, CheckpointError, OperationState, ResumePolicy
from vv_agent.events import DiagnosticEvent, RunEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.state import CheckpointStore
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, CompletionReason, LLMResponse, Message, ToolCall


class _CrashAfterFourthCycle:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.crash = True
        self.history_reads = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def load_checkpoint_history(self, checkpoint_key: str) -> Any:
        self.history_reads += 1
        return self.inner.load_checkpoint_history(checkpoint_key)

    def commit_checkpoint(self, checkpoint: Any, **kwargs: Any) -> bool:
        written = self.inner.commit_checkpoint(checkpoint, **kwargs)
        if written and checkpoint.cycle_index == 4 and self.crash:
            self.crash = False
            raise SystemExit("crash after fourth committed cycle")
        return written


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
@pytest.mark.parametrize("entrypoint", ["runtime", "runner"])
@pytest.mark.parametrize("stop", ["cancel", "budget_cancel", "budget_exhausted"])
def test_restored_early_stop_preserves_checkpoint_tail_and_can_finalize(
    tmp_path: Path, kind: str, entrypoint: str, stop: str
) -> None:
    inner = InMemoryCheckpointStore() if kind == "memory" else SqliteCheckpointStore(tmp_path / "early-stop.sqlite")
    store = _CrashAfterFourthCycle(inner)
    seed = _minimal_checkpoint(key="early-stop")
    token = CancellationToken()
    events: list[RunEvent] = []
    model_calls = 0
    injections = 0

    class Meter:
        amount = 0

        def read(self) -> HostCost:
            return HostCost(unit="credits", amount_microunits=self.amount)

    meter = Meter()
    limits = (
        RunBudgetLimits(max_total_tokens=10000, max_host_cost=HostCost(unit="credits", amount_microunits=100))
        if stop != "cancel"
        else None
    )

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        assert store.history_reads == 0
        model_calls += 1
        return LLMResponse(
            content=f"cycle {model_calls}",
            raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )

    def inject(index: int, _messages: list[Message], shared: dict[str, Any]) -> list[Message]:
        nonlocal injections
        injections += 1
        shared["last_injected_cycle"] = index
        return [Message(role="user", content=f"durable instruction {index}")]

    def new_controller() -> CheckpointResumeController:
        return CheckpointResumeController(
            config=CheckpointConfig(store=store, key=seed.checkpoint_key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
            task_id=seed.task_id,
            run_id=seed.root_run_id,
            trace_id=seed.trace_id,
            run_definition=deepcopy(seed.run_definition),
            run_definition_digest=seed.run_definition_digest,
            initial_messages=[],
            initial_shared_state={},
            initial_budget_usage=None,
            extensions=[],
            reconciliation_provider=None,
            event_sink=events.append,
        )

    controller = new_controller() if entrypoint == "runtime" else None
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[complete] * 8),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id=seed.task_id,
        model="test-model",
        prompt_bundle=build_raw_system_prompt_bundle("continue"),
        user_prompt="retain history",
        max_cycles=8,
        no_tool_policy="continue",
        use_workspace=False,
    )
    config = _config(
        cast(CheckpointStore, store),
        key=seed.checkpoint_key,
        provider=_provider(lambda: ScriptedLLM(steps=[complete] * 8)),
        max_cycles=8,
        no_tool_policy="continue",
        capability_refs={
            "before_cycle_messages": {"id": "test.early-stop-injection", "version": "1"},
            "host_cost_meter": {"id": "test.early-stop-meter", "version": "1"},
        },
    )
    config.budget_limits = limits
    config.host_cost_meter = meter
    config.before_cycle_messages = inject
    config.cancellation_token = token
    config.stream = events.append
    agent = Agent(name="early-stop", instructions="continue", model="test-model")

    def run() -> Any:
        if controller is None:
            return Runner.run_sync(agent, "retain history", run_config=config).raw_result
        return runtime.run(
            task,
            checkpoint_controller=controller,
            ctx=ExecutionContext(cancellation_token=token, event_handler=events.append),
            budget_limits=limits,
            initial_budget_usage=controller.budget_usage,
            host_cost_meter=meter,
            before_cycle_messages=inject,
        )

    try:
        if controller is not None:
            assert controller.admit() is None
        with pytest.raises(SystemExit, match="crash after fourth committed cycle"):
            run()
        assert store.history_reads == 0
        durable = store.load_checkpoint(seed.checkpoint_key)
        assert durable is not None
        assert [cycle.index for cycle in durable.cycles] == [4]
        assert durable.history["cycle_count"] == 3
        if controller is not None:
            controller.close()
            controller = new_controller()
            assert controller.admit() is None
        if stop == "budget_exhausted":
            meter.amount = 100
        else:
            token.cancel()
        events.clear()
        result = run()
        assert result.status is AgentStatus.FAILED
        assert result.completion_reason is (
            CompletionReason.BUDGET_EXHAUSTED if stop == "budget_exhausted" else CompletionReason.CANCELLED
        )
        assert [cycle.index for cycle in result.cycles] == [1, 2, 3, 4]
        assert [call.cycle_index for call in result.token_usage.model_calls] == [1, 2, 3, 4]
        assert result.token_usage.total_tokens == 60
        assert result.messages == durable.messages
        assert result.shared_state == durable.shared_state
        assert result.shared_state["last_injected_cycle"] == 4
        assert result.partial_output == "cycle 4"
        assert model_calls == injections == 4
        if limits is not None:
            assert result.budget_usage is not None
            assert result.budget_usage.total_tokens == 60
        if stop == "budget_exhausted":
            assert result.budget_exhaustion is not None
            assert result.budget_exhaustion.enforcement_boundary is BudgetEnforcementBoundary.RUN_START
        else:
            assert [
                event.cycle_index for event in events if isinstance(event, DiagnosticEvent) and event.code == "run_cancelled"
            ] == [4]
        if controller is not None:
            assert store.history_reads == 1
            result = controller.finalize(controller.prepare_terminal(result))
            controller.close()
            controller = new_controller()
            replay = controller.admit()
        else:
            replay = run()
        terminal = store.load_checkpoint(seed.checkpoint_key)
        assert terminal is not None and terminal.terminal_result is not None
        assert [cycle.index for cycle in terminal.terminal_result.cycles] == [4]
        assert terminal.messages == durable.messages
        assert terminal.shared_state == durable.shared_state
        assert replay is not None and replay.cycles == result.cycles
        assert replay.token_usage == result.token_usage
        assert model_calls == injections == 4
    finally:
        if controller is not None:
            controller.close()
        if isinstance(inner, SqliteCheckpointStore):
            inner.close()


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
@pytest.mark.parametrize("stop", ["cancel", "budget_exhausted"])
@pytest.mark.parametrize("operation", ["deferred", "unknown"])
def test_budget_early_stop_cannot_finalize_or_rerun_outstanding_operation(
    tmp_path: Path, kind: str, stop: str, operation: str
) -> None:
    store = InMemoryCheckpointStore() if kind == "memory" else SqliteCheckpointStore(tmp_path / "barrier.sqlite")
    model_calls = 0
    effects = 0
    token = CancellationToken()

    class Meter:
        amount = 0

        def read(self) -> HostCost:
            return HostCost(unit="credits", amount_microunits=self.amount)

    meter = Meter()

    @function_tool(name="outstanding", tool_metadata={"idempotency": "unknown"})
    def outstanding(context: ToolContext) -> ToolCallOutcome:
        nonlocal effects
        effects += 1
        if operation == "unknown":
            raise SystemExit("unconfirmed external effect")
        return context.defer()

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(
            content=f"cycle {model_calls}",
            tool_calls=[ToolCall(id="outstanding-5", name="outstanding", arguments={})] if model_calls == 5 else [],
            raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )

    config = _config(
        store,
        key="outstanding",
        provider=_provider(lambda: ScriptedLLM(steps=[complete] * 8)),
        max_cycles=8,
        no_tool_policy="continue",
        ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
        capability_refs={"host_cost_meter": {"id": "test.barrier-meter", "version": "1"}},
    )
    config.budget_limits = RunBudgetLimits(max_total_tokens=10000, max_host_cost=HostCost(unit="credits", amount_microunits=100))
    config.host_cost_meter = meter
    config.cancellation_token = token
    agent = Agent(name="barrier", instructions="continue", model="test-model", tools=[outstanding])

    def run() -> Any:
        return Runner.run_sync(agent, "retain outstanding operation", run_config=config)

    try:
        if operation == "unknown":
            with pytest.raises(SystemExit, match="unconfirmed external effect"):
                run()
            # Expire only this disposable test's abandoned lease.
            if isinstance(store, SqliteCheckpointStore):
                with store._conn:
                    store._conn.execute("UPDATE checkpoints SET lease_expires_at_ms = 1 WHERE checkpoint_key = 'outstanding'")
            else:
                with store._lock:
                    store._store["outstanding"].lease_expires_at_ms = 1
        else:
            assert run().status is AgentStatus.DEFERRED
        before = store.load_checkpoint("outstanding")
        assert before is not None
        if stop == "cancel":
            token.cancel()
        else:
            meter.amount = 100
        try:
            resumed = run()
        except CheckpointError as error:
            # An unowned unknown operation may fail closed at terminal preparation;
            # a cancellation/budget candidate cannot replace its durable authority.
            assert operation == "unknown"
            assert error.code == "checkpoint_claim_active"
        else:
            assert resumed.status is (AgentStatus.RECONCILIATION_REQUIRED if operation == "unknown" else AgentStatus.DEFERRED)
            assert [cycle.index for cycle in resumed.raw_result.cycles] == [1, 2, 3, 4]
            assert resumed.token_usage.total_tokens == 75
            assert resumed.raw_result.budget_exhaustion is None
            assert all(event.type != "budget_exhausted" for event in resumed.events)
        after = store.load_checkpoint("outstanding")
        assert after is not None and after.terminal_result is None
        assert after.cycles == before.cycles
        assert after.messages == before.messages
        assert after.history == before.history
        assert after.tool_journal[0].state in (
            {OperationState.STARTED, OperationState.AMBIGUOUS} if operation == "unknown" else {OperationState.DEFERRED}
        )
        assert model_calls == 5
        assert effects == 1
    finally:
        if isinstance(store, SqliteCheckpointStore):
            store.close()
