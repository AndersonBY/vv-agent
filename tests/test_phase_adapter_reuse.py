"""The historical SDK phase entrypoints share the recovery host bridge."""

from contextlib import nullcontext
from pathlib import Path

from support.kernel_runtime import KernelRuntime

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel import _after_cycle as cycle
from vv_agent.kernel import _compaction_phase as compaction_core
from vv_agent.kernel import _driver_inputs as core_inputs
from vv_agent.kernel import _driver_planning as core_planning
from vv_agent.kernel._driver_requests import CreateBoundary, PlanModel
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.run_config import RunConfig
from vv_agent.session import _driver_inputs, _driver_planning, _recovery_phase, compaction, kernel, lifecycle, output
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse


def test_recovery_bridge_composes_ledger_requests_before_the_host(monkeypatch, tmp_path):
    ledger_requests = (
        PlanModel,
        CreateBoundary,
        core_inputs.CompleteResult,
        core_inputs.YieldResults,
        core_planning.ClosePlanning,
        compaction_core.CreateRecord,
        compaction_core.CloseCompaction,
        cycle.CompleteSkippedTool,
        cycle.CloseAfterCycle,
        cycle.RecordAfterCycle,
    )
    original_handle = _recovery_phase._Host.handle

    def observe_host(self, request):
        assert not isinstance(request, ledger_requests), type(request).__name__
        return original_handle(self, request)

    monkeypatch.setattr(_recovery_phase._Host, "handle", observe_host)
    runtime = KernelRuntime(
        llm_client=ScriptedLLM([LLMResponse("done")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    result = runtime.run(
        AgentTask(
            task_id="phase-adapter-ledger",
            model="dummy-model",
            prompt_bundle=build_raw_system_prompt_bundle("sys"),
            user_prompt="go",
            max_cycles=1,
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_answer == "done"


def test_historical_facades_use_the_real_sqlite_driver_bridge():
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "phase-adapter", "/tmp"), consumers=())
            tx.push("s", InboxItem("initial", "user", {"content": "go"}))

        runtime = kernel.Runtime(
            agent=Agent("kernel", "Be precise."),
            config=RunConfig(workspace=Path("/tmp"), session_memory_enabled=False),
            resolved=ResolvedModelConfig("dummy", "dummy", "dummy", "dummy", []),
            llm=ScriptedLLM([]),
            heartbeat_store=lambda: nullcontext(store),
        )
        lease = store.acquire("s", owner="phase-adapter", ttl_ms=15_000)
        assert lease is not None
        driver = kernel._Driver(store, "s", runtime, kernel._Scope(lease, runtime, now_ms=store._now()))

        assert _driver_inputs.apply_input(driver)
        assert driver.state.active_turn_id is None
        assert _driver_inputs.start_turn(driver)
        assert driver.state.active_turn_id == "s/turn/initial"
        assert _driver_planning.effective_model(driver)["model"] == "dummy"
        plan = _driver_planning.model_plan(driver)
        assert plan is not None and plan.kind == "op_planned"

        manager = compaction.manager_for(driver)
        source = driver.transcript()
        replacement = compaction.replacement_record(driver, manager, source, source, mode="micro")
        assert replacement.kind == "context_compacted"
        assert compaction.compact_context(driver) is False
        assert lifecycle.after_cycle(driver) is False
        assert output.prepare_output(driver, "direct") == ("completed", None, "direct")
