from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vv_agent import Agent, ApprovalRequestedEvent, MemorySession, RunConfig, Runner, ToolContext, ToolPolicy, function_tool
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.result import RunState
from vv_agent.runtime import InlineBackend
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="approval-model",
        selected_model="approval-model",
        model_id="approval-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="approval-model")],
    )


def test_interrupted_result_snapshot_and_runner_resume_execute_approved_call_once(tmp_path: Path) -> None:
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def delete_file(path: str) -> str:
        executions.append(path)
        return f"deleted {path}"

    llm_calls = 0

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        nonlocal llm_calls
        llm_calls += 1
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="delete",
                        tool_calls=[ToolCall(id="delete-call", name="delete_file", arguments={"path": "danger.txt"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="approver", instructions="Delete after approval.", model="approval-model", tools=[delete_file]),
        "delete danger.txt",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.WAIT_USER
    assert executions == []
    assert len(result.approvals) == 1
    approval = result.approvals[0]
    assert approval.tool_name == "delete_file"
    assert approval.tool_call_id == "delete-call"
    assert approval.arguments == {"path": "danger.txt"}
    requested = next(event for event in result.events if isinstance(event, ApprovalRequestedEvent))
    assert requested.request_id == approval.interruption_id

    state = result.into_state()
    assert isinstance(state, RunState)
    state.approve(approval.interruption_id)
    assert state.approvals[0].approved is True
    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "deleted danger.txt"
    assert executions == ["danger.txt"]
    assert llm_calls == 1


def test_run_handle_resume_uses_interrupted_result_state(tmp_path: Path) -> None:
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def guarded_write(value: str) -> str:
        executions.append(value)
        return value

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="write",
                        tool_calls=[ToolCall(id="write-call", name="guarded_write", arguments={"value": "approved"})],
                    )
                ]
            ),
            _resolved(),
        )

    handle = Runner.start(
        Agent(name="writer", instructions="Write after approval.", model="approval-model", tools=[guarded_write]),
        "write",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )
    interrupted = handle.result(timeout=2)
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = handle.resume(state)

    assert resumed is not None
    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "approved"
    assert executions == ["approved"]


def test_manual_approval_resume_rechecks_current_policy_and_preserves_tool_context(tmp_path: Path) -> None:
    executions: list[str] = []
    allowed = True
    manager = object()
    backend = InlineBackend()

    @function_tool(needs_approval=True)
    def guarded_write(context: ToolContext, value: str) -> str:
        executions.append(value)
        assert context.sub_task_manager is manager
        assert context.ctx is not None
        assert context.ctx.metadata["execution_backend"] is backend
        assert context.run_context is not None
        context.shared_state["resumed"] = value
        return value

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="write",
                        tool_calls=[ToolCall(id="guarded-call", name="guarded_write", arguments={"value": "approved"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="writer", instructions="Write after approval.", model="approval-model", tools=[guarded_write]),
        "write",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            sub_task_manager=manager,
            execution_backend=backend,
            shared_state={"original": True},
            tool_policy=ToolPolicy(can_use_tool=lambda _name, _arguments: allowed),
        ),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])
    allowed = False

    with pytest.raises(RuntimeError, match="not allowed"):
        Runner.resume(state)

    assert executions == []

    allowed = True
    result = Runner.run_sync(
        Agent(name="writer", instructions="Write after approval.", model="approval-model", tools=[guarded_write]),
        "write",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            sub_task_manager=manager,
            execution_backend=backend,
            shared_state={"original": True},
            tool_policy=ToolPolicy(can_use_tool=lambda _name, _arguments: allowed),
        ),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.raw_result.shared_state == {"original": True, "todo_list": [], "resumed": "approved"}
    assert executions == ["approved"]


def test_only_wait_user_results_convert_to_run_state() -> None:
    result = Runner.run_sync(
        Agent(
            name="done",
            instructions="Finish.",
            model=ScriptedLLM(
                steps=[LLMResponse(content="done", tool_calls=[ToolCall(id="finish", name="task_finish", arguments={})])]
            ),
        ),
        "go",
    )

    with pytest.raises(ValueError, match="only interrupted runs"):
        result.into_state()


def test_approval_snapshot_nested_arguments_are_isolated_and_resume_is_once_only(tmp_path: Path) -> None:
    executions: list[dict[str, object]] = []
    session = MemorySession("approval-once")

    @function_tool(needs_approval=True)
    def guarded(payload: dict[str, object]) -> str:
        executions.append(payload)
        return str(payload["nested"])

    arguments = {"payload": {"nested": {"value": "original"}}}
    result = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run once.",
            model=ScriptedLLM(
                steps=[LLMResponse(content="guard", tool_calls=[ToolCall(id="guard", name="guarded", arguments=arguments)])]
            ),
            tools=[guarded],
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, session=session),
    )
    state = result.into_state()
    approval = state.approvals[0]
    arguments["payload"]["nested"]["value"] = "mutated"  # type: ignore[index]
    approval.arguments["payload"]["nested"]["value"] = "also mutated"  # type: ignore[index]
    state.approve(approval.interruption_id)

    resumed = Runner.resume(state)

    assert executions == [{"nested": {"value": "original"}}]
    with pytest.raises(RuntimeError, match="approval_already_consumed"):
        Runner.resume(state)
    session_tool_results = [item for item in session.get_items() if item.role == "tool" and item.tool_call_id == "guard"]
    assert [item.content for item in session_tool_results] == [resumed.final_output]
    resumed_cycle = resumed.raw_result.cycles[0]
    assert len([item for item in resumed_cycle.tool_results if item.tool_call_id == "guard"]) == 1


def test_approval_resume_claim_is_shared_across_concurrent_state_uses(tmp_path: Path) -> None:
    executions = 0
    execution_lock = threading.Lock()

    @function_tool(needs_approval=True)
    def guarded(value: str) -> str:
        nonlocal executions
        with execution_lock:
            executions += 1
        return value

    result = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run once.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="guard",
                        tool_calls=[ToolCall(id="guard", name="guarded", arguments={"value": "ok"})],
                    )
                ]
            ),
            tools=[guarded],
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])
    barrier = threading.Barrier(2)

    def resume() -> str:
        barrier.wait(timeout=2)
        try:
            return Runner.resume(state).final_output or ""
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: resume(), range(2)))

    assert executions == 1
    assert sorted(outcomes) == ["approval_already_consumed", "ok"]
