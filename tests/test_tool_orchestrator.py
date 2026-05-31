from __future__ import annotations

from vv_agent.tools import ToolContext
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.function import function_tool
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import ToolCall, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def test_function_tool_is_adapted_to_tool_executor() -> None:
    @function_tool
    def echo(value: str) -> str:
        return value

    executor = echo.to_executor()

    assert executor.name == "echo"
    assert executor.exposure == ToolExposure.DIRECT
    assert executor.spec(None).name == "echo"


def test_orchestrator_rejects_tool_not_allowed_for_batch(tmp_path) -> None:
    @function_tool
    def hidden() -> str:
        return "no"

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    orchestrator = ToolOrchestrator.from_tools([hidden])
    result = orchestrator.run_one(
        ToolCall(id="call_1", name="hidden", arguments={}),
        context=context,
        allowed_tool_names={"other"},
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "tool_not_allowed"


def test_orchestrator_unknown_tool_returns_error(tmp_path) -> None:
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = ToolOrchestrator.from_tools([]).run_one(
        ToolCall(id="call_1", name="missing", arguments={}),
        context=context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "tool_not_found"


def test_orchestrator_normalizes_tool_exceptions(tmp_path) -> None:
    @function_tool
    def fails() -> str:
        raise RuntimeError("boom")

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = ToolOrchestrator.from_tools([fails]).run_one(
        ToolCall(id="call_1", name="fails", arguments={}),
        context=context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "tool_execution_failed"


def test_orchestrator_emits_tool_started_and_completed_events(tmp_path) -> None:
    @function_tool
    def echo(value: str) -> str:
        return value

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )
    events = []

    result = ToolOrchestrator.from_tools([echo]).run_one(
        ToolCall(id="call_1", name="echo", arguments={"value": "ok"}),
        context=context,
        event_sink=events.append,
    )

    assert result.status_code == ToolResultStatus.SUCCESS
    assert [event.type for event in events] == ["tool_call_started", "tool_call_completed"]
    assert events[0].tool_name == "echo"
    assert events[1].tool_call_id == "call_1"
