from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from v_agent.tools.base import ToolContext, ToolSpec
from v_agent.tools.dispatcher import dispatch_tool_call
from v_agent.tools.registry import ToolRegistry
from v_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus
from v_agent.workspace import LocalWorkspaceBackend


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def ok_handler(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult(tool_call_id="", status="success", content="ok")

    def wait_handler(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            content="need user",
            directive=ToolDirective.WAIT_USER,
        )

    registry.register(ToolSpec(name="_ok", handler=ok_handler))
    registry.register(ToolSpec(name="_wait", handler=wait_handler))
    return registry


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path, shared_state={"todo_list": []},
        cycle_index=1, workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_dispatch_tool_call_success_sets_tool_call_id(tmp_path: Path) -> None:
    result = dispatch_tool_call(
        registry=_build_registry(),
        context=_context(tmp_path),
        call=ToolCall(id="c1", name="_ok", arguments={}),
    )

    assert result.status_code == ToolResultStatus.SUCCESS
    assert result.tool_call_id == "c1"


def test_dispatch_tool_call_wait_user_maps_to_wait_response(tmp_path: Path) -> None:
    result = dispatch_tool_call(
        registry=_build_registry(),
        context=_context(tmp_path),
        call=ToolCall(id="c2", name="_wait", arguments={}),
    )

    assert result.directive == ToolDirective.WAIT_USER
    assert result.status_code == ToolResultStatus.WAIT_RESPONSE


def test_dispatch_tool_call_unknown_tool_returns_error(tmp_path: Path) -> None:
    result = dispatch_tool_call(
        registry=_build_registry(),
        context=_context(tmp_path),
        call=ToolCall(id="c3", name="_missing", arguments={}),
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "tool_not_found"


def test_dispatch_tool_call_invalid_arguments_returns_error(tmp_path: Path) -> None:
    result = dispatch_tool_call(
        registry=_build_registry(),
        context=_context(tmp_path),
        call=ToolCall(id="c4", name="_ok", arguments=cast(Any, "{not-json}")),
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "invalid_arguments_json"
