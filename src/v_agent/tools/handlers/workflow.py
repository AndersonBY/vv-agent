from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus


def _not_enabled(tool_name: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code="workflow_tools_not_enabled",
        content=to_json(
            {
                "error": f"{tool_name} is not enabled in this runtime",
                "error_code": "workflow_tools_not_enabled",
            }
        ),
    )


def create_workflow(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("_create_workflow")


def run_workflow(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("_run_workflow")
