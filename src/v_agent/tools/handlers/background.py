from __future__ import annotations

from typing import Any

from v_agent.runtime.background_sessions import background_session_manager
from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus


def check_background_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context
    session_id = str(arguments.get("session_id", "")).strip()
    if not session_id:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="session_id_required",
            content=to_json({"error": "`session_id` is required"}),
        )

    result = background_session_manager.check(session_id)
    status = str(result.get("status", "missing"))

    if status == "running":
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(result),
            metadata=result,
        )

    if status in {"completed"}:
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.SUCCESS,
            content=to_json(result),
            metadata=result,
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code="background_command_failed",
        content=to_json(result),
        metadata=result,
    )
