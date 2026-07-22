from __future__ import annotations

from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, select_metadata, to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus


def check_background_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context
    session_id = str(arguments.get("session_id", "")).strip()
    if not session_id:
        return builtin_error("`session_id` is required", "session_id_required")

    result = background_session_manager.check(session_id)
    status = str(result.get("status", "missing"))

    if status == "running":
        return ToolExecutionResult(
            tool_call_id="",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(result),
            metadata=select_metadata(
                result,
                "status",
                "session_id",
                "elapsed_seconds",
                "shell",
            ),
        )

    if status in {"completed"}:
        return ToolExecutionResult(
            tool_call_id="",
            status_code=ToolResultStatus.SUCCESS,
            content=to_json(result),
            metadata=select_metadata(
                result,
                "status",
                "session_id",
                "exit_code",
                "shell",
            ),
        )

    error = str(result.get("error") or "")
    if not error:
        error = "Background command timed out" if status == "timeout" else "Background command failed"
    details = {key: value for key, value in result.items() if key != "error"}
    metadata = select_metadata(
        result,
        "status",
        "session_id",
        "exit_code",
        "shell",
    )
    return builtin_error(
        error,
        "background_command_failed",
        details=details,
        metadata=metadata,
    )
