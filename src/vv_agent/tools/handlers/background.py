from __future__ import annotations

from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, select_metadata, to_json
from vv_agent.types import ToolArtifactRef, ToolExecutionResult, ToolResultStatus


def check_background_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    session_id = str(arguments.get("session_id", "")).strip()
    if not session_id:
        return builtin_error("`session_id` is required", "session_id_required")

    result = background_session_manager.check_for_tool(
        session_id,
        context.workspace_backend,
        context.task_id,
        context.tool_call_id,
    )
    status = str(result.get("status", "missing"))

    artifact_error = result.get("artifact_error")
    if isinstance(artifact_error, str) and artifact_error:
        artifact_error_code = result.get("artifact_error_code")
        if artifact_error_code != "artifact_path_invalid":
            artifact_error_code = "artifact_persist_failed"
        return builtin_error(
            f"failed to persist complete command output: {artifact_error}",
            artifact_error_code,
        )

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

    if status in {"completed", "failed", "timeout"}:
        success = status == "completed"
        content = str(result.get("output") or "")
        if not content and not success:
            content = "Background command failed"
        metadata = select_metadata(
            result,
            "status",
            "session_id",
            "exit_code",
            "shell",
        )
        artifact = None
        if result.get("output_truncated") is True:
            raw_artifact = result.get("artifact")
            try:
                artifact = ToolArtifactRef.from_dict(raw_artifact) if isinstance(raw_artifact, dict) else None
            except ValueError:
                artifact = None
            if artifact is None:
                return builtin_error(
                    "complete background output has no recoverable artifact",
                    "artifact_persist_failed",
                )
        return ToolExecutionResult(
            tool_call_id="",
            status_code=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
            error_code=None if success else "background_command_failed",
            content=content,
            metadata=metadata,
            truncated=True if artifact is not None else None,
            truncation_reason="output_limit" if artifact is not None else None,
            original_bytes=result.get("output_original_bytes") if artifact is not None else None,
            visible_bytes=result.get("output_visible_bytes") if artifact is not None else None,
            artifact=artifact,
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
