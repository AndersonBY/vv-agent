from __future__ import annotations

from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, select_metadata, to_json
from vv_agent.types import ToolArtifactRef, ToolExecutionResult, ToolResultStatus


def check_background_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    return _manage_background_command(context, arguments, stop=False)


def stop_background_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    return _manage_background_command(context, arguments, stop=True)


def _manage_background_command(context: ToolContext, arguments: dict[str, Any], *, stop: bool) -> ToolExecutionResult:
    session_id = str(arguments.get("session_id", "")).strip()
    if not session_id:
        return builtin_error("`session_id` is required", "session_id_required")
    operation = background_session_manager.stop_for_tool if stop else background_session_manager.check_for_tool
    payload = operation(
        session_id,
        context.workspace_backend,
        context.task_id,
        context.tool_call_id,
        workspace=context.workspace,
    )
    return background_command_result(payload)


def background_command_result(
    payload: dict[str, Any],
    *,
    foreground_cwd: str | None = None,
    force_handle: bool = False,
) -> ToolExecutionResult:
    """Project a completed management operation without treating its process as a tool continuation."""
    metadata = select_metadata(payload, "status", "session_id", "elapsed_seconds", "exit_code", "shell")
    status = payload.get("status")
    if payload.get("artifact_error"):
        return builtin_error(
            f"failed to persist complete command output: {payload['artifact_error']}",
            str(payload.get("artifact_error_code") or "artifact_persist_failed"),
            details=select_metadata(payload, "status", "session_id"),
            metadata=metadata,
        )
    if payload.get("output_error"):
        return builtin_error(
            f"failed to read command output: {payload['output_error']}",
            "command_failed",
            details=select_metadata(payload, "status", "session_id"),
            metadata=metadata,
        )
    if status not in {"running", "stopping", "unknown", "completed", "failed", "timeout", "stopped"}:
        error_code = str(payload.get("error_code") or "background_command_failed")
        return builtin_error(
            str(payload.get("error") or "Background command failed"),
            error_code,
            details=payload,
            metadata=metadata,
        )

    recovery: dict[str, Any] = {}
    truncated = bool(payload.get("output_truncated"))
    if truncated:
        raw_artifact = payload.get("artifact")
        try:
            artifact = ToolArtifactRef.from_dict(raw_artifact) if isinstance(raw_artifact, dict) else None
        except ValueError:
            artifact = None
        if artifact is None:
            return builtin_error(
                "complete command output has no recoverable artifact",
                "artifact_persist_failed",
                details=select_metadata(payload, "status", "session_id"),
                metadata=metadata,
            )
        recovery.update(truncated=True, truncation_reason="output_limit", artifact=artifact)

    ongoing = status in {"running", "stopping", "unknown"}
    exit_code = payload.get("exit_code")
    success = ongoing or (status != "timeout" and exit_code == 0)
    error_code = "command_failed" if foreground_cwd is not None else "background_command_failed"
    if ongoing or force_handle or status == "timeout":
        body = {key: value for key, value in payload.items() if key not in {"command", "output_json_bytes"}}
        if status == "timeout":
            body["message"] = "Command timed out"
        elif status in {"stopping", "unknown"}:
            body["message"] = "Process-tree termination is unconfirmed; check the session again."
        content = to_json(body)
        if truncated:
            full_output_json_bytes = int(payload["output_json_bytes"])
            # JSON escapes and the receipt envelope are part of visible content.
            # The artifact itself retains the unwrapped complete output prefix.
            while full_output_json_bytes < len(to_json(body["output"]).encode("utf-8")):
                output = body["output"]
                marker = output.index("\n... output omitted; full text in artifact ...\n")
                body["output"] = output[: marker - 1] + output[marker:]
                body["output_visible_bytes"] = len(body["output"].encode("utf-8"))
            content = to_json(body)
            visible_bytes = len(content.encode("utf-8"))
            recovery.update(
                visible_bytes=visible_bytes,
                original_bytes=visible_bytes - len(to_json(body["output"]).encode("utf-8")) + full_output_json_bytes,
            )
        return ToolExecutionResult(
            tool_call_id="",
            content=content,
            metadata=metadata,
            status_code=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
            error_code=None if success else error_code,
            **recovery,
        )

    content = str(payload.get("output") or "")
    if not content and not success:
        content = f"command exited with code {exit_code}" if status != "timeout" else "Command timed out"
    if foreground_cwd is not None:
        metadata = {"cwd": foreground_cwd, **select_metadata(payload, "exit_code", "shell")}
        if not success:
            metadata["error_code"] = error_code
    if truncated:
        recovery.update(original_bytes=payload["output_original_bytes"], visible_bytes=len(content.encode("utf-8")))
    return ToolExecutionResult(
        tool_call_id="",
        content=content,
        metadata=metadata,
        status_code=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
        error_code=None if success else error_code,
        **recovery,
    )
