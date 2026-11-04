from __future__ import annotations

from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, select_metadata, to_json
from vv_agent.types import ToolArtifactRef, ToolExecutionResult, ToolResultStatus
from vv_agent.workspace.artifacts import BOUNDED_TEXT_CHARS, PREVIEW_HEAD_CHARS, PREVIEW_MARKER

_UNAVAILABLE_MARKER = "\n... complete text could not be saved; retention unavailable; do not rerun the command ...\n"


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
    truncated = bool(payload.get("output_truncated"))
    raw_artifact = payload.get("artifact")
    try:
        artifact = ToolArtifactRef.from_dict(raw_artifact) if isinstance(raw_artifact, dict) else None
    except ValueError:
        artifact = None
    retention = None
    output = str(payload.get("output") or "")
    head = output[:PREVIEW_HEAD_CHARS]
    tail = output[PREVIEW_HEAD_CHARS + len(PREVIEW_MARKER) :] if truncated else output[PREVIEW_HEAD_CHARS:]
    marker = PREVIEW_MARKER
    if payload.get("artifact_error") or payload.get("output_error") or (truncated and artifact is None):
        retention = {"status": "unavailable", "reason": "artifact_persist_failed"}
        metadata["retention"] = retention
        payload = dict(payload)
        artifact = None
        marker = _UNAVAILABLE_MARKER
        tail = tail[-(BOUNDED_TEXT_CHARS - len(head) - len(marker)) :]
        payload["output"] = head + marker + tail
        payload["output_visible_bytes"] = len(payload["output"].encode("utf-8"))
        payload["retention"] = retention
        for key in ("artifact", "artifact_error", "artifact_error_code", "output_error"):
            payload.pop(key, None)
    if status not in {"running", "stopping", "unknown", "completed", "failed", "timeout", "stopped"}:
        error_code = str(payload.get("error_code") or "background_command_failed")
        return builtin_error(
            str(payload.get("error") or "Background command failed"),
            error_code,
            details=payload,
            metadata=metadata,
        )

    recovery: dict[str, Any] = {}
    if truncated and artifact is not None:
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
        if truncated or retention is not None:
            full_output_json_bytes = int(payload.get("output_json_bytes", 0))
            # JSON escapes and the receipt envelope are part of visible content.
            # The artifact itself retains the unwrapped complete output prefix.
            for _ in range(BOUNDED_TEXT_CHARS + 1):
                output_exceeds_source = truncated and full_output_json_bytes < len(to_json(body["output"]).encode("utf-8"))
                unavailable_handle_exceeds_limit = retention is not None and len(to_json(body)) > BOUNDED_TEXT_CHARS
                if not output_exceeds_source and not unavailable_handle_exceeds_limit:
                    break
                before = len(head) + len(tail)
                if head:
                    head = head[:-1]
                elif tail:
                    tail = tail[1:]
                else:
                    raise AssertionError("capture marker exceeds original output budget")
                assert len(head) + len(tail) < before
                body["output"] = head + marker + tail
                body["output_visible_bytes"] = len(body["output"].encode("utf-8"))
            else:
                raise AssertionError("capture preview did not shrink within its bound")
            content = to_json(body)
            visible_bytes = len(content.encode("utf-8"))
            if recovery:
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
        if retention is not None:
            metadata["retention"] = retention
        if not success:
            metadata["error_code"] = error_code
    if recovery:
        recovery.update(original_bytes=payload["output_original_bytes"], visible_bytes=len(content.encode("utf-8")))
    return ToolExecutionResult(
        tool_call_id="",
        content=content,
        metadata=metadata,
        status_code=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
        error_code=None if success else error_code,
        **recovery,
    )
