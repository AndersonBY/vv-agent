"""Pure tool completion policy, image notifications and skipped results."""

from __future__ import annotations

import json

from vv_agent.kernel._execution_values import AgentTask, CompletionReason
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._tool_values import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus


def apply_tool_use_behavior(
    *,
    task: AgentTask,
    call: ToolCall,
    result: ToolExecutionResult,
) -> CompletionReason | None:
    if result.directive != ToolDirective.CONTINUE or result.status_code != ToolResultStatus.SUCCESS:
        return None
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    behavior = str(metadata.get("_vv_agent_tool_use_behavior") or "run_llm_again")
    should_stop = behavior == "stop_on_first_tool"
    if behavior == "stop_at_tool_names":
        raw_names = metadata.get("_vv_agent_stop_at_tool_names")
        stop_names = {str(name) for name in raw_names} if isinstance(raw_names, list) else set()
        should_stop = call.name in stop_names
    if should_stop:
        result.directive = ToolDirective.FINISH
        return CompletionReason.STOP_ON_FIRST_TOOL if behavior == "stop_on_first_tool" else CompletionReason.STOP_AT_TOOL_NAME
    return None


def build_image_notification(*, result: ToolExecutionResult, include_image: bool) -> Message | None:
    if not include_image:
        return None
    if result.image_url:
        # For data URLs keep text empty to avoid duplicating base64 payload as plain text.
        content = f"[Image loaded] {result.image_path}" if result.image_path else ""
        return Message(
            role="user",
            content=content,
            image_url=result.image_url,
        )
    if result.image_path:
        return Message(role="user", content=f"[Image loaded] {result.image_path}")
    return None


def build_skipped_result(
    call: ToolCall,
    *,
    error_code: str,
    message: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id=call.id,
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=json.dumps(
            {
                "ok": False,
                "error": message,
                "error_code": error_code,
            },
            ensure_ascii=False,
        ),
    )
