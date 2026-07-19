from __future__ import annotations

from dataclasses import replace
from typing import Any

from vv_agent.types import Message


def sanitize_for_resume(messages: list[Message]) -> list[Message]:
    """Remove incomplete or invalid tail/history messages before resuming a session."""
    sanitized = list(messages)
    sanitized = filter_empty_assistant_messages(sanitized)
    sanitized = filter_orphan_tool_results(sanitized)
    return filter_unresolved_tool_uses(sanitized)


def filter_empty_assistant_messages(messages: list[Message]) -> list[Message]:
    """Drop assistant messages that contain no visible text, tool calls, or reasoning."""
    result: list[Message] = []
    for message in messages:
        if message.role != "assistant":
            result.append(message)
            continue
        if _get_text_content(message).strip():
            result.append(message)
            continue
        if _get_tool_calls(message):
            result.append(message)
            continue
        if _has_thinking_content(message):
            result.append(message)
            continue
    return result


def filter_unresolved_tool_uses(messages: list[Message]) -> list[Message]:
    """Keep only tool calls completed by the immediately following tool-result block."""
    return _filter_tool_turns(messages, drop_orphan_results=False)


def filter_orphan_tool_results(messages: list[Message]) -> list[Message]:
    """Drop tool results not paired with the immediately preceding assistant turn."""
    return _filter_tool_turns(messages, drop_unresolved_calls=False)


def _filter_tool_turns(
    messages: list[Message],
    *,
    drop_orphan_results: bool = True,
    drop_unresolved_calls: bool = True,
) -> list[Message]:
    result: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            if not drop_orphan_results:
                result.append(message)
            index += 1
            continue
        tool_calls = _get_tool_calls(message) if message.role == "assistant" else []
        if not tool_calls:
            result.append(message)
            index += 1
            continue

        result_end = index + 1
        while result_end < len(messages) and messages[result_end].role == "tool":
            result_end += 1
        tool_results = messages[index + 1 : result_end]

        call_counts: dict[str, int] = {}
        for tool_call in tool_calls:
            call_id = _get_id(tool_call)
            if call_id is not None:
                call_counts[call_id] = call_counts.get(call_id, 0) + 1
        result_counts: dict[str, int] = {}
        for tool_result in tool_results:
            call_id = _get_tool_call_id(tool_result)
            if call_id is not None:
                result_counts[call_id] = result_counts.get(call_id, 0) + 1

        ordered_calls = [
            (call_id, tool_call)
            for tool_call in tool_calls
            if (call_id := _get_id(tool_call)) is not None
        ]
        ordered_results = [
            (call_id, tool_result)
            for tool_result in tool_results
            if (call_id := _get_tool_call_id(tool_result)) is not None
        ]
        paired_calls: list[dict[str, Any]] = []
        paired_results: list[Message] = []
        for (call_id, tool_call), (result_id, tool_result) in zip(ordered_calls, ordered_results, strict=False):
            if call_id != result_id or call_counts[call_id] != 1 or result_counts[result_id] != 1:
                break
            paired_calls.append(tool_call)
            paired_results.append(tool_result)
        visible_calls = paired_calls if drop_unresolved_calls else tool_calls
        if visible_calls:
            result.append(replace(message, tool_calls=visible_calls))
        if not drop_orphan_results:
            result.extend(tool_results)
        elif paired_calls:
            result.extend(paired_results)
        index = result_end
    return result


def filter_thinking_only_messages(messages: list[Message]) -> list[Message]:
    """Explicitly drop assistant messages that only carry reasoning payloads."""
    result: list[Message] = []
    for message in messages:
        if message.role != "assistant":
            result.append(message)
            continue
        if _has_thinking_content(message) and not _get_text_content(message).strip() and not _get_tool_calls(message):
            continue
        result.append(message)
    return result


def _get_text_content(message: Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return " ".join(parts)
    return str(content or "")


def _get_tool_calls(message: Message) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _get_tool_call_id(message: Message) -> str | None:
    tool_call_id = getattr(message, "tool_call_id", None)
    text = str(tool_call_id or "").strip()
    return text or None


def _get_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        candidate = tool_call.get("id")
        if candidate:
            text = str(candidate).strip()
            if text:
                return text
        function_payload = tool_call.get("function")
        if isinstance(function_payload, dict):
            function_id = function_payload.get("id")
            if function_id:
                text = str(function_id).strip()
                if text:
                    return text
        return None
    candidate = getattr(tool_call, "id", None)
    if candidate:
        text = str(candidate).strip()
        if text:
            return text
    function_payload = getattr(tool_call, "function", None)
    function_id = getattr(function_payload, "id", None)
    if function_id:
        text = str(function_id).strip()
        if text:
            return text
    return None


def _has_thinking_content(message: Message) -> bool:
    reasoning_content = getattr(message, "reasoning_content", None)
    if str(reasoning_content or "").strip():
        return True
    content = message.content
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and str(block.get("type") or "").strip().lower() in {"thinking", "reasoning"} for block in content
    )
