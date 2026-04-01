from __future__ import annotations

from dataclasses import replace
from typing import Any

from vv_agent.types import Message


def sanitize_for_resume(messages: list[Message]) -> list[Message]:
    """Remove incomplete or invalid tail/history messages before resuming a session."""
    sanitized = list(messages)
    sanitized = filter_empty_assistant_messages(sanitized)
    sanitized = filter_thinking_only_messages(sanitized)
    sanitized = filter_orphan_tool_results(sanitized)
    sanitized = filter_unresolved_tool_uses(sanitized)
    return sanitized


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
    """Remove unresolved tail tool calls, keeping earlier history intact."""
    if not messages:
        return []

    result_ids: set[str] = set()
    for message in messages:
        if message.role != "tool":
            continue
        call_id = _get_tool_call_id(message)
        if call_id:
            result_ids.add(call_id)

    result = list(messages)
    while result:
        tail_index = len(result) - 1
        while tail_index >= 0 and result[tail_index].role == "tool":
            tail_index -= 1
        if tail_index < 0:
            break
        last_message = result[tail_index]
        if last_message.role != "assistant":
            break

        tool_calls = _get_tool_calls(last_message)
        if not tool_calls:
            break

        unresolved_ids = [call_id for call_id in (_get_id(item) for item in tool_calls) if call_id not in result_ids]
        if not unresolved_ids:
            break

        if len(unresolved_ids) == len(tool_calls):
            result.pop(tail_index)
            continue

        remaining_tool_calls = [item for item in tool_calls if _get_id(item) not in set(unresolved_ids)]
        result[tail_index] = replace(last_message, tool_calls=remaining_tool_calls or None)
        break

    return result


def filter_orphan_tool_results(messages: list[Message]) -> list[Message]:
    """Drop tool results whose tool call no longer exists in history."""
    call_ids: set[str] = set()
    for message in messages:
        if message.role != "assistant":
            continue
        for tool_call in _get_tool_calls(message):
            call_id = _get_id(tool_call)
            if call_id:
                call_ids.add(call_id)

    result: list[Message] = []
    for message in messages:
        if message.role != "tool":
            result.append(message)
            continue
        call_id = _get_tool_call_id(message)
        if call_id and call_id not in call_ids:
            continue
        result.append(message)
    return result


def filter_thinking_only_messages(messages: list[Message]) -> list[Message]:
    """Drop assistant messages that only carry reasoning/thinking payloads."""
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
            return str(candidate)
        function_payload = tool_call.get("function")
        if isinstance(function_payload, dict):
            function_id = function_payload.get("id")
            if function_id:
                return str(function_id)
        return None
    candidate = getattr(tool_call, "id", None)
    if candidate:
        return str(candidate)
    function_payload = getattr(tool_call, "function", None)
    function_id = getattr(function_payload, "id", None)
    if function_id:
        return str(function_id)
    return None


def _has_thinking_content(message: Message) -> bool:
    reasoning_content = getattr(message, "reasoning_content", None)
    if str(reasoning_content or "").strip():
        return True
    content = message.content
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and str(block.get("type") or "").strip().lower() in {"thinking", "reasoning"}
        for block in content
    )
