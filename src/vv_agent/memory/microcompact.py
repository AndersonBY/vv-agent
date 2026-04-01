from __future__ import annotations

from dataclasses import dataclass, replace

from vv_agent.constants import (
    BASH_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.types import Message

COMPACTABLE_TOOLS: set[str] = {
    READ_FILE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    BASH_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
}

CLEARED_MARKER = "[Old tool result content cleared by microcompact]"


@dataclass(slots=True)
class MicrocompactConfig:
    trigger_ratio: float = 0.75
    keep_recent_cycles: int = 3
    min_result_length: int = 500
    compactable_tools: set[str] | None = None


def microcompact(
    messages: list[Message],
    current_cycle: int,
    config: MicrocompactConfig | None = None,
) -> tuple[list[Message], int]:
    if not messages:
        return messages, 0

    effective_config = config or MicrocompactConfig()
    compactable_tools = effective_config.compactable_tools or COMPACTABLE_TOOLS
    tool_call_names = _build_tool_call_name_map(messages)
    inferred_cycles = _infer_message_cycles(messages)
    max_inferred_cycle = inferred_cycles[-1] if inferred_cycles else 0
    # `current_cycle` is the upcoming cycle index, while inferred cycles track
    # completed assistant turns currently present in the retained window.
    # Clamp against `max_inferred + 1` to preserve the existing keep_recent
    # semantics without over-clearing after earlier compaction has truncated history.
    effective_current_cycle = min(max(int(current_cycle), 0), max_inferred_cycle + 1)
    protected_cycle = max(effective_current_cycle - max(effective_config.keep_recent_cycles, 0), 0)

    updated_messages: list[Message] = []
    cleared_count = 0
    for message, inferred_cycle in zip(messages, inferred_cycles, strict=False):
        if not _should_clear_message(
            message,
            inferred_cycle=inferred_cycle,
            protected_cycle=protected_cycle,
            min_result_length=effective_config.min_result_length,
            compactable_tools=compactable_tools,
            tool_call_names=tool_call_names,
        ):
            updated_messages.append(message)
            continue

        updated_messages.append(_replace_content(message))
        cleared_count += 1
    return updated_messages, cleared_count


def is_microcompacted_tool_content(content: str) -> bool:
    return str(content or "").startswith(CLEARED_MARKER)


def _build_tool_call_name_map(messages: list[Message]) -> dict[str, str]:
    tool_call_names: dict[str, str] = {}
    for message in messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or "").strip()
            function_payload = tool_call.get("function")
            if not tool_call_id or not isinstance(function_payload, dict):
                continue
            tool_name = str(function_payload.get("name") or "").strip()
            if tool_name:
                tool_call_names[tool_call_id] = tool_name
    return tool_call_names


def _infer_message_cycles(messages: list[Message]) -> list[int]:
    """Infer per-message cycle indices from assistant turns in the current message window.

    The runtime appends one assistant message per completed cycle, so the inferred
    cycle increments on each assistant message and the following tool/user messages
    inherit that cycle number until the next assistant appears.

    This is an approximation over the currently retained message slice, not a
    canonical persisted cycle id. Callers should clamp any external `current_cycle`
    against the inferred window size (effectively `max_inferred + 1`, because the
    runtime passes the upcoming cycle index) to avoid over-clearing after prior
    compaction has truncated older turns.
    """
    current_cycle = 0
    inferred_cycles: list[int] = []
    for message in messages:
        if message.role == "assistant":
            current_cycle += 1
        inferred_cycles.append(current_cycle)
    return inferred_cycles


def _should_clear_message(
    message: Message,
    *,
    inferred_cycle: int,
    protected_cycle: int,
    min_result_length: int,
    compactable_tools: set[str],
    tool_call_names: dict[str, str],
) -> bool:
    if message.role != "tool":
        return False
    if inferred_cycle >= protected_cycle:
        return False
    if len(message.content) <= max(min_result_length, 1):
        return False
    if is_microcompacted_tool_content(message.content):
        return False
    tool_name = tool_call_names.get(str(message.tool_call_id or "").strip(), "")
    return bool(tool_name and tool_name in compactable_tools)


def _replace_content(message: Message) -> Message:
    metadata = dict(message.metadata)
    metadata["microcompacted"] = True
    metadata["microcompact_original_chars"] = len(message.content)
    return replace(message, content=CLEARED_MARKER, metadata=metadata)
