from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}
SYSTEM_PROMPT_SECTIONS_KEY = "system_prompt_sections"
PROMPT_CACHE_ENABLED_KEY = "anthropic_prompt_cache_enabled"
_MAX_BREAKPOINTS = 4
_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}
_HISTORY_PRIORITY = {"tool_result": 0, "text": 1}


def apply_claude_prompt_cache(
    *,
    endpoint_type: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    extra_body: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    normalized_endpoint = str(endpoint_type or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    request_metadata = metadata if isinstance(metadata, dict) else {}

    if normalized_endpoint not in {"anthropic", "anthropic_vertex"}:
        return messages, tools, extra_body
    if not normalized_model.startswith("claude"):
        return messages, tools, extra_body
    if request_metadata.get(PROMPT_CACHE_ENABLED_KEY, True) is False:
        return messages, tools, extra_body

    planned_messages = deepcopy(messages)
    planned_tools = deepcopy(tools)
    planned_extra_body = deepcopy(extra_body) if isinstance(extra_body, dict) else None

    token_threshold = _minimum_cacheable_tokens(normalized_model)
    breakpoint_budget = _MAX_BREAKPOINTS

    budget_ref = [breakpoint_budget]
    system_char_count = _apply_system_cache_breakpoint(
        messages=planned_messages,
        metadata=request_metadata,
        token_threshold=token_threshold,
        breakpoint_budget_ref=budget_ref,
    )
    breakpoint_budget = max(0, budget_ref[0])

    budget_ref = [breakpoint_budget]
    tool_char_count = _apply_tool_cache_breakpoint(
        tools=planned_tools,
        prefix_char_count=system_char_count,
        token_threshold=token_threshold,
        breakpoint_budget_ref=budget_ref,
    )
    breakpoint_budget = max(0, budget_ref[0])

    history_prefix_char_count = system_char_count + tool_char_count
    _apply_history_cache_breakpoint(
        messages=planned_messages,
        prefix_char_count=history_prefix_char_count,
        token_threshold=token_threshold,
        breakpoint_budget=breakpoint_budget,
    )

    return planned_messages, planned_tools, planned_extra_body


def _apply_system_cache_breakpoint(
    *,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    token_threshold: int,
    breakpoint_budget_ref: list[int],
) -> int:
    if not messages or breakpoint_budget_ref[0] <= 0:
        return 0

    system_index = next((index for index, message in enumerate(messages) if message.get("role") == "system"), None)
    if system_index is None:
        return 0

    system_message = messages[system_index]
    sections = _normalize_system_prompt_sections(metadata.get(SYSTEM_PROMPT_SECTIONS_KEY))
    if sections:
        blocks = [{"type": "text", "text": section["text"]} for section in sections]
    else:
        blocks = _ensure_content_blocks(system_message)

    if not blocks:
        return 0

    system_message["content"] = blocks
    prefix_char_count = sum(_estimate_block_chars(block) for block in blocks)
    if _estimate_tokens(prefix_char_count) < token_threshold:
        return prefix_char_count

    stable_indexes = (
        [index for index, section in enumerate(sections) if section.get("stable", True)]
        if sections
        else list(range(len(blocks)))
    )
    if not stable_indexes:
        return prefix_char_count

    blocks[stable_indexes[-1]]["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)
    breakpoint_budget_ref[0] -= 1
    return prefix_char_count


def _apply_tool_cache_breakpoint(
    *,
    tools: list[dict[str, Any]],
    prefix_char_count: int,
    token_threshold: int,
    breakpoint_budget_ref: list[int],
) -> int:
    if not tools or breakpoint_budget_ref[0] <= 0:
        return 0

    tool_char_count = sum(_estimate_tool_chars(tool) for tool in tools)
    if _estimate_tokens(prefix_char_count + tool_char_count) < token_threshold:
        return tool_char_count

    tools[-1]["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)
    breakpoint_budget_ref[0] -= 1
    return tool_char_count


def _apply_history_cache_breakpoint(
    *,
    messages: list[dict[str, Any]],
    prefix_char_count: int,
    token_threshold: int,
    breakpoint_budget: int,
) -> None:
    if breakpoint_budget <= 0:
        return

    candidate = _find_history_breakpoint(messages)
    if candidate is None:
        return

    message_index, block_index = candidate
    history_char_count = prefix_char_count
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            continue
        blocks = _ensure_content_blocks(message)
        if index < message_index:
            history_char_count += sum(_estimate_block_chars(block) for block in blocks)
            continue
        history_char_count += sum(_estimate_block_chars(block) for block in blocks[: block_index + 1])
        break

    if _estimate_tokens(history_char_count) < token_threshold:
        return

    target_blocks = _ensure_content_blocks(messages[message_index])
    if block_index < len(target_blocks):
        target_blocks[block_index]["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)


def _find_history_breakpoint(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    best: tuple[int, int, int] | None = None
    fallback: tuple[int, int] | None = None

    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if message.get("role") == "system":
            continue
        blocks = _ensure_content_blocks(message)
        for block_index in range(len(blocks) - 1, -1, -1):
            block = blocks[block_index]
            block_type = str(block.get("type") or "text").strip().lower()
            if block_type in _THINKING_BLOCK_TYPES:
                continue
            if block.get("cache_control") is not None:
                continue
            if _estimate_block_chars(block) <= 0:
                continue
            if block_type in _HISTORY_PRIORITY:
                priority = _HISTORY_PRIORITY[block_type]
                if best is None or priority < best[2]:
                    best = (message_index, block_index, priority)
                break
            if fallback is None:
                fallback = (message_index, block_index)
        if best is not None:
            break

    if best is not None:
        return best[0], best[1]
    return fallback


def _ensure_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        changed = False
        for item in content:
            if isinstance(item, dict):
                blocks.append(item)
            elif isinstance(item, str):
                blocks.append({"type": "text", "text": item})
                changed = True
        if changed:
            message["content"] = blocks
        return blocks
    if isinstance(content, str):
        block = {"type": "text", "text": content}
        message["content"] = [block]
        return [block]
    return []


def _normalize_system_prompt_sections(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    sections: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        sections.append(
            {
                "id": str(item.get("id") or "").strip(),
                "text": text,
                "stable": bool(item.get("stable", True)),
            }
        )
    return sections


def _estimate_tokens(char_count: int) -> int:
    if char_count <= 0:
        return 0
    return (char_count + 3) // 4


def _estimate_tool_chars(tool: dict[str, Any]) -> int:
    return len(json.dumps(tool, ensure_ascii=False, sort_keys=True))


def _estimate_block_chars(block: dict[str, Any]) -> int:
    block_type = str(block.get("type") or "text").strip().lower()
    if block_type == "text":
        return len(str(block.get("text") or ""))
    if block_type == "tool_result":
        return len(json.dumps(block.get("content"), ensure_ascii=False))
    if block_type == "tool_use":
        return len(str(block.get("name") or "")) + len(json.dumps(block.get("input"), ensure_ascii=False))
    if block_type in _THINKING_BLOCK_TYPES:
        return 0
    return len(json.dumps(block, ensure_ascii=False))


def _minimum_cacheable_tokens(model: str) -> int:
    normalized_model = str(model or "").strip().lower()
    if "opus-4-6" in normalized_model or "opus-4-5" in normalized_model:
        return 4096
    if "haiku-4-5" in normalized_model:
        return 4096
    if "haiku" in normalized_model:
        return 2048
    return 1024
