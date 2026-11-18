"""Pure text rendering and recognition for compacted tool results."""

from __future__ import annotations

import json

COMPACT_MARKER_OPENING = "<Tool Result Compact>"
COMPACT_MARKER_CLOSING = "</Tool Result Compact>"


def build_compacted_tool_content(
    content: str,
    *,
    artifact_path: str,
    tool_name: str,
    excerpt_head_chars: int = 200,
    excerpt_tail_chars: int = 200,
) -> str:
    content = content_without_recovery_envelope(content)
    head_length = max(excerpt_head_chars, 0)
    tail_length = max(excerpt_tail_chars, 0)
    head = content[:head_length] if head_length else ""
    tail = content[-tail_length:] if tail_length and len(content) > head_length else ""
    excerpt_parts = [head] if head else []
    if tail:
        if head:
            excerpt_parts.append("...<snip>...")
        excerpt_parts.append(tail)
    excerpt = "\n".join(excerpt_parts).strip()
    return (
        f"{COMPACT_MARKER_OPENING}\n"
        f"tool_name: {tool_name}\n"
        f"artifact_path: {artifact_path}\n"
        "retrieval_hint: use read_file on artifact_path if needed\n"
        "excerpt:\n"
        f"{excerpt}\n"
        f"{COMPACT_MARKER_CLOSING}"
    )


def is_microcompacted_tool_content(content: str) -> bool:
    return str(content or "").startswith(COMPACT_MARKER_OPENING)


def has_recovery_envelope(content: str) -> bool:
    _, separator, last_line = str(content or "").rpartition("\n")
    if not separator:
        return False
    try:
        payload = json.loads(last_line)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "vv_agent_recovery" in payload


def content_without_recovery_envelope(content: str) -> str:
    prefix, separator, _last_line = str(content or "").rpartition("\n")
    return prefix if separator and has_recovery_envelope(content) else content
