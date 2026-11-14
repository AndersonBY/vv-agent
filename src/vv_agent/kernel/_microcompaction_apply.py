"""Microcompaction application; archive and token observations are requested in order."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from vv_agent.kernel._compaction_text import build_compacted_tool_content
from vv_agent.kernel._context_messages import atomic_start
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._microcompaction import (
    EXCERPT_METADATA_KEY,
    MicrocompactCandidate,
    MicrocompactPlan,
    replace_with_compacted_marker,
)
from vv_agent.kernel._token_count import TokenCount


@dataclass(frozen=True, slots=True)
class MicrocompactionApplyResult:
    messages: list[Message]
    archived_count: int = 0
    reclaimed_tokens: int = 0
    artifact_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class ArchiveToolMessage:
    message: Message
    candidate: MicrocompactCandidate


@dataclass(frozen=True, slots=True)
class ToolMarkerInput:
    message: Message


def compaction_excerpt(message: Message) -> str:
    excerpt = message.metadata.get(EXCERPT_METADATA_KEY)
    return excerpt if isinstance(excerpt, str) else message.content


def apply_microcompaction(
    messages: list[Message],
    *,
    plan: MicrocompactPlan,
    protected_start: int | None,
    model: str,
) -> Generator[ArchiveToolMessage | ToolMarkerInput | TokenCount, Any, MicrocompactionApplyResult]:
    updated_messages = list(messages)
    remaining_tokens = plan.current_tokens
    archived_count = 0
    reclaimed_tokens = 0
    artifact_failure_count = 0
    protected = atomic_start(messages, protected_start) if protected_start is not None else len(messages)
    for candidate in plan.candidates:
        if candidate.message_index >= protected:
            continue
        if remaining_tokens <= plan.target_tokens:
            break
        message = updated_messages[candidate.message_index]
        artifact = yield ArchiveToolMessage(message, candidate)
        if artifact is None:
            artifact_failure_count += 1
            continue
        excerpt, excerpt_head_chars, excerpt_tail_chars = yield ToolMarkerInput(message)
        marker = build_compacted_tool_content(
            excerpt,
            artifact_path=artifact.path,
            tool_name=candidate.tool_name,
            excerpt_head_chars=excerpt_head_chars,
            excerpt_tail_chars=excerpt_tail_chars,
        )
        replacement = replace_with_compacted_marker(
            message,
            candidate,
            artifact=artifact,
            marker=marker,
        )
        actual_reclaimed = max(
            (yield TokenCount(message, model)) - (yield TokenCount(replacement, model)),
            0,
        )
        if actual_reclaimed == 0:
            continue
        updated_messages[candidate.message_index] = replacement
        archived_count += 1
        reclaimed_tokens += actual_reclaimed
        remaining_tokens = max(remaining_tokens - actual_reclaimed, 0)
    return MicrocompactionApplyResult(
        messages=updated_messages,
        archived_count=archived_count,
        reclaimed_tokens=reclaimed_tokens,
        artifact_failure_count=artifact_failure_count,
    )
