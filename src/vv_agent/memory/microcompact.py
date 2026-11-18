"""Synchronous public driver for the core planner and shared text helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from vv_agent.kernel._compaction_text import (
    COMPACT_MARKER_CLOSING,
    COMPACT_MARKER_OPENING,
    build_compacted_tool_content,
    content_without_recovery_envelope,
    has_recovery_envelope,
    is_microcompacted_tool_content,
)
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._microcompaction import (
    EXCERPT_METADATA_KEY,
    MicrocompactCandidate,
    MicrocompactPlan,
    replace_with_compacted_marker,
)
from vv_agent.kernel._microcompaction import (
    plan_microcompact as _plan_microcompact,
)
from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy
from vv_agent.kernel._token_count import TokenCount
from vv_agent.kernel._tool_metadata import ToolResultRetention
from vv_agent.memory._counting import _drain_counts


def plan_microcompact(
    messages: list[Message],
    *,
    current_cycle: int,
    current_tokens: int,
    target_tokens: int,
    policy: MicrocompactionPolicy,
    result_retentions: Mapping[str, ToolResultRetention],
    artifact_path_estimate_for: Callable[[str], str],
    estimate_message_tokens: Callable[[Message], int],
    excerpt_head_chars: int = 200,
    excerpt_tail_chars: int = 200,
) -> MicrocompactPlan:
    steps = _plan_microcompact(
        messages,
        current_cycle=current_cycle,
        current_tokens=current_tokens,
        target_tokens=target_tokens,
        policy=policy,
        result_retentions=result_retentions,
        artifact_path_estimate_for=artifact_path_estimate_for,
        excerpt_head_chars=excerpt_head_chars,
        excerpt_tail_chars=excerpt_tail_chars,
    )

    def count(request: TokenCount) -> int:
        # This planner only requests estimates for individual messages.
        assert isinstance(request.messages, Message)
        return estimate_message_tokens(request.messages)

    return _drain_counts(steps, count)


__all__ = [
    "COMPACT_MARKER_CLOSING",
    "COMPACT_MARKER_OPENING",
    "EXCERPT_METADATA_KEY",
    "MicrocompactCandidate",
    "MicrocompactPlan",
    "build_compacted_tool_content",
    "content_without_recovery_envelope",
    "has_recovery_envelope",
    "is_microcompacted_tool_content",
    "plan_microcompact",
    "replace_with_compacted_marker",
]
