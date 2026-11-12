"""Public microcompaction bindings to the pure core planner and text helpers."""

from vv_agent.kernel._compaction_text import (
    COMPACT_MARKER_CLOSING,
    COMPACT_MARKER_OPENING,
    build_compacted_tool_content,
    content_without_recovery_envelope,
    has_recovery_envelope,
    is_microcompacted_tool_content,
)
from vv_agent.kernel._microcompaction import (
    EXCERPT_METADATA_KEY,
    MicrocompactCandidate,
    MicrocompactPlan,
    plan_microcompact,
    replace_with_compacted_marker,
)

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
