from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.types import Message, ToolArtifactRef

COMPACT_MARKER_OPENING = "<Tool Result Compact>"
COMPACT_MARKER_CLOSING = "</Tool Result Compact>"
EXCERPT_METADATA_KEY = "_vv_agent_microcompact_excerpt"


@dataclass(frozen=True, slots=True)
class MicrocompactCandidate:
    message_index: int
    tool_name: str
    tool_call_id: str
    existing_artifact: ToolArtifactRef | None
    estimated_reclaimable_tokens: int


@dataclass(frozen=True, slots=True)
class MicrocompactPlan:
    candidates: tuple[MicrocompactCandidate, ...] = ()
    current_tokens: int = 0
    target_tokens: int = 0

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def estimated_reclaimable_tokens(self) -> int:
        return sum(candidate.estimated_reclaimable_tokens for candidate in self.candidates)


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
) -> MicrocompactPlan:
    if not messages or current_tokens <= target_tokens:
        return MicrocompactPlan(current_tokens=max(current_tokens, 0), target_tokens=max(target_tokens, 0))

    tool_call_names = _build_tool_call_name_map(messages)
    inferred_cycles = _infer_message_cycles(messages)
    max_inferred_cycle = inferred_cycles[-1] if inferred_cycles else 0
    effective_current_cycle = max(max(int(current_cycle), 0), max_inferred_cycle + 1)
    protected_cycle = max(effective_current_cycle - policy.keep_recent_cycles, 0)
    candidates: list[MicrocompactCandidate] = []

    for index, (message, inferred_cycle) in enumerate(zip(messages, inferred_cycles, strict=False)):
        tool_name = _candidate_tool_name(
            message,
            inferred_cycle=inferred_cycle,
            protected_cycle=protected_cycle,
            min_result_chars=policy.min_result_chars,
            tool_call_names=tool_call_names,
            result_retentions=result_retentions,
        )
        if tool_name is None:
            continue
        tool_call_id = str(message.tool_call_id or "").strip()
        existing_artifact = _existing_artifact(message)
        artifact_path = existing_artifact.path if existing_artifact is not None else artifact_path_estimate_for(tool_call_id)
        excerpt_value = message.metadata.get(EXCERPT_METADATA_KEY)
        excerpt_source = excerpt_value if isinstance(excerpt_value, str) else message.content
        marker = build_compacted_tool_content(
            excerpt_source,
            artifact_path=artifact_path,
            tool_name=tool_name,
        )
        marker_message = replace(message, content=marker)
        reclaimable = max(estimate_message_tokens(message) - estimate_message_tokens(marker_message), 0)
        if reclaimable == 0:
            continue
        candidates.append(
            MicrocompactCandidate(
                message_index=index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                existing_artifact=existing_artifact,
                estimated_reclaimable_tokens=reclaimable,
            )
        )

    return MicrocompactPlan(
        candidates=tuple(candidates),
        current_tokens=max(current_tokens, 0),
        target_tokens=max(target_tokens, 0),
    )


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


def replace_with_compacted_marker(
    message: Message,
    candidate: MicrocompactCandidate,
    *,
    artifact: ToolArtifactRef,
    marker: str,
) -> Message:
    return _replace_content(
        message,
        marker,
        artifact=artifact,
        tool_name=candidate.tool_name,
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


def _candidate_tool_name(
    message: Message,
    *,
    inferred_cycle: int,
    protected_cycle: int,
    min_result_chars: int,
    tool_call_names: dict[str, str],
    result_retentions: Mapping[str, ToolResultRetention],
) -> str | None:
    if message.role != "tool" or inferred_cycle >= protected_cycle:
        return None
    if len(message.content) <= min_result_chars or is_microcompacted_tool_content(message.content):
        return None
    if _existing_artifact(message) is None and has_recovery_envelope(message.content):
        return None
    tool_name = tool_call_names.get(str(message.tool_call_id or "").strip())
    if tool_name is None:
        return None
    if result_retentions.get(tool_name, ToolResultRetention.ARCHIVE) == ToolResultRetention.PRESERVE:
        return None
    return tool_name


def _replace_content(
    message: Message,
    content: str,
    *,
    artifact: ToolArtifactRef,
    tool_name: str,
) -> Message:
    del tool_name
    metadata = dict(message.metadata)
    metadata.pop(EXCERPT_METADATA_KEY, None)
    return replace(message, content=content, metadata=metadata, artifact_ref=artifact)


def _existing_artifact(message: Message) -> ToolArtifactRef | None:
    return message.artifact_ref if isinstance(message.artifact_ref, ToolArtifactRef) else None


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
    current_cycle = 0
    inferred_cycles: list[int] = []
    for message in messages:
        if message.role == "assistant":
            current_cycle += 1
        inferred_cycles.append(current_cycle)
    return inferred_cycles


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
