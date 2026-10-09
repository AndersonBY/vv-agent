from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vv_agent.memory.session_memory import SessionMemoryConfig
from vv_agent.memory.token_utils import resolve_model_token_limits
from vv_agent.model_settings import ModelSettings
from vv_agent.types import AgentTask
from vv_agent.workspace.local import LocalWorkspaceBackend

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.tools.registry import ToolRegistry

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.memory.message_sanitizer import filter_empty_assistant_messages
from vv_agent.memory.microcompact import (
    EXCERPT_METADATA_KEY,
    MicrocompactCandidate,
    MicrocompactPlan,
    build_compacted_tool_content,
    has_recovery_envelope,
    plan_microcompact,
    replace_with_compacted_marker,
)
from vv_agent.memory.session_memory import SessionMemory
from vv_agent.memory.token_utils import compute_compaction_threshold, count_messages_tokens
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.types import COMPACTION_METADATA_KEY, Message, ToolArtifactRef, ToolExecutionResult, validate_compaction_metadata
from vv_agent.workspace.artifacts import persist_text_artifact
from vv_agent.workspace.base import WorkspaceBackend

_MEMORY_SUMMARY_NAME = "memory_summary"
_ORIGINAL_USER_REQUEST_PATTERN = re.compile(r"<Original User Request>\s*(.*?)\s*</Original User Request>", re.DOTALL)
_COMPRESSED_AGENT_MEMORY_PATTERN = re.compile(
    r"<Compressed Agent Memory>\s*([\s\S]*?)\s*</Compressed Agent Memory>",
    re.IGNORECASE,
)
_ANALYSIS_BLOCK_PATTERN = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_SUMMARY_BLOCK_PATTERN = re.compile(r"<summary>\s*([\s\S]*?)\s*</summary>", re.IGNORECASE)

_MEMORY_WARNING_PROMPTS = {
    "zh-CN": (
        "当前记忆已使用容量超过 {memory_threshold_percentage}%,"
        "建议立即整理、记录对话中的关键信息、资料, 并储存至工作区, 避免记忆压缩后资料丢失。\n\n"
    ),
    "en-US": (
        "The current memory usage has exceeded {memory_threshold_percentage}%. "
        "It is recommended to immediately organize and record key information and materials "
        "from the conversation, and store them in the workspace to prevent data loss after "
        "memory compression.\n\n"
    ),
}

_COMPRESS_MEMORY_PROMPTS = {
    "zh-CN": """你正在总结一段用户与 AI 编程助手的对话。
请先在 <analysis> 标签中进行思考 (该部分后续会被剥离), 然后输出结构化 JSON 摘要。

<analysis>
请逐步思考: 哪些信息必须保留, 哪些用户原话不能丢, 哪些文件/错误/当前状态会影响后续继续执行。
</analysis>

<Previous Summary>
{previous_summary_jcs}
</Previous Summary>

<Conversation Prefix>
{conversation_prefix_jcs}
</Conversation Prefix>

请将以上对话压缩为结构化 JSON「Task Status Summary」, 让 Agent 能快速恢复任务, 并保留用户约束、关键决策、文件操作与当前工作状态。

要求:
- 只输出 JSON, 不要 Markdown。
- 字段内容简洁、可检索, 短句表达。
- 没有信息的字段使用 [] 或 ""。
- `original_user_messages` 字段至关重要: 尽量保留用户原话, 不要做概括式改写。

JSON Schema:
{{
  "summary_version": "2.0",
  "original_user_messages": ["..."],
  "user_constraints": ["..."],
  "decisions": ["..."],
  "files_examined_or_modified": [
    {{"path": "...", "action": "read|created|modified|deleted", "summary": "..."}}
  ],
  "errors_and_fixes": [
    {{"error": "...", "fix": "...", "file": "..."}}
  ],
  "progress": ["最多保留 {event_limit} 条关键进展"],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "current_work_state": "...",
  "next_steps": ["..."]
}}
""",
    "en-US": """You are summarizing a conversation between a user and an AI coding assistant.
Provide your analysis in <analysis> tags first (this section will be stripped), then output a structured JSON summary.

<analysis>
Think step by step about what information is critical to preserve, especially the user's exact wording,
the current work state, file operations, and any errors that were resolved.
</analysis>

<Previous Summary>
{previous_summary_jcs}
</Previous Summary>

<Conversation Prefix>
{conversation_prefix_jcs}
</Conversation Prefix>

Please compress the conversation into a structured JSON "Task Status Summary".
This summary should allow the Agent to quickly resume the task
while preserving user constraints, key decisions, file operations, and critical context.

Requirements:
- Output JSON only, no Markdown.
- Keep fields concise and searchable; use short sentences.
- If a field has no data, use [] or "" as appropriate.
- The "original_user_messages" field is critical. Preserve user messages verbatim or near-verbatim.

JSON Schema:
{{
  "summary_version": "2.0",
  "original_user_messages": ["..."],
  "user_constraints": ["..."],
  "decisions": ["..."],
  "files_examined_or_modified": [
    {{"path": "...", "action": "read|created|modified|deleted", "summary": "..."}}
  ],
  "errors_and_fixes": [
    {{"error": "...", "fix": "...", "file": "..."}}
  ],
  "progress": ["Preserve up to {event_limit} critical events"],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "current_work_state": "...",
  "next_steps": ["..."]
}}
""",
}


SummaryCallback = Callable[[str, str | None, str | None], str | None]
CompactionMode = Literal["none", "micro", "structural", "summary", "emergency"]
_COMPACTION_MODE_RANK: dict[CompactionMode, int] = {
    "none": 0,
    "micro": 1,
    "structural": 2,
    "summary": 3,
    "emergency": 4,
}


def _strongest_compaction_mode(*modes: CompactionMode) -> CompactionMode:
    return max(modes, key=_COMPACTION_MODE_RANK.__getitem__, default="none")


ReservedOutputSource = Literal[
    "model_settings",
    "task_metadata",
    "framework_fallback",
    "framework_fallback_capped_by_model_capability",
]


@dataclass(frozen=True, slots=True)
class MemoryCompactionResult:
    messages: list[Message]
    mode: CompactionMode
    changed: bool
    archived_count: int = 0
    reclaimed_tokens: int = 0
    artifact_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class MicrocompactionApplyResult:
    messages: list[Message]
    archived_count: int = 0
    reclaimed_tokens: int = 0
    artifact_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class SummaryPlan:
    """Exact history and prompt for a separately persisted summary model operation."""

    messages: list[Message]
    systems: list[Message]
    previous: list[Message]
    prefix: list[Message]
    tail: list[Message]
    prompt: str
    keep_recent: int


@dataclass(slots=True)
class MemoryManager:
    compact_threshold: int = 250_000
    keep_recent_messages: int = 10
    model: str = ""
    model_context_window: int = 279_000
    model_max_output_tokens: int | None = None
    reserved_output_tokens: int = 16_000
    reserved_output_source: ReservedOutputSource = "framework_fallback"
    autocompact_buffer_tokens: int = 13_000
    language: str = "zh-CN"
    warning_threshold_percentage: int = 90
    include_memory_warning: bool = False
    tool_result_excerpt_head: int = 200
    tool_result_excerpt_tail: int = 200
    microcompaction_policy: MicrocompactionPolicy = field(default_factory=MicrocompactionPolicy)
    tool_result_retentions: dict[str, ToolResultRetention] = field(default_factory=dict)
    workspace_backend: WorkspaceBackend | None = None
    recovery_tool_available: bool = False
    artifact_scope: str = "run"
    summary_event_limit: int = 40
    summary_backend: str | None = None
    summary_model: str | None = None
    summary_callback: SummaryCallback | None = None
    base_system_prompt: str = ""
    session_memory: SessionMemory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.microcompaction_policy, MicrocompactionPolicy):
            self.microcompaction_policy = MicrocompactionPolicy.from_dict(self.microcompaction_policy)

    @property
    def effective_context_window(self) -> int:
        """Token budget available to prompt messages after reserving model output."""
        return max(self.model_context_window - self.reserved_output_tokens, 0)

    @property
    def autocompact_threshold(self) -> int:
        """Prompt token threshold that triggers automatic compaction."""
        return compute_compaction_threshold(
            configured_threshold=self.compact_threshold,
            model_context_window=self.model_context_window,
            reserved_output_tokens=self.reserved_output_tokens,
            autocompact_buffer_tokens=self.autocompact_buffer_tokens,
        )

    @property
    def warning_threshold(self) -> int:
        """Token threshold that emits a memory warning before compaction."""
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.warning_threshold_percentage / 100)

    @property
    def microcompact_trigger_threshold(self) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.microcompaction_policy.trigger_ratio)

    @property
    def microcompact_target_threshold(self) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        return int(threshold * self.microcompaction_policy.target_ratio)

    def compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        force: bool = False,
    ) -> tuple[list[Message], bool]:
        result = self.compact_with_result(
            messages,
            cycle_index=cycle_index,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
            force=force,
        )
        return result.messages, result.changed

    def compact_with_result(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        force: bool = False,
        microcompact_plan: MicrocompactPlan | None = None,
        microcompact_planned: bool = False,
    ) -> MemoryCompactionResult:
        original_messages = list(messages)
        strongest_mode: CompactionMode = "none"
        if not messages:
            return MemoryCompactionResult(messages=messages, mode="none", changed=False)

        # Apply a precomputed plan to exactly the view it was built against.
        working_messages = list(messages)
        if self._summary_parts(working_messages, self.keep_recent_messages) is None:
            return MemoryCompactionResult(messages=working_messages, mode="none", changed=False)

        message_length = self.effective_tokens(
            working_messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        self._maybe_extract_session_memory(
            working_messages,
            cycle_index=cycle_index,
            current_tokens=message_length,
        )
        microcompacted_messages = working_messages
        microcompact_result = MicrocompactionApplyResult(messages=working_messages)
        if not force:
            effective_plan = microcompact_plan
            if not microcompact_planned:
                effective_plan = self.plan_microcompaction(
                    working_messages,
                    cycle_index=cycle_index,
                    current_tokens=message_length,
                )
            if effective_plan is not None and effective_plan.candidates:
                microcompact_result = self.apply_microcompaction(
                    working_messages,
                    plan=effective_plan,
                )
                microcompacted_messages = microcompact_result.messages
            if microcompact_result.archived_count > 0:
                strongest_mode = _strongest_compaction_mode(strongest_mode, "micro")
                message_length = max(
                    message_length - microcompact_result.reclaimed_tokens,
                    0,
                )

        sanitized_messages = filter_empty_assistant_messages(microcompacted_messages)
        if sanitized_messages != microcompacted_messages:
            strongest_mode = "structural"
            microcompacted_messages = sanitized_messages

        if not force and message_length <= self.autocompact_threshold:
            maybe_warned_messages, warning_inserted = self._maybe_append_memory_warning(
                microcompacted_messages,
                message_length=message_length,
            )
            if warning_inserted:
                strongest_mode = _strongest_compaction_mode(strongest_mode, "structural")
            return self._compaction_result(
                original_messages,
                maybe_warned_messages,
                strongest_mode,
                archived_count=microcompact_result.archived_count,
                reclaimed_tokens=microcompact_result.reclaimed_tokens,
                artifact_failure_count=microcompact_result.artifact_failure_count,
            )

        summarized_messages, summarized = self.compress_memory(microcompacted_messages, cycle_index=cycle_index)
        if summarized:
            strongest_mode = _strongest_compaction_mode(strongest_mode, "summary")
        return self._compaction_result(
            original_messages,
            summarized_messages,
            strongest_mode,
            archived_count=microcompact_result.archived_count,
            reclaimed_tokens=microcompact_result.reclaimed_tokens,
            artifact_failure_count=microcompact_result.artifact_failure_count,
        )

    @staticmethod
    def _compaction_result(
        before_messages: list[Message],
        after_messages: list[Message],
        mode: CompactionMode,
        *,
        archived_count: int = 0,
        reclaimed_tokens: int = 0,
        artifact_failure_count: int = 0,
    ) -> MemoryCompactionResult:
        content_changed = before_messages != after_messages
        if not content_changed:
            mode = "none"
        elif mode == "none":
            mode = "structural"
        return MemoryCompactionResult(
            messages=after_messages,
            mode=mode,
            changed=content_changed,
            archived_count=archived_count,
            reclaimed_tokens=reclaimed_tokens,
            artifact_failure_count=artifact_failure_count,
        )

    def emergency_compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        drop_ratio: float = 0.2,
    ) -> list[Message]:
        return self._summarize_prefix(messages, keep_recent=self.keep_recent_messages, drop_ratio=drop_ratio)[0]

    def compress_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], bool]:
        return self._summarize_prefix(messages, keep_recent=self.keep_recent_messages)

    @staticmethod
    def _summary_parts(
        messages: list[Message],
        keep_recent: int,
    ) -> tuple[list[Message], list[Message], list[Message], list[Message]] | None:
        # Validate each original contiguous block; never repair or match ids globally.
        blocks: list[tuple[int, int]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "tool":
                return None
            end = index + 1
            if message.tool_calls:
                if message.role != "assistant":
                    return None
                ids = [call.get("id") for call in message.tool_calls]
                if any(not isinstance(call_id, str) or not call_id for call_id in ids) or len(set(ids)) != len(ids):
                    return None
                end += len(ids)
                results = messages[index + 1 : end]
                # A short slice is only possible at EOF: retain that unfinished
                # block in the tail. A non-tool interruption inside a batch or
                # a mismatched result is still invalid.
                if any(
                    result.role != "tool" or result.tool_call_id != call_id for result, call_id in zip(results, ids, strict=False)
                ):
                    return None
            blocks.append((index, end))
            index = end
        retained_ids = {id(message) for message in filter_empty_assistant_messages(messages)}
        raw_indices = [
            i for i, m in enumerate(messages) if id(m) in retained_ids and m.role != "system" and m.name != _MEMORY_SUMMARY_NAME
        ]
        tail_count = max(keep_recent, 1)
        cut = raw_indices[-tail_count] if len(raw_indices) > tail_count else (raw_indices[0] if raw_indices else 0)
        for start, end in blocks:
            if start < cut < end:
                cut = start
        systems = [m for m in messages if m.role == "system" and m.name != _MEMORY_SUMMARY_NAME]
        previous = [m for m in messages if m.name == _MEMORY_SUMMARY_NAME]
        prefix = [messages[i] for i in raw_indices if i < cut]
        tail = [messages[i] for i in raw_indices if i >= cut]
        return systems, previous, prefix, tail

    def plan_summary(
        self, messages: list[Message], *, keep_recent: int | None = None, drop_ratio: float = 0
    ) -> SummaryPlan | None:
        """Plan without inference, pruning, or callbacks; preserve complete block boundaries."""
        keep = self.keep_recent_messages if keep_recent is None else keep_recent
        keep = max(1, int(keep * (1 - min(max(drop_ratio, 0.0), 0.95))))
        parts = self._summary_parts(messages, keep)
        if parts is None or not parts[2]:
            return None
        systems, previous, prefix, tail = parts
        return SummaryPlan(
            list(messages), systems, previous, prefix, tail, self._build_compress_memory_prompt(prefix, previous=previous), keep
        )

    def _summarize_prefix(
        self, messages: list[Message], *, keep_recent: int, drop_ratio: float = 0
    ) -> tuple[list[Message], bool]:
        if self.summary_callback is None:
            return list(messages), False
        plan = self.plan_summary(messages, keep_recent=keep_recent, drop_ratio=drop_ratio)
        if plan is None:
            return list(messages), False
        return self.accept_summary(plan, self._generate_summary(plan.prompt))

    def accept_summary(self, plan: SummaryPlan, text: str, *, notify: bool = True) -> tuple[list[Message], bool]:
        """Validate a retained model result. Log owners defer notification until commit."""
        messages, systems, previous, prefix, tail = plan.messages, plan.systems, plan.previous, plan.prefix, plan.tail
        summary = self._normalize_summary_payload(self._parse_summary_payload(self._normalize_summary_output(text)))
        if not any(value for key, value in summary.items() if key not in {"summary_version", "user_constraints"}):
            return list(messages), False
        try:
            evidence = self._collect_evidence(previous, prefix)
        except (TypeError, ValueError):
            return list(messages), False
        if (evidence["artifacts"] or evidence["cursors"]) and not self.recovery_tool_available:
            return list(messages), False
        paths = summary["files_examined_or_modified"]
        seen = {item["path"] for item in paths}
        prior_paths = []
        for message in previous:
            match = _COMPRESSED_AGENT_MEMORY_PATTERN.search(message.content)
            if match:
                prior_paths.extend(
                    self._normalize_summary_payload(self._parse_summary_payload(match.group(1)))["files_examined_or_modified"]
                )
        for item in [*prior_paths, *self._collect_file_actions(prefix, first_path_wins=True)]:
            if item["path"] not in seen:
                paths.append(item)
                seen.add(item["path"])
        originals = "\n\n".join(summary["original_user_messages"])
        content = (
            f"<Original User Request>\n{originals}\n</Original User Request>\n\n"
            f"<Compressed Agent Memory>\n{canonical_json_bytes(summary).decode()}\n</Compressed Agent Memory>\n\n"
            + self._render_evidence(evidence)
        )
        candidate = [
            *systems,
            Message(role="user", name=_MEMORY_SUMMARY_NAME, content=content, metadata={COMPACTION_METADATA_KEY: evidence}),
            *tail,
        ]
        tokens = self._calculate_message_length(candidate)
        if tokens >= self._calculate_message_length(messages) or tokens > self.effective_context_window:
            return list(messages), False
        if notify and self.session_memory is not None:
            self.session_memory.on_compaction(current_tokens=tokens)
        return candidate, True

    @staticmethod
    def _normalize_summary_payload(payload: dict[str, object]) -> dict[str, Any]:
        normalized: dict[str, Any] = {"summary_version": "2.0"}
        for key in (
            "original_user_messages",
            "user_constraints",
            "decisions",
            "progress",
            "key_facts",
            "open_issues",
            "next_steps",
        ):
            value = payload.get(key)
            normalized[key] = list(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else []
        state = payload.get("current_work_state")
        normalized["current_work_state"] = state if isinstance(state, str) else ""
        for key, fields in (
            ("files_examined_or_modified", ("path", "action", "summary")),
            ("errors_and_fixes", ("error", "fix", "file")),
        ):
            records = payload.get(key)
            normalized[key] = []
            for item in records if isinstance(records, list) else []:
                if not isinstance(item, dict):
                    continue
                record = cast(dict[str, object], item)
                if key == "files_examined_or_modified":
                    path = record.get("path")
                    if (
                        not isinstance(path, str)
                        or not path.strip()
                        or record.get("action") not in ("read", "created", "modified", "deleted")
                    ):
                        continue
                elif not isinstance(record.get("error"), str):
                    continue
                normalized[key].append({field: value if isinstance(value := record.get(field), str) else "" for field in fields})
        return normalized

    def compaction_evidence(self, messages: list[Message]) -> dict[str, list[dict[str, Any]]]:
        """Collect validated recovery references without reading files or dropping evidence."""
        parts = self._summary_parts(messages, 1)
        if parts is None:
            raise ValueError("invalid compaction blocks")
        return self._collect_evidence(parts[1], [*parts[2], *parts[3]])

    @staticmethod
    def _collect_evidence(previous: list[Message], prefix: list[Message]) -> dict[str, list[dict[str, Any]]]:
        evidence: dict[str, list[dict[str, Any]]] = {"artifacts": [], "cursors": []}
        seen: dict[str, set[bytes]] = {"artifacts": set(), "cursors": set()}

        def merge(manifest: dict[str, Any]) -> None:
            validate_compaction_metadata({COMPACTION_METADATA_KEY: manifest})
            for key in evidence:
                for record in manifest[key]:
                    identity = canonical_json_bytes(record)
                    if identity not in seen[key]:
                        seen[key].add(identity)
                        evidence[key].append(record)

        for message in previous:
            if COMPACTION_METADATA_KEY in message.metadata:
                merge(message.metadata[COMPACTION_METADATA_KEY])
        calls: dict[str, Any] = {}
        for message in prefix:
            if message.role == "assistant":
                calls = {call["id"]: call["function"] for call in message.tool_calls or []}
            if message.role != "tool":
                continue
            assert message.tool_call_id is not None  # Validated by _summary_parts.
            function = calls[message.tool_call_id]
            arguments = json.loads(function["arguments"])
            common = {
                "tool_call_id": message.tool_call_id,
                "tool_name": function["name"],
                "arguments": canonical_json_bytes(arguments).decode(),
            }
            refs: dict[str, Any] = {"artifacts": [], "cursors": []}
            if message.artifact_ref is not None:
                refs["artifacts"].append({**common, "artifact_ref": message.artifact_ref.to_dict()})
            if has_recovery_envelope(message.content):
                body, _, raw = message.content.rpartition("\n")
                envelope = json.loads(raw)
                if set(envelope) != {"vv_agent_recovery"}:
                    raise ValueError("invalid recovery envelope")
                recovery = envelope["vv_agent_recovery"]
                if not isinstance(recovery, dict) or set(recovery) - {
                    "truncated",
                    "truncation_reason",
                    "original_bytes",
                    "visible_bytes",
                    "artifact",
                    "cursor",
                }:
                    raise ValueError("invalid recovery fields")
                result = ToolExecutionResult.from_dict(
                    {
                        "tool_call_id": message.tool_call_id,
                        "content": body,
                        "status_code": "SUCCESS",
                        "directive": "continue",
                        **recovery,
                    }
                )
                if result.artifact:
                    refs["artifacts"].append({**common, "artifact_ref": result.artifact.to_dict()})
                if result.cursor:
                    refs["cursors"].append({**common, "cursor": result.cursor.to_dict()})
            merge(refs)
        return evidence

    @staticmethod
    def _render_evidence(evidence: dict[str, list[dict[str, Any]]]) -> str:
        lines = ["<Persisted Artifacts>"]
        for key, pointer in (("artifacts", "artifact_ref"), ("cursors", "cursor")):
            for record in evidence[key]:
                visible = {field: record[field] for field in ("tool_call_id", "tool_name", "arguments")}
                if key == "artifacts":
                    visible.update(
                        artifact_path=record[pointer]["path"], retrieval_hint="use read_file on artifact_path if needed"
                    )
                else:
                    visible.update(
                        path=record[pointer]["path"],
                        offset_chars=record[pointer]["offset_chars"],
                        retrieval_hint="use read_file on path if needed",
                    )
                lines.append("- " + canonical_json_bytes(visible).decode())
        return "\n".join([*lines, "</Persisted Artifacts>"])

    def apply_session_memory_context(self, messages: list[Message]) -> list[Message]:
        self._capture_base_system_prompt(messages)
        rendered_system_prompt = self._render_system_prompt()
        if not rendered_system_prompt:
            return list(messages)

        updated_messages = list(messages)
        if updated_messages and updated_messages[0].role == "system":
            if updated_messages[0].content == rendered_system_prompt:
                return updated_messages
            updated_messages[0] = replace(updated_messages[0], content=rendered_system_prompt)
            return updated_messages
        return [Message(role="system", content=rendered_system_prompt), *updated_messages]

    def strip_session_memory_context(self, messages: list[Message]) -> list[Message]:
        self._capture_base_system_prompt(messages)
        if not messages:
            return []
        if not self.base_system_prompt:
            return list(messages)

        updated_messages = list(messages)
        if updated_messages[0].role == "system" and updated_messages[0].content != self.base_system_prompt:
            updated_messages[0] = replace(updated_messages[0], content=self.base_system_prompt)
        return updated_messages

    def _calculate_message_length(self, messages: list[Message]) -> int:
        if not messages:
            return 0
        payload = [message.to_openai_message() for message in messages]
        return count_messages_tokens(payload, model=self.model)

    def _estimate_tool_message_length(self, messages: list[Message], recent_tool_call_ids: set[str] | None) -> int:
        if not recent_tool_call_ids:
            return 0
        tool_messages = [
            message.to_openai_message()
            for message in messages
            if message.role == "tool" and message.tool_call_id in recent_tool_call_ids
        ]
        if not tool_messages:
            return 0
        return count_messages_tokens(tool_messages, model=self.model)

    def effective_tokens(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None,
        recent_tool_call_ids: set[str] | None,
    ) -> int:
        if total_tokens is not None and total_tokens > 0:
            return total_tokens + self._estimate_tool_message_length(messages, recent_tool_call_ids)
        return self._calculate_message_length(messages)

    def estimate_memory_usage_percentage(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
    ) -> int:
        threshold = self.autocompact_threshold
        if threshold <= 0:
            return 0
        used_tokens = self.effective_tokens(
            messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        return int((used_tokens / threshold) * 100)

    def should_preemptive_microcompact(
        self,
        messages: list[Message],
        *,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
    ) -> bool:
        threshold = self.microcompact_trigger_threshold
        if threshold <= 0:
            return False
        effective_length = self.effective_tokens(
            messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        return effective_length > threshold

    def microcompact_messages(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], int]:
        plan = self.plan_microcompaction(
            messages,
            cycle_index=cycle_index,
            current_tokens=self._calculate_message_length(messages),
        )
        if plan is None:
            return messages, 0
        result = self.apply_microcompaction(messages, plan=plan)
        return result.messages, result.archived_count

    def plan_microcompaction(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None,
        current_tokens: int,
        recovery_tool_available: bool | None = None,
    ) -> MicrocompactPlan | None:
        can_recover = self.recovery_tool_available if recovery_tool_available is None else recovery_tool_available
        if (
            not can_recover
            or cycle_index is None
            or self.microcompact_trigger_threshold <= 0
            or current_tokens <= self.microcompact_trigger_threshold
        ):
            return None
        plan = plan_microcompact(
            messages,
            current_cycle=cycle_index,
            current_tokens=current_tokens,
            target_tokens=self.microcompact_target_threshold,
            policy=self.microcompaction_policy,
            result_retentions=self.tool_result_retentions,
            artifact_path_estimate_for=self._estimate_tool_artifact_path,
            estimate_message_tokens=self._estimate_message_tokens,
            excerpt_head_chars=self.tool_result_excerpt_head,
            excerpt_tail_chars=self.tool_result_excerpt_tail,
        )
        if current_tokens > self.autocompact_threshold:
            parts = self._summary_parts(messages, self.keep_recent_messages)
            if parts is None:
                return None
            protected = {id(message) for message in parts[3]}
            plan = replace(plan, candidates=tuple(c for c in plan.candidates if id(messages[c.message_index]) not in protected))
        return plan

    def apply_microcompaction(
        self,
        messages: list[Message],
        *,
        plan: MicrocompactPlan,
    ) -> MicrocompactionApplyResult:
        updated_messages = list(messages)
        remaining_tokens = plan.current_tokens
        archived_count = 0
        reclaimed_tokens = 0
        artifact_failure_count = 0
        for candidate in plan.candidates:
            if remaining_tokens <= plan.target_tokens:
                break
            message = updated_messages[candidate.message_index]
            artifact = self._archive_tool_message(message, candidate)
            if artifact is None:
                artifact_failure_count += 1
                continue
            marker = build_compacted_tool_content(
                self._compaction_excerpt(message),
                artifact_path=artifact.path,
                tool_name=candidate.tool_name,
                excerpt_head_chars=self.tool_result_excerpt_head,
                excerpt_tail_chars=self.tool_result_excerpt_tail,
            )
            replacement = replace_with_compacted_marker(
                message,
                candidate,
                artifact=artifact,
                marker=marker,
            )
            actual_reclaimed = max(
                self._estimate_message_tokens(message) - self._estimate_message_tokens(replacement),
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

    def _estimate_message_tokens(self, message: Message) -> int:
        return count_messages_tokens([message.to_openai_message()], model=self.model)

    def _render_system_prompt(self) -> str:
        session_context = self.session_memory.render_as_system_context() if self.session_memory is not None else ""
        if self.base_system_prompt and session_context:
            return f"{self.base_system_prompt}\n\n{session_context}"
        return session_context or self.base_system_prompt

    def _capture_base_system_prompt(self, messages: list[Message]) -> None:
        if self.base_system_prompt:
            return
        if not messages or messages[0].role != "system":
            return
        current_content = messages[0].content
        marker = "\n\n<Session Memory>"
        if marker in current_content:
            current_content = current_content.split(marker, 1)[0]
        if current_content:
            self.base_system_prompt = current_content

    def _maybe_extract_session_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None,
        current_tokens: int,
    ) -> bool:
        if self.session_memory is None or current_tokens <= 0:
            return False

        text_message_count = sum(1 for message in messages if message.role in {"user", "assistant"})
        if not self.session_memory.should_extract(current_tokens, text_message_count):
            return False

        extracted = self.session_memory.extract(
            messages,
            current_cycle=cycle_index or 0,
            current_tokens=current_tokens,
        )
        return extracted > 0

    def _maybe_append_memory_warning(self, messages: list[Message], *, message_length: int) -> tuple[list[Message], bool]:
        if not self.include_memory_warning:
            return messages, False
        if self.autocompact_threshold <= 0:
            return messages, False

        if message_length < self.warning_threshold:
            return messages, False

        template = _MEMORY_WARNING_PROMPTS.get(self.language, _MEMORY_WARNING_PROMPTS["en-US"])
        warning_text = template.format(memory_threshold_percentage=self.warning_threshold_percentage)
        for message in reversed(messages[-10:]):
            if message.role == "user" and warning_text in message.content:
                return messages, False

        warned = list(messages)
        warned.append(Message(role="user", content=warning_text))
        return warned, True

    def _estimate_tool_artifact_path(self, tool_call_id: str) -> str:
        del tool_call_id
        return ".vv-agent/artifacts/estimate/call-00000000000000000000000000000000.txt"

    def _archive_tool_message(
        self,
        message: Message,
        candidate: MicrocompactCandidate,
    ) -> ToolArtifactRef | None:
        backend = self.workspace_backend
        if backend is None:
            return None
        try:
            if candidate.existing_artifact is not None:
                if self._artifact_is_intact(candidate.existing_artifact):
                    return candidate.existing_artifact
                return None
            if has_recovery_envelope(message.content):
                return None
            return persist_text_artifact(
                backend,
                self.artifact_scope,
                candidate.tool_call_id,
                message.content,
                reuse_existing=True,
            )
        except Exception as exc:
            from vv_agent.runtime.cancellation import CancelledError

            if isinstance(exc, CancelledError) or getattr(exc, "vv_agent_control_flow", False):
                raise
            return None

    @staticmethod
    def _compaction_excerpt(message: Message) -> str:
        excerpt = message.metadata.get(EXCERPT_METADATA_KEY)
        return excerpt if isinstance(excerpt, str) else message.content

    def _artifact_is_intact(self, artifact: ToolArtifactRef) -> bool:
        backend = self.workspace_backend
        if backend is None:
            return False
        try:
            from vv_agent.workspace.streaming import scan_text

            scanned = scan_text(backend, artifact.path, lambda _text: None)
        except (OSError, ValueError):
            return False
        return scanned.valid_utf8 and scanned.size_bytes == artifact.size_bytes and scanned.sha256 == artifact.sha256

    def _build_compress_memory_prompt(self, messages: list[Message], *, previous: list[Message] | None = None) -> str:
        template = _COMPRESS_MEMORY_PROMPTS.get(self.language, _COMPRESS_MEMORY_PROMPTS["en-US"])

        def project(message: Message) -> dict[str, Any]:
            value = message.to_dict()
            value.pop("artifact_ref", None)
            if "image_url" in value:
                del value["image_url"]
                value["content"] = f"[image omitted from summary input: {message.content or 'image'}]"
            metadata = value.get("metadata", {})
            metadata.pop(COMPACTION_METADATA_KEY, None)
            if not metadata:
                value.pop("metadata", None)
            return value

        return template.format(
            previous_summary_jcs=canonical_json_bytes([project(m) for m in previous or []]).decode(),
            conversation_prefix_jcs=canonical_json_bytes([project(m) for m in messages]).decode(),
            event_limit=max(self.summary_event_limit, 1),
        )

    def _generate_summary(self, prompt: str) -> str:
        if self.summary_callback is not None:
            try:
                summarized = self.summary_callback(prompt, self.summary_backend, self.summary_model)
                if isinstance(summarized, str):
                    return summarized
            except Exception as exc:
                from vv_agent.runtime.cancellation import CancelledError

                if isinstance(exc, CancelledError) or getattr(exc, "vv_agent_control_flow", False):
                    raise
                logging.getLogger(__name__).debug("Memory summary callback failed", exc_info=True)
        return ""

    def _build_local_summary(self, messages: list[Message], artifacts: list[dict[str, str]]) -> str:
        events = self._build_summary_events(messages[2:])
        artifact_facts = [
            f"{item.get('path', '')} (tool={item.get('tool', 'unknown')})" for item in artifacts if item.get("path")
        ]
        payload = {
            "summary_version": "2.0",
            "original_user_messages": self._collect_original_user_messages(messages),
            "user_constraints": [],
            "decisions": [],
            "files_examined_or_modified": self._collect_file_actions(messages),
            "errors_and_fixes": self._collect_errors_and_fixes(messages),
            "progress": events,
            "key_facts": artifact_facts,
            "open_issues": [],
            "current_work_state": self._build_current_work_state(messages),
            "next_steps": [],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _normalize_summary_output(self, text: str) -> str:
        cleaned = self._strip_markdown_code_fence(text)
        cleaned = _ANALYSIS_BLOCK_PATTERN.sub("", cleaned).strip()
        summary_match = _SUMMARY_BLOCK_PATTERN.search(cleaned)
        if summary_match:
            cleaned = summary_match.group(1).strip()
        return cleaned

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if len(lines) < 2:
            return cleaned
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _parse_summary_payload(self, text: str) -> dict[str, object]:
        cleaned = text.strip()
        if not cleaned:
            return {}

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _collect_original_user_messages(self, messages: list[Message]) -> list[str]:
        user_messages: list[str] = []
        seen: set[str] = set()

        def append_unique(value: object) -> None:
            if not isinstance(value, str):
                return
            normalized = value.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            user_messages.append(normalized)

        for message in messages[1:]:
            if message.role != "user":
                continue
            content = message.content.strip()
            if not content:
                continue

            compressed_match = _COMPRESSED_AGENT_MEMORY_PATTERN.search(content)
            if compressed_match:
                summary_data = self._parse_summary_payload(compressed_match.group(1))
                previous_originals = summary_data.get("original_user_messages")
                if isinstance(previous_originals, list):
                    for previous_original in previous_originals:
                        append_unique(previous_original)

            original_match = _ORIGINAL_USER_REQUEST_PATTERN.search(content)
            if original_match and not compressed_match:
                append_unique(original_match.group(1))

            if compressed_match or original_match or "<Compressed Agent Memory>" in content:
                continue
            append_unique(content)
        return user_messages

    def _collect_file_actions(self, messages: list[Message], *, first_path_wins: bool = False) -> list[dict[str, str]]:
        action_priority = {"modified": 0, "created": 1, "deleted": 2, "read": 3}
        tool_action_map = {
            "read_file": "read",
            "file_info": "read",
            "write_file": "modified",
            "edit_file": "modified",
        }
        actions_by_path: dict[str, dict[str, str]] = {}
        ordered_paths: list[str] = []

        for message in messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function_payload = tool_call.get("function")
                if not isinstance(function_payload, dict):
                    continue
                tool_name = function_payload.get("name")
                if not isinstance(tool_name, str):
                    continue
                action = tool_action_map.get(tool_name)
                if action is None:
                    continue
                arguments = self._parse_tool_arguments(function_payload.get("arguments"))
                path = arguments.get("path") if first_path_wins else self._extract_file_path_from_arguments(arguments)
                if not isinstance(path, str) or not path.strip():
                    continue

                summary = self._summarize_file_action(tool_name, path)
                existing = actions_by_path.get(path)
                if existing is None:
                    actions_by_path[path] = {"path": path, "action": action, "summary": summary}
                    ordered_paths.append(path)
                    continue

                if first_path_wins:
                    continue
                if action_priority[action] < action_priority.get(existing["action"], 99):
                    existing["action"] = action
                existing["summary"] = summary

        return [actions_by_path[path] for path in ordered_paths]

    @staticmethod
    def _parse_tool_arguments(raw_arguments: object) -> dict[str, object]:
        if isinstance(raw_arguments, dict):
            normalized: dict[str, object] = {}
            for key, value in raw_arguments.items():
                normalized[str(key)] = value
            return normalized
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            return {}
        try:
            parsed: object = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        normalized: dict[str, object] = {}
        for key, value in parsed.items():
            normalized[str(key)] = value
        return normalized

    @staticmethod
    def _extract_file_path_from_arguments(arguments: dict[str, object]) -> str | None:
        for key in ("path", "file_path", "filepath", "target_file"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _summarize_file_action(tool_name: str, path: str) -> str:
        if tool_name == "read_file":
            return f"Read {path}"
        if tool_name == "file_info":
            return f"Inspected {path}"
        if tool_name == "write_file":
            return f"Updated {path}"
        if tool_name == "edit_file":
            return f"Modified {path}"
        return f"Touched {path}"

    def _collect_errors_and_fixes(self, messages: list[Message]) -> list[dict[str, str]]:
        error_entries: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if message.role != "tool":
                continue
            lowered = message.content.lower()
            if not any(token in lowered for token in ("error", "exception", "traceback", "failed")):
                continue
            fix = ""
            for follow_message in messages[index + 1 :]:
                if follow_message.role == "assistant" and follow_message.content.strip():
                    fix = self._summarize_message_content(follow_message.content)
                    break
            error_entries.append(
                {
                    "error": self._summarize_message_content(message.content),
                    "fix": fix,
                    "file": "",
                }
            )
            if len(error_entries) >= 5:
                break
        return error_entries

    def _build_current_work_state(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role not in {"assistant", "user"}:
                continue
            content = message.content.strip()
            if content:
                return self._summarize_message_content(content)
        return ""

    @staticmethod
    def _summarize_message_content(content: str, *, limit: int = 240) -> str:
        normalized = " ".join(content.split()).strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def _build_summary_events(self, middle: list[Message]) -> list[str]:
        events: list[str] = []
        limit = max(self.summary_event_limit, 1)
        for idx, message in enumerate(middle[:limit], start=1):
            content = message.content.replace("\n", " ").strip()
            if len(content) > 160:
                content = f"{content[:157]}..."
            note = f"{idx:02d}. {message.role}: {content}"
            if message.tool_call_id:
                note += f" (tool_call_id={message.tool_call_id})"
            if message.tool_calls:
                tool_names: list[str] = []
                for tool_call in message.tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function_payload = tool_call.get("function")
                    if isinstance(function_payload, dict):
                        tool_name = function_payload.get("name")
                        if isinstance(tool_name, str) and tool_name:
                            tool_names.append(tool_name)
                if tool_names:
                    note += f" (tool_calls={','.join(tool_names)})"
            events.append(note)
        if len(middle) > limit:
            events.append(f"... {len(middle) - limit} more messages omitted ...")
        return events


def build_memory_manager(
    *,
    task: AgentTask,
    workspace_path: Path,
    workspace_backend: WorkspaceBackend | None = None,
    ctx: ExecutionContext | None = None,
    tool_registry: ToolRegistry,
) -> MemoryManager:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}

    def read_optional_int(key: str, *, minimum: int = 0) -> int | None:
        if key not in metadata:
            return None
        raw = metadata.get(key)
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return max(value, minimum)

    def read_int(key: str, default: int, *, minimum: int = 0) -> int:
        value = read_optional_int(key, minimum=minimum)
        return max(default, minimum) if value is None else value

    warning_threshold = max(1, min(task.memory_threshold_percentage, 100))
    summary_backend = read_optional_string(metadata, "memory_summary_backend")
    summary_model = read_optional_string(metadata, "memory_summary_model") or task.model
    session_memory_extraction_backend = read_optional_string(metadata, "session_memory_extraction_backend") or summary_backend
    session_memory_extraction_model = read_optional_string(metadata, "session_memory_extraction_model") or summary_model
    session_memory_enabled = read_session_memory_enabled(metadata)
    model_context_window = metadata_token_limit(
        metadata,
        "model_context_window",
        minimum=1,
    )
    model_max_output_tokens = metadata_token_limit(
        metadata,
        "model_max_output_tokens",
        minimum=0,
    )
    if model_context_window is None or model_max_output_tokens is None:
        fallback_context_window, fallback_max_output_tokens = resolve_model_token_limits(task.model)
        if model_context_window is None:
            model_context_window = fallback_context_window
        if model_max_output_tokens is None:
            model_max_output_tokens = fallback_max_output_tokens
    effective_model_settings = task.model_settings
    if ctx is not None:
        runtime_model_settings = ctx.metadata.get("_vv_agent_model_settings")
        if isinstance(runtime_model_settings, ModelSettings):
            effective_model_settings = runtime_model_settings
    request_max_tokens = effective_model_settings.max_tokens if effective_model_settings is not None else None
    task_reserved_output_tokens = metadata_token_limit(
        metadata,
        "reserved_output_tokens",
        minimum=0,
    )
    if request_max_tokens is not None:
        reserved_output_tokens = request_max_tokens
        reserved_output_source = "model_settings"
    elif task_reserved_output_tokens is not None:
        reserved_output_tokens = task_reserved_output_tokens
        reserved_output_source = "task_metadata"
    else:
        reserved_output_tokens = 16_000
        reserved_output_source = "framework_fallback"
        if model_max_output_tokens is not None and model_max_output_tokens < reserved_output_tokens:
            reserved_output_tokens = model_max_output_tokens
            reserved_output_source = "framework_fallback_capped_by_model_capability"
    autocompact_buffer_tokens = read_int("autocompact_buffer_tokens", 13_000, minimum=0)
    if model_context_window is None:
        planning_prompt_capacity = task.memory_compact_threshold if task.memory_compact_threshold > 0 else 250_000
        model_context_window = min(
            planning_prompt_capacity + reserved_output_tokens + autocompact_buffer_tokens,
            (1 << 64) - 1,
        )
    session_memory: SessionMemory | None = None
    if session_memory_enabled:
        session_memory_scope = read_optional_string(metadata, "session_id", "task_id") or str(task.task_id or "").strip()
        session_memory = SessionMemory(
            SessionMemoryConfig(
                min_tokens_before_extraction=read_int("session_memory_min_tokens", 10_000, minimum=1),
                max_tokens=read_int("session_memory_max_tokens", 40_000, minimum=1),
                min_text_messages=read_int("session_memory_min_text_messages", 5, minimum=1),
                storage_dir=str(metadata.get("session_memory_storage_dir", ".memory/session")),
                extraction_callback=None,
                extraction_backend=session_memory_extraction_backend,
                extraction_model=session_memory_extraction_model,
                token_model=task.model or "",
            ),
            workspace=workspace_path if task.use_workspace else None,
            storage_scope=session_memory_scope,
        )
        session_memory.load()
    tool_result_retentions: dict[str, ToolResultRetention] = {}
    for tool_name in tool_registry.list_tool_names():
        declared_metadata = tool_registry.tool_metadata(tool_name)
        tool_result_retentions[tool_name] = (
            declared_metadata.result_retention if declared_metadata is not None else ToolResultRetention.ARCHIVE
        )
    return MemoryManager(
        compact_threshold=max(task.memory_compact_threshold, 0),
        keep_recent_messages=read_int("memory_keep_recent_messages", 10, minimum=1),
        model=task.model or "",
        model_context_window=model_context_window,
        model_max_output_tokens=model_max_output_tokens,
        reserved_output_tokens=reserved_output_tokens,
        reserved_output_source=reserved_output_source,
        autocompact_buffer_tokens=autocompact_buffer_tokens,
        language=str(metadata.get("language", "zh-CN")),
        warning_threshold_percentage=warning_threshold,
        include_memory_warning=bool(metadata.get("include_memory_warning", False)),
        tool_result_excerpt_head=read_int("tool_result_excerpt_head", 200),
        tool_result_excerpt_tail=read_int("tool_result_excerpt_tail", 200),
        microcompaction_policy=task.microcompaction_policy,
        tool_result_retentions=tool_result_retentions,
        workspace_backend=workspace_backend or (LocalWorkspaceBackend(workspace_path) if task.use_workspace else None),
        artifact_scope=task.task_id,
        summary_event_limit=read_int("summary_event_limit", 40, minimum=1),
        summary_backend=summary_backend,
        summary_model=summary_model,
        summary_callback=None,
        base_system_prompt=task.prompt_bundle.flatten(),
        session_memory=session_memory,
    )


def read_optional_string(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, str):
            value = raw.strip()
            if value:
                return value
    return None


def read_session_memory_enabled(metadata: dict[str, Any]) -> bool:
    explicit = metadata.get("session_memory_enabled", False)
    if not isinstance(explicit, bool):
        raise ValueError("session_memory_enabled must be a boolean")
    return explicit


def metadata_token_limit(metadata: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value
