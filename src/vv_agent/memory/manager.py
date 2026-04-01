from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from vv_agent.constants import READ_FILE_TOOL_NAME
from vv_agent.memory.microcompact import MicrocompactConfig, is_microcompacted_tool_content, microcompact
from vv_agent.memory.post_compact_restore import PostCompactRestoreConfig, restore_key_files
from vv_agent.memory.session_memory import SessionMemory
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.types import Message

_MEMORY_SUMMARY_NAME = "memory_summary"
_COMPACT_MARKER = "<Tool Result Compact>"
_ORIGINAL_USER_REQUEST_PATTERN = re.compile(r"<Original User Request>\s*(.*?)\s*</Original User Request>", re.DOTALL)
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

<Conversation History>
{messages}
</Conversation History>

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

<Conversation History>
{messages}
</Conversation History>

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


@dataclass(slots=True)
class MemoryManager:
    compact_threshold: int = 128_000
    keep_recent_messages: int = 10
    model: str = ""
    model_context_window: int = 200_000
    reserved_output_tokens: int = 16_000
    autocompact_buffer_tokens: int = 13_000
    language: str = "zh-CN"
    warning_threshold_percentage: int = 90
    include_memory_warning: bool = False
    tool_result_compact_threshold: int = 2_000
    tool_result_keep_last: int = 3
    tool_result_excerpt_head: int = 200
    tool_result_excerpt_tail: int = 200
    tool_calls_keep_last: int = 3
    assistant_no_tool_keep_last: int = 1
    microcompact_trigger_ratio: float = 0.75
    microcompact_keep_recent_cycles: int = 3
    microcompact_min_result_length: int = 500
    microcompact_compactable_tools: set[str] | None = None
    tool_result_artifact_dir: str = ".memory/tool_results"
    workspace: Path | None = None
    summary_event_limit: int = 40
    summary_backend: str | None = None
    summary_model: str | None = None
    summary_callback: SummaryCallback | None = None
    base_system_prompt: str = ""
    session_memory: SessionMemory | None = None

    @property
    def effective_context_window(self) -> int:
        """Token budget available to prompt messages after reserving model output."""
        return max(self.model_context_window - self.reserved_output_tokens, 0)

    @property
    def autocompact_threshold(self) -> int:
        """Prompt token threshold that triggers automatic compaction."""
        threshold = self.effective_context_window - self.autocompact_buffer_tokens
        if threshold > 0:
            return threshold
        return max(self.compact_threshold, 0)

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
        ratio = min(max(self.microcompact_trigger_ratio, 0.0), 1.0)
        return int(threshold * ratio)

    def compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        force: bool = False,
    ) -> tuple[list[Message], bool]:
        if not messages:
            return messages, False

        cleaned = [msg for msg in messages if not (msg.role == "system" and msg.name == _MEMORY_SUMMARY_NAME)]
        summary_removed = len(cleaned) != len(messages)
        sanitized_messages, sanitized = self._sanitize_empty_assistant_messages(cleaned)
        working_messages = self.apply_session_memory_context(sanitized_messages)

        message_length = self._calculate_effective_length(
            working_messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        extracted = self._maybe_extract_session_memory(
            working_messages,
            cycle_index=cycle_index,
            current_tokens=message_length,
        )
        if extracted:
            working_messages = self.apply_session_memory_context(sanitized_messages)
            message_length = self._calculate_effective_length(
                working_messages,
                total_tokens=None,
                recent_tool_call_ids=recent_tool_call_ids,
            )
        if not force and message_length <= self.autocompact_threshold:
            maybe_warned_messages, warning_inserted = self._maybe_append_memory_warning(
                working_messages,
                message_length=message_length,
            )
            return maybe_warned_messages, summary_removed or sanitized or warning_inserted

        microcompacted_messages = working_messages
        microcompacted = False
        if not force:
            microcompacted_messages, cleared = self.microcompact_messages(
                working_messages,
                cycle_index=cycle_index,
            )
            microcompacted = cleared > 0
            if microcompacted:
                message_length = self._calculate_effective_length(
                    microcompacted_messages,
                    total_tokens=None,
                    recent_tool_call_ids=None,
                )
                if message_length <= self.autocompact_threshold:
                    return microcompacted_messages, summary_removed or sanitized or microcompacted

        compacted_messages, compacted = self._compact_messages(microcompacted_messages, cycle_index=cycle_index)
        message_length = self._calculate_effective_length(
            compacted_messages,
            total_tokens=None,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        if not force and message_length <= self.autocompact_threshold:
            return compacted_messages, summary_removed or compacted or sanitized or microcompacted

        summarized_messages, summarized = self.compress_memory(compacted_messages, cycle_index=cycle_index)
        if summarized and self.session_memory is not None:
            post_compaction_messages = self.apply_session_memory_context(summarized_messages)
            self.session_memory.on_compaction(
                current_tokens=self._calculate_message_length(post_compaction_messages),
            )
        return summarized_messages, summary_removed or compacted or sanitized or microcompacted or summarized

    def emergency_compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        drop_ratio: float = 0.2,
    ) -> list[Message]:
        # Kept for API symmetry with other compaction entry points.
        del cycle_index
        if len(messages) <= 2:
            return list(messages)

        system_message = messages[0] if messages and messages[0].role == "system" else None
        non_system = list(messages[1:] if system_message else messages)
        if not non_system:
            return [system_message] if system_message else []

        keep_count = max(self.keep_recent_messages, 1)
        clamped_ratio = min(max(drop_ratio, 0.0), 0.95)
        max_droppable = max(len(non_system) - keep_count, 0)
        drop_count = min(max(1, int(len(non_system) * clamped_ratio)), max_droppable) if max_droppable else 0
        start_index = min(drop_count, len(non_system))
        if len(non_system) - start_index < keep_count:
            start_index = max(len(non_system) - keep_count, 0)
        start_index = self._adjust_compaction_start_for_tool_context(non_system, start_index)

        kept = non_system[start_index:]
        kept, _ = self._normalize_orphan_tool_messages(kept)
        kept, _ = self._sanitize_empty_assistant_messages(kept)
        if system_message is not None:
            return [system_message, *kept]
        return kept

    def compress_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], bool]:
        summary_messages = self.strip_session_memory_context(messages)
        if len(summary_messages) <= 2:
            return summary_messages, False

        # Collect tool-call info before compaction to preserve arguments in artifact metadata.
        tool_call_id_to_info = self._build_tool_call_id_to_info_map(summary_messages)
        compacted_messages, _ = self._compact_messages(summary_messages, cycle_index=cycle_index)
        artifacts = self._collect_compacted_artifacts(compacted_messages, tool_call_id_to_info)

        prompt = self._build_compress_memory_prompt(compacted_messages)
        compressed_memory = self._normalize_summary_output(
            self._generate_summary(prompt, compacted_messages, artifacts)
        )
        summary_data = self._parse_summary_payload(compressed_memory)
        restored_context = restore_key_files(
            summary_data,
            self.workspace,
            PostCompactRestoreConfig(token_model=self.model),
        )

        if restored_context:
            compressed_memory = f"{compressed_memory}\n\n{restored_context}"

        if artifacts:
            artifact_section = ["<Persisted Artifacts>"]
            for artifact in artifacts:
                tool_name = artifact.get("tool", "unknown")
                tool_args = artifact.get("arguments", "")
                artifact_section.append(
                    f"- {artifact['path']} (tool: {tool_name}, arguments: {tool_args})"
                )
            artifact_section.append("</Persisted Artifacts>")
            compressed_memory = f"{compressed_memory}\n\n" + "\n".join(artifact_section)

        original_request = self._extract_original_user_request(compacted_messages)
        new_messages = [compacted_messages[0]]
        new_messages.append(
            Message(
                role="user",
                content=(
                    "<Original User Request>\n"
                    f"{original_request}\n"
                    "</Original User Request>\n\n"
                    "<Compressed Agent Memory>\n"
                    f"{compressed_memory}\n"
                    "</Compressed Agent Memory>"
                ),
            )
        )
        return new_messages, True

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

    def _calculate_effective_length(
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
        used_tokens = self._calculate_effective_length(
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
        effective_length = self._calculate_effective_length(
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
        if cycle_index is None:
            return messages, 0
        return microcompact(
            messages,
            current_cycle=cycle_index,
            config=MicrocompactConfig(
                trigger_ratio=self.microcompact_trigger_ratio,
                keep_recent_cycles=self.microcompact_keep_recent_cycles,
                min_result_length=self.microcompact_min_result_length,
                compactable_tools=self.microcompact_compactable_tools,
            ),
        )

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

    @staticmethod
    def _is_empty_content(content: str) -> bool:
        return not content.strip()

    def _sanitize_empty_assistant_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        updated = False
        sanitized: list[Message] = []
        for message in messages:
            if message.role == "assistant" and not message.tool_calls and self._is_empty_content(message.content):
                updated = True
                continue
            sanitized.append(message)
        return sanitized, updated

    def _adjust_compaction_start_for_tool_context(self, messages: list[Message], start_index: int) -> int:
        if start_index <= 0 or start_index >= len(messages):
            return max(start_index, 0)

        required_tool_call_ids = {
            str(message.tool_call_id or "").strip()
            for message in messages[start_index:]
            if message.role == "tool" and str(message.tool_call_id or "").strip()
        }
        if not required_tool_call_ids:
            return start_index

        adjusted_start = start_index
        for index in range(start_index - 1, -1, -1):
            message = messages[index]
            if message.role != "assistant" or not message.tool_calls:
                continue
            message_tool_call_ids = {
                str(tool_call.get("id") or "").strip()
                for tool_call in message.tool_calls
                if isinstance(tool_call, dict) and str(tool_call.get("id") or "").strip()
            }
            if not message_tool_call_ids.intersection(required_tool_call_ids):
                continue
            adjusted_start = index
            required_tool_call_ids.difference_update(message_tool_call_ids)
            if not required_tool_call_ids:
                break
        return adjusted_start

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

    def _compact_messages(self, messages: list[Message], *, cycle_index: int | None) -> tuple[list[Message], bool]:
        updated = False
        compacted = messages

        compacted, stripped = self._strip_stale_tool_calls(compacted)
        updated = updated or stripped

        compacted, normalized = self._normalize_orphan_tool_messages(compacted)
        updated = updated or normalized

        compacted, collapsed_assistant = self._collapse_assistant_no_tool_messages(compacted)
        updated = updated or collapsed_assistant

        compacted, compacted_images = self._compact_processed_image_messages(compacted)
        updated = updated or compacted_images

        compacted, compacted_tools = self._compact_tool_messages(compacted, cycle_index=cycle_index)
        updated = updated or compacted_tools

        compacted, sanitized = self._sanitize_empty_assistant_messages(compacted)
        updated = updated or sanitized

        return compacted, updated

    def _strip_stale_tool_calls(self, messages: list[Message]) -> tuple[list[Message], bool]:
        keep_count = max(self.tool_calls_keep_last, 0)
        tool_call_indices = [idx for idx, message in enumerate(messages) if message.role == "assistant" and message.tool_calls]
        keep_indices = set(tool_call_indices[-keep_count:]) if keep_count else set()

        updated = False
        stripped: list[Message] = []
        for idx, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls and idx not in keep_indices:
                updated = True
                new_message = replace(message, tool_calls=None)
                if self._is_empty_content(new_message.content):
                    continue
                stripped.append(new_message)
                continue
            stripped.append(message)
        return stripped, updated

    def _normalize_orphan_tool_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        updated = False
        pending_tool_calls: dict[str, int] = {}
        normalized: list[Message] = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    raw_tool_call_id = tool_call.get("id")
                    tool_call_id = str(raw_tool_call_id or "").strip()
                    if not tool_call_id:
                        continue
                    pending_tool_calls[tool_call_id] = pending_tool_calls.get(tool_call_id, 0) + 1
                normalized.append(message)
                continue

            if message.role == "tool":
                tool_call_id = str(message.tool_call_id or "").strip()
                if not tool_call_id:
                    updated = True
                    continue
                remaining = pending_tool_calls.get(tool_call_id, 0)
                if remaining <= 0:
                    updated = True
                    continue
                pending_tool_calls[tool_call_id] = remaining - 1
            normalized.append(message)
        return normalized, updated

    def _collapse_assistant_no_tool_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        keep_last = max(self.assistant_no_tool_keep_last, 0)
        if keep_last <= 0:
            return messages, False

        updated = False
        collapsed: list[Message] = []
        run_buffer: list[Message] = []

        def flush_buffer() -> None:
            nonlocal updated
            if not run_buffer:
                return
            if len(run_buffer) > keep_last:
                updated = True
                collapsed.extend(run_buffer[-keep_last:])
            else:
                collapsed.extend(run_buffer)
            run_buffer.clear()

        for message in messages:
            if message.role == "assistant" and not message.tool_calls:
                run_buffer.append(message)
                continue
            flush_buffer()
            collapsed.append(message)

        flush_buffer()
        return collapsed, updated

    def _compact_processed_image_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        assistant_indices = [idx for idx, message in enumerate(messages) if message.role == "assistant"]
        if not assistant_indices:
            return messages, False

        updated = False
        compacted: list[Message] = []
        for idx, message in enumerate(messages):
            if message.role == "user" and message.image_url:
                has_following_assistant = any(assistant_index > idx for assistant_index in assistant_indices)
                if has_following_assistant:
                    updated = True
                    compacted.append(
                        replace(
                            message,
                            content=f"{message.content} [image payload compacted]".strip(),
                            image_url=None,
                        )
                    )
                    continue
            compacted.append(message)
        return compacted, updated

    def _compact_tool_messages(self, messages: list[Message], *, cycle_index: int | None) -> tuple[list[Message], bool]:
        if self.tool_result_compact_threshold <= 0:
            return messages, False

        tool_call_id_to_info = self._build_tool_call_id_to_info_map(messages)
        tool_indices = [idx for idx, message in enumerate(messages) if message.role == "tool"]
        keep_count = max(self.tool_result_keep_last, 0)
        keep_indices = set(tool_indices[-keep_count:]) if keep_count else set()

        updated = False
        compacted: list[Message] = []
        for idx, message in enumerate(messages):
            if message.role != "tool" or idx in keep_indices:
                compacted.append(message)
                continue

            if len(message.content) <= self.tool_result_compact_threshold:
                compacted.append(message)
                continue
            if self._is_compacted_tool_content(message.content):
                compacted.append(message)
                continue

            tool_info = tool_call_id_to_info.get(message.tool_call_id or "")
            tool_name = tool_info.get("name") if tool_info else None
            artifact_path = self._persist_tool_content(message.content, message.tool_call_id, cycle_index=cycle_index)
            compacted_content = self._build_compacted_tool_content(
                message.content,
                artifact_path=artifact_path,
                tool_name=tool_name,
            )
            compacted.append(replace(message, content=compacted_content))
            updated = True

        return compacted, updated

    @staticmethod
    def _is_compacted_tool_content(content: str) -> bool:
        return content.startswith(_COMPACT_MARKER) or is_microcompacted_tool_content(content)

    def _build_tool_artifact_path(self, tool_call_id: str | None, *, cycle_index: int | None) -> Path:
        safe_tool_call_id = (tool_call_id or f"tool_result_{uuid.uuid4().hex}").strip()
        safe_tool_call_id = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_tool_call_id)
        filename = f"{safe_tool_call_id}.txt"
        base = Path(self.tool_result_artifact_dir)
        if cycle_index is None:
            return base / filename
        return base / f"cycle_{cycle_index}" / filename

    def _persist_tool_content(self, content: str, tool_call_id: str | None, *, cycle_index: int | None) -> str | None:
        if self.workspace is None:
            return None
        artifact_rel_path = self._build_tool_artifact_path(tool_call_id, cycle_index=cycle_index)
        target = (self.workspace / artifact_rel_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return artifact_rel_path.as_posix()

    def _build_compacted_tool_content(
        self,
        content: str,
        *,
        artifact_path: str | None,
        tool_name: str | None,
    ) -> str:
        head_length = max(self.tool_result_excerpt_head, 0)
        tail_length = max(self.tool_result_excerpt_tail, 0)
        total_length = len(content)

        head = content[:head_length] if head_length > 0 else ""
        tail = content[-tail_length:] if tail_length > 0 and total_length > head_length else ""

        excerpt_parts: list[str] = []
        if head:
            excerpt_parts.append(head)
        if tail:
            if head:
                excerpt_parts.append("...<snip>...")
            excerpt_parts.append(tail)
        excerpt = "\n".join(excerpt_parts).strip()
        truncated_chars = max(total_length - len(head) - len(tail), 0)
        artifact_line = artifact_path or "N/A"
        tool_line = f"tool_name: {tool_name}\n" if tool_name else ""

        return (
            "<Tool Result Compact>\n"
            f"{tool_line}"
            f"artifact_path: {artifact_line}\n"
            f"total_chars: {total_length}\n"
            f"truncated_chars: {truncated_chars}\n"
            f"retrieval_hint: use {READ_FILE_TOOL_NAME} on artifact_path if needed\n"
            "excerpt:\n"
            f"{excerpt}\n"
            "</Tool Result Compact>"
        )

    def _build_tool_call_id_to_info_map(self, messages: list[Message]) -> dict[str, dict[str, str]]:
        tool_call_id_to_info: dict[str, dict[str, str]] = {}
        for message in messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function_payload = tool_call.get("function")
                if not tool_call_id or not isinstance(function_payload, dict):
                    continue

                info: dict[str, str] = {}
                tool_name = function_payload.get("name")
                if isinstance(tool_name, str) and tool_name:
                    info["name"] = tool_name
                tool_arguments = function_payload.get("arguments")
                if isinstance(tool_arguments, str) and tool_arguments:
                    info["arguments"] = tool_arguments
                if info:
                    tool_call_id_to_info[str(tool_call_id)] = info
        return tool_call_id_to_info

    def _collect_compacted_artifacts(
        self,
        messages: list[Message],
        tool_call_id_to_info: dict[str, dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        for message in messages:
            if message.role != "tool":
                continue
            if not self._is_compacted_tool_content(message.content):
                continue

            artifact_info: dict[str, str] = {}
            for line in message.content.splitlines():
                stripped = line.strip()
                if stripped.startswith("tool_name:"):
                    artifact_info["tool"] = stripped[len("tool_name:") :].strip()
                elif stripped.startswith("artifact_path:"):
                    path = stripped[len("artifact_path:") :].strip()
                    if path and path != "N/A":
                        artifact_info["path"] = path

            if tool_call_id_to_info:
                tool_call_id = message.tool_call_id
                if tool_call_id and tool_call_id in tool_call_id_to_info:
                    tool_info = tool_call_id_to_info[tool_call_id]
                    if "tool" not in artifact_info and "name" in tool_info:
                        artifact_info["tool"] = tool_info["name"]
                    if "arguments" in tool_info:
                        artifact_info["arguments"] = tool_info["arguments"]

            if artifact_info.get("path"):
                artifacts.append(artifact_info)
        return artifacts

    def _build_compress_memory_prompt(self, messages: list[Message]) -> str:
        prompt_template = _COMPRESS_MEMORY_PROMPTS.get(self.language, _COMPRESS_MEMORY_PROMPTS["en-US"])
        serialized_messages = [message.to_openai_message() for message in messages]
        return prompt_template.format(
            messages=json.dumps(serialized_messages, ensure_ascii=False),
            event_limit=max(self.summary_event_limit, 1),
        )

    def _generate_summary(
        self,
        prompt: str,
        messages: list[Message],
        artifacts: list[dict[str, str]],
    ) -> str:
        if self.summary_callback is not None:
            try:
                summarized = self.summary_callback(prompt, self.summary_backend, self.summary_model)
                if isinstance(summarized, str) and summarized.strip():
                    return summarized.strip()
            except Exception:
                logging.getLogger(__name__).debug("Memory summary callback failed", exc_info=True)

        return self._build_local_summary(messages, artifacts)

    def _build_local_summary(self, messages: list[Message], artifacts: list[dict[str, str]]) -> str:
        events = self._build_summary_events(messages[2:])
        artifact_facts = [
            f"{item.get('path', '')} (tool={item.get('tool', 'unknown')})"
            for item in artifacts
            if item.get("path")
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

    def _extract_original_user_request(self, messages: list[Message]) -> str:
        for message in messages[1:]:
            if message.role != "user":
                continue
            content = message.content.strip()
            if not content:
                continue
            match = _ORIGINAL_USER_REQUEST_PATTERN.search(content)
            if match:
                return match.group(1).strip()
            return content
        return ""

    def _collect_original_user_messages(self, messages: list[Message]) -> list[str]:
        user_messages: list[str] = []
        for message in messages[1:]:
            if message.role != "user":
                continue
            content = message.content.strip()
            if not content:
                continue
            match = _ORIGINAL_USER_REQUEST_PATTERN.search(content)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    user_messages.append(extracted)
                continue
            if "<Compressed Agent Memory>" in content:
                continue
            user_messages.append(content)
        return user_messages

    def _collect_file_actions(self, messages: list[Message]) -> list[dict[str, str]]:
        action_priority = {"modified": 0, "created": 1, "deleted": 2, "read": 3}
        tool_action_map = {
            "read_file": "read",
            "file_info": "read",
            "write_file": "modified",
            "file_str_replace": "modified",
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
                path = self._extract_file_path_from_arguments(arguments)
                if not path:
                    continue

                summary = self._summarize_file_action(tool_name, path)
                existing = actions_by_path.get(path)
                if existing is None:
                    actions_by_path[path] = {"path": path, "action": action, "summary": summary}
                    ordered_paths.append(path)
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
        if tool_name == "file_str_replace":
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
