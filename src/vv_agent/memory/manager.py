from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from vv_agent.constants import READ_FILE_TOOL_NAME
from vv_agent.types import Message

_MEMORY_SUMMARY_NAME = "memory_summary"
_COMPACT_MARKER = "<Tool Result Compact>"
_ORIGINAL_USER_REQUEST_PATTERN = re.compile(r"<Original User Request>\\n?(.*?)\\n?</Original User Request>", re.DOTALL)

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
    "zh-CN": """<Conversation History>
{messages}
</Conversation History>

请将以上对话压缩为结构化 JSON「Task Status Summary」, 让 Agent 能快速恢复任务, 并保留用户约束、关键决策与核心上下文。

要求:
- 只输出 JSON, 不要 Markdown。
- 字段内容简洁、可检索, 短句表达。
- 没有信息的字段使用 [] 或 ""。
- 不要在 JSON 中包含文件路径, artifact 信息会由系统自动附加。

JSON Schema:
{{
  "summary_version": 1,
  "user_constraints": ["..."],
  "decisions": ["..."],
  "progress": ["..."],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "next_steps": ["..."]
}}
""",
    "en-US": """<Conversation History>
{messages}
</Conversation History>

Please compress the conversation into a structured JSON "Task Status Summary".
This summary should allow the Agent to quickly resume the task
while preserving user constraints, key decisions, and critical context.

Requirements:
- Output JSON only, no Markdown.
- Keep fields concise and searchable; use short sentences.
- If a field has no data, use [] or "" as appropriate.
- Do not include any file paths in the JSON; artifact information will be automatically appended by the system.

JSON Schema:
{{
  "summary_version": 1,
  "user_constraints": ["..."],
  "decisions": ["..."],
  "progress": ["..."],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "next_steps": ["..."]
}}
""",
}


SummaryCallback = Callable[[str, str | None, str | None], str | None]


@dataclass(slots=True)
class MemoryManager:
    compact_threshold: int = 128_000
    keep_recent_messages: int = 10
    language: str = "zh-CN"
    warning_threshold_percentage: int = 90
    include_memory_warning: bool = False
    tool_result_compact_threshold: int = 2_000
    tool_result_keep_last: int = 3
    tool_result_excerpt_head: int = 200
    tool_result_excerpt_tail: int = 200
    tool_calls_keep_last: int = 3
    assistant_no_tool_keep_last: int = 1
    tool_result_artifact_dir: str = ".memory/tool_results"
    workspace: Path | None = None
    summary_event_limit: int = 40
    summary_backend: str | None = None
    summary_model: str | None = None
    summary_callback: SummaryCallback | None = None

    def compact(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        total_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
    ) -> tuple[list[Message], bool]:
        if not messages:
            return messages, False

        cleaned = [msg for msg in messages if not (msg.role == "system" and msg.name == _MEMORY_SUMMARY_NAME)]
        summary_removed = len(cleaned) != len(messages)
        sanitized_messages, sanitized = self._sanitize_empty_assistant_messages(cleaned)

        message_length = self._calculate_effective_length(
            sanitized_messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        if message_length <= self.compact_threshold:
            maybe_warned_messages, warning_inserted = self._maybe_append_memory_warning(
                sanitized_messages,
                message_length=message_length,
            )
            return maybe_warned_messages, summary_removed or sanitized or warning_inserted

        compacted_messages, compacted = self._compact_messages(sanitized_messages, cycle_index=cycle_index)
        message_length = self._calculate_effective_length(
            compacted_messages,
            total_tokens=total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        if message_length <= self.compact_threshold:
            return compacted_messages, summary_removed or compacted or sanitized

        summarized_messages, summarized = self.compress_memory(compacted_messages, cycle_index=cycle_index)
        return summarized_messages, summary_removed or compacted or sanitized or summarized

    def compress_memory(
        self,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
    ) -> tuple[list[Message], bool]:
        if len(messages) <= 2:
            return messages, False

        # Collect tool-call info before compaction to preserve arguments in artifact metadata.
        tool_call_id_to_info = self._build_tool_call_id_to_info_map(messages)
        compacted_messages, _ = self._compact_messages(messages, cycle_index=cycle_index)
        artifacts = self._collect_compacted_artifacts(compacted_messages, tool_call_id_to_info)

        prompt = self._build_compress_memory_prompt(compacted_messages)
        compressed_memory = self._generate_summary(prompt, compacted_messages, artifacts)
        compressed_memory = self._strip_markdown_code_fence(compressed_memory)

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

    def _calculate_message_length(self, messages: list[Message]) -> int:
        if len(messages) <= 2:
            return 0
        payload = [message.to_openai_message() for message in messages]
        return len(json.dumps(payload[2:], ensure_ascii=False))

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
        return len(json.dumps(tool_messages, ensure_ascii=False))

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

    def _maybe_append_memory_warning(self, messages: list[Message], *, message_length: int) -> tuple[list[Message], bool]:
        if not self.include_memory_warning:
            return messages, False
        if self.compact_threshold <= 0:
            return messages, False

        usage_percentage = int((message_length / self.compact_threshold) * 100)
        if usage_percentage < self.warning_threshold_percentage:
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
        allowed_tool_call_ids = {
            tool_call.get("id")
            for message in messages
            if message.role == "assistant" and message.tool_calls
            for tool_call in message.tool_calls
            if isinstance(tool_call, dict) and tool_call.get("id")
        }

        updated = False
        normalized: list[Message] = []
        for message in messages:
            if message.role == "tool" and (
                not message.tool_call_id or message.tool_call_id not in allowed_tool_call_ids
            ):
                updated = True
                continue
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
        return content.startswith(_COMPACT_MARKER)

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
        return prompt_template.format(messages=json.dumps(serialized_messages, ensure_ascii=False))

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
            "summary_version": 1,
            "user_constraints": [],
            "decisions": [],
            "progress": events,
            "key_facts": artifact_facts,
            "open_issues": [],
            "next_steps": [],
        }
        return json.dumps(payload, ensure_ascii=False)

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
