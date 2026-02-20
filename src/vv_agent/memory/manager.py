from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from vv_agent.types import Message

_MEMORY_SUMMARY_NAME = "memory_summary"
_COMPACT_MARKER = "<Tool Result Compact>"
_MEMORY_WARNING_PROMPTS = {
    "zh-CN": (
        "当前记忆已使用容量超过 {memory_threshold_percentage}% "
        "建议立即整理并把关键资料写入 workspace 以避免压缩后信息丢失。"
    ),
    "en-US": (
        "Memory usage has exceeded {memory_threshold_percentage}% "
        "Please persist key information into workspace before compression."
    ),
}


@dataclass(slots=True)
class MemoryManager:
    threshold_chars: int = 24_000
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

    def compact(self, messages: list[Message], *, cycle_index: int | None = None) -> tuple[list[Message], bool]:
        if not messages:
            return messages, False

        cleaned = [msg for msg in messages if not (msg.role == "system" and msg.name == _MEMORY_SUMMARY_NAME)]
        summary_removed = len(cleaned) != len(messages)
        sanitized_messages, sanitized = self._sanitize_empty_assistant_messages(cleaned)

        message_length = self._calculate_message_length(sanitized_messages)
        if message_length <= self.threshold_chars:
            maybe_warned_messages, warning_inserted = self._maybe_append_memory_warning(
                sanitized_messages,
                message_length=message_length,
            )
            return maybe_warned_messages, summary_removed or sanitized or warning_inserted

        compacted_messages, compacted = self._compact_messages(sanitized_messages, cycle_index=cycle_index)
        message_length = self._calculate_message_length(compacted_messages)
        if message_length <= self.threshold_chars:
            return compacted_messages, summary_removed or compacted or sanitized

        summarized_messages, summarized = self._summarize_messages(compacted_messages)
        return summarized_messages, summary_removed or compacted or sanitized or summarized

    def _calculate_message_length(self, messages: list[Message]) -> int:
        if len(messages) <= 2:
            return 0
        payload = [message.to_openai_message() for message in messages]
        return len(json.dumps(payload[2:], ensure_ascii=False))

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
        if self.threshold_chars <= 0:
            return messages, False

        usage_percentage = int((message_length / self.threshold_chars) * 100)
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

            artifact_path = self._persist_tool_content(message.content, message.tool_call_id, cycle_index=cycle_index)
            compacted_content = self._build_compacted_tool_content(message.content, artifact_path=artifact_path)
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

    def _build_compacted_tool_content(self, content: str, *, artifact_path: str | None) -> str:
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

        return (
            "<Tool Result Compact>\n"
            f"artifact_path: {artifact_line}\n"
            f"total_chars: {total_length}\n"
            f"truncated_chars: {truncated_chars}\n"
            "retrieval_hint: use _read_file on artifact_path if needed\n"
            "excerpt:\n"
            f"{excerpt}\n"
            "</Tool Result Compact>"
        )

    def _summarize_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        keep_recent = max(self.keep_recent_messages, 0)
        head_size = 2 if len(messages) >= 2 else 1
        if len(messages) <= head_size:
            return messages, False
        if keep_recent <= 0:
            keep_recent = 1

        recent_start = max(head_size, len(messages) - keep_recent)
        while recent_start > head_size and recent_start < len(messages) and messages[recent_start].role == "tool":
            recent_start -= 1

        head = messages[:head_size]
        middle = messages[head_size:recent_start]
        recent = messages[recent_start:]
        if not middle:
            return messages, False

        events = self._build_summary_events(middle)
        artifacts = self._collect_compacted_artifacts(messages)
        summary_payload = {
            "summary_version": 1,
            "events": events,
            "persisted_artifacts": artifacts,
            "retrieval_hint": "use _read_file with artifact_path when details are needed",
        }
        summary_text = "Compressed memory summary:\n" + json.dumps(summary_payload, ensure_ascii=False)
        summary_message = Message(role="system", name=_MEMORY_SUMMARY_NAME, content=summary_text)
        return [*head, summary_message, *recent], True

    def _build_summary_events(self, middle: list[Message]) -> list[str]:
        events: list[str] = []
        limit = max(self.summary_event_limit, 1)
        for idx, message in enumerate(middle[:limit], start=1):
            role = message.role
            content = message.content.replace("\n", " ").strip()
            if len(content) > 160:
                content = f"{content[:157]}..."
            note = f"{idx:02d}. {role}: {content}"
            if message.tool_call_id:
                note += f" (tool_call_id={message.tool_call_id})"
            if message.tool_calls:
                tool_names = []
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
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

    def _collect_compacted_artifacts(self, messages: list[Message]) -> list[str]:
        artifact_paths: list[str] = []
        for message in messages:
            if message.role != "tool":
                continue
            if not self._is_compacted_tool_content(message.content):
                continue
            for line in message.content.splitlines():
                stripped = line.strip()
                if stripped.startswith("artifact_path:"):
                    path = stripped[len("artifact_path:"):].strip()
                    if path and path != "N/A":
                        artifact_paths.append(path)
        return artifact_paths
