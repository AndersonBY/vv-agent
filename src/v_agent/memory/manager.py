from __future__ import annotations

from dataclasses import dataclass

from v_agent.types import Message


@dataclass(slots=True)
class MemoryManager:
    threshold_chars: int = 24_000
    keep_recent_messages: int = 10

    def compact(self, messages: list[Message]) -> tuple[list[Message], bool]:
        if not messages:
            return messages, False

        cleaned = [msg for msg in messages if not (msg.role == "system" and msg.name == "memory_summary")]
        total_chars = sum(len(msg.content) for msg in cleaned)
        if total_chars <= self.threshold_chars:
            return cleaned, False

        if len(cleaned) < self.keep_recent_messages + 2:
            return cleaned, False

        head = cleaned[:1]
        recent_start = max(1, len(cleaned) - self.keep_recent_messages)
        # Preserve assistant->tool call structure at the compaction boundary.
        # Some OpenAI-compatible providers reject histories that start with a dangling tool message.
        while recent_start > 1 and cleaned[recent_start].role == "tool":
            recent_start -= 1

        recent = cleaned[recent_start:]
        middle = cleaned[1:recent_start]
        summary_lines: list[str] = []
        limit = 40
        for idx, msg in enumerate(middle[:limit], start=1):
            text = msg.content.replace("\n", " ").strip()
            if len(text) > 120:
                text = f"{text[:117]}..."
            summary_lines.append(f"{idx:02d}. {msg.role}: {text}")
        if len(middle) > limit:
            summary_lines.append(f"... {len(middle) - limit} more messages omitted ...")

        summary_text = "Compressed memory summary:\n" + "\n".join(summary_lines)
        summary_message = Message(role="system", name="memory_summary", content=summary_text)

        return [*head, summary_message, *recent], True
