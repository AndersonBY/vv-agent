from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vv_agent.memory.token_utils import count_tokens
from vv_agent.types import Message

DEFAULT_MIN_TOKENS = 10_000
DEFAULT_MAX_TOKENS = 40_000
DEFAULT_MIN_TEXT_MESSAGES = 5
DEFAULT_GROWTH_RATIO = 0.5
_SESSION_MEMORY_FILENAME = "session_memory.json"
_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*?\]")
_SESSION_MEMORY_CATEGORIES = (
    "user_intent",
    "decision",
    "file_change",
    "error_fix",
    "key_fact",
)

ExtractionCallback = Callable[[str, str | None, str | None], str | None]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionMemoryConfig:
    """Configuration for persistent session memory extraction and storage."""

    min_tokens_before_extraction: int = DEFAULT_MIN_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_text_messages: int = DEFAULT_MIN_TEXT_MESSAGES
    growth_ratio: float = DEFAULT_GROWTH_RATIO
    storage_dir: str = ".memory/session"
    extraction_callback: ExtractionCallback | None = None
    extraction_backend: str | None = None
    extraction_model: str | None = None
    token_model: str = ""


@dataclass(slots=True)
class SessionMemoryEntry:
    """A single durable memory item extracted from the conversation."""

    category: str
    content: str
    source_cycle: int
    importance: int = 5

    def __post_init__(self) -> None:
        normalized_category = (self.category or "key_fact").strip().lower()
        if normalized_category not in _SESSION_MEMORY_CATEGORIES:
            normalized_category = "key_fact"
        self.category = normalized_category
        self.content = str(self.content or "").strip()
        self.importance = min(max(int(self.importance or 5), 1), 10)
        self.source_cycle = int(self.source_cycle or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "content": self.content,
            "source_cycle": self.source_cycle,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMemoryEntry:
        return cls(
            category=str(data.get("category", "key_fact")),
            content=str(data.get("content", "")),
            source_cycle=int(data.get("source_cycle", 0) or 0),
            importance=int(data.get("importance", 5) or 5),
        )


@dataclass(slots=True)
class SessionMemoryState:
    """Runtime state for session memory extraction and persistence."""

    entries: list[SessionMemoryEntry] = field(default_factory=list)
    last_extracted_message_index: int = -1
    tokens_at_last_extraction: int = 0
    initialized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "last_extracted_message_index": self.last_extracted_message_index,
            "tokens_at_last_extraction": self.tokens_at_last_extraction,
            "initialized": self.initialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMemoryState:
        raw_entries = data.get("entries", [])
        entries = [
            SessionMemoryEntry.from_dict(item)
            for item in raw_entries
            if isinstance(item, dict)
        ]
        return cls(
            entries=entries,
            last_extracted_message_index=int(data.get("last_extracted_message_index", -1) or -1),
            tokens_at_last_extraction=int(data.get("tokens_at_last_extraction", 0) or 0),
            initialized=bool(data.get("initialized", False)),
        )


class SessionMemory:
    """Persistent structured memory that survives conversation compaction."""

    def __init__(self, config: SessionMemoryConfig, workspace: Path | None = None) -> None:
        self.config = config
        self.workspace = workspace
        self.state = SessionMemoryState()

    def should_extract(self, current_tokens: int, message_count: int) -> bool:
        """Return True when enough new context has accumulated for extraction."""

        if self.config.extraction_callback is None or current_tokens <= 0 or message_count <= 0:
            return False

        if not self.state.initialized:
            return (
                current_tokens >= self.config.min_tokens_before_extraction
                and message_count >= self.config.min_text_messages
            )

        growth_threshold = max(int(self.config.min_tokens_before_extraction * self.config.growth_ratio), 1)
        growth = current_tokens - self.state.tokens_at_last_extraction
        if growth < 0:
            growth = current_tokens
        return growth >= growth_threshold

    def extract(
        self,
        messages: list[Message],
        *,
        current_cycle: int,
        current_tokens: int,
    ) -> int:
        """Extract durable facts from new messages and persist them."""

        if self.config.extraction_callback is None or not messages:
            return 0

        start_index = 0
        if 0 <= self.state.last_extracted_message_index < len(messages):
            start_index = self.state.last_extracted_message_index + 1

        new_messages = [
            message
            for index, message in enumerate(messages)
            if index >= start_index and not self._should_skip_message(message)
        ]

        if not new_messages:
            self._record_extraction(len(messages) - 1, current_tokens)
            return 0

        prompt = self._build_extraction_prompt(new_messages)
        try:
            raw_result = self.config.extraction_callback(
                prompt,
                self.config.extraction_backend,
                self.config.extraction_model,
            )
        except Exception:
            logger.debug("Session memory extraction callback failed", exc_info=True)
            return 0

        parsed_entries = self._parse_extraction_result(raw_result, current_cycle)
        merged_count = self._merge_entries(parsed_entries)
        self._prune_to_budget()
        self._record_extraction(len(messages) - 1, current_tokens)
        self._save()
        return merged_count

    def render_as_system_context(self) -> str:
        """Render session memory as a system prompt section."""

        if not self.state.entries:
            return ""

        grouped: dict[str, list[str]] = {}
        for entry in self.state.entries:
            grouped.setdefault(entry.category, []).append(entry.content)

        ordered_categories = [
            category for category in _SESSION_MEMORY_CATEGORIES if grouped.get(category)
        ]
        ordered_categories.extend(
            category
            for category in sorted(grouped)
            if category not in _SESSION_MEMORY_CATEGORIES
        )

        parts = ["<Session Memory>"]
        for category in ordered_categories:
            parts.append(f"## {category}")
            for item in grouped[category]:
                parts.append(f"- {item}")
        parts.append("</Session Memory>")
        return "\n".join(parts)

    def on_compaction(self, *, current_tokens: int | None = None) -> None:
        """Reset transcript tracking after a full compaction while keeping memories."""

        self.state.last_extracted_message_index = -1
        if current_tokens is not None and current_tokens >= 0:
            self.state.tokens_at_last_extraction = current_tokens
        self._save()

    def load(self) -> None:
        """Load persisted session memory from disk if available."""

        path = self._storage_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load session memory from %s", path, exc_info=True)
            return
        if isinstance(data, dict):
            self.state = SessionMemoryState.from_dict(data)

    def _storage_path(self) -> Path | None:
        if self.workspace is None:
            return None
        workspace_root = self.workspace.resolve()
        storage_dir = Path(self.config.storage_dir)
        base = storage_dir if storage_dir.is_absolute() else self.workspace / storage_dir
        resolved = (base / _SESSION_MEMORY_FILENAME).resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            logger.warning("Session memory storage_dir escapes workspace, refusing: %s", resolved)
            return None
        return resolved

    def _save(self) -> None:
        path = self._storage_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to persist session memory to %s", path, exc_info=True)

    def _record_extraction(self, last_message_index: int, current_tokens: int) -> None:
        self.state.last_extracted_message_index = last_message_index
        self.state.tokens_at_last_extraction = max(int(current_tokens or 0), 0)
        self.state.initialized = True

    @staticmethod
    def _should_skip_message(message: Message) -> bool:
        if message.role == "system":
            return True
        return message.role == "user" and "<Compressed Agent Memory>" in message.content

    def _build_extraction_prompt(self, messages: list[Message]) -> str:
        serialized_messages = [self._message_to_text(message) for message in messages]
        return (
            "Analyze the following conversation messages and extract durable facts that should survive context "
            "compression.\n\n"
            "Categories:\n"
            "- user_intent: goals, constraints, preferences, explicit asks\n"
            "- decision: decisions or chosen approaches\n"
            "- file_change: files created/modified/deleted and why\n"
            "- error_fix: failures and their resolutions\n"
            "- key_fact: other important context that should not be forgotten\n\n"
            "Requirements:\n"
            "- Return JSON array only.\n"
            "- Keep each content field concise and deduplicatable.\n"
            "- Skip transient chatter and repeated information.\n"
            "- importance is 1-10 where 10 means critical.\n\n"
            "Output format:\n"
            '[{"category":"...", "content":"...", "importance": 5}]\n\n'
            f"Messages:\n{json.dumps(serialized_messages, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _message_to_text(message: Message) -> dict[str, Any]:
        payload = message.to_openai_message(include_reasoning_content=False)
        content = payload.get("content")
        if isinstance(content, str) and len(content) > 2_000:
            payload["content"] = f"{content[:1200]}\n...[truncated]...\n{content[-400:]}"
        return payload

    def _parse_extraction_result(self, raw: str | None, cycle: int) -> list[SessionMemoryEntry]:
        if not raw:
            return []

        text = raw.strip()
        match = _JSON_ARRAY_PATTERN.search(text)
        if match is None:
            return []

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        entries: list[SessionMemoryEntry] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            entries.append(
                SessionMemoryEntry(
                    category=str(item.get("category", "key_fact")),
                    content=content,
                    source_cycle=cycle,
                    importance=int(item.get("importance", 5) or 5),
                )
            )
        return entries

    def _merge_entries(self, entries: list[SessionMemoryEntry]) -> int:
        if not entries:
            return 0

        merged = 0
        existing_keys = {
            self._entry_key(entry): index
            for index, entry in enumerate(self.state.entries)
        }
        for entry in entries:
            key = self._entry_key(entry)
            existing_index = existing_keys.get(key)
            if existing_index is None:
                self.state.entries.append(entry)
                existing_keys[key] = len(self.state.entries) - 1
                merged += 1
                continue

            existing = self.state.entries[existing_index]
            existing.importance = max(existing.importance, entry.importance)
            existing.source_cycle = max(existing.source_cycle, entry.source_cycle)
        return merged

    @staticmethod
    def _entry_key(entry: SessionMemoryEntry) -> tuple[str, str]:
        normalized_content = " ".join(entry.content.lower().split())
        return entry.category, normalized_content

    def _prune_to_budget(self) -> None:
        if self.config.max_tokens <= 0 or not self.state.entries:
            return

        current_tokens = count_tokens(self.render_as_system_context(), model=self.config.token_model)
        while current_tokens > self.config.max_tokens and self.state.entries:
            drop_index, _ = min(
                enumerate(self.state.entries),
                key=lambda item: (item[1].importance, item[1].source_cycle, item[0]),
            )
            self.state.entries.pop(drop_index)
            current_tokens = count_tokens(self.render_as_system_context(), model=self.config.token_model)
