from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from vv_agent.sessions.base import _normalize_message
from vv_agent.types import Message


@dataclass(slots=True)
class MemorySession:
    session_id: str
    _items: list[Message] = field(default_factory=list)

    def get_items(self, limit: int | None = None) -> list[Message]:
        items = deepcopy(self._items)
        if limit is not None:
            normalized_limit = max(int(limit), 0)
            return [] if normalized_limit == 0 else items[-normalized_limit:]
        return items

    def add_items(self, items: list[Message]) -> None:
        normalized = [_normalize_message(item) for item in items]
        self._items.extend(normalized)

    def pop_item(self) -> Message | None:
        if not self._items:
            return None
        return deepcopy(self._items.pop())

    def clear(self) -> None:
        self._items.clear()

    def clear_session(self) -> None:
        self.clear()


@dataclass(slots=True)
class MemorySessionStore:
    _sessions: dict[str, MemorySession] = field(default_factory=dict)

    def session(self, session_id: str) -> MemorySession:
        if session_id not in self._sessions:
            self._sessions[session_id] = MemorySession(session_id)
        return self._sessions[session_id]
