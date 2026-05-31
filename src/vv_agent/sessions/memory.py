from __future__ import annotations

from dataclasses import dataclass, field

from vv_agent.types import Message


@dataclass(slots=True)
class MemorySession:
    session_id: str
    _items: list[Message] = field(default_factory=list)

    def get_items(self, limit: int | None = None) -> list[Message]:
        items = list(self._items)
        if limit is not None:
            return items[-max(int(limit), 0) :]
        return items

    def add_items(self, items: list[Message]) -> None:
        self._items.extend(items)

    def pop_item(self) -> Message | None:
        if not self._items:
            return None
        return self._items.pop()

    def clear_session(self) -> None:
        self._items.clear()
