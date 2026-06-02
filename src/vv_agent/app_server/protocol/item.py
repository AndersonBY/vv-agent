from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ThreadItem:
    item_id: str
    thread_id: str
    turn_id: str
    item_type: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0
    updated_at: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "itemId": self.item_id,
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "type": self.item_type,
            "status": self.status,
            "payload": dict(self.payload),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
