from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TurnStartParams:
    thread_id: str
    input: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"threadId": self.thread_id, "input": [dict(item) for item in self.input]}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
