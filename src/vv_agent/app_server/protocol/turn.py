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


@dataclass(frozen=True, slots=True)
class TurnSteerParams:
    thread_id: str
    expected_turn_id: str
    input: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threadId": self.thread_id,
            "expectedTurnId": self.expected_turn_id,
            "input": [dict(item) for item in self.input],
        }


@dataclass(frozen=True, slots=True)
class TurnFollowUpParams:
    thread_id: str
    expected_turn_id: str
    input: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threadId": self.thread_id,
            "expectedTurnId": self.expected_turn_id,
            "input": [dict(item) for item in self.input],
        }


@dataclass(frozen=True, slots=True)
class TurnInterruptParams:
    thread_id: str
    expected_turn_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {"threadId": self.thread_id, "expectedTurnId": self.expected_turn_id}
        if self.reason:
            payload["reason"] = self.reason
        return payload
