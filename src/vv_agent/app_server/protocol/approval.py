from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    TIMEOUT = "timeout"

    @classmethod
    def from_wire(cls, value: ApprovalDecision | str) -> ApprovalDecision:
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass(frozen=True, slots=True)
class ApprovalRequestParams:
    request_id: str
    thread_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    preview: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "toolCallId": self.tool_call_id,
            "toolName": self.tool_name,
            "preview": self.preview,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class ApprovalResolveParams:
    request_id: str
    thread_id: str
    turn_id: str
    decision: ApprovalDecision
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ApprovalDecision.from_wire(self.decision))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requestId": self.request_id,
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "decision": self.decision.value,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
