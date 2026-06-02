from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
