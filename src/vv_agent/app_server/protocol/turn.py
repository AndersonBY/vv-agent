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
class TurnResumeParams:
    thread_id: str
    turn_id: str
    checkpoint_key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "checkpointKey": self.checkpoint_key,
        }


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    key: str
    resume_attempt: int
    cycle_index: int
    status: str
    terminal_acknowledged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "resumeAttempt": self.resume_attempt,
            "cycleIndex": self.cycle_index,
            "status": self.status,
            "terminalAcknowledged": self.terminal_acknowledged,
        }


@dataclass(frozen=True, slots=True)
class InterruptionSummary:
    reason: str
    operation_id: str
    operation_kind: str
    cycle_index: int
    risk: str
    idempotency_support: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "operationId": self.operation_id,
            "operationKind": self.operation_kind,
            "cycleIndex": self.cycle_index,
            "risk": self.risk,
            "idempotencySupport": self.idempotency_support,
        }


@dataclass(frozen=True, slots=True)
class TurnResumeResponse:
    thread_id: str
    turn_id: str
    run_id: str
    status: str
    final_output: Any | None = None
    completion_reason: str | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None
    checkpoint: CheckpointSummary | None = None
    interruption: InterruptionSummary | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "runId": self.run_id,
            "status": self.status,
        }
        optional_fields = {
            "finalOutput": self.final_output,
            "completionReason": self.completion_reason,
            "completionToolName": self.completion_tool_name,
            "partialOutput": self.partial_output,
            "error": self.error,
        }
        payload.update({name: value for name, value in optional_fields.items() if value is not None})
        if self.checkpoint is not None:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.interruption is not None:
            payload["interruption"] = self.interruption.to_dict()
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
