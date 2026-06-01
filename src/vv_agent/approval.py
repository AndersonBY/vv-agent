from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    action: Literal["allow", "deny", "allow_session", "timeout"]
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> ApprovalDecision:
        return cls(action="allow")

    @classmethod
    def deny(cls, reason: str = "") -> ApprovalDecision:
        return cls(action="deny", reason=reason)

    @classmethod
    def timeout(cls, reason: str = "") -> ApprovalDecision:
        return cls(action="timeout", reason=reason)

    @classmethod
    def from_input(cls, decision: ApprovalDecision | str) -> ApprovalDecision:
        if isinstance(decision, ApprovalDecision):
            return decision
        normalized = decision.strip().lower()
        if normalized in {"allow", "approve", "approved"}:
            return cls.allow()
        if normalized in {"deny", "reject", "rejected"}:
            return cls.deny()
        if normalized == "allow_session":
            return cls(action="allow_session")
        if normalized == "timeout":
            return cls.timeout()
        raise ValueError(f"Unknown approval decision: {decision}")


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    trace_id: str = ""
    agent_name: str = ""
    cycle_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        run_id: str,
        trace_id: str,
        agent_name: str,
        cycle_index: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        return cls(
            request_id=f"approval_{uuid.uuid4().hex}",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=dict(arguments),
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            cycle_index=cycle_index,
            metadata=dict(metadata or {}),
        )


class ApprovalProvider(Protocol):
    def should_request(self, request: ApprovalRequest) -> bool:
        raise NotImplementedError

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        raise NotImplementedError


class ApprovalBroker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, ApprovalDecision] = {}
        self._cancel_decision: ApprovalDecision | None = None

    def register(self, request: ApprovalRequest) -> None:
        with self._condition:
            if self._cancel_decision is not None:
                self._decisions[request.request_id] = self._cancel_decision
                self._condition.notify_all()
                return
            self._pending[request.request_id] = request

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        with self._condition:
            if request_id not in self._pending:
                return False
            normalized = ApprovalDecision.from_input(decision)
            self._pending.pop(request_id, None)
            self._decisions[request_id] = normalized
            self._condition.notify_all()
            return True

    def cancel_pending(self, reason: str = "Run was cancelled.") -> int:
        decision = ApprovalDecision.deny(reason)
        with self._condition:
            self._cancel_decision = decision
            request_ids = list(self._pending)
            for request_id in request_ids:
                self._pending.pop(request_id, None)
                self._decisions[request_id] = decision
            if request_ids:
                self._condition.notify_all()
            return len(request_ids)

    def wait(self, request_id: str, timeout: float | None) -> ApprovalDecision:
        with self._condition:
            resolved = self._condition.wait_for(lambda: request_id in self._decisions, timeout=timeout)
            if not resolved:
                self._pending.pop(request_id, None)
                return ApprovalDecision.timeout("Approval request timed out.")
            decision = self._decisions.pop(request_id)
            self._pending.pop(request_id, None)
            return decision
