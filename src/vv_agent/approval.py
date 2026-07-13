from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from vv_agent.runtime.cancellation import CancellationToken


class ApprovalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    action: Literal["allow", "deny", "allow_session", "timeout"]
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> ApprovalDecision:
        return cls(action="allow")

    @classmethod
    def allow_session(cls) -> ApprovalDecision:
        return cls(action="allow_session")

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
            arguments=deepcopy(arguments),
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            cycle_index=cycle_index,
            metadata=deepcopy(metadata or {}),
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
        self._session_allowed_tools: set[str] = set()
        self._cancel_decision: ApprovalDecision | None = None

    def is_session_allowed(self, tool_name: str) -> bool:
        with self._condition:
            return tool_name in self._session_allowed_tools

    def session_allowed_tools(self) -> frozenset[str]:
        with self._condition:
            return frozenset(self._session_allowed_tools)

    def pending_request(self, request_id: str) -> ApprovalRequest | None:
        with self._condition:
            return self._pending.get(request_id)

    def register(self, request: ApprovalRequest) -> None:
        with self._condition:
            if self._cancel_decision is not None:
                self._decisions[request.request_id] = self._cancel_decision
                self._condition.notify_all()
                return
            self._pending[request.request_id] = request

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        with self._condition:
            request = self._pending.get(request_id)
            if request is None:
                return False
            normalized = ApprovalDecision.from_input(decision)
            self._pending.pop(request_id, None)
            if normalized.action == "allow_session":
                self._session_allowed_tools.add(request.tool_name)
            self._decisions[request_id] = normalized
            self._condition.notify_all()
            return True

    def discard(self, request_id: str) -> bool:
        with self._condition:
            removed = self._pending.pop(request_id, None) is not None
            self._decisions.pop(request_id, None)
            if removed:
                self._condition.notify_all()
            return removed

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

    def reset_cancelled(self) -> None:
        with self._condition:
            self._cancel_decision = None

    def wait(self, request_id: str, timeout: float | None) -> ApprovalDecision:
        with self._condition:
            resolved = self._condition.wait_for(lambda: request_id in self._decisions, timeout=timeout)
            if not resolved:
                self._pending.pop(request_id, None)
                return ApprovalDecision.timeout("Approval request timed out.")
            decision = self._decisions.pop(request_id)
            self._pending.pop(request_id, None)
            return decision


def bind_request_cancellation(
    broker: ApprovalBroker,
    request_id: str,
    cancellation_token: CancellationToken | None,
) -> None:
    if cancellation_token is None:
        return

    def cancel_request() -> None:
        broker.resolve(
            request_id,
            ApprovalDecision.deny(cancellation_token.reason or "Operation was cancelled"),
        )

    cancellation_token.on_cancel(cancel_request)
