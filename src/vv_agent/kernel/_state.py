"""Pure execution state values shared by the reducer, projection and stores."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vv_agent.kernel._records import Record

if TYPE_CHECKING:
    from vv_agent.kernel._context import InputProtection


@dataclass
class Attempt:
    plan: Record
    state: str = "planned"
    dispatch: Record | None = None
    wait: dict[str, Any] | None = None
    child_handle: dict[str, Any] | None = None
    unknown: Record | None = None
    result: Record | None = None
    context: str | None = None
    unknown_consumed: bool = False
    approval: str | None = None
    superseded: bool = False
    prepared: Record | None = None
    approval_answer: Record | None = None

    @property
    def yielded(self) -> bool:
        return self.result is not None and "yielded_by" in self.result._payload

    @property
    def execution_plan(self) -> Record:
        if self.prepared is None:
            return self.plan
        p = self.prepared._payload
        return replace(
            self.plan,
            payload=self.plan._payload
            | {
                **{
                    key: p[key] for key in ("request", "request_digest", "tool", "op_kind", "provider_binding", "idempotency_key")
                },
                "budget_admission": {"hook_result": p["hook_result"], "shared_state": p["shared_state"]},
            },
        )

    @property
    def started(self) -> bool:
        return self.dispatch is not None


@dataclass
class Operation:
    turn_id: str
    kind: str
    attempts: dict[int, Attempt] = field(default_factory=dict)
    selected_attempt: int | None = None

    @property
    def state(self) -> str:
        return self.attempts[max(self.attempts)].state


@dataclass
class Turn:
    start: Record
    ended: bool = False
    cancelled: bool = False
    suspended: bool = False
    wait: Record | None = None


@dataclass
class ExecutionState:
    _protection: InputProtection | None = field(default=None, init=False, repr=False, compare=False)

    session_id: str | None = None
    operations: dict[str, Operation] = field(default_factory=dict)
    turns: dict[str, Turn] = field(default_factory=dict)
    active_turn_id: str | None = None
    closed: bool = False
    archived: bool = False
    cancel_requested: bool = False
    suspend_requested: bool = False
    phase: str = "idle"
    due_ms: tuple[int, ...] = ()
    next_drive_ms: int | None = None
    terminal_seq: int = 0
    applied_inputs: dict[str, Record] = field(default_factory=dict)
    admitted_inputs: set[str] = field(default_factory=set)
    child_completions: dict[tuple[str, int, str], Record] | None = None
    usage_observations: dict[str, int] = field(default_factory=dict)
    usage_values: dict[str, Record] = field(default_factory=dict)
    compactions: list[Record] = field(default_factory=list)
    boundaries: dict[tuple[str, str, str], Record] = field(default_factory=dict)
    preferred_endpoint_id: str | None = None
    model_selection: Record | None = None

    @property
    def waits(self) -> dict[tuple[str, int], dict[str, Any]]:
        return {
            (oid, number): attempt.wait
            for oid, op in self.operations.items()
            for number, attempt in op.attempts.items()
            if attempt.state == "parked" and attempt.wait is not None
        }


def child_handles(handle: dict[str, Any]) -> list[dict[str, Any]]:
    base = {key: value for key, value in handle.items() if key != "siblings"}
    return [base, *(base | sibling for sibling in handle.get("siblings", []))]
