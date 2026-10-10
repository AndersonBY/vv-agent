"""Child admission and terminal delivery on the host's session transaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vv_agent.types import ToolExecutionResult, ToolResultStatus

from .records import InboxItem, Record, SessionSpec
from .store import SessionStore, SessionTx, StoredRecord

if TYPE_CHECKING:
    from .reducer import ExecutionState


@dataclass(frozen=True)
class ChildSession:
    spec: SessionSpec
    content: str
    generation: int = 0
    consumers: tuple[str, ...] = ()
    background: bool = False


class InvalidChildBatch(ValueError):
    """Custom admission must return a nonempty, uniformly scheduled child batch."""

    code = "invalid_child_batch"


def child_handles(handle: dict[str, Any]) -> list[dict[str, Any]]:
    base = {key: value for key, value in handle.items() if key != "siblings"}
    return [base, *(base | sibling for sibling in handle.get("siblings", []))]


def cancel_children(store: SessionStore, tx: SessionTx, state: ExecutionState, sid: str, tid: str) -> list[Record]:
    for op in state.operations.values():
        if op.turn_id != tid:
            continue
        for attempt in op.attempts.values():
            if attempt.child_handle is None:
                continue
            for handle in child_handles(attempt.child_handle):
                child_state, _, _ = store.read_state(handle["session_id"])
                child_tid = child_state.active_turn_id or handle["turn_id"]
                child_turn = child_state.turns.get(child_tid)
                if child_turn is None or not child_turn.ended:
                    tx.push(
                        handle["session_id"],
                        InboxItem(
                            f"parent-cancel/{sid}/{tid}/{attempt.plan.operation_id}/{attempt.plan.attempt}",
                            "control",
                            {"action": "cancel"},
                            child_tid,
                            child_turn.start._payload["generation"] if child_turn else handle["generation"],
                        ),
                    )
    return []


def create_child(tx: SessionTx, child: ChildSession, plan: Record, generation: int) -> dict[str, Any]:
    handle = {
        "kind": "child",
        "session_id": child.spec.session_id,
        "turn_id": f"{child.spec.session_id}/turn/start",
        "delivery_target": {
            "session_id": plan.session_id,
            "turn_id": plan.turn_id,
            "generation": generation,
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
        },
        "generation": child.generation,
        "background": child.background,
    }
    tx.create(
        replace(
            child.spec,
            parent_session_id=plan.session_id,
            parent_operation_id=plan.operation_id,
            attributes=dict(child.spec.attributes or {}) | {"child_handle": handle},
        ),
        consumers=(*child.consumers, "child_delivery"),
    )
    tx.push(child.spec.session_id, InboxItem("start", "user", {"content": child.content}, generation=child.generation))
    return handle


def completion_input(handle: dict[str, Any], terminal: StoredRecord) -> InboxItem:
    target = handle["delivery_target"]
    record = terminal.record
    return InboxItem(
        f"child/{handle['session_id']}/{record.record_id}",
        "child_result",
        {
            "session_id": handle["session_id"],
            "turn_id": handle["turn_id"],
            "operation_id": target["operation_id"],
            "attempt": target["attempt"],
            "result": record.payload["result"],
            "status": record._payload["status"],
            "terminal_seq": terminal.seq,
            "terminal_digest": record.digest,
        },
        target["turn_id"],
        target["generation"],
    )


def child_delivery(
    store: SessionStore, tx: SessionTx, session_id: str, *, hook: Callable[[str], None] = lambda _point: None
) -> int:
    """Push and ack on one transaction; the caller owns commit and optional host writes."""
    batch = tx.consumer_batch(session_id, "child_delivery")
    if batch is None:
        return 0
    created = store.read(session_id, limit=1).records[0].record
    handle = created.payload["attributes"]["child_handle"]
    for terminal in batch.records:
        if terminal.record.kind == "turn_ended" and terminal.record.turn_id == handle["turn_id"]:
            hook("before_parent_inbox")
            tx.push(created.payload["parent_session_id"], completion_input(handle, terminal))
            hook("after_parent_inbox")
    tx.ack(batch)
    hook("after_child_ack")
    return batch.through_seq


def verify_completion(store: SessionStore, parent_id: str, handle: dict[str, Any], item: InboxItem) -> StoredRecord | None:
    handle = next((h for h in child_handles(handle) if h["session_id"] == item.payload["session_id"]), {})
    if not handle:
        return None
    p, target = item.payload, handle["delivery_target"]
    if (
        target["session_id"] != parent_id
        or target["turn_id"] != item.target_turn_id
        or target["operation_id"] != p["operation_id"]
        or target["attempt"] != p["attempt"]
        or handle["session_id"] != p["session_id"]
        or handle["turn_id"] != p["turn_id"]
    ):
        return None
    created = store.read(p["session_id"], limit=1).records[0].record
    if (
        created.payload["parent_session_id"] != parent_id
        or created.payload["parent_operation_id"] != p["operation_id"]
        or created.payload["attributes"].get("child_handle") != handle
    ):
        return None
    page = store.read(p["session_id"], after_seq=p["terminal_seq"] - 1, through_seq=p["terminal_seq"])
    if not page.records:
        return None
    terminal = page.records[0]
    r = terminal.record
    if r.kind != "turn_ended" or r.turn_id != p["turn_id"] or r.digest != p["terminal_digest"]:
        return None
    # Canonical input bytes distinguish JSON values such as true and 1.
    expected = completion_input(handle, terminal)
    if (
        replace(expected, input_id=item.input_id, available_ms=item.available_ms, generation=item.generation).encode()
        != item.encode()
    ):
        return None
    return terminal


def child_outcome(plan: Record, terminal: StoredRecord) -> ToolExecutionResult:
    p = terminal.record.payload
    return ToolExecutionResult(
        tool_call_id=plan.payload["request"]["id"],
        content=p["result"] if isinstance(p["result"], str) else str(p["result"]),
        status_code=ToolResultStatus.SUCCESS if p["status"] == "completed" else ToolResultStatus.ERROR,
        error_code=None if p["status"] == "completed" else f"child_{p['status']}",
        metadata={
            "child_session_id": terminal.record.session_id,
            "child_turn_id": terminal.record.turn_id,
            "status": p["status"],
        },
    )
