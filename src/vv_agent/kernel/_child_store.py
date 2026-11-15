"""Child ledger algorithms over caller-owned storage transactions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, Principal
from vv_agent.kernel._records import InboxItem, Record, SessionSpec
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._state import child_handles as child_handles
from vv_agent.kernel._store import DEFAULT_TENANT_ID, SessionStore, SessionTx, StoredRecord
from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus


def local_content_ref(value: Any, *, slot: str, tenant_id: str = DEFAULT_TENANT_ID) -> dict[str, Any]:
    """Return the local adapter's deterministic S1 reference for clear runtime data."""
    body = canonical_json_bytes(value)
    identity = canonical_json_bytes({"slot": slot, "tenant_id": tenant_id, "value": value})
    return ContentRef(
        content_id="content/local/" + sha256(identity).hexdigest(),
        mac=sha256(body).hexdigest(),
        size=len(body),
        key_id="key/local/" + sha256(tenant_id.encode()).hexdigest(),
        labels=ContentLabels(classification=Classification.INTERNAL),
    ).to_dict()


def input_content_ref(value: Any, *, slot: str, tenant_id: str = DEFAULT_TENANT_ID) -> dict[str, Any]:
    """Normalize runtime content to the S1 reference form accepted by inbox schemas."""
    try:
        ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return local_content_ref(value, slot=slot, tenant_id=tenant_id)
    return value


def is_content_ref(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return False
    return True


def service_principal(tenant_id: str, subject_id: str, *, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a trusted local service identity for inputs without a host-authenticated user."""
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("tenant_id must be a non-empty string")
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a non-empty string")
    value = attributes if attributes is not None else {"role": subject_id}
    return Principal(
        tenant_id=tenant_id,
        subject_id=subject_id,
        subject_kind="service",
        attributes=ContentRef.from_dict(local_content_ref(value, slot=f"principal/{subject_id}/attributes", tenant_id=tenant_id)),
    ).to_dict()


def _store_tenant_id(store: SessionStore) -> str:
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("session store must expose a non-empty tenant binding")
    return tenant_id


@dataclass(frozen=True)
class ChildSession:
    spec: SessionSpec
    content: str
    generation: int = 0
    consumers: tuple[str, ...] = ()
    background: bool = False
    yield_on_steer: bool = False


@dataclass(frozen=True)
class ChildTool:
    """Transactional admission with an optional read-only interruption observer."""

    admit: Callable[[Record], ChildSession | Sequence[ChildSession]]
    on_yield: Callable[[Record, list[dict[str, Any]], str], ToolExecutionResult] | None = None


@dataclass(frozen=True)
class SummaryChildTool:
    """Prepare a logged fresh summary, then admit children on the host transaction.

    The predicate must be pure. Change handler_version when its behavior changes.
    The admission callback receives None when the predicate returns False.
    Keep admission short and transactional; see docs/host-integration.md#transaction-callbacks.
    """

    requires_summary: Callable[[Record], bool]
    admit: Callable[[Record, str | None], ChildSession | Sequence[ChildSession]]
    on_yield: Callable[[Record, list[dict[str, Any]], str], ToolExecutionResult] | None = None


class InvalidChildBatch(ValueError):
    """Custom admission must return a nonempty, uniformly scheduled child batch."""

    code = "invalid_child_batch"


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
                            actor=service_principal(_store_tenant_id(store), "child-control"),
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
    attributes = child.spec.attributes
    if is_content_ref(attributes):
        raise ValueError("child admission attributes require content hydration before adding child_handle")
    attributes = dict(attributes or {}) | {"child_handle": handle}
    tenant_id = child.spec.principal["tenant_id"]
    tx_store = getattr(tx, "store", None)
    if tx_store is not None and _store_tenant_id(tx_store) != tenant_id:
        raise ValueError("child principal must use the parent transaction tenant")
    tx.create(
        replace(
            child.spec,
            parent_session_id=plan.session_id,
            parent_operation_id=plan.operation_id,
            attributes=local_content_ref(
                attributes,
                slot=f"session/{child.spec.session_id}/attributes",
                tenant_id=tenant_id,
            ),
        ),
        consumers=(*child.consumers, "child_delivery"),
    )
    tx.push(
        child.spec.session_id,
        InboxItem(
            "start",
            "user",
            {
                "content": input_content_ref(
                    child.content,
                    slot=f"session/{child.spec.session_id}/input/start",
                    tenant_id=tenant_id,
                )
            },
            generation=child.generation,
            actor=service_principal(tenant_id, "child-admission"),
        ),
    )
    return handle


def completion_input(handle: dict[str, Any], terminal: StoredRecord, *, actor: dict[str, Any] | None = None) -> InboxItem:
    target = handle["delivery_target"]
    record = terminal.record
    actor = actor or service_principal(DEFAULT_TENANT_ID, "child-delivery")
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
        actor=actor,
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
    actor = service_principal(_store_tenant_id(store), "child-delivery")
    for terminal in batch.records:
        if terminal.record.kind == "turn_ended" and terminal.record.turn_id == handle["turn_id"]:
            hook("before_parent_inbox")
            tx.push(created.payload["parent_session_id"], completion_input(handle, terminal, actor=actor))
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
    expected = completion_input(handle, terminal, actor=service_principal(_store_tenant_id(store), "child-delivery"))
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


def ordinary_result_for_children(
    store: SessionStore,
    plan: Record,
    handles: list[dict[str, Any]],
    *,
    first_snapshot: tuple[ExecutionState, tuple[StoredRecord, ...], int] | None = None,
) -> ToolExecutionResult:
    """Project ordinary child terminals, reusing the caller's first snapshot."""
    first = store.read_state(handles[0]["session_id"]) if first_snapshot is None else first_snapshot
    results = []
    for i, handle in enumerate(handles):
        snapshot = first if i == 0 else store.read_state(handle["session_id"])
        terminal = next(r for r in snapshot[1] if r.record.kind == "turn_ended" and r.record.turn_id == handle["turn_id"])
        results.append(child_outcome(plan, terminal))
    if len(results) == 1:
        return results[0]
    return ToolExecutionResult(
        plan._payload["request"]["id"],
        json.dumps([result.to_dict() for result in results]),
        status_code=ToolResultStatus.ERROR
        if any(r.status_code == ToolResultStatus.ERROR for r in results)
        else ToolResultStatus.SUCCESS,
        metadata={"children": [result.metadata for result in results]},
    )
