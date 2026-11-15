"""Child ledger algorithms over caller-owned storage transactions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentRef, Principal
from vv_agent.kernel._records import InboxItem, Record, SessionSpec
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._state import child_handles as child_handles
from vv_agent.kernel._store import SessionStore, SessionTx, StoredRecord
from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus


def is_content_ref(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return False
    return True


def service_principal(tenant_id: str, subject_id: str, *, attributes: ContentRef | dict[str, Any]) -> dict[str, Any]:
    """Build a service identity from a content reference already put by a trusted ingress."""
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("tenant_id must be a non-empty string")
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a non-empty string")
    ref = attributes if isinstance(attributes, ContentRef) else ContentRef.from_dict(attributes)
    return Principal(
        tenant_id=tenant_id,
        subject_id=subject_id,
        subject_kind="service",
        attributes=ref,
    ).to_dict()


def _store_tenant_id(store: SessionStore) -> str:
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("session store must expose a non-empty tenant binding")
    return tenant_id


def _content_writer(tx: SessionTx, *, tenant_id: str, session_id: str):
    from vv_agent.session.content_runtime import SessionContentWriter

    return SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)


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
    tenant_id = _store_tenant_id(store)
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
                    child_id = handle["session_id"]
                    writer = _content_writer(tx, tenant_id=tenant_id, session_id=child_id)
                    actor, attributes_ref = writer.service_principal("child-control")
                    input_id = f"parent-cancel/{sid}/{tid}/{attempt.plan.operation_id}/{attempt.plan.attempt}"
                    item = InboxItem(
                        input_id,
                        "control",
                        {"action": "cancel"},
                        child_tid,
                        child_turn.start._payload["generation"] if child_turn else handle["generation"],
                        actor=actor,
                    )
                    tx.push(child_id, item)
                    writer.link_input_actor(input_id, attributes_ref)
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
    principal = Principal.from_dict(child.spec.principal)
    tenant_id = principal.tenant_id
    tx_store = getattr(tx, "store", None)
    if tx_store is not None and _store_tenant_id(tx_store) != tenant_id:
        raise ValueError("child principal must use the parent transaction tenant")
    parent_writer = _content_writer(tx, tenant_id=tenant_id, session_id=plan.session_id)
    child_writer = _content_writer(tx, tenant_id=tenant_id, session_id=child.spec.session_id)

    def child_spec(writer):
        principal_attributes = parent_writer.get_json(principal.attributes)
        child_principal_attributes = writer.put_json(principal_attributes)
        child_principal = service_principal(
            tenant_id,
            principal.subject_id,
            attributes=child_principal_attributes,
        )
        if is_content_ref(child.spec.workspace):
            workspace_ref = writer.copy_json_from(parent_writer, child.spec.workspace)
        else:
            workspace_ref = writer.put_json(child.spec.workspace)
        attributes_ref = writer.put_json(attributes)
        return replace(
            child.spec,
            principal=child_principal,
            workspace=workspace_ref.to_dict(),
            parent_session_id=plan.session_id,
            parent_operation_id=plan.operation_id,
            attributes=attributes_ref.to_dict(),
        )

    child_writer.create_session(child_spec, consumers=(*child.consumers, "child_delivery"))
    content_ref = child_writer.put_json(child.content)
    actor, actor_attributes = child_writer.service_principal("child-admission")
    input_id = "start"
    tx.push(
        child.spec.session_id,
        InboxItem(
            input_id,
            "user",
            {"content": content_ref.to_dict()},
            generation=child.generation,
            actor=actor,
        ),
    )
    child_writer.link_input_content(input_id, content_ref)
    child_writer.link_input_actor(input_id, actor_attributes)
    return handle


def completion_input(
    handle: dict[str, Any],
    terminal: StoredRecord,
    *,
    actor: dict[str, Any] | None = None,
    result_ref: ContentRef | dict[str, Any] | None = None,
) -> InboxItem:
    target = handle["delivery_target"]
    record = terminal.record
    result = result_ref if result_ref is not None else record.payload["result"]
    if isinstance(result, ContentRef):
        result = result.to_dict()
    return InboxItem(
        f"child/{handle['session_id']}/{record.record_id}",
        "child_result",
        {
            "session_id": handle["session_id"],
            "turn_id": handle["turn_id"],
            "operation_id": target["operation_id"],
            "attempt": target["attempt"],
            "result": result,
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
    stored_created = store.read(session_id, limit=1).records[0]
    created = stored_created.record
    tenant_id = _store_tenant_id(store)
    parent_id = created.payload["parent_session_id"]
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("child delivery requires a parent session")
    child_writer = _content_writer(tx, tenant_id=tenant_id, session_id=session_id)
    parent_writer = _content_writer(tx, tenant_id=tenant_id, session_id=parent_id)
    from vv_agent.session.content_runtime import SESSION_ATTRIBUTES

    attributes_ref = created.payload["attributes"]
    if is_content_ref(attributes_ref):
        child_writer.assert_record(stored_created.seq, attributes_ref, SESSION_ATTRIBUTES)
        attributes = child_writer.get_json(attributes_ref)
    else:
        attributes = attributes_ref
    handle = attributes["child_handle"]
    actor, actor_attributes = parent_writer.service_principal("child-delivery")
    for terminal in batch.records:
        if terminal.record.kind == "turn_ended" and terminal.record.turn_id == handle["turn_id"]:
            hook("before_parent_inbox")
            result = terminal.record.payload["result"]
            if result is None:
                result_ref = None
            elif is_content_ref(result):
                result_ref = parent_writer.copy_json_from(child_writer, result)
            else:
                result_ref = parent_writer.put_json(result)
            item = completion_input(handle, terminal, actor=actor, result_ref=result_ref)
            tx.push(parent_id, item)
            if result_ref is not None:
                parent_writer.link_child_result(item.input_id, result_ref)
            parent_writer.link_input_actor(item.input_id, actor_attributes)
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
    created_stored = store.read(p["session_id"], limit=1).records[0]
    created = created_stored.record
    created_attributes_ref = created.payload["attributes"]
    if is_content_ref(created_attributes_ref):
        from vv_agent.session.content_runtime import SESSION_ATTRIBUTES, read_record_json

        created_attributes = read_record_json(
            store,
            tenant_id=_store_tenant_id(store),
            session_id=p["session_id"],
            record_seq=created_stored.seq,
            ref=created_attributes_ref,
            slot=SESSION_ATTRIBUTES,
        )
    else:
        created_attributes = created_attributes_ref
    if (
        created.payload["parent_session_id"] != parent_id
        or created.payload["parent_operation_id"] != p["operation_id"]
        or created_attributes.get("child_handle") != handle
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
    expected = completion_input(handle, terminal, actor=item.actor, result_ref=item.payload["result"])
    if (
        replace(expected, input_id=item.input_id, available_ms=item.available_ms, generation=item.generation).encode()
        != item.encode()
    ):
        return None
    tenant_id = _store_tenant_id(store)
    with store.atomic() as tx:
        from vv_agent.session.content_runtime import CHILD_RESULT, INPUT_ACTOR_ATTRIBUTES

        parent_writer = _content_writer(tx, tenant_id=tenant_id, session_id=parent_id)
        child_writer = _content_writer(tx, tenant_id=tenant_id, session_id=p["session_id"])
        if item.actor is None:
            return None
        parent_writer.assert_input(item.input_id, item.actor["attributes"], INPUT_ACTOR_ATTRIBUTES)
        result_ref = item.payload["result"]
        if result_ref is not None:
            parent_writer.assert_input(item.input_id, result_ref, CHILD_RESULT)
            delivered_result = parent_writer.get_json(result_ref)
            terminal_result = r.payload["result"]
            if is_content_ref(terminal_result):
                terminal_result = child_writer.get_json(terminal_result)
            if canonical_json_bytes(delivered_result) != canonical_json_bytes(terminal_result):
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
