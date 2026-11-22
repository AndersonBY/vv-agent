"""Child ledger algorithms over caller-owned storage transactions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from vv_agent.kernel._boundary_hydration import SessionHydration
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    CHILD_RESULT,
    INPUT_ACTOR_ATTRIBUTES,
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentBindingMismatch,
    ContentRef,
    ContentSessionWriter,
    ContentUnavailable,
    Principal,
)
from vv_agent.kernel._records import InboxItem, Record, SessionSpec
from vv_agent.kernel._requirements import normalize_requirements
from vv_agent.kernel._sql import SessionNotLive
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._state import child_handles as child_handles
from vv_agent.kernel._store import ContentSessionTx, SessionStore, SessionTx, StoredRecord
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


def _bound_actor(
    writer: ContentSessionWriter,
    actor: Principal | Mapping[str, Any] | None,
    *,
    subject_id: str,
) -> tuple[dict[str, Any], ContentRef]:
    if actor is None:
        return writer.service_principal(subject_id)
    persisted = isinstance(actor, Principal)
    payload = actor.to_dict() if persisted else dict(actor)
    if set(payload) != {"tenant_id", "subject_id", "subject_kind", "attributes"}:
        raise ValueError("input actor must be a complete Principal")
    if payload["tenant_id"] != writer.tenant_id:
        raise ValueError("input actor must use the session tenant")
    if not isinstance(payload["subject_id"], str) or not payload["subject_id"]:
        raise ValueError("input actor subject_id must be a non-empty string")
    if payload["subject_kind"] not in {"user", "service", "agent"}:
        raise ValueError("input actor subject_kind must be user, service or agent")
    attributes = payload["attributes"]
    if persisted:
        ref = actor.attributes
        try:
            verified = writer.assert_record(1, ref, SESSION_PRINCIPAL_ATTRIBUTES)
        except ContentUnavailable as exc:
            raise ValueError("persisted child actor attributes require the parent principal owner link") from exc
        ref = writer.copy_json_from(writer, verified)
    elif isinstance(attributes, ContentRef):
        raise ValueError("host ingress actor attributes must be a plaintext mapping")
    elif isinstance(attributes, Mapping):
        ref = writer.put_json(dict(attributes))
    else:
        raise ValueError("host ingress actor attributes must be a plaintext mapping")
    return payload | {"attributes": ref.to_dict()}, ref


def _store_tenant_id(store: SessionStore) -> str:
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("session store must expose a non-empty tenant binding")
    return tenant_id


def _content_writer(tx: SessionTx, *, tenant_id: str, session_id: str):
    if not isinstance(tx, ContentSessionTx):
        raise ValueError("trusted child content requires an explicit transaction capability")
    return tx.content_writer(tenant_id=tenant_id, session_id=session_id)


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
    if child.spec.parent_session_id not in {None, plan.session_id}:
        raise ValueError("child spec parent must match the admitting parent")
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
    attributes = dict(child.spec.attributes or {}) | {"child_handle": handle}
    principal = Principal.from_dict(child.spec.principal)
    tenant_id = principal.tenant_id
    tx_store = getattr(tx, "store", None)
    if tx_store is None:
        tx_store = getattr(getattr(tx, "_tx", None), "store", None)
    if tx_store is not None and _store_tenant_id(tx_store) != tenant_id:
        raise ValueError("child principal must use the parent transaction tenant")
    if tx_store is not None:
        parent_page = tx_store.read(plan.session_id, limit=1)
        if not parent_page.records:
            raise ValueError("child admission requires a durable parent session")
        parent_created = parent_page.records[0].record
        if parent_created.feature_level == 2 and (child.spec.feature_level != 2 or child.spec.bootstrap is None):
            raise ValueError("child admission in a level-2 parent requires an explicit trusted child bootstrap")
    requirements = None
    if child.spec.feature_level == 2:
        if child.spec.bootstrap is None:
            raise ValueError("level-2 child admission requires bootstrap requirements")
        requirements = normalize_requirements(child.spec.bootstrap)
    parent_writer = _content_writer(tx, tenant_id=tenant_id, session_id=plan.session_id)
    child_writer = _content_writer(tx, tenant_id=tenant_id, session_id=child.spec.session_id)
    require_child_creation = getattr(child_writer, "_require_child_creation", None)
    if require_child_creation is not None:
        require_child_creation()

    def child_spec(writer):
        source_principal_attributes = parent_writer.assert_record(1, principal.attributes, SESSION_PRINCIPAL_ATTRIBUTES)
        child_principal_attributes = writer.copy_json_from(parent_writer, source_principal_attributes)
        child_principal = Principal(
            tenant_id=tenant_id,
            subject_id=principal.subject_id,
            subject_kind=principal.subject_kind,
            attributes=child_principal_attributes,
        ).to_dict()
        if is_content_ref(child.spec.workspace):
            source_workspace = parent_writer.assert_record(1, child.spec.workspace, SESSION_WORKSPACE)
            workspace_ref = writer.copy_json_from(parent_writer, source_workspace)
        else:
            workspace_ref = writer.put_json(child.spec.workspace)
        return replace(
            child.spec,
            principal=child_principal,
            workspace=workspace_ref.to_dict(),
            parent_session_id=plan.session_id,
            parent_operation_id=plan.operation_id,
            attributes=attributes,
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
            feature_level=2 if requirements is not None else 1,
            requirements=requirements,
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
    store: SessionStore,
    tx: SessionTx,
    session_id: str,
    *,
    hook: Callable[[str], None] = lambda _point: None,
    actor: Principal | Mapping[str, Any] | None = None,
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

    attributes_ref = created.payload["attributes"]
    if is_content_ref(attributes_ref):
        child_writer.assert_record(stored_created.seq, attributes_ref, SESSION_ATTRIBUTES)
        attributes = child_writer.get_json(attributes_ref)
    else:
        attributes = attributes_ref
    handle = attributes["child_handle"]
    result_actor, actor_attributes = _bound_actor(parent_writer, actor, subject_id="child-delivery")
    for terminal in batch.records:
        if terminal.record.kind == "turn_ended" and terminal.record.turn_id == handle["turn_id"]:
            hook("before_parent_inbox")
            result = terminal.record.payload["result"]
            if result is None:
                result_ref = None
            else:
                if not is_content_ref(result):
                    raise ValueError("child terminal result requires a ContentRef")
                child_writer.assert_record(terminal.seq, result, "turn_ended_result")
                result_ref = parent_writer.copy_json_from(child_writer, result)
            item = completion_input(handle, terminal, actor=result_actor, result_ref=result_ref)
            tx.push(parent_id, item)
            if result_ref is not None:
                parent_writer.link_child_result(item.input_id, result_ref)
            parent_writer.link_input_actor(item.input_id, actor_attributes)
            hook("after_parent_inbox")
    tx.ack(batch)
    hook("after_child_ack")
    return batch.through_seq


@dataclass(frozen=True)
class ChildCompletion:
    session_id: str
    turn_id: str
    status: str
    result: Any
    terminal_seq: int
    terminal_digest: str
    from_parent_copy: bool


class ChildCompletionPending:
    """The child is fenced, but its complete tombstone is not available yet."""


PENDING_CHILD_COMPLETION = ChildCompletionPending()
_UNHYDRATED_RESULT = object()
_UNVERIFIED = object()


def _child_is_deleted(store: SessionStore, session_id: str) -> bool:
    checker = getattr(store, "child_is_deleted", None)
    return checker(session_id) is True if callable(checker) else False


def _verified_parent_copy(store: SessionStore, parent_id: str, item: InboxItem) -> Any:
    if item.actor is None:
        return _UNVERIFIED
    tenant_id = _store_tenant_id(store)
    if item.actor["tenant_id"] != tenant_id:
        return _UNVERIFIED
    with store.atomic() as tx:
        parent_writer = _content_writer(tx, tenant_id=tenant_id, session_id=parent_id)
        try:
            parent_writer.assert_input(item.input_id, item.actor["attributes"], INPUT_ACTOR_ATTRIBUTES)
            result_ref = item.payload["result"]
            if result_ref is None:
                return None
            parent_writer.assert_input(item.input_id, result_ref, CHILD_RESULT)
            return parent_writer.get_json(result_ref)
        except (ContentBindingMismatch, ContentUnavailable):
            return _UNVERIFIED


def _matched_completion_handle(parent_id: str, handles: dict[str, Any], item: InboxItem) -> dict[str, Any] | None:
    p = item.payload
    handle = next((h for h in child_handles(handles) if h["session_id"] == p["session_id"]), None)
    if handle is None:
        return None
    target = handle["delivery_target"]
    if (
        target["session_id"] != parent_id
        or target["turn_id"] != item.target_turn_id
        or target["generation"] != item.generation
        or target["operation_id"] != p["operation_id"]
        or target["attempt"] != p["attempt"]
        or handle["session_id"] != p["session_id"]
        or handle["turn_id"] != p["turn_id"]
    ):
        return None
    return handle


def _verify_live_completion(
    store: SessionStore,
    parent_id: str,
    handle: dict[str, Any],
    item: InboxItem,
) -> ChildCompletion | None:
    p = item.payload
    matched_handle = _matched_completion_handle(parent_id, handle, item)
    if matched_handle is None:
        return None
    created_stored = store.read(p["session_id"], limit=1).records[0]
    created = created_stored.record
    created_attributes_ref = created.payload["attributes"]
    if is_content_ref(created_attributes_ref):
        with store.atomic() as tx:
            writer = _content_writer(tx, tenant_id=_store_tenant_id(store), session_id=p["session_id"])
            writer.assert_record(created_stored.seq, created_attributes_ref, SESSION_ATTRIBUTES)
            created_attributes = writer.get_json(created_attributes_ref)
    else:
        created_attributes = created_attributes_ref
    if (
        created.payload["parent_session_id"] != parent_id
        or created.payload["parent_operation_id"] != p["operation_id"]
        or created_attributes.get("child_handle") != matched_handle
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
    expected = completion_input(matched_handle, terminal, actor=item.actor, result_ref=item.payload["result"])
    if (
        replace(
            expected,
            input_id=item.input_id,
            available_ms=item.available_ms,
            generation=item.generation,
            admission_proof=None,
        ).encode()
        != replace(item, admission_proof=None).encode()
    ):
        return None
    tenant_id = _store_tenant_id(store)
    delivered_result = _verified_parent_copy(store, parent_id, item)
    if delivered_result is _UNVERIFIED:
        return None
    with store.atomic() as tx:
        child_writer = _content_writer(tx, tenant_id=tenant_id, session_id=p["session_id"])
        result_ref = item.payload["result"]
        if result_ref is not None:
            terminal_result = r.payload["result"]
            if not is_content_ref(terminal_result):
                return None
            child_writer.assert_record(terminal.seq, terminal_result, "turn_ended_result")
            terminal_result = child_writer.get_json(terminal_result)
            if canonical_json_bytes(delivered_result) != canonical_json_bytes(terminal_result):
                return None
    return ChildCompletion(
        session_id=p["session_id"],
        turn_id=p["turn_id"],
        status=r._payload["status"],
        result=delivered_result,
        terminal_seq=terminal.seq,
        terminal_digest=r.digest,
        from_parent_copy=False,
    )


def _verify_parent_completion(
    store: SessionStore,
    parent_id: str,
    handle: dict[str, Any],
    item: InboxItem,
) -> ChildCompletion | None:
    p = item.payload
    if _matched_completion_handle(parent_id, handle, item) is None:
        return None
    delivered_result = _verified_parent_copy(store, parent_id, item)
    if delivered_result is _UNVERIFIED:
        return None
    return ChildCompletion(
        session_id=p["session_id"],
        turn_id=p["turn_id"],
        status=p["status"],
        result=delivered_result,
        terminal_seq=p["terminal_seq"],
        terminal_digest=p["terminal_digest"],
        from_parent_copy=True,
    )


def verify_completion(
    store: SessionStore,
    parent_id: str,
    handle: dict[str, Any],
    item: InboxItem,
) -> ChildCompletion | ChildCompletionPending | None:
    """Verify a live child terminal or its already-admitted parent-owned copy."""

    if _child_is_deleted(store, item.payload["session_id"]):
        return _verify_parent_completion(store, parent_id, handle, item)
    try:
        return _verify_live_completion(store, parent_id, handle, item)
    except KeyError:
        return (
            _verify_parent_completion(store, parent_id, handle, item)
            if _child_is_deleted(store, item.payload["session_id"])
            else None
        )
    except SessionNotLive:
        # A pending tombstone fences the child before completion. Keep the parent
        # inbox recoverable; the complete tombstone switches this to the parent copy.
        return PENDING_CHILD_COMPLETION


def _hydrate_terminal_result(store: SessionStore, terminal: StoredRecord) -> Any:
    """Hydrate only the terminal result through its record-owner content link."""

    result = terminal.record._payload["result"]
    if not is_content_ref(result):
        return result
    ref = ContentRef.from_dict(result)
    hydration = SessionHydration.from_store(
        store,
        tenant_id=_store_tenant_id(store),
        session_id=terminal.record.session_id,
    )
    return hydration.read(
        f"record:{terminal.seq}",
        "turn_ended_result",
        ref,
        record_kind=terminal.record.kind,
        record_payload=terminal.record._payload,
    )


def _child_outcome(
    plan: Record,
    *,
    session_id: str,
    turn_id: str,
    status: str,
    result: Any,
) -> ToolExecutionResult:
    if is_content_ref(result):
        raise ValueError("child turn result requires content hydration")
    return ToolExecutionResult(
        tool_call_id=plan.payload["request"]["id"],
        content=result if isinstance(result, str) else str(result),
        status_code=ToolResultStatus.SUCCESS if status == "completed" else ToolResultStatus.ERROR,
        error_code=None if status == "completed" else f"child_{status}",
        metadata={
            "child_session_id": session_id,
            "child_turn_id": turn_id,
            "status": status,
        },
    )


def child_outcome(
    plan: Record,
    terminal: StoredRecord,
    *,
    hydrated_result: Any = _UNHYDRATED_RESULT,
) -> ToolExecutionResult:
    p = terminal.record.payload
    result = p["result"] if hydrated_result is _UNHYDRATED_RESULT else hydrated_result
    turn_id = terminal.record.turn_id
    assert turn_id is not None
    return _child_outcome(
        plan,
        session_id=terminal.record.session_id,
        turn_id=turn_id,
        status=p["status"],
        result=result,
    )


def completion_outcome(plan: Record, completion: ChildCompletion) -> ToolExecutionResult:
    return _child_outcome(
        plan,
        session_id=completion.session_id,
        turn_id=completion.turn_id,
        status=completion.status,
        result=completion.result,
    )


def _result_from_outcomes(plan: Record, results: Sequence[ToolExecutionResult]) -> ToolExecutionResult:
    if len(results) == 1:
        return results[0]
    return ToolExecutionResult(
        plan._payload["request"]["id"],
        json.dumps([result.to_dict() for result in results]),
        status_code=ToolResultStatus.ERROR
        if any(result.status_code == ToolResultStatus.ERROR for result in results)
        else ToolResultStatus.SUCCESS,
        metadata={"children": [result.metadata for result in results]},
    )


def result_from_completions(plan: Record, completions: Sequence[ChildCompletion]) -> ToolExecutionResult:
    """Project admitted parent copies without reading a live child session."""

    return _result_from_outcomes(plan, [completion_outcome(plan, completion) for completion in completions])


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
        results.append(child_outcome(plan, terminal, hydrated_result=_hydrate_terminal_result(store, terminal)))
    return _result_from_outcomes(plan, results)
