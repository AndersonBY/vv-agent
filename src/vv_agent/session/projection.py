"""Pure durable-record -> existing RunEvent projection. Never executes or acknowledges."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from vv_agent.budget import BudgetEnforcementBoundary, BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.events import (
    AgentStartedEvent,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    BudgetExhaustedEvent,
    BudgetSnapshotEvent,
    CycleStartedEvent,
    DiagnosticEvent,
    HandoffCompletedEvent,
    HandoffStartedEvent,
    HostInteractionRequestedEvent,
    HostInteractionResponseConsumedEvent,
    ModelCallCompletedEvent,
    ModelCallFailedEvent,
    ModelCallStartedEvent,
    ModelRetryDuplicateRiskEvent,
    OperationAmbiguousEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    RunStateChangedEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallPlannedEvent,
    ToolCallStartedEvent,
    event_from_dict,
)
from vv_agent.interaction import HostInteractionRequest
from vv_agent.kernel._canonical_json import canonical_json_bytes, validate_sha256
from vv_agent.kernel._content import (
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentBindingMismatch,
    ContentRef,
    ContentStoreError,
    ContentUnavailable,
)
from vv_agent.kernel._content import (
    ContentIntegrityError as BackendContentIntegrityError,
)
from vv_agent.kernel._content import (
    KeyUnavailable as BackendKeyUnavailable,
)
from vv_agent.kernel._content_publication import (
    _MISSING,
    _REFERENCE_DIGESTS,
    NestedContentSlot,
    _content_slot_specs,
    _expand_nested_slots,
    _payload_slots,
    declared_content_ref,
    input_content_slots,
)
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import (
    ContentReader,
    HydrationError,
    HydrationSlot,
    IntegrityFailure,
    KeyUnavailable,
    MissingContent,
    MissingContentLink,
    ReferenceDigestMismatch,
    SessionHydrator,
    TypedSchemaFailure,
    _parse_definition,
    reference_digest,
)
from vv_agent.kernel._reducer import logical_inputs
from vv_agent.types import ModelCallOperation, ToolExecutionResult

from .children import child_handles
from .content_reader import input_owner, record_owner
from .content_runtime import durable_content_reader
from .interactions import user_interaction
from .records import InboxItem, copy_json, digest
from .runtime import model_usage
from .store import StoredRecord
from .tool_results import project_tool_result


@dataclass(frozen=True, slots=True)
class SessionHydration:
    """One caller-owned reader bound to a tenant and session."""

    reader: ContentReader = field(repr=False)
    tenant_id: str
    session_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.reader, ContentReader):
            raise TypeError("reader must implement ContentReader")

    @classmethod
    def from_store(cls, store: Any, *, tenant_id: str, session_id: str) -> SessionHydration:
        return cls(
            durable_content_reader(store, tenant_id=tenant_id, session_id=session_id),
            tenant_id=tenant_id,
            session_id=session_id,
        )

    def read(
        self,
        owner: str,
        slot: str,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
        record_kind: str | None = None,
        record_payload: dict[str, Any] | None = None,
    ) -> Any:
        hydrator = SessionHydrator(
            self.reader,
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            owner=owner,
        )
        try:
            selected = HydrationSlot(slot)
        except ValueError:
            selected = None
        nested = _nested_content_slots(record_kind, slot, record_payload)
        if nested:
            value = self._read_exact_json(
                hydrator,
                owner,
                slot,
                ref,
                expected_reference_digest=expected_reference_digest,
            )
            for path, nested_slot, nested_value in _expand_nested_slots(value, nested, slot):
                if nested_value is _MISSING or nested_value is None:
                    continue
                nested_ref = declared_content_ref(nested_value)
                if nested_ref is None:
                    raise TypedSchemaFailure(f"{nested_slot} requires a ContentRef")
                _path_set(
                    value,
                    path,
                    self.read(
                        owner,
                        nested_slot,
                        nested_ref,
                        record_kind=record_kind,
                        record_payload=record_payload,
                    ),
                )
            if selected is HydrationSlot.DEFINITION:
                value = _parse_definition(value)
        elif selected is None:
            value = self._read_exact_json(
                hydrator,
                owner,
                slot,
                ref,
                expected_reference_digest=expected_reference_digest,
            )
        else:
            value = hydrator.hydrate(selected, ref, expected_reference_digest=expected_reference_digest)
        return _projection_value(slot, value)

    def _read_exact_json(
        self,
        hydrator: SessionHydrator,
        owner: str,
        slot: str,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> Any:
        if hydrator.binding != (self.tenant_id, self.session_id, owner):
            raise MissingContentLink("content hydrator owner binding does not match")
        if expected_reference_digest is not None:
            try:
                expected_reference_digest = validate_sha256(
                    expected_reference_digest,
                    f"{slot} expected reference digest",
                )
            except ValueError as exc:
                raise TypedSchemaFailure(str(exc)) from exc
            if not hmac.compare_digest(reference_digest(ref), expected_reference_digest):
                raise ReferenceDigestMismatch(f"{slot} reference digest does not match")
        try:
            body = self.reader.read(self.tenant_id, self.session_id, owner, slot, ref)
        except (HydrationError, MissingContentLink, MissingContent, KeyUnavailable, IntegrityFailure):
            raise
        except ContentUnavailable as exc:
            raise MissingContent("content is unavailable") from exc
        except BackendKeyUnavailable as exc:
            raise KeyUnavailable("session content key is unavailable") from exc
        except BackendContentIntegrityError as exc:
            raise IntegrityFailure("content failed integrity validation") from exc
        except ContentBindingMismatch as exc:
            raise MissingContentLink("content owner link is unavailable") from exc
        except ContentStoreError as exc:
            raise HydrationError("content read failed") from exc
        if type(body) is not bytes:
            raise IntegrityFailure(f"{slot} reader returned a non-bytes value")
        if len(body) != ref.size:
            raise IntegrityFailure(f"{slot} size does not match ContentRef")
        try:
            value = json.loads(body)
            if canonical_json_bytes(value, slot) != body:
                raise ValueError("content is not canonical JSON")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypedSchemaFailure(f"{slot} is not valid canonical JSON") from exc
        return value


def hydration_for_store(store: Any, *, tenant_id: str, session_id: str) -> SessionHydration:
    return SessionHydration.from_store(store, tenant_id=tenant_id, session_id=session_id)


def hydration_for_session_store(store: Any, session_id: str) -> SessionHydration:
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("session store requires a tenant binding for content hydration")
    return hydration_for_store(store, tenant_id=tenant_id, session_id=session_id)


class _RetainedInboxItem(InboxItem):
    """Logical input fields with the committed S1 encoding used for identity checks."""

    def __init__(self, item: InboxItem, body: bytes) -> None:
        super().__init__(**item.to_dict())
        object.__setattr__(self, "_retained_body", body)

    def encode(self) -> bytes:
        return self._retained_body


def _projection_value(slot: str, value: Any) -> Any:
    if slot == HydrationSlot.DEFINITION.value:
        return {
            "agent_name": value.agent_name,
            "memory_settings": copy_json(value.memory_settings),
            "child_tools": list(value.child_tools),
            "model_binding": copy_json(value.model_binding),
            "capabilities": copy_json(value.capabilities),
            "task": value.task.to_dict(),
            "tools": copy_json(list(value.tools)),
        }
    if slot == HydrationSlot.BUDGET.value:
        return value.to_dict()
    if slot == HydrationSlot.TASK.value:
        return value.to_dict()
    if slot == HydrationSlot.MESSAGES.value:
        return [message.to_dict() for message in value]
    return copy_json(value)


def hydrate_input_payload(item: InboxItem, hydration: SessionHydration | None) -> dict[str, Any]:
    payload = copy_json(item.payload)
    for content_slot in input_content_slots(item.kind):
        try:
            value = _path_get(payload, content_slot.path)
        except (KeyError, IndexError):
            continue
        ref = declared_content_ref(value)
        if ref is None:
            continue
        if hydration is None:
            raise MissingContentLink(f"input:{item.input_id}/{content_slot.slot} requires a content hydration reader")
        _path_set(payload, content_slot.path, hydration.read(input_owner(item.input_id), content_slot.slot, ref))
    return payload


def hydrate_input(item: InboxItem, hydration: SessionHydration | None) -> InboxItem:
    return replace(item, payload=hydrate_input_payload(item, hydration))


def hydrate_record_payload(
    stored: StoredRecord,
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
) -> dict[str, Any]:
    """Hydrate every declared C slot without changing the committed record envelope."""

    record = stored.record
    source = copy_json(record._payload)
    payload = copy_json(source)
    owner = record_owner(stored.seq)
    declared_slots = list(_payload_slots(record.kind, source))
    if record.kind == "session_created":
        declared_slots.extend(
            (
                (("principal", "attributes"), SESSION_PRINCIPAL_ATTRIBUTES, None),
                (("workspace",), SESSION_WORKSPACE, None),
                (("attributes",), SESSION_ATTRIBUTES, None),
            )
        )
    for path, slot, _value in declared_slots:
        if not include_input and record.kind == "input_applied" and path[:1] == ("input",):
            continue
        try:
            value = _path_get(payload, path)
        except (KeyError, IndexError):
            continue
        if value is None:
            continue
        ref = declared_content_ref(value)
        if ref is None:
            continue
        if hydration is None:
            raise MissingContentLink(f"{owner}/{slot} requires a content hydration reader")
        if record.session_id != hydration.session_id:
            raise MissingContentLink("record session does not match the content reader binding")
        expected = _REFERENCE_DIGESTS.get((record.kind, slot))
        _path_set(
            payload,
            path,
            hydration.read(
                owner,
                slot,
                ref,
                expected_reference_digest=payload.get(expected) if expected is not None else None,
                record_kind=record.kind,
                record_payload=source,
            ),
        )
    return payload


def hydrate_stored_record(
    stored: StoredRecord,
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
) -> StoredRecord:
    payload = hydrate_record_payload(stored, hydration, include_input=include_input)
    retained_body = stored.record.encode()
    record = replace(stored.record, payload=payload)
    object.__setattr__(record, "_body", retained_body)
    if record.kind == "turn_started":
        task_data = copy_json(payload["definition"]["task"])
        object.__setattr__(record, "_task", lambda: AgentTask.from_dict(copy_json(task_data)))
    return replace(stored, record=record)


def hydrate_records(
    records: Iterable[StoredRecord],
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
) -> tuple[StoredRecord, ...]:
    return tuple(hydrate_stored_record(stored, hydration, include_input=include_input) for stored in records)


def fold_hydrated_records(
    records: tuple[StoredRecord, ...],
    source_records: tuple[StoredRecord, ...],
    fold_factory: Callable[[], Any],
    hydration: SessionHydration | None,
) -> Any:
    """Fold hydrated values while retaining the committed record encoding for dedupe."""

    source_by_identity = {(row.record.record_id, row.seq): row for row in source_records}
    hydrated_by_identity = {}
    for row in records:
        identity = (row.record.record_id, row.seq)
        source = source_by_identity.get(identity)
        if source is None:
            raise MissingContentLink("hydrated record identity does not match source")
        if identity in hydrated_by_identity:
            raise MissingContentLink("hydrated record identity is duplicated")
        hydrated_by_identity[identity] = row

    fold_records = []
    input_rows = []
    for source in source_records:
        identity = (source.record.record_id, source.seq)
        row = hydrated_by_identity.get(identity)
        if row is None:
            row = hydrate_stored_record(source, hydration)
        elif row.record.encode() != source.record.encode():
            raise MissingContentLink("hydrated record body does not match source")
        if source.record.kind == "input_applied":
            logical_payload = copy_json(row.record._payload)
            input_rows.append((source, row))
            fold_payload = copy_json(logical_payload)
            fold_payload["input"] = copy_json(source.record._payload["input"])
            fold_record = replace(row.record, payload=fold_payload)
            object.__setattr__(fold_record, "_body", source.record.encode())
            object.__setattr__(fold_record, "_business_payload", logical_payload)
            row = replace(row, record=fold_record)
        fold_records.append(row)

    logical_items = {}
    hydrated_inputs = {}
    for source, row in input_rows:
        committed = InboxItem(**source.record._payload["input"])
        logical = InboxItem(**row.record._payload["input"])
        logical_items[row.record.record_id] = _RetainedInboxItem(logical, committed.encode())
        hydrated_inputs[row.record.record_id] = row
    with logical_inputs(logical_items):
        state = fold_factory().extend(
            (row.record for row in fold_records),
            consumed_inputs=logical_items.values(),
            bodies=(row.record.encode() for row in source_records),
        )
    state.applied_inputs = {
        input_id: hydrated_inputs[record.record_id].record if record.record_id in hydrated_inputs else record
        for input_id, record in state.applied_inputs.items()
    }
    for operation in state.operations.values():
        for attempt in operation.attempts.values():
            if attempt.approval_answer is not None:
                hydrated = hydrated_inputs.get(attempt.approval_answer.record_id)
                if hydrated is not None:
                    attempt.approval_answer = hydrated.record
    return state


def _path_part(part: str | int) -> str | int:
    return int(part) if isinstance(part, str) and part.isdecimal() else part


def _nested_content_slots(
    kind: str | None,
    slot: str,
    payload: dict[str, Any] | None,
) -> tuple[NestedContentSlot, ...]:
    if kind is None or payload is None:
        return ()
    return next((content_slot.nested for content_slot in _content_slot_specs(kind, payload) if content_slot.slot == slot), ())


def _path_get(value: Any, path: tuple[str | int, ...]) -> Any:
    for part in path:
        value = value[_path_part(part)]
    return value


def _path_set(value: Any, path: tuple[str | int, ...], replacement: Any) -> None:
    if not path:
        raise ValueError("content slot path cannot be empty")
    parent = _path_get(value, path[:-1]) if len(path) > 1 else value
    parent[_path_part(path[-1])] = replacement


def project_records(records: Iterable[StoredRecord], *, hydration: SessionHydration | None = None) -> list[RunEvent]:
    """Pass a prefix including plans; callers filter emitted metadata.session_seq by cursor."""
    plans, events, turns = {}, [], {}
    delegations: dict[tuple[str, int], dict] = {}
    interactions: dict[tuple[str, int], HostInteractionRequest] = {}
    compactions: dict[str, str | None] = {}
    for stored in records:
        r = stored.record
        p = hydrate_record_payload(stored, hydration)
        identity_digest = digest([r.session_id, r.record_id])
        common: dict[str, Any] = {
            "run_id": r.turn_id or r.session_id,
            "trace_id": r.turn_id or r.session_id,
            "session_id": r.session_id,
            "event_id": f"sk/{identity_digest}/event",
            "created_at": stored.created_ms / 1000,
            "metadata": {"session_seq": stored.seq},
        }
        if r.turn_id in turns:
            common["agent_name"] = turns[r.turn_id]["definition"].get("agent_name")
        event: RunEvent | None = None
        if r.kind in {"op_unknown", "input_applied"} and p.get("reason") == "provider_result_invalid":
            invalid_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/provider_result_invalid"}
            events.append(
                DiagnosticEvent(
                    **invalid_common,
                    level="warning",
                    code="provider_result_invalid",
                    details={
                        "operation_id": r.operation_id or p.get("target_operation_id"),
                        "attempt": r.attempt or p.get("input", {}).get("payload", {}).get("attempt"),
                    },
                )
            )
        if r.kind == "input_applied" and p["reason"] == "host compaction request" and p["disposition"] in {"queued", "applied"}:
            compactions[p["input"]["input_id"]] = r.turn_id
        if r.kind == "turn_started":
            for key in p["input_ids"]:
                if key in compactions:
                    compactions[key] = r.turn_id
            turns[r.turn_id] = p
            common["agent_name"] = p["definition"]["task"].get("metadata", {}).get("agent_name")
            if p["definition"]["task"].get("metadata", {}).get("vv_session", {}).get("input_blocked"):
                continue
            event = RunStartedEvent(**common, input=p["definition"]["task"]["user_prompt"])
            events.append(event)
            agent_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/agent"}
            event = AgentStartedEvent(**agent_common)
        elif r.kind == "turn_ended":
            remaining = [key for key, tid in compactions.items() if tid == r.turn_id]
            if remaining:
                failure_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/compaction_failed"}
                events.append(
                    DiagnosticEvent(
                        **failure_common,
                        level="warning",
                        code="context_compaction_failed",
                        details={"request_ids": remaining, "error": p["reason"] or p["status"]},
                    )
                )
            if p["status"] == "completed":
                completed_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                events.append(
                    DiagnosticEvent(
                        **completed_common,
                        level="info",
                        code="run_completed",
                        details={"final_answer": p["result"], "completion_reason": p["reason"] or "no_tool_finish"},
                    )
                )
                event = RunCompletedEvent(**common, final_output=p["result"], status="completed")
            elif p["status"] in {"cancelled", "aborted"}:
                event = RunCancelledEvent(**common, reason=p["reason"] or p["status"])
            else:
                error = p["reason"] or "failed"
                if p["reason"] == "model_request_rejected":
                    error = f"{error}: {p['result']}"
                event = RunFailedEvent(**common, error=error, status="failed")
        elif r.kind == "boundary_recorded":
            data, stage = p["data"], p["stage"]
            if stage in {"memory_started", "memory_completed"}:
                if stage == "memory_completed":
                    for key in data["event"].get("metadata", {}).get("requested_by", []):
                        compactions.pop(key, None)
                event = event_from_dict(data["event"] | {"metadata": data["event"].get("metadata", {}) | common["metadata"]})
            elif stage == "after_cycle":
                code = (
                    "after_cycle_failed"
                    if data["error"]
                    else "after_cycle_steered"
                    if data["action"] == "steer"
                    else "after_cycle_decision"
                )
                event = DiagnosticEvent(
                    **common, level="info", code=code, details={k: data[k] for k in ("action", "disallow_tools", "error")}
                )
            elif stage == "budget":
                exhaustion = BudgetExhaustion.from_dict(data["exhaustion"]) if data["exhaustion"] else None
                boundary = (
                    exhaustion.enforcement_boundary
                    if exhaustion
                    else (
                        BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE
                        if "/tool_complete/" in p["boundary_id"]
                        else BudgetEnforcementBoundary.TOOL_BATCH_PREFLIGHT
                        if p["boundary_id"].endswith("/tool_batch")
                        else BudgetEnforcementBoundary.TERMINAL
                        if p["boundary_id"] == "terminal"
                        else BudgetEnforcementBoundary.MODEL_CALL_COMPLETE
                        if "/complete/" in p["boundary_id"]
                        else BudgetEnforcementBoundary.RUN_START
                        if p["boundary_id"] == "run_start"
                        else BudgetEnforcementBoundary.CYCLE_START
                    )
                )
                kwargs: dict[str, Any] = common | {
                    "enforcement_boundary": boundary,
                    "budget_usage": BudgetUsageSnapshot.from_dict(data["usage"]),
                }
                event = (
                    BudgetExhaustedEvent(**kwargs, budget_exhaustion=exhaustion) if exhaustion else BudgetSnapshotEvent(**kwargs)
                )
            elif stage == "session_memory_saved":
                event = DiagnosticEvent(
                    **common, level="info", code="session_memory_saved", details={"entry_count": len(data["state"]["entries"])}
                )
        elif r.kind == "turn_parked":
            common["metadata"]["interaction"] = user_interaction(r.session_id, r.turn_id, turns[r.turn_id]["generation"], p)
            event = RunStateChangedEvent(**common, state="wait_user")
        elif r.kind == "input_applied" and p["disposition"] == "applied":
            item = p["input"]
            if item["kind"] == "control":
                action = item["payload"]["action"]
                event = RunStateChangedEvent(
                    **common,
                    state=action,
                    cancel_requested={"from": False, "to": True} if action in {"cancel", "abort", "close"} else None,
                )
            elif item["kind"] == "approval_answer":
                answer = item["payload"]
                plan = plans[(answer["operation_id"], answer["attempt"])]
                approval_common: dict[str, Any] = common | {
                    "metadata": common["metadata"]
                    | {"reason": answer.get("reason", ""), "decision_metadata": answer.get("metadata", {})}
                }
                event = ApprovalResolvedEvent(
                    **approval_common,
                    request_id=answer["request_id"],
                    tool_name=plan["request"]["name"],
                    tool_call_id=plan["request"]["id"],
                    action="allow" if answer["decision"] == "approve" else answer["decision"],
                )
            elif item["kind"] == "child_result":
                answer = item["payload"]
                parent_plan = plans[(answer["operation_id"], answer["attempt"])]
                marker = delegations.get((answer["operation_id"], answer["attempt"]), {})
                if marker.get("mode") == "handoff":
                    handoff_common = {k: v for k, v in common.items() if k != "agent_name"}
                    handoff_common["metadata"] = common["metadata"] | marker["metadata"]
                    event = HandoffCompletedEvent(
                        **handoff_common,
                        source_agent=common.get("agent_name") or "agent",
                        target_agent=marker["agent_name"],
                        tool_call_id=parent_plan["request"]["id"],
                        child_session_id=answer["session_id"],
                        child_run_id=answer["turn_id"],
                        status=answer["status"],
                    )
                else:
                    event = SubRunCompletedEvent(
                        **common,
                        parent_tool_call_id=parent_plan["request"]["id"],
                        child_session_id=answer["session_id"],
                        task_id=answer["turn_id"],
                        status=answer["status"],
                        final_output=str(answer["result"]),
                    )
            elif item["kind"] == "user" and p["target_wait_id"] and p["target_operation_id"] is None:
                event = RunStateChangedEvent(**common, state="running")
            elif item["kind"] == "user" and p["target_operation_id"]:
                plan = plans[(p["target_operation_id"], 1)]
                cycle = int(plan["dependencies"][0].rsplit("/", 1)[1])
                event = HostInteractionResponseConsumedEvent(
                    **common,
                    interaction_id=p["target_wait_id"],
                    logical_cycle=cycle,
                    operation_id=p["target_operation_id"],
                    tool_call_id=plan["request"]["id"],
                    request_digest=str(interactions[(p["target_operation_id"], 1)].request_digest),
                    command_id=item["input_id"],
                    response_digest=p["input_digest"],
                    cycle_index=cycle,
                )
        elif r.kind.startswith("op_"):
            assert r.operation_id is not None and r.attempt is not None
            key = (r.operation_id, r.attempt)
            if r.kind == "op_planned":
                plans[key] = p
            elif r.kind == "op_prepared":
                plans[key] = plans[key] | {"request": p["request"], "tool": p["tool"], "request_digest": p["request_digest"]}
            plan = plans[key]
            kind, request = plan["op_kind"], plan["request"]
            source = r.operation_id if kind == "model" else plan["dependencies"][0]
            cycle = (
                request.get("metadata", {})
                .get("vv_session", {})
                .get(
                    "cycle_index", int(source.rsplit("/", 1)[1]) if plan["purpose"] not in {"compaction", "session_memory"} else 1
                )
            )
            if kind != "model":
                cycle = plans[(source, 1)]["request"]["metadata"].get("vv_session", {}).get("cycle_index", cycle)
            if r.kind == "op_planned" and kind == "model" and plan["purpose"] == "primary" and r.attempt == 1:
                cycle_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/cycle"}
                events.append(CycleStartedEvent(**cycle_common, cycle_index=cycle))
            if r.kind == "op_completed" and kind == "model" and plan["purpose"] == "primary" and p["context"] == "normal":
                response_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/response"}
                events.append(
                    DiagnosticEvent(
                        **response_common,
                        cycle_index=cycle,
                        level="info",
                        code="cycle_llm_response",
                        details={
                            "assistant_message": p["result"].get("content", ""),
                            "tool_calls": p["result"].get("tool_calls", []),
                        },
                    )
                )
            if kind == "model":
                identity: dict[str, Any] = {
                    **common,
                    "call_id": f"{r.operation_id}/{r.attempt}",
                    "operation_id": r.operation_id,
                    "attempt": r.attempt,
                    "operation": {
                        "compaction": ModelCallOperation.MEMORY_COMPACTION,
                        "session_memory": ModelCallOperation.SESSION_MEMORY,
                        "output_repair": ModelCallOperation.OUTPUT_REPAIR,
                    }.get(plan["purpose"], ModelCallOperation.AGENT_CYCLE),
                    "metadata": common["metadata"] | {"purpose": plan["purpose"]},
                    "cycle_index": cycle,
                    "backend": request.get("metadata", {})
                    .get("vv_session", {})
                    .get("model_binding", {})
                    .get(
                        "backend",
                        turns[r.turn_id]["definition"].get("model_binding", {}).get("backend", plan["provider_binding"]),
                    ),
                    "model": request["model"],
                }
                if r.kind == "op_started":
                    event = ModelCallStartedEvent(**identity)
                elif r.kind == "op_completed":
                    if not p["execution_started"]:
                        event = ModelCallFailedEvent(
                            **identity,
                            outcome="definitive",
                            usage=model_usage(p["usage"]),
                            error_code=p["result"].get("error_code", "not_executed"),
                        )
                    elif p["result"].get("error_code"):
                        if p["result"].get("error_code") == "model_request_rejected":
                            identity["metadata"] = identity["metadata"] | {"model_rejection": p["result"]["rejection"]}
                        event = ModelCallFailedEvent(
                            **identity, outcome="definitive", usage=model_usage(p["usage"]), error_code=p["result"]["error_code"]
                        )
                    else:
                        event = ModelCallCompletedEvent(**identity, usage=model_usage(p["usage"]))
                elif r.kind == "op_unknown":
                    events.append(
                        ModelCallFailedEvent(
                            **identity, outcome="ambiguous", usage=model_usage(None), error_code="model_outcome_unknown"
                        )
                    )
                    common["event_id"] = f"sk/{identity_digest}/duplicate-risk"
                    event = ModelRetryDuplicateRiskEvent(
                        **common,
                        operation_id=r.operation_id,
                        operation_kind="model",
                        risk="duplicate_model_request_and_cost",
                        cycle_index=cycle,
                    )
            else:
                tool: dict[str, Any] = {"tool_name": request["name"], "tool_call_id": request["id"], "cycle_index": cycle}
                if r.kind in {"op_planned", "op_started"}:
                    if r.kind == "op_started":
                        diagnostic_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                        events.append(
                            DiagnosticEvent(
                                **diagnostic_common,
                                cycle_index=cycle,
                                level="info",
                                code="tool_started",
                                details={
                                    "tool_name": request["name"],
                                    "tool_call_id": request["id"],
                                    "tool_arguments": request["arguments"],
                                },
                            )
                        )
                    cls = ToolCallPlannedEvent if r.kind == "op_planned" else ToolCallStartedEvent
                    event = cls(
                        **common, **tool, arguments=copy_json(request["arguments"]), tool_metadata=copy_json(plan["tool"]) or None
                    )
                elif r.kind == "op_completed":
                    projected_record = replace(r, payload=p)
                    object.__setattr__(projected_record, "_body", r.encode())
                    descriptor = project_tool_result(replace(stored, record=projected_record))
                    common["metadata"] = common["metadata"] | {"tool_result": descriptor}
                    result = ToolExecutionResult.from_dict(p["result"])
                    tool_result_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                    events.append(
                        DiagnosticEvent(
                            **tool_result_common,
                            cycle_index=cycle,
                            level="info",
                            code="tool_result",
                            details=descriptor,
                        )
                    )
                    event = ToolCallCompletedEvent(
                        **common,
                        **tool,
                        status=result.status_code.value.lower(),
                        directive=result.directive.value,
                        error_code=result.error_code,
                        execution_started=p["execution_started"],
                        duration_ms=None,
                        operation_id=r.operation_id,
                        attempt=r.attempt,
                    )
                elif r.kind == "op_unknown":
                    event = OperationAmbiguousEvent(
                        **common,
                        operation_id=r.operation_id,
                        operation_kind="tool",
                        risk="tool_outcome_unknown",
                        cycle_index=cycle,
                        idempotency_support=(plan["tool"] or {}).get("idempotency", "unknown"),
                    )
                elif r.kind == "op_parked":
                    h = p["handle"]
                    if h["kind"] == "approval":
                        event = ApprovalRequestedEvent(**common, **tool, request_id=h["request_id"], message="Approval required")
                    elif h["kind"] == "user":
                        common["metadata"]["interaction"] = user_interaction(
                            r.session_id,
                            r.turn_id,
                            turns[r.turn_id]["generation"],
                            p,
                            operation_id=r.operation_id,
                            attempt=r.attempt,
                        )
                        interaction = HostInteractionRequest(
                            interaction_id=h["interaction_id"],
                            logical_cycle=cycle,
                            operation_id=r.operation_id,
                            tool_call_id=request["id"],
                            prompt=h["question"],
                        )
                        interactions[key] = interaction
                        event = HostInteractionRequestedEvent(
                            **common,
                            interaction_id=h["interaction_id"],
                            logical_cycle=cycle,
                            operation_id=r.operation_id,
                            tool_call_id=request["id"],
                            request_digest=str(interaction.request_digest),
                            prompt=h["question"],
                            cycle_index=cycle,
                        )
                    elif h["kind"] == "child":
                        marker = p.get("delegation", {})
                        delegations[key] = marker
                        if marker.get("mode") == "handoff":
                            handoff_common = {k: v for k, v in common.items() if k != "agent_name"}
                            handoff_common["metadata"] = common["metadata"] | marker["metadata"]
                            event = HandoffStartedEvent(
                                **handoff_common,
                                source_agent=common.get("agent_name") or "agent",
                                target_agent=marker["agent_name"],
                                tool_call_id=request["id"],
                                child_session_id=h["session_id"],
                            )
                        else:
                            for index, member in enumerate(child_handles(h)):
                                child_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/child/{index}"}
                                events.append(
                                    SubRunStartedEvent(
                                        **child_common,
                                        parent_tool_call_id=request["id"],
                                        child_session_id=member["session_id"],
                                        task_id=member["turn_id"],
                                    )
                                )
                    else:
                        event = RunStateChangedEvent(**common, state="parked")
        if event is not None:
            events.append(event)
    for projected in events:
        object.__setattr__(projected, "version", "v6")
    return deepcopy(events)
