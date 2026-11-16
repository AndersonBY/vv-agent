"""Kernel-side publication plans for record content.

The helper stays independent of session backends, SQL, and cryptography. It
turns producer values into references, declares the record fields that own
those references, and links them after append has assigned record sequences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ._content import (
    APPROVAL_METADATA,
    APPROVAL_REASON,
    APPROVAL_SCOPE,
    CHILD_RESULT,
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    INPUT_SELECTION,
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentRef,
    ContentSessionWriter,
)
from ._hydration import reference_digest
from ._records import InboxItem, Record, copy_json, digest, make_record, record_identity

_CONTENT_REF_FIELDS = frozenset({"content_id", "mac", "size", "key_id", "labels"})
_MISSING = object()

PROVIDER_RESULT = "provider.result"
PROVIDER_USAGE = "provider.usage"
PROVIDER_EVIDENCE = "provider.evidence"
PROVIDER_HANDLE_JOB_ID = "provider.handle.job_id"
PROVIDER_HANDLE_EVIDENCE = "provider.handle.evidence"
PROVIDER_HANDLE_QUERY_REF = "provider.handle.query_ref"
PROVIDER_HANDLE_CANCEL_REF = "provider.handle.cancel_ref"


@dataclass(frozen=True, slots=True)
class NestedContentSlot:
    """One explicitly typed content-reference array below an encrypted container."""

    path: tuple[str, ...]
    slot: str


@dataclass(frozen=True, slots=True)
class ContentSlot:
    """One statically schema-owned content slot and its explicit nested paths."""

    path: tuple[str, ...]
    slot: str
    nested: tuple[NestedContentSlot, ...] = ()


_REQUEST_NESTED = (NestedContentSlot(("messages", "*"), "messages"),)
_SEED_MESSAGES = (NestedContentSlot(("seed", "messages", "*"), "seed.messages"),)
_DEFINITION_NESTED = (
    NestedContentSlot(("task", "initial_messages", "*"), "task.initial_messages"),
    NestedContentSlot(
        ("task", "metadata", "vv_session", "input_messages", "*"),
        "task.metadata.vv_session.input_messages",
    ),
)
_CHILD_ADMISSION_DEFINITION_NESTED = (
    NestedContentSlot(
        ("child_admission", "definition", "task", "initial_messages", "*"),
        "child_admission.definition.task.initial_messages",
    ),
    NestedContentSlot(
        ("child_admission", "definition", "task", "metadata", "vv_session", "input_messages", "*"),
        "child_admission.definition.task.metadata.vv_session.input_messages",
    ),
)
_SESSION_ATTRIBUTES_NESTED = _SEED_MESSAGES + _CHILD_ADMISSION_DEFINITION_NESTED
_HOOK_RESULT_NESTED = (
    NestedContentSlot(("metadata", "children", "*"), "metadata.children"),
    NestedContentSlot(("metadata", "running_children", "*"), "metadata.running_children"),
)

# Each path is explicit. ``*`` expands one declared JSON array element and the
# resulting link slot is the field name followed by its stable zero-based index.
_CONTENT_SLOTS: Mapping[str, tuple[ContentSlot, ...]] = {
    "session_created": (
        ContentSlot(("principal", "attributes"), SESSION_PRINCIPAL_ATTRIBUTES, _SEED_MESSAGES),
        ContentSlot(("workspace",), SESSION_WORKSPACE),
        ContentSlot(("attributes",), SESSION_ATTRIBUTES, _SESSION_ATTRIBUTES_NESTED),
    ),
    "turn_started": (
        ContentSlot(("definition",), "definition", _DEFINITION_NESTED),
        ContentSlot(("budget",), "budget"),
        ContentSlot(("environment", "descriptor"), "environment_descriptor"),
        ContentSlot(("environment", "operational_policy"), "environment_operational_policy"),
    ),
    "input_applied": (
        ContentSlot(("input", "actor", "attributes"), INPUT_ACTOR_ATTRIBUTES),
        ContentSlot(("reason",), "input_reason"),
        ContentSlot(("model_binding",), "model_binding"),
    ),
    "op_planned": (
        ContentSlot(("request",), "request", _REQUEST_NESTED),
        ContentSlot(("tool",), "tool"),
        ContentSlot(("budget_admission",), "budget_admission"),
    ),
    "op_prepared": (
        ContentSlot(("request",), "request", _REQUEST_NESTED),
        ContentSlot(("tool",), "tool"),
        ContentSlot(("hook_result",), "hook_result", _HOOK_RESULT_NESTED),
        ContentSlot(("shared_state",), "shared_state"),
    ),
    "turn_parked": (ContentSlot(("question",), "turn_question"),),
    "op_started": (),
    "op_parked": (
        ContentSlot(("handle", "job_id"), "provider_handle_job_id"),
        ContentSlot(("handle", "evidence"), "provider_handle_evidence"),
        ContentSlot(("handle", "query_ref"), "provider_handle_query_ref"),
        ContentSlot(("handle", "cancel_ref"), "provider_handle_cancel_ref"),
        ContentSlot(("handle", "scope"), "approval_scope"),
        ContentSlot(("handle", "question"), "user_question"),
        ContentSlot(("interaction_result",), "interaction_result"),
        ContentSlot(("delegation", "agent_name"), "delegation_agent_name"),
        ContentSlot(("delegation", "metadata"), "delegation_metadata"),
    ),
    "op_completed": (
        ContentSlot(("result",), "result"),
        ContentSlot(("usage",), "usage"),
        ContentSlot(("shared_state",), "shared_state"),
        ContentSlot(("evidence",), "evidence"),
        ContentSlot(("usage_counts", "budget", "host_cost", "unit"), "usage_counts.host_cost.unit"),
        ContentSlot(("usage_counts", "budget", "host_cost", "currency"), "usage_counts.host_cost.currency"),
    ),
    "op_unknown": (
        ContentSlot(("reason",), "op_unknown_reason"),
        ContentSlot(("dispatch_evidence",), "op_unknown_dispatch_evidence"),
        ContentSlot(("observation",), "op_unknown_observation"),
    ),
    "context_compacted": (
        ContentSlot(("replacement", "*"), "replacement"),
        ContentSlot(("evidence_manifest",), "evidence_manifest"),
    ),
    "usage_observed": (ContentSlot(("usage",), "usage"), ContentSlot(("source",), "usage_source")),
    "turn_ended": (
        ContentSlot(("reason",), "turn_ended_reason"),
        ContentSlot(("result",), "turn_ended_result"),
        ContentSlot(("budget",), "turn_ended_budget"),
    ),
    "boundary_recorded": (),
}

_INPUT_SLOTS: Mapping[str, tuple[ContentSlot, ...]] = {
    "user": (ContentSlot(("content",), INPUT_CONTENT),),
    "steer": (ContentSlot(("content",), INPUT_CONTENT),),
    "follow_up": (ContentSlot(("content",), INPUT_CONTENT),),
    "model_selection": (ContentSlot(("selection",), INPUT_SELECTION),),
    "provider_result": (
        ContentSlot(("result",), PROVIDER_RESULT),
        ContentSlot(("usage",), PROVIDER_USAGE),
        ContentSlot(("evidence",), PROVIDER_EVIDENCE),
    ),
    "approval_answer": (
        ContentSlot(("scope",), APPROVAL_SCOPE),
        ContentSlot(("reason",), APPROVAL_REASON),
        ContentSlot(("metadata",), APPROVAL_METADATA),
    ),
    "child_result": (ContentSlot(("result",), CHILD_RESULT),),
    "control": (),
    "provider_evidence": (
        ContentSlot(("handle", "job_id"), PROVIDER_HANDLE_JOB_ID),
        ContentSlot(("handle", "evidence"), PROVIDER_HANDLE_EVIDENCE),
        ContentSlot(("handle", "query_ref"), PROVIDER_HANDLE_QUERY_REF),
        ContentSlot(("handle", "cancel_ref"), PROVIDER_HANDLE_CANCEL_REF),
    ),
}

_BOUNDARY_CONTENT_SLOTS: Mapping[str, tuple[ContentSlot, ...]] = {
    "before_memory": (ContentSlot(("messages", "*"), "message"), ContentSlot(("shared_state",), "shared_state")),
    "after_cycle": (
        ContentSlot(("steering_messages", "*"), "message"),
        ContentSlot(("stop", "message"), "stop_message"),
        ContentSlot(("shared_state",), "shared_state"),
        ContentSlot(("error",), "boundary_error"),
    ),
    "memory_started": (ContentSlot(("event",), "memory_event"),),
    "memory_completed": (ContentSlot(("event",), "memory_event"),),
    "session_memory_saved": (ContentSlot(("state", "entries", "*", "content"), "session_memory_content"),),
    "output_checked": (
        ContentSlot(("reason",), "output_reason"),
        ContentSlot(("value",), "output_value"),
        ContentSlot(("partial_output",), "partial_output"),
    ),
    "budget": (ContentSlot(("usage",), "budget_usage"), ContentSlot(("exhaustion",), "budget_exhaustion")),
}


def input_content_slots(kind: str) -> tuple[ContentSlot, ...]:
    """Return the closed payload slots for one of the nine inbox kinds."""

    return _INPUT_SLOTS[kind]


def input_record_slots(kind: str) -> tuple[ContentSlot, ...]:
    """Project inbox payload slots under the embedded ``input.payload`` owner path."""

    return tuple(
        ContentSlot(("input", "payload", *item.path), f"input.payload.{'.'.join(item.path)}")
        for item in input_content_slots(kind)
    )


_CONTENT_SLOTS["input_applied"] = (
    ContentSlot(("input", "actor", "attributes"), INPUT_ACTOR_ATTRIBUTES),
    ContentSlot(("reason",), "input_reason"),
    ContentSlot(("model_binding",), "model_binding"),
)

_REFERENCE_DIGESTS: Mapping[tuple[str, str], str] = {
    ("turn_started", "definition"): "definition_digest",
    ("op_planned", "request"): "request_digest",
    ("op_prepared", "request"): "request_digest",
    ("op_completed", "result"): "result_digest",
}


class _RecordLinkStore(Protocol):
    def link_record(
        self,
        tenant_id: str,
        session_id: str,
        record_seq: int,
        ref: ContentRef,
        slot: str,
    ) -> None: ...


@runtime_checkable
class RecordContentSessionWriter(ContentSessionWriter, Protocol):
    tenant_id: str
    session_id: str
    durable: _RecordLinkStore


@dataclass(frozen=True, slots=True)
class TurnStartedContent:
    definition: ContentRef
    budget: ContentRef
    definition_digest: str


@dataclass(frozen=True, slots=True)
class DeclaredRecordContentLink:
    record_id: str
    slot: str
    ref: ContentRef


@dataclass(frozen=True, slots=True)
class RecordContentLink:
    record_id: str
    record_seq: int
    ref: ContentRef
    slot: str


@dataclass(frozen=True, slots=True)
class RecordContentPublication:
    records: tuple[Record, ...]
    links: tuple[DeclaredRecordContentLink, ...]


@dataclass(frozen=True, slots=True)
class DeclaredInputContentLink:
    input_id: str
    slot: str
    ref: ContentRef


@dataclass(frozen=True, slots=True)
class PublishedNestedContentLink:
    slot: str
    ref: ContentRef


@dataclass(frozen=True, slots=True)
class PublishedContentValue:
    ref: ContentRef
    nested: tuple[PublishedNestedContentLink, ...] = ()


@dataclass(frozen=True, slots=True)
class InputContentPublication:
    item: InboxItem
    links: tuple[DeclaredInputContentLink, ...]


@dataclass(frozen=True, slots=True)
class SessionCreatedContent:
    principal: dict[str, Any]
    workspace: ContentRef
    attributes: ContentRef
    nested: tuple[PublishedNestedContentLink, ...]


def declared_content_ref(value: Any) -> ContentRef | None:
    """Parse a reference-form value from an explicit replay or link path."""

    if isinstance(value, ContentRef):
        return value
    try:
        return ContentRef.from_dict(value)
    except (TypeError, ValueError):
        if isinstance(value, Mapping) and set(value) == _CONTENT_REF_FIELDS:
            raise ValueError("malformed ContentRef") from None
        return None


def producer_content_ref(value: Any) -> ContentRef | None:
    """Return an explicitly typed producer reference; mappings are opaque JSON."""

    return value if isinstance(value, ContentRef) else None


def _producer_ref(writer: ContentSessionWriter, value: Any) -> ContentRef | None:
    """Accept a mapping only when this writer explicitly returned its ref."""

    typed = producer_content_ref(value)
    if typed is not None:
        published_refs = getattr(writer, "_published_refs", None)
        if isinstance(published_refs, set):
            published_refs.add(typed)
        return typed
    try:
        ref = ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return None
    return ref if ref in getattr(writer, "_published_refs", ()) else None


def make_record_before_publication(
    kind: str,
    *,
    session_id: str,
    payload: dict[str, Any],
    turn_id: str | None = None,
    operation_id: str | None = None,
    attempt: int | None = None,
) -> Record:
    """Build a content record that must pass through ``publish_record_content`` before encoding."""

    payload = copy_json(payload)
    if kind == "turn_started":
        payload.setdefault("policy_digest", None)
        payload.setdefault(
            "requirements",
            {"level": 1, "handler_version": payload.get("handler_version"), "capabilities": []},
        )
        payload.setdefault("tokenizer_binding", None)
        payload.setdefault("environment", None)
    elif kind in {"op_planned", "op_prepared", "op_completed"}:
        payload.setdefault("policy_decision", None)
    if not has_record_content_payload(kind, payload):
        return make_record(
            kind,
            session_id=session_id,
            turn_id=turn_id,
            operation_id=operation_id,
            attempt=attempt,
            payload=payload,
        )
    return Record(
        record_identity(kind, session_id, turn_id, operation_id, attempt, payload),
        kind,
        session_id,
        turn_id,
        operation_id,
        attempt,
        payload,
    )


def make_retained_record_before_publication(
    kind: str,
    *,
    session_id: str,
    payload: dict[str, Any],
    turn_id: str | None = None,
    operation_id: str | None = None,
    attempt: int | None = None,
) -> Record:
    """Rebuild a committed record while preserving refs at closed content slots."""

    retained = deepcopy(payload)
    for path, _slot, value in _payload_slots(kind, retained):
        if value is _MISSING or value is None:
            continue
        ref = declared_content_ref(value)
        if ref is not None:
            _set_path(retained, path, ref)
    return make_record_before_publication(
        kind,
        session_id=session_id,
        turn_id=turn_id,
        operation_id=operation_id,
        attempt=attempt,
        payload=retained,
    )


def publish_json(writer: ContentSessionWriter, value: Any) -> ContentRef:
    """Preserve a typed or explicitly returned ref and put other mappings as JSON."""

    declared = _producer_ref(writer, value)
    return writer.put_json(value) if declared is None else declared


def publish_content_container(
    writer: ContentSessionWriter,
    value: Any,
    *,
    slot: str,
    nested: tuple[NestedContentSlot, ...] = (),
) -> PublishedContentValue:
    """Publish one outer container and only its explicitly declared nested refs."""

    declared = _producer_ref(writer, value)
    if declared is not None:
        if not nested:
            return PublishedContentValue(declared)
        read_json = getattr(writer, "get_json", None)
        if not callable(read_json):
            raise ValueError(f"declared content container cannot be inspected: {slot}")
        content = read_json(declared)
        prepared = deepcopy(content)
        nested_links: list[PublishedNestedContentLink] = []
        changed = False
        for path, nested_slot, nested_value in _expand_nested_slots(prepared, nested, slot):
            if nested_value is _MISSING or nested_value is None:
                continue
            nested_declared = declared_content_ref(nested_value)
            nested_ref = nested_declared if nested_declared is not None else writer.put_json(nested_value)
            changed = changed or nested_declared is None
            _set_path(prepared, path, nested_ref.to_dict())
            nested_links.append(PublishedNestedContentLink(nested_slot, nested_ref))
        ref = writer.put_json(prepared) if changed else declared
        return PublishedContentValue(ref, tuple(nested_links))

    prepared = deepcopy(value)
    nested_links: list[PublishedNestedContentLink] = []
    for path, nested_slot, nested_value in _expand_nested_slots(prepared, nested, slot):
        if nested_value is _MISSING or nested_value is None:
            continue
        declared = _producer_ref(writer, nested_value)
        nested_ref = writer.put_json(nested_value) if declared is None else declared
        _set_path(prepared, path, nested_ref.to_dict())
        nested_links.append(PublishedNestedContentLink(nested_slot, nested_ref))
    return PublishedContentValue(writer.put_json(prepared), tuple(nested_links))


def declare_input_content(writer: ContentSessionWriter, item: InboxItem) -> InputContentPublication:
    """Publish provider inbox values and retain their owner-slot declarations."""

    payload = deepcopy(item.payload)
    links: list[DeclaredInputContentLink] = []
    for content_slot in input_content_slots(item.kind):
        for path, slot, value in _expand_slot(payload, content_slot.path, content_slot.slot):
            if value is _MISSING or value is None:
                continue
            declared = producer_content_ref(value)
            ref = writer.assert_input(item.input_id, declared, slot) if declared is not None else writer.put_json(value)
            _set_path(payload, path, ref.to_dict())
            links.append(DeclaredInputContentLink(item.input_id, slot, ref))
    return InputContentPublication(replace(item, payload=payload), tuple(links))


def publish_session_created_content(
    writer: ContentSessionWriter,
    *,
    principal: dict[str, Any],
    workspace: Any,
    attributes: Any,
) -> SessionCreatedContent:
    """Publish creation values while keeping declared message arrays as owned child objects."""

    published_principal = copy_json(principal)
    principal_attributes = publish_content_container(
        writer,
        published_principal["attributes"],
        slot=SESSION_PRINCIPAL_ATTRIBUTES,
        nested=_SEED_MESSAGES,
    )
    published_principal["attributes"] = principal_attributes.ref.to_dict()
    workspace_ref = publish_json(writer, workspace)
    published_attributes = publish_content_container(
        writer,
        attributes,
        slot=SESSION_ATTRIBUTES,
        nested=_SESSION_ATTRIBUTES_NESTED,
    )
    return SessionCreatedContent(
        principal=published_principal,
        workspace=workspace_ref,
        attributes=published_attributes.ref,
        nested=principal_attributes.nested + published_attributes.nested,
    )


def link_declared_input_content(
    writer: ContentSessionWriter,
    item: InboxItem,
) -> None:
    """Link inbox payload and actor references after the inbox row exists."""

    for content_slot in input_content_slots(item.kind):
        for _path, slot, value in _expand_slot(item.payload, content_slot.path, content_slot.slot):
            if value is _MISSING or value is None:
                continue
            ref = declared_content_ref(value)
            if ref is None:
                raise ValueError(f"input content slot is not a ContentRef: {item.input_id}/{slot}")
            writer.link_input(item.input_id, ref, slot)

    if item.actor is not None:
        attributes = item.actor["attributes"]
        ref = declared_content_ref(attributes)
        if ref is None:
            raise ValueError(f"input actor attributes are not a ContentRef: {item.input_id}")
        writer.link_input(item.input_id, ref, INPUT_ACTOR_ATTRIBUTES)


def publish_turn_started(
    writer: ContentSessionWriter,
    definition: Any,
    budget: Any,
    *,
    admission_definition_digest: str | None = None,
) -> TurnStartedContent:
    """Publish one turn definition and budget through the transaction writer."""

    declared = _producer_ref(writer, definition)
    if declared is None and admission_definition_digest is not None and digest(definition) != admission_definition_digest:
        raise ValueError("child admission definition digest mismatch")
    definition_ref = publish_content_container(
        writer,
        definition,
        slot="definition",
        nested=_DEFINITION_NESTED,
    ).ref
    budget_ref = publish_json(writer, budget)
    return TurnStartedContent(
        definition=definition_ref,
        budget=budget_ref,
        definition_digest=reference_digest(definition_ref),
    )


def has_record_content(records: Iterable[Record]) -> bool:
    """Return whether any explicitly declared content slot has a value."""

    return any(has_record_content_payload(record.kind, record.payload) for record in records)


def has_record_content_payload(kind: str, payload: dict[str, Any]) -> bool:
    """Return whether one producer payload declares a non-null content slot."""

    return any(value is not _MISSING and value is not None for _path, _slot, value in _payload_slots(kind, payload))


def publish_record_content(writer: ContentSessionWriter, records: Iterable[Record]) -> RecordContentPublication:
    """Publish declared producer fields and return immutable records plus links."""

    source = tuple(records)
    published: list[Record] = []
    links: list[DeclaredRecordContentLink] = []
    seen_links: set[tuple[str, str]] = set()
    for record in source:
        payload = deepcopy(record.payload)
        changed = False
        for content_slot in _content_slot_specs(record.kind, payload):
            for path, slot, value in _expand_slot(payload, content_slot.path, content_slot.slot):
                if value is _MISSING or value is None:
                    continue
                if record.kind == "input_applied" and (slot == INPUT_ACTOR_ATTRIBUTES or slot.startswith("input.payload.")):
                    ref = declared_content_ref(value)
                    if ref is None:
                        raise ValueError(f"applied input content slot is not a ContentRef: {record.record_id}/{slot}")
                    published_content = PublishedContentValue(ref)
                else:
                    published_content = publish_content_container(
                        writer,
                        value,
                        slot=slot,
                        nested=content_slot.nested,
                    )
                ref = published_content.ref
                _set_path(payload, path, ref.to_dict())
                changed = True
                _append_record_link(record, slot, ref, links, seen_links)
                for nested_link in published_content.nested:
                    _append_record_link(record, nested_link.slot, nested_link.ref, links, seen_links)
                digest_field = _REFERENCE_DIGESTS.get((record.kind, slot))
                if digest_field is not None:
                    payload[digest_field] = reference_digest(ref)
        published.append(replace(record, payload=payload) if changed else record)

    return RecordContentPublication(tuple(published), tuple(links))


def resolve_record_content_links(
    links: Iterable[DeclaredRecordContentLink], record_sequences: tuple[tuple[str, int], ...]
) -> tuple[RecordContentLink, ...]:
    """Resolve declarations to append-assigned record sequences."""

    sequences = dict(record_sequences)
    resolved: list[RecordContentLink] = []
    for link in links:
        record_seq = sequences.get(link.record_id)
        if record_seq is None:
            raise ValueError(f"appended {link.record_id} record has no committed sequence")
        resolved.append(RecordContentLink(link.record_id, record_seq, link.ref, link.slot))
    return tuple(resolved)


def declared_record_content(
    records: Iterable[Record], record_sequences: tuple[tuple[str, int], ...]
) -> tuple[RecordContentLink, ...]:
    """Declare and resolve refs already present in a committed record batch."""

    declarations: list[DeclaredRecordContentLink] = []
    for record in records:
        for _path, slot, value in _slots(record):
            if value is _MISSING or value is None:
                continue
            ref = declared_content_ref(value)
            if ref is None:
                raise ValueError(f"record content slot is not a ContentRef: {record.record_id}/{slot}")
            declarations.append(DeclaredRecordContentLink(record.record_id, slot, ref))
    return resolve_record_content_links(declarations, record_sequences)


def require_record_link_writer(writer: ContentSessionWriter) -> RecordContentSessionWriter:
    if not isinstance(writer, RecordContentSessionWriter):
        raise ValueError("session content writer lacks durable record-link capability")
    return writer


def link_declared_record_content(
    writer: ContentSessionWriter,
    links: Iterable[RecordContentLink],
) -> None:
    """Link every declared ref to the record sequence assigned by append."""

    record_writer = require_record_link_writer(writer)
    for link in links:
        record_writer.durable.link_record(
            record_writer.tenant_id,
            record_writer.session_id,
            link.record_seq,
            link.ref,
            link.slot,
        )


def _append_record_link(
    record: Record,
    slot: str,
    ref: ContentRef,
    links: list[DeclaredRecordContentLink],
    seen_links: set[tuple[str, str]],
) -> None:
    key = (record.record_id, slot)
    if key in seen_links:
        raise ValueError(f"record content slot is declared more than once: {record.record_id}/{slot}")
    seen_links.add(key)
    links.append(DeclaredRecordContentLink(record.record_id, slot, ref))


def _expand_nested_slots(
    root: Any,
    nested: tuple[NestedContentSlot, ...],
    outer_slot: str,
) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    for content_slot in nested:
        slot = f"{outer_slot}.{content_slot.slot}"
        yield from _expand_slot(root, content_slot.path, slot)


def _content_slot_specs(kind: str, source: dict[str, Any]) -> tuple[ContentSlot, ...]:
    if kind == "boundary_recorded":
        stage = source.get("stage")
        rules = _BOUNDARY_CONTENT_SLOTS.get(stage, ()) if isinstance(stage, str) else ()
        return tuple(ContentSlot(("data", *content_slot.path), content_slot.slot, content_slot.nested) for content_slot in rules)
    if kind == "input_applied":
        input_value = source.get("input")
        input_kind = input_value.get("kind") if isinstance(input_value, dict) else None
        if isinstance(input_kind, str):
            return (
                ContentSlot(("input", "actor", "attributes"), INPUT_ACTOR_ATTRIBUTES),
                *input_record_slots(input_kind),
                ContentSlot(("reason",), "input_reason"),
                ContentSlot(("model_binding",), "model_binding"),
            )
    try:
        return _CONTENT_SLOTS[kind]
    except KeyError:
        raise ValueError(f"record kind has no closed content inventory: {kind}") from None


def _slots(record: Record, *, payload: dict[str, Any] | None = None) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    yield from _payload_slots(record.kind, record.payload if payload is None else payload)


def _payload_slots(kind: str, source: dict[str, Any]) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    for content_slot in _content_slot_specs(kind, source):
        yield from _expand_slot(source, content_slot.path, content_slot.slot)


def _expand_slot(
    root: Any,
    path: tuple[str, ...],
    slot: str,
    *,
    resolved: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if not path:
        yield resolved, slot, root
        return
    part, remaining = path[0], path[1:]
    if part == "*":
        if root is None:
            yield resolved, slot, _MISSING
            return
        if not isinstance(root, list):
            raise ValueError(f"declared content array has non-array value at {'/'.join(resolved)}")
        for index, item in enumerate(root):
            yield from _expand_slot(item, remaining, f"{slot}[{index}]", resolved=(*resolved, str(index)))
        return
    if root is None:
        yield resolved, slot, _MISSING
        return
    if not isinstance(root, dict):
        raise ValueError(f"declared content slot has non-object value at {'/'.join(resolved)}")
    if part not in root:
        yield (*resolved, part), slot, _MISSING
        return
    yield from _expand_slot(root[part], remaining, slot, resolved=(*resolved, part))


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = root
    for index, part in enumerate(path):
        last = index == len(path) - 1
        if part.isdecimal():
            if not isinstance(current, list):
                raise ValueError(f"content slot path enters non-array at {part}")
            item_index = int(part)
            if last:
                current[item_index] = value
            else:
                current = current[item_index]
        else:
            if last:
                current[part] = value
            else:
                current = current[part]
