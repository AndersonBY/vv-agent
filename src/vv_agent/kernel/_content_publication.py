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

from ._content import ContentRef, ContentSessionWriter
from ._hydration import reference_digest
from ._records import Record, digest

_CONTENT_REF_FIELDS = frozenset({"content_id", "mac", "size", "key_id", "labels"})
_MISSING = object()

# Each path is explicit. ``*`` expands one declared JSON array element and the
# resulting link slot is the field name followed by its stable zero-based index.
_CONTENT_SLOTS: Mapping[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "turn_started": ((("definition",), "definition"), (("budget",), "budget")),
    "input_applied": ((("reason",), "input_reason"), (("model_binding",), "model_binding")),
    "op_planned": (
        (("request",), "request"),
        (("tool",), "tool"),
        (("budget_admission",), "budget_admission"),
    ),
    "op_prepared": (
        (("request",), "request"),
        (("tool",), "tool"),
        (("hook_result",), "hook_result"),
        (("shared_state",), "shared_state"),
    ),
    "turn_parked": ((("question",), "turn_question"),),
    "op_parked": (
        (("handle", "job_id"), "provider_handle_job_id"),
        (("handle", "evidence"), "provider_handle_evidence"),
        (("handle", "query_ref"), "provider_handle_query_ref"),
        (("handle", "cancel_ref"), "provider_handle_cancel_ref"),
        (("handle", "scope"), "approval_scope"),
        (("handle", "question"), "user_question"),
        (("interaction_result",), "interaction_result"),
        (("delegation", "agent_name"), "delegation_agent_name"),
        (("delegation", "metadata"), "delegation_metadata"),
    ),
    "op_completed": (
        (("result",), "result"),
        (("usage",), "usage"),
        (("shared_state",), "shared_state"),
        (("evidence",), "evidence"),
    ),
    "op_unknown": (
        (("reason",), "op_unknown_reason"),
        (("dispatch_evidence",), "op_unknown_dispatch_evidence"),
        (("observation",), "op_unknown_observation"),
    ),
    "context_compacted": (
        (("replacement", "*"), "replacement"),
        (("evidence_manifest",), "evidence_manifest"),
    ),
    "usage_observed": ((("usage",), "usage"), (("source",), "usage_source")),
    "turn_ended": (
        (("reason",), "turn_ended_reason"),
        (("result",), "turn_ended_result"),
        (("budget",), "turn_ended_budget"),
    ),
}

_BOUNDARY_CONTENT_SLOTS: Mapping[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "before_memory": ((("messages", "*"), "message"), (("shared_state",), "shared_state")),
    "after_cycle": (
        (("steering_messages", "*"), "message"),
        (("stop", "message"), "stop_message"),
        (("shared_state",), "shared_state"),
        (("error",), "boundary_error"),
    ),
    "memory_started": ((("event",), "memory_event"),),
    "memory_completed": ((("event",), "memory_event"),),
    "session_memory_saved": ((("state", "entries", "*", "content"), "session_memory_content"),),
    "output_checked": (
        (("reason",), "output_reason"),
        (("value",), "output_value"),
        (("partial_output",), "partial_output"),
    ),
    "budget": ((("usage",), "budget_usage"), (("exhaustion",), "budget_exhaustion")),
}

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


def declared_content_ref(value: Any) -> ContentRef | None:
    """Return a valid declared reference, or ``None`` for a plain producer value."""

    if isinstance(value, ContentRef):
        return value
    try:
        return ContentRef.from_dict(value)
    except (TypeError, ValueError):
        if isinstance(value, Mapping) and set(value) == _CONTENT_REF_FIELDS:
            raise ValueError("malformed ContentRef") from None
        return None


def publish_json(writer: ContentSessionWriter, value: Any) -> ContentRef:
    """Preserve a legal reference and put only plain producer content."""

    declared = declared_content_ref(value)
    return writer.put_json(value) if declared is None else declared


def publish_turn_started(
    writer: ContentSessionWriter,
    definition: Any,
    budget: Any,
    *,
    admission_definition_digest: str | None = None,
) -> TurnStartedContent:
    """Publish one turn definition and budget through the transaction writer."""

    declared = declared_content_ref(definition)
    if declared is None and admission_definition_digest is not None and digest(definition) != admission_definition_digest:
        raise ValueError("child admission definition digest mismatch")
    definition_ref = writer.put_json(definition) if declared is None else declared
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
        for path, slot, value in _slots(record, payload=payload):
            if value is _MISSING or value is None:
                continue
            ref = publish_json(writer, value)
            _set_path(payload, path, ref.to_dict())
            changed = True
            key = (record.record_id, slot)
            if key in seen_links:
                raise ValueError(f"record content slot is declared more than once: {record.record_id}/{slot}")
            seen_links.add(key)
            links.append(DeclaredRecordContentLink(record.record_id, slot, ref))
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


def _slots(record: Record, *, payload: dict[str, Any] | None = None) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    yield from _payload_slots(record.kind, record.payload if payload is None else payload)


def _payload_slots(kind: str, source: dict[str, Any]) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if kind == "boundary_recorded":
        stage = source.get("stage")
        rules = _BOUNDARY_CONTENT_SLOTS.get(stage, ()) if isinstance(stage, str) else ()
        root = source.get("data")
        if isinstance(root, dict):
            for path, slot in rules:
                yield from _expand_slot(root, path, slot)
        return

    for path, slot in _CONTENT_SLOTS.get(kind, ()):
        yield from _expand_slot(source, path, slot)


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
        if not isinstance(root, list):
            raise ValueError(f"declared content array has non-array value at {'/'.join(resolved)}")
        for index, item in enumerate(root):
            yield from _expand_slot(item, remaining, f"{slot}[{index}]", resolved=(*resolved, str(index)))
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
