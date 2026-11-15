"""Kernel-side publication plans for record content.

The helper stays independent of session backends, SQL, and cryptography. It
turns producer values into references, declares the record fields that own
those references, and links them after append has assigned record sequences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ._content import ContentRef, ContentSessionWriter
from ._hydration import reference_digest
from ._records import Record, digest

_CONTENT_REF_FIELDS = frozenset({"content_id", "mac", "size", "key_id", "labels"})


class _RecordLinkStore(Protocol):
    def link_record(
        self,
        tenant_id: str,
        session_id: str,
        record_seq: int,
        ref: ContentRef,
        slot: str,
    ) -> None: ...


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
class RecordContentLink:
    record_id: str
    record_seq: int
    ref: ContentRef
    slot: str


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


def declared_record_content(
    records: Iterable[Record], record_sequences: tuple[tuple[str, int], ...]
) -> tuple[RecordContentLink, ...]:
    """Declare record-owned refs that must be linked after append."""

    sequences = dict(record_sequences)
    declared: dict[tuple[str, str], RecordContentLink] = {}
    for record in records:
        if record.kind != "turn_started":
            continue
        record_seq = sequences.get(record.record_id)
        if record_seq is None:
            raise ValueError("appended turn_started record has no committed sequence")
        for field, slot in (("definition", "definition"), ("budget", "budget")):
            value = record._payload[field]
            ref = ContentRef.from_dict(value) if not isinstance(value, ContentRef) else value
            key = (record.record_id, slot)
            if key in declared:
                raise ValueError("record content slot is declared more than once")
            declared[key] = RecordContentLink(record.record_id, record_seq, ref, slot)
    return tuple(declared.values())


def link_declared_record_content(
    writer: ContentSessionWriter,
    links: Iterable[RecordContentLink],
) -> None:
    """Link every declared ref to the record sequence assigned by append."""

    record_writer = cast(RecordContentSessionWriter, writer)
    for link in links:
        record_writer.durable.link_record(
            record_writer.tenant_id,
            record_writer.session_id,
            link.record_seq,
            link.ref,
            link.slot,
        )
