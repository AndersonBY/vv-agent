"""Hydrate durable records and fold a read-only logical view without SDK dependencies."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ._canonical_json import canonical_json_bytes, validate_sha256
from ._content import (
    INPUT_ACTOR_ATTRIBUTES,
    ContentBindingMismatch,
    ContentRef,
    ContentStoreError,
    ContentUnavailable,
)
from ._content import ContentIntegrityError as BackendContentIntegrityError
from ._content import KeyUnavailable as BackendKeyUnavailable
from ._content_publication import (
    _MISSING,
    _REFERENCE_DIGESTS,
    NestedContentSlot,
    _content_slot_specs,
    _expand_nested_slots,
    _payload_slots,
    declared_content_ref,
    input_content_slots,
)
from ._content_reader import DurableContentReader
from ._execution_values import AgentTask
from ._hydration import (
    ContentReader,
    DefinitionContent,
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
    _parse_principal_attributes,
    reference_digest,
)
from ._records import InboxItem, Record, copy_json
from ._reducer import Fold, TransitionError, logical_inputs
from ._state import ExecutionState
from ._store import StoredRecord

if TYPE_CHECKING:
    from ._driver import CoreDriver


_LOGICAL_PREFLIGHT_KINDS = frozenset(
    {
        "input_applied",
        "op_planned",
        "op_prepared",
        "op_started",
        "op_parked",
        "op_completed",
        "op_unknown",
        "turn_parked",
        "turn_ended",
        "boundary_recorded",
        "context_compacted",
    }
)


@dataclass(frozen=True, slots=True)
class SessionHydration:
    """One caller-owned content reader bound to a tenant and session."""

    reader: ContentReader = field(repr=False)
    tenant_id: str
    session_id: str
    _validated_json_refs: set[ContentRef] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.reader, ContentReader):
            raise TypeError("reader must implement ContentReader")

    @classmethod
    def from_store(cls, store: Any, *, tenant_id: str, session_id: str) -> SessionHydration:
        """Bind the core durable reader over one caller-owned session store."""

        from ._content_runtime import durable_content_reader

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
        projected, _ = self._read_projected(
            owner,
            slot,
            ref,
            expected_reference_digest=expected_reference_digest,
            record_kind=record_kind,
            record_payload=record_payload,
        )
        return projected

    def _read_projected(
        self,
        owner: str,
        slot: str,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
        record_kind: str | None = None,
        record_payload: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        hydrator = SessionHydrator(
            self.reader,
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            owner=owner,
            validated_json_refs=self._validated_json_refs,
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
            if type(value) is not dict:
                raise TypedSchemaFailure(f"{slot} must be a JSON object")
            try:
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
            except ValueError as exc:
                raise TypedSchemaFailure(f"{slot} nested content inventory is invalid") from exc
            if selected is HydrationSlot.DEFINITION:
                value = _parse_definition(value)
            elif selected is HydrationSlot.PRINCIPAL_ATTRIBUTES:
                value = _parse_principal_attributes(value)
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
        return _projection_value(slot, value), value

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
        except HydrationError:
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
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypedSchemaFailure(f"{slot} is not valid canonical JSON") from exc
        validated_refs = self._validated_json_refs
        if validated_refs is None or ref not in validated_refs:
            if canonical_json_bytes(value, slot) != body:
                raise TypedSchemaFailure(f"{slot} is not canonical JSON")
            if validated_refs is not None:
                validated_refs.add(ref)
        return value


class RetainedInboxItem(InboxItem):
    """Logical input fields with the committed S1 encoding used for identity checks."""

    _retained_body: bytes

    def __init__(self, item: InboxItem, body: bytes) -> None:
        super().__init__(**item.to_dict())
        object.__setattr__(self, "_retained_body", body)

    def encode(self) -> bytes:
        return self._retained_body


@dataclass(frozen=True, slots=True)
class HydratedFoldResult:
    """One in-memory fold result; never persisted or independently acknowledged."""

    fold: Fold
    state: ExecutionState
    hydrated_records: tuple[StoredRecord, ...]


def hydrate_session_input_payload(
    item: InboxItem,
    hydration: SessionHydration | None,
) -> dict[str, Any]:
    """Hydrate declared input payload slots through their exact input owner."""

    payload = copy_json(item.payload)
    owner = f"input:{item.input_id}"
    for content_slot in input_content_slots(item.kind):
        try:
            value = _path_get(payload, content_slot.path)
        except (KeyError, IndexError):
            continue
        ref = declared_content_ref(value)
        if ref is None:
            continue
        if hydration is None:
            raise MissingContentLink(f"{owner}/{content_slot.slot} requires a content hydration reader")
        _path_set(payload, content_slot.path, hydration.read(owner, content_slot.slot, ref))
    return payload


def hydrate_session_input(
    item: InboxItem,
    hydration: SessionHydration | None,
    *,
    include_actor_attributes: bool = False,
) -> InboxItem:
    """Hydrate one input while optionally resolving the actor identity proof."""

    if hydration is None or not isinstance(hydration.reader, DurableContentReader):
        return _hydrate_session_input(item, hydration, include_actor_attributes=include_actor_attributes)
    with hydration.reader._read_frame(input_ids=(item.input_id,)) as reader:
        return _hydrate_session_input(
            item,
            replace(hydration, reader=reader),
            include_actor_attributes=include_actor_attributes,
        )


def _hydrate_session_input(
    item: InboxItem,
    hydration: SessionHydration | None,
    *,
    include_actor_attributes: bool,
) -> InboxItem:
    """Hydrate one input through an already owner-bound local reader."""

    payload = hydrate_session_input_payload(item, hydration)
    actor = copy_json(item.actor)
    if include_actor_attributes and actor is not None:
        attributes = _path_get(actor, ("attributes",))
        ref = declared_content_ref(attributes)
        if ref is None:
            raise TypedSchemaFailure(f"input:{item.input_id}/{INPUT_ACTOR_ATTRIBUTES} must be a ContentRef")
        if hydration is None:
            raise MissingContentLink(f"input:{item.input_id}/{INPUT_ACTOR_ATTRIBUTES} requires a content hydration reader")
        _path_set(
            actor,
            ("attributes",),
            hydration.read(f"input:{item.input_id}", INPUT_ACTOR_ATTRIBUTES, ref),
        )
    return replace(item, payload=payload, actor=actor)


def hydrate_session_record_payload(
    stored: StoredRecord,
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
    include_actor_attributes: bool = False,
    _decoded_values: dict[tuple[str | int, ...], Any] | None = None,
) -> dict[str, Any]:
    """Hydrate every declared content slot while preserving the committed envelope."""

    record = stored.record
    source = record._payload
    payload = copy_json(source)
    owner = f"record:{stored.seq}"
    for path, slot, value in _payload_slots(record.kind, source):
        if not include_input and record.kind == "input_applied" and path[:1] == ("input",):
            continue
        if not include_actor_attributes and record.kind == "input_applied" and path == ("input", "actor", "attributes"):
            continue
        if value is _MISSING or value is None:
            continue
        try:
            ref = declared_content_ref(value)
        except ValueError as exc:
            raise TypedSchemaFailure(f"{owner}/{slot} contains a malformed ContentRef") from exc
        if ref is None:
            continue
        if hydration is None:
            raise MissingContentLink(f"{owner}/{slot} requires a content hydration reader")
        if record.session_id != hydration.session_id:
            raise MissingContentLink("record session does not match the content reader binding")
        expected = _REFERENCE_DIGESTS.get((record.kind, slot))
        projected, hydrated_value = hydration._read_projected(
            owner,
            slot,
            ref,
            expected_reference_digest=payload.get(expected) if expected is not None else None,
            record_kind=record.kind,
            record_payload=source,
        )
        if _decoded_values is not None:
            _decoded_values[path] = hydrated_value
        _path_set(
            payload,
            path,
            projected,
        )
    return payload


def _retain_record_payload(record: Record, body: bytes, payload: dict[str, Any]) -> None:
    record._retain(
        body,
        {
            "record_id": record.record_id,
            "kind": record.kind,
            "session_id": record.session_id,
            "turn_id": record.turn_id,
            "operation_id": record.operation_id,
            "attempt": record.attempt,
            "payload": payload,
            "schema_version": record.schema_version,
            "feature_level": record.feature_level,
        },
    )


def hydrate_session_stored_record(
    stored: StoredRecord,
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
    include_actor_attributes: bool = False,
) -> StoredRecord:
    retained_body = stored.record.encode()
    decoded_values: dict[tuple[str | int, ...], Any] = {}
    payload = hydrate_session_record_payload(
        stored,
        hydration,
        include_input=include_input,
        include_actor_attributes=include_actor_attributes,
        _decoded_values=decoded_values,
    )
    record = replace(stored.record, payload=payload)
    _retain_record_payload(record, retained_body, payload)
    if record.kind == "turn_started":
        definition = decoded_values.get(("definition",))
        if not isinstance(definition, DefinitionContent):
            raise TypedSchemaFailure("turn_started definition hydration returned the wrong type")
        task = definition.task
        object.__setattr__(record, "_task_value", task)
    return replace(stored, record=record)


def hydrate_session_records(
    records: Iterable[StoredRecord],
    hydration: SessionHydration | None,
    *,
    include_input: bool = True,
    include_actor_attributes: bool = False,
) -> tuple[StoredRecord, ...]:
    return _hydrate_session_records_batch(
        tuple(records),
        hydration,
        include_input=include_input,
        include_actor_attributes=include_actor_attributes,
    )


def _hydrate_session_records_batch(
    sources: tuple[StoredRecord, ...],
    hydration: SessionHydration | None,
    *,
    include_input: bool,
    include_actor_attributes: bool,
) -> tuple[StoredRecord, ...]:
    if not sources:
        return ()
    if hydration is None or not isinstance(hydration.reader, DurableContentReader):
        return tuple(
            hydrate_session_stored_record(
                stored,
                hydration,
                include_input=include_input,
                include_actor_attributes=include_actor_attributes,
            )
            for stored in sources
        )
    with hydration.reader._read_frame(
        record_seqs=(stored.seq for stored in sources),
        validate_each_read=False,
    ) as reader:
        framed = replace(hydration, reader=reader, _validated_json_refs=set())
        return tuple(
            hydrate_session_stored_record(
                stored,
                framed,
                include_input=include_input,
                include_actor_attributes=include_actor_attributes,
            )
            for stored in sources
        )


def _prepare_fold_records(
    sources: Iterable[StoredRecord],
    hydrated: Iterable[StoredRecord],
) -> tuple[list[Record], dict[str, InboxItem], dict[str, StoredRecord]]:
    """Prepare raw record identity and logical input values for one fold."""

    fold_records: list[Record] = []
    logical_items: dict[str, InboxItem] = {}
    hydrated_inputs: dict[str, StoredRecord] = {}
    for source, row in zip(sources, hydrated, strict=True):
        record = row.record
        if source.record.kind == "input_applied":
            logical_payload = copy_json(record._payload)
            fold_payload = copy_json(logical_payload)
            fold_payload["input"] = copy_json(source.record._payload["input"])
            fold_record = replace(record, payload=fold_payload)
            _retain_record_payload(fold_record, source.record.encode(), fold_payload)
            object.__setattr__(fold_record, "_business_payload", logical_payload)
            logical_items[fold_record.record_id] = RetainedInboxItem(
                InboxItem(**logical_payload["input"]),
                InboxItem(**source.record._payload["input"]).encode(),
            )
            hydrated_inputs[row.record.record_id] = row
            record = fold_record
        fold_records.append(record)
    return fold_records, logical_items, hydrated_inputs


def hydrate_and_fold_records(
    source_records: Iterable[StoredRecord],
    fold_factory: Callable[[], Fold],
    hydration: SessionHydration | None,
    *,
    hydrated_records: Iterable[StoredRecord] | None = None,
    base_fold: Fold | None = None,
    record_bodies: Iterable[bytes] | None = None,
    include_input: bool = True,
    raw_applied_inputs: bool = False,
) -> HydratedFoldResult:
    """Fold hydrated values while retaining committed bytes as record identity."""

    sources = tuple(source_records)
    source_by_identity = {(row.record.record_id, row.seq): row for row in sources}
    hydrated_by_identity: dict[tuple[str, int], StoredRecord] = {}

    def add_hydrated(row: StoredRecord) -> None:
        identity = (row.record.record_id, row.seq)
        source = source_by_identity.get(identity)
        if source is None:
            raise MissingContentLink("hydrated record identity does not match source")
        if identity in hydrated_by_identity:
            raise MissingContentLink("hydrated record identity is duplicated")
        if row.record.encode() != source.record.encode():
            raise MissingContentLink("hydrated record body does not match source")
        hydrated_by_identity[identity] = row

    if hydrated_records is None:
        for row in _hydrate_session_records_batch(
            sources,
            hydration,
            include_input=include_input,
            include_actor_attributes=False,
        ):
            add_hydrated(row)
    else:
        for row in tuple(hydrated_records):
            add_hydrated(row)
        missing = tuple(source for source in sources if (source.record.record_id, source.seq) not in hydrated_by_identity)
        if missing:
            for row in _hydrate_session_records_batch(
                missing,
                hydration,
                include_input=include_input,
                include_actor_attributes=False,
            ):
                add_hydrated(row)
    hydrated = tuple(hydrated_by_identity[(source.record.record_id, source.seq)] for source in sources)
    fold_records, logical_items, hydrated_inputs = _prepare_fold_records(sources, hydrated)

    bodies = tuple(record_bodies) if record_bodies is not None else tuple(row.record.encode() for row in sources)
    if len(bodies) != len(sources):
        raise ValueError("record body count does not match source record count")
    current = base_fold.fork() if base_fold is not None else fold_factory()
    with logical_inputs(logical_items):
        state = current.extend(fold_records, consumed_inputs=logical_items.values(), bodies=bodies)

    if raw_applied_inputs:
        raw_by_id = {row.record.record_id: row.record for row in sources if row.record.kind == "input_applied"}
        state.applied_inputs = {
            input_id: raw_by_id.get(record.record_id, record) for input_id, record in state.applied_inputs.items()
        }
        if state.child_completions is not None:
            hydrated_by_id = {row.record.record_id: row.record for row in hydrated}
            state.child_completions = {
                key: hydrated_by_id.get(record.record_id, record) for key, record in state.child_completions.items()
            }
    else:
        state.applied_inputs = {
            input_id: hydrated_inputs[record.record_id].record if record.record_id in hydrated_inputs else record
            for input_id, record in state.applied_inputs.items()
        }
    for operation in state.operations.values():
        for attempt in operation.attempts.values():
            if attempt.approval_answer is not None:
                hydrated_input = hydrated_inputs.get(attempt.approval_answer.record_id)
                if hydrated_input is not None:
                    attempt.approval_answer = hydrated_input.record
    return HydratedFoldResult(current, state, hydrated)


def fold_hydrated_records(
    records: tuple[StoredRecord, ...],
    source_records: tuple[StoredRecord, ...],
    fold_factory: Callable[[], Fold],
    hydration: SessionHydration | None,
    *,
    base_fold: Fold | None = None,
    raw_applied_inputs: bool = False,
) -> ExecutionState:
    """SDK-compatible state view over the shared hydrated fold implementation."""

    result = hydrate_and_fold_records(
        source_records,
        fold_factory,
        hydration,
        hydrated_records=records,
        base_fold=base_fold,
        raw_applied_inputs=raw_applied_inputs,
    )
    return result.state


def validate_logical_preflight(
    driver: CoreDriver,
    records: list[Record],
    encoded_records: list[Record],
    *,
    base_fold: Fold | None = None,
) -> None:
    """Validate a provisional semantic transition without mutating the raw fold."""

    if driver.content_reader is None or not any(record.kind in _LOGICAL_PREFLIGHT_KINDS for record in records):
        return
    semantic_payloads: list[dict[str, Any]] = []
    for record in records:
        payload = copy_json(record._payload)
        for path, slot, value in _payload_slots(record.kind, record._payload):
            if record.kind == "input_applied" and path[:1] == ("input",):
                continue
            if value is None:
                continue
            try:
                retained = declared_content_ref(value)
            except ValueError as exc:
                raise TransitionError("logical preflight received a malformed ContentRef") from exc
            if retained is not None:
                expected_key = _REFERENCE_DIGESTS.get((record.kind, slot))
                if expected_key is not None:
                    expected = record._payload.get(expected_key)
                    if not isinstance(expected, str) or not hmac.compare_digest(reference_digest(retained), expected):
                        raise ReferenceDigestMismatch(f"{record.kind}/{slot} reference digest does not match {expected_key}")
                owner = driver._prepared_source_record(record, slot)
                value = None
                if owner is not None:
                    value = driver.read_content_json(f"record:{owner.seq}", slot, retained)
                elif record.kind == "op_planned" and slot in {"request", "tool", "budget_admission"}:
                    previous = next(
                        (
                            stored
                            for stored in reversed(driver.records)
                            if stored.record.kind == "op_planned"
                            and stored.record.operation_id == record.operation_id
                            and (stored.record.attempt or 0) < (record.attempt or 0)
                        ),
                        None,
                    )
                    if previous is not None:
                        previous_value = driver._content_field(previous.record, slot)
                        try:
                            previous_ref = ContentRef.from_dict(previous_value) if previous_value is not None else None
                        except (TypeError, ValueError):
                            previous_ref = None
                        if previous_ref == retained:
                            value = driver.hydrate_plan(previous.record)._payload[slot]
                if value is None:
                    value = driver.hydrate_record_field(record, slot, retained)
                _path_set(payload, path, value)
        semantic_payloads.append(payload)
    if len(records) != len(encoded_records):
        raise TransitionError("logical preflight record and committed body counts differ")

    reader, tenant_id = driver._content_reader_binding()
    hydration = SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
    base_seq = driver.records[-1].seq if driver.records else 0
    synthetic = tuple(
        StoredRecord(
            record=encoded,
            seq=base_seq + index,
            commit_id="preflight",
            writer_epoch=0,
            created_ms=0,
        )
        for index, encoded in enumerate(encoded_records, 1)
    )
    pending_rows: list[StoredRecord] = []
    for index, (source, semantic) in enumerate(zip(synthetic, records, strict=True)):
        payload = semantic_payloads[index]
        if semantic.kind == "input_applied":
            logical = hydrate_session_input(
                InboxItem(**payload["input"]),
                hydration,
                include_actor_attributes=True,
            )
            payload["input"] = logical.to_dict()
        hydrated_record = replace(semantic, payload=payload)
        hydrated_record._retain(source.record.encode(), hydrated_record.to_dict())
        pending_rows.append(replace(source, record=hydrated_record))
    pending = tuple(pending_rows)
    if base_fold is None:
        persisted = hydrate_session_records(driver.records, hydration)
        hydrate_and_fold_records(
            driver.records + synthetic,
            driver._fold_factory,
            hydration,
            hydrated_records=persisted + pending,
            record_bodies=tuple(row.record.encode() for row in driver.records)
            + tuple(record.encode() for record in encoded_records),
            include_input=True,
        )
    else:
        hydrate_and_fold_records(
            synthetic,
            driver._fold_factory,
            hydration,
            hydrated_records=pending,
            base_fold=base_fold,
            record_bodies=tuple(record.encode() for record in encoded_records),
            include_input=True,
        )


def definition_payload(driver: CoreDriver, record: Record) -> dict[str, Any]:
    """Return the hydrated definition as its current canonical JSON container."""
    return _definition_value(driver.hydrated_definition(record))


def active_task(driver: CoreDriver) -> AgentTask:
    """Resolve the active frozen task through the driver's owner-bound definition."""
    tid = driver.state.active_turn_id
    if tid is None:
        raise TypedSchemaFailure("active task requires an active turn")
    return driver.hydrated_definition(driver.state.turns[tid].start).task


def hydrate_record(driver: CoreDriver, record: Record) -> Record:
    """Hydrate one retained record without changing its committed envelope."""
    return hydrate_stored_record(driver, driver.stored_record(record))


def hydrate_stored_record(driver: CoreDriver, stored: StoredRecord) -> Record:
    """Hydrate one driver-owned record for a read-only producer decision."""
    reader, tenant_id = driver._content_reader_binding()
    hydration = SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
    return hydrate_session_stored_record(stored, hydration).record


def hydrate_stored_records(driver: CoreDriver, records: Iterable[StoredRecord]) -> tuple[StoredRecord, ...]:
    """Hydrate a chronological record prefix while retaining commit metadata."""
    reader, tenant_id = driver._content_reader_binding()
    hydration = SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
    return hydrate_session_records(records, hydration)


def hydrate_stored_payload(
    driver: CoreDriver,
    stored: StoredRecord,
    *,
    include_input: bool = True,
    include_actor_attributes: bool = False,
) -> dict[str, Any]:
    """Replace declared content paths while preserving every other stored field."""
    reader, tenant_id = driver._content_reader_binding()
    hydration = SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
    return hydrate_session_record_payload(
        stored,
        hydration,
        include_input=include_input,
        include_actor_attributes=include_actor_attributes,
    )


def hydrate_boundary_data(driver: CoreDriver, record: Record) -> dict[str, Any]:
    """Return the hydrated data object for one boundary record."""
    if record.kind != "boundary_recorded":
        raise TypedSchemaFailure("boundary hydration requires a boundary_recorded record")
    hydrated = hydrate_record(driver, record)
    data = hydrated.payload.get("data")
    if type(data) is not dict:
        raise TypedSchemaFailure("boundary_recorded.data must be an object")
    return copy_json(data)


def hydrate_context_compacted(driver: CoreDriver, record: Record) -> Record:
    """Hydrate a compaction replacement and its evidence manifest in array order."""
    if record.kind != "context_compacted":
        raise TypedSchemaFailure("content compaction hydration requires context_compacted")
    return hydrate_record(driver, record)


def hydrate_turn_ended(driver: CoreDriver, record: Record) -> Record:
    """Hydrate terminal reason, result and budget values."""
    if record.kind != "turn_ended":
        raise TypedSchemaFailure("terminal hydration requires turn_ended")
    return hydrate_record(driver, record)


def hydrate_input(driver: CoreDriver, item: InboxItem) -> InboxItem:
    """Hydrate a driver-owned inbox item, including its actor identity proof."""
    reader, tenant_id = driver._content_reader_binding()
    hydration = SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
    return hydrate_session_input(item, hydration, include_actor_attributes=True)


def _projection_value(slot: str, value: Any) -> Any:
    if slot == HydrationSlot.DEFINITION.value:
        return _definition_value(value)
    if slot == HydrationSlot.BUDGET.value:
        return value.to_dict()
    if slot == HydrationSlot.TASK.value:
        return value.to_dict()
    if slot == HydrationSlot.MESSAGES.value:
        return [message.to_dict() for message in value]
    return copy_json(value)


def _definition_value(definition: DefinitionContent) -> dict[str, Any]:
    return {
        "agent_name": definition.agent_name,
        "memory_settings": copy_json(definition.memory_settings),
        "child_tools": list(definition.child_tools),
        "model_binding": copy_json(definition.model_binding),
        "capabilities": copy_json(definition.capabilities),
        "task": definition.task.to_dict(),
        "tools": copy_json(list(definition.tools)),
    }


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
        value = value[int(part)] if isinstance(part, str) and part.isdecimal() else value[part]
    return value


def _path_set(value: Any, path: tuple[str | int, ...], replacement: Any) -> None:
    if not path:
        raise ValueError("content path cannot be empty")
    parent = _path_get(value, path[:-1]) if len(path) > 1 else value
    key = int(path[-1]) if isinstance(path[-1], str) and path[-1].isdecimal() else path[-1]
    parent[key] = replacement


__all__ = [
    "HydratedFoldResult",
    "RetainedInboxItem",
    "SessionHydration",
    "active_task",
    "definition_payload",
    "fold_hydrated_records",
    "hydrate_and_fold_records",
    "hydrate_boundary_data",
    "hydrate_context_compacted",
    "hydrate_input",
    "hydrate_record",
    "hydrate_session_input",
    "hydrate_session_input_payload",
    "hydrate_session_record_payload",
    "hydrate_session_records",
    "hydrate_session_stored_record",
    "hydrate_stored_payload",
    "hydrate_stored_record",
    "hydrate_stored_records",
    "hydrate_turn_ended",
    "validate_logical_preflight",
]
