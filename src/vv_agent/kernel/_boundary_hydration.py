"""Hydrate durable records and fold a read-only logical view without SDK dependencies."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
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
    PROVIDER_LOOKUP,
    NestedContentSlot,
    _content_slot_specs,
    _expand_nested_slots,
    _payload_slots,
    declared_content_ref,
    input_content_slots,
)
from ._content_reader import DurableContentReader, owner_scope
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
from ._records import InboxItem, Record, copy_json, validate_provider_lookup_evidence
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
        hydrate_nested: bool = True,
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
        nested = _nested_content_slots(record_kind, slot, record_payload) if hydrate_nested else ()
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


class _ScopedContentFrameReader:
    """Reuse one snapshot while preserving the original safe path on a frame miss."""

    def __init__(
        self,
        reader: DurableContentReader,
        fallback: ContentReader,
        runtime: Any | None,
        driver: CoreDriver,
        *,
        record_seqs: Iterable[int] = (),
        input_ids: Iterable[str] = (),
    ) -> None:
        self.reader = reader
        self.fallback = fallback
        self.runtime = runtime
        self.driver = driver
        self._values: dict[tuple[str, str, str, str, ContentRef], bytes] = {}
        self._fingerprints: dict[tuple[str, str, str, str, ContentRef], str] = {}
        self._record_seqs = frozenset(record_seqs)
        self._input_ids = frozenset(input_ids)

    def covers(self, *, record_seqs: Iterable[int], input_ids: Iterable[str]) -> bool:
        return self._record_seqs.issuperset(record_seqs) and self._input_ids.issuperset(input_ids)

    @property
    def fingerprints(self) -> dict[tuple[str, str, str, str, ContentRef], str]:
        return dict(self._fingerprints)

    @property
    def binding(self) -> tuple[str, str]:
        return self.reader.binding

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        key = (tenant_id, session_id, owner, slot, ref)
        cached = self._values.get(key)
        if cached is not None:
            return cached
        from_fallback = False
        try:
            value = self.reader.read(tenant_id, session_id, owner, slot, ref)
        except MissingContentLink:
            value = self.fallback.read(tenant_id, session_id, owner, slot, ref)
            from_fallback = True
        except KeyUnavailable:
            if self.runtime is not None:
                self.runtime._key_unavailable = True
            raise
        if from_fallback:
            return value
        self._values[key] = value
        self._fingerprints[key] = self._fingerprint(tenant_id, session_id, owner, slot, ref)
        return value

    def fingerprint(
        self,
        tenant_id: str,
        session_id: str,
        owner: str,
        slot: str,
        ref: ContentRef,
    ) -> str:
        self.assert_link(tenant_id, session_id, owner, slot, ref)
        return self._fingerprint(tenant_id, session_id, owner, slot, ref)

    def _fingerprint(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> str:
        store = getattr(self.reader, "_store", None)
        frame = getattr(store, "_frame", store)
        parsed = getattr(frame, "_parsed_content", {}).get(ref.content_id)
        row = getattr(frame, "_content", {}).get(ref.content_id)
        if parsed is None and row is None:
            fallback = self.fallback
            if isinstance(fallback, _ScopedContentFrameReader):
                return fallback.fingerprint(tenant_id, session_id, owner, slot, ref)
            raise IntegrityFailure("content frame fingerprint is unavailable")

        values: Iterable[Any]
        if parsed is not None:
            values = parsed[1:]
        else:
            assert row is not None
            values = row[5:]
        digest = hashlib.sha256()
        for value in values:
            if value is None:
                digest.update(b"N")
            elif isinstance(value, bytes):
                digest.update(b"B" + len(value).to_bytes(8, "big") + value)
            else:
                encoded = str(value).encode("utf-8")
                digest.update(b"V" + len(encoded).to_bytes(8, "big") + encoded)
        return digest.hexdigest()

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        owner: str,
        slot: str,
        ref: ContentRef,
    ) -> ContentRef:
        """Validate one owner/slot link without retaining a second plaintext copy."""

        owner_kind, owner_value = owner_scope(owner)
        try:
            return self.reader._store.assert_link(
                tenant_id,
                session_id,
                ref,
                slot,
                **{owner_kind: owner_value},
            )
        except (ContentUnavailable, ContentBindingMismatch) as exc:
            raise MissingContentLink(f"content owner link is unavailable: {owner}/{slot}") from exc
        except BackendKeyUnavailable as exc:
            if self.runtime is not None:
                self.runtime._key_unavailable = True
            raise KeyUnavailable("session content key is unavailable") from exc
        except BackendContentIntegrityError as exc:
            raise IntegrityFailure("content failed integrity validation") from exc
        except ContentStoreError as exc:
            raise HydrationError("content link validation failed") from exc

    @contextmanager
    def _read_frame(
        self,
        *,
        record_seqs: Iterable[int] = (),
        input_ids: Iterable[str] = (),
        validate_each_read: bool = True,
    ) -> Iterator[_ScopedContentFrameReader]:
        record_seq_values = tuple(record_seqs)
        input_id_values = tuple(input_ids)
        if self.covers(record_seqs=record_seq_values, input_ids=input_id_values):
            yield self
            return
        with _session_content_frame(
            self.driver,
            record_seqs=record_seq_values,
            input_ids=input_id_values,
        ) as hydration:
            yield hydration.reader


@contextmanager
def _session_content_frame(
    driver: CoreDriver,
    *,
    record_seqs: Iterable[int] = (),
    input_ids: Iterable[str] = (),
) -> Iterator[SessionHydration]:
    """Temporarily bind one logical batch to a disposable content frame."""

    reader, tenant_id = driver._content_reader_binding()
    record_seq_values = tuple(record_seqs)
    input_id_values = tuple(input_ids)
    if isinstance(reader, _ScopedContentFrameReader) and reader.covers(
        record_seqs=record_seq_values,
        input_ids=input_id_values,
    ):
        yield SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
        return
    fallback_reader = reader if isinstance(reader, _ScopedContentFrameReader) else getattr(reader, "fallback", reader)
    root_reader = fallback_reader
    while isinstance(root_reader, _ScopedContentFrameReader):
        root_reader = root_reader.fallback
    tracked_reader = getattr(root_reader, "reader", None)
    tracking_runtime = getattr(root_reader, "runtime", None) if isinstance(tracked_reader, DurableContentReader) else None
    durable_reader = tracked_reader if isinstance(tracked_reader, DurableContentReader) else root_reader
    if not isinstance(durable_reader, DurableContentReader):
        yield SessionHydration(reader, tenant_id=tenant_id, session_id=driver.sid)
        return

    with durable_reader._read_frame(
        record_seqs=record_seq_values,
        input_ids=input_id_values,
        validate_each_read=False,
    ) as frame_reader:
        previous_reader, previous_hydrators = driver.content_reader, driver._hydrators
        driver_reader = _ScopedContentFrameReader(
            frame_reader,
            fallback_reader,
            tracking_runtime,
            driver,
            record_seqs=record_seq_values,
            input_ids=input_id_values,
        )
        driver.content_reader, driver._hydrators = driver_reader, {}
        try:
            yield SessionHydration(driver_reader, tenant_id=tenant_id, session_id=driver.sid)
        finally:
            fingerprints = getattr(driver, "_logical_frame_fingerprints", None)
            if isinstance(fingerprints, dict):
                fingerprints.update(driver_reader.fingerprints)
            driver.content_reader, driver._hydrators = previous_reader, previous_hydrators


def hydrate_session_input_payload(
    item: InboxItem,
    hydration: SessionHydration | None,
) -> dict[str, Any]:
    """Hydrate declared input payload slots through their exact input owner."""

    payload = copy_json(item.payload)
    owner = f"input:{item.input_id}"
    if item.feature_level == 3 and item.kind == "provider_evidence":
        try:
            ref = declared_content_ref(payload.get(PROVIDER_LOOKUP))
        except ValueError as exc:
            raise TypedSchemaFailure(f"{owner}/{PROVIDER_LOOKUP} contains a malformed ContentRef") from exc
        if ref is None:
            raise TypedSchemaFailure(f"{owner}/{PROVIDER_LOOKUP} requires a ContentRef")
        if hydration is None:
            raise MissingContentLink(f"{owner}/{PROVIDER_LOOKUP} requires a content hydration reader")
        lookup = hydration.read(owner, PROVIDER_LOOKUP, ref)
        validate_provider_lookup_evidence(item, lookup, session_id=hydration.session_id)
        payload[PROVIDER_LOOKUP] = lookup
        return payload
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


def validate_logical_content_frame(
    records: Iterable[StoredRecord],
    hydration: SessionHydration,
    *,
    deep_record_ids: Iterable[str],
    expected_fingerprints: Mapping[tuple[str, str, str, str, ContentRef], str] | None = None,
) -> frozenset[str]:
    """Validate every raw owner link and deeply reopen only active logical content."""

    sources = tuple(records)
    reader = hydration.reader
    assert_link = getattr(reader, "assert_link", None)
    if not callable(assert_link):
        raise MissingContentLink("logical content frame does not support link-only validation")

    def checked_ref(owner: str, slot: str, value: Any) -> None:
        if value is _MISSING or value is None:
            return
        try:
            ref = declared_content_ref(value)
        except ValueError as exc:
            raise TypedSchemaFailure(f"{owner}/{slot} contains a malformed ContentRef") from exc
        if ref is None:
            raise TypedSchemaFailure(f"{owner}/{slot} requires a ContentRef")
        assert_link(hydration.tenant_id, hydration.session_id, owner, slot, ref)

    for stored in sources:
        record = stored.record
        owner = f"record:{stored.seq}"
        for _path, slot, value in _payload_slots(record.kind, record._payload):
            checked_ref(owner, slot, value)
        if record.kind != "input_applied":
            continue
        item = InboxItem(**record._payload["input"])
        input_owner = f"input:{item.input_id}"
        for content_slot in input_content_slots(item.kind):
            try:
                value = _path_get(item.payload, content_slot.path)
            except (KeyError, IndexError):
                continue
            checked_ref(input_owner, content_slot.slot, value)

    fingerprint = getattr(reader, "fingerprint", None)
    if not callable(fingerprint):
        raise MissingContentLink("logical content frame does not support content fingerprint validation")
    for (tenant_id, session_id, owner, slot, ref), expected in (expected_fingerprints or {}).items():
        actual = fingerprint(tenant_id, session_id, owner, slot, ref)
        if not hmac.compare_digest(actual, expected):
            raise IntegrityFailure("content frame changed since logical hydration")

    selected_ids = frozenset(deep_record_ids)
    selected = tuple(stored for stored in sources if stored.record.record_id in selected_ids)
    hydrate_session_records(selected, hydration)
    return selected_ids


def hydrate_session_input(
    item: InboxItem,
    hydration: SessionHydration | None,
    *,
    include_actor_attributes: bool = False,
) -> InboxItem:
    """Hydrate one input while optionally resolving the actor identity proof."""

    if hydration is None or not callable(getattr(hydration.reader, "_read_frame", None)):
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
    hydrate_nested: bool = True,
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
            hydrate_nested=hydrate_nested,
        )
        if _decoded_values is not None:
            _decoded_values[path] = hydrated_value
        _path_set(
            payload,
            path,
            projected,
        )
    if include_input and _lookup_input_owner(record) is not None:
        if hydration is None or record.session_id != hydration.session_id:
            raise MissingContentLink("applied lookup requires its original session's inbox reader")
        item = InboxItem(**source["input"])
        # The proof and encrypted lookup remain bound to the original inbox row.
        logical = _hydrate_session_input(item, hydration, include_actor_attributes=False)
        payload["input"]["payload"] = logical.payload
    return payload


def _lookup_input_owner(record: Record) -> str | None:
    if record.kind == "input_applied":
        item = record._payload["input"]
        if item["kind"] == "provider_evidence" and item["feature_level"] == 3:
            return item["input_id"]
    return None


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
    hydrate_nested: bool = True,
) -> StoredRecord:
    retained_body = stored.record.encode()
    decoded_values: dict[tuple[str | int, ...], Any] = {}
    payload = hydrate_session_record_payload(
        stored,
        hydration,
        include_input=include_input,
        include_actor_attributes=include_actor_attributes,
        hydrate_nested=hydrate_nested,
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
    shallow_nested_record_ids: Iterable[str] = (),
) -> tuple[StoredRecord, ...]:
    return _hydrate_session_records_batch(
        tuple(records),
        hydration,
        include_input=include_input,
        include_actor_attributes=include_actor_attributes,
        shallow_nested_record_ids=shallow_nested_record_ids,
    )


def _hydrate_session_records_batch(
    sources: tuple[StoredRecord, ...],
    hydration: SessionHydration | None,
    *,
    include_input: bool,
    include_actor_attributes: bool,
    shallow_nested_record_ids: Iterable[str] = (),
) -> tuple[StoredRecord, ...]:
    if not sources:
        return ()
    shallow_ids = frozenset(shallow_nested_record_ids)
    if hydration is None or not callable(getattr(hydration.reader, "_read_frame", None)):
        return tuple(
            hydrate_session_stored_record(
                stored,
                hydration,
                include_input=include_input,
                include_actor_attributes=include_actor_attributes,
                hydrate_nested=stored.record.record_id not in shallow_ids,
            )
            for stored in sources
        )
    with hydration.reader._read_frame(
        record_seqs=(stored.seq for stored in sources),
        input_ids=(
            input_id for stored in sources if include_input and (input_id := _lookup_input_owner(stored.record)) is not None
        ),
        validate_each_read=False,
    ) as reader:
        framed = replace(hydration, reader=reader, _validated_json_refs=set())
        return tuple(
            hydrate_session_stored_record(
                stored,
                framed,
                include_input=include_input,
                include_actor_attributes=include_actor_attributes,
                hydrate_nested=stored.record.record_id not in shallow_ids,
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
    deep_record_ids: Iterable[str] | None = None,
    base_fold: Fold | None = None,
    record_bodies: Iterable[bytes] | None = None,
    include_input: bool = True,
    raw_applied_inputs: bool = False,
) -> HydratedFoldResult:
    """Fold hydrated values while retaining committed bytes as record identity."""

    sources = tuple(source_records)
    source_by_identity = {(row.record.record_id, row.seq): row for row in sources}
    hydrated_by_identity: dict[tuple[str, int], StoredRecord] = {}
    deep_ids = None if deep_record_ids is None else frozenset(deep_record_ids)
    shallow_nested_record_ids = (
        set()
        if deep_ids is None
        else {
            stored.record.record_id
            for stored in sources
            if stored.record.kind == "op_planned" and stored.record.record_id not in deep_ids
        }
    )

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
            shallow_nested_record_ids=shallow_nested_record_ids,
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
                shallow_nested_record_ids=shallow_nested_record_ids,
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


def _logical_preflight_frame_owners(
    driver: CoreDriver,
    records: list[Record],
    *,
    base_fold: Fold | None,
) -> tuple[set[int], set[str]]:
    record_seqs = {stored.seq for stored in driver.records} if base_fold is None else set()
    input_ids = {record._payload["input"]["input_id"] for record in records if record.kind == "input_applied"}
    if base_fold is None:
        input_ids.update(input_id for stored in driver.records if (input_id := _lookup_input_owner(stored.record)) is not None)

    for record in records:
        for _path, slot, value in _payload_slots(record.kind, record._payload):
            if record.kind == "input_applied" and _path[:1] == ("input",):
                continue
            if value is None:
                continue
            try:
                retained = declared_content_ref(value)
            except ValueError:
                continue
            if retained is None:
                continue
            owner = driver._prepared_source_record(record, slot)
            if owner is not None:
                record_seqs.add(owner.seq)
                continue
            if record.kind != "op_planned" or slot not in {"request", "tool", "budget_admission"}:
                continue
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
            if previous is not None and driver._content_field(previous.record, slot) == retained.to_dict():
                record_seqs.add(previous.seq)

    return record_seqs, input_ids


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
    record_seqs, input_ids = _logical_preflight_frame_owners(driver, records, base_fold=base_fold)
    with _session_content_frame(driver, record_seqs=record_seqs, input_ids=input_ids) as hydration:
        _validate_logical_preflight_in_frame(driver, records, encoded_records, hydration, base_fold=base_fold)


def _validate_logical_preflight_in_frame(
    driver: CoreDriver,
    records: list[Record],
    encoded_records: list[Record],
    hydration: SessionHydration,
    *,
    base_fold: Fold | None,
) -> None:
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
