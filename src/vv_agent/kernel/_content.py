"""Content references and lifecycle rules for the S1 session store boundary.

The module contains no encryption backend or durable storage. It validates the
closed wire values and returns the decisions that providers and stores must
persist atomically. Identifier fields are opaque, non-empty strings.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from ._canonical_json import canonical_json_bytes
from ._record_codec import RecordError, validate
from ._record_schema import CONTENT_LABELS, CONTENT_REF, PRINCIPAL
from ._records import InboxItem, Record, SessionSpec

SESSION_PRINCIPAL_ATTRIBUTES = "session.principal.attributes"
SESSION_WORKSPACE = "session.workspace"
SESSION_ATTRIBUTES = "session.attributes"
INPUT_CONTENT = "input.content"
INPUT_SELECTION = "input.selection"
INPUT_ACTOR_ATTRIBUTES = "input.actor.attributes"
APPROVAL_SCOPE = "approval.scope"
APPROVAL_REASON = "approval.reason"
APPROVAL_METADATA = "approval.metadata"
CHILD_RESULT = "child.result"


class ContentErrorCode(StrEnum):
    INVALID_CONTENT_REF = "invalid_content_ref"
    BINDING_MISMATCH = "binding_mismatch"
    INVALID_DESTROY_OUTCOME = "invalid_destroy_outcome"
    INVALID_TOMBSTONE = "invalid_tombstone"
    TOMBSTONE_HEAD_MISMATCH = "tombstone_head_mismatch"
    FORGOTTEN_SESSION_REUSE = "forgotten_session_reuse"
    INVALID_EXPORT = "invalid_export"
    UNSUPPORTED_EXPORT_VERSION = "unsupported_export_version"
    EXPORT_MODE_MISMATCH = "export_mode_mismatch"


class ContentLifecycleError(ValueError):
    """A classified rejection at the content lifecycle boundary."""

    code = ContentErrorCode.INVALID_CONTENT_REF


class InvalidContentRef(ContentLifecycleError):
    code = ContentErrorCode.INVALID_CONTENT_REF


class ContentBindingMismatch(ContentLifecycleError):
    code = ContentErrorCode.BINDING_MISMATCH


class ContentStoreError(RuntimeError):
    """Base class for content backend failures."""


class ContentUnavailable(ContentStoreError):
    """The requested tenant, session, or content object is absent."""


class KeyUnavailable(ContentStoreError):
    """The session data key is absent, destroyed, or cannot be unwrapped."""


class ContentIntegrityError(ContentStoreError):
    """Content metadata, ciphertext, or authentication material is invalid."""


class ContentLinkConflict(ContentStoreError):
    """An owner slot is already linked to different content."""


class InvalidDestroyOutcome(ContentLifecycleError):
    code = ContentErrorCode.INVALID_DESTROY_OUTCOME


class InvalidTombstone(ContentLifecycleError):
    code = ContentErrorCode.INVALID_TOMBSTONE


class TombstoneHeadMismatch(ContentLifecycleError):
    code = ContentErrorCode.TOMBSTONE_HEAD_MISMATCH


class ForgottenSessionReuse(ContentLifecycleError):
    code = ContentErrorCode.FORGOTTEN_SESSION_REUSE


class InvalidExport(ContentLifecycleError):
    code = ContentErrorCode.INVALID_EXPORT


class UnsupportedExportVersion(ContentLifecycleError):
    code = ContentErrorCode.UNSUPPORTED_EXPORT_VERSION


class ExportModeMismatch(ContentLifecycleError):
    code = ContentErrorCode.EXPORT_MODE_MISMATCH


def _closed_mapping(value: Any, fields: set[str], error: type[ContentLifecycleError], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise error(f"{label} must be a closed object")
    return value


def _text(value: Any, field_name: str, error: type[ContentLifecycleError]) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{field_name} must be a non-empty string")
    return value


def _integer(
    value: Any,
    field_name: str,
    error: type[ContentLifecycleError],
    *,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise error(f"{field_name} must be an integer >= {minimum}")
    return value


def _hash(value: Any, field_name: str, error: type[ContentLifecycleError]) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise error(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _signature(value: Any, field_name: str, error: type[ContentLifecycleError]) -> str:
    text = _text(value, field_name, error)
    try:
        decoded = base64.b64decode(text, validate=True)
    except ValueError as exc:
        raise error(f"{field_name} must be canonical base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != text:
        raise error(f"{field_name} must be canonical base64")
    return text


def _validate_shared(value: dict[str, Any], schema: dict[str, Any], error: type[ContentLifecycleError], label: str) -> None:
    try:
        validate(value, schema, canonical=True)
    except RecordError as exc:
        raise error(f"{label} does not match the contract schema") from exc


class SourceKind(StrEnum):
    SYSTEM = "system"
    USER = "user"
    INTERNAL = "internal"
    UNTRUSTED = "untrusted"


class Classification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


_SOURCE_ORDER = {kind: index for index, kind in enumerate(SourceKind)}


@dataclass(frozen=True, slots=True)
class ContentLabels:
    source: tuple[SourceKind, ...] = ()
    classification: Classification = Classification.INTERNAL

    def __post_init__(self) -> None:
        try:
            source = tuple(SourceKind(item) for item in self.source)
            classification = Classification(self.classification)
        except (TypeError, ValueError) as exc:
            raise InvalidContentRef("ContentLabels contains an unknown value") from exc
        if len(source) != len(set(source)):
            raise InvalidContentRef("ContentRef labels.source contains duplicates")
        if [_SOURCE_ORDER[item] for item in source] != sorted(_SOURCE_ORDER[item] for item in source):
            raise InvalidContentRef("ContentRef labels.source is not in source-kind order")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "classification", classification)

    def to_dict(self) -> dict[str, Any]:
        return {"source": [item.value for item in self.source], "classification": self.classification.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContentLabels:
        payload = _closed_mapping(value, {"source", "classification"}, InvalidContentRef, "ContentLabels")
        _validate_shared(payload, CONTENT_LABELS, InvalidContentRef, "ContentLabels")
        return cls(tuple(SourceKind(item) for item in payload["source"]), Classification(payload["classification"]))


@dataclass(frozen=True, slots=True)
class ContentRef:
    content_id: str
    mac: str
    size: int
    key_id: str
    labels: ContentLabels

    def __post_init__(self) -> None:
        _text(self.content_id, "ContentRef.content_id", InvalidContentRef)
        _hash(self.mac, "ContentRef.mac", InvalidContentRef)
        _integer(self.size, "ContentRef.size", InvalidContentRef, minimum=0)
        _text(self.key_id, "ContentRef.key_id", InvalidContentRef)
        if not isinstance(self.labels, ContentLabels):
            raise InvalidContentRef("ContentRef.labels must be ContentLabels")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "mac": self.mac,
            "size": self.size,
            "key_id": self.key_id,
            "labels": self.labels.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContentRef:
        payload = _closed_mapping(value, {"content_id", "mac", "size", "key_id", "labels"}, InvalidContentRef, "ContentRef")
        _validate_shared(payload, CONTENT_REF, InvalidContentRef, "ContentRef")
        return cls(
            content_id=payload["content_id"],
            mac=payload["mac"],
            size=payload["size"],
            key_id=payload["key_id"],
            labels=ContentLabels.from_dict(payload["labels"]),
        )


@runtime_checkable
class ContentSessionWriter(Protocol):
    """Kernel-visible content operations already bound to one session transaction."""

    def put_json(self, value: Any, *, labels: ContentLabels = ...) -> ContentRef: ...

    def get_json(self, ref: ContentRef | Mapping[str, Any]) -> Any: ...

    def copy_json_from(
        self,
        source: ContentSessionWriter,
        ref: ContentRef | Mapping[str, Any],
    ) -> ContentRef: ...

    def service_principal(
        self,
        subject_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ContentRef]: ...

    def link_input_actor(self, input_id: str, ref: ContentRef) -> None: ...

    def link_input_content(self, input_id: str, ref: ContentRef) -> None: ...

    def link_child_result(self, input_id: str, ref: ContentRef) -> None: ...

    def assert_input(self, input_id: str, ref: ContentRef | Mapping[str, Any], slot: str) -> ContentRef: ...

    def assert_record(self, record_seq: int, ref: ContentRef | Mapping[str, Any], slot: str) -> ContentRef: ...

    def create_session(
        self,
        spec_factory: Callable[[ContentSessionWriter], SessionSpec],
        *,
        consumers: tuple[str, ...],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ContentBinding:
    tenant_id: str
    session_id: str
    key_id: str

    def __post_init__(self) -> None:
        _text(self.tenant_id, "tenant_id", ContentBindingMismatch)
        _text(self.session_id, "session_id", ContentBindingMismatch)
        _text(self.key_id, "session key_id", ContentBindingMismatch)


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """Session-local deduplication identity: ``(session key, mac, labels)``."""

    key_id: str
    mac: str
    labels: ContentLabels

    def __post_init__(self) -> None:
        _text(self.key_id, "ContentIdentity.key_id", ContentBindingMismatch)
        _hash(self.mac, "ContentIdentity.mac", InvalidContentRef)
        if not isinstance(self.labels, ContentLabels):
            raise InvalidContentRef("ContentIdentity.labels must be ContentLabels")

    @classmethod
    def from_ref(cls, ref: ContentRef, binding: ContentBinding) -> ContentIdentity:
        if ref.key_id != binding.key_id:
            raise ContentBindingMismatch("ContentRef key does not belong to the session binding")
        return cls(binding.key_id, ref.mac, ref.labels)


def bind_content_ref(ref: ContentRef, binding: ContentBinding) -> ContentIdentity:
    return ContentIdentity.from_ref(ref, binding)


class KeyCapabilityState(StrEnum):
    ACTIVE = "active"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


class KeyDestroyStatus(StrEnum):
    DESTROYED = "destroyed"
    DESTROYING = "destroying"
    ALREADY_DESTROYED = "already_destroyed"
    KEY_UNAVAILABLE = "key_unavailable"
    BINDING_MISMATCH = "binding_mismatch"


@dataclass(frozen=True, slots=True)
class SessionKeyCapability:
    tenant_id: str
    session_id: str
    key_id: str
    provider_id: str
    capability_id: str
    state: KeyCapabilityState

    def __post_init__(self) -> None:
        _text(self.tenant_id, "capability.tenant_id", ContentBindingMismatch)
        _text(self.session_id, "capability.session_id", ContentBindingMismatch)
        _text(self.key_id, "capability.key_id", ContentBindingMismatch)
        _text(self.provider_id, "capability.provider_id", InvalidDestroyOutcome)
        _text(self.capability_id, "capability.capability_id", InvalidDestroyOutcome)
        try:
            object.__setattr__(self, "state", KeyCapabilityState(self.state))
        except ValueError as exc:
            raise InvalidDestroyOutcome("capability state is not recognized") from exc

    def require_binding(self, tenant_id: str, session_id: str, key_id: str) -> None:
        if (self.tenant_id, self.session_id, self.key_id) != (tenant_id, session_id, key_id):
            raise ContentBindingMismatch("Capability identity includes tenant, session, and key.")

    def destroy_status(self, tenant_id: str, key_id: str) -> KeyDestroyStatus:
        if (self.tenant_id, self.key_id) != (tenant_id, key_id):
            return KeyDestroyStatus.BINDING_MISMATCH
        return {
            KeyCapabilityState.ACTIVE: KeyDestroyStatus.DESTROYED,
            KeyCapabilityState.DESTROYING: KeyDestroyStatus.DESTROYING,
            KeyCapabilityState.DESTROYED: KeyDestroyStatus.ALREADY_DESTROYED,
        }[self.state]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SessionKeyCapability:
        fields = {"tenant_id", "session_id", "key_id", "provider_id", "capability_id", "state"}
        payload = _closed_mapping(value, fields, InvalidDestroyOutcome, "key capability")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class KeyDestroyRequest:
    tenant_id: str
    key_id: str
    domain: str = "vv-agent/S1/key-destroy"

    def __post_init__(self) -> None:
        _text(self.domain, "destroy domain", InvalidDestroyOutcome)
        _text(self.tenant_id, "destroy tenant_id", ContentBindingMismatch)
        _text(self.key_id, "destroy key_id", ContentBindingMismatch)
        if self.domain != "vv-agent/S1/key-destroy":
            raise InvalidDestroyOutcome("destroy domain is closed")


@dataclass(frozen=True, slots=True)
class KeyDestroyOutcome:
    accepted: bool
    status: KeyDestroyStatus
    evidence_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            status = KeyDestroyStatus(self.status)
        except ValueError as exc:
            raise InvalidDestroyOutcome("destroy status is not recognized") from exc
        accepted_statuses = {
            KeyDestroyStatus.DESTROYED,
            KeyDestroyStatus.DESTROYING,
            KeyDestroyStatus.ALREADY_DESTROYED,
        }
        rejected_statuses = {KeyDestroyStatus.KEY_UNAVAILABLE, KeyDestroyStatus.BINDING_MISMATCH}
        if type(self.accepted) is not bool or self.accepted != (status in accepted_statuses):
            raise InvalidDestroyOutcome("destroy acceptance does not match status")
        if status in rejected_statuses and (not isinstance(self.reason, str) or not self.reason):
            raise InvalidDestroyOutcome("rejected destroy outcomes require a reason")
        if (
            self.accepted and status in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED}
        ) or self.evidence_id is not None:
            _text(self.evidence_id, "destroy evidence_id", InvalidDestroyOutcome)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class WrappedDataKey:
    """A session-bound data-key envelope returned by a key provider."""

    tenant_id: str
    session_id: str
    key_id: str
    wrapped_key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.tenant_id, "data key tenant_id", ContentBindingMismatch)
        _text(self.session_id, "data key session_id", ContentBindingMismatch)
        _text(self.key_id, "data key key_id", ContentBindingMismatch)
        if not isinstance(self.wrapped_key, bytes) or not self.wrapped_key:
            raise ContentBindingMismatch("wrapped_key must be non-empty bytes")


@runtime_checkable
class KeyProvider(Protocol):
    """Backend contract for session data keys and atomic key revocation."""

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey: ...

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes: ...

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome: ...


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    subject_id: str
    subject_kind: str
    attributes: ContentRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "attributes": self.attributes.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Principal:
        payload = _closed_mapping(value, {"tenant_id", "subject_id", "subject_kind", "attributes"}, InvalidTombstone, "Principal")
        _validate_shared(payload, PRINCIPAL, InvalidTombstone, "Principal")
        return cls(
            tenant_id=payload["tenant_id"],
            subject_id=payload["subject_id"],
            subject_kind=payload["subject_kind"],
            attributes=ContentRef.from_dict(payload["attributes"]),
        )


@dataclass(frozen=True, slots=True)
class SessionHead:
    tenant_id: str
    session_id: str
    final_seq: int
    chain_digest: str
    key_id: str

    def __post_init__(self) -> None:
        _text(self.tenant_id, "head tenant_id", ContentBindingMismatch)
        _text(self.session_id, "head session_id", ContentBindingMismatch)
        _integer(self.final_seq, "head final_seq", InvalidTombstone, minimum=1)
        _hash(self.chain_digest, "head chain_digest", InvalidTombstone)
        _text(self.key_id, "head key_id", ContentBindingMismatch)


class TombstoneState(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class ForgetTombstone:
    tenant_id: str
    session_id: str
    final_seq: int
    chain_digest: str
    deleted_ms: int
    actor: Principal
    key_id: str
    signature: str
    state: TombstoneState
    scope_root_session_id: str

    def __post_init__(self) -> None:
        _text(self.tenant_id, "tombstone tenant_id", ContentBindingMismatch)
        _text(self.session_id, "tombstone session_id", ContentBindingMismatch)
        _integer(self.final_seq, "tombstone final_seq", InvalidTombstone, minimum=1)
        _hash(self.chain_digest, "tombstone chain_digest", InvalidTombstone)
        _integer(self.deleted_ms, "tombstone deleted_ms", InvalidTombstone, minimum=0)
        if not isinstance(self.actor, Principal):
            raise InvalidTombstone("tombstone actor must be Principal")
        _text(self.key_id, "tombstone key_id", ContentBindingMismatch)
        _signature(self.signature, "tombstone signature", InvalidTombstone)
        try:
            object.__setattr__(self, "state", TombstoneState(self.state))
        except ValueError as exc:
            raise InvalidTombstone("tombstone state is not recognized") from exc
        _text(self.scope_root_session_id, "tombstone scope_root_session_id", InvalidTombstone)

    def verify(self, head: SessionHead) -> None:
        """Verify that a complete tombstone still identifies the forgotten head."""
        if (
            self.tenant_id != head.tenant_id
            or self.session_id != head.session_id
            or self.key_id != head.key_id
            or self.actor.tenant_id != self.tenant_id
            or self.actor.attributes.key_id != self.key_id
        ):
            raise ContentBindingMismatch("tombstone tenant, session, actor, or key binding does not match")
        if self.scope_root_session_id != self.session_id:
            raise ForgottenSessionReuse("A forgotten session identity cannot be reused.")
        if self.state is not TombstoneState.COMPLETE:
            raise InvalidTombstone("forget verification requires a complete tombstone")
        if (self.final_seq, self.chain_digest) != (head.final_seq, head.chain_digest):
            raise TombstoneHeadMismatch("The tombstone head must match the retained final record.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForgetTombstone:
        fields = {
            "tenant_id",
            "session_id",
            "final_seq",
            "chain_digest",
            "deleted_ms",
            "actor",
            "key_id",
            "signature",
            "state",
            "scope_root_session_id",
        }
        payload = _closed_mapping(value, fields, InvalidTombstone, "forget tombstone")
        return cls(
            tenant_id=payload["tenant_id"],
            session_id=payload["session_id"],
            final_seq=payload["final_seq"],
            chain_digest=payload["chain_digest"],
            deleted_ms=payload["deleted_ms"],
            actor=Principal.from_dict(payload["actor"]),
            key_id=payload["key_id"],
            signature=payload["signature"],
            state=payload["state"],
            scope_root_session_id=payload["scope_root_session_id"],
        )


class ExportContentMode(StrEnum):
    SKELETON = "skeleton"
    INCLUDE_CONTENT = "include-content"


@dataclass(frozen=True, slots=True)
class ExportHeader:
    tenant_id: str
    session_id: str
    head_seq: int
    chain_digest: str
    content_mode: ExportContentMode
    export_version: int = 1
    storage_major: str = "S1"

    def __post_init__(self) -> None:
        if type(self.export_version) is not int or self.export_version != 1:
            raise UnsupportedExportVersion("Only export_version 1 is current.")
        if self.storage_major != "S1":
            raise UnsupportedExportVersion("Only storage_major S1 is current.")
        _text(self.tenant_id, "export tenant_id", InvalidExport)
        _text(self.session_id, "export session_id", InvalidExport)
        _integer(self.head_seq, "export head_seq", InvalidExport, minimum=0)
        _hash(self.chain_digest, "export chain_digest", InvalidExport)
        try:
            object.__setattr__(self, "content_mode", ExportContentMode(self.content_mode))
        except ValueError as exc:
            raise ExportModeMismatch("content_mode is closed.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": "header",
            "export_version": self.export_version,
            "storage_major": self.storage_major,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "head_seq": self.head_seq,
            "chain_digest": self.chain_digest,
            "content_mode": self.content_mode.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExportHeader:
        fields = {
            "line",
            "export_version",
            "storage_major",
            "tenant_id",
            "session_id",
            "head_seq",
            "chain_digest",
            "content_mode",
        }
        payload = _closed_mapping(value, fields, InvalidExport, "export header")
        if payload["line"] != "header":
            raise InvalidExport("export header line discriminator is invalid")
        return cls(
            tenant_id=payload["tenant_id"],
            session_id=payload["session_id"],
            head_seq=payload["head_seq"],
            chain_digest=payload["chain_digest"],
            content_mode=payload["content_mode"],
            export_version=payload["export_version"],
            storage_major=payload["storage_major"],
        )


class ExportLineKind(StrEnum):
    RECORD = "record"
    INBOX = "inbox"
    CONTENT = "content"
    CHECKPOINT = "checkpoint"
    MANIFEST = "manifest"


@dataclass(frozen=True, slots=True)
class ExportLine:
    kind: ExportLineKind
    value: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExportLine:
        if type(value) is not dict:
            raise InvalidExport("export line must be an object")
        try:
            canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise InvalidExport("export line must be canonical JSON data") from exc
        kind_value = value.get("line")
        try:
            kind = ExportLineKind(kind_value)
        except (TypeError, ValueError) as exc:
            raise InvalidExport("export line discriminator is invalid") from exc
        payload = cast(dict[str, Any], deepcopy(value))
        _validate_export_line(kind, payload)
        return cls(kind, payload)


def _validate_storage_envelope(value: Any) -> None:
    fields = {
        "schema_version",
        "tenant_id",
        "session_id",
        "seq",
        "record_digest",
        "chain_digest",
        "writer_epoch",
        "created_ms",
    }
    envelope = _closed_mapping(value, fields, InvalidExport, "storage envelope")
    if envelope["schema_version"] != "S1":
        raise InvalidExport("storage envelope schema_version must be S1")
    _text(envelope["tenant_id"], "storage envelope tenant_id", InvalidExport)
    _text(envelope["session_id"], "storage envelope session_id", InvalidExport)
    _integer(envelope["seq"], "storage envelope seq", InvalidExport, minimum=1)
    _hash(envelope["record_digest"], "storage envelope record_digest", InvalidExport)
    _hash(envelope["chain_digest"], "storage envelope chain_digest", InvalidExport)
    _integer(envelope["writer_epoch"], "storage envelope writer_epoch", InvalidExport, minimum=0)
    _integer(envelope["created_ms"], "storage envelope created_ms", InvalidExport, minimum=0)


def _validate_export_line(kind: ExportLineKind, value: dict[str, Any]) -> None:
    if kind is ExportLineKind.RECORD:
        _closed_mapping(value, {"line", "storage_envelope", "record"}, InvalidExport, "record export line")
        _validate_storage_envelope(value["storage_envelope"])
        if type(value["record"]) is not dict:
            raise InvalidExport("record export line requires a record object")
        try:
            record = Record.parse(canonical_json_bytes(value["record"]))
        except (RecordError, TypeError, ValueError) as exc:
            raise InvalidExport("record export line contains an invalid S1 record") from exc
        if value["storage_envelope"]["record_digest"] != record.digest:
            raise InvalidExport("record export line digest does not match the record")
        if value["storage_envelope"]["session_id"] != record.session_id:
            raise InvalidExport("record export line session does not match its record")
    elif kind is ExportLineKind.INBOX:
        _closed_mapping(value, {"line", "input_seq", "received_ms", "consumed_seq", "input"}, InvalidExport, "inbox export line")
        _integer(value["input_seq"], "inbox input_seq", InvalidExport, minimum=1)
        _integer(value["received_ms"], "inbox received_ms", InvalidExport, minimum=0)
        if value["consumed_seq"] is not None:
            _integer(value["consumed_seq"], "inbox consumed_seq", InvalidExport, minimum=0)
        if type(value["input"]) is not dict:
            raise InvalidExport("inbox export line requires an input object")
        try:
            InboxItem.parse(canonical_json_bytes(value["input"]))
        except (RecordError, TypeError, ValueError) as exc:
            raise InvalidExport("inbox export line contains an invalid S1 inbox item") from exc
    elif kind is ExportLineKind.CONTENT:
        _closed_mapping(value, {"line", "content_id", "key_id", "slot", "decoded_value"}, InvalidExport, "content export line")
        _text(value["content_id"], "content export content_id", InvalidExport)
        _text(value["key_id"], "content export key_id", InvalidExport)
        _text(value["slot"], "content export slot", InvalidExport)
    elif kind is ExportLineKind.CHECKPOINT:
        fields = {"line", "seq", "chain_digest", "key_id", "signature", "signed_ms"}
        _closed_mapping(value, fields, InvalidExport, "checkpoint export line")
        _integer(value["seq"], "checkpoint seq", InvalidExport, minimum=1)
        _hash(value["chain_digest"], "checkpoint chain_digest", InvalidExport)
        _text(value["key_id"], "checkpoint key_id", InvalidExport)
        _signature(value["signature"], "checkpoint signature", InvalidExport)
        _integer(value["signed_ms"], "checkpoint signed_ms", InvalidExport, minimum=0)
    else:
        fields = {"line", "record_lines", "inbox_lines", "checkpoint_lines", "head_seq", "chain_digest"}
        _closed_mapping(value, fields, InvalidExport, "manifest export line")
        _integer(value["record_lines"], "manifest record_lines", InvalidExport, minimum=0)
        _integer(value["inbox_lines"], "manifest inbox_lines", InvalidExport, minimum=0)
        _integer(value["checkpoint_lines"], "manifest checkpoint_lines", InvalidExport, minimum=0)
        _integer(value["head_seq"], "manifest head_seq", InvalidExport, minimum=0)
        _hash(value["chain_digest"], "manifest chain_digest", InvalidExport)


def _walk(value: Any):
    yield value
    if type(value) is dict:
        for item in value.values():
            yield from _walk(item)
    elif type(value) is list:
        for item in value:
            yield from _walk(item)


def _content_refs(value: Any) -> list[ContentRef]:
    refs: list[ContentRef] = []
    for node in _walk(value):
        if type(node) is dict and set(node) == {"content_id", "mac", "size", "key_id", "labels"}:
            refs.append(ContentRef.from_dict(node))
    return refs


@dataclass(frozen=True, slots=True)
class ExportDocument:
    header: ExportHeader
    lines: tuple[ExportLine, ...]

    def verify(self, head: SessionHead) -> None:
        """Verify export framing, manifest counts, and content-mode isolation."""
        if (
            self.header.tenant_id != head.tenant_id
            or self.header.session_id != head.session_id
            or self.header.head_seq != head.final_seq
            or self.header.chain_digest != head.chain_digest
        ):
            raise ContentBindingMismatch("export header does not match the retained session head")
        manifests = [line for line in self.lines if line.kind is ExportLineKind.MANIFEST]
        if len(manifests) != 1 or self.lines[-1].kind is not ExportLineKind.MANIFEST:
            raise InvalidExport("export requires exactly one final manifest line")
        if self.header.content_mode is ExportContentMode.SKELETON and any(
            line.kind is ExportLineKind.CONTENT for line in self.lines
        ):
            raise ExportModeMismatch("skeleton content_mode excludes content lines.")
        manifest = manifests[0].value
        if manifest["head_seq"] != self.header.head_seq or manifest["chain_digest"] != self.header.chain_digest:
            raise InvalidExport("manifest does not match the export header")
        counts = {
            ExportLineKind.RECORD: 0,
            ExportLineKind.INBOX: 0,
            ExportLineKind.CHECKPOINT: 0,
        }
        for line in self.lines:
            if line.kind in counts:
                counts[line.kind] += 1
        if (
            manifest["record_lines"] != counts[ExportLineKind.RECORD]
            or manifest["inbox_lines"] != counts[ExportLineKind.INBOX]
            or manifest["checkpoint_lines"] != counts[ExportLineKind.CHECKPOINT]
        ):
            raise InvalidExport("manifest line counts do not match the export")

        available_refs: dict[str, ContentRef] = {}
        for line in self.lines:
            if line.kind is ExportLineKind.RECORD:
                envelope = line.value["storage_envelope"]
                if envelope["tenant_id"] != self.header.tenant_id:
                    raise ContentBindingMismatch("record envelope tenant does not match the export")
                refs = _content_refs(line.value["record"])
            elif line.kind is ExportLineKind.INBOX:
                item = InboxItem.parse(canonical_json_bytes(line.value["input"]))
                if item.actor is None or item.actor["tenant_id"] != self.header.tenant_id:
                    raise ContentBindingMismatch("inbox actor tenant does not match the export")
                refs = _content_refs(line.value["input"])
            else:
                continue
            for ref in refs:
                previous = available_refs.get(ref.content_id)
                if previous is not None and previous != ref:
                    raise ContentBindingMismatch("one content id maps to conflicting content references")
                available_refs[ref.content_id] = ref

        for line in self.lines:
            if line.kind is not ExportLineKind.CONTENT:
                continue
            ref = available_refs.get(line.value["content_id"])
            if ref is None or ref.key_id != line.value["key_id"]:
                raise ContentBindingMismatch("content export line does not match a referenced content key")

    def to_dict(self) -> dict[str, Any]:
        return {"header": self.header.to_dict(), "lines": [line.to_dict() for line in self.lines]}

    @classmethod
    def from_dict(cls, header: Mapping[str, Any], lines: Sequence[Mapping[str, Any]]) -> ExportDocument:
        if isinstance(lines, str | bytes) or not isinstance(lines, Sequence):
            raise InvalidExport("export lines must be an array")
        return cls(ExportHeader.from_dict(header), tuple(ExportLine.from_dict(line) for line in lines))


__all__ = [
    "APPROVAL_METADATA",
    "APPROVAL_REASON",
    "APPROVAL_SCOPE",
    "CHILD_RESULT",
    "INPUT_ACTOR_ATTRIBUTES",
    "INPUT_CONTENT",
    "INPUT_SELECTION",
    "SESSION_ATTRIBUTES",
    "SESSION_PRINCIPAL_ATTRIBUTES",
    "SESSION_WORKSPACE",
    "Classification",
    "ContentBinding",
    "ContentBindingMismatch",
    "ContentErrorCode",
    "ContentIdentity",
    "ContentIntegrityError",
    "ContentLabels",
    "ContentLifecycleError",
    "ContentLinkConflict",
    "ContentRef",
    "ContentSessionWriter",
    "ContentStoreError",
    "ContentUnavailable",
    "ExportContentMode",
    "ExportDocument",
    "ExportHeader",
    "ExportLine",
    "ExportLineKind",
    "ExportModeMismatch",
    "ForgetTombstone",
    "ForgottenSessionReuse",
    "InvalidContentRef",
    "InvalidDestroyOutcome",
    "InvalidExport",
    "InvalidTombstone",
    "KeyCapabilityState",
    "KeyDestroyOutcome",
    "KeyDestroyRequest",
    "KeyDestroyStatus",
    "KeyProvider",
    "KeyUnavailable",
    "Principal",
    "SessionHead",
    "SessionKeyCapability",
    "SourceKind",
    "TombstoneHeadMismatch",
    "TombstoneState",
    "UnsupportedExportVersion",
    "bind_content_ref",
]
