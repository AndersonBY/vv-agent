"""Standard-library SQL content rules for the S1 session boundary.

This module owns content framing, owner links, lifecycle state and every
content SQL statement. A caller supplies the storage backend and the narrow
``ContentCipher`` port; this module opens no connections, creates no threads,
reads no clocks and imports no SDK or cryptography package.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._canonical_json import canonical_json_bytes
from ._content import (
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentIntegrityError,
    ContentLabels,
    ContentLinkConflict,
    ContentRef,
    ContentUnavailable,
    ExportLine,
    ExportLineKind,
    KeyDestroyStatus,
    KeyProvider,
    KeyUnavailable,
    WrappedDataKey,
)
from ._content_publication import _MISSING, _expand_nested_slots, declared_content_ref
from ._content_store import (
    ContentCipher,
    ContentLink,
    ContentState,
    SessionKeyRecord,
    SessionKeyState,
)

_DATA_KEY_BYTES = 32
_NONCE_BYTES = 12
_STAGING_PIN_MS = 5 * 60 * 1000
_CONTENT_COLUMNS = "content_id,key_id,mac,size,labels,state,nonce,ciphertext,object_locator"
_READ_FRAME_OWNER_BATCH = 256

ObjectLocatorCleaner = Callable[[str, str, bytes], None]


@dataclass(frozen=True, slots=True)
class ContentExportRef:
    """One exported reference with its persisted SQL owner and declared slot."""

    record_seq: int | None
    input_id: str | None
    slot: str
    ref: ContentRef
    nested: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if (self.record_seq is None) == (self.input_id is None):
            raise ContentBindingMismatch("content export requires exactly one owner")
        if self.record_seq is not None and (type(self.record_seq) is not int or self.record_seq <= 0):
            raise ContentBindingMismatch("record_seq must be a positive integer")
        if self.input_id is not None and (not isinstance(self.input_id, str) or not self.input_id):
            raise ContentBindingMismatch("input_id must be a non-empty string")
        _link_slot(self.slot)
        if not isinstance(self.ref, ContentRef):
            raise TypeError("content export ref must be ContentRef")
        if not isinstance(self.nested, tuple):
            raise TypeError("content export nested declarations must be a tuple")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContentBindingMismatch(f"{field_name} must be a non-empty string")
    return value


def _random_bytes(cipher: ContentCipher, count: int) -> bytes:
    value = cipher.random_bytes(count)
    if not isinstance(value, bytes) or len(value) != count:
        raise ContentIntegrityError("content randomness is malformed")
    return value


def _random_token(cipher: ContentCipher, count: int) -> str:
    return base64.urlsafe_b64encode(_random_bytes(cipher, count)).rstrip(b"=").decode("ascii")


def _content_aad(
    tenant_id: str,
    session_id: str,
    content_id: str,
    key_id: str,
    mac: str,
    size: int,
    labels: ContentLabels,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/content",
            "framing_version": 1,
            "purpose": "content-ciphertext",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "content_id": content_id,
            "key_id": key_id,
            "mac": mac,
            "size": size,
            "labels": labels.to_dict(),
        },
        "content AAD",
    )


def _content_mac(data_key: bytes, tenant_id: str, session_id: str, plaintext: bytes) -> str:
    framing = canonical_json_bytes(
        {
            "domain": "vv-agent/S1/content-mac",
            "framing_version": 1,
            "purpose": "content-ref",
            "tenant_id": tenant_id,
            "session_id": session_id,
        },
        "content MAC framing",
    )
    return hmac.digest(data_key, framing + b"\0" + plaintext, hashlib.sha256).hex()


@runtime_checkable
class _SQLContentBackend(Protocol):
    """Driver-specific execution with one shared, explicit parameter token."""

    lock_clause: str

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None: ...

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]: ...

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int: ...

    def require_transaction(self) -> None: ...

    def now_ms(self) -> int: ...


def _blob(value: object, field_name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise ContentIntegrityError(f"{field_name} is not binary data")


def _labels_from_storage(value: object) -> tuple[ContentLabels, bytes]:
    raw = _blob(value, "content labels")
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("labels are not an object")
        labels = ContentLabels.from_dict(payload)
        if canonical_json_bytes(payload) != raw:
            raise ValueError("labels are not canonical JSON")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContentIntegrityError("stored content labels are malformed") from exc
    return labels, raw


def _content_from_row(
    row: tuple[Any, ...],
) -> tuple[ContentRef, ContentState, bytes, bytes | None, bytes | None]:
    content_id, key_id, mac, size, labels_value, state_value, nonce_value, ciphertext_value, locator_value = row
    labels, _labels_bytes = _labels_from_storage(labels_value)
    try:
        state = ContentState(state_value)
        ref = ContentRef(content_id=content_id, mac=mac, size=size, key_id=key_id, labels=labels)
    except (TypeError, ValueError) as exc:
        raise ContentIntegrityError("stored content metadata is malformed") from exc
    nonce = _blob(nonce_value, "content nonce")
    ciphertext = None if ciphertext_value is None else _blob(ciphertext_value, "content ciphertext")
    locator = None if locator_value is None else _blob(locator_value, "content object locator")
    return ref, state, nonce, ciphertext, locator


def _key_from_row(tenant_id: str, session_id: str, row: tuple[Any, ...]) -> SessionKeyRecord:
    key_id, provider_id, wrapped_key, state_value = row
    try:
        state = SessionKeyState(state_value)
    except (TypeError, ValueError) as exc:
        raise KeyUnavailable("session data key state is unavailable") from exc
    return SessionKeyRecord(
        tenant_id=tenant_id,
        session_id=session_id,
        key_id=key_id,
        wrapped_key=_blob(wrapped_key, "wrapped session key"),
        provider_id=provider_id,
        state=state,
    )


def _owner_identity(
    *,
    record_seq: int | None,
    input_id: str | None,
) -> tuple[str, int | str]:
    if (record_seq is None) == (input_id is None):
        raise ContentBindingMismatch("exactly one content owner is required")
    if record_seq is not None:
        if type(record_seq) is not int or record_seq <= 0:
            raise ContentBindingMismatch("record_seq must be a positive integer")
        return "record_seq", record_seq
    return "input_id", _text(input_id, "input_id")


def _link_slot(value: object) -> str:
    slot = _text(value, "slot")
    if "\x00" in slot:
        raise ContentBindingMismatch("slot must not contain NUL")
    return slot


class _ContentReadFrame:
    """One disposable content snapshot with lazy local key and plaintext use."""

    def __init__(
        self,
        *,
        tenant_id: str,
        session_id: str,
        live_row: tuple[Any, ...] | None,
        key_row: tuple[Any, ...] | None,
        link_rows: Iterable[tuple[Any, ...]],
        content_rows: Iterable[tuple[Any, ...]],
        key_provider: KeyProvider,
        cipher: ContentCipher,
        provider_id: str,
    ) -> None:
        self.tenant_id = _text(tenant_id, "tenant_id")
        self.session_id = _text(session_id, "session_id")
        self._live_row = live_row
        self._key_row = key_row
        self._links: dict[tuple[str, int | str, str], str] = {}
        self._content: dict[str, tuple[Any, ...]] = {}
        for row in link_rows:
            record_seq, input_id, slot = row[:3]
            if (record_seq is None) == (input_id is None):
                raise ContentIntegrityError("content link has an invalid owner")
            owner_kind = "record_seq" if record_seq is not None else "input_id"
            owner_value = record_seq if record_seq is not None else input_id
            self._links[(owner_kind, owner_value, slot)] = row[3]
        for content in content_rows:
            self._content[content[0]] = content
        self._key_provider = key_provider
        self._cipher = cipher
        self._provider_id = provider_id
        self._session_key: SessionKeyRecord | None = None
        self._parsed_content: dict[str, tuple[ContentRef, ContentState, bytes, bytes | None, bytes | None]] = {}
        self._data_key: bytes | None = None
        self._plaintext: dict[ContentRef, bytes] = {}

    def _require_binding(self, tenant_id: str, session_id: str) -> None:
        if (tenant_id, session_id) != (self.tenant_id, self.session_id):
            raise ContentBindingMismatch("content frame binding does not match")

    def _require_live(self) -> None:
        if self._live_row is None:
            raise ContentUnavailable("content session is unavailable")
        if self._live_row[0] != "live":
            raise ContentUnavailable("content session is fenced for deletion")

    def _active_session_key(self) -> SessionKeyRecord:
        key = self._session_key
        if key is None:
            if self._key_row is None:
                raise ContentUnavailable("content session is unavailable")
            key = _key_from_row(self.tenant_id, self.session_id, self._key_row)
            self._session_key = key
        if key.provider_id != self._provider_id:
            raise KeyUnavailable("session data key provider does not match")
        if key.state is not SessionKeyState.ACTIVE:
            raise KeyUnavailable("session data key is not active")
        return key

    def _validated_content(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
    ) -> tuple[SessionKeyRecord, ContentState, bytes, bytes | None, bytes | None]:
        self._require_binding(tenant_id, session_id)
        if not isinstance(ref, ContentRef):
            raise TypeError("ref must be ContentRef")
        self._require_live()
        key = self._active_session_key()
        parsed = self._parsed_content.get(ref.content_id)
        if parsed is None:
            row = self._content.get(ref.content_id)
            if row is None:
                raise ContentUnavailable("content is unavailable")
            parsed = _content_from_row(row)
            self._parsed_content[ref.content_id] = parsed
            del self._content[ref.content_id]
        stored, state, nonce, ciphertext, locator = parsed
        if stored.key_id != key.key_id or ref.key_id != key.key_id:
            raise ContentBindingMismatch("content key does not belong to the session binding")
        if stored != ref:
            raise ContentIntegrityError("content metadata does not match its reference")
        if state is ContentState.GC_PENDING:
            raise ContentUnavailable("content is unavailable")
        if state is ContentState.STAGED:
            raise ContentUnavailable("content staging pin is expired")
        return key, state, nonce, ciphertext, locator

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        slot: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> ContentRef:
        self._require_binding(tenant_id, session_id)
        slot = _link_slot(slot)
        owner_column, owner = _owner_identity(record_seq=record_seq, input_id=input_id)
        self._validated_content(tenant_id, session_id, ref)
        if self._links.get((owner_column, owner, slot)) != ref.content_id:
            raise ContentUnavailable("content link is unavailable")
        return ref

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes:
        key, _state, nonce, ciphertext, locator = self._validated_content(tenant_id, session_id, ref)
        if locator is not None or ciphertext is None:
            raise ContentUnavailable("content ciphertext is unavailable")
        if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
            raise ContentIntegrityError("content metadata or ciphertext is malformed")
        cached = self._plaintext.get(ref)
        if cached is not None:
            return cached
        if self._data_key is None:
            data_key = self._key_provider.unwrap(
                tenant_id,
                session_id,
                WrappedDataKey(key.tenant_id, key.session_id, key.key_id, key.wrapped_key),
            )
            if not isinstance(data_key, bytes) or len(data_key) != _DATA_KEY_BYTES:
                raise KeyUnavailable("session data key has an invalid size")
            self._data_key = data_key
        plaintext = self._cipher.decrypt(
            self._data_key,
            nonce,
            ciphertext,
            _content_aad(
                tenant_id,
                session_id,
                ref.content_id,
                ref.key_id,
                ref.mac,
                ref.size,
                ref.labels,
            ),
        )
        if len(plaintext) != ref.size:
            raise ContentIntegrityError("content size does not match its reference")
        actual_mac = _content_mac(self._data_key, tenant_id, session_id, plaintext)
        if not hmac.compare_digest(actual_mac, ref.mac):
            raise ContentIntegrityError("content failed HMAC authentication")
        self._plaintext[ref] = plaintext
        return plaintext

    def close(self) -> None:
        self._session_key = None
        self._parsed_content.clear()
        self._data_key = None
        self._plaintext.clear()
        self._links.clear()
        self._content.clear()
        self._live_row = None
        self._key_row = None


class CoreSQLContentStore:
    """Shared durable content implementation over a caller-owned SQL backend.

    Mutating methods require an existing outer session transaction. The adapter
    deliberately has no ``atomic`` context manager: it never starts, commits, or
    rolls back a business transaction.
    """

    def __init__(
        self,
        backend: _SQLContentBackend,
        key_provider: KeyProvider,
        cipher: ContentCipher,
        *,
        provider_id: str | None = None,
        object_locator_cleaner: ObjectLocatorCleaner | None = None,
    ) -> None:
        self._backend = backend
        self._key_provider = key_provider
        self._cipher = cipher
        inferred_provider_id = getattr(key_provider, "provider_id", None)
        self._provider_id = _text(provider_id if provider_id is not None else inferred_provider_id, "provider_id")
        if object_locator_cleaner is not None and not callable(object_locator_cleaner):
            raise TypeError("object_locator_cleaner must be callable")
        self._object_locator_cleaner = object_locator_cleaner

    def set_object_locator_cleaner(self, cleaner: ObjectLocatorCleaner) -> None:
        """Attach the idempotent external object deletion capability."""
        if not callable(cleaner):
            raise TypeError("object locator cleaner must be callable")
        self._object_locator_cleaner = cleaner

    def _require_transaction(self) -> None:
        self._backend.require_transaction()

    def _require_live(self, tenant_id: str, session_id: str) -> None:
        row = self._backend.fetchone(
            "SELECT deletion_state FROM sk_session WHERE tenant_id={p} AND session_id={p}",
            (tenant_id, session_id),
        )
        if row is None:
            raise ContentUnavailable("content session is unavailable")
        if row[0] != "live":
            raise ContentUnavailable("content session is fenced for deletion")

    def _key_row(self, tenant_id: str, session_id: str) -> tuple[Any, ...] | None:
        return self._backend.fetchone(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id={p} AND session_id={p}",
            (tenant_id, session_id),
        )

    def _session_key(self, tenant_id: str, session_id: str) -> SessionKeyRecord:
        row = self._key_row(tenant_id, session_id)
        if row is None:
            raise ContentUnavailable("content session is unavailable")
        return _key_from_row(tenant_id, session_id, row)

    def _active_session_key(self, tenant_id: str, session_id: str) -> SessionKeyRecord:
        key = self._session_key(tenant_id, session_id)
        if key.provider_id != self._provider_id:
            raise KeyUnavailable("session data key provider does not match")
        if key.state is not SessionKeyState.ACTIVE:
            raise KeyUnavailable("session data key is not active")
        return key

    @staticmethod
    def _wrapped_key(key: SessionKeyRecord) -> WrappedDataKey:
        return WrappedDataKey(key.tenant_id, key.session_id, key.key_id, key.wrapped_key)

    def _restore_key(self, key: SessionKeyRecord) -> None:
        restore = getattr(self._key_provider, "restore_data_key", None)
        if callable(restore):
            restore(self._wrapped_key(key))

    def _unwrap(self, tenant_id: str, session_id: str, key: SessionKeyRecord) -> bytes:
        wrapped = self._wrapped_key(key)
        data_key = self._key_provider.unwrap(tenant_id, session_id, wrapped)
        if len(data_key) != _DATA_KEY_BYTES:
            raise KeyUnavailable("session data key has an invalid size")
        return data_key

    @contextmanager
    def _read_frame(
        self,
        tenant_id: str,
        session_id: str,
        *,
        record_seqs: Iterable[int] = (),
        input_ids: Iterable[str] = (),
    ) -> Iterator[_ContentReadFrame]:
        """Capture requested owners, then release the SQL snapshot before KMS work."""

        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        records = tuple(sorted({_owner_identity(record_seq=value, input_id=None)[1] for value in record_seqs}))
        inputs = tuple(sorted({_owner_identity(record_seq=None, input_id=value)[1] for value in input_ids}))
        read_snapshot = getattr(self._backend, "read_snapshot", None)
        if not callable(read_snapshot):
            raise RuntimeError("content backend does not provide a read snapshot")

        link_rows: list[tuple[Any, ...]] = []
        content_rows: list[tuple[Any, ...]] = []
        with read_snapshot():
            live_row = self._backend.fetchone(
                "SELECT deletion_state FROM sk_session WHERE tenant_id={p} AND session_id={p}",
                (tenant_id, session_id),
            )
            key_row = self._key_row(tenant_id, session_id)
            for start in range(0, len(records), _READ_FRAME_OWNER_BATCH):
                chunk = records[start : start + _READ_FRAME_OWNER_BATCH]
                placeholders = ",".join("{p}" for _ in chunk)
                link_rows.extend(
                    self._backend.fetchall(
                        "SELECT record_seq,input_id,slot,content_id FROM sk_content_link "
                        f"WHERE tenant_id={{p}} AND session_id={{p}} AND record_seq IN ({placeholders}) "
                        "ORDER BY record_seq,slot",
                        (tenant_id, session_id, *chunk),
                    )
                )
            for start in range(0, len(inputs), _READ_FRAME_OWNER_BATCH):
                chunk = inputs[start : start + _READ_FRAME_OWNER_BATCH]
                placeholders = ",".join("{p}" for _ in chunk)
                link_rows.extend(
                    self._backend.fetchall(
                        "SELECT record_seq,input_id,slot,content_id FROM sk_content_link "
                        f"WHERE tenant_id={{p}} AND session_id={{p}} AND input_id IN ({placeholders}) "
                        "ORDER BY input_id,slot",
                        (tenant_id, session_id, *chunk),
                    )
                )
            content_ids = sorted({row[3] for row in link_rows})
            for start in range(0, len(content_ids), _READ_FRAME_OWNER_BATCH):
                chunk = content_ids[start : start + _READ_FRAME_OWNER_BATCH]
                placeholders = ",".join("{p}" for _ in chunk)
                content_rows.extend(
                    self._backend.fetchall(
                        f"SELECT {_CONTENT_COLUMNS} FROM sk_content "
                        f"WHERE tenant_id={{p}} AND session_id={{p}} AND content_id IN ({placeholders})",
                        (tenant_id, session_id, *chunk),
                    )
                )

        frame = _ContentReadFrame(
            tenant_id=tenant_id,
            session_id=session_id,
            live_row=live_row,
            key_row=key_row,
            link_rows=link_rows,
            content_rows=content_rows,
            key_provider=self._key_provider,
            cipher=self._cipher,
            provider_id=self._provider_id,
        )
        try:
            yield frame
        finally:
            frame.close()

    def create_session_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        self._require_transaction()
        self._require_live(tenant_id, session_id)
        existing = self._session_key(tenant_id, session_id) if self._key_row(tenant_id, session_id) is not None else None
        if existing is not None:
            if existing.provider_id != self._provider_id or existing.state is not SessionKeyState.ACTIVE:
                raise KeyUnavailable("session data key is unavailable")
            self._restore_key(existing)
            return self._wrapped_key(existing)

        wrapped = self._key_provider.generate_data_key(tenant_id, session_id)
        if (wrapped.tenant_id, wrapped.session_id) != (tenant_id, session_id):
            raise ContentBindingMismatch("generated data key does not match the session binding")
        key = SessionKeyRecord(
            tenant_id=tenant_id,
            session_id=session_id,
            key_id=wrapped.key_id,
            wrapped_key=wrapped.wrapped_key,
            provider_id=self._provider_id,
            state=SessionKeyState.ACTIVE,
        )
        self.put_session_key(key)
        stored = self._active_session_key(tenant_id, session_id)
        if stored != key:
            raise ContentBindingMismatch("session key identity has different bytes")
        return wrapped

    def put_session_key(self, key: SessionKeyRecord) -> None:
        self._require_transaction()
        self._require_live(key.tenant_id, key.session_id)
        if key.provider_id != self._provider_id:
            raise KeyUnavailable("session data key provider does not match")
        inserted = self._backend.run(
            "INSERT INTO sk_session_key(tenant_id,session_id,key_id,provider_id,wrapped_key,state,created_ms) "
            "VALUES ({p},{p},{p},{p},{p},{p},{p}) ON CONFLICT(tenant_id,session_id) DO NOTHING",
            (
                key.tenant_id,
                key.session_id,
                key.key_id,
                key.provider_id,
                key.wrapped_key,
                key.state.value,
                self._backend.now_ms(),
            ),
        )
        if inserted:
            return
        existing = self._session_key(key.tenant_id, key.session_id)
        if (existing.key_id, existing.wrapped_key, existing.provider_id) != (
            key.key_id,
            key.wrapped_key,
            key.provider_id,
        ):
            raise ContentBindingMismatch("session key identity has different bytes")
        if existing.state is key.state:
            return
        if existing.state is SessionKeyState.ACTIVE and key.state is SessionKeyState.DESTROYING:
            self._backend.run(
                "UPDATE sk_session_key SET state={p} WHERE tenant_id={p} AND session_id={p} AND key_id={p} AND state='active'",
                (key.state.value, key.tenant_id, key.session_id, key.key_id),
            )
            return
        raise KeyUnavailable("session data key state transition is not allowed")

    def put(
        self,
        tenant_id: str,
        session_id: str,
        plaintext: bytes,
        labels: ContentLabels,
        *,
        state: ContentState = ContentState.STAGED,
    ) -> ContentRef:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        if not isinstance(plaintext, bytes):
            raise TypeError("plaintext must be bytes")
        if not isinstance(labels, ContentLabels):
            raise TypeError("labels must be ContentLabels")
        try:
            content_state = ContentState(state)
        except ValueError as exc:
            raise ValueError("put state must be staged or live") from exc
        if content_state not in {ContentState.STAGED, ContentState.LIVE}:
            raise ValueError("put state must be staged or live")
        self._require_transaction()
        self._require_live(tenant_id, session_id)

        wrapped = self.create_session_key(tenant_id, session_id)
        key = self._active_session_key(tenant_id, session_id)
        data_key = self._unwrap(tenant_id, session_id, key)
        mac = _content_mac(data_key, tenant_id, session_id, plaintext)
        _labels, labels_bytes = _labels_from_storage(canonical_json_bytes(labels.to_dict()))
        existing_row = self._backend.fetchone(
            f"SELECT {_CONTENT_COLUMNS} FROM sk_content "
            "WHERE tenant_id={p} AND session_id={p} AND key_id={p} AND mac={p} AND labels={p}"
            f"{self._backend.lock_clause}",
            (tenant_id, session_id, wrapped.key_id, mac, labels_bytes),
        )
        if existing_row is not None:
            existing, existing_state, _nonce, _ciphertext, _locator = _content_from_row(existing_row)
            if existing.mac != mac or existing.labels != labels or existing.size != len(plaintext):
                raise ContentIntegrityError("deduplicated content metadata is inconsistent")
            return self._reuse_content(tenant_id, session_id, existing, existing_state, content_state)

        content_id = f"content_{_random_token(self._cipher, 32)}"
        size = len(plaintext)
        nonce = _random_bytes(self._cipher, _NONCE_BYTES)
        ciphertext = self._cipher.encrypt(
            data_key,
            nonce,
            plaintext,
            _content_aad(tenant_id, session_id, content_id, wrapped.key_id, mac, size, labels),
        )
        ref = ContentRef(content_id=content_id, mac=mac, size=size, key_id=wrapped.key_id, labels=labels)
        now_ms = self._backend.now_ms()
        inserted = self._backend.run(
            "INSERT INTO sk_content(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,created_ms,state,"
            "pin_until_ms,ciphertext) VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p}) "
            "ON CONFLICT(tenant_id,session_id,key_id,mac,labels) DO NOTHING",
            (
                tenant_id,
                session_id,
                content_id,
                wrapped.key_id,
                mac,
                size,
                labels_bytes,
                nonce,
                now_ms,
                content_state.value,
                now_ms + _STAGING_PIN_MS if content_state is ContentState.STAGED else None,
                ciphertext,
            ),
        )
        if inserted:
            return ref

        # A concurrent caller may have won the unique identity between selection and insertion.
        row = self._backend.fetchone(
            f"SELECT {_CONTENT_COLUMNS} FROM sk_content "
            "WHERE tenant_id={p} AND session_id={p} AND key_id={p} AND mac={p} AND labels={p}"
            f"{self._backend.lock_clause}",
            (tenant_id, session_id, wrapped.key_id, mac, labels_bytes),
        )
        if row is None:
            raise ContentUnavailable("content insertion did not produce a row")
        existing, existing_state, _nonce, _ciphertext, _locator = _content_from_row(row)
        return self._reuse_content(tenant_id, session_id, existing, existing_state, content_state)

    def _reuse_content(
        self,
        tenant_id: str,
        session_id: str,
        existing: ContentRef,
        existing_state: ContentState,
        requested_state: ContentState,
    ) -> ContentRef:
        if existing_state is ContentState.GC_PENDING:
            raise ContentUnavailable("deduplicated content is unavailable")
        if existing_state is ContentState.STAGED:
            if requested_state is ContentState.LIVE:
                self._backend.run(
                    "UPDATE sk_content SET state='live',pin_until_ms=NULL "
                    "WHERE tenant_id={p} AND session_id={p} AND content_id={p} AND state='staged'",
                    (tenant_id, session_id, existing.content_id),
                )
            else:
                self._backend.run(
                    "UPDATE sk_content SET pin_until_ms={p} "
                    "WHERE tenant_id={p} AND session_id={p} AND content_id={p} AND state='staged'",
                    (self._backend.now_ms() + _STAGING_PIN_MS, tenant_id, session_id, existing.content_id),
                )
        return existing

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        if not isinstance(ref, ContentRef):
            raise TypeError("ref must be ContentRef")
        self._require_live(tenant_id, session_id)
        key = self._active_session_key(tenant_id, session_id)
        row = self._backend.fetchone(
            f"SELECT {_CONTENT_COLUMNS} FROM sk_content WHERE tenant_id={{p}} AND session_id={{p}} AND content_id={{p}}",
            (tenant_id, session_id, ref.content_id),
        )
        if row is None:
            raise ContentUnavailable("content is unavailable")
        stored, state, nonce, ciphertext, locator = _content_from_row(row)
        if stored.key_id != key.key_id or ref.key_id != key.key_id:
            raise ContentBindingMismatch("content key does not belong to the session binding")
        if stored != ref:
            raise ContentIntegrityError("content metadata does not match its reference")
        if state is ContentState.GC_PENDING or locator is not None or ciphertext is None:
            raise ContentUnavailable("content ciphertext is unavailable")
        if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
            raise ContentIntegrityError("content metadata or ciphertext is malformed")

        data_key = self._unwrap(tenant_id, session_id, key)
        plaintext = self._cipher.decrypt(
            data_key,
            nonce,
            ciphertext,
            _content_aad(
                tenant_id,
                session_id,
                stored.content_id,
                stored.key_id,
                stored.mac,
                stored.size,
                stored.labels,
            ),
        )
        if len(plaintext) != ref.size:
            raise ContentIntegrityError("content size does not match its reference")
        actual_mac = _content_mac(data_key, tenant_id, session_id, plaintext)
        if not hmac.compare_digest(actual_mac, ref.mac):
            raise ContentIntegrityError("content failed HMAC authentication")
        return plaintext

    def export_content_lines(
        self,
        tenant_id: str,
        session_id: str,
        refs: Iterable[ContentExportRef],
    ) -> tuple[ExportLine, ...]:
        """Decrypt referenced canonical JSON values for an explicit content export.

        The caller must first verify the retained ledger snapshot. This method
        requires the caller's transaction and validates every read against the
        persisted tenant/session/owner/slot link. Declared nested references are
        read from the authenticated outer value and follow the same link checks.
        """
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        self._require_transaction()
        requested: list[ContentExportRef] = []
        seen: dict[tuple[int | None, str | None, str, ContentRef], tuple[Any, ...]] = {}
        by_content_id: dict[str, ContentRef] = {}
        for request in refs:
            if not isinstance(request, ContentExportRef):
                raise TypeError("content export references must be ContentExportRef values")
            previous = by_content_id.get(request.ref.content_id)
            if previous is not None and previous != request.ref:
                raise ContentBindingMismatch("one content id has conflicting export references")
            identity = (request.record_seq, request.input_id, request.slot, request.ref)
            previous_nested = seen.get(identity)
            if previous_nested is not None and previous_nested != request.nested:
                raise ContentBindingMismatch("one content export slot has conflicting nested declarations")
            if previous_nested is not None:
                continue
            seen[identity] = request.nested
            by_content_id[request.ref.content_id] = request.ref
            requested.append(request)

        lines: list[ExportLine] = []
        for request in requested:
            self._export_content_ref(tenant_id, session_id, request, seen=set(), lines=lines)
        return tuple(lines)

    def _export_content_ref(
        self,
        tenant_id: str,
        session_id: str,
        request: ContentExportRef,
        *,
        seen: set[tuple[int | None, str | None, str, ContentRef]],
        lines: list[ExportLine],
    ) -> None:
        identity = (request.record_seq, request.input_id, request.slot, request.ref)
        if identity in seen:
            return
        seen.add(identity)
        self.assert_link(
            tenant_id,
            session_id,
            request.ref,
            request.slot,
            record_seq=request.record_seq,
            input_id=request.input_id,
        )
        plaintext = self.get(tenant_id, session_id, request.ref)
        try:
            decoded_value = json.loads(plaintext)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContentIntegrityError("content export value is not valid JSON") from exc
        if canonical_json_bytes(decoded_value, "content export value") != plaintext:
            raise ContentIntegrityError("content export value is not canonical JSON")
        lines.append(
            ExportLine(
                ExportLineKind.CONTENT,
                {
                    "line": "content",
                    "content_id": request.ref.content_id,
                    "key_id": request.ref.key_id,
                    "slot": request.slot,
                    "decoded_value": decoded_value,
                },
            )
        )

        try:
            nested_values = tuple(_expand_nested_slots(decoded_value, request.nested, request.slot))
        except ValueError as exc:
            raise ContentIntegrityError("content export container does not match its declared nested slots") from exc
        for _path, slot, value in nested_values:
            if value is _MISSING or value is None:
                continue
            try:
                nested_ref = declared_content_ref(value)
            except ValueError as exc:
                raise ContentIntegrityError(f"{slot} nested content slot contains a malformed ContentRef") from exc
            if nested_ref is None:
                raise ContentIntegrityError(f"{slot} nested content slot does not contain a ContentRef")
            self._export_content_ref(
                tenant_id,
                session_id,
                ContentExportRef(
                    record_seq=request.record_seq,
                    input_id=request.input_id,
                    slot=slot,
                    ref=nested_ref,
                ),
                seen=seen,
                lines=lines,
            )

    def _require_owner(self, tenant_id: str, session_id: str, owner_column: str, owner: int | str) -> None:
        table = "sk_record" if owner_column == "record_seq" else "sk_inbox"
        column = "seq" if owner_column == "record_seq" else "input_id"
        if (
            self._backend.fetchone(
                f"SELECT 1 FROM {table} WHERE tenant_id={{p}} AND session_id={{p}} AND {column}={{p}}",
                (tenant_id, session_id, owner),
            )
            is None
        ):
            raise ContentUnavailable("content owner is unavailable")

    def _validated_content(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        *,
        for_update: bool = False,
    ) -> ContentState:
        if not isinstance(ref, ContentRef):
            raise TypeError("ref must be ContentRef")
        self._require_live(tenant_id, session_id)
        key = self._active_session_key(tenant_id, session_id)
        row = self._backend.fetchone(
            f"SELECT {_CONTENT_COLUMNS},pin_until_ms FROM sk_content "
            "WHERE tenant_id={p} AND session_id={p} AND content_id={p}"
            f"{self._backend.lock_clause if for_update else ''}",
            (tenant_id, session_id, ref.content_id),
        )
        if row is None:
            raise ContentUnavailable("content is unavailable")
        stored, state, _nonce, _ciphertext, _locator = _content_from_row(row[:-1])
        if stored.key_id != key.key_id or ref.key_id != key.key_id:
            raise ContentBindingMismatch("content key does not belong to the session binding")
        if stored != ref:
            raise ContentIntegrityError("content metadata does not match its reference")
        if state is ContentState.GC_PENDING:
            raise ContentUnavailable("content is unavailable")
        if state is ContentState.STAGED and (not for_update or row[-1] is None or row[-1] <= self._backend.now_ms()):
            raise ContentUnavailable("content staging pin is expired")
        return state

    def _link(
        self,
        tenant_id: str,
        session_id: str,
        owner_column: str,
        owner: int | str,
        ref: ContentRef,
        slot: str,
    ) -> None:
        self._require_transaction()
        slot = _link_slot(slot)
        self._validated_content(tenant_id, session_id, ref, for_update=True)
        self._require_owner(tenant_id, session_id, owner_column, owner)
        record_seq = owner if owner_column == "record_seq" else None
        input_id = owner if owner_column == "input_id" else None
        self._backend.run(
            "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,input_id,slot) "
            "VALUES ({p},{p},{p},{p},{p},{p}) ON CONFLICT DO NOTHING",
            (tenant_id, session_id, ref.content_id, record_seq, input_id, slot),
        )
        existing = self._backend.fetchone(
            f"SELECT content_id FROM sk_content_link WHERE tenant_id={{p}} AND session_id={{p}} "
            f"AND {owner_column}={{p}} AND slot={{p}}",
            (tenant_id, session_id, owner, slot),
        )
        if existing is None or existing[0] != ref.content_id:
            raise ContentLinkConflict("content owner slot is already linked")
        self._backend.run(
            "UPDATE sk_content SET state='live',pin_until_ms=NULL WHERE tenant_id={p} AND session_id={p} "
            "AND content_id={p} AND state IN ('staged','live')",
            (tenant_id, session_id, ref.content_id),
        )

    def link_record(self, tenant_id: str, session_id: str, record_seq: int, ref: ContentRef, slot: str) -> None:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        owner_column, owner = _owner_identity(record_seq=record_seq, input_id=None)
        assert isinstance(owner, int)
        self._link(tenant_id, session_id, owner_column, owner, ref, slot)

    def link_input(self, tenant_id: str, session_id: str, input_id: str, ref: ContentRef, slot: str) -> None:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        owner_column, owner = _owner_identity(record_seq=None, input_id=input_id)
        assert isinstance(owner, str)
        self._link(tenant_id, session_id, owner_column, owner, ref, slot)

    def read_links(
        self,
        tenant_id: str,
        session_id: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> tuple[ContentLink, ...]:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        owner_column, owner = _owner_identity(record_seq=record_seq, input_id=input_id)
        self._require_live(tenant_id, session_id)
        key = self._active_session_key(tenant_id, session_id)
        rows = self._backend.fetchall(
            f"SELECT l.record_seq,l.input_id,l.slot,{_content_columns_prefixed('c')} "
            "FROM sk_content_link l "
            "JOIN sk_content c ON c.tenant_id=l.tenant_id AND c.session_id=l.session_id "
            "AND c.content_id=l.content_id "
            f"AND c.tenant_id={{p}} AND c.session_id={{p}} AND l.{owner_column}={{p}} "
            "ORDER BY l.slot",
            (tenant_id, session_id, owner),
        )
        links: list[ContentLink] = []
        for row in rows:
            stored, state, _nonce, _ciphertext, _locator = _content_from_row(tuple(row[3:]))
            if stored.key_id != key.key_id:
                raise ContentBindingMismatch("content link key does not belong to the session binding")
            if state is ContentState.GC_PENDING:
                raise ContentUnavailable("content link is unavailable")
            links.append(
                ContentLink(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    record_seq=row[0],
                    input_id=row[1],
                    slot=row[2],
                    ref=stored,
                )
            )
        return tuple(links)

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        slot: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> ContentRef:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        slot = _link_slot(slot)
        owner_column, owner = _owner_identity(record_seq=record_seq, input_id=input_id)
        self._require_live(tenant_id, session_id)
        self._validated_content(tenant_id, session_id, ref)
        row = self._backend.fetchone(
            f"SELECT 1 FROM sk_content_link WHERE tenant_id={{p}} AND session_id={{p}} "
            f"AND {owner_column}={{p}} AND slot={{p}} AND content_id={{p}}",
            (tenant_id, session_id, owner, slot, ref.content_id),
        )
        if row is None:
            raise ContentUnavailable("content link is unavailable")
        return ref

    def collect_garbage(self, tenant_id: str, session_id: str) -> int:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        self._require_transaction()
        self._require_live(tenant_id, session_id)
        now_ms = self._backend.now_ms()
        candidates = self._backend.fetchall(
            "SELECT c.content_id,c.object_locator FROM sk_content c "
            "WHERE c.tenant_id={p} AND c.session_id={p} "
            "AND c.state IN ('staged','gc_pending') "
            "AND (c.pin_until_ms IS NULL OR c.pin_until_ms<={p}) "
            "AND NOT EXISTS (SELECT 1 FROM sk_content_link l WHERE l.tenant_id=c.tenant_id "
            "AND l.session_id=c.session_id AND l.content_id=c.content_id) "
            f"ORDER BY c.content_id{self._backend.lock_clause}",
            (tenant_id, session_id, now_ms),
        )
        collected = 0
        cleaner = self._object_locator_cleaner
        for content_id, object_locator in candidates:
            marked = self._backend.run(
                "UPDATE sk_content SET state='gc_pending',pin_until_ms=NULL "
                "WHERE tenant_id={p} AND session_id={p} AND content_id={p} "
                "AND state IN ('staged','gc_pending') "
                "AND (pin_until_ms IS NULL OR pin_until_ms<={p}) "
                "AND NOT EXISTS (SELECT 1 FROM sk_content_link l WHERE l.tenant_id=sk_content.tenant_id "
                "AND l.session_id=sk_content.session_id AND l.content_id=sk_content.content_id)",
                (tenant_id, session_id, content_id, now_ms),
            )
            if not marked:
                continue
            if object_locator is not None:
                if cleaner is None:
                    continue
                try:
                    locator = _blob(object_locator, "content object locator")
                    cleaned = cleaner(tenant_id, session_id, locator) is None
                except Exception:
                    cleaned = False
                if not cleaned:
                    # The object may have been deleted even when the remote result was lost.
                    # Keep the row as a durable retry marker; cleaners must be idempotent.
                    continue
            collected += self._backend.run(
                "DELETE FROM sk_content WHERE tenant_id={p} AND session_id={p} AND content_id={p} "
                "AND state='gc_pending' "
                "AND NOT EXISTS (SELECT 1 FROM sk_content_link l WHERE l.tenant_id=sk_content.tenant_id "
                "AND l.session_id=sk_content.session_id AND l.content_id=sk_content.content_id)",
                (tenant_id, session_id, content_id),
            )
        return collected

    def forget_session(self, tenant_id: str, session_id: str) -> None:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        self._require_transaction()
        row = self._key_row(tenant_id, session_id)
        if row is None:
            content_count = self._backend.fetchone(
                "SELECT COUNT(*) FROM sk_content WHERE tenant_id={p} AND session_id={p}",
                (tenant_id, session_id),
            )
            link_count = self._backend.fetchone(
                "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id={p} AND session_id={p}",
                (tenant_id, session_id),
            )
            if (content_count is not None and content_count[0]) or (link_count is not None and link_count[0]):
                raise KeyUnavailable("session data key is unavailable; content was not deleted")
            return

        key = self._session_key(tenant_id, session_id)
        if key.provider_id != self._provider_id:
            raise KeyUnavailable("session data key provider does not match")
        if key.state is SessionKeyState.ACTIVE:
            # A provider that can still confirm ALREADY_DESTROYED may purge rows.
            with suppress(KeyUnavailable):
                self._restore_key(key)
        locator_rows = self._backend.fetchall(
            "SELECT object_locator FROM sk_content WHERE tenant_id={p} AND session_id={p} "
            "AND object_locator IS NOT NULL ORDER BY content_id",
            (tenant_id, session_id),
        )
        cleaner = self._object_locator_cleaner
        if locator_rows and cleaner is None:
            raise ContentCleanupUnavailable("external content object cleanup capability is unavailable; content was not deleted")

        outcome = self._key_provider.destroy(tenant_id, key.key_id)
        if outcome.status is KeyDestroyStatus.BINDING_MISMATCH:
            raise ContentBindingMismatch(outcome.reason or "session key binding mismatch")
        if outcome.status not in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED}:
            raise KeyUnavailable(outcome.reason or "session data key is not destroyed")

        # Key revocation is confirmed before any content object or row is removed.
        for (locator_value,) in locator_rows:
            locator = _blob(locator_value, "content object locator")
            assert cleaner is not None
            if cleaner(tenant_id, session_id, locator) is not None:
                raise ContentCleanupUnavailable("external content object cleanup did not confirm deletion")
        self._backend.run(
            "DELETE FROM sk_content_link WHERE tenant_id={p} AND session_id={p}",
            (tenant_id, session_id),
        )
        self._backend.run(
            "DELETE FROM sk_content WHERE tenant_id={p} AND session_id={p}",
            (tenant_id, session_id),
        )
        self._backend.run(
            "DELETE FROM sk_session_key WHERE tenant_id={p} AND session_id={p} AND key_id={p}",
            (tenant_id, session_id, key.key_id),
        )


def _content_columns_prefixed(alias: str) -> str:
    return ",".join(f"{alias}.{name}" for name in _CONTENT_COLUMNS.split(","))


__all__ = [
    "ContentExportRef",
    "CoreSQLContentStore",
    "ObjectLocatorCleaner",
    "_SQLContentBackend",
    "_content_aad",
    "_content_from_row",
    "_content_mac",
    "_key_from_row",
    "_random_bytes",
    "_random_token",
]
