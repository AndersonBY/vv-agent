"""Encrypted content backends for the S1 session boundary.

The in-memory backend remains development-only.  Durable adapters accept a
caller-owned database connection and never open or commit a transaction of
their own.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentIntegrityError,
    ContentLabels,
    ContentLinkConflict,
    ContentRef,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyProvider,
    KeyUnavailable,
    WrappedDataKey,
)
from vv_agent.kernel._content_store import ContentLink, ContentState, SessionKeyRecord, SessionKeyState

_DATA_KEY_BYTES = 32
_NONCE_BYTES = 12
_KEY_WRAP_PREFIX = b"vv-agent/S1/key-wrap/v1\0"
_STAGING_PIN_MS = 5 * 60 * 1000


@dataclass(slots=True)
class _LocalKeyState:
    session_id: str
    destroyed: bool = False


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContentBindingMismatch(f"{field_name} must be a non-empty string")
    return value


def _key_wrap_aad(tenant_id: str, session_id: str, key_id: str) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/key-wrap",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "key_id": key_id,
        },
        "data-key AAD",
    )


class LocalKeyProvider:
    """Development/test-only key provider with a caller-injected AES-256 master key."""

    provider_id = "local-development-v1"

    def __init__(self, master_key: bytes) -> None:
        if not isinstance(master_key, bytes) or len(master_key) != _DATA_KEY_BYTES:
            raise ValueError("master_key must be exactly 32 bytes")
        self._master_key = master_key
        self._states: dict[tuple[str, str], _LocalKeyState] = {}
        self._lock = RLock()

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        key_id = f"key_{secrets.token_urlsafe(32)}"
        nonce = secrets.token_bytes(_NONCE_BYTES)
        data_key = secrets.token_bytes(_DATA_KEY_BYTES)
        aad = _key_wrap_aad(tenant_id, session_id, key_id)
        wrapped = _KEY_WRAP_PREFIX + nonce + AESGCM(self._master_key).encrypt(nonce, data_key, aad)
        with self._lock:
            self._states[(tenant_id, key_id)] = _LocalKeyState(session_id)
        return WrappedDataKey(tenant_id, session_id, key_id, wrapped)

    def _decrypt_wrapped_key(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        wrapped = wrapped_key.wrapped_key
        prefix_length = len(_KEY_WRAP_PREFIX)
        if not wrapped.startswith(_KEY_WRAP_PREFIX) or len(wrapped) < prefix_length + _NONCE_BYTES + 16:
            raise KeyUnavailable("wrapped data key is malformed")
        nonce = wrapped[prefix_length : prefix_length + _NONCE_BYTES]
        ciphertext = wrapped[prefix_length + _NONCE_BYTES :]
        try:
            data_key = AESGCM(self._master_key).decrypt(
                nonce,
                ciphertext,
                _key_wrap_aad(tenant_id, session_id, wrapped_key.key_id),
            )
        except InvalidTag as exc:
            raise KeyUnavailable("wrapped data key failed authentication") from exc
        if len(data_key) != _DATA_KEY_BYTES:
            raise KeyUnavailable("session data key has an invalid size")
        return data_key

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        if not isinstance(wrapped_key, WrappedDataKey):
            raise KeyUnavailable("wrapped data key is unavailable")
        with self._lock:
            state = self._states.get((tenant_id, wrapped_key.key_id))
            if (
                state is None
                or state.destroyed
                or state.session_id != session_id
                or wrapped_key.tenant_id != tenant_id
                or wrapped_key.session_id != session_id
            ):
                raise KeyUnavailable("session data key is unavailable")

        return self._decrypt_wrapped_key(tenant_id, session_id, wrapped_key)

    def restore_data_key(self, wrapped_key: WrappedDataKey) -> None:
        """Restore an active process-local state for a durable wrapped envelope."""
        if not isinstance(wrapped_key, WrappedDataKey):
            raise KeyUnavailable("wrapped data key is unavailable")
        tenant_id = _text(wrapped_key.tenant_id, "tenant_id")
        session_id = _text(wrapped_key.session_id, "session_id")
        self._decrypt_wrapped_key(tenant_id, session_id, wrapped_key)
        with self._lock:
            state = self._states.get((tenant_id, wrapped_key.key_id))
            if state is not None and (state.destroyed or state.session_id != session_id):
                raise KeyUnavailable("session data key is unavailable")
            if state is None:
                self._states[(tenant_id, wrapped_key.key_id)] = _LocalKeyState(session_id)

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
        tenant_id = _text(tenant_id, "tenant_id")
        key_id = _text(key_id, "key_id")
        with self._lock:
            state = self._states.get((tenant_id, key_id))
            if state is None:
                owner = next((item_tenant for item_tenant, item_key in self._states if item_key == key_id), None)
                if owner is not None:
                    return KeyDestroyOutcome(
                        False,
                        KeyDestroyStatus.BINDING_MISMATCH,
                        None,
                        "Session key belongs to another tenant.",
                    )
                return KeyDestroyOutcome(
                    False,
                    KeyDestroyStatus.KEY_UNAVAILABLE,
                    None,
                    "Session key is unavailable.",
                )
            status = KeyDestroyStatus.ALREADY_DESTROYED if state.destroyed else KeyDestroyStatus.DESTROYED
            state.destroyed = True
        return KeyDestroyOutcome(True, status, f"local-key-destroy-{secrets.token_urlsafe(18)}")


@dataclass(frozen=True, slots=True)
class _StoredContent:
    ref: ContentRef
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)


def _content_aad(
    tenant_id: str,
    session_id: str,
    content_id: str,
    key_id: str,
    size: int,
    labels: ContentLabels,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/content",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "content_id": content_id,
            "key_id": key_id,
            "size": size,
            "labels": labels.to_dict(),
        },
        "content AAD",
    )


class InMemoryEncryptedContentStore:
    """Process-local encrypted content backend; this store is not durable."""

    def __init__(self, key_provider: KeyProvider) -> None:
        self._key_provider = key_provider
        self._session_keys: dict[tuple[str, str], WrappedDataKey] = {}
        self._contents: dict[tuple[str, str, str], _StoredContent] = {}
        self._dedup: dict[tuple[str, str, str, str, ContentLabels], str] = {}
        self._lock = RLock()

    def put(
        self,
        tenant_id: str,
        session_id: str,
        plaintext: bytes,
        labels: ContentLabels,
    ) -> ContentRef:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        if not isinstance(plaintext, bytes):
            raise TypeError("plaintext must be bytes")
        if not isinstance(labels, ContentLabels):
            raise TypeError("labels must be ContentLabels")

        with self._lock:
            wrapped_key = self._session_keys.get((tenant_id, session_id))
            if wrapped_key is None:
                wrapped_key = self._key_provider.generate_data_key(tenant_id, session_id)
                if wrapped_key.tenant_id != tenant_id or wrapped_key.session_id != session_id or not wrapped_key.key_id:
                    raise ContentBindingMismatch("generated data key does not match the session binding")
                self._session_keys[(tenant_id, session_id)] = wrapped_key

            data_key = self._key_provider.unwrap(tenant_id, session_id, wrapped_key)
            if len(data_key) != _DATA_KEY_BYTES:
                raise KeyUnavailable("session data key has an invalid size")
            mac = hmac.digest(data_key, plaintext, hashlib.sha256).hex()
            identity = (tenant_id, session_id, wrapped_key.key_id, mac, labels)
            existing_id = self._dedup.get(identity)
            if existing_id is not None:
                return self._contents[(tenant_id, session_id, existing_id)].ref

            content_id = f"content_{secrets.token_urlsafe(32)}"
            size = len(plaintext)
            nonce = secrets.token_bytes(_NONCE_BYTES)
            aad = _content_aad(tenant_id, session_id, content_id, wrapped_key.key_id, size, labels)
            ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, aad)
            ref = ContentRef(content_id=content_id, mac=mac, size=size, key_id=wrapped_key.key_id, labels=labels)
            self._contents[(tenant_id, session_id, content_id)] = _StoredContent(ref, nonce, ciphertext)
            self._dedup[identity] = content_id
            return ref

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        if not isinstance(ref, ContentRef):
            raise TypeError("ref must be ContentRef")

        with self._lock:
            wrapped_key = self._session_keys.get((tenant_id, session_id))
            if wrapped_key is None:
                raise ContentUnavailable("content session is unavailable")
            stored = self._contents.get((tenant_id, session_id, ref.content_id))
            if stored is None:
                raise ContentUnavailable("content is unavailable")
            if (
                wrapped_key.tenant_id != tenant_id
                or wrapped_key.session_id != session_id
                or stored.ref.key_id != wrapped_key.key_id
                or ref.key_id != wrapped_key.key_id
            ):
                raise ContentBindingMismatch("content key does not belong to the session binding")
            if (
                stored.ref.content_id != ref.content_id
                or ref.mac != stored.ref.mac
                or ref.size != stored.ref.size
                or ref.labels != stored.ref.labels
                or not isinstance(stored.nonce, bytes)
                or len(stored.nonce) != _NONCE_BYTES
                or not isinstance(stored.ciphertext, bytes)
                or len(stored.ciphertext) < 16
            ):
                raise ContentIntegrityError("content metadata or ciphertext is malformed")

            data_key = self._key_provider.unwrap(tenant_id, session_id, wrapped_key)
            if len(data_key) != _DATA_KEY_BYTES:
                raise KeyUnavailable("session data key has an invalid size")
            try:
                plaintext = AESGCM(data_key).decrypt(
                    stored.nonce,
                    stored.ciphertext,
                    _content_aad(
                        tenant_id,
                        session_id,
                        stored.ref.content_id,
                        stored.ref.key_id,
                        stored.ref.size,
                        stored.ref.labels,
                    ),
                )
            except (InvalidTag, ValueError) as exc:
                raise ContentIntegrityError("content failed AEAD authentication") from exc
            if len(plaintext) != ref.size:
                raise ContentIntegrityError("content size does not match its reference")
            actual_mac = hmac.digest(data_key, plaintext, hashlib.sha256).hex()
            if not hmac.compare_digest(actual_mac, ref.mac):
                raise ContentIntegrityError("content failed HMAC authentication")
            return plaintext

    def forget_session(self, tenant_id: str, session_id: str) -> None:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        session = (tenant_id, session_id)
        with self._lock:
            wrapped_key = self._session_keys.get(session)
            if wrapped_key is None:
                return
            if wrapped_key.tenant_id != tenant_id or wrapped_key.session_id != session_id:
                raise ContentBindingMismatch("session key binding is inconsistent")
            outcome = self._key_provider.destroy(tenant_id, wrapped_key.key_id)
            if outcome.status is KeyDestroyStatus.BINDING_MISMATCH:
                raise ContentBindingMismatch(outcome.reason or "session key binding mismatch")
            if outcome.status not in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED}:
                raise KeyUnavailable(outcome.reason or "session data key is not destroyed")

            del self._session_keys[session]
            content_ids = [item[2] for item in self._contents if item[:2] == session]
            for content_id in content_ids:
                del self._contents[(tenant_id, session_id, content_id)]
            for identity in [item for item in self._dedup if item[:2] == session]:
                del self._dedup[identity]


@runtime_checkable
class _SQLContentBackend(Protocol):
    """Driver-specific execution with one shared, explicit parameter token."""

    lock_clause: str

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None: ...

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]: ...

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int: ...

    def require_transaction(self) -> None: ...

    def now_ms(self) -> int: ...


_CONTENT_COLUMNS = "content_id,key_id,mac,size,labels,state,nonce,ciphertext,object_locator"


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


class SQLContentStore:
    """Shared durable content implementation over a caller-owned SQL backend.

    Mutating methods require an existing outer session transaction. The adapter
    deliberately has no ``atomic`` context manager: it never starts, commits, or
    rolls back a business transaction.
    """

    def __init__(
        self,
        backend: _SQLContentBackend,
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._key_provider = key_provider
        inferred_provider_id = getattr(key_provider, "provider_id", None)
        self._provider_id = _text(provider_id if provider_id is not None else inferred_provider_id, "provider_id")

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
        self._restore_key(key)
        data_key = self._key_provider.unwrap(tenant_id, session_id, wrapped)
        if len(data_key) != _DATA_KEY_BYTES:
            raise KeyUnavailable("session data key has an invalid size")
        return data_key

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
        mac = hmac.digest(data_key, plaintext, hashlib.sha256).hex()
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

        content_id = f"content_{secrets.token_urlsafe(32)}"
        size = len(plaintext)
        nonce = secrets.token_bytes(_NONCE_BYTES)
        ciphertext = AESGCM(data_key).encrypt(
            nonce,
            plaintext,
            _content_aad(tenant_id, session_id, content_id, wrapped.key_id, size, labels),
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
        try:
            plaintext = AESGCM(data_key).decrypt(
                nonce,
                ciphertext,
                _content_aad(tenant_id, session_id, stored.content_id, stored.key_id, stored.size, stored.labels),
            )
        except (InvalidTag, ValueError) as exc:
            raise ContentIntegrityError("content failed AEAD authentication") from exc
        if len(plaintext) != ref.size:
            raise ContentIntegrityError("content size does not match its reference")
        actual_mac = hmac.digest(data_key, plaintext, hashlib.sha256).hex()
        if not hmac.compare_digest(actual_mac, ref.mac):
            raise ContentIntegrityError("content failed HMAC authentication")
        return plaintext

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
            "SELECT c.content_id FROM sk_content c WHERE c.tenant_id={p} AND c.session_id={p} "
            "AND c.state IN ('staged','gc_pending') "
            "AND (c.pin_until_ms IS NULL OR c.pin_until_ms<={p}) "
            "AND NOT EXISTS (SELECT 1 FROM sk_content_link l WHERE l.tenant_id=c.tenant_id "
            "AND l.session_id=c.session_id AND l.content_id=c.content_id) "
            f"ORDER BY c.content_id{self._backend.lock_clause}",
            (tenant_id, session_id, now_ms),
        )
        collected = 0
        for (content_id,) in candidates:
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
        outcome = self._key_provider.destroy(tenant_id, key.key_id)
        if outcome.status is KeyDestroyStatus.BINDING_MISMATCH:
            raise ContentBindingMismatch(outcome.reason or "session key binding mismatch")
        if outcome.status not in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED}:
            raise KeyUnavailable(outcome.reason or "session data key is not destroyed")

        # Key revocation is confirmed before any durable content row is removed.
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
