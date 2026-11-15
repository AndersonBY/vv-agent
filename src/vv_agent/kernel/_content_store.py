"""Encrypted content-store foundation for the S1 session boundary.

The included backend stores ciphertext only in process memory. It is intended for
development and tests; durable SQLite and PostgreSQL adapters remain future work.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ._canonical_json import canonical_json_bytes
from ._content import (
    ContentBindingMismatch,
    ContentLabels,
    ContentRef,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyProvider,
    WrappedDataKey,
)

_DATA_KEY_BYTES = 32
_NONCE_BYTES = 12
_KEY_WRAP_PREFIX = b"vv-agent/S1/key-wrap/v1\0"


class ContentStoreError(RuntimeError):
    """Base class for content backend failures."""


class ContentUnavailable(ContentStoreError):
    """The requested tenant, session, or content object is absent."""


class KeyUnavailable(ContentStoreError):
    """The session data key is absent, destroyed, or cannot be unwrapped."""


class ContentIntegrityError(ContentStoreError):
    """Content metadata, ciphertext, or authentication material is invalid."""


@runtime_checkable
class ContentStore(Protocol):
    """Backend-neutral encrypted content operations for one session binding."""

    def put(
        self,
        tenant_id: str,
        session_id: str,
        plaintext: bytes,
        labels: ContentLabels,
    ) -> ContentRef: ...

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes: ...

    def forget_session(self, tenant_id: str, session_id: str) -> None: ...


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
