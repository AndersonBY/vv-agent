"""Encrypted content backends for the S1 session boundary.

The in-memory backend remains development-only.  Durable adapters accept a
caller-owned database connection and never open or commit a transaction of
their own.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentBindingMismatch,
    ContentIntegrityError,
    ContentLabels,
    ContentRef,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyProvider,
    KeyUnavailable,
    WrappedDataKey,
    _ExactKeyIntent,
)
from vv_agent.kernel._content_sql import (
    ContentExportRef,
    CoreSQLContentStore,
    ObjectLocatorCleaner,
    _content_aad,
    _content_mac,
    _key_from_row,
    _SQLContentBackend,
    _text,
)
from vv_agent.kernel._content_store import ContentLink, ContentState, SessionKeyRecord, SessionKeyState

_DATA_KEY_BYTES = 32
_NONCE_BYTES = 12
_KEY_WRAP_PREFIX = b"vv-agent/S1/key-wrap/v1\0"
_KEY_STATE_ENV = "VV_AGENT_LOCAL_KEY_STATE"


def _key_wrap_aad(tenant_id: str, session_id: str, key_id: str, *, provider_instance_id: str | None = None) -> bytes:
    binding = {
        "domain": "vv-agent/S1/key-wrap",
        "framing_version": 1,
        "purpose": "session-data-key",
        "tenant_id": tenant_id,
        "session_id": session_id,
        "key_id": key_id,
    }
    if provider_instance_id is not None:
        binding["provider_instance_id"] = provider_instance_id
    return canonical_json_bytes(binding, "data-key AAD")


def _default_local_key_state_path(master_key: bytes) -> Path:
    configured = os.environ.get(_KEY_STATE_ENV)
    if configured:
        return Path(configured).expanduser()
    state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    fingerprint = hashlib.sha256(master_key).hexdigest()[:24]
    return state_root / "vv-agent" / f"local-content-keys-{fingerprint}.sqlite3"


class AESGCMCipher:
    """Production AES-256-GCM adapter for the core content cipher port."""

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return AESGCM(key).encrypt(nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise ContentIntegrityError("content failed AEAD authentication") from exc

    def random_bytes(self, n: int) -> bytes:
        return secrets.token_bytes(n)


class LocalKeyProvider:
    """Development/test-only key provider with a caller-injected AES-256 master key."""

    provider_id = "local-development-v1"

    def __init__(self, master_key: bytes, *, state_path: str | Path | None = None) -> None:
        if not isinstance(master_key, bytes) or len(master_key) != _DATA_KEY_BYTES:
            raise ValueError("master_key must be exactly 32 bytes")
        self._master_key = master_key
        self._state_path = Path(state_path).expanduser() if state_path is not None else _default_local_key_state_path(master_key)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        try:
            self._state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._connection = self._connect()
            self._initialize_state()
        except BaseException:
            self.close()
            raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._state_path, timeout=30, check_same_thread=False)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
        except BaseException:
            connection.close()
            raise
        return connection

    def _state_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("LocalKeyProvider is closed")
        return self._connection

    def close(self) -> None:
        """Close the provider-owned state connection."""
        with self._lock:
            connection = self._connection
            if connection is None:
                return
            connection.close()
            self._connection = None

    def _initialize_state(self) -> None:
        master_fingerprint = hashlib.sha256(self._master_key).hexdigest()
        with self._lock, self._state_connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS local_content_key_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton=1),"
                "master_fingerprint TEXT NOT NULL"
                ") STRICT"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS local_content_keys ("
                "tenant_id TEXT NOT NULL,"
                "session_id TEXT NOT NULL,"
                "key_id TEXT PRIMARY KEY,"
                "destroyed INTEGER NOT NULL CHECK (destroyed IN (0,1)),"
                "evidence_id TEXT"
                ") STRICT"
            )
            connection.execute(
                "INSERT OR IGNORE INTO local_content_key_state(singleton,master_fingerprint) VALUES (1,?)",
                (master_fingerprint,),
            )
            stored_fingerprint = connection.execute(
                "SELECT master_fingerprint FROM local_content_key_state WHERE singleton=1"
            ).fetchone()
            if stored_fingerprint != (master_fingerprint,):
                raise ValueError("local key state belongs to a different master key")
        with suppress(OSError):
            self._state_path.chmod(0o600)

    def _key_state(self, key_id: str) -> tuple[str, str, bool, str | None] | None:
        with self._lock:
            connection = self._state_connection()
            row = connection.execute(
                "SELECT tenant_id,session_id,destroyed,evidence_id FROM local_content_keys WHERE key_id=?",
                (key_id,),
            ).fetchone()
        if row is None:
            return None
        return row[0], row[1], bool(row[2]), row[3]

    def _persist_new_key(self, tenant_id: str, session_id: str, key_id: str) -> None:
        with self._lock, self._state_connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO local_content_keys(tenant_id,session_id,key_id,destroyed,evidence_id) VALUES (?,?,?,0,NULL)",
                    (tenant_id, session_id, key_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ContentBindingMismatch("session data key identifier is already registered") from exc

    def _require_active_key(self, tenant_id: str, session_id: str, key_id: str) -> None:
        state = self._key_state(key_id)
        if state is None or state[2] or state[:2] != (tenant_id, session_id):
            raise KeyUnavailable("session data key is unavailable")

    def _provision_exact_key_lifecycle(self) -> str:
        """Provision a development authority whose state must stay outside primary restore."""
        instance_id = secrets.token_hex(32)
        with self._lock, self._state_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE local_exact_key_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton=1), "
                "instance_id TEXT NOT NULL"
                ") STRICT"
            )
            connection.execute(
                "CREATE TABLE local_exact_keys (key_id TEXT PRIMARY KEY, binding BLOB NOT NULL, evidence_id TEXT) STRICT"
            )
            connection.execute("INSERT INTO local_exact_key_state VALUES (1,?)", (instance_id,))
        return instance_id

    @staticmethod
    def _exact_key_instance_id(connection: sqlite3.Connection) -> str:
        try:
            row = connection.execute("SELECT instance_id FROM local_exact_key_state WHERE singleton=1").fetchone()
        except sqlite3.OperationalError as exc:
            raise KeyUnavailable("exact key provider state is unavailable") from exc
        if row is None:
            raise KeyUnavailable("exact key provider state is unavailable")
        return row[0]

    def _exact_key_provider_instance_id(self) -> str:
        with self._lock:
            return self._exact_key_instance_id(self._state_connection())

    def _exact_key_binding(self, connection: sqlite3.Connection, intent: _ExactKeyIntent) -> bytes:
        if intent.provider_instance_id != self._exact_key_instance_id(connection):
            raise ContentBindingMismatch("exact key belongs to another provider instance")
        return hmac.digest(
            self._master_key,
            canonical_json_bytes(
                [
                    "vv-agent/S1/exact-key-owner/v1",
                    intent.provider_instance_id,
                    intent.key_id,
                    intent.tenant_id,
                    intent.session_id,
                    intent.admission_root_id,
                    intent.admission_id,
                ],
                "exact key owner",
            ),
            "sha256",
        )

    def _allocate_exact_key(self, intent: _ExactKeyIntent) -> WrappedDataKey:
        """Create once. A duplicate or terminal ID cannot issue a second envelope."""
        with self._lock, self._state_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            binding = self._exact_key_binding(connection, intent)
            inserted = connection.execute(
                "INSERT INTO local_exact_keys(key_id,binding) VALUES (?,?) ON CONFLICT(key_id) DO NOTHING",
                (intent.key_id, binding),
            )
            if inserted.rowcount != 1:
                raise KeyUnavailable("exact key ID has already been used")
            return self._wrap_data_key(
                intent.tenant_id, intent.session_id, intent.key_id, provider_instance_id=intent.provider_instance_id
            )

    def _retire_exact_key(self, intent: _ExactKeyIntent) -> KeyDestroyOutcome:
        """Commit terminal evidence even if the exact allocation has yet to arrive."""
        with self._lock, self._state_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            binding = self._exact_key_binding(connection, intent)
            row = connection.execute(
                "SELECT binding,evidence_id FROM local_exact_keys WHERE key_id=?", (intent.key_id,)
            ).fetchone()
            if row is not None:
                owner_binding, evidence_id = row
                if not hmac.compare_digest(owner_binding, binding):
                    raise ContentBindingMismatch("exact key owner does not match")
                if evidence_id is not None:
                    return KeyDestroyOutcome(True, KeyDestroyStatus.ALREADY_DESTROYED, evidence_id)
            evidence_id = f"local-key-destroy-{secrets.token_urlsafe(18)}"
            connection.execute(
                "INSERT INTO local_exact_keys(key_id,binding,evidence_id) VALUES (?,?,?) "
                "ON CONFLICT(key_id) DO UPDATE SET evidence_id=excluded.evidence_id",
                (intent.key_id, binding, evidence_id),
            )
        return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, evidence_id)

    def _verify_exact_retirement(self, intent: _ExactKeyIntent) -> str | None:
        """Read existing terminal evidence without creating a provider record."""
        with self._lock:
            connection = self._state_connection()
            binding = self._exact_key_binding(connection, intent)
            row = connection.execute(
                "SELECT binding,evidence_id FROM local_exact_keys WHERE key_id=?", (intent.key_id,)
            ).fetchone()
        if row is None or not hmac.compare_digest(row[0], binding):
            return None
        return row[1]

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        tenant_id = _text(tenant_id, "tenant_id")
        session_id = _text(session_id, "session_id")
        key_id = f"key_{secrets.token_urlsafe(32)}"
        wrapped = self._wrap_data_key(tenant_id, session_id, key_id)
        self._persist_new_key(tenant_id, session_id, key_id)
        return wrapped

    def _wrap_data_key(
        self, tenant_id: str, session_id: str, key_id: str, *, provider_instance_id: str | None = None
    ) -> WrappedDataKey:
        nonce = secrets.token_bytes(_NONCE_BYTES)
        data_key = secrets.token_bytes(_DATA_KEY_BYTES)
        aad = _key_wrap_aad(tenant_id, session_id, key_id, provider_instance_id=provider_instance_id)
        wrapped = _KEY_WRAP_PREFIX + nonce + AESGCM(self._master_key).encrypt(nonce, data_key, aad)
        return WrappedDataKey(tenant_id, session_id, key_id, wrapped)

    def _decrypt_wrapped_key(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        provider_instance_id = None
        if wrapped_key.key_id.startswith(_EXACT_KEY_ID_PREFIX):
            try:
                row = (
                    self._state_connection()
                    .execute(
                        "SELECT instance_id FROM local_exact_key_state CROSS JOIN local_exact_keys "
                        "WHERE singleton=1 AND key_id=? AND evidence_id IS NULL",
                        (wrapped_key.key_id,),
                    )
                    .fetchone()
                )
            except sqlite3.OperationalError as exc:
                raise KeyUnavailable("exact key provider state is unavailable") from exc
            if row is None:
                raise KeyUnavailable("exact key is unavailable")
            provider_instance_id = row[0]
        else:
            self._require_active_key(tenant_id, session_id, wrapped_key.key_id)
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
                _key_wrap_aad(tenant_id, session_id, wrapped_key.key_id, provider_instance_id=provider_instance_id),
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
        if wrapped_key.tenant_id != tenant_id or wrapped_key.session_id != session_id:
            raise KeyUnavailable("wrapped data key does not belong to the session binding")
        with self._lock:
            return self._decrypt_wrapped_key(tenant_id, session_id, wrapped_key)

    def restore_data_key(self, wrapped_key: WrappedDataKey) -> None:
        """Validate persistent lifecycle state for a durable wrapped envelope."""
        if not isinstance(wrapped_key, WrappedDataKey):
            raise KeyUnavailable("wrapped data key is unavailable")
        tenant_id = _text(wrapped_key.tenant_id, "tenant_id")
        session_id = _text(wrapped_key.session_id, "session_id")
        with self._lock:
            self._decrypt_wrapped_key(tenant_id, session_id, wrapped_key)

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
        tenant_id = _text(tenant_id, "tenant_id")
        key_id = _text(key_id, "key_id")
        if key_id.startswith(_EXACT_KEY_ID_PREFIX):
            return KeyDestroyOutcome(False, KeyDestroyStatus.KEY_UNAVAILABLE, None, "Exact key intent is required.")
        with self._lock, self._state_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT tenant_id,session_id,destroyed,evidence_id FROM local_content_keys WHERE key_id=?",
                (key_id,),
            ).fetchone()
            if row is None:
                return KeyDestroyOutcome(
                    False,
                    KeyDestroyStatus.KEY_UNAVAILABLE,
                    None,
                    "Session key is unavailable.",
                )
            owner_tenant, _owner_session, destroyed, evidence_id = row
            if owner_tenant != tenant_id:
                return KeyDestroyOutcome(
                    False,
                    KeyDestroyStatus.BINDING_MISMATCH,
                    None,
                    "Session key belongs to another tenant.",
                )
            if destroyed:
                return KeyDestroyOutcome(True, KeyDestroyStatus.ALREADY_DESTROYED, evidence_id)
            evidence_id = f"local-key-destroy-{secrets.token_urlsafe(18)}"
            connection.execute(
                "UPDATE local_content_keys SET destroyed=1,evidence_id=? WHERE key_id=? AND destroyed=0",
                (evidence_id, key_id),
            )
        return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, evidence_id)


@dataclass(frozen=True, slots=True)
class _StoredContent:
    ref: ContentRef
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)


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
            mac = _content_mac(data_key, tenant_id, session_id, plaintext)
            identity = (tenant_id, session_id, wrapped_key.key_id, mac, labels)
            existing_id = self._dedup.get(identity)
            if existing_id is not None:
                return self._contents[(tenant_id, session_id, existing_id)].ref

            content_id = f"content_{secrets.token_urlsafe(32)}"
            size = len(plaintext)
            nonce = secrets.token_bytes(_NONCE_BYTES)
            aad = _content_aad(tenant_id, session_id, content_id, wrapped_key.key_id, mac, size, labels)
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
                        stored.ref.mac,
                        stored.ref.size,
                        stored.ref.labels,
                    ),
                )
            except (InvalidTag, ValueError) as exc:
                raise ContentIntegrityError("content failed AEAD authentication") from exc
            if len(plaintext) != ref.size:
                raise ContentIntegrityError("content size does not match its reference")
            actual_mac = _content_mac(data_key, tenant_id, session_id, plaintext)
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


class SQLContentStore(CoreSQLContentStore):
    """SDK SQL adapter with the production AES-256-GCM cipher bound by default."""

    def __init__(
        self,
        backend: _SQLContentBackend,
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
        object_locator_cleaner: ObjectLocatorCleaner | None = None,
    ) -> None:
        super().__init__(
            backend,
            key_provider,
            AESGCMCipher(),
            provider_id=provider_id,
            object_locator_cleaner=object_locator_cleaner,
        )


__all__ = [
    "AESGCMCipher",
    "ContentExportRef",
    "ContentLink",
    "ContentState",
    "InMemoryEncryptedContentStore",
    "LocalKeyProvider",
    "SQLContentStore",
    "SessionKeyRecord",
    "SessionKeyState",
    "_SQLContentBackend",
    "_content_aad",
    "_content_mac",
    "_key_from_row",
]
