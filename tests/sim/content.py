"""Deterministic Sim host ports over real AES-256-GCM encryption."""

from __future__ import annotations

import base64
import hashlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    ContentIntegrityError,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    WrappedDataKey,
)

_KEY_ENVELOPE_PREFIX = b"vv-agent/S1/sim-key-wrap/v1\0"


class SimContentEntropy:
    """A SHA-256 byte stream whose state cannot be advanced by scheduler choices."""

    def __init__(self, seed: int, domain: bytes) -> None:
        self._prefix = b"vv-agent/S1/sim-content/v1\0" + str(seed).encode("ascii") + b"\0" + domain + b"\0"
        self._counter = 0

    def random_bytes(self, count: int) -> bytes:
        if type(count) is not int or count < 0:
            raise ValueError("random byte count must be a non-negative integer")
        output = bytearray()
        while len(output) < count:
            output.extend(hashlib.sha256(self._prefix + self._counter.to_bytes(8, "big")).digest())
            self._counter += 1
        return bytes(output[:count])


class SimAESGCMCipher:
    """Real AES-GCM with a deterministic stream reserved for content allocation."""

    def __init__(self, seed: int) -> None:
        self._entropy = SimContentEntropy(seed, b"cipher")
        self.decrypts = 0

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return AESGCM(key).encrypt(nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        self.decrypts += 1
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise ContentIntegrityError("content failed AEAD authentication") from exc

    def random_bytes(self, count: int) -> bytes:
        return self._entropy.random_bytes(count)


class SimContentKeyProvider:
    """Deterministic, process-local scripted provider with destroy state."""

    provider_id = "sim-scripted-v1"

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._entropy = SimContentEntropy(seed, b"key-provider")
        self._bindings: dict[str, tuple[str, str]] = {}
        self._destroyed: set[str] = set()
        self.revision = 0

    @staticmethod
    def _data_key(seed: int, key_id: str) -> bytes:
        return hashlib.sha256(
            b"vv-agent/S1/sim-data-key/v1\0" + str(seed).encode("ascii") + b"\0" + key_id.encode("ascii")
        ).digest()

    @staticmethod
    def _envelope(tenant_id: str, session_id: str, key_id: str) -> bytes:
        binding = canonical_json_bytes({"key_id": key_id, "session_id": session_id, "tenant_id": tenant_id})
        return _KEY_ENVELOPE_PREFIX + binding

    def _bind_validated(
        self,
        wrapped_key: WrappedDataKey,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(wrapped_key, WrappedDataKey):
            raise KeyUnavailable("session data key is unavailable")
        tenant_id = wrapped_key.tenant_id if tenant_id is None else tenant_id
        session_id = wrapped_key.session_id if session_id is None else session_id
        if not isinstance(tenant_id, str) or not tenant_id or not isinstance(session_id, str) or not session_id:
            raise KeyUnavailable("session data key binding does not match")
        if (wrapped_key.tenant_id, wrapped_key.session_id) != (tenant_id, session_id):
            raise KeyUnavailable("session data key binding does not match")
        expected = self._envelope(tenant_id, session_id, wrapped_key.key_id)
        if wrapped_key.wrapped_key != expected:
            raise KeyUnavailable("session data key envelope failed authentication")
        binding = (tenant_id, session_id)
        registered = self._bindings.setdefault(wrapped_key.key_id, binding)
        if registered != binding:
            raise KeyUnavailable("session data key binding conflicts with the session key")

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        key_id = "key_" + base64.urlsafe_b64encode(self._entropy.random_bytes(32)).rstrip(b"=").decode("ascii")
        self._bindings[key_id] = (tenant_id, session_id)
        return WrappedDataKey(tenant_id, session_id, key_id, self._envelope(tenant_id, session_id, key_id))

    def restore_data_key(self, wrapped_key: WrappedDataKey) -> None:
        self._bind_validated(wrapped_key)

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        self._bind_validated(wrapped_key, tenant_id=tenant_id, session_id=session_id)
        if wrapped_key.key_id in self._destroyed:
            raise KeyUnavailable("session data key is unavailable")
        return self._data_key(self._seed, wrapped_key.key_id)

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
        binding = self._bindings.get(key_id)
        if binding is None:
            return KeyDestroyOutcome(False, KeyDestroyStatus.KEY_UNAVAILABLE, None, "session data key is unavailable")
        if binding[0] != tenant_id:
            return KeyDestroyOutcome(False, KeyDestroyStatus.BINDING_MISMATCH, None, "session data key tenant does not match")
        if key_id in self._destroyed:
            return KeyDestroyOutcome(True, KeyDestroyStatus.ALREADY_DESTROYED, f"sim-destroy/{key_id}")
        self._destroyed.add(key_id)
        self.revision += 1
        return KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, f"sim-destroy/{key_id}")


__all__ = ["SimAESGCMCipher", "SimContentEntropy", "SimContentKeyProvider"]
