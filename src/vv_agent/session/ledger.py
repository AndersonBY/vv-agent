"""Local Ed25519 signing and public-key verification for S1 ledgers."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from vv_agent.kernel._sql import LedgerIntegrityError


@dataclass(frozen=True)
class _Ed25519KeyBinding:
    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False)

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)


class Ed25519LedgerVerifier:
    """Verify with trusted public keys, including retired keys retained for reads."""

    def __init__(self, public_keys: Mapping[str, Ed25519PublicKey]) -> None:
        self._public_keys = dict(public_keys)
        if not self._public_keys or any(not isinstance(key_id, str) or not key_id.strip() for key_id in self._public_keys):
            raise ValueError("ledger verification requires nonempty key identifiers")
        if any(not isinstance(key, Ed25519PublicKey) for key in self._public_keys.values()):
            raise ValueError("ledger verification requires Ed25519 public keys")

    def bind_key(self) -> _Ed25519KeyBinding:
        raise LedgerIntegrityError("public-key ledger verification cannot sign")

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        public_key = self._public_keys.get(key_id)
        if public_key is None:
            return False
        try:
            public_key.verify(signature, message)
        except InvalidSignature:
            return False
        return True


class Ed25519LedgerSigner(Ed25519LedgerVerifier):
    """Sign with one active key and verify retained history across key rotation."""

    def __init__(
        self,
        key_id: str,
        private_key: Ed25519PrivateKey,
        *,
        retired_keys: Mapping[str, Ed25519PublicKey] | None = None,
    ) -> None:
        public_keys = dict(retired_keys or {})
        if key_id in public_keys:
            raise ValueError("active ledger key identifier is already assigned to a retired key")
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("ledger signing requires an Ed25519 private key")
        public_keys[key_id] = private_key.public_key()
        super().__init__(public_keys)
        self._binding = _Ed25519KeyBinding(key_id, private_key)

    def bind_key(self) -> _Ed25519KeyBinding:
        return self._binding
