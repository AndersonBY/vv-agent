"""Local Ed25519 signing and public-key verification for S1 ledgers."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from operator import call
from types import BuiltinMethodType, MappingProxyType

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from vv_agent.kernel._sql import _CHECKPOINT_DOMAIN, _INPUT_ADMISSION_DOMAIN, LedgerIntegrityError

# Checkpoints and admissions sign metadata. Tombstones also sign actor data.
_CACHEABLE_PROOF_PREFIXES = (
    (b'{"chain_digest":"', f'","domain":"{_CHECKPOINT_DOMAIN}",'.encode()),
    (b'{"body_digest":"', f'","domain":"{_INPUT_ADMISSION_DOMAIN}",'.encode()),
)


@cache
def _native_ed25519_key_type() -> type[Ed25519PublicKey]:
    # The RFC 8032 test key identifies the native implementation through cryptography's public API.
    return type(
        Ed25519PublicKey.from_public_bytes(bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"))
    )


@dataclass(frozen=True)
class _Ed25519KeyBinding:
    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False)

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)


class Ed25519LedgerVerifier:
    """Verify with trusted public keys, including retired keys retained for reads."""

    def __init__(self, public_keys: Mapping[str, Ed25519PublicKey]) -> None:
        self._public_keys = MappingProxyType(dict(public_keys))
        if not self._public_keys or any(not isinstance(key_id, str) or not key_id.strip() for key_id in self._public_keys):
            raise ValueError("ledger verification requires nonempty key identifiers")
        if any(not isinstance(key, Ed25519PublicKey) for key in self._public_keys.values()):
            raise ValueError("ledger verification requires Ed25519 public keys")
        self._verify_cached = lru_cache(maxsize=1024)(call)

    def bind_key(self) -> _Ed25519KeyBinding:
        raise LedgerIntegrityError("public-key ledger verification cannot sign")

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        public_key = self._public_keys.get(key_id)
        if public_key is None:
            return False
        try:
            verify_signature = public_key.verify
            if (
                type(verify_signature) is BuiltinMethodType
                and verify_signature.__self__ is public_key
                and type(public_key) is _native_ed25519_key_type()
                and type(message) is bytes
                and len(message) <= 1024
                and type(signature) is bytes
                and len(signature) == 64
                and any(
                    message.startswith(prefix) and message.startswith(domain, len(prefix) + 64)
                    for prefix, domain in _CACHEABLE_PROOF_PREFIXES
                )
            ):
                # The bound method identifies the trusted key. Exceptions never populate the cache.
                self._verify_cached(verify_signature, signature, message)
            else:
                verify_signature(signature, message)
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
