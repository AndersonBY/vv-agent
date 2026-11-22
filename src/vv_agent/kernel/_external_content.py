"""Private SQLite content capabilities used by the object-backed runtime."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, cast

from ._allocation_journal import Allocation, AllocationOwner, KeyIntent, SQLiteAllocationJournal
from ._canonical_json import canonical_json_bytes
from ._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyProvider,
    KeyUnavailable,
    WrappedDataKey,
    _ExactKeyIntent,
)

_AES_GCM_TAG_BYTES = 16
_ALLOCATION_PIN_MS = 5_000


class _ObjectStore(Protocol):
    @property
    def _storage_identity(self) -> tuple[str, str, str]: ...

    def _uses_journal(self, journal: SQLiteAllocationJournal) -> bool: ...

    def put(self, owner: AllocationOwner, ciphertext: bytes, *, pin_until_ms: int, check_fence: Callable[[], None]) -> bytes: ...

    def get(self, owner: AllocationOwner, locator: bytes, *, ciphertext_size: int) -> bytes: ...

    def delete_claimed(self, claim: Allocation, *, check_fence: Callable[[], None]) -> None: ...


class _ExactAllocator(KeyProvider, Protocol):
    def _allocate_exact_key(self, intent: _ExactKeyIntent) -> WrappedDataKey: ...


@dataclass(frozen=True, slots=True)
class _SQLiteObjectContent:
    journal: SQLiteAllocationJournal
    objects: _ObjectStore
    provider_instance_id: str
    terminal_bound_delete: bool = False

    def _binding_digest(self) -> str:
        identity = self.journal.identity
        endpoint, bucket, prefix = self.objects._storage_identity
        if not all(isinstance(value, str) and value for value in (endpoint, bucket, prefix)):
            raise ContentCleanupUnavailable("external content requires a stable object storage identity")
        binding = canonical_json_bytes(
            [identity.journal_id, identity.database_id, identity.namespace, self.provider_instance_id, endpoint, bucket, prefix]
        )
        return sha256(binding).hexdigest()

    def verify_installation(self, database_id: str, provider: KeyProvider) -> str:
        if self.journal.verify_identity().database_id != database_id:
            raise ContentUnavailable("external content journal belongs to a different SQLite installation")
        if not self.objects._uses_journal(self.journal):
            raise ContentUnavailable("external content object adapter requires the same allocation journal")
        provider_instance_id = getattr(provider, "_exact_key_provider_instance_id", None)
        if (
            not callable(provider_instance_id)
            or provider_instance_id() != self.provider_instance_id
            or not all(
                callable(getattr(provider, method, None))
                for method in ("_allocate_exact_key", "_retire_exact_key", "_verify_exact_retirement")
            )
        ):
            raise ContentUnavailable("external content requires the provisioned exact key provider")
        return self._binding_digest()

    def _key_prefix(self) -> str:
        return _EXACT_KEY_ID_PREFIX + self._binding_digest()[:32]

    def require_erasure_binding(self, key_id: str) -> None:
        if not self.terminal_bound_delete or not key_id.startswith(self._key_prefix()):
            raise ContentCleanupUnavailable("external erasure requires the matching journal and terminal bound DELETE")

    def erasure_inventory(self, tenant_id: str, session_id: str, key_id: str) -> tuple[KeyIntent | None, dict[bytes, Allocation]]:
        self.require_erasure_binding(key_id)
        expected_key = self._root_intent(tenant_id, session_id, key_id)
        keys = self.journal.scoped_key_intents(tenant_id, (session_id,))
        if keys and (len(keys) != 1 or keys[0].intent != expected_key):
            raise ContentCleanupUnavailable("external key inventory does not match the frozen root")
        allocations = {}
        for row in self.journal.scoped_allocations(tenant_id, (session_id,)):
            owner = row.owner
            if owner is None or owner != self._owner(expected_key, owner.content_id):
                raise ContentCleanupUnavailable("external object inventory does not match the frozen root")
            allocations[b"\x01" + row.allocation_id] = row
        return keys[0] if keys else None, allocations

    def _root_intent(self, tenant_id: str, session_id: str, key_id: str) -> _ExactKeyIntent:
        return _ExactKeyIntent(key_id, self.provider_instance_id, tenant_id, session_id, session_id, "session/create")

    def verify_retired(self, provider: KeyProvider, tenant_id: str, session_id: str, key_id: str) -> None:
        verify = getattr(provider, "_verify_exact_retirement", None)
        if not callable(verify) or not verify(self._root_intent(tenant_id, session_id, key_id)):
            raise ContentCleanupUnavailable("external key retirement evidence is unavailable")

    def retire(self, provider: KeyProvider, key: KeyIntent, *, check_fence: Callable[[], None]) -> KeyIntent:
        check_fence()
        if key.state == "retired":
            verify = getattr(provider, "_verify_exact_retirement", None)
            if not callable(verify) or verify(key.intent) != key.evidence_id:
                raise ContentCleanupUnavailable("external key retirement evidence changed")
            check_fence()
            return key
        retire = getattr(provider, "_retire_exact_key", None)
        if not callable(retire):
            raise ContentCleanupUnavailable("external content requires exact key retirement")
        claim = self.journal.claim_key_intent(key, require_expired_pin=False)
        if claim is None:
            raise ContentCleanupUnavailable("external key retirement claim changed")
        check_fence()
        outcome = retire(claim.intent)
        check_fence()
        if not isinstance(outcome, KeyDestroyOutcome) or outcome.status not in {
            KeyDestroyStatus.DESTROYED,
            KeyDestroyStatus.ALREADY_DESTROYED,
        }:
            raise ContentCleanupUnavailable("external key retirement is unconfirmed")
        retired = self.journal.record_key_retirement(claim, outcome)
        if retired is None:
            raise ContentCleanupUnavailable("external key retirement record changed")
        return retired

    @staticmethod
    def _owner(key: _ExactKeyIntent, content_id: str) -> AllocationOwner:
        return AllocationOwner(
            key.tenant_id,
            key.session_id,
            content_id,
            key.admission_root_id,
            key.admission_id,
            key.provider_instance_id,
            key.key_id,
        )

    def _key(self, tenant_id: str, session_id: str, key_id: str) -> KeyIntent:
        if not key_id.startswith(self._key_prefix()):
            raise ContentBindingMismatch("external content exact key does not match its storage binding")
        key = self.journal.get_key_intent(self.provider_instance_id, key_id)
        if key is None or key.state != "allocating":
            raise KeyUnavailable("external content exact key is unavailable")
        if (key.intent.tenant_id, key.intent.session_id) != (tenant_id, session_id):
            raise ContentBindingMismatch("external content exact key does not match its SQL owner")
        return key

    def allocate(
        self, provider: KeyProvider, tenant_id: str, session_id: str, *, check_fence: Callable[[], None]
    ) -> WrappedDataKey:
        check_fence()
        if not callable(getattr(provider, "_allocate_exact_key", None)):
            raise KeyUnavailable("external content requires an exact key provider")
        pending = self.journal.key_intents_for_admission(tenant_id, session_id, session_id, "session/create")
        if pending:
            raise KeyUnavailable("external content key intent requires reconciliation")
        intent = _ExactKeyIntent(
            self._key_prefix() + secrets.token_hex(16),
            self.provider_instance_id,
            tenant_id,
            session_id,
            session_id,
            "session/create",
        )
        reservation = self.journal.reserve_key_intent(intent, pin_until_ms=self.pin())
        check_fence()
        if reservation is None or self.journal.claim_key_allocation(reservation) is None:
            raise KeyUnavailable("external content could not confirm key allocation authority")
        check_fence()
        wrapped = cast(_ExactAllocator, provider)._allocate_exact_key(intent)
        check_fence()
        if wrapped.key_id != intent.key_id:
            raise ContentBindingMismatch("external content provider returned a different exact key")
        return wrapped

    def pin(self) -> int:
        return self.journal._pin_until_ms(_ALLOCATION_PIN_MS)

    def put(
        self,
        tenant_id: str,
        session_id: str,
        content_id: str,
        key_id: str,
        ciphertext: bytes,
        *,
        plaintext_size: int,
        check_fence: Callable[[], None],
    ) -> bytes:
        if len(ciphertext) != plaintext_size + _AES_GCM_TAG_BYTES:
            raise ContentUnavailable("external content requires AES-GCM ciphertext length")
        check_fence()
        key = self._key(tenant_id, session_id, key_id).intent
        owner = self._owner(key, content_id)
        pin_until_ms = self.pin()
        check_fence()
        return self.objects.put(owner, ciphertext, pin_until_ms=pin_until_ms, check_fence=check_fence)

    def get(self, tenant_id: str, session_id: str, content_id: str, key_id: str, locator: bytes, *, size: int) -> bytes:
        key = self._key(tenant_id, session_id, key_id).intent
        owner = self._owner(key, content_id)
        ciphertext_size = size + _AES_GCM_TAG_BYTES
        result = self.objects.get(owner, locator, ciphertext_size=ciphertext_size)
        if not isinstance(result, bytes) or len(result) != ciphertext_size:
            raise ContentUnavailable("external ciphertext is incomplete")
        return result
