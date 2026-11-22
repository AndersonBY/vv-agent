"""Private SQLite content capabilities used by the object-backed runtime."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from ._allocation_journal import AllocationOwner, KeyIntent, SQLiteAllocationJournal
from ._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentBindingMismatch,
    ContentUnavailable,
    KeyProvider,
    KeyUnavailable,
    WrappedDataKey,
    _ExactKeyIntent,
)

_AES_GCM_TAG_BYTES = 16


class _ObjectStore(Protocol):
    def put(self, owner: AllocationOwner, ciphertext: bytes, *, pin_until_ms: int, check_fence: Callable[[], None]) -> bytes: ...

    def get(self, owner: AllocationOwner, locator: bytes, *, ciphertext_size: int) -> bytes: ...


class _ExactAllocator(KeyProvider, Protocol):
    def _allocate_exact_key(self, intent: _ExactKeyIntent) -> WrappedDataKey: ...


@dataclass(frozen=True, slots=True)
class _SQLiteObjectContent:
    journal: SQLiteAllocationJournal
    objects: _ObjectStore
    provider_instance_id: str

    def _key(self, tenant_id: str, session_id: str, key_id: str) -> KeyIntent:
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
            _EXACT_KEY_ID_PREFIX + secrets.token_hex(32),
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

    @staticmethod
    def pin() -> int:
        return time.time_ns() // 1_000_000 + 5_000

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
        owner = AllocationOwner(
            tenant_id, session_id, content_id, key.admission_root_id, key.admission_id, self.provider_instance_id, key_id
        )
        return self.objects.put(owner, ciphertext, pin_until_ms=self.pin(), check_fence=check_fence)

    def get(self, tenant_id: str, session_id: str, content_id: str, key_id: str, locator: bytes, *, size: int) -> bytes:
        key = self._key(tenant_id, session_id, key_id).intent
        owner = AllocationOwner(
            tenant_id, session_id, content_id, key.admission_root_id, key.admission_id, self.provider_instance_id, key_id
        )
        ciphertext_size = size + _AES_GCM_TAG_BYTES
        result = self.objects.get(owner, locator, ciphertext_size=ciphertext_size)
        if not isinstance(result, bytes) or len(result) != ciphertext_size:
            raise ContentUnavailable("external ciphertext is incomplete")
        return result
