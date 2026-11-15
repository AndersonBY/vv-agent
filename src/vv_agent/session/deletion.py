"""Durable S1 session forget and purge lifecycle.

The service fences the SQL ledger before touching session keys or content. A
failed key revocation therefore leaves a retryable pending tombstone and a
closed execution boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vv_agent.kernel._content import ForgetTombstone, Principal, TombstoneState
from vv_agent.kernel._sql import LedgerSigner, SessionNotLive, SQLStore
from vv_agent.kernel._store import Conflict, StoredInput, StoredRecord

from .content_store import SQLContentStore


class DeletionSignerUnavailable(RuntimeError):
    """Trusted deletion was requested without a usable bound ledger signer."""


class DeletionAuthenticity(StrEnum):
    SIGNED = "signed"
    HASH_ONLY = "hash-only"


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    operation: str
    session_id: str
    session_ids: tuple[str, ...]
    authenticity: DeletionAuthenticity
    remote_effect_state: str = "unknown"


@dataclass(frozen=True, slots=True)
class SkeletonSnapshot:
    tenant_id: str
    session_id: str
    head_seq: int
    inbox_seq: int
    chain_digest: str
    records: tuple[StoredRecord, ...]
    inputs: tuple[StoredInput, ...]
    tombstone: ForgetTombstone | None


@dataclass(frozen=True, slots=True)
class _Authority:
    bound: Any | None
    verifier: LedgerSigner | None
    authenticity: DeletionAuthenticity

    @property
    def allow_unsigned(self) -> bool:
        return self.authenticity is DeletionAuthenticity.HASH_ONLY


class SessionDeletionStore:
    """Fence, erase, and converge one tenant's session without hidden retries."""

    def __init__(
        self,
        store: SQLStore,
        content_store: SQLContentStore,
        *,
        signer: LedgerSigner | None = None,
        allow_unsigned: bool = False,
    ) -> None:
        self._store = store
        self._content_store = content_store
        self._signer = signer
        self._allow_unsigned = allow_unsigned

    def _authority(self, scoped: SQLStore) -> _Authority:
        configured, configured_unsigned, configured_signer = scoped._input_admission_policy()
        if self._allow_unsigned:
            if self._signer is not None or (configured and not configured_unsigned):
                raise ValueError("hash-only deletion cannot claim a trusted signer")
            return _Authority(None, None, DeletionAuthenticity.HASH_ONLY)

        signer = self._signer or configured_signer
        if signer is None or not callable(getattr(signer, "verify", None)):
            raise DeletionSignerUnavailable("deletion requires a configured ledger signer")
        bound = signer.bind_key()
        key_id = getattr(bound, "key_id", None)
        if not isinstance(key_id, str) or not key_id or not callable(getattr(bound, "sign", None)):
            raise DeletionSignerUnavailable("ledger signer returned an invalid key binding")
        return _Authority(bound, signer, DeletionAuthenticity.SIGNED)

    def _scoped(self, tenant_id: str) -> SQLStore:
        scoped = self._store.for_tenant(tenant_id)
        if not isinstance(scoped, SQLStore):
            bound = getattr(scoped, "_bound", None)
            if not isinstance(bound, SQLStore):
                raise TypeError("deletion store requires a concrete SQL session store")
            scoped = bound
        return scoped

    def forget_session(self, tenant_id: str, session_id: str, *, actor: Principal) -> DeletionReceipt:
        """Destroy one session key/content after fencing its complete subtree."""
        scoped = self._scoped(tenant_id)
        authority = self._authority(scoped)
        session_ids = scoped.begin_forget(
            session_id,
            actor,
            bound_signer=authority.bound,
            verifier=authority.verifier,
            allow_unsigned=authority.allow_unsigned,
        )
        for member in session_ids:
            with scoped._transaction():
                self._content_store.forget_session(tenant_id, member)
        scoped.complete_forget(session_id)
        return DeletionReceipt(
            "forget",
            session_id,
            session_ids,
            authority.authenticity,
        )

    def purge_session(self, tenant_id: str, session_id: str, *, actor: Principal) -> DeletionReceipt:
        """Cascade descendants, erase their content, then delete owned SQL rows."""
        scoped = self._scoped(tenant_id)
        authority = self._authority(scoped)
        session_ids = scoped.begin_purge(
            session_id,
            actor,
            bound_signer=authority.bound,
            verifier=authority.verifier,
            allow_unsigned=authority.allow_unsigned,
        )
        for member in session_ids:
            with scoped._transaction():
                self._content_store.forget_session(tenant_id, member)
        completed = scoped.complete_purge(session_id)
        return DeletionReceipt(
            "purge",
            session_id,
            completed,
            authority.authenticity,
        )

    def read_skeleton(self, tenant_id: str, session_id: str) -> SkeletonSnapshot:
        """Read and verify retained reference-form data without content keys."""
        scoped = self._scoped(tenant_id)
        authority = self._authority(scoped)
        with scoped._transaction():
            row = scoped._one(
                "SELECT head_seq,inbox_seq,deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, session_id),
            )
            if row[2] not in {"forgetting", "forgotten"}:
                raise SessionNotLive("skeleton access requires a fenced forget state")
            head_seq, inbox_seq = row[:2]
            verified, chains = scoped._verified_records(session_id, head_seq)
            records = tuple(scoped._stored(session_id, item) for item in verified)
            input_rows = scoped._rows(
                "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
                "admission_signature,admission_signed_ms FROM sk_inbox "
                "WHERE tenant_id=%s AND session_id=%s ORDER BY input_seq",
                (tenant_id, session_id),
            )
            inputs = tuple(scoped._stored_input(session_id, item) for item in input_rows)
            tombstone = scoped._tombstone(session_id)
            if tombstone is not None:
                scoped._verify_tombstone_object(
                    tombstone,
                    authority.verifier,
                    allow_unsigned=authority.allow_unsigned,
                )
            return SkeletonSnapshot(
                tenant_id,
                session_id,
                head_seq,
                inbox_seq,
                chains[-1],
                records,
                inputs,
                tombstone,
            )

    def verify_tombstones(self, tenant_id: str, root_session_id: str) -> tuple[ForgetTombstone, ...]:
        """Authenticate every complete tombstone retained by a purge scope."""
        scoped = self._scoped(tenant_id)
        authority = self._authority(scoped)
        rows = scoped._rows(
            "SELECT session_id FROM sk_tombstone WHERE tenant_id=%s AND scope_root_session_id=%s ORDER BY session_id",
            (tenant_id, root_session_id),
        )
        if not rows:
            raise Conflict("deletion tombstone not found in tenant")
        tombstones: list[ForgetTombstone] = []
        for (session_id,) in rows:
            tombstone = scoped._tombstone(session_id)
            if tombstone is None or tombstone.scope_root_session_id != root_session_id:
                raise Conflict("deletion tombstone scope is inconsistent")
            if tombstone.state is not TombstoneState.COMPLETE:
                raise Conflict("deletion tombstone is still pending")
            scoped._verify_tombstone_object(
                tombstone,
                authority.verifier,
                allow_unsigned=authority.allow_unsigned,
            )
            tombstones.append(tombstone)
        return tuple(tombstones)


__all__ = [
    "DeletionAuthenticity",
    "DeletionReceipt",
    "DeletionSignerUnavailable",
    "SessionDeletionStore",
    "SkeletonSnapshot",
]
