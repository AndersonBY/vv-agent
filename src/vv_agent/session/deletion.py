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
from vv_agent.kernel._content_sql import CoreSQLContentStore
from vv_agent.kernel._sql import LedgerIntegrityError, LedgerSigner, SessionNotLive, SQLStore
from vv_agent.kernel._store import (
    SESSION_DELETION_INVENTORY,
    Conflict,
    StoredInput,
    StoredRecord,
)

SESSION_DELETION_TABLES = SESSION_DELETION_INVENTORY

PURGE_DELETE_TABLES = tuple(table for table in SESSION_DELETION_TABLES if table != "sk_tombstone")
_CONTENT_ERASE_TABLES = ("sk_session_key", "sk_content", "sk_content_link")


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


@dataclass(frozen=True, slots=True)
class _ScopeMember:
    session_id: str
    parent_session_id: str | None
    head_seq: int
    chain_digest: str | None
    deletion_state: str


class SessionDeletionStore:
    """Fence, erase, and converge one tenant's session without hidden retries."""

    def __init__(
        self,
        store: SQLStore,
        content_store: CoreSQLContentStore,
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

    @staticmethod
    def _validate_scope_ids(root_session_id: str, session_ids: tuple[str, ...]) -> None:
        if not session_ids or len(session_ids) != len(set(session_ids)) or root_session_id not in session_ids:
            raise LedgerIntegrityError("session deletion scope is empty, duplicated, or missing its root")
        if any(not isinstance(session_id, str) or not session_id or "\x00" in session_id for session_id in session_ids):
            raise LedgerIntegrityError("session deletion scope contains an invalid session identity")

    def _read_scope(
        self,
        scoped: SQLStore,
        tenant_id: str,
        root_session_id: str,
        session_ids: tuple[str, ...],
        *,
        require_present: bool,
    ) -> tuple[_ScopeMember, ...]:
        """Verify the returned primary-key inventory against this fenced root subtree."""
        self._validate_scope_ids(root_session_id, session_ids)
        rows = scoped._rows(
            """
            WITH RECURSIVE deletion_scope(session_id) AS (
                SELECT session_id FROM sk_session WHERE tenant_id=%s AND session_id=%s
                UNION ALL
                SELECT child.session_id
                FROM sk_session child
                JOIN deletion_scope ON child.tenant_id=%s
                    AND child.parent_session_id=deletion_scope.session_id
            )
            SELECT s.session_id,s.parent_session_id,s.head_seq,r.chain_digest,s.deletion_state
            FROM sk_session s
            JOIN deletion_scope ON deletion_scope.session_id=s.session_id
            LEFT JOIN sk_record r ON r.tenant_id=s.tenant_id AND r.session_id=s.session_id
                AND r.seq=s.head_seq
            WHERE s.tenant_id=%s
            ORDER BY s.session_id
            """,
            (tenant_id, root_session_id, tenant_id, tenant_id),
        )
        actual_ids = {row[0] for row in rows}
        if not rows:
            if require_present:
                raise LedgerIntegrityError("deletion scope root is missing while session rows remain")
            return ()
        if actual_ids != set(session_ids):
            raise LedgerIntegrityError("session deletion scope does not match the root descendant inventory")
        members = tuple(_ScopeMember(*row) for row in rows)
        by_id = {member.session_id: member for member in members}
        for member in members:
            if member.session_id == root_session_id:
                continue
            if member.parent_session_id not in by_id:
                raise LedgerIntegrityError("session deletion scope contains a member outside its root parent chain")
        return members

    @staticmethod
    def _scope_root_is_ancestor(member_id: str, scope_root: str, members_by_id: dict[str, _ScopeMember]) -> bool:
        current = member_id
        seen: set[str] = set()
        while current != scope_root:
            if current in seen:
                return False
            seen.add(current)
            member = members_by_id.get(current)
            if member is None:
                return False
            parent_session_id = member.parent_session_id
            if parent_session_id is None or parent_session_id not in members_by_id:
                return False
            current = parent_session_id
        return True

    def _verify_scope_evidence(
        self,
        scoped: SQLStore,
        tenant_id: str,
        root_session_id: str,
        session_ids: tuple[str, ...],
        *,
        require_present: bool,
        expected_states: set[str],
        require_complete: bool,
        authority: _Authority,
    ) -> tuple[ForgetTombstone, ...]:
        members = self._read_scope(
            scoped,
            tenant_id,
            root_session_id,
            session_ids,
            require_present=require_present,
        )
        members_by_id = {member.session_id: member for member in members}
        if members and any(member.deletion_state not in expected_states for member in members):
            raise LedgerIntegrityError("deletion scope did not retain its fenced state")

        scope_ids = set(session_ids)
        root_tombstone: ForgetTombstone | None = None
        tombstones: list[ForgetTombstone] = []
        for session_id in session_ids:
            tombstone = scoped._tombstone(session_id)
            if tombstone is None:
                raise LedgerIntegrityError("deletion scope member is missing its tombstone")
            if require_complete and tombstone.state is not TombstoneState.COMPLETE:
                raise LedgerIntegrityError("deletion scope retained an incomplete tombstone")
            if not require_complete and tombstone.state not in {TombstoneState.PENDING, TombstoneState.COMPLETE}:
                raise LedgerIntegrityError("deletion scope retained an invalid tombstone state")
            if tombstone.tenant_id != tenant_id or tombstone.actor.tenant_id != tenant_id:
                raise LedgerIntegrityError("deletion tombstone crosses the tenant boundary")
            if tombstone.scope_root_session_id not in scope_ids:
                raise LedgerIntegrityError("deletion tombstone scope root is outside the deletion scope")
            scoped._verify_tombstone_object(
                tombstone,
                authority.verifier,
                allow_unsigned=authority.allow_unsigned,
            )
            if members:
                member = members_by_id[session_id]
                if tombstone.final_seq != member.head_seq or tombstone.chain_digest != member.chain_digest:
                    raise LedgerIntegrityError("deletion tombstone does not match the member's retained head")
                if not self._scope_root_is_ancestor(
                    session_id,
                    tombstone.scope_root_session_id,
                    members_by_id,
                ):
                    raise LedgerIntegrityError("deletion tombstone scope root is not a member ancestor")
            if session_id == root_session_id:
                root_tombstone = tombstone
            tombstones.append(tombstone)
        if not members and (root_tombstone is None or root_tombstone.scope_root_session_id != root_session_id):
            raise LedgerIntegrityError("tombstone-only deletion scope is missing its root anchor")
        return tuple(tombstones)

    @staticmethod
    def _count_rows_for_sessions(
        scoped: SQLStore,
        tenant_id: str,
        tables: tuple[str, ...],
        session_ids: tuple[str, ...],
        *,
        batch_size: int = 256,
    ) -> dict[str, int]:
        counts = dict.fromkeys(tables, 0)
        for offset in range(0, len(session_ids), batch_size):
            batch = session_ids[offset : offset + batch_size]
            placeholders = ",".join(["%s"] * len(batch))
            query = " UNION ALL ".join(
                f"SELECT %s,COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id IN ({placeholders})" for table in tables
            )
            params: list[Any] = []
            for table in tables:
                params.extend((table, tenant_id, *batch))
            for table, count in scoped._rows(query, tuple(params)):
                counts[table] += count
        return counts

    def _verify_content_erased(
        self,
        scoped: SQLStore,
        tenant_id: str,
        session_ids: tuple[str, ...],
    ) -> None:
        counts = self._count_rows_for_sessions(scoped, tenant_id, _CONTENT_ERASE_TABLES, session_ids)
        for table, count in counts.items():
            if count:
                raise LedgerIntegrityError(f"content key/data cleanup left {table} rows")

    def _erase_pending_content(
        self,
        scoped: SQLStore,
        tenant_id: str,
        tombstones: tuple[ForgetTombstone, ...],
    ) -> None:
        completed = tuple(item.session_id for item in tombstones if item.state is TombstoneState.COMPLETE)
        pending = tuple(item.session_id for item in tombstones if item.state is TombstoneState.PENDING)
        if completed:
            counts = self._count_rows_for_sessions(scoped, tenant_id, _CONTENT_ERASE_TABLES, completed)
            for table, count in counts.items():
                if count:
                    raise LedgerIntegrityError(f"complete deletion tombstone outlived {table} content")
        for member in pending:
            with scoped._transaction():
                self._content_store.forget_session(tenant_id, member)

    def _verify_forget_outcome(
        self,
        scoped: SQLStore,
        tenant_id: str,
        root_session_id: str,
        session_ids: tuple[str, ...],
        authority: _Authority,
    ) -> None:
        self._verify_scope_evidence(
            scoped,
            tenant_id,
            root_session_id,
            session_ids,
            require_present=True,
            expected_states={"forgotten"},
            require_complete=True,
            authority=authority,
        )
        self._verify_content_erased(scoped, tenant_id, session_ids)

    def _verify_purge_outcome(
        self,
        scoped: SQLStore,
        tenant_id: str,
        root_session_id: str,
        session_ids: tuple[str, ...],
        authority: _Authority,
    ) -> None:
        self._verify_scope_evidence(
            scoped,
            tenant_id,
            root_session_id,
            session_ids,
            require_present=False,
            expected_states={"purging"},
            require_complete=True,
            authority=authority,
        )
        counts = self._count_rows_for_sessions(scoped, tenant_id, PURGE_DELETE_TABLES, session_ids)
        for table, count in counts.items():
            if count:
                raise LedgerIntegrityError(f"purge left {table} rows for a fenced session")

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
        tombstones = self._verify_scope_evidence(
            scoped,
            tenant_id,
            session_id,
            session_ids,
            require_present=True,
            expected_states={"forgetting", "forgotten"},
            require_complete=False,
            authority=authority,
        )
        self._erase_pending_content(scoped, tenant_id, tombstones)
        scoped.complete_forget(session_id)
        self._verify_forget_outcome(scoped, tenant_id, session_id, session_ids, authority)
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
        tombstones = self._verify_scope_evidence(
            scoped,
            tenant_id,
            session_id,
            session_ids,
            require_present=False,
            expected_states={"purging"},
            require_complete=False,
            authority=authority,
        )
        self._erase_pending_content(scoped, tenant_id, tombstones)
        completed = scoped.complete_purge(session_id)
        if completed != session_ids:
            raise LedgerIntegrityError("purge completion returned a different session scope")
        self._verify_purge_outcome(scoped, tenant_id, session_id, completed, authority)
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
    "PURGE_DELETE_TABLES",
    "SESSION_DELETION_TABLES",
    "DeletionAuthenticity",
    "DeletionReceipt",
    "DeletionSignerUnavailable",
    "SessionDeletionStore",
    "SkeletonSnapshot",
]
