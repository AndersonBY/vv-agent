"""Shared SQL transaction rules for the PostgreSQL reference and single-host SQLite store."""

from __future__ import annotations

import base64
import hmac
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from copy import copy
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Protocol

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch, ForgetTombstone, Principal, TombstoneState, _tombstone_message
from vv_agent.kernel._records import InboxItem, Record, SessionSpec, _MessageCache
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import (
    DEFAULT_TENANT_ID,
    CommitReceipt,
    Conflict,
    ConsumerBatch,
    CreateReceipt,
    InputReceipt,
    Lease,
    LeaseLost,
    LeasePoll,
    ReadPage,
    SequenceConflict,
    SessionStore,
    StoredInput,
    StoredRecord,
    WorkCursor,
    WorkItem,
)

_DRIVE_RUNNABLE_SQL = """
    (s.lease_until_ms IS NULL OR s.lease_until_ms<=t.ms)
    AND s.drive_retry_at_ms<=t.ms
    AND (s.next_drive_ms<=t.ms OR EXISTS (
        SELECT 1 FROM sk_inbox i
        WHERE i.tenant_id=s.tenant_id AND i.session_id=s.session_id
        AND i.consumed_seq IS NULL AND i.available_ms<=t.ms))
"""

_INPUT_ADMISSION_DOMAIN = "vv-agent/S1/input"
_CHAIN_DOMAIN = b"vv-agent/S1"
_CHECKPOINT_DOMAIN = "vv-agent/S1/checkpoint"


class InputAdmissionProofError(Conflict):
    """A signed inbox row no longer matches its original trusted admission."""


class LedgerIntegrityError(Conflict):
    """Stored ledger bytes, chain positions, commit metadata or checkpoints do not agree."""


class UnsupportedFeatureLevel(Conflict):
    """A producer cannot publish work above the database's write-level ceiling."""


class SessionNotLive(Conflict):
    """Normal execution is closed while a session is being forgotten or purged."""


class LedgerSigner(Protocol):
    """The minimum signing capability used by transactional input admission."""

    def bind_key(self) -> Any:
        """Return an immutable binding exposing ``key_id`` and ``sign(message)``."""
        ...

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool: ...


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _natural(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _level(value: Any, name: str) -> int:
    _positive(value, name)
    return value


def _record_level(record: Record) -> int:
    required = _level(record.feature_level, "record feature_level")
    if record.kind == "turn_started":
        requirements = record._payload.get("requirements")
        if isinstance(requirements, dict):
            required = max(required, _level(requirements.get("level", 1), "turn requirement level"))
    elif record.kind == "input_applied":
        input_item = record._payload.get("input")
        if isinstance(input_item, dict):
            required = max(required, _level(input_item.get("feature_level", 1), "input feature_level"))
    return required


def _identity(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("identity must be nonempty text without NUL")


def _tenant(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("tenant_id must be nonempty text without NUL")


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("chain identity exceeds U32 length framing")
    return len(encoded).to_bytes(4, "big") + encoded


def _initial_chain_digest(tenant_id: str, session_id: str) -> str:
    return sha256(_CHAIN_DOMAIN + _length_prefixed(tenant_id) + _length_prefixed(session_id)).hexdigest()


def _digest_bytes(value: str, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise LedgerIntegrityError(f"{name} must be a lowercase SHA-256 digest")
    return bytes.fromhex(value)


def _extend_chain_digest(previous: str, seq: int, record_digest: str) -> str:
    _positive(seq, "chain seq")
    return sha256(
        _digest_bytes(previous, "previous chain digest") + seq.to_bytes(8, "big") + _digest_bytes(record_digest, "record digest")
    ).hexdigest()


def _checkpoint_message(
    tenant_id: str,
    session_id: str,
    seq: int,
    chain_digest: str,
    key_id: str,
    signed_ms: int,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": _CHECKPOINT_DOMAIN,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "seq": seq,
            "chain_digest": chain_digest,
            "key_id": key_id,
            "signed_ms": signed_ms,
        }
    )


def _commit_manifest(
    *,
    head_seq: int,
    head_chain_digest: str,
    records: tuple[tuple[str, str], ...],
    consume_input_ids: tuple[str, ...],
    consumers: tuple[str, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "head_seq": head_seq,
            "head_chain_digest": head_chain_digest,
            "records": [{"record_id": record_id, "digest": digest} for record_id, digest in records],
            "consume_input_ids": list(consume_input_ids),
            "consumers": list(consumers),
        }
    )


def _input_admission_message(
    tenant_id: str,
    session_id: str,
    input_id: str,
    input_seq: int,
    received_ms: int,
    body_digest: str,
    key_id: str,
    signed_ms: int,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": _INPUT_ADMISSION_DOMAIN,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "input_id": input_id,
            "input_seq": input_seq,
            "received_ms": received_ms,
            "body_digest": body_digest,
            "key_id": key_id,
            "signed_ms": signed_ms,
        }
    )


def _input_body(item: InboxItem) -> bytes:
    """Encode the clear input body; proof bytes live in SQL columns."""

    return replace(item, admission_proof=None).encode()


def stored_input_proof(stored: StoredInput) -> dict[str, Any] | None:
    """Build the exact record proof copied from an authenticated inbox row."""

    proof = stored.item.admission_proof
    if proof is None:
        return None
    return {
        "input_seq": stored.input_seq,
        "received_ms": stored.received_ms,
        "key_id": proof["admission_key_id"],
        "signature": proof["admission_signature"],
        "signed_ms": proof["admission_signed_ms"],
    }


@dataclass
class _Prefix:
    tenant_id: str
    session_id: str
    epoch: int
    head_digest: str
    reducer: Fold
    records: tuple[StoredRecord, ...]


@dataclass(frozen=True, slots=True)
class DeletionScopeEntry:
    tenant_id: str
    session_id: str
    parent_session_id: str | None
    head_seq: int
    inbox_seq: int
    deletion_state: str
    depth: int


class SQLStore:
    """One connection per thread. atomic() nests as a savepoint in host transactions."""

    lock_clause = ""
    clock_sql = ""
    _input_admission_configured = False
    _input_admission_allow_unsigned = False
    _input_admission_signer: LedgerSigner | None = None
    _fold_cache: _Prefix | None = None
    _previous_prefix: _Prefix | None = None
    _tenant_id = DEFAULT_TENANT_ID
    _root_store: SQLStore | None = None
    _worker_level = 1

    def __init__(self, *, fold_factory: Callable[[], Fold]):
        self.fold_factory = fold_factory

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def worker_level(self) -> int:
        return self._worker_level

    def configure_worker(self, *, level: int = 1) -> None:
        """Set the maximum feature level this process can execute or project."""
        root = self._root_store or self
        root._worker_level = _level(level, "worker level")

    def for_tenant(self, tenant_id: str) -> SessionStore:
        """Return a lightweight facade whose public operations are tenant-scoped."""
        _tenant(tenant_id)
        return _TenantSQLStore(self._root_store or self, tenant_id)

    def _scoped(self, tenant_id: str) -> SQLStore:
        scoped = copy(self)
        scoped._root_store = self._root_store or self
        scoped._tenant_id = tenant_id
        scoped._fold_cache = None
        scoped._previous_prefix = None
        return scoped

    @property
    def fold_factory(self) -> Callable[[], Fold]:
        return self._fold_factory

    @fold_factory.setter
    def fold_factory(self, factory: Callable[[], Fold]) -> None:
        self._fold_factory = factory
        self._fold_cache = self._previous_prefix = None

    def _cached_prefix(self, session_id: str) -> _Prefix | None:
        cached = self._fold_cache
        if cached is None:
            self._previous_prefix = None
            return None
        if cached.tenant_id == self.tenant_id and cached.session_id == session_id:
            return cached
        previous = self._previous_prefix
        return (
            previous
            if previous is not None and previous.tenant_id == self.tenant_id and previous.session_id == session_id
            else None
        )

    def _retain_prefix(self, prefix: _Prefix) -> None:
        # Two slots keep parent/child projection from evicting each other's validated log.
        if self._fold_cache is None:
            self._previous_prefix = None
        elif (self._fold_cache.tenant_id, self._fold_cache.session_id) != (prefix.tenant_id, prefix.session_id):
            self._previous_prefix = self._fold_cache
        self._fold_cache = prefix

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        raise NotImplementedError

    def _transaction(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def _transaction_id(self) -> object:
        raise NotImplementedError

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...]:
        rows = self._rows(query, params)
        if not rows:
            raise Conflict("session or cursor not found")
        return rows[0]

    def _now(self) -> int:
        return self._one(f"SELECT {self.clock_sql}")[0]

    def configure_input_admission(self, signer: LedgerSigner | None = None, *, allow_unsigned: bool = False) -> None:
        """Select signed inbox admission or the explicit local hash-only mode."""

        if signer is not None and allow_unsigned:
            raise ValueError("input admission cannot be both signed and unsigned")
        if signer is None and not allow_unsigned:
            raise ValueError("input admission requires a signer or explicit allow_unsigned=True")
        if signer is not None and (
            not callable(getattr(signer, "bind_key", None)) or not callable(getattr(signer, "verify", None))
        ):
            raise TypeError("ledger signer must provide bind_key() and verify()")
        root = self._root_store or self
        root._input_admission_configured = True
        root._input_admission_allow_unsigned = allow_unsigned
        root._input_admission_signer = signer

    def _input_admission_policy(self) -> tuple[bool, bool, LedgerSigner | None]:
        root = self._root_store or self
        return (
            bool(root._input_admission_configured),
            bool(root._input_admission_allow_unsigned),
            root._input_admission_signer,
        )

    def _write_ceiling(self) -> int:
        """Read and lock the database-wide write ceiling inside the caller transaction."""
        rows = self._rows(f"SELECT storage_major,write_level FROM sk_settings WHERE singleton=1{self.lock_clause}")
        if len(rows) != 1 or rows[0][0] != "S1":
            raise LedgerIntegrityError("settings do not identify an S1 store")
        return _level(rows[0][1], "write_level")

    def _admit_level(self, *, ceiling: int, required: int, current: int = 1) -> int:
        effective = max(_level(required, "required feature level"), _level(current, "session level"))
        if effective > ceiling:
            raise UnsupportedFeatureLevel(f"feature level {effective} exceeds database write level {ceiling}")
        return effective

    def _input_level(self, item: InboxItem) -> int:
        required = _level(item.feature_level, "input feature_level")
        if item.kind == "child_result":
            child_id = item.payload.get("session_id")
            if not isinstance(child_id, str):
                raise Conflict("child_result requires a child session_id")
            child = self._rows(
                "SELECT level,deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, child_id),
            )
            if len(child) != 1:
                raise Conflict("child_result references a missing child session")
            if child[0][1] != "live":
                raise SessionNotLive("child result session is fenced for deletion")
            required = max(required, _level(child[0][0], "child session level"))
        return required

    def _record_level(self, record: Record) -> int:
        required = _record_level(record)
        if record.kind == "input_applied":
            input_item = record._payload["input"]
            required = max(required, self._input_level(InboxItem(**input_item)))
        return required

    def _pending_input_level(self, session_id: str) -> int:
        required = 1
        rows = self._rows(
            "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
            "admission_signature,admission_signed_ms FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s AND consumed_seq IS NULL ORDER BY input_seq",
            (self.tenant_id, session_id),
        )
        for row in rows:
            required = max(required, self._input_level(self._stored_input(session_id, row).item))
        return required

    def _verify_level_cache(self, session_id: str, records: tuple[StoredRecord, ...], cached_level: int) -> None:
        required = max((self._record_level(stored.record) for stored in records), default=1)
        required = max(required, self._pending_input_level(session_id))
        if _level(cached_level, "session level") < required:
            raise LedgerIntegrityError("session level cache is understated")

    def _bind_checkpoint_signer(self) -> Any | None:
        configured, allow_unsigned, signer = self._input_admission_policy()
        if not configured or allow_unsigned:
            return None
        if signer is None:
            raise LedgerIntegrityError("ledger signer is unavailable")
        bound = signer.bind_key()
        key_id = getattr(bound, "key_id", None)
        if not isinstance(key_id, str) or not key_id or not callable(getattr(bound, "sign", None)):
            raise LedgerIntegrityError("ledger signer returned an invalid key binding")
        return bound

    def _verify_checkpoints(self, session_id: str, chains: tuple[str, ...]) -> None:
        configured, allow_unsigned, signer = self._input_admission_policy()
        signed = configured and not allow_unsigned
        if signed and signer is None:
            raise LedgerIntegrityError("ledger signer is unavailable for checkpoint verification")
        rows = self._rows(
            "SELECT seq,chain_digest,key_id,signature,signed_ms FROM sk_checkpoint "
            "WHERE tenant_id=%s AND session_id=%s AND seq<=%s ORDER BY seq",
            (self.tenant_id, session_id, len(chains)),
        )
        if not signed:
            if rows:
                raise LedgerIntegrityError("ledger signer is required to verify stored checkpoints")
            return
        if len(rows) != len(chains):
            raise LedgerIntegrityError("ledger checkpoint is missing")
        assert signer is not None
        for expected_seq, row in enumerate(rows, 1):
            seq, stored_chain, key_id, signature, signed_ms = row
            if seq != expected_seq or stored_chain != chains[seq - 1]:
                raise LedgerIntegrityError("ledger checkpoint does not match the record chain")
            if not isinstance(key_id, str) or not key_id or not isinstance(signature, bytes):
                raise LedgerIntegrityError("ledger checkpoint has invalid signing material")
            _natural(signed_ms, "checkpoint signed_ms")
            message = _checkpoint_message(self.tenant_id, session_id, seq, stored_chain, key_id, signed_ms)
            if not signer.verify(key_id, message, bytes(signature)):
                raise LedgerIntegrityError("ledger checkpoint signature is invalid")

    def _verify_commits(self, session_id: str, chains: tuple[str, ...]) -> None:
        record_rows = self._rows(
            "SELECT seq,record_id,digest,commit_id,commit_digest,commit_position FROM sk_record "
            "WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
            (self.tenant_id, session_id),
        )
        if len(record_rows) != len(chains):
            raise LedgerIntegrityError("ledger record count does not match the session head")
        records_by_seq = {row[0]: row for row in record_rows}
        records_by_id = {row[1]: row for row in record_rows}
        if len(records_by_seq) != len(record_rows) or len(records_by_id) != len(record_rows):
            raise LedgerIntegrityError("duplicate ledger record identity")

        commits = self._rows(
            "SELECT commit_id,body,digest,record_sequences,head_seq,created_ms FROM sk_commit "
            "WHERE tenant_id=%s AND session_id=%s ORDER BY created_ms,commit_id",
            (self.tenant_id, session_id),
        )
        claimed: dict[int, tuple[str, str, int]] = {}
        for commit_id, body, checksum, sequence_body, stored_head, created_ms in commits:
            _identity(commit_id)
            _natural(stored_head, "commit head_seq")
            _natural(created_ms, "commit created_ms")
            if not isinstance(body, bytes) or sha256(body).hexdigest() != checksum:
                raise LedgerIntegrityError("commit body digest mismatch")
            try:
                manifest = json.loads(body)
                sequences = json.loads(sequence_body)
                canonical = canonical_json_bytes(manifest, "commit body")
                canonical_sequences = canonical_json_bytes(sequences, "commit record_sequences")
            except (TypeError, ValueError) as exc:
                raise LedgerIntegrityError("commit metadata is not canonical JCS") from exc
            if canonical != body or canonical_sequences != sequence_body:
                raise LedgerIntegrityError("commit metadata is not canonical JCS")
            expected_fields = {"head_seq", "head_chain_digest", "records", "consume_input_ids", "consumers"}
            if not isinstance(manifest, dict) or set(manifest) != expected_fields:
                raise LedgerIntegrityError("commit body has an invalid shape")
            head_seq = manifest["head_seq"]
            if type(head_seq) is not int or not 0 <= head_seq <= len(chains):
                raise LedgerIntegrityError("commit head_seq is invalid")
            expected_head_chain = _initial_chain_digest(self.tenant_id, session_id) if head_seq == 0 else chains[head_seq - 1]
            if manifest["head_chain_digest"] != expected_head_chain:
                raise LedgerIntegrityError("commit head_chain_digest does not match the original head")

            records = manifest["records"]
            if not isinstance(records, list):
                raise LedgerIntegrityError("commit records are invalid")
            manifest_records: list[tuple[str, str]] = []
            for item in records:
                if not isinstance(item, dict) or set(item) != {"record_id", "digest"}:
                    raise LedgerIntegrityError("commit record reference is invalid")
                record_id, record_digest = item["record_id"], item["digest"]
                _identity(record_id)
                _digest_bytes(record_digest, "commit record digest")
                manifest_records.append((record_id, record_digest))
            if len({record_id for record_id, _ in manifest_records}) != len(manifest_records):
                raise LedgerIntegrityError("commit repeats a record identity")

            if not isinstance(sequences, list) or len(sequences) != len(manifest_records):
                raise LedgerIntegrityError("commit record sequences do not match its body")
            ordered_sequences: list[tuple[str, int]] = []
            for item in sequences:
                if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                    raise LedgerIntegrityError("commit record sequence is invalid")
                if type(item[1]) is not int or item[1] <= 0:
                    raise LedgerIntegrityError("commit record sequence is invalid")
                ordered_sequences.append((item[0], item[1]))
            if tuple(record_id for record_id, _ in ordered_sequences) != tuple(record_id for record_id, _ in manifest_records):
                raise LedgerIntegrityError("commit record order does not match its body")

            for position, ((record_id, record_digest), (sequence_id, seq)) in enumerate(
                zip(manifest_records, ordered_sequences, strict=True)
            ):
                if record_id != sequence_id:
                    raise LedgerIntegrityError("commit record identity does not match its sequence")
                record = records_by_id.get(record_id)
                if record is None or record[0] != seq or record[2] != record_digest:
                    raise LedgerIntegrityError("commit record does not match the ledger")
                if seq > head_seq:
                    if (record[3], record[4], record[5]) != (commit_id, checksum, position):
                        raise LedgerIntegrityError("commit record metadata does not match the ledger")
                    if seq in claimed:
                        raise LedgerIntegrityError("ledger record belongs to multiple commits")
                    claimed[seq] = (commit_id, checksum, position)
                elif (record[3], record[4]) == (commit_id, checksum):
                    raise LedgerIntegrityError("overlapping commit extends the ledger chain")

            resulting_head = max([head_seq, *(seq for _, seq in ordered_sequences)])
            if stored_head != resulting_head:
                raise LedgerIntegrityError("commit head_seq column does not match its body")

            consume_ids = manifest["consume_input_ids"]
            consumers = manifest["consumers"]
            if not isinstance(consume_ids, list) or not all(isinstance(input_id, str) and input_id for input_id in consume_ids):
                raise LedgerIntegrityError("commit identities are invalid")
            if len(set(consume_ids)) != len(consume_ids):
                raise LedgerIntegrityError("commit repeats a consumed input identity")
            if not isinstance(consumers, list) or not all(isinstance(consumer, str) and consumer for consumer in consumers):
                raise LedgerIntegrityError("commit identities are invalid")
            if consumers != sorted(set(consumers)):
                raise LedgerIntegrityError("commit consumers are not sorted and unique")
            for input_id in consume_ids:
                inbox = self._rows(
                    "SELECT consumed_seq FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                    (self.tenant_id, session_id, input_id),
                )
                if len(inbox) != 1 or type(inbox[0][0]) is not int:
                    raise LedgerIntegrityError("commit consumed input is missing or unconsumed")
                applied = records_by_seq.get(inbox[0][0])
                if applied is None or applied[3] != commit_id:
                    raise LedgerIntegrityError("commit consumed input does not match an applied record")
            for consumer in consumers:
                if not self._rows(
                    "SELECT 1 FROM sk_consumer WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
                    (self.tenant_id, session_id, consumer),
                ):
                    raise LedgerIntegrityError("commit creation consumer is missing")

        if set(claimed) != set(range(1, len(chains) + 1)):
            raise LedgerIntegrityError("ledger sequence is not fully claimed by commits")

    def _stored_input(self, session_id: str, row: tuple[Any, ...]) -> StoredInput:
        """Validate and reconstruct one inbox row before any content hydration."""

        body, seq, received, consumed, checksum, key_id, signature, signed_ms = row
        if sha256(body).hexdigest() != checksum:
            raise Conflict("stored input digest mismatch")
        base = InboxItem.parse(body)
        if base.admission_proof is not None:
            raise InputAdmissionProofError("stored input body duplicates admission proof")
        columns = (key_id, signature, signed_ms)
        if any(value is None for value in columns) and not all(value is None for value in columns):
            raise InputAdmissionProofError("stored input admission proof is incomplete")
        proof = None
        if all(value is not None for value in columns):
            proof = {
                "admission_key_id": key_id,
                "admission_signature": base64.b64encode(bytes(signature)).decode("ascii"),
                "admission_signed_ms": signed_ms,
            }
        item = replace(base, admission_proof=proof)
        item.encode()
        if item.actor is None or item.actor["tenant_id"] != self.tenant_id:
            raise Conflict("input actor belongs to another tenant")

        configured, allow_unsigned, signer = self._input_admission_policy()
        if configured:
            if allow_unsigned:
                if proof is not None:
                    raise InputAdmissionProofError("unsigned input admission must not contain a proof")
            else:
                if signer is None or proof is None:
                    raise InputAdmissionProofError("signed input admission proof is missing")
                message = _input_admission_message(
                    self.tenant_id,
                    session_id,
                    item.input_id,
                    seq,
                    received,
                    checksum,
                    proof["admission_key_id"],
                    proof["admission_signed_ms"],
                )
                if not signer.verify(proof["admission_key_id"], message, bytes(signature)):
                    raise InputAdmissionProofError("signed input admission signature is invalid")
        return StoredInput(item, seq, received, consumed)

    @contextmanager
    def _bound_append(self, lease: Lease) -> Iterator[None]:
        """Keep final-write limits inside the append transaction."""
        with self._transaction():
            yield

    def list_sessions(self, *, limit: int = 100, after: str | None = None) -> tuple[str, ...]:
        """Read-only catalog, including idle sessions absent from list_runnable."""
        _positive(limit, "limit")
        if after is not None:
            _identity(after)
        return tuple(
            row[0]
            for row in self._rows(
                "SELECT session_id FROM sk_session WHERE tenant_id=%s AND deletion_state='live' "
                "AND (%s OR session_id > %s) ORDER BY session_id LIMIT %s",
                (self.tenant_id, after is None, after or "", limit),
            )
        )

    @contextmanager
    def atomic(self) -> Iterator[SQLSessionTx]:
        with self._transaction():
            yield SQLSessionTx(self)

    def _lock(self, session_id: str) -> tuple[Any, ...]:
        return self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms,level,deletion_state "
            f"FROM sk_session WHERE tenant_id=%s AND session_id=%s{self.lock_clause}",
            (self.tenant_id, session_id),
        )

    @staticmethod
    def _require_live_row(row: tuple[Any, ...]) -> None:
        if len(row) < 8 or row[7] != "live":
            raise SessionNotLive("session is fenced for deletion")

    def _require_live(self, session_id: str, *, for_update: bool = False) -> tuple[Any, ...]:
        row = self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms,level,deletion_state "
            f"FROM sk_session WHERE tenant_id=%s AND session_id=%s"
            f"{self.lock_clause if for_update else ''}",
            (self.tenant_id, session_id),
        )
        self._require_live_row(row)
        return row

    def _check_lease(self, session_id: str, lease: Lease, row: tuple[Any, ...], now: int) -> None:
        self._require_live_row(row)
        if (
            lease.tenant_id != self.tenant_id
            or lease.session_id != session_id
            or row[2] != lease.epoch
            or row[3] != lease.owner
            or row[4] is None
            or row[4] <= now
            or lease.expires_at_ms <= now
        ):
            raise LeaseLost("lease expired or fenced")

    def _deletion_scope(self, root_session_id: str) -> tuple[DeletionScopeEntry, ...]:
        """Lock the current root/descendant inventory in deterministic depth order."""
        rows = self._rows(
            """
            WITH RECURSIVE deletion_scope(session_id,depth) AS (
                SELECT session_id,0 FROM sk_session WHERE tenant_id=%s AND session_id=%s
                UNION ALL
                SELECT child.session_id,deletion_scope.depth+1
                FROM sk_session child
                JOIN deletion_scope ON child.tenant_id=%s
                    AND child.parent_session_id=deletion_scope.session_id
            )
            SELECT s.tenant_id,s.session_id,s.parent_session_id,s.head_seq,s.inbox_seq,s.deletion_state,
                   deletion_scope.depth
            FROM sk_session s
            JOIN deletion_scope ON deletion_scope.session_id=s.session_id
            WHERE s.tenant_id=%s
            ORDER BY deletion_scope.depth DESC,s.session_id
            """
            + self.lock_clause,
            (self.tenant_id, root_session_id, self.tenant_id, self.tenant_id),
        )
        return tuple(DeletionScopeEntry(*row) for row in rows)

    def _stored(self, session_id: str, row: tuple[Any, ...], messages: _MessageCache | None = None) -> StoredRecord:
        body, seq, commit, epoch, created, checksum = row
        if sha256(body).hexdigest() != checksum:
            raise Conflict("stored record digest mismatch")
        cached = self._cached_prefix(session_id)
        if cached is not None and 0 < seq <= len(cached.records):
            prior = cached.records[seq - 1].record
            if cached.reducer.seen[prior.record_id] == body:
                # The same session/sequence and exact validated bytes can share the immutable record.
                return StoredRecord(prior, seq, commit, epoch, created)
        record = Record.parse(body) if messages is None else Record._parse(body, messages)
        if record.encode() != body:
            raise LedgerIntegrityError("stored record body is not canonical JCS")
        return StoredRecord(record, seq, commit, epoch, created)

    def _verified_records(self, session_id: str, through: int) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
        rows = self._rows(
            "SELECT body,seq,commit_id,writer_epoch,created_ms,digest,chain_digest,record_id,"
            "schema_version,kind,turn_id,operation_id,attempt,commit_position "
            "FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq<=%s ORDER BY seq",
            (self.tenant_id, session_id, through),
        )
        if len(rows) != through:
            raise LedgerIntegrityError("log sequence gap")
        previous = _initial_chain_digest(self.tenant_id, session_id)
        chains: list[str] = []
        stored_rows: list[tuple[Any, ...]] = []
        for expected_seq, row in enumerate(rows, 1):
            (
                body,
                seq,
                commit_id,
                epoch,
                created_ms,
                checksum,
                stored_chain,
                record_id,
                schema_version,
                kind,
                turn_id,
                operation_id,
                attempt,
                commit_position,
            ) = row
            if seq != expected_seq:
                raise LedgerIntegrityError("log sequence gap or reorder")
            actual_digest = sha256(body).hexdigest()
            if actual_digest != checksum:
                raise LedgerIntegrityError("stored record digest mismatch")
            try:
                value = json.loads(body)
                if canonical_json_bytes(value, "stored record body") != body:
                    raise LedgerIntegrityError("stored record body is not canonical JCS")
            except (TypeError, ValueError) as exc:
                raise LedgerIntegrityError("stored record body is not canonical JCS") from exc
            if not isinstance(value, dict) or (
                value.get("record_id"),
                value.get("session_id"),
                value.get("schema_version"),
                value.get("kind"),
                value.get("turn_id"),
                value.get("operation_id"),
                value.get("attempt"),
            ) != (
                record_id,
                session_id,
                schema_version,
                kind,
                turn_id,
                operation_id,
                attempt,
            ):
                raise LedgerIntegrityError("stored record SQL identity does not match its body")
            _natural(epoch, "record writer_epoch")
            _natural(created_ms, "record created_ms")
            _natural(commit_position, "record commit_position")
            actual_chain = _extend_chain_digest(previous, expected_seq, actual_digest)
            if stored_chain != actual_chain:
                raise LedgerIntegrityError("stored record chain digest mismatch")
            previous = actual_chain
            chains.append(actual_chain)
            stored_rows.append((body, seq, commit_id, epoch, created_ms, checksum))
        self._verify_checkpoints(session_id, tuple(chains))
        self._verify_commits(session_id, tuple(chains))
        return stored_rows, tuple(chains)

    def _prefix(self, session_id: str, row: tuple[Any, ...]) -> _Prefix:
        head, epoch = row[0], row[2]
        cached_level = row[3] if len(row) == 4 else row[6]
        if head == 0:
            raise LedgerIntegrityError("log sequence gap")
        stored_rows, chains = self._verified_records(session_id, head)
        cached = self._cached_prefix(session_id)
        if cached is not None:
            valid = (
                (cached.tenant_id, cached.session_id) == (self.tenant_id, session_id)
                and len(cached.records) <= head
                and bool(cached.records)
            )
            if valid:
                valid = cached.head_digest == chains[len(cached.records) - 1]
            if not valid:
                self._fold_cache = cached = None
                self._previous_prefix = None
            elif cached.epoch != epoch:
                # The immutable prefix survives a lease change; execution authority does not.
                cached = _Prefix(self.tenant_id, session_id, epoch, cached.head_digest, cached.reducer, cached.records)
        through = len(cached.records) if cached else 0
        if cached is not None and through == head:
            self._verify_level_cache(session_id, cached.records, cached_level)
            self._retain_prefix(cached)
            return cached
        records, bodies = [], []
        reducer = cached.reducer.fork() if cached else self.fold_factory()
        head_digest = chains[-1]
        cursor = through
        while cursor < head:
            rows = stored_rows[cursor : min(head, cursor + 256)]
            for row_data in rows:
                if row_data[1] != cursor + 1:
                    raise LedgerIntegrityError("log sequence gap or reorder")
                stored = self._stored(session_id, row_data, reducer._messages)
                stored.record._share_messages(reducer._messages)
                records.append(stored)
                bodies.append(row_data[0])
                cursor = stored.seq
        consumed = []
        for input_row in self._rows(
            "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,admission_signature,admission_signed_ms "
            "FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_seq<=%s "
            "AND consumed_seq>%s AND consumed_seq<=%s ORDER BY input_seq",
            (self.tenant_id, session_id, row[1], through, head),
        ):
            consumed.append(self._stored_input(session_id, input_row).item)
        reducer.extend((r.record for r in records), consumed_inputs=consumed, bodies=bodies)
        prefix = _Prefix(
            self.tenant_id,
            session_id,
            epoch,
            head_digest,
            reducer,
            (cached.records if cached else ()) + tuple(records),
        )
        self._verify_level_cache(session_id, prefix.records, cached_level)
        self._retain_prefix(prefix)
        return prefix

    def read_state(self, session_id: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
        # Preserve the public read boundary used by host snapshot/reconnect hooks.
        self.read(session_id, limit=1)
        # One statement captures a committed head; immutable prefix reads need no row lock.
        row = self._head(session_id)
        prefix = self._prefix(session_id, row)
        return prefix.reducer.snapshot(), prefix.records, row[1]

    def _head(self, session_id: str) -> tuple[Any, ...]:
        return self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,level FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        )

    def _chain_anchor(self, session_id: str, seq: int) -> str:
        if seq == 0:
            return _initial_chain_digest(self.tenant_id, session_id)
        return self._one(
            "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
            (self.tenant_id, session_id, seq),
        )[0]

    def _schedule(self, session_id: str, state: ExecutionState, *, preserve_due: bool = False) -> None:
        self._rows(
            "UPDATE sk_session SET phase=%s,next_drive_ms=CASE WHEN %s THEN next_drive_ms ELSE %s END,"
            "active_turn_id=%s,terminal_seq=%s WHERE tenant_id=%s AND session_id=%s",
            (
                state.phase,
                preserve_due,
                state.next_drive_ms,
                state.active_turn_id,
                state.terminal_seq,
                self.tenant_id,
                session_id,
            ),
        )

    def rebuild_schedule(self, session_id: str) -> ExecutionState:
        self._require_live(session_id)
        while True:
            captured = self._head(session_id)
            self._fold_cache = None
            reducer = self._prefix(session_id, captured).reducer
            snapshot = reducer.snapshot()
            with self._transaction():
                row = self._lock(session_id)
                self._require_live_row(row)
                if row[:3] != captured[:3]:
                    continue
                self._schedule(session_id, reducer.state)
                return snapshot

    def read(self, session_id: str, *, after_seq: int = 0, through_seq: int | None = None, limit: int = 1024) -> ReadPage:
        _positive(limit, "limit")
        _natural(after_seq, "after_seq")
        if through_seq is not None:
            _natural(through_seq, "through_seq")
        self._require_live(session_id)
        # Records are immutable: cap by the captured head without locking writers.
        head, inbox = self._one(
            "SELECT head_seq,inbox_seq FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        )
        verified, _chains = self._verified_records(session_id, head)
        boundary = min(head, through_seq) if through_seq is not None else head
        rows = [row for row in verified if after_seq < row[1] <= boundary][:limit]
        return ReadPage(head, inbox, tuple(self._stored(session_id, row) for row in rows))

    def lease_status(self, session_id: str) -> tuple[int, int, int]:
        self._require_live(session_id)
        head, epoch, until, now = self._one(
            f"SELECT head_seq,lease_epoch,lease_until_ms,{self.clock_sql} FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        )
        return head, epoch, max(0, (until or now) - now)

    def read_input(self, session_id: str, input_id: str) -> StoredInput | None:
        self._require_live(session_id)
        rows = self._rows(
            "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,admission_signature,admission_signed_ms "
            "FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            (self.tenant_id, session_id, input_id),
        )
        if not rows:
            return None
        return self._stored_input(session_id, rows[0])

    def control_records(self, session_id: str, *, through_seq: int) -> tuple[StoredRecord, ...]:
        """Read consumed control facts for catalog filtering without folding turns."""
        _natural(through_seq, "through_seq")
        self._require_live(session_id)
        rows = self._rows(
            "SELECT r.body,r.seq,r.commit_id,r.writer_epoch,r.created_ms,r.digest "
            "FROM sk_inbox i JOIN sk_record r ON r.tenant_id=i.tenant_id "
            "AND r.session_id=i.session_id AND r.seq=i.consumed_seq "
            "WHERE i.tenant_id=%s AND i.session_id=%s AND i.kind='control' "
            "AND r.seq<=%s ORDER BY r.seq",
            (self.tenant_id, session_id, through_seq),
        )
        return tuple(self._stored(session_id, row) for row in rows)

    def projection_context(self, session_id: str, records: tuple[StoredRecord, ...]) -> tuple[StoredRecord, ...]:
        """Indexed dependencies for a bounded event page, without folding session history."""
        self._require_live(session_id)
        found = {r.seq: r for r in records}
        through = max(found, default=0)
        turns = {r.record.turn_id for r in records if r.record.turn_id}
        seen = set()

        def operation(oid, attempt):
            if (oid, attempt) in seen:
                return
            seen.add((oid, attempt))
            rows = self._rows(
                "SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record "
                "WHERE tenant_id=%s AND session_id=%s AND operation_id=%s AND attempt=%s AND seq<=%s "
                "AND kind IN ('op_planned','op_prepared','op_parked') ORDER BY seq",
                (self.tenant_id, session_id, oid, attempt, through),
            )
            for row in rows:
                stored = self._stored(session_id, row)
                found[stored.seq] = stored
                if stored.record.kind == "op_planned" and stored.record._payload["op_kind"] != "model":
                    operation(stored.record._payload["dependencies"][0], 1)

        for r in records:
            if r.record.operation_id:
                operation(r.record.operation_id, r.record.attempt)
            elif r.record.kind == "input_applied":
                data = r.record._payload
                item = data["input"]
                oid = data["target_operation_id"] or item["payload"].get("operation_id")
                if oid:
                    operation(oid, item["payload"].get("attempt", 1))
        for tid in turns:
            rows = self._rows(
                "SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record "
                "WHERE tenant_id=%s AND session_id=%s AND turn_id=%s AND kind='turn_started' "
                "AND seq<=%s ORDER BY seq LIMIT 1",
                (self.tenant_id, session_id, tid, through),
            )
            for row in rows:
                stored = self._stored(session_id, row)
                found[stored.seq] = stored
        for r in records:
            if r.record.kind != "turn_ended":
                continue
            # Read only this turn's indexed intent and lifecycle dependencies.
            rows = self._rows(
                "SELECT r.body,r.seq,r.commit_id,r.writer_epoch,r.created_ms,r.digest FROM sk_record r "
                "LEFT JOIN sk_inbox i ON i.tenant_id=r.tenant_id "
                "AND i.session_id=r.session_id AND i.consumed_seq=r.seq "
                "WHERE r.tenant_id=%s AND r.session_id=%s AND r.turn_id=%s AND r.seq<=%s AND "
                "(r.kind='boundary_recorded' OR (r.kind='input_applied' AND i.input_id LIKE 'compaction/%%')) ORDER BY r.seq",
                (self.tenant_id, session_id, r.record.turn_id, through),
            )
            for row in rows:
                stored = self._stored(session_id, row)
                found[stored.seq] = stored
            start = next(
                row.record
                for row in found.values()
                if row.record.kind == "turn_started" and row.record.turn_id == r.record.turn_id
            )
            for key in start._payload["input_ids"]:
                if key.startswith("compaction/"):
                    item = self.read_input(session_id, key)
                    if item is not None and item.consumed_seq is not None and item.consumed_seq <= through:
                        row = self.read(
                            session_id, after_seq=item.consumed_seq - 1, through_seq=item.consumed_seq, limit=1
                        ).records[0]
                        found[row.seq] = row
        return tuple(found[k] for k in sorted(found))

    def peek_inbox(
        self, session_id: str, *, through_input_seq: int | None = None, limit: int = 256, ready_only: bool = True
    ) -> tuple[StoredInput, ...]:
        _positive(limit, "limit")
        if through_input_seq is not None:
            _natural(through_input_seq, "through_input_seq")
        self._require_live(session_id)
        readiness = f"AND available_ms<={self.clock_sql} " if ready_only else ""
        rows = self._rows(
            "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,admission_signature,admission_signed_ms "
            "FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s "
            "AND consumed_seq IS NULL " + readiness + "AND (%s OR input_seq<=%s) ORDER BY input_seq LIMIT %s",
            (self.tenant_id, session_id, through_input_seq is None, through_input_seq or 0, limit),
        )
        return tuple(self._stored_input(session_id, row) for row in rows)

    def acquire(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None:
        _identity(owner)
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(session_id)
            self._require_live_row(row)
            now = self._now()  # Must be read after the row lock, not at transaction start.
            if row[6] > self._worker_level or (row[4] is not None and row[4] > now) or row[5] > now:
                return None
            lease = Lease(session_id, owner, row[2] + 1, now + ttl_ms, self.tenant_id)
            self._rows(
                "UPDATE sk_session SET lease_epoch=%s,lease_owner=%s,lease_until_ms=%s WHERE tenant_id=%s AND session_id=%s",
                (lease.epoch, owner, lease.expires_at_ms, self.tenant_id, session_id),
            )
            return lease

    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll:
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(lease.session_id)
            now = self._now()
            self._check_lease(lease.session_id, lease, row, now)
            renewed = Lease(lease.session_id, lease.owner, lease.epoch, now + ttl_ms, lease.tenant_id)
            self._rows(
                "UPDATE sk_session SET lease_until_ms=%s WHERE tenant_id=%s AND session_id=%s",
                (renewed.expires_at_ms, self.tenant_id, lease.session_id),
            )
            controls = tuple(
                self._stored_input(lease.session_id, r).item
                for r in self._rows(
                    "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
                    "admission_signature,admission_signed_ms "
                    "FROM sk_inbox WHERE tenant_id=%s AND session_id=%s "
                    "AND consumed_seq IS NULL AND available_ms<=%s AND kind='control' ORDER BY input_seq",
                    (self.tenant_id, lease.session_id, now),
                )
            )
            return LeasePoll(renewed, now, row[1], controls)

    def release(self, lease: Lease) -> bool:
        with self._transaction():
            row = self._lock(lease.session_id)
            try:
                self._check_lease(lease.session_id, lease, row, self._now())
            except LeaseLost:
                return False
            self._rows(
                "UPDATE sk_session SET lease_owner=NULL,lease_until_ms=NULL WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, lease.session_id),
            )
            return True

    def next_drive_delay_ms(self, session_id: str, *, lease: Lease | None = None) -> int | None:
        """Read the SQL schedule using database time; provider queries require the held lease."""
        with self._transaction():
            row = self._lock(session_id)
            self._require_live_row(row)
            now = self._now()
            if lease is not None:
                self._check_lease(session_id, lease, row, now)
            due = self._one(
                "SELECT next_drive_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, session_id),
            )[0]
            return None if due is None else max(0, due - now)

    def defer_idle_drive(self, lease: Lease, *, poll_ms: int) -> bool:
        """Defer a stale idle schedule without changing execution facts or future due work."""
        return self._defer_idle_drive(lease, poll_ms=poll_ms)

    def _defer_idle_drive(self, lease: Lease, *, poll_ms: int, lease_guard: Callable[[], Lease] | None = None) -> bool:
        _positive(poll_ms, "poll_ms")
        sid = lease.session_id
        self._require_live(sid)
        captured = self._head(sid)
        state = self._prefix(sid, captured).reducer.state
        with self._transaction():
            if lease_guard is not None:
                lease = lease_guard()
            row = self._lock(sid)
            now = self._now()
            self._check_lease(sid, lease, row, now)
            scheduled = self._one(
                "SELECT next_drive_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, sid),
            )[0]
            if scheduled is None or scheduled > now:
                return False
            if row[:3] != captured:
                raise SequenceConflict("head or inbox watermark changed")
            due = [now + poll_ms, *(t for t in state.due_ms if t > now)]
            available = self._one(
                "SELECT min(available_ms) FROM sk_inbox WHERE tenant_id=%s AND session_id=%s "
                "AND consumed_seq IS NULL AND available_ms>%s",
                (self.tenant_id, sid, now),
            )[0]
            if available is not None:
                due.append(available)
            return bool(
                self._rows(
                    "UPDATE sk_session SET next_drive_ms=%s "
                    "WHERE tenant_id=%s AND session_id=%s AND next_drive_ms<=%s RETURNING session_id",
                    (min(due), self.tenant_id, sid, now),
                )
            )

    def defer_drive(self, lease: Lease, *, retry_after_ms: int) -> None:
        """Fence failure/readiness retry independently of due work and ready inbox."""
        _positive(retry_after_ms, "retry_after_ms")
        with self._transaction():
            row = self._lock(lease.session_id)
            now = self._now()
            self._check_lease(lease.session_id, lease, row, now)
            self._rows(
                "UPDATE sk_session SET drive_retry_at_ms=CASE WHEN drive_retry_at_ms>%s "
                "THEN drive_retry_at_ms ELSE %s END WHERE tenant_id=%s AND session_id=%s",
                (now + retry_after_ms, now + retry_after_ms, self.tenant_id, lease.session_id),
            )

    def defer_projection(self, session_id: str, consumer: str, *, retry_after_ms: int) -> None:
        """Back off one consumer without claiming or delaying execution."""
        _positive(retry_after_ms, "retry_after_ms")
        with self._transaction():
            self._require_live(session_id)
            self._one(
                f"SELECT last_seq FROM sk_consumer WHERE tenant_id=%s AND session_id=%s AND consumer=%s{self.lock_clause}",
                (self.tenant_id, session_id, consumer),
            )
            until = self._now() + retry_after_ms
            self._rows(
                "UPDATE sk_consumer SET project_retry_at_ms=CASE WHEN project_retry_at_ms>%s "
                "THEN project_retry_at_ms ELSE %s END "
                "WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
                (until, until, self.tenant_id, session_id, consumer),
            )

    def is_runnable(self, session_id: str) -> bool:
        # Check after release in a fresh transaction so a pusher racing the holder is visible.
        with self._transaction():
            return bool(
                self._rows(
                    f"WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms) "
                    "SELECT 1 FROM sk_session s CROSS JOIN clock t "
                    f"WHERE s.tenant_id=%s AND s.session_id=%s AND s.schema_version='S1' "
                    f"AND s.deletion_state='live' AND s.level<=%s AND {_DRIVE_RUNNABLE_SQL}",
                    (self.tenant_id, session_id, self._worker_level),
                )
            )

    def list_runnable(self, *, limit: int = 100, after: WorkCursor | None = None) -> tuple[WorkItem, ...]:
        _positive(limit, "limit")
        cursor = (after.session_id, after.kind, after.consumer) if after else ("", "", "")
        rows = self._rows(
            f"""
            WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms), driving AS (
                SELECT s.session_id,'drive' AS work_kind,'' AS consumer
                FROM sk_session s CROSS JOIN clock t
                WHERE s.tenant_id=%s AND s.session_id >= %s
                  AND (s.session_id,'drive','')>(%s,%s,%s)
                  AND s.schema_version='S1' AND s.deletion_state='live' AND s.level<=%s
                  AND {_DRIVE_RUNNABLE_SQL}
                ORDER BY s.session_id LIMIT %s
            ), projecting AS (
                SELECT c.session_id,'project' AS work_kind,c.consumer FROM sk_consumer c
                JOIN sk_session s ON s.tenant_id=c.tenant_id AND s.session_id=c.session_id
                CROSS JOIN clock t
                WHERE c.tenant_id=%s AND c.session_id >= %s
                  AND (c.session_id,'project',c.consumer)>(%s,%s,%s)
                  AND s.schema_version='S1' AND s.deletion_state='live' AND s.level<=%s
                  AND c.last_seq<s.head_seq AND c.project_retry_at_ms<=t.ms
                ORDER BY c.session_id,c.consumer LIMIT %s
            ), work AS (
                SELECT * FROM driving UNION ALL SELECT * FROM projecting
            ) SELECT session_id,work_kind,consumer FROM work
            ORDER BY session_id,work_kind,consumer LIMIT %s
            """,
            (
                self.tenant_id,
                cursor[0],
                *cursor,
                self._worker_level,
                limit,
                self.tenant_id,
                cursor[0],
                *cursor,
                self._worker_level,
                limit,
                limit,
            ),
        )
        return tuple(WorkItem(*row) for row in rows)

    @staticmethod
    def _verify_tombstone_object(tombstone: ForgetTombstone, verifier: Any | None, *, allow_unsigned: bool) -> None:
        message = _tombstone_message(tombstone)
        signature = base64.b64decode(tombstone.signature, validate=True)
        if allow_unsigned:
            expected = sha256(message).digest()
            if tombstone.key_id != "hash-only/sha256" or not hmac.compare_digest(signature, expected):
                raise LedgerIntegrityError("hash-only tombstone does not match its canonical message")
            return
        if verifier is None or not callable(getattr(verifier, "verify", None)):
            raise SessionNotLive("a ledger signer is required to verify the deletion tombstone")
        if not verifier.verify(tombstone.key_id, message, signature):
            raise LedgerIntegrityError("deletion tombstone signature is invalid")

    def _tombstone(self, session_id: str, *, for_update: bool = False) -> ForgetTombstone | None:
        row = self._rows(
            "SELECT final_seq,chain_digest,deleted_ms,actor,key_id,signature,state,scope_root_session_id "
            "FROM sk_tombstone WHERE tenant_id=%s AND session_id=%s" + (self.lock_clause if for_update else ""),
            (self.tenant_id, session_id),
        )
        if not row:
            return None
        final_seq, chain_digest, deleted_ms, actor_body, key_id, signature, state, scope_root = row[0]
        try:
            actor_value = json.loads(actor_body)
            if canonical_json_bytes(actor_value, "tombstone actor") != actor_body:
                raise ValueError("actor is not canonical JCS")
            return ForgetTombstone.from_dict(
                {
                    "tenant_id": self.tenant_id,
                    "session_id": session_id,
                    "final_seq": final_seq,
                    "chain_digest": chain_digest,
                    "deleted_ms": deleted_ms,
                    "actor": actor_value,
                    "key_id": key_id,
                    "signature": base64.b64encode(bytes(signature)).decode("ascii"),
                    "state": state,
                    "scope_root_session_id": scope_root,
                }
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError("stored deletion tombstone is invalid") from exc

    def _write_pending_tombstone(
        self,
        entry: DeletionScopeEntry,
        actor: Principal,
        *,
        scope_root_session_id: str,
        bound_signer: Any | None,
        verifier: Any | None,
        allow_unsigned: bool,
        now: int,
    ) -> ForgetTombstone:
        existing = self._tombstone(entry.session_id, for_update=True)
        chain_digest = self._chain_anchor(entry.session_id, entry.head_seq)
        if entry.head_seq < 1:
            raise LedgerIntegrityError("deletion tombstone requires a retained record head")
        if existing is not None:
            if (
                existing.final_seq,
                existing.chain_digest,
                existing.actor,
                existing.state,
                existing.scope_root_session_id,
            ) != (entry.head_seq, chain_digest, actor, TombstoneState.PENDING, scope_root_session_id):
                raise LedgerIntegrityError("existing deletion tombstone does not match the fenced scope")
            self._verify_tombstone_object(existing, verifier, allow_unsigned=allow_unsigned)
            return existing

        if allow_unsigned:
            key_id = "hash-only/sha256"
        else:
            if bound_signer is None:
                raise SessionNotLive("a bound ledger signer is required before deletion fencing")
            key_id = getattr(bound_signer, "key_id", None)
            if not isinstance(key_id, str) or not key_id or not callable(getattr(bound_signer, "sign", None)):
                raise LedgerIntegrityError("ledger signer returned an invalid tombstone binding")
        unsigned = ForgetTombstone(
            tenant_id=self.tenant_id,
            session_id=entry.session_id,
            final_seq=entry.head_seq,
            chain_digest=chain_digest,
            deleted_ms=now,
            actor=actor,
            key_id=key_id,
            signature="AA==",
            state=TombstoneState.PENDING,
            scope_root_session_id=scope_root_session_id,
        )
        message = _tombstone_message(unsigned)
        signature = sha256(message).digest() if allow_unsigned else bound_signer.sign(message)
        if not isinstance(signature, bytes) or not signature:
            raise LedgerIntegrityError("ledger signer returned an invalid tombstone signature")
        tombstone = replace(
            unsigned,
            signature=base64.b64encode(signature).decode("ascii"),
        )
        self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
        self._rows(
            "INSERT INTO sk_tombstone(tenant_id,session_id,final_seq,chain_digest,deleted_ms,actor,key_id,signature,"
            "state,scope_root_session_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s)",
            (
                self.tenant_id,
                entry.session_id,
                entry.head_seq,
                chain_digest,
                now,
                canonical_json_bytes(actor.to_dict(), "tombstone actor"),
                tombstone.key_id,
                signature,
                scope_root_session_id,
            ),
        )
        return tombstone

    def begin_forget(
        self,
        session_id: str,
        actor: Principal,
        *,
        bound_signer: Any | None,
        verifier: Any | None,
        allow_unsigned: bool,
    ) -> tuple[str, ...]:
        """Fence a forget scope and durably bind its pending tombstone."""
        if actor.tenant_id != self.tenant_id:
            raise ContentBindingMismatch("deletion actor belongs to another tenant")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            if not scope:
                raise Conflict("session not found in tenant")
            root = next(entry for entry in scope if entry.session_id == session_id)
            if root.deletion_state == "purging":
                raise SessionNotLive("session purge is already in progress")
            if root.deletion_state == "forgotten":
                tombstone = self._tombstone(session_id, for_update=True)
                if tombstone is None or tombstone.state is not TombstoneState.COMPLETE:
                    raise LedgerIntegrityError("forgotten session is missing its complete tombstone")
                self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
                return (session_id,)
            now = self._now()
            if root.deletion_state == "live":
                updated = self._rows(
                    "UPDATE sk_session SET deletion_state='forgetting',lease_epoch=lease_epoch+1,lease_owner=NULL,"
                    "lease_until_ms=NULL,next_drive_ms=NULL,drive_retry_at_ms=0 "
                    "WHERE tenant_id=%s AND session_id=%s AND deletion_state='live' RETURNING session_id",
                    (self.tenant_id, session_id),
                )
                if len(updated) != 1:
                    raise Conflict("session deletion fence changed concurrently")
            self._write_pending_tombstone(
                root,
                actor,
                scope_root_session_id=session_id,
                bound_signer=bound_signer,
                verifier=verifier,
                allow_unsigned=allow_unsigned,
                now=now,
            )
            return (session_id,)

    def complete_forget(self, session_id: str) -> None:
        """Publish the complete tombstone after content/key erasure succeeds."""
        with self._transaction():
            scope = self._deletion_scope(session_id)
            if not scope:
                raise Conflict("session not found in tenant")
            root = next(entry for entry in scope if entry.session_id == session_id)
            if root.deletion_state == "forgotten":
                return
            if root.deletion_state != "forgetting":
                raise SessionNotLive("session is not in the forgetting state")
            tombstone = self._tombstone(session_id, for_update=True)
            if tombstone is None or tombstone.state is not TombstoneState.PENDING:
                raise LedgerIntegrityError("pending forget tombstone is missing")
            self._rows(
                "UPDATE sk_tombstone SET state='complete' WHERE tenant_id=%s AND session_id=%s AND state='pending'",
                (self.tenant_id, session_id),
            )
            self._rows(
                "UPDATE sk_session SET deletion_state='forgotten',lease_owner=NULL,lease_until_ms=NULL,next_drive_ms=NULL "
                "WHERE tenant_id=%s AND session_id=%s AND deletion_state='forgetting'",
                (self.tenant_id, session_id),
            )

    def begin_purge(
        self,
        session_id: str,
        actor: Principal,
        *,
        bound_signer: Any | None,
        verifier: Any | None,
        allow_unsigned: bool,
    ) -> tuple[str, ...]:
        """Freeze root/descendants, persist pending tombstones, and start purge."""
        if actor.tenant_id != self.tenant_id:
            raise ContentBindingMismatch("deletion actor belongs to another tenant")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            now = self._now()
            if not scope:
                tombstones = self._rows(
                    "SELECT session_id FROM sk_tombstone WHERE tenant_id=%s AND scope_root_session_id=%s ORDER BY session_id",
                    (self.tenant_id, session_id),
                )
                if not tombstones:
                    raise Conflict("session not found in tenant")
                for (member,) in tombstones:
                    tombstone = self._tombstone(member, for_update=True)
                    if tombstone is None or tombstone.state is not TombstoneState.COMPLETE:
                        raise LedgerIntegrityError("incomplete purge tombstone outlives its session")
                    self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
                return tuple(sorted(row[0] for row in tombstones))
            for entry in scope:
                self._rows(
                    "UPDATE sk_session SET deletion_state='purging',lease_epoch=lease_epoch+1,lease_owner=NULL,"
                    "lease_until_ms=NULL,next_drive_ms=NULL,drive_retry_at_ms=0 "
                    "WHERE tenant_id=%s AND session_id=%s AND deletion_state!='purging'",
                    (self.tenant_id, entry.session_id),
                )
                existing = self._tombstone(entry.session_id, for_update=True)
                if existing is not None and existing.state is TombstoneState.COMPLETE:
                    if (
                        existing.final_seq != entry.head_seq
                        or existing.scope_root_session_id != session_id
                        or existing.actor != actor
                    ):
                        raise LedgerIntegrityError("completed tombstone does not match the purge scope")
                    self._verify_tombstone_object(existing, verifier, allow_unsigned=allow_unsigned)
                    continue
                self._write_pending_tombstone(
                    entry,
                    actor,
                    scope_root_session_id=session_id,
                    bound_signer=bound_signer,
                    verifier=verifier,
                    allow_unsigned=allow_unsigned,
                    now=now,
                )
            return tuple(entry.session_id for entry in scope)

    def complete_purge(self, session_id: str) -> tuple[str, ...]:
        """Delete a fenced scope child-first and retain only complete tombstones."""
        with self._transaction():
            scope = self._deletion_scope(session_id)
            if not scope:
                rows = self._rows(
                    "SELECT session_id FROM sk_tombstone WHERE tenant_id=%s AND scope_root_session_id=%s "
                    "AND state='complete' ORDER BY session_id",
                    (self.tenant_id, session_id),
                )
                if not rows:
                    raise Conflict("purge scope has no retained complete tombstones")
                return tuple(row[0] for row in rows)
            if any(entry.deletion_state != "purging" for entry in scope):
                raise SessionNotLive("purge scope is not fully fenced")
            for entry in scope:
                tombstone = self._tombstone(entry.session_id, for_update=True)
                if tombstone is None or tombstone.state not in {TombstoneState.PENDING, TombstoneState.COMPLETE}:
                    raise LedgerIntegrityError("purge tombstone is missing or incomplete")
            tables = (
                "sk_content_link",
                "sk_content",
                "sk_inbox",
                "sk_checkpoint",
                "sk_consumer",
                "sk_commit",
                "sk_record",
                "sk_session_key",
                "sk_session",
            )
            for entry in scope:
                for table in tables:
                    self._rows(
                        f"DELETE FROM {table} WHERE tenant_id=%s AND session_id=%s",
                        (self.tenant_id, entry.session_id),
                    )
            for entry in scope:
                self._rows(
                    "UPDATE sk_tombstone SET state='complete' WHERE tenant_id=%s AND session_id=%s AND state='pending'",
                    (self.tenant_id, entry.session_id),
                )
            return tuple(entry.session_id for entry in scope)


class _TenantSQLStore:
    """Tenant-bound public view over one concrete SQL store."""

    def __init__(self, root: SQLStore, tenant_id: str):
        _tenant(tenant_id)
        self._root = root
        self._bound = root._scoped(tenant_id)
        self.tenant_id = tenant_id

    def for_tenant(self, tenant_id: str) -> SessionStore:
        _tenant(tenant_id)
        if tenant_id == self.tenant_id:
            return self
        return self._root.for_tenant(tenant_id)

    @property
    def fold_factory(self) -> Callable[[], Fold]:
        return self._root.fold_factory

    @fold_factory.setter
    def fold_factory(self, factory: Callable[[], Fold]) -> None:
        self._root.fold_factory = factory
        self._bound.fold_factory = factory

    def heartbeat_store(self) -> AbstractContextManager[SessionStore]:
        @contextmanager
        def scoped() -> Iterator[SessionStore]:
            with self._root.heartbeat_store() as store:
                yield store.for_tenant(self.tenant_id)

        return scoped()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)


class SQLSessionTx:
    def __init__(self, store: SQLStore):
        self.store = store
        self.transaction_id = store._transaction_id()
        self.batches: dict[tuple[str, str], ConsumerBatch] = {}
        self._lease_guard: Callable[[], Lease] | None = None

    def _active(self) -> None:
        if self.store._transaction_id() != self.transaction_id:
            raise RuntimeError("SessionTx must be used within its original caller transaction")

    def _commit(
        self,
        session_id: str,
        commit_id: str,
        body: bytes,
        *metadata: Any,
    ) -> None:
        if len(metadata) == 3:
            sequences, head, now = metadata
            manifest_body = body
        elif len(metadata) == 4:
            legacy_body, sequences, head, now = metadata
            if commit_id != "session/create":
                raise LedgerIntegrityError("legacy commit metadata is only valid for session creation")
            try:
                legacy = json.loads(legacy_body)
                consumers = tuple(legacy["consumers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LedgerIntegrityError("legacy commit body is invalid") from exc
        else:
            raise TypeError("commit metadata requires sequences, head and timestamp")
        if not isinstance(body, bytes) or not isinstance(sequences, tuple):
            raise LedgerIntegrityError("commit metadata is invalid")
        if len(metadata) == 3:
            try:
                manifest = json.loads(manifest_body)
                head_seq = manifest["head_seq"]
                head_chain_digest = manifest["head_chain_digest"]
                consume_input_ids = tuple(manifest["consume_input_ids"])
                consumers = tuple(manifest["consumers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LedgerIntegrityError("commit manifest is invalid") from exc
        else:
            head_seq, consume_input_ids = 0, ()
        _natural(head, "commit resulting head")
        _natural(now, "commit created_ms")
        if head != max([head_seq, *(seq for _, seq in sequences)]):
            raise LedgerIntegrityError("commit resulting head does not match its record sequences")
        if len(metadata) == 3 and head_chain_digest != self.store._chain_anchor(session_id, head_seq):
            raise LedgerIntegrityError("commit head_chain_digest does not match the persisted head")
        record_refs: list[tuple[str, str]] = []
        for record_id, seq in sequences:
            _identity(record_id)
            _positive(seq, "commit record seq")
            record_refs.append(
                (
                    record_id,
                    self.store._one(
                        "SELECT digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND record_id=%s AND seq=%s",
                        (self.store.tenant_id, session_id, record_id, seq),
                    )[0],
                )
            )
        stored_body = _commit_manifest(
            head_seq=head_seq,
            head_chain_digest=head_chain_digest if len(metadata) == 3 else self.store._chain_anchor(session_id, head_seq),
            records=tuple(record_refs),
            consume_input_ids=consume_input_ids,
            consumers=consumers,
        )
        if len(metadata) == 3 and stored_body != manifest_body:
            raise LedgerIntegrityError("commit manifest does not match persisted record digests")
        stored_digest = sha256(stored_body).hexdigest()
        for _record_id, seq in sequences:
            if seq <= head_seq:
                continue
            updated = self.store._rows(
                "UPDATE sk_record SET commit_digest=%s WHERE tenant_id=%s AND session_id=%s "
                "AND seq=%s AND commit_id=%s RETURNING seq",
                (stored_digest, self.store.tenant_id, session_id, seq, commit_id),
            )
            if len(updated) != 1:
                raise LedgerIntegrityError("commit references a record outside its transaction")
        self.store._rows(
            "INSERT INTO sk_commit(tenant_id,session_id,commit_id,body,digest,record_sequences,head_seq,created_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                self.store.tenant_id,
                session_id,
                commit_id,
                stored_body,
                stored_digest,
                canonical_json_bytes(sequences),
                head,
                now,
            ),
        )

    def _write(
        self,
        record: Record,
        *,
        seq: int,
        commit_id: str,
        commit_digest: str,
        chain_digest: str | None = None,
        position: int,
        epoch: int,
        now: int,
        body: bytes | None = None,
    ) -> None:
        body = record.encode() if body is None else body
        digest = sha256(body).hexdigest()
        expected_chain = _extend_chain_digest(self.store._chain_anchor(record.session_id, seq - 1), seq, digest)
        if chain_digest is not None and chain_digest != expected_chain:
            raise LedgerIntegrityError("record chain digest does not extend the persisted head")
        chain_digest = expected_chain
        self.store._rows(
            "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,schema_version,kind,turn_id,operation_id,"
            "attempt,body,digest,chain_digest,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
            "VALUES (%s,%s,%s,%s,'S1',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                self.store.tenant_id,
                record.session_id,
                seq,
                record.record_id,
                record.kind,
                record.turn_id,
                record.operation_id,
                record.attempt,
                body,
                digest,
                chain_digest,
                commit_id,
                commit_digest,
                position,
                epoch,
                now,
            ),
        )
        bound = self.store._bind_checkpoint_signer()
        if bound is not None:
            self._checkpoint(bound, session_id=record.session_id, seq=seq, chain_digest=chain_digest, now=now)

    def _checkpoint(self, bound: Any, *, session_id: str, seq: int, chain_digest: str, now: int) -> None:
        key_id = bound.key_id
        message = _checkpoint_message(self.store.tenant_id, session_id, seq, chain_digest, key_id, now)
        signature = bound.sign(message)
        if not isinstance(signature, bytes) or not signature:
            raise LedgerIntegrityError("ledger signer returned an invalid signature")
        self.store._rows(
            "INSERT INTO sk_checkpoint(tenant_id,session_id,seq,chain_digest,key_id,signature,signed_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (self.store.tenant_id, session_id, seq, chain_digest, key_id, signature, now),
        )

    def lock_session(self, session_id: str) -> None:
        self._active()
        self.store._require_live_row(self.store._lock(session_id))

    def create(self, spec: SessionSpec, *, consumers: tuple[str, ...]) -> CreateReceipt:
        self._active()
        for consumer in consumers:
            _identity(consumer)
        if len(consumers) != len(set(consumers)):
            raise Conflict("duplicate consumer")
        commit_id = "session/create"
        s = self.store
        principal = spec.principal
        tenant_id = principal.get("tenant_id") if isinstance(principal, dict) else None
        if not isinstance(tenant_id, str):
            raise Conflict("session principal requires a tenant_id")
        _tenant(tenant_id)
        if tenant_id != s.tenant_id:
            raise Conflict("session principal belongs to another tenant")
        with s._transaction():
            ceiling = s._write_ceiling()
            parent_level = 1
            if spec.parent_session_id is not None:
                parent = s._rows(
                    f"SELECT level,deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s{s.lock_clause}",
                    (tenant_id, spec.parent_session_id),
                )
                if len(parent) != 1:
                    raise Conflict("parent session not found in tenant")
                if parent[0][1] != "live":
                    raise SessionNotLive("parent session is fenced for deletion")
                parent_level = _level(parent[0][0], "parent session level")
            if s._rows(
                "SELECT 1 FROM sk_tombstone WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, spec.session_id),
            ):
                raise SessionNotLive("a deleted session identity cannot be reused")
            existing_session = s._rows(
                "SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, spec.session_id),
            )
            if existing_session and existing_session[0][0] != "live":
                raise SessionNotLive("a deleted session identity cannot be reused")
            now = s._now()
            inserted = s._rows(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,parent_session_id,parent_operation_id,"
                "phase,level,deletion_state,created_ms) "
                "VALUES (%s,%s,'S1',%s,%s,'idle',1,'live',%s) "
                "ON CONFLICT(tenant_id,session_id) DO NOTHING RETURNING session_id",
                (tenant_id, spec.session_id, spec.parent_session_id, spec.parent_operation_id, now),
            )
            row = s._lock(spec.session_id)
            existing = (
                s._one(
                    "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
                    (tenant_id, spec.session_id),
                )[0]
                if not inserted
                else None
            )
            record = spec.record(_existing=Record.parse(existing)) if existing is not None else spec.record()
            ordered_consumers = tuple(sorted(consumers))
            manifest = _commit_manifest(
                head_seq=0,
                head_chain_digest=_initial_chain_digest(tenant_id, spec.session_id),
                records=((record.record_id, record.digest),),
                consume_input_ids=(),
                consumers=ordered_consumers,
            )
            if not inserted:
                s._prefix(spec.session_id, row)
                old = s._one(
                    "SELECT body,created_ms FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                    (tenant_id, spec.session_id, commit_id),
                )
                if old[0] != manifest or existing != record.encode():
                    raise Conflict("create identity has different bytes")
                return CreateReceipt(spec.session_id, 1, old[1], True)
            required = s._admit_level(
                ceiling=ceiling,
                required=max(s._record_level(record), parent_level),
            )
            s._rows(
                "UPDATE sk_session SET level=%s WHERE tenant_id=%s AND session_id=%s",
                (required, tenant_id, spec.session_id),
            )
            chain_digest = _extend_chain_digest(_initial_chain_digest(tenant_id, spec.session_id), 1, record.digest)
            commit_digest = sha256(manifest).hexdigest()
            state = s.fold_factory().extend((record,))
            self._write(
                record,
                seq=1,
                commit_id=commit_id,
                commit_digest=commit_digest,
                chain_digest=chain_digest,
                position=0,
                epoch=0,
                now=now,
            )
            self._commit(spec.session_id, commit_id, manifest, ((record.record_id, 1),), 1, now)
            for consumer in consumers:
                s._rows(
                    "INSERT INTO sk_consumer(tenant_id,session_id,consumer) VALUES (%s,%s,%s)",
                    (tenant_id, spec.session_id, consumer),
                )
            s._rows(
                "UPDATE sk_session SET head_seq=1 WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, spec.session_id),
            )
            s._schedule(spec.session_id, state)
            return CreateReceipt(spec.session_id, 1, now)

    def push(self, session_id: str, item: InboxItem) -> InputReceipt:
        self._active()
        if item.admission_proof is not None:
            raise InputAdmissionProofError("input admission proof is created inside the SQL transaction")
        body = _input_body(item)
        s = self.store
        assert item.actor is not None
        if item.actor["tenant_id"] != s.tenant_id:
            raise Conflict("input actor belongs to another tenant")
        configured, allow_unsigned, signer = s._input_admission_policy()
        if configured and not allow_unsigned and signer is None:
            raise InputAdmissionProofError("signed input admission signer is unavailable")
        with s._transaction():
            ceiling = s._write_ceiling()
            row = s._lock(session_id)
            s._require_live_row(row)
            old = s._rows(
                "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,admission_signature,admission_signed_ms "
                "FROM sk_inbox "
                "WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                (s.tenant_id, session_id, item.input_id),
            )
            if old:
                stored = s._stored_input(session_id, old[0])
                if old[0][0] != body:
                    raise Conflict("input identity has different bytes")
                return InputReceipt(item.input_id, stored.input_seq, stored.received_ms, True)
            required = s._admit_level(
                ceiling=ceiling,
                required=s._input_level(item),
                current=row[6],
            )
            now, seq = s._now(), row[1] + 1
            key_id = signature = signed_ms = None
            if configured and not allow_unsigned:
                assert signer is not None
                bound = signer.bind_key()
                key_id = getattr(bound, "key_id", None)
                if not isinstance(key_id, str) or not key_id:
                    raise InputAdmissionProofError("ledger signer returned an invalid key identity")
                message = _input_admission_message(
                    s.tenant_id,
                    session_id,
                    item.input_id,
                    seq,
                    now,
                    sha256(body).hexdigest(),
                    key_id,
                    now,
                )
                signature = bound.sign(message)
                if not isinstance(signature, bytes) or not signature:
                    raise InputAdmissionProofError("ledger signer returned an invalid signature")
                signed_ms = now
            s._rows(
                "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,"
                "admission_key_id,admission_signature,admission_signed_ms,available_ms,received_ms) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    s.tenant_id,
                    session_id,
                    item.input_id,
                    seq,
                    item.kind,
                    body,
                    sha256(body).hexdigest(),
                    key_id,
                    signature,
                    signed_ms,
                    item.available_ms,
                    now,
                ),
            )
            s._rows(
                "UPDATE sk_session SET inbox_seq=%s,level=%s WHERE tenant_id=%s AND session_id=%s",
                (seq, required, s.tenant_id, session_id),
            )
            return InputReceipt(item.input_id, seq, now)

    def append(
        self,
        session_id: str,
        *,
        lease: Lease,
        expected_seq: int,
        commit_id: str,
        records: tuple[Record, ...],
        consume_input_ids: tuple[str, ...] = (),
        expected_inbox_seq: int | None = None,
    ) -> CommitReceipt:
        self._active()
        _identity(commit_id)
        _natural(expected_seq, "expected_seq")
        if expected_inbox_seq is not None:
            _natural(expected_inbox_seq, "expected_inbox_seq")
        if len(set(consume_input_ids)) != len(consume_input_ids):
            raise Conflict("duplicate consume_input_ids")
        for record in records:
            if (
                record.kind == "op_completed"
                and (input_id := record._payload.get("yielded_by")) is not None
                and (
                    input_id not in consume_input_ids
                    or not any(
                        r.kind == "input_applied"
                        and r.turn_id == record.turn_id
                        and r._payload["input"]["input_id"] == input_id
                        and r._payload["input"]["kind"] == "steer"
                        and r._payload["disposition"] == "applied"
                        for r in records
                    )
                )
            ):
                raise Conflict("yield completion and steering must commit together")
        encoded = []
        for record in records:
            encoded.append(record.encode())
            if record.session_id != session_id:
                raise Conflict("record belongs to another session")
        if len({r.record_id for r in records}) != len(records):
            raise Conflict("duplicate record_id in batch")
        s = self.store
        s._require_live(session_id)
        captured = s._head(session_id)
        prefix = None
        reducer = None
        # Commit replay needs no history or live lease. Prepare new immutable history
        # before the write lock; locked admission below rechecks both CAS and epoch.
        if expected_seq == captured[0] and not s._rows(
            "SELECT 1 FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (s.tenant_id, session_id, commit_id),
        ):
            prefix = s._prefix(session_id, captured)
            reducer = prefix.reducer.fork()
        if self._lease_guard is not None:
            lease = self._lease_guard()
        with s._bound_append(lease) if self._lease_guard is not None else s._transaction():
            ceiling = s._write_ceiling()
            row = s._lock(session_id)
            head_chain_digest = s._chain_anchor(session_id, expected_seq)
            body = _commit_manifest(
                head_seq=expected_seq,
                head_chain_digest=head_chain_digest,
                records=tuple((record.record_id, record.digest) for record in records),
                consume_input_ids=tuple(consume_input_ids),
                consumers=(),
            )
            old = s._rows(
                "SELECT body,record_sequences,head_seq FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                (s.tenant_id, session_id, commit_id),
            )
            if old:
                s._prefix(session_id, row)
                stored = {
                    r.record_id: s._one(
                        "SELECT body,digest,seq FROM sk_record WHERE tenant_id=%s AND session_id=%s AND record_id=%s",
                        (s.tenant_id, session_id, r.record_id),
                    )
                    for r in records
                }
                stored_manifest = json.loads(old[0][0])
                stored_sequences = tuple((record_id, seq) for record_id, seq in json.loads(old[0][1]))
                body = _commit_manifest(
                    head_seq=stored_manifest["head_seq"],
                    head_chain_digest=stored_manifest["head_chain_digest"],
                    records=tuple((r.record_id, stored[r.record_id][1]) for r in records),
                    consume_input_ids=tuple(consume_input_ids),
                    consumers=tuple(stored_manifest["consumers"]),
                )
                if (
                    old[0][0] != body
                    or any(stored[r.record_id][0] != encoded[i] for i, r in enumerate(records))
                    or tuple(record_id for record_id, _ in stored_sequences) != tuple(r.record_id for r in records)
                    or old[0][2] != max([stored_manifest["head_seq"], *(seq for _, seq in stored_sequences)])
                ):
                    raise Conflict("commit identity has different bytes")
                return CommitReceipt(commit_id, stored_sequences, old[0][2], True)
            s._check_lease(session_id, lease, row, s._now())
            if expected_seq != row[0] or (expected_inbox_seq is not None and expected_inbox_seq != row[1]):
                raise SequenceConflict("head or inbox watermark changed")
            if prefix is None or reducer is None or row[2] != captured[2]:
                raise SequenceConflict("head or epoch changed during append preparation")
            if prefix.head_digest != head_chain_digest:
                raise LedgerIntegrityError("commit head_chain_digest does not match the original head")
            required = s._admit_level(
                ceiling=ceiling,
                required=max((s._record_level(record) for record in records), default=1),
                current=row[6],
            )
            sequences: list[tuple[str, int]] = []
            new: list[tuple[int, Record, int]] = []
            head = row[0]
            for position, record in enumerate(records):
                prior = s._rows(
                    "SELECT body,seq FROM sk_record WHERE tenant_id=%s AND session_id=%s AND record_id=%s",
                    (s.tenant_id, session_id, record.record_id),
                )
                if prior:
                    if prior[0][0] != encoded[position]:
                        raise Conflict("record identity has different bytes")
                    sequences.append((record.record_id, prior[0][1]))
                else:
                    head += 1
                    sequences.append((record.record_id, head))
                    new.append((position, record, head))
            applications = {r._payload["input"]["input_id"]: (r, seq) for _, r, seq in new if r.kind == "input_applied"}
            if set(applications) != set(consume_input_ids):
                raise Conflict("new input_applied and consume_input_ids must match exactly")
            consumed = []
            now = s._now()
            for input_id in consume_input_ids:
                incoming = s._one(
                    "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
                    "admission_signature,admission_signed_ms,available_ms FROM sk_inbox "
                    "WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                    (s.tenant_id, session_id, input_id),
                )
                stored = s._stored_input(session_id, incoming[:8])
                applied = applications[input_id][0]
                applied_input = applied._payload["input"]
                if (
                    stored.consumed_seq is not None
                    or incoming[8] > now
                    or stored.item.to_dict() != applied_input
                    or stored_input_proof(stored) != applied._payload["input_proof"]
                ):
                    raise Conflict("input already consumed, unavailable, or different bytes")
                consumed.append(stored.item)
            state = reducer.extend((r for _, r, _ in new), consumed_inputs=consumed, bodies=(encoded[p] for p, _, _ in new))
            now = s._now()
            s._check_lease(session_id, lease, row, now)  # Validation may outlive the lease.
            commit_digest = sha256(body).hexdigest()
            previous_chain = prefix.head_digest
            for position, record, seq in new:
                if record.kind == "op_started":
                    if record._payload["epoch"] != lease.epoch:
                        raise Conflict("dispatch epoch does not match writer")
                    assert record.operation_id is not None and record.attempt is not None
                    op = state.operations[record.operation_id]
                    planned = op.attempts[record.attempt].plan
                    if (planned._payload["not_before_ms"] or 0) > now:
                        raise Conflict("dispatch before not-before")
                chain_digest = _extend_chain_digest(previous_chain, seq, record.digest)
                self._write(
                    record,
                    seq=seq,
                    commit_id=commit_id,
                    commit_digest=commit_digest,
                    chain_digest=chain_digest,
                    position=position,
                    epoch=lease.epoch,
                    now=now,
                    body=encoded[position],
                )
                previous_chain = chain_digest
            for input_id, (_, seq) in applications.items():
                s._rows(
                    "UPDATE sk_inbox SET consumed_seq=%s WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                    (seq, s.tenant_id, session_id, input_id),
                )
            s._rows(
                "UPDATE sk_session SET head_seq=%s,level=%s WHERE tenant_id=%s AND session_id=%s",
                (head, required, s.tenant_id, session_id),
            )
            # Consuming unrelated input must not erase an idle provider's SQL deferral.
            s._schedule(
                session_id,
                state,
                preserve_due=state.active_turn_id == prefix.reducer.state.active_turn_id
                and state.due_ms == prefix.reducer.state.due_ms
                and all(r.kind in {"input_applied", "usage_observed"} for _, r, _ in new),
            )
            added = tuple(StoredRecord(r, seq, commit_id, lease.epoch, now) for _, r, seq in new)
            receipt = CommitReceipt(commit_id, tuple(sequences), head, records=added, inbox_seq=row[1])
            self._commit(session_id, commit_id, body, receipt.record_sequences, head, now)
            checksum = previous_chain
            s._retain_prefix(_Prefix(s.tenant_id, session_id, lease.epoch, checksum, reducer, prefix.records + added))
            s._check_lease(session_id, lease, row, s._now())
            return receipt

    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None:
        self._active()
        _positive(limit, "limit")
        s = self.store
        last, level, deletion_state = s._one(
            "SELECT c.last_seq,s.level,s.deletion_state FROM sk_consumer c "
            "JOIN sk_session s ON s.tenant_id=c.tenant_id AND s.session_id=c.session_id "
            f"WHERE c.tenant_id=%s AND c.session_id=%s AND c.consumer=%s{s.lock_clause}",
            (s.tenant_id, session_id, consumer),
        )
        if deletion_state != "live":
            raise SessionNotLive("session is fenced for deletion")
        if level > s._worker_level:
            return None
        # One statement snapshot. Include all remaining rows of the commit touching the soft limit.
        rows = s._rows(
            """
            WITH page AS MATERIALIZED (
                SELECT seq,commit_id FROM sk_record
                WHERE tenant_id=%s AND session_id=%s AND seq>%s ORDER BY seq LIMIT %s
            ), boundary AS (
                SELECT max(r.seq) AS seq FROM sk_record r
                WHERE r.tenant_id=%s AND r.session_id=%s
                AND r.commit_id=(SELECT commit_id FROM page ORDER BY seq DESC LIMIT 1)
            )
            SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record
            WHERE tenant_id=%s AND session_id=%s AND seq>%s
              AND seq<=(SELECT seq FROM boundary) ORDER BY seq
            """,
            (s.tenant_id, session_id, last, limit, s.tenant_id, session_id, s.tenant_id, session_id, last),
        )
        key = (session_id, consumer)
        if not rows:
            self.batches.pop(key, None)
            return None
        records = tuple(s._stored(session_id, row) for row in rows)
        batch = ConsumerBatch(session_id, consumer, last + 1, records[-1].seq, records)
        self.batches[key] = batch
        return batch

    def ack(self, batch: ConsumerBatch) -> None:
        self._active()
        self.store._require_live(batch.session_id)
        key = (batch.session_id, batch.consumer)
        if self.batches.get(key) is not batch:
            raise Conflict("ack requires the batch issued by this transaction")
        updated = self.store._rows(
            "UPDATE sk_consumer SET last_seq=%s "
            "WHERE tenant_id=%s AND session_id=%s AND consumer=%s "
            "AND last_seq=%s RETURNING last_seq",
            (batch.through_seq, self.store.tenant_id, *key, batch.from_seq - 1),
        )
        if not updated:
            raise Conflict("consumer cursor changed")
        del self.batches[key]
