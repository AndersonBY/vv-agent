"""Shared SQL transaction rules for the PostgreSQL reference and single-host SQLite store."""

from __future__ import annotations

import base64
import hmac
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from copy import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from vv_agent.kernel._canonical_json import canonical_json_bytes, validate_sha256
from vv_agent.kernel._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentUnavailable,
    ForgetTombstone,
    KeyUnavailable,
    Principal,
    SessionHead,
    TombstoneState,
    _tombstone_message,
)
from vv_agent.kernel._records import InboxItem, Record, SessionSpec, _MessageCache
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._requirements import SessionRequirements, normalize_requirements
from vv_agent.kernel._state import ExecutionState, queued_input_starts_turn
from vv_agent.kernel._store import (
    DEFAULT_TENANT_ID,
    SESSION_PURGE_DELETE_ORDER,
    ClaimedWork,
    CommitReceipt,
    Conflict,
    ConsumerBatch,
    CreateReceipt,
    InputReceipt,
    Lease,
    LeaseLost,
    LeasePoll,
    ProjectionConsumers,
    ReadPage,
    SequenceConflict,
    SessionStore,
    StoredInput,
    StoredRecord,
    WorkCursor,
    WorkerSupports,
    WorkItem,
    WorkKinds,
)

if TYPE_CHECKING:
    from ._external_content import _RootObjectErasure


_ExternalRoot = bool | tuple["_RootObjectErasure", Callable[[], None]]
type _ObjectDeadline = tuple[int, Callable[[], int]]


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
_RECORD_READ_PAGE_SIZE = 256
_COMMIT_PARSE_CACHE_LIMIT = 1024
_CONFIGURED_VERIFIER = object()
_STORED_INBOX_COLUMNS = (
    "body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
    "admission_signature,admission_signed_ms,input_id,kind,available_ms"
)


@dataclass(frozen=True)
class _RoutingCache:
    requirements: SessionRequirements
    active_turn_id: str | None
    route_level: int | None = None


@dataclass(slots=True)
class _TransactionState:
    depth: int = 0
    epoch: int = 0
    sqlite_write_fence: bool = False


class InputAdmissionProofError(Conflict):
    """A signed inbox row no longer matches its original trusted admission."""


class LedgerIntegrityError(Conflict):
    """Stored ledger bytes, chain positions, commit metadata or checkpoints do not agree."""


class UnsupportedFeatureLevel(Conflict):
    """A producer cannot publish work above the database's write-level ceiling."""


class UnsupportedWorkerCapabilities(UnsupportedFeatureLevel):
    """A worker lacks the handler or capability required by active work."""


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
    try:
        validate_sha256(value, name)
    except ValueError as exc:
        raise LedgerIntegrityError(f"{name} must be a lowercase SHA-256 digest") from exc
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


def _require_expected_head(
    expected_head: SessionHead | tuple[int, str] | None,
    *,
    tenant_id: str,
    session_id: str,
    head_seq: int,
    chain_digest: str,
) -> None:
    """Compare a caller-owned head anchor with the SQL row before trusting it."""

    if expected_head is None:
        return
    if isinstance(expected_head, SessionHead):
        expected = (
            expected_head.tenant_id,
            expected_head.session_id,
            expected_head.final_seq,
            expected_head.chain_digest,
        )
    elif (
        type(expected_head) is tuple
        and len(expected_head) == 2
        and type(expected_head[0]) is int
        and isinstance(expected_head[1], str)
    ):
        expected = (tenant_id, session_id, expected_head[0], expected_head[1])
    else:
        raise TypeError("expected_head must be SessionHead or (final_seq, chain_digest)")
    actual = (tenant_id, session_id, head_seq, chain_digest)
    if expected != actual:
        raise ContentBindingMismatch(f"ledger head does not match expected_head at seq {head_seq}")


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


@dataclass(frozen=True, slots=True)
class _ProjectionIndexes:
    operations: dict[tuple[str, int], tuple[StoredRecord, ...]]
    turn_starts: dict[str, StoredRecord]
    turn_dependencies: dict[str, tuple[StoredRecord, ...]]
    inputs_by_id: dict[str, StoredInput]


@dataclass(frozen=True, slots=True)
class _CommitPayload:
    head_seq: int
    head_chain_digest: Any
    records: tuple[tuple[str, str], ...]
    sequences: tuple[tuple[str, int], ...]
    consume_input_ids: tuple[str, ...]
    consumers: tuple[str, ...]


@dataclass
class _Prefix:
    tenant_id: str
    session_id: str
    epoch: int
    head_digest: str
    reducer: Fold
    records: tuple[StoredRecord, ...]
    stored_inputs: tuple[StoredInput, ...] = ()
    inbox_seq: int = 0
    chains: tuple[str, ...] = ()
    _projection_indexes: _ProjectionIndexes | None = field(default=None, init=False, repr=False)

    def projection_indexes(self) -> _ProjectionIndexes:
        if self._projection_indexes is not None:
            return self._projection_indexes
        operations: dict[tuple[str, int], list[StoredRecord]] = {}
        turn_starts: dict[str, StoredRecord] = {}
        dependencies: dict[str, list[StoredRecord]] = {}
        for stored in self.records:
            record = stored.record
            if (
                record.operation_id is not None
                and record.attempt is not None
                and record.kind
                in {
                    "op_planned",
                    "op_prepared",
                    "op_parked",
                }
            ):
                operations.setdefault((record.operation_id, record.attempt), []).append(stored)
            if record.kind == "turn_started" and record.turn_id is not None:
                turn_starts.setdefault(record.turn_id, stored)
            if record.kind == "boundary_recorded" and record.turn_id is not None:
                dependencies.setdefault(record.turn_id, []).append(stored)
            elif record.kind == "input_applied" and record.turn_id is not None:
                input_id = record._payload["input"]["input_id"]
                if input_id.startswith("compaction/"):
                    dependencies.setdefault(record.turn_id, []).append(stored)
        self._projection_indexes = _ProjectionIndexes(
            operations={key: tuple(values) for key, values in operations.items()},
            turn_starts=turn_starts,
            turn_dependencies={key: tuple(values) for key, values in dependencies.items()},
            inputs_by_id={stored.item.input_id: stored for stored in self.stored_inputs},
        )
        return self._projection_indexes

    @property
    def head_seq(self) -> int:
        return len(self.records)


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
    _worker_handler_filter_sql = (
        "(s.handler_version IS NULL OR s.handler_version IN (SELECT CAST(value AS TEXT) FROM json_each(%s)))"
    )
    _worker_capability_filter_sql = (
        "NOT EXISTS (SELECT 1 FROM json_each(CAST(s.required_capabilities AS TEXT)) AS required "
        "WHERE CAST(required.value AS TEXT) NOT IN (SELECT CAST(value AS TEXT) FROM json_each(%s)))"
    )
    clock_sql = ""
    _input_admission_configured = False
    _input_admission_allow_unsigned = False
    _input_admission_signer: LedgerSigner | None = None
    _fold_cache: _Prefix | None = None
    _previous_prefix: _Prefix | None = None
    _read_snapshot_active = False
    _tenant_id = DEFAULT_TENANT_ID
    _root_store: SQLStore | None = None
    _worker_supports = WorkerSupports()
    _all_tenants = False
    _reader_level = 2

    def __init__(self, *, fold_factory: Callable[[], Fold]):
        self.fold_factory = fold_factory
        self._transaction_state = _TransactionState()

    @cached_property
    def _commit_parse_cache(self) -> dict[tuple[bytes, bytes], _CommitPayload]:
        return {}

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def worker_level(self) -> int:
        return self.worker_supports.max_level

    @property
    def max_feature_level(self) -> int:
        """Return the durable write ceiling without changing it."""

        return self._write_ceiling()

    @property
    def worker_supports(self) -> WorkerSupports:
        return (self._root_store or self)._worker_supports

    @property
    def reader_level(self) -> int:
        return (self._root_store or self)._reader_level

    @property
    def claims_all_tenants(self) -> bool:
        return self._root_store is None and bool(self._all_tenants)

    def _claim_tenant_id(self) -> str | None:
        return None if self.claims_all_tenants else self.tenant_id

    def configure_reader(self, *, level: int = 2) -> None:
        """Set the local S1 reader format level without declaring execution support."""

        root = self._root_store or self
        root._reader_level = _level(level, "reader level")

    def configure_worker(
        self,
        *,
        level: int = 1,
        handler_versions: tuple[str, ...] = ("1",),
        capabilities: tuple[str, ...] = (),
        all_tenants: bool = False,
    ) -> None:
        """Set the immutable execution support declared by this worker process."""

        root = self._root_store or self
        if all_tenants and self._root_store is not None:
            raise ValueError("cross-tenant worker policy must be configured on the root store")
        worker_level = _level(level, "worker level")
        root._worker_supports = WorkerSupports(
            max_level=worker_level,
            handler_versions=handler_versions,
            capabilities=capabilities,
        )
        root._reader_level = max(root._reader_level, worker_level)
        if self._root_store is None:
            root._all_tenants = bool(all_tenants)

    def _runnable_lock_clause(self) -> str:
        return ""

    def heartbeat_store(self) -> AbstractContextManager[SessionStore]:
        raise NotImplementedError

    def heartbeat_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        """Return the store's repeatable renewal connection factory."""

        return self.heartbeat_store

    def _require_claim_capable(self) -> None:
        """Reject store bindings that must not own worker leases."""

    def for_tenant(self, tenant_id: str) -> SessionStore:
        """Return a lightweight facade whose public operations are tenant-scoped."""
        _tenant(tenant_id)
        return cast(SessionStore, _TenantSQLStore(self._root_store or self, tenant_id))

    def _scoped(self, tenant_id: str) -> SQLStore:
        scoped = copy(self)
        scoped._root_store = self._root_store or self
        scoped._tenant_id = tenant_id
        scoped._fold_cache = None
        scoped._previous_prefix = None
        scoped.__dict__.pop("_commit_parse_cache", None)
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

    def _transaction_in_progress(self) -> bool:
        """Return whether this store already has a caller-owned transaction."""

        depth = getattr(self, "_depth", 0)
        connection = getattr(self, "connection", None)
        return bool(depth or getattr(connection, "in_transaction", False))

    def _require_claim_outside_transaction(self) -> None:
        if self._transaction_in_progress():
            raise RuntimeError("claim_runnable requires an idle connection outside a host transaction")

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

    def _ledger_integrity_mode(self) -> str:
        """Return the configured policy without consulting mutable proof rows."""

        configured, allow_unsigned, _signer = self._input_admission_policy()
        if not configured:
            return "unconfigured"
        return "hash-only" if allow_unsigned else "signed"

    def _require_configured_ledger(self, operation: str) -> None:
        """Reject trust-dependent operations until the host chooses a proof mode."""

        if self._ledger_integrity_mode() == "unconfigured":
            raise LedgerIntegrityError(f"{operation} requires an explicit signed signer or allow_unsigned hash-only mode")

    def _resolve_ledger_verifier(
        self,
        verifier: Any,
        signed: bool | None,
    ) -> tuple[bool, Any | None]:
        """Resolve explicit proof mode once for a verification boundary."""

        configured, allow_unsigned, configured_verifier = self._input_admission_policy()
        if verifier is _CONFIGURED_VERIFIER:
            verifier = configured_verifier
        if not configured and verifier is None:
            self._require_configured_ledger("ledger verification")
        resolved_signed = configured and not allow_unsigned if signed is None else signed
        return resolved_signed, verifier

    def _require_input_integrity_gate(self) -> None:
        """Require an explicit mode before reading any inbox projection."""

        self._require_configured_ledger("ledger input read")

    @contextmanager
    def _inherit_runtime_configuration(self, manager: AbstractContextManager[SQLStore]) -> Iterator[SQLStore]:
        """Configure an independent connection with this store's process-wide policy."""

        with manager as store:
            supports = self.worker_supports
            store.configure_reader(level=self.reader_level)
            store.configure_worker(
                level=supports.max_level,
                handler_versions=supports.handler_versions,
                capabilities=supports.capabilities,
                all_tenants=self.claims_all_tenants,
            )
            configured, allow_unsigned, signer = self._input_admission_policy()
            if configured:
                store.configure_input_admission(signer, allow_unsigned=allow_unsigned)
            yield store

    def _write_ceiling(self) -> int:
        """Read and lock the database-wide write ceiling inside the caller transaction."""
        # A shared settings lock prevents a concurrent ceiling update while
        # avoiding a lock-order cycle with content transactions that lock the
        # session key before checking the ceiling.
        settings_lock = " FOR SHARE" if self.lock_clause else ""
        rows = self._rows(f"SELECT storage_major,write_level FROM sk_settings WHERE singleton=1{settings_lock}")
        if len(rows) != 1 or rows[0][0] != "S1":
            raise LedgerIntegrityError("settings do not identify an S1 store")
        return _level(rows[0][1], "write_level")

    def _admit_level(self, *, ceiling: int, required: int, current: int = 1) -> int:
        effective = max(_level(required, "required feature level"), _level(current, "session level"))
        if effective > ceiling:
            raise UnsupportedFeatureLevel(f"feature level {effective} exceeds database write level {ceiling}")
        return effective

    @staticmethod
    def _session_capabilities(value: Any) -> tuple[str, ...]:
        body = value.tobytes() if isinstance(value, memoryview) else value
        if not isinstance(body, bytes):
            raise LedgerIntegrityError("session required_capabilities is not bytea")
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError("session required_capabilities is not canonical JSON") from exc
        if (
            not isinstance(decoded, list)
            or any(not isinstance(item, str) or not item or "\x00" in item for item in decoded)
            or decoded != sorted(set(decoded))
            or canonical_json_bytes(decoded, "session required_capabilities") != body
        ):
            raise LedgerIntegrityError("session required_capabilities is not canonical sorted unique text")
        return tuple(decoded)

    def _worker_requirements_from_row(self, row: tuple[Any, ...]) -> tuple[int, str | None, tuple[str, ...]]:
        if len(row) >= 10:
            level, handler_version, capabilities = row[6], row[8], self._session_capabilities(row[9])
        elif len(row) >= 6:
            level, handler_version, capabilities = row[3], row[4], self._session_capabilities(row[5])
        else:
            level, handler_version, capabilities = row[3], None, ()
        if handler_version is not None and (not isinstance(handler_version, str) or not handler_version):
            raise LedgerIntegrityError("session handler_version is invalid")
        return _level(level, "session feature level"), handler_version, capabilities

    def _worker_accepts_row(self, row: tuple[Any, ...]) -> bool:
        level, handler_version, capabilities = self._worker_requirements_from_row(row)
        return self.worker_supports.accepts(
            level=level,
            handler_version=handler_version,
            capabilities=capabilities,
        )

    def _require_worker_accepts_row(self, row: tuple[Any, ...]) -> None:
        level, handler_version, capabilities = self._worker_requirements_from_row(row)
        self._require_worker_supports(
            level=level,
            handler_version=handler_version,
            capabilities=capabilities,
        )

    def _require_worker_supports(
        self,
        *,
        level: int,
        handler_version: str | None,
        capabilities: Iterable[str],
    ) -> None:
        if level > self.worker_supports.max_level:
            raise UnsupportedFeatureLevel(
                f"session feature level {level} exceeds worker feature level {self.worker_supports.max_level}"
            )
        if handler_version is not None and handler_version not in self.worker_supports.handler_versions:
            raise UnsupportedWorkerCapabilities(f"session handler_version {handler_version} is not supported by this worker")
        missing = set(capabilities).difference(self.worker_supports.capabilities)
        if missing:
            raise UnsupportedWorkerCapabilities(
                "session required capabilities " + ",".join(sorted(missing)) + " are not supported by this worker"
            )

    def _require_reader_level(self, level: Any) -> None:
        required = _level(level, "reader requirement level")
        if required > self.reader_level:
            raise UnsupportedFeatureLevel(f"session feature level {required} exceeds reader feature level {self.reader_level}")

    def _require_reader_row(self, row: tuple[Any, ...]) -> None:
        self._require_reader_level(row[6] if len(row) >= 10 else row[3])

    def _worker_filter_params(self) -> tuple[str, str]:
        handlers = canonical_json_bytes(list(self.worker_supports.handler_versions), "worker handler_versions")
        capabilities = canonical_json_bytes(list(self.worker_supports.capabilities), "worker capabilities")
        return handlers.decode("utf-8"), capabilities.decode("utf-8")

    def _worker_inbox_candidate_predicate(
        self,
        *,
        session_alias: str = "s",
        inbox_alias: str = "candidate",
        clock_alias: str = "t",
    ) -> tuple[str, tuple[Any, ...]]:
        """Match a due route-defining inbox row from its body-free projection."""

        handlers, capabilities = self._worker_filter_params()
        handler_sql = self._worker_handler_filter_sql.replace("s.", f"{inbox_alias}.routing_")
        capability_sql = self._worker_capability_filter_sql.replace("s.", f"{inbox_alias}.routing_")
        sql = (
            f"EXISTS (SELECT 1 FROM sk_inbox {inbox_alias} "
            f"WHERE {inbox_alias}.tenant_id={session_alias}.tenant_id "
            f"AND {inbox_alias}.session_id={session_alias}.session_id "
            f"AND {inbox_alias}.consumed_seq IS NULL "
            f"AND {inbox_alias}.available_ms<={clock_alias}.ms "
            f"AND {inbox_alias}.kind IN ('user','follow_up','model_selection') "
            f"AND {inbox_alias}.routing_level<=%s AND {handler_sql} AND {capability_sql})"
        )
        return sql, (self.worker_level, handlers, capabilities)

    def _worker_candidate_predicate(self, *, session_alias: str = "s") -> tuple[str, tuple[Any, ...]]:
        """Match the session cache or a compatible due inbox without decoding either body."""

        handlers, capabilities = self._worker_filter_params()
        handler_sql = self._worker_handler_filter_sql.replace("s.", f"{session_alias}.")
        capability_sql = self._worker_capability_filter_sql.replace("s.", f"{session_alias}.")
        inbox_sql, inbox_params = self._worker_inbox_candidate_predicate(session_alias=session_alias)
        sql = f"{session_alias}.level<=%s AND (({handler_sql} AND {capability_sql}) OR {inbox_sql})"
        return sql, (self.worker_level, handlers, capabilities, *inbox_params)

    def _worker_accepts_ready_inbox(self, session_id: str) -> bool:
        predicate, params = self._worker_inbox_candidate_predicate()
        return bool(
            self._rows(
                f"WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms) "
                "SELECT 1 FROM sk_session s CROSS JOIN clock t "
                "WHERE s.tenant_id=%s AND s.session_id=%s AND s.deletion_state='live' "
                f"AND s.level<=%s AND {predicate}",
                (self.tenant_id, session_id, self.worker_level, *params),
            )
        )

    def _input_level(self, item: InboxItem, *, parent_session_id: str | None = None) -> int:
        required = _level(item.feature_level, "input feature_level")
        if item.kind == "child_result":
            child_id = item.payload.get("session_id")
            if not isinstance(child_id, str):
                raise Conflict("child_result requires a child session_id")
            child = self._rows(
                "SELECT level,deletion_state,parent_session_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, child_id),
            )
            if len(child) != 1:
                if parent_session_id is None:
                    # Persisted admission already raised the parent's level while the
                    # child was current. Rebuild must not make consumption depend on
                    # the child's later deletion lifecycle.
                    return required
                raise Conflict("child_result references a missing child session")
            if child[0][1] != "live":
                if parent_session_id is None:
                    return required
                raise SessionNotLive("child result session is fenced for deletion")
            if parent_session_id is not None and child[0][2] != parent_session_id:
                raise Conflict("child_result does not belong to this parent session")
            required = max(required, _level(child[0][0], "child session level"))
        return required

    def child_is_deleted(self, session_id: str) -> bool:
        """Return true only for a completed deletion bound by a verified tombstone."""

        row = self._rows(
            "SELECT deletion_state,head_seq FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        )
        tombstone = self._tombstone(session_id)
        if tombstone is None or tombstone.state is not TombstoneState.COMPLETE:
            return False
        _configured, allow_unsigned, verifier = self._input_admission_policy()
        self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
        if not row:
            # Purge removes the session row only after publishing its complete
            # tombstone. The tenant-scoped tombstone proves the identity was deleted.
            return True
        deletion_state, head = row[0]
        if deletion_state != "forgotten":
            return False
        if tombstone.final_seq != head or tombstone.chain_digest != self._chain_anchor(session_id, head):
            raise LedgerIntegrityError("completed tombstone does not match the forgotten session head")
        return True

    def _record_level(self, record: Record) -> int:
        # Historical input_applied records retain their own validated skeleton. Looking up
        # a child session again would make replay depend on deletion lifecycle.
        return _record_level(record)

    def _pending_input_level(self, stored_inputs: tuple[StoredInput, ...]) -> int:
        required = 1
        for stored in stored_inputs:
            if stored.consumed_seq is None:
                required = max(required, self._input_level(stored.item))
        return required

    def _historical_child_input_level(self, records: tuple[StoredRecord, ...]) -> int:
        child_ids = {
            stored.record._payload["input"]["payload"]["session_id"]
            for stored in records
            if stored.record.kind == "input_applied" and stored.record._payload["input"]["kind"] == "child_result"
        }
        if not child_ids:
            return 1
        placeholders = ",".join("%s" for _ in child_ids)
        rows = self._rows(
            f"SELECT level,deletion_state FROM sk_session WHERE tenant_id=%s AND session_id IN ({placeholders})",
            (self.tenant_id, *sorted(child_ids)),
        )
        return max((_level(level, "child session level") for level, state in rows if state == "live"), default=1)

    def _level_floor(
        self,
        records: tuple[StoredRecord, ...],
        stored_inputs: tuple[StoredInput, ...],
        *,
        minimum_level: int = 1,
    ) -> int:
        required = max(
            max((self._record_level(stored.record) for stored in records), default=1),
            self._historical_child_input_level(records),
        )
        return max(minimum_level, required, self._pending_input_level(stored_inputs))

    def _verify_level_cache(
        self,
        records: tuple[StoredRecord, ...],
        stored_inputs: tuple[StoredInput, ...],
        cached_level: int,
    ) -> None:
        cached = _level(cached_level, "session level")
        required = self._level_floor(records, stored_inputs)
        if cached < required:
            raise LedgerIntegrityError("session level cache is understated")

    @staticmethod
    def _inbox_requirements(
        item: InboxItem,
        *,
        inherited: SessionRequirements | None = None,
    ) -> SessionRequirements:
        if item.feature_level == 1 or (item.feature_level == 3 and item.requirements is None):
            return inherited or SessionRequirements(item.feature_level, None, ())
        assert item.requirements is not None
        return normalize_requirements(item.requirements)

    def _bootstrap_requirements(self, prefix: _Prefix) -> SessionRequirements:
        created = prefix.records[0].record
        if created.kind != "session_created":
            raise LedgerIntegrityError("session creation record is missing")
        if created.feature_level == 2:
            return normalize_requirements(created._payload["bootstrap"])
        return SessionRequirements(1, None, ())

    @staticmethod
    def _ready_pending_inputs(stored_inputs: tuple[StoredInput, ...], now: int) -> tuple[StoredInput, ...]:
        return tuple(stored for stored in stored_inputs if stored.consumed_seq is None and stored.item.available_ms <= now)

    @staticmethod
    def _selection_requirements(
        item: InboxItem,
    ) -> SessionRequirements | None:
        return None if item.payload["selection"] is None else SQLStore._inbox_requirements(item)

    def _derive_routing_cache(
        self,
        prefix: _Prefix,
        *,
        now: int,
        minimum_level: int = 1,
    ) -> _RoutingCache:
        state = prefix.reducer.state
        pending = self._ready_pending_inputs(prefix.stored_inputs, now)
        format_floor = self._level_floor(
            prefix.records,
            prefix.stored_inputs,
            minimum_level=minimum_level,
        )
        bootstrap = self._bootstrap_requirements(prefix)
        if state.active_turn_id is not None:
            turn = state.turns[state.active_turn_id].start
            base = normalize_requirements(turn._payload["requirements"])
            capabilities = set(base.capabilities)
            level = base.level
            applied_selection = (
                InboxItem(**state.model_selection._payload["input"]) if state.model_selection is not None else None
            )
            applied_requirements = self._selection_requirements(applied_selection) if applied_selection is not None else None
            if applied_requirements is not None:
                level = max(level, applied_requirements.level)
                capabilities.update(applied_requirements.capabilities)
            for stored in pending:
                selection = stored.item
                if selection.kind != "model_selection":
                    continue
                if selection.feature_level == 2 and self._inbox_requirements(selection).handler_version != base.handler_version:
                    continue
                requirements = self._selection_requirements(selection)
                if requirements is not None:
                    level = max(level, requirements.level)
                    capabilities.update(requirements.capabilities)
            return _RoutingCache(
                SessionRequirements.normalized(
                    level=max(level, format_floor),
                    handler_version=base.handler_version,
                    capabilities=capabilities,
                ),
                state.active_turn_id,
                level,
            )

        stored_input_sequences = {stored.item.input_id: stored.input_seq for stored in prefix.stored_inputs}
        queued_candidates: list[tuple[int, str, SessionRequirements]] = []
        for input_id, applied in state.applied_inputs.items():
            if input_id in state.admitted_inputs or not queued_input_starts_turn(applied):
                continue
            input_item = InboxItem(**applied._payload["input"])
            if input_id not in stored_input_sequences:
                raise LedgerIntegrityError("queued input is missing from the stored inbox prefix")
            queued_candidates.append(
                (
                    stored_input_sequences[input_id],
                    input_item.kind,
                    self._inbox_requirements(input_item, inherited=bootstrap),
                )
            )

        pending_candidates: list[tuple[int, int, str, SessionRequirements]] = []
        for stored in pending:
            input_item = stored.item
            if input_item.kind in {"user", "follow_up"} and input_item.target_turn_id is None:
                pending_candidates.append(
                    (
                        stored.input_seq,
                        input_item.available_ms,
                        input_item.kind,
                        self._inbox_requirements(input_item, inherited=bootstrap),
                    )
                )

        effective_selection = InboxItem(**state.model_selection._payload["input"]) if state.model_selection is not None else None
        for stored in pending:
            selection = stored.item
            if selection.kind != "model_selection":
                continue
            publication_candidates = [
                (kind, requirements) for input_seq, kind, requirements in queued_candidates if input_seq < stored.input_seq
            ]
            publication_candidates.extend(
                (kind, requirements)
                for input_seq, available_ms, kind, requirements in pending_candidates
                if input_seq < stored.input_seq and available_ms <= stored.received_ms
            )
            route = (
                min(enumerate(publication_candidates), key=lambda entry: (entry[1][0] != "user", entry[0]))[1][1]
                if publication_candidates
                else bootstrap
            )
            if selection.feature_level == 2 and self._inbox_requirements(selection).handler_version != route.handler_version:
                continue
            effective_selection = None if selection.payload["selection"] is None else selection

        candidates = [(kind, requirements) for _, kind, requirements in queued_candidates]
        candidates.extend((kind, requirements) for _, _, kind, requirements in pending_candidates)
        route = min(enumerate(candidates), key=lambda entry: (entry[1][0] != "user", entry[0]))[1][1] if candidates else bootstrap
        selected = self._selection_requirements(effective_selection) if effective_selection is not None else None
        capabilities = set(route.capabilities)
        level = route.level
        if selected is not None:
            capabilities.update(selected.capabilities)
            level = max(level, selected.level)
        return _RoutingCache(
            SessionRequirements.normalized(
                level=max(level, format_floor),
                handler_version=route.handler_version,
                capabilities=capabilities,
            ),
            None,
            level,
        )

    def _routing_cache_from_row(self, row: tuple[Any, ...]) -> _RoutingCache:
        if len(row) >= 11:
            level, handler, capabilities, active_turn_id = row[6], row[8], row[9], row[10]
        else:
            level, handler, capabilities, active_turn_id = row[3], row[4], row[5], row[6]
        if handler is not None and (not isinstance(handler, str) or not handler):
            raise LedgerIntegrityError("session handler_version is invalid")
        if active_turn_id is not None and (not isinstance(active_turn_id, str) or not active_turn_id):
            raise LedgerIntegrityError("session active_turn_id is invalid")
        return _RoutingCache(
            SessionRequirements(_level(level, "session level"), handler, self._session_capabilities(capabilities)),
            active_turn_id,
        )

    def _routing_cache_is_understated(
        self,
        actual: _RoutingCache,
        expected: _RoutingCache,
        *,
        minimum_level: int,
    ) -> bool:
        return (
            actual.requirements.level < minimum_level
            or (
                expected.requirements.handler_version is not None
                and actual.requirements.handler_version != expected.requirements.handler_version
            )
            or not set(expected.requirements.capabilities).issubset(actual.requirements.capabilities)
        )

    def _routing_projection_times(
        self,
        session_id: str,
        *,
        head: int,
        records: tuple[StoredRecord, ...],
        stored_inputs: tuple[StoredInput, ...],
        now: int,
    ) -> tuple[int, ...]:
        """Return every evidence cut at which this captured prefix can legitimately project."""

        times = {now}
        mode = self._ledger_integrity_mode()
        if mode == "signed":
            checkpoint_rows = self._rows(
                "SELECT signed_ms FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq<=%s",
                (self.tenant_id, session_id, head),
            )
            if not checkpoint_rows:
                raise LedgerIntegrityError("signed session has no authenticated checkpoint")
            lower_bound = max(row[0] for row in checkpoint_rows)
            for stored in stored_inputs:
                proof = stored.item.admission_proof
                if proof is None:
                    raise LedgerIntegrityError("signed input has no admission proof")
                lower_bound = max(lower_bound, proof["admission_signed_ms"], stored.received_ms)
            times.add(lower_bound)
        else:
            # Explicit weak modes trust their SQL metadata boundary. Records and
            # inputs are still hash-checked by the same captured-prefix reader.
            lower_bound = max(
                max((stored.created_ms for stored in records), default=0),
                max((stored.received_ms for stored in stored_inputs), default=0),
            )
            times.add(lower_bound)

        for stored in stored_inputs:
            available = stored.item.available_ms
            times.add(available)
        return tuple(sorted(moment for moment in times if lower_bound <= moment <= now))

    def _deleted_child_changes_projection(self, prefix: _Prefix, actual: _RoutingCache, expected: _RoutingCache) -> bool:
        """Allow a verified live-child routing cut to be refreshed after deletion."""

        if (
            actual.requirements.level <= expected.requirements.level
            or actual.requirements.handler_version != expected.requirements.handler_version
            or actual.requirements.capabilities != expected.requirements.capabilities
        ):
            return False
        child_ids = {stored.item.payload["session_id"] for stored in prefix.stored_inputs if stored.item.kind == "child_result"}
        child_ids.update(
            stored.record._payload["input"]["payload"]["session_id"]
            for stored in prefix.records
            if stored.record.kind == "input_applied" and stored.record._payload["input"]["kind"] == "child_result"
        )
        return any(self.child_is_deleted(session_id) for session_id in child_ids)

    def _require_routing_cache(
        self,
        session_id: str,
        row: tuple[Any, ...],
        *,
        prefix: _Prefix | None = None,
    ) -> _RoutingCache:
        routing_prefix = prefix or self._prefix(session_id, row, require_worker=False)
        expected_time = self._now()
        actual = self._routing_cache_from_row(row)
        expected = self._derive_routing_cache(routing_prefix, now=expected_time)
        minimum_level = self._level_floor(routing_prefix.records, routing_prefix.stored_inputs)
        matches = actual.requirements == expected.requirements and actual.active_turn_id == expected.active_turn_id
        if not matches:
            projection_times = self._routing_projection_times(
                session_id,
                head=row[0],
                records=routing_prefix.records,
                stored_inputs=routing_prefix.stored_inputs,
                now=expected_time,
            )
            stale = self._deleted_child_changes_projection(routing_prefix, actual, expected) or any(
                (
                    historical := self._derive_routing_cache(
                        routing_prefix,
                        now=moment,
                    )
                ).requirements
                == actual.requirements
                and historical.active_turn_id == actual.active_turn_id
                for moment in projection_times
            )
            if not stale:
                if self._routing_cache_is_understated(actual, expected, minimum_level=minimum_level):
                    raise LedgerIntegrityError("session routing cache understates the verified requirements")
                raise LedgerIntegrityError("session routing cache does not match the verified requirements")
            # The row lock makes this a compare-and-refresh of one evidence prefix.
            # The cache was a legal projection at a cut after the current source
            # boundary; advance it to database time before worker compatibility.
            self._write_routing_cache(session_id, expected, minimum_level=minimum_level)
        return expected

    def _write_routing_cache(self, session_id: str, cache: _RoutingCache, *, minimum_level: int = 1) -> None:
        _positive(minimum_level, "minimum session level")
        persisted_level = max(cache.requirements.level, minimum_level)
        self._rows(
            "UPDATE sk_session SET level=%s,handler_version=%s,required_capabilities=%s,active_turn_id=%s "
            "WHERE tenant_id=%s AND session_id=%s",
            (
                persisted_level,
                cache.requirements.handler_version,
                canonical_json_bytes(list(cache.requirements.capabilities), "session required_capabilities"),
                cache.active_turn_id,
                self.tenant_id,
                session_id,
            ),
        )

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

    def _verify_checkpoints(
        self,
        session_id: str,
        chains: tuple[str, ...],
        *,
        through: int,
        verifier: Any = _CONFIGURED_VERIFIER,
        signed: bool | None = None,
    ) -> None:
        signed, verifier = self._resolve_ledger_verifier(verifier, signed)
        if signed and verifier is None:
            raise LedgerIntegrityError("ledger verifier is unavailable for checkpoint verification")
        rows = self._rows(
            "SELECT seq,chain_digest,key_id,signature,signed_ms FROM sk_checkpoint "
            "WHERE tenant_id=%s AND session_id=%s AND seq<=%s ORDER BY seq",
            (self.tenant_id, session_id, through),
        )
        if not signed:
            if rows:
                raise LedgerIntegrityError(f"ledger verifier is required to verify stored checkpoint at seq {rows[0][0]}")
            return
        assert verifier is not None
        for expected_seq in range(1, len(chains) + 1):
            if expected_seq > len(rows):
                raise LedgerIntegrityError(f"ledger checkpoint is missing at seq {expected_seq}")
            row = rows[expected_seq - 1]
            seq, stored_chain, key_id, signature, signed_ms = row
            if seq != expected_seq:
                raise LedgerIntegrityError(f"ledger checkpoint sequence gap or reorder at seq {expected_seq}")
            if stored_chain != chains[seq - 1]:
                raise LedgerIntegrityError(f"ledger checkpoint does not match the record chain at seq {seq}")
            if not isinstance(key_id, str) or not key_id or not isinstance(signature, bytes):
                raise LedgerIntegrityError(f"ledger checkpoint has invalid signing material at seq {seq}")
            try:
                _natural(signed_ms, "checkpoint signed_ms")
            except ValueError as exc:
                raise LedgerIntegrityError(f"ledger checkpoint has invalid signed_ms at seq {seq}") from exc
            message = _checkpoint_message(self.tenant_id, session_id, seq, stored_chain, key_id, signed_ms)
            if not verifier.verify(key_id, message, bytes(signature)):
                raise LedgerIntegrityError(f"ledger checkpoint signature is invalid at seq {seq}")
        if len(rows) > len(chains):
            raise LedgerIntegrityError(f"ledger checkpoint outlives the record chain at seq {len(chains) + 1}")

    def _commit_payload(self, body: bytes, sequence_body: bytes) -> _CommitPayload:
        """Validate immutable commit bytes once while SQL metadata is checked on every pass."""

        if not isinstance(sequence_body, bytes):
            raise LedgerIntegrityError("commit metadata is not canonical JCS")
        key = (body, sequence_body)
        cache = self._commit_parse_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
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
        if type(head_seq) is not int or head_seq < 0:
            raise LedgerIntegrityError("commit head_seq is invalid")

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

        parsed = _CommitPayload(
            head_seq=head_seq,
            head_chain_digest=manifest["head_chain_digest"],
            records=tuple(manifest_records),
            sequences=tuple(ordered_sequences),
            consume_input_ids=tuple(consume_ids),
            consumers=tuple(consumers),
        )
        if len(cache) >= _COMMIT_PARSE_CACHE_LIMIT:
            cache.clear()
        cache[key] = parsed
        return parsed

    def _verify_commits(
        self,
        session_id: str,
        chains: tuple[str, ...],
        *,
        through: int,
        record_rows: list[tuple[Any, ...]],
        applied_inputs: dict[str, int],
    ) -> None:
        if len(record_rows) != len(chains):
            raise LedgerIntegrityError("ledger record count does not match the session head")
        records_by_seq = {row[0]: row for row in record_rows}
        records_by_id = {row[1]: row for row in record_rows}
        if len(records_by_seq) != len(record_rows) or len(records_by_id) != len(record_rows):
            raise LedgerIntegrityError("duplicate ledger record identity")

        # Fetch the inventories in one statement. Input consumption is bounded
        # by record sequence here, independently of the inbox read watermark.
        inventory = self._rows(
            "SELECT 2 AS row_kind,commit_id AS identity,body,digest,record_sequences,head_seq AS seq,created_ms "
            "FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND head_seq<=%s "
            "UNION ALL SELECT 0,input_id,NULL,NULL,NULL,consumed_seq,NULL FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s AND consumed_seq IS NOT NULL AND consumed_seq<=%s "
            "UNION ALL SELECT 1,consumer,NULL,NULL,NULL,NULL,NULL FROM sk_consumer "
            "WHERE tenant_id=%s AND session_id=%s ORDER BY row_kind,created_ms,identity",
            (self.tenant_id, session_id, through, self.tenant_id, session_id, through, self.tenant_id, session_id),
        )
        consumed_by_id: dict[str, Any] = {}
        committed_consumers: set[str] = set()
        claimed: dict[int, tuple[str, str, int]] = {}
        for row_kind, identity, body, checksum, sequence_body, seq, created_ms in inventory:
            if row_kind == 0:
                if identity in consumed_by_id:
                    raise LedgerIntegrityError("commit consumed input identity is duplicated")
                consumed_by_id[identity] = seq
                continue
            if row_kind == 1:
                committed_consumers.add(identity)
                continue
            commit_id, stored_head = identity, seq
            _identity(commit_id)
            _natural(stored_head, "commit head_seq")
            _natural(created_ms, "commit created_ms")
            if not isinstance(body, bytes) or sha256(body).hexdigest() != checksum:
                raise LedgerIntegrityError("commit body digest mismatch")
            payload = self._commit_payload(body, sequence_body)
            head_seq = payload.head_seq
            if head_seq > len(chains):
                raise LedgerIntegrityError("commit head_seq is invalid")
            expected_head_chain = _initial_chain_digest(self.tenant_id, session_id) if head_seq == 0 else chains[head_seq - 1]
            if payload.head_chain_digest != expected_head_chain:
                raise LedgerIntegrityError("commit head_chain_digest does not match the original head")

            for position, ((record_id, record_digest), (sequence_id, seq)) in enumerate(
                zip(payload.records, payload.sequences, strict=True)
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

            resulting_head = max([head_seq, *(seq for _, seq in payload.sequences)])
            if stored_head != resulting_head:
                raise LedgerIntegrityError("commit head_seq column does not match its body")

            for input_id in payload.consume_input_ids:
                consumed_seq = consumed_by_id.get(input_id)
                if type(consumed_seq) is not int:
                    raise LedgerIntegrityError("commit consumed input is missing or unconsumed")
                applied = records_by_seq.get(consumed_seq)
                if applied is None or applied[3] != commit_id:
                    raise LedgerIntegrityError("commit consumed input does not match an applied record")
            for consumer in payload.consumers:
                if consumer not in committed_consumers:
                    raise LedgerIntegrityError("commit creation consumer is missing")

        if set(claimed) != set(range(1, len(chains) + 1)):
            raise LedgerIntegrityError("ledger sequence is not fully claimed by commits")
        for input_id, applied_seq in applied_inputs.items():
            if consumed_by_id.get(input_id) != applied_seq:
                raise InputAdmissionProofError(f"stored input consumption does not match its proof at seq {applied_seq}")
        for input_id, consumed_seq in consumed_by_id.items():
            if input_id not in applied_inputs:
                raise InputAdmissionProofError(f"stored input was consumed without an applied record at seq {consumed_seq}")

    def _verify_head_bounds(self, session_id: str) -> None:
        """Reject append-only rows that are not covered by the session head."""

        head, checkpoint, record, commit = self._one(
            "SELECT s.head_seq,"
            "(SELECT COALESCE(MAX(c.seq),0) FROM sk_checkpoint c "
            "WHERE c.tenant_id=%s AND c.session_id=%s),"
            "(SELECT COALESCE(MAX(r.seq),0) FROM sk_record r "
            "WHERE r.tenant_id=%s AND r.session_id=%s),"
            "(SELECT COALESCE(MAX(k.head_seq),0) FROM sk_commit k "
            "WHERE k.tenant_id=%s AND k.session_id=%s) "
            "FROM sk_session s WHERE s.tenant_id=%s AND s.session_id=%s",
            (self.tenant_id, session_id, self.tenant_id, session_id, self.tenant_id, session_id, self.tenant_id, session_id),
        )
        if checkpoint > head:
            raise LedgerIntegrityError(f"ledger checkpoint outlives the session head at seq {checkpoint}")
        if record > head:
            raise LedgerIntegrityError(f"ledger record outlives the session head at seq {record}")
        if commit > head:
            raise LedgerIntegrityError(f"ledger commit outlives the session head at seq {commit}")

    def _stored_input(
        self,
        session_id: str,
        row: tuple[Any, ...],
        *,
        verifier: Any = _CONFIGURED_VERIFIER,
        signed: bool | None = None,
        enforce_reader: bool = True,
    ) -> StoredInput:
        """Validate and reconstruct one inbox row before any content hydration."""

        body, seq, received, consumed, checksum, key_id, signature, signed_ms, input_id, kind, available_ms = row
        if sha256(body).hexdigest() != checksum:
            raise InputAdmissionProofError(f"stored input digest mismatch at input_seq {seq}")
        base = InboxItem.parse(body)
        if (input_id, kind, available_ms) != (base.input_id, base.kind, base.available_ms):
            raise InputAdmissionProofError(f"stored input metadata does not match its body at input_seq {seq}")
        if enforce_reader:
            self._require_reader_level(base.feature_level)
        if base.admission_proof is not None:
            raise InputAdmissionProofError(f"stored input body duplicates admission proof at input_seq {seq}")
        columns = (key_id, signature, signed_ms)
        if any(value is None for value in columns) and not all(value is None for value in columns):
            raise InputAdmissionProofError(f"stored input admission proof is incomplete at input_seq {seq}")
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
            raise InputAdmissionProofError(f"input actor belongs to another tenant at input_seq {seq}")

        signed, verifier = self._resolve_ledger_verifier(verifier, signed)
        if proof is not None and not signed:
            raise InputAdmissionProofError(f"ledger verifier is required for input admission proof at input_seq {seq}")
        if signed:
            if verifier is None or proof is None:
                raise InputAdmissionProofError(f"signed input admission proof is missing at input_seq {seq}")
            admission_signed_ms = proof["admission_signed_ms"]
            if type(admission_signed_ms) is not int:
                raise InputAdmissionProofError(f"admission signed_ms is invalid at input_seq {seq}")
            message = _input_admission_message(
                self.tenant_id,
                session_id,
                item.input_id,
                seq,
                received,
                checksum,
                proof["admission_key_id"],
                admission_signed_ms,
            )
            if not verifier.verify(proof["admission_key_id"], message, bytes(signature)):
                raise InputAdmissionProofError(f"signed input admission signature is invalid at input_seq {seq}")
        return StoredInput(item, seq, received, consumed)

    def _stored_inputs(self, session_id: str, inbox_seq: int) -> tuple[StoredInput, ...]:
        """Validate the complete contiguous inbox snapshot used by a ledger boundary."""

        # The session row and inbox are updated in one transaction. Cap this
        # second statement at the watermark captured with the worker level so a
        # concurrent level-raising push cannot enter this snapshot.
        rows = self._rows(
            f"SELECT {_STORED_INBOX_COLUMNS} "
            "FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_seq<=%s ORDER BY input_seq",
            (self.tenant_id, session_id, inbox_seq),
        )
        for expected_seq in range(1, inbox_seq + 1):
            if expected_seq > len(rows):
                raise InputAdmissionProofError(f"stored input is missing at input_seq {expected_seq}")
            if rows[expected_seq - 1][1] != expected_seq:
                raise InputAdmissionProofError(f"stored input sequence gap or reorder at input_seq {expected_seq}")
        if len(rows) != inbox_seq:
            raise InputAdmissionProofError(f"stored input count exceeds session inbox_seq at input_seq {inbox_seq + 1}")
        return tuple(self._stored_input(session_id, row) for row in rows)

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
            tx = SQLSessionTx(self)
            yield tx
            tx._finalize_lease_fence()

    def _lock(self, session_id: str) -> tuple[Any, ...]:
        return self._session_row(session_id, for_update=True)

    def _require_create_isolation(self) -> None:
        """Require isolation that can see a tombstone after locking its session identity."""

    @staticmethod
    def _require_live_row(row: tuple[Any, ...]) -> None:
        if len(row) < 8 or row[7] != "live":
            raise SessionNotLive("session is fenced for deletion")

    def _require_live(self, session_id: str, *, for_update: bool = False) -> tuple[Any, ...]:
        row = self._session_row(session_id, for_update=for_update)
        self._require_live_row(row)
        return row

    def _session_row(self, session_id: str, *, for_update: bool = False) -> tuple[Any, ...]:
        return self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms,level,deletion_state,"
            "handler_version,required_capabilities,active_turn_id "
            f"FROM sk_session WHERE tenant_id=%s AND session_id=%s"
            f"{self.lock_clause if for_update else ''}",
            (self.tenant_id, session_id),
        )

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
        prior = self._cached_record_for_body(cached, seq, body)
        if prior is not None:
            # Fresh SQL metadata wraps the same validated session/sequence/body.
            return StoredRecord(prior, seq, commit, epoch, created)
        record = Record.parse(body) if messages is None else Record._parse(body, messages)
        if record.encode() != body:
            raise LedgerIntegrityError("stored record body is not canonical JCS")
        return StoredRecord(record, seq, commit, epoch, created)

    @staticmethod
    def _cached_record_for_body(cached: _Prefix | None, seq: int, body: bytes) -> Record | None:
        """Return a validated record only for its exact immutable canonical bytes."""

        if cached is None or not 0 < seq <= len(cached.records):
            return None
        record = cached.records[seq - 1].record
        return record if record.encode() == body else None

    def _verified_records(
        self,
        session_id: str,
        through: int,
        *,
        verifier: Any = _CONFIGURED_VERIFIER,
        signed: bool | None = None,
        parsed_records: dict[int, Record] | None = None,
    ) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
        explicitly_unsigned = signed is False
        signed, verifier = self._resolve_ledger_verifier(verifier, signed)
        if explicitly_unsigned and self._ledger_integrity_mode() == "unconfigured":
            raise LedgerIntegrityError("ledger integrity requires an explicit signed signer or allow_unsigned hash-only mode")
        previous = _initial_chain_digest(self.tenant_id, session_id)
        chains: list[str] = []
        stored_rows: list[tuple[Any, ...]] = []
        commit_record_rows: list[tuple[Any, ...]] = []
        applied_inputs: dict[str, int] = {}
        cached = self._cached_prefix(session_id)
        cached_chain_count = len(cached.chains) if cached is not None else 0
        cursor = 0
        while cursor < through:
            rows = self._rows(
                "SELECT body,seq,commit_id,writer_epoch,created_ms,digest,chain_digest,record_id,"
                "schema_version,kind,turn_id,operation_id,attempt,commit_position,commit_digest "
                "FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq>%s AND seq<=%s "
                "ORDER BY seq LIMIT %s",
                (self.tenant_id, session_id, cursor, through, _RECORD_READ_PAGE_SIZE),
            )
            if not rows:
                raise LedgerIntegrityError(f"log sequence gap at seq {cursor + 1}")
            for expected_seq, row in enumerate(rows, cursor + 1):
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
                    commit_digest,
                ) = row
                if seq != expected_seq:
                    raise LedgerIntegrityError(f"log sequence gap or reorder at seq {expected_seq}; found {seq}")
                record = self._cached_record_for_body(cached, seq, body)
                if record is None:
                    cached_chain_count = 0
                    actual_digest = sha256(body).hexdigest()
                else:
                    actual_digest = record.digest
                if actual_digest != checksum:
                    raise LedgerIntegrityError(f"stored record digest mismatch at seq {seq}")
                if record is None and parsed_records is not None:
                    try:
                        parsed = Record.parse(body)
                    except Exception:
                        # Report parser failures in the fold after it verifies the ledger and inbox.
                        pass
                    else:
                        if parsed.encode() == body:
                            record = parsed
                            parsed_records[seq] = parsed
                if record is None:
                    try:
                        value = json.loads(body)
                        if canonical_json_bytes(value, "stored record body") != body:
                            raise LedgerIntegrityError(f"stored record body is not canonical JCS at seq {seq}")
                    except (TypeError, ValueError) as exc:
                        raise LedgerIntegrityError(f"stored record body is not canonical JCS at seq {seq}") from exc
                    if not isinstance(value, dict):
                        raise LedgerIntegrityError(f"stored record SQL identity does not match its body at seq {seq}")
                    body_identity = (
                        value.get("record_id"),
                        value.get("session_id"),
                        value.get("schema_version"),
                        value.get("kind"),
                        value.get("turn_id"),
                        value.get("operation_id"),
                        value.get("attempt"),
                    )
                else:
                    body_identity = (
                        record.record_id,
                        record.session_id,
                        record.schema_version,
                        record.kind,
                        record.turn_id,
                        record.operation_id,
                        record.attempt,
                    )
                if body_identity != (
                    record_id,
                    session_id,
                    schema_version,
                    kind,
                    turn_id,
                    operation_id,
                    attempt,
                ):
                    raise LedgerIntegrityError(f"stored record SQL identity does not match its body at seq {seq}")
                if kind == "input_applied":
                    payload = value.get("payload") if record is None else record._payload
                    item = payload.get("input") if isinstance(payload, dict) else None
                    input_id = item.get("input_id") if isinstance(item, dict) else None
                    if not isinstance(input_id, str) or input_id in applied_inputs:
                        raise InputAdmissionProofError(f"applied input identity is invalid at seq {seq}")
                    applied_inputs[input_id] = seq
                try:
                    _natural(epoch, "record writer_epoch")
                    _natural(created_ms, "record created_ms")
                    _natural(commit_position, "record commit_position")
                except ValueError as exc:
                    raise LedgerIntegrityError(f"stored record metadata is invalid at seq {seq}") from exc
                # A cached chain is valid only through the exact retained body prefix.
                if cached is not None and seq <= cached_chain_count:
                    actual_chain = cached.chains[seq - 1]
                else:
                    actual_chain = _extend_chain_digest(previous, expected_seq, actual_digest)
                if stored_chain != actual_chain:
                    raise LedgerIntegrityError(f"stored record chain digest mismatch at seq {seq}")
                previous = actual_chain
                chains.append(actual_chain)
                stored_rows.append((body, seq, commit_id, epoch, created_ms, checksum))
                commit_record_rows.append((seq, record_id, checksum, commit_id, commit_digest, commit_position))
            cursor = expected_seq
        if cursor != through:
            raise LedgerIntegrityError(f"log sequence gap at seq {cursor + 1}")
        # ``through`` is the caller's logical snapshot boundary. These tables are
        # append-only and committed with the session head, so bounded queries
        # validate that exact prefix without holding a transaction across hydration.
        self._verify_checkpoints(session_id, tuple(chains), through=through, verifier=verifier, signed=signed)
        self._verify_commits(
            session_id, tuple(chains), through=through, record_rows=commit_record_rows, applied_inputs=applied_inputs
        )
        self._verify_head_bounds(session_id)
        return stored_rows, tuple(chains)

    def _prefix(self, session_id: str, row: tuple[Any, ...], *, require_worker: bool = True) -> _Prefix:
        head, epoch = row[0], row[2]
        cached_level = row[6] if len(row) >= 10 else row[3]
        if require_worker:
            self._require_worker_accepts_row(row)
        if head == 0:
            raise LedgerIntegrityError("log sequence gap")
        transaction_state = getattr(self, "_transaction_state", None)
        if transaction_state is not None and transaction_state.depth:
            self._read_snapshot_prefix = None
        snapshot_prefix = getattr(self, "_read_snapshot_prefix", None)
        if (
            getattr(self, "_read_snapshot_active", False)
            and snapshot_prefix is not None
            and snapshot_prefix.tenant_id == self.tenant_id
            and snapshot_prefix.session_id == session_id
            and snapshot_prefix.head_seq == head
            and snapshot_prefix.epoch == epoch
            and snapshot_prefix.inbox_seq == row[1]
        ):
            self._verify_level_cache(snapshot_prefix.records, snapshot_prefix.stored_inputs, cached_level)
            return snapshot_prefix
        parsed_records: dict[int, Record] = {}
        stored_rows, chains = self._verified_records(session_id, head, parsed_records=parsed_records)
        stored_inputs = self._stored_inputs(session_id, row[1])
        cached = self._cached_prefix(session_id)
        retained_records: tuple[StoredRecord, ...] = ()
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
            else:
                # The chain covers record bodies. SQL supplies the wrapper metadata.
                retained_records = tuple(
                    stored
                    if stored.commit_id == commit_id and stored.writer_epoch == writer_epoch and stored.created_ms == created_ms
                    else StoredRecord(stored.record, seq, commit_id, writer_epoch, created_ms)
                    for stored, (_, seq, commit_id, writer_epoch, created_ms, _) in zip(cached.records, stored_rows, strict=False)
                )
        through = len(retained_records)
        if cached is not None and through == head:
            cached = _Prefix(
                self.tenant_id,
                session_id,
                epoch,
                cached.head_digest,
                cached.reducer,
                retained_records,
                stored_inputs,
                row[1],
                chains,
            )
            self._verify_level_cache(cached.records, cached.stored_inputs, cached_level)
            self._retain_prefix(cached)
            if getattr(self, "_read_snapshot_active", False):
                self._read_snapshot_prefix = cached
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
                parsed = parsed_records.get(row_data[1])
                if parsed is None:
                    stored = self._stored(session_id, row_data, reducer._messages)
                else:
                    stored = StoredRecord(parsed, row_data[1], row_data[2], row_data[3], row_data[4])
                stored.record._share_messages(reducer._messages)
                records.append(stored)
                bodies.append(row_data[0])
                cursor = stored.seq
        consumed = [
            stored.item for stored in stored_inputs if stored.consumed_seq is not None and through < stored.consumed_seq <= head
        ]
        reducer.extend((r.record for r in records), consumed_inputs=consumed, bodies=bodies)
        prefix = _Prefix(
            self.tenant_id,
            session_id,
            epoch,
            head_digest,
            reducer,
            retained_records + tuple(records),
            stored_inputs,
            row[1],
            chains,
        )
        self._verify_level_cache(prefix.records, prefix.stored_inputs, cached_level)
        self._retain_prefix(prefix)
        if getattr(self, "_read_snapshot_active", False):
            self._read_snapshot_prefix = prefix
        return prefix

    def read_state(
        self,
        session_id: str,
        *,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
        row = self._require_live(session_id)
        self._require_reader_row(row)
        if expected_head is not None:
            _require_expected_head(
                expected_head,
                tenant_id=self.tenant_id,
                session_id=session_id,
                head_seq=row[0],
                chain_digest=self._chain_anchor(session_id, row[0]),
            )
        prefix = self._prefix(session_id, row, require_worker=False)
        return prefix.reducer.snapshot(), prefix.records, row[1]

    def _head(self, session_id: str) -> tuple[Any, ...]:
        return self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,level,handler_version,required_capabilities,active_turn_id "
            "FROM sk_session WHERE tenant_id=%s AND session_id=%s",
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
        self._require_configured_ledger("schedule rebuild")
        self._require_live(session_id)
        while True:
            captured = self._head(session_id)
            self._require_worker_accepts_row(captured)
            self._fold_cache = None
            reducer = self._prefix(session_id, captured).reducer
            snapshot = reducer.snapshot()
            with self._transaction():
                row = self._lock(session_id)
                self._require_live_row(row)
                self._require_worker_accepts_row(row)
                if row[:3] != captured[:3]:
                    continue
                self._schedule(session_id, reducer.state)
                return snapshot

    def read(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        through_seq: int | None = None,
        limit: int = 1024,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> ReadPage:
        _positive(limit, "limit")
        _natural(after_seq, "after_seq")
        if through_seq is not None:
            _natural(through_seq, "through_seq")
        row = self._require_live(session_id)
        self._require_reader_row(row)
        # Records are immutable: cap by the captured head without locking writers.
        head, inbox = row[:2]
        if expected_head is not None:
            _require_expected_head(
                expected_head,
                tenant_id=self.tenant_id,
                session_id=session_id,
                head_seq=head,
                chain_digest=self._chain_anchor(session_id, head),
            )
        self._require_input_integrity_gate()
        boundary = min(head, through_seq) if through_seq is not None else head
        if head == 0:
            self._stored_inputs(session_id, inbox)
            return ReadPage(head, inbox, ())
        prefix = self._prefix(session_id, row, require_worker=False)
        return ReadPage(head, inbox, prefix.records[after_seq:boundary][:limit])

    def verify_ledger(
        self,
        session_id: str,
        *,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> tuple[int, str]:
        """Verify the complete stored ledger and the optional external head anchor."""

        row = self._session_row(session_id)
        forgotten = row[7] == "forgotten"
        if not forgotten:
            self._require_live_row(row)
        self._require_reader_row(row)
        head, inbox = row[:2]
        chain_digest = self._chain_anchor(session_id, head)
        self._require_input_integrity_gate()
        if forgotten:
            tombstone = self._tombstone(session_id)
            if tombstone is None or tombstone.state is not TombstoneState.COMPLETE:
                raise LedgerIntegrityError("forgotten session requires a complete tombstone")
            _, allow_unsigned, verifier = self._input_admission_policy()
            self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
            if (tombstone.final_seq, tombstone.chain_digest) != (head, chain_digest):
                raise LedgerIntegrityError("completed tombstone does not match the forgotten session head")
        _require_expected_head(
            expected_head,
            tenant_id=self.tenant_id,
            session_id=session_id,
            head_seq=head,
            chain_digest=chain_digest,
        )
        _stored_rows, chains = self._verified_records(session_id, head)
        self._stored_inputs(session_id, inbox)
        if not chains or chains[-1] != chain_digest:
            raise LedgerIntegrityError(f"ledger head chain mismatch at seq {head}")
        return head, chains[-1]

    def lease_status(self, session_id: str) -> tuple[int, int, int]:
        self._require_live(session_id)
        head, epoch, until, now = self._one(
            f"SELECT head_seq,lease_epoch,lease_until_ms,{self.clock_sql} FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        )
        return head, epoch, max(0, (until or now) - now)

    def read_input(self, session_id: str, input_id: str) -> StoredInput | None:
        row = self._require_live(session_id)
        self._require_reader_row(row)
        self._require_input_integrity_gate()
        self._stored_inputs(session_id, row[1])
        rows = self._rows(
            f"SELECT {_STORED_INBOX_COLUMNS} "
            "FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s AND input_id=%s AND input_seq<=%s",
            (self.tenant_id, session_id, input_id, row[1]),
        )
        if not rows:
            return None
        return self._stored_input(session_id, rows[0])

    def control_records(self, session_id: str, *, through_seq: int) -> tuple[StoredRecord, ...]:
        """Read consumed control facts for catalog filtering without folding turns."""
        _natural(through_seq, "through_seq")
        row = self._require_live(session_id)
        self._require_reader_row(row)
        head, inbox = row[:2]
        if through_seq > head:
            raise SequenceConflict("control records cannot read beyond the captured head")
        if through_seq == 0:
            return ()
        # ``through_seq`` may cut through a multi-record commit. Verify the
        # complete captured-head prefix, then select only indexed rows at or
        # before the caller's logical boundary.
        verified, _chains = self._verified_records(session_id, head)
        self._require_input_integrity_gate()
        self._stored_inputs(session_id, inbox)
        verified_by_seq = {stored_row[1]: stored_row for stored_row in verified}
        rows = self._rows(
            "SELECT r.body,r.seq,r.commit_id,r.writer_epoch,r.created_ms,r.digest "
            "FROM sk_inbox i JOIN sk_record r ON r.tenant_id=i.tenant_id "
            "AND r.session_id=i.session_id AND r.seq=i.consumed_seq "
            "WHERE i.tenant_id=%s AND i.session_id=%s AND i.kind='control' "
            "AND r.seq<=%s ORDER BY r.seq",
            (self.tenant_id, session_id, through_seq),
        )
        return tuple(self._stored(session_id, verified_by_seq[row[1]]) for row in rows)

    def projection_prefix(self, session_id: str) -> _Prefix:
        """Capture one verified session prefix for caller-owned projection delivery."""

        row = self._require_live(session_id)
        self._require_reader_row(row)
        self._require_input_integrity_gate()
        return self._prefix(session_id, row, require_worker=False)

    def _validate_projection_prefix(self, prefix: _Prefix, session_id: str, row: tuple[Any, ...]) -> None:
        if (prefix.tenant_id, prefix.session_id) != (self.tenant_id, session_id):
            raise LedgerIntegrityError("projection prefix belongs to another session")
        if prefix.head_seq != len(prefix.records) or prefix.head_seq > row[0]:
            raise LedgerIntegrityError("projection prefix does not match the captured head")
        if prefix.head_digest != self._chain_anchor(session_id, prefix.head_seq):
            raise LedgerIntegrityError("projection prefix chain no longer matches the stored head")

    def read_projection_page(
        self,
        prefix: _Prefix,
        session_id: str,
        *,
        after_seq: int = 0,
        through_seq: int | None = None,
        limit: int = 256,
    ) -> ReadPage:
        """Return one page from a prefix verified once by this delivery."""

        _positive(limit, "limit")
        _natural(after_seq, "after_seq")
        if through_seq is not None:
            _natural(through_seq, "through_seq")
        row = self._require_live(session_id)
        self._require_reader_row(row)
        self._validate_projection_prefix(prefix, session_id, row)
        boundary = min(prefix.head_seq, through_seq) if through_seq is not None else prefix.head_seq
        return ReadPage(prefix.head_seq, prefix.inbox_seq, prefix.records[after_seq:boundary][:limit])

    def projection_context(
        self,
        session_id: str,
        records: tuple[StoredRecord, ...],
        *,
        prefix: _Prefix | None = None,
    ) -> tuple[StoredRecord, ...]:
        """Indexed dependencies for a bounded event page from one verified delivery prefix."""

        row = self._require_live(session_id)
        self._require_reader_row(row)
        prefix = self.projection_prefix(session_id) if prefix is None else prefix
        self._validate_projection_prefix(prefix, session_id, row)
        through = max((stored.seq for stored in records), default=prefix.head_seq)
        if through > prefix.head_seq:
            raise SequenceConflict("projection records cannot read beyond the captured head")
        for stored in records:
            retained = prefix.records[stored.seq - 1] if 0 < stored.seq <= prefix.head_seq else None
            if retained is None or retained.record.encode() != stored.record.encode():
                raise LedgerIntegrityError("projection input does not match the verified ledger prefix")

        indexes = prefix.projection_indexes()
        found = {stored.seq: stored for stored in records}
        seen: set[tuple[str, int]] = set()

        def operation(operation_id: str, attempt: int) -> None:
            key = (operation_id, attempt)
            if key in seen:
                return
            seen.add(key)
            for stored in indexes.operations.get(key, ()):
                if stored.seq > through:
                    continue
                found[stored.seq] = stored
                if stored.record.kind == "op_planned" and stored.record._payload["op_kind"] != "model":
                    dependencies = stored.record._payload["dependencies"]
                    if dependencies:
                        operation(dependencies[0], 1)

        for stored in records:
            record = stored.record
            if record.operation_id is not None and record.attempt is not None:
                operation(record.operation_id, record.attempt)
            elif record.kind == "input_applied":
                data = record._payload
                item = data["input"]
                operation_id = data["target_operation_id"] or item["payload"].get("operation_id")
                if operation_id:
                    operation(operation_id, item["payload"].get("attempt", 1))

        for turn_id in {record.record.turn_id for record in records if record.record.turn_id}:
            start = indexes.turn_starts.get(turn_id)
            if start is not None and start.seq <= through:
                found[start.seq] = start

        for stored in records:
            record = stored.record
            if record.kind != "turn_ended":
                continue
            turn_id = record.turn_id
            if turn_id is None:
                raise LedgerIntegrityError("turn projection dependency has no verified start")
            dependencies = indexes.turn_dependencies.get(turn_id, ())
            for dependency in dependencies:
                if dependency.seq <= through:
                    found[dependency.seq] = dependency
            start = indexes.turn_starts.get(turn_id)
            if start is None:
                raise LedgerIntegrityError("turn projection dependency has no verified start")
            for input_id in start.record._payload["input_ids"]:
                if not input_id.startswith("compaction/"):
                    continue
                item = indexes.inputs_by_id.get(input_id)
                if item is None or item.consumed_seq is None or item.consumed_seq > through:
                    continue
                applied = prefix.records[item.consumed_seq - 1] if item.consumed_seq <= prefix.head_seq else None
                if applied is not None:
                    found[applied.seq] = applied
        return tuple(found[seq] for seq in sorted(found))

    def peek_inbox(
        self, session_id: str, *, through_input_seq: int | None = None, limit: int = 256, ready_only: bool = True
    ) -> tuple[StoredInput, ...]:
        _positive(limit, "limit")
        if through_input_seq is not None:
            _natural(through_input_seq, "through_input_seq")
        row = self._require_live(session_id)
        self._require_reader_row(row)
        self._require_input_integrity_gate()
        self._stored_inputs(session_id, row[1])
        through = row[1] if through_input_seq is None else min(through_input_seq, row[1])
        readiness = f"AND available_ms<={self.clock_sql} " if ready_only else ""
        rows = self._rows(
            f"SELECT {_STORED_INBOX_COLUMNS} "
            "FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s "
            "AND consumed_seq IS NULL " + readiness + "AND input_seq<=%s ORDER BY input_seq LIMIT %s",
            (self.tenant_id, session_id, through, limit),
        )
        return tuple(self._stored_input(session_id, row) for row in rows)

    def acquire(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None:
        self._require_configured_ledger("lease acquisition")
        _identity(owner)
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(session_id)
            self._require_live_row(row)
            if not self._worker_accepts_row(row) and not self._worker_accepts_ready_inbox(session_id):
                return None
            routing = self._require_routing_cache(session_id, row)
            if not self.worker_supports.accepts(
                level=routing.requirements.level,
                handler_version=routing.requirements.handler_version,
                capabilities=routing.requirements.capabilities,
            ):
                return None
            # Explicit-session drives may inspect idle state; runnable scans apply due-work readiness.
            now = self._now()  # Must be read after the row lock.
            if (row[4] is not None and row[4] > now) or row[5] > now:
                return None
            lease = Lease(session_id, owner, row[2] + 1, now + ttl_ms, self.tenant_id)
            self._rows(
                "UPDATE sk_session SET lease_epoch=%s,lease_owner=%s,lease_until_ms=%s WHERE tenant_id=%s AND session_id=%s",
                (lease.epoch, owner, lease.expires_at_ms, self.tenant_id, session_id),
            )
            return lease

    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll:
        self._require_configured_ledger("lease renewal")
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(lease.session_id)
            now = self._now()
            self._check_lease(lease.session_id, lease, row, now)
            renewed = Lease(lease.session_id, lease.owner, lease.epoch, now + ttl_ms, lease.tenant_id)
            updated = self._rows(
                "UPDATE sk_session SET lease_until_ms=%s WHERE tenant_id=%s AND session_id=%s "
                "AND lease_epoch=%s AND lease_owner=%s AND lease_until_ms=%s AND lease_until_ms>%s "
                "RETURNING session_id",
                (
                    renewed.expires_at_ms,
                    self.tenant_id,
                    lease.session_id,
                    lease.epoch,
                    lease.owner,
                    lease.expires_at_ms,
                    now,
                ),
            )
            if len(updated) != 1:
                raise LeaseLost("lease expired or fenced")
            controls = tuple(
                self._stored_input(lease.session_id, r).item
                for r in self._rows(
                    f"SELECT {_STORED_INBOX_COLUMNS} "
                    "FROM sk_inbox WHERE tenant_id=%s AND session_id=%s "
                    "AND consumed_seq IS NULL AND available_ms<=%s AND kind='control' ORDER BY input_seq",
                    (self.tenant_id, lease.session_id, now),
                )
            )
            return LeasePoll(renewed, now, row[1], controls)

    def release(self, lease: Lease) -> bool:
        self._require_configured_ledger("lease release")
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
                self._require_worker_accepts_row(row)
            next_drive_ms, drive_retry_at_ms = self._one(
                "SELECT next_drive_ms,drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, session_id),
            )
            future_due = [due for due in (next_drive_ms, drive_retry_at_ms) if due is not None and due > now]
            if future_due:
                return min(future_due) - now
            if next_drive_ms is not None and next_drive_ms <= now:
                return 0
            return None

    def defer_idle_drive(self, lease: Lease, *, poll_ms: int) -> bool:
        """Defer a stale idle schedule without changing execution facts or future due work."""
        return self._defer_idle_drive(lease, poll_ms=poll_ms)

    def _defer_idle_drive(self, lease: Lease, *, poll_ms: int, lease_guard: Callable[[], Lease] | None = None) -> bool:
        self._require_configured_ledger("idle drive deferral")
        _positive(poll_ms, "poll_ms")
        sid = lease.session_id
        self._require_live(sid)
        captured = self._head(sid)
        self._require_worker_accepts_row(captured)
        state = self._prefix(sid, captured).reducer.state
        with self._transaction():
            if lease_guard is not None:
                lease = lease_guard()
            row = self._lock(sid)
            now = self._now()
            self._check_lease(sid, lease, row, now)
            self._require_worker_accepts_row(row)
            scheduled = self._one(
                "SELECT next_drive_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (self.tenant_id, sid),
            )[0]
            if scheduled is None or scheduled > now:
                return False
            if (row[0], row[1], row[2], row[6], row[8], row[9], row[10]) != captured:
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
        self._require_configured_ledger("drive deferral")
        _positive(retry_after_ms, "retry_after_ms")
        with self._transaction():
            row = self._lock(lease.session_id)
            now = self._now()
            self._check_lease(lease.session_id, lease, row, now)
            self._require_worker_accepts_row(row)
            self._rows(
                "UPDATE sk_session SET drive_retry_at_ms=CASE WHEN drive_retry_at_ms>%s "
                "THEN drive_retry_at_ms ELSE %s END WHERE tenant_id=%s AND session_id=%s",
                (now + retry_after_ms, now + retry_after_ms, self.tenant_id, lease.session_id),
            )

    def defer_projection(self, session_id: str, consumer: str, *, retry_after_ms: int) -> None:
        """Back off one consumer without claiming or delaying execution."""
        self._require_configured_ledger("projection deferral")
        _positive(retry_after_ms, "retry_after_ms")
        with self._transaction():
            _, level, deletion_state, handler_version, required_capabilities = self._one(
                "SELECT c.last_seq,s.level,s.deletion_state,s.handler_version,s.required_capabilities "
                "FROM sk_consumer c JOIN sk_session s ON s.tenant_id=c.tenant_id AND s.session_id=c.session_id "
                f"WHERE c.tenant_id=%s AND c.session_id=%s AND c.consumer=%s{self.lock_clause}",
                (self.tenant_id, session_id, consumer),
            )
            if deletion_state != "live":
                raise SessionNotLive("session is fenced for deletion")
            self._require_reader_level(level)
            if handler_version is not None and (not isinstance(handler_version, str) or not handler_version):
                raise LedgerIntegrityError("session handler_version is invalid")
            self._session_capabilities(required_capabilities)
            until = self._now() + retry_after_ms
            self._rows(
                "UPDATE sk_consumer SET project_retry_at_ms=CASE WHEN project_retry_at_ms>%s "
                "THEN project_retry_at_ms ELSE %s END "
                "WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
                (until, until, self.tenant_id, session_id, consumer),
            )

    def is_runnable(self, session_id: str) -> bool:
        _identity(session_id)
        return bool(
            self._runnable_work(
                limit=1,
                after=None,
                apply_worker_filter=True,
                work_kinds=("drive",),
                projection_consumers=(),
                session_id=session_id,
            )
        )

    def _runnable_work(
        self,
        *,
        limit: int,
        after: WorkCursor | None,
        apply_worker_filter: bool,
        work_kinds: WorkKinds | None,
        projection_consumers: ProjectionConsumers | None,
        session_id: str | None = None,
    ) -> tuple[WorkItem, ...]:
        _positive(limit, "limit")
        if work_kinds is not None:
            if isinstance(work_kinds, str) or any(kind not in {"drive", "project"} for kind in work_kinds):
                raise ValueError("work_kinds must contain only 'drive' and 'project'")
            if len(work_kinds) != len(set(work_kinds)):
                raise ValueError("work_kinds must not contain duplicates")
            selected_kinds = set(work_kinds)
        else:
            selected_kinds = {"drive", "project"}
        if projection_consumers is not None:
            if isinstance(projection_consumers, str):
                raise TypeError("projection_consumers must be an iterable of identifiers")
            for consumer in projection_consumers:
                _identity(consumer)
            if len(projection_consumers) != len(set(projection_consumers)):
                raise ValueError("projection_consumers must not contain duplicates")
            if projection_consumers:
                placeholders = ",".join("%s" for _ in projection_consumers)
                project_filter = f"AND c.consumer IN ({placeholders})"
                project_filter_params: tuple[Any, ...] = tuple(projection_consumers)
            else:
                project_filter = "AND 1=0"
                project_filter_params = ()
        else:
            project_filter = ""
            project_filter_params = ()
        drive_filter = "" if "drive" in selected_kinds else "AND 1=0"
        if "project" not in selected_kinds:
            project_filter = "AND 1=0"
            project_filter_params = ()
        tenant_id = self._claim_tenant_id()
        if after is not None and tenant_id is not None and after.tenant_id != tenant_id:
            raise ValueError("work cursor belongs to another tenant")
        if session_id is not None:
            _identity(session_id)
        cursor = (
            (
                after.tenant_id,
                after.session_id,
                after.kind,
                after.consumer,
            )
            if after
            else ("", "", "", "")
        )
        if apply_worker_filter:
            candidate_predicate, worker_params = self._worker_candidate_predicate()
            worker_predicate = f"AND {candidate_predicate}"
        else:
            worker_predicate = ""
            worker_params = ()
        reader_predicate = "AND s.level<=%s"
        reader_params = (self.reader_level,)
        drive_scope = "" if tenant_id is None else "s.tenant_id=%s AND "
        drive_session_filter = "" if session_id is None else "AND s.session_id=%s "
        project_scope = "" if tenant_id is None else "c.tenant_id=%s AND "
        scope_params: tuple[Any, ...] = () if tenant_id is None else (tenant_id,)
        select_tenant = "s.tenant_id" if tenant_id is None else "%s"
        project_tenant = "c.tenant_id" if tenant_id is None else "%s"
        drive_select_params: tuple[Any, ...] = () if tenant_id is None else (tenant_id,)
        drive_session_params: tuple[Any, ...] = () if session_id is None else (session_id,)
        project_select_params: tuple[Any, ...] = () if tenant_id is None else (tenant_id,)
        rows = self._rows(
            f"""
            WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms), driving AS (
                SELECT {select_tenant} AS tenant_id,s.session_id,'drive' AS work_kind,'' AS consumer
                FROM sk_session s CROSS JOIN clock t
                WHERE {drive_scope}(s.tenant_id,s.session_id,'drive','')>(%s,%s,%s,%s)
                  {drive_session_filter}
                  AND s.schema_version='S1' AND s.deletion_state='live'
                  {worker_predicate}
                  {drive_filter}
                  AND {_DRIVE_RUNNABLE_SQL}
                ORDER BY s.tenant_id,s.session_id LIMIT %s
            ), projecting AS (
                SELECT {project_tenant} AS tenant_id,c.session_id,'project' AS work_kind,c.consumer FROM sk_consumer c
                JOIN sk_session s ON s.tenant_id=c.tenant_id AND s.session_id=c.session_id
                CROSS JOIN clock t
                WHERE {project_scope}(c.tenant_id,c.session_id,'project',c.consumer)>(%s,%s,%s,%s)
                  AND s.schema_version='S1' AND s.deletion_state='live'
                  {reader_predicate}
                  {project_filter}
                  AND c.last_seq<s.head_seq AND c.project_retry_at_ms<=t.ms
                ORDER BY c.tenant_id,c.session_id,c.consumer LIMIT %s
            ), work AS (
                SELECT * FROM driving UNION ALL SELECT * FROM projecting
            ) SELECT tenant_id,session_id,work_kind,consumer FROM work
            ORDER BY tenant_id,session_id,work_kind,consumer LIMIT %s
            """,
            (
                *drive_select_params,
                *scope_params,
                *cursor,
                *drive_session_params,
                *worker_params,
                limit,
                *project_select_params,
                *scope_params,
                *cursor,
                *reader_params,
                *project_filter_params,
                limit,
                limit,
            ),
        )
        return tuple(WorkItem(row[1], row[2], row[3], row[0]) for row in rows)

    def list_runnable(
        self,
        *,
        limit: int = 100,
        after: WorkCursor | None = None,
        work_kinds: WorkKinds | None = None,
        projection_consumers: ProjectionConsumers | None = None,
    ) -> tuple[WorkItem, ...]:
        """Return runnable work as a read-only hint; no row is claimed or locked."""

        return self._runnable_work(
            limit=limit,
            after=after,
            apply_worker_filter=False,
            work_kinds=work_kinds,
            projection_consumers=projection_consumers,
        )

    def _claim_drive(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None:
        candidate_predicate, worker_params = self._worker_candidate_predicate()
        claim_lock_clause = self._runnable_lock_clause()
        rows = self._rows(
            f"""
            WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms)
            SELECT s.head_seq,s.inbox_seq,s.lease_epoch,s.lease_owner,s.lease_until_ms,s.drive_retry_at_ms,
                   s.level,s.deletion_state,s.handler_version,s.required_capabilities,s.active_turn_id,t.ms
            FROM sk_session s CROSS JOIN clock t
            WHERE s.tenant_id=%s AND s.session_id=%s
              AND s.schema_version='S1' AND s.deletion_state='live'
              AND {candidate_predicate}
              AND {_DRIVE_RUNNABLE_SQL}
            {claim_lock_clause}
            """,
            (
                self.tenant_id,
                session_id,
                *worker_params,
            ),
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        routing = self._require_routing_cache(session_id, row)
        if not self.worker_supports.accepts(
            level=routing.requirements.level,
            handler_version=routing.requirements.handler_version,
            capabilities=routing.requirements.capabilities,
        ):
            return None
        previous_epoch = row[2]
        now = row[11]
        epoch = previous_epoch + 1
        expires_at_ms = now + ttl_ms
        updated = self._rows(
            "UPDATE sk_session SET lease_epoch=%s,lease_owner=%s,lease_until_ms=%s "
            "WHERE tenant_id=%s AND session_id=%s AND lease_epoch=%s RETURNING session_id",
            (epoch, owner, expires_at_ms, self.tenant_id, session_id, previous_epoch),
        )
        if len(updated) != 1:
            return None
        return Lease(session_id, owner, epoch, expires_at_ms, self.tenant_id)

    def claim_runnable(
        self,
        *,
        owner: str,
        ttl_ms: int,
        limit: int = 100,
        after: WorkCursor | None = None,
        work_kinds: WorkKinds | None = None,
        projection_consumers: ProjectionConsumers | None = None,
    ) -> tuple[ClaimedWork, ...]:
        """Atomically claim work, holding at most one drive lease."""

        self._require_configured_ledger("runnable work claim")
        _identity(owner)
        _positive(ttl_ms, "ttl_ms")
        _positive(limit, "limit")
        self._require_claim_capable()
        self._require_claim_outside_transaction()
        with self._transaction():
            claimed: list[ClaimedWork] = []
            cursor = after
            while len(claimed) < limit:
                work = self._runnable_work(
                    limit=limit - len(claimed),
                    after=cursor,
                    apply_worker_filter=True,
                    work_kinds=work_kinds,
                    projection_consumers=projection_consumers,
                )
                if not work:
                    break
                for item in work:
                    cursor = item.cursor
                    if item.kind == "project":
                        claimed.append(ClaimedWork(item))
                        continue
                    claim_store = self._scoped(item.tenant_id) if self.claims_all_tenants else self
                    lease = claim_store._claim_drive(item.session_id, owner=owner, ttl_ms=ttl_ms)
                    if lease is not None:
                        claimed.append(ClaimedWork(item, lease))
                        # Callers may page projection work, but a drive lease
                        # must be consumed before another one is taken.
                        return tuple(claimed)
                    if len(claimed) == limit:
                        break
            return tuple(claimed)

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
        state: TombstoneState = TombstoneState.PENDING,
    ) -> ForgetTombstone:
        existing = self._tombstone(entry.session_id, for_update=True)
        chain_digest = self._chain_anchor(entry.session_id, entry.head_seq)
        if entry.head_seq < 1:
            raise LedgerIntegrityError("deletion tombstone requires a retained record head")
        if existing is not None:
            if (existing.final_seq, existing.chain_digest) != (entry.head_seq, chain_digest):
                raise LedgerIntegrityError("existing deletion tombstone does not match the retained final head")
            self._verify_tombstone_object(existing, verifier, allow_unsigned=allow_unsigned)
            if (existing.actor, existing.scope_root_session_id) == (actor, scope_root_session_id):
                if existing.state is state:
                    return existing
                if existing.state is TombstoneState.COMPLETE and state is TombstoneState.PENDING:
                    self._rows(
                        "UPDATE sk_tombstone SET state='pending' WHERE tenant_id=%s AND session_id=%s",
                        (self.tenant_id, entry.session_id),
                    )
                    return replace(existing, state=state)
                raise LedgerIntegrityError("existing deletion tombstone has an incompatible lifecycle state")
            if existing.state is TombstoneState.PENDING:
                raise LedgerIntegrityError("a pending deletion tombstone belongs to a different actor or scope")

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
            state=state,
            scope_root_session_id=scope_root_session_id,
        )
        message = _tombstone_message(unsigned)
        if allow_unsigned:
            signature = sha256(message).digest()
        else:
            assert bound_signer is not None
            signature = bound_signer.sign(message)
        if not isinstance(signature, bytes) or not signature:
            raise LedgerIntegrityError("ledger signer returned an invalid tombstone signature")
        tombstone = replace(
            unsigned,
            signature=base64.b64encode(signature).decode("ascii"),
        )
        self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
        values = (
            entry.head_seq,
            chain_digest,
            now,
            canonical_json_bytes(actor.to_dict(), "tombstone actor"),
            tombstone.key_id,
            signature,
            state.value,
            scope_root_session_id,
            self.tenant_id,
            entry.session_id,
        )
        if existing is None:
            self._rows(
                "INSERT INTO sk_tombstone(tenant_id,session_id,final_seq,chain_digest,deleted_ms,actor,key_id,signature,"
                "state,scope_root_session_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (self.tenant_id, entry.session_id, *values[:8]),
            )
        else:
            self._rows(
                "UPDATE sk_tombstone SET final_seq=%s,chain_digest=%s,deleted_ms=%s,actor=%s,key_id=%s,"
                "signature=%s,state=%s,scope_root_session_id=%s WHERE tenant_id=%s AND session_id=%s",
                values,
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
        _external_root: _ExternalRoot = False,
    ) -> tuple[str, ...]:
        """Fence a forget scope and durably bind its pending tombstone."""
        self._require_configured_ledger("session forget")
        if actor.tenant_id != self.tenant_id:
            raise ContentBindingMismatch("deletion actor belongs to another tenant")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            self._guard_external_erasure(session_id, scope, enabled=_external_root)
            if not scope:
                raise Conflict("session not found in tenant")
            if any(entry.deletion_state == "purging" for entry in scope):
                raise SessionNotLive("session purge is already in progress")
            if any(entry.deletion_state not in {"live", "forgetting", "forgotten"} for entry in scope):
                raise SessionNotLive("session deletion state is not forgettable")
            now = self._now()
            for entry in scope:
                if entry.deletion_state != "live":
                    continue
                updated = self._rows(
                    "UPDATE sk_session SET deletion_state='forgetting',lease_epoch=lease_epoch+1,lease_owner=NULL,"
                    "lease_until_ms=NULL,next_drive_ms=NULL,drive_retry_at_ms=0 "
                    "WHERE tenant_id=%s AND session_id=%s AND deletion_state='live' RETURNING session_id",
                    (self.tenant_id, entry.session_id),
                )
                if len(updated) != 1:
                    raise Conflict("session deletion fence changed concurrently")
            for entry in scope:
                tombstone = self._tombstone(entry.session_id, for_update=True)
                if tombstone is not None and tombstone.state is TombstoneState.COMPLETE:
                    chain_digest = self._chain_anchor(entry.session_id, entry.head_seq)
                    if tombstone.final_seq != entry.head_seq or tombstone.chain_digest != chain_digest:
                        raise LedgerIntegrityError("completed tombstone does not match the retained final head")
                    if entry.deletion_state != "forgotten":
                        raise LedgerIntegrityError("completed tombstone belongs to a session that is not forgotten")
                    self._write_pending_tombstone(
                        entry,
                        actor,
                        scope_root_session_id=session_id,
                        bound_signer=bound_signer,
                        verifier=verifier,
                        allow_unsigned=allow_unsigned,
                        now=now,
                        state=TombstoneState.COMPLETE,
                    )
                    continue
                if entry.deletion_state == "forgotten":
                    raise LedgerIntegrityError("forgotten session is missing its complete tombstone")
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

    def _reject_external_erasure(self, session_ids: tuple[str, ...]) -> None:
        from ._content_runtime import _verify_content_binding, content_runtime_binding_for

        binding = content_runtime_binding_for(self)
        if binding is not None and binding._object_content is not None:
            raise ContentCleanupUnavailable("external content erasure is unavailable")
        try:
            _verify_content_binding(self, binding)
        except ContentUnavailable as exc:
            raise ContentCleanupUnavailable("external content erasure requires its configured binding") from exc
        for session_id in session_ids:
            (has_exact_key, has_locator) = self._one(
                "SELECT EXISTS (SELECT 1 FROM sk_session_key WHERE tenant_id=%s AND session_id=%s "
                "AND substr(key_id,1,%s)=%s), EXISTS (SELECT 1 FROM sk_content "
                "WHERE tenant_id=%s AND session_id=%s AND object_locator IS NOT NULL)",
                (self.tenant_id, session_id, len(_EXACT_KEY_ID_PREFIX), _EXACT_KEY_ID_PREFIX, self.tenant_id, session_id),
            )
            if has_exact_key or has_locator:
                raise ContentCleanupUnavailable("external content erasure is unavailable")

    def _guard_external_erasure(
        self, session_id: str, scope: tuple[DeletionScopeEntry, ...], *, enabled: _ExternalRoot, completing: bool = False
    ) -> None:
        if not enabled:
            self._reject_external_erasure(tuple(entry.session_id for entry in scope))
            return
        from ._content_runtime import _verify_content_binding, content_runtime_binding_for
        from ._external_content import _PostgresObjectContent, _SQLiteObjectContent

        binding = content_runtime_binding_for(self)
        if binding is None or any(entry.session_id != session_id or entry.parent_session_id is not None for entry in scope):
            raise ContentCleanupUnavailable("external erasure supports only a bound root")
        if isinstance(enabled, tuple):
            erasure, check_fence = enabled
            if not isinstance(binding._object_content, _PostgresObjectContent) or erasure.content is not binding._object_content:
                raise ContentCleanupUnavailable("PostgreSQL external erasure requires its configured binding")
            if not self._transaction_in_progress():
                raise ContentCleanupUnavailable("PostgreSQL external erasure requires its active primary fence")
            _, allow_unsigned, verifier = self._input_admission_policy()
            if allow_unsigned or verifier is None:
                raise ContentCleanupUnavailable("PostgreSQL external erasure requires signed ledger authority")
            check_fence()
        elif isinstance(binding._object_content, _SQLiteObjectContent):
            erasure = binding._object_content.erasure()
        else:
            raise ContentCleanupUnavailable("external erasure supports only a bound SQLite root")
        _verify_content_binding(self, binding)
        params = (self.tenant_id, session_id)
        rows = self._rows("SELECT key_id,provider_id,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s", params)
        if rows:
            ((key_id, provider_id, state),) = rows
            if provider_id != getattr(binding.key_provider, "provider_id", None):
                raise ContentCleanupUnavailable("external erasure requires the original key provider")
            key, allocations = erasure.erasure_inventory(*params, key_id)
            if not completing:
                if state == "active" and key is None:
                    raise ContentCleanupUnavailable("external erasure lacks the committed key intent")
                return
            if state != "destroying" or key is not None or allocations:
                raise ContentCleanupUnavailable("external erasure evidence is incomplete")
            erasure.verify_retired(binding.key_provider, *params, key_id)
        elif scope and scope[0].deletion_state != "forgotten":
            raise ContentCleanupUnavailable("external erasure requires the durable SQL key")
        if erasure.journal.scoped_key_intents(params[0], (session_id,)) or erasure.journal.scoped_allocations(
            params[0], (session_id,)
        ):
            raise ContentCleanupUnavailable("external erasure scope is not empty")
        if self._rows("SELECT 1 FROM sk_content WHERE tenant_id=%s AND session_id=%s LIMIT 1", params) or self._rows(
            "SELECT 1 FROM sk_content_link WHERE tenant_id=%s AND session_id=%s LIMIT 1", params
        ):
            raise ContentCleanupUnavailable("external erasure retained SQL content")
        if completing and rows:
            tombstone = self._tombstone(session_id, for_update=True)
            if tombstone is None or tombstone.state is not TombstoneState.PENDING:
                raise ContentCleanupUnavailable("external erasure requires a pending tombstone")
            _, allow_unsigned, verifier = self._input_admission_policy()
            self._verify_tombstone_object(tombstone, verifier, allow_unsigned=allow_unsigned)
            self._rows("DELETE FROM sk_session_key WHERE tenant_id=%s AND session_id=%s AND state='destroying'", params)

    def complete_forget(self, session_id: str, *, _external_root: _ExternalRoot = False) -> None:
        """Publish the complete tombstone after content/key erasure succeeds."""
        self._require_configured_ledger("session forget completion")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            self._guard_external_erasure(session_id, scope, enabled=_external_root, completing=True)
            if not scope:
                raise Conflict("session not found in tenant")
            if any(entry.deletion_state not in {"forgetting", "forgotten"} for entry in scope):
                raise SessionNotLive("forget scope is not fully fenced")
            for entry in scope:
                tombstone = self._tombstone(entry.session_id, for_update=True)
                if tombstone is None:
                    raise LedgerIntegrityError("forget tombstone is missing")
                if tombstone.final_seq != entry.head_seq or tombstone.scope_root_session_id != session_id:
                    raise LedgerIntegrityError("forget tombstone does not match the locked scope")
                if entry.deletion_state == "forgotten":
                    if tombstone.state is not TombstoneState.COMPLETE:
                        raise LedgerIntegrityError("forgotten session is missing its complete tombstone")
                    continue
                if tombstone.state is not TombstoneState.PENDING:
                    raise LedgerIntegrityError("pending forget tombstone is missing")
                self._rows(
                    "UPDATE sk_tombstone SET state='complete' WHERE tenant_id=%s AND session_id=%s AND state='pending'",
                    (self.tenant_id, entry.session_id),
                )
                self._rows(
                    "UPDATE sk_session SET deletion_state='forgotten',lease_owner=NULL,lease_until_ms=NULL,"
                    "next_drive_ms=NULL WHERE tenant_id=%s AND session_id=%s AND deletion_state='forgetting'",
                    (self.tenant_id, entry.session_id),
                )

    def begin_purge(
        self,
        session_id: str,
        actor: Principal,
        *,
        bound_signer: Any | None,
        verifier: Any | None,
        allow_unsigned: bool,
        _external_root: _ExternalRoot = False,
    ) -> tuple[str, ...]:
        """Freeze root/descendants, persist pending tombstones, and start purge."""
        self._require_configured_ledger("session purge")
        if actor.tenant_id != self.tenant_id:
            raise ContentBindingMismatch("deletion actor belongs to another tenant")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            self._guard_external_erasure(session_id, scope, enabled=_external_root)
            if _external_root and any(entry.deletion_state == "forgotten" for entry in scope):
                raise ContentCleanupUnavailable("external purge after forget is unavailable")
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

    def complete_purge(self, session_id: str, *, _external_root: _ExternalRoot = False) -> tuple[str, ...]:
        """Delete a fenced scope child-first and retain only complete tombstones."""
        self._require_configured_ledger("session purge completion")
        with self._transaction():
            scope = self._deletion_scope(session_id)
            self._guard_external_erasure(session_id, scope, enabled=_external_root, completing=True)
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
                if (
                    tombstone.final_seq != entry.head_seq
                    or tombstone.chain_digest != self._chain_anchor(entry.session_id, entry.head_seq)
                    or tombstone.scope_root_session_id != session_id
                ):
                    raise LedgerIntegrityError("purge tombstone does not match the fenced scope or final head")
            for entry in scope:
                for table in SESSION_PURGE_DELETE_ORDER:
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
            return cast(SessionStore, self)
        return self._root.for_tenant(tenant_id)

    @property
    def fold_factory(self) -> Callable[[], Fold]:
        return self._root.fold_factory

    @fold_factory.setter
    def fold_factory(self, factory: Callable[[], Fold]) -> None:
        self._root.fold_factory = factory
        self._bound.fold_factory = factory

    def heartbeat_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        root_factory = self._root.heartbeat_store_factory()

        @contextmanager
        def scoped() -> Iterator[SessionStore]:
            with root_factory() as store:
                yield store.for_tenant(self.tenant_id)

        return scoped

    def heartbeat_store(self) -> AbstractContextManager[SessionStore]:
        return self.heartbeat_store_factory()()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)


class SQLSessionTx:
    def __init__(self, store: SQLStore):
        self.store = store
        self.transaction_id = store._transaction_id()
        self.batches: dict[tuple[str, str], ConsumerBatch] = {}
        self._lease_guard: Callable[[], Lease] | None = None
        self._check_persistence_deadline: Callable[[], None] | None = None
        self._check_external_deadline: Callable[[], None] | None = None
        self._freeze_persistence_deadline: Callable[[], None] | None = None
        self._external_object_deadline: _ObjectDeadline | None = None
        self._leased_object_deadline: _ObjectDeadline | None = None
        self._pending_lease_fence: tuple[str, Lease] | None = None

    def bind_lease_guard(self, guard: Callable[[], Lease]) -> None:
        self._lease_guard = guard

    def _bind_persistence_deadline(
        self, check_deadline: Callable[[], None], freeze_deadline: Callable[[], None], object_deadline: _ObjectDeadline
    ) -> bool:
        self._check_persistence_deadline = check_deadline
        self._freeze_persistence_deadline = freeze_deadline
        self._leased_object_deadline = object_deadline
        return bool(self.store.lock_clause)

    def _bind_external_deadline(self, check_deadline: Callable[[], None], object_deadline: _ObjectDeadline | None = None) -> None:
        if self._check_external_deadline is None:
            self._check_external_deadline = check_deadline
            self._external_object_deadline = object_deadline

    def _object_write_deadline(self) -> _ObjectDeadline:
        external = self._external_object_deadline
        leased = self._leased_object_deadline
        if external is None:
            raise KeyUnavailable("external content requires a persistence deadline")
        if leased is None:
            return external
        return min(external, leased, key=lambda bound: bound[0] - bound[1]())

    def _active(self) -> None:
        if self.store._transaction_id() != self.transaction_id:
            raise RuntimeError("SessionTx must be used within its original caller transaction")

    def _finalize_lease_fence(self) -> None:
        pending = self._pending_lease_fence
        if pending is None:
            if self._check_external_deadline is not None:
                self._check_content_persistence_deadline()
            return
        self._active()
        session_id, lease = pending
        row = self.store._lock(session_id)
        self.store._check_lease(session_id, lease, row, self.store._now())
        self._check_content_persistence_deadline()

    def _check_content_persistence_deadline(self) -> None:
        if self._check_external_deadline is not None:
            self._check_external_deadline()
        if self._check_persistence_deadline is not None:
            self._check_persistence_deadline()

    def _commit(
        self,
        session_id: str,
        commit_id: str,
        manifest_body: bytes,
        sequences: tuple[tuple[str, int], ...],
        head: int,
        now: int,
    ) -> None:
        if not isinstance(manifest_body, bytes) or not isinstance(sequences, tuple):
            raise LedgerIntegrityError("commit metadata is invalid")
        try:
            manifest = json.loads(manifest_body)
            head_seq = manifest["head_seq"]
            head_chain_digest = manifest["head_chain_digest"]
            consume_input_ids = tuple(manifest["consume_input_ids"])
            consumers = tuple(manifest["consumers"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerIntegrityError("commit manifest is invalid") from exc
        _natural(head, "commit resulting head")
        _natural(now, "commit created_ms")
        if head != max([head_seq, *(seq for _, seq in sequences)]):
            raise LedgerIntegrityError("commit resulting head does not match its record sequences")
        if head_chain_digest != self.store._chain_anchor(session_id, head_seq):
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
            head_chain_digest=head_chain_digest,
            records=tuple(record_refs),
            consume_input_ids=consume_input_ids,
            consumers=consumers,
        )
        if stored_body != manifest_body:
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
        verifier = self.store._input_admission_policy()[2]
        if verifier is None or not verifier.verify(key_id, message, signature):
            raise LedgerIntegrityError(f"ledger signer key binding does not verify checkpoint at seq {seq}")
        self.store._rows(
            "INSERT INTO sk_checkpoint(tenant_id,session_id,seq,chain_digest,key_id,signature,signed_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (self.store.tenant_id, session_id, seq, chain_digest, key_id, signature, now),
        )

    def lock_session(self, session_id: str) -> None:
        self._active()
        self.store._require_live_row(self._lock_for_persistence(session_id))

    def _lock_for_persistence(self, session_id: str) -> tuple[Any, ...]:
        if self._freeze_persistence_deadline is not None:
            self._freeze_persistence_deadline()
        row = self.store._lock(session_id)
        if self._check_persistence_deadline is not None:
            self._check_persistence_deadline()
        return row

    def preflight_create(self, session_id: str) -> bool:
        """Check the creation identity before preparing content and return whether it exists."""
        self._active()
        store = self.store
        store._require_configured_ledger("session creation")
        _identity(session_id)
        store._require_create_isolation()
        tenant_id = store.tenant_id
        deletion_state, tombstoned = store._one(
            "SELECT (SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s), "
            "EXISTS (SELECT 1 FROM sk_tombstone WHERE tenant_id=%s AND session_id=%s)",
            (tenant_id, session_id, tenant_id, session_id),
        )
        if tombstoned or deletion_state not in {None, "live"}:
            raise SessionNotLive("a deleted session identity cannot be reused")
        return deletion_state == "live"

    def create(self, spec: SessionSpec, *, consumers: tuple[str, ...]) -> CreateReceipt:
        self._active()
        self.store._require_configured_ledger("session creation")
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
            s._require_create_isolation()
            ceiling = s._write_ceiling()
            if spec.parent_session_id is not None:
                if self._freeze_persistence_deadline is not None:
                    self._freeze_persistence_deadline()
                parent = s._rows(
                    f"SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s{s.lock_clause}",
                    (tenant_id, spec.parent_session_id),
                )
                if self._check_persistence_deadline is not None:
                    self._check_persistence_deadline()
                if len(parent) != 1:
                    raise Conflict("parent session not found in tenant")
                if parent[0][0] != "live":
                    raise SessionNotLive("parent session is fenced for deletion")
            if self._freeze_persistence_deadline is not None:
                self._freeze_persistence_deadline()
            now = s._now()
            inserted = s._rows(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,parent_session_id,parent_operation_id,"
                "phase,level,deletion_state,created_ms) "
                "VALUES (%s,%s,'S1',%s,%s,'idle',1,'live',%s) "
                "ON CONFLICT(tenant_id,session_id) DO NOTHING RETURNING session_id",
                (tenant_id, spec.session_id, spec.parent_session_id, spec.parent_operation_id, now),
            )
            row = self._lock_for_persistence(spec.session_id)
            # Serialize admission against the deletion fence before inspecting the identity.
            if row[7] != "live" or s._rows(
                "SELECT 1 FROM sk_tombstone WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, spec.session_id),
            ):
                raise SessionNotLive("a deleted session identity cannot be reused")
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
                # Create replay compares the already committed bytes and needs no
                # worker execution capability for the existing history.
                s._prefix(spec.session_id, row, require_worker=False)
                old = s._one(
                    "SELECT body,created_ms FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                    (tenant_id, spec.session_id, commit_id),
                )
                if old[0] != manifest or existing != record.encode():
                    raise Conflict("create identity has different bytes")
                return CreateReceipt(spec.session_id, 1, old[1], True)
            bootstrap = (
                normalize_requirements(record._payload["bootstrap"])
                if record.feature_level == 2
                else SessionRequirements(1, None, ())
            )
            routing = _RoutingCache(
                SessionRequirements.normalized(
                    level=bootstrap.level,
                    handler_version=bootstrap.handler_version,
                    capabilities=bootstrap.capabilities,
                ),
                None,
            )
            s._admit_level(ceiling=ceiling, required=max(routing.requirements.level, record.feature_level))
            s._write_routing_cache(
                spec.session_id,
                routing,
                minimum_level=record.feature_level,
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
        self.store._require_configured_ledger("input push")
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
            row = self._lock_for_persistence(session_id)
            s._require_live_row(row)
            old = s._rows(
                f"SELECT {_STORED_INBOX_COLUMNS} FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                (s.tenant_id, session_id, item.input_id),
            )
            if old:
                stored = s._stored_input(session_id, old[0])
                if old[0][0] != body:
                    raise Conflict("input identity has different bytes")
                if item.admission_proof is not None and item.admission_proof != stored.item.admission_proof:
                    raise InputAdmissionProofError("replayed input admission proof does not match the original")
                return InputReceipt(item.input_id, stored.input_seq, stored.received_ms, True)
            prefix_before = s._prefix(session_id, row, require_worker=False)
            routing_before = s._require_routing_cache(session_id, row, prefix=prefix_before)
            if item.kind == "model_selection" and item.feature_level == 2:
                selection = s._inbox_requirements(item)
                current_handler = routing_before.requirements.handler_version
                if selection.handler_version != current_handler:
                    raise Conflict("model selection handler does not match the current route")
            if item.admission_proof is not None:
                raise InputAdmissionProofError("input admission proof is created inside the SQL transaction")
            required = s._admit_level(
                ceiling=ceiling,
                required=max(
                    s._input_level(item, parent_session_id=session_id),
                    s._level_floor(prefix_before.records, prefix_before.stored_inputs),
                ),
                current=1,
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
                if not signer.verify(key_id, message, signature):
                    raise InputAdmissionProofError(
                        f"ledger signer key binding does not verify input admission at input_seq {seq}"
                    )
                signed_ms = now
            projected = s._inbox_requirements(item, inherited=routing_before.requirements)
            s._rows(
                "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,"
                "admission_key_id,admission_signature,admission_signed_ms,available_ms,received_ms,"
                "routing_level,routing_handler_version,routing_required_capabilities) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
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
                    projected.level,
                    projected.handler_version,
                    canonical_json_bytes(list(projected.capabilities), "input routing required_capabilities"),
                ),
            )
            s._rows(
                "UPDATE sk_session SET inbox_seq=%s,level=%s WHERE tenant_id=%s AND session_id=%s",
                (seq, required, s.tenant_id, session_id),
            )
            routing_row = (row[0], seq, *row[2:6], required, *row[7:])
            prefix = s._prefix(session_id, routing_row, require_worker=False)
            routing = s._derive_routing_cache(prefix, now=now, minimum_level=required)
            s._write_routing_cache(session_id, routing, minimum_level=required)
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
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> CommitReceipt:
        self._active()
        self.store._require_configured_ledger("ledger append")
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
        captured = s._head(session_id) if expected_head is not None else None
        latest_turn = next((record for record in reversed(records) if record.kind == "turn_started"), None)
        if expected_head is not None:
            assert captured is not None
            if s._ledger_integrity_mode() == "unconfigured":
                raise LedgerIntegrityError(
                    "expected_head appends require an explicit signed signer or allow_unsigned hash-only mode"
                )
            _require_expected_head(
                expected_head,
                tenant_id=s.tenant_id,
                session_id=session_id,
                head_seq=captured[0],
                chain_digest=s._chain_anchor(session_id, captured[0]),
            )
        if self._lease_guard is not None:
            lease = self._lease_guard()
        with s._bound_append(lease) if self._lease_guard is not None else s._transaction():
            ceiling = s._write_ceiling()
            row = s._lock(session_id)
            s._require_live_row(row)
            head_chain_digest = s._chain_anchor(session_id, expected_seq)
            if expected_head is not None:
                _require_expected_head(
                    expected_head,
                    tenant_id=s.tenant_id,
                    session_id=session_id,
                    head_seq=row[0],
                    chain_digest=s._chain_anchor(session_id, row[0]),
                )
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
                # Commit replay verifies the existing ledger without acquiring
                # execution capability.
                s._prefix(session_id, row, require_worker=False)
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
                if self._lease_guard is not None:
                    self._pending_lease_fence = (session_id, lease)
                return CommitReceipt(commit_id, stored_sequences, old[0][2], True)
            s._require_worker_accepts_row(row)
            prefix = s._prefix(session_id, row, require_worker=False)
            reducer = prefix.reducer.fork()
            routing_before = s._require_routing_cache(session_id, row, prefix=prefix)
            s._require_worker_supports(
                level=routing_before.requirements.level,
                handler_version=routing_before.requirements.handler_version,
                capabilities=routing_before.requirements.capabilities,
            )
            required = s._admit_level(
                ceiling=ceiling,
                required=max((s._record_level(record) for record in records), default=1),
                current=routing_before.requirements.level,
            )
            if latest_turn is not None:
                actual = normalize_requirements(latest_turn._payload["requirements"])
                bound = routing_before.requirements
                if actual.handler_version is None:
                    raise Conflict("turn_started handler must be explicit")
                if bound.handler_version is not None and actual.handler_version != bound.handler_version:
                    raise Conflict("turn_started handler does not match the pre-admission bound")
                route_level = routing_before.route_level or bound.level
                if actual.level < route_level:
                    raise Conflict("turn_started lowers the pre-admission requirement level")
                if not set(actual.capabilities).issubset(bound.capabilities):
                    raise Conflict("turn_started requirements exceed the pre-admission bound")
                s._require_worker_accepts_row(
                    (0, 0, 0, actual.level, actual.handler_version, canonical_json_bytes(list(actual.capabilities)))
                )
            s._check_lease(session_id, lease, row, s._now())
            if expected_seq != row[0] or (expected_inbox_seq is not None and expected_inbox_seq != row[1]):
                raise SequenceConflict("head or inbox watermark changed")
            if prefix.head_digest != head_chain_digest:
                raise LedgerIntegrityError("commit head_chain_digest does not match the original head")
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
                    f"SELECT {_STORED_INBOX_COLUMNS} FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                    (s.tenant_id, session_id, input_id),
                )
                stored = s._stored_input(session_id, incoming)
                applied = applications[input_id][0]
                applied_input = applied._payload["input"]
                if (
                    stored.consumed_seq is not None
                    or stored.item.available_ms > now
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
            new_chains: list[str] = []
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
                new_chains.append(chain_digest)
            for input_id, (_, seq) in applications.items():
                s._rows(
                    "UPDATE sk_inbox SET consumed_seq=%s WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                    (seq, s.tenant_id, session_id, input_id),
                )
            stored_inputs = s._stored_inputs(session_id, row[1]) if applications else prefix.stored_inputs
            added = tuple(StoredRecord(record, seq, commit_id, lease.epoch, now) for _, record, seq in new)
            if new:
                routing_prefix = _Prefix(
                    s.tenant_id,
                    session_id,
                    lease.epoch,
                    previous_chain,
                    reducer,
                    prefix.records + added,
                    stored_inputs,
                    row[1],
                    prefix.chains + tuple(new_chains),
                )
                routing = s._derive_routing_cache(routing_prefix, now=now, minimum_level=required)
                floor = s._level_floor(routing_prefix.records, stored_inputs, minimum_level=required)
                s._write_routing_cache(session_id, routing, minimum_level=max(required, floor))
            else:
                level = max(routing_before.requirements.level, s._level_floor(prefix.records, prefix.stored_inputs))
                s._rows(
                    "UPDATE sk_session SET level=%s WHERE tenant_id=%s AND session_id=%s",
                    (level, s.tenant_id, session_id),
                )
            s._rows(
                "UPDATE sk_session SET head_seq=%s WHERE tenant_id=%s AND session_id=%s",
                (head, s.tenant_id, session_id),
            )
            # Consuming unrelated input must not erase an idle provider's SQL deferral.
            s._schedule(
                session_id,
                state,
                preserve_due=state.active_turn_id == prefix.reducer.state.active_turn_id
                and state.due_ms == prefix.reducer.state.due_ms
                and all(r.kind in {"input_applied", "usage_observed"} for _, r, _ in new),
            )
            receipt = CommitReceipt(commit_id, tuple(sequences), head, records=added, inbox_seq=row[1])
            self._commit(session_id, commit_id, body, receipt.record_sequences, head, now)
            s._retain_prefix(
                _Prefix(
                    s.tenant_id,
                    session_id,
                    lease.epoch,
                    previous_chain,
                    reducer,
                    prefix.records + added,
                    stored_inputs,
                    row[1],
                    prefix.chains + tuple(new_chains),
                )
            )
            s._check_lease(session_id, lease, row, s._now())
            if self._lease_guard is not None:
                self._pending_lease_fence = (session_id, lease)
            return receipt

    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None:
        self._active()
        _positive(limit, "limit")
        s = self.store
        last, head, inbox, level, deletion_state, handler_version, required_capabilities = s._one(
            "SELECT c.last_seq,s.head_seq,s.inbox_seq,s.level,s.deletion_state,s.handler_version,"
            "s.required_capabilities FROM sk_consumer c "
            "JOIN sk_session s ON s.tenant_id=c.tenant_id AND s.session_id=c.session_id "
            f"WHERE c.tenant_id=%s AND c.session_id=%s AND c.consumer=%s{s.lock_clause}",
            (s.tenant_id, session_id, consumer),
        )
        if deletion_state != "live":
            raise SessionNotLive("session is fenced for deletion")
        s._require_reader_level(level)
        if handler_version is not None and (not isinstance(handler_version, str) or not handler_version):
            raise LedgerIntegrityError("session handler_version is invalid")
        s._session_capabilities(required_capabilities)
        if last > head:
            raise LedgerIntegrityError("consumer cursor is beyond the verified ledger head")
        verified, _chains = s._verified_records(session_id, head)
        s._require_input_integrity_gate()
        s._stored_inputs(session_id, inbox)
        verified_by_seq = {stored_row[1]: stored_row for stored_row in verified}
        # One statement snapshot. Include all remaining rows of the commit touching the soft limit.
        rows = s._rows(
            """
            WITH page AS MATERIALIZED (
                SELECT seq,commit_id FROM sk_record
                WHERE tenant_id=%s AND session_id=%s AND seq>%s AND seq<=%s ORDER BY seq LIMIT %s
            ), boundary AS (
                SELECT max(r.seq) AS seq FROM sk_record r
                WHERE r.tenant_id=%s AND r.session_id=%s
                AND r.seq<=%s AND r.commit_id=(SELECT commit_id FROM page ORDER BY seq DESC LIMIT 1)
            )
            SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record
            WHERE tenant_id=%s AND session_id=%s AND seq>%s
              AND seq<=(SELECT seq FROM boundary) ORDER BY seq
            """,
            (s.tenant_id, session_id, last, head, limit, s.tenant_id, session_id, head, s.tenant_id, session_id, last),
        )
        key = (session_id, consumer)
        if not rows:
            self.batches.pop(key, None)
            return None
        records = tuple(s._stored(session_id, verified_by_seq[row[1]]) for row in rows)
        batch = ConsumerBatch(session_id, consumer, last + 1, records[-1].seq, records)
        self.batches[key] = batch
        return batch

    def ack(self, batch: ConsumerBatch) -> None:
        self._active()
        self.store._require_configured_ledger("consumer acknowledgement")
        row = self.store._lock(batch.session_id)
        self.store._require_live_row(row)
        self.store._require_reader_level(row[6])
        if row[8] is not None and (not isinstance(row[8], str) or not row[8]):
            raise LedgerIntegrityError("session handler_version is invalid")
        self.store._session_capabilities(row[9])
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
