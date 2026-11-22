"""Single-host SQLite session store. Never share the database across hosts."""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from hashlib import sha256
from threading import RLock
from typing import Any

from vv_agent.kernel._content import ContentUnavailable, KeyProvider
from vv_agent.kernel._content_sql import CoreSQLContentStore
from vv_agent.kernel._content_store import ContentCipher
from vv_agent.kernel._lease_scope import Clock
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLSessionTx, SQLStore, _TenantSQLStore
from vv_agent.kernel._store import SessionStore

SCHEMA_VERSION = 4
STORAGE_MAJOR = "S1"
DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS = 250
SCHEMA_FINGERPRINT_VERSION = "S1-sqlite-schema-v3"
# Update this digest with intentional DDL changes and contract review.
SCHEMA_FINGERPRINT = "60446855526b178427bea806638e7fbf6d8632bb6accac1b5a57832c6de9666f"
SCHEMA_TABLES = frozenset(
    {
        "sk_settings",
        "sk_session",
        "sk_record",
        "sk_inbox",
        "sk_consumer",
        "sk_commit",
        "sk_checkpoint",
        "sk_session_key",
        "sk_content",
        "sk_content_link",
        "sk_tombstone",
    }
)

DDL = """
CREATE TABLE sk_settings (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    storage_major TEXT NOT NULL CHECK (storage_major = 'S1'),
    write_level INTEGER NOT NULL CHECK (write_level >= 1),
    database_id TEXT NOT NULL CHECK (length(database_id) = 64 AND database_id NOT GLOB '*[^0-9a-f]*'),
    external_content_binding TEXT CHECK (
        length(external_content_binding) = 64 AND external_content_binding NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;
CREATE TABLE sk_session (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
    schema_version TEXT NOT NULL CHECK (schema_version = 'S1'),
    parent_session_id TEXT, parent_operation_id TEXT,
    head_seq INTEGER NOT NULL DEFAULT 0 CHECK (head_seq >= 0),
    inbox_seq INTEGER NOT NULL DEFAULT 0 CHECK (inbox_seq >= 0),
    lease_epoch INTEGER NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
    lease_owner TEXT, lease_until_ms INTEGER CHECK (lease_until_ms IS NULL OR lease_until_ms >= 0),
    phase TEXT NOT NULL CHECK (phase IN ('idle','active','parked','suspended','closed')),
    active_turn_id TEXT, next_drive_ms INTEGER CHECK (next_drive_ms IS NULL OR next_drive_ms >= 0),
    drive_retry_at_ms INTEGER NOT NULL DEFAULT 0 CHECK (drive_retry_at_ms >= 0),
    terminal_seq INTEGER NOT NULL DEFAULT 0 CHECK (terminal_seq >= 0 AND terminal_seq <= head_seq),
    level INTEGER NOT NULL CHECK (level >= 1), handler_version TEXT,
    required_capabilities BLOB NOT NULL DEFAULT X'5B5D',
    deletion_state TEXT NOT NULL CHECK (deletion_state IN ('live','forgetting','forgotten','purging')),
    created_ms INTEGER NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id),
    CHECK ((lease_owner IS NULL) = (lease_until_ms IS NULL)),
    CHECK ((parent_session_id IS NULL) = (parent_operation_id IS NULL)),
    FOREIGN KEY (tenant_id,parent_session_id) REFERENCES sk_session(tenant_id,session_id)
) STRICT;
CREATE INDEX sk_session_due ON sk_session(level,next_drive_ms,tenant_id,session_id)
    WHERE next_drive_ms IS NOT NULL AND deletion_state = 'live';
CREATE INDEX sk_session_parent ON sk_session(tenant_id,parent_session_id) WHERE parent_session_id IS NOT NULL;
CREATE INDEX sk_session_deletion ON sk_session(tenant_id,deletion_state,session_id);
CREATE TABLE sk_record (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
    seq INTEGER NOT NULL CHECK (seq > 0), record_id TEXT NOT NULL,
    schema_version TEXT NOT NULL CHECK (schema_version = 'S1'),
    kind TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 64 AND kind GLOB '[a-z]*'
        AND kind NOT GLOB '*[^a-z0-9_]*'),
    turn_id TEXT, operation_id TEXT, attempt INTEGER CHECK (attempt > 0), body BLOB NOT NULL,
    digest TEXT NOT NULL CHECK (length(digest)=64 AND digest NOT GLOB '*[^0-9a-f]*'),
    chain_digest TEXT NOT NULL CHECK (length(chain_digest)=64 AND chain_digest NOT GLOB '*[^0-9a-f]*'),
    commit_id TEXT NOT NULL,
    commit_digest TEXT NOT NULL CHECK (length(commit_digest)=64 AND commit_digest NOT GLOB '*[^0-9a-f]*'),
    commit_position INTEGER NOT NULL CHECK (commit_position >= 0),
    writer_epoch INTEGER NOT NULL CHECK (writer_epoch >= 0), created_ms INTEGER NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,seq), UNIQUE (tenant_id,session_id,record_id),
    UNIQUE (tenant_id,session_id,commit_id,commit_position),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
) STRICT;
CREATE INDEX sk_record_operation ON sk_record(tenant_id,session_id,operation_id,attempt,seq) WHERE operation_id IS NOT NULL;
CREATE INDEX sk_record_turn ON sk_record(tenant_id,session_id,turn_id,seq);
CREATE TABLE sk_inbox (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, input_id TEXT NOT NULL,
    input_seq INTEGER NOT NULL CHECK (input_seq > 0),
    kind TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 64 AND kind GLOB '[a-z]*'
        AND kind NOT GLOB '*[^a-z0-9_]*'),
    body BLOB NOT NULL, digest TEXT NOT NULL CHECK (length(digest)=64 AND digest NOT GLOB '*[^0-9a-f]*'),
    admission_key_id TEXT, admission_signature BLOB,
    admission_signed_ms INTEGER CHECK (admission_signed_ms IS NULL OR admission_signed_ms >= 0),
    available_ms INTEGER NOT NULL CHECK (available_ms >= 0), received_ms INTEGER NOT NULL CHECK (received_ms >= 0),
    consumed_seq INTEGER CHECK (consumed_seq IS NULL OR consumed_seq > 0),
    routing_level INTEGER NOT NULL DEFAULT 1 CHECK (routing_level >= 1), routing_handler_version TEXT,
    routing_required_capabilities BLOB NOT NULL DEFAULT X'5B5D',
    PRIMARY KEY (tenant_id,session_id,input_id), UNIQUE (tenant_id,session_id,input_seq),
    CHECK ((admission_key_id IS NULL) = (admission_signature IS NULL)
        AND (admission_key_id IS NULL) = (admission_signed_ms IS NULL)),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
    FOREIGN KEY (tenant_id,session_id,consumed_seq) REFERENCES sk_record(tenant_id,session_id,seq)
) STRICT;
CREATE INDEX sk_inbox_ready ON sk_inbox(tenant_id,session_id,available_ms,input_seq) WHERE consumed_seq IS NULL;
CREATE TABLE sk_consumer (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, consumer TEXT NOT NULL,
    last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
    project_retry_at_ms INTEGER NOT NULL DEFAULT 0 CHECK (project_retry_at_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,consumer),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
) STRICT;
CREATE INDEX sk_consumer_retry ON sk_consumer(tenant_id,project_retry_at_ms,last_seq,session_id,consumer);
-- Commit metadata is necessary for empty/fully overlapping commits and their original head_seq.
-- It owns no execution state. Record bodies remain solely in sk_record.
CREATE TABLE sk_commit (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, commit_id TEXT NOT NULL,
    body BLOB NOT NULL, digest TEXT NOT NULL CHECK (length(digest)=64 AND digest NOT GLOB '*[^0-9a-f]*'),
    record_sequences BLOB NOT NULL, head_seq INTEGER NOT NULL CHECK (head_seq >= 0),
    created_ms INTEGER NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,commit_id),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
) STRICT;
CREATE TABLE sk_checkpoint (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, seq INTEGER NOT NULL CHECK (seq > 0),
    chain_digest TEXT NOT NULL CHECK (length(chain_digest)=64 AND chain_digest NOT GLOB '*[^0-9a-f]*'),
    key_id TEXT NOT NULL, signature BLOB NOT NULL, signed_ms INTEGER NOT NULL CHECK (signed_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,seq),
    FOREIGN KEY (tenant_id,session_id,seq) REFERENCES sk_record(tenant_id,session_id,seq)
) STRICT;
CREATE TABLE sk_session_key (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, key_id TEXT NOT NULL,
    provider_id TEXT NOT NULL, wrapped_key BLOB NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active','destroying')), created_ms INTEGER NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id), UNIQUE (tenant_id,key_id),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
) STRICT;
CREATE TABLE sk_content (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, content_id TEXT NOT NULL,
    key_id TEXT NOT NULL, mac TEXT NOT NULL CHECK (length(mac)=64 AND mac NOT GLOB '*[^0-9a-f]*'),
    size INTEGER NOT NULL CHECK (size >= 0), labels BLOB NOT NULL, nonce BLOB NOT NULL CHECK (length(nonce)=12),
    created_ms INTEGER NOT NULL CHECK (created_ms >= 0),
    state TEXT NOT NULL CHECK (state IN ('staged','live','gc_pending')),
    pin_until_ms INTEGER CHECK (pin_until_ms IS NULL OR pin_until_ms >= 0),
    ciphertext BLOB, object_locator BLOB,
    PRIMARY KEY (tenant_id,session_id,content_id),
    UNIQUE (tenant_id,session_id,key_id,mac,labels),
    CHECK ((ciphertext IS NOT NULL) <> (object_locator IS NOT NULL))
) STRICT;
CREATE INDEX sk_content_gc ON sk_content(state,pin_until_ms,created_ms,tenant_id,session_id,content_id)
    WHERE state IN ('staged','gc_pending');
CREATE TABLE sk_content_link (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, content_id TEXT NOT NULL,
    record_seq INTEGER, input_id TEXT, slot TEXT NOT NULL,
    CHECK ((record_seq IS NOT NULL) + (input_id IS NOT NULL) = 1),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
    FOREIGN KEY (tenant_id,session_id,record_seq) REFERENCES sk_record(tenant_id,session_id,seq),
    FOREIGN KEY (tenant_id,session_id,input_id) REFERENCES sk_inbox(tenant_id,session_id,input_id),
    FOREIGN KEY (tenant_id,session_id,content_id) REFERENCES sk_content(tenant_id,session_id,content_id)
) STRICT;
CREATE UNIQUE INDEX sk_content_link_record_slot
    ON sk_content_link(tenant_id,session_id,record_seq,slot) WHERE record_seq IS NOT NULL;
CREATE UNIQUE INDEX sk_content_link_input_slot
    ON sk_content_link(tenant_id,session_id,input_id,slot) WHERE input_id IS NOT NULL;
CREATE INDEX sk_content_link_content ON sk_content_link(tenant_id,session_id,content_id);
CREATE TABLE sk_tombstone (
    tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
    final_seq INTEGER NOT NULL CHECK (final_seq >= 0),
    chain_digest TEXT NOT NULL CHECK (length(chain_digest)=64 AND chain_digest NOT GLOB '*[^0-9a-f]*'),
    deleted_ms INTEGER NOT NULL CHECK (deleted_ms >= 0), actor BLOB NOT NULL,
    key_id TEXT NOT NULL, signature BLOB NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending','complete')),
    scope_root_session_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id,session_id)
) STRICT;
CREATE INDEX sk_tombstone_state ON sk_tombstone(state,tenant_id,session_id);
"""


def _schema_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sk\\_%' ESCAPE '\\'")
    }


def _schema_objects(connection: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    """Return a deterministic, stored-text schema manifest for fingerprinting."""

    return connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE (name LIKE 'sk\\_%' ESCAPE '\\' OR tbl_name LIKE 'sk\\_%' ESCAPE '\\') "
        "AND type IN ('table','index','trigger','view') AND (type IN ('table','view') OR sql IS NOT NULL) "
        "ORDER BY type,name"
    ).fetchall()


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    material = {"version": SCHEMA_FINGERPRINT_VERSION, "objects": _schema_objects(connection)}
    encoded = json.dumps(material, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _schema_is_current(connection: sqlite3.Connection) -> bool:
    if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        return False
    if _schema_tables(connection) != SCHEMA_TABLES or _schema_fingerprint(connection) != SCHEMA_FINGERPRINT:
        return False
    try:
        settings = connection.execute("SELECT singleton,storage_major,write_level FROM sk_settings").fetchall()
    except sqlite3.Error:
        return False
    return (
        len(settings) == 1
        and settings[0][0] == 1
        and settings[0][1] == STORAGE_MAJOR
        and type(settings[0][2]) is int
        and settings[0][2] >= 1
    )


def _positive_timeout(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class SQLiteStore(SQLStore):
    """A serialized connection per instance; BEGIN IMMEDIATE fences local writers."""

    clock_sql = "session_now_ms()"

    def heartbeat_store(self):
        return self._heartbeat_connection()

    def drive_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        """Capture a normal independent drive connection opener for this store."""

        path = self.path
        fold_factory = self.fold_factory
        clock = self.clock
        busy_timeout_ms = self.busy_timeout_ms
        heartbeat_busy_timeout_ms = self.heartbeat_busy_timeout_ms
        tenant_id = self.tenant_id
        supports = self.worker_supports
        reader_level = self.reader_level
        all_tenants = self.claims_all_tenants
        configured, allow_unsigned, signer = self._input_admission_policy()

        @contextmanager
        def open_store() -> Iterator[SessionStore]:
            with SQLiteStore.standalone(
                path,
                fold_factory=fold_factory,
                clock=clock,
                busy_timeout_ms=busy_timeout_ms,
                heartbeat_busy_timeout_ms=heartbeat_busy_timeout_ms,
            ) as store:
                store.configure_reader(level=reader_level)
                store.configure_worker(
                    level=supports.max_level,
                    handler_versions=supports.handler_versions,
                    capabilities=supports.capabilities,
                    all_tenants=all_tenants,
                )
                if configured:
                    store.configure_input_admission(signer, allow_unsigned=allow_unsigned)
                yield store if all_tenants else store.for_tenant(tenant_id)

        return open_store

    def _heartbeat_connection(self):
        return self._inherit_runtime_configuration(
            SQLiteStore.standalone(
                self.path,
                fold_factory=self.fold_factory,
                clock=self.clock,
                busy_timeout_ms=self.heartbeat_busy_timeout_ms,
                heartbeat_busy_timeout_ms=self.heartbeat_busy_timeout_ms,
            )
        )

    def __init__(
        self,
        path: str,
        *,
        fold_factory: Callable[[], Fold],
        clock: Clock,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        heartbeat_busy_timeout_ms: int = DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS,
    ):
        if path == ":memory:":
            raise ValueError("session SQLiteStore requires a file path; ':memory:' is not supported")
        _positive_timeout(busy_timeout_ms, "busy_timeout_ms")
        _positive_timeout(heartbeat_busy_timeout_ms, "heartbeat_busy_timeout_ms")
        super().__init__(fold_factory=fold_factory)
        self.clock = clock
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self.heartbeat_busy_timeout_ms = heartbeat_busy_timeout_ms
        self.connection = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        objects = _schema_objects(self.connection)
        if not _schema_is_current(self.connection) and (version != 0 or objects):
            self.connection.close()
            raise ValueError("unsupported session SQLite schema version")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.create_function("session_now_ms", 0, self.clock.now_ms)
        self._connection_lock = RLock()

    @classmethod
    @contextmanager
    def standalone(
        cls,
        path: str,
        *,
        fold_factory: Callable[[], Fold],
        clock: Clock,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        heartbeat_busy_timeout_ms: int = DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS,
    ) -> Iterator[SQLiteStore]:
        with cls._owned(
            cls(
                path,
                fold_factory=fold_factory,
                clock=clock,
                busy_timeout_ms=busy_timeout_ms,
                heartbeat_busy_timeout_ms=heartbeat_busy_timeout_ms,
            )
        ) as store:
            yield store

    @staticmethod
    @contextmanager
    def _owned[Store: SQLiteStore](store: Store) -> Iterator[Store]:
        try:
            yield store
        finally:
            with store._connection_lock:
                store.connection.close()

    def _rows(self, query: str, params: tuple = ()) -> list[tuple]:
        with self._connection_lock:
            return self.connection.execute(query.replace("%s", "?"), params).fetchall()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._connection_lock:
            state = self._transaction_state
            self._read_snapshot_prefix = None
            outer = state.depth == 0 and not self.connection.in_transaction
            savepoint = f"session_{state.depth}"
            if outer:
                self.connection.execute("BEGIN IMMEDIATE")
                state.epoch += 1
                state.sqlite_write_fence = True
            else:
                self.connection.execute(f"SAVEPOINT {savepoint}")
            state.depth += 1
            try:
                yield
                self.connection.execute("COMMIT" if outer else f"RELEASE {savepoint}")
            except BaseException:
                self._fold_cache = None
                self._previous_prefix = None
                self.connection.execute("ROLLBACK" if outer else f"ROLLBACK TO {savepoint}")
                if not outer:
                    self.connection.execute(f"RELEASE {savepoint}")
                raise
            finally:
                state.depth -= 1
                if outer:
                    state.sqlite_write_fence = False

    def _require_content_write_fence(self, tx: SQLSessionTx) -> None:
        with self._connection_lock:
            tx._active()
            if not self._transaction_state.sqlite_write_fence or self._transaction_state.depth != 1:
                raise RuntimeError("external content requires a store-owned SQLite write transaction")
            tx._check_content_persistence_deadline()

    def _transaction_id(self) -> object:
        with self._connection_lock:
            if not self._transaction_state.depth or not self.connection.in_transaction:
                raise RuntimeError("SessionTx requires an open transaction")
            return self._transaction_state.epoch

    @property
    def _depth(self) -> int:
        return self._transaction_state.depth

    @property
    def _epoch(self) -> int:
        return self._transaction_state.epoch

    def _object_installation(self) -> tuple[str, str | None] | None:
        with self._connection_lock:
            try:
                row = self.connection.execute(
                    "SELECT database_id,external_content_binding FROM sk_settings WHERE singleton=1"
                ).fetchone()
            except sqlite3.OperationalError:
                if self.connection.execute("PRAGMA user_version").fetchone() == (0,):
                    return None
                raise
            if row is None:
                raise ValueError("SQLite S1 installation settings are missing")
            return row

    def _provision_object_content(self, verify_installation: Callable[[str], str]) -> None:
        """Pin the verified object configuration on an installed, quiesced primary."""
        with self._connection_lock:
            if self._transaction_in_progress():
                raise ContentUnavailable("external content provisioning requires an idle SQLite connection")
            with self._transaction():
                installation = self._object_installation()
                if installation is None or installation[1] is not None:
                    raise ContentUnavailable("external content requires an installed SQLite store with no prior binding")
                database_id, _ = installation
                digest = verify_installation(database_id)
                self._rows("UPDATE sk_settings SET external_content_binding=%s WHERE singleton=1", (digest,))

    def install_schema(self) -> None:
        with self._connection_lock:
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if _schema_is_current(self.connection):
                return
            if version != 0 or _schema_objects(self.connection):
                raise ValueError("unsupported session SQLite schema version")
            self.connection.execute("PRAGMA journal_mode=WAL")
            with self._transaction():
                for statement in DDL.split(";"):
                    if statement.strip():
                        self.connection.execute(statement)
                self.connection.execute(
                    "INSERT INTO sk_settings(singleton,storage_major,write_level,database_id) VALUES (1,'S1',1,?)",
                    (secrets.token_hex(32),),
                )
                self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                if not _schema_is_current(self.connection):
                    raise ValueError("session SQLite schema fingerprint does not match this runtime")


class _SQLiteContentBackend:
    lock_clause = ""

    def __init__(
        self,
        connection: sqlite3.Connection,
        connection_lock: RLock | None,
        *,
        tenant_context_id: str | None = None,
        runtime_mode: str = "unbound-internal",
    ) -> None:
        self.connection = connection
        self.connection_lock = connection_lock
        self.tenant_context_id = tenant_context_id
        self.runtime_mode = runtime_mode

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        if self.connection_lock is None:
            yield
            return
        with self.connection_lock:
            yield

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """Join a caller transaction or hold one short read snapshot."""

        with self._serialized():
            if self.connection.in_transaction:
                yield
                return
            self.connection.execute("BEGIN DEFERRED")
            try:
                yield
            except BaseException:
                with suppress(BaseException):
                    self.connection.execute("ROLLBACK")
                raise
            try:
                self.connection.execute("COMMIT")
            except BaseException:
                with suppress(BaseException):
                    self.connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _query(query: str) -> str:
        return query.replace("{p}", "?")

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        with self._serialized():
            row = self.connection.execute(self._query(query), params).fetchone()
        return None if row is None else tuple(row)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._serialized():
            rows = self.connection.execute(self._query(query), params).fetchall()
        return [tuple(row) for row in rows]

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self._serialized():
            return self.connection.execute(self._query(query), params).rowcount

    def require_transaction(self) -> None:
        with self._serialized():
            if not self.connection.in_transaction:
                raise RuntimeError("durable content writes require the caller's outer session transaction")

    def now_ms(self) -> int:
        row = self.fetchone("SELECT session_now_ms()")
        if row is None:
            raise RuntimeError("SQLite content clock is unavailable")
        return row[0]


class CoreSQLiteContentStore(CoreSQLContentStore):
    """SQLite content adapter over a caller-owned connection or transaction."""

    def __init__(
        self,
        source: SQLiteStore | _TenantSQLStore | SQLSessionTx | sqlite3.Connection,
        key_provider: KeyProvider,
        cipher: ContentCipher,
        *,
        provider_id: str | None = None,
    ) -> None:
        connection: Any
        connection_lock: RLock | None = None
        tenant_context_id: str | None = None
        runtime_mode = "unbound-internal"
        if isinstance(source, SQLSessionTx):
            if not isinstance(source.store, SQLiteStore):
                raise TypeError("SQLiteContentStore transaction must belong to SQLiteStore")
            connection = source.store.connection
            connection_lock = source.store._connection_lock
            if source.store._root_store is not None:
                tenant_context_id = source.store.tenant_id
                runtime_mode = "tenant"
        elif isinstance(source, _TenantSQLStore):
            bound = source._bound
            if not isinstance(bound, SQLiteStore):
                raise TypeError("SQLiteContentStore tenant view must belong to SQLiteStore")
            connection = bound.connection
            connection_lock = bound._connection_lock
            tenant_context_id = source.tenant_id
            runtime_mode = "tenant"
        else:
            connection = source.connection if isinstance(source, SQLiteStore) else source
            if isinstance(source, SQLiteStore):
                connection_lock = source._connection_lock
                if source._root_store is not None:
                    tenant_context_id = source.tenant_id
                    runtime_mode = "tenant"
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("SQLiteContentStore requires a SQLite connection or SQLiteStore")
        super().__init__(
            _SQLiteContentBackend(
                connection,
                connection_lock,
                tenant_context_id=tenant_context_id,
                runtime_mode=runtime_mode,
            ),
            key_provider,
            cipher,
            provider_id=provider_id,
        )
