"""Single-host SQLite session store. Never share the database across hosts."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Any

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._content_store import SQLContentStore
from vv_agent.kernel._lease_scope import Clock
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLSessionTx, SQLStore

SCHEMA_VERSION = 4
STORAGE_MAJOR = "S1"
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
    write_level INTEGER NOT NULL CHECK (write_level >= 1)
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
    kind TEXT NOT NULL CHECK (kind IN ('session_created','turn_started','input_applied','op_planned','op_started',
        'op_prepared','turn_parked','op_parked','op_completed','op_unknown','context_compacted','boundary_recorded','usage_observed','turn_ended')),
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
    kind TEXT NOT NULL CHECK (kind IN ('user','steer','follow_up','provider_result','approval_answer','child_result',
        'control','provider_evidence','model_selection')),
    body BLOB NOT NULL, digest TEXT NOT NULL CHECK (length(digest)=64 AND digest NOT GLOB '*[^0-9a-f]*'),
    admission_key_id TEXT, admission_signature BLOB,
    admission_signed_ms INTEGER CHECK (admission_signed_ms IS NULL OR admission_signed_ms >= 0),
    available_ms INTEGER NOT NULL CHECK (available_ms >= 0), received_ms INTEGER NOT NULL CHECK (received_ms >= 0),
    consumed_seq INTEGER CHECK (consumed_seq IS NULL OR consumed_seq > 0),
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


class SQLiteStore(SQLStore):
    """A serialized connection per instance; BEGIN IMMEDIATE fences local writers."""

    clock_sql = "session_now_ms()"

    def heartbeat_store(self):
        return nullcontext(self) if self.path == ":memory:" else self._heartbeat_connection()

    def _heartbeat_connection(self):
        return SQLiteStore.standalone(self.path, fold_factory=self.fold_factory, clock=self.clock)

    def __init__(self, path: str, *, fold_factory: Callable[[], Fold], clock: Clock):
        super().__init__(fold_factory=fold_factory)
        self.clock = clock
        self.path = path
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=5, check_same_thread=False)
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        tables = _schema_tables(self.connection)
        settings = (
            self.connection.execute("SELECT storage_major,write_level FROM sk_settings WHERE singleton=1").fetchone()
            if "sk_settings" in tables
            else None
        )
        valid_s1 = version == SCHEMA_VERSION and tables == SCHEMA_TABLES and settings == (STORAGE_MAJOR, 1)
        if not valid_s1 and (version != 0 or tables):
            self.connection.close()
            raise ValueError("unsupported session SQLite schema version")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA synchronous=FULL")
        if version == 0:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.create_function("session_now_ms", 0, self.clock.now_ms)
        self._connection_lock = RLock()
        self._epoch = 0
        self._depth = 0

    @classmethod
    @contextmanager
    def standalone(cls, path: str, *, fold_factory: Callable[[], Fold], clock: Clock) -> Iterator[SQLiteStore]:
        with cls._owned(cls(path, fold_factory=fold_factory, clock=clock)) as store:
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
            outer = self._depth == 0
            savepoint = f"session_{self._depth}"
            if outer:
                self.connection.execute("BEGIN IMMEDIATE")
                self._epoch += 1
            else:
                self.connection.execute(f"SAVEPOINT {savepoint}")
            self._depth += 1
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
                self._depth -= 1

    def _transaction_id(self) -> object:
        with self._connection_lock:
            if not self._depth or not self.connection.in_transaction:
                raise RuntimeError("SessionTx requires an open transaction")
            return self._epoch

    def install_schema(self) -> None:
        with self._transaction():
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version == SCHEMA_VERSION:
                tables = _schema_tables(self.connection)
                settings = (
                    self.connection.execute("SELECT storage_major,write_level FROM sk_settings WHERE singleton=1").fetchone()
                    if "sk_settings" in tables
                    else None
                )
                if tables == SCHEMA_TABLES and settings == (STORAGE_MAJOR, 1):
                    return
                raise ValueError("unsupported session SQLite schema version")
            if version != 0 or _schema_tables(self.connection):
                raise ValueError("unsupported session SQLite schema version")
            for statement in DDL.split(";"):
                if statement.strip():
                    self.connection.execute(statement)
            self.connection.execute("INSERT INTO sk_settings(singleton,storage_major,write_level) VALUES (1,'S1',1)")
            self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


class _SQLiteContentBackend:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    @staticmethod
    def _query(query: str) -> str:
        return query.replace("{p}", "?")

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        row = self.connection.execute(self._query(query), params).fetchone()
        return None if row is None else tuple(row)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in self.connection.execute(self._query(query), params).fetchall()]

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        return self.connection.execute(self._query(query), params).rowcount

    def require_transaction(self) -> None:
        if not self.connection.in_transaction:
            raise RuntimeError("durable content writes require the caller's outer session transaction")

    def now_ms(self) -> int:
        row = self.fetchone("SELECT session_now_ms()")
        if row is None:
            raise RuntimeError("SQLite content clock is unavailable")
        return row[0]


class SQLiteContentStore(SQLContentStore):
    """SQLite content adapter bound to a caller-owned connection or transaction."""

    def __init__(
        self,
        source: SQLiteStore | SQLSessionTx | sqlite3.Connection,
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
    ) -> None:
        connection: Any
        if isinstance(source, SQLSessionTx):
            if not isinstance(source.store, SQLiteStore):
                raise TypeError("SQLiteContentStore transaction must belong to SQLiteStore")
            connection = source.store.connection
        else:
            connection = source.connection if isinstance(source, SQLiteStore) else source
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("SQLiteContentStore requires a SQLite connection or SQLiteStore")
        super().__init__(_SQLiteContentBackend(connection), key_provider, provider_id=provider_id)
