"""Single-host SQLite session store. Never share the database across hosts."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

from .sql import SQLStore

DDL = """
CREATE TABLE sk_session (
    session_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    parent_session_id TEXT REFERENCES sk_session(session_id), parent_operation_id TEXT,
    head_seq INTEGER NOT NULL DEFAULT 0 CHECK (head_seq >= 0),
    inbox_seq INTEGER NOT NULL DEFAULT 0 CHECK (inbox_seq >= 0),
    lease_epoch INTEGER NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
    lease_owner TEXT, lease_until_ms INTEGER,
    phase TEXT NOT NULL CHECK (phase IN ('idle','active','parked','suspended','closed')),
    active_turn_id TEXT, next_drive_ms INTEGER, terminal_seq INTEGER NOT NULL DEFAULT 0,
    CHECK ((lease_owner IS NULL) = (lease_until_ms IS NULL)),
    CHECK (terminal_seq BETWEEN 0 AND head_seq)
) STRICT;
CREATE INDEX sk_session_due ON sk_session(next_drive_ms,session_id) WHERE next_drive_ms IS NOT NULL;
CREATE INDEX sk_session_parent ON sk_session(parent_session_id);
CREATE TABLE sk_record (
    session_id TEXT NOT NULL REFERENCES sk_session(session_id), seq INTEGER NOT NULL CHECK (seq > 0),
    record_id TEXT NOT NULL, schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    kind TEXT NOT NULL CHECK (kind IN ('session_created','turn_started','input_applied','op_planned','op_started',
        'op_prepared','turn_parked','op_parked','op_completed','op_unknown','context_compacted','boundary_recorded','usage_observed','turn_ended')),
    turn_id TEXT, operation_id TEXT, attempt INTEGER CHECK (attempt > 0), body BLOB NOT NULL,
    digest TEXT NOT NULL CHECK (length(digest)=64), commit_id TEXT NOT NULL,
    commit_digest TEXT NOT NULL CHECK (length(commit_digest)=64),
    commit_position INTEGER NOT NULL CHECK (commit_position >= 0), writer_epoch INTEGER NOT NULL, created_ms INTEGER NOT NULL,
    PRIMARY KEY(session_id,seq), UNIQUE(session_id,record_id), UNIQUE(session_id,commit_id,commit_position)
) STRICT;
CREATE INDEX sk_record_operation ON sk_record(session_id,operation_id,attempt,seq) WHERE operation_id IS NOT NULL;
CREATE INDEX sk_record_turn ON sk_record(session_id,turn_id,seq);
CREATE TABLE sk_inbox (
    session_id TEXT NOT NULL REFERENCES sk_session(session_id), input_id TEXT NOT NULL,
    input_seq INTEGER NOT NULL CHECK (input_seq > 0),
    kind TEXT NOT NULL CHECK (kind IN ('user','steer','follow_up','provider_result','approval_answer','child_result',
        'control','provider_evidence')),
    body BLOB NOT NULL, digest TEXT NOT NULL CHECK (length(digest)=64),
    available_ms INTEGER NOT NULL, received_ms INTEGER NOT NULL, consumed_seq INTEGER,
    PRIMARY KEY(session_id,input_id), UNIQUE(session_id,input_seq),
    FOREIGN KEY(session_id,consumed_seq) REFERENCES sk_record(session_id,seq)
) STRICT;
CREATE INDEX sk_inbox_ready ON sk_inbox(session_id,available_ms,input_seq) WHERE consumed_seq IS NULL;
CREATE TABLE sk_consumer (
    session_id TEXT NOT NULL REFERENCES sk_session(session_id), consumer TEXT NOT NULL,
    last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0), PRIMARY KEY(session_id,consumer)
) STRICT;
-- Commit metadata is necessary for empty/fully overlapping commits and their original head_seq.
-- It owns no execution state. Record bodies remain solely in sk_record.
CREATE TABLE sk_commit (
    session_id TEXT NOT NULL REFERENCES sk_session(session_id), commit_id TEXT NOT NULL,
    body BLOB NOT NULL, digest TEXT NOT NULL CHECK(length(digest)=64),
    record_sequences BLOB NOT NULL, head_seq INTEGER NOT NULL CHECK(head_seq >= 0), created_ms INTEGER NOT NULL,
    PRIMARY KEY(session_id,commit_id)
) STRICT;
"""


class SQLiteStore(SQLStore):
    """A serialized connection per instance; BEGIN IMMEDIATE fences local writers."""

    clock_sql = "session_now_ms()"

    def __init__(self, path: str):
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=5, check_same_thread=False)
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        existing = self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='sk_session'").fetchone()
        if version != 1 and (version != 0 or existing):
            self.connection.close()
            raise ValueError("unsupported session SQLite schema version")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.create_function("session_now_ms", 0, lambda: time.time_ns() // 1_000_000)
        self._connection_lock = RLock()
        self._epoch = 0
        self._depth = 0

    @classmethod
    @contextmanager
    def standalone(cls, path: str) -> Iterator[SQLiteStore]:
        store = cls(path)
        try:
            yield store
        finally:
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
            for statement in DDL.split(";"):
                if statement.strip():
                    self.connection.execute(statement)
            self.connection.execute("PRAGMA user_version=1")
