"""Internal PostgreSQL reference store, bound to caller-owned psycopg v3 connections."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, cast

from .sql import SQLSessionTx, SQLStore

if TYPE_CHECKING:
    from psycopg import Connection

CLOCK = "floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint"
DDL = """
CREATE TABLE sk_session (
    session_id text PRIMARY KEY,
    schema_version smallint NOT NULL CHECK (schema_version = 1),
    parent_session_id text REFERENCES sk_session(session_id), parent_operation_id text,
    head_seq bigint NOT NULL DEFAULT 0 CHECK (head_seq >= 0),
    inbox_seq bigint NOT NULL DEFAULT 0 CHECK (inbox_seq >= 0),
    lease_epoch bigint NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
    lease_owner text, lease_until_ms bigint,
    phase text NOT NULL CHECK (phase IN ('idle','active','parked','suspended','closed')),
    active_turn_id text, next_drive_ms bigint, terminal_seq bigint NOT NULL DEFAULT 0,
    CHECK ((lease_owner IS NULL) = (lease_until_ms IS NULL)),
    CHECK (terminal_seq BETWEEN 0 AND head_seq)
);
CREATE INDEX sk_session_due ON sk_session(next_drive_ms,session_id) WHERE next_drive_ms IS NOT NULL;
CREATE INDEX sk_session_parent ON sk_session(parent_session_id);
CREATE TABLE sk_record (
    session_id text NOT NULL REFERENCES sk_session(session_id), seq bigint NOT NULL CHECK (seq > 0),
    record_id text NOT NULL, schema_version smallint NOT NULL CHECK (schema_version = 1),
    kind text NOT NULL CHECK (kind IN ('session_created','turn_started','input_applied','op_planned','op_started',
        'op_prepared','turn_parked','op_parked','op_completed','op_unknown','context_compacted','boundary_recorded','usage_observed','turn_ended')),
    turn_id text, operation_id text, attempt integer CHECK (attempt > 0), body bytea NOT NULL,
    digest text NOT NULL CHECK (length(digest)=64), commit_id text NOT NULL,
    commit_digest text NOT NULL CHECK (length(commit_digest)=64),
    commit_position integer NOT NULL CHECK (commit_position >= 0), writer_epoch bigint NOT NULL, created_ms bigint NOT NULL,
    PRIMARY KEY(session_id,seq), UNIQUE(session_id,record_id), UNIQUE(session_id,commit_id,commit_position)
);
CREATE INDEX sk_record_operation ON sk_record(session_id,operation_id,attempt,seq) WHERE operation_id IS NOT NULL;
CREATE INDEX sk_record_turn ON sk_record(session_id,turn_id,seq);
CREATE TABLE sk_inbox (
    session_id text NOT NULL REFERENCES sk_session(session_id), input_id text NOT NULL,
    input_seq bigint NOT NULL CHECK (input_seq > 0),
    kind text NOT NULL CHECK (kind IN ('user','steer','follow_up','deferred_result','approval_answer','child_result',
        'control','provider_evidence')),
    body bytea NOT NULL, digest text NOT NULL CHECK (length(digest)=64),
    available_ms bigint NOT NULL, received_ms bigint NOT NULL, consumed_seq bigint,
    PRIMARY KEY(session_id,input_id), UNIQUE(session_id,input_seq),
    FOREIGN KEY(session_id,consumed_seq) REFERENCES sk_record(session_id,seq)
);
CREATE INDEX sk_inbox_ready ON sk_inbox(session_id,available_ms,input_seq) WHERE consumed_seq IS NULL;
CREATE TABLE sk_consumer (
    session_id text NOT NULL REFERENCES sk_session(session_id), consumer text NOT NULL,
    last_seq bigint NOT NULL DEFAULT 0 CHECK (last_seq >= 0), PRIMARY KEY(session_id,consumer)
);
-- Commit metadata is necessary for empty/fully overlapping commits and their original head_seq.
-- It owns no execution state; record bodies remain solely in sk_record.
CREATE TABLE sk_commit (
    session_id text NOT NULL REFERENCES sk_session(session_id), commit_id text NOT NULL,
    body bytea NOT NULL, digest text NOT NULL CHECK(length(digest)=64),
    record_sequences bytea NOT NULL, head_seq bigint NOT NULL CHECK(head_seq >= 0), created_ms bigint NOT NULL,
    PRIMARY KEY(session_id,commit_id)
);
"""


class PostgresStore(SQLStore):
    """One connection per thread. atomic() nests as a savepoint in host transactions."""

    lock_clause = " FOR UPDATE"
    read_lock_clause = " FOR SHARE"
    clock_sql = CLOCK

    def __init__(self, connection: Connection[Any]):
        self.connection = connection

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        import psycopg

        with psycopg.connect(conninfo, autocommit=True) as connection:
            yield cls(connection)

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        from psycopg.rows import tuple_row

        with self.connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(cast(LiteralString, query), params)
            return cursor.fetchall() if cursor.description is not None else []

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            with self.connection.transaction():
                yield
        except BaseException:
            self._fold_cache = None
            raise

    def _transaction_id(self) -> object:
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("SessionTx requires an open transaction")
        return self._one("SELECT pg_current_xact_id()::text")[0]

    def install_schema(self) -> None:
        with self._transaction():
            self._rows(DDL)


def join_transaction(connection: Connection[Any]) -> SQLSessionTx:
    """Join the host's open transaction; never open a connection or commit it."""
    return SQLSessionTx(PostgresStore(connection))
