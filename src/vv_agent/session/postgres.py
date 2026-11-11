"""Internal PostgreSQL reference store, bound to caller-owned psycopg v3 connections."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock, get_ident
from typing import TYPE_CHECKING, Any, LiteralString, cast

from .sql import SQLSessionTx, SQLStore
from .store import Lease

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
    active_turn_id text, next_drive_ms bigint,
    drive_retry_at_ms bigint NOT NULL DEFAULT 0 CHECK (drive_retry_at_ms >= 0), terminal_seq bigint NOT NULL DEFAULT 0,
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
    kind text NOT NULL CHECK (kind IN ('user','steer','follow_up','provider_result','approval_answer','child_result',
        'control','provider_evidence','model_selection')),
    body bytea NOT NULL, digest text NOT NULL CHECK (length(digest)=64),
    available_ms bigint NOT NULL, received_ms bigint NOT NULL, consumed_seq bigint,
    PRIMARY KEY(session_id,input_id), UNIQUE(session_id,input_seq),
    FOREIGN KEY(session_id,consumed_seq) REFERENCES sk_record(session_id,seq)
);
CREATE INDEX sk_inbox_ready ON sk_inbox(session_id,available_ms,input_seq) WHERE consumed_seq IS NULL;
CREATE TABLE sk_consumer (
    session_id text NOT NULL REFERENCES sk_session(session_id), consumer text NOT NULL,
    last_seq bigint NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
    project_retry_at_ms bigint NOT NULL DEFAULT 0 CHECK (project_retry_at_ms >= 0), PRIMARY KEY(session_id,consumer)
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


def _connection_conninfo(connection: Connection[Any]) -> str:
    from psycopg.conninfo import make_conninfo

    info = connection.info
    # ConnectionInfo.dsn deliberately omits the password.
    return make_conninfo(info.dsn, password=info.password) if info.password else info.dsn


class PostgresStore(SQLStore):
    """A serialized connection; atomic() nests as a savepoint in host transactions."""

    # Session keys never change. This still serializes writers while allowing
    # child inserts to retain their parent foreign-key KEY SHARE locks.
    lock_clause = " FOR NO KEY UPDATE"
    clock_sql = CLOCK

    def __init__(self, connection: Connection[Any]):
        self._initialize(connection)

    def _initialize(self, connection, *, source=None, input_only=False):
        from psycopg.pq import TransactionStatus

        from .ownership import OwnershipError, register_host, restore_host

        if connection.closed:
            raise OwnershipError(stage="initialization-closed")
        super().__init__()
        self.connection = connection
        self._connection_lock = RLock()
        self._target = None
        self._conninfo = None
        self._initial_thread = get_ident()
        from .ownership import native_keys

        self._ownership_keys = native_keys(self)
        previous = register_host(self, input_only=input_only) if source is None else None
        try:
            self._configure_target = source is not None
            if source is not None:
                self._target = source._target
                self._conninfo = source._conninfo
                return
            if input_only or not connection.autocommit or connection.info.transaction_status != TransactionStatus.IDLE:
                return
            self._conninfo = _connection_conninfo(connection)
            self._target = self._freeze_target()
        except BaseException:
            if previous is not None:
                restore_host(self, previous)
            raise

    def _freeze_target(self):
        from .ownership import OwnershipError

        db = self._one("SELECT oid FROM pg_database WHERE datname=current_database()")[0]
        relations = self._rows(
            "SELECT c.relname,c.oid,n.oid,n.nspname FROM "
            "unnest(ARRAY['sk_session','sk_record','sk_inbox','sk_consumer','sk_commit']) AS t(name) "
            "JOIN pg_class c ON c.oid=to_regclass(t.name) JOIN pg_namespace n ON n.oid=c.relnamespace "
            "ORDER BY c.relname"
        )
        if not relations:
            return None
        if len(relations) != 5 or len({r[2] for r in relations}) != 1 or relations[0][3].startswith("pg_temp_"):
            raise OwnershipError(stage="initialization-tables")
        info = self.connection.info
        return (db, relations[0][3], tuple(relations), info.host, info.port)

    def _validate_checkout(self, source):
        from psycopg.pq import TransactionStatus

        from .ownership import OwnershipError

        if (
            self.connection.closed
            or not self.connection.autocommit
            or self.connection.info.transaction_status != TransactionStatus.IDLE
        ):
            raise OwnershipError(stage="idle")
        if self._configure_target:
            from psycopg import sql

            self.connection.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(source._target[1])))
            self._configure_target = False
        if source._target is None or self._freeze_target() != source._target:
            raise OwnershipError(stage="target")

    def execution_store(self):
        from .ownership import OwnershipError

        if self._target is None:
            raise OwnershipError(stage="setup-only")
        return self._checkout()

    def heartbeat_store(self):
        return self.execution_store()

    @contextmanager
    def _checkout(self):
        import psycopg
        from psycopg.conninfo import make_conninfo

        assert self._conninfo is not None
        store = None
        connection = None
        error = None
        yielded = False
        try:
            connection = psycopg.connect(make_conninfo(self._conninfo, connect_timeout=3), autocommit=True)
            store = PostgresStore.__new__(PostgresStore)
            store._initialize(connection, source=self)
            from .ownership import OwnershipError, native_keys, register_host

            if set(native_keys(store)) & set(self._ownership_keys):
                raise OwnershipError(stage="factory-alias")
            register_host(store)
            store._validate_checkout(self)
            with connection:
                yielded = True
                yield store

        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                if connection is not None and not connection.closed:
                    try:
                        connection.close()
                    except BaseException as cleanup:
                        if error is None:
                            raise
                        error.add_note(f"checkout close failed: {type(cleanup).__name__}")
            finally:
                if store is not None and (not yielded or (connection is not None and connection.closed)):
                    from .ownership import forget_host

                    forget_host(store)

    @contextmanager
    def _bound_append(self, lease: Lease) -> Iterator[None]:
        # Rollback restores settings on failure; success must restore them explicitly.
        with self._transaction():
            previous = self._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")
            self._rows(
                f"WITH remaining AS (SELECT GREATEST(1,%s-({self.clock_sql})) AS ms) "
                "SELECT set_config('lock_timeout',ms::text || 'ms',true), "
                "set_config('statement_timeout',ms::text || 'ms',true) FROM remaining",
                (lease.expires_at_ms,),
            )
            yield
            self._rows(
                "SELECT set_config('lock_timeout',%s,true),set_config('statement_timeout',%s,true)",
                previous,
            )

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        import psycopg

        with psycopg.connect(conninfo, autocommit=True) as connection:
            yield cls(connection)

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        from psycopg.rows import tuple_row

        with self._connection_lock, self.connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(cast(LiteralString, query), params, binary=query.lstrip().startswith("SELECT"))
            return cursor.fetchall() if cursor.description is not None else []

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            with self._connection_lock, self.connection.transaction():
                yield
        except BaseException:
            self._fold_cache = None
            self._previous_prefix = None
            raise

    def _transaction_id(self) -> object:
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("SessionTx requires an open transaction")
        return self._one("SELECT pg_current_xact_id()::text")[0]

    def install_schema(self) -> None:
        with self._transaction():
            self._rows(DDL)
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status == TransactionStatus.IDLE:
            self._conninfo = _connection_conninfo(self.connection)
            self._target = self._freeze_target()


def join_transaction(connection: Connection[Any]) -> SQLSessionTx:
    """Join the host's open transaction; never open a connection or commit it."""
    store = PostgresStore.__new__(PostgresStore)
    store._initialize(connection, input_only=True)
    return SQLSessionTx(store)
