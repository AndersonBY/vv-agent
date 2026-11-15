import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from time import monotonic
from uuid import uuid4

import pytest

from vv_agent.kernel import _postgres, _sqlite
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._sqlite import SQLiteStore

TABLES = {
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
SESSION_TABLES = TABLES - {"sk_settings"}

EXPECTED_COLUMNS = {
    "sk_settings": {"singleton", "storage_major", "write_level"},
    "sk_session": {
        "tenant_id",
        "session_id",
        "schema_version",
        "parent_session_id",
        "parent_operation_id",
        "head_seq",
        "inbox_seq",
        "lease_epoch",
        "lease_owner",
        "lease_until_ms",
        "phase",
        "active_turn_id",
        "next_drive_ms",
        "drive_retry_at_ms",
        "terminal_seq",
        "level",
        "handler_version",
        "required_capabilities",
        "deletion_state",
        "created_ms",
    },
    "sk_record": {
        "tenant_id",
        "session_id",
        "seq",
        "record_id",
        "schema_version",
        "kind",
        "turn_id",
        "operation_id",
        "attempt",
        "body",
        "digest",
        "chain_digest",
        "commit_id",
        "commit_digest",
        "commit_position",
        "writer_epoch",
        "created_ms",
    },
    "sk_inbox": {
        "tenant_id",
        "session_id",
        "input_id",
        "input_seq",
        "kind",
        "body",
        "digest",
        "admission_key_id",
        "admission_signature",
        "admission_signed_ms",
        "available_ms",
        "received_ms",
        "consumed_seq",
    },
    "sk_consumer": {"tenant_id", "session_id", "consumer", "last_seq", "project_retry_at_ms"},
    "sk_commit": {
        "tenant_id",
        "session_id",
        "commit_id",
        "body",
        "digest",
        "record_sequences",
        "head_seq",
        "created_ms",
    },
    "sk_checkpoint": {
        "tenant_id",
        "session_id",
        "seq",
        "chain_digest",
        "key_id",
        "signature",
        "signed_ms",
    },
    "sk_session_key": {
        "tenant_id",
        "session_id",
        "key_id",
        "provider_id",
        "wrapped_key",
        "state",
        "created_ms",
    },
    "sk_content": {
        "tenant_id",
        "session_id",
        "content_id",
        "key_id",
        "mac",
        "size",
        "labels",
        "nonce",
        "created_ms",
        "state",
        "pin_until_ms",
        "ciphertext",
        "object_locator",
    },
    "sk_content_link": {"tenant_id", "session_id", "content_id", "record_seq", "input_id", "slot"},
    "sk_tombstone": {
        "tenant_id",
        "session_id",
        "final_seq",
        "chain_digest",
        "deleted_ms",
        "actor",
        "key_id",
        "signature",
        "state",
        "scope_root_session_id",
    },
}

ZERO_DIGEST = "0" * 64


class Clock:
    def now_ms(self) -> int:
        return 0

    def monotonic(self) -> float:
        return 0.0

    def monotonic_ns(self) -> int:
        return 0

    def sleep(self, seconds: float) -> None:
        pass


def _fold_factory():
    raise AssertionError("schema tests do not execute a fold")


@contextmanager
def sqlite_store(path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore.standalone(str(path), fold_factory=_fold_factory, clock=Clock()) as store:
        yield store


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _sqlite_index_columns(connection: sqlite3.Connection, index: str) -> tuple[str, ...]:
    return tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index})"))


def _sqlite_foreign_keys(connection: sqlite3.Connection, table: str) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
    grouped: dict[int, list[tuple[int, str, str, str]]] = {}
    for row in connection.execute(f"PRAGMA foreign_key_list({table})"):
        grouped.setdefault(row[0], []).append((row[1], row[2], row[3], row[4]))
    return {(rows[0][1], tuple((source, target) for _, _, source, target in sorted(rows))) for rows in grouped.values()}


def _insert_sqlite_session(connection: sqlite3.Connection, tenant_id: str = "tenant-a", session_id: str = "s") -> None:
    connection.execute(
        "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
        "VALUES (?,?,'S1','idle',1,'live',0)",
        (tenant_id, session_id),
    )


def _insert_sqlite_record(connection: sqlite3.Connection, tenant_id: str = "tenant-a") -> None:
    connection.execute(
        "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,kind,body,digest,chain_digest,"
        "schema_version,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
        "VALUES (?,'s',1,'r','session_created',X'7B7D',?,?,'S1','c',?,0,0,0)",
        (tenant_id, ZERO_DIGEST, ZERO_DIGEST, ZERO_DIGEST),
    )


def test_sqlite_installs_complete_s1_schema_and_reopens(tmp_path):
    path = tmp_path / "s1.sqlite"
    with sqlite_store(path) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone() == (0,)
        store.install_schema()
        assert store.connection.execute("PRAGMA user_version").fetchone() == (_sqlite.SCHEMA_VERSION,)
        assert store.connection.execute("SELECT * FROM sk_settings").fetchall() == [(1, "S1", 1)]

        actual_tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sk\\_%' ESCAPE '\\'"
            )
        }
        assert actual_tables == TABLES
        assert {table: _sqlite_columns(store.connection, table) for table in TABLES} == EXPECTED_COLUMNS

        for table in SESSION_TABLES:
            assert next(row for row in store.connection.execute(f"PRAGMA table_info({table})") if row[1] == "tenant_id")[3] == 1
            assert store.connection.execute(f"PRAGMA foreign_key_check({table})").fetchall() == []
            indexes = store.connection.execute(f"PRAGMA index_list({table})").fetchall()
            assert indexes
            for index in indexes:
                assert "tenant_id" in _sqlite_index_columns(store.connection, index[1])

        session_pk = tuple(
            row[1] for row in sorted(store.connection.execute("PRAGMA table_info(sk_session)"), key=lambda row: row[5]) if row[5]
        )
        record_pk = tuple(
            row[1] for row in sorted(store.connection.execute("PRAGMA table_info(sk_record)"), key=lambda row: row[5]) if row[5]
        )
        assert session_pk == ("tenant_id", "session_id")
        assert record_pk == ("tenant_id", "session_id", "seq")

        assert (
            "sk_session",
            (("tenant_id", "tenant_id"), ("parent_session_id", "session_id")),
        ) in _sqlite_foreign_keys(store.connection, "sk_session")
        assert (
            "sk_record",
            (
                ("tenant_id", "tenant_id"),
                ("session_id", "session_id"),
                ("consumed_seq", "seq"),
            ),
        ) in _sqlite_foreign_keys(store.connection, "sk_inbox")

        store.install_schema()

    with sqlite_store(path) as reopened:
        assert reopened.connection.execute("PRAGMA user_version").fetchone() == (_sqlite.SCHEMA_VERSION,)
        assert reopened.connection.execute("SELECT storage_major FROM sk_settings").fetchone() == ("S1",)


@pytest.mark.parametrize("version", [1, 3, 5])
def test_sqlite_rejects_non_s1_user_version(tmp_path, version):
    path = tmp_path / f"version-{version}.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sk_session(session_id TEXT PRIMARY KEY)")
    connection.execute(f"PRAGMA user_version={version}")
    connection.close()

    with pytest.raises(ValueError, match="unsupported session SQLite schema version"), sqlite_store(path):
        pass


def test_sqlite_rejects_unversioned_or_incomplete_s1_tables(tmp_path):
    unversioned = tmp_path / "unversioned.sqlite"
    connection = sqlite3.connect(unversioned)
    connection.execute("CREATE TABLE sk_session(session_id TEXT PRIMARY KEY)")
    connection.close()
    with pytest.raises(ValueError, match="unsupported session SQLite schema version"), sqlite_store(unversioned):
        pass

    incomplete = tmp_path / "incomplete.sqlite"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE sk_session(session_id TEXT PRIMARY KEY)")
    connection.execute(f"PRAGMA user_version={_sqlite.SCHEMA_VERSION}")
    connection.close()
    with pytest.raises(ValueError, match="unsupported session SQLite schema version"), sqlite_store(incomplete):
        pass


def test_sqlite_s1_constraints_are_enforced(tmp_path):
    with sqlite_store(tmp_path / "constraints.sqlite") as store:
        store.install_schema()
        connection = store.connection
        _insert_sqlite_session(connection)
        assert connection.execute("SELECT required_capabilities FROM sk_session").fetchone() == (b"[]",)
        _insert_sqlite_record(connection)
        connection.execute(
            "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms) "
            "VALUES ('tenant-a','s','i',1,'user',X'7B7D',?,0,0)",
            (ZERO_DIGEST,),
        )
        connection.execute(
            "INSERT INTO sk_content(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,"
            "created_ms,state,ciphertext) VALUES ('tenant-a','s','c','k',?,0,X'7B7D',?,0,'staged',X'00')",
            (ZERO_DIGEST, b"\0" * 12),
        )
        connection.execute(
            "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,slot) VALUES ('tenant-a','s','c',1,'input')"
        )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sk_inbox SET admission_key_id='k' WHERE input_id='i'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES ('tenant-a','old','1','idle',1,'live',0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sk_content SET ciphertext=NULL WHERE content_id='c'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,input_id,slot) "
                "VALUES ('tenant-a','s','c',1,'i','both')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_sqlite_session(connection, "tenant-b", "child")
            connection.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,parent_session_id,parent_operation_id,"
                "phase,level,deletion_state,created_ms) "
                "VALUES ('tenant-c','orphan','S1','missing','op','idle',1,'live',0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sk_settings SET storage_major='S2'")


@pytest.fixture(scope="session")
def postgres_admin_dsn() -> str:
    psycopg = pytest.importorskip("psycopg", reason="PostgreSQL tests need the postgres extra")
    configured = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    for dsn in ([configured] if configured else []) + ["dbname=postgres"]:
        try:
            with psycopg.connect(dsn, autocommit=True, connect_timeout=3):
                return dsn
        except psycopg.OperationalError:
            continue
    pytest.skip("No PostgreSQL via VV_AGENT_TEST_POSTGRES_DSN or local unix socket")


@pytest.fixture
def postgres_database(postgres_admin_dsn: str) -> Iterator[str]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = f"vvsk_schema_{uuid4().hex}"
    with psycopg.connect(postgres_admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(postgres_admin_dsn, dbname=name)
        finally:
            deadline = monotonic() + 10
            while True:
                try:
                    admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
                    break
                except psycopg.errors.ObjectInUse:
                    if monotonic() >= deadline:
                        raise
                    Event().wait(0.05)


@pytest.mark.postgres
def test_postgres_rejects_non_s1_schema(postgres_database):
    import psycopg

    with psycopg.connect(postgres_database, autocommit=True) as connection:
        connection.execute("CREATE TABLE sk_session(session_id text PRIMARY KEY)")
        store = PostgresStore(connection, fold_factory=_fold_factory)
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            store.install_schema()


@pytest.mark.postgres
def test_postgres_installs_complete_s1_schema(postgres_database):
    import psycopg

    with psycopg.connect(postgres_database, autocommit=True) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        store.install_schema()

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname=current_schema() AND tablename LIKE 'sk\\_%' ESCAPE '\\'"
            )
        }
        assert tables == TABLES
        assert connection.execute("SELECT * FROM sk_settings").fetchall() == [(1, "S1", 1)]
        columns = {
            row[0]: {
                item[0]
                for item in connection.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s",
                    (row[0],),
                )
            }
            for row in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname=current_schema() AND tablename=ANY(%s)", (list(TABLES),)
            )
        }
        assert columns == EXPECTED_COLUMNS

        for table in SESSION_TABLES:
            assert connection.execute(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s AND column_name='tenant_id'",
                (table,),
            ).fetchone() == ("NO",)
            indexes = connection.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND tablename=%s",
                (table,),
            ).fetchall()
            assert indexes
            assert all("tenant_id" in row[0] for row in indexes)

        constraints = connection.execute(
            "SELECT conrelid::regclass::text,pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE connamespace=current_schema()::regnamespace AND conrelid::regclass::text=ANY(%s)",
            (sorted(SESSION_TABLES),),
        ).fetchall()
        foreign_keys = [definition for table, definition in constraints if definition.startswith("FOREIGN KEY")]
        assert foreign_keys
        assert all(definition.startswith("FOREIGN KEY (tenant_id") for definition in foreign_keys)

        assert _postgres.RLS_ISOLATION == "deferred"
        assert "not installed" in _postgres.RLS_DEFERRED_REASON
        assert not any(
            row[0]
            for row in connection.execute(
                "SELECT rowsecurity FROM pg_tables WHERE schemaname=current_schema() AND tablename=ANY(%s)",
                (sorted(SESSION_TABLES),),
            )
        )

        connection.execute(
            "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
            "VALUES ('tenant-a','s','S1','idle',1,'live',0)"
        )
        assert connection.execute("SELECT required_capabilities FROM sk_session").fetchone() == (b"[]",)
        connection.execute(
            "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,kind,body,digest,chain_digest,"
            "schema_version,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
            "VALUES ('tenant-a','s',1,'r','session_created',%s,%s,%s,'S1','c',%s,0,0,0)",
            (b"{}", ZERO_DIGEST, ZERO_DIGEST, ZERO_DIGEST),
        )
        connection.execute(
            "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms) "
            "VALUES ('tenant-a','s','i',1,'user',%s,%s,0,0)",
            (b"{}", ZERO_DIGEST),
        )
        connection.execute(
            "INSERT INTO sk_content(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,"
            "created_ms,state,ciphertext) VALUES ('tenant-a','s','c','k',%s,0,%s,%s,0,'staged',%s)",
            (ZERO_DIGEST, b"{}", b"\0" * 12, b"ciphertext"),
        )
        with pytest.raises(psycopg.IntegrityError):
            connection.execute("UPDATE sk_inbox SET admission_key_id='k' WHERE input_id='i'")
        with pytest.raises(psycopg.IntegrityError):
            connection.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES ('tenant-a','old','1','idle',1,'live',0)"
            )
        with pytest.raises(psycopg.IntegrityError):
            connection.execute(
                "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,kind,body,digest,chain_digest,"
                "schema_version,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
                "VALUES ('tenant-b','s',2,'r2','session_created',%s,%s,%s,'S1','c2',%s,0,0,0)",
                (b"{}", ZERO_DIGEST, ZERO_DIGEST, ZERO_DIGEST),
            )
        with pytest.raises(psycopg.IntegrityError):
            connection.execute(
                "INSERT INTO sk_content(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,"
                "created_ms,state,ciphertext,object_locator) VALUES ('tenant-a','s','bad','k',%s,0,%s,%s,0,'staged',%s,%s)",
                (ZERO_DIGEST, b"{}", b"\0" * 12, b"ciphertext", b"locator"),
            )
        with pytest.raises(psycopg.IntegrityError):
            connection.execute(
                "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,input_id,slot) "
                "VALUES ('tenant-a','s','c',1,'i','both')"
            )
        with pytest.raises(psycopg.IntegrityError):
            connection.execute("UPDATE sk_settings SET storage_major='S2'")
