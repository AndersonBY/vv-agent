"""Content backends expose one short, caller-safe SQL read snapshot."""

from __future__ import annotations

import sqlite3

import pytest
from test_s1_postgres_rls import (
    TENANT_A,
    TENANT_B,
    RLSDatabase,
    _dsn,
    _provision,
)
from test_s1_postgres_rls import (
    disposable_pg as disposable_pg,
)
from test_s1_postgres_rls import (
    rls_database as rls_database,
)

from vv_agent.kernel._postgres import CorePostgresContentStore, _PostgresContentBackend
from vv_agent.kernel._sqlite import CoreSQLiteContentStore, _SQLiteContentBackend
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore

from .helpers import create_fixture_session, wrap_fixture_store


def _provider() -> LocalKeyProvider:
    return LocalKeyProvider(b"k" * 32)


def _sqlite_content_query() -> str:
    return (
        "SELECT s.session_id,k.key_id,c.content_id,c.state "
        "FROM sk_session AS s "
        "JOIN sk_session_key AS k USING (tenant_id,session_id) "
        "JOIN sk_content AS c USING (tenant_id,session_id) "
        "WHERE s.tenant_id={p} AND s.session_id={p} ORDER BY c.content_id"
    )


def test_sqlite_snapshot_is_stable_while_another_connection_commits(tmp_path):
    path = tmp_path / "snapshot.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        wrapped = wrap_fixture_store(store)
        create_fixture_session(wrapped, "s")
        content = CoreSQLiteContentStore(store, _provider(), AESGCMCipher())
        backend = content._backend
        assert isinstance(backend, _SQLiteContentBackend)
        assert backend.connection is store.connection
        assert backend.connection_lock is store._connection_lock

        writer = sqlite3.connect(path, isolation_level=None)
        try:
            with backend.read_snapshot():
                assert store.connection.in_transaction
                before = backend.fetchall(_sqlite_content_query(), ("default", "s"))
                assert before
                changed = writer.execute(
                    "UPDATE sk_content SET state='gc_pending' "
                    "WHERE tenant_id='default' AND session_id='s' AND state IN ('staged','live')"
                ).rowcount
                assert changed > 0
                during = backend.fetchall(_sqlite_content_query(), ("default", "s"))
                assert during == before

            assert not store.connection.in_transaction
            after = backend.fetchall(_sqlite_content_query(), ("default", "s"))
            assert after != before
            assert {row[3] for row in after} == {"gc_pending"}
        finally:
            writer.close()


def test_sqlite_snapshot_joins_caller_transaction_without_commit_or_rollback(tmp_path):
    path = tmp_path / "caller-transaction.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        wrapped = wrap_fixture_store(store)
        create_fixture_session(wrapped, "s")
        backend = CoreSQLiteContentStore(store, _provider(), AESGCMCipher())._backend
        assert isinstance(backend, _SQLiteContentBackend)
        assert backend.connection is store.connection
        original = backend.fetchone(
            "SELECT state FROM sk_content WHERE tenant_id={p} AND session_id={p} ORDER BY content_id LIMIT 1",
            ("default", "s"),
        )
        assert original in {("staged",), ("live",)}

        store.connection.execute("BEGIN")
        store.connection.execute(
            "UPDATE sk_content SET state='gc_pending' WHERE tenant_id='default' AND session_id='s' AND state IN ('staged','live')"
        )
        with pytest.raises(RuntimeError, match="caller owns failure"), backend.read_snapshot():
            assert store.connection.in_transaction
            assert backend.fetchone(
                "SELECT state FROM sk_content WHERE tenant_id={p} AND session_id={p} ORDER BY content_id LIMIT 1",
                ("default", "s"),
            ) == ("gc_pending",)
            raise RuntimeError("caller owns failure")
        assert store.connection.in_transaction
        store.connection.execute("ROLLBACK")
        assert (
            backend.fetchone(
                "SELECT state FROM sk_content WHERE tenant_id={p} AND session_id={p} ORDER BY content_id LIMIT 1",
                ("default", "s"),
            )
            == original
        )


def _seed_tenant_content(database: RLSDatabase) -> None:
    import psycopg

    with psycopg.connect(
        _dsn(database, database.superuser_role, database.superuser_password),
        autocommit=True,
    ) as admin:
        for tenant_id, session_id, suffix in (
            (TENANT_A, "a-owned", "a"),
            (TENANT_B, "b-owned", "b"),
        ):
            admin.execute(
                "INSERT INTO sk_session_key"
                "(tenant_id,session_id,key_id,provider_id,wrapped_key,state,created_ms) "
                "VALUES (%s,%s,%s,'test',%s,'active',0)",
                (tenant_id, session_id, f"key-{suffix}", b"wrapped-" + suffix.encode()),
            )
            admin.execute(
                "INSERT INTO sk_content"
                "(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,created_ms,state,pin_until_ms,"
                "ciphertext,object_locator) "
                "VALUES (%s,%s,%s,%s,%s,1,%s,%s,0,'live',NULL,%s,NULL)",
                (
                    tenant_id,
                    session_id,
                    f"content-{suffix}",
                    f"key-{suffix}",
                    suffix * 64,
                    b"labels",
                    suffix.encode() * 12,
                    b"ciphertext",
                ),
            )


def _content_query() -> str:
    return (
        "SELECT s.session_id,k.key_id,c.content_id,c.state "
        "FROM sk_session AS s "
        "JOIN sk_session_key AS k USING (tenant_id,session_id) "
        "JOIN sk_content AS c USING (tenant_id,session_id) "
        "WHERE s.tenant_id={p} AND s.session_id={p}"
    )


@pytest.mark.postgres
def test_postgres_snapshot_preserves_verified_tenant_scope_and_isolation(rls_database):
    import psycopg
    from psycopg.pq import TransactionStatus

    _provision(rls_database)
    _seed_tenant_content(rls_database)
    tenant_dsn = _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password)
    with PostgresStore.tenant_standalone(tenant_dsn, tenant_id=TENANT_A) as store:
        assert store.credential_tenant_id == TENANT_A
        content = CorePostgresContentStore(store, _provider(), AESGCMCipher())
        backend = content._backend
        assert isinstance(backend, _PostgresContentBackend)
        assert backend.connection is store.connection
        assert backend.credential_tenant_id == TENANT_A
        assert backend.connection_lock is store._connection_lock

        with (
            psycopg.connect(
                _dsn(rls_database, rls_database.superuser_role, rls_database.superuser_password),
                autocommit=True,
            ) as writer,
            backend.read_snapshot(),
        ):
            assert store.connection.info.transaction_status == TransactionStatus.INTRANS
            before = backend.fetchall(_content_query(), (TENANT_A, "a-owned"))
            assert before == [("a-owned", "key-a", "content-a", "live")]
            assert backend.fetchall(_content_query(), (TENANT_B, "b-owned")) == []
            writer.execute(
                "UPDATE sk_content SET state='gc_pending' WHERE tenant_id=%s AND session_id=%s AND content_id='content-a'",
                (TENANT_A, "a-owned"),
            )
            during = backend.fetchall(_content_query(), (TENANT_A, "a-owned"))
            assert during == before

        assert store.connection.info.transaction_status == TransactionStatus.IDLE
        with backend.read_snapshot():
            assert backend.fetchall(_content_query(), (TENANT_A, "a-owned")) == [("a-owned", "key-a", "content-a", "gc_pending")]
        with pytest.raises(psycopg.errors.UndefinedTable), backend.read_snapshot():
            backend.fetchone("SELECT * FROM missing_content_snapshot_table")
        assert store.connection.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.postgres
def test_postgres_snapshot_joins_caller_transaction_and_rejects_inerror(rls_database):
    from psycopg.pq import TransactionStatus

    _provision(rls_database)
    tenant_dsn = _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password)
    with PostgresStore.tenant_standalone(tenant_dsn, tenant_id=TENANT_A) as store:
        with pytest.raises(RuntimeError, match="caller owns failure"), store.atomic() as tx:
            isolation_row = store.connection.execute("SHOW transaction_isolation").fetchone()
            assert isolation_row is not None
            isolation = isolation_row[0]
            store.connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (TENANT_A,))
            store.connection.execute(
                "UPDATE sk_session SET phase='parked' WHERE tenant_id=%s AND session_id='a-owned'",
                (TENANT_A,),
            )
            content = CorePostgresContentStore(tx, _provider(), AESGCMCipher())
            backend = content._backend
            assert isinstance(backend, _PostgresContentBackend)
            assert backend.connection is store.connection
            assert backend.connection_lock is store._connection_lock
            assert backend.credential_tenant_id == TENANT_A
            store.connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (TENANT_B,))
            with backend.read_snapshot():
                assert store.connection.info.transaction_status == TransactionStatus.INTRANS
                assert (
                    backend.fetchone(
                        "SELECT phase FROM sk_session WHERE tenant_id={p} AND session_id={p}",
                        (TENANT_A, "a-owned"),
                    )
                    is None
                )
                current_context = store.connection.execute("SELECT current_setting('vv.tenant_id',true)").fetchone()
                assert current_context == (TENANT_B,)
            assert store.connection.info.transaction_status == TransactionStatus.INTRANS
            joined_isolation_row = store.connection.execute("SHOW transaction_isolation").fetchone()
            assert joined_isolation_row is not None
            assert joined_isolation_row[0] == isolation
            raise RuntimeError("caller owns failure")
        assert store.connection.info.transaction_status == TransactionStatus.IDLE
        with backend.read_snapshot():
            assert backend.fetchone(
                "SELECT phase FROM sk_session WHERE tenant_id={p} AND session_id={p}",
                (TENANT_A, "a-owned"),
            ) == ("idle",)

        store.connection.execute("BEGIN")
        import psycopg

        with pytest.raises(psycopg.errors.DivisionByZero):
            store.connection.execute("SELECT 1/0")
        assert store.connection.info.transaction_status == TransactionStatus.INERROR
        with pytest.raises(RuntimeError, match="INERROR"), backend.read_snapshot():
            pass
        assert store.connection.info.transaction_status == TransactionStatus.INERROR
        store.connection.rollback()
        assert store.connection.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.postgres
def test_shared_gateway_snapshot_does_not_infer_tenant_scope(rls_database):
    _provision(rls_database)
    _seed_tenant_content(rls_database)
    shared_dsn = _dsn(rls_database, rls_database.gateway_role, rls_database.gateway_password)
    with PostgresStore.shared_gateway_standalone(shared_dsn) as store:
        backend = CorePostgresContentStore(store, _provider(), AESGCMCipher())._backend
        assert isinstance(backend, _PostgresContentBackend)
        assert backend.connection is store.connection
        assert backend.credential_tenant_id is None
        with backend.read_snapshot():
            assert backend.fetchall(_content_query(), (TENANT_A, "a-owned")) == []

        with store.connection.transaction():
            store.connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (TENANT_A,))
            with backend.read_snapshot():
                assert backend.fetchall(_content_query(), (TENANT_A, "a-owned")) == [("a-owned", "key-a", "content-a", "live")]
