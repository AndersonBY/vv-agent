from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from uuid import uuid4

import pytest
from test_s1_postgres_rls import TENANT_A, TENANT_B, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel._content import ContentUnavailable, KeyUnavailable
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.kernel._store import Conflict
from vv_agent.session.postgres import PostgresStore


@pytest.fixture
def object_publication(rls_database):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = rls_database
    _provision(database)
    cluster_dsn = _dsn(database, database.cluster.admin_role, database.cluster.admin_password)
    owner_dsn = _dsn(database, database.migration_role, database.migration_password)
    primary = {
        TENANT_A: _dsn(database, database.tenant_a_role, database.tenant_a_password),
        TENANT_B: _dsn(database, database.tenant_b_role, database.tenant_b_password),
    }
    nonce = uuid4().hex
    journal_database = f"vv_journal_{nonce}"
    password = uuid4().hex
    roles = {
        f"vv_journal_{index}_{nonce}": role
        for index, role in enumerate(
            [JournalRole("anonymous")]
            + [
                role
                for tenant in primary
                for role in (JournalRole("binder", tenant), JournalRole("scope", tenant), JournalRole("key-intent", tenant))
            ]
        )
    }
    with psycopg.connect(cluster_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(journal_database)))
        try:
            for role in roles:
                admin.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOINHERIT").format(sql.Identifier(role), sql.Literal(password))
                )
            with PostgresStore.schema_admin_standalone(owner_dsn) as owner:
                (database_id,) = owner._one("SELECT database_id FROM sk_settings")
                with psycopg.connect(make_conninfo(cluster_dsn, dbname=journal_database), autocommit=True) as journal_admin:
                    identity = PostgresAllocationJournal.provision(
                        journal_admin, database_id=database_id, namespace="objects-test", roles=roles
                    )
                logins = {
                    role: PostgresAllocationJournal(
                        make_conninfo(cluster_dsn, dbname=journal_database, user=login, password=password), identity, role
                    )
                    for login, role in roles.items()
                }
                journals = {
                    tenant: PostgresObjectJournal(
                        anonymous=logins[JournalRole("anonymous")],
                        binder=logins[JournalRole("binder", tenant)],
                        scope=logins[JournalRole("scope", tenant)],
                        keys=logins[JournalRole("key-intent", tenant)],
                    )
                    for tenant in primary
                }
                yield owner, primary, journals
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(journal_database)))
            for role in roles:
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def test_postgres_publication_checks_tenant_binding_and_outer_transaction(object_publication, monkeypatch):
    from psycopg import IsolationLevel

    owner, primary, journals = object_publication
    journal = journals[TENANT_A]

    def check_deadline():
        pass

    with PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as store:
        published_sql = []
        one = store._one

        def capture_one(query, params=()):
            if "pg_try_advisory_xact_lock" in query:
                published_sql.append(query)
            return one(query, params)

        monkeypatch.setattr(store, "_one", capture_one)
        with (
            pytest.raises(ContentUnavailable, match="binding does not match"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("unprovisioned primary accepted publication")
        owner._provision_object_journal(journal)
        with pytest.raises(ContentUnavailable, match="unbound PostgreSQL installation"):
            owner._provision_object_journal(journal)
        with pytest.raises(ContentUnavailable, match="schema owner"):
            store._provision_object_journal(journal)
        with (
            pytest.raises(ContentUnavailable, match="authenticated journal tenant"),
            store._object_publication_atomic(journals[TENANT_B], check_deadline=check_deadline),
        ):
            pytest.fail("different journal tenant accepted publication")
        with (
            store.connection.transaction(),
            pytest.raises(ContentUnavailable, match="store-owned outer"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("ambient transaction accepted publication")
        with (
            store.atomic(),
            pytest.raises(ContentUnavailable, match="store-owned outer"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("store savepoint accepted publication")
        store.connection.isolation_level = IsolationLevel.REPEATABLE_READ
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("repeatable read accepted publication")
        store.connection.isolation_level = IsolationLevel.READ_COMMITTED
        store.connection.read_only = True
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("read-only transaction accepted publication")
        store.connection.read_only = False
        store._one("SELECT set_config('vv.tenant_id',%s,false)", (TENANT_B,))
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("inconsistent tenant context accepted publication")
        store._one("SELECT set_config('vv.tenant_id',%s,false)", (TENANT_A,))
        for change in ({"database_id": "0" * 64}, {"journal_id": "0" * 64}, {"namespace": "other-namespace"}):
            substituted = replace(journal.identity, **change)
            with monkeypatch.context() as patch:
                patch.setattr(PostgresObjectJournal, "identity", property(lambda _journal, value=substituted: value))
                with (
                    pytest.raises(ContentUnavailable, match="binding does not match"),
                    store._object_publication_atomic(journal, check_deadline=check_deadline),
                ):
                    pytest.fail("changed journal identity accepted publication")
        ((binding,),) = owner._rows(
            "UPDATE sk_settings SET object_journal_binding=%s RETURNING object_journal_binding", ("0" * 64,)
        )
        assert binding == "0" * 64
        with (
            pytest.raises(ContentUnavailable, match="binding does not match"),
            store._object_publication_atomic(journal, check_deadline=check_deadline),
        ):
            pytest.fail("changed durable binding accepted publication")
        assert published_sql == []


def test_postgres_publication_is_nonblocking_and_releases_on_commit_and_rollback(object_publication):
    owner, primary, journals = object_publication
    journal = journals[TENANT_A]
    owner._provision_object_journal(journal)
    expired = False

    def check_deadline():
        if expired:
            raise KeyUnavailable("test publication deadline expired")

    def enter(store):
        with store._object_publication_atomic(journal, check_deadline=check_deadline):
            pytest.fail("another thread entered with the same primary connection")

    with (
        PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as first,
        PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as same,
        PostgresStore.tenant_standalone(primary[TENANT_B], tenant_id=TENANT_B) as other,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        same._one("SELECT set_config('statement_timeout','400ms',false)")
        for expires in (False, True):
            outcome = pytest.raises(KeyUnavailable, match="test publication deadline expired") if expires else nullcontext()
            with outcome, first._object_publication_atomic(journal, check_deadline=check_deadline) as (_, check_fence):
                contender = pool.submit(enter, first)
                with pytest.raises(Conflict, match="connection is busy"):
                    contender.result(timeout=2)
                with (
                    pytest.raises(Conflict, match="publication is busy"),
                    same._object_publication_atomic(journal, check_deadline=check_deadline),
                ):
                    pytest.fail("same tenant entered while the primary transaction held the lock")
                with other._object_publication_atomic(journals[TENANT_B], check_deadline=check_deadline):
                    pass
                check_fence()
                expired = expires
            expired = False
            with same._object_publication_atomic(journal, check_deadline=check_deadline):
                pass
            with pytest.raises(RuntimeError, match="SessionTx requires an open transaction"):
                check_fence()
