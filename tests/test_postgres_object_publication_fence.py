from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import replace
from uuid import uuid4

import pytest
from session.test_s3_reservation_adapter import _Client
from test_s1_postgres_rls import TENANT_A, TENANT_B, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel._content import ContentRef, ContentUnavailable, KeyUnavailable
from vv_agent.kernel._content_runtime import ContentRuntimeUnavailable, _provision_postgres_object_content, bind_content_runtime
from vv_agent.kernel._content_sql import _read_ciphertext
from vv_agent.kernel._external_content import _PostgresObjectContent
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.kernel._store import Conflict
from vv_agent.session.content_runtime import SESSION_WORKSPACE
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.fork import fork_thread
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem
from vv_agent.session.s3_content import S3CiphertextStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock


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
                owner.install_tenant_rls(TENANT_A, database.tenant_a_role, database.tenant_a_password)
                owner.install_tenant_rls(TENANT_B, database.tenant_b_role, database.tenant_b_password)
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


_TEST_CONTENT_DIGEST = "1" * 64


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
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("unprovisioned primary accepted publication")
        with pytest.raises(ValueError, match="SHA-256 hex digest"):
            owner._provision_object_journal(journal, content_digest="incomplete")
        owner._provision_object_journal(journal, content_digest=_TEST_CONTENT_DIGEST)
        with (
            pytest.raises(ValueError, match="SHA-256 hex digest"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest="incomplete"),
        ):
            pytest.fail("invalid digest accepted publication")
        with pytest.raises(ContentUnavailable, match="unbound PostgreSQL installation"):
            owner._provision_object_journal(journal, content_digest=_TEST_CONTENT_DIGEST)
        with pytest.raises(ContentUnavailable, match="schema owner"):
            store._provision_object_journal(journal, content_digest=_TEST_CONTENT_DIGEST)
        with (
            pytest.raises(ContentUnavailable, match="authenticated journal tenant"),
            store._object_publication_atomic(
                journals[TENANT_B], check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST
            ),
        ):
            pytest.fail("different journal tenant accepted publication")
        with (
            store.connection.transaction(),
            pytest.raises(ContentUnavailable, match="store-owned outer"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("ambient transaction accepted publication")
        with (
            store.atomic(),
            pytest.raises(ContentUnavailable, match="store-owned outer"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("store savepoint accepted publication")
        store.connection.isolation_level = IsolationLevel.REPEATABLE_READ
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("repeatable read accepted publication")
        store.connection.isolation_level = IsolationLevel.READ_COMMITTED
        store.connection.read_only = True
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("read-only transaction accepted publication")
        store.connection.read_only = False
        store._one("SELECT set_config('vv.tenant_id',%s,false)", (TENANT_B,))
        with (
            pytest.raises(ContentUnavailable, match="READ COMMITTED"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("inconsistent tenant context accepted publication")
        store._one("SELECT set_config('vv.tenant_id',%s,false)", (TENANT_A,))
        for change in ({"database_id": "0" * 64}, {"journal_id": "0" * 64}, {"namespace": "other-namespace"}):
            substituted = replace(journal.identity, **change)
            with monkeypatch.context() as patch:
                patch.setattr(PostgresObjectJournal, "identity", property(lambda _journal, value=substituted: value))
                with (
                    pytest.raises(ContentUnavailable, match="binding does not match"),
                    store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
                ):
                    pytest.fail("changed journal identity accepted publication")
        ((binding,),) = owner._rows(
            "UPDATE sk_settings SET object_journal_binding=%s RETURNING object_journal_binding", ("0" * 64,)
        )
        assert binding == "0" * 64
        with (
            pytest.raises(ContentUnavailable, match="binding does not match"),
            store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
        ):
            pytest.fail("changed durable binding accepted publication")
        assert published_sql == []


def test_postgres_publication_is_nonblocking_and_releases_on_commit_and_rollback(object_publication):
    owner, primary, journals = object_publication
    journal = journals[TENANT_A]
    owner._provision_object_journal(journal, content_digest=_TEST_CONTENT_DIGEST)
    expired = False

    def check_deadline():
        if expired:
            raise KeyUnavailable("test publication deadline expired")

    def enter(store):
        with store._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST):
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
            with (
                outcome,
                first._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST) as (
                    _,
                    check_fence,
                ),
            ):
                contender = pool.submit(enter, first)
                with pytest.raises(Conflict, match="connection is busy"):
                    contender.result(timeout=2)
                with (
                    pytest.raises(Conflict, match="publication is busy"),
                    same._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST),
                ):
                    pytest.fail("same tenant entered while the primary transaction held the lock")
                with other._object_publication_atomic(
                    journals[TENANT_B], check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST
                ):
                    pass
                check_fence()
                expired = expires
            expired = False
            with same._object_publication_atomic(journal, check_deadline=check_deadline, content_digest=_TEST_CONTENT_DIGEST):
                pass
            with pytest.raises(RuntimeError, match="SessionTx requires an open transaction"):
                check_fence()


def test_postgres_sdk_root_object_create_push_and_cold_read(object_publication, tmp_path, monkeypatch):
    owner, primary, journals = object_publication
    journal = journals[TENANT_A]

    class Client(_Client):
        def put_object(self, **request):
            if "IfMatch" in request:
                allocation_id = bytes.fromhex(request["Key"].rsplit("/", 1)[1])
                row = journal._scope.get(allocation_id)
                assert row is not None and row.owner is not None
                key = journal.get_key_intent(row.owner.provider_instance_id, row.owner.key_id)
                assert key is not None and key.state == "allocating"
                assert row.owner.tenant_id == TENANT_A and row.owner.session_id == "objects"
                self.owner = row.owner
            return super().put_object(**request)

    key_path = tmp_path / "keys.sqlite"
    master_key = b"x" * 32
    client = Client(lambda allocation_id: journal._scope.get(allocation_id) or journal.get_anonymous(allocation_id))

    with closing(LocalKeyProvider(master_key, state_path=key_path)) as provider:
        provider_instance = provider._provision_exact_key_lifecycle()
        objects = _PostgresObjectContent(journal, S3CiphertextStore(client, bucket="fixture", journal=journal), provider_instance)
        _provision_postgres_object_content(owner, key_provider=provider, objects=objects)
        with PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as store:
            store.configure_input_admission(allow_unsigned=True)
            bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
            item = InboxItem("input", "user", {"content": "object payload"})
            with closing(SessionDriver(store=store)) as driver:
                with (
                    PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as contender,
                    contender._object_publication_atomic(
                        journal, content_digest=objects._binding_digest(), check_deadline=lambda: None
                    ),
                    pytest.raises(Conflict, match="publication is busy"),
                ):
                    driver.create("objects", "/rejected")
                assert not client.calls
                assert journal.key_intents_for_admission(TENANT_A, "objects", "objects", "session/create") == ()
                driver.create("objects", "/workspace")
                before = len(client.calls)
                with pytest.raises(ContentRuntimeUnavailable, match="root user input"):
                    driver.push("objects", replace(item, target_turn_id="turn", generation=1))
                with pytest.raises(ContentRuntimeUnavailable, match="worker controls"):
                    driver.control("objects", "close", "control")
                with pytest.raises(ContentRuntimeUnavailable, match="fork is unavailable"):
                    fork_thread(driver, None, {})
                assert len(client.calls) == before
                assert store.peek_inbox("objects") == ()
                assert not driver.push("objects", item).replayed
                before = len(client.calls)
                assert driver.push("objects", item).replayed
                assert len(client.calls) == before

                now = 0.0
                push_in_transaction = driver._push_in_transaction

                def push_then_expire(*args, **kwargs):
                    nonlocal now
                    receipt = push_in_transaction(*args, **kwargs)
                    now = 5.0
                    return receipt

                with monkeypatch.context() as patch:
                    patch.setattr(SystemClock, "monotonic", lambda self: now)
                    patch.setattr(driver, "_push_in_transaction", push_then_expire)
                    with pytest.raises(KeyUnavailable, match="persistence budget expired"):
                        driver.push("objects", InboxItem("failed", "user", {"content": "object payload"}))
                assert [row.item.input_id for row in store.peek_inbox("objects")] == ["input"]
                with pytest.raises(ContentRuntimeUnavailable, match="root create or push"), driver.store.atomic():
                    pytest.fail("generic mutation entered PostgreSQL object store")
                with pytest.raises(ContentRuntimeUnavailable, match="worker execution"), driver._open_drive_store():
                    pytest.fail("worker opened a PostgreSQL object store")
                assert set(
                    store._rows(
                        "SELECT ciphertext IS NULL, octet_length(object_locator) FROM sk_content WHERE session_id='objects'"
                    )
                ) == {(True, 33)}
                assert store._one("SELECT key_id FROM sk_session_key WHERE session_id='objects'")[0].startswith("s1-exact-key-")

    with (
        closing(LocalKeyProvider(master_key, state_path=key_path)) as provider,
        PostgresStore.tenant_standalone(primary[TENANT_A], tenant_id=TENANT_A) as store,
    ):
        store.configure_input_admission(allow_unsigned=True)
        with pytest.raises(ContentUnavailable, match="requires its external content binding"):
            bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher())
        objects = _PostgresObjectContent(journal, S3CiphertextStore(client, bucket="fixture", journal=journal), provider_instance)
        with monkeypatch.context() as patch:
            patch.setattr(client, "endpoint_url", "http://other-fixture.invalid")
            with pytest.raises(ContentUnavailable, match="binding does not match"):
                bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        with closing(SessionDriver(store=store)) as driver:
            created = store.read("objects", limit=1).records[0].record
            workspace = ContentRef.from_dict(created.payload["workspace"])
            persisted = store.peek_inbox("objects")[0].item
            ref = ContentRef.from_dict(persisted.payload["content"])
            assert driver.read_record_content("objects", 1, workspace, slot=SESSION_WORKSPACE) == "/workspace"
            assert driver.read_input_content("objects", persisted.input_id, ref) == "object payload"
            with pytest.raises(ContentUnavailable, match="exact object locator"):
                _read_ciphertext(objects, TENANT_A, "objects", ref, bytes(12), bytes(16), None)
