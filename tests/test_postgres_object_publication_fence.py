from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import replace
from functools import partial
from threading import Event
from uuid import uuid4

import pytest
from session.test_s3_reservation_adapter import _Client
from test_s1_postgres_rls import TENANT_A, TENANT_B, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent import Agent, RunConfig
from vv_agent.kernel._content import ContentRef, ContentUnavailable, KeyUnavailable
from vv_agent.kernel._content_runtime import ContentRuntimeUnavailable, _provision_postgres_object_content, bind_content_runtime
from vv_agent.kernel._content_sql import _read_ciphertext
from vv_agent.kernel._external_content import _PostgresObjectContent
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.kernel._store import Conflict
from vv_agent.memory.token_utils import count_tokens
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.content_runtime import SESSION_WORKSPACE
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.fork import fork_thread
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem
from vv_agent.session.result import project_result
from vv_agent.session.s3_content import S3CiphertextStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock
from vv_agent.types import LLMResponse


@pytest.fixture
def object_publication(rls_database):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = rls_database
    _provision(database)
    cluster_dsn = _dsn(database, database.cluster.admin_role, database.cluster.admin_password)
    owner_dsn = _dsn(database, database.migration_role, database.migration_password)
    worker_dsn = _dsn(database, database.worker_role, database.worker_password)
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
                owner.install_worker_rls(database.worker_role, database.worker_password)
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
                yield owner, primary, journals, worker_dsn
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(journal_database)))
            for role in roles:
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


_TEST_CONTENT_DIGEST = "1" * 64


def test_postgres_publication_checks_tenant_binding_and_outer_transaction(object_publication, monkeypatch):
    from psycopg import IsolationLevel

    owner, primary, journals, _ = object_publication
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
    owner, primary, journals, _ = object_publication
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


class _SdkObjectClient(_Client):
    def __init__(self, journal: PostgresObjectJournal, session_id: str, *, check_key_intent: bool = False):
        super().__init__(self._lookup)
        self.journal = journal
        self.session_id = session_id
        self.check_key_intent = check_key_intent

    def _lookup(self, allocation_id: bytes):
        row = self.journal._scope.get(allocation_id) or self.journal.get_anonymous(allocation_id)
        if row is not None and row.state == "bound":
            assert row.owner is not None
            assert row.owner.tenant_id == self.journal.tenant_id and row.owner.session_id == self.session_id
            if self.check_key_intent:
                key = self.journal.get_key_intent(row.owner.provider_instance_id, row.owner.key_id)
                assert key is not None and key.state == "allocating"
            self.owner = row.owner
        return row


def test_postgres_sdk_root_object_create_push_and_cold_read(object_publication, tmp_path, monkeypatch):
    owner, primary, journals, _ = object_publication
    journal = journals[TENANT_A]

    key_path = tmp_path / "keys.sqlite"
    master_key = b"x" * 32
    client = _SdkObjectClient(journal, "objects", check_key_intent=True)

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
                with pytest.raises(ContentRuntimeUnavailable, match="fenced root publication transaction"), driver.store.atomic():
                    pytest.fail("generic mutation entered PostgreSQL object store")
                with pytest.raises(ContentRuntimeUnavailable, match="requires open_claim_store"):
                    driver.start("objects", Agent("test", "Answer the user.", model="m"), RunConfig(), None)
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


@pytest.mark.parametrize("mode", ["start-input", "queued-input", "resume-receipt", "contended-publication"])
def test_postgres_sdk_worker_publishes_root_turn(object_publication, tmp_path, monkeypatch, mode):
    owner, primary, journals, worker_dsn = object_publication
    journal = journals[TENANT_A]
    session_id, input_id = "worker-root", "first-input"
    prompt, answer = "read the stored worker input", "published worker result"
    key_path = tmp_path / "keys.sqlite"
    master_key = b"x" * 32
    client = _SdkObjectClient(journal, session_id)
    open_tenant = partial(PostgresStore.tenant_standalone, primary[TENANT_A], tenant_id=TENANT_A)
    open_claim_store = partial(PostgresStore.worker_standalone, worker_dsn)
    claims = []
    renewals = []
    drive_pids = []
    model_calls = []
    contended = Event()
    claim_runnable, renew = PostgresStore.claim_runnable, PostgresStore.renew

    def record_claim(store, **kwargs):
        claimed = claim_runnable(store, **kwargs)
        for work in claimed:
            if work.lease is not None:
                assert store.runtime_mode == "worker" and not store.claims_all_tenants
                claims.append((store.connection.info.backend_pid, work.lease))
        return claimed

    def record_renewal(store, lease, **kwargs):
        poll = renew(store, lease, **kwargs)
        assert store.credential_tenant_id == TENANT_A
        renewals.append((store.connection.info.backend_pid, poll.lease))
        return poll

    monkeypatch.setattr(PostgresStore, "claim_runnable", record_claim)
    monkeypatch.setattr(PostgresStore, "renew", record_renewal)

    def model_response(request):
        assert any(message.role == "user" and message.content == prompt for message in request.messages)
        model_calls.append(request)
        return LLMResponse(answer)

    agent = Agent("test", "Answer the user.", model="m")
    # Initialize the scripted model's tokenizer before timing the PostgreSQL worker.
    count_tokens(prompt, "m")
    config = RunConfig(model_provider=ScriptedModelProvider.from_callback("scripted", "m", model_response))
    with closing(LocalKeyProvider(master_key, state_path=key_path)) as provider, open_tenant() as store:
        provider_instance = provider._provision_exact_key_lifecycle()
        objects = _PostgresObjectContent(journal, S3CiphertextStore(client, bucket="fixture", journal=journal), provider_instance)
        _provision_postgres_object_content(owner, key_provider=provider, objects=objects)
        store.configure_input_admission(allow_unsigned=True)
        bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        with closing(SessionDriver(store=store, open_claim_store=open_claim_store)) as driver:
            if mode == "contended-publication":
                defer = PostgresStore.defer_drive

                def observe_deferral(store, lease, *, retry_after_ms):
                    result = defer(store, lease, retry_after_ms=retry_after_ms)
                    if lease.session_id == session_id:
                        contended.set()
                    return result

                monkeypatch.setattr(PostgresStore, "defer_drive", observe_deferral)

            def runtime_factory(drive_store, agent, config, task, session_id):
                assert drive_store.credential_tenant_id == TENANT_A
                drive_pids.append(drive_store.connection.info.backend_pid)
                runtime = driver.runtime(agent, config, task, session_id=session_id, store=drive_store)
                if mode == "resume-receipt":

                    def stop_after_receipt(point, record):
                        if point == "after_commit" and record is not None and record.kind == "op_completed":
                            raise RuntimeError("stop after committed model receipt")

                    runtime.hook = stop_after_receipt
                return runtime

            driver.runtime_factory = runtime_factory
            driver.create(session_id, str(tmp_path))
            if mode in {"queued-input", "contended-publication"}:
                driver.push(session_id, InboxItem(input_id, "user", {"content": prompt}))
                if mode == "contended-publication":
                    with (
                        open_tenant() as contender,
                        contender._object_publication_atomic(
                            journal, content_digest=objects._binding_digest(), check_deadline=lambda: None
                        ),
                    ):
                        handle = driver.start(session_id, agent, config, None)
                        assert contended.wait(10)
                        assert model_calls == []
                else:
                    handle = driver.start(session_id, agent, config, None)
            else:
                handle = driver.start(session_id, agent, config, prompt, input_id=input_id)
            try:
                if mode == "resume-receipt":
                    with pytest.raises(RuntimeError, match="stop after committed model receipt"):
                        handle.result(timeout=20)
                    assert store.read_state(session_id)[0].active_turn_id == handle.run_id
                else:
                    assert handle.result(timeout=20).final_output == answer
            except TimeoutError as exc:
                raise exc from driver._worker.last_error
            run_id = handle.run_id

        root_pid = store.connection.info.backend_pid
        claim_pids = {pid for pid, _ in claims}
        assert all((claim.tenant_id, claim.session_id) == (TENANT_A, session_id) for _, claim in claims)
        if mode == "contended-publication":
            assert len({claim.epoch for _, claim in claims}) >= 2
        elif mode != "resume-receipt":
            assert len(claims) == 1
        assert drive_pids and claim_pids
        assert root_pid not in set(drive_pids) | claim_pids
        assert claim_pids.isdisjoint(drive_pids)
        assert renewals
        claimed = {(lease.owner, lease.epoch, lease.session_id, lease.tenant_id) for _, lease in claims}
        for heartbeat_pid, renewed in renewals:
            assert heartbeat_pid not in {root_pid} | claim_pids | set(drive_pids)
            assert (renewed.owner, renewed.epoch, renewed.session_id, renewed.tenant_id) in claimed
        assert store.peek_inbox(session_id) == ()
        assert store._one("SELECT lease_owner,lease_until_ms FROM sk_session WHERE session_id=%s", (session_id,)) == (None, None)
        assert store._one(
            "SELECT count(*) FROM sk_content WHERE session_id=%s AND (ciphertext IS NOT NULL OR object_locator IS NULL)",
            (session_id,),
        ) == (0,)

    with closing(LocalKeyProvider(master_key, state_path=key_path)) as provider, open_tenant() as cold_store:
        cold_store.configure_input_admission(allow_unsigned=True)
        bind_content_runtime(cold_store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        with closing(SessionDriver(store=cold_store, open_claim_store=open_claim_store)) as cold:
            if mode == "resume-receipt":
                resumed = cold.start(session_id, agent, config, None)
                assert resumed.run_id == run_id
                assert resumed.result(timeout=20).final_output == answer
            assert len(model_calls) == 1
            body, consumed_seq = cold_store._one(
                "SELECT body,consumed_seq FROM sk_inbox WHERE session_id=%s AND input_id=%s", (session_id, input_id)
            )
            item = InboxItem.parse(body)
            assert consumed_seq is not None
            assert cold.read_input_content(session_id, input_id, ContentRef.from_dict(item.payload["content"])) == prompt
            assert project_result(cold.store, session_id, run_id).final_output == answer
