"""A/B method matrix for PostgreSQL tenant RLS in trusted storage S1."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from queue import Queue
from typing import Any

import pytest
from session.helpers import item, record
from support.s1_fixture_values import (
    append_fixture_records,
    create_fixture_session,
    push_fixture_input,
    wrap_fixture_store,
)
from test_s1_postgres_rls import TENANT_A, TENANT_B, RLSDatabase, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel import _postgres
from vv_agent.kernel._content import ContentLabels, ContentUnavailable, Principal
from vv_agent.kernel._content_store import ContentState
from vv_agent.kernel._postgres import SESSION_TABLES
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import Conflict, Lease
from vv_agent.session.content_store import ContentExportRef, LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.export import export_session
from vv_agent.session.postgres import PostgresContentStore, PostgresStore

SHARED = "matrix-shared"
B_ONLY = "matrix-b-only"
FORGET = "matrix-forget"
PURGE = "matrix-purge"
CONTENT_DELETE = "matrix-content-delete"
CHECKPOINT = "matrix-checkpoint"
SEEDED_SESSIONS = (SHARED, FORGET, PURGE, CONTENT_DELETE, CHECKPOINT)

STORE_METHODS = (
    "read",
    "read_state",
    "read_input",
    "peek_inbox",
    "control_records",
    "projection_context",
    "lease_status",
    "next_drive_delay_ms",
    "is_runnable",
    "rebuild_schedule",
    "verify_ledger",
    "list_sessions",
    "list_runnable",
    "acquire",
    "release",
)
TRANSACTION_METHODS = ("create", "push", "append")
CONTENT_METHODS = (
    "put",
    "create_session_key",
    "put_session_key",
    "get",
    "assert_link",
    "link_input",
    "link_record",
    "read_links",
    "export_content_lines",
    "collect_garbage",
    "forget_session",
    "set_object_locator_cleaner",
)
EXPORT_DELETE_METHODS = (
    "export_session",
    "forget_session",
    "read_skeleton",
    "verify_tombstones",
    "purge_session",
)


@pytest.fixture
def key_provider(tmp_path: Path) -> Iterator[LocalKeyProvider]:
    provider = LocalKeyProvider(b"m" * 32, state_path=tmp_path / "matrix-keys.sqlite3")
    try:
        yield provider
    finally:
        provider.close()


@contextmanager
def _open_tenant(database: RLSDatabase, tenant_id: str) -> Iterator[PostgresStore]:
    if tenant_id == TENANT_A:
        role, password = database.tenant_a_role, database.tenant_a_password
    else:
        role, password = database.tenant_b_role, database.tenant_b_password
    with PostgresStore.tenant_standalone(_dsn(database, role, password), tenant_id=tenant_id) as store:
        store.configure_input_admission(allow_unsigned=True)
        yield store


def _provision_with_settings_lock(database: RLSDatabase) -> None:
    """Install the session-owned read/lock view on top of the shared RLS fixture."""

    _provision(database)
    with PostgresStore.schema_admin_standalone(_dsn(database, database.migration_role, database.migration_password)) as store:
        store.install_schema()
        store.install_tenant_rls(TENANT_A, database.tenant_a_role, database.tenant_a_password)
        store.install_tenant_rls(TENANT_B, database.tenant_b_role, database.tenant_b_password)
        store.install_worker_rls(database.worker_role, database.worker_password)
        store.install_shared_gateway_rls(database.gateway_role, database.gateway_password)


def _seed_tenant(store: PostgresStore, provider: LocalKeyProvider, input_id: str) -> dict[str, Any]:
    wrapped = wrap_fixture_store(store, provider)
    for session_id in SEEDED_SESSIONS:
        create_fixture_session(wrapped, session_id, consumers=())

    push_fixture_input(wrapped, SHARED, item(input_id))
    append_fixture_records(
        wrapped,
        SHARED,
        [record("turn_started", sid=SHARED, tid=f"turn-{input_id}", oid=f"op-{input_id}")],
        commit_id=f"matrix-{input_id}",
    )

    refs: dict[str, Any] = {}
    with store.atomic() as tx:
        content = PostgresContentStore(tx, provider)
        refs[SHARED] = content.put(
            store.tenant_id,
            SHARED,
            _shared_content(store.tenant_id),
            ContentLabels(),
            state=ContentState.LIVE,
        )
        refs[CONTENT_DELETE] = content.put(
            store.tenant_id,
            CONTENT_DELETE,
            f"delete:{store.tenant_id}".encode(),
            ContentLabels(),
            state=ContentState.LIVE,
        )

    with store.atomic():
        row = store._one(
            "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
            (store.tenant_id, CHECKPOINT),
        )
        store._rows(
            "INSERT INTO sk_checkpoint(tenant_id,session_id,seq,chain_digest,key_id,signature,signed_ms) "
            "VALUES (%s,%s,1,%s,'matrix',%s,0)",
            (store.tenant_id, CHECKPOINT, row[0], b"\x00"),
        )
    return refs


def _shared_content(tenant_id: str) -> bytes:
    return json.dumps({"tenant_id": tenant_id}, sort_keys=True, separators=(",", ":")).encode()


def _seed_pair(database: RLSDatabase, provider: LocalKeyProvider) -> tuple[dict[str, Any], dict[str, Any]]:
    with ExitStack() as stack:
        tenant_a = stack.enter_context(_open_tenant(database, TENANT_A))
        tenant_b = stack.enter_context(_open_tenant(database, TENANT_B))
        refs_a = _seed_tenant(tenant_a, provider, "a-input")
        refs_b = _seed_tenant(tenant_b, provider, "b-input")
        create_fixture_session(wrap_fixture_store(tenant_b, provider), B_ONLY, consumers=())
    return refs_a, refs_b


def _actor(
    store: PostgresStore,
    key_provider: LocalKeyProvider,
    session_id: str,
    subject: str,
) -> Principal:
    from vv_agent.session.content_runtime import SessionContentWriter

    with store.atomic() as tx:
        writer = SessionContentWriter.bind(
            tx,
            tenant_id=store.tenant_id,
            session_id=session_id,
            key_provider=key_provider,
        )
        principal, _attributes = writer.service_principal(subject)
    return Principal.from_dict(principal)


def _set_context(connection: Any, tenant_id: str) -> None:
    connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (tenant_id,))


def _tenant_table_counts(connection: Any, tenant_id: str) -> dict[str, int]:
    from psycopg import sql

    with connection.transaction():
        _set_context(connection, tenant_id)
        return {
            table: connection.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE tenant_id=%s").format(sql.Identifier(table)),
                (tenant_id,),
            ).fetchone()[0]
            for table in SESSION_TABLES
        }


def _wait_for_blocked_settings_lock(database: RLSDatabase, role: str) -> None:
    import psycopg

    deadline = time.monotonic() + 10
    with psycopg.connect(_dsn(database, database.migration_role, database.migration_password), autocommit=True) as observer:
        while time.monotonic() < deadline:
            blocked_row = observer.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_locks l JOIN pg_stat_activity a USING(pid) "
                "WHERE a.datname=current_database() AND a.usename=%s AND NOT l.granted)",
                (role,),
            ).fetchone()
            assert blocked_row is not None
            blocked = blocked_row[0]
            if blocked:
                return
            time.sleep(0.05)
    raise AssertionError(f"{role} did not wait on the sk_settings row lock")


def test_matrix_api_inventory_is_explicit() -> None:
    for method in STORE_METHODS:
        assert callable(getattr(PostgresStore, method))
    for method in TRANSACTION_METHODS:
        assert callable(getattr(SQLSessionTx, method))
    for method in CONTENT_METHODS:
        assert callable(getattr(PostgresContentStore, method))
    for method in EXPORT_DELETE_METHODS:
        target = export_session if method == "export_session" else getattr(SessionDeletionStore, method, None)
        assert callable(target)


@pytest.mark.postgres
def test_runtime_ceiling_lock_does_not_expand_settings_write_privilege(rls_database) -> None:
    import psycopg

    _provision_with_settings_lock(rls_database)
    with psycopg.connect(_dsn(rls_database, rls_database.migration_role, rls_database.migration_password)) as owner:
        assert _postgres._schema_fingerprint(owner) == _postgres.SCHEMA_FINGERPRINT
        assert _postgres._rls_state_is_current(owner)
        assert owner.execute(
            "SELECT c.relkind,c.relowner=%s::regrole FROM pg_class c WHERE c.oid='vv_settings_ceiling'::regclass",
            (rls_database.migration_role,),
        ).fetchone() == ("v", True)

    with _open_tenant(rls_database, TENANT_A) as store:
        connection = store.connection
        with connection.transaction():
            privileges = connection.execute(
                "SELECT has_table_privilege(current_user,'sk_settings','UPDATE'),"
                "has_table_privilege(current_user,'vv_settings_ceiling','UPDATE'),"
                "has_column_privilege(current_user,'vv_settings_ceiling','lock_token','UPDATE'),"
                "has_table_privilege(current_user,'vv_settings_ceiling','SELECT')"
            ).fetchone()
        assert privileges == (False, False, True, True)

        for clause in ("FOR SHARE", "FOR NO KEY UPDATE"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege) as raised, connection.transaction():
                connection.execute(f"SELECT storage_major,write_level FROM sk_settings WHERE singleton=1 {clause}")
            assert raised.value.sqlstate == "42501"

        with connection.transaction():
            assert store._write_ceiling() == 1
        for statement in (
            "UPDATE sk_settings SET write_level=2 WHERE singleton=1",
            "UPDATE vv_settings_ceiling SET write_level=2 WHERE singleton=1",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), connection.transaction():
                connection.execute(statement)
        with pytest.raises(psycopg.errors.DatabaseError, match="cannot update column"), connection.transaction():
            connection.execute("UPDATE vv_settings_ceiling SET lock_token=1")
        with connection.transaction():
            assert connection.execute("SELECT write_level FROM sk_settings").fetchone() == (1,)


@pytest.mark.postgres
def test_operator_raise_waits_for_in_flight_runtime_ceiling_lock(rls_database) -> None:
    import psycopg

    _provision_with_settings_lock(rls_database)
    writer_ready = threading.Event()
    writer_release = threading.Event()
    writer_done = threading.Event()
    update_done = threading.Event()
    failures: Queue[BaseException] = Queue()
    observed: Queue[int] = Queue()

    def writer() -> None:
        try:
            with (
                PostgresStore.tenant_standalone(
                    _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password), tenant_id=TENANT_A
                ) as store,
                store.connection.transaction(),
            ):
                observed.put(store._write_ceiling())
                writer_ready.set()
                if not writer_release.wait(10):
                    raise TimeoutError("writer lock was not released")
        except BaseException as exc:
            failures.put(exc)
            writer_ready.set()
        finally:
            writer_done.set()

    def operator_update() -> None:
        try:
            with psycopg.connect(
                _dsn(rls_database, rls_database.migration_role, rls_database.migration_password), autocommit=True
            ) as connection:
                connection.execute("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        except BaseException as exc:
            failures.put(exc)
        finally:
            update_done.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    update_thread = threading.Thread(target=operator_update, daemon=True)
    try:
        writer_thread.start()
        assert writer_ready.wait(10)
        if not failures.empty():
            raise failures.get()
        update_thread.start()
        _wait_for_blocked_settings_lock(rls_database, rls_database.migration_role)
        assert not update_done.is_set()
    finally:
        writer_release.set()
        writer_thread.join(10)
        update_thread.join(10)

    assert not writer_thread.is_alive() and not update_thread.is_alive()
    if not failures.empty():
        raise failures.get()
    assert writer_done.is_set() and update_done.is_set()
    assert observed.get_nowait() == 1
    with psycopg.connect(
        _dsn(rls_database, rls_database.migration_role, rls_database.migration_password), autocommit=True
    ) as connection:
        assert connection.execute("SELECT write_level FROM sk_settings").fetchone() == (2,)


@pytest.mark.postgres
def test_runtime_ceiling_waits_for_in_flight_operator_raise(rls_database) -> None:
    import psycopg

    _provision_with_settings_lock(rls_database)
    update_ready = threading.Event()
    update_commit = threading.Event()
    update_done = threading.Event()
    writer_done = threading.Event()
    failures: Queue[BaseException] = Queue()
    observed: Queue[int] = Queue()

    def operator_update() -> None:
        connection = None
        try:
            connection = psycopg.connect(_dsn(rls_database, rls_database.migration_role, rls_database.migration_password))
            connection.execute("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
            update_ready.set()
            if not update_commit.wait(10):
                raise TimeoutError("operator update was not released")
            connection.commit()
        except BaseException as exc:
            failures.put(exc)
            update_ready.set()
        finally:
            if connection is not None:
                connection.close()
            update_done.set()

    def writer() -> None:
        try:
            with (
                PostgresStore.tenant_standalone(
                    _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password), tenant_id=TENANT_A
                ) as store,
                store.connection.transaction(),
            ):
                observed.put(store._write_ceiling())
        except BaseException as exc:
            failures.put(exc)
        finally:
            writer_done.set()

    update_thread = threading.Thread(target=operator_update, daemon=True)
    writer_thread = threading.Thread(target=writer, daemon=True)
    try:
        update_thread.start()
        assert update_ready.wait(10)
        if not failures.empty():
            raise failures.get()
        writer_thread.start()
        _wait_for_blocked_settings_lock(rls_database, rls_database.tenant_a_role)
        assert not writer_done.is_set()
    finally:
        update_commit.set()
        update_thread.join(10)
        writer_thread.join(10)

    assert not update_thread.is_alive() and not writer_thread.is_alive()
    if not failures.empty():
        raise failures.get()
    assert update_done.is_set() and writer_done.is_set()
    assert observed.get_nowait() == 2


@pytest.mark.postgres
def test_tenant_store_content_and_export_method_matrix(rls_database, key_provider: LocalKeyProvider) -> None:
    _provision_with_settings_lock(rls_database)
    refs_a, refs_b = _seed_pair(rls_database, key_provider)

    with ExitStack() as stack:
        tenant_a = stack.enter_context(_open_tenant(rls_database, TENANT_A))
        tenant_b = stack.enter_context(_open_tenant(rls_database, TENANT_B))
        content_a = PostgresContentStore(tenant_a, key_provider)
        content_b = PostgresContentStore(tenant_b, key_provider)

        sessions_a = tenant_a.list_sessions()
        sessions_b = tenant_b.list_sessions()
        assert sessions_a and set(sessions_a) <= set(SEEDED_SESSIONS) | {"a-owned"}
        assert sessions_b and set(sessions_b) <= set(SEEDED_SESSIONS) | {B_ONLY, "b-owned"}
        assert B_ONLY not in sessions_a

        page_a = tenant_a.read(SHARED)
        page_b = tenant_b.read(SHARED)
        assert page_a.head_seq == page_b.head_seq == 2
        assert page_a.records[0].record.encode() != page_b.records[0].record.encode()
        assert tenant_a.read_state(SHARED)[1] == page_a.records
        assert tenant_a.read_input(SHARED, "a-input") is not None
        assert tenant_a.read_input(SHARED, "b-input") is None
        assert [stored.item.input_id for stored in tenant_a.peek_inbox(SHARED)] == ["a-input"]
        assert tenant_a.lease_status(SHARED)[0] == page_a.head_seq
        assert tenant_a.verify_ledger(SHARED)[0] == page_a.head_seq

        with pytest.raises(Conflict):
            tenant_a.read(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.read_state(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.read_input(B_ONLY, "b-input")
        with pytest.raises(Conflict):
            tenant_a.peek_inbox(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.control_records(B_ONLY, through_seq=1)
        with pytest.raises(Conflict):
            tenant_a.projection_context(B_ONLY, ())
        with pytest.raises(Conflict):
            tenant_a.lease_status(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.next_drive_delay_ms(B_ONLY)
        assert not tenant_a.is_runnable(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.rebuild_schedule(B_ONLY)
        with pytest.raises(Conflict):
            tenant_a.verify_ledger(B_ONLY)
        with pytest.raises(ValueError, match="another tenant"):
            tenant_a.for_tenant(TENANT_B)

        lease = tenant_a.acquire(SHARED, owner="matrix-a", ttl_ms=60_000)
        assert lease is not None
        assert tenant_a.release(lease)
        with pytest.raises(Conflict):
            tenant_a.acquire(B_ONLY, owner="matrix-a", ttl_ms=60_000)
        runnable_a = tenant_a.list_runnable(limit=100)
        assert runnable_a and {work.tenant_id for work in runnable_a} == {TENANT_A}
        assert B_ONLY not in {work.session_id for work in runnable_a}

        b_head_before = tenant_b.lease_status(SHARED)[0]
        with pytest.raises(Conflict), tenant_a.atomic() as tx:
            tx.append(
                B_ONLY,
                lease=Lease(B_ONLY, "forged-a", 0, lease.expires_at_ms, TENANT_A),
                expected_seq=0,
                commit_id="matrix-forged-b",
                records=(record("turn_started", sid=B_ONLY, tid="forged-b", oid="forged-b"),),
            )
        assert tenant_b.lease_status(SHARED)[0] == b_head_before
        assert tenant_b.read(B_ONLY).head_seq == 1

        assert content_a.get(TENANT_A, SHARED, refs_a[SHARED]) == _shared_content(TENANT_A)
        assert content_b.get(TENANT_B, SHARED, refs_b[SHARED]) == _shared_content(TENANT_B)
        with tenant_a.atomic():
            assert content_a.create_session_key(TENANT_A, SHARED).tenant_id == TENANT_A
            content_a.put_session_key(content_a._active_session_key(TENANT_A, SHARED))
        with pytest.raises(ContentUnavailable), tenant_a.atomic():
            content_a.create_session_key(TENANT_B, SHARED)
        with pytest.raises(ContentUnavailable), tenant_a.atomic():
            content_a.put_session_key(content_b._active_session_key(TENANT_B, SHARED))
        with tenant_a.atomic():
            content_a.link_input(TENANT_A, SHARED, "a-input", refs_a[SHARED], "matrix.content")
            content_a.link_record(TENANT_A, SHARED, 1, refs_a[SHARED], "matrix.record")
        with pytest.raises(ContentUnavailable), tenant_a.atomic():
            content_a.link_record(TENANT_B, SHARED, 1, refs_b[SHARED], "matrix.record")
        assert (
            content_a.assert_link(
                TENANT_A,
                SHARED,
                refs_a[SHARED],
                "matrix.content",
                input_id="a-input",
            )
            == refs_a[SHARED]
        )
        with pytest.raises(ContentUnavailable):
            content_a.assert_link(
                TENANT_B,
                SHARED,
                refs_b[SHARED],
                "matrix.content",
                input_id="b-input",
            )
        assert {link.slot for link in content_a.read_links(TENANT_A, SHARED, input_id="a-input")} >= {"matrix.content"}
        content_a.set_object_locator_cleaner(lambda *_args: None)
        with pytest.raises(ContentUnavailable):
            content_a.get(TENANT_B, SHARED, refs_b[SHARED])
        with pytest.raises(ContentUnavailable):
            content_a.read_links(TENANT_B, SHARED, record_seq=1)
        with pytest.raises(ContentUnavailable), tenant_a.atomic():
            content_a.export_content_lines(
                TENANT_B,
                SHARED,
                (ContentExportRef(None, "b-input", "matrix.content", refs_b[SHARED]),),
            )
        with tenant_a.atomic():
            assert content_a.export_content_lines(
                TENANT_A,
                SHARED,
                (ContentExportRef(None, "a-input", "matrix.content", refs_a[SHARED]),),
            )
        with pytest.raises(ContentUnavailable), tenant_a.atomic():
            content_a.put(TENANT_B, SHARED, b"forged", ContentLabels())
        with tenant_a.atomic():
            content_a.forget_session(TENANT_B, SHARED)
            with pytest.raises(ContentUnavailable):
                content_a.collect_garbage(TENANT_B, SHARED)
        assert content_b.get(TENANT_B, SHARED, refs_b[SHARED]) == _shared_content(TENANT_B)

        export_a = export_session(tenant_a, SHARED)
        export_b = export_session(tenant_b, SHARED)
        assert export_a.header.tenant_id == TENANT_A
        assert export_b.header.tenant_id == TENANT_B
        assert export_a.lines != export_b.lines
        with pytest.raises(Conflict):
            export_session(tenant_a, B_ONLY)

        with tenant_a.connection.transaction():
            _set_context(tenant_a.connection, TENANT_B)
            assert tenant_a.list_sessions() == ()
            with pytest.raises(Conflict):
                tenant_a.read(SHARED)
            with pytest.raises(ContentUnavailable):
                content_a.get(TENANT_A, SHARED, refs_a[SHARED])
            _set_context(tenant_a.connection, TENANT_A)
            assert SHARED in tenant_a.list_sessions()
            assert content_a.get(TENANT_A, SHARED, refs_a[SHARED]) == _shared_content(TENANT_A)


@pytest.mark.postgres
def test_forget_and_purge_a_to_b_matrix(rls_database, key_provider: LocalKeyProvider) -> None:
    _provision_with_settings_lock(rls_database)
    _seed_pair(rls_database, key_provider)

    with ExitStack() as stack:
        tenant_a = stack.enter_context(_open_tenant(rls_database, TENANT_A))
        tenant_b = stack.enter_context(_open_tenant(rls_database, TENANT_B))
        content_a = PostgresContentStore(tenant_a, key_provider)
        deletion = SessionDeletionStore(tenant_a, content_a)
        actor_a = _actor(tenant_a, key_provider, FORGET, "service/retention-a")
        actor_b = _actor(tenant_b, key_provider, FORGET, "service/retention-b")

        with pytest.raises(Conflict):
            deletion.forget_session(TENANT_A, B_ONLY, actor=actor_a)
        with pytest.raises(ValueError, match="another tenant"):
            deletion.forget_session(TENANT_B, B_ONLY, actor=actor_b)
        with pytest.raises(PermissionError, match="actor belongs to another tenant"):
            deletion.forget_session(TENANT_A, FORGET, actor=actor_b)
        assert tenant_b.read(FORGET).head_seq == 1

        forgotten = deletion.forget_session(TENANT_A, FORGET, actor=actor_a)
        assert forgotten.session_ids == (FORGET,)
        skeleton = deletion.read_skeleton(TENANT_A, FORGET)
        assert skeleton.tenant_id == TENANT_A and skeleton.tombstone is not None
        assert deletion.verify_tombstones(TENANT_A, FORGET)[0].scope_root_session_id == FORGET
        with pytest.raises(Conflict):
            deletion.read_skeleton(TENANT_A, B_ONLY)
        with pytest.raises(ValueError, match="another tenant"):
            deletion.read_skeleton(TENANT_B, B_ONLY)
        assert tenant_b.read(FORGET).head_seq == 1
        assert tenant_b.lease_status(FORGET)[0] == 1

        purged = deletion.purge_session(TENANT_A, PURGE, actor=actor_a)
        assert purged.session_ids == (PURGE,)
        assert deletion.verify_tombstones(TENANT_A, PURGE)[0].scope_root_session_id == PURGE
        with pytest.raises(Conflict):
            deletion.purge_session(TENANT_A, B_ONLY, actor=actor_a)
        with pytest.raises(ValueError, match="another tenant"):
            deletion.verify_tombstones(TENANT_B, B_ONLY)
        assert tenant_b.read(PURGE).head_seq == 1
        assert tenant_b.read(B_ONLY).head_seq == 1


@pytest.mark.postgres
def test_rls_is_authority_across_shared_connection_context_switches(
    rls_database,
    key_provider: LocalKeyProvider,
) -> None:
    import psycopg
    from psycopg import sql

    _provision_with_settings_lock(rls_database)
    refs_a, refs_b = _seed_pair(rls_database, key_provider)

    with ExitStack() as stack:
        tenant_a = stack.enter_context(_open_tenant(rls_database, TENANT_A))
        tenant_b = stack.enter_context(_open_tenant(rls_database, TENANT_B))
        content_a = PostgresContentStore(tenant_a, key_provider)
        connection = tenant_a.connection

        with connection.transaction():
            _set_context(connection, TENANT_B)
            for table in sorted(SESSION_TABLES):
                assert connection.execute(
                    sql.SQL("SELECT count(*) FROM {} WHERE tenant_id=%s").format(sql.Identifier(table)),
                    (TENANT_B,),
                ).fetchone() == (0,)
            assert (
                connection.execute(
                    "UPDATE sk_session SET phase='closed' WHERE tenant_id=%s AND session_id=%s",
                    (TENANT_B, SHARED),
                ).rowcount
                == 0
            )
            assert (
                connection.execute(
                    "DELETE FROM sk_content WHERE tenant_id=%s AND session_id=%s",
                    (TENANT_B, SHARED),
                ).rowcount
                == 0
            )

        with pytest.raises(psycopg.errors.InsufficientPrivilege), connection.transaction():
            _set_context(connection, TENANT_B)
            connection.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES (%s,'forged-b','S1','idle',1,'live',0)",
                (TENANT_B,),
            )

        with connection.transaction():
            _set_context(connection, TENANT_A)
            rows_a = connection.execute("SELECT tenant_id,session_id FROM sk_session ORDER BY session_id").fetchall()
            assert rows_a and {row[0] for row in rows_a} == {TENANT_A}
            _set_context(connection, TENANT_B)
            assert connection.execute("SELECT tenant_id,session_id FROM sk_session").fetchall() == []
            _set_context(connection, TENANT_A)
            assert (TENANT_A, SHARED) in connection.execute("SELECT tenant_id,session_id FROM sk_session").fetchall()

        assert tenant_b.read(SHARED).head_seq == 2
        assert tenant_b.lease_status(SHARED)[0] == 2
        assert content_a.get(TENANT_A, SHARED, refs_a[SHARED]) == _shared_content(TENANT_A)
        assert (
            tenant_b._one(
                "SELECT count(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
                (TENANT_B, SHARED),
            )[0]
            >= 1
        )
        assert refs_b[SHARED].content_id


FK_CASES = (
    (
        "sk_session_tenant_id_parent_session_id_fkey",
        "INSERT INTO sk_session(tenant_id,session_id,schema_version,parent_session_id,parent_operation_id,"
        "phase,level,deletion_state,created_ms) VALUES (%s,'fk-child','S1',%s,'op/fk','idle',1,'live',0)",
    ),
    (
        "sk_record_tenant_id_session_id_fkey",
        "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,schema_version,kind,body,digest,chain_digest,"
        "commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
        "VALUES (%s,%s,77,'fk-record','S1','turn_started',%s,%s,%s,'fk-commit',%s,0,0,0)",
    ),
    (
        "sk_inbox_tenant_id_session_id_fkey",
        "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms) "
        "VALUES (%s,%s,'fk-input',77,'user',%s,%s,0,0)",
    ),
    (
        "sk_inbox_tenant_id_session_id_consumed_seq_fkey",
        "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms,"
        "consumed_seq) VALUES (%s,%s,'fk-consumed',78,'control',%s,%s,0,0,999)",
    ),
    (
        "sk_consumer_tenant_id_session_id_fkey",
        "INSERT INTO sk_consumer(tenant_id,session_id,consumer) VALUES (%s,%s,'fk-consumer')",
    ),
    (
        "sk_commit_tenant_id_session_id_fkey",
        "INSERT INTO sk_commit(tenant_id,session_id,commit_id,body,digest,record_sequences,head_seq,created_ms) "
        "VALUES (%s,%s,'fk-commit',%s,%s,%s,0,0)",
    ),
    (
        "sk_checkpoint_tenant_id_session_id_seq_fkey",
        "INSERT INTO sk_checkpoint(tenant_id,session_id,seq,chain_digest,key_id,signature,signed_ms) "
        "VALUES (%s,%s,1,%s,'fk',%s,0)",
    ),
    (
        "sk_session_key_tenant_id_session_id_fkey",
        "INSERT INTO sk_session_key(tenant_id,session_id,key_id,provider_id,wrapped_key,state,created_ms) "
        "VALUES (%s,%s,'fk-key','fk',%s,'active',0)",
    ),
    (
        "sk_content_link_tenant_id_session_id_fkey",
        "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,slot) "
        "VALUES (%s,%s,'missing-content',1,'fk-session')",
    ),
    (
        "sk_content_link_tenant_id_session_id_content_id_fkey",
        "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,slot) "
        "VALUES (%s,%s,'missing-content',1,'fk-content')",
    ),
    (
        "sk_content_link_tenant_id_session_id_record_seq_fkey",
        "INSERT INTO sk_content_link(tenant_id,session_id,content_id,record_seq,slot) VALUES (%s,%s,%s,999,'fk-record')",
    ),
    (
        "sk_content_link_tenant_id_session_id_input_id_fkey",
        "INSERT INTO sk_content_link(tenant_id,session_id,content_id,input_id,slot) VALUES (%s,%s,%s,'missing-input','fk-input')",
    ),
)


@pytest.mark.postgres
def test_all_declared_foreign_keys_reject_cross_tenant_references(
    rls_database,
    key_provider: LocalKeyProvider,
) -> None:
    import psycopg

    _provision_with_settings_lock(rls_database)
    refs_a, _ = _seed_pair(rls_database, key_provider)
    content_id = refs_a[SHARED].content_id
    body = b"fk-body"
    digest = "0" * 64
    signature = b"\x00"

    with _open_tenant(rls_database, TENANT_A) as connection_store:
        connection = connection_store.connection
        with connection.transaction():
            _set_context(connection, TENANT_A)
            declared = {
                (table, constraint)
                for table, constraint in connection.execute(
                    "SELECT conrelid::regclass::text,conname FROM pg_constraint "
                    "WHERE contype='f' AND connamespace=current_schema()::regnamespace"
                ).fetchall()
            }
            assert declared == {(constraint.split("_tenant_id", 1)[0], constraint) for constraint, _ in FK_CASES}

        parameter_sets = {
            "sk_session_tenant_id_parent_session_id_fkey": (TENANT_A, B_ONLY),
            "sk_record_tenant_id_session_id_fkey": (TENANT_A, B_ONLY, body, digest, digest, digest),
            "sk_inbox_tenant_id_session_id_fkey": (TENANT_A, B_ONLY, body, digest),
            "sk_inbox_tenant_id_session_id_consumed_seq_fkey": (TENANT_A, SHARED, body, digest),
            "sk_consumer_tenant_id_session_id_fkey": (TENANT_A, B_ONLY),
            "sk_commit_tenant_id_session_id_fkey": (TENANT_A, B_ONLY, body, digest, b"[]"),
            "sk_checkpoint_tenant_id_session_id_seq_fkey": (TENANT_A, B_ONLY, digest, signature),
            "sk_session_key_tenant_id_session_id_fkey": (TENANT_A, B_ONLY, b"wrapped"),
            "sk_content_link_tenant_id_session_id_fkey": (TENANT_A, B_ONLY),
            "sk_content_link_tenant_id_session_id_content_id_fkey": (TENANT_A, SHARED),
            "sk_content_link_tenant_id_session_id_record_seq_fkey": (TENANT_A, SHARED, content_id),
            "sk_content_link_tenant_id_session_id_input_id_fkey": (TENANT_A, SHARED, content_id),
        }

        before = _tenant_table_counts(connection, TENANT_A)
        failures: dict[str, str] = {}
        for constraint, statement in FK_CASES:
            with pytest.raises(psycopg.errors.ForeignKeyViolation) as raised, connection.transaction():
                _set_context(connection, TENANT_A)
                connection.execute(statement, parameter_sets[constraint])
            failures[constraint] = raised.value.diag.constraint_name or ""

        content_link_constraints = {constraint for constraint, _ in FK_CASES if constraint.startswith("sk_content_link_")}
        assert failures["sk_content_link_tenant_id_session_id_fkey"] in content_link_constraints
        for constraint, actual in failures.items():
            if constraint.startswith("sk_content_link_") and constraint.endswith("_session_id_fkey"):
                continue
            assert actual == constraint

        with connection.transaction():
            _set_context(connection, TENANT_A)
            assert connection.execute(
                "SELECT count(*) FROM sk_content_link WHERE tenant_id=%s AND slot LIKE 'fk-%%'",
                (TENANT_A,),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT count(*) FROM sk_session WHERE tenant_id=%s AND session_id='fk-child'",
                (TENANT_A,),
            ).fetchone() == (0,)
        assert _tenant_table_counts(connection, TENANT_A) == before
