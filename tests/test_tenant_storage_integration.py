"""Tenant-bound facade integration tests for the S1 SQL stores.

SQLite coverage exercises application-level store scoping on real files. PostgreSQL
runs the same facade assertions against the base schema; credential-level RLS has
separate TCP/password installation and startup coverage in test_s1_postgres_rls.py.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier, Event, RLock
from time import monotonic
from uuid import uuid4

import pytest

from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._records import InboxItem, SessionSpec, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLStore
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._store import Conflict
from vv_agent.session.surfaces import SessionDriver

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
SHARED_SESSION = "shared-session"


class _Clock:
    def __init__(self) -> None:
        self._ms = 1_700_000_000_000
        self._lock = RLock()

    def now_ms(self) -> int:
        with self._lock:
            return self._ms

    def monotonic(self) -> float:
        return self.now_ms() / 1000

    def monotonic_ns(self) -> int:
        return self.now_ms() * 1_000_000

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._ms += int(seconds * 1000)


def _fold_factory() -> Fold:
    def unused(*args, **kwargs):
        raise AssertionError("tenant integration records do not request fold callbacks")

    return Fold(normalize_memory_settings=unused, count_messages=unused)


def _content_ref(label: str) -> dict:
    body = label.encode("utf-8")
    return {
        "content_id": f"content/test/{label}",
        "mac": sha256(body).hexdigest(),
        "size": len(body),
        "key_id": "key/test",
        "labels": {"source": [], "classification": "internal"},
    }


def _principal(tenant_id: str) -> dict:
    return {
        "tenant_id": tenant_id,
        "subject_id": f"subject-{tenant_id}",
        "subject_kind": "user",
        "attributes": _content_ref(f"principal-{tenant_id}"),
    }


def _session(
    tenant_id: str,
    session_id: str,
    *,
    parent_session_id: str | None = None,
) -> SessionSpec:
    return SessionSpec(
        session_id,
        _principal(tenant_id),
        _content_ref(f"workspace-{tenant_id}-{session_id}"),
        parent_session_id=parent_session_id,
        parent_operation_id="parent-operation" if parent_session_id is not None else None,
        attributes=_content_ref(f"attributes-{tenant_id}-{session_id}"),
    )


def _input(tenant_id: str, input_id: str) -> InboxItem:
    return InboxItem(
        input_id,
        "user",
        {"content": _content_ref(f"input-{tenant_id}-{input_id}")},
        actor=_principal(tenant_id),
    )


def _usage_record(session_id: str):
    return make_record(
        "usage_observed",
        session_id=session_id,
        payload={
            "meter_id": "shared-meter",
            "observation": 1,
            "mode": "cumulative",
            "usage": _content_ref("shared-usage"),
            "source": _content_ref("shared-source"),
        },
    )


def _create(store: SQLStore, tenant_id: str, session_id: str, *, consumers: tuple[str, ...] = ("host",)):
    with store.atomic() as tx:
        return tx.create(_session(tenant_id, session_id), consumers=consumers)


def test_raw_kernel_postgres_store_cannot_be_a_runtime_entrypoint():
    raw_store = object.__new__(PostgresStore)
    with pytest.raises(ValueError, match="raw kernel PostgreSQL stores"):
        SessionDriver(store=raw_store, tenant_id=TENANT_A)


@pytest.fixture(scope="session")
def tenant_postgres_admin_dsn() -> str:
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
def tenant_store(request, tmp_path) -> Iterator[SQLStore]:
    if request.param == "sqlite":
        with SQLiteStore.standalone(
            str(tmp_path / "tenant.sqlite"),
            fold_factory=_fold_factory,
            clock=_Clock(),
        ) as store:
            store.install_schema()
            yield store
        return

    dsn = request.getfixturevalue("tenant_postgres_admin_dsn")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = f"vvsk_tenant_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            with PostgresStore.standalone(make_conninfo(dsn, dbname=name), fold_factory=_fold_factory) as store:
                store.install_schema()
                yield store
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


@pytest.mark.parametrize(
    "tenant_store",
    [
        pytest.param("sqlite", id="sqlite-file"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
    ],
    indirect=True,
)
def test_same_session_id_round_trip_is_tenant_local(tenant_store):
    tenant_a = tenant_store.for_tenant(TENANT_A)
    tenant_b = tenant_store.for_tenant(TENANT_B)

    created_a = _create(tenant_a, TENANT_A, SHARED_SESSION)
    created_b = _create(tenant_b, TENANT_B, SHARED_SESSION)
    assert created_a.session_id == created_b.session_id == SHARED_SESSION

    with tenant_a.atomic() as tx:
        pushed_a = tx.push(SHARED_SESSION, _input(TENANT_A, "shared-input"))
    with tenant_b.atomic() as tx:
        pushed_b = tx.push(SHARED_SESSION, _input(TENANT_B, "shared-input"))
    assert not pushed_a.replayed and not pushed_b.replayed

    page_a = tenant_a.read(SHARED_SESSION)
    page_b = tenant_b.read(SHARED_SESSION)
    assert page_a.head_seq == page_b.head_seq == 1
    assert page_a.records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == TENANT_A
    assert page_b.records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == TENANT_B
    assert tenant_a.peek_inbox(SHARED_SESSION)[0].item.actor["tenant_id"] == TENANT_A
    assert tenant_b.peek_inbox(SHARED_SESSION)[0].item.actor["tenant_id"] == TENANT_B

    lease_a = tenant_a.acquire(SHARED_SESSION, owner="worker", ttl_ms=10_000)
    lease_b = tenant_b.acquire(SHARED_SESSION, owner="worker", ttl_ms=10_000)
    assert lease_a is not None and lease_b is not None
    renewed_a = tenant_a.renew(lease_a, ttl_ms=20_000)
    renewed_b = tenant_b.renew(lease_b, ttl_ms=20_000)
    assert renewed_a.lease.epoch == renewed_b.lease.epoch == 1
    assert tenant_a.release(renewed_a.lease)
    assert tenant_b.release(renewed_b.lease)


@pytest.mark.parametrize(
    "tenant_store",
    [
        pytest.param("sqlite", id="sqlite-file"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
    ],
    indirect=True,
)
def test_tenant_queries_and_wrong_tenant_writes_cannot_touch_tenant_b(tenant_store):
    tenant_a = tenant_store.for_tenant(TENANT_A)
    tenant_b = tenant_store.for_tenant(TENANT_B)
    _create(tenant_a, TENANT_A, "a-owned")
    _create(tenant_b, TENANT_B, "b-owned")
    with tenant_a.atomic() as tx:
        tx.push("a-owned", _input(TENANT_A, "a-input"))
    with tenant_b.atomic() as tx:
        tx.push("b-owned", _input(TENANT_B, "existing-input"))

    assert tenant_a.list_sessions() == ("a-owned",)
    assert tenant_b.list_sessions() == ("b-owned",)
    assert "b-owned" not in {item.session_id for item in tenant_a.list_runnable()}
    assert "a-owned" not in {item.session_id for item in tenant_b.list_runnable()}

    with pytest.raises(Conflict):
        tenant_a.read("b-owned")
    with pytest.raises(Conflict):
        tenant_a.peek_inbox("b-owned", ready_only=False)
    with pytest.raises(Conflict):
        tenant_a.acquire("b-owned", owner="wrong-tenant-worker", ttl_ms=20_000)

    b_head = tenant_b.read("b-owned").head_seq
    b_inputs = tuple(item.item.input_id for item in tenant_b.peek_inbox("b-owned", ready_only=False))
    b_lease = tenant_b.acquire("b-owned", owner="b-worker", ttl_ms=20_000)
    assert b_lease is not None
    b_runnable = tenant_b.list_runnable()

    with pytest.raises(Conflict), tenant_a.atomic() as tx:
        tx.push("b-owned", _input(TENANT_B, "wrong-tenant-input"))
    with pytest.raises(Conflict), tenant_a.atomic() as tx:
        tx.append(
            "b-owned",
            lease=b_lease,
            expected_seq=b_head,
            commit_id="wrong-tenant-commit",
            records=(_usage_record("b-owned"),),
        )
    with pytest.raises(Conflict), tenant_a.atomic() as tx:
        tx.consumer_batch("b-owned", "host")
    with pytest.raises(Conflict):
        tenant_a.defer_drive(b_lease, retry_after_ms=60_000)
    with pytest.raises(Conflict):
        tenant_a.defer_idle_drive(b_lease, poll_ms=60_000)
    with pytest.raises(Conflict):
        tenant_a.defer_projection("b-owned", "host", retry_after_ms=60_000)

    assert tenant_b.read("b-owned").head_seq == b_head
    assert tuple(item.item.input_id for item in tenant_b.peek_inbox("b-owned", ready_only=False)) == b_inputs
    assert tenant_b.list_runnable() == b_runnable
    renewed_b = tenant_b.renew(b_lease, ttl_ms=20_000)

    with tenant_b.atomic() as tx:
        batch = tx.consumer_batch("b-owned", "host")
        assert batch is not None
        tx.ack(batch)
    assert tenant_b.release(renewed_b.lease)


@pytest.mark.parametrize(
    "tenant_store",
    [
        pytest.param("sqlite", id="sqlite-file"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
    ],
    indirect=True,
)
def test_identity_replay_is_tenant_local_across_input_record_consumer_and_commit(tenant_store):
    tenant_a = tenant_store.for_tenant(TENANT_A)
    tenant_b = tenant_store.for_tenant(TENANT_B)
    _create(tenant_a, TENANT_A, SHARED_SESSION)
    _create(tenant_b, TENANT_B, SHARED_SESSION)

    item_a, item_b = _input(TENANT_A, "same-input"), _input(TENANT_B, "same-input")
    with tenant_a.atomic() as tx:
        first_a = tx.push(SHARED_SESSION, item_a)
    with tenant_a.atomic() as tx:
        replay_a = tx.push(SHARED_SESSION, item_a)
    with tenant_b.atomic() as tx:
        first_b = tx.push(SHARED_SESSION, item_b)
    assert not first_a.replayed and not first_b.replayed
    assert replay_a.replayed
    assert replay_a.input_id == first_a.input_id
    assert replay_a.input_seq == first_a.input_seq
    assert replay_a.received_ms == first_a.received_ms

    lease_a = tenant_a.acquire(SHARED_SESSION, owner="worker-a", ttl_ms=20_000)
    lease_b = tenant_b.acquire(SHARED_SESSION, owner="worker-b", ttl_ms=20_000)
    assert lease_a is not None and lease_b is not None
    record_a, record_b = _usage_record(SHARED_SESSION), _usage_record(SHARED_SESSION)
    assert record_a.encode() == record_b.encode()

    with tenant_a.atomic() as tx:
        commit_a = tx.append(
            SHARED_SESSION,
            lease=lease_a,
            expected_seq=1,
            commit_id="same-commit",
            records=(record_a,),
        )
    with tenant_a.atomic() as tx:
        replay_commit_a = tx.append(
            SHARED_SESSION,
            lease=lease_a,
            expected_seq=1,
            commit_id="same-commit",
            records=(record_a,),
        )
    with tenant_b.atomic() as tx:
        commit_b = tx.append(
            SHARED_SESSION,
            lease=lease_b,
            expected_seq=1,
            commit_id="same-commit",
            records=(record_b,),
        )
    assert not commit_a.replayed and not commit_b.replayed
    assert replay_commit_a.replayed
    assert replay_commit_a.head_seq == commit_a.head_seq == commit_b.head_seq == 2

    assert tenant_a.read(SHARED_SESSION).head_seq == tenant_b.read(SHARED_SESSION).head_seq == 2
    assert tuple(item.item.input_id for item in tenant_a.peek_inbox(SHARED_SESSION)) == ("same-input",)
    assert tuple(item.item.input_id for item in tenant_b.peek_inbox(SHARED_SESSION)) == ("same-input",)
    with tenant_a.atomic() as tx:
        batch_a = tx.consumer_batch(SHARED_SESSION, "host")
        assert batch_a is not None
        tx.ack(batch_a)
    with tenant_b.atomic() as tx:
        batch_b = tx.consumer_batch(SHARED_SESSION, "host")
        assert batch_b is not None
        tx.ack(batch_b)
    assert tuple(item.commit_id for item in batch_a.records) == tuple(item.commit_id for item in batch_b.records)
    assert batch_a.records[-1].record.record_id == batch_b.records[-1].record.record_id
    assert tenant_a.release(lease_a)
    assert tenant_b.release(lease_b)


@pytest.mark.parametrize(
    "tenant_store",
    [
        pytest.param("sqlite", id="sqlite-file"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
    ],
    indirect=True,
)
def test_child_session_inherits_parent_tenant_and_rejects_cross_tenant_parent(tenant_store):
    tenant_a = tenant_store.for_tenant(TENANT_A)
    tenant_b = tenant_store.for_tenant(TENANT_B)
    _create(tenant_a, TENANT_A, "tenant-a-parent")

    with tenant_a.atomic() as tx:
        child = tx.create(_session(TENANT_A, "tenant-a-child", parent_session_id="tenant-a-parent"), consumers=())
    assert child.session_id == "tenant-a-child"
    assert tenant_a.read("tenant-a-child").records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == TENANT_A
    assert set(tenant_a.list_sessions()) == {"tenant-a-child", "tenant-a-parent"}
    assert tenant_b.list_sessions() == ()
    with pytest.raises(Conflict):
        tenant_b.read("tenant-a-child")

    with pytest.raises(Conflict), tenant_b.atomic() as tx:
        tx.create(_session(TENANT_B, "cross-tenant-child", parent_session_id="tenant-a-parent"), consumers=())
    with pytest.raises(Conflict):
        tenant_b.read("cross-tenant-child")

    with pytest.raises(Conflict), tenant_a.atomic() as tx:
        tx.create(_session(TENANT_A, "missing-parent-child", parent_session_id="missing-parent"), consumers=())
    with pytest.raises(Conflict), tenant_a.atomic() as tx:
        tx.create(_session(TENANT_B, "wrong-principal-child", parent_session_id="tenant-a-parent"), consumers=())
    assert "missing-parent-child" not in tenant_a.list_sessions()
    assert "wrong-principal-child" not in tenant_a.list_sessions()


def test_tenant_facade_survives_concurrent_and_cold_sqlite_connections(tmp_path):
    path = str(tmp_path / "independent.sqlite")
    clock = _Clock()
    with SQLiteStore.standalone(path, fold_factory=_fold_factory, clock=clock) as schema:
        schema.install_schema()

    barrier = Barrier(2)

    def initialize(store: SQLiteStore, tenant_id: str, owned_session: str) -> None:
        facade = store.for_tenant(tenant_id)
        barrier.wait()
        _create(facade, tenant_id, SHARED_SESSION)
        _create(facade, tenant_id, owned_session)

    with (
        SQLiteStore.standalone(path, fold_factory=_fold_factory, clock=clock) as connection_a,
        SQLiteStore.standalone(path, fold_factory=_fold_factory, clock=clock) as connection_b,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(initialize, connection_a, TENANT_A, "a-only"),
                executor.submit(initialize, connection_b, TENANT_B, "b-only"),
            )
            for future in futures:
                future.result()

        facade_a = connection_a.for_tenant(TENANT_A)
        assert "b-only" not in facade_a.list_sessions()
        with pytest.raises(Conflict):
            facade_a.read("b-only")
        page = facade_a.read(SHARED_SESSION)
        assert page.records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == TENANT_A

        facade_b = connection_b.for_tenant(TENANT_B)
        assert "a-only" not in facade_b.list_sessions()
        with pytest.raises(Conflict):
            facade_b.read("a-only")
        page = facade_b.read(SHARED_SESSION)
        assert page.records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == TENANT_B

    with SQLiteStore.standalone(path, fold_factory=_fold_factory, clock=clock) as cold:
        for tenant_id in (TENANT_A, TENANT_B):
            facade = cold.for_tenant(tenant_id)
            page = facade.read(SHARED_SESSION)
            assert page.records[0].record.to_dict()["payload"]["principal"]["tenant_id"] == tenant_id
        with pytest.raises(Conflict):
            cold.for_tenant(TENANT_A).read("b-only")
