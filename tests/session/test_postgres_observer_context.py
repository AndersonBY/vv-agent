"""PostgreSQL observers bind only explicit tenant views to transaction-local RLS context."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, current_thread, main_thread
from typing import Any, cast

import pytest
from test_s1_postgres_rls import TENANT_A, TENANT_B, RLSDatabase, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel._content import (
    SESSION_WORKSPACE,
    ContentRef,
    KeyDestroyOutcome,
    KeyProvider,
    WrappedDataKey,
)
from vv_agent.kernel._postgres import CorePostgresContentStore, RuntimeBinding, _PostgresContentBackend
from vv_agent.session.content_runtime import (
    bind_content_key_provider,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.surfaces import _read_snapshot
from vv_agent.session.worker import LocalWorker, WorkerSupport

from .helpers import append_fixture_records, create_fixture_session, record


def _provider(tmp_path: Path) -> LocalKeyProvider:
    return LocalKeyProvider(b"o" * 32, state_path=tmp_path / "observer-keys.sqlite3")


def _grant_seed_ceiling_write(database: RLSDatabase) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(
        _dsn(database, database.migration_role, database.migration_password),
        autocommit=True,
    ) as connection:
        connection.execute(sql.SQL("GRANT UPDATE ON TABLE sk_settings TO {}").format(sql.Identifier(database.worker_role)))


def _open_seed_store(database: RLSDatabase) -> Any:
    import psycopg

    return psycopg.connect(
        _dsn(database, database.worker_role, database.worker_password),
        autocommit=True,
    )


def _seed_store(connection: Any) -> PostgresStore:
    return PostgresStore(connection, runtime_binding=RuntimeBinding("worker"))


def _create_session(
    database: RLSDatabase,
    *,
    tenant_id: str,
    session_id: str,
    key_provider: KeyProvider,
) -> None:
    with _open_seed_store(database) as connection, connection.transaction():
        connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (tenant_id,))
        store = _seed_store(connection)
        bind_content_key_provider(store, key_provider)
        wrapped = wrap_session_content_store(store.for_tenant(tenant_id), key_provider)
        create_fixture_session(wrapped, session_id, consumers=())


def _append_turn(
    database: RLSDatabase,
    *,
    tenant_id: str,
    session_id: str,
    key_provider: KeyProvider,
) -> None:
    with _open_seed_store(database) as connection, connection.transaction():
        connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (tenant_id,))
        store = _seed_store(connection)
        bind_content_key_provider(store, key_provider)
        wrapped = wrap_session_content_store(store.for_tenant(tenant_id), key_provider)
        append_fixture_records(
            wrapped,
            session_id,
            [record("turn_started", sid=session_id, tid="turn-observer", oid="op-observer")],
            commit_id="observer-turn",
        )


class _TrackingKeyProvider:
    def __init__(self, provider: LocalKeyProvider, connection: Any) -> None:
        self._provider = provider
        self._connection = connection
        self.provider_id = provider.provider_id
        self.unwrap_count = 0

    def generate_data_key(self, tenant_id: str, session_id: str) -> WrappedDataKey:
        return self._provider.generate_data_key(tenant_id, session_id)

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: WrappedDataKey) -> bytes:
        from psycopg.pq import TransactionStatus

        assert self._connection.info.transaction_status == TransactionStatus.IDLE
        self.unwrap_count += 1
        return self._provider.unwrap(tenant_id, session_id, wrapped_key)

    def destroy(self, tenant_id: str, key_id: str) -> KeyDestroyOutcome:
        return self._provider.destroy(tenant_id, key_id)


@pytest.mark.postgres
@pytest.mark.parametrize("mode", ["tenant", "shared-gateway"])
def test_observer_independent_connection_reads_bound_tenant_and_content(rls_database, tmp_path, mode):
    _provision(rls_database)
    _grant_seed_ceiling_write(rls_database)
    provider = _provider(tmp_path)
    try:
        _create_session(
            rls_database,
            tenant_id=TENANT_A,
            session_id="observer-a",
            key_provider=provider,
        )
        _create_session(
            rls_database,
            tenant_id=TENANT_B,
            session_id="observer-b",
            key_provider=provider,
        )

        if mode == "tenant":
            root = PostgresStore.tenant_standalone(
                _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password),
                tenant_id=TENANT_A,
            )
        else:
            root = PostgresStore.shared_gateway_standalone(
                _dsn(rls_database, rls_database.gateway_role, rls_database.gateway_password)
            )

        first = Event()
        second = Event()
        observed_heads: list[int] = []
        observer_connections: list[Any] = []
        worker_threads: list[Any] = []
        observed_content: list[bytes] = []

        with root as root_store:
            root_connection = root_store.connection
            opener = root_store.drive_store_factory()

            @contextmanager
            def open_store() -> Iterator[Any]:
                with opener() as store:
                    concrete = cast(Any, store)
                    if concrete.tenant_id != TENANT_A:
                        concrete = concrete.for_tenant(TENANT_A)
                    observer_connections.append(concrete.connection)
                    worker_threads.append(current_thread())
                    yield concrete
                    from psycopg.pq import TransactionStatus

                    assert concrete.connection.info.transaction_status == TransactionStatus.IDLE

            def observe(store: Any) -> None:
                observer = store
                assert observer.tenant_id == TENANT_A
                assert observer.connection is not root_store.connection
                with _read_snapshot(observer):
                    page = observer.read("observer-a")
                    state, records, inbox_seq = observer.read_state("observer-a")
                    assert state is not None
                    assert len(records) == page.head_seq
                    assert inbox_seq == page.inbox_seq == 0
                    assert observer.peek_inbox("observer-a") == ()
                    assert observer.lease_status("observer-a")[0] == page.head_seq

                    backend = cast(
                        _PostgresContentBackend,
                        CorePostgresContentStore(observer, provider, AESGCMCipher())._backend,
                    )
                    workspace_ref = ContentRef.from_dict(page.records[0].record._payload["workspace"])
                    with backend.read_snapshot():
                        rows = backend.fetchall(
                            "SELECT s.session_id,k.key_id,c.content_id FROM sk_session AS s "
                            "JOIN sk_session_key AS k USING (tenant_id,session_id) "
                            "JOIN sk_content AS c USING (tenant_id,session_id) "
                            "WHERE s.tenant_id={p} AND s.session_id={p}",
                            (TENANT_A, "observer-a"),
                        )
                        assert rows and all(row[0] == "observer-a" for row in rows)
                        assert (
                            backend.fetchall(
                                "SELECT session_id FROM sk_session WHERE tenant_id={p} AND session_id={p}",
                                (TENANT_B, "observer-b"),
                            )
                            == []
                        )

                if not isinstance(backend, _PostgresContentBackend):
                    raise AssertionError("observer content backend is not PostgreSQL")

                tracking = _TrackingKeyProvider(provider, observer.connection)
                content_observer = wrap_session_content_store(observer, tracking)
                reader = durable_content_reader(content_observer, tenant_id=TENANT_A, session_id="observer-a")
                with reader._read_frame(record_seqs=(1,)) as frame:
                    observed_content.append(
                        frame.read(
                            TENANT_A,
                            "observer-a",
                            "record:1",
                            SESSION_WORKSPACE,
                            workspace_ref,
                        )
                    )
                assert tracking.unwrap_count == 1

                observed_heads.append(page.head_seq)
                if page.head_seq == 1:
                    first.set()
                elif page.head_seq == 2:
                    second.set()

            worker = LocalWorker(
                WorkerSupport(
                    open_store=open_store,
                    observe=observe,
                    projection_consumers=(),
                ),
                scan_interval=0.01,
                name=f"postgres-observer-{mode}",
            )
            try:
                worker.start()
                assert first.wait(3), worker.last_error
                _append_turn(
                    rls_database,
                    tenant_id=TENANT_A,
                    session_id="observer-a",
                    key_provider=provider,
                )
                assert second.wait(3), worker.last_error
            finally:
                assert worker.close(timeout=3), worker.last_error

        assert 1 in observed_heads and 2 in observed_heads
        assert observer_connections
        assert all(connection is not root_connection for connection in observer_connections)
        assert all(connection.closed for connection in observer_connections)
        assert worker_threads and all(thread is not main_thread() for thread in worker_threads)
        assert json.loads(observed_content[0]) == "/tmp"
    finally:
        provider.close()


@pytest.mark.postgres
def test_bare_gateway_and_tenant_credential_forgery_remain_fail_closed(rls_database, tmp_path):
    import psycopg
    from psycopg.pq import TransactionStatus

    from vv_agent.kernel._hydration import MissingContentLink
    from vv_agent.kernel._store import Conflict

    _provision(rls_database)
    _grant_seed_ceiling_write(rls_database)
    provider = _provider(tmp_path)
    try:
        _create_session(
            rls_database,
            tenant_id=TENANT_A,
            session_id="gateway-a",
            key_provider=provider,
        )
        _create_session(
            rls_database,
            tenant_id=TENANT_B,
            session_id="tenant-b",
            key_provider=provider,
        )

        with PostgresStore.shared_gateway_standalone(
            _dsn(rls_database, rls_database.gateway_role, rls_database.gateway_password)
        ) as gateway:
            with pytest.raises(Conflict):
                gateway.read("gateway-a")
            backend = cast(
                _PostgresContentBackend,
                CorePostgresContentStore(gateway, provider, AESGCMCipher())._backend,
            )
            with backend.read_snapshot():
                assert backend.fetchall("SELECT tenant_id,session_id FROM sk_session ORDER BY tenant_id,session_id") == []
            assert gateway.connection.info.transaction_status == TransactionStatus.IDLE

        tenant_dsn = _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password)
        with psycopg.connect(tenant_dsn, autocommit=True) as tenant_a:
            store = PostgresStore(tenant_a, tenant_id=TENANT_A)
            with pytest.raises(ValueError, match="another tenant"):
                store.for_tenant(TENANT_B)
            page = store.read("gateway-a")
            workspace_ref = ContentRef.from_dict(page.records[0].record._payload["workspace"])
            bind_content_key_provider(store, provider)
            content_store = wrap_session_content_store(store, provider)
            reader = durable_content_reader(content_store, tenant_id=TENANT_A, session_id="gateway-a")
            with tenant_a.transaction():
                tenant_a.execute("SELECT set_config('vv.tenant_id',%s,true)", (TENANT_B,))
                with pytest.raises(Conflict):
                    store.read("tenant-b")
                with reader._read_frame(record_seqs=(1,)) as frame:
                    with pytest.raises(MissingContentLink):
                        frame.read(
                            TENANT_A,
                            "gateway-a",
                            "record:1",
                            SESSION_WORKSPACE,
                            workspace_ref,
                        )
                    assert tenant_a.execute("SELECT current_setting('vv.tenant_id',true)").fetchone() == (TENANT_B,)
            assert tenant_a.info.transaction_status == TransactionStatus.IDLE
    finally:
        provider.close()
