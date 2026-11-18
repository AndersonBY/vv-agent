"""B1 regressions for tenant-bound drive and content transactions."""

from __future__ import annotations

from pathlib import Path

import pytest
from support.s1_fixture_values import create_fixture_session, wrap_fixture_store
from test_s1_postgres_rls import TENANT_A, TENANT_B, RLSDatabase, _dsn
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database
from test_s1_postgres_rls_matrix import _provision_with_settings_lock

from vv_agent import Agent, RunConfig
from vv_agent.kernel._content import ContentBindingMismatch, ContentLabels
from vv_agent.llm.base import LlmRequest
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.postgres import PostgresContentStore, PostgresStore, join_transaction
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

TENANT_X = "tenant-x"
TENANT_Y = "tenant-y"
MASTER_KEY = b"r" * 32


def _run_driver(driver: SessionDriver, session_id: str, workspace: Path) -> str:
    driver.create(session_id, str(workspace))

    def complete(_request: LlmRequest) -> LLMResponse:
        return LLMResponse("done")

    handle = driver.start(
        session_id,
        Agent("tenant-drive-agent", "Finish the request.", model="tenant-drive"),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback("tenant-drive", "tenant-drive", complete),
            workspace=workspace,
        ),
        "go",
    )
    return handle.result(timeout=10).final_output


def test_sqlite_tenant_x_driver_completes_without_reading_tenant_y(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "sqlite-keys.sqlite3")
    path = tmp_path / "tenant-routing.sqlite"
    try:
        with SQLiteStore.standalone(str(path)) as store:
            store.install_schema()
            create_fixture_session(wrap_fixture_store(store.for_tenant(TENANT_Y), provider), "shared-drive", consumers=())
            driver = SessionDriver(store=store, tenant_id=TENANT_X, content_key_provider=provider)
            try:
                assert _run_driver(driver, "shared-drive", tmp_path) == "done"
                x_page = store.for_tenant(TENANT_X).read("shared-drive")
                y_page = store.for_tenant(TENANT_Y).read("shared-drive")
            finally:
                driver.close()

        assert x_page.records[-1].record.kind == "turn_ended"
        assert y_page.head_seq == 1
        assert y_page.records[0].record.kind == "session_created"
    finally:
        provider.close()


def test_sqlite_facade_cannot_destroy_another_tenants_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "sqlite-destroy-keys.sqlite3")
    destroyed: list[str] = []
    original_destroy = provider.destroy

    def track_destroy(tenant_id: str, key_id: str):
        destroyed.append(key_id)
        return original_destroy(tenant_id, key_id)

    monkeypatch.setattr(provider, "destroy", track_destroy)
    try:
        with SQLiteStore.standalone(str(tmp_path / "sqlite-destroy.sqlite")) as store:
            store.install_schema()
            tenant_y = wrap_fixture_store(store.for_tenant(TENANT_Y), provider)
            create_fixture_session(tenant_y, "cross-tenant", consumers=())
            create_fixture_session(tenant_y, "source-facade", consumers=())
            create_fixture_session(tenant_y, "internal-delete", consumers=())

            tenant_x = store.for_tenant(TENANT_X)
            with pytest.raises(ContentBindingMismatch, match="another tenant"), tenant_x.atomic() as tx:
                SQLiteContentStore(tx, provider).forget_session(TENANT_Y, "cross-tenant")

            assert destroyed == []
            assert store._one(
                "SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
                (TENANT_Y, "cross-tenant"),
            ) == (1,)

            with pytest.raises(ContentBindingMismatch, match="another tenant"), tenant_x.atomic():
                SQLiteContentStore(tenant_x, provider).forget_session(TENANT_Y, "source-facade")

            assert destroyed == []
            assert store._one(
                "SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
                (TENANT_Y, "source-facade"),
            ) == (1,)

            with store.atomic() as tx:
                SQLiteContentStore(tx, provider).forget_session(TENANT_Y, "internal-delete")

            assert destroyed
            assert store._one(
                "SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
                (TENANT_Y, "internal-delete"),
            ) == (0,)
    finally:
        provider.close()


@pytest.mark.postgres
def test_postgres_worker_drives_only_authenticated_tenant_x(
    rls_database: RLSDatabase,
    tmp_path: Path,
) -> None:
    _provision_with_settings_lock(rls_database)
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "postgres-keys.sqlite3")
    try:
        with PostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_b_role, rls_database.tenant_b_password),
            tenant_id=TENANT_B,
        ) as tenant_y:
            create_fixture_session(wrap_fixture_store(tenant_y, provider), "shared-drive", consumers=())

        with PostgresStore.worker_standalone(
            _dsn(rls_database, rls_database.worker_role, rls_database.worker_password)
        ) as worker:
            driver = SessionDriver(store=worker, tenant_id=TENANT_A, content_key_provider=provider)
            try:
                assert _run_driver(driver, "shared-drive", tmp_path) == "done"
                x_page = driver.store.read("shared-drive")
            finally:
                driver.close()

        with PostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_b_role, rls_database.tenant_b_password),
            tenant_id=TENANT_B,
        ) as tenant_y:
            y_page = tenant_y.read("shared-drive")

        assert x_page.records[-1].record.kind == "turn_ended"
        assert y_page.head_seq == 1
        assert y_page.records[0].record.kind == "session_created"
    finally:
        provider.close()


@pytest.mark.postgres
def test_tenant_bound_worker_cannot_destroy_another_tenants_session_key(
    rls_database: RLSDatabase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provision_with_settings_lock(rls_database)
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "destroy-keys.sqlite3")
    destroyed: list[str] = []
    original_destroy = provider.destroy

    def track_destroy(tenant_id: str, key_id: str):
        destroyed.append(key_id)
        return original_destroy(tenant_id, key_id)

    monkeypatch.setattr(provider, "destroy", track_destroy)
    try:
        with PostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_b_role, rls_database.tenant_b_password),
            tenant_id=TENANT_B,
        ) as tenant_b:
            create_fixture_session(wrap_fixture_store(tenant_b, provider), "cross-tenant", consumers=())

        with PostgresStore.worker_standalone(
            _dsn(rls_database, rls_database.worker_role, rls_database.worker_password)
        ) as worker:
            tenant_a = worker.for_tenant(TENANT_A)
            with pytest.raises(ContentBindingMismatch, match="another tenant"), tenant_a.atomic() as tx:
                PostgresContentStore(tx, provider).forget_session(TENANT_B, "cross-tenant")

        assert destroyed == []
    finally:
        provider.close()


@pytest.mark.postgres
def test_join_transaction_binds_authenticated_tenant_for_content_io(
    rls_database: RLSDatabase,
    tmp_path: Path,
) -> None:
    _provision_with_settings_lock(rls_database)
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "join-keys.sqlite3")
    try:
        with PostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password),
            tenant_id=TENANT_A,
        ) as tenant_a:
            create_fixture_session(wrap_fixture_store(tenant_a, provider), "joined-content", consumers=())
            with tenant_a.connection.transaction():
                tx = join_transaction(tenant_a.connection)
                content = PostgresContentStore(tx, provider)
                ref = content.put(TENANT_A, "joined-content", b"tenant-a-content", ContentLabels())
                assert content.get(TENANT_A, "joined-content", ref) == b"tenant-a-content"
                assert tenant_a.connection.execute("SELECT current_setting('vv.tenant_id',true)").fetchone() == (TENANT_A,)
    finally:
        provider.close()
