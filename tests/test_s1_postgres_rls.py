"""Credential-level PostgreSQL tenant isolation for the S1 store."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from vv_agent.kernel import _postgres
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.session.postgres import PostgresStore as SessionPostgresStore

TENANT_A = "tenant-a'quoted"
TENANT_B = "tenant-b"


def _fold_factory():
    raise AssertionError("RLS tests do not execute a fold")


@dataclass(frozen=True)
class PGCluster:
    host: str
    port: int
    admin_role: str
    admin_password: str


@dataclass(frozen=True)
class RLSDatabase:
    cluster: PGCluster
    name: str
    migration_role: str
    migration_password: str
    superuser_role: str
    superuser_password: str
    bypass_role: str
    bypass_password: str
    tenant_a_role: str
    tenant_a_password: str
    tenant_b_role: str
    tenant_b_password: str
    worker_role: str
    worker_password: str
    gateway_role: str
    gateway_password: str


def _postgres_bindir() -> Path | None:
    pg_config = shutil.which("pg_config")
    if pg_config is not None:
        result = subprocess.run([pg_config, "--bindir"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            path = Path(result.stdout.strip())
            if (path / "initdb").exists() and (path / "pg_ctl").exists():
                return path
    for path in sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True):
        if (path / "initdb").exists() and (path / "pg_ctl").exists():
            return path
    return None


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or f"command failed: {command[0]}")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture(scope="module")
def disposable_pg(tmp_path_factory) -> Iterator[PGCluster]:
    psycopg = pytest.importorskip("psycopg", reason="PostgreSQL tests need the postgres extra")
    from psycopg.conninfo import make_conninfo

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("PostgreSQL initdb refuses to run as root")
    bindir = _postgres_bindir()
    if bindir is None:
        pytest.skip("PostgreSQL initdb and pg_ctl are unavailable")
    root = tmp_path_factory.mktemp("postgres-rls")
    data = root / "data"
    sockets = root / "sockets"
    sockets.mkdir()
    password_file = root / "password"
    admin_role = f"vv_rls_admin_{uuid4().hex}"
    admin_password = f"{uuid4().hex}'admin"
    password_file.write_text(admin_password, encoding="utf-8")
    _run(
        [
            str(bindir / "initdb"),
            "-D",
            str(data),
            "-U",
            admin_role,
            "--pwfile",
            str(password_file),
            "--auth-local=trust",
            "--auth-host=scram-sha-256",
            "--no-locale",
            "--encoding=UTF8",
            "--no-sync",
        ]
    )
    port = _free_port()
    log = root / "postgres.log"
    started = False
    try:
        _run(
            [
                str(bindir / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(log),
                "-o",
                f"-p {port} -h 127.0.0.1 -k {sockets}",
                "-w",
                "start",
            ]
        )
        started = True
        cluster = PGCluster("127.0.0.1", port, admin_role, admin_password)
        dsn = make_conninfo(
            host=cluster.host,
            hostaddr=cluster.host,
            port=cluster.port,
            user=cluster.admin_role,
            password=cluster.admin_password,
            dbname="postgres",
            connect_timeout=3,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                with psycopg.connect(dsn):
                    break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        yield cluster
    finally:
        if started:
            _run([str(bindir / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"])


def _dsn(database: RLSDatabase, role: str, password: str) -> str:
    from psycopg.conninfo import make_conninfo

    return make_conninfo(
        host=database.cluster.host,
        hostaddr=database.cluster.host,
        port=database.cluster.port,
        user=role,
        password=password,
        dbname=database.name,
        connect_timeout=3,
    )


@pytest.fixture
def rls_database(disposable_pg: PGCluster) -> Iterator[RLSDatabase]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    nonce = uuid4().hex
    database = RLSDatabase(
        cluster=disposable_pg,
        name=f"vv_rls_{nonce}",
        migration_role=f"vv_rls_migration_{nonce}",
        migration_password=f"{uuid4().hex}'migration",
        superuser_role=f"vv_rls_super_{nonce}",
        superuser_password=f"{uuid4().hex}'super",
        bypass_role=f"vv_rls_bypass_{nonce}",
        bypass_password=f"{uuid4().hex}'bypass",
        tenant_a_role=f'vv tenant "A" {nonce}',
        tenant_a_password=f"{uuid4().hex}'tenant-a",
        tenant_b_role=f"vv_rls_tenant_b_{nonce}",
        tenant_b_password=f"{uuid4().hex}'tenant-b",
        worker_role=f"vv_rls_worker_{nonce}",
        worker_password=f"{uuid4().hex}'worker",
        gateway_role=f"vv_rls_gateway_{nonce}",
        gateway_password=f"{uuid4().hex}'gateway",
    )
    admin_dsn = make_conninfo(
        host=disposable_pg.host,
        hostaddr=disposable_pg.host,
        port=disposable_pg.port,
        user=disposable_pg.admin_role,
        password=disposable_pg.admin_password,
        dbname="postgres",
        connect_timeout=3,
    )
    roles = [
        (database.migration_role, database.migration_password, sql.SQL("CREATEROLE")),
        (database.superuser_role, database.superuser_password, sql.SQL("SUPERUSER")),
        (database.bypass_role, database.bypass_password, sql.SQL("BYPASSRLS")),
    ]
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        for role, password, attribute in roles:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} {}").format(sql.Identifier(role), sql.Literal(password), attribute)
            )
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(database.name), sql.Identifier(database.migration_role))
        )
    try:
        yield database
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database.name)))
            cleanup_roles = [
                database.tenant_a_role,
                database.tenant_b_role,
                database.worker_role,
                database.gateway_role,
                database.bypass_role,
                database.superuser_role,
                database.migration_role,
            ]
            for role in cleanup_roles:
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _provision(database: RLSDatabase):
    import psycopg

    with psycopg.connect(_dsn(database, database.migration_role, database.migration_password), autocommit=True) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        for tenant_id, session_id in ((TENANT_A, "a-owned"), (TENANT_B, "b-owned")):
            connection.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES (%s,%s,'S1','idle',1,'live',0)",
                (tenant_id, session_id),
            )
        tenant_a = store.install_tenant_rls(TENANT_A, database.tenant_a_role, database.tenant_a_password)
        tenant_b = store.install_tenant_rls(TENANT_B, database.tenant_b_role, database.tenant_b_password)
        worker = store.install_worker_rls(database.worker_role, database.worker_password)
        gateway = store.install_shared_gateway_rls(database.gateway_role, database.gateway_password)
    return tenant_a, tenant_b, worker, gateway


def _query_in_context(connection, tenant_id: str, query: str, params: tuple[object, ...] = ()):
    with connection.transaction():
        connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (tenant_id,))
        cursor = connection.execute(query, params)
        return cursor.fetchall() if cursor.description is not None else []


def test_installation_forces_rls_and_rejects_unsafe_runtime_roles(rls_database):
    import psycopg

    tenant_a, tenant_b, worker, gateway = _provision(rls_database)
    assert tenant_a.mode == "tenant" and tenant_a.credential_isolation
    assert tenant_b.mode == "tenant" and tenant_b.credential_isolation
    assert worker.mode == "worker" and not worker.credential_isolation
    assert gateway.mode == "shared-gateway" and not gateway.credential_isolation
    assert _postgres.RLS_ISOLATION == "explicit-tenant-credentials"
    assert "install_tenant_rls" in _postgres.RLS_INSTALL_REASON
    assert "credential isolation" in _postgres.RLS_SHARED_GATEWAY_REASON

    with psycopg.connect(_dsn(rls_database, rls_database.migration_role, rls_database.migration_password)) as owner:
        guarded = owner.execute(
            "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(%s) ORDER BY c.relname",
            (sorted(_postgres.SESSION_TABLES),),
        ).fetchall()
        assert guarded == [(table, True, True) for table in sorted(_postgres.SESSION_TABLES)]
        assert owner.execute(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=current_schema() AND p.prosecdef"
        ).fetchone() == (0,)
        for role, password in (
            (rls_database.migration_role, rls_database.migration_password),
            (rls_database.superuser_role, rls_database.superuser_password),
            (rls_database.bypass_role, rls_database.bypass_password),
        ):
            with (
                psycopg.connect(_dsn(rls_database, role, password)) as unsafe,
                pytest.raises(ValueError, match="credential-bound"),
            ):
                PostgresStore.verify_tenant_connection(unsafe, TENANT_A)

    with psycopg.connect(
        _dsn(rls_database, rls_database.migration_role, rls_database.migration_password), autocommit=True
    ) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        with pytest.raises(ValueError, match="matching vv-agent RLS marker"):
            store.install_tenant_rls(TENANT_A, rls_database.migration_role, rls_database.migration_password)


def test_tenant_credentials_are_bound_to_tenant_and_forced_context(rls_database):
    import psycopg
    from psycopg import sql

    _provision(rls_database)
    with psycopg.connect(
        _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password), autocommit=True
    ) as tenant_a:
        PostgresStore.verify_tenant_connection(tenant_a, TENANT_A)
        with PostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password),
            tenant_id=TENANT_A,
            fold_factory=_fold_factory,
        ):
            pass
        with SessionPostgresStore.tenant_standalone(
            _dsn(rls_database, rls_database.tenant_a_role, rls_database.tenant_a_password), tenant_id=TENANT_A
        ):
            pass
        assert tenant_a.info.host == rls_database.cluster.host
        assert tenant_a.info.password == rls_database.tenant_a_password
        with tenant_a.transaction():
            assert tenant_a.execute("SELECT session_id FROM sk_session ORDER BY session_id").fetchall() == []
        assert _query_in_context(tenant_a, TENANT_A, "SELECT session_id FROM sk_session ORDER BY session_id") == [("a-owned",)]
        with pytest.raises(psycopg.errors.InsufficientPrivilege), tenant_a.transaction():
            tenant_a.execute(
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES (%s,'a-no-context','S1','idle',1,'live',0)",
                (TENANT_A,),
            )
        assert _query_in_context(tenant_a, TENANT_B, "SELECT session_id FROM sk_session ORDER BY session_id") == []
        assert _query_in_context(tenant_a, TENANT_A, "SELECT session_id FROM sk_session ORDER BY session_id") == [("a-owned",)]
        _query_in_context(
            tenant_a,
            TENANT_A,
            "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
            "VALUES (%s,'a-new','S1','idle',1,'live',0)",
            (TENANT_A,),
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _query_in_context(
                tenant_a,
                TENANT_B,
                "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                "VALUES (%s,'b-forged','S1','idle',1,'live',0)",
                (TENANT_B,),
            )
        for statement in (
            sql.SQL("SET ROLE {}").format(sql.Identifier(rls_database.migration_role)),
            sql.SQL("SET SESSION AUTHORIZATION {}").format(sql.Identifier(rls_database.migration_role)),
            sql.SQL("TRUNCATE sk_session"),
            sql.SQL("ALTER TABLE sk_session ADD COLUMN forbidden integer"),
            sql.SQL("CREATE POLICY forbidden ON sk_session USING (true)"),
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), tenant_a.transaction():
                tenant_a.execute(statement)
        for role, password in (
            (rls_database.migration_role, rls_database.migration_password),
            (rls_database.worker_role, rls_database.worker_password),
        ):
            with (
                pytest.raises(ValueError, match="credential-bound"),
                PostgresStore.tenant_standalone(
                    _dsn(rls_database, role, password), tenant_id=TENANT_A, fold_factory=_fold_factory
                ),
            ):
                pass

    with psycopg.connect(
        _dsn(rls_database, rls_database.tenant_b_role, rls_database.tenant_b_password), autocommit=True
    ) as tenant_b:
        PostgresStore.verify_tenant_connection(tenant_b, TENANT_B)
        assert _query_in_context(tenant_b, TENANT_B, "SELECT session_id FROM sk_session ORDER BY session_id") == [("b-owned",)]
        assert _query_in_context(tenant_b, TENANT_A, "SELECT session_id FROM sk_session ORDER BY session_id") == []


def test_worker_and_shared_gateway_are_explicit_non_tenant_paths(rls_database):
    import psycopg

    _, _, worker_installation, gateway_installation = _provision(rls_database)
    with psycopg.connect(_dsn(rls_database, rls_database.worker_role, rls_database.worker_password), autocommit=True) as worker:
        assert worker.execute("SELECT tenant_id,session_id FROM sk_session ORDER BY tenant_id").fetchall() == [
            (TENANT_A, "a-owned"),
            (TENANT_B, "b-owned"),
        ]
        with pytest.raises(ValueError, match="credential-bound"):
            PostgresStore.verify_tenant_connection(worker, TENANT_A)
    assert worker_installation.credential_isolation is False

    with psycopg.connect(
        _dsn(rls_database, rls_database.gateway_role, rls_database.gateway_password), autocommit=True
    ) as gateway:
        assert _query_in_context(gateway, TENANT_A, "SELECT session_id FROM sk_session") == [("a-owned",)]
        assert _query_in_context(gateway, TENANT_B, "SELECT session_id FROM sk_session") == [("b-owned",)]
    assert gateway_installation.credential_isolation is False
