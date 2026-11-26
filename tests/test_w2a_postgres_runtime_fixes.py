"""Regression coverage for PostgreSQL transaction, runtime binding and schema boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4

import pytest
from session.helpers import content_ref, principal

from vv_agent.kernel._postgres import SCHEMA_TABLES
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.session.records import SessionSpec
from vv_agent.session.postgres import PostgresStore, join_transaction
from vv_agent.session.reducer import Fold as SessionFold

TENANT_ID = "tenant-a"


def _fold_factory():
    raise AssertionError("runtime fix tests do not execute a fold")


@dataclass
class PGDatabase:
    admin: Any
    dsn: str
    roles: list[str] = field(default_factory=list)

    def add_role(self) -> tuple[str, str]:
        role = f"vv_w2a_fix_{uuid4().hex}"
        self.roles.append(role)
        return role, uuid4().hex


class HostPostgresStore(PostgresStore):
    """Django-style host adapter that intentionally does not initialize the superclass."""

    def __init__(self, raw):
        self.raw = raw
        self._connection_lock = RLock()

    @property
    def connection(self):
        return self.raw


class _InputSigner:
    def bind_key(self, *_args, **_kwargs):
        raise AssertionError("test signer key binding is not executed")

    def verify(self, *_args, **_kwargs):
        return True


class _CheckpointSigner:
    class _Binding:
        key_id = "test-checkpoint-key"

        def sign(self, _message):
            return b"test-checkpoint-signature"

    def bind_key(self):
        return self._Binding()

    def verify(self, *_args, **_kwargs):
        return True


@pytest.fixture(scope="session")
def postgres_admin():
    psycopg = pytest.importorskip("psycopg", reason="PostgreSQL tests need the postgres extra")
    configured = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    connection = None
    for dsn in ([configured] if configured else []) + ["dbname=postgres"]:
        try:
            connection = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
            break
        except psycopg.OperationalError:
            continue
    if connection is None:
        pytest.skip("No PostgreSQL via VV_AGENT_TEST_POSTGRES_DSN or local unix socket")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def pg_database(postgres_admin):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    admin_dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN") or postgres_admin.info.dsn
    name = f"vv_w2a_fix_{uuid4().hex}"
    try:
        postgres_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("PostgreSQL test role cannot create disposable databases")
    database = PGDatabase(postgres_admin, make_conninfo(admin_dsn, dbname=name))
    try:
        yield database
    finally:
        with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=3) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
            for role in database.roles:
                admin.execute(sql.SQL("REASSIGN OWNED BY {} TO CURRENT_USER").format(sql.Identifier(role)))
                admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _create_session(store) -> None:
    spec = SessionSpec(
        "s",
        principal(),
        content_ref("workspace/s"),
        attributes=content_ref("attributes/s"),
    )
    with store.atomic() as tx:
        tx.create(spec, consumers=())


def _install(database: PGDatabase) -> None:
    import psycopg

    with psycopg.connect(database.dsn, autocommit=True, connect_timeout=3) as connection:
        PostgresStore(connection).install_schema()


def _role_dsn(database: PGDatabase, role: str, password: str) -> str:
    from psycopg.conninfo import make_conninfo

    host = database.admin.info.host
    insecure_socket = not host or host.startswith("/")
    return make_conninfo(
        database.dsn,
        host="127.0.0.1" if insecure_socket else host,
        hostaddr="127.0.0.1" if insecure_socket else host,
        port=database.admin.info.port,
        user=role,
        password=password,
    )


def _require_createrole(database: PGDatabase) -> None:
    privileged = database.admin.execute("SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()
    if not privileged or not privileged[0]:
        pytest.skip("PostgreSQL test role lacks CREATEROLE for runtime binding coverage")


def _create_login(database: PGDatabase, *, comment: str | None = None) -> tuple[str, str, str]:
    from psycopg import sql

    role, password = database.add_role()
    database.admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
    if comment is not None:
        database.admin.execute(sql.SQL("COMMENT ON ROLE {} IS {}").format(sql.Identifier(role), sql.Literal(comment)))
    return role, password, _role_dsn(database, role, password)


def _provision_runtime_roles(database: PGDatabase) -> dict[str, str]:
    import psycopg
    from psycopg import sql

    _require_createrole(database)
    with psycopg.connect(database.dsn, autocommit=True, connect_timeout=3) as connection:
        store = CorePostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        schema = connection.execute("SELECT current_schema()").fetchone()
        assert schema is not None
        connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema[0])))
        provisioned = {}
        for mode in ("tenant", "worker", "shared-gateway"):
            role, password = database.add_role()
            if mode == "tenant":
                store.install_tenant_rls(TENANT_ID, role, password)
            elif mode == "worker":
                store.install_worker_rls(role, password)
            else:
                store.install_shared_gateway_rls(role, password)
            provisioned[mode] = _role_dsn(database, role, password)
    return provisioned


@pytest.mark.postgres
def test_default_connection_probes_remain_idle_and_atomic_writes_commit(pg_database):
    import psycopg
    from psycopg.pq import TransactionStatus

    _install(pg_database)
    with psycopg.connect(pg_database.dsn, connect_timeout=3) as connection:
        assert connection.info.transaction_status == TransactionStatus.IDLE
        store = PostgresStore(connection)
        assert connection.info.transaction_status == TransactionStatus.IDLE
        with store.atomic():
            store._rows("CREATE TABLE committed_probe(value integer PRIMARY KEY)")
            store._rows("INSERT INTO committed_probe VALUES (1)")
        assert connection.info.transaction_status == TransactionStatus.IDLE

    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as reader:
        assert reader.execute("SELECT value FROM committed_probe").fetchall() == [(1,)]


@pytest.mark.postgres
def test_join_transaction_rejects_idle_and_preserves_existing_transaction(pg_database):
    import psycopg
    from psycopg.pq import TransactionStatus

    _install(pg_database)
    with psycopg.connect(pg_database.dsn, connect_timeout=3) as connection:
        with pytest.raises(RuntimeError, match="open INTRANS"):
            join_transaction(connection)
        assert connection.info.transaction_status == TransactionStatus.IDLE

        with connection.transaction():
            connection.execute("CREATE TABLE joined_probe(value integer PRIMARY KEY)")
            assert connection.info.transaction_status == TransactionStatus.INTRANS
            transaction = join_transaction(connection)
            assert connection.info.transaction_status == TransactionStatus.INTRANS
            transaction.store._rows("INSERT INTO joined_probe VALUES (1)")
            assert connection.info.transaction_status == TransactionStatus.INTRANS
        assert connection.info.transaction_status == TransactionStatus.IDLE

    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as reader:
        assert reader.execute("SELECT value FROM joined_probe").fetchall() == [(1,)]


@pytest.mark.postgres
def test_host_subclass_without_super_init_has_unbound_runtime_metadata(pg_database):
    import psycopg
    from psycopg.pq import TransactionStatus

    _install(pg_database)
    with psycopg.connect(pg_database.dsn, connect_timeout=3) as connection:
        host = HostPostgresStore(connection)
        assert connection.info.transaction_status == TransactionStatus.IDLE
        assert host.runtime_mode == "unbound-internal"
        assert host.runtime_binding.credential_isolation is False
        assert host.credential_isolation is False
        assert host.credential_tenant_id is None
        assert host.for_tenant(TENANT_ID).tenant_id == TENANT_ID
        assert connection.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.postgres
def test_session_runtime_modes_heartbeats_and_owner_rejection_are_consistent(pg_database):
    import psycopg
    from psycopg.pq import TransactionStatus

    provisioned = _provision_runtime_roles(pg_database)

    with PostgresStore.tenant_standalone(provisioned["tenant"], tenant_id=TENANT_ID) as tenant:
        assert tenant.credential_isolation is True
        assert tenant.runtime_mode == "tenant"
        with pytest.raises(ValueError, match="another tenant"):
            tenant.for_tenant("tenant-b")
        with tenant.heartbeat_store() as heartbeat:
            assert heartbeat.credential_isolation is True
            assert heartbeat.runtime_mode == "tenant"

    for mode in ("worker", "shared-gateway"):
        opener = PostgresStore.worker_standalone if mode == "worker" else PostgresStore.shared_gateway_standalone
        with opener(provisioned[mode]) as store:
            assert store.credential_isolation is False
            assert store.runtime_mode == mode
            with store.heartbeat_store() as heartbeat:
                assert heartbeat.credential_isolation is False
                assert heartbeat.runtime_mode == mode

    with (
        pytest.raises(ValueError, match="unbound PostgreSQL roles"),
        PostgresStore.standalone(pg_database.dsn),
    ):
        pass
    with (
        psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection,
        pytest.raises(ValueError, match="unbound PostgreSQL roles"),
    ):
        PostgresStore(connection)
    with (
        pytest.raises(ValueError, match="unbound PostgreSQL roles"),
        CorePostgresStore.standalone(pg_database.dsn, fold_factory=_fold_factory),
    ):
        pass
    with psycopg.connect(pg_database.dsn, connect_timeout=3) as connection:
        with (
            connection.transaction(),
            pytest.raises(ValueError, match="unbound PostgreSQL roles"),
        ):
            connection.execute("SELECT 1")
            assert connection.info.transaction_status == TransactionStatus.INTRANS
            join_transaction(connection)
        assert connection.info.transaction_status == TransactionStatus.IDLE
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        internal = CorePostgresStore(connection, fold_factory=_fold_factory)
        assert internal.runtime_mode == "unbound-internal"
        assert internal.runtime_binding.credential_isolation is False
    with PostgresStore.schema_admin_standalone(pg_database.dsn) as admin:
        assert admin.runtime_mode == "unbound-internal"
        assert admin.credential_isolation is False
        admin.install_schema()
    with CorePostgresStore.schema_admin_standalone(pg_database.dsn, fold_factory=_fold_factory) as admin:
        assert admin.runtime_mode == "unbound-internal"
        admin.install_schema()


@pytest.mark.parametrize("facade", ["core", "session"])
@pytest.mark.postgres
def test_postgres_heartbeat_inherits_worker_level_and_input_admission(pg_database, facade):
    signer = _InputSigner()

    def exercise(store):
        store.install_schema()
        store.configure_worker(level=2)
        store.configure_input_admission(_CheckpointSigner())
        _create_session(store)
        store.configure_input_admission(signer)
        lease = store.acquire("s", owner="worker/root", ttl_ms=10_000)
        assert lease is not None

        with store.heartbeat_store() as heartbeat:
            assert heartbeat.connection is not store.connection
            assert heartbeat.worker_level == 2
            assert heartbeat._input_admission_policy() == (True, False, signer)
            renewed = heartbeat.renew(lease, ttl_ms=10_000)
            assert renewed.lease.epoch == lease.epoch
            assert heartbeat.release(renewed.lease)

    if facade == "core":
        with CorePostgresStore.standalone(pg_database.dsn, fold_factory=SessionFold) as store:
            exercise(store)
    else:
        with PostgresStore.standalone(pg_database.dsn) as store:
            exercise(store)


@pytest.mark.postgres
def test_benign_role_comment_does_not_claim_a_vv_agent_runtime_marker(pg_database):
    import psycopg
    from psycopg import sql

    _require_createrole(pg_database)
    _install(pg_database)
    role, _password, role_dsn = _create_login(pg_database, comment="ordinary database administrator")
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as admin:
        schema = admin.execute("SELECT current_schema()").fetchone()
        assert schema is not None
        admin.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema[0])))
        admin.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema[0]), sql.Identifier(role)))
        admin.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(sql.Identifier(schema[0]), sql.Identifier(role))
        )
        admin.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(pg_database.admin.info.dbname), sql.Identifier(role)
            )
        )

    with PostgresStore.standalone(role_dsn) as store:
        assert store.runtime_mode == "unbound-internal"
        assert store.credential_isolation is False


@pytest.mark.parametrize(
    "relation_kind",
    ["partial-table", "view", "views-with-managed-names", "materialized-view", "foreign-table"],
)
@pytest.mark.postgres
def test_any_nonempty_sk_relation_set_is_rejected_as_noncurrent(pg_database, relation_kind):
    import psycopg
    from psycopg import sql
    from psycopg.pq import TransactionStatus

    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as setup:
        if relation_kind == "partial-table":
            setup.execute("CREATE TABLE sk_session(session_id text PRIMARY KEY)")
        elif relation_kind == "view":
            setup.execute("CREATE VIEW sk_unexpected AS SELECT 1 AS value")
        elif relation_kind == "views-with-managed-names":
            for table in sorted(SCHEMA_TABLES):
                setup.execute(sql.SQL("CREATE VIEW {} AS SELECT 1 AS value").format(sql.Identifier(table)))
        elif relation_kind == "materialized-view":
            setup.execute("CREATE MATERIALIZED VIEW sk_unexpected AS SELECT 1 AS value")
        else:
            try:
                setup.execute("CREATE EXTENSION IF NOT EXISTS postgres_fdw")
                setup.execute("CREATE SERVER sk_foreign_server FOREIGN DATA WRAPPER postgres_fdw")
                setup.execute(
                    "CREATE FOREIGN TABLE sk_unexpected(value integer) SERVER sk_foreign_server OPTIONS (table_name 'remote')"
                )
            except psycopg.errors.InsufficientPrivilege:
                pytest.skip("PostgreSQL role cannot create postgres_fdw objects")

    with psycopg.connect(pg_database.dsn, connect_timeout=3) as connection:
        assert connection.info.transaction_status == TransactionStatus.IDLE
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            PostgresStore(connection)
        assert connection.info.transaction_status == TransactionStatus.IDLE
