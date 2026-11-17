"""W2a PostgreSQL schema parity and explicit runtime binding boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import pytest
from support.s1_fixture_values import content_ref, principal

from vv_agent.kernel import _postgres
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._record_schema import INPUT_PAYLOADS, PAYLOADS
from vv_agent.kernel._records import InboxItem, Record, make_record
from vv_agent.session.postgres import PostgresStore as SessionPostgresStore

TENANT_ID = "tenant-a'quoted"
ZERO_DIGEST = "0" * 64


def _fold_factory():
    raise AssertionError("schema tests do not execute a fold")


@dataclass
class PGDatabase:
    admin: Any
    dsn: str
    name: str
    roles: list[str] = field(default_factory=list)

    def add_role(self) -> tuple[str, str]:
        role = f"vv_w2a_{uuid4().hex}"
        self.roles.append(role)
        return role, uuid4().hex


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

    name = f"vv_w2a_{uuid4().hex}"
    try:
        postgres_admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("PostgreSQL test role cannot create disposable databases")
    database = PGDatabase(postgres_admin, make_conninfo(postgres_admin.info.dsn, dbname=name), name)
    try:
        yield database
    finally:
        with psycopg.connect(postgres_admin.info.dsn, autocommit=True, connect_timeout=3) as admin:
            _wait_for_no_live_database_connections(database)
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
            for role in database.roles:
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _install(database: PGDatabase) -> None:
    import psycopg

    with psycopg.connect(database.dsn, autocommit=True, connect_timeout=3) as connection:
        PostgresStore(connection, fold_factory=_fold_factory).install_schema()


def _wait_for_no_live_database_connections(database: PGDatabase) -> None:
    deadline = monotonic() + 2
    while True:
        active = database.admin.execute(
            "SELECT pid,state,query FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid() ORDER BY pid",
            (database.name,),
        ).fetchall()
        if not active:
            return
        if monotonic() >= deadline:
            raise AssertionError(f"disposable database {database.name} still has live connections: {active!r}")
        sleep(0.02)


def _assert_fresh_connection_reopen_rejected(database: PGDatabase) -> None:
    import psycopg

    with (
        psycopg.connect(database.dsn, autocommit=True, connect_timeout=3) as connection,
        pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"),
    ):
        PostgresStore(connection, fold_factory=_fold_factory)


def _install_runtime_roles(database: PGDatabase):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    privileged = database.admin.execute("SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()
    if not privileged or not privileged[0]:
        pytest.skip("PostgreSQL test role lacks CREATEROLE for runtime binding coverage")

    provisioned = {}
    with psycopg.connect(database.dsn, autocommit=True, connect_timeout=3) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        schema_row = connection.execute("SELECT current_schema()").fetchone()
        assert schema_row is not None
        schema = schema_row[0]
        connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema)))
        for mode in ("tenant", "worker", "shared-gateway"):
            role, password = database.add_role()
            if mode == "tenant":
                installation = store.install_tenant_rls(TENANT_ID, role, password)
            elif mode == "worker":
                installation = store.install_worker_rls(role, password)
            else:
                installation = store.install_shared_gateway_rls(role, password)
            host = database.admin.info.host
            hostaddr = host if host and not host.startswith("/") else "127.0.0.1"
            provisioned[mode] = (
                make_conninfo(
                    database.dsn,
                    host=host if host and not host.startswith("/") else "127.0.0.1",
                    hostaddr=hostaddr,
                    port=database.admin.info.port,
                    user=role,
                    password=password,
                ),
                installation,
            )
    return provisioned


@pytest.mark.postgres
def test_postgres_schema_install_reopen_and_write_level_are_idempotent(pg_database):
    import psycopg

    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        store.install_schema()
        assert _postgres._schema_fingerprint(connection) == _postgres.SCHEMA_FINGERPRINT
        assert store.runtime_binding == _postgres.UNBOUND_INTERNAL_BINDING
        connection.execute("UPDATE sk_settings SET write_level=2")
        store.install_schema()

    with PostgresStore.standalone(pg_database.dsn, fold_factory=_fold_factory) as reopened:
        assert reopened._one("SELECT storage_major,write_level FROM sk_settings") == ("S1", 2)
        assert _postgres._schema_is_current(reopened.connection)
        reopened.install_schema()


@pytest.mark.parametrize(
    "drift",
    [
        "ALTER TABLE sk_session DROP COLUMN handler_version",
        "DROP INDEX sk_record_turn",
        "ALTER TABLE sk_record DROP CONSTRAINT sk_record_tenant_id_session_id_record_id_key",
        "DROP TABLE sk_checkpoint",
        "CREATE TABLE sk_extra(value integer)",
        "CREATE FUNCTION sk_forbidden_trigger() RETURNS trigger LANGUAGE plpgsql "
        "AS $$BEGIN RETURN NEW; END$$; CREATE TRIGGER sk_forbidden BEFORE INSERT ON sk_session "
        "FOR EACH ROW EXECUTE FUNCTION sk_forbidden_trigger()",
        "ALTER TABLE sk_settings DROP CONSTRAINT sk_settings_write_level_check; UPDATE sk_settings SET write_level=0",
    ],
)
@pytest.mark.postgres
def test_postgres_schema_rejects_catalog_drift(pg_database, drift):
    import psycopg

    _install(pg_database)
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        connection.execute(drift)
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            PostgresStore(connection, fold_factory=_fold_factory).install_schema()

    _assert_fresh_connection_reopen_rejected(pg_database)
    _wait_for_no_live_database_connections(pg_database)


@pytest.mark.postgres
def test_postgres_rejects_partial_initialization_on_install_and_reopen(pg_database):
    import psycopg

    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        connection.execute("CREATE TABLE sk_session(session_id text PRIMARY KEY)")
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            PostgresStore(connection, fold_factory=_fold_factory).install_schema()

    _assert_fresh_connection_reopen_rejected(pg_database)
    _wait_for_no_live_database_connections(pg_database)


@pytest.mark.postgres
def test_postgres_record_kind_check_is_additive_but_codec_remains_closed(pg_database):
    import psycopg

    assert len(PAYLOADS) == 14
    _install(pg_database)
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        connection.execute(
            "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
            "VALUES ('tenant-a','s','S1','idle',1,'live',0)"
        )

        def insert_kind(seq, kind):
            connection.execute(
                "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,schema_version,kind,body,digest,"
                "chain_digest,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
                "VALUES ('tenant-a','s',%s,%s,'S1',%s,%s,%s,%s,%s,%s,0,0,0)",
                (seq, f"r{seq}", kind, b"{}", ZERO_DIGEST, ZERO_DIGEST, f"c{seq}", ZERO_DIGEST),
            )

        for seq, kind in enumerate(PAYLOADS, start=2):
            insert_kind(seq, kind)
        additive_kinds = ("turn_policy_recorded", "future_record", "turn__record", "future_" + "a" * 57)
        for seq, kind in enumerate(additive_kinds, start=20):
            insert_kind(seq, kind)

        for seq, kind in enumerate(("", "Turn_record", "turn-record", "_future_record", "a" * 65), start=30):
            with pytest.raises(psycopg.errors.CheckViolation):
                insert_kind(seq, kind)

    valid = make_record(
        "session_created",
        session_id="s",
        payload={
            "principal": principal(),
            "workspace": content_ref("workspace"),
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": content_ref("attributes"),
        },
    )
    future = valid.to_dict()
    future["kind"] = "future_record"
    with pytest.raises(RecordError):
        Record.parse(canonical_json_bytes(future))
    with pytest.raises(RecordError, match="unknown record kind"):
        make_record("future_record", session_id="s", payload={})


@pytest.mark.postgres
def test_postgres_inbox_kind_check_is_additive_but_codec_remains_closed(pg_database):
    import psycopg

    assert len(INPUT_PAYLOADS) == 9
    _install(pg_database)
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        connection.execute(
            "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
            "VALUES ('tenant-a','s','S1','idle',1,'live',0)"
        )

        def insert_kind(input_seq, kind):
            connection.execute(
                "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,available_ms,received_ms) "
                "VALUES ('tenant-a','s',%s,%s,%s,%s,%s,0,0)",
                (f"i{input_seq}", input_seq, kind, b"{}", ZERO_DIGEST),
            )

        for input_seq, kind in enumerate(INPUT_PAYLOADS, start=2):
            insert_kind(input_seq, kind)
        additive_kinds = ("future_input", "future__input", "future_" + "a" * 57)
        for input_seq, kind in enumerate(additive_kinds, start=20):
            insert_kind(input_seq, kind)

        for input_seq, kind in enumerate(("", "Future_input", "future-input", "_future_input", "a" * 65), start=30):
            with pytest.raises(psycopg.errors.CheckViolation):
                insert_kind(input_seq, kind)

    valid = InboxItem(
        input_id="current",
        kind="user",
        payload={"content": content_ref("request")},
        actor=principal(),
    )
    future = valid.to_dict()
    future["kind"] = "future_input"
    with pytest.raises(RecordError):
        InboxItem.parse(canonical_json_bytes(future))


@pytest.mark.postgres
def test_postgres_tenant_worker_gateway_bindings_and_heartbeats_are_preserved(pg_database):
    import psycopg

    provisioned = _install_runtime_roles(pg_database)
    tenant_dsn, tenant_installation = provisioned["tenant"]
    worker_dsn, worker_installation = provisioned["worker"]
    gateway_dsn, gateway_installation = provisioned["shared-gateway"]
    assert tenant_installation.credential_isolation is True
    assert worker_installation.credential_isolation is False
    assert gateway_installation.credential_isolation is False

    with PostgresStore.tenant_standalone(tenant_dsn, tenant_id=TENANT_ID, fold_factory=_fold_factory) as tenant:
        assert tenant.runtime_mode == "tenant"
        assert tenant.runtime_binding.credential_isolation is True
        assert tenant.credential_isolation is True
        assert tenant.credential_tenant_id == TENANT_ID
        with tenant.heartbeat_store() as heartbeat:
            assert heartbeat.runtime_mode == "tenant"
            assert heartbeat.runtime_binding.tenant_id == TENANT_ID
    with (
        pytest.raises(ValueError, match="credential-bound"),
        PostgresStore.tenant_standalone(tenant_dsn, tenant_id="other-tenant", fold_factory=_fold_factory),
    ):
        pass
    with (
        psycopg.connect(worker_dsn, autocommit=True, connect_timeout=3) as worker,
        pytest.raises(ValueError, match="credential-bound"),
    ):
        PostgresStore.verify_tenant_connection(worker, TENANT_ID)

    with psycopg.connect(tenant_dsn, autocommit=True, connect_timeout=3) as connection:
        tenant = SessionPostgresStore(connection, tenant_id=TENANT_ID)
        assert tenant.credential_isolation is True
        with tenant.heartbeat_store() as heartbeat:
            assert heartbeat.credential_isolation is True
            assert heartbeat.runtime_mode == "tenant"

    for dsn, mode, installation, core_opener in (
        (worker_dsn, "worker", worker_installation, PostgresStore.worker_standalone),
        (gateway_dsn, "shared-gateway", gateway_installation, PostgresStore.shared_gateway_standalone),
    ):
        with core_opener(dsn, fold_factory=_fold_factory) as store:
            assert installation.credential_isolation is False
            assert store.runtime_binding.credential_isolation is False
            assert store.runtime_mode == mode
            with store.heartbeat_store() as heartbeat:
                assert heartbeat.runtime_binding.credential_isolation is False
                assert heartbeat.runtime_mode == mode
        with SessionPostgresStore.standalone(dsn) as store:
            assert store.credential_isolation is False
            assert store.runtime_mode == mode
            with store.heartbeat_store() as heartbeat:
                assert heartbeat.credential_isolation is False
                assert heartbeat.runtime_mode == mode


@pytest.mark.postgres
def test_postgres_schema_rejects_rls_policy_drift(pg_database):
    import psycopg

    _install_runtime_roles(pg_database)
    with psycopg.connect(pg_database.dsn, autocommit=True, connect_timeout=3) as connection:
        store = PostgresStore(connection, fold_factory=_fold_factory)
        store.install_schema()
        connection.execute("CREATE POLICY rogue_allow_all ON sk_session USING (true) WITH CHECK (true)")
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            store.install_schema()
