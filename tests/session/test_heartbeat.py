"""Heartbeat connections retain credentials and connection failures retain their cause."""

from contextlib import ExitStack, contextmanager
from threading import Event
from unittest.mock import Mock
from uuid import uuid4

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, Runner
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, _Scope
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.runtime import static_worker_supports
from vv_agent.session.store import Lease, LeaseLost, LeaseRetryExhausted
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .test_runner_parity import RESOLVED


@pytest.mark.parametrize("password", ["", "test 'password\\value"])
def test_connection_conninfo_preserves_password(password, monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict

    from vv_agent.session.postgres import _connection_conninfo

    monkeypatch.setattr(psycopg, "connect", Mock(side_effect=AssertionError("unit test must not connect")))
    connection = Mock()
    connection.info.dsn = "host=127.0.0.1 port=55483 dbname=isolated user=test"
    connection.info.password = password
    parameters = conninfo_to_dict(_connection_conninfo(connection))
    assert parameters == conninfo_to_dict(connection.info.dsn) | ({"password": password} if password else {})
    psycopg.connect.assert_not_called()


@pytest.mark.postgres
def test_real_connection_conninfo_preserves_password(password_postgres_dsn):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    from vv_agent.session.postgres import _connection_conninfo

    with psycopg.connect(password_postgres_dsn, connect_timeout=3) as connection:
        assert conninfo_to_dict(_connection_conninfo(connection)) == conninfo_to_dict(connection.info.dsn) | (
            {"password": connection.info.password} if connection.info.password else {}
        )


def test_heartbeat_open_failure_reaches_public_caller(monkeypatch, tmp_path):
    error = ConnectionError("heartbeat connection refused")
    driver = SessionDriver()
    load_state = _Driver.load_state
    build_runtime = driver._build_runtime_for_claim

    @contextmanager
    def unavailable():
        raise error
        yield

    def after_heartbeat_failure(self):
        assert self.scope.ready.wait(5)
        assert self.scope.lost
        assert not self.scope.lease_keeper.is_alive(self.scope)
        load_state(self)

    def runtime_with_unavailable_heartbeat(store, sid):
        runtime = build_runtime(store, sid)
        runtime.heartbeat_store = unavailable
        return runtime

    monkeypatch.setattr(_Driver, "load_state", after_heartbeat_failure)
    monkeypatch.setattr(driver, "_build_runtime_for_claim", runtime_with_unavailable_heartbeat)
    monkeypatch.setattr("vv_agent.session.surfaces.SessionDriver", lambda: driver)
    config = RunConfig(model_provider=FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), RESOLVED), workspace=tmp_path)
    try:
        with pytest.raises(LeaseRetryExhausted) as caught:
            Runner.run_sync(Agent("heartbeat", "Answer."), "go", run_config=config)
        assert isinstance(caught.value.__cause__, LeaseLost)
        assert caught.value.__cause__.__cause__ is error
        runtime = driver.runtime(Agent("heartbeat", "Answer."), config)
        runtime.heartbeat_store = unavailable
        scope = _Scope(Lease("s", "owner", 1, 0), runtime)
        scope.start()
        try:
            assert scope.ready.wait(5)
            assert scope.lost
            with pytest.raises(LeaseLost) as lost, scope.committing():
                pass
            assert lost.value.__cause__ is error
        finally:
            scope.close()
        assert not scope.lease_keeper.is_alive(scope)
    finally:
        driver.close()


@pytest.fixture
def password_postgres_dsn(database):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    role, password = f"vvsk_test_login_{uuid4().hex}", uuid4().hex
    with psycopg.connect(database, autocommit=True) as admin:
        privileged = admin.execute("SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user").fetchone()
        if not (privileged and privileged[0]):
            pytest.skip("test PostgreSQL role lacks CREATEROLE for a temporary password login")
        admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
        admin.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(role), sql.Identifier(admin.info.user)))
        try:
            admin.execute(sql.SQL("GRANT CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            dsn = make_conninfo(
                database,
                host="127.0.0.1",
                hostaddr="127.0.0.1",
                port=admin.info.port,
                user=role,
                password=password,
            )
            try:
                with psycopg.connect(make_conninfo(dsn, password=uuid4().hex), connect_timeout=3):
                    pytest.skip("local pg_hba does not require password authentication over TCP")
            except psycopg.OperationalError as exc:
                if "pg_hba.conf" in str(exc):
                    pytest.skip("local pg_hba does not allow this password-authenticated TCP connection")
                if "password authentication failed" not in str(exc):
                    raise
            yield dsn
        finally:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.postgres
@pytest.mark.parametrize("caller_owned", [False, True])
def test_runner_heartbeat_over_password_authenticated_tcp(password_postgres_dsn, caller_owned, monkeypatch, tmp_path):
    import psycopg

    renewed = Event()
    renew = PostgresStore.renew
    with ExitStack() as stack:
        if caller_owned:
            connection = stack.enter_context(psycopg.connect(password_postgres_dsn, autocommit=True))
            store = PostgresStore(connection)
        else:
            store = stack.enter_context(PostgresStore.standalone(password_postgres_dsn))
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        assert store.connection.info.host == "127.0.0.1"
        assert store.connection.info.password

        def heartbeat_renew(self, lease, *, ttl_ms):
            result = renew(self, lease, ttl_ms=ttl_ms)
            if self.connection is not store.connection:
                renewed.set()
            return result

        def response(_request):
            assert renewed.wait(5), "password-authenticated heartbeat did not renew"
            return LLMResponse("done")

        agent = Agent("heartbeat", "Answer.")
        config = RunConfig(model_provider=FixedModelProvider(ScriptedLLM([response]), RESOLVED), workspace=tmp_path)
        supports = static_worker_supports(agent, config, resolved=None, extra_capabilities=())
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"h" * 32),
            worker_level=2,
            capabilities=supports,
        )
        monkeypatch.setattr(PostgresStore, "renew", heartbeat_renew)
        monkeypatch.setattr("vv_agent.session.surfaces.SessionDriver", lambda: driver)
        try:
            result = Runner.run_sync(agent, "go", run_config=config)
            assert result.final_output == "done"
            assert renewed.is_set()
        finally:
            driver.close()
        assert not store.connection.closed
