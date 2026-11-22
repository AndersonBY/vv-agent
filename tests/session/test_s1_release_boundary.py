import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem

from .conftest import open_store
from .helpers import create_fixture_session, push_fixture_input, wrap_fixture_store

PRE_S1_READER = """
import json
import sys
from importlib.metadata import version

from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem, RecordError, SessionSpec
from vv_agent.session.sqlite import SQLiteStore

assert version('vv-agent') == '0.36.0'
kind, database, operation = json.load(sys.stdin)
factory = SQLiteStore if kind == 'sqlite' else PostgresStore
stage = 'open'
try:
    with factory.standalone(database) as store:
        stage = 'operation'
        if operation == 'seed':
            store.install_schema()
            with store.atomic() as tx:
                tx.create(SessionSpec('s', 'test', '/tmp'), consumers=('events',))
                tx.push('s', InboxItem('initial', 'user', {'content': 'go'}))
        if operation in {'seed', 'history'}:
            state, records, inbox_seq = store.read_state('s')
            pending = store.peek_inbox('s', ready_only=False)
            assert records[0].record.kind == 'session_created'
            assert inbox_seq == 1 and [row.item.input_id for row in pending] == ['initial']
            print(json.dumps({
                'phase': state.phase,
                'history': [[r.seq, r.commit_id, r.writer_epoch, r.created_ms, r.record.encode().hex()] for r in records],
                'inbox': [[r.input_seq, r.received_ms, r.consumed_seq, r.item.encode().hex()] for r in pending],
            }))
        elif operation == 'read':
            store.read_state('s')
        elif operation == 'create':
            with store.atomic() as tx:
                tx.create(SessionSpec('old-created', 'test', '/tmp'), consumers=('events',))
        elif operation == 'push':
            with store.atomic() as tx:
                tx.push('s', InboxItem('old-input', 'user', {'content': 'stale'}))
        elif operation == 'acquire':
            store.acquire('s', owner='old-worker', ttl_ms=60_000)
        else:
            assert operation == 'open', operation
except RecordError:
    print(json.dumps([stage, 'RecordError']))
except ValueError as error:
    if str(error) != 'unsupported session SQLite schema version':
        raise
    print(json.dumps([stage, 'unsupported SQLite schema']))
except Exception as error:
    from psycopg import Error

    if not isinstance(error, Error):
        raise
    if stage == 'open' and 'permission denied for database' in str(error):
        print(json.dumps([stage, 'CONNECT denied']))
    else:
        print(json.dumps([stage, error.sqlstate or type(error).__name__]))
else:
    if operation not in {'seed', 'history'}:
        print(json.dumps([stage, 'success']))
"""


@pytest.fixture
def released_reader():
    python = os.environ.get("VV_AGENT_0360_PYTHON")
    if not python:
        pytest.skip("Set VV_AGENT_0360_PYTHON to an isolated PyPI 0.36.0 interpreter")

    def run(database: str | Path, operation: str):
        backend = "sqlite" if isinstance(database, Path) else "postgres"
        result = subprocess.run(
            [python, "-I", "-c", PRE_S1_READER],
            input=json.dumps([backend, str(database), operation]),
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"0.36.0 {operation} subprocess failed"
        return json.loads(result.stdout)

    return run


@pytest.mark.persistent_store
def test_s1_refusal_preserves_released_history_and_inbox(database, released_reader):
    before = released_reader(database, "seed")
    with (
        pytest.raises(ValueError, match=r"unsupported (session SQLite|PostgreSQL session) schema version"),
        open_store(database),
    ):
        pytest.fail("S1 opened a store created by 0.36.0")
    assert released_reader(database, "history") == before


def _s1_snapshot(store):
    return (
        store.list_sessions(),
        store.read("s"),
        store.peek_inbox("s", ready_only=False),
        store._rows("SELECT lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE session_id='s'"),
    )


@pytest.mark.persistent_store
def test_released_pre_s1_cannot_read_or_publish_s1_records(database, released_reader):
    with open_store(database) as bare_store:
        bare_store.install_schema()
        store = wrap_fixture_store(bare_store)
        create_fixture_session(store, "s")
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
        before = _s1_snapshot(store)

    if isinstance(database, Path):
        expected = {op: ["open", "unsupported SQLite schema"] for op in ("open", "read", "create", "push", "acquire")}
    else:
        expected = {
            "open": ["operation", "success"],
            "read": ["operation", "RecordError"],
            "create": ["operation", "42P10"],
            "push": ["operation", "22P02"],
        }
    for operation, outcome in expected.items():
        assert released_reader(database, operation) == outcome
        with open_store(database) as reader:
            assert _s1_snapshot(reader) == before


@pytest.mark.postgres
def test_s1_cutover_fences_released_postgres_credentials(database, postgres_admin_dsn, released_reader):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    old_database, old_role, new_role = (f"cutover_{uuid4().hex}" for _ in range(3))
    old_password, new_password = uuid4().hex, uuid4().hex
    current_database = conninfo_to_dict(database)["dbname"]
    assert isinstance(current_database, str)
    old_admin_dsn = make_conninfo(database, dbname=old_database)
    old_dsn = make_conninfo(database, dbname=old_database, user=old_role, password=old_password)
    stale_dsn = make_conninfo(database, user=old_role, password=old_password)
    current_dsn = make_conninfo(database, user=new_role, password=new_password)
    with psycopg.connect(postgres_admin_dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOINHERIT PASSWORD {}").format(sql.Identifier(old_role), sql.Literal(old_password))
        )
        try:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(old_database)))
            for name in (old_database, current_database):
                admin.execute(sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(sql.Identifier(name)))
            admin.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(old_database), sql.Identifier(old_role))
            )
            historical = released_reader(old_admin_dsn, "seed")
            with psycopg.connect(old_admin_dsn, autocommit=True) as owner:
                owner.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(old_role)))
                assert owner.execute(
                    "SELECT has_table_privilege(%s,'sk_session','SELECT'),"
                    "has_table_privilege(%s,'sk_session','INSERT, UPDATE, DELETE')",
                    (old_role, old_role),
                ).fetchone() == (True, False)
            with PostgresStore.standalone(database) as owner:
                owner.install_schema()
                owner.install_tenant_rls("default", new_role, new_password)
                assert owner.connection.execute(
                    "SELECT has_table_privilege(%s,'sk_session','INSERT, UPDATE, DELETE'),"
                    "has_table_privilege(%s,'sk_record','INSERT'),has_table_privilege(%s,'sk_inbox','INSERT')",
                    (old_role, old_role, old_role),
                ).fetchone() == (False, False, False)
            admin.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(current_database), sql.Identifier(new_role))
            )
            assert admin.execute(
                "SELECT has_database_privilege(%s,%s,'CONNECT'),has_database_privilege(%s,%s,'CONNECT')",
                (new_role, old_database, old_role, current_database),
            ).fetchone() == (False, False)

            with PostgresStore.tenant_standalone(current_dsn, tenant_id="default") as tenant:
                store = wrap_fixture_store(tenant)
                create_fixture_session(store, "s")
                push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
                before = _s1_snapshot(store)

            for operation in ("open", "read", "create", "push", "acquire"):
                assert released_reader(stale_dsn, operation) == ["open", "CONNECT denied"]
            with PostgresStore.tenant_standalone(current_dsn, tenant_id="default") as tenant:
                tenant.configure_input_admission(allow_unsigned=True)
                assert _s1_snapshot(tenant) == before
                lease = tenant.acquire("s", owner="s1-worker", ttl_ms=60_000)
                assert lease is not None
                assert tenant.release(lease)
            assert released_reader(old_dsn, "history") == historical
        finally:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(old_database)))
            if admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (new_role,)).fetchone():
                with psycopg.connect(database, autocommit=True) as owner:
                    owner.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(new_role)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}, {}").format(sql.Identifier(old_role), sql.Identifier(new_role)))
