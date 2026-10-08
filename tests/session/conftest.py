import os
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from time import monotonic
from uuid import uuid4

import pytest

from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore

Database = str | Path | SQLiteStore


def open_store(database: Database):
    if isinstance(database, SQLiteStore):
        # In-memory heartbeat and control threads share the owner's locked connection.
        return nullcontext(database)
    if isinstance(database, Path):
        return SQLiteStore.standalone(str(database))
    return PostgresStore.standalone(database)


def pytest_generate_tests(metafunc):
    if "store_backend" not in metafunc.fixturenames:
        return
    if metafunc.definition.get_closest_marker("postgres"):
        backends = ["postgres"]
    elif metafunc.definition.get_closest_marker("sqlite_file"):
        backends = ["sqlite"]
    elif metafunc.definition.get_closest_marker("persistent_store"):
        backends = ["postgres", "sqlite"]
    else:
        backends = ["postgres", "sqlite", "sqlite_memory"]
    metafunc.parametrize("store_backend", backends)


@pytest.fixture(scope="session")
def postgres_admin_dsn():
    psycopg = pytest.importorskip("psycopg", reason="PostgreSQL tests need the postgres extra; central CI runs them")
    configured = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    for dsn in ([configured] if configured else []) + ["dbname=postgres"]:
        try:
            with psycopg.connect(dsn, autocommit=True, connect_timeout=3):
                return dsn
        except psycopg.OperationalError:
            continue
    pytest.skip("No PostgreSQL via VV_AGENT_TEST_POSTGRES_DSN or local unix socket; central CI runs these tests")


@pytest.fixture
def database(store_backend, tmp_path: Path, request) -> Iterator[Database]:
    """Only store tests allocate databases; each PostgreSQL test owns a disposable database."""
    if store_backend == "sqlite":
        yield tmp_path / "session.sqlite"
        return
    if store_backend == "sqlite_memory":
        with SQLiteStore.standalone(":memory:") as memory:
            yield memory
        return
    dsn = request.getfixturevalue("postgres_admin_dsn")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = f"vvsk_test_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(dsn, dbname=name)
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


@pytest.fixture
def store(database: Database):
    with open_store(database) as store:
        store.install_schema()
        yield store
