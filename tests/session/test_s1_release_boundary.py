import os
import subprocess
from pathlib import Path

import pytest

from vv_agent.session.records import InboxItem

from .conftest import open_store
from .helpers import create_fixture_session, push_fixture_input, wrap_fixture_store

PRE_S1_READER = """
import sys
from importlib.metadata import version

from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem, RecordError, SessionSpec
from vv_agent.session.sqlite import SQLiteStore

assert version('vv-agent') == '0.36.0'
kind, database, operation = sys.argv[1:]
factory = SQLiteStore if kind == 'sqlite' else PostgresStore
if operation == 'create':
    with factory.standalone(database) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec('s', 'test', '/tmp'), consumers=('events',))
            tx.push('s', InboxItem('initial', 'user', {'content': 'go'}))
        assert store.read_state('s')[1][0].record.kind == 'session_created'
else:
    try:
        with factory.standalone(database) as store:
            store.read_state('s')
    except RecordError:
        assert kind == 'postgres'
    except ValueError as error:
        assert kind == 'sqlite' and str(error) == 'unsupported session SQLite schema version'
    else:
        raise AssertionError('0.36.0 accepted an S1 session')
"""


@pytest.mark.persistent_store
@pytest.mark.parametrize("old_store", [True, False], ids=["s1-reads-p1", "p1-reads-s1"])
def test_released_pre_s1_and_s1_reject_each_others_stores(database, old_store):
    python = os.environ.get("VV_AGENT_0360_PYTHON")
    if not python:
        pytest.skip("Set VV_AGENT_0360_PYTHON to an isolated PyPI 0.36.0 interpreter")
    backend = "sqlite" if isinstance(database, Path) else "postgres"

    def run_old(operation):
        result = subprocess.run(
            [python, "-I", "-c", PRE_S1_READER, backend, str(database), operation],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    if old_store:
        run_old("create")
        with (
            pytest.raises(ValueError, match=r"unsupported (session SQLite|PostgreSQL session) schema version"),
            open_store(database),
        ):
            pytest.fail("S1 opened a store created by 0.36.0")
        return

    with open_store(database) as bare_store:
        bare_store.install_schema()
        store = wrap_fixture_store(bare_store)
        create_fixture_session(store, "s")
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
        before = store.read("s")
        pending = store.peek_inbox("s")
        assert before.records[0].record.kind == "session_created"
        assert pending[0].item.input_id == "initial"
        run_old("read")
        assert store.read("s") == before
        assert store.peek_inbox("s") == pending
