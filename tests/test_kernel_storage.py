"""Concrete storage can execute actual core SQL/Fold rules without importing the SDK."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_real_sqlite_storage_imports_and_runs_without_sdk(tmp_path, flags):
    script = r"""
import importlib.abc
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import SessionSpec, InboxItem, make_record
from vv_agent.kernel._store import LeaseLost
from vv_agent.kernel._sql import SQLSessionTx
assert not any(n.split('.')[0] in {'psycopg','django'} for n in sys.modules)

class Clock:
    ms = 1000
    def now_ms(self): return self.ms
    def monotonic(self): return self.ms / 1000
    def monotonic_ns(self): return self.ms * 1000000
    def sleep(self, seconds): self.ms += int(seconds * 1000)
clock = Clock()
def unused(*args, **kwargs):
    raise AssertionError('these session/usage records do not request compaction effects')
def factory():
    return Fold(normalize_memory_settings=unused, count_messages=unused)

class Crash(BaseException): pass
path = sys.argv[2]
with SQLiteStore.standalone(path, clock=clock, fold_factory=factory) as store:
    store.install_schema()
    assert store.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert store.connection.execute('PRAGMA synchronous').fetchone()[0] == 2
    with store.atomic() as tx:
        assert type(tx) is SQLSessionTx
        tx.create(SessionSpec('s','test','workspace'), consumers=())
        first = tx.push('s', InboxItem('i','user',{'content':'go'}))
        assert tx.push('s', InboxItem('i','user',{'content':'go'})).replayed
    lease = store.acquire('s', owner='first', ttl_ms=1000)
    r = make_record('usage_observed', session_id='s', payload={
        'meter_id':'m','observation':1,'mode':'cumulative','usage':{},'source':'host'})
    with store.atomic() as tx:
        receipt = tx.append('s',lease=lease,expected_seq=1,commit_id='usage',records=(r,))
    with store.atomic() as tx:
        replay = tx.append('s',lease=lease,expected_seq=1,commit_id='usage',records=(r,))
        assert replay.replayed and replay.head_seq == receipt.head_seq == 2
    with store.atomic() as tx:
        tx.create(SessionSpec('outer','test','workspace'), consumers=())
        try:
            with store.atomic() as nested:
                nested.create(SessionSpec('rolled-back','test','workspace'), consumers=())
                raise Crash()
        except Crash: pass
    assert store.list_sessions() == ('outer','s')
    with store.heartbeat_store() as heartbeat:
        assert type(heartbeat) is SQLiteStore and heartbeat is not store
        assert heartbeat.fold_factory is factory and heartbeat.clock is clock
        assert heartbeat.read('s').head_seq == 2
        heartbeat.read_state('s')
        assert type(heartbeat._fold_cache.reducer) is Fold
        assert heartbeat.renew(lease, ttl_ms=1000).db_now_ms == clock.ms
    clock.ms += 1001
    replacement = store.acquire('s', owner='second', ttl_ms=1000)
    assert replacement.epoch == lease.epoch + 1
    try:
        with store.atomic() as tx:
            tx.append('s',lease=lease,expected_seq=2,commit_id='stale',records=())
    except LeaseLost: pass
    else: raise AssertionError('old lease admitted')
    assert store.release(replacement)
with SQLiteStore.standalone(path, clock=clock, fold_factory=factory) as cold:
    assert cold.read('s').head_seq == 2 and cold.fold_factory is factory
with SQLiteStore.standalone(':memory:', clock=clock, fold_factory=factory) as memory:
    with memory.heartbeat_store() as heartbeat:
        assert heartbeat is memory
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(tmp_path / "core.sqlite")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sdk_storage_defaults_and_actual_sql_inheritance(tmp_path):
    from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
    from vv_agent.kernel._sql import SQLSessionTx as CoreTx
    from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
    from vv_agent.session.postgres import PostgresStore
    from vv_agent.session.reducer import Fold
    from vv_agent.session.sql import SQLSessionTx, SQLStore
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.session.timing import SystemClock

    assert issubclass(PostgresStore, CorePostgresStore) and issubclass(PostgresStore, SQLStore)
    assert SQLSessionTx is CoreTx
    with SQLiteStore.standalone(str(tmp_path / "sdk.sqlite")) as store:
        assert isinstance(store, CoreSQLiteStore) and isinstance(store, SQLStore)
        assert store.fold_factory is Fold and type(store.clock) is SystemClock
        store.install_schema()
        with store.heartbeat_store() as heartbeat:
            assert type(heartbeat) is SQLiteStore
            assert heartbeat.fold_factory is Fold and heartbeat.clock is store.clock
    assert SQLiteStore._transaction is CoreSQLiteStore._transaction
    assert PostgresStore._transaction is CorePostgresStore._transaction
    assert PostgresStore._bound_append is CorePostgresStore._bound_append
