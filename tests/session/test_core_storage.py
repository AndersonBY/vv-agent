"""Core PostgreSQL connection ownership uses the real adapter and existing SQL rules."""

from dataclasses import replace

import pytest

from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._records import SessionSpec
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import LeaseLost
from vv_agent.session.reducer import Fold


@pytest.mark.postgres
def test_core_postgres_preserves_host_connection_transaction_and_heartbeat(store):
    core = CorePostgresStore(store.connection, fold_factory=Fold)
    assert core.connection is store.connection
    core._rows("CREATE TABLE host_write (id integer PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="rollback"), store.connection.transaction():
        with core.atomic() as tx:
            assert type(tx) is SQLSessionTx
            tx.create(SessionSpec("rolled-back", "host", "workspace"), consumers=())
            core._rows("INSERT INTO host_write VALUES (1)")
        raise RuntimeError("rollback")
    assert core.list_sessions() == () and core._rows("SELECT * FROM host_write") == []
    with core.atomic() as tx:
        tx.create(SessionSpec("s", "host", "workspace"), consumers=())
    lease = core.acquire("s", owner="first", ttl_ms=15000)
    assert lease is not None
    before = core._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")
    with core.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="empty", records=())
    assert core._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')") == before
    with core.heartbeat_store() as heartbeat:
        assert type(heartbeat) is CorePostgresStore and heartbeat.fold_factory is Fold
        assert heartbeat.connection is not core.connection
        assert heartbeat._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')") == ("250ms", "1s")
        poll = heartbeat.renew(lease, ttl_ms=15000)
        assert poll.lease.epoch == lease.epoch and poll.db_now_ms > 0
        with pytest.raises(LeaseLost), heartbeat.atomic() as tx:
            tx.append("s", lease=replace(lease, epoch=lease.epoch + 1), expected_seq=1, commit_id="forged", records=())
    assert not core.connection.closed


def test_raw_core_sqlite_keeps_sdk_host_identity_latches_and_trace_boundary(tmp_path, monkeypatch):
    from vv_agent.kernel._sqlite import SQLiteStore
    from vv_agent.session.kernel import drive
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.session.timing import SystemClock
    from vv_agent.session.tracing import deliver_spans
    from vv_agent.types import LLMResponse

    from .test_recovery_matrix import runtime, start

    class NoHostWallClock(SystemClock):
        def now_ms(self):
            raise AssertionError("SQL-backed scope must receive actual DB now")

    path = tmp_path / "core-drive.sqlite"
    with SQLiteStore.standalone(str(path), fold_factory=Fold, clock=SystemClock()) as core:
        surface = SessionDriver(store=core)
        assert surface.store is core and surface.path == str(path)
        assert core._one("PRAGMA user_version")[0] == 3
        start(core)
        rt = runtime(path, [LLMResponse("done")])
        rt.clock = NoHostWallClock()
        rt.heartbeat_store = core.heartbeat_store
        guarded = []
        defer = core._defer_idle_drive

        def capture(*args, **kwargs):
            assert kwargs.get("lease_guard") is not None
            guarded.append(True)
            return defer(*args, **kwargs)

        monkeypatch.setattr(core, "_defer_idle_drive", capture)
        assert drive(core, "s", runtime=rt) and guarded
        with core.atomic(), pytest.raises(ValueError, match="top-level transaction"):
            deliver_spans(core, "s", [])
        surface.close()
        assert core._one("SELECT 1") == (1,), "host-owned connection must remain open"


@pytest.mark.postgres
def test_raw_core_postgres_factory_gets_driver_heartbeat_timeouts(store, monkeypatch):
    from contextlib import nullcontext

    from vv_agent.session._driver_scope import _LeaseHeartbeat

    core = CorePostgresStore(store.connection, fold_factory=Fold)
    with core.atomic() as tx:
        tx.create(SessionSpec("s", "host", "workspace"), consumers=())
    lease = core.acquire("s", owner="first", ttl_ms=15000)
    assert lease is not None
    core._rows("SELECT set_config('lock_timeout','0',false),set_config('statement_timeout','0',false)")
    hb = _LeaseHeartbeat(lease, lambda: nullcontext(core), 15000, 0.25, now_ms=core._now())
    renew = core.renew

    def stop_after_real_renew(*args, **kwargs):
        receipt = renew(*args, **kwargs)
        hb.request_stop()
        return receipt

    monkeypatch.setattr(core, "renew", stop_after_real_renew)
    hb._heartbeat()  # Actual worker body, synchronously; factory bypasses heartbeat_store options.
    assert not hb.lost and hb.heartbeat_error is None and hb.ready.is_set()
    assert core._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')") == ("250ms", "1s")


@pytest.mark.postgres
def test_real_postgres_core_import_precedes_lazy_driver_and_never_loads_sdk(database, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = r"""
import importlib.abc
import os
import sys
sys.path.insert(0, sys.argv[1])
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith('django'), fullname
        assert (not fullname.startswith('vv_agent.') or fullname == 'vv_agent.kernel'
                or fullname.startswith('vv_agent.kernel.')), fullname
sys.meta_path.insert(0, NoSDK())
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import SessionSpec
assert not any(n.split('.')[0] in {'psycopg','django'} for n in sys.modules)
def unused(*a, **kw): raise AssertionError('no compaction in session creation')
def factory(): return Fold(normalize_memory_settings=unused, count_messages=unused)
with PostgresStore.standalone(os.environ['VV_CORE_STORAGE_TEST_DSN'], fold_factory=factory) as store:
    store.install_schema()
    with store.atomic() as tx:
        tx.create(SessionSpec('core-import','test','workspace'), consumers=())
    lease = store.acquire('core-import', owner='owner', ttl_ms=15000)
    assert lease is not None
    with store.heartbeat_store() as heartbeat:
        assert heartbeat.renew(lease, ttl_ms=15000).db_now_ms > 0
        assert heartbeat.read_state('core-import')[0].session_id == 'core-import'
        assert type(heartbeat._fold_cache.reducer) is Fold
assert 'psycopg' in sys.modules
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(Path(__file__).resolve().parents[2] / "src")],
        cwd=tmp_path,
        env=os.environ | {"VV_CORE_STORAGE_TEST_DSN": database},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).replace(database, "<test-dsn>")
