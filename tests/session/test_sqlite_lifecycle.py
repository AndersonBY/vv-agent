import sqlite3
from threading import Event, current_thread

import pytest

from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .test_recovery_matrix import records_of, runtime, start


def test_memory_heartbeat_borrows_owner_connection_and_owner_closes_it(monkeypatch):
    renewed = Event()
    heartbeat_threads = set()
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        renew = store.renew

        def observe_renew(lease, *, ttl_ms):
            result = renew(lease, ttl_ms=ttl_ms)
            if current_thread().name == "session-heartbeat":
                heartbeat_threads.add(current_thread())
                renewed.set()
            return result

        def answer(_request):
            assert renewed.wait(2)
            return LLMResponse("done")

        monkeypatch.setattr(store, "renew", observe_renew)
        rt = runtime(store, [answer], heartbeat_seconds=0.01)
        drive(store, "s", runtime=rt)
        assert renewed.is_set()
        assert records_of(store, "turn_ended")[0].payload["status"] == "completed"
        assert all(not thread.is_alive() for thread in heartbeat_threads)
        assert store.connection.execute("SELECT 1").fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store.connection.execute("SELECT 1")
