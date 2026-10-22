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


@pytest.mark.parametrize("owner", ["standalone", "driver"])
def test_close_waits_for_inflight_read_in_another_thread(owner):
    import subprocess
    import sys
    import textwrap

    # Isolate the native SQLite shutdown race so a regression cannot kill pytest.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            from threading import Event, Thread
            from vv_agent.session.sqlite import SQLiteStore
            from vv_agent.session.surfaces import SessionDriver

            if sys.argv[1] == "standalone":
                manager = SQLiteStore.standalone(":memory:")
                store = manager.__enter__()
                close_owner = lambda: manager.__exit__(None, None, None)
            else:
                driver = SessionDriver()
                store = driver.store
                close_owner = driver.close
            entered, release, closing, closed = (Event() for _ in range(4))
            rows, errors = [], []
            def slow():
                entered.set()
                assert release.wait(3)
                return 7
            store.connection.create_function("slow", 0, slow)
            def read():
                try:
                    rows.extend(store._rows("SELECT slow()"))
                except BaseException as error:
                    errors.append(error)
            def close():
                closing.set()
                close_owner()
                closed.set()
            reader = Thread(target=read)
            closer = Thread(target=close)
            reader.start()
            assert entered.wait(2)
            closer.start()
            assert closing.wait(2)
            assert not closed.wait(0.05)
            release.set()
            reader.join(3)
            closer.join(3)
            assert not reader.is_alive() and not closer.is_alive()
            assert rows == [(7,)] and not errors
            assert closed.is_set()
        """),
            owner,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
