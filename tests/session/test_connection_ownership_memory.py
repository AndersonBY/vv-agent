"""The one memory C is serialized; committed facades cannot see a local outer tx."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vv_agent.session.kernel import drive
from vv_agent.session.ownership import OwnershipError, checkout
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .test_recovery_matrix import runtime, start


@pytest.mark.parametrize("outcome", ["commit", "rollback", "commitfailure"])
def test_memory_serialization(outcome, monkeypatch):
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        kernel = SessionDriver(store=store)
        wakes = []
        entered = Event()
        finished = Event()

        def observer():
            entered.set()
            count = store._one("SELECT count(*) FROM host_local")[0]
            finished.set()
            return count

        store._rows("CREATE TABLE host_local (id integer PRIMARY KEY)")
        store._rows("CREATE TABLE ref (id integer PRIMARY KEY)")
        store._rows("CREATE TABLE deferred (id integer REFERENCES ref(id) DEFERRABLE INITIALLY DEFERRED)")

        class Rollback(Exception):
            pass

        import sqlite3

        failure = Rollback if outcome == "rollback" else sqlite3.IntegrityError
        with ThreadPoolExecutor(1) as pool:
            try:
                with store.atomic() as tx:
                    tx.connection.execute("INSERT INTO host_local VALUES (1)")
                    if outcome == "commitfailure":
                        tx.connection.execute("INSERT INTO deferred VALUES (99)")
                    # Controlled local reads are legal.
                    assert store._one("SELECT count(*) FROM host_local")[0] == 1
                    assert store._defer_memory_wake("s", lambda: wakes.append("s"))
                    assert tx.touched == {"s"} and wakes == []
                    rows = store._rows
                    sql = []

                    def counted(*a, **kw):
                        sql.append(a[0])
                        return rows(*a, **kw)

                    monkeypatch.setattr(store, "_rows", counted)
                    for operation in [
                        lambda: drive(store, "s", runtime=runtime(store, [])),
                        lambda: kernel.messages("s"),
                        lambda: kernel.waits("s"),
                    ]:
                        with pytest.raises(OwnershipError) as caught:
                            operation()
                        assert caught.value.code == "committed_read_in_transaction"
                    assert sql == []
                    future = pool.submit(observer)
                    assert entered.wait(2)
                    assert not finished.is_set()
                    if outcome == "rollback":
                        raise Rollback()
            except failure:
                assert outcome != "commit"
            assert future.result(5) == (1 if outcome == "commit" else 0)
            assert wakes == ["s"]
        with checkout(store) as borrowed:
            assert borrowed.connection is store.connection
        assert store._one("SELECT 1")[0] == 1


def test_committed_facades_reject_before_native_sql(monkeypatch):
    from vv_agent.app_server import AppServer
    from vv_agent.session.events import SessionRunEventStore
    from vv_agent.session.tracing import deliver_spans

    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        server = AppServer(store=store)
        try:
            with store.atomic() as tx:
                sql = []
                store.connection.set_trace_callback(sql.append)
                adapter = server.run_adapter
                operations = [
                    lambda: adapter.public_thread_status("s"),
                    lambda: adapter.read_context("s"),
                    lambda: adapter.recover(None, "s"),
                    lambda: adapter._publish_turns("s"),
                    lambda: adapter.subscribe("s", "synthetic"),
                    lambda: list(SessionRunEventStore(store, "s").replay(run_id="s/turn/initial")),
                    lambda: SessionRunEventStore(store, "s").consume(lambda event: None),
                    lambda: deliver_spans(store, "s", []),
                ]
                for operation in operations:
                    with pytest.raises(OwnershipError) as caught:
                        operation()
                    assert caught.value.code == "committed_read_in_transaction"
                adapter.wake("s")
                assert tx.touched == {"s"}
                assert sql == []
                store.connection.set_trace_callback(None)
                adapter._notification_stop.set()
        finally:
            server.run_adapter.close()
            server.run_adapter.join()
            server.kernel.close()
