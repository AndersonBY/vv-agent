"""E/P preserve business SQL timeouts; HB's existing bounded limits stay independent."""

from contextlib import contextmanager
from threading import current_thread

import pytest

from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import drive
from vv_agent.session.ownership import checkout
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import SessionSpec

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.postgres
@pytest.mark.parametrize("configured", [False, True])
def test_business_timeout_independent_of_heartbeat(store, database, monkeypatch, configured):
    start(store)
    renew = PostgresStore.renew
    heartbeat = []
    business = []

    def observed(candidate, *a, **kw):
        if current_thread().name == "session-heartbeat":
            heartbeat.append(candidate._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')"))
        return renew(candidate, *a, **kw)

    monkeypatch.setattr(PostgresStore, "renew", observed)
    factory = store.execution_store

    @contextmanager
    def explicit():
        with factory() as candidate:
            candidate.connection.execute("SET statement_timeout='2200ms'")
            yield candidate

    if configured:
        monkeypatch.setattr(store, "execution_store", explicit)

    def admit(plan, *, tx):
        business.append(tx.store._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')"))
        tx.connection.execute("SELECT pg_sleep(1.2)")
        return ChildSession(SessionSpec("timeout-child", "test", "/tmp"), "go")

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    assert drive(store, "s", runtime=rt)
    assert business == [("0", "2200ms" if configured else "0")]
    assert len(heartbeat) >= 2
    assert all(lock == "250ms" and statement in {"1s", "1000ms"} for lock, statement in heartbeat)
    monkeypatch.setattr(store, "execution_store", factory)
    with checkout(store, role="projection") as reader:
        assert reader._one("SELECT current_setting('statement_timeout')")[0] == "0"
        reader.connection.execute("SELECT pg_sleep(1.2)")
    print("CO2-TIME01", configured, business, "HB_renew", len(heartbeat), flush=True)
