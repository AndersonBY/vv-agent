"""S1 content transactions retain the latest lease and SQL append budget."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock
from time import monotonic, sleep

import pytest

from vv_agent.kernel._sql import SQLStore
from vv_agent.session import kernel
from vv_agent.session.children import ChildSession, ChildTool
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import Record
from vv_agent.session.sql import SQLSessionTx
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import fixture_child_spec, fixture_content_reader, read_fixture_state, wrap_fixture_store
from .test_children import parent_runtime
from .test_recovery_matrix import runtime as base_runtime
from .test_recovery_matrix import start

pytestmark = pytest.mark.postgres


class HostPostgresStore(PostgresStore):
    """Exercise the caller-owned connection shape used by embedding hosts."""

    def __init__(self, raw):
        self.raw = raw
        self._connection_lock = RLock()

    @property
    def connection(self):
        return self.raw


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def wait_until(predicate, *, timeout=10):
    deadline = monotonic() + timeout
    while not predicate():
        assert monotonic() < deadline, "condition did not become true"
        sleep(0.01)


class PlannedCut(BaseException):
    pass


def test_content_tx_slow_encode_appends_with_renewed_lease_and_budget(store, database, monkeypatch):
    start(store)
    rt = base_runtime(
        database,
        [LLMResponse("done")],
        content_reader=fixture_content_reader(store, "s"),
        ttl_ms=4000,
        heartbeat_seconds=0.025,
    )
    rt.heartbeat_store = store.heartbeat_store
    entered, release = Event(), Event()
    encode = Record.encode
    renew = SQLStore.renew
    bound_append = PostgresStore._bound_append
    renewals = []
    budgets = []
    target_dbname: str | None = None
    target_owner: str | None = None

    def slow_encode(self):
        if self.kind == "turn_started" and not entered.is_set():
            entered.set()
            assert release.wait(10), "slow encode was not released"
        return encode(self)

    def observed_renew(self, lease, *, ttl_ms):
        poll = renew(self, lease, ttl_ms=ttl_ms)
        try:
            dbname = self.connection.info.dbname
        except AttributeError:
            return poll
        if (
            target_dbname is not None
            and target_owner is not None
            and dbname == target_dbname
            and lease.session_id == "s"
            and lease.owner == target_owner
        ):
            renewals.append(poll.lease)
        return poll

    @contextmanager
    def observed_bound(self, lease):
        with bound_append(self, lease):
            lock_ms, statement_ms = self._one(
                "SELECT "
                "EXTRACT(EPOCH FROM current_setting('lock_timeout')::interval) * 1000, "
                "EXTRACT(EPOCH FROM current_setting('statement_timeout')::interval) * 1000"
            )
            if entered.is_set() and not budgets:
                budgets.append((lease, float(lock_ms), float(statement_ms)))
            yield

    monkeypatch.setattr(Record, "encode", slow_encode)
    monkeypatch.setattr(SQLStore, "renew", observed_renew)
    monkeypatch.setattr(PostgresStore, "_bound_append", observed_bound)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(kernel.drive, store, "s", runtime=rt)
        assert entered.wait(10), "drive did not reach content transaction encoding"
        with open_store(database) as observer:
            target_dbname = observer.connection.info.dbname
            target_owner = observer._one(
                "SELECT lease_owner FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (observer.tenant_id, "s"),
            )[0]
        assert target_owner is not None
        wait_until(lambda: len({lease.expires_at_ms for lease in renewals}) >= 2)
        renewed_before_append = max(renewals, key=lambda lease: lease.expires_at_ms)
        release.set()
        assert future.result(timeout=20)

    assert renewals
    assert {lease.session_id for lease in renewals} == {"s"}
    assert {lease.owner for lease in renewals} == {target_owner}
    assert len(budgets) == 1
    appended_lease, lock_ms, statement_ms = budgets[0]
    assert appended_lease.expires_at_ms >= renewed_before_append.expires_at_ms
    assert (appended_lease.owner, appended_lease.epoch) == (renewed_before_append.owner, renewed_before_append.epoch)
    assert 0 < lock_ms <= rt.ttl_ms
    assert 0 < statement_ms <= rt.ttl_ms
    state, records, _ = read_fixture_state(store, "s")
    assert state.active_turn_id is None
    assert records[-1].record.kind == "turn_ended"
    assert records[-1].writer_epoch == appended_lease.epoch


def test_content_tx_statement_timeout_rolls_back_business_parent_and_child(store, database, monkeypatch):
    from psycopg.errors import QueryCanceled

    raw_host = HostPostgresStore(store.connection)
    host = wrap_fixture_store(raw_host)
    host._rows("CREATE TABLE business_child (id integer PRIMARY KEY)")
    start(host)
    rt = parent_runtime(host, database)

    def admit(_plan):
        for child_id in ("business-0", "business-1"):
            host._rows("INSERT INTO business_child VALUES (%s)", (int(child_id[-1]),))
        return tuple(ChildSession(fixture_child_spec(host, "s", child_id), "work") for child_id in ("business-0", "business-1"))

    rt.children["child_task"] = ChildTool(admit)

    def cut(point, record):
        if point == "after_commit" and record and record.kind == "op_planned" and record.payload["op_kind"] == "tool":
            raise PlannedCut()

    rt.hook = cut
    with pytest.raises(PlannedCut):
        kernel.drive(host, "s", runtime=rt)
    rt.hook = lambda *_: None
    rt.ttl_ms = 2500
    prefix = read_fixture_state(host, "s")[1]
    content_count = host._one("SELECT count(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", (host.tenant_id, "s"))[0]
    lease = host.acquire("s", owner="content-timeout", ttl_ms=2500)
    assert lease is not None
    scope = kernel._Scope(lease, rt, now_ms=host._now())
    driver = kernel._Driver(host, "s", rt, scope)
    driver.bind_turn()
    scope.poll(host)
    attempt = next(op.attempts[1] for op in driver.state.operations.values() if op.kind == "tool")
    commit = SQLSessionTx._commit
    commit_budgets = []

    def slow_commit(self, *args):
        if args[0] != "s":
            return commit(self, *args)
        budget = self.store._one(
            "SELECT "
            "EXTRACT(EPOCH FROM current_setting('lock_timeout')::interval) * 1000, "
            "EXTRACT(EPOCH FROM current_setting('statement_timeout')::interval) * 1000"
        )
        commit_budgets.append(budget)
        assert 0 < float(budget[1]) <= 2500, budget
        self.store._rows("SELECT pg_sleep(4)")
        return commit(self, *args)

    monkeypatch.setattr(SQLSessionTx, "_commit", slow_commit)
    try:
        with pytest.raises(QueryCanceled):
            driver.dispatch(attempt)
    finally:
        scope.close()
        host.release(scope.lease)

    assert commit_budgets and all(0 < float(statement_ms) <= 2500 for _, statement_ms in commit_budgets)
    assert host._rows("SELECT * FROM business_child") == []
    assert host.list_sessions() == ("s",)
    assert read_fixture_state(host, "s")[1] == prefix
    assert (
        host._one("SELECT count(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s", (host.tenant_id, "s"))[0]
        == content_count
    )
    assert host._one("SELECT count(*) FROM sk_inbox WHERE session_id != 's'")[0] == 0
    with open_store(database) as raw_reader:
        reader = wrap_fixture_store(raw_reader)
        assert reader._one("SELECT count(*) FROM business_child")[0] == 0
        assert reader.list_sessions() == ("s",)
