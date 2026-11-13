"""Real SQL producers exercise the shared ledger and SDK callback boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vv_agent.budget import HostCost
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._records import SessionSpec, copy_json, digest
from vv_agent.session import kernel
from vv_agent.session.reducer import Fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost, SequenceConflict

from .helpers import record
from .test_recovery_matrix import runtime


class Clock:
    def __init__(self, events):
        self.events, self.ns = events, 0

    def now_ms(self):
        return 1000

    def monotonic(self):
        return 1.0

    def monotonic_ns(self):
        self.ns += 1_000_000
        self.events.append(("clock", self.ns))
        return self.ns

    def sleep(self, seconds):
        raise AssertionError("ledger does not schedule sleeps")


class Cut(BaseException):
    pass


def run_case(driver_type, path, case, monkeypatch):
    events, phases = [], []
    clock = Clock(events)
    cut = Cut(case)
    with SQLiteStore.standalone(str(path), clock=clock) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "w"), consumers=())
        rt = runtime(path, [])
        rt.clock = clock
        observed = []

        def cost():
            events.append(("cost",))
            observed.append(True)
            if case == "cost_error":
                raise ValueError("unavailable")
            if case == "cost_baseexception":
                raise cut
            return HostCost("credits", 3)

        rt.config = replace(rt.config, host_cost_meter=SimpleNamespace(read=cost))

        def hook(point, rec):
            events.append((point, None if rec is None else rec.kind))
            if point == case:
                raise cut

        rt.hook = hook

        def emit(level, message, *, extra):
            events.append(("phase", extra["phase"]))
            phases.append((level, message, dict(extra)))
            if case == "log_error":
                raise cut

        monkeypatch.setattr(kernel.logger, "log", emit)
        lease = store.acquire("s", owner="ledger", ttl_ms=15000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        if driver_type is CoreDriver:
            # Core consumes already converted observations; the SDK producer below
            # exercises its actual conversion/error handling, not a replacement runtime.
            driver = CoreDriver(
                store,
                "s",
                scope,
                fold_factory=store.fold_factory,
                clock=clock,
                observe_cost=lambda: {"unit": "credits", "currency": None, "amount_microunits": 3},
                hook=hook,
                emit_phase=lambda duration, fields: emit(
                    30 if duration >= rt.ttl_ms / 2 else 10, "session transaction phase", extra=fields
                ),
            )
        else:
            driver = driver_type(store, "s", rt, scope)
        assert driver.store is store and driver.scope is scope
        if case in {"active", "cost_error", "cost_baseexception"}:
            # These are original record inputs; all construction, append and folding are real.
            task = rt.compile("go", "t")
            definition = rt.definition(task)
            driver.commit([record("turn_started", definition=definition, definition_digest=rt.definition_digest(task))])
        incoming = driver.record(
            "usage_observed",
            {
                "meter_id": "host",
                "observation": 1,
                "mode": "cumulative",
                "usage": {},
                "source": "host",
            },
        )
        pending = [incoming]

        def prepare(tx):
            events.append(("prepare",))
            tx.create(SessionSpec("business", "p", "w"), consumers=())
            if case == "prepare":
                raise cut
            return []

        if case == "guard":
            from vv_agent.session.records import InboxItem

            with store.atomic() as tx:
                tx.push("s", InboxItem("late", "user", {"content": "late"}))
        if case == "lease":
            scope.lease = replace(scope.lease, epoch=scope.lease.epoch + 1)
        before = len(driver.records)
        error = None
        try:
            driver.commit(pending, guarded=case == "guard", prepare=prepare)
            if case == "replay":
                driver.commit(pending)
        except BaseException as exc:
            error = type(exc).__name__
            if case in {"prepare", "before_commit", "after_commit", "log_error", "cost_baseexception"}:
                assert exc is cut
            elif case == "guard":
                assert isinstance(exc, SequenceConflict)
            elif case == "lease":
                assert isinstance(exc, LeaseLost)
            else:
                raise
        committed = case not in {"prepare", "before_commit", "guard", "lease", "cost_baseexception"}
        assert ("business" in store.list_sessions()) == committed
        assert (len(driver.records) > before) == (committed and case not in {"after_commit", "log_error"})
        rows = store.read("s").records
        result = {
            "error": error,
            "records": [r.record.to_dict() for r in rows],
            "driver_records": [r.record.to_dict() for r in driver.records],
            "events": events[:],
            "phases": phases[:],
            "watermark": driver.watermark,
            "active_since": driver.active_since,
            "active_tid": driver.active_tid,
            "observed_cost": len(observed),
        }
        if case == "replay":
            assert len(rows) == 2 and len(driver.records) == 2
            assert len([e for e in events if e[0] == "after_commit"]) == 2
        return result


@pytest.mark.parametrize("case", ["normal", "replay", "prepare", "before_commit", "after_commit", "log_error", "guard", "lease"])
def test_sdk_and_core_run_identical_real_commit_algorithm(tmp_path, monkeypatch, case):
    core = run_case(CoreDriver, tmp_path / "core.sqlite", case, monkeypatch)
    sdk = run_case(kernel._Driver, tmp_path / "sdk.sqlite", case, monkeypatch)
    assert core == sdk


@pytest.mark.parametrize("case", ["active", "cost_error", "cost_baseexception"])
def test_sdk_cost_conversion_and_partial_commit_boundary(tmp_path, monkeypatch, case):
    result = run_case(kernel._Driver, tmp_path / "cost.sqlite", case, monkeypatch)
    assert result["observed_cost"] == 1
    usage = [r for r in result["records"] if r["kind"] == "usage_observed" and r["payload"]["source"] == "kernel-active-scope"]
    if case == "cost_baseexception":
        assert result["error"] == "Cut" and not usage
    else:
        assert len(usage) == 1 and usage[0]["payload"]["usage"]["elapsed_ms"] > 0
        assert (usage[0]["payload"]["usage"]["host_cost"] is None) == (case == "cost_error")


def test_sdk_uses_core_ledger_methods_and_refreshes_live_bindings(tmp_path):
    names = (
        "load_state",
        "refresh",
        "extend",
        "bind_turn",
        "record",
        "boundary",
        "boundary_record",
        "pending_batch",
        "durable_shared_state",
        "commit",
    )
    for name in names:
        assert getattr(kernel._Driver, name) is getattr(CoreDriver, name)
    path = tmp_path / "refresh.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "w"), consumers=())
        rt = runtime(path, [])
        lease = store.acquire("s", owner="refresh", ttl_ms=15000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        driver = kernel._Driver(store, "s", rt, scope)
        clock = Clock([])
        rt.clock = clock
        assert driver.clock is clock
        calls = []

        def factory():
            calls.append(True)
            return Fold()

        store.fold_factory = factory
        driver.load_state()
        assert store._fold_cache is not None
        assert calls and driver.fold is not store._fold_cache.reducer
        assert driver.state == store.read_state("s")[0]
        assert driver.store is store and driver.scope is scope


def test_core_boundary_and_pending_state_follow_actual_retained_records(tmp_path):
    path = tmp_path / "state.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "w"), consumers=())
        rt = runtime(path, [])
        lease = store.acquire("s", owner="state", ttl_ms=15000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        driver = kernel._Driver(store, "s", rt, scope)
        task = rt.compile("go", "t")
        definition = rt.definition(task)
        driver.commit([record("turn_started", definition=definition, definition_digest=rt.definition_digest(task))])
        assert driver.durable_shared_state() == task.initial_shared_state and not driver.pending_batch()
        messages = [m.to_dict() for m in driver.transcript()]
        boundary = driver.boundary_record(
            "before_memory", "b", {"messages": messages, "shared_state": {"items": [1]}}, source_digest=digest(messages)
        )
        driver.commit([boundary])
        assert driver.boundary("before_memory", "b") is not None
        snapshot = driver.durable_shared_state()
        snapshot["items"].append(2)
        assert driver.durable_shared_state() == {"items": [1]}
        plan = driver.model_plan()
        assert plan is not None
        driver.commit([plan])
        assert driver.pending_batch()
        driver.fold = Fold()
        driver.load_state()
        assert driver.pending_batch()
        assert driver.durable_shared_state() == copy_json(boundary.payload["data"]["shared_state"])
