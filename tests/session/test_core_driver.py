"""Real SQL producers exercise the shared ledger and SDK callback boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vv_agent.budget import HostCost, RunBudgetLimits
from vv_agent.kernel._content_publication import publish_turn_started
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._records import SessionSpec, digest
from vv_agent.session import kernel
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.reducer import Fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost, SequenceConflict

from .helpers import (
    create_fixture_session,
    fixture_content_reader,
    item,
    push_fixture_input,
    read_fixture_state,
    wrap_fixture_store,
)
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


def commit_turn_started(driver):
    task = driver.runtime.compile("test", "t", seed={"messages": [], "shared_state": {}})
    definition = driver.runtime.definition(task)

    def prepare(tx):
        content = publish_turn_started(driver.content_writer(tx), definition, RunBudgetLimits().to_dict())
        return [
            driver.record(
                "turn_started",
                {
                    "input_ids": [],
                    "definition": content.definition.to_dict(),
                    "definition_digest": content.definition_digest,
                    "handler_version": driver.runtime.handler_version,
                    "budget": content.budget.to_dict(),
                    "binding": "task",
                    "generation": 0,
                },
                tid="t",
            )
        ]

    driver.commit([], guarded=True, prepare=prepare)


def run_case(driver_type, path, case, monkeypatch):
    events, phases = [], []
    clock = Clock(events)
    cut = Cut(case)
    with SQLiteStore.standalone(str(path), clock=clock) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        create_fixture_session(store, "s", consumers=())
        rt = runtime(path, [])
        rt.clock = clock
        rt.content_reader = fixture_content_reader(store, "s")
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
                content_reader=rt.content_reader,
            )
        else:
            driver = driver_type(store, "s", rt, scope)
        assert driver.store is store and driver.scope is scope
        if case in {"active", "cost_error", "cost_baseexception"}:
            commit_turn_started(driver)
        incoming = driver.record(
            "usage_observed",
            {
                "meter_id": "host",
                "observation": 1,
                "mode": "cumulative",
                "usage": {"elapsed_ms": 7},
                "source": "host",
            },
        )
        pending = [incoming]

        def prepare(tx):
            events.append(("prepare",))
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="business")
            writer.create_session(
                lambda content: SessionSpec(
                    "business",
                    content.service_principal("fixture/business")[0],
                    content.put_json("/tmp").to_dict(),
                    attributes={"seed": {"messages": [], "shared_state": {}}},
                ),
                consumers=(),
            )
            if case == "prepare":
                raise cut
            return []

        replacement_lease = None
        if case == "guard":
            push_fixture_input(store, "s", item("late"))
        if case == "lease":
            scope.poll(store)
            assert store.release(scope.lease)
            replacement_lease = store.acquire("s", owner="replacement", ttl_ms=15000)
            assert replacement_lease is not None
            assert replacement_lease.epoch == scope.lease.epoch + 1
            with pytest.raises(LeaseLost):
                scope.poll(store)
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
        hydrated_rows = read_fixture_state(store, "s")[1]
        result = {
            "error": error,
            "records": [r.record.to_dict() for r in rows],
            "hydrated_records": [r.record.to_dict() for r in hydrated_rows],
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
        if replacement_lease is not None:
            assert store.release(replacement_lease)
        return result


@pytest.mark.parametrize("case", ["normal", "replay", "prepare", "before_commit", "after_commit", "log_error", "guard", "lease"])
def test_sdk_and_core_run_identical_real_commit_algorithm(tmp_path, monkeypatch, case):
    core = run_case(CoreDriver, tmp_path / "core.sqlite", case, monkeypatch)
    sdk = run_case(kernel._Driver, tmp_path / "sdk.sqlite", case, monkeypatch)
    assert core["hydrated_records"] == sdk["hydrated_records"]
    assert {key: value for key, value in core.items() if key not in {"records", "driver_records"}} == {
        key: value for key, value in sdk.items() if key not in {"records", "driver_records"}
    }


@pytest.mark.parametrize("case", ["active", "cost_error", "cost_baseexception"])
def test_sdk_cost_conversion_and_partial_commit_boundary(tmp_path, monkeypatch, case):
    result = run_case(kernel._Driver, tmp_path / "cost.sqlite", case, monkeypatch)
    assert result["observed_cost"] == 1
    usage = [
        r for r in result["hydrated_records"] if r["kind"] == "usage_observed" and r["payload"]["source"] == "kernel-active-scope"
    ]
    if case == "cost_baseexception":
        assert result["error"] == "Cut" and not usage
    else:
        assert len(usage) == 1 and usage[0]["payload"]["usage"]["elapsed_ms"] > 0
        assert (usage[0]["payload"]["usage"]["host_cost"] is None) == (case == "cost_error")


def test_sdk_uses_core_ledger_methods_and_refreshes_live_bindings(tmp_path):
    names = (
        "refresh",
        "record",
        "boundary",
        "boundary_record",
        "pending_batch",
        "durable_shared_state",
    )
    for name in names:
        assert getattr(kernel._Driver, name) is getattr(CoreDriver, name)
    path = tmp_path / "refresh.sqlite"
    with SQLiteStore.standalone(str(path)) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        create_fixture_session(store, "s", consumers=())
        rt = runtime(path, [])
        rt.content_reader = fixture_content_reader(store, "s")
        lease = store.acquire("s", owner="refresh", ttl_ms=15000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        driver = kernel._Driver(store, "s", rt, scope)
        original_clock = driver.clock
        rt.clock = Clock([])
        assert driver.clock is original_clock
        assert driver.clock is not rt.clock
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
    with SQLiteStore.standalone(str(path)) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw_store)
        create_fixture_session(store, "s", consumers=())
        rt = runtime(path, [])
        rt.content_reader = fixture_content_reader(store, "s")
        lease = store.acquire("s", owner="state", ttl_ms=15000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        driver = kernel._Driver(store, "s", rt, scope)
        commit_turn_started(driver)
        assert driver.durable_shared_state() == {"todo_list": []} and not driver.pending_batch()
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
        assert driver.durable_shared_state() == {"items": [1]}
