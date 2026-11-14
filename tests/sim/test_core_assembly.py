"""Focused host-binding checks for the real core SimWorld assembly."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sim import assembly
from sim.assembly import ScheduledHeartbeat, Scope, SimDriver, _cancellation_factory
from sim.scenarios import Provider, run_scenario
from sim.world import SimWorld
from vv_agent._memory_settings import normalize_memory_settings
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation
from vv_agent.kernel._budget_values import BudgetUnavailableReason, HostCost
from vv_agent.kernel._compaction_phase import CompactionMemory
from vv_agent.kernel._driver_outcomes import OutcomeRead
from vv_agent.kernel._lease_scope import HeartbeatStopped, LeaseOwnership


def capture_factory_heartbeats(monkeypatch):
    actual = assembly.ScheduledHeartbeat
    captured = []

    def create(*args, **kwargs):
        heartbeat = actual(*args, **kwargs)
        captured.append(heartbeat)
        return heartbeat

    monkeypatch.setattr(assembly, "ScheduledHeartbeat", create)
    return captured


def core_worker(world: SimWorld, *, runtime_setup=None):
    world.create()
    runtime = world.runtime("s", "owner")
    if runtime_setup is not None:
        runtime_setup(runtime)
    store = world.connection("owner")
    lease = store.acquire("s", owner="owner", ttl_ms=runtime.ttl_ms)
    assert lease is not None
    scope = Scope(lease, runtime, now_ms=store._now())
    worker = SimDriver(store, "s", runtime, scope)
    worker.refresh()
    return worker, runtime


def test_budget_meter_is_bound_once_per_run_across_nested_runs(tmp_path):
    class Meter:
        def __init__(self, amount):
            self.amount = amount

        def read(self):
            return HostCost("credits", self.amount)

    class Clock:
        def __init__(self):
            self.calls = 0

        def now_ms(self):
            return 0

        def monotonic(self):
            return 0.0

        def monotonic_ns(self):
            self.calls += 1
            return self.calls

        def sleep(self, seconds):
            del seconds

    class ReplacementClock(Clock):
        def monotonic_ns(self):
            raise AssertionError("binding must not reread mutable runtime.clock")

    with SimWorld(tmp_path / "budget-binding.sqlite", seed=17) as world:
        budget_clock = Clock()
        worker, runtime = core_worker(world, runtime_setup=lambda selected: setattr(selected, "clock", budget_clock))
        first, replacement = Meter(3), Meter(1001)
        budget_clock.calls = 0
        runtime.clock = ReplacementClock()
        runtime.config = replace(runtime.config, host_cost_meter=first)

        def inner_steps():
            yield OutcomeRead.BIND_BUDGET_METER
            yield BudgetObservation.CLOCK_NS
            return (yield BudgetObservation.HOST_COST)

        def outer_steps():
            yield OutcomeRead.BIND_BUDGET_METER
            runtime.config = replace(runtime.config, host_cost_meter=replacement)
            inner = worker.run(inner_steps())
            yield BudgetObservation.CLOCK_NS
            return inner, (yield BudgetObservation.HOST_COST)

        inner, outer = worker.run(outer_steps())
        assert inner.reading == HostCost("credits", 1001)
        assert outer.reading == HostCost("credits", 3)
        assert budget_clock.calls == 2

        runtime.config = replace(runtime.config, host_cost_meter=None)

        def unavailable_steps():
            yield OutcomeRead.BIND_BUDGET_METER
            return (yield BudgetObservation.HOST_COST)

        missing = worker.run(unavailable_steps())
        assert isinstance(missing, HostCostObservation)
        assert missing.unavailable_reason is BudgetUnavailableReason.METER_MISSING

        class BrokenMeter:
            def read(self):
                raise RuntimeError("meter unavailable")

        runtime.config = replace(runtime.config, host_cost_meter=BrokenMeter())
        failed = worker.run(unavailable_steps())
        assert isinstance(failed, HostCostObservation)
        assert failed.unavailable_reason is BudgetUnavailableReason.METER_ERROR


def test_scheduled_heartbeat_constructs_actual_lease_ownership(tmp_path):
    with SimWorld(tmp_path / "heartbeat-construction.sqlite", seed=17) as world:
        world.create()
        runtime = world.runtime("s", "owner")
        store = world.connection("owner")
        now_ms = store._now()
        lease = store.acquire("s", owner="owner", ttl_ms=runtime.ttl_ms)
        assert lease is not None
        heartbeat = ScheduledHeartbeat(
            world,
            "owner",
            lease,
            runtime.ttl_ms,
            runtime.heartbeat_seconds,
            runtime.clock,
            now_ms=now_ms,
        )
        assert isinstance(heartbeat, LeaseOwnership)
        assert heartbeat.expires_deadline == runtime.clock.monotonic() + (lease.expires_at_ms - now_ms) / 1000
        assert store.release(lease)


def test_factory_runtime_owns_driver_scope_hook_and_ttl(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "factory-runtime.sqlite", seed=17) as world:
        world.create()
        factory_heartbeats = capture_factory_heartbeats(monkeypatch)
        provider = Provider(world)
        base = world.runtime("s", "worker-0", provider=provider)
        factory_runtime = world.runtime("s", "worker-0", provider=provider)
        factory_runtime.handler_version = "factory-runtime-v2"
        factory_runtime.ttl_ms = 22_000
        world.runtimes[("s", "worker-0")] = base

        hook_points = []
        original_hook = factory_runtime.hook

        def observed_hook(point, record):
            hook_points.append(point)
            original_hook(point, record)

        factory_runtime.hook = observed_hook
        store = world.connection("worker-0")
        renew = store.renew
        ttl_values = []

        def observed_renew(lease, *, ttl_ms):
            ttl_values.append(ttl_ms)
            return renew(lease, ttl_ms=ttl_ms)

        monkeypatch.setattr(store, "renew", observed_renew)
        assert world.drive(provider=provider, factory=lambda _sid: factory_runtime)

        _state, rows, _ = world.connection("observer").read_state("s")
        started = next(row.record for row in rows if row.record.kind == "turn_started")
        assert started.payload["handler_version"] == "factory-runtime-v2"
        assert 22_000 in ttl_values and ttl_values[-1] == 22_000
        assert "after_drive" in hook_points
        assert len(factory_heartbeats) == 1
        heartbeat = factory_heartbeats[0]
        assert heartbeat.stop.is_set()
        assert f"renew/{heartbeat.scheduler.participant}" not in world.tasks
        with pytest.raises(HeartbeatStopped):
            heartbeat.poll(world.connection("worker-0/factory/heartbeat"))


def test_factory_heartbeat_stops_on_runtime_not_ready(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "factory-not-ready.sqlite", seed=17) as world:
        world.create()
        factory_heartbeats = capture_factory_heartbeats(monkeypatch)
        assert world.drive(factory=world.not_ready)
        assert world.calls == []
        assert len(factory_heartbeats) == 1
        heartbeat = factory_heartbeats[0]
        assert heartbeat.stop.is_set()
        assert f"renew/{heartbeat.scheduler.participant}" not in world.tasks
        with pytest.raises(HeartbeatStopped):
            heartbeat.poll(world.connection(f"{heartbeat.scheduler.participant}/heartbeat"))


def test_sim_cancellation_children_are_independent_and_propagate_parent_cancel():
    parent = _cancellation_factory()
    child = parent.child()
    grandchild = child.child()
    assert child is not parent and grandchild is not child
    assert child._event is not parent._event and child._lock is not parent._lock
    parent.cancel("root cancel")
    assert parent.cancelled and child.cancelled and grandchild.cancelled
    assert {parent.reason, child.reason, grandchild.reason} == {"root cancel"}


def test_context_compaction_event_uses_core_effective_threshold(tmp_path):
    configured = {
        "compact_threshold": 5_000,
        "model_context_window": 2_048,
        "reserved_output_tokens": 800,
        "autocompact_buffer_tokens": 748,
    }
    with SimWorld(tmp_path / "compaction-threshold.sqlite", seed=17) as world:
        summary = run_scenario(world, "context_compacted", memory_settings=configured)
        _state, rows, _ = world.connection("observer").read_state("s")
        started = next(
            row.record
            for row in rows
            if row.record.kind == "boundary_recorded" and row.record.payload["stage"] == "memory_started"
        )
        event = started.payload["data"]["event"]
        manager = CompactionMemory(normalize_memory_settings(world.memory_settings, None), None, False)

    assert summary["scenario"] == "context_compacted"
    assert event["configured_threshold"] == 5_000
    assert manager.autocompact_threshold == 500
    assert event["effective_threshold"] == manager.autocompact_threshold
    assert event["microcompact_threshold"] == int(manager.autocompact_threshold * 0.95)
    assert event["microcompact_target"] == int(manager.autocompact_threshold * 0.90)
