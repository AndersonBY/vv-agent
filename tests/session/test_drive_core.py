"""The extracted drive lifecycle uses real leases, SQL, errors and host callbacks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Event, Lock, get_ident, local
from typing import Any, cast

import pytest

from vv_agent.kernel._drive import DriveBinding, DriveStep, drive_core
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import SessionSpec, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._store import StoredRecord


class FixedClock:
    def __init__(self) -> None:
        self.milliseconds = 1_000_000
        self.sleeps: list[float] = []
        self.on_sleep = None

    def now_ms(self) -> int:
        return self.milliseconds

    def monotonic(self) -> float:
        return self.milliseconds / 1000

    def monotonic_ns(self) -> int:
        return self.milliseconds * 1_000_000

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))


def unused(*args, **kwargs):
    raise AssertionError("the drive lifecycle must not fold")


def fold_factory() -> Fold:
    return Fold(normalize_memory_settings=unused, count_messages=unused)


class Keeper:
    def is_alive(self, heartbeat: object) -> bool:
        return False


class FactoryHeartbeat(LeaseOwnership):
    def __init__(self, lease, heartbeat_store, ttl_ms: int, clock: FixedClock):
        self.heartbeat_store = heartbeat_store
        self.lease_keeper = Keeper()
        super().__init__(
            lease,
            ttl_ms,
            0.25,
            clock=clock,
            stop=Event(),
            ready=Event(),
            lock=Lock(),
            poll_lock=Lock(),
            commit_lock=Lock(),
            latch_entry=local(),
            actor_identity=get_ident,
        )

    def start(self) -> None:
        with self.heartbeat_store() as store:
            self.poll(store)
        self.heartbeat_finished()
        self.request_stop()

    def close(self) -> None:
        self.request_stop()
        self.check_closed(False)


class Scope(LeaseOwnership):
    def __init__(self, lease, heartbeat_store, ttl_ms: int, clock: FixedClock):
        self.heartbeat_store = heartbeat_store
        self.close_count = 0
        super().__init__(
            lease,
            ttl_ms,
            0.25,
            clock=clock,
            stop=Event(),
            ready=Event(),
            lock=Lock(),
            poll_lock=Lock(),
            commit_lock=Lock(),
            latch_entry=local(),
            actor_identity=get_ident,
        )

    def start(self) -> None:
        with self.heartbeat_store() as store:
            self.poll(store)

    def close(self) -> None:
        self.close_count += 1
        self.request_stop()
        self.check_closed(False)


class Runtime:
    ttl_ms = 15_000
    poll_ms = 37

    def __init__(self, events: list[str]):
        self.events = events

    def hook(self, point: str, record) -> None:
        self.events.append(point)

    def wake(self, session_id: str) -> None:
        self.events.append("wake")


class SqlAppendPhase:
    def __init__(self, store, session_id: str, scope: Scope, events: list[str]):
        self.store, self.session_id, self.scope, self.events = store, session_id, scope, events
        self.records = []
        self.calls = 0

    def step(self) -> bool:
        self.calls += 1
        if self.calls > 1:
            self.events.append("phase:idle")
            return False
        record = make_record(
            "usage_observed",
            session_id=self.session_id,
            payload={"meter_id": "drive-core", "observation": 1, "mode": "cumulative", "usage": {}, "source": "host"},
        )
        with self.store.atomic() as tx, self.scope.committing() as lease:
            tx.append(
                self.session_id,
                lease=lease,
                expected_seq=self.store.read(self.session_id).head_seq,
                commit_id="drive-core-step",
                records=(record,),
            )
        self.records = list(self.store.read(self.session_id).records[-1:])
        self.events.append("phase:append")
        return True


class RetryReady(Exception):
    def __init__(self):
        self.retry_after_ms = 37


class PhaseFailure(Exception):
    pass


def binding(store, session_id: str, events: list[str], clock: FixedClock, *, factory: bool):
    runtime = None if factory else Runtime(events)
    scopes: list[Scope] = []
    drivers: list[SqlAppendPhase] = []

    def make_heartbeat(lease, ttl_ms):
        return FactoryHeartbeat(lease, store.heartbeat_store, ttl_ms, clock)

    def make_runtime():
        events.append("runtime")
        return Runtime(events)

    def make_scope(lease, selected):
        events.append("scope")
        scope = Scope(lease, store.heartbeat_store, selected.ttl_ms, clock)
        scopes.append(scope)
        return scope

    def make_driver(selected_store, selected_session_id, selected, scope):
        events.append("driver")
        phase = SqlAppendPhase(selected_store, selected_session_id, scope, events)
        drivers.append(phase)
        return phase

    value = DriveBinding(
        clock=clock,
        ttl_ms=15_000,
        runtime=runtime,
        make_factory_heartbeat=make_heartbeat if factory else None,
        make_runtime=make_runtime if factory else None,
        factory_retry_after_ms=lambda error: error.retry_after_ms if isinstance(error, RetryReady) else None,
        heartbeat_alive=lambda heartbeat: cast(Any, heartbeat).lease_keeper.is_alive(heartbeat),
        make_scope=make_scope,
        make_driver=make_driver,
        after_drive=lambda selected: cast(Any, selected).hook("after_drive", None),
        wake_if_runnable=lambda selected: cast(Any, selected).wake(session_id),
    )
    return value, scopes, drivers


def create_session(store, session_id: str = "s") -> None:
    with store.atomic() as tx:
        tx.create(SessionSpec(session_id, "p", "w"), consumers=())


@pytest.mark.persistent_store
def test_factory_handoff_and_idle_guard_run_on_the_real_store(store, monkeypatch):
    create_session(store)
    events: list[str] = []
    clock = FixedClock()
    configured, scopes, drivers = binding(store, "s", events, clock, factory=True)
    defer_idle = store._defer_idle_drive
    latched = []

    def observe_idle(lease, *, poll_ms, lease_guard):
        def guarded():
            scope = scopes[0]
            current = lease_guard()
            assert current == scope.lease
            assert scope.commit_lock.locked()
            return current

        result = defer_idle(lease, poll_ms=poll_ms, lease_guard=guarded)
        latched.append((scopes[0].commit_lock.locked(), poll_ms, result))
        events.append("defer_idle")
        return result

    monkeypatch.setattr(store, "_defer_idle_drive", observe_idle)

    assert drive_core(store, "s", make_binding=lambda: configured)
    assert events == ["runtime", "scope", "driver", "phase:append", "phase:idle", "defer_idle", "after_drive"]
    assert latched == [(True, 37, False)]
    assert len(scopes) == 1 and scopes[0].close_count == 1
    assert len(drivers) == 1 and drivers[0].calls == 2
    assert store.read("s").records[-1].record.kind == "usage_observed"
    owner, retry_at = store._one("SELECT lease_owner,drive_retry_at_ms FROM sk_session WHERE session_id='s'")
    assert owner is None and retry_at == 0


@pytest.mark.persistent_store
def test_factory_readiness_and_failure_backoff_follow_existing_order(store):
    create_session(store)
    events: list[str] = []
    clock = FixedClock()
    configured, scopes, drivers = binding(store, "s", events, clock, factory=True)
    configured = DriveBinding(
        **{field: getattr(configured, field) for field in configured.__dataclass_fields__ if field != "make_runtime"},
        make_runtime=lambda: (_ for _ in ()).throw(RetryReady()),
    )

    started_ms = store._now()
    assert drive_core(store, "s", make_binding=lambda: configured) is True
    owner, retry_at = store._one("SELECT lease_owner,drive_retry_at_ms FROM sk_session WHERE session_id='s'")
    assert owner is None and retry_at >= started_ms + 37
    assert events == [] and not scopes and not drivers


@pytest.mark.persistent_store
def test_unclassified_factory_error_defers_releases_and_reraises_same_error(store):
    create_session(store)
    events: list[str] = []
    error = ValueError("factory failed")
    configured, scopes, drivers = binding(store, "s", events, FixedClock(), factory=True)
    configured = DriveBinding(
        **{field: getattr(configured, field) for field in configured.__dataclass_fields__ if field != "make_runtime"},
        make_runtime=lambda: (_ for _ in ()).throw(error),
    )

    started_ms = store._now()
    with pytest.raises(ValueError) as caught:
        drive_core(store, "s", failure_backoff_ms=1234, make_binding=lambda: configured)
    assert caught.value is error
    owner, retry_at = store._one("SELECT lease_owner,drive_retry_at_ms FROM sk_session WHERE session_id='s'")
    assert owner is None and retry_at >= started_ms + 1234
    assert events == [] and not scopes and not drivers


def test_phase_error_closes_releases_hooks_and_does_not_backoff_direct_runtime(tmp_path):
    from vv_agent.kernel._sqlite import SQLiteStore

    clock = FixedClock()
    with SQLiteStore.standalone(str(tmp_path / "error.sqlite"), fold_factory=fold_factory, clock=clock) as store:
        store.install_schema()
        create_session(store)
        events: list[str] = []
        configured, scopes, _ = binding(store, "s", events, clock, factory=False)
        error = PhaseFailure("phase failed")

        class FailingPhase:
            records: tuple[StoredRecord, ...] = ()

            def step(self):
                raise error

        configured = DriveBinding(
            **{field: getattr(configured, field) for field in configured.__dataclass_fields__ if field != "make_driver"},
            make_driver=lambda *args: cast(DriveStep, FailingPhase()),
        )
        with pytest.raises(PhaseFailure) as caught:
            drive_core(store, "s", make_binding=lambda: configured)
        assert caught.value is error
        assert events == ["scope", "after_drive"]
        assert scopes[0].close_count == 2
        owner, retry_at = store._one("SELECT lease_owner,drive_retry_at_ms FROM sk_session WHERE session_id='s'")
        assert owner is None and retry_at == 0


def test_lease_wait_uses_exact_fifty_then_one_hundred_millisecond_delays(tmp_path):
    from vv_agent.kernel._sqlite import SQLiteStore

    clock = FixedClock()
    with SQLiteStore.standalone(str(tmp_path / "wait.sqlite"), fold_factory=fold_factory, clock=clock) as store:
        store.install_schema()
        create_session(store)
        holder = store.acquire("s", owner="holder", ttl_ms=15_000)
        assert holder is not None
        clock.on_sleep = lambda count: store.release(holder) if count == 2 else None
        events: list[str] = []
        configured, _, _ = binding(store, "s", events, clock, factory=False)

        assert drive_core(store, "s", wait_for_lease=True, make_binding=lambda: configured)
        assert clock.sleeps == [0.05, 0.1]
        assert store.lease_status("s")[:2] == (2, 2)


def test_option_validation_precedes_binding_and_store_effects(tmp_path):
    from vv_agent.kernel._sqlite import SQLiteStore

    called = False

    def make_binding():
        nonlocal called
        called = True
        raise AssertionError("binding must not be constructed")

    with SQLiteStore.standalone(str(tmp_path / "invalid.sqlite"), fold_factory=fold_factory, clock=FixedClock()) as store:
        store.install_schema()
        with pytest.raises(ValueError, match="failure_backoff_ms must be a positive integer"):
            drive_core(store, "s", failure_backoff_ms=0, make_binding=make_binding)
    assert called is False


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_drive_core_is_sdk_free_and_runs_factory_transfer_on_real_sqlite(tmp_path, flags):
    script = r"""
import importlib.abc
import sys
from threading import Event, Lock, get_ident, local
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        allowed = (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                   or fullname.split('.')[0] in sys.stdlib_module_names)
        assert allowed, fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._drive import DriveBinding, drive_core
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import SessionSpec, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import SQLiteStore
class Clock:
    def now_ms(self): return 1000000
    def monotonic(self): return 1000.0
    def monotonic_ns(self): return 1000000000000
    def sleep(self, seconds): raise AssertionError('no wait')
class Keeper:
    def is_alive(self, heartbeat): return False
class Heartbeat(LeaseOwnership):
    def __init__(self, lease, source, ttl_ms, clock):
        self.source, self.lease_keeper = source, Keeper()
        super().__init__(lease, ttl_ms, .25, clock=clock, stop=Event(), ready=Event(), lock=Lock(),
            poll_lock=Lock(), commit_lock=Lock(), latch_entry=local(), actor_identity=get_ident)
    def start(self):
        with self.source() as store: self.poll(store)
        self.heartbeat_finished(); self.request_stop()
    def close(self):
        self.request_stop(); self.check_closed(False)
class Scope(Heartbeat):
    def start(self):
        with self.source() as store: self.poll(store)
class Runtime:
    ttl_ms=15000
    poll_ms=37
    def __init__(self): self.events=[]
    def hook(self, point, record): self.events.append(point)
    def wake(self, session_id): self.events.append('wake')
class Phase:
    def __init__(self, store, scope): self.store, self.scope, self.records, self.calls=store, scope, [], 0
    def step(self):
        self.calls += 1
        if self.calls > 1: return False
        rec=make_record('usage_observed', session_id='s', payload={'meter_id':'core','observation':1,
            'mode':'cumulative','usage':{},'source':'host'})
        with self.store.atomic() as tx:
            with self.scope.committing() as lease:
                tx.append('s', lease=lease, expected_seq=1, commit_id='core-step', records=(rec,))
        self.records=list(self.store.read('s').records[-1:])
        return True
def unused(*args, **kwargs): raise AssertionError('not folded')
def factory(): return Fold(normalize_memory_settings=unused, count_messages=unused)
clock=Clock()
with SQLiteStore.standalone(sys.argv[2], clock=clock, fold_factory=factory) as store:
    store.install_schema()
    with store.atomic() as tx: tx.create(SessionSpec('s','p','w'), consumers=())
    runtime=None
    def make_runtime():
        global runtime
        runtime=Runtime(); return runtime
    def make_scope(lease, selected): return Scope(lease, store.heartbeat_store, selected.ttl_ms, clock)
    def make_driver(selected_store, sid, selected, scope): return Phase(selected_store, scope)
    binding=DriveBinding(clock=clock, ttl_ms=15000, runtime=None,
        make_factory_heartbeat=lambda lease,ttl: Heartbeat(lease, store.heartbeat_store, ttl, clock),
        make_runtime=make_runtime, factory_retry_after_ms=lambda error: None,
        heartbeat_alive=lambda heartbeat: heartbeat.lease_keeper.is_alive(heartbeat),
        make_scope=make_scope, make_driver=make_driver,
        after_drive=lambda selected: selected.hook('after_drive', None),
        wake_if_runnable=lambda selected: selected.wake('s'))
    assert drive_core(store, 's', make_binding=lambda: binding)
    assert store.read('s').records[-1].record.kind=='usage_observed'
    assert store._one("SELECT lease_owner FROM sk_session WHERE session_id='s'")[0] is None
    assert runtime.events==['after_drive']
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[2] / "src"), str(tmp_path / "core.sqlite")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
