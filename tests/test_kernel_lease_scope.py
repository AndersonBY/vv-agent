"""Actual core lease rules run with actor-local synchronization and scripted receipts."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = r"""
import importlib.abc
import inspect
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel.")
                or fullname.split(".")[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._lease_scope import LeaseOwnership, HeartbeatStopped, apply_controls, cancel_if_lost
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._store import Lease, LeaseLost, LeasePoll
source, start_line = inspect.getsourcelines(apply_controls)
cancel_line = start_line + next(i for i, line in enumerate(source) if "self.token.cancel(item.payload" in line)

class Actors:
    current = 1
    def __init__(self): self.trace = []
    @contextmanager
    def use(self, actor):
        previous = self.current
        self.current = actor
        try: yield
        finally: self.current = previous

class Entry:
    def __init__(self, actors): self.actors, self.values = actors, {}
    @property
    def poll(self): return self.values.get((self.actors.current, "poll"), False)
    @poll.setter
    def poll(self, value): self.values[self.actors.current, "poll"] = value
    @property
    def commit(self): return self.values.get((self.actors.current, "commit"), False)
    @commit.setter
    def commit(self, value): self.values[self.actors.current, "commit"] = value

class Mutex:
    def __init__(self, actors, name):
        self.actors, self.name, self.owner = actors, name, None
        self.on_enter = self.on_exit = None
    def acquire(self, blocking=True, timeout=-1):
        if self.owner is not None: return False
        self.owner = self.actors.current
        self.actors.trace.append((self.owner, "enter", self.name))
        return True
    def release(self):
        assert self.owner == self.actors.current
        self.actors.trace.append((self.owner, "exit", self.name))
        self.owner = None
    def locked(self): return self.owner is not None
    def __enter__(self):
        assert self.acquire(), "a contending actor must be scheduled after release"
        if self.on_enter is not None: self.on_enter()
        return self
    def __exit__(self, *args):
        try:
            if self.on_exit is not None: self.on_exit()
        finally: self.release()

class Flag:
    def __init__(self): self.value = False
    def is_set(self): return self.value
    def set(self): self.value = True
    def wait(self, timeout=None): raise AssertionError("core must not schedule a wait")

class ManualClock:
    def __init__(self): self.wall, self.mono, self.calls = 0, 100.0, []
    def now_ms(self): self.calls.append("wall"); return self.wall
    def monotonic(self): self.calls.append("mono"); return self.mono
    def monotonic_ns(self): return int(self.mono * 1e9)
    def sleep(self, seconds): raise AssertionError("core must not schedule sleep")

class Sink:
    def __init__(self): self.reasons = []
    def cancel(self, reason): self.reasons.append(reason)

class ReceiptStore:
    # Only the external renewal boundary is scripted; no lease policy is implemented here.
    def __init__(self, callback): self.callback, self.calls = callback, 0
    def renew(self, lease, *, ttl_ms):
        self.calls += 1
        return self.callback(lease, ttl_ms)

def make():
    actors, clock = Actors(), ManualClock()
    state = LeaseOwnership(Lease("s", "owner", 1, 15000), 15000, 0.25,
        clock=clock, stop=Flag(), ready=Flag(), lock=Mutex(actors, "S"),
        poll_lock=Mutex(actors, "P"), commit_lock=Mutex(actors, "C"),
        latch_entry=Entry(actors), actor_identity=lambda: actors.current)
    assert clock.calls == ["wall", "mono"] and state.expires_deadline == 115.0
    state.token, state.turn_id, state.generation = Sink(), "turn", 1
    return state, actors, clock

@contextmanager
def raises(kind, message):
    try: yield
    except kind as error:
        assert message in str(error), str(error)
    else: raise AssertionError("expected " + kind.__name__)

# Imports are ordinary package imports; construction/execution cannot allocate host primitives or read host time.
def forbidden(*args, **kwargs): raise AssertionError("host primitive or clock used by core")
for name in ("Thread", "Event", "Lock", "RLock", "local"):
    setattr(threading, name, forbidden)
for name in ("time", "time_ns", "monotonic", "monotonic_ns", "sleep"):
    setattr(time, name, forbidden)
def audit(event, args):
    if event in {"open", "os.listdir", "os.scandir", "socket.connect", "socket.getaddrinfo", "subprocess.Popen"}:
        raise AssertionError("core effect: " + event)
sys.addaudithook(audit)

# L1: poll has P only while renewing; a distinct actor can enter C and observe the old lease.
state, actors, clock = make()
old = state.lease
clock.mono = 101.0
control = InboxItem("cancel", "control", {"action": "cancel"}, "turn", 1)
receipt = LeasePoll(replace(old, expires_at_ms=17000), 1000, 3, (control,))
def renewal(lease, ttl_ms):
    assert lease is old and ttl_ms == 15000
    assert state.poll_lock.owner == 1 and not state.lock.locked() and not state.commit_lock.locked()
    with actors.use(2), state.committing() as captured:
        assert captured is old and state._commit_thread == 2
        with raises(RuntimeError, "poll while holding commit/poll latch"):
            state.poll(store)
    clock.mono = 103.0
    return receipt
store = ReceiptStore(renewal)
assert state.poll(store) is receipt
assert state.expires_deadline == 117.0 and state.lease is receipt.lease
assert state.successful_ready and state.ready.is_set() and store.calls == 1
assert actors.trace == [(1,"enter","P"),(1,"enter","S"),(1,"exit","S"),
    (2,"enter","C"),(2,"enter","S"),(2,"exit","S"),(2,"exit","C"),
    (1,"enter","S"),(1,"exit","S"),(1,"exit","P")]
assert state._poll_thread is state._commit_thread is None
assert not any(state._latch_entry.values.values())

# Publication may proceed under another actor's commit latch, without replacing its captured lease.
state, actors, clock = make()
old = state.lease
store = ReceiptStore(lambda lease, ttl: LeasePoll(replace(lease, expires_at_ms=16000), 1000, 0, ()))
with state.committing() as captured:
    with actors.use(2): state.poll(store)
    assert state.commit_lock.owner == 1 and captured is old and state.lease is not old

# Entry guards cover acquired-latch and release-tail windows, not merely owner markers.
for kind in ("poll", "commit"):
    state, actors, clock = make()
    store = ReceiptStore(lambda lease, ttl: LeasePoll(lease, 0, 0, ()))
    def rejected():
        with raises(RuntimeError, "poll while holding commit/poll latch"): state.poll(store)
    latch = state.poll_lock if kind == "poll" else state.commit_lock
    latch.on_enter = latch.on_exit = rejected
    if kind == "poll": state.poll(store)
    else:
        with state.committing(): pass
    assert not any(state._latch_entry.values.values()) and not latch.locked()

# Real expiry and changed-identity decisions are executed, not replaced by always-valid leases.
state, actors, clock = make()
clock.mono = state.expires_deadline
with raises(LeaseLost, "lease expired"), state.committing(): pass
assert state.lost
state, actors, clock = make()
old = state.lease
with raises(LeaseLost, "changed lease identity"):
    state.poll(ReceiptStore(lambda lease, ttl: LeasePoll(replace(lease, epoch=2), 0, 0, ())))
assert state.lost and state.lease is old and not state.successful_ready

# L2: stop after committed renewal suppresses publication but preserves the returned receipt.
state, actors, clock = make()
old = state.lease
new = LeasePoll(replace(old, expires_at_ms=30000), 1000, 4, (control,))
def stopped_renew(lease, ttl):
    state.request_stop()
    return new
assert state.poll(ReceiptStore(stopped_renew)) is new
assert state.lease is old and not state.successful_ready and not state.lost
state.heartbeat_finished()
assert state.ready.is_set()
with raises(HeartbeatStopped, "heartbeat stopped"): state.poll(ReceiptStore(stopped_renew))
state.heartbeat_failed(HeartbeatStopped("normal stop"))
assert not state.lost and state.heartbeat_error is None

# L3: target matching does not freeze the token; preserve the known late-token behavior.
state, actors, clock = make()
original, replacement = state.token, Sink()
wrong = InboxItem("wrong", "control", {"action":"abort"}, "other", 2)
apply_controls(state, LeasePoll(state.lease, 0, 0, (wrong,)))
assert not original.reasons
def switch(frame, event, arg):
    if frame.f_code is apply_controls.__code__ and event == "line" and frame.f_lineno == cancel_line:
        state.token, state.turn_id = replacement, "next"
        sys.settrace(None)
    return switch
sys.settrace(switch)
try: apply_controls(state, LeasePoll(state.lease, 0, 0, (control,)))
finally: sys.settrace(None)
assert not original.reasons and replacement.reasons == ["cancel"]

# Retry budget, terminal SQLite error classification, and worker handoff guard retain their rules.
state, actors, clock = make()
failure = TimeoutError("transient")
def timeout(lease, ttl): raise failure
store = ReceiptStore(timeout)
state.heartbeat_step(store, (TimeoutError,))
assert state.heartbeat_error is failure and not state.lost and store.calls == 1
clock.mono = 114.0
with raises(TimeoutError, "transient"): state.heartbeat_step(store, (TimeoutError,))
assert store.calls == 1  # Exhausted margin stops before another external renewal.
state.heartbeat_failed(failure)
cancel_if_lost(state)
assert state.lost and state.token.reasons == ["lease_lost"]
state, actors, clock = make()
locked = sqlite3.OperationalError("database is locked")
def busy(lease, ttl): raise locked
with raises(sqlite3.OperationalError, "locked"): state.heartbeat_step(ReceiptStore(busy), (TimeoutError,))
state.heartbeat_failed(locked)
assert state.lost and state.heartbeat_error is locked
state, actors, clock = make()
state.request_stop()
state.check_closed(False)
assert not state.lost
with raises(LeaseLost, "lease cannot be released or handed off"): state.check_closed(True)
assert state.lost
assert all(n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.")
           for n in sys.modules if n.startswith("vv_agent"))
"""


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_actual_lease_rules_core_only(tmp_path, flags):
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", SCRIPT, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sdk_uses_actual_core_methods_and_public_exception():
    from vv_agent import HeartbeatStopped
    from vv_agent.kernel import _lease_scope as core
    from vv_agent.session import _driver_scope as sdk
    from vv_agent.session import kernel, timing

    assert HeartbeatStopped is kernel.HeartbeatStopped is sdk.HeartbeatStopped is core.HeartbeatStopped
    assert issubclass(sdk._LeaseHeartbeat, core.LeaseOwnership)
    assert sdk._LeaseHeartbeat.poll is core.LeaseOwnership.poll
    assert sdk._LeaseHeartbeat.committing is core.LeaseOwnership.committing
    assert sdk._LeaseHeartbeat.heartbeat_step is core.LeaseOwnership.heartbeat_step
    assert sdk.apply_controls is core.apply_controls and sdk.cancel_if_lost is core.cancel_if_lost
    assert timing.Clock is sdk.Clock is core.Clock
