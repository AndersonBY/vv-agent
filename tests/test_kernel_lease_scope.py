"""Actual core lease rules remain host-free, lock-free and snapshot based."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = r"""
import importlib.abc
import sqlite3
import sys
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

class Flag:
    def __init__(self): self.value = False
    def is_set(self): return self.value
    def set(self): self.value = True
    def wait(self, timeout=None):
        assert timeout is None or timeout >= 0
        return self.value

class ManualClock:
    def __init__(self):
        self.wall = 1_000
        self.mono = 100.0
        self.calls = []
    def now_ms(self): self.calls.append("wall"); return self.wall
    def monotonic(self): self.calls.append("mono"); return self.mono
    def monotonic_ns(self): return int(self.mono * 1_000_000_000)
    def sleep(self, seconds): raise AssertionError(f"core scheduled sleep({seconds})")

class Sink:
    def __init__(self, on_cancel=None): self.reasons, self.on_cancel = [], on_cancel
    def cancel(self, reason):
        self.reasons.append(reason)
        if self.on_cancel is not None: self.on_cancel()

class ReceiptStore:
    def __init__(self, callback): self.callback, self.calls = callback, 0
    def renew(self, lease, *, ttl_ms):
        self.calls += 1
        return self.callback(lease, ttl_ms)

@contextmanager
def raises(kind, message):
    try:
        yield
    except kind as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("expected " + kind.__name__)

def make(tokens=None):
    clock = ManualClock()
    lease = Lease("s", "owner", 1, 5_000)
    if tokens is None:
        shared = Sink()
        token_factory = lambda: shared
    else:
        tokens = list(tokens)
        token_factory = lambda: tokens.pop(0)
    state = LeaseOwnership(
        lease,
        4_000,
        0.04,
        clock=clock,
        stop=Flag(),
        ready=Flag(),
        now_ms=clock.wall,
        token_factory=token_factory,
    )
    state.bind_turn("turn", 1, {})
    assert not any(hasattr(state, name) for name in ("lock", "poll_lock", "commit_lock", "thread"))
    assert state.latest_poll is None
    assert state.expires_deadline == 104.0 and state.safety_deadline == 103.0
    with state.committing() as anchored:
        assert anchored is lease
    return state, clock, lease

# A renewal publishes lease, database time and controls as one immutable receipt.
state, clock, old = make()
control = InboxItem("c", "control", {"action": "cancel"}, target_turn_id="turn", generation=1, actor={})
receipt = LeasePoll(replace(old, expires_at_ms=8_000), 2_000, 3, (control,))
store = ReceiptStore(lambda lease, ttl: receipt)
assert state.poll(store) is receipt
assert state.latest_poll is receipt and state.lease is receipt.lease
assert state.latest_poll.db_now_ms == 2_000 and state.latest_poll.controls == (control,)

# Consumed control ids never cancel again; the applied mapping is an uncopied view.
state, clock, old = make()
applied = {"c": object()}
state.bind_turn("turn", 1, applied)
assert state.control_binding.applied_inputs is applied
apply_controls(state, receipt)
assert not state.token.reasons
state.bind_turn("turn", 1, {})
apply_controls(state, receipt)
assert state.token.reasons == ["cancel"]

state, clock, old = make()
assert state.poll(store) is receipt

# Committing only yields the latest lease. A SQL epoch fence remains the authority.
with state.committing() as lease:
    assert lease is receipt.lease
sql_epoch = 1
def fenced_append(lease):
    if lease.epoch != sql_epoch:
        raise LeaseLost("stale lease epoch")
with state.committing() as lease:
    sql_epoch = 2
    with raises(LeaseLost, "stale lease epoch"):
        fenced_append(lease)

# Stop racing a committed renewal retains its latest receipt for close handoff.
def stopped(lease, ttl):
    state.request_stop()
    return LeasePoll(replace(lease, expires_at_ms=9_000), 3_000, 4, ())
stopped_receipt = state.poll(ReceiptStore(stopped))
assert stopped_receipt.lease is state.lease and state.latest_poll is stopped_receipt
assert state.successful_ready and not state.lost
state.heartbeat_finished()
assert state.ready.is_set()
with raises(HeartbeatStopped, "heartbeat stopped"):
    state.poll(ReceiptStore(lambda lease, ttl: receipt))
state.heartbeat_failed(HeartbeatStopped("normal stop"))
assert not state.lost and state.heartbeat_error is None

# Changed identity and elapsed expiry are loss, not retryable contention.
state, clock, old = make()
with raises(LeaseLost, "changed lease identity"):
    state.poll(ReceiptStore(lambda lease, ttl: LeasePoll(replace(lease, epoch=2), 1_500, 0, ())))
assert state.lost and state.lease is old
state, clock, old = make()
state.poll(ReceiptStore(lambda lease, ttl: LeasePoll(replace(lease, expires_at_ms=8_000), 2_000, 0, ())))
clock.mono = state.expires_deadline
with raises(LeaseLost, "lease expired"), state.committing():
    pass
assert state.lost

# A late control cancels the token captured for its original turn only.
replacement = Sink()
state, clock, old = make([Sink(), Sink(), replacement])
wrong = InboxItem("wrong", "control", {"action": "abort"}, target_turn_id="other", generation=2, actor={})
apply_controls(state, LeasePoll(state.lease, 0, 0, (wrong,)))
assert not state.token.reasons
state.token.on_cancel = state.reset_token
apply_controls(state, LeasePoll(state.lease, 0, 0, (control,)))
assert state.token is replacement and state.turn_id == "turn" and state.generation == 1
assert not replacement.reasons

# Transient exhaustion is retryable before the ratio safety deadline, then loss.
state, clock, old = make()
failure = TimeoutError("transient")
timeout_store = ReceiptStore(lambda lease, ttl: (_ for _ in ()).throw(failure))
state.heartbeat_step(timeout_store, (TimeoutError,))
assert state.heartbeat_error is failure and not state.lost and timeout_store.calls == 1
clock.mono = state.safety_deadline
with raises(LeaseLost, "safety deadline"):
    state.heartbeat_step(timeout_store, (TimeoutError,))
assert timeout_store.calls == 1
state.heartbeat_failed(state.heartbeat_error)
cancel_if_lost(state)
assert state.lost and state.token.reasons == ["lease_lost"]

# Non-lock storage errors stay raw; the wrapper classifies only explicit lock codes.
state, clock, old = make()
locked = sqlite3.OperationalError("disk I/O error")
with raises(sqlite3.OperationalError, "disk I/O error"):
    state.heartbeat_step(ReceiptStore(lambda lease, ttl: (_ for _ in ()).throw(locked)), (TimeoutError,))

# Close cannot hand off while a background renewal remains alive.
state, clock, old = make()
state.request_stop()
state.check_closed(False)
assert not state.lost
with raises(LeaseLost, "lease cannot be released or handed off"):
    state.check_closed(True)
assert state.lost

assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
           for name in sys.modules if name.startswith("vv_agent"))
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
    assert sdk._LeaseHeartbeat.confirmed_poll is core.LeaseOwnership.confirmed_poll
    assert sdk._LeaseHeartbeat.bind_turn is core.LeaseOwnership.bind_turn
    assert sdk._LeaseHeartbeat.reset_token is core.LeaseOwnership.reset_token
    assert sdk._Scope.heartbeat_confirmed is not core.LeaseOwnership.heartbeat_confirmed
    assert sdk.apply_controls is core.apply_controls and sdk.cancel_if_lost is core.cancel_if_lost
    assert timing.Clock is sdk.Clock is core.Clock
