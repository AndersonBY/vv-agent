"""Cancellation traces use the real SDK and core methods with stdlib synchronization."""

from __future__ import annotations

import pytest


def cancellation_trace(factory):
    from threading import Event

    trace = []
    token = factory()
    assert list(vars(token))[:4] == ["_event", "_callbacks", "_lock", "_reason"]
    trace.append(["initial", token.cancelled, token.reason, token._lock.locked()])
    token.check()

    class ObservedEvent(Event):
        def set(self):
            trace.append(["publish", token._reason, self.is_set(), token._lock.locked(), len(token._callbacks)])
            super().set()

    token._event = ObservedEvent()

    def observe(name, current=token):
        trace.append([name, current.cancelled, current.reason, current._lock.locked(), len(current._callbacks)])

    def first():
        observe("first")
        token.on_cancel(lambda: observe("nested-immediate"))
        token.cancel("must not overwrite")

    token.on_cancel(first)
    for failure in (KeyboardInterrupt, SystemExit, GeneratorExit, StopIteration):

        def raising(failure=failure):
            observe(failure.__name__)
            raise failure("callback failure")

        token.on_cancel(raising)
    child = token.child()
    grandchild = child.child()
    child.on_cancel(lambda: observe("child", child))
    grandchild.on_cancel(lambda: observe("grandchild", grandchild))
    token.on_cancel(lambda: observe("last"))
    token.cancel(" \t first reason \n")
    token.cancel("second reason")
    token.on_cancel(lambda: observe("late-immediate"))
    for failure in (KeyboardInterrupt, SystemExit, GeneratorExit, StopIteration):

        def raising_late(failure=failure):
            observe("late-" + failure.__name__)
            raise failure("late callback failure")

        token.on_cancel(raising_late)
    late_child = token.child()
    observe("late-child", late_child)
    for current in (token, child, grandchild, late_child):
        try:
            current.check()
        except Exception as error:
            trace.append(["check", type(error).__module__, type(error).__name__, str(error)])
        else:
            raise AssertionError("cancelled check returned")
    parent = factory()
    independent = parent.child()
    independent.cancel("child reason")
    trace.append(["child-only", parent.cancelled, parent.reason, independent.reason])
    parent.cancel("parent reason")
    trace.append(["first-reasons", parent.reason, independent.reason])
    for reason in (None, "", " \t\n", "\u00a0 trimmed \n"):
        current = factory()
        if reason is None:
            current.cancel()
        else:
            current.cancel(reason)
        trace.append(["reason", current.reason])
    return trace


@pytest.fixture(params=["sdk", "core"])
def token_factory(request):
    if request.param == "sdk":
        from vv_agent.runtime.cancellation import CancellationToken

        return CancellationToken
    from threading import Event, Lock

    from vv_agent.kernel._cancellation import Cancellation

    def factory():
        return Cancellation(signal=Event(), lock=Lock(), child_factory=factory)

    return factory


def test_callback_state_and_child_trace(token_factory):
    trace = cancellation_trace(token_factory)
    assert trace[0] == ["initial", False, None, False]
    assert trace[1] == ["publish", "first reason", False, True, 7]
    callbacks = [entry for entry in trace if len(entry) == 5 and entry[0] != "publish"]
    assert [entry[0] for entry in callbacks] == [
        "first",
        "nested-immediate",
        "KeyboardInterrupt",
        "SystemExit",
        "GeneratorExit",
        "StopIteration",
        "grandchild",
        "child",
        "last",
        "late-immediate",
        "late-KeyboardInterrupt",
        "late-SystemExit",
        "late-GeneratorExit",
        "late-StopIteration",
        "late-child",
    ]
    assert all(entry[1:] == [True, "first reason", False, 0] for entry in callbacks)
    assert trace[-6:] == [
        ["child-only", False, None, "child reason"],
        ["first-reasons", "parent reason", "child reason"],
        ["reason", "Operation was cancelled"],
        ["reason", "Operation was cancelled"],
        ["reason", "Operation was cancelled"],
        ["reason", "trimmed"],
    ]


def test_callback_block_does_not_hold_lock_or_delay_late_registration(token_factory):
    from threading import Event, Thread

    token = token_factory()
    entered, release = Event(), Event()
    trace = []

    def first():
        trace.append("first-enter")
        entered.set()
        release.wait(3)
        trace.append("first-exit")

    token.on_cancel(first)
    token.on_cancel(lambda: trace.append("second"))
    worker = Thread(target=lambda: token.cancel("first reason"), daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert token._lock.acquire(blocking=False)
        token._lock.release()
        token.cancel("later reason")
        token.on_cancel(lambda: trace.append("late"))
        assert trace == ["first-enter", "late"]
        assert token.reason == "first reason"
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert trace == ["first-enter", "late", "first-exit", "second"]


def test_sdk_child_stays_sdk_type_for_subclass_parent():
    from vv_agent.runtime.cancellation import CancellationToken

    class SpecializedToken(CancellationToken):
        def __init__(self, required):
            super().__init__()
            self.required = required

    parent = SpecializedToken("parent-only")
    child = parent.child()
    assert type(child) is CancellationToken
    parent.cancel("inherited")
    assert child.reason == "inherited"
    assert type(parent.child()) is CancellationToken


# Actual SDK callback/state trace captured at committed 618889b before extraction.
BASELINE_TRACE_SHA256 = "bd9d58e8ba2f44428d76ba320189e68aa5451907826a5be8e57c8bbfcd53294e"


def test_sdk_and_core_match_committed_sdk_trace(token_factory):
    import hashlib
    import json

    body = json.dumps(cancellation_trace(token_factory), ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(body).hexdigest() == BASELINE_TRACE_SHA256


def test_sdk_methods_public_signatures_and_exception_identity():
    import inspect
    import pickle
    import sys
    from typing import assert_type, get_type_hints

    from vv_agent.kernel._cancellation import Cancellation
    from vv_agent.kernel._cancellation import CancelledError as CoreCancelledError
    from vv_agent.runtime import CancellationToken, CancelledError

    expected = {
        "__init__": "(self) -> 'None'",
        "cancel": "(self, reason: 'str' = 'Operation was cancelled') -> 'None'",
        "check": "(self) -> 'None'",
        "on_cancel": "(self, cb: 'Callable[[], None]') -> 'None'",
        "child": "(self) -> 'CancellationToken'",
        "_invoke_callback": "(callback: 'Callable[[], None]') -> 'None'",
    }
    assert str(inspect.signature(CancellationToken)) == "() -> 'None'"
    assert {name: str(inspect.signature(getattr(CancellationToken, name))) for name in expected} == expected
    for name in ("cancelled", "reason", "cancel", "check", "on_cancel", "_invoke_callback"):
        assert getattr(CancellationToken, name) is getattr(Cancellation, name)
    assert CancelledError is CoreCancelledError
    assert CancelledError.__module__ == "vv_agent.runtime.cancellation"
    parent = CancellationToken()
    base_calls = []

    def observe_call(frame, event, arg):
        if event == "call" and frame.f_code is Cancellation.child.__code__:
            base_calls.append(frame.f_locals["self"])

    previous_profile = sys.getprofile()
    sys.setprofile(observe_call)
    try:
        child = assert_type(parent.child(), CancellationToken)
    finally:
        sys.setprofile(previous_profile)
    assert base_calls == [parent]
    assert_type(parent.cancelled, bool)
    assert_type(parent.reason, str | None)
    parent.cancel("typed reason")
    with pytest.raises(CancelledError) as raised:
        child.check()
    assert type(raised.value) is CoreCancelledError
    restored = pickle.loads(pickle.dumps(raised.value))
    assert type(restored) is CancelledError and restored.args == ("typed reason",)
    assert get_type_hints(CancellationToken.child)["return"] is CancellationToken
    assert get_type_hints(Cancellation.child)["return"] is Cancellation


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_before_sdk_import_with_actual_host_synchronization(tmp_path, flags):
    import inspect
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = r"""
import importlib.abc
import json
import sys
from pathlib import Path
from threading import Event, Lock
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._cancellation import Cancellation, CancelledError

def factory():
    signal, lock = Event(), Lock()
    token = Cancellation(signal=signal, lock=lock, child_factory=factory)
    assert token._event is signal and token._lock is lock
    return token

def audit(event, args):
    if event in {'open', 'os.listdir', 'os.scandir', 'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'}:
        raise AssertionError('unexpected core effect: ' + event)
sys.addaudithook(audit)
"""
    script += "\n" + inspect.getsource(cancellation_trace)
    script += r"""
trace = cancellation_trace(factory)
assert CancelledError.__module__ == 'vv_agent.runtime.cancellation'
assert all(n in {'vv_agent', 'vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
print(json.dumps(trace, ensure_ascii=False))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from vv_agent.runtime.cancellation import CancellationToken

    assert json.loads(result.stdout) == cancellation_trace(CancellationToken)


def test_actual_sim_scope_uses_core_methods_on_sql_lease_loss(tmp_path):
    from support.sdk_runtime import SDKWorld, sdk_scope

    from vv_agent.kernel._cancellation import Cancellation
    from vv_agent.runtime.cancellation import CancellationToken
    from vv_agent.session.store import LeaseLost

    with SDKWorld(tmp_path / "cancellation.sqlite", seed=17) as world:
        world.create()
        scope = sdk_scope(world, participant="owner")
        assert type(scope.token) is CancellationToken
        for name in ("cancel", "on_cancel", "check"):
            assert getattr(type(scope.token), name) is getattr(Cancellation, name)
        child = scope.token.child()
        world.expire()
        successor = world.connection("successor").acquire("s", owner="successor", ttl_ms=scope.ttl_ms)
        assert successor is not None and successor.epoch > scope.lease.epoch
        with pytest.raises(LeaseLost):
            world.keepers["owner"].poll(scope)
        assert scope.token.cancelled and child.cancelled
        assert scope.token.reason == child.reason == "lease_lost"


def test_sdk_child_annotation_resolves_in_fresh_process(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = """
import sys
from typing import get_type_hints
sys.path.insert(0, sys.argv[1])
from vv_agent.runtime.cancellation import CancellationToken
assert get_type_hints(CancellationToken.child)["return"] is CancellationToken
assert type(CancellationToken().child()) is CancellationToken
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(Path(__file__).resolve().parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
