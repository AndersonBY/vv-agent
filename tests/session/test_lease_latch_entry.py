"""Latch guards cover entry, waiting, ownership and the release tail per Scope."""

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
import threading
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.session import kernel
from vv_agent.session.store import Lease, LeasePoll

MESSAGE = "poll while holding commit/poll latch"


def _heartbeat():
    return kernel._LeaseHeartbeat(Lease("s", "owner", 1, 15000), lambda: None, 15000, 0.25, now_ms=0)


class _Store:
    def renew(self, lease, *, ttl_ms):
        return LeasePoll(replace(lease, expires_at_ms=lease.expires_at_ms + 1), 1, 0, ())


@contextmanager
def _operation(heartbeat, kind):
    if kind == "commit":
        with heartbeat.committing():
            yield
    else:
        heartbeat.poll(_Store())
        yield


def _reject(heartbeat):
    with pytest.raises(RuntimeError) as caught:
        heartbeat.poll(_Store())
    assert str(caught.value) == MESSAGE


def _clean(heartbeat):
    assert not heartbeat.poll_lock.locked() and not heartbeat.commit_lock.locked()
    assert heartbeat._poll_thread is None and heartbeat._commit_thread is None
    heartbeat.poll(_Store())
    with heartbeat.committing():
        pass


def _case_trace(kind):
    heartbeat = _heartbeat()
    fn = kernel._LeaseHeartbeat.poll if kind == "poll" else kernel._LeaseHeartbeat.committing.__wrapped__
    lines, start = inspect.getsourcelines(fn)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    name = kind + "_lock"
    held = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == name for item in node.items)
    )
    target = start + held.body[0].lineno - 1
    facts, errors = [], []

    def trace(frame, event, arg):
        if frame.f_code is fn.__code__ and event == "line" and frame.f_lineno == target:
            sys.settrace(None)
            fact = {"kind": kind, "line": target, "locked": getattr(heartbeat, name).locked()}
            facts.append(fact)
            assert fact["locked"], "trace did not reach the acquired-latch window"
            try:
                heartbeat.poll(_Store())
            except RuntimeError as error:
                assert str(error) == MESSAGE
                fact["rejected"] = True
            else:
                fact["rejected"] = False
        return trace

    def worker():
        try:
            sys.settrace(trace)
            with _operation(heartbeat, kind):
                pass
            sys.settrace(None)
            _clean(heartbeat)
        except BaseException:
            errors.append(traceback.format_exc())
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(1)
    print("LATCH_WINDOW " + json.dumps({"facts": facts, "alive": thread.is_alive(), "errors": errors}), flush=True)
    if thread.is_alive():
        os._exit(70)
    assert not errors, errors
    assert len(facts) == 1 and facts[0]["locked"] and facts[0]["rejected"], facts
    _clean(heartbeat)


def _case_exit(kind, path):
    heartbeat = _heartbeat()
    original = getattr(heartbeat, kind + "_lock")
    primary = ValueError("controlled latch body failure")
    if path == "acquire":
        # This adapter really times out acquiring an already-held stdlib Lock;
        # its own context protocol raises, without changing stdlib semantics.
        entered, release = threading.Event(), threading.Event()
        errors = []

        def holder():
            try:
                with original:
                    entered.set()
                    assert release.wait(3)
            except BaseException:
                errors.append(traceback.format_exc())

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        try:
            assert entered.wait(3)

            class TimedLatch:
                def __enter__(self):
                    _reject(heartbeat)
                    acquired = original.acquire(timeout=0.05)
                    assert not acquired, "holder released before acquisition failure"
                    raise TimeoutError("test adapter acquisition timed out")

                def __exit__(self, *_):
                    raise AssertionError("failed acquisition must not exit")

            setattr(heartbeat, kind + "_lock", TimedLatch())
            with pytest.raises(TimeoutError, match="test adapter acquisition timed out"), _operation(heartbeat, kind):
                pass
        finally:
            release.set()
            thread.join(3)
            setattr(heartbeat, kind + "_lock", original)
        assert not thread.is_alive() and not errors, errors
    elif kind == "commit":
        with pytest.raises(ValueError) as caught, heartbeat.committing():
            raise primary
        assert caught.value is primary
    else:

        class FailingStore:
            def renew(self, lease, *, ttl_ms):
                raise primary

        with pytest.raises(ValueError) as caught:
            heartbeat.poll(FailingStore())
        assert caught.value is primary
    _clean(heartbeat)


def _case_interval(kind, phase):
    heartbeat = _heartbeat()
    actual = getattr(heartbeat, kind + "_lock")
    observed = []

    class ObservedLatch:
        def __enter__(self):
            if phase == "enter":
                _reject(heartbeat)
                observed.append("enter")
            actual.acquire()

        def __exit__(self, *_):
            actual.release()
            if phase == "released":
                assert not actual.locked()
                _reject(heartbeat)
                observed.append("released")

    setattr(heartbeat, kind + "_lock", ObservedLatch())
    try:
        with _operation(heartbeat, kind):
            pass
    finally:
        setattr(heartbeat, kind + "_lock", actual)
    assert observed == [phase]
    _clean(heartbeat)


def _case_threads(kind):
    heartbeat = _heartbeat()
    original = getattr(heartbeat, kind + "_lock")
    first, release, attempting = threading.Event(), threading.Event(), threading.Event()
    inputs, outputs, errors = [], [], []
    active = maximum = 0

    class Store:
        def renew(self, lease, *, ttl_ms):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            inputs.append(lease)
            if kind == "poll" and len(inputs) == 1:
                first.set()
                assert release.wait(3)
            result = _Store().renew(lease, ttl_ms=ttl_ms)
            outputs.append(result.lease)
            active -= 1
            return result

    class WaitingLatch:
        def __enter__(self):
            if threading.current_thread().name == "second":
                _reject(heartbeat)
                attempting.set()
            original.acquire()

        def __exit__(self, *_):
            original.release()

    setattr(heartbeat, kind + "_lock", WaitingLatch())

    def worker(which):
        try:
            if kind == "poll":
                heartbeat.poll(Store())
            else:
                with heartbeat.committing():
                    if which == "first":
                        first.set()
                        assert release.wait(3)
        except BaseException:
            errors.append(traceback.format_exc())

    a = threading.Thread(target=worker, args=("first",), name="first", daemon=True)
    b = threading.Thread(target=worker, args=("second",), name="second", daemon=True)
    a.start()
    try:
        assert first.wait(3)
        owner = heartbeat._poll_thread if kind == "poll" else heartbeat._commit_thread
        assert owner == a.ident
        b.start()
        assert attempting.wait(3), errors
        assert (heartbeat._poll_thread if kind == "poll" else heartbeat._commit_thread) == a.ident
        if kind == "commit":
            # Another thread's C marker must not reject this thread's poll.
            heartbeat.poll(Store())
    finally:
        release.set()
        a.join(3)
        if b.ident is not None:
            b.join(3)
        setattr(heartbeat, kind + "_lock", original)
    assert not a.is_alive() and not b.is_alive() and not errors, errors
    if kind == "poll":
        assert maximum == 1 and len(inputs) == 2 and inputs[1] is outputs[0]
        assert heartbeat.lease is outputs[1]
    _clean(heartbeat)


def _case_scopes(kind):
    first, second = _heartbeat(), _heartbeat()
    assert first is not second
    if kind == "commit":
        with first.committing():
            second.poll(_Store())
    else:

        class Store:
            def renew(self, lease, *, ttl_ms):
                second.poll(_Store())
                return _Store().renew(lease, ttl_ms=ttl_ms)

        first.poll(Store())
    _clean(first)
    _clean(second)


def _run_child(tmp_path, case, *args):
    source = str(Path(kernel.__file__).resolve().parents[2])
    code = (
        "import runpy,sys;sys.path.insert(0,"
        + repr(source)
        + ");"
        + "scope=runpy.run_path("
        + repr(str(Path(__file__).resolve()))
        + ");"
        + "scope["
        + repr("_case_" + case)
        + "](*"
        + repr(args)
        + ")"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-W", "error", "-c", code],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "LANG": "C.UTF-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    print(result.stdout, end="")
    assert result.returncode == 0, "latch child failed: " + result.stdout
    assert "Exception in thread" not in result.stdout and "Exception ignored in" not in result.stdout


@pytest.mark.parametrize("kind", ["poll", "commit"])
def test_trace_acquired_latch_rejects_poll_before_owner_registration(tmp_path, kind):
    _run_child(tmp_path, "trace", kind)


@pytest.mark.parametrize("kind", ["poll", "commit"])
@pytest.mark.parametrize("path", ["body", "acquire"])
def test_entry_mark_clears_after_body_or_acquisition_failure(tmp_path, kind, path):
    _run_child(tmp_path, "exit", kind, path)


@pytest.mark.parametrize("kind", ["poll", "commit"])
@pytest.mark.parametrize("phase", ["enter", "released"])
def test_entry_mark_covers_lock_entry_and_release_tail(tmp_path, kind, phase):
    _run_child(tmp_path, "interval", kind, phase)


@pytest.mark.parametrize("kind", ["poll", "commit"])
def test_waiters_are_thread_local_and_poll_remains_serial(tmp_path, kind):
    _run_child(tmp_path, "threads", kind)


@pytest.mark.parametrize("kind", ["poll", "commit"])
def test_independent_scopes_do_not_reject_each_other(tmp_path, kind):
    _run_child(tmp_path, "scopes", kind)
