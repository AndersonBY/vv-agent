"""Observe this native test's accepted hint and finite natural completion graph."""

import json
import traceback
from threading import Event, Thread, current_thread, get_ident
from time import monotonic_ns

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session.app_server import _KernelRunAdapter


class NativeCohort:
    def __init__(self, adapter, store, monkeypatch):
        self.adapter, self.store = adapter, store
        self.hint = None
        self.handoff, self.owner_wait = Event(), Event()
        self.events = []
        self.waited = []
        self.pump_exits = []
        defer, recover, drain, pump = (
            adapter._defer_recovery,
            adapter.recover,
            adapter._drain_recoveries,
            adapter._pump_events,
        )

        def deferred(*args, **kwargs):
            result = defer(*args, **kwargs)
            if result:
                with adapter._notification_lock:
                    hint = adapter._pending_recoveries.get("completion")
                    if hint is not None:
                        assert self.hint is None
                        self.hint = hint
                        self.observe("pending", hint=id(hint), sources=[id(s) for s in hint[4]])
            return result

        def recovered(*args, **kwargs):
            result = recover(*args, **kwargs)
            with adapter._notification_lock:
                if "completion" in adapter._recovering:
                    self.observe("processing_owner_registered", hint=id(self.hint))
            return result

        def drained():
            result = drain()
            with adapter._notification_lock:
                if (
                    self.hint is not None
                    and "completion" not in adapter._pending_recoveries
                    and "completion" not in adapter._recovering
                    and not self.handoff.is_set()
                ):
                    self.observe("hint_handoff_exit", hint=id(self.hint))
                    self.handoff.set()
            return result

        def pumped(connection, started, stream):
            try:
                return pump(connection, started, stream)
            finally:
                self.pump_exits.append((current_thread(), started.handle))
                self.observe("pump_exit", owner=id(started.handle))

        monkeypatch.setattr(adapter, "_defer_recovery", deferred)
        monkeypatch.setattr(adapter, "recover", recovered)
        monkeypatch.setattr(adapter, "_drain_recoveries", drained)
        monkeypatch.setattr(adapter, "_pump_events", pumped)

    def observe(self, point, **values):
        self.events.append(
            {
                "point": point,
                "ns": monotonic_ns(),
                "thread": get_ident(),
                "handles": [id(h) for h in self.adapter.kernel.handles],
                "pumps": [id(p) for p in self.adapter._pumps],
                **values,
            }
        )

    def outside(self):
        assert not self.adapter.kernel._background_lock._is_owned()
        assert not self.store._connection_lock._is_owned()
        assert not any(s.turn_lock._is_owned() for s in self.adapter._state_manager._states.values())

    def wait(self):
        self.outside()
        assert self.hint is not None
        assert self.handoff.wait(5)
        assert any(e["point"] == "processing_owner_registered" for e in self.events)
        # Three owners bound this case: parked, first wake, one deferred hint.
        # A pump returns only after synchronous completion handoff/registration.
        for _ in range(3):
            remaining = [p for p in self.adapter._pumps if p not in self.waited]
            if not remaining:
                break
            pump = remaining[0]
            owner = self.adapter.kernel.handles[self.adapter._pumps.index(pump)]
            self.observe("registered_owner_wait", pump=id(pump), owner=id(owner), done=owner._done)
            if not owner.done():
                self.owner_wait.set()
            self.outside()
            pump.join(5)
            assert not pump.is_alive()
            self.waited.append(pump)
        assert self.waited == self.adapter._pumps
        assert len(self.adapter.kernel.handles) == len(self.waited) <= 3
        for handle in self.adapter.kernel.handles:
            self.outside()
            handle.join(5)
        self.observe("finite_cutoff")
        assert not self.adapter._notification_stop.is_set()
        assert not self.adapter._pending_recoveries and not self.adapter._recovering
        assert not self.adapter._lease_retries
        assert self.adapter._state_manager.active_turn("completion") is None
        assert all(h.done() and h._error is None and not h._thread.is_alive() for h in self.adapter.kernel.handles)
        assert all(h.runtime.hook is h._original_hook for h in self.adapter.kernel.handles)

    def snapshot(self):
        a = self.adapter
        return {
            "events": list(self.events),
            "hint": id(self.hint),
            "handoff": self.handoff.is_set(),
            "waited": [id(p) for p in self.waited],
            "handles": [
                {
                    "id": id(h),
                    "sid": h.session_id,
                    "tid": h.run_id,
                    "done": h._done,
                    "error": None
                    if h._error is None
                    else {
                        "id": id(h._error),
                        "type": type(h._error).__name__,
                        "trace": "".join(traceback.format_exception(h._error)),
                    },
                    "thread": h._thread.ident,
                    "alive": h._thread.is_alive(),
                    "runtime": id(h.runtime),
                    "hook_restored": h.runtime.hook is h._original_hook,
                    "E": id(getattr(h, "_execution_store", None)),
                    "E_raw": id(getattr(getattr(h, "_execution_store", None), "connection", None)),
                    "owner_wake": h._owner_wake_requested,
                    "rerun": h._rerun_requested,
                }
                for h in a.kernel.handles
            ],
            "pumps": [{"id": id(p), "thread": p.ident, "alive": p.is_alive()} for p in a._pumps],
            "pump_exits": [{"pump": id(p), "owner": id(h)} for p, h in self.pump_exits],
            "observer": None
            if a._notification_thread is None
            else {"id": id(a._notification_thread), "alive": a._notification_thread.is_alive()},
            "stop": a._notification_stop.is_set(),
            "pending": {sid: id(hint) for sid, hint in a._pending_recoveries.items()},
            "processing": list(a._recovering),
        }


@pytest.mark.persistent_store
def test_fix4_deferred_observer_after_join_waits_for_natural_owner(store, tmp_path, monkeypatch):
    from . import test_connection_ownership_native_commit as native

    reached, allow_observer, worker_reached, allow_worker = [Event() for _ in range(4)]
    selected, late, audits, controller_errors = [], [], [], []
    original_drain, original_join, original_pass = (
        _KernelRunAdapter._drain_recoveries,
        _KernelRunAdapter.join,
        RunHandle._drive_pass,
    )

    def drain(adapter):
        if adapter._pending_recoveries and not selected:
            cohort = adapter._native_test_cohort
            cohort.outside()
            selected.append(cohort)
            reached.set()
            assert allow_observer.wait(5)
        return original_drain(adapter)

    def passed(handle):
        if selected and allow_observer.is_set() and not late:
            selected[0].outside()
            late.append(handle)
            worker_reached.set()
            assert allow_worker.wait(5)
        return original_pass(handle)

    def joined(adapter):
        result = original_join(adapter)
        if adapter._pending_recoveries and not allow_observer.is_set():
            assert reached.wait(5)
            cohort = selected[0]
            cohort.outside()
            audits.append({"original_join_return": cohort.snapshot()})
            assert len(adapter.kernel.handles) == 2
            assert all(h.done() and h._error is None for h in adapter.kernel.handles)
            allow_observer.set()
            assert worker_reached.wait(5)
            controller.start()
        return result

    def release_after_acceptance_wait():
        try:
            assert worker_reached.wait(5)
            cohort = selected[0]
            assert cohort.owner_wait.wait(5)
            assert cohort.handoff.is_set() and not late[0].done() and late[0]._error is None
            audits.append({"acceptance_waiting_for_worker": cohort.snapshot()})
        except BaseException as exc:
            controller_errors.append(exc)
        finally:
            allow_worker.set()

    monkeypatch.setattr(_KernelRunAdapter, "_drain_recoveries", drain)
    monkeypatch.setattr(_KernelRunAdapter, "join", joined)
    monkeypatch.setattr(RunHandle, "_drive_pass", passed)
    controller = Thread(target=release_after_acceptance_wait, name="native-cohort-test-controller")
    try:
        native.test_native_companion_accepted_work(store, tmp_path, monkeypatch, True)
        assert all(h.done() and h._error is None and not h._thread.is_alive() for h in selected[0].adapter.kernel.handles)
    finally:
        allow_observer.set()
        allow_worker.set()
        if controller.ident is not None:
            controller.join(5)
        audits.append({"controller_errors": ["".join(traceback.format_exception(e)) for e in controller_errors]})
        with (tmp_path / "controlled-lifecycle.json").open("x") as out:
            json.dump(audits, out, indent=2)
    assert not controller.is_alive()
    if controller_errors:
        raise controller_errors[0]
    assert len(late) == 1 and len(audits) == 3
