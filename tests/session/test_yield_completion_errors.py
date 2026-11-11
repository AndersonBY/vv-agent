"""Completion failures remain controlled without inferring a terminal outcome."""

from threading import Event
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, DefaultAppServerHost
from vv_agent.run_handle import RunHandle
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall

from .test_children import child_task
from .test_yield_review6_edges import payload
from .test_yieldable_waits import YieldProvider


class CountingProvider(YieldProvider):
    submits = 0

    def submit(self, plan, *, context):
        self.submits += 1
        return super().submit(plan, context=context)


def parked_server(store, tmp_path, monkeypatch):
    provider = CountingProvider(declared=False, callback=None)
    calls = []

    def model(request):
        calls.append(request)
        return LLMResponse("done") if len(calls) > 1 else LLMResponse("", [ToolCall("nested-work", "child_task", {})])

    server = AppServer(
        store=store,
        host=DefaultAppServerHost(
            agent=Agent("completion", "Work.", tools=[child_task]),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m", model),
                tool_providers={"child_task": provider},
                workspace=tmp_path,
            ),
        ),
    )
    notices = []
    notify = server.run_adapter._notify_subscribers

    def capture(sid, method, params):
        assert not cast(Any, server.kernel._background_lock)._is_owned()
        notices.append((sid, method, params))
        return notify(sid, method, params)

    monkeypatch.setattr(server.run_adapter, "_notify_subscribers", capture)
    server.store.create_thread_receipt(input_id="completion", agent_key="default")
    server.run_adapter.start_turn(
        connection_id=None,
        thread_id="completion",
        input_id="start",
        expected_turn_id=None,
        input=[{"type": "text", "text": "go"}],
    )
    server.run_adapter.join()
    assert len(calls) == 1 and provider.submits == 1
    notices.clear()
    return server, provider, calls, notices


@pytest.mark.persistent_store
@pytest.mark.parametrize("damage", ["digest", "record_shape"])
def test_active_corruption_restores_exact_bytes_and_can_wake(store, tmp_path, monkeypatch, damage):
    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    original = store.read("completion", limit=1).records[0].record
    ready, release = Event(), Event()
    real_pass = RunHandle._drive_pass
    captured = []

    def pause(handle):
        if handle.session_id == "completion" and not captured:
            captured.append(handle)
            ready.set()
            assert release.wait(10)
        return real_pass(handle)

    monkeypatch.setattr(RunHandle, "_drive_pass", pause)
    try:
        server.wake("completion")
        assert ready.wait(10)
        from hashlib import sha256

        body = original.encode() if damage == "digest" else b"{}"
        checksum = "0" * 64 if damage == "digest" else sha256(body).hexdigest()
        with store.atomic():
            store._rows("UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=1", (body, checksum, "completion"))
        store._fold_cache = store._previous_prefix = None
        release.set()
        server.run_adapter.join()
        warnings = [p for sid, method, p in notices if method == "error/warning" and sid == "completion"]
        assert len(warnings) == 1 and "completion:" in warnings[0]["message"]
        assert warnings[0]["code"] == "event_stream"
        assert not any(method == "turn/completed" for _, method, _ in notices)
        assert server.run_adapter._state_manager.active_turn("completion") is None
        assert captured[0].done() and captured[0]._error is not None
        assert len(calls) == provider.submits == 1 and not provider.cancels
        with store.atomic():
            store._rows(
                "UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=1",
                (original.encode(), original.digest, "completion"),
            )
        store._fold_cache = store._previous_prefix = None
        state = store.read_state("completion")[0]
        assert state.active_turn_id and not state.turns[state.active_turn_id].ended
        server.kernel.push(
            "completion", InboxItem("healthy-result", "provider_result", payload(provider), state.active_turn_id, 0)
        )
        server.wake("completion")
        server.run_adapter.join()
        state, records, _ = store.read_state("completion")
        assert state.active_turn_id is None and len(calls) == 2
        assert (
            sum(r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == "healthy-result" for r in records)
            == 1
        )
        assert provider.submits == 1 and not provider.cancels and not store.peek_inbox("completion")
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        assert len([1 for _, method, _ in notices if method == "error/warning"]) == 1
    finally:
        release.set()
        server.run_adapter.close()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    "phase", ["result_read", "completion_read", "status_projection", "publication_read", "publication_projection"]
)
def test_completion_read_failures_report_once_then_recover(store, tmp_path, monkeypatch, phase):
    import sys
    from threading import current_thread

    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    rows = store._rows
    hits = []
    armed = True
    if phase.startswith("publication_"):
        from vv_agent.session.app_server import _NotificationPosition

        # A local publication cursor, without an independent observer racing this cut.
        server.run_adapter._notification_positions["completion"] = _NotificationPosition(0)
    if phase == "publication_projection":
        import vv_agent.session.app_server as adapter

        mapper = adapter.map_run_event

        def broken_projection(*args, **kwargs):
            frame = sys._getframe(1)
            if armed and current_thread().name == "session-app-server-pump" and frame.f_code.co_name == "_turn_notifications":
                hits.append(1)
                raise ValueError("temporary projection storage failure")
            return mapper(*args, **kwargs)

        monkeypatch.setattr(adapter, "map_run_event", broken_projection)

    def unavailable(*args, **kwargs):
        frame = sys._getframe(1)
        names = set()
        while frame:
            names.add(frame.f_code.co_name)
            frame = frame.f_back
        target = (
            (phase == "result_read" and "result" in names)
            or (phase == "completion_read" and "_complete_turn" in names)
            or (phase == "status_projection" and "_complete_turn" in names and "thread_status" in names)
            or (phase == "publication_read" and "_turn_notifications" in names)
        )
        if armed and current_thread().name == "session-app-server-pump" and target:
            hits.append(1)
            raise OSError("temporary projection storage failure")
        return rows(*args, **kwargs)

    # Inject at SQL execution, leaving completion, state reads and projections real.
    monkeypatch.setattr(store, "_rows", unavailable)
    try:
        server.wake("completion")
        server.run_adapter.join()
        assert hits == [1]
        warnings = [p for _, method, p in notices if method == "error/warning"]
        assert len(warnings) == 1 and "completion: temporary projection storage failure" in warnings[0]["message"]
        assert not any(method == "turn/completed" for _, method, _ in notices)
        assert server.run_adapter._state_manager.active_turn("completion") is None
        armed = False
        state = store.read_state("completion")[0]
        assert state.active_turn_id and not state.turns[state.active_turn_id].ended
        server.kernel.push(
            "completion", InboxItem("healthy-result", "provider_result", payload(provider), state.active_turn_id, 0)
        )
        server.wake("completion")
        server.run_adapter.join()
        assert store.read_state("completion")[0].active_turn_id is None
        assert len(calls) == 2 and provider.submits == 1 and not provider.cancels
        assert not store.peek_inbox("completion") and all(h.done() for h in server.kernel.handles)
        assert len([1 for _, method, _ in notices if method == "error/warning"]) == 1
    finally:
        armed = False
        server.run_adapter.close()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("phase", ["healthy", "worker_error", "lease_exhausted"])
def test_completion_notification_callback_errors_propagate(store, tmp_path, monkeypatch, phase):
    server, _provider, _calls, _notices = parked_server(store, tmp_path, monkeypatch)
    captured = []
    complete = server.run_adapter._complete_turn
    monkeypatch.setattr(server.run_adapter, "_complete_turn", lambda connection, started, **kwargs: captured.append(started))
    lease = None
    try:
        if phase == "lease_exhausted":
            lease = store.acquire("completion", owner="other-owner", ttl_ms=15000)
            assert lease is not None
            server.run_adapter.recover(None, "completion", lease_retry=False)
        else:
            server.wake("completion")
        server.run_adapter.join()
        assert len(captured) == 1

        callbacks = []

        def broken_notify(*args, **kwargs):
            callbacks.append(1)
            raise RuntimeError("notification callback defect")

        monkeypatch.setattr(server.run_adapter, "_notify_subscribers", broken_notify)
        with pytest.raises(RuntimeError, match="notification callback defect"):
            complete(None, captured[0], result=None, error=OSError("drive failed") if phase == "worker_error" else None)
        assert server.run_adapter._state_manager.active_turn("completion") is None
        assert callbacks == [1]
    finally:
        if lease is not None:
            store.release(lease)
        server.run_adapter.close()
        server.kernel.close()
