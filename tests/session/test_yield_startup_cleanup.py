"""Unstarted App Server owners fail once and leave durable replies recoverable."""

from contextlib import contextmanager
from threading import current_thread

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session.records import InboxItem

from .test_yield_completion_errors import parked_server
from .test_yield_review6_edges import payload


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    "phase",
    [
        "transaction_exit",
        "projection",
        "status",
        "active",
        "response",
        "notification",
        "handle_start",
        "pump_start",
        "pump_started_raise",
    ],
)
def test_unstarted_owner_failure_can_consume_healthy_reply(store, tmp_path, monkeypatch, phase):
    import vv_agent.session.app_server as module

    server, provider, calls, _notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    before = len(server.kernel.handles)
    start_handle = RunHandle.start
    error = RuntimeError(f"startup {phase}")
    hits = []

    def fail():
        hits.append(1)
        raise error

    if phase == "transaction_exit":
        atomic = store.atomic

        @contextmanager
        def transaction_exit():
            with atomic() as tx:
                yield tx
                if not hits and len(server.kernel.handles) > before:
                    fail()

        monkeypatch.setattr(store, "atomic", transaction_exit)
    elif phase in {"projection", "status"}:
        target = adapter._store if phase == "projection" else adapter
        method = "_turn_record" if phase == "projection" else "public_thread_status"
        monkeypatch.setattr(target, method, lambda *a, **kw: fail())
    elif phase == "active":
        publish = adapter._state_manager.set_active_turn

        def publish_then_fail(**kwargs):
            publish(**kwargs)
            fail()

        monkeypatch.setattr(adapter._state_manager, "set_active_turn", publish_then_fail)
    elif phase == "response":
        monkeypatch.setattr(adapter._router, "send_response", lambda *a, **kw: fail())
    elif phase == "notification":
        notify = adapter._notify_subscribers

        def broken(sid, method, params):
            if method == "thread/status/changed" and current_thread().name == "MainThread":
                fail()
            return notify(sid, method, params)

        monkeypatch.setattr(adapter, "_notify_subscribers", broken)
    elif phase == "handle_start":
        monkeypatch.setattr(RunHandle, "start", lambda h: fail())
    else:
        start = module.threading.Thread.start

        def pump_start(thread):
            if thread.name == "session-app-server-pump":
                if phase == "pump_started_raise":
                    start(thread)
                fail()
            return start(thread)

        monkeypatch.setattr(module.threading.Thread, "start", pump_start)
    try:
        # Exercise startup directly so the exact exception, rather than a warning
        # emitted by the public recovery boundary, is observable here.
        with pytest.raises(RuntimeError) as raised:
            adapter._start(None, "completion", None, request_id="test" if phase == "response" else None)
        assert raised.value is error and hits == [1]
        h = server.kernel.handles[-1]
        assert len(server.kernel.handles) == before + 1
        assert h.done() and h._error is error and h._thread.ident is None
        assert adapter._state_manager.active_turn("completion") is None
        with pytest.raises(RuntimeError) as result_error:
            h.result(timeout=0)
        assert result_error.value is error
        with pytest.raises(RuntimeError) as restart_error:
            start_handle(h)
        assert restart_error.value is error and h._thread.ident is None
        monkeypatch.undo()
        tid = store.read_state("completion")[0].active_turn_id
        server.kernel.push("completion", InboxItem("healthy-reply", "provider_result", payload(provider), tid, 0))
        server.wake("completion")
        server.wake("completion")
        adapter.join()
        state, records, _ = store.read_state("completion")
        assert state.active_turn_id is None and len(calls) == 2
        assert not store.peek_inbox("completion") and provider.submits == 1 and not provider.cancels
        assert (
            sum(r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == "healthy-reply" for r in records)
            == 1
        )
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        assert all(not p.is_alive() for p in adapter._pumps)
    finally:
        monkeypatch.undo()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
def test_exception_after_worker_start_preserves_completion_owner(store, tmp_path, monkeypatch):
    from threading import Event

    server, provider, calls, _ = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    started, release = Event(), Event()
    real_pass = RunHandle._drive_pass
    real_start = RunHandle.start
    error = RuntimeError("caller failed after worker started")
    armed = []

    def gated(handle):
        started.set()
        assert release.wait(5)
        return real_pass(handle)

    def start_then_fail(handle):
        real_start(handle)
        if not armed:
            armed.append(1)
            assert started.wait(5)
            raise error

    monkeypatch.setattr(RunHandle, "_drive_pass", gated)
    monkeypatch.setattr(RunHandle, "start", start_then_fail)
    try:
        with pytest.raises(RuntimeError) as raised:
            server.wake("completion")
        assert raised.value is error
        h = server.kernel.handles[-1]
        assert h._thread.ident is not None and not h.done() and h._error is None
        assert adapter._state_manager.active_turn("completion").handle is h
        # A healthy wake reaches the existing owner without a second dispatch.
        server.wake("completion")
        release.set()
        adapter.join()
        assert h.done() and h._error is None
        assert adapter._state_manager.active_turn("completion") is None
        monkeypatch.undo()
        tid = store.read_state("completion")[0].active_turn_id
        server.kernel.push("completion", InboxItem("reply-after-start", "provider_result", payload(provider), tid, 0))
        server.wake("completion")
        adapter.join()
        assert store.read_state("completion")[0].active_turn_id is None
        assert len(calls) == 2 and provider.submits == 1 and not provider.cancels
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        assert all(not p.is_alive() for p in adapter._pumps)
    finally:
        release.set()
        monkeypatch.undo()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("phase", ["read", "preparation", "notification"])
def test_public_wake_callback_error_is_exposed_once(store, tmp_path, monkeypatch, phase):
    server, _provider, _calls, _ = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    error = RuntimeError("host callback defect")
    callbacks = []
    if phase == "read":
        monkeypatch.setattr(store, "read_state", lambda *a, **kw: (_ for _ in ()).throw(OSError("read fault")))
    elif phase == "preparation":
        monkeypatch.setattr(adapter, "public_thread_status", lambda *a: (_ for _ in ()).throw(OSError("projection fault")))

    def broken(*args):
        callbacks.append(1)
        raise error

    monkeypatch.setattr(adapter, "_notify_subscribers", broken)
    try:
        with pytest.raises(RuntimeError) as raised:
            server.wake("completion")
        assert raised.value is error and callbacks == [1]
        assert adapter._state_manager.active_turn("completion") is None
        assert all(h.done() for h in server.kernel.handles)
    finally:
        monkeypatch.undo()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
def test_failed_publication_cannot_clear_a_different_real_owner(store, tmp_path, monkeypatch):
    from threading import Event

    server, provider, calls, _ = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    publish = adapter._state_manager.set_active_turn
    drive = RunHandle._drive_pass
    error = RuntimeError("publication caller failed")
    cut, successor = [], []
    release = Event()

    def gated(handle):
        assert release.wait(5)
        return drive(handle)

    def replace_owner(**kwargs):
        publish(**kwargs)
        if not cut:
            cut.append(kwargs["handle"])
            assert adapter._state_manager.clear_active_turn("completion", kwargs["turn_id"], handle=kwargs["handle"])
            successor.append(adapter._start(None, "completion", None).handle)
            raise error

    monkeypatch.setattr(RunHandle, "_drive_pass", gated)
    monkeypatch.setattr(adapter._state_manager, "set_active_turn", replace_owner)
    try:
        with pytest.raises(RuntimeError) as raised:
            adapter._start(None, "completion", None)
        assert raised.value is error
        assert cut[0].done() and cut[0]._error is error and cut[0]._thread.ident is None
        assert successor[0]._thread.ident is not None and not successor[0].done()
        assert adapter._state_manager.active_turn("completion").handle is successor[0]
        release.set()
        adapter.join()
        monkeypatch.undo()
        tid = store.read_state("completion")[0].active_turn_id
        server.kernel.push("completion", InboxItem("identity-reply", "provider_result", payload(provider), tid, 0))
        server.wake("completion")
        adapter.join()
        assert store.read_state("completion")[0].active_turn_id is None and len(calls) == 2
        assert provider.submits == 1 and not provider.cancels
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
    finally:
        release.set()
        monkeypatch.undo()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("failed_thread", ["session-surface-driver", "session-app-server-pump"])
@pytest.mark.parametrize("lease_unavailable", [False, True])
def test_connection_open_failure_keeps_real_worker_completion_owned(
    store, tmp_path, monkeypatch, failed_thread, lease_unavailable
):
    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    factory = store.heartbeat_store
    hits = []
    error = OSError("independent connection unavailable")
    held = store.acquire("completion", owner="another-real-owner", ttl_ms=15000) if lease_unavailable else None

    @contextmanager
    def connection():
        if current_thread().name == failed_thread and not hits:
            hits.append(1)
            raise error
        with factory() as opened:
            yield opened

    monkeypatch.setattr(store, "heartbeat_store", connection)
    try:
        server.wake("completion")
        adapter.join()
        assert hits == [1] and adapter._state_manager.active_turn("completion") is None
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        warnings = [params for _, method, params in notices if method == "error/warning"]
        assert len(warnings) == 1 and "independent connection unavailable" in warnings[0]["message"]
        assert not any(method == "turn/completed" for _, method, _ in notices)
        monkeypatch.undo()
        if held is not None:
            store.release(held)
        tid = store.read_state("completion")[0].active_turn_id
        server.kernel.push("completion", InboxItem("connection-reply", "provider_result", payload(provider), tid, 0))
        server.wake("completion")
        adapter.join()
        assert store.read_state("completion")[0].active_turn_id is None and len(calls) == 2
        assert provider.submits == 1 and not provider.cancels and not store.peek_inbox("completion")
    finally:
        monkeypatch.undo()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("blocked_lease", [False, True])
def test_wake_during_connection_cleanup_stays_with_started_worker(store, tmp_path, monkeypatch, blocked_lease):
    from threading import Event

    from vv_agent.app_server import AppServer

    server, provider, calls, _ = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    factory = store.heartbeat_store
    closing, release = Event(), Event()
    cut = []
    held = store.acquire("completion", owner="manual-cleanup-lease", ttl_ms=15000) if blocked_lease else None
    other = None

    @contextmanager
    def connection():
        with factory() as opened:
            yield opened
            if current_thread().name == "session-surface-driver" and not cut:
                cut.append(1)
                closing.set()
                assert release.wait(5)

    monkeypatch.setattr(store, "heartbeat_store", connection)
    before = len(server.kernel.handles)
    try:
        server.wake("completion")
        assert closing.wait(5)
        handle = server.kernel.handles[-1]
        assert not handle.done() and handle._thread.is_alive()
        assert adapter._state_manager.active_turn("completion").handle is handle
        if blocked_lease:
            adapter.close()
        tid = store.read_state("completion")[0].active_turn_id
        server.kernel.push("completion", InboxItem("cleanup-reply", "provider_result", payload(provider), tid, 0))
        server.wake("completion")
        assert len(server.kernel.handles) == before + 1
        release.set()
        adapter.join()
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        if blocked_lease:
            assert not adapter._lease_retries and adapter._notification_thread is None
            assert len(calls) == 1 and store.peek_inbox("completion")
            assert held is not None
            store.release(held)
            other = AppServer(store=store, host=server.host)
            other.wake("completion")
            other.run_adapter.join()
        state, records, _ = store.read_state("completion")
        assert state.active_turn_id is None and len(calls) == 2
        assert provider.submits == 1 and not provider.cancels and not store.peek_inbox("completion")
        assert (
            sum(r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == "cleanup-reply" for r in records)
            == 1
        )
        assert len(server.kernel.handles) <= before + 2
    finally:
        release.set()
        monkeypatch.undo()
        if other is not None:
            other.run_adapter.close()
            other.run_adapter.join()
            other.kernel.close()
        adapter.close()
        adapter.join()
        server.kernel.close()
