"""Parked owner cancellation and child wake regressions from review A5."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread, current_thread

import pytest
from test_app_server_distributed import request, submission

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.run_handle import RunHandle
from vv_agent.session.children import ChildSession
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_children import child_task
from .test_recovery_matrix import runtime
from .test_yieldable_waits import YieldProvider


def interactive(store, tmp_path, monkeypatch, *, park=True, blocking=False):
    calls = []
    entered = Event()
    release = Event()

    def model(req):
        calls.append(req)
        if park and len(calls) == 1:
            return LLMResponse("", [ToolCall("work", "child_task", {})])
        entered.set()
        if blocking:
            assert release.wait(10)
        return LLMResponse("done")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(model_provider=ScriptedModelProvider.from_callback("test", "m", model), workspace=tmp_path),
        store=store,
    )
    session = client.create_session(agent=Agent("test", "Work.", tools=[child_task]), session_id="a5-owner")
    original = client.driver.runtime
    provider = YieldProvider(declared=True, callback=None)

    def assemble(*a, **kw):
        rt = original(*a, **kw)
        rt.providers["child_task"] = provider
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)
    return client, session, calls, entered, release


def controls(store, sid):
    state, rows, _ = store.read_state(sid)
    return state, [
        r.record.payload["input"]
        for r in rows
        if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "control"
    ]


def guard_store(monkeypatch, driver, session=None):
    hits = []
    for name in ["read_state", "read", "peek_inbox", "atomic", "read_input", "is_runnable", "next_drive_delay_ms"]:
        original = getattr(driver.store, name)

        def checked(*a, _name=name, _fn=original, **kw):
            if (session is not None and session._lock._is_owned()) or driver._background_lock._is_owned():
                hits.append((_name, current_thread().name))
                raise AssertionError("store under session/B lock")
            return _fn(*a, **kw)

        monkeypatch.setattr(driver.store, name, checked)
    return hits


@pytest.mark.persistent_store
def test_end_cancel_without_steer_parked_turn(store, tmp_path, monkeypatch):
    client, s, calls, _, _ = interactive(store, tmp_path, monkeypatch)
    accepted = []

    def listener(event, payload):
        if event == "session_run_end" and not accepted:
            accepted.append(s.cancel())

    hits = guard_store(monkeypatch, client.driver, s)
    unsub = s.subscribe(listener)
    try:
        s.prompt("start", auto_follow_up=False)
        state, cs = controls(store, s.session_id)
        print(
            "PARK_END_CANCEL",
            dict(
                accepted=accepted,
                running=s.running,
                calls=len(calls),
                active_turn=state.active_turn_id,
                controls=len(cs),
                intent=s._cancel_requested,
            ),
            flush=True,
        )
        assert accepted == [True] and not hits
        assert_cancelled(store, s, client, calls)
        assert state.active_turn_id is None and len(cs) == 1, "accepted cancel lost when parked owner has no steer successor"
    finally:
        unsub()
        client.driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("phase", ["claim", "after_extract", "before_start", "initial", "running"])
@pytest.mark.parametrize("n", [2, 12])
def test_cancel_steer_wake_interleavings(store, tmp_path, monkeypatch, phase, n):
    initial = phase == "initial"
    active = phase == "running"
    client, s, calls, entered, release = interactive(store, tmp_path, monkeypatch, park=not initial, blocking=active)
    ready = Event()
    allow = Event()
    errors = []
    if not initial:
        s.prompt("start", auto_follow_up=False)
    old_cancel = RunHandle.cancel
    old_start = RunHandle.start
    old_set = s._set_active_run_handle

    def cancel(h, *a, **kw):
        if phase == "after_extract" and current_thread().name == "interactive-steer" and not ready.is_set():
            ready.set()
            assert allow.wait(10)
        return old_cancel(h, *a, **kw)

    def start(h):
        if phase == "before_start" and h.session_id == s.session_id and not ready.is_set():
            ready.set()
            assert allow.wait(10)
        return old_start(h)

    def set_handle(h):
        if h is not None and phase == "after_extract" and not ready.is_set():
            # Queue one intent before the real publication; block only after extraction in cancel().
            assert s.cancel()
        return old_set(h)

    monkeypatch.setattr(RunHandle, "cancel", cancel)
    monkeypatch.setattr(RunHandle, "start", start)
    monkeypatch.setattr(s, "_set_active_run_handle", set_handle)
    if phase == "claim":
        orig = s._run_steered

        def claim():
            ready.set()
            assert allow.wait(10)
            orig()

        monkeypatch.setattr(s, "_run_steered", claim)
    if initial:

        def listener(event, payload):
            if event == "session_run_start":
                ready.set()
                assert allow.wait(10)

        s.subscribe(listener)

        def prompt():
            try:
                s.prompt("start", auto_follow_up=False)
            except BaseException as e:
                errors.append(e)

        owner = Thread(target=prompt)
        owner.start()
    else:
        s.steer("kick")
        owner = None
        if active:
            assert entered.wait(5)
            ready.set()
    assert ready.wait(5)
    hits = guard_store(monkeypatch, client.driver, s)
    barrier = Barrier(n)

    def action(i):
        barrier.wait(5)
        if i % 3 == 0 or initial:
            return s.cancel()
        if i % 3 == 1:
            return s.steer(f"concurrent-{i}")
        return client.driver._wake(s.session_id)

    try:
        with ThreadPoolExecutor(n) as pool:
            _results = [f.result(8) for f in [pool.submit(action, i) for i in range(n)]]
        allow.set()
        release.set()
        if owner:
            owner.join(10)
            assert not owner.is_alive()
        s._join_steered()
        assert not errors, errors
        state, cs = controls(store, s.session_id)
        print(
            "CANCEL_INTERLEAVE",
            dict(
                backend=type(store).__name__,
                phase=phase,
                n=n,
                calls=len(calls),
                controls=len(cs),
                status=client.driver.handles[-1].state().status,
                hits=hits,
            ),
            flush=True,
        )
        assert not hits and len(cs) == 1 and state.active_turn_id is None
        ended = next(r.record.payload for r in store.read_state(s.session_id)[1] if r.record.kind == "turn_ended")
        assert ended["status"] == "cancelled"
        assert cs[0]["input_id"] == f"cancel/{client.driver.handles[-1].run_id}"
        assert cs[0]["target_turn_id"] == client.driver.handles[-1].run_id and cs[0]["payload"] == {"action": "cancel"}
        print(
            "CANCEL_STORE_SHAPE",
            dict(
                phase=phase,
                input_id="cancel/<turn>",
                kind=cs[0]["kind"],
                target="<turn>",
                payload=cs[0]["payload"],
                ended=ended["status"],
            ),
            flush=True,
        )
        assert client.driver.handles[-1].state().cancelled
        assert len(calls) == (0 if initial else 2 if active else 1)
        assert not s._cancel_requested and not s.running
    finally:
        allow.set()
        release.set()
        client.driver.close()


@pytest.mark.persistent_store
def test_delayed_old_cancel_does_not_kill_replacement(store, tmp_path, monkeypatch):
    client, s, calls, entered, release = interactive(store, tmp_path, monkeypatch, park=False, blocking=True)
    errors = []

    def prompt(text):
        try:
            s.prompt(text, auto_follow_up=False)
        except BaseException as e:
            errors.append(e)

    owner = Thread(target=prompt, args=("first",))
    owner.start()
    assert entered.wait(5)
    old = s.active_run_handle
    old_token = s._active_cancellation_token
    at_cancel = Event()
    allow = Event()
    original = RunHandle.cancel

    def cancel(h, *a, **kw):
        if current_thread().name == "late-cancel" and h is old:
            at_cancel.set()
            assert allow.wait(10)
        return original(h, *a, **kw)

    monkeypatch.setattr(RunHandle, "cancel", cancel)
    ct = Thread(target=s.cancel, name="late-cancel")
    ct.start()
    assert at_cancel.wait(5)
    release.set()
    owner.join(10)
    assert not owner.is_alive()
    release.clear()
    entered.clear()
    newer = Thread(target=prompt, args=("second",))
    newer.start()
    assert entered.wait(5)
    new = s.active_run_handle
    new_token = s._active_cancellation_token
    try:
        allow.set()
        ct.join(10)
        assert not ct.is_alive()
        assert new is not old and new_token is not old_token
        assert not new_token.cancelled
        release.set()
        newer.join(10)
        assert not newer.is_alive()
        assert not errors, errors
        _state, cs = controls(store, s.session_id)
        print(
            "DELAYED_CANCEL_REPLACEMENT",
            dict(
                backend=type(store).__name__,
                calls=len(calls),
                old_token=old_token.cancelled,
                new_token=new_token.cancelled,
                controls=len(cs),
                status=new.state().status,
            ),
            flush=True,
        )
        assert new.state().status == "completed" and len(cs) == 0 and not s._cancel_requested
    finally:
        allow.set()
        release.set()
        client.driver.close()


def server_for(store, tmp_path, monkeypatch, *, child=False, database=None, yield_child=True):
    calls = []
    provider = YieldProvider(declared=not child, callback=None)

    def model(req):
        calls.append(req)
        return LLMResponse("", [ToolCall("work", "child_task", {})]) if len(calls) == 1 else LLMResponse("done")

    srv = AppServer(
        transport=ChannelTransport(connection_id="a5", outbound_capacity=10000),
        store=store,
        host=DefaultAppServerHost(
            agent=Agent("test", "Work.", tools=[child_task]),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m", model),
                tool_providers={} if child else {"child_task": provider},
            ),
        ),
    )
    assert "result" in request(srv, "init", "initialize", {"clientInfo": {"name": "a5"}})
    assert "result" in request(srv, "create", "thread/start", {"inputId": "thread"})
    if child:
        original = srv.kernel.runtime

        def child_model(req):
            if any(m.role == "tool" for m in req.messages):
                return LLMResponse("child done")
            return LLMResponse("", [ToolCall("nested-work", "child_task", {})])

        def assemble(*a, **kw):
            rt = original(*a, **kw)
            rt.children["child_task"] = lambda plan: ChildSession(
                SessionSpec(
                    "child", "test", "/tmp", attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}}
                ),
                "work",
                yield_on_steer=yield_child,
            )
            monkeypatch.setattr(
                rt,
                "child_runtime",
                lambda *_: runtime(database, [child_model, child_model], [child_task], providers={"child_task": provider}),
            )
            return rt

        monkeypatch.setattr(srv.kernel, "runtime", assemble)
    return srv, provider, calls


def shutdown(srv):
    srv.run_adapter.close()
    srv.kernel.close()
    srv.run_adapter.join()


def settle_server(srv):
    for _ in range(10):
        # Completion pumps may register a successor before starting its thread.
        srv.run_adapter.join()
        with srv.kernel._background_lock:
            handles = tuple(srv.kernel.handles)
        before = len(handles)
        for handle in handles:
            handle.join(10)
            assert handle.done()
        srv.run_adapter.join()
        if len(srv.kernel.handles) == before:
            return
    pytest.fail("child delivery did not settle")


@pytest.mark.persistent_store
@pytest.mark.parametrize("yield_child", [False, True])
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_child_only_provider_input_host_wake(store, database, tmp_path, monkeypatch, yield_child, kind):
    srv, provider, calls = server_for(store, tmp_path, monkeypatch, child=True, database=database, yield_child=yield_child)
    try:
        request(srv, "start", "turn/start", submission())
        srv.run_adapter.join()
        root = srv.kernel.handles[0]
        h = provider.handle
        childstate = store.read_state("child")[0]
        tid = childstate.active_turn_id
        before = root._position("thread")
        base = len(srv.kernel.handles)
        assert root.done()
        payload = {"operation_id": h["operation_id"], "attempt": h["attempt"], "request_digest": h["request_digest"]}
        if kind == "provider_result":
            payload.update(
                provider_binding=h["provider"],
                result=ToolExecutionResult("nested-work", "real completion").to_dict(),
                usage={},
                evidence=[h["evidence"]],
            )
        else:
            payload.update(handle=h)
        with store.atomic() as tx:
            tx.push("child", InboxItem("child-callback", kind, payload, tid, 0))
        assert root._position("thread") == before
        root.runtime.wake("child")
        srv.run_adapter.join()
        pending = [x.item.input_id for x in store.peek_inbox("child")]
        print(
            "CHILD_HOST_WAKE",
            dict(
                backend=type(store).__name__,
                yield_child=yield_child,
                kind=kind,
                root_watermark=before,
                unchanged=root._position("thread") == before,
                new_handles=len(srv.kernel.handles) - base,
                pending=pending,
            ),
            flush=True,
        )
        assert not pending
        assert 1 <= len([h for h in srv.kernel.handles[base:] if h.session_id == "thread"]) <= 2
        assert_callback_applied(store, "child", "child-callback")
        if kind == "provider_evidence":
            assert store.read_state("child")[0].active_turn_id == tid
            assert len(calls) == 1
            push_provider(store, "child", tid, provider, "final-child", "provider_result")
            root.runtime.wake("child")
            srv.run_adapter.join()
        assert store.read_state("child")[0].turns[tid].ended and not store.read_state("child")[0].turns[tid].cancelled
        assert store.read_state("thread")[0].active_turn_id is None and len(calls) == 2
    finally:
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("n", [4, 12])
def test_complete_host_wake_admission_three_way(store, tmp_path, monkeypatch, n):
    snapshot = Event()
    allow = Event()
    admitted = Event()
    wake_ready = Barrier(n + 1)
    captured = []
    srv, _provider, calls = server_for(store, tmp_path, monkeypatch)
    try:
        original_complete = srv.run_adapter._complete_turn

        def complete(connection_id, started, **kw):
            with srv.run_adapter._state_manager.turn_lock("thread"):
                if not captured:
                    assert started.handle.done()
                    srv.kernel.store.read_state("thread")
                    captured.append(started.handle)
                    snapshot.set()
                    assert allow.wait(10)
                return original_complete(connection_id, started, **kw)

        monkeypatch.setattr(srv.run_adapter, "_complete_turn", complete)
        tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
        assert snapshot.wait(5)
        original_recover = srv.run_adapter.recover

        def recover(*a, **kw):
            if current_thread().name != "session-app-server-pump":
                admitted.set()
            return original_recover(*a, **kw)

        monkeypatch.setattr(srv.run_adapter, "recover", recover)
        errors = []

        def admission():
            try:
                assert "result" in request(srv, "late", "turn/steer", submission(identity="late", expected=tid, text="fresh"))
            except BaseException as e:
                errors.append(e)

        def wake(i):
            wake_ready.wait(5)
            srv.wake("thread")

        t = Thread(target=admission, name="a5-admission")
        t.start()
        assert admitted.wait(5)
        with ThreadPoolExecutor(n) as pool:
            fs = [pool.submit(wake, i) for i in range(n)]
            wake_ready.wait(5)
            allow.set()
            for f in fs:
                f.result(10)
        t.join(10)
        assert not t.is_alive()
        assert not errors, errors
        srv.run_adapter.join()
        pending = store.peek_inbox("thread")
        handles = len(srv.kernel.handles)
        print(
            "THREE_WAY",
            dict(backend=type(store).__name__, n=n, calls=len(calls), pending=len(pending), handles=handles),
            flush=True,
        )
        assert len(calls) == 2 and not pending and 2 <= handles <= n + 2
    finally:
        allow.set()
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_native_child_inherited_host_wake(store, tmp_path, kind):
    external = YieldProvider(declared=False, callback=None)
    calls = []
    child = Agent("child", "Work.", model="child", tools=[child_task])
    root_agent = Agent("root", "Delegate.", model="root", tools=[child.as_tool(name="run_child")])

    def model(req):
        calls.append(req.model)
        if any(m.role == "tool" for m in req.messages):
            return LLMResponse(req.model + " done")
        return (
            LLMResponse("", [ToolCall("nested-work", "child_task", {})])
            if req.model == "child"
            else LLMResponse("", [ToolCall("delegate", "run_child", {"task_description": "work"})])
        )

    srv = AppServer(
        transport=ChannelTransport(connection_id="a5-native", outbound_capacity=10000),
        store=store,
        host=DefaultAppServerHost(
            agent=root_agent,
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "root", model), tool_providers={"child_task": external}
            ),
        ),
    )
    try:
        request(srv, "init", "initialize", {"clientInfo": {"name": "native"}})
        request(srv, "create", "thread/start", {"inputId": "thread"})
        request(srv, "start", "turn/start", submission())
        srv.run_adapter.join()
        root = srv.kernel.handles[0]
        rootstate = store.read_state("thread")[0]
        child_sid = next(w["handle"]["session_id"] for w in rootstate.waits.values() if w["handle"]["kind"] == "child")
        child_tid = store.read_state(child_sid)[0].active_turn_id
        child_rt = root.runtime.child_runtime(store, child_sid)
        before = root._position("thread")
        base = len(srv.kernel.handles)
        h = external.handle
        payload = {"operation_id": h["operation_id"], "attempt": h["attempt"], "request_digest": h["request_digest"]}
        if kind == "provider_result":
            payload.update(
                provider_binding=h["provider"],
                result=ToolExecutionResult("nested-work", "done").to_dict(),
                usage={},
                evidence=[h["evidence"]],
            )
        else:
            payload.update(handle=h)
        with store.atomic() as tx:
            tx.push(child_sid, InboxItem("native-child-callback", kind, payload, child_tid, 0))
        assert root._position("thread") == before
        child_rt.wake(child_sid)
        srv.run_adapter.join()
        pending = [x.item.input_id for x in store.peek_inbox(child_sid)]
        print(
            "NATIVE_CHILD_HOST_WAKE",
            dict(
                backend=type(store).__name__,
                kind=kind,
                calls=calls[:],
                root_watermark_unchanged=root._position("thread") == before,
                new_handles=len(srv.kernel.handles) - base,
                pending=pending,
            ),
            flush=True,
        )
        assert not pending and base + 1 <= len(srv.kernel.handles) <= base + 2
        assert_callback_applied(store, child_sid, "native-child-callback")
        if kind == "provider_evidence":
            assert calls == ["root", "child"]
            push_provider(store, child_sid, child_tid, external, "native-final", "provider_result")
            child_rt.wake(child_sid)
            srv.run_adapter.join()
        assert calls == ["root", "child", "child", "root"]
        assert (
            store.read_state(child_sid)[0].turns[child_tid].ended
            and not store.read_state(child_sid)[0].turns[child_tid].cancelled
        )
        assert store.read_state("thread")[0].active_turn_id is None
    finally:
        shutdown(srv)


@pytest.mark.persistent_store
def test_lost_park_cancel_can_still_call_model(store, tmp_path, monkeypatch):
    client, s, calls, _, _ = interactive(store, tmp_path, monkeypatch)
    accepted = []

    def listener(event, payload):
        if event == "session_run_end" and not accepted:
            accepted.append(s.cancel())

    unsubscribe = s.subscribe(listener)
    try:
        s.prompt("start", auto_follow_up=False)
        unsubscribe()
        before, cs = controls(store, s.session_id)
        assert_cancelled(store, s, client, calls)
        s.steer("later input after accepted cancel")
        s._join_steered()
        print(
            "LOST_CANCEL_LATER_MODEL",
            dict(
                backend=type(store).__name__,
                accepted=accepted,
                active_after_cancel=before.active_turn_id,
                cancel_controls=len(cs),
                calls=len(calls),
                later=client.driver.handles[-1].state().status,
            ),
            flush=True,
        )
        assert len(calls) == 1, "model called again on turn whose cancel was accepted then discarded"
        assert_cancelled(store, s, client, calls)
    finally:
        client.driver.close()


def assert_cancelled(store, session, client, calls):
    state, items = controls(store, session.session_id)
    tid = next(iter(state.turns))
    assert state.active_turn_id is None and len(state.turns) == 1
    assert state.turns[tid].ended and state.turns[tid].cancelled
    assert len(items) == 1
    assert items[0]["input_id"] == f"cancel/{tid}"
    assert items[0]["target_turn_id"] == tid and items[0]["payload"] == {"action": "cancel"}
    assert client.driver.handles[-1].state().status == "cancelled"
    assert not session.running and not session._cancel_requested and len(calls) == 1


def assert_callback_applied(store, sid, identity):
    rows = store.read_state(sid)[1]
    applied = [r for r in rows if r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == identity]
    assert len(applied) == 1


def push_provider(store, sid, tid, provider, identity, kind):
    h = provider.handle
    payload = {"operation_id": h["operation_id"], "attempt": h["attempt"], "request_digest": h["request_digest"]}
    if kind == "provider_result":
        payload.update(
            provider_binding=h["provider"],
            result=ToolExecutionResult("nested-work", "done").to_dict(),
            usage={},
            evidence=[h["evidence"]],
        )
    else:
        payload.update(handle=h)
    with store.atomic() as tx:
        tx.push(sid, InboxItem(identity, kind, payload, tid, 0))


@pytest.mark.persistent_store
@pytest.mark.parametrize("yield_child", [False, True])
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
@pytest.mark.parametrize("n", [4, 12])
@pytest.mark.parametrize("root_work", [False, True])
def test_concurrent_child_wakes_share_composite_watermark(
    store, database, tmp_path, monkeypatch, yield_child, kind, n, root_work
):
    srv, provider, calls = server_for(store, tmp_path, monkeypatch, child=True, database=database, yield_child=yield_child)
    hits = guard_store(monkeypatch, srv.kernel)
    try:
        tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
        srv.run_adapter.join()
        root = srv.kernel.handles[0]
        child_tid = store.read_state("child")[0].active_turn_id
        before = len(srv.kernel.handles)
        push_provider(store, "child", child_tid, provider, "concurrent-child", kind)
        if root_work:
            with store.atomic() as tx:
                tx.push("thread", InboxItem("root-steer", "steer", {"content": "root fresh work"}, tid, 0))
        # Every caller passes the fast path before anyone enters recover's turn lock.
        ready = Barrier(n)
        original = srv.run_adapter.recover

        def recover(*a, **kw):
            if current_thread().name.startswith("child-waker"):
                ready.wait(10)
            return original(*a, **kw)

        monkeypatch.setattr(srv.run_adapter, "recover", recover)
        with ThreadPoolExecutor(n, thread_name_prefix="child-waker") as pool:
            futures = [pool.submit(root.runtime.wake, "child") for _ in range(n)]
            for future in futures:
                future.result(20)
        srv.run_adapter.join()
        assert not store.peek_inbox("child")
        assert_callback_applied(store, "child", "concurrent-child")
        added = len([h for h in srv.kernel.handles[before:] if h.session_id == "thread"])
        assert 1 <= added <= n
        monkeypatch.setattr(srv.run_adapter, "recover", original)
        if root_work and not yield_child and kind == "provider_evidence":
            # Acceptance is no terminal result: a blocking child still defers the steer.
            assert [i.item.input_id for i in store.peek_inbox("thread")] == ["root-steer"]
            push_provider(store, "child", child_tid, provider, "concurrent-final", "provider_result")
            root.runtime.wake("child")
            srv.run_adapter.join()
        assert not store.peek_inbox("thread")
        if root_work:
            assert_callback_applied(store, "thread", "root-steer")
            assert any("root fresh work" in str(req.messages) for req in calls)
        settle_server(srv)
        base = len(srv.kernel.handles)
        for _ in range(n):
            root.runtime.wake("child")
        srv.run_adapter.join()
        assert base <= len(srv.kernel.handles) <= base + n and not hits
        print(
            "COMPOSITE_CHILD_WAKE",
            dict(
                backend=type(store).__name__,
                n=n,
                root_work=root_work,
                yield_child=yield_child,
                kind=kind,
                new_roots=added,
                pending=0,
                idle_successors=len(srv.kernel.handles) - base,
            ),
            flush=True,
        )
    finally:
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_local_owner_sees_child_only_watermark(store, tmp_path, kind):
    provider = YieldProvider(declared=False, callback=None)
    calls = []
    child = Agent("child", "Work.", model="child", tools=[child_task])
    agent = Agent("root", "Delegate.", model="root", tools=[child.as_tool(name="run_child")])

    def model(req):
        calls.append(req.model)
        if any(m.role == "tool" for m in req.messages):
            return LLMResponse(req.model + " done")
        if req.model == "child":
            return LLMResponse("", [ToolCall("nested-work", "child_task", {})])
        return LLMResponse("", [ToolCall("delegate", "run_child", {"task_description": "work"})])

    config = RunConfig(
        model_provider=ScriptedModelProvider.from_callback("test", "root", model),
        tool_providers={"child_task": provider},
        workspace=tmp_path,
    )
    driver = SessionDriver(store=store)
    driver.create("local-root", str(tmp_path))
    try:
        root = driver.start("local-root", agent, config, "go", input_id="start")
        root.result(timeout=10)
        assert root.done()
        state = store.read_state("local-root")[0]
        child_sid = next(w["handle"]["session_id"] for w in state.waits.values() if w["handle"]["kind"] == "child")
        tid = store.read_state(child_sid)[0].active_turn_id
        before = root._position("local-root")
        push_provider(store, child_sid, tid, provider, "local-child-callback", kind)
        assert root._position("local-root") == before
        # The host owns successor creation; push and the marker never create one.
        assert driver.handles == [root]
        resumed = driver.start("local-root", agent, config, None)
        resumed.result(timeout=10)
        assert not store.peek_inbox(child_sid)
        assert_callback_applied(store, child_sid, "local-child-callback")
        if kind == "provider_evidence":
            assert calls == ["root", "child"]
            push_provider(store, child_sid, tid, provider, "local-final", "provider_result")
            resumed = driver.start("local-root", agent, config, None)
            resumed.result(timeout=10)
        assert calls == ["root", "child", "child", "root"]
        assert store.read_state("local-root")[0].active_turn_id is None
        print("LOCAL_COMPOSITE_WAKE", dict(backend=type(store).__name__, kind=kind, pending=0, calls=calls), flush=True)
    finally:
        driver.close()
