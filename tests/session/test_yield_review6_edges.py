"""A6 independent real-store owner/watermark probes."""

from typing import Any

import pytest
from test_app_server_distributed import request, submission

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.session.children import ChildSession, child_handles
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_children import child_task
from .test_recovery_matrix import runtime
from .test_yield_parked_owner import controls, interactive, settle_server, shutdown
from .test_yieldable_waits import YieldProvider


def payload(provider, kind="provider_result"):
    h = provider.handle
    p = {"operation_id": h["operation_id"], "attempt": h["attempt"], "request_digest": h["request_digest"]}
    if kind == "provider_result":
        p.update(
            provider_binding=h["provider"],
            result=ToolExecutionResult("nested-work", "real final").to_dict(),
            usage={},
            evidence=[h["evidence"]],
        )
    else:
        p.update(handle=h)
    return p


def owned_sessions(store, sid):
    found = set()
    pending = [sid]
    while pending:
        sid = pending.pop()
        if sid in found:
            continue
        found.add(sid)
        for op in store.read_state(sid)[0].operations.values():
            for attempt in op.attempts.values():
                if attempt.child_handle:
                    pending.extend(c["session_id"] for c in child_handles(attempt.child_handle))
    return found


def native(store, tmp_path, *, depth=2, root_wait_after_steer=False):
    external = YieldProvider(declared=False, callback=None)
    calls = []
    agents: list[Agent] = []
    for i in reversed(range(depth)):
        agents.insert(
            0,
            Agent(
                f"a{i}", "Work.", model=f"m{i}", tools=[child_task] if i == depth - 1 else [agents[0].as_tool(name="run_child")]
            ),
        )

    def model(req):
        calls.append(req.model)
        if any(m.role == "tool" for m in req.messages):
            if req.model == "m0" and root_wait_after_steer:
                return LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Next?"})])
            return LLMResponse(req.model + " done")
        if req.model == f"m{depth - 1}":
            return LLMResponse("", [ToolCall("nested-work", "child_task", {})])
        return LLMResponse("", [ToolCall("delegate", "run_child", {"task_description": "work"})])

    srv = AppServer(
        transport=ChannelTransport(connection_id="a6", outbound_capacity=10000),
        store=store,
        host=DefaultAppServerHost(
            agent=agents[0],
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m0", model),
                tool_providers={"child_task": external},
                workspace=tmp_path,
            ),
        ),
    )
    request(srv, "init", "initialize", {"clientInfo": {"name": "a6"}})
    request(srv, "create", "thread/start", {"inputId": "thread"})
    tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
    srv.run_adapter.join()
    root = srv.kernel.handles[0]
    sid = "thread"
    chain = [sid]
    while True:
        state = store.read_state(sid)[0]
        waits = [w for w in state.waits.values() if w["handle"]["kind"] == "child"]
        if not waits:
            break
        sid = child_handles(waits[0]["handle"])[0]["session_id"]
        chain.append(sid)
    assert len(chain) == depth
    return srv, root, chain, external, calls, tid


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_deep_host_wake(store, tmp_path, kind):
    srv, root, chain, p, calls, _ = native(store, tmp_path, depth=5)
    try:
        assert len(chain) == 5
        leaf = chain[-1]
        tid = store.read_state(leaf)[0].active_turn_id
        rt = root.runtime
        for sid in chain[1:]:
            rt = rt.child_runtime(store, sid)
        with store.atomic() as tx:
            tx.push(leaf, InboxItem("deep-callback", kind, payload(p, kind), tid, 0))
        rt.wake(leaf)
        srv.run_adapter.join()
        pending = [x.item.input_id for x in store.peek_inbox(leaf)]
        print(
            "DEEP_WAKE", dict(backend=type(store).__name__, depth=len(chain), kind=kind, pending=pending, calls=calls), flush=True
        )
        assert not pending
        if kind == "provider_evidence":
            with store.atomic() as tx:
                tx.push(leaf, InboxItem("deep-final", "provider_result", payload(p), tid, 0))
            rt.wake(leaf)
            srv.run_adapter.join()
        assert store.read_state("thread")[0].active_turn_id is None
    finally:
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_terminal_child_wait_uncleared(store, tmp_path, kind):
    srv, root, chain, p, calls, _ = native(store, tmp_path)
    try:
        leaf = chain[-1]
        tid = store.read_state(leaf)[0].active_turn_id
        rt = root.runtime.child_runtime(store, leaf)
        before = root._position("thread")
        if kind == "provider_evidence":
            with store.atomic() as tx:
                tx.push(leaf, InboxItem("child-acceptance", kind, payload(p, kind), tid, 0))
            drive(store, leaf, runtime=rt, _one_turn=True)
        with store.atomic() as tx:
            tx.push(leaf, InboxItem("child-final", "provider_result", payload(p), tid, 0))
        # A real kernel worker finished execution, but died before child_delivery.
        drive(store, leaf, runtime=rt, _one_turn=True)
        assert store.read_state(leaf)[0].turns[tid].ended
        assert not store.is_runnable(leaf)
        assert root._position("thread") == before
        assert any(w["handle"]["kind"] == "child" for w in store.read_state("thread")[0].waits.values())
        base = len(srv.kernel.handles)
        rt.wake(leaf)
        srv.run_adapter.join()
        stranded = store.read_state("thread")[0].active_turn_id is not None
        print(
            "TERMINAL_UNDELIVERED_WAKE",
            dict(backend=type(store).__name__, new_handles=len(srv.kernel.handles) - base, stranded=stranded, calls=calls[:]),
            flush=True,
        )
        print(
            "TERMINAL_UNDELIVERED_SETTLED", dict(active=store.read_state("thread")[0].active_turn_id, calls=calls[:]), flush=True
        )
        assert not stranded, "real terminal child + host wake was filtered before recover could redeliver"
    finally:
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
@pytest.mark.parametrize("root_ended", [False, True])
@pytest.mark.parametrize("arrival", ["idle", "done_before_complete"])
def test_yielded_child_host_wake(store, database, tmp_path, monkeypatch, kind, root_ended, arrival):
    from threading import Event

    release_completion = Event()
    p = YieldProvider(declared=False, callback=None)
    calls = []

    def model(req):
        calls.append("root")
        return (
            LLMResponse("", [ToolCall("work", "child_task", {})])
            if len(calls) == 1
            else LLMResponse("Waiting for next instruction")
        )

    srv = AppServer(
        transport=ChannelTransport(connection_id="a6", outbound_capacity=10000),
        store=store,
        host=DefaultAppServerHost(
            agent=Agent("root", "Delegate.", tools=[child_task]),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m", model),
                workspace=tmp_path,
                no_tool_policy=None if root_ended else "wait_user",
            ),
        ),
    )
    original = srv.kernel.runtime

    def assemble(*a, **kw):
        rt = original(*a, **kw)
        rt.children["child_task"] = lambda plan: ChildSession(
            SessionSpec("yielded-child", "test", "/tmp"), "work", yield_on_steer=True
        )

        def child_rt(*args):
            state = store.read_state("yielded-child")[0]
            started = bool(state.operations)
            steps = (
                [LLMResponse("child done")]
                if started
                else [LLMResponse("", [ToolCall("nested-work", "child_task", {})]), LLMResponse("child done")]
            )
            return runtime(database, steps, [child_task], providers={"child_task": p})

        monkeypatch.setattr(rt, "child_runtime", child_rt)
        return rt

    monkeypatch.setattr(srv.kernel, "runtime", assemble)
    request(srv, "init", "initialize", {"clientInfo": {"name": "a6"}})
    request(srv, "create", "thread/start", {"inputId": "thread"})
    tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
    srv.run_adapter.join()
    chain = ["thread", "yielded-child"]
    try:
        request(srv, "steer", "turn/steer", submission(identity="yield-now", expected=tid, text="Ask me while child continues"))
        srv.run_adapter.join()
        active = next(h for h in reversed(srv.kernel.handles) if h.session_id == "thread")
        leaf = chain[-1]
        ctid = store.read_state(leaf)[0].active_turn_id
        assert ctid is not None
        state = store.read_state("thread")[0]
        assert state.active_turn_id == (None if root_ended else tid)
        assert not any(w["handle"]["kind"] == "child" for w in state.waits.values())
        base = len(srv.kernel.handles)
        captured = []
        if arrival == "done_before_complete":
            completion_ready = Event()
            original_complete = srv.run_adapter._complete_turn

            def complete(connection_id, started, **kw):
                if not captured:
                    assert started.handle.done()
                    captured.append(started.handle)
                    completion_ready.set()
                    assert release_completion.wait(10)
                return original_complete(connection_id, started, **kw)

            monkeypatch.setattr(srv.run_adapter, "_complete_turn", complete)
            active.runtime.wake("thread")
            assert completion_ready.wait(5)
            assert not captured[0]._owner_wake_requested
        with store.atomic() as tx:
            tx.push(leaf, InboxItem("yielded-child-callback", kind, payload(p, kind), ctid, 0))
        active.runtime.wake(leaf)
        if captured:
            assert captured[0]._owner_wake_requested
        release_completion.set()
        settle_server(srv)
        pending = [x.item.input_id for x in store.peek_inbox(leaf)]
        print(
            "YIELDED_CHILD_WAKE",
            dict(
                backend=type(store).__name__,
                kind=kind,
                root_ended=root_ended,
                arrival=arrival,
                new_handles=len(srv.kernel.handles) - base,
                pending=pending,
                calls=calls[:],
            ),
            flush=True,
        )
        print(
            "YIELDED_CHILD_SETTLED", dict(pending=[x.item.input_id for x in store.peek_inbox(leaf)], calls=calls[:]), flush=True
        )
        assert len(calls) == 2
        assert len(store.read_state("thread")[0].turns) == 1
        assert "yielded-child-callback" not in pending, "yielded child still executing but no longer included in owner watermark"
    finally:
        release_completion.set()
        shutdown(srv)


@pytest.mark.persistent_store
def test_cancel_then_close(store, tmp_path, monkeypatch):
    client, s, calls, _, _ = interactive(store, tmp_path, monkeypatch)
    accepted = []
    errors = []

    def listener(event, data):
        if event == "session_run_end" and not accepted:
            accepted.extend([s.cancel(), s.cancel()])
            s.close()

    unsub = s.subscribe(listener)
    try:
        try:
            s.prompt("go", auto_follow_up=False)
        except Exception as exc:
            errors.append(repr(exc))
        state, cs = controls(store, s.session_id)
        print(
            "CANCEL_CLOSE",
            dict(
                backend=type(store).__name__,
                accepted=accepted,
                closed=state.closed,
                active=state.active_turn_id,
                controls=len(cs),
                intent=s._cancel_requested,
                running=s.running,
                calls=len(calls),
                errors=errors,
            ),
            flush=True,
        )
        assert accepted == [True, True] and state.closed and state.active_turn_id is None
        assert not s._cancel_requested and not s.running and len(calls) == 1 and not errors
        assert sum(x["input_id"].startswith("cancel/") for x in cs) == 1
    finally:
        unsub()
        client.driver.close()


@pytest.mark.persistent_store
def test_cancel_target_completed_before_redemption(store, tmp_path, monkeypatch):
    client, s, calls, _, _ = interactive(store, tmp_path, monkeypatch)
    accepted = []
    finished = []

    def listener(event, data):
        if event == "session_run_end" and not accepted:
            accepted.append(s.cancel())
            h = client.driver.handles[-1]
            p = h.runtime.providers["child_task"]
            tid = h.run_id
            with store.atomic() as tx:
                pl = payload(p)
                pl["result"] = ToolExecutionResult("work", "completed outside owner").to_dict()
                tx.push(s.session_id, InboxItem("provider-final", "provider_result", pl, tid, 0))
            drive(store, s.session_id, runtime=h.runtime, _one_turn=True)
            finished.append(store.read_state(s.session_id)[0].turns[tid].ended)

    unsub = s.subscribe(listener)
    try:
        s.prompt("go", auto_follow_up=False)
        unsub()
        _state, cs = controls(store, s.session_id)
        intent = s._cancel_requested
        s.prompt("next turn", auto_follow_up=False)
        statuses = [r.record.payload["status"] for r in store.read_state(s.session_id)[1] if r.record.kind == "turn_ended"]
        print(
            "CANCEL_COMPLETED_TARGET",
            dict(
                backend=type(store).__name__,
                accepted=accepted,
                finished=finished,
                controls=len(cs),
                intent=intent,
                statuses=statuses,
                calls=len(calls),
            ),
            flush=True,
        )
        assert finished == [True] and accepted == [True] and not cs and not intent
        assert statuses == ["completed", "completed"]
    finally:
        unsub()
        client.driver.close()


@pytest.mark.persistent_store
def test_root_cancel_deep_descendants(store, tmp_path, monkeypatch):
    calls = []
    external = YieldProvider(declared=False, callback=None)
    leaf = Agent("leaf", "Work.", model="m3", tools=[child_task])
    child = leaf
    for i in [2, 1, 0]:
        child = Agent(f"a{i}", "Delegate.", model=f"m{i}", tools=[child.as_tool(name="run_child")])

    def model(req):
        calls.append(req.model)
        return (
            LLMResponse("", [ToolCall("nested-work", "child_task", {})])
            if req.model == "m3"
            else LLMResponse("", [ToolCall("delegate", "run_child", {"task_description": "work"})])
        )

    client = InteractiveAgentClient(
        options=AgentSessionOptions(model_provider=ScriptedModelProvider.from_callback("test", "m0", model), workspace=tmp_path),
        store=store,
    )
    s = client.create_session(agent=child, session_id="root-cancel")
    original = client.driver.runtime

    def rt(*a, **kw):
        result = original(*a, **kw)
        result.providers["child_task"] = external
        return result

    monkeypatch.setattr(client.driver, "runtime", rt)
    accepted = []
    chain = []

    def listener(event, data):
        if event == "session_run_end" and not accepted:
            h = client.driver.handles[0]
            chain.extend(owned_sessions(store, h.session_id))
            accepted.extend([s.cancel(), s.cancel(), s.cancel()])

    unsub = s.subscribe(listener)
    try:
        s.prompt("go", auto_follow_up=False)
        out: list[dict[str, Any]] = []
        for sid in chain:
            state, cs = controls(store, sid)
            ends = [r.record.payload["status"] for r in store.read_state(sid)[1] if r.record.kind == "turn_ended"]
            out.append(
                dict(
                    sid=sid,
                    active=state.active_turn_id,
                    controls=len(cs),
                    ends=ends,
                    pending=[x.item.kind for x in store.peek_inbox(sid)],
                )
            )
        print("CANCEL_DESCENDANTS", dict(backend=type(store).__name__, accepted=accepted, calls=calls, sessions=out), flush=True)
        assert len(chain) == 4 and accepted == [True] * 3 and len(calls) == 4
        assert all(
            x["active"] is None and x["ends"] == ["cancelled"] and x["controls"] == 1 and "control" not in x["pending"]
            for x in out
        )
    finally:
        unsub()
        client.driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("queued_followup", [False, True])
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_cancel_target_ends_after_exit_snapshot(store, tmp_path, monkeypatch, queued_followup, kind):
    client, s, calls, _, _ = interactive(store, tmp_path, monkeypatch)
    accepted = []
    starts = []
    done = []
    original = client.driver.start

    def start_after_finish(*a, **kw):
        starts.append(1)
        if len(starts) == 2:
            h = client.driver.handles[-1]
            oldtid = h.run_id
            p = h.runtime.providers["child_task"]
            with store.atomic() as tx:
                if kind == "provider_evidence":
                    tx.push(s.session_id, InboxItem("boundary-acceptance", kind, payload(p, kind), oldtid, 0))
                pl = payload(p)
                pl["result"] = ToolExecutionResult("work", "final at publication boundary").to_dict()
                tx.push(s.session_id, InboxItem("boundary-final", "provider_result", pl, oldtid, 0))
            drive(store, s.session_id, runtime=h.runtime, _one_turn=True)
            assert store.read_state(s.session_id)[0].turns[oldtid].ended
            done.append(oldtid)
            if queued_followup:
                s.follow_up("independent later work")
        return original(*a, **kw)

    monkeypatch.setattr(client.driver, "start", start_after_finish)

    def listener(event, data):
        if event == "session_run_end" and not accepted:
            accepted.append(s.cancel())

    unsub = s.subscribe(listener)
    try:
        s.prompt("go", auto_follow_up=False)
        _state, cs = controls(store, s.session_id)
        ends = [
            (r.record.turn_id, r.record.payload["status"])
            for r in store.read_state(s.session_id)[1]
            if r.record.kind == "turn_ended"
        ]
        print(
            "CANCEL_PUBLICATION_COMPLETED",
            dict(
                backend=type(store).__name__,
                queued_followup=queued_followup,
                accepted=accepted,
                ends=ends,
                control_targets=[x["target_turn_id"] for x in cs],
                oldtid=done[0],
                calls=len(calls),
                intent=s._cancel_requested,
            ),
            flush=True,
        )
        assert not any(tid != done[0] and status == "cancelled" for tid, status in ends), (
            "cancel accepted for old turn poisoned an independent follow-up after old provider completion"
        )
        assert not cs and not s._cancel_requested
    finally:
        unsub()
        client.driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
@pytest.mark.parametrize("work", ["child_only", "terminal_undelivered", "yielded_child", "deadline", "root_steer"])
def test_host_wake_work_types(store, database, tmp_path, monkeypatch, kind, work):
    """Each retained work category converges through host wake alone."""
    from vv_agent.session.providers import Definitive

    from .test_yield_parked_owner import server_for

    if work == "child_only":
        test_deep_host_wake(store, tmp_path, kind)
    elif work == "terminal_undelivered":
        test_terminal_child_wait_uncleared(store, tmp_path, kind)
    elif work == "yielded_child":
        test_yielded_child_host_wake(store, database, tmp_path, monkeypatch, kind, False, "idle")
    else:
        monkeypatch.setattr(type(store), "clock_sql", "1000000")
        srv, provider, calls = server_for(store, tmp_path, monkeypatch)
        try:
            tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
            srv.run_adapter.join()
            assert len(calls) == 1
            queries = provider.queries
            if work == "deadline":
                monkeypatch.setattr(
                    provider,
                    "query",
                    lambda h: Definitive(ToolExecutionResult("work", "finished at deadline").to_dict(), ("provider-accepted",)),
                )
                monkeypatch.setattr(type(store), "clock_sql", "1060000")
            else:
                with store.atomic() as tx:
                    tx.push("thread", InboxItem("enum-steer", "steer", {"content": "change direction"}, tid, 0))
            srv.wake("thread")
            srv.run_adapter.join()
            state, records, _ = store.read_state("thread")
            assert not store.peek_inbox("thread")
            assert state.turns[tid].ended and not state.active_turn_id
            assert len(calls) == 2
            assert (
                sum(r.record.kind == "op_completed" and r.record.operation_id == provider.handle["operation_id"] for r in records)
                == 1
            )
            assert provider.cancels == 0
            if work == "root_steer":
                assert provider.queries == queries
        finally:
            shutdown(srv)
    print("HOST_WAKE_WORK_TYPE", dict(backend=type(store).__name__, work=work, kind=kind, pending=0), flush=True)


@pytest.mark.persistent_store
def test_exit_pass_preserves_input_for_another_turn(store, database):
    from vv_agent.run_handle import RunHandle
    from vv_agent.session.surfaces import SessionDriver

    from .test_recovery_matrix import start

    start(store)
    rt = runtime(database, [LLMResponse("first"), LLMResponse("second")])
    drive(store, "s", runtime=rt, _one_turn=True)
    driver = SessionDriver(store=store)
    with store.atomic() as tx:
        tx.push("s", InboxItem("next", "follow_up", {"content": "independent work"}))
    old = RunHandle(driver, "s", "s/turn/initial", rt, None)
    try:
        old.start()
        old.result(timeout=10)
        state, _, _ = store.read_state("s")
        assert state.active_turn_id is None and len(state.turns) == 1
        assert len(rt.llm.steps) == 1
        assert [i.item.input_id for i in store.peek_inbox("s")] == ["next"]
        successor = RunHandle(driver, "s", "s/turn/next", rt, None)
        successor.start()
        successor.result(timeout=10)
        state, _, _ = store.read_state("s")
        assert len(state.turns) == 2 and all(t.ended for t in state.turns.values())
        assert not rt.llm.steps and not store.peek_inbox("s")
    finally:
        driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("already_queued", [False, True])
def test_child_continuation_selects_its_reserved_turn(store, monkeypatch, already_queued):
    from test_sub_task_manager_continuation_recovery import _child, _start

    from vv_agent.session.surfaces import SessionDriver

    driver = SessionDriver(store=store)
    wakes = []
    try:
        parent = _start(driver, [LLMResponse("child done"), LLMResponse("parent done"), LLMResponse("continued")])
        parent.result(timeout=10)
        child = _child(driver, parent)
        sid = child["session_id"]
        parent.runtime.wake = wakes.append
        schedule = parent.runtime.hook
        monkeypatch.setattr(parent.runtime, "hook", lambda *_: None)
        manager = parent.runtime.child_tasks(store, "parent")
        assert manager.message(sid, "continuation", "more") == "continued"
        if already_queued:
            from vv_agent.session.store import LeaseLost

            queued_runtime = parent.runtime.child_runtime(store, sid)

            def cut(point, record):
                if point == "after_commit" and record is not None and record.kind == "input_applied":
                    raise LeaseLost("continuation queued before admission")

            monkeypatch.setattr(queued_runtime, "hook", cut)
            with pytest.raises(LeaseLost, match="continuation queued before admission"):
                drive(store, sid, runtime=queued_runtime, _one_turn=True, _wake_on_exit=False)
            state = store.read_state(sid)[0]
            assert len(state.turns) == 1 and not store.peek_inbox(sid)
            assert "continuation" not in state.admitted_inputs
            assert state.applied_inputs["continuation"]._payload["disposition"] == "queued"
        monkeypatch.setattr(parent.runtime, "hook", schedule)
        schedule("after_input", None)
        handle = driver._background[sid]
        handle.result(timeout=10)
        state, records, _ = store.read_state(sid)
        assert handle.run_id == f"{sid}/turn/continuation"
        assert len(state.turns) == 2 and state.active_turn_id is None
        assert [r.record.payload["result"] for r in records if r.record.kind == "turn_ended"] == ["child done", "continued"]
        assert sum(r.record.kind == "op_started" and r.record.payload["mode"] == "sync" for r in records) == 2
        assert not store.peek_inbox(sid) and wakes == [sid]
        print(
            "CHILD_CONTINUATION_TARGET",
            dict(backend=type(store).__name__, already_queued=already_queued, turn=handle.run_id, pending=0),
            flush=True,
        )
    finally:
        driver.close()


@pytest.mark.persistent_store
def test_prepublication_cancel_with_completed_history(store, tmp_path, monkeypatch):
    client, session, calls, _, _ = interactive(store, tmp_path, monkeypatch, park=False)
    accepted = []
    session.prompt("first run", auto_follow_up=False)
    first = next(iter(store.read_state(session.session_id)[0].turns))

    def listener(event, payload):
        if event == "session_run_start" and not accepted:
            accepted.append(session.cancel())

    unsubscribe = session.subscribe(listener)
    try:
        session.prompt("new reserved run", auto_follow_up=False)
        state, records, _ = store.read_state(session.session_id)
        turns = [(r.record.turn_id, r.record.payload["status"]) for r in records if r.record.kind == "turn_ended"]
        _, controls_ = controls(store, session.session_id)
        print(
            "RETAINED_HISTORY_PREPUBLICATION_CANCEL",
            dict(
                backend=type(store).__name__,
                accepted=accepted,
                calls=len(calls),
                turns=turns,
                control_targets=[x["target_turn_id"] for x in controls_],
            ),
            flush=True,
        )
        assert accepted == [True] and len(calls) == 1
        assert len(turns) == 2 and turns[0] == (first, "completed") and turns[1][1] == "cancelled"
        assert len(controls_) == 1 and controls_[0]["target_turn_id"] == turns[1][0]
        assert not state.active_turn_id and not session.running and not session._cancel_requested
        assert not store.peek_inbox(session.session_id)
    finally:
        unsubscribe()
        client.driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("already_admitted", [False, True])
def test_new_user_admission_outlives_old_handle(store, tmp_path, monkeypatch, already_admitted):
    from threading import Event

    from vv_agent.run_handle import RunHandle
    from vv_agent.session.store import LeaseLost

    from .test_yield_parked_owner import server_for

    srv, _, calls = server_for(store, tmp_path, monkeypatch)
    ready, release = Event(), Event()
    captured = []
    try:
        tid = request(srv, "start", "turn/start", submission())["result"]["turnId"]
        srv.run_adapter.join()
        original = RunHandle._drive_pass

        def drive_pass(handle):
            progressed = original(handle)
            if (
                handle.session_id == "thread"
                and handle.run_id == tid
                and not captured
                and store.read_state("thread")[0].turns[tid].ended
            ):
                captured.append(handle)
                ready.set()
                assert release.wait(10)
            return progressed

        monkeypatch.setattr(RunHandle, "_drive_pass", drive_pass)
        response = request(srv, "finish", "turn/steer", submission(identity="finish", expected=tid, text="finish old turn"))
        assert "result" in response and ready.wait(5)
        assert not captured[0].done()
        response = request(srv, "next", "turn/start", submission(identity="new-turn", expected=tid, text="new work"))
        assert response["result"]["accepted"]
        next_tid = response["result"]["turnId"]
        if already_admitted:

            def cut(point, record):
                if point == "after_commit" and record is not None and record.kind == "turn_started":
                    assert record.turn_id == next_tid
                    raise LeaseLost("new turn admitted before old handle exit")

            _, agent, config = srv.run_adapter._binding("thread")
            config = srv.run_adapter._with_app_server_controls(config, connection_id="a5", thread_id="thread", turn_id=next_tid)
            external = srv.kernel.runtime(agent, config)
            external.wake = captured[0].runtime.wake
            monkeypatch.setattr(external, "hook", cut)
            with pytest.raises(LeaseLost, match="new turn admitted before old handle exit"):
                drive(store, "thread", runtime=external, _one_turn=True)
            assert store.read_state("thread")[0].active_turn_id == next_tid
        release.set()
        settle_server(srv)
        state, records, _ = store.read_state("thread")
        ends = [(r.record.turn_id, r.record.payload["status"]) for r in records if r.record.kind == "turn_ended"]
        queued = [
            key
            for key, record in state.applied_inputs.items()
            if record._payload["disposition"] == "queued" and key not in state.admitted_inputs
        ]
        print(
            "NEW_USER_OWNER_HANDOFF",
            dict(
                backend=type(store).__name__,
                already_admitted=already_admitted,
                calls=len(calls),
                ends=ends,
                queued=queued,
                active=state.active_turn_id,
            ),
            flush=True,
        )
        assert len(calls) == 3
        assert ends == [(tid, "completed"), (next_tid, "completed")]
        assert "new-turn" in state.admitted_inputs and state.active_turn_id is None
        assert not queued and not store.peek_inbox("thread")
    finally:
        release.set()
        shutdown(srv)


@pytest.mark.persistent_store
@pytest.mark.parametrize("kind", ["provider_result", "provider_evidence"])
def test_local_child_push_after_last_child_pass_marks_root(store, database, monkeypatch, kind):
    from vv_agent.run_handle import RunHandle
    from vv_agent.session.surfaces import SessionDriver

    from .test_recovery_matrix import start
    from .test_yield_parked_owner import guard_store
    from .test_yieldable_waits import waiting_runtime

    start(store)
    rt, observed = waiting_runtime(database, declared=False)
    provider = YieldProvider(declared=False, callback=None)
    child_rt = runtime(
        database,
        [LLMResponse("", [ToolCall("work", "child_task", {})]), LLMResponse("child done")],
        [child_task],
        providers={"child_task": provider},
    )
    drive(store, "s", runtime=rt)
    drive(store, "yield-child-0", runtime=child_rt)
    monkeypatch.setattr(
        rt,
        "child_runtime",
        lambda *_: runtime(database, [LLMResponse("child done")], [child_task], providers={"child_task": provider}),
    )
    driver = SessionDriver(store=store)
    parent = RunHandle(driver, "s", "s/turn/initial", rt, None)
    driver.handles.append(parent)
    hits = guard_store(monkeypatch, driver)
    original_tree, original_wake = parent._drive_tree, driver._wake
    injected, wake_results = [], []

    def wake(sid):
        result = original_wake(sid)
        if sid == "yield-child-0":
            wake_results.append(result)
        return result

    def tree(sid, runtime_, tid, **kw):
        progressed = original_tree(sid, runtime_, tid, **kw)
        if sid == "yield-child-0" and not injected:
            injected.append(True)
            data = payload(provider, kind)
            if kind == "provider_result":
                data["result"] = ToolExecutionResult("work", "late child completion").to_dict()
            driver.push(sid, InboxItem("after-child-pass", kind, data, "yield-child-0/turn/start", 0))
        return progressed

    monkeypatch.setattr(driver, "_wake", wake)
    monkeypatch.setattr(parent, "_drive_tree", tree)
    try:
        parent.start()
        parent.result(timeout=10)
        pending = [x.item.input_id for x in store.peek_inbox("yield-child-0")]
        print(
            "LOCAL_CHILD_EXIT_MARKER",
            dict(
                backend=type(store).__name__,
                kind=kind,
                pending=pending,
                wake_results=wake_results,
                observed=len(observed),
                handles=len(driver.handles),
                lock_hits=hits,
            ),
            flush=True,
        )
        assert injected and not pending and wake_results == [True] and not hits
        assert len(driver.handles) == 1
        assert len(observed) == (1 if kind == "provider_result" else 0)
        if kind == "provider_evidence":
            data = payload(provider)
            data["result"] = ToolExecutionResult("work", "final after evidence").to_dict()
            driver.push("yield-child-0", InboxItem("local-final", "provider_result", data, "yield-child-0/turn/start", 0))
            assert len(driver.handles) == 1
            successor = RunHandle(driver, "s", parent.run_id, rt, None)
            driver.handles.append(successor)
            successor.start()
            successor.result(timeout=10)
        assert len(observed) == 1
        assert not store.peek_inbox("s") and not store.peek_inbox("yield-child-0")
    finally:
        driver.close()
