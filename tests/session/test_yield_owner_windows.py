"""Owner cancellation and concurrent wake regressions from review A4."""

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier, Event, Thread, current_thread

import pytest

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, ScriptedModelProvider
from vv_agent.run_handle import RunHandle
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_children import child_task
from .test_yield_wake_protocol import make_server
from .test_yieldable_waits import YieldProvider


def interactive(tmp_path, monkeypatch, *, blocking=False):
    calls = []
    entered = Event()
    release = Event()

    def model(req):
        calls.append(req)
        if len(calls) == 1:
            return LLMResponse("", [ToolCall("work", "child_task", {})])
        entered.set()
        if blocking:
            assert release.wait(8)
        return LLMResponse("done")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(model_provider=ScriptedModelProvider.from_callback("test", "m", model), workspace=tmp_path)
    )
    session = client.create_session(agent=Agent("test", "Use tool.", tools=[child_task]), session_id="rev4-owner")
    original = client.driver.runtime
    external = YieldProvider(declared=True, callback=None)

    def assemble(*a, **kw):
        rt = original(*a, **kw)
        rt.providers["child_task"] = external
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)
    session.prompt("start", auto_follow_up=False)
    return client, session, calls, entered, release


def test_done_before_condition_notify_admission(tmp_path, monkeypatch):
    from test_app_server_distributed import request, submission

    checked = Event()
    finish_check = Event()
    decided = Event()
    allow_notify = Event()
    waking = Event()
    original = RunHandle._drive_pass
    captured = []

    def check(self):
        result = original(self)
        if self.session_id == "thread" and not result and not captured:
            captured.append(self)
            checked.set()
            assert finish_check.wait(8)
        return result

    monkeypatch.setattr(RunHandle, "_drive_pass", check)
    with ExitStack() as stack:
        server, _provider, calls = make_server(stack, tmp_path)
        stack.callback(allow_notify.set)
        stack.callback(finish_check.set)
        tid = request(server, "start", "turn/start", submission())["result"]["turnId"]
        assert checked.wait(5)
        handle = captured[0]
        notify = handle._condition.notify_all

        def controlled_notify():
            if handle._done and not decided.is_set():
                decided.set()
                assert allow_notify.wait(8)
            return notify()

        monkeypatch.setattr(handle._condition, "notify_all", controlled_notify)
        wake = server.kernel._wake

        def controlled_wake(sid):
            if sid == "thread":
                waking.set()
            return wake(sid)

        monkeypatch.setattr(server.kernel, "_wake", controlled_wake)
        finish_check.set()
        assert decided.wait(5)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                request, server, "late", "turn/steer", submission(identity="late", expected=tid, text="notify window")
            )
            assert waking.wait(5)
            assert not future.done()
            allow_notify.set()
            assert "result" in future.result(8)
        server.run_adapter.join()
        pending = server.kernel.store.peek_inbox("thread")
        print("DONE_BEFORE_NOTIFY", dict(calls=len(calls), handles=len(server.kernel.handles), pending=len(pending)))
        assert len(calls) == 2 and not pending


def test_two_host_wakes_idle_same_watermark(tmp_path, monkeypatch):
    from test_app_server_distributed import request, submission

    with ExitStack() as stack:
        server, _provider, calls = make_server(stack, tmp_path)
        # Keep the resumed turn parked, so stale second wake cannot hide behind terminal state.
        llm = server.kernel.handles
        request(server, "start", "turn/start", submission())
        server.run_adapter.join()
        old = llm[-1]
        with server.kernel.store.atomic() as tx:
            tx.push("thread", InboxItem("stale-wake", "steer", {"content": "wrong generation"}, old.run_id, 99))
        second_checked = Event()
        release_second = Event()
        original = server.run_adapter.recover

        def recover(*a, **kw):
            if current_thread().name == "second-host-wake":
                second_checked.set()
                assert release_second.wait(8)
            return original(*a, **kw)

        monkeypatch.setattr(server.run_adapter, "recover", recover)
        t = Thread(target=server.run_adapter.wake, args=("thread",), name="second-host-wake")
        t.start()
        assert second_checked.wait(5)
        server.run_adapter.wake("thread")
        server.run_adapter.join()
        baseline = len(llm)
        assert llm[-1].done()
        release_second.set()
        t.join(8)
        assert not t.is_alive()
        server.run_adapter.join()
        print(
            "TWO_IDLE_HOST_WAKES",
            dict(before=baseline, after=len(llm), calls=len(calls), watermarks=[h._drive_positions.get("thread") for h in llm]),
        )
        assert baseline <= len(llm) <= baseline + 1
        assert len(calls) == 1
        assert [i.item.input_id for i in server.kernel.store.peek_inbox("thread")] == ["stale-wake"]
        with server.kernel.store.atomic() as tx:
            tx.push("thread", InboxItem("finish-idle-wakes", "steer", {"content": "finish"}, old.run_id, 0))
        server.wake("thread")
        server.run_adapter.join()
        assert len(calls) == 2 and not server.kernel.store.peek_inbox("thread")


def test_interactive_claimed_cancel_before_handle(tmp_path, monkeypatch):
    client, session, calls, _entered, _release = interactive(tmp_path, monkeypatch)
    claimed = Event()
    allow = Event()
    original = session._run_steered

    def hold():
        claimed.set()
        assert allow.wait(8)
        original()

    monkeypatch.setattr(session, "_run_steered", hold)
    try:
        session.steer("new direction")
        assert claimed.wait(5)
        snapshot = session.state()
        accepted = session.cancel()
        print(
            "INTERACTIVE_CLAIM_CANCEL",
            dict(running=snapshot.running, handle=snapshot.active_run_handle is not None, cancel=accepted),
        )
        allow.set()
        session._join_steered()
        state, _, _ = client.driver.store.read_state(session.session_id)
        ended = state.turns[next(reversed(state.turns))].ended
        print("INTERACTIVE_CLAIM_RESULT", dict(calls=len(calls), ended=ended.payload if hasattr(ended, "payload") else ended))
        assert accepted, "running claimed successor cannot be cancelled until handle is published"
        assert snapshot.running and snapshot.active_run_handle is None
        assert len(calls) == 1
        assert_cancelled(client, session)
    finally:
        allow.set()
        client.driver.close()


def test_active_owner_strongly_retained_during_gc(tmp_path, monkeypatch):
    client, session, _calls, entered, release = interactive(tmp_path, monkeypatch, blocking=True)
    try:
        session.steer("keep alive")
        assert entered.wait(5)
        reference = weakref.ref(session)
        del session
        gc.collect()
        retained = reference()
        assert retained is not None
        assert retained.running and retained.active_run_handle is not None
        assert retained.cancel()
        print("ACTIVE_GC", dict(alive=True, running=True, cancel=True))
    finally:
        release.set()
        client.driver.close()


def test_two_interactive_idle_steers_one_owner(tmp_path, monkeypatch):
    client, session, _calls, entered, release = interactive(tmp_path, monkeypatch, blocking=True)
    barrier = Barrier(2)

    def send(i):
        barrier.wait(5)
        session.steer(f"message-{i}")

    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(send, i) for i in range(2)]
            for f in futures:
                f.result(5)
        assert entered.wait(5)
        assert session.running and session.active_run_handle is not None
        assert len([h for h in client.driver.handles if not h.done()]) == 1
        assert session.cancel()
        release.set()
        session._join_steered()
        print("TWO_INTERACTIVE_STEERS", dict(handles=len(client.driver.handles), running=session.running))
        assert not session.running and session.active_run_handle is None
    finally:
        release.set()
        client.driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("new_after_replacement,notify_owner", [(False, False), (True, False), (True, True)])
def test_child_previous_identity_swap(store, database, monkeypatch, new_after_replacement, notify_owner):
    from .test_yield_local_recovery import parked_tree

    rt, _observed, provider, driver, parent = parked_tree(store, database, monkeypatch)
    sid = "yield-child-0"
    old = driver._background[sid]
    child = old._background_child[1]
    # A fresh stale input makes one legitimate new worker, which parks at a new watermark.
    driver.push(sid, InboxItem("first-watermark", "steer", {"content": "stale"}, child["turn_id"], 99))
    checked = Event()
    allow = Event()
    original = rt.child_runtime

    def binding(*a, **kw):
        result = original(*a, **kw)
        if current_thread().name == "old-child-scheduler":
            checked.set()
            assert allow.wait(8)
        return result

    monkeypatch.setattr(rt, "child_runtime", binding)
    thread = Thread(target=parent._schedule_child, args=(child,), kwargs={"independent": True}, name="old-child-scheduler")
    try:
        thread.start()
        assert checked.wait(5)
        parent._schedule_child(child, independent=True)
        replacement = driver._background[sid]
        replacement.join(5)
        assert replacement is not old and replacement.done()
        # Wait until replacement's finally scheduling has also finished.
        assert not replacement._thread.is_alive()
        baseline_pending = [i.item.input_id for i in store.peek_inbox(sid)]
        if new_after_replacement:
            h = provider.handle
            driver.push(
                sid,
                InboxItem(
                    "second-watermark",
                    "provider_result",
                    {
                        "operation_id": h["operation_id"],
                        "attempt": h["attempt"],
                        "request_digest": h["request_digest"],
                        "provider_binding": h["provider"],
                        "result": ToolExecutionResult("work", "real completion").to_dict(),
                        "usage": {},
                        "evidence": [h["evidence"]],
                    },
                    child["turn_id"],
                    0,
                ),
            )
        if new_after_replacement and notify_owner:
            # Correct host protocol: after commit, ask the child execution owner to schedule.
            parent._schedule_child(child, independent=True)
            driver._background[sid].result(timeout=8)
        before = len(driver.handles)
        allow.set()
        thread.join(8)
        assert not thread.is_alive()
        driver._background[sid].join(5)
        pending = [i.item.input_id for i in store.peek_inbox(sid)]
        print(
            "CHILD_IDENTITY_SWAP",
            dict(
                new_input=new_after_replacement,
                owner_notified=notify_owner,
                before=before,
                after=len(driver.handles),
                pending=pending,
                replacement_unchanged=driver._background[sid] is replacement,
            ),
        )
        stranded = "second-watermark" in pending
        if stranded:
            parent._schedule_child(child, independent=True)
            driver._background[sid].result(timeout=8)
            state, _, _ = store.read_state(sid)
            print(
                "CHILD_MANUAL_RESCAN", dict(ended=bool(state.turns[child["turn_id"]].ended), pending=len(store.peek_inbox(sid)))
            )
            assert state.turns[child["turn_id"]].ended
        if not new_after_replacement:
            assert pending == baseline_pending and before <= len(driver.handles) <= before + 1
        assert stranded == (new_after_replacement and not notify_owner), "owner notification must service the new child watermark"
    finally:
        allow.set()
        thread.join(8)
        driver.close()


@pytest.mark.sqlite_file
@pytest.mark.parametrize("phase", ["before_clear", "after_clear", "after_check"])
def test_marker_interleavings_real_input(store, database, monkeypatch, phase):
    import inspect

    from .test_yield_local_recovery import parked_tree

    _rt, observed, _provider, driver, parent = parked_tree(store, database, monkeypatch)
    source, line = inspect.getsourcelines(RunHandle._run)
    matches = {
        "before_clear": next(line + i for i, s in enumerate(source) if "with self.kernel._background_lock:" in s),
        "after_clear": next(line + i for i, s in enumerate(source) if "progressed = self._drive_pass()" in s),
        "after_check": next(line + i for i, s in enumerate(source) if "if self._rerun_requested or progressed:" in s) - 1,
    }
    paused = Event()
    allow = Event()

    def trace(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is RunHandle._run.__code__
            and frame.f_locals.get("self") is parent
            and frame.f_lineno == matches[phase]
            and not paused.is_set()
        ):
            paused.set()
            assert allow.wait(8)
        return trace

    target = parent._thread._target

    def traced():
        sys.settrace(trace)
        try:
            target()
        finally:
            sys.settrace(None)

    parent._thread._target = traced
    try:
        parent.start()
        assert paused.wait(5)
        parent.steer("actual input in " + phase)
        allow.set()
        parent.result(timeout=8)
        pending = store.peek_inbox("s")
        print("MARKER_INTERLEAVING", dict(phase=phase, observed=len(observed), pending=len(pending), handles=len(driver.handles)))
        assert len(observed) == 1 and not pending
    finally:
        allow.set()
        driver.close()


def test_app_complete_snapshot_vs_admission(tmp_path, monkeypatch):
    from test_app_server_distributed import request, submission

    snapshot = Event()
    allow = Event()
    admission_waiting = Event()
    captured = []
    with ExitStack() as stack:
        server, _provider, calls = make_server(stack, tmp_path)
        stack.callback(allow.set)
        original_complete = server.run_adapter._complete_turn

        def complete(connection_id, started, **kw):
            with server.run_adapter._state_manager.turn_lock("thread"):
                if not captured:
                    assert started.handle.done()
                    server.kernel.store.read_state("thread")
                    captured.append(started.handle)
                    snapshot.set()
                    assert allow.wait(8)
                return original_complete(connection_id, started, **kw)

        monkeypatch.setattr(server.run_adapter, "_complete_turn", complete)
        tid = request(server, "start", "turn/start", submission())["result"]["turnId"]
        assert snapshot.wait(5)
        recover = server.run_adapter.recover

        def recovery(*a, **kw):
            if current_thread().name != "session-app-server-pump":
                admission_waiting.set()
            return recover(*a, **kw)

        monkeypatch.setattr(server.run_adapter, "recover", recovery)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                request, server, "late", "turn/steer", submission(identity="late", expected=tid, text="complete versus admission")
            )
            assert admission_waiting.wait(5)
            assert not future.done()
            allow.set()
            assert "result" in future.result(8)
        server.run_adapter.join()
        pending = server.kernel.store.peek_inbox("thread")
        print("APP_COMPLETE_ADMISSION", dict(calls=len(calls), pending=len(pending), handles=len(server.kernel.handles)))
        assert len(calls) == 2 and not pending


def test_interactive_end_feedback_steer_then_cancel(tmp_path, monkeypatch):
    # Public event listener, no paused private methods or synthetic state.
    calls = []
    cancelled = []

    def model(req):
        calls.append(req)
        return LLMResponse("", [ToolCall("work", "child_task", {})]) if len(calls) == 1 else LLMResponse("done")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(model_provider=ScriptedModelProvider.from_callback("test", "m", model), workspace=tmp_path)
    )
    session = client.create_session(agent=Agent("test", "Use tool.", tools=[child_task]), session_id="end-cancel")
    original = client.driver.runtime
    external = YieldProvider(declared=True, callback=None)

    def assemble(*a, **kw):
        rt = original(*a, **kw)
        rt.providers["child_task"] = external
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)

    def listener(event, payload):
        if event == "session_run_end" and not cancelled:
            session.steer("new direction from callback")
            cancelled.append(session.cancel())
            print(
                "END_FEEDBACK_CANCEL",
                dict(running=session.running, handle=session.active_run_handle is not None, cancel=cancelled[-1]),
            )

    session.subscribe(listener)
    try:
        run = session.prompt("start", auto_follow_up=False)
        print("END_FEEDBACK_RESULT", dict(calls=len(calls), status=run.status.value, handles=len(client.driver.handles)))
        assert cancelled == [True], "owner running at steer handoff drops public cancel"
        assert len(calls) == 1
        assert_cancelled(client, session)
    finally:
        client.driver.close()


def assert_cancelled(client, session):
    snapshot = session.state()
    handle = client.driver.handles[-1]
    state, records, _ = client.driver.store.read_state(session.session_id)
    assert not snapshot.running and snapshot.active_run_handle is None
    assert handle.state().cancelled and handle.state().status == "cancelled"
    assert snapshot.latest_run.completion_reason.value == "cancelled"
    assert state.active_turn_id is None
    assert len(state.turns) == 1
    assert next(r.record for r in records if r.record.kind == "turn_ended").payload["status"] == "cancelled"
    controls = [
        r.record.payload["input"]
        for r in records
        if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "control"
    ]
    assert len(controls) == 1
    assert controls[0]["target_turn_id"] == handle.run_id
    assert controls[0]["payload"] == {"action": "cancel"}
    print("INTERACTIVE_CANCELLED", dict(status=handle.state().status, running=snapshot.running, controls=len(controls)))


def test_initial_run_cancel_before_turn_admission(tmp_path):
    calls = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda request: calls.append(request) or LLMResponse("done"))
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("test", "Work."))

    def listener(event, payload):
        if event == "session_run_start":
            assert session.state().running
            assert session.cancel()

    session.subscribe(listener)
    try:
        session.prompt("start", auto_follow_up=False)
        assert not calls
        assert_cancelled(client, session)
    finally:
        client.driver.close()


def test_cancel_without_successor_does_not_cancel_later_run(tmp_path):
    provider = ScriptedModelProvider.from_callback("test", "m", lambda request: LLMResponse("done"))
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("test", "Work."))
    accepted = []

    def listener(event, payload):
        if event == "session_run_end":
            accepted.append(session.cancel())

    unsubscribe = session.subscribe(listener)
    try:
        session.prompt("start", auto_follow_up=False)
        assert accepted == [True]
        assert not session.running and not session.cancel()
        unsubscribe()
        run = session.prompt("later", auto_follow_up=False)
        assert run.status.value == "completed"
        assert not client.driver.handles[-1].state().cancelled
        print("CANCEL_NO_SUCCESSOR", dict(cancel=accepted[0], later=run.status.value))
    finally:
        client.driver.close()


@pytest.mark.parametrize("phase", ["prepare", "execute"])
def test_cancelled_owner_failure_does_not_cancel_retry(tmp_path, monkeypatch, phase):
    client, session, calls, _entered, _release = interactive(tmp_path, monkeypatch)
    claimed, allow = Event(), Event()
    original = session._run_steered

    def hold():
        claimed.set()
        assert allow.wait(8)
        original()

    def fail(*args, **kwargs):
        raise RuntimeError("injected owner setup failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(session, "_run_steered", hold)
            if phase == "prepare":
                patch.setattr(type(session), "messages", property(fail))
            else:
                patch.setattr(session, "_execute_run", fail)
            session.steer("retryable input")
            assert claimed.wait(5)
            assert session.cancel()
            allow.set()
            session._join_steered()
            assert not session.running and session.active_run_handle is None
        run = session.continue_run()
        assert len(calls) == 2 and run.status.value == "completed"
        assert not client.driver.handles[-1].state().cancelled
        print("CANCEL_FAILED_OWNER", dict(phase=phase, later=run.status.value))
    finally:
        allow.set()
        client.driver.close()


@pytest.mark.parametrize("threads", [4, 16])
def test_many_host_wakes_recheck_latest_watermark(tmp_path, monkeypatch, threads):
    from test_app_server_distributed import request, submission

    with ExitStack() as stack:
        server, _, calls = make_server(stack, tmp_path)
        request(server, "start", "turn/start", submission())
        server.run_adapter.join()
        handles = server.kernel.handles
        turn_id = handles[-1].run_id
        original = server.run_adapter.recover
        barrier = Barrier(threads + 1)
        release = Event()
        stack.callback(release.set)

        def recover(*args, **kwargs):
            if current_thread().name.startswith("concurrent-host-wake"):
                barrier.wait(8)
                assert release.wait(8)
            return original(*args, **kwargs)

        monkeypatch.setattr(server.run_adapter, "recover", recover)
        with server.kernel.store.atomic() as tx:
            tx.push("thread", InboxItem("first-watermark", "steer", {"content": "stale"}, turn_id, 99))
        with ThreadPoolExecutor(threads, thread_name_prefix="concurrent-host-wake") as pool:
            futures = [pool.submit(server.run_adapter.wake, "thread") for _ in range(threads)]
            try:
                barrier.wait(8)
                server.run_adapter.wake("thread")
                server.run_adapter.join()
                before = len(handles)
                release.set()
                for future in futures:
                    future.result(8)
            finally:
                release.set()
        server.run_adapter.join()
        print("N_IDLE_HOST_WAKES", dict(threads=threads, before=before, after=len(handles)))
        assert before <= len(handles) <= before + threads
        assert len(calls) == 1
        before = len(handles)
        with server.kernel.store.atomic() as tx:
            tx.push("thread", InboxItem("fresh-watermark", "steer", {"content": "real input"}, turn_id, 0))
        barrier = Barrier(threads)

        def wake():
            barrier.wait(8)
            server.run_adapter.wake("thread")

        with ThreadPoolExecutor(threads) as pool:
            futures = [pool.submit(wake) for _ in range(threads)]
            for future in futures:
                future.result(8)
        server.run_adapter.join()
        assert len(calls) == 2 and before < len(handles) <= before + threads
        assert not server.kernel.store.peek_inbox("thread")
        print("N_FRESH_HOST_WAKES", dict(threads=threads, calls=len(calls), new_handles=len(handles) - before))
