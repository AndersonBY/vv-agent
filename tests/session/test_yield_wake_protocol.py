"""Owner wake handoff and native-lock regressions from review A3."""

import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from threading import Event, current_thread

import pytest

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, ScriptedModelProvider
from vv_agent.run_handle import RunHandle
from vv_agent.session.records import InboxItem
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_children import child_task
from .test_yield_local_recovery import parked_tree
from .test_yieldable_waits import YieldProvider


def make_server(stack, tmp_path):
    from test_app_server_distributed import node, request

    provider = YieldProvider(declared=True, callback=None)
    calls = []

    def complete(req):
        calls.append(req)
        return LLMResponse("", [ToolCall("work", "child_task", {})]) if len(calls) == 1 else LLMResponse("done")

    server = node(
        stack,
        tmp_path / "server.sqlite",
        "rev3-a",
        ScriptedModelProvider.from_callback("test", "m", complete),
        agent=Agent("test", "Use tool.", tools=[child_task]),
    )
    server.run_adapter._host._run_config = replace(server.run_adapter._host._run_config, tool_providers={"child_task": provider})
    assert "result" in request(server, "create", "thread/start", {"inputId": "thread"})
    return (server, provider, calls)


@pytest.mark.parametrize("arrival", ["after_check", "after_handle_done", "after_done"])
@pytest.mark.parametrize("kind", ["steer", "provider_result", "provider_evidence"])
def test_app_server_owns_late_input_handoff(tmp_path, monkeypatch, arrival, kind):
    from test_app_server_distributed import request, submission

    checked, release = (Event(), Event())
    completing, complete_release = Event(), Event()
    original = RunHandle._drive_pass
    captured = []

    def check(self):
        result = original(self)
        if self.session_id == "thread" and not result and (not captured):
            captured.append(self)
            checked.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(RunHandle, "_drive_pass", check)
    with ExitStack() as stack:
        server, provider, calls = make_server(stack, tmp_path)
        stack.callback(release.set)
        stack.callback(complete_release.set)
        original_complete = server.run_adapter._complete_turn

        def complete(*args, **kwargs):
            if arrival == "after_handle_done" and not completing.is_set():
                completing.set()
                assert complete_release.wait(5)
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(server.run_adapter, "_complete_turn", complete)
        tid = request(server, "start", "turn/start", submission())["result"]["turnId"]
        assert checked.wait(5)
        if arrival == "after_done":
            release.set()
            server.run_adapter.join()
        elif arrival == "after_handle_done":
            release.set()
            assert completing.wait(5)
            assert captured[0].done()
        if kind == "steer":
            receipt = request(server, "late", "turn/steer", submission(identity="late", expected=tid, text="late steer"))
            assert "result" in receipt
        else:
            h = provider.handle
            payload = {"operation_id": h["operation_id"], "attempt": h["attempt"], "request_digest": h["request_digest"]}
            if kind == "provider_result":
                payload.update(
                    provider_binding=h["provider"],
                    result=ToolExecutionResult("work", "real completion").to_dict(),
                    usage={},
                    evidence=[h["evidence"]],
                )
            else:
                payload.update(handle=h, yield_on_steer=True)
            with server.kernel.store.atomic() as tx:
                tx.push("thread", InboxItem("late", kind, payload, tid, 0))
            server.run_adapter.wake("thread")
        release.set()
        complete_release.set()
        server.run_adapter.join()
        state, _records, _ = server.kernel.store.read_state("thread")
        pending = [x.item.input_id for x in server.kernel.store.peek_inbox("thread")]
        print(
            "APP_LATE_INPUT",
            dict(
                arrival=arrival,
                kind=kind,
                calls=len(calls),
                pending=pending,
                handles=len(server.kernel.handles),
                active=server.run_adapter._state_manager.active_turn("thread") is not None,
            ),
        )
        assert not pending, "App Server owns successor but never starts it for late input"
        assert len(calls) == (1 if kind == "provider_evidence" else 2)
        if kind != "provider_evidence":
            assert state.turns[tid].ended is not None
        count = len(server.kernel.handles)
        for _ in range(10):
            server.run_adapter.wake("thread")
        server.run_adapter.join()
        assert count <= len(server.kernel.handles) <= count + 10
        assert not server.kernel.store.peek_inbox("thread")
        assert len(calls) == (1 if kind == "provider_evidence" else 2)


def test_interactive_successor_retains_public_control_owner(tmp_path, monkeypatch):
    entered, release = (Event(), Event())
    requests = []

    def model(req):
        requests.append(req)
        if len(requests) == 1:
            return LLMResponse("", [ToolCall("work", "child_task", {})])
        entered.set()
        assert release.wait(5)
        return LLMResponse("done")

    provider = ScriptedModelProvider.from_callback("test", "m", model)
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("test", "Use tool.", tools=[child_task]), session_id="interactive-rev3-a")
    external = YieldProvider(declared=True, callback=None)
    original = client.driver.runtime

    def assemble(*args, **kwargs):
        rt = original(*args, **kwargs)
        rt.providers["child_task"] = external
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)
    try:
        session.prompt("start", auto_follow_up=False)
        assert client.driver.handles[-1].done()
        session.steer("new direction after park")
        assert entered.wait(5)
        active = [h for h in client.driver.handles if not h.done()]
        assert len(active) == 1
        state = session.state()
        cancelled = session.cancel()
        print(
            "INTERACTIVE_SUCCESSOR_OWNER",
            dict(
                real_active_handles=len(active),
                state_running=state.running,
                public_handle=state.active_run_handle is not None,
                cancel_accepted=cancelled,
            ),
        )
        assert state.running
        assert state.active_run_handle is active[0]
        assert cancelled, "steered run must remain under AgentSession control"
    finally:
        release.set()
        client.driver.close()


def _local_lock_probe(store, database, monkeypatch, *, event_sink=False):
    _rt, observed, _provider, driver, parent = parked_tree(store, database, monkeypatch)
    checking, at_wake = Event(), Event()
    admission_thread = current_thread()
    original_check, original_wake = parent._drive_pass, driver._wake
    captured = []

    def check():
        if not captured:
            captured.append(True)
            checking.set()
            assert at_wake.wait(5)
        return original_check()

    def wake(sid):
        if sid == "s" and current_thread() is admission_thread:
            at_wake.set()
        return original_wake(sid)

    monkeypatch.setattr(parent, "_drive_pass", check)
    monkeypatch.setattr(driver, "_wake", wake)
    try:
        parent.start()
        assert checking.wait(5)
        if event_sink:
            from vv_agent.session.events import SessionRunEventStore

            assert SessionRunEventStore(store, "s", "events").consume(lambda event: parent.steer("event feedback steer"))
        else:
            with store.atomic():
                parent.steer("host transactional steer")
        parent.result(timeout=5)
        assert observed
        assert not store.peek_inbox("s")
        print(
            "EVENT_CONSUMER_LOCK" if event_sink else "LOCAL_TRANSACTION_LOCK",
            {"native_locks": True, "observed": len(observed)},
            flush=True,
        )
    finally:
        at_wake.set()
        driver.close()


def _child_rpc_lock_probe(tmp_path, monkeypatch):
    from test_app_server_distributed import request, submission

    from vv_agent.session.children import ChildSession
    from vv_agent.session.records import SessionSpec

    from .test_recovery_matrix import runtime

    final_locked, at_wake, actual_wake = (Event(), Event(), Event())
    captured, timeouts = ([], [])
    original_check, original_wake = (RunHandle._drive_pass, SessionDriver._wake)

    def check(self):
        result = original_check(self)
        if self.session_id == "nested-child" and not result and (not captured):
            captured.append(self)
            final_locked.set()
            assert at_wake.wait(5)
        return result

    def wake(self, sid):
        if sid == "nested-child":
            actual_wake.set()
            at_wake.set()
        return original_wake(self, sid)

    monkeypatch.setattr(RunHandle, "_drive_pass", check)
    monkeypatch.setattr(SessionDriver, "_wake", wake)
    with ExitStack() as stack:
        server, _, _calls = make_server(stack, tmp_path)
        original_runtime = server.kernel.runtime
        provider = YieldProvider(declared=False, callback=None)

        def assemble(*args, **kwargs):
            rt = original_runtime(*args, **kwargs)
            rt.providers.pop("child_task", None)
            rt.children["child_task"] = lambda plan: ChildSession(
                SessionSpec(
                    "nested-child",
                    "test",
                    "/tmp",
                    attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}},
                ),
                "work",
                yield_on_steer=True,
            )
            rt.child_runtime = lambda *_: runtime(
                Path(server.kernel.store.path),
                [LLMResponse("", [ToolCall("nested-work", "child_task", {})])],
                [child_task],
                providers={"child_task": provider},
            )
            return rt

        monkeypatch.setattr(server.kernel, "runtime", assemble)
        request(server, "start", "turn/start", submission())
        assert final_locked.wait(5)

        error = None
        try:
            reply = request(
                server,
                "child-steer",
                "turn/steer",
                submission(sid="nested-child", identity="late-child", expected="nested-child/turn/start", text="child input"),
            )
            if "error" in reply:
                error = reply["error"]
        except RuntimeError as exc:
            error = str(exc)
        finally:
            at_wake.set()
        server.run_adapter.join()
        print("APP_CHILD_RPC", dict(error=error, actual_wake=actual_wake.is_set()), flush=True)
        assert actual_wake.is_set(), "RPC never reached child local wake"
        print("APP_CHILD_LOCK", dict(timeouts=timeouts, error=error, native_locks=True))
        assert error is None, error


def native_lock_probe(mode, directory):
    import faulthandler

    from vv_agent.session.sqlite import SQLiteStore

    faulthandler.dump_traceback_later(60 if mode == "stress" else 15, exit=True)
    with pytest.MonkeyPatch.context() as patch:
        if mode == "stress":
            stress_probe(directory, patch)
        elif mode == "child_rpc":
            _child_rpc_lock_probe(Path(directory), patch)
        else:
            database = Path(directory) / "local.sqlite"
            with SQLiteStore.standalone(str(database)) as store:
                store.install_schema()
                _local_lock_probe(store, database, patch, event_sink=mode == "event_sink")
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("mode", ["local_transaction", "child_rpc", "event_sink", "stress"])
def test_native_locks_complete_in_subprocess(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-c",
        "from session.test_yield_wake_protocol import native_lock_probe; import sys; native_lock_probe(sys.argv[1], sys.argv[2])",
        mode,
        str(tmp_path),
    ]
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    result = subprocess.run(command, capture_output=True, text=True, timeout=70 if mode == "stress" else 25, cwd=root, env=env)
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr


def stress_probe(directory, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from random import Random
    from threading import Barrier
    from time import monotonic, sleep

    calls = []

    def model(req):
        calls.append(req)
        sleep(Random(4000 + len(calls)).uniform(0, 0.003))
        return LLMResponse("", [ToolCall(f"work-{len(calls)}", "child_task", {})])

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=ScriptedModelProvider.from_callback("test", "m", model), workspace=Path(directory)
        )
    )
    session = client.create_session(agent=Agent("stress", "Use tool.", tools=[child_task], max_cycles=200), session_id="stress")
    external = YieldProvider(declared=True, callback=None)
    original = client.driver.runtime

    def assemble(*args, **kwargs):
        rt = original(*args, **kwargs)
        rt.providers["child_task"] = external
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)
    expected = set()
    try:
        session.prompt("start", auto_follow_up=False)
        for round_no in range(8):
            barrier = Barrier(4)

            def producer(worker, round_no=round_no, barrier=barrier):
                rng = Random(1000 * round_no + worker)
                barrier.wait(5)
                sent = []
                for index in range(3):
                    sleep(rng.uniform(0, 0.02))
                    text = f"steer-{round_no}-{worker}-{index}"
                    session.steer(text)
                    sent.append(text)
                return sent

            with ThreadPoolExecutor(4) as pool:
                futures = [pool.submit(producer, worker) for worker in range(4)]
                for future in futures:
                    expected.update(future.result(10))
            deadline = monotonic() + 10
            while session.running and monotonic() < deadline:
                sleep(0.005)
            assert not session.running
            assert not any(not h.done() for h in client.driver.handles)
            assert not client.driver.store.peek_inbox("stress")
        _, records, _ = client.driver.store.read_state("stress")
        applied = [
            r.record.payload["input"]["payload"]["content"]
            for r in records
            if r.record.kind == "input_applied"
            and r.record.payload["input"]["kind"] == "steer"
            and r.record.payload["disposition"] == "applied"
        ]
        assert len(applied) == len(expected) == 96
        assert set(applied) == expected
        seen = {str(m.content) for req in calls for m in req.messages if m.role == "user"}
        assert expected <= seen
        # Owner handoffs may include one empty pass; work must stay bounded.
        watermarks = [h._drive_positions["stress"][1] for h in client.driver.handles]
        assert len(client.driver.handles) <= len(expected) + 1
        print(
            "STEER_EXIT_STRESS",
            {"inputs": len(applied), "handles": len(watermarks), "rounds": 8, "threads": 4, "model_calls": len(calls)},
            flush=True,
        )
    finally:
        client.driver.close()


def test_interactive_run_end_feedback_keeps_session_owner(tmp_path, monkeypatch):
    calls = []

    def model(req):
        calls.append(req)
        if len(calls) == 1:
            return LLMResponse("", [ToolCall("work", "child_task", {})])
        return LLMResponse("done")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(model_provider=ScriptedModelProvider.from_callback("test", "m", model), workspace=tmp_path)
    )
    session = client.create_session(agent=Agent("test", "Use tool.", tools=[child_task]), session_id="end-feedback")
    original = client.driver.runtime
    provider = YieldProvider(declared=True, callback=None)

    def assemble(*args, **kwargs):
        rt = original(*args, **kwargs)
        rt.providers["child_task"] = provider
        return rt

    monkeypatch.setattr(client.driver, "runtime", assemble)
    sent = []

    def listener(event, payload):
        if event == "session_run_end" and not sent:
            sent.append(True)
            session.steer("steer after handle ended before owner exit")

    session.subscribe(listener)
    try:
        run = session.prompt("start", auto_follow_up=False)
        assert run.final_output == "done"
        assert len(calls) == len(client.driver.handles) == 2
        assert any("before owner exit" in str(m.content) for m in calls[-1].messages)
        assert not session.running
        assert session.active_run_handle is None
    finally:
        client.driver.close()
