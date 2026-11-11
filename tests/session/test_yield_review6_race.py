"""Public controller actions race a real manual lease owner; no pre-drive adapter join."""

from dataclasses import replace
from threading import Event, Thread, current_thread
from time import monotonic

import pytest
from support.interactions import displayed_target

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, DefaultAppServerHost
from vv_agent.session.kernel import drive
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall


@pytest.mark.persistent_store
@pytest.mark.parametrize("terminal", ["cancel", "abort"])
def test_manual_drive_during_async_recover(store, tmp_path, monkeypatch, terminal):
    agent = Agent("assistant", "Ask.", model="m")
    provider = ScriptedModelProvider.new(
        "test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose"})]), LLMResponse("done")]
    )
    config = RunConfig(model_provider=provider, workspace=tmp_path)
    kernel = SessionDriver(store=store)
    server = AppServer(host=DefaultAppServerHost(agent=agent, run_config=config), store=store)
    tid = server.store.create_thread_receipt(input_id="race-root", agent_key="default")["threadId"]
    handle = kernel.start(tid, agent, config, "go")
    handle.result(timeout=10)
    adapter = server.run_adapter
    server.store.runtimes[tid] = handle.runtime

    def action(identity, kind, **rest):
        return adapter.controller_action(
            connection_id="c", thread_id=tid, turn_id=handle.run_id, action_id=identity, action={"kind": kind, **rest}
        )

    action("suspend", "suspend")
    adapter.join()
    gate = Event()
    acquire_allowed = Event()
    manual_holds = Event()
    release_manual = Event()
    errors = []
    acquire = store.acquire
    armed = [True]

    def gated(sid, *a, **kw):
        if sid == tid and current_thread().name == "session-surface-driver" and armed[0]:
            armed[0] = False
            gate.set()
            assert acquire_allowed.wait(8)
        return acquire(sid, *a, **kw)

    monkeypatch.setattr(store, "acquire", gated)

    def hook(point, record):
        if point == "after_commit":
            manual_holds.set()
            assert release_manual.wait(8)

    manual_rt = replace(handle.runtime, hook=hook)

    def manual():
        try:
            assert drive(store, tid, runtime=manual_rt, _one_turn=True)
        except BaseException as e:
            errors.append(repr(e))

    thread = Thread(target=manual, name="a6-manual")
    try:
        receipt = action(
            "reply", "respond", target=displayed_target(kernel, tid).to_dict(), answer={"kind": "text", "text": "value"}
        )
        assert receipt["accepted"] and gate.wait(5)
        # Start the manual drive while the asynchronous recover has not acquired its lease.
        thread.start()
        assert manual_holds.wait(5)
        acquire_allowed.set()
        h = server.kernel.handles[-1]
        h.join(3)
        assert h.done() and h._lease_unavailable
        with store.atomic():
            action("resume", "resume")
            action("terminal", terminal)
        immediate = store.read_state(tid)[0].phase
        release_manual.set()
        thread.join(5)
        assert not thread.is_alive() and not errors, errors
        # App Server's real observer/lease retry must finish without a fresh input/recover.
        deadline = monotonic() + 5
        while monotonic() < deadline and not store.read_state(tid)[0].turns[handle.run_id].ended:
            Event().wait(0.02)
        state, rows, _ = store.read_state(tid)
        ends = [r.record.payload["status"] for r in rows if r.record.kind == "turn_ended"]
        print(
            "NO_JOIN_PRODUCT",
            dict(
                backend=type(store).__name__,
                terminal=terminal,
                accepted=receipt["accepted"],
                lease_unavailable=h._lease_unavailable,
                immediate=immediate,
                ends=ends,
                pending=len(store.peek_inbox(tid)),
                errors=errors,
            ),
            flush=True,
        )
        assert state.turns[handle.run_id].ended and ends == ["cancelled" if terminal == "cancel" else "aborted"]
        assert sum(r.record.kind == "op_started" and r.record.payload["mode"] == "sync" for r in rows) == 1
        assert not store.peek_inbox(tid)
    finally:
        release_manual.set()
        acquire_allowed.set()
        if thread.ident:
            thread.join(5)
        adapter.close()
        adapter.join()
        server.kernel.close()
        kernel.close()
