"""Local entries accept declarations but wait for normal child/provider results."""

from dataclasses import replace
from typing import cast

import pytest

from vv_agent import (
    Agent,
    AgentSessionOptions,
    InteractiveAgentClient,
    RunConfig,
    RunHandle,
    Runner,
    ScriptedModelProvider,
    function_tool,
)
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget
from vv_agent.kernel import _child_delegation
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session.children import ChildSession, ChildTool, child_delivery
from vv_agent.session.delegation import assemble as assemble_child_specs
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import Runtime
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

from .test_children import child_task
from .test_yieldable_waits import YieldProvider


@pytest.mark.parametrize("entry", ["runner", "interactive", "app_server"])
@pytest.mark.parametrize("wait_kind", ["child", "provider"])
def test_local_declaration_keeps_steer_pending_until_normal_result(store, tmp_path, monkeypatch, entry, wait_kind):
    driver = SessionDriver(store=store)
    monkeypatch.setattr("vv_agent.session.surfaces.SessionDriver", lambda *args, **kwargs: driver)
    accepted = YieldProvider()
    rendered = []

    def on_yield(*args):
        rendered.append(args)
        raise AssertionError("local waits must not render a yield")

    accepted.on_yield = on_yield
    original_runtime = driver.runtime

    def runtime(*args, **kwargs):
        value = original_runtime(*args, **kwargs)
        value.poll_ms = 60_000
        if wait_kind == "provider":
            value.providers["child_task"] = accepted
        return value

    monkeypatch.setattr(driver, "runtime", runtime)

    @function_tool
    def wait_child() -> ToolExecutionResult:
        return ToolExecutionResult(
            "", "Need child input", directive=ToolDirective.WAIT_USER, metadata={"question": "Need child input"}
        )

    if wait_kind == "child":
        assemble = _child_delegation.assemble

        def declared_children(*args, **kwargs):
            children = cast(list[ChildSession], (yield from assemble(*args, **kwargs)))
            return [replace(child, yield_on_steer=True) for child in children]

        monkeypatch.setattr(_child_delegation, "assemble", declared_children)
        child = Agent("worker", "Work.", model="m", tools=[wait_child])
        agent = Agent("parent", "Delegate.", model="m", tools=[child.as_tool()])
        steps = [
            LLMResponse("", [ToolCall("child-call", "worker", {"task_description": "work"})]),
            LLMResponse("", [ToolCall("wait", "wait_child", {})]),
            LLMResponse("child done"),
        ]
    else:
        agent = Agent("parent", "Work.", model="m", tools=[child_task])
        steps = [LLMResponse("", [ToolCall("child-call", "child_task", {})])]
    observed = []

    def finish(request):
        observed.append(request)
        return LLMResponse("parent done")

    provider = ScriptedModelProvider.from_steps("test", "m", [*steps, finish])
    server = None
    try:
        if entry == "runner":
            handle = Runner.start(agent, "go", run_config=RunConfig(model_provider=provider, workspace=tmp_path))
            assert handle.result(timeout=10).status == AgentStatus.WAIT_USER
            sid, tid = handle.session_id, handle.run_id

            def resume():
                return Runner.resume(sid, tid)

            handle.steer("Change direction now")
        elif entry == "interactive":
            client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
            session = client.create_session(agent=agent)
            assert session.prompt("go", auto_follow_up=False).status == AgentStatus.WAIT_USER
            sid = session.session_id
            tid = store.read_state(sid)[0].active_turn_id
            resume = session.continue_run
            session.steer("Change direction now")
        else:
            transport = ChannelTransport(connection_id="local-test", outbound_capacity=10000)
            server = AppServer(
                transport=transport,
                store=store,
                host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, workspace=tmp_path)),
            )
            identity = 0

            def rpc(method, params):
                nonlocal identity
                identity += 1
                server.processor.process_message(
                    transport.connection_id, {"jsonrpc": "2.0", "id": identity, "method": method, "params": params}
                )
                while True:
                    wire = transport.receive_outbound(timeout=10)
                    if wire.get("id") == identity:
                        assert "result" in wire, wire
                        return wire["result"]

            rpc("initialize", {"clientInfo": {"name": "local-test"}})
            server.processor.process_message(transport.connection_id, {"jsonrpc": "2.0", "method": "initialized"})
            sid = "local-thread"
            rpc("thread/start", {"inputId": sid})
            tid = rpc(
                "turn/start",
                {"threadId": sid, "inputId": "initial", "expectedTurnId": None, "input": [{"type": "text", "text": "go"}]},
            )["turnId"]
            server.run_adapter.join()

            def resume():
                rpc("turn/resume", {"threadId": sid, "turnId": tid})
                server.run_adapter.join()

            rpc(
                "turn/steer",
                {
                    "threadId": sid,
                    "inputId": "steer",
                    "expectedTurnId": tid,
                    "input": [{"type": "text", "text": "Change direction now"}],
                },
            )
            server.run_adapter.join()

        resume()
        state, records, _ = store.read_state(sid)
        parked = next(r.record for r in records if r.record.kind == "op_parked")
        assert parked.payload["yield_on_steer"] is True
        assert not observed
        assert not rendered
        assert state.active_turn_id == tid
        assert any(i.item.kind == "steer" for i in store.peek_inbox(sid))
        assert not any(r.record.payload.get("yielded_by") for r in records)
        if wait_kind == "child":
            wait = driver.waits(sid, driver.handles[-1].runtime)[0][2]["interaction"]
            target = InteractionReplyTarget.from_dict(
                {
                    k: wait[k]
                    for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
                }
            )
            driver.answer(sid, driver.handles[-1].runtime, target, InteractionAnswer("text", text="child answer"), "answer")
        else:
            h = accepted.handle
            driver.push(
                sid,
                InboxItem(
                    "provider-done",
                    "provider_result",
                    {
                        "operation_id": h["operation_id"],
                        "attempt": h["attempt"],
                        "request_digest": h["request_digest"],
                        "provider_binding": h["provider"],
                        "result": ToolExecutionResult("child-call", "work done").to_dict(),
                        "usage": {},
                        "evidence": [h["evidence"]],
                    },
                    tid,
                    0,
                ),
            )
        resume()
        state, records, _ = store.read_state(sid)
        assert state.active_turn_id is None
        assert len(observed) == 1
        assert any("Change direction now" in str(m.content) for m in observed[0].messages)
        assert not any("Wait interrupted by new input" in str(m.content) for m in observed[0].messages)
        assert not any(r.record.payload.get("yielded_by") for r in records)
        assert not store.peek_inbox(sid)
        if wait_kind == "provider":
            assert accepted.submits == 1 and accepted.queries == accepted.cancels == 0
            assert not rendered
    finally:
        if server is not None:
            server.run_adapter.close()
        driver.close()


@pytest.mark.parametrize("entry", ["runner", "interactive"])
def test_local_declaration_preserves_inbox_paging_and_consumption_order(store, tmp_path, monkeypatch, entry):
    assemble = assemble_child_specs

    def exercise(declared):
        driver = SessionDriver(store=store)
        admitted, rendered, observed = [], [], []
        original_runtime = driver.runtime

        def on_yield(*args):
            rendered.append(args)
            raise AssertionError("local waits must not render a yield")

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)

            def admit(plan):
                state, _, _ = store.read_state(plan.session_id)
                task = state.turns[plan.turn_id].start.task()
                context = value.context(task, plan, CancellationToken(), store=store, session_id=plan.session_id)
                specs = cast(list[ChildSession], assemble(value, plan, context, task, store))
                admitted.extend(replace(child, yield_on_steer=declared) for child in specs)
                return admitted

            value.children["worker"] = ChildTool(admit, on_yield=on_yield)
            return value

        @function_tool
        def wait_child() -> ToolExecutionResult:
            return ToolExecutionResult("", "Need child input", directive=ToolDirective.WAIT_USER)

        def finish(request):
            observed.append(request)
            return LLMResponse("parent done")

        child = Agent("worker", "Work.", model="m", tools=[wait_child])
        agent = Agent("parent", "Delegate.", model="m", tools=[child.as_tool()])
        provider = ScriptedModelProvider.from_steps(
            "test",
            "m",
            [
                LLMResponse("", [ToolCall("child-call", "worker", {"task_description": "work"})]),
                LLMResponse("", [ToolCall("wait", "wait_child", {})]),
                LLMResponse("child done"),
                finish,
            ],
        )
        config = RunConfig(model_provider=provider, workspace=tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr("vv_agent.session.surfaces.SessionDriver", lambda *args, **kwargs: driver)
            patch.setattr(driver, "runtime", runtime)
            try:
                if entry == "runner":
                    handle = Runner.start(agent, "go", run_config=config)
                    assert handle.result(timeout=10).status == AgentStatus.WAIT_USER
                    sid, tid = handle.session_id, handle.run_id

                    def resume():
                        return Runner.resume(sid, tid)
                else:
                    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
                    session = client.create_session(agent=agent)
                    assert session.prompt("go", auto_follow_up=False).status == AgentStatus.WAIT_USER
                    sid = session.session_id
                    tid = store.read_state(sid)[0].active_turn_id
                    resume = session.continue_run

                _, records, before = store.read_state(sid)
                parked = next(r.record for r in records if r.record.kind == "op_parked")
                assert parked.payload.get("yield_on_steer", False) is declared
                assert len(admitted) == 1
                child_sid, child_runtime, wait = driver.waits(sid, driver.handles[-1].runtime)[0]
                target = InteractionReplyTarget.from_dict(
                    {
                        k: wait["interaction"][k]
                        for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
                    }
                )
                driver.answer(sid, driver.handles[-1].runtime, target, InteractionAnswer("text", text="answer"), "answer")
                child_handle = RunHandle(driver, child_sid, target.turn_id, cast(Runtime, child_runtime), None)
                driver.handles.append(child_handle)
                child_handle.start()
                assert child_handle.result(timeout=10).status == AgentStatus.COMPLETED
                assert not store.peek_inbox(sid)

                with store.atomic() as tx:
                    tx.push(sid, InboxItem("steer", "steer", {"content": "Change direction now"}, tid, 0))
                    for index in range(255):
                        tx.push(sid, InboxItem(f"stale-{index}", "follow_up", {"content": "old"}, "stale-turn", 0))
                    child_delivery(store, tx, child_sid)
                _, _, watermark = store.read_state(sid)
                page = store.peek_inbox(sid, through_input_seq=watermark, limit=257)
                assert watermark == before + 257
                assert len(page) == 257
                assert [item.item.input_id for item in page[:256]] == ["steer", *[f"stale-{i}" for i in range(255)]]
                assert page[256].item.kind == "child_result"
                terminal_id = page[256].item.input_id
                peeks = []
                original_peek = store.peek_inbox

                def peek(session_id, *, through_input_seq=None, limit=256, ready_only=True):
                    result = original_peek(session_id, through_input_seq=through_input_seq, limit=limit, ready_only=ready_only)
                    if session_id == sid:
                        peeks.append((through_input_seq, limit, len(result)))
                    return result

                patch.setattr(store, "peek_inbox", peek)
                assert resume().status == AgentStatus.COMPLETED
                state, records, _ = store.read_state(sid)
                assert state.active_turn_id is None
                assert not any(r.record.payload.get("yielded_by") for r in records)
                assert not rendered
                assert len(observed) == 1
                assert any("Change direction now" in str(m.content) for m in observed[0].messages)
                assert not original_peek(sid)
                consumed = [
                    (r.record.payload["input"]["input_id"], r.record.payload["disposition"])
                    for r in records
                    if r.record.kind == "input_applied"
                    and r.record.payload["input"]["input_id"] in {i.item.input_id for i in page}
                ]
                consumed = [
                    ("child_result" if identity == terminal_id else identity, disposition) for identity, disposition in consumed
                ]
                assert consumed == [
                    *[(f"stale-{i}", "rejected") for i in range(255)],
                    ("child_result", "applied"),
                    ("steer", "applied"),
                ]
                assert peeks
                assert all(limit == 256 for _, limit, _ in peeks)
                captured = [p for p in peeks if p[0] is not None]
                assert captured[0] == (watermark, 256, 256)
                assert all(through == watermark for through, _, _ in captured)
                return consumed, peeks
            finally:
                driver.close()

    baseline = exercise(False)
    assert exercise(True) == baseline
