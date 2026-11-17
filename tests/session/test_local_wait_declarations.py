"""Local entries use the same declared-wait yield rules as the SQL kernel."""

from dataclasses import replace
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Any, cast

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
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_inputs import CommitInput, InputSetting, _steer_held_in_state, apply_input
from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session.children import ChildSession, ChildTool, child_delivery
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.delegation import assemble as assemble_child_specs
from vv_agent.session.projection import SessionHydration, hydrate_records
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.result import project_result
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

from .helpers import content_ref, fixture_key_provider
from .test_children import child_task
from .test_yieldable_waits import YieldProvider


@pytest.mark.parametrize("entry", ["runner", "interactive", "app_server"])
@pytest.mark.parametrize("wait_kind", ["child", "provider"])
def test_local_declaration_yields_steer_and_continues_parent(store, tmp_path, monkeypatch, entry, wait_kind):
    with store.atomic():
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    driver = SessionDriver(store=store, content_key_provider=fixture_key_provider())

    def driver_factory(*args, **kwargs):
        del args, kwargs
        return driver

    monkeypatch.setattr("vv_agent.session.surfaces.SessionDriver", driver_factory)
    monkeypatch.setattr("vv_agent.session.app_server.SessionDriver", driver_factory)
    accepted = YieldProvider()
    rendered = []
    yielded_content = "Yielded local observation"

    def on_yield(plan, handles, reason):
        assert reason == "steer"
        rendered.append((plan, handles, reason))
        return ToolExecutionResult(plan._payload["request"]["id"], yielded_content)

    accepted.on_yield = on_yield
    original_runtime = driver.runtime

    def runtime(*args, **kwargs):
        value = original_runtime(*args, **kwargs)
        value.poll_ms = 60_000
        if wait_kind == "provider":
            value.providers["child_task"] = accepted
        claim_store = kwargs.get("store")
        if claim_store is None:
            # AppServer may build a pre-admission definition shell on the caller.
            # Transactional child bindings require the worker's explicit store.
            return value
        if wait_kind == "child":

            def admit(plan):
                state, _, _ = driver._read_state(plan.session_id, claim_store)
                task = state.turns[plan.turn_id].start.task()
                context = value.context(task, plan, CancellationToken(), store=claim_store, session_id=plan.session_id)
                children = cast(list[ChildSession], assemble_child_specs(value, plan, context, task, claim_store))
                return [replace(child, yield_on_steer=True) for child in children]

            value.children["worker"] = ChildTool(admit, on_yield=on_yield)
        return value

    monkeypatch.setattr(driver, "runtime", runtime)

    @function_tool
    def wait_child() -> ToolExecutionResult:
        return ToolExecutionResult(
            "", "Need child input", directive=ToolDirective.WAIT_USER, metadata={"question": "Need child input"}
        )

    late_child_observed = []

    def late_child_finish(request):
        late_child_observed.append(request)
        return LLMResponse("child done")

    if wait_kind == "child":
        child = Agent("worker", "Work.", model="m", tools=[wait_child])
        agent = Agent("parent", "Delegate.", model="m", tools=[child.as_tool()])
        steps = [
            LLMResponse("", [ToolCall("child-call", "worker", {"task_description": "work"})]),
            LLMResponse("", [ToolCall("wait", "wait_child", {})]),
        ]
    else:
        agent = Agent("parent", "Work.", model="m", tools=[child_task])
        steps = [LLMResponse("", [ToolCall("child-call", "child_task", {})])]
    observed = []

    def finish(request):
        observed.append(request)
        return LLMResponse("parent done")

    provider = ScriptedModelProvider.from_steps(
        "test",
        "m",
        [*steps, finish, late_child_finish] if wait_kind == "child" else [*steps, finish],
    )
    server = None
    try:
        if entry == "runner":
            handle = Runner.start(agent, "go", run_config=RunConfig(model_provider=provider, workspace=tmp_path))
            first_result = handle.result(timeout=10)
            assert first_result.status == AgentStatus.WAIT_USER
            sid, tid = handle.session_id, handle.run_id

            def resume():
                return Runner.resume(sid, tid)

            def steer():
                handle.steer("Change direction now")
        elif entry == "interactive":
            client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
            session = client.create_session(agent=agent)
            assert session.prompt("go", auto_follow_up=False).status == AgentStatus.WAIT_USER
            sid = session.session_id
            tid = store.read_state(sid)[0].active_turn_id
            resume = session.continue_run

            def steer():
                session.steer("Change direction now")
        else:
            transport = ChannelTransport(connection_id="local-test", outbound_capacity=10000)
            server = AppServer(
                transport=transport,
                store=store,
                host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, workspace=tmp_path)),
            )
            assert server.kernel is driver
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
            deadline = monotonic() + 10
            while monotonic() < deadline:
                app_state = store.read_state(sid)[0]
                if app_state.waits:
                    tid = app_state.active_turn_id or tid
                    break
                sleep(0.02)
            else:
                raise AssertionError("AppServer worker did not reach the declared wait")

            def resume():
                rpc("turn/resume", {"threadId": sid, "turnId": tid})
                server.run_adapter.join()

            def steer():
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

        if wait_kind == "child":
            deadline = monotonic() + 10
            while monotonic() < deadline:
                parent_projection = driver._runtime_view(sid)
                waits = driver.waits(sid, parent_projection)
                if waits:
                    child_sid, _, first_wait = waits[0]
                    break
                sleep(0.02)
            else:
                raise AssertionError("declared child wait was not observable")
            child_target = InteractionReplyTarget.from_dict(
                {
                    k: first_wait["interaction"][k]
                    for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
                }
            )
        steer()
        resume()
        state, records, _ = store.read_state(sid)
        assert state.active_turn_id is None
        steer_input = next(
            r.record.payload["input"]["input_id"]
            for r in records
            if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "steer"
        )
        receipts = [r for r in records if r.record.kind == "op_completed" and r.record.payload.get("yielded_by") == steer_input]
        assert len(receipts) == 1
        hydration = SessionHydration.from_store(store, tenant_id=driver.tenant_id, session_id=sid)
        receipt = hydrate_records((receipts[0],), hydration)[0].record.payload["result"]
        assert receipt.get("content") == yielded_content, receipt
        assert receipt.get("metadata", {}).get("yield_callback_failed") is not True
        assert len(rendered) == 1
        assert len(observed) == 1
        assert any("Change direction now" in str(m.content) for m in observed[0].messages)
        assert not store.peek_inbox(sid)

        if wait_kind == "child":
            child_projection = driver._runtime_view(child_sid)
            driver.answer(child_sid, child_projection, child_target, InteractionAnswer("text", text="child answer"), "answer")
            deadline = monotonic() + 8
            while monotonic() < deadline:
                child_state = store.read_state(child_sid)[0]
                if child_state.active_turn_id is None and child_state.turns and all(t.ended for t in child_state.turns.values()):
                    break
                sleep(0.02)
            else:
                raise AssertionError("yielded child did not complete after its normal answer")
            assert len(late_child_observed) == 1
            while monotonic() < deadline and store.peek_inbox(sid):
                sleep(0.02)
            assert not store.peek_inbox(sid)
        else:
            assert not late_child_observed
            assert accepted.submits == 1 and accepted.queries == accepted.cancels == 0
            h = accepted.handle
            driver.push(
                sid,
                InboxItem(
                    "late-provider",
                    "provider_result",
                    {
                        "operation_id": h["operation_id"],
                        "attempt": h["attempt"],
                        "request_digest": h["request_digest"],
                        "provider_binding": h["provider"],
                        "result": ToolExecutionResult("child-call", "late result").to_dict(),
                        "usage": {},
                        "evidence": [h["evidence"]],
                    },
                    tid,
                    0,
                ),
            )
            deadline = monotonic() + 8
            while monotonic() < deadline and store.peek_inbox(sid):
                sleep(0.02)
            assert not store.peek_inbox(sid)
            assert accepted.queries == accepted.cancels == 0
        assert len(rendered) == len(receipts) == len(observed) == 1
    finally:
        if server is not None:
            server.run_adapter.close()
        driver.close()


def test_undeclared_local_child_keeps_steer_until_normal_result(store, tmp_path, monkeypatch):
    with store.atomic():
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    driver = SessionDriver(store=store, content_key_provider=fixture_key_provider())
    monkeypatch.setattr("vv_agent.session.surfaces.SessionDriver", lambda *args, **kwargs: driver)
    rendered = []

    def on_yield(*args):
        rendered.append(args)
        raise AssertionError("undeclared waits must not yield")

    original_runtime = driver.runtime

    def runtime(*args, **kwargs):
        value = original_runtime(*args, **kwargs)
        claim_store = kwargs.get("store")
        assert claim_store is not None

        def admit(plan):
            state, _, _ = driver._read_state(plan.session_id, claim_store)
            task = state.turns[plan.turn_id].start.task()
            context = value.context(task, plan, CancellationToken(), store=claim_store, session_id=plan.session_id)
            children = cast(list[ChildSession], assemble_child_specs(value, plan, context, task, claim_store))
            return [replace(child, yield_on_steer=False) for child in children]

        value.children["worker"] = ChildTool(admit, on_yield=on_yield)
        return value

    monkeypatch.setattr(driver, "runtime", runtime)

    @function_tool
    def wait_child() -> ToolExecutionResult:
        return ToolExecutionResult("", "Need child input", directive=ToolDirective.WAIT_USER)

    child = Agent("worker", "Work.", model="m", tools=[wait_child])
    agent = Agent("parent", "Delegate.", model="m", tools=[child.as_tool()])
    observed = []

    def finish(request):
        observed.append(request)
        return LLMResponse("parent done")

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
    try:
        handle = Runner.start(agent, "go", run_config=RunConfig(model_provider=provider, workspace=tmp_path))
        assert handle.result(timeout=10).status == AgentStatus.WAIT_USER
        sid, tid = handle.session_id, handle.run_id
        projection = driver._runtime_view(sid)
        child_sid, _, wait = driver.waits(sid, projection)[0]
        target = InteractionReplyTarget.from_dict(
            {
                k: wait["interaction"][k]
                for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
            }
        )
        handle.steer("Change direction now")
        assert Runner.resume(sid, tid).status == AgentStatus.WAIT_USER
        assert not rendered and not observed
        assert any(item.item.kind == "steer" for item in store.peek_inbox(sid))

        child_projection = driver._runtime_view(child_sid)
        driver.answer(child_sid, child_projection, target, InteractionAnswer("text", text="child answer"), "answer")
        assert Runner.resume(sid, tid).status == AgentStatus.COMPLETED
        _, records, _ = store.read_state(sid)
        assert len(observed) == 1
        assert not rendered
        assert not any(row.record.payload.get("yielded_by") for row in records)
        assert not store.peek_inbox(sid)
    finally:
        driver.close()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("mixed", True),
        ("all_yieldable", False),
        ("stale_generation", True),
        ("turn_wait", True),
        ("empty_wait", False),
        ("stale_target", False),
        ("one_turn", True),
        ("one_turn_untargeted", False),
        ("cancelled_turn", True),
        ("suspended_turn", True),
        ("cancelled_token", True),
        ("child_not_yieldable", True),
        ("child_yieldable", False),
    ],
)
def test_steer_held_fact_matches_core_whole_batch_eligibility(case, expected):
    def parked(*, yield_on_steer=True, child=None):
        return SimpleNamespace(
            state="parked",
            wait={"yield_on_steer": yield_on_steer, "yieldable_children": ["child"]},
            unknown=None,
            child_handle=child,
            plan=SimpleNamespace(operation_id="op", attempt=1),
        )

    operations: dict[str, Any] = {}
    child_handles = None
    if case == "mixed":
        operations = {
            "yieldable": SimpleNamespace(turn_id="turn", attempts={1: parked()}),
            "blocking": SimpleNamespace(turn_id="turn", attempts={1: parked(yield_on_steer=False)}),
        }
    elif case in {
        "all_yieldable",
        "one_turn",
        "one_turn_untargeted",
        "stale_generation",
        "cancelled_token",
        "turn_wait",
        "cancelled_turn",
        "suspended_turn",
        "stale_target",
    }:
        operations = {"op": SimpleNamespace(turn_id="turn", attempts={1: parked()})}
    elif case in {"child_not_yieldable", "child_yieldable"}:
        child_handles = {
            "session_id": "child" if case == "child_yieldable" else "other",
            "siblings": [],
        }
        operations = {"op": SimpleNamespace(turn_id="turn", attempts={1: parked(child=child_handles)})}

    turn = SimpleNamespace(
        cancelled=case == "cancelled_turn",
        suspended=case == "suspended_turn",
        wait={"kind": "user"} if case == "turn_wait" else None,
        start=SimpleNamespace(_payload={"generation": 1}),
    )
    state = SimpleNamespace(
        active_turn_id="turn",
        turns={"turn": turn},
        operations=operations,
        child_completions={},
    )
    item = InboxItem(
        "steer",
        "steer",
        {"content": "redirect"},
        None if case == "one_turn_untargeted" else "stale-turn" if case == "stale_target" else "turn",
        generation=0 if case == "stale_generation" else None,
    )
    assert (
        _steer_held_in_state(
            state,
            item,
            one_turn=case in {"one_turn", "one_turn_untargeted"},
            cancelled=case == "cancelled_token",
        )
        is expected
    )


@pytest.mark.parametrize(
    ("one_turn", "target_tid", "generation", "expected_held"),
    [
        (False, "turn", 0, True),
        (True, None, None, False),
    ],
)
def test_steer_held_fact_matches_real_core_apply_input(one_turn, target_tid, generation, expected_held):
    turn_start = Record(
        "turn/s/current",
        "turn_started",
        "s",
        "turn",
        None,
        None,
        {"generation": 1},
    )
    operation_plan = Record("op/op", "op_planned", "s", "turn", "op", 1, {})
    attempt = Attempt(
        plan=operation_plan,
        state="parked",
        wait={"yield_on_steer": True, "yieldable_children": []},
    )
    state = ExecutionState(
        session_id="s",
        turns={"turn": Turn(turn_start)},
        active_turn_id="turn",
        operations={"op": Operation(turn_id="turn", kind="tool", attempts={1: attempt})},
    )
    item = InboxItem(
        "steer",
        "steer",
        {"content": content_ref("redirect")},
        target_tid,
        generation,
        actor={
            "tenant_id": "default",
            "subject_id": "test",
            "subject_kind": "service",
            "attributes": content_ref("actor"),
        },
    )
    assert _steer_held_in_state(state, item, one_turn=one_turn) is expected_held

    driver = object.__new__(CoreDriver)
    driver.sid = "s"
    driver.watermark = 1
    driver.state = state
    driver.content_reader = None
    driver.store = SimpleNamespace(peek_inbox=lambda *_args, **_kwargs: [SimpleNamespace(item=item, input_seq=1, received_ms=1)])
    pending = apply_input(driver, one_turn=one_turn)
    assert next(pending) is InputSetting.CANCELLED
    try:
        request = pending.send(False)
    except StopIteration as done:
        assert expected_held
        assert done.value is False
    else:
        assert not expected_held
        assert isinstance(request, CommitInput)
        assert request.records[0]._payload["disposition"] == "applied"
        pending.close()


@pytest.mark.parametrize("entry", ["runner", "interactive", "app_server"])
def test_local_declaration_preserves_inbox_paging_and_consumption_order(store, tmp_path, monkeypatch, entry):
    assemble = assemble_child_specs
    stale_input_count = 255
    page_limit = stale_input_count + 2

    def exercise(declared):
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        driver = SessionDriver(store=store, content_key_provider=fixture_key_provider())
        admitted, rendered, observed = [], [], []
        accepted = YieldProvider()
        original_runtime = driver.runtime

        def on_yield(plan, handles, reason):
            assert reason == "steer"
            rendered.append((plan, handles, reason))
            return ToolExecutionResult(plan._payload["request"]["id"], "Yielded local observation")

        accepted.on_yield = on_yield

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)
            value.poll_ms = 60_000
            if declared:
                value.providers["child_task"] = accepted
            claim_store = kwargs.get("store")
            if claim_store is None:
                # AppServer registers a pre-admission definition shell before the worker's store exists.
                return value

            def admit(plan):
                state, _, _ = driver._read_state(plan.session_id, claim_store)
                task = state.turns[plan.turn_id].start.task()
                context = value.context(task, plan, CancellationToken(), store=claim_store, session_id=plan.session_id)
                specs = cast(list[ChildSession], assemble(value, plan, context, task, claim_store))
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
        agent = Agent("parent", "Delegate.", model="m", tools=[child.as_tool(), child_task])
        provider = ScriptedModelProvider.from_steps(
            "test",
            "m",
            [
                LLMResponse(
                    "",
                    [
                        ToolCall("child-call", "worker", {"task_description": "work"}),
                        *([ToolCall("provider-call", "child_task", {})] if declared else []),
                    ],
                ),
                LLMResponse("", [ToolCall("wait", "wait_child", {})]),
                LLMResponse("child done"),
                finish,
            ],
        )
        config = RunConfig(model_provider=provider, workspace=tmp_path)
        server = None
        with monkeypatch.context() as patch:
            patch.setattr("vv_agent.session.surfaces.SessionDriver", lambda *args, **kwargs: driver)
            patch.setattr("vv_agent.session.app_server.SessionDriver", lambda *args, **kwargs: driver)
            patch.setattr(driver, "runtime", runtime)
            try:
                if entry == "runner":
                    handle = Runner.start(agent, "go", run_config=config)
                    assert handle.result(timeout=10).status == AgentStatus.WAIT_USER
                    sid, tid = handle.session_id, handle.run_id

                    def resume():
                        return Runner.resume(sid, tid)
                elif entry == "interactive":
                    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
                    session = client.create_session(agent=agent)
                    assert session.prompt("go", auto_follow_up=False).status == AgentStatus.WAIT_USER
                    sid = session.session_id
                    tid = store.read_state(sid)[0].active_turn_id
                    resume = session.continue_run
                else:
                    transport = ChannelTransport(connection_id="local-test", outbound_capacity=10000)
                    server = AppServer(
                        transport=transport,
                        store=store,
                        host=DefaultAppServerHost(
                            agent=agent,
                            run_config=RunConfig(model_provider=provider, workspace=tmp_path),
                        ),
                    )
                    assert server.kernel is driver
                    identity = 0

                    def rpc(method, params):
                        nonlocal identity
                        identity += 1
                        server.processor.process_message(
                            transport.connection_id,
                            {"jsonrpc": "2.0", "id": identity, "method": method, "params": params},
                        )
                        while True:
                            wire = transport.receive_outbound(timeout=10)
                            if wire.get("id") == identity:
                                assert "result" in wire, wire
                                return wire["result"]

                    rpc("initialize", {"clientInfo": {"name": "local-test"}})
                    server.processor.process_message(transport.connection_id, {"jsonrpc": "2.0", "method": "initialized"})
                    sid = f"local-thread-{'declared' if declared else 'undeclared'}"
                    rpc("thread/start", {"inputId": sid})
                    tid = rpc(
                        "turn/start",
                        {
                            "threadId": sid,
                            "inputId": "initial",
                            "expectedTurnId": None,
                            "input": [{"type": "text", "text": "go"}],
                        },
                    )["turnId"]
                    deadline = monotonic() + 10
                    while monotonic() < deadline:
                        app_state = store.read_state(sid)[0]
                        if app_state.waits:
                            tid = app_state.active_turn_id or tid
                            break
                        sleep(0.02)
                    else:
                        raise AssertionError("AppServer worker did not reach the declared wait")

                    def resume():
                        rpc("turn/resume", {"threadId": sid, "turnId": tid})
                        server.run_adapter.join()
                        return project_result(driver.store, sid, tid)

                _, records, before = store.read_state(sid)
                parked = next(r.record for r in records if r.record.kind == "op_parked")
                assert parked.payload.get("yield_on_steer", False) is declared
                assert len(admitted) == 1
                deadline = monotonic() + 10
                while monotonic() < deadline:
                    projection = driver._runtime_view(sid)
                    waits = driver.waits(sid, projection)
                    if waits:
                        child_sid, _, wait = waits[0]
                        break
                    sleep(0.02)
                else:
                    raise AssertionError("declared child wait was not observable")
                target = InteractionReplyTarget.from_dict(
                    {
                        k: wait["interaction"][k]
                        for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
                    }
                )
                original_deliver = driver._deliver_child_session

                def hold_child_projection(drive_store, child_id):
                    if child_id == child_sid:
                        return 0
                    return original_deliver(drive_store, child_id)

                patch.setattr(driver, "_deliver_child_session", hold_child_projection)
                driver.answer(sid, projection, target, InteractionAnswer("text", text="answer"), "answer")
                child_handle = RunHandle(driver, child_sid, target.turn_id, None, None)
                driver.handles.append(child_handle)
                assert child_handle.result(timeout=10).status == AgentStatus.COMPLETED
                assert not store.peek_inbox(sid)
                assert not store.is_runnable(sid)
                actor = driver._session_principal(sid).to_dict()
                actor_ref = ContentRef.from_dict(actor["attributes"])
                transaction_id = cast(Any, driver.store)._root._transaction_id
                atomic_write_ids: dict[str, object] = {}
                with driver.store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=driver.tenant_id, session_id=sid)
                    driver._push_in_transaction(
                        tx,
                        sid,
                        InboxItem("steer", "steer", {"content": "Change direction now"}, tid, 0),
                        actor=actor,
                    )
                    atomic_write_ids["steer"] = transaction_id()
                    template = driver._prepare_item(
                        sid,
                        InboxItem("stale-template", "follow_up", {"content": "old"}, "stale-turn", 0),
                        writer,
                    )
                    content_ref = ContentRef.from_dict(template.payload["content"])
                    for index in range(stale_input_count):
                        input_id = f"stale-{index}"
                        item = driver._route_inbox_item(sid, replace(template, input_id=input_id, actor=actor))
                        tx.push(sid, item)
                        writer.link_input_content(input_id, content_ref)
                        writer.link_input_actor(input_id, actor_ref)
                        atomic_write_ids[f"stale/{index}"] = transaction_id()

                    def record_delivery(point):
                        if point == "after_parent_inbox":
                            atomic_write_ids["child_delivery"] = transaction_id()

                    assert child_delivery(driver.store, tx, child_sid, hook=record_delivery) > 0
                assert set(atomic_write_ids) == {"steer", "child_delivery", *[f"stale/{i}" for i in range(stale_input_count)]}
                assert len(set(atomic_write_ids.values())) == 1
                patch.setattr(driver, "_deliver_child_session", original_deliver)
                _, _, watermark = store.read_state(sid)
                first_page = store.peek_inbox(sid, through_input_seq=watermark, limit=page_limit - 1)
                page = store.peek_inbox(sid, through_input_seq=watermark, limit=page_limit)
                assert watermark == before + page_limit
                assert len(first_page) == page_limit - 1
                assert first_page[-1].item.input_id == f"stale-{stale_input_count - 1}"
                assert len(page) == page_limit
                assert [item.item.input_id for item in page[:-1]] == [
                    "steer",
                    *[f"stale-{i}" for i in range(stale_input_count)],
                ]
                assert page[-1].item.kind == "child_result"
                with driver.store.atomic() as tx:
                    assert child_delivery(driver.store, tx, child_sid) == 0
                assert [item.item.input_id for item in store.peek_inbox(sid, limit=page_limit)] == [
                    item.item.input_id for item in page
                ]
                assert resume().status == AgentStatus.COMPLETED
                state, records, _ = store.read_state(sid)
                assert state.active_turn_id is None
                yielded = [
                    r for r in records if r.record.kind == "op_completed" and r.record.payload.get("yielded_by") == "steer"
                ]
                if declared:
                    assert yielded
                    assert len(rendered) == 1
                else:
                    assert not yielded
                    assert not rendered
                assert len(observed) == 1
                assert any("Change direction now" in str(m.content) for m in observed[0].messages)
                assert not store.peek_inbox(sid)
                consumed = [
                    (
                        r.seq,
                        r.commit_id,
                        r.record.payload["input"]["input_id"],
                        r.record.payload["disposition"],
                    )
                    for r in records
                    if r.record.kind == "input_applied"
                    and r.record.payload["input"]["input_id"] in {i.item.input_id for i in page}
                ]
                stale_consumed = [row for row in consumed if row[2].startswith("stale-")]
                assert [identity for _, _, identity, _ in stale_consumed] == [f"stale-{i}" for i in range(stale_input_count)]
                assert [seq for seq, _, _, _ in stale_consumed] == sorted(seq for seq, _, _, _ in stale_consumed)
                assert all(disposition == "rejected" for _, _, _, disposition in stale_consumed)
                child_consumed = [
                    (seq, commit_id, disposition)
                    for seq, commit_id, identity, disposition in consumed
                    if identity == page[-1].item.input_id
                ]
                steer_consumed = [
                    (seq, commit_id, disposition) for seq, commit_id, identity, disposition in consumed if identity == "steer"
                ]
                assert len(child_consumed) == len(steer_consumed) == 1
                assert len(consumed) == len({identity for _, _, identity, _ in consumed}) == page_limit
                if declared:
                    receipt_commit_ids = {r.commit_id for r in yielded}
                    assert len(receipt_commit_ids) == 1
                    yielded_commit_id = next(iter(receipt_commit_ids))
                    assert steer_consumed[0][1] == yielded_commit_id
                    assert child_consumed[0][1] == steer_consumed[0][1]
                assert all(commit_id for _, commit_id, _, _ in consumed)
                assert len({commit_id for _, commit_id, _ in child_consumed}) == 1
                assert len({commit_id for _, commit_id, _ in steer_consumed}) == 1
                assert child_consumed[0][0] < steer_consumed[0][0]
                assert child_consumed[0][2] == steer_consumed[0][2] == "applied"
                return (
                    tuple((identity, disposition) for _, _, identity, disposition in stale_consumed),
                    child_consumed[0][2],
                    steer_consumed[0][2],
                    child_consumed[0][0] < steer_consumed[0][0],
                )
            finally:
                if server is not None:
                    server.run_adapter.close()
                driver.close()

    baseline = exercise(False)
    assert exercise(True) == baseline
