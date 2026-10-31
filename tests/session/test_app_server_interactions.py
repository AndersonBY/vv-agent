"""Real JSON-RPC producers; all applicable cases run on memory/file SQLite and PostgreSQL."""

import json
import runpy
from pathlib import Path

import pytest

from vv_agent import Agent, InteractionAnswer, InteractionReplyTarget
from vv_agent.canonical_json import canonical_json_sha256
from vv_agent.session.interactions import reply_content
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store

AUTHOR = runpy.run_path(str(Path(__file__).parents[2] / "scripts/app_server_interaction_fixtures.py"))
InteractionServer, InteractionHost, target_of = (AUTHOR[k] for k in ("InteractionServer", "InteractionHost", "target_of"))
THREAD, TURN = AUTHOR["THREAD"], AUTHOR["TURN"]


def error(wire, reason, code=-32602):
    assert wire["error"]["code"] == code, wire
    assert wire["error"].get("data", {}).get("reason", wire["error"]["message"]) == reason, wire


def text(value="free answer"):
    return {"kind": "text", "text": value}


def unchanged(server, form, before):
    assert server.form() == form
    state, rows, _ = server.server.kernel.store.read_state(THREAD)
    assert not state.turns[TURN].ended
    assert (server.host.calls, sum(r.record.kind == "op_planned" for r in rows)) == before
    assert server.host.asks == sum(r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "user" for r in rows)


@pytest.mark.parametrize(
    "spec,answer,expected",
    [
        ({}, text("  a full reply  "), {"reply_mode": "free_text", "text": "  a full reply  "}),
        (
            {"options": [" A ", "B", "A"]},
            {"kind": "single", "selected": ["B"], "custom": []},
            {"reply_mode": "selection", "selection_type": "single", "selected": ["B"], "custom": []},
        ),
        (
            {"options": ["A", "B"], "selection_type": "multi"},
            {"kind": "multi", "selected": ["A", "B"], "custom": []},
            {"reply_mode": "selection", "selection_type": "multi", "selected": ["A", "B"], "custom": []},
        ),
        (
            {"options": ["A"], "allow_custom_options": True},
            {"kind": "single", "selected": [], "custom": ["  Other  "]},
            {"reply_mode": "selection", "selection_type": "single", "selected": [], "custom": ["Other"]},
        ),
        (
            {"options": ["A", "B"], "allow_custom_options": False},
            text("NOT_AN_OPTION"),
            {"reply_mode": "free_text", "text": "NOT_AN_OPTION"},
        ),
        (
            {"options": ["A", "B"], "selection_type": "multi", "allow_custom_options": False},
            text("NOT_AN_OPTION"),
            {"reply_mode": "free_text", "text": "NOT_AN_OPTION"},
        ),
    ],
)
def test_forms_live_snapshots_replay_rebuild_and_typed_answers(store, database, spec, answer, expected):
    host = InteractionHost(spec)
    server = InteractionServer(store, host)
    try:
        server.start()
        form = server.form()
        assert form["kind"] == "user" and form["generation"] == 0
        assert form["requestDigest"] == canonical_json_sha256(
            {k: v for k, v in form.items() if k not in {"kind", "requestDigest"}}
        )
        read = server.rpc("thread/read", {"threadId": THREAD})["result"]
        resumed = server.rpc("thread/resume", {"threadId": THREAD, "subscribe": False})["result"]
        event = next(e for e in server.events() if e["type"] == "host_interaction_requested")
        assert read["state"] == resumed["state"] == server.status()
        assert event["metadata"]["interaction"] == form == read["state"]["pendingInteraction"]
        assert any(
            m.get("method") == "thread/status/changed" and m["params"]["pendingInteraction"] == form for m in server.messages
        )
        server.server.processor.disconnect_connection(server.transport.connection_id)
        server.close()
        if database is not store:
            store.connection.close()
        with open_store(database) as reopened:
            # Different current host form cannot rewrite the retained form.
            changed = InteractionHost({"question": "CHANGED", "options": ["X", "Y"]})
            server = InteractionServer(reopened, changed, connection="rebuilt")
            assert server.form() == form
            assert changed.calls == 0
            result = server.answer(form, answer)
            assert result["result"]["accepted"]
            assert changed.calls == 1 and changed.asks == 0
            model_reply = next(m.content for m in changed.requests[-1].messages if m.role == "tool" and m.tool_call_id == "ask-0")
            assert json.loads(model_reply) == expected
            server.close()
            # Retry after resolution and another rebuild uses the receipt first.
            server = InteractionServer(reopened, InteractionHost(), connection="retry")
            assert server.answer(form, answer)["result"] == result["result"]
            assert server.host.calls == 0
            error(server.answer(form, text("changed")), "submission_conflict", -32024)
            if answer.get("custom"):
                error(
                    server.answer(form, answer | {"custom": [v.strip() for v in answer["custom"]]}), "submission_conflict", -32024
                )
    finally:
        server.close()


@pytest.mark.parametrize(
    "answer",
    [
        {"kind": "single", "selected": ["NOT_AN_OPTION"], "custom": []},
        {"kind": "single", "selected": [], "custom": ["custom"]},
        {"kind": "single", "selected": ["A", "A"], "custom": []},
        {"kind": "single", "selected": [], "custom": []},
        {"kind": "multi", "selected": ["A"], "custom": []},
        {"kind": "text", "text": ""},
        {"kind": "text", "text": " "},
        {"kind": "text", "text": "汉" * 21846},
        {"kind": "single", "selected": [], "custom": ["x" * 1025]},
        {"kind": "single", "selected": ["A"], "custom": [], "text": "forged"},
    ],
)
def test_rejected_answers_keep_wait_and_do_not_call_model_or_tool(store, answer):
    server = InteractionServer(store, InteractionHost({"options": ["A", "B"]}))
    try:
        server.start()
        form = server.form()
        before = (server.host.calls, sum(r.record.kind == "op_planned" for r in store.read_state(THREAD)[1]))
        error(server.answer(form, answer), "invalid_interaction_answer")
        unchanged(server, form, before)
        assert "result" in server.answer(form, text("NOT_AN_OPTION"), action_id="valid")
    finally:
        server.close()


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("sessionId", "unrelated", -32602),
        ("turnId", "wrong-turn", -32025),
        ("generation", 7, -32602),
        ("operationId", "wrong-operation", -32602),
        ("attempt", 2, -32602),
        ("interactionId", "wrong-interaction", -32602),
        ("requestDigest", "0" * 64, -32602),
    ],
)
def test_rejected_target_keeps_wait_and_routes_nowhere(store, field, value, code):
    server = InteractionServer(store, InteractionHost({"options": ["A"]}))
    try:
        server.start()
        form = server.form()
        before = (server.host.calls, sum(r.record.kind == "op_planned" for r in store.read_state(THREAD)[1]))
        target = target_of(form) | {field: value}
        response = server.rpc(
            "turn/action",
            {
                "threadId": THREAD,
                "turnId": TURN,
                "actionId": "wrong",
                "action": {"kind": "respond", "target": target, "answer": text()},
            },
        )
        assert response["error"]["code"] == code, response
        if code == -32602:
            error(response, "stale_interaction")
        unchanged(server, form, before)
        assert not store.peek_inbox(THREAD)
    finally:
        server.close()


def test_stale_answer_to_next_question_and_cancelled_question(store):
    server = InteractionServer(store, InteractionHost(questions=2))
    try:
        server.start()
        old = server.form()
        assert "result" in server.answer(old, text())
        new = server.form()
        assert new["requestDigest"] != old["requestDigest"] and new["interactionId"] != old["interactionId"]
        before = (server.host.calls, sum(r.record.kind == "op_planned" for r in store.read_state(THREAD)[1]))
        error(server.answer(old, text(), action_id="late"), "stale_interaction")
        unchanged(server, new, before)
        response = server.rpc("turn/interrupt", {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel"})
        assert "result" in response
        server.join()
        calls = server.host.calls
        error(server.answer(new, text(), action_id="after-cancel"), "stale_interaction")
        assert server.host.calls == calls and server.form() is None
        assert store.read_state(THREAD)[0].turns[TURN].ended
    finally:
        server.close()


@pytest.mark.parametrize("typed,invalid", [(True, True), (True, False), (False, False), (False, True)])
def test_direct_inbox_uses_kernel_validation_and_labels_free_text(store, typed, invalid):
    server = InteractionServer(store, InteractionHost({"options": ["A"], "allow_custom_options": False}))
    try:
        server.start()
        form = server.form()
        target = InteractionReplyTarget.from_dict(target_of(form))
        if typed:
            answer = (
                InteractionAnswer("single", selected=("NOT_AN_OPTION",))
                if invalid
                else InteractionAnswer("text", text="NOT_AN_OPTION")
            )
            content = reply_content(target, answer)
        else:
            content = {"interaction_id": target.interaction_id, "text": "NOT_AN_OPTION"}
            if invalid:
                content["interaction_id"] = "wrong-id"
        before = server.host.calls
        server.server.kernel.push(THREAD, InboxItem("direct", "user", {"content": content}, TURN, 0))
        runtime = server.server.store.runtimes[THREAD]
        drive(store, THREAD, runtime=runtime, _one_turn=True)
        receipt = store.read_state(THREAD)[0].applied_inputs["direct"]._payload
        assert receipt["disposition"] == ("rejected" if invalid else "applied")
        if invalid:
            assert server.form() == form and server.host.calls == before
        else:
            tool = next(m for m in server.host.requests[-1].messages if m.role == "tool" and m.tool_call_id == "ask-0")
            assert json.loads(tool.content) == {"reply_mode": "free_text", "text": "NOT_AN_OPTION"}
    finally:
        server.close()


def test_turn_level_wait_has_null_operation_and_identical_replay_form(store):
    server = InteractionServer(store, InteractionHost(no_tool=True))
    try:
        server.start()
        form = server.form()
        assert form["operationId"] is form["attempt"] is None
        event = next(e for e in server.events() if e["type"] == "run_state_changed" and "interaction" in e["metadata"])
        assert event["metadata"]["interaction"] == form
        assert server.rpc("thread/read", {"threadId": THREAD})["result"]["state"]["pendingInteraction"] == form
        assert "result" in server.answer(form, text("reply"))
        assert any(m.role == "user" and m.content == "reply" for m in server.host.requests[-1].messages)
    finally:
        server.close()


def test_two_children_can_be_answered_in_order_and_require_root_and_child_authorization(store):
    def model(request):
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse("child done" if request.model == "child" else "parent done")
        if request.model == "child":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Child question", "options": ["A", "B"]})])
        return LLMResponse(
            "", [ToolCall("one", "run_one", {"task_description": "one"}), ToolCall("two", "run_two", {"task_description": "two"})]
        )

    agent = Agent(
        "parent",
        "Delegate.",
        model="root",
        tools=[
            Agent("one", "Ask.", model="child").as_tool(name="run_one"),
            Agent("two", "Ask.", model="child").as_tool(name="run_two"),
        ],
    )
    server = InteractionServer(store, InteractionHost(agent=agent, model=model))
    try:
        server.start()
        waits = server.server.kernel.waits(THREAD)
        assert len(waits) == 2 and len({sid for sid, _, _ in waits}) == 2
        first = server.form()
        assert first["sessionId"] != THREAD and server.status()["status"] == "child_running"
        assert not store.read_state(THREAD)[0].turns[TURN].ended
        server.host.denied.add(first["sessionId"])
        before = server.host.calls
        error(server.answer(first, text()), "permission_denied", -32027)
        assert server.host.calls == before and server.form() == first
        server.host.denied.clear()
        assert "result" in server.answer(first, text())
        assert ("turn/action", THREAD) in server.host.authorizations and (
            "turn/action",
            first["sessionId"],
        ) in server.host.authorizations
        second = server.form()
        assert second["sessionId"] != first["sessionId"]
        assert not store.read_state(THREAD)[0].turns[TURN].ended
        calls = server.host.calls
        error(server.answer(first, text(), action_id="finished-child"), "stale_interaction")
        assert server.host.calls == calls and server.form() == second
        assert "result" in server.answer(second, {"kind": "single", "selected": ["A"], "custom": []}, action_id="second")
        assert store.read_state(THREAD)[0].turns[TURN].ended
        assert server.rpc("thread/read", {"threadId": THREAD})["result"]["turns"][-1]["result"]["finalOutput"] == "parent done"
    finally:
        server.close()


def test_canonical_fixture_is_from_real_producer():
    actual = AUTHOR["produce"]()
    path = Path(__file__).parents[1] / "fixtures/parity/app_server_interactions.json"
    assert actual == json.loads(path.read_text())


def test_nested_child_target_follows_the_admitted_graph(store):
    leaf = Agent("leaf", "Ask.", model="leaf")
    middle = Agent("middle", "Delegate.", model="middle", tools=[leaf.as_tool(name="run_leaf")])
    root = Agent("parent", "Delegate.", model="root", tools=[middle.as_tool(name="run_middle")])

    def model(request):
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse(request.model + " done")
        if request.model == "leaf":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Nested question"})])
        name = "run_middle" if request.model == "root" else "run_leaf"
        return LLMResponse("", [ToolCall("delegate", name, {"task_description": "work"})])

    server = InteractionServer(store, InteractionHost(agent=root, model=model))
    try:
        server.start()
        form = server.form()
        from vv_agent.session.tool_result_read import validate_child_graph

        chain, tid = validate_child_graph(store, THREAD, form["sessionId"])
        assert len(chain) == 2 and tid == TURN
        assert "result" in server.answer(form, text())
        assert store.read_state(THREAD)[0].turns[TURN].ended
        assert server.host.calls == 6
    finally:
        server.close()


def test_parent_cancel_closes_both_waiting_children_and_rejects_late_answer(store):
    child = Agent("child", "Ask.", model="child")
    parent = Agent("parent", "Delegate.", model="root", tools=[child.as_tool(name="run_one"), child.as_tool(name="run_two")])

    def model(request):
        if request.model == "child":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Need input"})])
        return LLMResponse(
            "", [ToolCall("one", "run_one", {"task_description": "one"}), ToolCall("two", "run_two", {"task_description": "two"})]
        )

    server = InteractionServer(store, InteractionHost(agent=parent, model=model))
    try:
        server.start()
        waits = server.server.kernel.waits(THREAD)
        assert len(waits) == 2
        form = server.form()
        calls = server.host.calls
        assert "result" in server.rpc("turn/interrupt", {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel-parent"})
        server.join()
        error(server.answer(form, text(), action_id="late-child"), "stale_interaction")
        assert server.host.calls == calls
        assert store.read_state(THREAD)[0].turns[TURN].ended
        for sid, _, wait in waits:
            state, _, _ = store.read_state(sid)
            assert state.turns[wait["turn_id"]].ended
    finally:
        server.close()


@pytest.mark.parametrize(
    "form",
    [
        {"question": "汉" * 5462},
        {"options": [str(i) for i in range(65)]},
        {"options": ["汉" * 342]},
        {"question": "q" * 16000, "options": [str(i) + "x" * 1000 for i in range(64)]},
    ],
)
def test_oversize_new_form_never_opens_a_wait(store, form):
    server = InteractionServer(store, InteractionHost(form))
    try:
        server.start()
        state, rows, _ = store.read_state(THREAD)
        assert not state.waits and server.form() is None
        assert not any(r.record.kind == "op_parked" for r in rows)
        errors = [
            r.record.payload["result"]
            for r in rows
            if r.record.kind == "op_completed" and r.record.payload["result"].get("error_code") == "interaction_form_too_large"
        ]
        assert errors and all(e["directive"] == "continue" and e["status_code"] == "ERROR" for e in errors)
        assert all(json.loads(e["content"])["error_code"] == "interaction_form_too_large" for e in errors)
    finally:
        server.close()


def test_child_message_cannot_blindly_consume_a_user_wait(store):
    child = Agent("child", "Ask.", model="child")
    parent = Agent("parent", "Delegate.", model="root", tools=[child.as_tool(name="run_child")])

    def model(request):
        if request.model == "child":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Need input"})])
        return LLMResponse("", [ToolCall("child", "run_child", {"task_description": "work"})])

    server = InteractionServer(store, InteractionHost(agent=parent, model=model))
    try:
        server.start()
        form = server.form()
        manager = server.server.store.runtimes[THREAD].child_tasks(store, THREAD)
        manager.message(form["sessionId"], "steering", "not a user answer")
        child_runtime = server.server.store.runtimes[THREAD].child_runtime(store, form["sessionId"])
        before = server.host.calls
        drive(store, form["sessionId"], runtime=child_runtime, _one_turn=True)
        assert server.form() == form and server.host.calls == before
        assert any(i.item.kind == "steer" for i in store.peek_inbox(form["sessionId"]))
    finally:
        server.close()


def test_historical_oversize_form_remains_parked_and_reports_explicit_error(store, tmp_path):
    from dataclasses import replace

    from vv_agent.session.records import SessionSpec
    from vv_agent.session.sqlite import SQLiteStore

    first = InteractionServer(store)
    try:
        first.start()
        original = store.read_state(THREAD)[1]
        created = original[0].record.payload
        original_calls = first.host.calls
        with SQLiteStore.standalone(str(tmp_path / "historical.sqlite")) as historical:
            historical.install_schema()
            with historical.atomic() as tx:
                tx.create(
                    SessionSpec(THREAD, created["principal"], created["workspace"], attributes=created["attributes"]),
                    consumers=("events", "app_server", "traces"),
                )
                inputs = [InboxItem(**row.record.payload["input"]) for row in original if row.record.kind == "input_applied"]
                for item in inputs:
                    tx.push(THREAD, item)
            lease = historical.acquire(THREAD, owner="old-writer", ttl_ms=60000)
            assert lease is not None
            old_records = []
            for row in original[1:]:
                rec = row.record
                if rec.kind == "op_parked" and rec.payload["handle"]["kind"] == "user":
                    payload = rec.payload
                    payload["handle"]["question"] = "h" * 16385
                    payload["interaction_result"]["metadata"]["question"] = "h" * 16385
                    rec = replace(rec, payload=payload)
                old_records.append(rec)
            with historical.atomic() as tx:
                tx.append(
                    THREAD,
                    lease=lease,
                    expected_seq=1,
                    commit_id="historical-import",
                    records=tuple(old_records),
                    consume_input_ids=tuple(item.input_id for item in inputs),
                )
            historical.release(lease)
            old = historical.read_state(THREAD)
            rebuilt = InteractionServer(historical)
            try:
                for method in ("thread/status", "thread/read", "thread/resume"):
                    params = {"threadId": THREAD}
                    if method == "thread/resume":
                        params["subscribe"] = False
                    error(rebuilt.rpc(method, params), "interaction_form_too_large")
                assert historical.read_state(THREAD)[1] == old[1]
                assert historical.read_state(THREAD)[0].waits == old[0].waits
                assert rebuilt.host.calls == 0 and first.host.calls == original_calls
            finally:
                rebuilt.close()
    finally:
        first.close()


def test_child_cancel_from_wire_driven_host_tool_rejects_its_late_answer(store):
    from threading import Event
    from time import monotonic

    from vv_agent import function_tool

    child_ids, servers = [], []

    @function_tool
    def stop_child() -> str:
        server = servers[0]
        runtime = server.server.store.runtimes[THREAD]
        runtime.child_tasks(store, THREAD).handle(child_ids[0]).cancel()
        return "cancel requested"

    child = Agent("child", "Ask.", model="child")
    parent = Agent("parent", "Launch, ask and cancel.", model="root", tools=[child.as_background_task(name="launch"), stop_child])

    def model(request):
        if request.model == "child":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Child question"})])
        tool_ids = {m.tool_call_id for m in request.messages if m.role == "tool"}
        if "launch" not in tool_ids:
            return LLMResponse("", [ToolCall("launch", "launch", {"task_description": "work"})])
        if "ask-0" not in tool_ids:
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Stop the child?"})])
        if "cancel-child" not in tool_ids:
            return LLMResponse("", [ToolCall("cancel-child", "stop_child", {})])
        return LLMResponse("parent done")

    server = InteractionServer(store, InteractionHost(agent=parent, model=model))
    servers.append(server)
    try:
        server.start()
        deadline = monotonic() + 10
        while len(server.server.kernel.waits(THREAD)) < 2 and monotonic() < deadline:
            Event().wait(0.01)
        waits = server.server.kernel.waits(THREAD)
        child_form = next(wait["interaction"] for sid, _, wait in waits if sid != THREAD)
        child_ids.append(child_form["sessionId"])
        root_form = next(wait["interaction"] for sid, _, wait in waits if sid == THREAD)
        assert server.form() == root_form
        assert "result" in server.answer(root_form, text("yes"))
        deadline = monotonic() + 10
        while not store.read_state(child_ids[0])[0].turns[child_form["turnId"]].ended and monotonic() < deadline:
            Event().wait(0.01)
        state, rows, _ = store.read_state(child_ids[0])
        failures = [
            r.record.payload["result"]["content"]
            for r in store.read_state(THREAD)[1]
            if r.record.kind == "op_completed" and r.record.payload["result"].get("error_code")
        ]
        assert not failures, "\n".join(failures)
        assert state.turns[child_form["turnId"]].ended
        assert any(r.record.kind == "turn_ended" and r.record.payload["status"] == "cancelled" for r in rows)
        calls = server.host.calls
        error(server.answer(child_form, text(), action_id="after-child-cancel"), "stale_interaction")
        assert server.host.calls == calls
    finally:
        server.close()


def test_approval_precedes_user_question_with_stable_identity(store):
    from vv_agent import function_tool

    effects = []

    @function_tool(needs_approval=True)
    def change() -> str:
        effects.append("changed")
        return "changed"

    parent = Agent(
        "parent",
        "Delegate.",
        model="root",
        tools=[
            Agent("user", "Ask.", model="user").as_tool(name="user_child"),
            Agent("approval", "Approve.", model="approval", tools=[change]).as_tool(name="approval_child"),
        ],
    )

    def model(request):
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse("done")
        if request.model == "user":
            return LLMResponse("", [ToolCall("ask-0", "ask_user", {"question": "Question"})])
        if request.model == "approval":
            return LLMResponse("", [ToolCall("change", "change", {})])
        return LLMResponse(
            "",
            [
                ToolCall("user", "user_child", {"task_description": "ask"}),
                ToolCall("approval", "approval_child", {"task_description": "approve"}),
            ],
        )

    server = InteractionServer(store, InteractionHost(agent=parent, model=model))
    try:
        from threading import Event
        from time import monotonic

        server.start(settle=False)
        deadline = monotonic() + 10
        while len(server.server.kernel.waits(THREAD)) < 2 and monotonic() < deadline:
            Event().wait(0.01)
        form = server.form()
        assert form["kind"] == "approval" and set(form) == {"kind", "sessionId", "turnId", "requestId"}
        user = next(w["interaction"] for _, _, w in server.server.kernel.waits(THREAD) if "interaction" in w)
        assert not effects
        result = server.rpc(
            "approval/resolve", {"threadId": THREAD, "turnId": TURN, "requestId": form["requestId"], "decision": "allow"}
        )
        assert "result" in result, result
        server.join()
        assert effects == ["changed"] and server.form() == user
        assert "result" in server.answer(user, text())
        assert store.read_state(THREAD)[0].turns[TURN].ended
    finally:
        server.close()


@pytest.mark.parametrize(
    "answer",
    [
        {"kind": "multi", "selected": ["A", "A"], "custom": []},
        {"kind": "multi", "selected": ["NOT_AN_OPTION"], "custom": []},
        {"kind": "multi", "selected": [], "custom": []},
        {"kind": "multi", "selected": [], "custom": ["Other", " Other "]},
        {"kind": "multi", "selected": [], "custom": [" A "]},
        {"kind": "multi", "selected": [], "custom": [str(i) for i in range(65)]},
        {"kind": "multi", "selected": [], "custom": [str(i).zfill(4) + "x" * 1020 for i in range(64)]},
    ],
)
def test_invalid_multi_answers_keep_the_retained_question(store, answer):
    server = InteractionServer(
        store, InteractionHost({"options": ["A", "B"], "selection_type": "multi", "allow_custom_options": True})
    )
    try:
        server.start()
        form = server.form()
        before = (server.host.calls, sum(r.record.kind == "op_planned" for r in store.read_state(THREAD)[1]))
        error(server.answer(form, answer), "invalid_interaction_answer")
        unchanged(server, form, before)
    finally:
        server.close()


@pytest.mark.parametrize(
    "mutation", ["missing_turn", "missing_generation", "wrong_generation", "forged_text", "extra_extension_field", "follow_up"]
)
def test_direct_typed_inbox_cannot_bypass_envelope_or_answer_validation(store, mutation):
    server = InteractionServer(store)
    try:
        server.start()
        form = server.form()
        target = InteractionReplyTarget.from_dict(target_of(form))
        content = reply_content(target, InteractionAnswer("text", text="answer"))
        tid, generation, kind = TURN, 0, "user"
        if mutation == "missing_turn":
            tid = None
        elif mutation == "missing_generation":
            generation = None
        elif mutation == "wrong_generation":
            generation = 1
        elif mutation == "forged_text":
            content["text"] = "forged"
        elif mutation == "extra_extension_field":
            content["metadata"]["interaction_response"]["extra"] = True
        else:
            kind = "follow_up"
        before = server.host.calls
        server.server.kernel.push(THREAD, InboxItem("bad-direct", kind, {"content": content}, tid, generation))
        drive(store, THREAD, runtime=server.server.store.runtimes[THREAD], _one_turn=True)
        assert store.read_state(THREAD)[0].applied_inputs["bad-direct"].payload["disposition"] == "rejected"
        assert server.form() == form and server.host.calls == before
    finally:
        server.close()
