"""Yielding completes an observation and applies steering in one real-store commit."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from vv_agent.session import children
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .helpers import (
    fixture_child_spec,
    fixture_key_provider,
    push_fixture_input,
    read_fixture_state,
    wrap_fixture_store,
)
from .test_batch_children import complete, deliver, rows
from .test_children import child_task, fixture_runtime, start


@pytest.fixture
def store(store):
    wrapped = wrap_fixture_store(store)
    with wrapped.atomic():
        wrapped._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    return wrapped


def waiting_runtime(store, database, *, count=1, declared=True, callback=None):
    observed = []

    def next_request(request):
        observed.append(request)
        return LLMResponse("Parent awaiting user")

    rt = fixture_runtime(
        store,
        database,
        "s",
        [LLMResponse("", [ToolCall("child-call", "child_task", {})]), next_request],
        [child_task],
    )
    rt.config.no_tool_policy = "wait_user"

    def admit(_plan):
        return [
            children.ChildSession(
                fixture_child_spec(store, "s", f"yield-child-{i}"),
                "work",
                yield_on_steer=declared,
            )
            for i in range(count)
        ]

    rt.children["child_task"] = admit if callback is None else children.ChildTool(admit, on_yield=callback)
    return rt, observed


def hydrated_rows(store, sid="s", kind=None):
    _, hydrated, _ = read_fixture_state(store, sid)
    return [r for r in hydrated if kind is None or r.record.kind == kind]


def steer(store, *, input_id="steer", generation=0):
    tid = read_state(store, "s")[0].active_turn_id
    push_fixture_input(store, "s", InboxItem(input_id, "steer", {"content": "Change direction now"}, tid, generation))


@pytest.mark.parametrize("count", [1, 3])
def test_steer_yields_foreground_children_atomically_and_later_notifies(store, database, count):
    start(store)
    rt, observed = waiting_runtime(store, database, count=count)
    drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[0]
    active_children = []
    for i in range(count):
        sid = f"yield-child-{i}"
        provider = YieldProvider(declared=False)
        child_rt = fixture_runtime(
            store,
            database,
            sid,
            [LLMResponse("", [ToolCall("work", "child_task", {})]), LLMResponse(sid + " done")],
            [child_task],
            providers={"child_task": provider},
        )
        drive(store, sid, runtime=child_rt)
        assert read_state(store, sid)[0].active_turn_id is not None
        active_children.append((sid, child_rt, provider))
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipts = [r for r in rows(store, kind="op_completed") if r.record.operation_id == park.record.operation_id]
    assert len(receipts) == 1
    receipt = receipts[0]
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert receipt.commit_id == applied.commit_id
    assert receipt.record.payload["yielded_by"] == "steer"
    messages = observed[0].messages
    assert any("Change direction now" in str(m.content) for m in messages)
    assert any("Wait interrupted by new input" in str(m.content) for m in messages)
    for sid, child_rt, provider in active_children:
        assert not rows(store, sid, "turn_ended")
        assert read_state(store, sid)[0].active_turn_id is not None
        assert not any(i.item.kind == "control" for i in store.peek_inbox(sid))
        from vv_agent.types import ToolExecutionResult

        h = provider.handle
        child_turn_id = read_state(store, sid)[0].active_turn_id
        assert child_turn_id is not None
        push_fixture_input(
            store,
            sid,
            InboxItem(
                "child-work-done",
                "provider_result",
                {
                    "operation_id": h["operation_id"],
                    "attempt": h["attempt"],
                    "request_digest": h["request_digest"],
                    "provider_binding": h["provider"],
                    "result": ToolExecutionResult("work", "work done").to_dict(),
                    "usage": {},
                    "evidence": [h["evidence"]],
                },
                child_turn_id,
                0,
            ),
        )
        drive(store, sid, runtime=child_rt)
        assert rows(store, sid, "turn_ended")
        deliver(store, sid)
    drive(store, "s", runtime=rt)
    notices = [
        r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["reason"] == "background child notification"
    ]
    assert len(notices) == count, [
        (
            r.record.payload["input"]["input_id"],
            r.record.payload["input"]["kind"],
            r.record.payload["disposition"],
            r.record.payload["reason"],
        )
        for r in rows(store, kind="input_applied")
        if r.record.payload["input"]["kind"] == "child_result"
    ]
    from vv_agent.session.context import project_context

    state, retained, _ = read_fixture_state(store, "s")
    assert sum(m.role == "user" and "Child completed:" in str(m.content) for m in project_context(retained, state)) == count
    assert len([r for r in rows(store, kind="op_completed") if r.record.operation_id == park.record.operation_id]) == 1


def test_undeclared_foreground_wait_keeps_steer_pending(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database, declared=False)
    drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[0]
    assert "yield_on_steer" not in park.record.payload
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert all("yielded_by" not in r.record.payload for r in rows(store, kind="op_completed"))


@pytest.mark.parametrize("completed_first", [False, True])
def test_same_watermark_child_terminal_precedes_steer(store, database, completed_first):
    seen_handles = []

    def render(plan, handles, reason):
        seen_handles.append(handles)
        return ToolExecutionResult(plan.payload["request"]["id"], "Child and steer observed")

    start(store)
    rt, observed = waiting_runtime(store, database, count=2, callback=render)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    if completed_first:
        deliver(store, "yield-child-0")
        steer(store)
    else:
        steer(store)
        deliver(store, "yield-child-0")
    push_fixture_input(store, "s", InboxItem("stale", "user", {"content": "stale"}, "s/turn/other", 0))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    child_applied = next(
        r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"
    )
    steer_applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    yielded = [r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer"]
    stale = next(r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "stale")
    assert child_applied.record.payload["reason"] == "batch member completed"
    assert len(yielded) == 1
    assert {child_applied.commit_id, steer_applied.commit_id, yielded[0].commit_id} == {steer_applied.commit_id}
    assert child_applied.seq < steer_applied.seq < stale.seq
    assert stale.record.payload["disposition"] == "rejected"
    assert stale.commit_id != steer_applied.commit_id
    assert len(seen_handles) == 1
    assert [handle["status"] for handle in seen_handles[0]] == ["completed", "running"]
    assert seen_handles[0][0]["result"] == "yield-child-0 done"


def test_yield_callback_hydrates_persisted_child_and_marks_current_terminal_pending(store, database):
    seen_handles = []

    def render(plan, handles, reason):
        seen_handles.append(handles)
        return ToolExecutionResult(plan.payload["request"]["id"], "Persisted child observed")

    start(store)
    rt, observed = waiting_runtime(store, database, count=3, callback=render)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    assert not seen_handles

    complete(store, database, "yield-child-1")
    deliver(store, "yield-child-1")
    steer(store)
    drive(store, "s", runtime=rt)

    assert len(observed) == 1
    assert len(seen_handles) == 1
    handles = seen_handles[0]
    assert [handle["status"] for handle in handles] == ["completed", "completed", "running"]
    assert handles[0]["result"] == "yield-child-0 done"
    assert handles[1]["result"] == "yield-child-1 done"
    assert handles[0]["terminal_digest"]
    child_inputs = [r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
    assert len(child_inputs) == 2
    assert child_inputs[0].commit_id != child_inputs[1].commit_id
    yielded = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer")
    steer_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert child_inputs[1].commit_id == steer_input.commit_id == yielded.commit_id

    with store.atomic() as tx:
        assert children.child_delivery(store, tx, "yield-child-1") == 0
    drive(store, "s", runtime=rt)
    assert len(seen_handles) == 1
    assert not rows(store, "yield-child-2", "turn_ended")


def test_complete_child_batch_commits_before_steer_without_yield(store, database, monkeypatch):
    from vv_agent.session import _recovery_phase

    child_results = []
    result_for_children = _recovery_phase.result_for_children

    def observe_child_result(*args, **kwargs):
        child_results.append((args, kwargs))
        return result_for_children(*args, **kwargs)

    monkeypatch.setattr(_recovery_phase, "result_for_children", observe_child_result)
    start(store)
    rt, observed = waiting_runtime(store, database, count=2)
    drive(store, "s", runtime=rt)
    for sid in ("yield-child-0", "yield-child-1"):
        complete(store, database, sid)
        deliver(store, sid)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    child_inputs = [r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
    assert len(child_inputs) == 2
    steer_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert len({r.commit_id for r in child_inputs}) == 1
    assert child_inputs[0].commit_id != steer_input.commit_id
    assert child_inputs[-1].seq < steer_input.seq
    assert len(child_results) == 1
    assert not any("yielded_by" in r.record.payload for r in rows(store, kind="op_completed"))


def test_held_steer_commits_child_records_once_and_keeps_steer_pending(store, database, monkeypatch):
    from vv_agent.session import _recovery_phase
    from vv_agent.tools.function import function_tool

    child_results = []
    result_for_children = _recovery_phase.result_for_children

    def observe_child_result(*args, **kwargs):
        child_results.append((args, kwargs))
        return result_for_children(*args, **kwargs)

    monkeypatch.setattr(_recovery_phase, "result_for_children", observe_child_result)

    @function_tool
    def blocker() -> str:
        raise AssertionError("provider owns blocker")

    start(store)
    rt, observed = waiting_runtime(store, database)
    rt.agent.tools.append(blocker)
    rt.providers["blocker"] = YieldProvider(declared=False)
    rt.llm.steps[0] = LLMResponse(
        "",
        [
            ToolCall("child-call", "child_task", {}),
            ToolCall("blocker-call", "blocker", {}),
        ],
    )
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    steer(store)

    drive(store, "s", runtime=rt)
    assert not observed
    assert len(child_results) == 1
    child_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result")
    assert child_input.record.payload["disposition"] == "applied"
    assert [item.item.input_id for item in store.peek_inbox("s")] == ["steer"]
    assert not any("yielded_by" in r.record.payload for r in rows(store, kind="op_completed"))


def test_child_result_then_inputs_in_inbox_order_before_steer(store, database):
    callbacks = []
    admitted = []

    def render(plan, handles, reason):
        callbacks.append((plan.operation_id, handles, reason))
        return ToolExecutionResult(plan.payload["request"]["id"], "Steered after ordered inputs")

    start(store)
    rt, observed = waiting_runtime(store, database, count=1, callback=render)

    def admit(_plan):
        sid = f"yield-child-{len(admitted)}"
        admitted.append(sid)
        return children.ChildSession(fixture_child_spec(store, "s", sid), "work", yield_on_steer=True)

    rt.children["child_task"] = children.ChildTool(admit, on_yield=render)
    rt.llm.steps.insert(
        1,
        LLMResponse("", [ToolCall("next-child-call", "child_task", {})]),
    )
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    push_fixture_input(store, "s", InboxItem("stale", "user", {"content": "stale"}, "s/turn/other", 0))
    push_fixture_input(store, "s", InboxItem("selection", "model_selection", {"selection": None}))
    push_fixture_input(store, "s", InboxItem("wrong-target", "control", {"action": "suspend"}, "other-turn", 0))
    input_seqs = {item.item.input_id: item.input_seq for item in store.peek_inbox("s")}
    drive(store, "s", runtime=rt)
    assert admitted == ["yield-child-0", "yield-child-1"]

    tid = read_state(store, "s")[0].active_turn_id
    push_fixture_input(store, "s", InboxItem("steer", "steer", {"content": "Change after child"}, tid, 0))
    input_seqs.update({item.item.input_id: item.input_seq for item in store.peek_inbox("s")})

    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert len(callbacks) == 1
    applied = [
        r
        for r in hydrated_rows(store, kind="input_applied")
        if r.record.payload["input"]["input_id"] in {"child", "stale", "selection", "wrong-target", "steer"}
        or r.record.payload["input"]["kind"] == "child_result"
    ]
    child_input = applied[0]
    assert [r.record.payload["input"]["input_id"] for r in applied] == [
        child_input.record.payload["input"]["input_id"],
        "stale",
        "selection",
        "wrong-target",
        "steer",
    ]
    applied_input_seqs = [input_seqs[r.record.payload["input"]["input_id"]] for r in applied]
    assert applied_input_seqs == sorted(applied_input_seqs)
    assert [r.seq for r in applied] == sorted(r.seq for r in applied)
    steer_input = applied[-1]
    yielded = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer")
    assert child_input.commit_id != steer_input.commit_id
    assert steer_input.commit_id == yielded.commit_id
    assert len({r.commit_id for r in applied}) == len(applied)
    dispositions = {r.record.payload["input"]["input_id"]: r.record.payload["disposition"] for r in applied}
    assert dispositions == {
        child_input.record.payload["input"]["input_id"]: "applied",
        "stale": "rejected",
        "selection": "applied",
        "wrong-target": "rejected",
        "steer": "applied",
    }


def test_two_child_results_and_steer_share_one_commit(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database, count=4)
    drive(store, "s", runtime=rt)
    for sid in ("yield-child-1", "yield-child-2"):
        complete(store, database, sid)
        deliver(store, sid)
    steer(store)

    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    child_inputs = [r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
    assert [r.record.payload["input"]["payload"]["session_id"] for r in child_inputs] == [
        "yield-child-1",
        "yield-child-2",
    ]
    steer_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    yielded = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer")
    assert {r.commit_id for r in [*child_inputs, steer_input, yielded]} == {steer_input.commit_id}
    assert not rows(store, "yield-child-0", "turn_ended")
    assert not rows(store, "yield-child-3", "turn_ended")


def test_pending_background_child_keeps_ready_child_and_steer_in_one_commit(store, database):
    from vv_agent.tools.function import function_tool

    @function_tool
    def background_task() -> str:
        raise AssertionError("session kernel owns background_task")

    @function_tool
    def foreground_task() -> str:
        raise AssertionError("session kernel owns foreground_task")

    observed = []

    def next_request(request):
        observed.append(request)
        return LLMResponse("Parent awaiting user")

    commits = []

    def observe_commit(point, _record):
        if point == "after_commit":
            state, _, _ = read_state(store, "s")
            commits.append(
                (
                    store.read("s", limit=1).head_seq,
                    state,
                    tuple(stored.item.input_id for stored in store.peek_inbox("s")),
                )
            )

    start(store)
    rt = fixture_runtime(
        store,
        database,
        "s",
        [
            LLMResponse(
                "",
                [
                    ToolCall("background-call", "background_task", {}),
                    ToolCall("foreground-call", "foreground_task", {}),
                ],
            ),
            next_request,
        ],
        [background_task, foreground_task],
    )
    rt.config.no_tool_policy = "wait_user"
    rt.hook = observe_commit
    rt.children["background_task"] = lambda _plan: children.ChildSession(
        fixture_child_spec(store, "s", "background-child"), "background work", background=True
    )
    rt.children["foreground_task"] = lambda _plan: [
        children.ChildSession(fixture_child_spec(store, "s", f"foreground-child-{index}"), "foreground work", yield_on_steer=True)
        for index in range(2)
    ]
    drive(store, "s", runtime=rt)

    complete(store, database, "background-child")
    deliver(store, "background-child")
    complete(store, database, "foreground-child-0")
    deliver(store, "foreground-child-0")
    steer(store)
    pending, ready, stored_steer = store.peek_inbox("s")
    assert [stored.item.kind for stored in (pending, ready, stored_steer)] == ["child_result", "child_result", "steer"]
    assert pending.item.payload["session_id"] == "background-child"
    assert ready.item.payload["session_id"] == "foreground-child-0"

    before_head = store.read("s", limit=1).head_seq
    drive(store, "s", runtime=rt)

    _, records, _ = read_fixture_state(store, "s")
    batches = []
    batch_head = before_head
    for commit_head, batch_state, inbox_ids in commits:
        batches.append(
            (
                [r for r in records if batch_head < r.seq <= commit_head],
                batch_state,
                inbox_ids,
            )
        )
        batch_head = commit_head
    steer_batch_index = next(
        index
        for index, (batch, _, _) in enumerate(batches)
        if any(
            r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == stored_steer.item.input_id
            for r in batch
        )
    )
    steer_batch, steer_batch_state, steer_batch_inbox = batches[steer_batch_index]
    pending_key = (
        pending.item.payload["operation_id"],
        pending.item.payload["attempt"],
        pending.item.payload["session_id"],
    )
    assert pending.item.input_id not in steer_batch_state.applied_inputs
    assert pending_key not in (steer_batch_state.child_completions or {})
    assert pending.item.input_id not in {
        r.record.payload["input"]["input_id"] for r in steer_batch if r.record.kind == "input_applied"
    }
    assert steer_batch_inbox == (pending.item.input_id,)

    ready_input = next(
        r
        for r in steer_batch
        if r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == ready.item.input_id
    )
    steer_input = next(
        r
        for r in steer_batch
        if r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == stored_steer.item.input_id
    )
    yielded = [r for r in steer_batch if r.record.kind == "op_completed" and r.record.payload.get("yielded_by") == "steer"]
    assert len(yielded) == 1
    assert [r.record.payload["input"]["input_id"] for r in steer_batch if r.record.kind == "input_applied"] == [
        ready.item.input_id,
        stored_steer.item.input_id,
    ]
    assert len({ready_input.commit_id, steer_input.commit_id, yielded[0].commit_id}) == 1
    assert ready_input.seq < steer_input.seq < yielded[0].seq
    assert yielded[0].record.operation_id == ready.item.payload["operation_id"]
    assert len(observed) == 1
    assert any("Change direction now" in str(message.content) for message in observed[0].messages)
    hydrated_yield = next(r for r in hydrated_rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer")
    assert hydrated_yield.record.payload["result"]["metadata"]["children"] == [
        {"session_id": "foreground-child-0", "status": "completed"},
        {"session_id": "foreground-child-1", "status": "running"},
    ]


def test_same_watermark_completed_child_extra_joins_steer_commit(store, database):
    from vv_agent.tools.function import function_tool

    @function_tool
    def provider_task() -> str:
        raise AssertionError("provider owns provider_task")

    start(store)
    rt, observed = waiting_runtime(store, database)
    rt.agent.tools.append(provider_task)
    rt.providers["provider_task"] = YieldProvider()
    rt.llm.steps[0] = LLMResponse(
        "",
        [
            ToolCall("child-call", "child_task", {}),
            ToolCall("provider-call", "provider_task", {}),
        ],
    )
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    steer(store)
    push_fixture_input(store, "s", InboxItem("stale", "user", {"content": "stale"}, "s/turn/other", 0))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1

    child_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result")
    child_completion = next(
        r for r in rows(store, kind="op_completed") if r.record.operation_id == child_input.record.payload["target_operation_id"]
    )
    steer_input = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    yielded = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer")
    stale = next(r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "stale")
    assert {child_input.commit_id, child_completion.commit_id, steer_input.commit_id, yielded.commit_id} == {
        steer_input.commit_id
    }
    assert child_input.seq < child_completion.seq < steer_input.seq < yielded.seq < stale.seq
    assert stale.record.payload["disposition"] == "rejected"
    assert stale.commit_id != steer_input.commit_id


@pytest.mark.parametrize("raises", [False, True])
def test_host_callback_or_default_fallback_is_model_visible(store, database, raises):
    from vv_agent.types import ToolExecutionResult

    calls = []

    def render(plan, handles, reason):
        calls.append((plan.operation_id, handles, reason))
        if raises:
            raise RuntimeError("private callback details")
        return ToolExecutionResult(plan.payload["request"]["id"], "Host rendered interruption")

    start(store)
    rt, observed = waiting_runtime(store, database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    assert calls[0][2] == "steer"
    expected = "Wait interrupted by new input" if raises else "Host rendered interruption"
    assert any(expected in str(m.content) for m in observed[0].messages)
    assert "private callback details" not in str([r.record.payload for r in rows(store)])


def test_mixed_child_members_wait_for_undeclared_sibling(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database, count=2)
    admit = rt.children["child_task"]
    rt.children["child_task"] = lambda plan: [replace(c, yield_on_steer=i == 0) for i, c in enumerate(admit(plan))]
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    complete(store, database, "yield-child-1")
    deliver(store, "yield-child-1")
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    result = next(
        r.record.payload["result"] for r in hydrated_rows(store, kind="op_completed") if "yielded_by" in r.record.payload
    )
    assert result["metadata"]["running_children"] == ["yield-child-0"]
    assert not rows(store, "yield-child-0", "turn_ended")


class YieldProvider:
    def __init__(self, *, declared=True, callback=False):
        self.handle: dict[str, Any] = {}
        self.submits = 0
        self.queries = 0
        self.cancels = 0
        self.declared = declared
        self.callback = callback
        self.on_yield: Callable[[Any, Any, str], ToolExecutionResult] | None = self._render if callback is not None else None

    def submit(self, plan, *, context):
        from vv_agent.session.providers import Accepted

        self.submits += 1
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "provider-accepted",
            "query_ref": None,
            "cancel_ref": None,
        }
        return Accepted(self.handle, yield_on_steer=self.declared)

    def query(self, handle):
        from vv_agent.session.providers import Accepted

        self.queries += 1
        return Accepted(handle, yield_on_steer=self.declared)

    def cancel(self, handle):
        self.cancels += 1
        raise AssertionError("a yielded provider handle must never be cancelled")

    def authenticate(self, item, plan):
        return item.payload["request_digest"] == self.handle["request_digest"]

    def _render(self, plan, handle, reason):
        from vv_agent.types import ToolExecutionResult

        assert handle == self.handle and reason == "steer"
        if not self.callback:
            raise RuntimeError("renderer unavailable")
        return ToolExecutionResult(plan.payload["request"]["id"], "Provider observation interrupted")


def provider_runtime(store, database, provider):
    rt, observed = waiting_runtime(store, database)
    rt.children.clear()
    rt.providers["child_task"] = provider
    return rt, observed


@pytest.mark.parametrize("callback", [None, False, True])
def test_accepted_provider_closes_without_cancel_and_late_result_is_audit(store, database, callback):
    from vv_agent.types import ToolExecutionResult

    start(store)
    provider = YieldProvider(callback=callback)
    rt, observed = provider_runtime(store, database, provider)
    drive(store, "s", runtime=rt)
    query_count = provider.queries
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert provider.queries == query_count
    receipt = next(r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    hydrated_receipt = next(r for r in hydrated_rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    assert "provider-accepted" in hydrated_receipt.record.payload["evidence"]
    expected = "Provider observation interrupted" if callback else "Wait interrupted by new input"
    assert any(expected in str(m.content) for m in observed[0].messages)
    h = provider.handle
    tid = receipt.record.turn_id
    late = InboxItem(
        "late",
        "provider_result",
        {
            "operation_id": h["operation_id"],
            "attempt": h["attempt"],
            "request_digest": h["request_digest"],
            "provider_binding": h["provider"],
            "result": ToolExecutionResult("child-call", "Late real output").to_dict(),
            "usage": {},
            "evidence": [h["evidence"]],
        },
        tid,
        0,
    )
    push_fixture_input(store, "s", late)
    push_fixture_input(
        store,
        "s",
        InboxItem(
            "late-accept",
            "provider_evidence",
            {
                "operation_id": h["operation_id"],
                "attempt": h["attempt"],
                "request_digest": h["request_digest"],
                "handle": h,
            },
            tid,
            0,
        ),
    )
    drive(store, "s", runtime=rt)
    state = read_fixture_state(store, "s")[0]
    assert all(
        state.applied_inputs[i].payload["reason"] == "yielded provider handle: audit only" for i in ("late", "late-accept")
    )
    assert next(r for r in rows(store) if r.seq == receipt.seq).record.encode() == receipt.record.encode()
    push_fixture_input(store, "s", InboxItem("cancel", "control", {"action": "cancel"}, tid, 0))
    drive(store, "s", runtime=rt)
    assert provider.cancels == 0 and provider.queries == query_count


def test_unfinished_undeclared_operation_blocks_whole_batch(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    provider = YieldProvider(declared=False)
    from vv_agent.tools.function import function_tool

    @function_tool
    def command() -> str:
        raise AssertionError("provider owns command")

    rt.agent.tools.append(command)
    rt.providers["command"] = provider
    rt.llm.steps[0] = LLMResponse("", [ToolCall("child-call", "child_task", {}), ToolCall("command", "command", {})])
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    h = provider.handle
    from vv_agent.types import ToolExecutionResult

    push_fixture_input(
        store,
        "s",
        InboxItem(
            "command-done",
            "provider_result",
            {
                "operation_id": h["operation_id"],
                "attempt": h["attempt"],
                "request_digest": h["request_digest"],
                "provider_binding": h["provider"],
                "result": ToolExecutionResult("command", "Command done").to_dict(),
                "usage": {},
                "evidence": [h["evidence"]],
            },
            "s/turn/initial",
            0,
        ),
    )
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert not rows(store, "yield-child-0", "turn_ended")


@pytest.mark.parametrize("action", ["finish", "cancel", "abort"])
def test_completion_after_parent_terminal_only_audits(store, database, action):
    start(store)
    rt, _ = waiting_runtime(store, database)
    if action == "finish":
        rt.config.no_tool_policy = "finish"
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    if action != "finish":
        push_fixture_input(store, "s", InboxItem(action, "control", {"action": action}, "s/turn/initial", 0))
        drive(store, "s", runtime=rt)
    # Finish or acknowledge the ordinary cancellation on the child.
    drive(store, "yield-child-0", runtime=fixture_runtime(store, database, "yield-child-0", [LLMResponse("child done")]))
    deliver(store, "yield-child-0")
    before = [r.record.encode() for r in rows(store, kind="op_completed")]
    drive(store, "s", runtime=rt)
    late = [r.record for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
    assert len(late) == 1
    assert late[0].payload["disposition"] == "noop"
    assert late[0].payload["reason"] == "late background completion: audit only"
    assert before == [r.record.encode() for r in rows(store, kind="op_completed")]


def yield_crash_worker(database, pipe, point, provider_wait):
    from .conftest import open_store

    with open_store(database) as raw_store:
        store = wrap_fixture_store(raw_store, fixture_key_provider())
        rt, _ = provider_runtime(store, database, YieldProvider()) if provider_wait else waiting_runtime(store, database)
        rt.llm.steps.pop(0)

        def hook(at, record):
            if at == point and record and record.kind == "op_completed" and "yielded_by" in record.payload:
                pipe.send("yield-cut")
                pipe.recv()

        rt.hook = hook
        drive(store, "s", runtime=rt)


def child_batch_yield_crash_worker(database, pipe, point):
    from .conftest import open_store

    with open_store(database) as raw_store:
        store = wrap_fixture_store(raw_store, fixture_key_provider())
        rt, _ = waiting_runtime(store, database, count=2)

        def hook(at, record):
            if at == point and record and record.kind == "op_completed" and "yielded_by" in record.payload:
                pipe.send("yield-cut")
                pipe.recv()

        rt.hook = hook
        drive(store, "s", runtime=rt)


@pytest.mark.persistent_store
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
@pytest.mark.parametrize("provider_wait", [False, True])
def test_process_kill_around_yield_commit_restores_both_or_neither(store, database, point, provider_wait):
    import multiprocessing

    from .test_recovery_matrix import force_expire

    start(store)
    rt, _ = provider_runtime(store, database, YieldProvider()) if provider_wait else waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    steer(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=yield_crash_worker, args=(database, child, point, provider_wait))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        assert parent.recv() == "yield-cut"
        process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("YIELD_KILL", {"provider": provider_wait, "point": point, "exit": process.exitcode})
        state = read_state(store, "s")[0]
        assert ("steer" in state.applied_inputs) == (point == "after_commit")
        assert bool([r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]) == (
            point == "after_commit"
        )
        assert any(i.item.input_id == "steer" for i in store.peek_inbox("s")) == (point == "before_commit")
        force_expire(database)
        callbacks = []

        def render(plan, handles, reason):
            from vv_agent.types import ToolExecutionResult

            callbacks.append(reason)
            return ToolExecutionResult(plan.payload["request"]["id"], "Restored observation")

        if provider_wait:
            provider = YieldProvider(declared=False)
            provider.on_yield = render
            resumed, observed = provider_runtime(store, database, provider)
        else:
            resumed, observed = waiting_runtime(store, database, declared=False, callback=render)
        resumed.llm.steps.pop(0)
        drive(store, "s", runtime=resumed)
        assert len(observed) == 1
        assert callbacks == (["steer"] if point == "before_commit" else [])
        receipts = [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
        assert len(receipts) == 1
        applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
        assert applied.commit_id == receipts[0].commit_id
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        parent.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_process_kill_around_child_steer_commit_replays_once(store, database, point):
    import multiprocessing

    from .test_recovery_matrix import force_expire

    start(store)
    rt, _ = waiting_runtime(store, database, count=2)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    steer(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=child_batch_yield_crash_worker, args=(database, child, point))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        assert parent.recv() == "yield-cut"
        process.kill()
        process.join(10)
        assert process.exitcode == -9
        print("CHILD_YIELD_KILL", {"point": point, "exit": process.exitcode})
        state = read_state(store, "s")[0]
        assert ("steer" in state.applied_inputs) == (point == "after_commit")
        assert any(i.item.input_id == "steer" for i in store.peek_inbox("s")) == (point == "before_commit")
        force_expire(database)
        callbacks = []

        def render(plan, handles, reason):
            callbacks.append(reason)
            return ToolExecutionResult(plan.payload["request"]["id"], "Restored child observation")

        resumed, observed = waiting_runtime(store, database, count=2, declared=False, callback=render)
        resumed.llm.steps.pop(0)
        drive(store, "s", runtime=resumed)
        assert len(observed) == 1
        assert callbacks == (["steer"] if point == "before_commit" else [])
        child_applied = next(
            r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"
        )
        steer_applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
        yielded = [r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "steer"]
        assert len(yielded) == 1
        assert {child_applied.commit_id, steer_applied.commit_id, yielded[0].commit_id} == {steer_applied.commit_id}
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        parent.close()


@pytest.mark.parametrize("generation", [99])
def test_stale_steer_cannot_yield(store, database, generation):
    start(store)
    rt, observed = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    steer(store, generation=generation)
    drive(store, "s", runtime=rt)
    assert not observed
    assert not any("yielded_by" in r.record.payload for r in rows(store))
    stale = next(r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert stale.record.payload["disposition"] == "rejected"
    assert stale.record.payload["reason"] == "stale turn or generation"
    assert not store.is_runnable("s")


def test_untargeted_stale_generation_is_rejected_not_held(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("stale", "steer", {"content": "old generation"}, generation=99))
    drive(store, "s", runtime=rt)
    assert not observed
    stale = next(r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "stale")
    assert stale.record.payload["disposition"] == "rejected"
    assert stale.record.payload["reason"] == "stale turn or generation"
    assert not store.is_runnable("s")


def test_targeted_steer_after_turn_end_is_rejected_not_held(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    tid = read_state(store, "s")[0].active_turn_id
    assert tid is not None
    push_fixture_input(store, "s", InboxItem("cancel", "control", {"action": "cancel"}, tid, 0))
    drive(store, "s", runtime=rt)
    assert read_state(store, "s")[0].active_turn_id is None

    push_fixture_input(store, "s", InboxItem("late-target", "steer", {"content": "late"}, tid, 0))
    drive(store, "s", runtime=rt)
    assert not observed
    stale = next(r for r in hydrated_rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "late-target")
    assert stale.record.payload["disposition"] == "rejected"
    assert stale.record.payload["reason"] == "stale turn or generation"
    assert not store.is_runnable("s")


def test_suspended_turn_does_not_yield_until_resumed(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("suspend", "control", {"action": "suspend"}, "s/turn/initial", 0))
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    push_fixture_input(store, "s", InboxItem("resume", "control", {"action": "resume"}, "s/turn/initial", 0))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1


def test_multiple_observations_share_one_steering_commit(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    rt.llm.steps[0] = LLMResponse("", [ToolCall(f"call-{i}", "child_task", {}) for i in range(2)])
    rt.children["child_task"] = lambda plan: children.ChildSession(
        fixture_child_spec(store, "s", plan.payload["request"]["id"]),
        "work",
        yield_on_steer=True,
    )
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipts = [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert len(receipts) == 2
    assert {r.commit_id for r in [*receipts, applied]} == {applied.commit_id}
    results = [r.record.payload["result"] for r in hydrated_rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
    assert [result["tool_call_id"] for result in results] == ["call-0", "call-1"]


@pytest.mark.parametrize("invalid", ["call_id", "directive", "output_type"])
def test_invalid_renderer_falls_back_without_stalling_steer(store, database, invalid):
    from vv_agent.types import ToolDirective, ToolExecutionResult

    def render(plan, handles, reason):
        if invalid == "output_type":
            return "not a tool result"
        return ToolExecutionResult(
            "wrong" if invalid == "call_id" else plan.payload["request"]["id"],
            "bad renderer",
            directive=ToolDirective.FINISH if invalid == "directive" else ToolDirective.CONTINUE,
        )

    start(store)
    rt, observed = waiting_runtime(store, database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    result = next(
        r.record.payload["result"] for r in hydrated_rows(store, kind="op_completed") if "yielded_by" in r.record.payload
    )
    assert result["metadata"]["yield_callback_failed"] is True
    assert any("Wait interrupted by new input" in str(m.content) for m in observed[0].messages)


def test_implicit_active_turn_steer_uses_the_same_atomic_rule(store, database):
    start(store)
    rt, observed = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("implicit", "steer", {"content": "Implicit active turn input"}))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipt = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "implicit")
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "implicit")
    assert applied.commit_id == receipt.commit_id
    assert any("Implicit active turn input" in str(m.content) for m in observed[0].messages)


@pytest.mark.parametrize("invalid", ["missing_steer", "missing_consume", "different_turn"])
def test_store_rejects_yield_completion_without_matching_steer_and_consume(store, database, monkeypatch, invalid):
    from vv_agent.session.sql import SQLSessionTx
    from vv_agent.session.store import Conflict

    start(store)
    rt, _ = waiting_runtime(store, database)
    drive(store, "s", runtime=rt)
    steer(store)

    def snapshot():
        return (
            tuple(
                (seq, bytes(body))
                for seq, body in store._rows("SELECT seq,body FROM sk_record WHERE session_id=%s ORDER BY seq", ("s",))
            ),
            tuple(
                (seq, bytes(body), consumed, available)
                for seq, body, consumed, available in store._rows(
                    "SELECT input_seq,body,consumed_seq,available_ms FROM sk_inbox WHERE session_id=%s ORDER BY input_seq", ("s",)
                )
            ),
        )

    before = snapshot()
    candidate = {}
    append = SQLSessionTx.append

    class PreparedYield(Exception):
        pass

    def capture(self, *args, **kwargs):
        if any(r.kind == "op_completed" and "yielded_by" in r.payload for r in kwargs["records"]):
            candidate.update(kwargs)
            raise PreparedYield
        return append(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(SQLSessionTx, "append", capture)
        with pytest.raises(PreparedYield):
            drive(store, "s", runtime=rt)
    assert snapshot() == before
    assert not any("yielded_by" in r.record.payload for r in rows(store))
    records = candidate["records"]
    consume = candidate["consume_input_ids"]
    assert consume == ("steer",)
    if invalid == "missing_steer":
        records = tuple(r for r in records if r.kind != "input_applied")
    elif invalid == "missing_consume":
        consume = ()
    else:
        records = tuple(replace(r, turn_id="other/turn") if r.kind == "input_applied" else r for r in records)
    lease = store.acquire("s", owner="split-commit", ttl_ms=15000)
    assert lease is not None
    try:
        with pytest.raises(Conflict, match="yield completion and steering must commit together"), store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=store.read("s").head_seq,
                commit_id="split",
                records=records,
                consume_input_ids=consume,
            )
    finally:
        store.release(lease)
    assert snapshot() == before


def test_yield_producer_matches_local_contract():
    import json
    import runpy
    from pathlib import Path

    from support.contract import fixture_root

    producer = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/yieldable_wait_fixtures.py"))
    assert producer["produce"]() == json.loads((fixture_root() / "yieldable_waits.json").read_text())
