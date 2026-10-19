"""Session-scoped selections use retained plans across connections and recovery."""

from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.events import ModelCallCompletedEvent
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.compaction import manager_for
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem, RecordError
from vv_agent.session.result import project_result
from vv_agent.session.store import Conflict
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .test_provider_usage import ReceiptProvider
from .test_recovery_matrix import MODEL_USAGE, runtime, start

SELECTION: dict[str, Any] = {
    "model": "B",
    "model_settings": {"reasoning": {"effort": "high"}, "max_tokens": 1024},
    "route": {"backend_type": "anthropic"},
}


@function_tool
def echo() -> str:
    return "ok"


def switch(store, selection=SELECTION, input_id="select", sid="s"):
    item = InboxItem(input_id, "model_selection", {"selection": selection})
    with store.atomic() as tx:
        return tx.push(sid, item)


def routed(database, steps_a, steps_b=(), **kwargs):
    rt = runtime(database, steps_a, [echo], **kwargs)
    b = ScriptedLLM(list(steps_b))

    def resolver(request):
        if request["model"] == "scripted" and request.get("route", {}) == {}:
            return rt.llm, rt.resolved
        if request["model"] != "B" or request["route"] != {"backend_type": "anthropic"}:
            raise ValueError("unknown model or route")
        return b, replace(
            rt.resolved,
            backend="anthropic",
            model_id="B",
            requested_model="B",
            selected_model="B",
            context_length=8192,
            max_output_tokens=2048,
        )

    rt.model_resolver = resolver
    return rt, b


def plans(store):
    return [
        r.record for r in store.read_state("s")[1] if r.record.kind == "op_planned" and r.record.payload["purpose"] == "primary"
    ]


@pytest.mark.parametrize("output_cap,request_max,expected_reserve", [(2048, 1024, 1024), (0, None, 0)])
def test_inflight_switch_cross_backend_usage_and_capacity(store, database, output_cap, request_max, expected_reserve):
    start(store)
    entered, release = Event(), Event()
    errors = []

    def first(request):
        assert request.model == "scripted"
        entered.set()
        assert release.wait(10)
        return LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})

    b_usage = {"prompt_tokens": 200, "completion_tokens": 50 if output_cap else 0, "prompt_tokens_details": {"cached_tokens": 0}}
    rt, b = routed(database, [first], [LLMResponse("done", raw={"usage": b_usage})])
    resolve = rt.model_resolver
    assert resolve is not None

    def selected_limits(request):
        client, config = resolve(request)
        return client, replace(config, max_output_tokens=output_cap) if request["model"] == "B" else config

    rt.model_resolver = selected_limits
    choice = SELECTION | {"model_settings": dict(SELECTION["model_settings"])}
    if request_max is None:
        choice["model_settings"].pop("max_tokens")

    def run():
        try:
            drive(store, "s", runtime=rt)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    try:
        assert entered.wait(10)
        with open_store(database) as other:
            switch(other, choice)
    finally:
        release.set()
        thread.join(15)
    assert not thread.is_alive() and not errors
    assert not b.steps
    a_plan, b_plan = plans(store)
    assert a_plan.payload["request"]["model"] == "scripted"
    request = b_plan.payload["request"]
    assert request["model"] == "B" and request["route"] == SELECTION["route"]
    assert request["model_settings"]["reasoning"] == {"effort": "high"}
    state, records, _ = store.read_state("s")
    events = [e for e in project_records(records) if isinstance(e, ModelCallCompletedEvent)]
    assert [(e.model, e.backend) for e in events] == [("scripted", "scripted"), ("B", "anthropic")]
    result = project_result(store, "s", a_plan.turn_id, runtime=rt)
    assert [(c.model, c.backend) for c in result.raw_result.token_usage.model_calls] == [
        ("scripted", "scripted"),
        ("B", "anthropic"),
    ]
    assert len({e.call_id for e in events}) == 2
    assert [e.usage.total_tokens for e in events] == [12300, 200 + b_usage["completion_tokens"]]
    # Capacity uses persisted selection even on a fresh worker, with no new resolution.
    state.active_turn_id = a_plan.turn_id
    driver = cast(Any, SimpleNamespace(state=state, runtime=rt, sid="s"))
    driver.effective_model = lambda: _Driver.effective_model(driver)
    manager = manager_for(driver)
    assert (manager.model, manager.model_context_window, manager.reserved_output_tokens) == ("B", 8192, expected_reserve)


def test_persistent_latest_wins_clear_reject_and_idempotency(store, database):
    start(store)
    rt, b = routed(database, [LLMResponse("A")], [LLMResponse("B1"), LLMResponse("B2")])
    switch(store, {"model": "bad", "route": {}}, "bad-first")
    switch(store)
    assert switch(store).replayed
    with pytest.raises(Conflict):
        switch(store, None)
    drive(store, "s", runtime=rt)
    assert plans(store)[0].payload["request"]["model"] == "B"
    # A later unresolvable selection cannot replace the accepted B selection.
    switch(store, {"model": "B", "route": {"backend_type": "denied"}}, "bad-route")
    with store.atomic() as tx:
        tx.push("s", InboxItem("turn2", "user", {"content": "next"}))
    drive(store, "s", runtime=rt)
    assert plans(store)[1].payload["request"]["model"] == "B"
    switch(store, SELECTION, "again")
    switch(store, None, "clear")
    with store.atomic() as tx:
        tx.push("s", InboxItem("turn3", "user", {"content": "next"}))
    drive(store, "s", runtime=rt)
    assert plans(store)[2].payload["request"]["model"] == "scripted"
    assert not b.steps
    state, _, _ = store.read_state("s")
    assert state.model_selection is None
    assert [state.applied_inputs[k].payload["disposition"] for k in ("bad-first", "bad-route")] == ["rejected", "rejected"]


class WorkerDeath(BaseException):
    pass


def test_selected_endpoints_and_retries_survive_runtime_reconstruction(store, database, monkeypatch):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
    from vv_agent.output_validation import OutputValidationResult

    start(store)
    calls = []
    repairs = []

    def repair(request):
        repairs.append(request.invalid_output)
        return "valid"

    def complete(client, request):
        assert len(client.endpoint_targets) == client.max_retries_per_endpoint == 1
        assert request.model_settings.retry.max_attempts == 1
        assert client.backend == ("anthropic" if request.model == "B" else "openai")
        endpoint = client.endpoint_targets[0].endpoint_id
        calls.append((request.model, endpoint))
        if endpoint == "a" and calls.count(("scripted", "a")) == 1:
            with open_store(database) as writer:
                switch(writer)
            return LLMResponse("", [ToolCall("a-tool", "echo", {})], raw={"usage": MODEL_USAGE})
        if endpoint == "b1":
            raise TimeoutError("receipt lost")
        if endpoint == "b2":
            return LLMResponse("", [ToolCall("b-tool", "echo", {})], raw={"usage": MODEL_USAGE})
        return LLMResponse("done", raw={"usage": MODEL_USAGE})

    monkeypatch.setattr(VvLlmClient, "complete", complete)

    def build(**kwargs):
        rt = runtime(database, [], [echo], **kwargs)
        rt.agent.output_validation_enabled = True
        rt.agent.output_validator = lambda value, ctx: (
            OutputValidationResult.accept() if value == "valid" else OutputValidationResult.reject("invalid", "bad answer")
        )
        rt.agent.output_repair = repair
        a = VvLlmClient([EndpointTarget("a", "unused", "https://example.invalid")], backend="openai", randomize_endpoints=False)
        b = VvLlmClient(
            [EndpointTarget(e, "unused", "https://example.invalid") for e in ("b1", "b2")],
            backend="anthropic",
            randomize_endpoints=False,
        )
        rt.llm = a
        rt.resolved = replace(rt.resolved, backend="openai")

        def resolve(request):
            if request["model"] == "scripted" and request.get("route", {}) == {}:
                return a, rt.resolved
            assert request["model"] == "B" and request["route"] == SELECTION["route"]
            return b, replace(
                rt.resolved,
                backend="anthropic",
                model_id="B",
                requested_model="B",
                selected_model="B",
                context_length=8192,
                max_output_tokens=2048,
            )

        rt.model_resolver = resolve
        return rt

    def die(point, record):
        if point == "after_commit" and record.kind == "op_unknown":
            raise WorkerDeath

    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=build(hook=die))
    with open_store(database) as writer:
        switch(writer, None, "clear-after-b-plan")
    drive(store, "s", runtime=build())
    assert calls == [("scripted", "a"), ("B", "b1"), ("B", "b2"), ("scripted", "a")]
    a, b1, b2, cleared = plans(store)
    assert b1.payload["request"] == b2.payload["request"]
    assert b1.operation_id == b2.operation_id and (b1.attempt, b2.attempt) == (1, 2)
    assert b1.payload["request"]["metadata"]["vv_session"]["endpoint_order"] == ["b1", "b2"]
    assert a.payload["request"]["model"] == cleared.payload["request"]["model"] == "scripted"
    result = project_result(store, "s", a.turn_id)
    assert result.final_output == "valid" and repairs == ["done"]
    assert [(c.model, c.backend) for c in result.token_usage.model_calls] == [
        ("scripted", "openai"),
        ("B", "anthropic"),
        ("B", "anthropic"),
        ("scripted", "openai"),
        ("scripted", "openai"),
    ]


@pytest.mark.parametrize("cut", ["after_commit", "before_external_call", "after_external_call"])
def test_recovery_keeps_planned_model_and_late_usage(store, database, cut):
    start(store)
    retained = []
    provider = ReceiptProvider({"content": "", "tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}, MODEL_USAGE)

    def die(point, plan):
        if point == cut and plan is not None and plan.kind == "op_planned" and plan.payload.get("purpose") == "primary":
            retained.append(plan)
            if cut == "after_external_call":
                provider.bind(plan)
            raise WorkerDeath

    # The commit hook receives the record at after_commit; before/after call receives plan.
    rt, _ = routed(database, [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})], hook=die)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=rt)
    original = plans(store)[0]
    switch(store)
    fresh, b = routed(
        database,
        [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
        [LLMResponse("B", raw={"usage": MODEL_USAGE})],
        providers={"model": provider},
    )
    if cut == "after_external_call":
        with store.atomic() as tx:
            tx.push("s", provider.item)
    drive(store, "s", runtime=fresh)
    assert plans(store)[0].encode() == original.encode()
    assert all(p.payload["request"]["model"] == "scripted" for p in plans(store) if p.operation_id == original.operation_id)
    assert plans(store)[-1].payload["request"]["model"] == "B"
    assert not b.steps
    if cut == "after_external_call":
        assert len(fresh.llm.steps) == 1  # authentic receipt, no duplicate call
    events = [e for e in project_records(store.read_state("s")[1]) if isinstance(e, ModelCallCompletedEvent)]
    assert [(e.model, e.usage.total_tokens) for e in events] == [("scripted", 12300), ("B", 12300)]


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"model": "", "route": {}},
        {"model": "B"},
        {"model": "B", "route": []},
        {"model": "B", "route": {}, "extra": 1},
        {"model": "B", "route": {}, "model_settings": {"bogus": 1}},
        {"model": "B", "route": {}, "model_settings": {"max_tokens": False}},
        {"model": "B", "route": {}, "model_settings": {"reasoning": "high"}},
        *[{"model": "B", "route": {}, "model_settings": {key: None}} for key in ("extra_headers", "extra_body", "extra_args")],
    ],
)
def test_selection_codec_rejects_invalid(selection):
    with pytest.raises(RecordError):
        InboxItem("s", "model_selection", {"selection": selection}).encode()


def test_selection_codec_roundtrip_and_scope():
    item = InboxItem("s", "model_selection", {"selection": SELECTION})
    assert InboxItem.parse(item.encode()) == item
    for changed in (replace(item, target_turn_id="t"), replace(item, generation=0), replace(item, schema_version=2)):
        with pytest.raises(RecordError):
            changed.encode()


def selection_worker(database, pipe, cut):
    def stop(point, plan):
        if point == cut and plan is not None and plan.kind == "op_planned" and plan.payload.get("purpose") == "primary":
            pipe.send(("barrier", plan.to_dict()))
            pipe.recv()

    rt, _ = routed(database, [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})], hook=stop)
    with open_store(database) as worker_store:
        drive(worker_store, "s", runtime=rt)


@pytest.mark.persistent_store
@pytest.mark.parametrize("cut", ["after_commit", "before_external_call", "after_external_call"])
def test_process_kill_recovery_uses_original_plan(store, database, cut):
    import multiprocessing

    from .test_recovery_matrix import kill, receive

    start(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=selection_worker, args=(database, child, cut))
    proc.start()
    child.close()
    try:
        original = receive(parent, "barrier")
        # Different connection, no lease: this must survive the holder's death.
        with open_store(database) as writer:
            switch(writer)
        kill(proc, parent, database)
        rt, b = routed(
            database,
            [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
            [LLMResponse("B", raw={"usage": MODEL_USAGE})],
        )
        drive(store, "s", runtime=rt)
        retained = plans(store)
        assert retained[0].to_dict() == original
        assert retained[0].payload["request"]["model"] == "scripted"
        assert retained[-1].payload["request"]["model"] == "B"
        assert not b.steps
        if cut != "after_commit":
            assert retained[1].payload["request"] == retained[0].payload["request"]
            assert retained[1].attempt == 2
    finally:
        if proc.is_alive():
            kill(proc, parent, database)


def test_selection_is_local_to_one_session_and_default_rejects(store, database):
    start(store)
    start(store, "child")
    switch(store)
    drive(store, "s", runtime=runtime(database, [LLMResponse("A")]))
    assert store.read_state("s")[0].applied_inputs["select"].payload["disposition"] == "rejected"
    rt, _ = routed(database, [], [LLMResponse("B")])
    switch(store, input_id="valid")
    with store.atomic() as tx:
        tx.push("s", InboxItem("turn2", "user", {"content": "next"}))
    drive(store, "s", runtime=rt)
    child_rt, _ = routed(database, [LLMResponse("child A")])
    drive(store, "child", runtime=child_rt)
    assert store.read_state("child")[0].model_selection is None
    child_plans = [r.record for r in store.read_state("child")[1] if r.record.kind == "op_planned"]
    assert child_plans[0].payload["request"]["model"] == "scripted"


def test_prompt_too_long_with_selection_keeps_cycle_and_single_billing(store, database):
    from .test_compaction import SUMMARY, history

    start(store)
    requests = []

    def too_long(request):
        requests.append(request)
        raise RuntimeError("maximum context length exceeded")

    def answer(request):
        requests.append(request)
        return LLMResponse("B", raw={"usage": MODEL_USAGE})

    rt, _ = routed(database, [LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})], [too_long, answer])
    resolve = rt.model_resolver
    assert resolve is not None

    def large_context(request):
        client, resolved = resolve(request)
        return client, replace(resolved, context_length=279000) if request["model"] == "B" else resolved

    rt.model_resolver = large_context
    rt.config = replace(rt.config, initial_messages=history(), max_cycles=1)
    rt.memory_manager.compact_threshold = 999999
    switch(store)
    drive(store, "s", runtime=rt)
    result = project_result(store, "s", "s/turn/initial", runtime=rt)
    assert result.final_output == "B"
    assert [r.model for r in requests] == ["B", "B"]
    primary = [c for c in result.token_usage.model_calls if c.operation.value == "agent_cycle"]
    assert len(primary) == 2 and [c.cycle_index for c in primary] == [1, 1]
    assert len({c.call_id for c in primary}) == 2
    assert primary[-1].usage.total_tokens == 12300


def test_parent_selection_does_not_propagate_to_real_child(store, database):
    from vv_agent.session.children import ChildSession, child_delivery
    from vv_agent.session.records import SessionSpec

    from .test_children import child_task

    start(store)
    rt, b = routed(database, [], [LLMResponse("", [ToolCall("child-call", "child_task", {})]), LLMResponse("parent B")])
    rt.agent.tools = [child_task]
    rt.children = {"child_task": lambda plan: ChildSession(SessionSpec("child", "test", "/tmp"), "child work")}
    switch(store)
    drive(store, "s", runtime=rt)
    assert store.read_state("s")[0].phase == "parked"
    drive(store, "child", runtime=runtime(database, [LLMResponse("child A")]))
    with store.atomic() as tx:
        assert child_delivery(store, tx, "child")
    drive(store, "s", runtime=rt)
    child_state, rows, _ = store.read_state("child")
    assert child_state.model_selection is None
    assert next(r.record for r in rows if r.record.kind == "op_planned").payload["request"]["model"] == "scripted"
    assert all(p.payload["request"]["model"] == "B" for p in plans(store))
    assert not b.steps
