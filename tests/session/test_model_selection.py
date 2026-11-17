"""Session-scoped selections use retained plans across connections and recovery."""

from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent.events import ModelCallCompletedEvent
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import publish_session_created_content
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.model import ModelError, ModelRef, ScriptedModelProvider
from vv_agent.session.compaction import manager_for
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.kernel import _Driver, _Scope, drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem, RecordError, SessionSpec
from vv_agent.session.result import project_result
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .helpers import (
    content_ref,
    create_fixture_session,
    fixture_child_spec,
    fixture_content_reader,
    fixture_key_provider,
    principal,
    push_fixture_input,
    wrap_fixture_store,
)
from .test_provider_usage import ReceiptProvider
from .test_recovery_matrix import MODEL_USAGE, runtime

SELECTION: dict[str, Any] = {
    "model": "B",
    "model_settings": {"reasoning": {"effort": "high"}, "max_tokens": 1024},
    "route": {"backend_type": "anthropic"},
}


@function_tool
def echo() -> str:
    return "ok"


class ModelSelectionReceiptProvider(ReceiptProvider):
    def authenticate(self, item, plan):
        return (
            item.kind == "provider_result"
            and item.target_turn_id == self.item.target_turn_id
            and item.generation == self.item.generation
            and item.payload == self.item.payload
            and item.payload["operation_id"] == plan.operation_id
            and item.payload["attempt"] == plan.attempt
            and item.payload["request_digest"] == plan.payload["request_digest"]
        )


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def push_user(store, runtime, input_id, content, sid="s"):
    item = InboxItem(
        input_id,
        "user",
        {"content": content},
        feature_level=2,
        requirements=runtime.requirements_bound().to_dict(),
    )
    return push_fixture_input(store, sid, item)


def create_level_two(store, requirements, sid="s"):
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=sid)

        def session_spec(content):
            actor, _ = content.service_principal("fixture/host")
            created = publish_session_created_content(
                content,
                principal=actor,
                workspace="/tmp",
                attributes={"seed": {"messages": [], "shared_state": {}}},
            )
            return SessionSpec(
                sid,
                created.principal,
                created.workspace.to_dict(),
                attributes=created.attributes.to_dict(),
                feature_level=2,
                bootstrap=requirements,
            )

        writer.create_session(session_spec, consumers=("events",))


def start(store, runtime, sid="s"):
    requirements = runtime.requirements_bound()
    create_level_two(store, requirements, sid)
    push_user(store, runtime, "initial", "go", sid)


def start_level_one(store, sid="s"):
    store._rows("UPDATE sk_settings SET write_level=1 WHERE singleton=1")
    create_fixture_session(store, sid)
    push_fixture_input(store, sid, InboxItem("initial", "user", {"content": "go"}))


def _model_selection_requirements(runtime, selection):
    if selection is None:
        selection = {
            "model": runtime.resolved.requested_model or runtime.resolved.model_id,
            "route": {},
        }
    return runtime.model_selection_requirements(selection)


def _worker_capabilities(runtime, selections):
    capabilities = set(runtime.requirements_bound().capabilities)
    for selection in selections:
        capabilities.update(_model_selection_requirements(runtime, selection).capabilities)
    return tuple(sorted(capabilities))


def _configure_worker_capabilities(store, runtime, capabilities):
    store.configure_worker(
        level=2,
        handler_versions=(runtime.handler_version,),
        capabilities=capabilities,
    )


def configure_worker(store, runtime, selections):
    capabilities = _worker_capabilities(runtime, selections)
    if store.worker_supports.accepts(
        level=2,
        handler_version=runtime.handler_version,
        capabilities=capabilities,
    ):
        return
    _configure_worker_capabilities(store, runtime, capabilities)


def configure_runtime(store, database, runtime, selections):
    capabilities = _worker_capabilities(runtime, selections)
    _configure_worker_capabilities(store, runtime, capabilities)

    @contextmanager
    def heartbeat_store():
        with open_store(database) as heartbeat:
            _configure_worker_capabilities(heartbeat, runtime, capabilities)
            yield heartbeat

    runtime.heartbeat_store = heartbeat_store


def switch(store, runtime, selection=SELECTION, input_id="select", sid="s"):
    requirements = _model_selection_requirements(runtime, selection)
    configure_worker(store, runtime, (selection,))
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    item = InboxItem(
        input_id,
        "model_selection",
        {"selection": selection},
        feature_level=2,
        requirements=requirements.to_dict(),
    )
    return push_fixture_input(store, sid, item)


def routed(database, steps_a, steps_b=(), *, store, sid="s", tools=(echo,), worker_selections=(SELECTION,), **kwargs):
    kwargs.setdefault("content_reader", fixture_content_reader(store, sid))
    rt = runtime(database, steps_a, list(tools), **kwargs)
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
    configure_runtime(store, database, rt, worker_selections)
    return rt, b


def plans(store, sid="s"):
    hydration = hydration_for_session_store(store, sid)
    return [
        SimpleNamespace(
            record=row.record,
            payload=hydrate_record_payload(row, hydration),
            operation_id=row.record.operation_id,
            turn_id=row.record.turn_id,
            attempt=row.record.attempt,
            encode=row.record.encode,
            to_dict=row.record.to_dict,
        )
        for row in store.read_state(sid)[1]
        if row.record.kind == "op_planned" and row.record.payload["purpose"] == "primary"
    ]


def routing_requirements(store, sid="s"):
    row = store._one(
        "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms,level,"
        "deletion_state,handler_version,required_capabilities,active_turn_id "
        "FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, sid),
    )
    return store._routing_cache_from_row(row).requirements


@pytest.mark.capability("model-selection")
@pytest.mark.parametrize("output_cap,request_max,expected_reserve", [(2048, 1024, 1024), (None, None, 16_000)])
def test_inflight_switch_cross_backend_usage_and_capacity(store, database, output_cap, request_max, expected_reserve):
    entered, release = Event(), Event()
    errors = []

    def first(request):
        assert request.model == "scripted"
        entered.set()
        assert release.wait(10)
        return LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})

    b_usage = {
        "prompt_tokens": 200,
        "completion_tokens": 50 if output_cap is not None else 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    rt, b = routed(database, [first], [LLMResponse("done", raw={"usage": b_usage})], store=store)
    start(store, rt)
    resolve = rt.model_resolver
    assert resolve is not None

    def selected_limits(request):
        client, config = resolve(request)
        return client, replace(config, max_output_tokens=output_cap) if request["model"] == "B" else config

    rt.model_resolver = selected_limits
    choice = SELECTION | {"model_settings": dict(SELECTION["model_settings"])}
    if request_max is None:
        choice["model_settings"].pop("max_tokens")
    configure_runtime(store, database, rt, (choice,))

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
            switch(wrap_fixture_store(other), rt, choice)
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
    _state, records, _ = store.read_state("s")
    events = [
        e
        for e in project_records(records, hydration=hydration_for_session_store(store, "s"))
        if isinstance(e, ModelCallCompletedEvent)
    ]
    assert [(e.model, e.backend) for e in events] == [("scripted", "scripted"), ("B", "anthropic")]
    result = project_result(store, "s", a_plan.turn_id, runtime=rt)
    assert [(c.model, c.backend) for c in result.raw_result.token_usage.model_calls] == [
        ("scripted", "scripted"),
        ("B", "anthropic"),
    ]
    assert len({e.call_id for e in events}) == 2
    assert [e.usage.total_tokens for e in events] == [12300, 200 + b_usage["completion_tokens"]]
    # Capacity uses persisted selection even on a fresh worker, with no new resolution.
    lease = store.acquire("s", owner="model-selection-capacity", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(store, "s", rt, _Scope(lease, rt, now_ms=store._now()))
    driver.state.active_turn_id = a_plan.turn_id
    manager = manager_for(driver)
    assert (manager.model, manager.model_context_window, manager.reserved_output_tokens) == ("B", 8192, expected_reserve)


@pytest.mark.capability("model-selection")
def test_persistent_latest_wins_clear_reject_and_idempotency(store, database):
    rt, b = routed(database, [LLMResponse("A")], [LLMResponse("B1"), LLMResponse("B2")], store=store)
    start(store, rt)
    _state, rows_before, watermark_before = store.read_state("s")
    inbox_before = len(store.peek_inbox("s"))
    with pytest.raises(ValueError, match="unknown model or route"):
        switch(store, rt, {"model": "bad", "route": {}}, "bad-first")
    _state, rows_after, watermark_after = store.read_state("s")
    assert (watermark_after, len(rows_after), len(store.peek_inbox("s"))) == (
        watermark_before,
        len(rows_before),
        inbox_before,
    )
    assert store.read_input("s", "bad-first") is None

    switch(store, rt)
    assert switch(store, rt).replayed
    with pytest.raises(Conflict):
        switch(store, rt, None)
    drive(store, "s", runtime=rt)
    assert plans(store)[0].payload["request"]["model"] == "B"

    state_before_route, rows_before, watermark_before = store.read_state("s")
    inbox_before = len(store.peek_inbox("s"))
    accepted = state_before_route.model_selection
    assert accepted is not None
    with pytest.raises(ValueError, match="unknown model or route"):
        switch(store, rt, {"model": "B", "route": {"backend_type": "denied"}}, "bad-route")
    state_after_route, rows_after, watermark_after = store.read_state("s")
    assert (watermark_after, len(rows_after), len(store.peek_inbox("s"))) == (
        watermark_before,
        len(rows_before),
        inbox_before,
    )
    assert store.read_input("s", "bad-route") is None
    assert state_after_route.model_selection is not None
    assert state_after_route.model_selection.encode() == accepted.encode()

    push_user(store, rt, "turn2", "next")
    drive(store, "s", runtime=rt)
    assert plans(store)[1].payload["request"]["model"] == "B"
    switch(store, rt, SELECTION, "again")
    switch(store, rt, None, "clear")
    push_user(store, rt, "turn3", "next")
    drive(store, "s", runtime=rt)
    assert plans(store)[2].payload["request"]["model"] == "scripted"
    assert not b.steps
    state, _, _ = store.read_state("s")
    assert state.model_selection is None
    assert "bad-first" not in state.applied_inputs
    assert "bad-route" not in state.applied_inputs


class WorkerDeath(BaseException):
    pass


@pytest.mark.capability("model-selection")
def test_selected_endpoints_and_retries_survive_runtime_reconstruction(store, database, monkeypatch):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
    from vv_agent.output_validation import OutputValidationResult

    calls = []
    repairs = []
    current_runtime = None

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
                assert current_runtime is not None
                switch(wrap_fixture_store(writer), current_runtime)
            return LLMResponse("", [ToolCall("a-tool", "echo", {})], raw={"usage": MODEL_USAGE})
        if endpoint == "b1":
            raise TimeoutError("receipt lost")
        if endpoint == "b2":
            return LLMResponse("", [ToolCall("b-tool", "echo", {})], raw={"usage": MODEL_USAGE})
        return LLMResponse("done", raw={"usage": MODEL_USAGE})

    monkeypatch.setattr(VvLlmClient, "complete", complete)

    def build(**kwargs):
        nonlocal current_runtime
        rt = runtime(database, [], [echo], content_reader=fixture_content_reader(store, "s"), **kwargs)
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
        configure_runtime(store, database, rt, (SELECTION,))
        current_runtime = rt
        return rt

    def die(point, record):
        if point == "after_commit" and record.kind == "op_unknown":
            raise WorkerDeath

    initial_runtime = build()
    initial_runtime.hook = die
    start(store, initial_runtime)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=initial_runtime)
    recovery_runtime = build()
    with open_store(database) as writer:
        switch(wrap_fixture_store(writer), recovery_runtime, None, "clear-after-b-plan")
    drive(store, "s", runtime=recovery_runtime)
    assert calls == [("scripted", "a"), ("B", "b1"), ("B", "b2"), ("scripted", "a")]
    a, b1, b2, cleared = plans(store)
    assert b1.payload["request"] == b2.payload["request"]
    assert b1.operation_id == b2.operation_id and (b1.attempt, b2.attempt) == (1, 2)
    b1_request_ref = ContentRef.from_dict(b1.record.payload["request"])
    b2_request_ref = ContentRef.from_dict(b2.record.payload["request"])
    assert b2_request_ref.content_id == b1_request_ref.content_id
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


@pytest.mark.capability("model-selection")
@pytest.mark.parametrize("cut", ["after_commit", "before_external_call", "after_external_call"])
def test_recovery_keeps_planned_model_and_late_usage(store, database, cut):
    retained = []
    provider = ModelSelectionReceiptProvider({"content": "", "tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}, MODEL_USAGE)

    def die(point, plan):
        if point == cut and plan is not None and plan.kind == "op_planned" and plan.payload.get("purpose") == "primary":
            retained.append(plan)
            if cut == "after_external_call":
                provider.bind(plan)
            raise WorkerDeath

    # The commit hook receives the record at after_commit; before/after call receives plan.
    rt, _ = routed(
        database,
        [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
        store=store,
        hook=die,
    )
    start(store, rt)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=rt)
    original = plans(store)[0]
    switch(store, rt)
    fresh, b = routed(
        database,
        [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
        [LLMResponse("B", raw={"usage": MODEL_USAGE})],
        store=store,
        providers={"model": provider},
    )
    if cut == "after_external_call":
        push_fixture_input(store, "s", provider.item)
    drive(store, "s", runtime=fresh)
    assert plans(store)[0].encode() == original.encode()
    assert all(p.payload["request"]["model"] == "scripted" for p in plans(store) if p.operation_id == original.operation_id)
    assert plans(store)[-1].payload["request"]["model"] == "B"
    assert not b.steps
    if cut == "after_external_call":
        assert len(fresh.llm.steps) == 1  # authentic receipt, no duplicate call
    events = [
        e
        for e in project_records(store.read_state("s")[1], hydration=hydration_for_session_store(store, "s"))
        if isinstance(e, ModelCallCompletedEvent)
    ]
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
    item = InboxItem(
        "s",
        "model_selection",
        {"selection": content_ref("selection")},
        actor=principal("fixture/model-selection"),
    )
    assert InboxItem.parse(item.encode()) == item
    for changed in (replace(item, target_turn_id="t"), replace(item, generation=0), replace(item, schema_version=2)):
        with pytest.raises(RecordError):
            changed.encode()


def selection_worker(database, pipe, cut):
    def stop(point, plan):
        if point == cut and plan is not None and plan.kind == "op_planned" and plan.payload.get("purpose") == "primary":
            pipe.send(("barrier", plan.to_dict()))
            pipe.recv()

    with open_store(database) as worker_store:
        worker_store = wrap_fixture_store(worker_store)
        rt, _ = routed(
            database,
            [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
            store=worker_store,
            hook=stop,
        )
        drive(worker_store, "s", runtime=rt)


@pytest.mark.capability("model-selection")
@pytest.mark.persistent_store
@pytest.mark.parametrize("cut", ["after_commit", "before_external_call", "after_external_call"])
def test_process_kill_recovery_uses_original_plan(store, database, cut):
    import multiprocessing

    from .test_recovery_matrix import kill, receive

    rt, b = routed(
        database,
        [LLMResponse("", [ToolCall("echo", "echo", {})], raw={"usage": MODEL_USAGE})],
        [LLMResponse("B", raw={"usage": MODEL_USAGE})],
        store=store,
    )
    start(store, rt)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=selection_worker, args=(database, child, cut))
    proc.start()
    child.close()
    try:
        original = receive(parent, "barrier")
        # Different connection, no lease: this must survive the holder's death.
        with open_store(database) as writer:
            switch(wrap_fixture_store(writer), rt)
        kill(proc, parent, database)
        drive(store, "s", runtime=rt)
        retained = plans(store)
        if cut == "after_commit":
            assert retained[0].record.to_dict() == original
        else:
            assert retained[0].record.record_id == original["record_id"]
            assert retained[0].payload == original["payload"]
        assert retained[0].payload["request"]["model"] == "scripted"
        assert retained[-1].payload["request"]["model"] == "B"
        assert not b.steps
        if cut != "after_commit":
            assert retained[1].payload["request"] == retained[0].payload["request"]
            assert retained[1].attempt == 2
    finally:
        if proc.is_alive():
            kill(proc, parent, database)


@pytest.mark.capability("model-selection")
def test_selection_is_local_to_one_session_and_default_rejects(store, database):
    producer_rt, _ = routed(database, [LLMResponse("A")], store=store)
    start(store, producer_rt)
    start(store, producer_rt, "child")
    switch(store, producer_rt)
    resolver = producer_rt.model_resolver
    producer_rt.model_resolver = None
    drive(store, "s", runtime=producer_rt)
    producer_rt.model_resolver = resolver
    assert store.read_state("s")[0].applied_inputs["select"].payload["disposition"] == "rejected"
    rt, _ = routed(database, [], [LLMResponse("B")], store=store)
    switch(store, rt, input_id="valid")
    push_user(store, rt, "turn2", "next")
    drive(store, "s", runtime=rt)
    child_rt, _ = routed(database, [LLMResponse("child A")], store=store, sid="child")
    drive(store, "child", runtime=child_rt)
    assert store.read_state("child")[0].model_selection is None
    assert plans(store, "child")[0].payload["request"]["model"] == "scripted"


def test_prompt_too_long_with_selection_keeps_cycle_and_single_billing(store, database):
    from .test_compaction import SUMMARY, history

    requests = []

    def too_long(request):
        requests.append(request)
        raise RuntimeError("maximum context length exceeded")

    def answer(request):
        requests.append(request)
        return LLMResponse("B", raw={"usage": MODEL_USAGE})

    rt, _ = routed(database, [LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})], [too_long, answer], store=store)
    start(store, rt)
    resolve = rt.model_resolver
    assert resolve is not None

    def large_context(request):
        client, resolved = resolve(request)
        return client, replace(resolved, context_length=279000) if request["model"] == "B" else resolved

    rt.model_resolver = large_context
    rt.config = replace(rt.config, initial_messages=history(), max_cycles=1)
    rt.memory_manager.compact_threshold = 999999
    configure_runtime(store, database, rt, (SELECTION,))
    switch(store, rt)
    drive(store, "s", runtime=rt)
    result = project_result(store, "s", "s/turn/initial", runtime=rt)
    assert result.final_output == "B"
    assert [r.model for r in requests] == ["B", "B"]
    primary = [c for c in result.token_usage.model_calls if c.operation.value == "agent_cycle"]
    assert len(primary) == 2 and [c.cycle_index for c in primary] == [1, 1]
    assert len({c.call_id for c in primary}) == 2
    assert primary[-1].usage.total_tokens == 12300


@pytest.mark.capability("model-selection")
def test_parent_selection_does_not_propagate_to_real_child(store, database):
    from vv_agent.session.children import ChildSession, child_delivery

    from .test_children import child_task

    rt, b = routed(
        database,
        [],
        [LLMResponse("", [ToolCall("child-call", "child_task", {})]), LLMResponse("parent B")],
        store=store,
        tools=(echo, child_task),
    )
    start(store, rt)
    child_requirements = rt.requirements_bound()
    rt.children = {
        "child_task": lambda plan: ChildSession(
            replace(
                fixture_child_spec(store, "s", "child"),
                feature_level=2,
                bootstrap=child_requirements,
            ),
            "child work",
        )
    }
    switch(store, rt)
    drive(store, "s", runtime=rt)
    assert store.read_state("s")[0].phase == "parked"
    child_runtime, _ = routed(
        database,
        [LLMResponse("child A")],
        store=store,
        sid="child",
        tools=(echo, child_task),
    )
    drive(store, "child", runtime=child_runtime)
    with store.atomic() as tx:
        assert child_delivery(store, tx, "child")
    drive(store, "s", runtime=rt)
    child_state, _rows, _ = store.read_state("child")
    assert child_state.model_selection is None
    assert plans(store, "child")[0].payload["request"]["model"] == "scripted"
    assert all(p.payload["request"]["model"] == "B" for p in plans(store))
    assert not b.steps


@pytest.mark.capability("model-selection")
def test_level_one_selection_is_admitted_during_active_cycle(store, database):
    errors = []

    def select_during_call(request):
        try:
            push_fixture_input(store, "s", InboxItem("active-select", "model_selection", {"selection": SELECTION}))
        except BaseException as exc:
            errors.append(exc)
        return LLMResponse("", [ToolCall("echo-call", "echo", {})])

    rt, _ = routed(
        database,
        [select_during_call, LLMResponse("A")],
        [LLMResponse("B")],
        store=store,
    )
    start_level_one(store)
    drive(store, "s", runtime=rt)
    assert not errors
    state, _, _ = store.read_state("s")
    assert state.applied_inputs["active-select"].payload["disposition"] == "applied"
    assert [plan.payload["request"]["model"] for plan in plans(store)] == ["scripted", "B"]


@pytest.mark.capability("model-selection")
def test_level_one_idle_selection_and_null_are_inherited_by_next_turn(store, database):
    rt, _ = routed(database, [LLMResponse("A"), LLMResponse("default")], [LLMResponse("B")], store=store)
    start_level_one(store)
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("idle-select", "model_selection", {"selection": SELECTION}))
    push_fixture_input(store, "s", InboxItem("next", "user", {"content": "next"}))
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("clear", "model_selection", {"selection": None}))
    push_fixture_input(store, "s", InboxItem("after-clear", "user", {"content": "after clear"}))
    drive(store, "s", runtime=rt)
    state, _, _ = store.read_state("s")
    assert state.applied_inputs["idle-select"].payload["disposition"] == "applied"
    assert state.applied_inputs["clear"].payload["disposition"] == "applied"
    assert state.model_selection is None
    assert [plan.payload["request"]["model"] for plan in plans(store)] == ["scripted", "B", "scripted"]


@pytest.mark.capability("model-selection")
def test_level_one_unresolvable_selection_keeps_later_valid_input_admissible(store, database):
    rt, _ = routed(database, [LLMResponse("A")], [LLMResponse("B")], store=store)
    start_level_one(store)
    drive(store, "s", runtime=rt)
    push_fixture_input(
        store,
        "s",
        InboxItem("unknown", "model_selection", {"selection": {"model": "unknown", "route": {}}}),
    )
    push_fixture_input(store, "s", InboxItem("valid", "model_selection", {"selection": SELECTION}))
    push_fixture_input(store, "s", InboxItem("next", "user", {"content": "next"}))
    drive(store, "s", runtime=rt)
    state, records, _ = store.read_state("s")
    rejected = state.applied_inputs["unknown"].payload
    assert rejected["disposition"] == "rejected"
    rejected_record = next(
        row for row in records if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "unknown"
    )
    hydrated = hydrate_record_payload(rejected_record, hydration_for_session_store(store, "s"))
    assert hydrated["reason"] == "model_selection_unresolvable"
    assert state.applied_inputs["valid"].payload["disposition"] == "applied"
    assert [plan.payload["request"]["model"] for plan in plans(store)] == ["scripted", "B"]


class ExactModelProvider(ScriptedModelProvider):
    def resolve(self, model: ModelRef):
        if model.model_name not in {"scripted", "B"}:
            raise ModelError("unknown model")
        return super().resolve(model)


@pytest.mark.capability("model-selection")
@pytest.mark.sqlite_file
def test_level_two_session_driver_typed_null_restores_base_and_next_valid_selection(store, database):
    rt, selected = routed(
        database,
        [LLMResponse("initial"), LLMResponse("base"), LLMResponse("base again")],
        [LLMResponse("selected"), LLMResponse("selected again")],
        store=store,
    )
    start(store, rt)
    drive(store, "s", runtime=rt)

    provider_requests: list[str] = []

    class RecordingProvider(ExactModelProvider):
        def resolve(self, model: ModelRef):
            provider_requests.append(model.model_name)
            return super().resolve(model)

    rt.config.model_provider = RecordingProvider(
        backend="anthropic",
        default_model="scripted",
        llm=rt.llm,
        context_length=128_000,
        max_output_tokens=16_384,
    ).with_token_limits(8192, 2048)
    runtime_requests: list[dict[str, Any]] = []
    runtime_resolver = rt.model_resolver
    assert runtime_resolver is not None

    def resolve(request):
        runtime_requests.append(request)
        if request["model"] == "B" and request["route"] == {}:
            return selected, replace(
                rt.resolved,
                backend="anthropic",
                model_id="B",
                requested_model="B",
                selected_model="B",
                context_length=8192,
                max_output_tokens=2048,
            )
        return runtime_resolver(request)

    rt.model_resolver = resolve
    driver = SessionDriver(
        store=store,
        content_key_provider=fixture_key_provider(),
        worker_level=2,
        handler_versions=(rt.handler_version,),
        capabilities=_worker_capabilities(rt, (SELECTION,)),
    )
    try:
        driver._register_runtime("s", rt, armed=True)
        provider_requests.clear()

        tables = ("sk_inbox", "sk_record", "sk_content", "sk_content_link")
        counts_before = tuple(
            store._one(f"SELECT COUNT(*) FROM {table} WHERE tenant_id='default' AND session_id='s'")[0] for table in tables
        )
        with pytest.raises(ModelError, match="unknown model"):
            driver.push("s", InboxItem("unknown", "model_selection", {"selection": {"model": "unknown", "route": {}}}))
        counts_after = tuple(
            store._one(f"SELECT COUNT(*) FROM {table} WHERE tenant_id='default' AND session_id='s'")[0] for table in tables
        )
        assert counts_after == counts_before
        assert store.read_input("s", "unknown") is None

        driver.push("s", InboxItem("select", "model_selection", {"selection": {"model": "B", "route": {}}}))
        driver.push("s", InboxItem("next", "user", {"content": "use selected"}))
        drive(store, "s", runtime=rt)
        selected_state, selected_records, _ = store.read_state("s")
        selected_payload = hydrate_record_payload(
            next(
                row
                for row in selected_records
                if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "select"
            ),
            hydration_for_session_store(store, "s"),
        )
        assert selected_state.model_selection is not None, selected_payload
        assert selected_payload["model_binding"]["model"] == "B"

        driver.push("s", InboxItem("clear", "model_selection", {"selection": None}))
        driver.push("s", InboxItem("after-clear", "user", {"content": "after clear"}))
        drive(store, "s", runtime=rt)
        cleared_state, records, _ = store.read_state("s")
        assert cleared_state.model_selection is None
        clear_record = next(
            row for row in records if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "clear"
        )
        clear_payload = hydrate_record_payload(clear_record, hydration_for_session_store(store, "s"))
        assert clear_payload["disposition"] == "applied"
        assert clear_payload["input"]["payload"] == {"selection": None}
        assert "model_binding" not in clear_payload

        driver.push("s", InboxItem("select-again", "model_selection", {"selection": {"model": "B", "route": {}}}))
        driver.push("s", InboxItem("after-valid", "user", {"content": "use selected again"}))
        drive(store, "s", runtime=rt)
        final_state, final_records, _ = store.read_state("s")
        assert final_state.model_selection is not None
        final_payload = hydrate_record_payload(
            next(
                row
                for row in final_records
                if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "select-again"
            ),
            hydration_for_session_store(store, "s"),
        )
        assert final_payload["model_binding"]["model"] == "B"
        assert [plan.payload["request"]["model"] for plan in plans(store)] == ["scripted", "B", "scripted", "B"]
        assert runtime_requests
        assert all(request["model"] is not None for request in runtime_requests)
        assert provider_requests == ["unknown", "B", "B"]

        clear_input = store.read_input("s", "clear")
        assert clear_input is not None
        assert clear_input.item.feature_level == 2
        assert clear_input.item.requirements == {
            "level": 2,
            "handler_version": rt.handler_version,
            "capabilities": [],
        }

        raw_inbox = b"".join(
            body
            for (body,) in store._rows(
                "SELECT body FROM sk_inbox WHERE tenant_id='default' AND session_id='s' ORDER BY input_seq"
            )
        )
        raw_records = b"".join(
            body
            for (body,) in store._rows("SELECT body FROM sk_record WHERE tenant_id='default' AND session_id='s' ORDER BY seq")
        )
        encrypted_content = b"".join(
            ciphertext
            for (ciphertext,) in store._rows(
                "SELECT ciphertext FROM sk_content WHERE tenant_id='default' AND session_id='s' ORDER BY content_id"
            )
        )
        for raw in (raw_inbox, raw_records):
            assert b'"selection":{"model":"B"' not in raw
            assert b"after clear" not in raw
            assert b"use selected again" not in raw
        for plaintext in (b'"model":"B"', b"after clear", b"use selected again"):
            assert plaintext not in encrypted_content
        assert not any(
            record.record.kind == "op_unknown" for record in records if record.record.turn_id == clear_record.record.turn_id
        )
    finally:
        driver.close()


@pytest.mark.capability("model-selection")
def test_level_two_wrong_handler_selection_writes_nothing(store, database):
    rt, _ = routed(database, [LLMResponse("A")], store=store)
    start(store, rt)
    requirements = replace(rt.model_selection_requirements(SELECTION), handler_version="handler/mismatched")
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    item = InboxItem(
        "wrong-handler",
        "model_selection",
        {"selection": SELECTION},
        feature_level=2,
        requirements=requirements.to_dict(),
    )
    _, records_before, watermark_before = store.read_state("s")
    inbox_before = len(store.peek_inbox("s"))
    content_before = store._rows("SELECT COUNT(*) FROM sk_content WHERE tenant_id='default' AND session_id='s'")[0][0]
    links_before = store._rows("SELECT COUNT(*) FROM sk_content_link WHERE tenant_id='default' AND session_id='s'")[0][0]
    with pytest.raises(Conflict, match="handler does not match"):
        push_fixture_input(store, "s", item)
    _, records_after, watermark_after = store.read_state("s")
    assert (watermark_after, len(records_after), len(store.peek_inbox("s")), store.read_input("s", "wrong-handler")) == (
        watermark_before,
        len(records_before),
        inbox_before,
        None,
    )
    assert store._rows("SELECT COUNT(*) FROM sk_content WHERE tenant_id='default' AND session_id='s'")[0][0] == content_before
    assert store._rows("SELECT COUNT(*) FROM sk_content_link WHERE tenant_id='default' AND session_id='s'")[0][0] == links_before


@pytest.mark.capability("model-selection")
def test_level_two_null_idle_handler_rejects_typed_selection(store, database):
    rt, _ = routed(database, [LLMResponse("A")], store=store)
    create_level_two(store, replace(rt.requirements_bound(), handler_version=None))
    requirements = rt.model_selection_requirements(SELECTION)
    item = InboxItem(
        "null-handler",
        "model_selection",
        {"selection": SELECTION},
        feature_level=2,
        requirements=requirements.to_dict(),
    )
    inbox_before = len(store.peek_inbox("s"))
    with pytest.raises(Conflict, match="handler does not match"):
        push_fixture_input(store, "s", item)
    assert len(store.peek_inbox("s")) == inbox_before
    assert store.read_input("s", "null-handler") is None


@pytest.mark.capability("model-selection")
def test_level_two_pending_user_sets_idle_route_for_selection_capabilities(store, database):
    rt, _ = routed(database, [LLMResponse("A")], [LLMResponse("B")], store=store)
    create_level_two(store, replace(rt.requirements_bound(), handler_version=None))
    push_user(store, rt, "next", "next")
    selection_requirements = rt.model_selection_requirements(SELECTION)
    push_fixture_input(
        store,
        "s",
        InboxItem(
            "select",
            "model_selection",
            {"selection": SELECTION},
            feature_level=2,
            requirements=selection_requirements.to_dict(),
        ),
    )
    routing = routing_requirements(store)
    assert routing.handler_version == rt.handler_version
    assert set(selection_requirements.capabilities) <= set(routing.capabilities)


@pytest.mark.capability("model-selection")
def test_level_one_selection_does_not_inherit_level_two_bootstrap_capabilities(store, database):
    bootstrap = SessionRequirements(2, "handler/h1", ("family:bootstrap",))
    create_level_two(store, bootstrap)
    user_requirements = SessionRequirements(2, "handler/h1", ("tool:user",))
    push_fixture_input(
        store,
        "s",
        InboxItem(
            "next",
            "user",
            {"content": "next"},
            feature_level=2,
            requirements=user_requirements.to_dict(),
        ),
    )
    push_fixture_input(store, "s", InboxItem("select", "model_selection", {"selection": SELECTION}))
    routing = routing_requirements(store)
    assert routing.handler_version == "handler/h1"
    assert routing.capabilities == ("tool:user",)
