"""Cached process-local values remain isolated at every mutable host boundary."""

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from support.s1_fixture_values import (
    append_fixture_records,
    clear_fixture_state_cache,
    content_ref,
    create_fixture_session,
    fixture_content_reader,
    principal,
    push_fixture_input,
    read_fixture_state,
    wrap_fixture_store,
)

from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.kernel._hydration import reference_digest
from vv_agent.kernel._sql import _extend_chain_digest
from vv_agent.session import records
from vv_agent.session.context import project_context
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec, digest, make_record
from vv_agent.session.reducer import Fold
from vv_agent.session.store import Conflict
from vv_agent.tools.function import function_tool
from vv_agent.tools.metadata import ToolMetadata
from vv_agent.types import LLMResponse, ToolCall

from .helpers import record
from .test_recovery_matrix import runtime, start


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def session_runtime(store, database, steps, tools=(), *, sid="s", **kwargs):
    kwargs.setdefault("content_reader", fixture_content_reader(store, sid))
    return runtime(database, steps, tools, **kwargs)


def read_persisted_state(store, database, session_id="s"):
    from .conftest import open_store

    with open_store(database) as persisted:
        return read_fixture_state(wrap_fixture_store(persisted), session_id)


def definition_and_task(database):
    rt = runtime(database, [])
    rt.model_binding(rt.llm, rt.resolved)
    task = rt.compile("go", "t")
    definition = rt.definition(task)
    return task, definition


def test_two_session_prefixes_reuse_validated_bytes_with_bounded_eviction(store, database, monkeypatch):
    for sid in ("parent", "child", "third"):
        create_fixture_session(store, sid)
    from .conftest import open_store

    with open_store(database) as reopened:
        store = wrap_fixture_store(reopened)
        clear_fixture_state_cache(store)
        store._root._previous_prefix = None
        for sid in ("parent", "child"):
            store.read_state(sid)
        calls = []
        stored = store._root._stored

        def counted(*args):
            calls.append(True)
            return stored(*args)

        monkeypatch.setattr(store._root, "_stored", counted)
        for _ in range(3):
            for sid in ("parent", "child"):
                store.read_state(sid)
        assert not calls
        store.read_state("third")
        assert {store._fold_cache.session_id, store._previous_prefix.session_id} == {"child", "third"}
        calls.clear()
        store.read_state("parent")
        assert calls, "the third session must evict the older prefix"
        clear_fixture_state_cache(store)
        calls.clear()
        store.read_state("third")
        assert calls and store._previous_prefix is None, "explicit disposal clears both slots"


@pytest.mark.parametrize("change", ["rollback", "same_head", "storage_digest", "schema"])
def test_displaced_parent_prefix_still_checks_rollback_and_database_bytes(store, database, change):
    for sid in ("s", "child"):
        create_fixture_session(store, sid, attributes={"seed": {"unseen_input_protection": False}})

    class Stop(BaseException):
        pass

    def stop_after_turn_start(point, committed):
        if point == "after_commit" and committed.kind == "turn_started":
            raise Stop

    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    rt = session_runtime(store, database, [LLMResponse("done")], hook=stop_after_turn_start)
    with pytest.raises(Stop):
        drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    turn_id = state.active_turn_id
    assert turn_id is not None
    row_count = len(rows)
    store.read_state("child")
    assert store._previous_prefix.session_id == "s"
    if change == "rollback":
        with pytest.raises(RuntimeError, match="rollback"), store._transaction():
            turn_ended = make_record_before_publication(
                "turn_ended",
                session_id="s",
                turn_id=turn_id,
                payload=record("turn_ended").payload | {"budget": RunBudgetLimits().to_dict()},
            )
            append_fixture_records(store, "s", (turn_ended,), commit_id="provisional")
            store.read_state("child")
            raise RuntimeError("rollback")
        state, rows, _ = store.read_state("s")
        assert len(rows) == row_count and not state.operations
        return
    turn_row = next(row for row in store.read("s").records if row.record.kind == "turn_started")
    value = turn_row.record.to_dict()
    value["turn_id"] = "replacement"
    value["record_id"] = "turn/replacement/started"
    if change == "schema":
        value["payload"]["extra"] = True
    body = canonical_json_bytes(value)
    checksum = "0" * 64 if change == "storage_digest" else sha256(body).hexdigest()
    previous_chain = store._rows(
        "SELECT chain_digest FROM sk_record WHERE session_id=%s AND seq=%s",
        ("s", turn_row.seq - 1),
    )[0][0]
    record_digest = sha256(body).hexdigest()
    chain_digest = _extend_chain_digest(previous_chain, turn_row.seq, record_digest)
    commit_id, manifest_body = store._rows(
        "SELECT c.commit_id,c.body FROM sk_commit c JOIN sk_record r "
        "ON r.tenant_id=c.tenant_id AND r.session_id=c.session_id AND r.commit_id=c.commit_id "
        "WHERE r.session_id=%s AND r.seq=%s",
        ("s", turn_row.seq),
    )[0]
    manifest = json.loads(manifest_body)
    manifest["records"][0] = {"digest": record_digest, "record_id": value["record_id"]}
    manifest_body = canonical_json_bytes(manifest)
    commit_digest = sha256(manifest_body).hexdigest()
    record_sequences = canonical_json_bytes([[value["record_id"], turn_row.seq]])
    with store.atomic():
        store._rows(
            "UPDATE sk_commit SET body=%s,digest=%s,record_sequences=%s WHERE session_id=%s AND commit_id=%s",
            (manifest_body, commit_digest, record_sequences, "s", commit_id),
        )
        store._rows(
            "UPDATE sk_record SET body=%s,digest=%s,chain_digest=%s,commit_digest=%s,record_id=%s,turn_id=%s "
            "WHERE session_id=%s AND seq=%s",
            (
                body,
                checksum,
                chain_digest,
                commit_digest,
                value["record_id"],
                value["turn_id"],
                "s",
                turn_row.seq,
            ),
        )
    if change == "same_head":
        assert store.read_state("s")[0].active_turn_id == "replacement"
    else:
        with pytest.raises(Conflict if change == "storage_digest" else records.RecordError):
            store.read_state("s")


@pytest.mark.parametrize("source", ["producer", "cold"])
def test_retained_json_is_not_parsed_again_and_nested_outputs_are_detached(source, monkeypatch):
    attributes = content_ref("attributes/s")
    rec = SessionSpec("s", principal("s"), content_ref("workspace/s"), attributes=attributes).record()
    body = rec.encode()
    if source == "cold":
        loads = json.loads
        calls = []

        def parse(*args, **kwargs):
            calls.append(1)
            return loads(*args, **kwargs)

        monkeypatch.setattr(json, "loads", parse)
        rec = records.Record.parse(body)
        assert len(calls) == 1

    def unexpected(*args, **kwargs):
        raise AssertionError("retained JSON must not be reparsed")

    monkeypatch.setattr(json, "loads", unexpected)
    original_attributes = rec.to_dict()["payload"]["attributes"]
    attributes["labels"]["source"].append("user")
    rec.payload["attributes"]["labels"]["source"].append("service")
    rec.to_dict()["payload"]["attributes"]["labels"]["source"].clear()
    assert rec.payload["attributes"] == original_attributes
    assert rec.encode() == body


def test_state_snapshot_detaches_waits_and_mutable_state_containers():
    fold = Fold()
    rows = [record(kind) for kind in ("session_created", "turn_started", "op_planned", "op_started", "op_parked")]
    fold.extend(rows)
    snapshot = fold.snapshot()
    assert snapshot.child_completions is None
    assert fold.fork().state.child_completions is None
    wait = snapshot.operations["o"].attempts[1].wait
    assert wait is not None
    wait["handle"]["query_ref"] = "changed"
    snapshot.turns["t"].cancelled = True
    snapshot.applied_inputs["new"] = rows[0]
    snapshot.admitted_inputs.add("new")
    snapshot.compactions.append(rows[0])
    snapshot.operations["o"].attempts.clear()
    fresh = fold.snapshot()
    fresh_wait = fresh.operations["o"].attempts[1].wait
    assert fresh_wait is not None and fresh_wait["handle"]["query_ref"] == content_ref("query")
    assert not fresh.turns["t"].cancelled
    assert not fresh.applied_inputs and not fresh.admitted_inputs and not fresh.compactions
    assert rows[-1].payload["handle"]["query_ref"] == content_ref("query")


def test_task_cache_and_context_outputs_cannot_mutate_definition(store, database, monkeypatch):
    rt = runtime(database, [])
    rt.config = replace(rt.config, shared_state={"nested": {"values": [1]}}, metadata={"host": {"values": [2]}})
    compiled = rt.compile("go", "t")
    definition = rt.definition(compiled)
    rec = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload
        | {
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits().to_dict(),
        },
    )
    object.__setattr__(rec, "_task_value", compiled)
    create_fixture_session(store, "s")
    append_fixture_records(store, "s", (rec,), commit_id="definition")
    from .conftest import open_store

    with open_store(database) as persisted:
        _, rows, _ = read_fixture_state(wrap_fixture_store(persisted), "s")
    rec = rows[-1].record
    task = rec.task()
    task.initial_shared_state["nested"]["values"].clear()
    task.metadata["host"]["values"].clear()
    task.extra_tool_names.clear()
    original = rec.task()

    def unexpected(*args, **kwargs):
        raise AssertionError("retained tasks must not be decoded again")

    monkeypatch.setattr(type(task), "from_dict", unexpected)
    assert rec.task().to_dict() == original.to_dict()
    fold = Fold()
    fold.extend(r.record for r in rows)
    messages = project_context(rows, fold.state)
    messages[0].metadata["host"]["values"].append(3)
    assert rec.task().metadata["host"]["values"] == [2]


def test_schema_plan_and_default_registry_are_reused_across_turns(store, database, monkeypatch):
    from vv_agent.kernel import _definition
    from vv_agent.session import runtime as bindings

    calls = {"registry": 0, "schemas": 0}
    build, plan = bindings.build_default_registry, _definition.plan_tool_schemas

    def registry():
        calls["registry"] += 1
        return build()

    def schemas(**kwargs):
        calls["schemas"] += 1
        return plan(**kwargs)

    monkeypatch.setattr(bindings, "build_default_registry", registry)
    monkeypatch.setattr(_definition, "plan_tool_schemas", schemas)
    start(store)
    rt = session_runtime(store, database, [LLMResponse("done"), LLMResponse("done")])
    drive(store, "s", runtime=rt)
    push_fixture_input(store, "s", InboxItem("next", "user", {"content": "again"}))
    drive(store, "s", runtime=rt)
    assert calls == {"registry": 1, "schemas": 1}


@function_tool(tool_metadata=ToolMetadata(capability_tags=["cache-test"]))
def cached_tool(value: str) -> str:
    return value


@pytest.mark.parametrize("change", ["schema", "capability", "policy", "memory", "children", "task", "bool_integer"])
def test_definition_cache_tracks_all_mutable_planning_inputs(database, change):
    rt = runtime(database, [], tools=[cached_tool])
    task = rt.compile("go", "t")
    task.metadata["typed_input"] = True
    before = rt.definition_digest(task)
    detached = rt.definition(task)
    detached["tools"][0]["function"]["description"] = "caller mutation"
    detached["task"]["metadata"]["caller"] = True
    assert rt.definition_digest(task) == before
    if change == "schema":
        schema = rt.registry.get_schema("cached_tool")
        schema["function"]["description"] = "updated"
        rt.registry.register_schema("cached_tool", schema)
    elif change == "capability":
        rt.registry.get_executor("cached_tool").tool_metadata.capability_tags.append("new-tag")
    elif change == "policy":
        task.metadata["_vv_agent_denied_capability_tags"] = ["cache-test"]
    elif change == "memory":
        rt.memory_manager.tool_result_retentions["cached_tool"] = "preserve"
    elif change == "children":
        rt.children["new-child"] = lambda plan: None
    elif change == "bool_integer":
        task.metadata["typed_input"] = 1
    else:
        task.extra_tool_names.clear()
        task.exclude_tools = ["cached_tool"]
    assert rt.definition_digest(task) != before
    if change in {"policy", "task"}:
        assert "cached_tool" not in [t["function"]["name"] for t in rt.definition(task)["tools"]]


def test_dynamic_enabled_tools_replan_on_changed_turn_context(store, database):
    enabled = {"value": False}
    observed = []

    @function_tool(is_enabled=lambda ctx, agent: enabled["value"])
    def dynamic() -> str:
        return "enabled"

    def response(request):
        observed.append("dynamic" in [s["function"]["name"] for s in request.tools])
        return LLMResponse("done")

    start(store)
    rt = session_runtime(store, database, [response, response, response], tools=[dynamic])
    for i, value in enumerate([False, True, False]):
        enabled["value"] = value
        if i:
            push_fixture_input(store, "s", InboxItem(str(i), "user", {"content": "go"}))
        drive(store, "s", runtime=rt)
    assert observed == [False, True, False]


def test_host_hook_and_provider_mutation_do_not_poison_retained_records(store, database):
    @function_tool
    def nested(values: list[str]) -> str:
        values.append("handler")
        return "tool done"

    def response(request):
        request.tools[0]["function"]["description"] = "provider mutation"
        request.metadata["nested"]["values"].append("provider")
        return LLMResponse("", [ToolCall("call", "nested", {"values": ["original"]})])

    class Hook:
        def before_llm(self, event):
            event.tool_schemas[0]["function"]["description"] = "hook mutation"
            event.task.metadata["nested"]["values"].append("hook")

        def after_tool_call(self, event):
            event.call.arguments["values"].append("hook")

    start(store)
    rt = session_runtime(store, database, [response, LLMResponse("done")], tools=[nested])
    rt.config = replace(rt.config, metadata={"nested": {"values": ["original"]}})
    rt.hooks.hooks.append(Hook())
    drive(store, "s", runtime=rt)
    _, rows, _ = read_fixture_state(store, "s")
    frozen = next(r.record for r in rows if r.record.kind == "turn_started")
    assert frozen.payload["definition"]["task"]["metadata"]["nested"]["values"] == ["original"]
    assert frozen.payload["definition"]["tools"][0]["function"]["description"] != "hook mutation"
    planned = next(r.record for r in rows if r.record.kind == "op_planned" and r.record.payload["op_kind"] == "tool")
    assert planned.payload["request"]["arguments"]["values"] == ["original"]
    assert deepcopy(frozen) is frozen


@pytest.mark.parametrize("field", ["reasoning", "extra_headers", "extra_body", "extra_args", "response_format"])
def test_task_cache_copies_nested_model_settings_at_output_boundary(tmp_path, field):
    from vv_agent import ModelSettings
    from vv_agent.model_settings import ResponseFormat

    value = {"nested": {"values": [1]}}
    if field == "extra_headers":
        value = {"header": "original"}
    elif field == "response_format":
        value = ResponseFormat.json_schema_format({"name": "test", "schema": {"type": "object"}})
    settings = ModelSettings.from_dict({field: value})
    rt = runtime(tmp_path / "unused.sqlite", [])
    rt.config = replace(rt.config, model_settings=settings)
    compiled = rt.compile("go", "t")
    definition = rt.definition(compiled)
    rec = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload
        | {
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits().to_dict(),
        },
    )
    object.__setattr__(rec, "_task_value", compiled)
    task = rec.task()
    assert task.model_settings is not None
    before = deepcopy(task.model_settings.to_dict())
    returned = getattr(task.model_settings, field)
    if field == "response_format":
        returned.json_schema["schema"]["type"] = "string"
    elif field == "extra_headers":
        returned["header"] = "changed"
    else:
        returned["nested"]["values"].clear()
    fresh = rec.task().model_settings
    assert fresh is not None and fresh.to_dict() == before


def test_provisional_fold_copies_only_modified_operations_and_turns():
    baseline = Fold()
    baseline.extend([record(kind) for kind in ("session_created", "turn_started", "op_planned", "op_started")])
    provisional = baseline.fork()
    provisional.extend([record("usage_observed")])
    assert provisional.state.operations["o"] is baseline.state.operations["o"]
    provisional.extend([record("op_completed")])
    assert provisional.state.operations["o"] is not baseline.state.operations["o"]
    assert provisional.state.operations["o"].state == "completed"
    assert baseline.state.operations["o"].state == "started"
    control = InboxItem("cancel", "control", {"action": "cancel"}, "t", 1, actor=principal("cancel"))
    applied = record("input_applied", input=control.to_dict(), input_digest=control.digest)
    provisional.extend([applied], consumed_inputs=[control])
    assert provisional.state.turns["t"].cancelled
    assert not baseline.state.turns["t"].cancelled
    assert not baseline.state.applied_inputs


def test_recovered_model_receipt_is_detached_before_tool_hooks(store, database):
    effects = []

    @function_tool
    def nested(values: list[str]) -> str:
        effects.append(list(values))
        return "tool done"

    class Hook:
        def before_tool_call(self, event):
            event.call.arguments["values"].append("hook")

    class Restart(BaseException):
        pass

    def stop_after_receipt(point, committed):
        if point == "after_commit" and committed.kind == "op_completed":
            raise Restart

    start(store)
    call = ToolCall("call", "nested", {"values": ["original"]})
    rt = session_runtime(
        store,
        database,
        [LLMResponse("", [call]), LLMResponse("done")],
        tools=[nested],
        hook=stop_after_receipt,
    )
    rt.hooks.hooks.append(Hook())
    with pytest.raises(Restart):
        drive(store, "s", runtime=rt)
    state_before, rows_before, _ = read_fixture_state(store, "s")
    receipt_id = next(r.record.operation_id for r in reversed(rows_before) if r.record.kind == "op_completed")
    assert receipt_id is not None
    receipt_result = rows_before[-1].record.payload["result"]
    recovered = session_runtime(store, database, [LLMResponse("done")], tools=[nested])
    recovered.hooks.hooks.append(Hook())
    drive(store, "s", runtime=recovered)
    state, rows, _ = read_fixture_state(store, "s")
    receipt = next(r.record for r in rows if r.record.kind == "op_completed" and r.record.operation_id == receipt_id)
    assert effects == [["original", "hook"]]
    assert receipt.payload["result"] == receipt_result
    store._fold_cache = None
    cold_state, cold_rows, _ = read_fixture_state(store, "s")
    assert cold_state == state
    assert [r.record.to_dict() for r in cold_rows] == [r.record.to_dict() for r in rows]
    assert state_before.operations[receipt_id].state == "completed"
    assert rows_before[-1].record.kind == "op_completed"


@pytest.mark.parametrize("invalid", [float("nan"), 2**100])
def test_failed_definition_validation_preserves_last_valid_cache(tmp_path, invalid):
    rt = runtime(tmp_path / "unused.sqlite", [])
    task = rt.compile("go", "t")
    definition = rt.definition(task)
    definition_digest = rt.definition_digest(task)
    task.metadata["invalid"] = invalid
    with pytest.raises(ValueError, match="RFC 8785 I-JSON"):
        rt.definition(task)
    del task.metadata["invalid"]
    assert rt.definition(task) == definition
    assert rt.definition_digest(task) == definition_digest


def test_snapshot_does_not_copy_fold_history_or_identity_indexes(monkeypatch):
    baseline = Fold()
    baseline.extend([record(kind) for kind in ("session_created", "turn_started", "op_planned", "op_started")])

    def unexpected(*args, **kwargs):
        raise AssertionError("a state snapshot must not fork the whole log")

    monkeypatch.setattr(Fold, "fork", unexpected)
    snapshot = baseline.snapshot()
    snapshot.operations["o"].attempts.clear()
    assert baseline.state.operations["o"].state == "started"


@pytest.mark.parametrize("change", ["schema", "capability", "memory", "children", "model_binding"])
def test_retained_definition_checks_live_bindings_without_serializing_task(database, monkeypatch, change):
    rt = runtime(database, [], tools=[cached_tool])
    task = rt.compile("go", "t")
    definition = rt.definition(task)
    retained = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload
        | {
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits().to_dict(),
        },
    )
    object.__setattr__(retained, "_task_value", task)
    before = rt.definition_digest(retained)
    assert before == digest(definition)

    def unexpected(*args, **kwargs):
        raise AssertionError("a frozen task must use its retained JSON")

    monkeypatch.setattr(type(task), "to_dict", unexpected)
    assert rt.definition_digest(retained) == before
    if change == "schema":
        schema = rt.registry.get_schema("cached_tool")
        schema["function"]["description"] = "updated"
        rt.registry.register_schema("cached_tool", schema)
    elif change == "capability":
        rt.registry.get_executor("cached_tool").tool_metadata.capability_tags.append("new-tag")
    elif change == "memory":
        rt.memory_manager.tool_result_retentions["cached_tool"] = "preserve"
    elif change == "children":
        rt.children["new-child"] = lambda plan: None
    else:
        rt.resolved = replace(rt.resolved, model_id="changed")
    assert rt.definition_digest(retained) != before


def test_recovered_completion_does_not_construct_a_second_receipt(store, database, monkeypatch):
    from vv_agent.kernel import _driver as core_driver

    completed = []
    make = core_driver.CoreDriver.record

    def counted(driver, kind, payload, plan=None, *, tid=None):
        if kind == "op_completed":
            completed.append(plan.operation_id if plan is not None else None)
        return make(driver, kind, payload, plan, tid=tid)

    monkeypatch.setattr(core_driver.CoreDriver, "record", counted)
    start(store)
    rt = session_runtime(store, database, [LLMResponse("done")])
    drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    assert state.active_turn_id is None
    assert completed == [r.record.operation_id for r in rows if r.record.kind == "op_completed"]
    assert len(completed) == 1


@pytest.mark.parametrize("context", ["normal", "audit"])
def test_preferred_endpoint_is_folded_from_successful_dispatch_and_survives_recovery(store, database, context):
    create_fixture_session(store, "s")
    baseline = Fold()
    task, definition = definition_and_task(database)
    binding = {"backend": "test", "model": "test", "endpoints": ["a", "b"], "context_length": None, "max_output_tokens": None}
    request = {"messages": [], "metadata": {"vv_session": {"endpoint_order": ["b", "a"], "model_binding": binding}}}
    turn = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload
        | {
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits().to_dict(),
        },
    )
    object.__setattr__(turn, "_task_value", task)
    sources = [
        turn,
        make_record_before_publication(
            "op_planned",
            session_id="s",
            turn_id="t",
            operation_id="o",
            attempt=1,
            payload=record("op_planned").payload | {"request": request, "request_digest": digest(request)},
        ),
        record("op_started", endpoint_id="b"),
    ]
    completed = make_record_before_publication(
        "op_completed",
        session_id="s",
        turn_id="t",
        operation_id="o",
        attempt=1,
        payload=record("op_completed").payload | {"request_digest": digest(request), "context": context},
    )
    prefix_len = 1 + len(sources)
    consumed_input_ids: tuple[str, ...] = ()
    consumed_inputs: tuple[InboxItem, ...] = ()
    if context == "audit":
        push_fixture_input(
            store,
            "s",
            InboxItem("cancel", "control", {"action": "cancel"}, "t", 1, actor=principal("cancel")),
        )
        stored_control = next(row.item for row in store.peek_inbox("s", ready_only=False) if row.item.input_id == "cancel")
        applied = record("input_applied", input=stored_control.to_dict(), input_digest=stored_control.digest)
        appended = (*sources, applied, completed)
        consumed_input_ids = (stored_control.input_id,)
        consumed_inputs = (stored_control,)
        prefix_len += 1
    else:
        appended = (*sources, completed)
    append_fixture_records(
        store,
        "s",
        appended,
        commit_id="endpoint",
        consume_input_ids=consumed_input_ids,
    )
    _, rows, _ = read_persisted_state(store, database)
    baseline.extend((r.record for r in rows[:prefix_len]), consumed_inputs=consumed_inputs)
    assert baseline.state.preferred_endpoint_id is None
    provisional = baseline.fork()
    provisional.extend(r.record for r in rows[prefix_len:])
    assert baseline.state.preferred_endpoint_id is None
    assert provisional.snapshot().preferred_endpoint_id == "b"
    recovered = Fold()
    recovered.extend((r.record for r in rows), consumed_inputs=consumed_inputs)
    assert recovered.state == provisional.state


@pytest.mark.parametrize(
    "result",
    [
        {"error_code": "failed", "value": None},
        {"error_code": "failed", "value": True},
        {"error_code": "failed", "value": 7},
        {"error_code": "failed", "value": "opaque"},
        {"error_code": "failed", "value": []},
        {"error_code": "failed", "value": {}},
    ],
)
def test_fold_keeps_opaque_receipts_and_does_not_prefer_failed_endpoints(store, database, result):
    create_fixture_session(store, "s")
    task, definition = definition_and_task(database)
    binding = {"backend": "test", "model": "test", "endpoints": ["a"], "context_length": None, "max_output_tokens": None}
    request = {"messages": [], "metadata": {"vv_session": {"endpoint_order": ["a"], "model_binding": binding}}}
    baseline = Fold()
    turn = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload
        | {
            "definition": definition,
            "definition_digest": digest(definition),
            "budget": RunBudgetLimits().to_dict(),
        },
    )
    object.__setattr__(turn, "_task_value", task)
    sources = [
        turn,
        make_record_before_publication(
            "op_planned",
            session_id="s",
            turn_id="t",
            operation_id="o",
            attempt=1,
            payload=record("op_planned").payload | {"request": request, "request_digest": digest(request)},
        ),
        record("op_started", endpoint_id="a"),
        make_record_before_publication(
            "op_completed",
            session_id="s",
            turn_id="t",
            operation_id="o",
            attempt=1,
            payload=record("op_completed").payload
            | {
                "request_digest": digest(request),
                "result": result,
                "result_digest": digest(result),
            },
        ),
    ]
    append_fixture_records(store, "s", sources, commit_id="opaque")
    _, rows, _ = read_persisted_state(store, database)
    baseline.extend(r.record for r in rows)
    receipt = baseline.state.operations["o"].attempts[1].result
    assert receipt is not None and receipt._payload["result"] == result
    assert baseline.state.operations["o"].state == "completed"
    assert baseline.state.preferred_endpoint_id is None


def test_definition_composition_matches_full_jcs_across_turns_and_nested_values(tmp_path):
    from vv_agent.types import Message

    rt = runtime(tmp_path / "unused.sqlite", [], tools=[cached_tool])
    for i in range(3):
        task = rt.compile("go", f"turn/{i}")
        task.metadata["nested"] = {"task": None, "tools": {"😀": [-0.0, 1e-27, True, i], "\uffff": '"task":null'}}
        task.initial_messages = [Message("user", "汉字")]
        definition = rt.definition(task)
        assert rt.definition_digest(task) == digest(definition)
        retained = make_record_before_publication(
            "turn_started",
            session_id="s",
            turn_id=task.task_id,
            payload=record("turn_started").payload
            | {
                "definition": definition,
                "definition_digest": digest(definition),
                "budget": RunBudgetLimits().to_dict(),
            },
        )
        object.__setattr__(retained, "_task_value", task)
        assert rt.definition_digest(retained) == digest(retained._payload["definition"])


def test_tool_schemas_share_only_validated_json_across_turns_and_host_copies(store, database):
    start(store)
    rt = session_runtime(store, database, [LLMResponse("done") for _ in range(3)], tools=[cached_tool])
    for i in range(3):
        if i:
            push_fixture_input(store, "s", InboxItem(str(i), "user", {"content": "again"}))
        drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    sources = [r.record for r in rows if r.record.kind in {"turn_started", "op_planned"}]
    schemas = [r._payload["definition" if r.kind == "turn_started" else "request"]["tools"] for r in sources]
    assert len(schemas) == 6 and len({canonical_json_bytes(value) for value in schemas}) == 1
    logical = [r.record.to_dict() for r in rows]
    raw_sources = [row.record for row in store.read("s").records if row.record.kind in {"turn_started", "op_planned"}]
    assert len(raw_sources) == len(sources)
    for raw, r in zip(raw_sources, sources, strict=True):
        container = "definition" if r.kind == "turn_started" else "request"
        raw_ref = ContentRef.from_dict(raw._payload[container])
        decoded = records.Record.parse(raw.encode())
        decoded_ref = ContentRef.from_dict(decoded._payload[container])
        assert raw.encode() == decoded.encode()
        assert decoded_ref == raw_ref
        expected = reference_digest(raw_ref)
        assert raw._payload[f"{container}_digest"] == expected
        assert decoded._payload[f"{container}_digest"] == expected

        before = canonical_json_bytes(r.to_dict()["payload"][container]["tools"])
        detached_payload, detached_value = r.payload, r.to_dict()
        detached_payload[container]["tools"].clear()
        detached_value["payload"][container]["tools"][0]["function"]["description"] = "host mutation"
        assert not detached_payload[container]["tools"]
        assert detached_value["payload"][container]["tools"][0]["function"]["description"] == "host mutation"
        assert canonical_json_bytes(r.to_dict()["payload"][container]["tools"]) == before
    store._fold_cache = None
    cold, reread, _ = read_fixture_state(store, "s")
    assert cold == state and [r.record.to_dict() for r in reread] == logical


@pytest.mark.parametrize("container", ["definition", "request"])
def test_retained_schema_encoding_matches_full_jcs_and_checks_embedded_digest(tmp_path, container):
    value = content_ref("汉字\U0001f600")
    kind = "turn_started" if container == "definition" else "op_planned"
    payload = record(kind).payload | {container: value, f"{container}_digest": digest(value)}
    retained = make_record(
        kind,
        session_id="s",
        turn_id="t",
        operation_id="o" if container == "request" else None,
        attempt=1 if container == "request" else None,
        payload=payload,
    )
    assert retained.encode() == canonical_json_bytes(retained.to_dict())
    assert retained._payload[container] == value
    assert records.Record.parse(retained.encode()).to_dict() == retained.to_dict()
    with pytest.raises(records.RecordError, match=f"{container} digest mismatch"):
        make_record(
            kind,
            session_id="s",
            turn_id="t",
            operation_id="o" if container == "request" else None,
            attempt=1 if container == "request" else None,
            payload=payload | {f"{container}_digest": "0" * 64},
        )


def test_retained_schema_source_requires_full_validation_before_reuse():
    with pytest.raises(records.RecordError, match="invalid session schema"):
        record("turn_started", definition={"tools": []}, definition_digest=digest({"tools": []}))
    with pytest.raises(records.RecordError, match="definition digest mismatch"):
        record("turn_started", definition_digest="0" * 64)
