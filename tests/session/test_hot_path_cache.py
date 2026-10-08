"""Cached process-local values remain isolated at every mutable host boundary."""

from copy import deepcopy
from dataclasses import replace

import pytest

from vv_agent.session import records
from vv_agent.session.context import project_context
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec, digest, make_record
from vv_agent.session.reducer import Fold
from vv_agent.session.store import StoredRecord
from vv_agent.tools.function import function_tool
from vv_agent.tools.metadata import ToolMetadata
from vv_agent.types import LLMResponse, ToolCall

from .helpers import record
from .test_recovery_matrix import runtime, start


@pytest.mark.parametrize("source", ["producer", "cold"])
def test_retained_json_is_not_parsed_again_and_nested_outputs_are_detached(source, monkeypatch):
    attributes = {"nested": [{"values": [1, 2]}]}
    rec = SessionSpec("s", "p", "w", attributes=attributes).record()
    body = rec.encode()
    if source == "cold":
        loads = records.json.loads
        calls = []

        def parse(*args, **kwargs):
            calls.append(1)
            return loads(*args, **kwargs)

        monkeypatch.setattr(records.json, "loads", parse)
        rec = records.Record.parse(body)
        assert len(calls) == 1

    def unexpected(*args, **kwargs):
        raise AssertionError("retained JSON must not be reparsed")

    monkeypatch.setattr(records.json, "loads", unexpected)
    attributes["nested"][0]["values"].clear()
    rec.payload["attributes"]["nested"][0]["values"].append(3)
    rec.to_dict()["payload"]["attributes"]["nested"].clear()
    assert rec.payload["attributes"] == {"nested": [{"values": [1, 2]}]}
    assert rec.encode() == body


def test_state_snapshot_detaches_waits_and_mutable_state_containers():
    fold = Fold()
    rows = [record(kind) for kind in ("session_created", "turn_started", "op_planned", "op_started", "op_parked")]
    fold.extend(rows)
    snapshot = fold.snapshot()
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
    assert fresh_wait is not None and fresh_wait["handle"]["query_ref"] == "query"
    assert not fresh.turns["t"].cancelled
    assert not fresh.applied_inputs and not fresh.admitted_inputs and not fresh.compactions
    assert rows[-1].payload["handle"]["query_ref"] == "query"


def test_task_cache_and_context_outputs_cannot_mutate_definition(database, monkeypatch):
    rt = runtime(database, [])
    rt.config = replace(rt.config, shared_state={"nested": {"values": [1]}}, metadata={"host": {"values": [2]}})
    definition = rt.definition(rt.compile("go", "t"))
    rec = make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload | {"definition": definition, "definition_digest": digest(definition)},
    )
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
    fold.extend([SessionSpec("s", "p", "w").record(), rec])
    rows = tuple(StoredRecord(r, i, "c", 1, 0) for i, r in enumerate([SessionSpec("s", "p", "w").record(), rec], 1))
    messages = project_context(rows, fold.state)
    messages[0].metadata["host"]["values"].append(3)
    assert rec.task().metadata["host"]["values"] == [2]


def test_schema_plan_and_default_registry_are_reused_across_turns(store, database, monkeypatch):
    from vv_agent.session import runtime as bindings

    calls = {"registry": 0, "schemas": 0}
    build, plan = bindings.build_default_registry, bindings.plan_tool_schemas

    def registry():
        calls["registry"] += 1
        return build()

    def schemas(**kwargs):
        calls["schemas"] += 1
        return plan(**kwargs)

    monkeypatch.setattr(bindings, "build_default_registry", registry)
    monkeypatch.setattr(bindings, "plan_tool_schemas", schemas)
    start(store)
    rt = runtime(database, [LLMResponse("done"), LLMResponse("done")])
    drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        tx.push("s", InboxItem("next", "user", {"content": "again"}))
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
    rt = runtime(database, [response, response, response], tools=[dynamic])
    for i, value in enumerate([False, True, False]):
        enabled["value"] = value
        if i:
            with store.atomic() as tx:
                tx.push("s", InboxItem(str(i), "user", {"content": "go"}))
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
    rt = runtime(database, [response, LLMResponse("done")], tools=[nested])
    rt.config = replace(rt.config, metadata={"nested": {"values": ["original"]}})
    rt.hooks.hooks.append(Hook())
    drive(store, "s", runtime=rt)
    _, rows, _ = store.read_state("s")
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
    definition = rt.definition(rt.compile("go", "t"))
    rec = make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload | {"definition": definition, "definition_digest": digest(definition)},
    )
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
    control = InboxItem("cancel", "control", {"action": "cancel"}, "t", 1)
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

    rt = runtime(database, [LLMResponse("done")], tools=[nested])
    rt.hooks.hooks.append(Hook())
    task = rt.compile("go", "t")
    task.task_id = "t"
    definition = rt.definition(task)
    request = {"messages": [], "tools": definition["tools"]}
    result = {"content": "", "tool_calls": [ToolCall("call", "nested", {"values": ["original"]}).to_dict()]}
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="seed", ttl_ms=15000)
    assert lease is not None
    try:
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="model-receipt",
                records=(
                    record("turn_started", definition=definition, definition_digest=digest(definition)),
                    record("op_planned", oid="t/model/1", request=request, request_digest=digest(request)),
                    record("op_started", oid="t/model/1", epoch=lease.epoch),
                    record(
                        "op_completed",
                        oid="t/model/1",
                        result=result,
                        result_digest=digest(result),
                        request_digest=digest(request),
                    ),
                ),
            )
    finally:
        store.release(lease)
    drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    receipt = next(r.record for r in rows if r.record.kind == "op_completed" and r.record.operation_id == "t/model/1")
    assert effects == [["original", "hook"]]
    assert receipt.payload["result"] == result
    store._fold_cache = None
    cold_state, cold_rows, _ = store.read_state("s")
    assert cold_state == state
    assert [r.record.to_dict() for r in cold_rows] == [r.record.to_dict() for r in rows]


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
    retained = make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload | {"definition": definition, "definition_digest": digest(definition)},
    )
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
    from vv_agent.session import kernel

    completed = []
    make = kernel.make_record

    def counted(kind, **kwargs):
        if kind == "op_completed":
            completed.append(kwargs["operation_id"])
        return make(kind, **kwargs)

    monkeypatch.setattr(kernel, "make_record", counted)
    start(store)
    rt = runtime(database, [LLMResponse("done")])
    drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    assert state.active_turn_id is None
    assert completed == [r.record.operation_id for r in rows if r.record.kind == "op_completed"]
    assert len(completed) == 1


@pytest.mark.parametrize("context", ["normal", "audit"])
def test_preferred_endpoint_is_folded_from_successful_dispatch_and_survives_recovery(context):
    baseline = Fold()
    definition = {"model_binding": {"endpoints": ["a", "b"]}}
    request = {"messages": [], "metadata": {"session_endpoint_order": ["b", "a"]}}
    rows = [
        record("session_created"),
        record("turn_started", definition=definition, definition_digest=digest(definition)),
        record("op_planned", request=request, request_digest=digest(request)),
        record("op_started", endpoint_id="b"),
    ]
    baseline.extend(rows)
    assert baseline.state.preferred_endpoint_id is None
    if context == "audit":
        control = InboxItem("cancel", "control", {"action": "cancel"}, "t", 1)
        applied = record("input_applied", input=control.to_dict(), input_digest=control.digest)
        baseline.extend([applied], consumed_inputs=[control])
        rows.append(applied)
    completed = record("op_completed", request_digest=digest(request), context=context)
    provisional = baseline.fork()
    provisional.extend([completed])
    assert baseline.state.preferred_endpoint_id is None
    assert provisional.snapshot().preferred_endpoint_id == "b"
    recovered = Fold()
    recovered.extend(
        [*rows, completed],
        consumed_inputs=[InboxItem(**r._payload["input"]) for r in rows if r.kind == "input_applied"],
    )
    assert recovered.state == provisional.state


@pytest.mark.parametrize("result", [None, True, 7, "opaque", [], {"error_code": "failed"}])
def test_fold_keeps_opaque_receipts_and_does_not_prefer_failed_endpoints(result):
    definition = {"model_binding": {"endpoints": ["a"]}}
    request = {"messages": [], "metadata": {"session_endpoint_order": ["a"]}}
    baseline = Fold()
    baseline.extend(
        [
            record("session_created"),
            record("turn_started", definition=definition, definition_digest=digest(definition)),
            record("op_planned", request=request, request_digest=digest(request)),
            record("op_started", endpoint_id="a"),
            record("op_completed", request_digest=digest(request), result=result, result_digest=digest(result)),
        ]
    )
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
        retained = make_record(
            "turn_started",
            session_id="s",
            turn_id=task.task_id,
            payload=record("turn_started").payload | {"definition": definition, "definition_digest": digest(definition)},
        )
        assert rt.definition_digest(retained) == digest(retained._payload["definition"])


def test_tool_schemas_share_only_validated_json_across_turns_and_host_copies(store, database):
    start(store)
    rt = runtime(database, [LLMResponse("done") for _ in range(3)], tools=[cached_tool])
    for i in range(3):
        if i:
            with store.atomic() as tx:
                tx.push("s", InboxItem(str(i), "user", {"content": "again"}))
        drive(store, "s", runtime=rt)
    state, rows, _ = store.read_state("s")
    sources = [r.record for r in rows if r.record.kind in {"turn_started", "op_planned"}]
    schemas = [r._payload["definition" if r.kind == "turn_started" else "request"]["tools"] for r in sources]
    assert len(schemas) == 6 and len({id(value) for value in schemas}) == 1
    logical = [r.record.to_dict() for r in rows]
    for r in sources:
        container = "definition" if r.kind == "turn_started" else "request"
        r.payload[container]["tools"].clear()
        r.to_dict()["payload"][container]["tools"][0]["function"]["description"] = "host mutation"
        assert digest(r.to_dict()["payload"][container]) == r._payload[f"{container}_digest"]
    store._fold_cache = None
    cold, reread, _ = store.read_state("s")
    assert cold == state and [r.record.to_dict() for r in reread] == logical


@pytest.mark.parametrize("container", ["definition", "request"])
def test_retained_schema_encoding_matches_full_jcs_and_checks_embedded_digest(tmp_path, container):
    from vv_agent.session.records import Record, RecordError, _RetainedTools

    rt = runtime(tmp_path / "unused.sqlite", [], tools=[cached_tool])
    definition = rt.definition(rt.compile("go", "t"))
    source = make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload=record("turn_started").payload | {"definition": definition, "definition_digest": digest(definition)},
    )
    value = {"tools": _RetainedTools(source), "😀": [-0.0, 1e-27], "\uffff": '"tools":null'}
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
    assert retained.encode() == records.canonical_json_bytes(retained.to_dict())
    assert retained._payload[container]["tools"] is source._payload["definition"]["tools"]
    assert Record.parse(retained.encode()).to_dict() == retained.to_dict()
    with pytest.raises(RecordError, match=f"{container} digest mismatch"):
        make_record(
            kind,
            session_id="s",
            turn_id="t",
            operation_id="o" if container == "request" else None,
            attempt=1 if container == "request" else None,
            payload=payload | {f"{container}_digest": "0" * 64},
        )


def test_retained_schema_source_requires_full_validation_before_reuse():
    from vv_agent.session.records import Record, RecordError, _RetainedTools

    value = record("turn_started").to_dict()
    value["payload"]["definition"] = {"tools": []}
    unvalidated = Record(**value)
    with pytest.raises(RecordError, match="definition digest mismatch"):
        digest({"tools": _RetainedTools(unvalidated)})
