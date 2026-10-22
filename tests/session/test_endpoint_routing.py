"""Current host routing with real kernel dispatch and a local transport double."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.reducer import fold
from vv_agent.session.result import project_result
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import ToolCall

from .test_model_selection import WorkerDeath, echo, plans, switch
from .test_provider_usage import ReceiptProvider
from .test_recovery_matrix import MODEL_USAGE, runtime, start


@pytest.fixture
def transport(monkeypatch):
    calls = []
    replies = []

    def create(*, endpoint_id, **_kwargs):
        def complete(**payload):
            calls.append((payload["model"], endpoint_id))
            response = replies.pop(0)
            if isinstance(response, Exception):
                raise response
            if callable(response):
                response = response()
            return [
                SimpleNamespace(
                    content=response.get("content", ""),
                    tool_calls=[
                        {"id": c["id"], "function": {"name": c["name"], "arguments": "{}"}}
                        for c in response.get("tool_calls", [])
                    ],
                    usage=MODEL_USAGE,
                )
            ]

        return SimpleNamespace(create_completion=complete)

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", create)
    return calls, replies


def client(ids, cls=VvLlmClient):
    return cls([EndpointTarget(e, "unused", "https://example.invalid") for e in ids], backend="openai", randomize_endpoints=False)


def host(database, settings, **kwargs):
    rt = runtime(database, [], [echo], **kwargs)
    rt.resolved = replace(rt.resolved, backend="openai", context_length=8192, max_output_tokens=1024)
    rt.llm = client(settings["scripted"])
    rt.model_resolver = lambda request: (
        client(settings[request["model"]]),
        replace(rt.resolved, model_id=request["model"], requested_model=request["model"], selected_model=request["model"]),
    )
    return rt


def rows(store):
    return [r.record for r in store.read_state("s")[1]]


def cut_plan(point, record):
    if point == "after_commit" and record.kind == "op_planned" and record.payload["purpose"] == "primary":
        raise WorkerDeath


@pytest.mark.parametrize("changed", [["a", "b", "c"], ["b"], ["renamed", "b"]])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("boundary", ["turn", "cycle"])
def test_endpoint_edits_between_turns_and_cycles(store, database, transport, changed, selected, boundary):
    calls, replies = transport
    settings = {"scripted": ["a", "b"], "B": ["a", "b"]}
    start(store)
    if selected:
        switch(store, {"model": "B", "route": {}})

    def edit():
        settings["B" if selected else "scripted"] = changed
        return {"tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}

    replies.extend([edit if boundary == "cycle" else {"content": "first"}, {"content": "done"}])
    rt = host(database, settings)
    drive(store, "s", runtime=rt)
    if boundary == "turn":
        settings["B" if selected else "scripted"] = changed
        with store.atomic() as tx:
            tx.push("s", InboxItem("next", "user", {"content": "next"}))
        drive(store, "s", runtime=host(database, settings))
    assert [e for _, e in calls] == ["a", "a" if "a" in changed else changed[0]]
    assert project_result(store, "s", plans(store)[-1].turn_id).final_output == "done"
    assert all(r.payload["reason"] != "handler_schema_or_capability_mismatch" for r in rows(store) if r.kind == "turn_ended")
    assert plans(store)[-1].payload["request"]["metadata"]["vv_session"]["endpoint_order"] == (
        changed if "a" not in changed else ["a", *[e for e in changed if e != "a"]]
    )


@pytest.mark.parametrize("retry", [False, True])
def test_removed_unstarted_endpoint_is_definitive_and_replans(store, database, transport, retry):
    calls, replies = transport
    settings = {"scripted": ["a", "b"]}
    start(store)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.attempt == (2 if retry else 1):
            raise WorkerDeath

    if retry:
        replies.append(TimeoutError("receipt lost"))
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=host(database, settings, hook=cut))
    settings["scripted"] = ["c"]
    replies.append({"content": "done"})
    drive(store, "s", runtime=host(database, settings))
    assert calls == ([("scripted", "a")] if retry else []) + [("scripted", "c")]
    unavailable = next(
        r for r in rows(store) if r.kind == "op_completed" and r.payload["result"].get("error_code") == "endpoint_unavailable"
    )
    assert (
        unavailable.payload["execution_started"] is False
        and unavailable.payload["usage"] == {}
        and unavailable.payload["evidence"] == []
    )
    assert not any(
        r.kind in {"op_started", "op_unknown"} and r.operation_id == unavailable.operation_id and r.attempt == unavailable.attempt
        for r in rows(store)
    )
    assert project_result(store, "s", plans(store)[-1].turn_id).final_output == "done"


def test_host_ordering_wins_over_previous_models_preference(store, database, transport):
    calls, replies = transport
    start(store)
    settings = {"scripted": ["B"], "other": ["A", "B"]}
    preferences = []

    class HostOrder(VvLlmClient):
        def ordered_targets(self, preferred_endpoint_id=None):
            if len(self.endpoint_targets) > 1:
                preferences.append(preferred_endpoint_id)
            return list(self.endpoint_targets)

    rt = host(database, settings)
    resolve = rt.model_resolver

    def resolver(request):
        _current, config = resolve(request)
        return client(settings[request["model"]], HostOrder), config

    rt.model_resolver = resolver

    def change():
        switch(store, {"model": "other", "route": {}})
        return {"tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}

    replies.extend([change, {"content": "done"}])
    drive(store, "s", runtime=rt)
    assert calls == [("scripted", "B"), ("other", "A")]
    assert preferences == ["B"]
    assert plans(store)[-1].payload["request"]["metadata"]["vv_session"]["endpoint_order"] == ["A", "B"]


@pytest.mark.parametrize("ids", [["a"], ["a", "b"], ["a", "b", "c"]])
def test_unknown_retry_limit_is_unchanged(store, database, transport, ids):
    calls, replies = transport
    start(store)
    replies.extend(TimeoutError("receipt lost") for _ in range(max(2, len(ids))))
    drive(store, "s", runtime=host(database, {"scripted": ids}))
    assert calls == [("scripted", ids[i % len(ids)]) for i in range(max(2, len(ids)))]
    unknowns = [r for r in rows(store) if r.kind == "op_unknown"]
    assert [r.payload["retry"] for r in unknowns] == ["retry"] * (max(2, len(ids)) - 1) + ["stop"]
    assert all(r.payload["duplicate_cost_risk"] for r in unknowns)


@pytest.mark.parametrize("cut", ["before_external_call", "after_external_call"])
@pytest.mark.parametrize("receipt", [True, False])
def test_dispatched_removed_endpoint_retains_evidence(store, database, transport, cut, receipt):
    calls, replies = transport
    start(store)
    settings = {"scripted": ["a", "b"]}
    provider = ReceiptProvider({"content": "authentic", "tool_calls": []}, MODEL_USAGE)

    def die(point, record):
        if point == cut:
            provider.bind(record)
            raise WorkerDeath

    replies.append({"content": "lost"})
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=host(database, settings, hook=die))
    original = next(r for r in rows(store) if r.kind == "op_started")
    settings["scripted"] = ["b"]
    if receipt:
        with store.atomic() as tx:
            tx.push("s", provider.item)
    else:
        replies[:] = [{"content": "retried"}]
    drive(store, "s", runtime=host(database, settings, providers={"model": provider}))
    assert calls == ([] if cut == "before_external_call" else [("scripted", "a")]) + ([] if receipt else [("scripted", "b")])
    assert next(r for r in rows(store) if r.kind == "op_started").encode() == original.encode()
    assert original.payload["endpoint_id"] == "a"
    result = next(r for r in rows(store) if r.kind == "op_completed")
    assert result.payload["execution_started"] and result.payload["usage"] == MODEL_USAGE
    assert result.payload["evidence"]
    unknowns = [r for r in rows(store) if r.kind == "op_unknown"]
    if receipt:
        assert not unknowns
    else:
        assert len(unknowns) == 1 and unknowns[0].payload["dispatch_evidence"] == [original.record_id]
        assert unknowns[0].payload["duplicate_cost_risk"]
    assert project_result(store, "s", plans(store)[0].turn_id).final_output == ("authentic" if receipt else "retried")


@pytest.mark.parametrize("field,value", [("backend", "other"), ("context_length", 4096), ("max_output_tokens", 512)])
def test_logical_model_drift_still_rejects(tmp_path, field, value):
    rt = host(tmp_path / "unused.sqlite", {"scripted": ["a"]})
    task = rt.compile("go", "t")
    definition = rt.definition_digest(task)
    binding = rt.model_binding(rt.llm, rt.resolved)
    assert binding["endpoints"] == ["a"]
    assert "endpoints" not in rt.definition(task)["model_binding"]
    request = {"model": "scripted", "route": {}, "metadata": {"vv_session": {"model_binding": binding | {"endpoints": ["old"]}}}}
    assert rt.resolve_model(request)[1] == rt.resolved
    rt.resolved = replace(rt.resolved, **{field: value})
    assert rt.definition_digest(task) != definition
    with pytest.raises(ValueError, match="binding differs"):
        rt.resolve_model(request)


def test_old_definition_record_bytes_load_and_continue(tmp_path, transport):
    """Use pre-change canonical producer records, never rewrite their digests."""
    from pathlib import Path

    source = Path(__file__).with_name("fixtures") / "routing_025_records.jsonl"
    saved = [Record.parse(line) for line in source.read_bytes().splitlines()]
    # This fixture ends immediately after the old producer froze the first plan.
    consumed = [InboxItem(**r.payload["input"]) for r in saved if r.kind == "input_applied"]
    state = fold(saved, consumed_inputs=consumed)
    assert state.active_turn_id is not None
    assert state.turns[state.active_turn_id].start.payload["definition"]["model_binding"]["endpoints"] == ["a", "b"]
    with SQLiteStore.standalone(str(tmp_path / "old.sqlite")) as store:
        store.install_schema()
        from vv_agent.session.records import SessionSpec

        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", "/tmp"), consumers=("events",))
            tx.push("s", consumed[0])
        # Import unchanged record bytes through the store's append boundary.
        poll = store.acquire("s", owner="import", ttl_ms=15000)
        assert poll is not None
        with store.atomic() as tx:
            tx.append(
                "s",
                records=tuple(saved[1:]),
                lease=poll,
                expected_seq=1,
                commit_id="import",
                consume_input_ids=(consumed[0].input_id,),
            )
        store.release(poll)
        calls, replies = transport
        replies.append({"content": "continued"})
        drive(store, "s", runtime=host(tmp_path / "old.sqlite", {"scripted": ["b"]}))
        assert calls == [("scripted", "b")]
        assert project_result(store, "s", plans(store)[-1].turn_id).final_output == "continued"


def test_removed_endpoint_does_not_create_missing_budget_usage(store, database, transport):
    from vv_agent.budget import RunBudgetLimits, UnavailableMetricPolicy

    settings = {"scripted": ["a", "b"]}
    start(store)
    rt = host(database, settings, hook=cut_plan)
    limits = RunBudgetLimits(max_total_tokens=100000, unavailable_metric_policy=UnavailableMetricPolicy.STOP)
    rt.config = replace(rt.config, budget_limits=limits)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=rt)
    settings["scripted"] = ["b"]
    fresh = host(database, settings)
    fresh.config = replace(fresh.config, budget_limits=limits)
    calls, replies = transport
    replies.append({"content": "done"})
    drive(store, "s", runtime=fresh)
    assert calls == [("scripted", "b")]
    result = project_result(store, "s", plans(store)[0].turn_id)
    assert result.final_output == "done"
    assert len(result.token_usage.model_calls) == 1
    assert result.token_usage.total_tokens == 12300
    assert all(
        not r.payload["data"]["usage"]["unavailable_dimensions"]
        for r in rows(store)
        if r.kind == "boundary_recorded" and r.payload["stage"] == "budget"
    )
    assert not any(
        r.kind == "boundary_recorded" and r.payload["stage"] == "budget" and r.payload["data"]["exhaustion"] for r in rows(store)
    )


def test_removed_repair_endpoint_does_not_invoke_callback(store, database, transport):
    from vv_agent.output_validation import OutputValidationResult

    settings = {"scripted": ["a"]}
    repairs = []
    rt = host(database, settings)
    rt.agent.output_validation_enabled = True
    rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid", "bad answer")
    rt.agent.output_repair = lambda request: repairs.append(request) or "valid"

    def remove(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.payload["purpose"] == "output_repair":
            settings["scripted"] = ["b"]

    rt.hook = remove
    start(store)
    calls, replies = transport
    replies.append({"content": "invalid"})
    drive(store, "s", runtime=rt)
    assert calls == [("scripted", "a")] and not repairs
    missing = next(
        r for r in rows(store) if r.kind == "op_completed" and r.payload["result"].get("error_code") == "endpoint_unavailable"
    )
    assert not missing.payload["execution_started"] and missing.payload["usage"] == {}
    assert not any(r.kind == "op_unknown" for r in rows(store))
    result = project_result(store, "s", plans(store)[0].turn_id)
    assert result.status == "failed" and result.completion_reason == "failed"
    assert next(r for r in rows(store) if r.kind == "turn_ended").payload["reason"] == "output_validation_failed"
