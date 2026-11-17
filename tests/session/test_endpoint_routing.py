"""Current host routing with real kernel dispatch and a local transport double."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vv_agent.kernel._boundary_hydration import hydrate_session_input
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session.kernel import drive
from vv_agent.session.projection import hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.result import project_result
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.function import function_tool
from vv_agent.types import ToolCall

from .conftest import open_store
from .helpers import (
    create_fixture_session,
    fixture_content_reader,
    push_fixture_input,
    read_fixture_state,
    wrap_fixture_store,
)
from .test_model_selection import WorkerDeath, echo, plans
from .test_provider_usage import ReceiptProvider
from .test_recovery_matrix import MODEL_USAGE, runtime


@pytest.fixture
def transport(monkeypatch):
    calls = []
    replies = []

    def create(*, endpoint_id, **_kwargs):
        def complete(**payload):
            calls.append((payload["model"], endpoint_id))
            response = replies.pop(0)
            if isinstance(response, Exception):
                from .test_model_rejection import adapter_error

                raise adapter_error(response)
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


def start(store, sid="s", *, requirements=None, initial=True):
    content_store = wrap_fixture_store(store)
    create_fixture_session(content_store, sid, consumers=("events",))
    if initial:
        push_initial(content_store, sid, requirements)
    return content_store


def push_initial(store, sid="s", requirements=None):
    if requirements is not None:
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    push_fixture_input(
        store,
        sid,
        InboxItem(
            "initial",
            "user",
            {"content": "go"},
            feature_level=2 if requirements is not None else 1,
            requirements=requirements.to_dict() if requirements is not None else None,
        ),
    )


def switch(store, runtime, selection, input_id="select", sid="s"):
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    requirements = runtime.model_selection_requirements(selection)
    item = InboxItem(
        input_id,
        "model_selection",
        {"selection": selection},
        feature_level=2,
        requirements=requirements.to_dict(),
    )
    return push_fixture_input(store, sid, item)


def host(database, settings, fixture_store=None, **kwargs):
    if fixture_store is not None:
        kwargs.setdefault("content_reader", fixture_content_reader(fixture_store, "s"))
    rt = runtime(database, [], [echo], **kwargs)
    rt.resolved = replace(rt.resolved, backend="openai", context_length=8192, max_output_tokens=1024)
    rt.llm = client(settings["scripted"])
    rt.model_resolver = lambda request: (
        client(settings[request["model"]]),
        replace(rt.resolved, model_id=request["model"], requested_model=request["model"], selected_model=request["model"]),
    )
    if fixture_store is not None:
        capabilities = set(rt.requirements_bound().capabilities)
        capabilities.update(
            capability
            for model in settings
            for capability in rt.model_selection_requirements({"model": model, "route": {}}).capabilities
        )
        fixture_store.configure_worker(
            level=2,
            handler_versions=(rt.handler_version,),
            capabilities=tuple(sorted(capabilities)),
        )

        @contextmanager
        def heartbeat_store():
            with open_store(database) as store:
                store.configure_worker(
                    level=2,
                    handler_versions=(rt.handler_version,),
                    capabilities=tuple(sorted(capabilities)),
                )
                yield store

        rt.heartbeat_store = heartbeat_store
    return rt


def rows(store):
    return [r.record for r in store.read_state("s")[1]]


def hydrated_rows(store):
    _, hydrated, _ = read_fixture_state(store, "s")
    return [row.record for row in hydrated]


def hydrated_input(store, input_id):
    hydration = hydration_for_session_store(store, "s")
    stored = next(row for row in store.peek_inbox("s") if row.item.input_id == input_id)
    return hydrate_session_input(stored.item, hydration, include_actor_attributes=True)


def cut_plan(point, record):
    if point == "after_commit" and record.kind == "op_planned" and record.payload["purpose"] == "primary":
        raise WorkerDeath


@pytest.mark.parametrize("changed", [["a", "b", "c"], ["b"], ["renamed", "b"]])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("boundary", ["turn", "cycle"])
def test_endpoint_edits_between_turns_and_cycles(store, database, transport, changed, selected, boundary):
    calls, replies = transport
    settings = {"scripted": ["a", "b"], "B": ["a", "b"]}
    store = start(store, initial=False)
    selection = {"model": "B", "route": {}}
    requirements = None
    requirement_runtime = None
    if selected:
        requirement_runtime = host(database, settings, store)
        selection_requirements = requirement_runtime.model_selection_requirements(selection)
        runtime_requirements = requirement_runtime.requirements_bound()
        requirements = SessionRequirements.normalized(
            level=max(selection_requirements.level, runtime_requirements.level),
            handler_version=runtime_requirements.handler_version,
            capabilities=(*selection_requirements.capabilities, *runtime_requirements.capabilities),
        )
    push_initial(store, requirements=requirements)
    if requirement_runtime is not None:
        switch(store, requirement_runtime, selection)
    rt = host(database, settings, store)

    def edit():
        settings["B" if selected else "scripted"] = changed
        return {"tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}

    replies.extend([edit if boundary == "cycle" else {"content": "first"}, {"content": "done"}])
    drive(store, "s", runtime=rt)
    if boundary == "turn":
        settings["B" if selected else "scripted"] = changed
        push_fixture_input(
            store,
            "s",
            InboxItem(
                "next",
                "user",
                {"content": "next"},
                feature_level=2 if requirements is not None else 1,
                requirements=requirements.to_dict() if requirements is not None else None,
            ),
        )
        drive(store, "s", runtime=host(database, settings, store))
    assert [e for _, e in calls] == ["a", "a" if "a" in changed else changed[0]]
    assert project_result(store, "s", plans(store)[-1].turn_id).final_output == "done"
    assert all(
        r.payload["reason"] != "handler_schema_or_capability_mismatch" for r in hydrated_rows(store) if r.kind == "turn_ended"
    )
    assert plans(store)[-1].payload["request"]["metadata"]["vv_session"]["endpoint_order"] == (
        changed if "a" not in changed else ["a", *[e for e in changed if e != "a"]]
    )


@pytest.mark.parametrize("retry", [False, True])
def test_removed_unstarted_endpoint_is_definitive_and_replans(store, database, transport, retry):
    calls, replies = transport
    settings = {"scripted": ["a", "b"]}
    store = start(store)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.attempt == (2 if retry else 1):
            raise WorkerDeath

    if retry:
        replies.append(TimeoutError("receipt lost"))
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=host(database, settings, store, hook=cut))
    settings["scripted"] = ["c"]
    replies.append({"content": "done"})
    drive(store, "s", runtime=host(database, settings, store))
    assert calls == ([("scripted", "a")] if retry else []) + [("scripted", "c")]
    unavailable = next(
        r
        for r in hydrated_rows(store)
        if r.kind == "op_completed" and r.payload["result"].get("error_code") == "endpoint_unavailable"
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
    store = start(store)
    settings = {"scripted": ["B"], "other": ["A", "B"]}
    preferences = []

    class HostOrder(VvLlmClient):
        def ordered_targets(self, preferred_endpoint_id=None):
            if len(self.endpoint_targets) > 1:
                preferences.append(preferred_endpoint_id)
            return list(self.endpoint_targets)

    rt = host(database, settings, store)
    resolve = rt.model_resolver

    def resolver(request):
        _current, config = resolve(request)
        return client(settings[request["model"]], HostOrder), config

    rt.model_resolver = resolver

    def change():
        switch(store, rt, {"model": "other", "route": {}})
        return {"tool_calls": [ToolCall("echo", "echo", {}).to_dict()]}

    replies.extend([change, {"content": "done"}])
    drive(store, "s", runtime=rt)
    assert calls == [("scripted", "B"), ("other", "A")]
    assert preferences == ["B"]
    assert plans(store)[-1].payload["request"]["metadata"]["vv_session"]["endpoint_order"] == ["A", "B"]


@pytest.mark.parametrize("ids", [["a"], ["a", "b"], ["a", "b", "c"]])
def test_unknown_retry_limit_is_unchanged(store, database, transport, ids):
    calls, replies = transport
    store = start(store)
    replies.extend(TimeoutError("receipt lost") for _ in range(max(2, len(ids))))
    drive(store, "s", runtime=host(database, {"scripted": ids}, store))
    assert calls == [("scripted", ids[i % len(ids)]) for i in range(max(2, len(ids)))]
    unknowns = [r for r in rows(store) if r.kind == "op_unknown"]
    assert [r.payload["retry"] for r in unknowns] == ["retry"] * (max(2, len(ids)) - 1) + ["stop"]
    assert all(r.payload["duplicate_cost_risk"] for r in unknowns)


@pytest.mark.parametrize("ids", [["a"], ["a", "b", "c"]])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 422, 429, 499, 500])
def test_rejection_endpoint_policy_is_distinct_from_unknown_retry(store, database, transport, monkeypatch, status, ids):
    from vv_agent.llm.errors import ModelRequestRejected

    from .test_model_rejection import http_error

    monkeypatch.setattr("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0)
    calls, replies = transport
    store = start(store)
    error = ModelRequestRejected(409, "Temporary endpoint refusal", retryable=True) if status == 409 else http_error(status)
    replies.extend([error, {"content": "done"}])
    drive(store, "s", runtime=host(database, {"scripted": ids}, store))
    retry = status in {409, 429, 500} or (status in {401, 403, 404} and len(ids) > 1)
    assert calls == [("scripted", "a")] + ([("scripted", ids[1 % len(ids)])] if retry else [])
    unknowns = [r for r in rows(store) if r.kind == "op_unknown"]
    assert len(unknowns) == int(status == 500)
    terminal = next(r for r in hydrated_rows(store) if r.kind == "turn_ended").payload
    assert terminal["status"] == ("completed" if retry else "failed")
    if not retry:
        assert terminal["reason"] == "model_request_rejected"
    if status != 500:
        assert not terminal["unconfirmed_operations"]
        assert not any(
            e.type == "model_retry_duplicate_risk"
            for e in project_records(
                store.read_state("s")[1],
                hydration=hydration_for_session_store(store, "s"),
            )
        )


@pytest.mark.parametrize("ids", [["a"], ["a", "b"], ["a", "b", "c"]])
@pytest.mark.parametrize("status", [401, 403, 404])
def test_endpoint_rejections_exhaust_without_backoff_and_keep_last_safe_error(store, database, transport, status, ids):
    from .test_model_rejection import http_error

    calls, replies = transport
    store = start(store)
    replies.extend(http_error(status, f"Denied at {endpoint}. Authorization: Bearer sk-secret") for endpoint in ids)
    drive(store, "s", runtime=host(database, {"scripted": ids}, store))
    assert calls == [("scripted", endpoint) for endpoint in ids]
    assert [p.attempt for p in plans(store)] == list(range(1, len(ids) + 1))
    assert all(p.payload["not_before_ms"] == 0 for p in plans(store) if p.attempt > 1)
    assert not any(r.kind == "op_unknown" for r in rows(store))
    terminal = next(r for r in hydrated_rows(store) if r.kind == "turn_ended").payload
    assert terminal["status"] == "failed" and terminal["reason"] == "model_request_rejected"
    assert terminal["unconfirmed_operations"] == []
    assert f"HTTP {status}: Model provider request failed." in terminal["result"]
    assert "sk-secret" not in str(rows(store))
    assert not any(
        e.type == "model_retry_duplicate_risk"
        for e in project_records(
            store.read_state("s")[1],
            hydration=hydration_for_session_store(store, "s"),
        )
    )


@pytest.mark.parametrize(
    "status,retryable,ids,retry",
    [
        (401, False, ["a", "b"], False),
        (429, False, ["a", "b"], False),
        (400, True, ["a", "b"], True),
        (401, True, ["a"], True),
        (401, None, ["a", "b"], True),
        (401, None, ["a"], False),
    ],
)
def test_host_rejection_retry_override_wins(store, database, transport, monkeypatch, status, retryable, ids, retry):
    from vv_agent.llm.errors import ModelRequestRejected

    monkeypatch.setattr("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0)
    calls, replies = transport
    store = start(store)
    replies.extend([ModelRequestRejected(status, "Rejected", retryable=retryable), {"content": "done"}])
    drive(store, "s", runtime=host(database, {"scripted": ids}, store))
    assert calls == [("scripted", "a")] + ([("scripted", ids[1 % len(ids)])] if retry else [])
    assert not any(r.kind == "op_unknown" for r in rows(store))
    terminal = next(r for r in hydrated_rows(store) if r.kind == "turn_ended").payload
    assert terminal["status"] == ("completed" if retry else "failed")


@pytest.mark.parametrize("cut", ["before_external_call", "after_external_call"])
@pytest.mark.parametrize("receipt", [True, False])
def test_dispatched_removed_endpoint_retains_evidence(store, database, transport, cut, receipt):
    calls, replies = transport
    store = start(store)
    settings = {"scripted": ["a", "b"]}

    class LogicalReceiptProvider(ReceiptProvider):
        def authenticate(self, item, plan):
            return (
                item.kind == self.item.kind
                and item.target_turn_id == self.item.target_turn_id
                and item.generation == self.item.generation
                and item.payload == self.item.payload
                and item.payload["operation_id"] == plan.operation_id
                and item.payload["attempt"] == plan.attempt
                and item.payload["request_digest"] == plan.payload["request_digest"]
            )

    provider = LogicalReceiptProvider({"content": "authentic", "tool_calls": []}, MODEL_USAGE)

    def die(point, record):
        if point == cut:
            provider.bind(record)
            raise WorkerDeath

    replies.append({"content": "lost"})
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=host(database, settings, store, hook=die))
    original = next(r for r in rows(store) if r.kind == "op_started")
    settings["scripted"] = ["b"]
    if receipt:
        push_fixture_input(store, "s", provider.item)
        provider.item = hydrated_input(store, provider.item.input_id)
    else:
        replies[:] = [{"content": "retried"}]
    drive(store, "s", runtime=host(database, settings, store, providers={"model": provider}))
    assert calls == ([] if cut == "before_external_call" else [("scripted", "a")]) + ([] if receipt else [("scripted", "b")])
    assert next(r for r in rows(store) if r.kind == "op_started").encode() == original.encode()
    assert original.payload["endpoint_id"] == "a"
    result = next(r for r in hydrated_rows(store) if r.kind == "op_completed")
    assert result.payload["execution_started"] and result.payload["usage"] == MODEL_USAGE
    assert result.payload["evidence"]
    unknowns = [r for r in hydrated_rows(store) if r.kind == "op_unknown"]
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


@pytest.mark.parametrize("suspended", [False, True])
def test_frozen_tool_schema_drift_rejects_old_definition_before_dispatch(tmp_path, transport, suspended):
    """Old records stay decodable; a changed frozen tool schema cannot resume."""

    @function_tool(description="legacy frozen schema")
    def old_echo() -> str:
        return "old"

    calls, replies = transport
    replies.append({"content": "continued"})
    path = tmp_path / "old.sqlite"
    with SQLiteStore.standalone(str(path)) as raw_store:
        raw_store.install_schema()
        store = wrap_fixture_store(raw_store)
        create_fixture_session(store, "s", consumers=("events",))
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
        old_runtime = host(
            path,
            {"scripted": ["a", "b"]},
            hook=cut_plan,
            content_reader=fixture_content_reader(store, "s"),
        )
        old_runtime.agent.tools = [old_echo]
        with pytest.raises(WorkerDeath):
            drive(store, "s", runtime=old_runtime)
        state, _, _ = store.read_state("s")
        assert state.active_turn_id is not None
        if suspended:
            push_fixture_input(
                store,
                "s",
                InboxItem("suspend", "control", {"action": "suspend"}, state.active_turn_id),
            )
            drive(store, "s", runtime=host(path, {"scripted": ["b"]}, content_reader=fixture_content_reader(store, "s")))
            assert store.read_state("s")[0].turns[state.active_turn_id].suspended
            assert not any(record.kind == "turn_ended" for record in rows(store))
            push_fixture_input(
                store,
                "s",
                InboxItem("resume", "control", {"action": "resume"}, state.active_turn_id),
            )
        before = [record.encode() for record in rows(store)]
        drive(store, "s", runtime=host(path, {"scripted": ["b"]}, content_reader=fixture_content_reader(store, "s")))
        assert calls == []
        assert [record.encode() for record in rows(store)[: len(before)]] == before
        _, hydrated, _ = read_fixture_state(store, "s")
        ended = next(row.record for row in hydrated if row.record.kind == "turn_ended")
        assert ended._payload["reason"] == "handler_schema_or_capability_mismatch"
        assert project_result(store, "s", state.active_turn_id).status == "failed"


def test_removed_endpoint_does_not_create_missing_budget_usage(store, database, transport):
    from vv_agent.budget import RunBudgetLimits, UnavailableMetricPolicy

    settings = {"scripted": ["a", "b"]}
    store = start(store)
    rt = host(database, settings, store, hook=cut_plan)
    limits = RunBudgetLimits(max_total_tokens=100000, unavailable_metric_policy=UnavailableMetricPolicy.STOP)
    rt.config = replace(rt.config, budget_limits=limits)
    with pytest.raises(WorkerDeath):
        drive(store, "s", runtime=rt)
    settings["scripted"] = ["b"]
    fresh = host(database, settings, store)
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
        for r in hydrated_rows(store)
        if r.kind == "boundary_recorded" and r.payload["stage"] == "budget"
    )
    assert not any(
        r.kind == "boundary_recorded" and r.payload["stage"] == "budget" and r.payload["data"]["exhaustion"]
        for r in hydrated_rows(store)
    )


def test_removed_repair_endpoint_does_not_invoke_callback(store, database, transport):
    from vv_agent.output_validation import OutputValidationResult

    settings = {"scripted": ["a"]}
    repairs = []
    store = start(store)
    rt = host(database, settings, store)
    rt.agent.output_validation_enabled = True
    rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid", "bad answer")
    rt.agent.output_repair = lambda request: repairs.append(request) or "valid"

    def remove(point, record):
        if (
            point == "after_commit"
            and record is not None
            and record.kind == "op_planned"
            and record.payload["purpose"] == "output_repair"
        ):
            settings["scripted"] = ["b"]

    rt.hook = remove
    calls, replies = transport
    replies.append({"content": "invalid"})
    drive(store, "s", runtime=rt)
    assert calls == [("scripted", "a")] and not repairs
    missing = next(
        r
        for r in hydrated_rows(store)
        if r.kind == "op_completed" and r.payload["result"].get("error_code") == "endpoint_unavailable"
    )
    assert not missing.payload["execution_started"] and missing.payload["usage"] == {}
    assert not any(r.kind == "op_unknown" for r in rows(store))
    result = project_result(store, "s", plans(store)[0].turn_id)
    assert result.status == "failed" and result.completion_reason == "failed"
    assert next(r for r in hydrated_rows(store) if r.kind == "turn_ended").payload["reason"] == "output_validation_failed"
