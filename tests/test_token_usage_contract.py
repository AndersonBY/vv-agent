from __future__ import annotations

import base64
import hmac
import json
from typing import Any

import pytest
from support.contract import fixture_root
from support.s1_capture import CapturedContent

from vv_agent.core_types import AgentTask, canonical_json_bytes
from vv_agent.kernel._boundary_hydration import SessionHydration
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.session.projection import hydrate_records
from vv_agent.session.records import Record
from vv_agent.session.store import StoredRecord
from vv_agent.types import (
    CacheUsage,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    TaskTokenUsage,
    TokenUsage,
    UsageSource,
)


def _contract() -> dict[str, Any]:
    return json.loads((fixture_root() / "token_usage.json").read_text(encoding="utf-8"))


def _compaction_sources(reference: str, session_id: str) -> dict[str, list[dict[str, Any]]]:
    fixture, fragment = reference.split("#", 1)
    assert fixture in {"session_projection.json", "session_projection_s1.json"}
    assert fragment == f"/source_records/{session_id}"
    root = fixture_root()
    source = json.loads((root / fixture).read_text(encoding="utf-8"))["source_records"]
    session_ids = {session_id, "summary"}
    # Contract 41 still points to the immutable Contract 39 fixture during S1 authoring.
    if fixture == "session_projection.json":
        catalog = json.loads((root / "session_content_catalog.json").read_text(encoding="utf-8"))
        keys = {entry["key_id"]: base64.b64decode(entry["key_base64"], validate=True) for entry in catalog["keys"]}
        fields = {"turn_started": "definition", "op_planned": "request", "op_completed": "result"}
        historical_records = {sid: [row["wire"] for row in source[sid]] for sid in session_ids}
        for sid, wires in historical_records.items():
            for wire in wires:
                field = fields.get(wire["kind"])
                if field is not None:
                    ref = wire["payload"][field]
                    entry = catalog["contents"][ref["content_id"]]
                    assert (entry["session_id"], entry["key_id"], entry["boundary"]) == (sid, ref["key_id"], field)
                    body = canonical_json_bytes(entry["value"], field)
                    assert len(body) == ref["size"] and hmac.digest(keys[ref["key_id"]], body, "sha256").hex() == ref["mac"]
                    wire["payload"][field] = entry["value"]
        return historical_records

    catalog = json.loads((root / "session_content_capture_s1.json").read_text(encoding="utf-8"))
    reader = CapturedContent(catalog)
    result = {}
    for sid in session_ids:
        records = tuple(
            StoredRecord(Record(**row["wire"]), row["seq"], row["commit_id"], row["writer_epoch"], row["created_ms"])
            for row in source[sid]
        )
        hydration = SessionHydration(reader, tenant_id="default", session_id=sid)
        result[sid] = [row.record.to_dict() for row in hydrate_records(records, hydration)]
    return result


def test_normalization_matches_canonical_token_usage_cases() -> None:
    for case in _contract()["normalization_cases"]:
        inputs = case["input"]
        usage = normalize_token_usage(
            inputs["raw_usage"],
            usage_source=inputs["usage_source_hint"],
            cache_status=inputs["cache_status_hint"],
        )

        assert usage.to_dict() == case["expected"], case["name"]


def test_aggregation_matches_canonical_cache_observation_cases() -> None:
    for case in _contract()["aggregation_cases"]:
        summary = TaskTokenUsage()
        for cycle_index, observation in enumerate(case["model_calls"], start=1):
            summary.add_model_call(
                ModelCallRecord(
                    call_id=f"op_model_cycle_{cycle_index}_main:attempt:1",
                    operation_id=f"op_model_cycle_{cycle_index}_main",
                    attempt=1,
                    operation=ModelCallOperation.AGENT_CYCLE,
                    cycle_index=cycle_index,
                    backend="test",
                    model="test-model",
                    status=ModelCallStatus.COMPLETED,
                    usage=TokenUsage(
                        total_tokens=1,
                        usage_source=UsageSource.PROVIDER_REPORTED,
                        cache_usage=CacheUsage.from_dict(observation),
                    ),
                    error_code=None,
                )
            )

        assert summary.cache_usage.to_dict() == case["expected"], case["name"]


def test_task_aggregation_uses_complete_model_call_ledger() -> None:
    cases = _contract()["task_aggregation_cases"]
    empty = TaskTokenUsage()
    assert empty.to_dict() == {
        "schema_version": "vv-agent.task-token-usage.v3",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cache_usage": cases[0]["expected"]["cache_usage"],
        "model_calls": [],
    }

    summary = TaskTokenUsage()
    for payload in cases[1]["model_calls"]:
        summary.add_model_call(ModelCallRecord.from_dict(payload))
    expected = cases[1]["expected"]
    assert summary.input_tokens == expected["input_tokens"]
    assert summary.output_tokens == expected["output_tokens"]
    assert summary.total_tokens == expected["total_tokens"]
    assert summary.reasoning_tokens == expected["reasoning_tokens"]
    assert summary.cache_usage.to_dict() == expected["cache_usage"]
    assert len(summary.model_calls) == expected["model_call_count"]
    assert TaskTokenUsage.from_dict(summary.to_dict()) == summary


def test_duplicate_model_call_ids_and_superseded_task_wire_are_rejected() -> None:
    payload = _contract()["task_aggregation_cases"][1]["model_calls"][0]
    record = ModelCallRecord.from_dict(payload)
    summary = TaskTokenUsage()
    summary.add_model_call(record)
    with pytest.raises(ValueError, match="model_call_id_duplicate"):
        summary.add_model_call(record)

    superseded = summary.to_dict()
    superseded["schema_version"] = "vv-agent.task-token-usage.v1"
    with pytest.raises(ValueError, match="unsupported TaskTokenUsage schema"):
        TaskTokenUsage.from_dict(superseded)


def test_current_model_call_and_task_usage_support_output_repair():
    payload = _contract()["task_aggregation_cases"][1]["model_calls"][0] | {"operation": "output_repair"}
    record = ModelCallRecord.from_dict(payload)
    assert record.operation is ModelCallOperation.OUTPUT_REPAIR
    summary = TaskTokenUsage()
    summary.add_model_call(record)
    assert TaskTokenUsage.from_dict(summary.to_dict()) == summary
    with pytest.raises(ValueError, match="unsupported"):
        ModelCallRecord.from_dict(payload | {"schema_version": "vv-agent.model-call.v1"})


def test_explicit_zero_usage_is_observable_and_superseded_wire_is_rejected() -> None:
    explicit_zero = normalize_token_usage(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    assert explicit_zero.has_usage() is True
    assert explicit_zero.cache_usage.read_input_tokens == 0
    with pytest.raises(ValueError, match="invalid TokenUsage fields"):
        TokenUsage.from_dict({"cached_tokens": 0})


def test_native_cache_write_usage_is_normalized_without_public_aliases() -> None:
    usage = normalize_token_usage(
        {
            "input_tokens": 300,
            "output_tokens": 50,
            "cache_read_input_tokens": 600,
            "cache_write_input_tokens": 100,
        }
    )

    assert usage.input_tokens == 1000
    assert usage.cache_usage.write_input_tokens == 100
    assert usage.cache_usage.uncached_input_tokens == 400


@pytest.mark.parametrize(
    "provider_usage",
    [
        # vv-llm Usage.model_dump(exclude_none=True) for Moonshot kimi-k3.
        {
            "completion_tokens": 91,
            "prompt_tokens": 10617,
            "total_tokens": 10708,
            "cache_creation_tokens": 10496,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 10496},
        },
        # The provider shape before vv-llm promotes the nested write count.
        {
            "completion_tokens": 91,
            "prompt_tokens": 10617,
            "total_tokens": 10708,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 10496},
        },
    ],
    ids=["vv-llm-normalized", "nested-only"],
)
def test_openai_compatible_cache_write_usage_is_provider_reported(provider_usage: dict[str, Any]) -> None:
    usage = normalize_token_usage(provider_usage)

    assert usage.input_tokens == 10617
    assert usage.total_tokens == 10708
    assert usage.cache_usage.status.value == "provider_reported"
    assert usage.cache_usage.read_input_tokens == 0
    assert usage.cache_usage.write_input_tokens == 10496
    assert usage.cache_usage.uncached_input_tokens == 10617


def test_vv_llm_cache_creation_is_not_added_to_inclusive_prompt_tokens() -> None:
    usage = normalize_token_usage(
        {
            "completion_tokens": 50,
            "prompt_tokens": 1000,
            "total_tokens": 1050,
            "cache_creation_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 600},
        }
    )

    assert usage.input_tokens == 1000
    assert usage.cache_usage.read_input_tokens == 600
    assert usage.cache_usage.write_input_tokens == 100
    assert usage.cache_usage.uncached_input_tokens == 400


@pytest.mark.parametrize("case", _contract()["compaction_cases"][:3], ids=lambda case: case["name"])
def test_compaction_accounting_real_producer(case):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider, ToolOutputText, function_tool
    from vv_agent.llm import ScriptedLLM
    from vv_agent.llm.scripted import ScriptStep
    from vv_agent.memory import MemoryManager
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.result import project_result
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.tools import ToolRegistry, build_default_registry
    from vv_agent.types import LLMResponse
    from vv_agent.workspace import MemoryWorkspaceBackend

    sid = case["input"]["session_id"]
    source = _compaction_sources(case["input"]["transcript_ref"], sid)
    wires = source[sid]
    start = next(r for r in wires if r["kind"] == "turn_started")
    definition = start["payload"]["definition"]
    task = AgentTask.from_dict(definition["task"])
    summary_source = source["summary"]
    summary_plan = next(r for r in summary_source if r["kind"] == "op_planned" and r["payload"]["purpose"] == "compaction")
    summary = next(
        r["payload"]["result"]["content"]
        for r in summary_source
        if r["kind"] == "op_completed" and r["operation_id"] == summary_plan["operation_id"]
    )
    steps: list[ScriptStep]
    if sid == "micro":
        steps = [LLMResponse("done")]
    elif sid == "summary":
        steps = [LLMResponse(summary), LLMResponse("done")]
    else:
        from vv_agent.types import ToolCall

        tool = next(r["payload"]["request"] for r in wires if r["kind"] == "op_planned" and r["payload"]["op_kind"] == "tool")
        steps = [LLMResponse(summary), LLMResponse("", [ToolCall.from_dict(tool)]), LLMResponse(summary), LLMResponse("done")]

    @function_tool
    def more() -> ToolOutputText:
        return ToolOutputText("new " * 1200)

    def registry():
        result = ToolRegistry()
        if sid == "second_summary":
            result.register_executor(more.to_executor())
        builtins = build_default_registry()
        for schema in definition["tools"]:
            name = schema["function"]["name"]
            if builtins.has_executor(name):
                result.register_executor(builtins.get_executor(name))
        return result

    provider = ScriptedModelProvider("scripted", "m", ScriptedLLM(steps), context_length=None, max_output_tokens=None)
    driver = SessionDriver()
    try:
        config = RunConfig(model_provider=provider, workspace="/fixture", tool_registry_factory=registry)
        runtime = driver.runtime(Agent("fixture", task.prompt_bundle, model="m"), config, task)
        settings = dict(definition["memory_settings"])
        settings["microcompaction_policy"] = MicrocompactionPolicy.from_dict(settings["microcompaction_policy"])
        runtime.memory_manager = MemoryManager(**settings, workspace_backend=MemoryWorkspaceBackend())
        # Reproduce the source fixture's imported-history creation policy.
        driver.create(sid, "/fixture", {"seed": {"unseen_input_protection": False}} if task.initial_messages else None)
        driver.push(sid, InboxItem("initial", "user", {"content": "go"}))
        if sid == "second_summary":

            def steer(point, record):
                if point == "after_commit" and record.kind == "op_completed" and record.operation_id.endswith("/tool/0"):
                    runtime.hook = lambda *_: None
                    driver.push(sid, InboxItem("continue", "steer", {"content": "continue"}, record.turn_id))

            runtime.hook = steer
        drive(driver.store, sid, runtime=runtime)
        result = project_result(driver.store, sid, f"{sid}/turn/initial", runtime=runtime)
        calls = [r.to_dict() for r in result.token_usage.model_calls if r.operation is ModelCallOperation.MEMORY_COMPACTION]
        expected = case["expected"]
        assert calls == expected["model_calls"]
        assert len(calls) == expected["new_model_dispatches"] == expected["new_model_call_records"]
        total = None if any(r["usage"]["total_tokens"] is None for r in calls) else sum(r["usage"]["total_tokens"] for r in calls)
        assert total == expected["new_budget_total_tokens"]
    finally:
        driver.close()
