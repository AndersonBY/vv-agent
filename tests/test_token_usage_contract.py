from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.types import (
    CacheUsage,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    TaskTokenUsage,
    TokenUsage,
    UsageSource,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "token_usage.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


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


@pytest.mark.parametrize("case", _contract()["compaction_cases"][:3], ids=lambda case: case["name"])
def test_compaction_accounting_real_producer(case):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider, ToolOutputText, function_tool
    from vv_agent.llm import ScriptedLLM
    from vv_agent.llm.scripted import ScriptStep
    from vv_agent.memory import MemoryManager
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem, Record
    from vv_agent.session.result import project_result
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.tools import ToolRegistry, build_default_registry
    from vv_agent.types import LLMResponse
    from vv_agent.workspace import MemoryWorkspaceBackend

    sid = case["input"]["session_id"]
    source = json.loads((FIXTURE_PATH.parent / "session_projection.json").read_text())["source_records"][sid]
    wires = [r["wire"] for r in source]
    start = next(r for r in wires if r["kind"] == "turn_started")
    definition = start["payload"]["definition"]
    task = Record(**start).task()
    summary_source = json.loads((FIXTURE_PATH.parent / "session_projection.json").read_text())["source_records"]["summary"]
    summary = next(r["wire"]["payload"]["result"]["content"] for r in summary_source if r["wire"]["kind"] == "op_completed")
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
        driver.create(sid, "/fixture")
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
