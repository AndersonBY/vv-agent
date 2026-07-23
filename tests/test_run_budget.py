from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, CompletionReason, FunctionTool, GuardrailResult, RunConfig, Runner, output_guardrail
from vv_agent.budget import (
    MAX_WIRE_INTEGER,
    BudgetDimension,
    BudgetEnforcementBoundary,
    BudgetEvaluator,
    BudgetExhaustionReason,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    HostCost,
    RunBudgetLimits,
    UnavailableMetricPolicy,
)
from vv_agent.events import DiagnosticEvent
from vv_agent.llm import LlmRequest
from vv_agent.llm.scripted import ScriptStep
from vv_agent.model import ScriptedModelProvider
from vv_agent.runtime import CancellationToken
from vv_agent.runtime.backends import InlineBackend, ThreadBackend
from vv_agent.tools import ToolOutputText
from vv_agent.types import CacheUsage, CacheUsageStatus, LLMResponse, TokenUsage, ToolCall, UsageSource

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_budget.json"
MODEL = "budget-model"


def _provider(steps: list[ScriptStep]) -> ScriptedModelProvider:
    return ScriptedModelProvider.from_steps("test", MODEL, steps)


class _Clock:
    def __init__(self, milliseconds: list[int]) -> None:
        self._readings = iter(value * 1_000_000 for value in milliseconds)
        self._last = 0

    def __call__(self) -> int:
        with suppress(StopIteration):
            self._last = next(self._readings)
        return self._last


class _Meter:
    def __init__(self, readings: list[HostCost | None | Exception]) -> None:
        self._readings: Iterator[HostCost | None | Exception] = iter(readings)
        self._last: HostCost | None | Exception = readings[-1] if readings else None

    def read(self) -> HostCost | None:
        with suppress(StopIteration):
            self._last = next(self._readings)
        if isinstance(self._last, Exception):
            raise self._last
        return self._last


def _usage(total: int, uncached: int | None) -> TokenUsage:
    return TokenUsage(
        total_tokens=total,
        usage_source=UsageSource.PROVIDER_REPORTED,
        cache_usage=CacheUsage(
            status=(CacheUsageStatus.PROVIDER_REPORTED if uncached is not None else CacheUsageStatus.ACCOUNTING_MISSING),
            uncached_input_tokens=uncached,
        ),
    )


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_budget_wire_examples_round_trip() -> None:
    fixture = _fixture()["wire_examples"]

    limits = RunBudgetLimits.from_dict(fixture["limits"])
    snapshot = BudgetUsageSnapshot.from_dict(fixture["snapshot"])

    assert limits.to_dict() == fixture["limits"]
    assert snapshot.to_dict() == fixture["snapshot"]


@pytest.mark.parametrize("case", _fixture()["invalid_cases"], ids=lambda case: case["name"])
def test_budget_limits_reject_contract_invalid_cases(case: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        RunBudgetLimits.from_dict(case["limits"])


def test_total_tokens_equal_limit_can_finish_but_next_cycle_is_rejected() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_total_tokens=10),
        clock_ns=_Clock([0, 0, 0, 0]),
    )

    assert evaluator.run_start() is None
    assert evaluator.cycle_start() is None
    assert evaluator.model_call_complete(_usage(10, 10)) is None
    exhaustion = evaluator.cycle_start()

    assert exhaustion is not None
    assert exhaustion.dimension is BudgetDimension.TOTAL_TOKENS
    assert exhaustion.reason is BudgetExhaustionReason.LIMIT_REACHED
    assert exhaustion.enforcement_boundary is BudgetEnforcementBoundary.CYCLE_START
    assert exhaustion.overshoot == 0


def test_total_tokens_allow_one_atomic_overshoot() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_total_tokens=10),
        clock_ns=_Clock([0, 0, 0, 0]),
    )

    assert evaluator.run_start() is None
    assert evaluator.cycle_start() is None
    exhaustion = evaluator.model_call_complete(_usage(12, 12))

    assert exhaustion is not None
    assert exhaustion.reason is BudgetExhaustionReason.LIMIT_EXCEEDED
    assert exhaustion.observed == 12
    assert exhaustion.overshoot == 2
    assert evaluator.snapshot().total_tokens == 12


def test_missing_uncached_usage_is_not_zero_and_policy_is_configurable() -> None:
    continuing = BudgetEvaluator(
        RunBudgetLimits(max_uncached_input_tokens=10),
        clock_ns=_Clock([0, 0, 0, 0]),
    )
    strict = BudgetEvaluator(
        RunBudgetLimits(
            max_uncached_input_tokens=10,
            unavailable_metric_policy=UnavailableMetricPolicy.STOP,
        ),
        clock_ns=_Clock([0, 0, 0, 0]),
    )

    assert continuing.run_start() is None
    assert continuing.cycle_start() is None
    assert continuing.model_call_complete(_usage(4, None)) is None
    assert continuing.snapshot().uncached_input_tokens is None
    assert continuing.snapshot().unavailable_dimensions[0].reason is BudgetUnavailableReason.USAGE_MISSING

    assert strict.run_start() is None
    assert strict.cycle_start() is None
    exhaustion = strict.model_call_complete(_usage(4, None))
    assert exhaustion is not None
    assert exhaustion.reason is BudgetExhaustionReason.METRIC_UNAVAILABLE
    assert exhaustion.unavailable_reason is BudgetUnavailableReason.USAGE_MISSING


def test_explicit_zero_uncached_usage_remains_available() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_uncached_input_tokens=1),
        clock_ns=_Clock([0, 0, 0, 0]),
    )

    evaluator.run_start()
    evaluator.cycle_start()

    assert evaluator.model_call_complete(_usage(3, 0)) is None
    assert evaluator.snapshot().uncached_input_tokens == 0
    assert evaluator.snapshot().unavailable_dimensions == ()


def test_tool_batch_preflight_has_no_partial_reservation() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_tool_calls=1, max_tool_calls_by_name={"alpha": 1}),
        clock_ns=_Clock([0, 0]),
    )
    evaluator.run_start()

    exhaustion = evaluator.preflight_tools(["alpha", "beta"])

    assert exhaustion is not None
    assert exhaustion.dimension is BudgetDimension.TOOL_CALLS
    assert exhaustion.attempted_increment == 2
    assert evaluator.snapshot().tool_calls == 0
    assert evaluator.snapshot().tool_calls_by_name == {}


def test_named_tool_budget_matches_exact_name() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_tool_calls_by_name={"search": 1}),
        clock_ns=_Clock([0, 0]),
    )
    evaluator.run_start()

    assert evaluator.preflight_tools(["lookup_records"]) is None
    exhaustion = evaluator.preflight_tools(["search", "search"])

    assert exhaustion is not None
    assert exhaustion.dimension is BudgetDimension.TOOL_CALLS_BY_NAME
    assert exhaustion.tool_name == "search"
    assert evaluator.snapshot().tool_calls_by_name == {"lookup_records": 1}


def test_host_meter_overshoot_and_non_monotonic_reading() -> None:
    limit = HostCost(unit="credits", amount_microunits=100)
    overshoot = BudgetEvaluator(
        RunBudgetLimits(max_host_cost=limit),
        host_cost_meter=_Meter(
            [
                HostCost(unit="credits", amount_microunits=0),
                HostCost(unit="credits", amount_microunits=0),
                HostCost(unit="credits", amount_microunits=120),
            ]
        ),
        clock_ns=_Clock([0, 0, 0, 0]),
    )

    assert overshoot.run_start() is None
    assert overshoot.cycle_start() is None
    exhaustion = overshoot.model_call_complete(_usage(1, 1))
    assert exhaustion is not None
    assert exhaustion.dimension is BudgetDimension.HOST_COST
    assert exhaustion.overshoot == 20

    non_monotonic = BudgetEvaluator(
        RunBudgetLimits(max_host_cost=limit),
        host_cost_meter=_Meter(
            [
                HostCost(unit="credits", amount_microunits=50),
                HostCost(unit="credits", amount_microunits=40),
            ]
        ),
        clock_ns=_Clock([0, 0, 0]),
    )
    assert non_monotonic.run_start() is None
    assert non_monotonic.cycle_start() is None
    assert non_monotonic.snapshot().host_cost is None
    assert non_monotonic.snapshot().unavailable_dimensions[0].reason is BudgetUnavailableReason.NON_MONOTONIC


def test_token_sum_wire_overflow_becomes_typed_unavailable() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(
            max_total_tokens=MAX_WIRE_INTEGER,
            unavailable_metric_policy=UnavailableMetricPolicy.STOP,
        ),
        initial_usage=BudgetUsageSnapshot(cycles=1, total_tokens=MAX_WIRE_INTEGER),
        clock_ns=_Clock([0, 0]),
    )

    exhaustion = evaluator.model_call_complete(_usage(1, 0))

    assert exhaustion is not None
    assert exhaustion.reason is BudgetExhaustionReason.METRIC_UNAVAILABLE
    assert exhaustion.unavailable_reason is BudgetUnavailableReason.INTEGER_OVERFLOW
    assert evaluator.snapshot().total_tokens is None


def test_distributed_active_elapsed_continues_from_snapshot_without_queue_time() -> None:
    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_wall_time_ms=1000),
        initial_usage=BudgetUsageSnapshot(elapsed_ms=120),
        clock_ns=_Clock([5000, 5030]),
    )

    assert evaluator.run_start() is None
    assert evaluator.snapshot().elapsed_ms == 150


def _scripted_response(step: dict[str, Any]) -> LLMResponse:
    usage = step["usage"]
    raw: dict[str, Any] = {}
    if usage is not None:
        total = usage["total_tokens"]
        uncached = usage["uncached_input_tokens"]
        usage_payload: dict[str, Any] = {
            "prompt_tokens": total if uncached is None else uncached,
            "completion_tokens": 0 if uncached is None else total - uncached,
            "total_tokens": total,
        }
        if uncached is not None:
            usage_payload["prompt_tokens_details"] = {"cached_tokens": 0}
        raw["usage"] = usage_payload
    return LLMResponse(
        content=step["assistant_output"],
        tool_calls=[ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"])) for call in step["tool_calls"]],
        raw=raw,
    )


@pytest.mark.parametrize("case", _fixture()["runner_cases"], ids=lambda case: case["name"])
def test_public_runner_budget_cases_match_contract(case: dict[str, Any], tmp_path: Path) -> None:
    model_calls = 0
    scripted_steps: list[ScriptStep] = []
    for step in case["steps"]:
        response = _scripted_response(step)

        def respond(_request: LlmRequest, response: LLMResponse = response) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return response

        scripted_steps.append(respond)

    tool_execution_count = 0

    def invoke_tool(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        nonlocal tool_execution_count
        tool_execution_count += 1
        return ToolOutputText(text="fixture tool result")

    tool_names = sorted({call["name"] for step in case["steps"] for call in step["tool_calls"]})
    tools = [
        FunctionTool(
            name=name,
            description="Execute a deterministic budget fixture tool.",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke=invoke_tool,
        )
        for name in tool_names
    ]
    meter = _Meter([HostCost.from_dict(reading) for reading in case["host_cost_readings"]])
    cancellation = CancellationToken()
    if case["pre_cancelled"]:
        cancellation.cancel("cancelled by fixture")
    limits = RunBudgetLimits.from_dict(case["limits"]) if case["limits"] is not None else None

    result = Runner.run_sync(
        Agent(
            name="run-budget-contract",
            instructions="Execute the deterministic budget fixture.",
            model=MODEL,
            tools=tools,
            no_tool_policy=case["no_tool_policy"],
        ),
        "run the fixture",
        run_config=RunConfig(
            workspace=tmp_path,
            max_cycles=max(2, len(case["steps"]) + 1),
            budget_limits=limits,
            host_cost_meter=meter if case["host_cost_readings"] else None,
            cancellation_token=cancellation,
            model_provider=_provider(scripted_steps),
        ),
    )

    expected = case["expected"]
    assert result.status.value == expected["status"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert model_calls == expected["model_calls"]
    assert tool_execution_count == expected["tool_execution_count"]
    if "partial_output" in expected:
        assert result.partial_output == expected["partial_output"]
    if "error" in expected:
        assert result.raw_result.error == expected["error"]
    if expected.get("budget_usage") is None and case["limits"] is None:
        assert result.budget_usage is None
    if "usage" in expected:
        assert result.budget_usage is not None
        projection = result.budget_usage.to_dict()
        for key, value in expected["usage"].items():
            assert projection[key] == value
    if "uncached_input_tokens" in expected:
        assert result.budget_usage is not None
        assert result.budget_usage.uncached_input_tokens == expected["uncached_input_tokens"]
    if "tool_calls" in expected:
        assert result.budget_usage is not None
        assert result.budget_usage.tool_calls == expected["tool_calls"]
    if "unavailable_dimensions" in expected:
        assert result.budget_usage is not None
        assert [item.to_dict() for item in result.budget_usage.unavailable_dimensions] == expected["unavailable_dimensions"]
    expected_exhaustion = expected["budget_exhaustion"]
    assert (result.budget_exhaustion.to_dict() if result.budget_exhaustion is not None else None) == expected_exhaustion
    if expected_exhaustion is not None:
        assert [event.type for event in result.events[-2:]] == ["budget_exhausted", "run_failed"]
    if "budget_event_types" in expected:
        assert [event.type for event in result.events if event.type.startswith("budget_")] == expected["budget_event_types"]


def test_budget_exhaustion_precedes_output_guardrails(tmp_path: Path) -> None:
    guardrail_calls = 0

    @output_guardrail
    def should_not_run(_context: Any, _output: Any) -> GuardrailResult:
        nonlocal guardrail_calls
        guardrail_calls += 1
        return GuardrailResult.block("must not replace budget failure")

    provider = _provider(
        [
            LLMResponse(
                content="draft over budget",
                raw={
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 0,
                        "total_tokens": 12,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            )
        ]
    )
    result = Runner.run_sync(
        Agent(
            name="budget-before-guardrail",
            instructions="Return the scripted draft.",
            model=MODEL,
            no_tool_policy="finish",
            output_guardrails=[should_not_run],
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            budget_limits=RunBudgetLimits(max_total_tokens=10),
            model_provider=provider,
        ),
    )

    assert result.completion_reason is CompletionReason.BUDGET_EXHAUSTED
    assert result.raw_result.error == "Run budget exhausted."
    assert guardrail_calls == 0


def test_completed_llm_cancellation_wins_without_losing_budget_usage(tmp_path: Path) -> None:
    cancellation = CancellationToken()

    def complete_then_cancel(_request: LlmRequest) -> LLMResponse:
        cancellation.cancel("cancelled after completed LLM call")
        return LLMResponse(
            content="completed draft",
            raw={
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 0,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
            },
        )

    result = Runner.run_sync(
        Agent(
            name="cancel-after-llm",
            instructions="Return the scripted response.",
            model=MODEL,
            no_tool_policy="finish",
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            budget_limits=RunBudgetLimits(max_total_tokens=10),
            cancellation_token=cancellation,
            model_provider=_provider([complete_then_cancel]),
        ),
    )

    assert result.status.value == "failed"
    assert result.completion_reason is CompletionReason.CANCELLED
    assert result.partial_output == "completed draft"
    assert result.budget_usage is not None
    assert result.budget_usage.cycles == 1
    assert result.budget_usage.total_tokens == 12
    assert result.budget_exhaustion is None
    assert "budget_exhausted" not in [event.type for event in result.events]
    assert result.events[-1].type == "run_cancelled"


def test_post_tool_host_cost_exhaustion_precedes_tool_finish(tmp_path: Path) -> None:
    tool_calls = 0

    def invoke_tool(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        nonlocal tool_calls
        tool_calls += 1
        return ToolOutputText(text="tool finished")

    meter = _Meter(
        [
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=120),
        ]
    )
    provider = _provider(
        [
            LLMResponse(
                content="costly tool draft",
                tool_calls=[ToolCall(id="finish-call", name="finish_work", arguments={})],
                raw={
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            )
        ]
    )
    result = Runner.run_sync(
        Agent(
            name="budget-before-tool-finish",
            instructions="Call the scripted tool.",
            model=MODEL,
            tools=[
                FunctionTool(
                    name="finish_work",
                    description="Finish after one deterministic side effect.",
                    params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    on_invoke=invoke_tool,
                )
            ],
            tool_use_behavior="stop_on_first_tool",
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            budget_limits=RunBudgetLimits(
                max_host_cost=HostCost(unit="credits", amount_microunits=100),
            ),
            host_cost_meter=meter,
            model_provider=provider,
        ),
    )

    assert tool_calls == 1
    assert result.status.value == "failed"
    assert result.completion_reason is CompletionReason.BUDGET_EXHAUSTED
    assert result.partial_output == "costly tool draft"
    assert result.budget_exhaustion is not None
    assert result.budget_exhaustion.enforcement_boundary is BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE


def test_completed_tool_cancellation_wins_without_losing_budget_usage(tmp_path: Path) -> None:
    cancellation = CancellationToken()
    tool_calls = 0

    def invoke_tool(_context: Any, _arguments: dict[str, Any]) -> ToolOutputText:
        nonlocal tool_calls
        tool_calls += 1
        return ToolOutputText(text="tool side effect completed")

    def cancel_after_tool_result(event: Any) -> None:
        if isinstance(event, DiagnosticEvent) and event.code == "tool_result":
            cancellation.cancel("cancelled after completed tool call")

    meter = _Meter(
        [
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=0),
            HostCost(unit="credits", amount_microunits=120),
        ]
    )
    provider = _provider(
        [
            LLMResponse(
                content="tool draft",
                tool_calls=[ToolCall(id="work-call", name="do_work", arguments={})],
                raw={
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 0,
                        "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            )
        ]
    )
    result = Runner.run_sync(
        Agent(
            name="cancel-after-tool",
            instructions="Call the scripted tool.",
            model=MODEL,
            tools=[
                FunctionTool(
                    name="do_work",
                    description="Perform one deterministic side effect.",
                    params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    on_invoke=invoke_tool,
                )
            ],
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            budget_limits=RunBudgetLimits(
                max_tool_calls=1,
                max_host_cost=HostCost(unit="credits", amount_microunits=100),
            ),
            host_cost_meter=meter,
            cancellation_token=cancellation,
            stream=cancel_after_tool_result,
            model_provider=provider,
        ),
    )

    assert tool_calls == 1
    assert result.status.value == "failed"
    assert result.completion_reason is CompletionReason.CANCELLED
    assert result.partial_output == "tool draft"
    assert result.budget_usage is not None
    assert result.budget_usage.tool_calls == 1
    assert result.budget_usage.host_cost == HostCost(unit="credits", amount_microunits=120)
    assert result.budget_exhaustion is None
    assert "budget_exhausted" not in [event.type for event in result.events]
    assert result.events[-1].type == "run_cancelled"


@pytest.mark.parametrize(
    "backend_factory",
    [InlineBackend, lambda: ThreadBackend(max_workers=1)],
    ids=["inline", "thread"],
)
def test_inline_and_thread_backends_share_budget_enforcement(
    backend_factory: Any,
    tmp_path: Path,
) -> None:
    provider = _provider(
        [
            LLMResponse(
                content="backend draft",
                raw={
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 0,
                        "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            )
        ]
    )
    result = Runner.run_sync(
        Agent(
            name="backend-budget",
            instructions="Return the scripted response.",
            model=MODEL,
            no_tool_policy="finish",
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            execution_backend=backend_factory(),
            budget_limits=RunBudgetLimits(max_total_tokens=1),
            model_provider=provider,
        ),
    )

    assert result.completion_reason is CompletionReason.BUDGET_EXHAUSTED
    assert result.budget_exhaustion is not None
    assert result.budget_exhaustion.enforcement_boundary is BudgetEnforcementBoundary.MODEL_CALL_COMPLETE


def test_per_run_budget_replaces_configured_runner_default_as_a_whole(tmp_path: Path) -> None:
    provider = _provider(
        [
            LLMResponse(
                content="within per-run limit",
                raw={
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 0,
                        "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            )
        ]
    )
    configured = Runner.configured(
        RunConfig(
            model_provider=provider,
            budget_limits=RunBudgetLimits(
                max_total_tokens=1,
                max_tool_calls=0,
            ),
        )
    )
    result = configured.run_sync(
        Agent(
            name="budget-precedence",
            instructions="Return the scripted response.",
            model=MODEL,
            no_tool_policy="finish",
        ),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            budget_limits=RunBudgetLimits(max_total_tokens=10),
        ),
    )

    assert result.status.value == "completed"
    assert result.budget_exhaustion is None
    assert result.budget_usage is not None
    assert result.budget_usage.total_tokens == 2
