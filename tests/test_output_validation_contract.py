from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    OutputRepairRequest,
    OutputValidationContext,
    OutputValidationResult,
    RunConfig,
    Runner,
    function_tool,
)
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model import ScriptedModelProvider
from vv_agent.result import RunResult
from vv_agent.types import AgentStatus, LLMResponse, ToolCall

FIXTURE_PATH = Path(__file__).parent / "fixtures/parity/output_validation.json"


def _fixture_cases() -> dict[str, dict[str, Any]]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "vv-agent.output-validation.v1"
    return {case["name"]: case for case in fixture["runner_cases"]}


def _run(
    output: str,
    *,
    model_step: Callable[[LlmRequest], LLMResponse] | None = None,
    **agent_kwargs: Any,
) -> RunResult:
    model = ScriptedLLM(steps=[model_step or LLMResponse(content=output)])
    agent = Agent(
        name="output-validation-agent",
        instructions="Return the requested value.",
        model="test-model",
        **agent_kwargs,
    )
    return Runner.run_sync(
        agent,
        "return a value",
        run_config=RunConfig(
            model_provider=ScriptedModelProvider(
                backend="test",
                default_model="test-model",
                llm=model,
            ),
            no_tool_policy="finish",
            max_cycles=1,
        ),
    )


def _observable_trace(result: RunResult) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_projection = []
    for event in result.events:
        payload = event.to_dict()
        event_projection.append(
            {
                key: payload[key]
                for key in ("type", "cycle_index", "agent_name", "model", "status", "final_output")
                if key in payload
            }
        )
    return event_projection, result.metadata["run_span"]["metadata"]


def test_output_validation_is_disabled_by_default_without_trace_or_terminal_changes() -> None:
    expected = _fixture_cases()["disabled"]["expected"]
    validator_calls = 0
    repair_calls = 0

    def would_fail(_output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        return OutputValidationResult.reject("unexpected")

    def repair(request: OutputRepairRequest) -> Any:
        nonlocal repair_calls
        repair_calls += 1
        return request.invalid_output

    baseline = _run("unchanged")
    result = _run(
        "unchanged",
        output_validator=would_fail,
        output_repair=repair,
    )

    assert validator_calls == expected["validator_calls"]
    assert repair_calls == expected["repair_calls"]
    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "unchanged"
    assert _observable_trace(result) == _observable_trace(baseline)


def test_valid_output_passes_without_repair_or_observation_changes() -> None:
    expected = _fixture_cases()["valid_without_repair"]["expected"]
    validator_calls = 0

    def validate(output: Any, context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        assert output == "valid"
        assert context.agent_name == "output-validation-agent"
        assert context.output_type is None
        return OutputValidationResult.accept()

    baseline = _run("valid")
    result = _run(
        "valid",
        output_validation_enabled=True,
        output_validator=validate,
    )

    assert validator_calls == expected["validator_calls"]
    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == expected["final_output"]
    assert _observable_trace(result) == _observable_trace(baseline)


def test_invalid_output_without_repair_is_persisted_as_typed_failure() -> None:
    expected = _fixture_cases()["invalid_without_repair_handler"]["expected"]
    result = _run(
        "invalid",
        output_validation_enabled=True,
        output_validator=lambda _output, _context: OutputValidationResult.reject(
            "format_invalid",
            "expected a valid marker",
        ),
    )

    assert result.status is AgentStatus.FAILED
    assert result.error_code == expected["error_code"]
    assert result.to_dict()["error_code"] == expected["error_code"]
    assert result.raw_result.to_dict()["error_code"] == expected["error_code"]
    assert result.raw_result.error == "output_validation_failed: format_invalid: expected a valid marker"
    assert result.partial_output == "invalid"
    assert result.events[-1].type == "run_failed"
    assert result.events[-1].to_dict()["error"] == ("output_validation_failed: format_invalid: expected a valid marker")


def test_one_tools_free_repair_is_revalidated_without_an_extra_model_call() -> None:
    expected = _fixture_cases()["one_repair_then_valid"]["expected"]
    validator_calls = 0
    repair_calls = 0
    model_requests: list[LlmRequest] = []

    def model_step(request: LlmRequest) -> LLMResponse:
        model_requests.append(request)
        return LLMResponse(content="invalid")

    def validate(output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        if output == "repaired":
            return OutputValidationResult.accept()
        return OutputValidationResult.reject("format_invalid", "repair required")

    def repair(request: OutputRepairRequest) -> str:
        nonlocal repair_calls
        repair_calls += 1
        assert request.invalid_output == "invalid"
        assert request.validation_code == "format_invalid"
        assert request.validation_message == "repair required"
        assert request.model == "repair-model"
        assert request.model_settings == {"temperature": 0}
        assert request.tools == ()
        return "repaired"

    result = _run(
        "invalid",
        model_step=model_step,
        output_validation_enabled=True,
        output_validator=validate,
        output_repair=repair,
        output_repair_model="repair-model",
        output_repair_model_settings={"temperature": 0},
    )

    assert validator_calls == expected["validator_calls"]
    assert repair_calls == expected["repair_calls"]
    assert len(model_requests) == 1
    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == expected["final_output"]
    assert result.raw_result.final_answer == "repaired"
    assert result.events[-1].type == "run_completed"
    assert result.events[-1].to_dict()["final_output"] == "repaired"


def test_still_invalid_repair_does_not_attempt_a_second_repair() -> None:
    expected = _fixture_cases()["repair_result_still_invalid"]["expected"]
    validator_calls = 0
    repair_calls = 0

    def validate(_output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        return OutputValidationResult.reject("still_invalid")

    def repair(_request: OutputRepairRequest) -> str:
        nonlocal repair_calls
        repair_calls += 1
        return "still invalid"

    result = _run(
        "invalid",
        output_validation_enabled=True,
        output_validator=validate,
        output_repair=repair,
    )

    assert validator_calls == expected["validator_calls"]
    assert repair_calls == expected["repair_calls"]
    assert result.status is AgentStatus.FAILED
    assert result.error_code == expected["error_code"]


def test_repair_provider_failure_is_typed_validation_failure() -> None:
    expected = _fixture_cases()["repair_provider_failure"]["expected"]
    validator_calls = 0
    repair_calls = 0

    def validate(_output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        return OutputValidationResult.reject("invalid")

    def repair(_request: OutputRepairRequest) -> str:
        nonlocal repair_calls
        repair_calls += 1
        raise RuntimeError("provider unavailable")

    result = _run(
        "invalid",
        output_validation_enabled=True,
        output_validator=validate,
        output_repair=repair,
    )

    assert validator_calls == expected["validator_calls"]
    assert repair_calls == expected["repair_calls"]
    assert result.status is AgentStatus.FAILED
    assert result.error_code == expected["error_code"]
    assert result.raw_result.error is not None
    assert "repair_provider_error: provider unavailable" in result.raw_result.error


def test_output_type_failure_can_be_repaired_then_typed_validator_rechecks_it() -> None:
    validator_calls = 0
    repair_calls = 0

    def validate(output: Any, context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        assert output == {"answer": "repaired"}
        assert context.output_type is dict
        return OutputValidationResult.accept()

    def repair(request: OutputRepairRequest) -> str:
        nonlocal repair_calls
        repair_calls += 1
        assert request.validation_code == "output_type_invalid"
        assert request.invalid_output == "not-json"
        assert request.tools == ()
        return '{"answer":"repaired"}'

    result = _run(
        "not-json",
        output_type=dict,
        output_validation_enabled=True,
        output_validator=validate,
        output_repair=repair,
    )

    assert validator_calls == 1
    assert repair_calls == 1
    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == {"answer": "repaired"}


def test_output_validation_configuration_rejects_unsafe_or_incomplete_values() -> None:
    with pytest.raises(ValueError, match="enabled output validation requires an output_validator"):
        Agent(
            name="missing-validator",
            instructions="Return a value.",
            output_validation_enabled=True,
        )
    with pytest.raises(ValueError, match="output_repair requires an output_validator"):
        Agent(
            name="missing-validator",
            instructions="Return a value.",
            output_repair=lambda request: request.invalid_output,
        )
    with pytest.raises(ValueError, match="output_validation_max_repairs must be 0 or 1"):
        Agent(
            name="too-many-repairs",
            instructions="Return a value.",
            output_validation_max_repairs=2,
        )


def test_approved_finish_validates_repaired_output_before_terminal_commit() -> None:
    validator_calls = 0
    repair_calls = 0

    @function_tool(needs_approval=True)
    def guarded_finish() -> str:
        return "invalid"

    def validate(output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        if output == "repaired":
            return OutputValidationResult.accept()
        return OutputValidationResult.reject("format_invalid")

    def repair(request: OutputRepairRequest) -> str:
        nonlocal repair_calls
        repair_calls += 1
        assert request.invalid_output == "invalid"
        return "repaired"

    agent = Agent(
        name="output-validation-agent",
        instructions="Finish with the guarded tool.",
        model="test-model",
        tools=[guarded_finish],
        tool_use_behavior="stop_on_first_tool",
        output_validation_enabled=True,
        output_validator=validate,
        output_repair=repair,
    )

    interrupted = Runner.run_sync(
        agent,
        "finish after approval",
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.from_steps(
                "test",
                "test-model",
                [
                    LLMResponse(
                        content="draft",
                        tool_calls=[ToolCall(id="finish", name="guarded_finish", arguments={})],
                    )
                ],
            )
        ),
    )
    assert interrupted.status is AgentStatus.WAIT_USER
    assert validator_calls == 0
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert validator_calls == 2
    assert repair_calls == 1
    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "repaired"
    assert resumed.events[-1].type == "run_completed"
    assert resumed.events[-1].to_dict()["final_output"] == "repaired"
