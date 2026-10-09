from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import (
    Agent,
    CompletionReason,
    FunctionTool,
    GuardrailResult,
    NoToolPolicy,
    RunConfig,
    Runner,
    output_guardrail,
)
from vv_agent.agent import ToolUseBehavior
from vv_agent.llm import LlmRequest
from vv_agent.llm.scripted import ScriptStep
from vv_agent.model import ScriptedModelProvider
from vv_agent.tools import ToolOutputText, build_default_registry
from vv_agent.types import LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "completion_policy.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _response(step: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=step["assistant_output"],
        tool_calls=[
            ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=dict(call["arguments"]),
            )
            for call in step["tool_calls"]
        ],
    )


def _provider(steps: list[ScriptStep]) -> ScriptedModelProvider:
    return ScriptedModelProvider.from_steps("test", "test-model", steps)


@pytest.mark.parametrize("case", _contract()["cases"], ids=lambda case: case["name"])
def test_public_completion_policy_matrix(case: dict[str, Any], tmp_path: Path) -> None:
    registry = build_default_registry()
    registry.register_tool(
        "handoff_result",
        lambda _context, _arguments: ToolExecutionResult(
            tool_call_id="",
            content="override done",
            directive=ToolDirective.FINISH,
            metadata={"final_message": "override done"},
        ),
        "Return the delegated result.",
    )
    requests: list[LlmRequest] = []
    scripted_steps: list[ScriptStep] = []
    for step in case["steps"]:
        response = _response(step)

        def respond(request: LlmRequest, response: LLMResponse = response) -> LLMResponse:
            requests.append(request)
            return response

        scripted_steps.append(respond)

    tools = []
    tool_results = [
        result
        for step in case["steps"]
        for result in step.get("tool_results", [])
        if any(call["name"] == "lookup" for call in step["tool_calls"])
    ]
    if tool_results:
        lookup_output = str(tool_results[0]["content"])
        tools.append(
            FunctionTool(
                name="lookup",
                description="Return a deterministic fixture value.",
                params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                on_invoke=lambda _context, _arguments: ToolOutputText(text=lookup_output),
            )
        )

    agent = Agent(
        name="completion-contract",
        instructions="Execute the scripted completion contract.",
        model="test-model",
        tools=tools,
        no_tool_policy=cast(NoToolPolicy | None, case["agent_policy"]),
        tool_use_behavior=cast(ToolUseBehavior, case["tool_use_behavior"]),
        stop_at_tool_names=list(case.get("stop_at_tool_names", [])),
    )
    configured = Runner.configured(
        RunConfig(
            model_provider=_provider(scripted_steps),
            tool_registry_factory=lambda: registry,
            no_tool_policy=cast(NoToolPolicy | None, case["runner_default_policy"]),
        )
    )
    result = configured.run_sync(
        agent,
        "run the completion fixture",
        run_config=RunConfig(
            workspace=tmp_path,
            max_cycles=case["max_cycles"],
            no_tool_policy=cast(NoToolPolicy | None, case["run_policy"]),
        ),
    )

    expected = case["expected"]
    assert result.status.value == expected["status"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.completion_tool_name == expected["completion_tool_name"]
    assert result.raw_result.final_answer == expected["final_answer"]
    assert result.raw_result.wait_reason == expected["wait_reason"]
    assert result.partial_output == expected["partial_output"]
    assert len(result.raw_result.cycles) == expected["cycles"]

    continuation_hint_emitted = any(
        message.role == "user" and message.content.startswith("Continue working on the task.")
        for request in requests[1:]
        for message in request.messages
    )
    assert continuation_hint_emitted is expected["continuation_hint_emitted"]
    assert all(
        "task_finish" not in {cast(dict[str, Any], tool["function"])["name"] for tool in request.tools} for request in requests
    )

    assert result.completion_reason.value == expected["completion_reason"]


def test_completion_controls_reject_unknown_policy() -> None:
    with pytest.raises(ValueError, match=r"Agent\.no_tool_policy must be one of"):
        Agent(
            name="invalid-agent-policy",
            instructions="Invalid policy.",
            no_tool_policy=cast(NoToolPolicy, "semantic_detector"),
        )
    with pytest.raises(ValueError, match=r"RunConfig\.no_tool_policy must be one of"):
        RunConfig(no_tool_policy=cast(NoToolPolicy, "semantic_detector"))


def test_completion_reason_inventory_matches_public_enum() -> None:
    contract = _contract()
    assert [reason.value for reason in CompletionReason] == contract["completion_reason_values"]
    assert contract["rules"]["assistant_text_is_not_classified"] is True
    assert contract["rules"]["completion_policy_does_not_change_tool_availability"] is True


def test_input_guardrail_failure_emits_the_canonical_reason() -> None:
    expected = next(
        case for case in _contract()["terminal_precedence_cases"] if case["name"] == "input_guardrail_fails_before_llm"
    )

    def block_input(_context: Any, _input: str) -> GuardrailResult:
        return GuardrailResult.block("blocked by input guardrail")

    result = Runner.run_sync(
        Agent(
            name="blocked-input",
            instructions="This model must not run.",
            model="test-model",
            input_guardrails=[block_input],
        ),
        "blocked",
        run_config=RunConfig(model_provider=_provider([])),
    )

    assert result.status.value == expected["expected_status"]
    assert result.completion_reason == CompletionReason(expected["expected_reason"])
    assert result.partial_output == expected["expected_partial_output"]
    assert result.completion_reason.value == expected["expected_reason"]


def test_output_guardrail_runs_only_after_same_turn_wait_is_resolved(tmp_path: Path) -> None:
    case = _contract()["output_guardrail_allow"]["case"]
    calls = []

    @output_guardrail
    def rewrite_output(_context: Any, value: Any) -> GuardrailResult:
        calls.append(value)
        return GuardrailResult.rewrite(case["guardrail_rewrite_output"])

    handle = Runner.start(
        Agent("guardrail-wait-contract", "Ask.", model="test-model", output_guardrails=[rewrite_output]),
        "ask",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_provider(
                [
                    LLMResponse(
                        case["candidate_observation"]["partial_output"],
                        [ToolCall("ask-contract", "ask_user", {"question": case["candidate_output"]})],
                    ),
                    LLMResponse("done"),
                ]
            ),
        ),
    )
    waiting = handle.result()
    assert waiting.status.value == "wait_user" and calls == []
    assert waiting.final_output == case["candidate_output"]
    assert not any(e.type in {"run_completed", "run_failed"} for e in waiting.events)
    handle.kernel.answer(handle.session_id, handle.runtime, "answer", "answer")
    resumed = handle.resume()
    assert resumed.status.value == "completed"
    assert resumed.final_output == case["guardrail_rewrite_output"]
    assert calls == ["done"]


def test_ordinary_llm_failure_returns_typed_terminal() -> None:
    expected = _contract()["ordinary_llm_failure"]
    result = Runner.run_sync(
        Agent(
            name="llm-failure-contract",
            instructions="This scripted queue is intentionally empty.",
            model="test-model",
        ),
        "go",
        run_config=RunConfig(model_provider=_provider([])),
    )
    terminals = [event for event in result.events if event.type in {"run_completed", "run_failed", "run_cancelled"}]

    assert expected["runner_outcome"] == "typed_result"
    assert result.status.value == expected["status"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.completion_tool_name == expected["completion_tool_name"]
    assert result.partial_output == expected["partial_output"]
    assert len(terminals) == expected["terminal_count"]
    assert terminals[0].type == expected["terminal_event"]
