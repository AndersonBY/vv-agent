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
    ModelSettings,
    NoToolPolicy,
    RunConfig,
    Runner,
    TaskTokenUsage,
    ToolPolicy,
)
from vv_agent.agent import ToolUseBehavior
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.llm.scripted import ScriptStep
from vv_agent.tools import ToolOutputText
from vv_agent.types import AgentResult, AgentStatus, LLMResponse, SubTaskOutcome, ToolCall

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "completion_policy_v1.json"


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


@pytest.mark.parametrize("case", _contract()["cases"], ids=lambda case: case["name"])
def test_public_completion_policy_matrix(case: dict[str, Any], tmp_path: Path) -> None:
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
        model=ScriptedLLM(steps=scripted_steps),
        tools=tools,
        no_tool_policy=cast(NoToolPolicy | None, case["agent_policy"]),
        tool_use_behavior=cast(ToolUseBehavior, case["tool_use_behavior"]),
        stop_at_tool_names=list(case.get("stop_at_tool_names", [])),
    )
    configured = Runner.configured(RunConfig(no_tool_policy=cast(NoToolPolicy | None, case["runner_default_policy"])))
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
        message.role == "user" and message.content.startswith("No tool call was produced.")
        for request in requests[1:]
        for message in request.messages
    )
    assert continuation_hint_emitted is expected["continuation_hint_emitted"]
    assert all(
        "task_finish" in {cast(dict[str, Any], tool["function"])["name"] for tool in request.tools} for request in requests
    )

    terminal = result.events[-1]
    assert terminal.to_dict()["completion_reason"] == expected["completion_reason"]
    if expected["completion_tool_name"] is not None:
        assert terminal.to_dict()["completion_tool_name"] == expected["completion_tool_name"]


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


def test_completion_fields_preserve_legacy_positional_construction() -> None:
    tool_policy = ToolPolicy()
    agent = Agent(
        "legacy-agent",
        "Preserve the original positional field order.",
        None,
        ModelSettings(),
        [],
        [],
        [],
        [],
        None,
        [],
        None,
        tool_policy,
    )
    run_config = RunConfig(None, None, None, None, None, None, None, None, tool_policy)
    usage = TaskTokenUsage()
    result = AgentResult(
        AgentStatus.COMPLETED,
        [],
        [],
        "legacy answer",
        None,
        None,
        {"legacy": True},
        usage,
    )
    sub_task = SubTaskOutcome(
        "task-1",
        "worker",
        AgentStatus.COMPLETED,
        None,
        "legacy child answer",
        None,
        None,
        None,
        2,
        [],
        {"model": "resolved"},
    )

    assert agent.tool_policy is tool_policy
    assert agent.no_tool_policy is None
    assert run_config.tool_policy is tool_policy
    assert run_config.no_tool_policy is None
    assert result.final_answer == "legacy answer"
    assert result.shared_state == {"legacy": True}
    assert result.token_usage is usage
    assert result.completion_reason is None
    assert sub_task.cycles == 2
    assert sub_task.resolved == {"model": "resolved"}
    assert sub_task.completion_reason is None


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
            model=ScriptedLLM(steps=[]),
            input_guardrails=[block_input],
        ),
        "blocked",
    )

    assert result.status.value == expected["expected_status"]
    assert result.completion_reason == CompletionReason(expected["expected_reason"])
    assert result.partial_output == expected["expected_partial_output"]
    assert result.events[-1].to_dict()["completion_reason"] == expected["expected_reason"]
