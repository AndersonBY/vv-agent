from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, MemorySession, RunBudgetLimits, RunConfig, Runner, output_guardrail
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.event_store import RunEventReplayQuery
from vv_agent.events import RunEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import CancellationToken
from vv_agent.types import AgentStatus, CompletionReason, LLMResponse, ToolCall

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "runner_terminal_v1.json"
FIXTURE_SHA256 = "2c6f7e7477d95a817a5fa2df7cf0b11be65e43a67206ff5181b593aec5845593"
TERMINAL_TYPES = {"run_completed", "run_failed", "run_cancelled"}


def _contract() -> dict[str, Any]:
    payload = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_SHA256
    return json.loads(payload)


def _agent(*, output_guardrails=None, assistant_message: str = "finish", final_message: str = "done") -> Agent:
    return Agent(
        name="terminal-agent",
        instructions="Finish.",
        model=ScriptedLLM(
            steps=[
                LLMResponse(
                    content=assistant_message,
                    tool_calls=[
                        ToolCall(
                            id="finish",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": final_message},
                        )
                    ],
                )
            ]
        ),
        output_guardrails=list(output_guardrails or []),
    )


def test_session_persists_before_the_only_success_terminal() -> None:
    expected = _contract()["success_with_session"]
    result = Runner.run_sync(
        _agent(),
        "go",
        run_config=RunConfig(session=MemorySession("terminal-session")),
    )
    types = [event.type for event in result.events]

    assert types[-2:] == expected["tail"]
    assert [event.type for event in result.events if event.type in TERMINAL_TYPES] == [expected["terminal"]]
    assert result.status.value == expected["status"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.events[-1].to_dict()["completion_reason"] == expected["completion_reason"]


def test_output_guardrail_block_short_circuits_and_owns_final_terminal() -> None:
    expected = _contract()["output_guardrail_block"]
    later_calls = 0

    @output_guardrail
    def block(_context, _output):
        return GuardrailResult.block(str(expected["error"]))

    @output_guardrail
    def later(_context, output):
        nonlocal later_calls
        later_calls += 1
        return GuardrailResult.rewrite(output)

    result = Runner.run_sync(
        _agent(
            output_guardrails=[block, later],
            assistant_message=expected["partial_output"],
            final_message="tool result must not become partial output",
        ),
        "go",
        run_config=RunConfig(session=MemorySession("blocked-session")),
    )
    types = [event.type for event in result.events]

    assert types[-2:] == expected["tail"]
    assert [event.type for event in result.events if event.type in TERMINAL_TYPES] == [expected["terminal"]]
    assert result.status == AgentStatus(expected["status"])
    assert result.final_output == expected["error"]
    assert result.raw_result.final_answer is None
    assert result.raw_result.error == expected["error"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.partial_output == expected["partial_output"]
    assert result.events[-1].to_dict()["completion_reason"] == expected["completion_reason"]
    assert result.events[-1].to_dict()["partial_output"] == expected["partial_output"]
    assert (later_calls > 0) is expected["later_guardrails_run"]


def test_max_cycles_preserves_partial_output_and_typed_reason() -> None:
    expected = _contract()["max_cycles"]
    result = Runner.run_sync(
        Agent(
            name="max-cycles-agent",
            instructions="Continue until stopped.",
            model=ScriptedLLM(steps=[LLMResponse(content=expected["partial_output"])]),
        ),
        "go",
        run_config=RunConfig(max_cycles=1),
    )

    assert result.status == AgentStatus(expected["status"])
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.partial_output == expected["partial_output"]
    assert result.events[-1].type == expected["terminal"]
    assert result.events[-1].to_dict()["completion_reason"] == expected["completion_reason"]
    assert result.events[-1].to_dict()["partial_output"] == expected["partial_output"]


def test_cancellation_has_typed_reason_and_single_terminal() -> None:
    expected = _contract()["cancellation"]
    cancellation = CancellationToken()
    cancellation.cancel("test cancellation")
    result = Runner.run_sync(
        Agent(
            name="cancelled-agent",
            instructions="This model must not be called.",
            model=ScriptedLLM(steps=[]),
        ),
        "go",
        run_config=RunConfig(cancellation_token=cancellation),
    )
    terminals = [event for event in result.events if event.type in TERMINAL_TYPES]

    assert len(terminals) == expected["terminal_count"]
    assert terminals[0].type == expected["terminal"]
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert terminals[0].to_dict()["completion_reason"] == expected["completion_reason"]


def test_cancellation_reason_precedes_output_guardrail_failure() -> None:
    cancellation = CancellationToken()
    cancellation.cancel("test cancellation")
    guardrail_calls = 0

    @output_guardrail
    def block(_context, _output):
        nonlocal guardrail_calls
        guardrail_calls += 1
        return GuardrailResult.block("guardrail also blocked")

    result = Runner.run_sync(
        Agent(
            name="cancelled-guardrail-agent",
            instructions="This model must not be called.",
            model=ScriptedLLM(steps=[]),
            output_guardrails=[block],
        ),
        "go",
        run_config=RunConfig(cancellation_token=cancellation),
    )

    assert result.status == AgentStatus.FAILED
    assert result.completion_reason == CompletionReason.CANCELLED
    assert result.final_output == result.raw_result.error
    assert "cancel" in str(result.final_output).lower()
    assert guardrail_calls == 0
    assert result.events[-1].type == "run_cancelled"
    assert result.events[-1].to_dict()["completion_reason"] == CompletionReason.CANCELLED.value


def test_budget_exhaustion_emits_observation_before_the_only_terminal() -> None:
    expected = _contract()["budget_exhausted"]
    result = Runner.run_sync(
        Agent(
            name="budget-terminal-agent",
            instructions="Return the scripted draft.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content=expected["partial_output"],
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
            ),
            no_tool_policy="finish",
        ),
        "go",
        run_config=RunConfig(budget_limits=RunBudgetLimits(max_total_tokens=10)),
    )
    terminals = [event for event in result.events if event.type in TERMINAL_TYPES]

    assert [event.type for event in result.events[-2:]] == expected["events_tail"]
    assert len(terminals) == expected["terminal_count"]
    assert terminals[0].type == expected["terminal"]
    assert result.status == AgentStatus(expected["status"])
    assert result.completion_reason == CompletionReason(expected["completion_reason"])
    assert result.completion_tool_name == expected["completion_tool_name"]
    assert result.partial_output == expected["partial_output"]
    assert result.raw_result.error == expected["error"]
    assert result.budget_usage is not None
    assert result.budget_exhaustion is not None
    assert terminals[0].to_dict()["budget_usage"] == result.budget_usage.to_dict()
    assert terminals[0].to_dict()["budget_exhaustion"] == result.budget_exhaustion.to_dict()


def test_event_store_fail_closed_is_a_normal_runner_error() -> None:
    expected = _contract()["event_store_fail_closed"]
    with pytest.raises(RuntimeError, match=str(expected["error"])):
        Runner.run_sync(
            _agent(),
            "go",
            run_config=RunConfig(event_store=_FailingStore(), event_store_fail_closed=True),
        )


class _FailingStore:
    def append(self, event: RunEvent) -> None:
        del event
        raise RuntimeError("store down")

    def replay(
        self,
        query: RunEventReplayQuery | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[RunEvent]:
        del query, run_id
        return iter(())
