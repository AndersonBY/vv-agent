from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, MemorySession, RunConfig, Runner, output_guardrail
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.event_store import RunEventReplayQuery
from vv_agent.events import RunEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, ToolCall

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "runner_terminal_v1.json"
FIXTURE_SHA256 = "4600b26cd3313a790cb84b0a0e1981f9046027b3095450e9b2522db42292939f"
TERMINAL_TYPES = {"run_completed", "run_failed", "run_cancelled"}


def _contract() -> dict[str, Any]:
    payload = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_SHA256
    return json.loads(payload)


def _agent(*, output_guardrails=None) -> Agent:
    return Agent(
        name="terminal-agent",
        instructions="Finish.",
        model=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="finish",
                    tool_calls=[
                        ToolCall(
                            id="finish",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": "done"},
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
        _agent(output_guardrails=[block, later]),
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
    assert (later_calls > 0) is expected["later_guardrails_run"]


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
