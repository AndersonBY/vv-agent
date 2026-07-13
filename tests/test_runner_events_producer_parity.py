from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, AssistantDeltaEvent, MemorySession, ModelSettings, RunConfig, Runner
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.types import LLMResponse, Message, ToolCall

RUNNER_EVENTS_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "runner_events_v1.jsonl"
RUNNER_EVENTS_FIXTURE_SHA256 = "15f23c49cac673766db17c42c96b403d2cc1ece8e876c40d772e8d198bfb8adc"


class StreamingGoldenLLM:
    model_id = "golden-model"

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, model_settings, request_metadata
        if stream_callback is not None:
            stream_callback(
                {
                    "event": "assistant_delta",
                    "content_delta": "complete ",
                    "cycle": 999,
                    "run_id": "run_spoofed",
                    "trace_id": "trace_spoofed",
                    "agent_name": "spoofed-agent",
                    "session_id": "session_spoofed",
                }
            )
            stream_callback({"event": "assistant_delta", "content_delta": "assistant message"})
        return LLMResponse(
            content="complete assistant message",
            tool_calls=[
                ToolCall(
                    id="finish_golden",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "done"},
                )
            ],
        )


def test_real_runner_events_match_cross_language_producer_fixture(tmp_path: Path) -> None:
    session = MemorySession("session_runner_parity")
    result = Runner.run_sync(
        Agent(
            name="runner-agent",
            instructions="Finish with task_finish.",
            model=StreamingGoldenLLM(),
        ),
        "golden input",
        run_config=RunConfig(
            workspace=tmp_path,
            session=session,
            tracing={"trace_id": "trace_runner_parity"},
            metadata={
                "_vv_agent_run_id": "run_spoofed",
                "_vv_agent_trace_id": "trace_spoofed",
                "_vv_agent_agent_name": "spoofed-agent",
                "_vv_agent_session_id": "session_spoofed",
            },
        ),
    )

    fixture_bytes = RUNNER_EVENTS_FIXTURE.read_bytes()
    expected = [json.loads(line) for line in fixture_bytes.decode("ascii").splitlines()]
    actual = [_normalize_event(event.to_dict()) for event in result.events]

    assert hashlib.sha256(fixture_bytes).hexdigest() == RUNNER_EVENTS_FIXTURE_SHA256
    assert actual == expected
    assert result.run_id.startswith("run_") and result.run_id != "run_spoofed"
    assert len({event.event_id for event in result.events}) == len(result.events)
    assert all(event.run_id == result.run_id for event in result.events)
    assert all(event.trace_id == "trace_runner_parity" for event in result.events)
    assert all(event.agent_name == "runner-agent" for event in result.events)
    assert all(event.session_id == "session_runner_parity" for event in result.events)

    canonical_lines = [
        json.dumps({key: actual_event[key] for key in expected_event}, separators=(",", ":"))
        for actual_event, expected_event in zip(actual, expected, strict=True)
    ]
    assert ("\n".join(canonical_lines) + "\n").encode("ascii") == fixture_bytes


def test_raw_stream_observer_failure_cannot_suppress_typed_run_handle_events(tmp_path: Path) -> None:
    raw_observer_calls = 0

    def broken_raw_observer(_payload: dict[str, Any]) -> None:
        nonlocal raw_observer_calls
        raw_observer_calls += 1
        raise RuntimeError("raw observer failed")

    with pytest.warns(RuntimeWarning, match="Runtime stream observer failed: raw observer failed"):
        handle = Runner.start(
            Agent(
                name="runner-agent",
                instructions="Finish with task_finish.",
                model=StreamingGoldenLLM(),
            ),
            "golden input",
            run_config=RunConfig(
                workspace=tmp_path,
                runtime_stream_callback=broken_raw_observer,
            ),
        )
        result = handle.result(timeout=2)

    journal = list(handle.events())
    assert result.status.value == "completed"
    assert raw_observer_calls == 2
    assert [event.delta for event in journal if isinstance(event, AssistantDeltaEvent)] == [
        "complete ",
        "assistant message",
    ]
    assert [event.event_id for event in journal] == [event.event_id for event in result.events]


def test_typed_stream_observer_failure_cannot_suppress_run_handle_journal(tmp_path: Path) -> None:
    observer_calls = 0

    def broken_typed_observer(event: Any) -> None:
        nonlocal observer_calls
        observer_calls += 1
        if event.type == "assistant_delta":
            raise RuntimeError("typed observer failed")

    with pytest.warns(RuntimeWarning, match="Run event stream observer failed: typed observer failed"):
        handle = Runner.start(
            Agent(
                name="runner-agent",
                instructions="Finish with task_finish.",
                model=StreamingGoldenLLM(),
            ),
            "golden input",
            run_config=RunConfig(workspace=tmp_path, stream=broken_typed_observer),
        )
        result = handle.result(timeout=2)

    journal = list(handle.events())
    assert result.status.value == "completed"
    assert observer_calls == len(result.events)
    assert [event.delta for event in journal if isinstance(event, AssistantDeltaEvent)] == [
        "complete ",
        "assistant message",
    ]
    assert [event.event_id for event in journal] == [event.event_id for event in result.events]


def _normalize_event(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["event_id"] = "evt_dynamic"
    normalized["run_id"] = "run_dynamic"
    normalized["created_at"] = 0.0
    normalized.pop("metadata", None)
    return normalized
