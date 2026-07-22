from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    AssistantDeltaEvent,
    DiagnosticEvent,
    MemorySession,
    ModelSettings,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    ReasoningDeltaEvent,
    RunCompletedEvent,
    RunConfig,
    Runner,
    ToolCallStartedEvent,
)
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.model import ScriptedModelProvider
from vv_agent.types import LLMResponse, Message, ToolCall

RUNNER_EVENTS_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "runner_events.jsonl"
STREAM_PROJECTION_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "llm_stream_projection.json"


def _provider(model: str, llm: Any) -> ScriptedModelProvider:
    return ScriptedModelProvider(backend="test", default_model=model, llm=llm)


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


class ContractStreamLLM:
    model_id = "stream-model"

    def __init__(self, raw_events: list[dict[str, Any]]) -> None:
        self.raw_events = raw_events
        self.calls = 0

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
        self.calls += 1
        if self.calls < 3:
            return LLMResponse(content=f"draft {self.calls}")

        assert stream_callback is not None
        for event in self.raw_events:
            stream_callback(dict(event))
        return LLMResponse(
            content="done",
            tool_calls=[
                ToolCall(
                    id="call_stream",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "done"},
                )
            ],
        )


def test_real_runner_projects_contract_stream_fixture_without_trusting_source_identity(tmp_path: Path) -> None:
    fixture_bytes = STREAM_PROJECTION_FIXTURE.read_bytes()
    contract = json.loads(fixture_bytes)
    synthetic = contract["synthetic_top_level"]
    provider_payloads = list(synthetic["provider_payloads"])
    llm = ContractStreamLLM(provider_payloads)
    callback_order: list[str] = []
    typed_wire_types = {mapping["wire_type"] for mapping in contract["mappings"].values()}

    def typed_observer(event: Any) -> None:
        if event.type in typed_wire_types:
            callback_order.append(event.type)

    result = Runner.run_sync(
        Agent(name="stream-agent", instructions="Finish on the third cycle.", model="stream-model"),
        "stream input",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_provider("stream-model", llm),
            session=MemorySession("session_stream_parity"),
            max_cycles=3,
            no_tool_policy="continue",
            tracing={"trace_id": "trace_stream_parity"},
            stream=typed_observer,
        ),
    )

    typed_events = [event for event in result.events if event.type in typed_wire_types]
    actual = [_normalize_event(event.to_dict()) for event in typed_events]

    assert actual == synthetic["expected_wire_events"]
    assert llm.calls == synthetic["context"]["cycle_index"] == 3
    assert len(typed_events) == synthetic["typed_event_count"] == 4
    assert [type(event) for event in typed_events] == [
        AssistantDeltaEvent,
        ReasoningDeltaEvent,
        ModelToolCallStartedEvent,
        ModelToolCallProgressEvent,
    ]
    assert callback_order == [
        "assistant_delta",
        "reasoning_delta",
        "model_tool_call_started",
        "model_tool_call_progress",
    ]
    assert all(event.run_id == result.run_id for event in typed_events)
    assert all(event.cycle_index == 3 for event in typed_events)

    execution_events = [event for event in result.events if isinstance(event, ToolCallStartedEvent)]
    assert len(execution_events) == 1
    assert execution_events[0].type == synthetic["execution_event_type"]
    assert execution_events[0].tool_call_id == "call_stream"
    assert result.events.index(execution_events[0]) > result.events.index(typed_events[-1])
    terminals = [event for event in result.events if isinstance(event, RunCompletedEvent)]
    assert len(terminals) == 1
    assert terminals[0].final_output == "done"


@pytest.mark.parametrize(
    "malformed_event",
    [
        {"type": "assistant_delta", "content_delta": "legacy discriminator"},
        {"event": "assistant_delta", "content_delta": 7},
        {"event": "reasoning_delta", "reasoning_delta": None},
        {"event": "tool_call_started", "tool_call_id": "", "function_name": "task_finish"},
        {
            "event": "tool_call_progress",
            "tool_call_id": "call_stream",
            "function_name": "task_finish",
            "arguments_chars": -1,
        },
    ],
)
def test_real_runner_drops_malformed_known_provider_stream_payloads(
    tmp_path: Path,
    malformed_event: dict[str, Any],
) -> None:
    llm = ContractStreamLLM([malformed_event])
    observed: list[Any] = []

    result = Runner.run_sync(
        Agent(name="stream-agent", instructions="Finish on the third cycle.", model="stream-model"),
        "stream input",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_provider("stream-model", llm),
            max_cycles=3,
            no_tool_policy="continue",
            stream=observed.append,
        ),
    )

    assert result.status.value == "completed"
    assert not any(
        isinstance(
            event,
            (
                AssistantDeltaEvent,
                ReasoningDeltaEvent,
                ModelToolCallStartedEvent,
                ModelToolCallProgressEvent,
            ),
        )
        for event in [*result.events, *observed]
    )


def test_real_runner_events_match_cross_language_producer_fixture(tmp_path: Path) -> None:
    session = MemorySession("session_runner_parity")
    llm = StreamingGoldenLLM()
    result = Runner.run_sync(
        Agent(
            name="runner-agent",
            instructions="Finish with task_finish.",
            model="golden-model",
        ),
        "golden input",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_provider("golden-model", llm),
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
    stable_events = [event for event in result.events if not isinstance(event, DiagnosticEvent)]
    actual = [_normalize_event(event.to_dict()) for event in stable_events]

    assert actual == expected
    assert result.run_id.startswith("run_") and result.run_id != "run_spoofed"
    assert len({event.event_id for event in result.events}) == len(result.events)
    assert all(event.run_id == result.run_id for event in result.events)
    assert all(event.trace_id == "trace_runner_parity" for event in result.events)
    assert all(event.agent_name == "runner-agent" for event in result.events)
    assert all(event.session_id == "session_runner_parity" for event in result.events)
    assert any(event.code == "cycle_llm_response" for event in result.events if isinstance(event, DiagnosticEvent))

    canonical_lines = [
        json.dumps({key: actual_event[key] for key in expected_event}, separators=(",", ":"))
        for actual_event, expected_event in zip(actual, expected, strict=True)
    ]
    assert ("\n".join(canonical_lines) + "\n").encode("ascii") == fixture_bytes


def test_typed_stream_observer_failure_cannot_suppress_run_handle_journal(tmp_path: Path) -> None:
    observer_calls = 0

    def broken_typed_observer(event: Any) -> None:
        nonlocal observer_calls
        observer_calls += 1
        if event.type == "assistant_delta":
            raise RuntimeError("typed observer failed")

    with pytest.warns(RuntimeWarning, match="Run event stream observer failed: typed observer failed"):
        llm = StreamingGoldenLLM()
        handle = Runner.start(
            Agent(
                name="runner-agent",
                instructions="Finish with task_finish.",
                model="golden-model",
            ),
            "golden input",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=_provider("golden-model", llm),
                stream=broken_typed_observer,
            ),
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
    if normalized.get("duration_ms") is not None:
        normalized["duration_ms"] = 0
    normalized.pop("metadata", None)
    return normalized
