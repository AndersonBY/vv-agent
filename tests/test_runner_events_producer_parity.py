from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from support.kernel_runtime import start_runner

from vv_agent import (
    Agent,
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    ReasoningDeltaEvent,
    RunCompletedEvent,
    RunConfig,
    Runner,
    ToolCallStartedEvent,
    function_tool,
)
from vv_agent.llm import LlmRequest
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall

RUNNER_EVENTS_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "runner_events.jsonl"
STREAM_PROJECTION_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "llm_stream_projection.json"


def _provider(model: str, llm: Any) -> ScriptedModelProvider:
    return ScriptedModelProvider(backend="test", default_model=model, llm=llm)


class StreamingGoldenLLM:
    model_id = "golden-model"

    def complete(self, request: LlmRequest) -> LLMResponse:
        return self._respond(request, None)

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        return self._respond(request, stream_callback)

    @staticmethod
    def _respond(request: LlmRequest, stream_callback) -> LLMResponse:
        del request
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
        return LLMResponse(content="complete assistant message")


class ContractStreamLLM:
    model_id = "stream-model"

    def __init__(self, raw_events: list[dict[str, Any]]) -> None:
        self.raw_events = raw_events
        self.calls = 0

    def complete(self, request: LlmRequest) -> LLMResponse:
        return self._respond(request, None)

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        return self._respond(request, stream_callback)

    def _respond(self, request: LlmRequest, stream_callback) -> LLMResponse:
        del request
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
                    name="echo",
                    arguments={"message": "done"},
                )
            ],
        )


def _stream_agent() -> Agent:
    @function_tool
    def echo(message: str) -> str:
        """Return the supplied message."""
        return message

    return Agent(
        name="stream-agent",
        instructions="Return the third-cycle tool result.",
        model="stream-model",
        tools=[echo],
        tool_use_behavior="stop_on_first_tool",
    )


def test_real_runner_projects_contract_stream_fixture_without_trusting_source_identity(tmp_path: Path) -> None:
    fixture_bytes = STREAM_PROJECTION_FIXTURE.read_bytes()
    contract = json.loads(fixture_bytes)
    synthetic = contract["synthetic_top_level"]
    provider_payloads = list(synthetic["provider_payloads"])
    llm = ContractStreamLLM(provider_payloads)
    callback_order: list[str] = []
    projected: list[Any] = []
    typed_wire_types = {mapping["wire_type"] for mapping in contract["mappings"].values()}

    def typed_observer(event: Any) -> None:
        if event.type in typed_wire_types:
            callback_order.append(event.type)
            projected.append(event)

    driver = SessionDriver()
    result = start_runner(
        driver,
        "c1c/stream",
        _stream_agent(),
        "stream input",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_provider("stream-model", llm),
            max_cycles=3,
            no_tool_policy="continue",
            stream=typed_observer,
        ),
    )

    result = result.result()
    driver.close()
    typed_events = projected
    actual = [_normalize_event(event.to_dict()) for event in typed_events]

    assert actual == [_normalize_event(e) for e in synthetic["expected_wire_events"]]
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
    assert not any(event.type in typed_wire_types for event in result.events)
    terminals = [event for event in result.events if isinstance(event, RunCompletedEvent)]
    assert len(terminals) == 1
    assert terminals[0].final_output == "done"


@pytest.mark.parametrize(
    "malformed_event",
    [
        {"type": "assistant_delta", "content_delta": "legacy discriminator"},
        {"event": "assistant_delta", "content_delta": 7},
        {"event": "reasoning_delta", "reasoning_delta": None},
        {"event": "tool_call_started", "tool_call_id": "", "function_name": "echo"},
        {
            "event": "tool_call_progress",
            "tool_call_id": "call_stream",
            "function_name": "echo",
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
        _stream_agent(),
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
    from vv_agent import RunBudgetLimits
    from vv_agent.runtime.hooks import BaseRuntimeHook

    class Hooks(BaseRuntimeHook):
        def before_tool_call(self, event):
            event.context.shared_state["prepared"] = True

    class AfterCycle:
        def after_cycle(self, snapshot):
            return None

    @function_tool
    def echo(text: str) -> str:
        return text

    with closing(SessionDriver()) as driver:
        result = start_runner(
            driver,
            "tools",
            Agent("fixture", "Be precise.", model="m", tools=[echo]),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.new(
                    "scripted", "m", [LLMResponse("", [ToolCall("echo", "echo", {"text": "ok"})]), LLMResponse("done")]
                ),
                budget_limits=RunBudgetLimits(max_total_tokens=100),
                hooks=[Hooks()],
                after_cycle_hooks=[AfterCycle()],
            ),
        ).result()
    expected = [json.loads(line) for line in RUNNER_EVENTS_FIXTURE.read_text().splitlines()]
    by_id = {e.event_id: e.to_dict() for e in result.events}
    assert [_normalize_event(by_id[e["event_id"]]) for e in expected] == [_normalize_event(e) for e in expected]
    assert result.run_id == "tools/turn/initial"
    assert len({e.event_id for e in result.events}) == len(result.events)
    assert all(e.session_id == "tools" for e in result.events)


def test_typed_stream_observer_failure_cannot_suppress_run_handle_journal(tmp_path: Path) -> None:
    observer_calls = []

    def observer(event):
        observer_calls.append(event)
        if event.type == "assistant_delta":
            raise RuntimeError("typed observer failed")

    handle = Runner.start(
        Agent("runner-agent", "Answer.", model="golden-model"),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=_provider("golden-model", StreamingGoldenLLM()), stream=observer),
    )
    result = handle.result(timeout=2)
    journal = list(handle.events())
    assert result.status.value == "completed"
    assert [e.delta for e in observer_calls if isinstance(e, AssistantDeltaEvent)] == ["complete ", "assistant message"]
    assert [e.delta for e in journal if isinstance(e, AssistantDeltaEvent)] == ["complete ", "assistant message"]
    assert not any(isinstance(e, AssistantDeltaEvent) for e in result.events)
    assert [e.event_id for e in journal if not isinstance(e, AssistantDeltaEvent)] == [e.event_id for e in result.events]


def _normalize_event(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["event_id"] = "evt_dynamic"
    normalized["run_id"] = "run_dynamic"
    normalized["created_at"] = 0.0
    normalized.pop("trace_id", None)
    normalized.pop("session_id", None)
    if normalized.get("duration_ms") is not None:
        normalized["duration_ms"] = 0
    if "budget_usage" in normalized:
        normalized["budget_usage"] = normalized["budget_usage"] | {"elapsed_ms": 0}
    normalized.pop("metadata", None)
    return normalized
