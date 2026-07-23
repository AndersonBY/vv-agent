from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vv_agent import Agent, ModelSettings, RunConfig, Runner, ToolCallCompletedEvent, ToolOutputText, function_tool
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model import ScriptedModelProvider
from vv_agent.types import LLMResponse, Message, ToolCall

TRACE_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "runner_trace.jsonl"
TRACE_FIELDS = (
    "type",
    "cycle_index",
    "agent_name",
    "call_id",
    "operation_id",
    "attempt",
    "operation",
    "backend",
    "model",
    "usage",
    "delta",
    "tool_name",
    "tool_call_id",
    "arguments",
    "status",
    "directive",
    "error_code",
    "execution_started",
    "duration_ms",
    "final_output",
)


class StreamingScriptedLLM(ScriptedLLM):
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
        return self.complete_request(
            LlmRequest(
                model=model,
                messages=list(messages),
                tools=list(tools),
                metadata=dict(request_metadata or {}),
                model_settings=model_settings,
            ),
            stream_callback=stream_callback,
        )

    def complete_request(self, request: LlmRequest, *, stream_callback=None) -> LLMResponse:
        response = super().complete_request(request, stream_callback=stream_callback)
        if stream_callback is not None:
            stream_callback({"event": "assistant_delta", "content_delta": response.content})
        return response


def test_real_runner_trace_matches_current_producer_fixture() -> None:
    @function_tool
    def lookup(query: str) -> ToolOutputText:
        """Look up a deterministic fixture value."""
        return ToolOutputText(
            text=f"found:{query}",
            metadata={"producer_marker": {"nested": True}},
        )

    llm = StreamingScriptedLLM(
        steps=[
            LLMResponse(
                content="lookup",
                tool_calls=[ToolCall(id="lookup-call", name="lookup", arguments={"query": "parity"})],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="finish-call", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    agent = Agent(
        name="trace-agent",
        instructions="Use lookup then finish.",
        model="direct",
        tools=[lookup],
    )

    result = Runner.run_sync(
        agent,
        "trace this",
        run_config=RunConfig(model_provider=ScriptedModelProvider(backend="test", default_model="direct", llm=llm)),
    )
    actual = [
        _trace_projection(event.to_dict())
        for event in result.events
        if event.type not in {"agent_started", "diagnostic", "session_persisted"}
    ]
    fixture_bytes = TRACE_FIXTURE.read_bytes()
    expected = [json.loads(line) for line in fixture_bytes.decode("ascii").splitlines()]

    assert actual == expected
    event_types = {event["type"] for event in actual}
    assert "cycle_llm_response" not in event_types
    assert "agent_started" not in event_types
    assert "session_persisted" not in event_types
    assert "agent_started" in {event.type for event in result.events}
    lookup_completed = next(
        event for event in result.events if isinstance(event, ToolCallCompletedEvent) and event.tool_name == "lookup"
    )
    assert lookup_completed.metadata["metadata"]["producer_marker"] == {"nested": True}
    assert lookup_completed.metadata["tool_arguments"] == {"query": "parity"}
    assert {event["status"] for event in actual if event["type"] == "tool_call_completed"} == {"success"}


def _trace_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in TRACE_FIELDS if key in payload}
