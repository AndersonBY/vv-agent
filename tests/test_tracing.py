from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    JsonlTraceExporter,
    MemorySession,
    RunCompletedEvent,
    RunConfig,
    RunEvent,
    RunEventReplayQuery,
    Runner,
    SessionPersistedEvent,
    Span,
    TraceProcessor,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall

TRACE_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "runner_trace_spans_v1.json"
TRACE_CONTRACT_SHA256 = "c44cd892fb2cff47c34c53e7fec861bc9cc4fb07b34bcfb993706ddfb3b1530d"


def _trace_contract() -> dict[str, Any]:
    payload = TRACE_CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == TRACE_CONTRACT_SHA256
    return json.loads(payload)


class CapturingTraceProcessor(TraceProcessor):
    def __init__(self) -> None:
        self.started: list[Span] = []
        self.ended: list[Span] = []

    def on_span_start(self, span: Span) -> None:
        self.started.append(span)

    def on_span_end(self, span: Span) -> None:
        self.ended.append(span)


class RecordingRunEventStore:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def append(self, event: RunEvent) -> None:
        self.events.append(event)

    def replay(
        self,
        query: RunEventReplayQuery | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[RunEvent]:
        del query, run_id
        return iter(self.events)


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def test_runner_emits_run_and_tool_trace_spans(tmp_path: Path) -> None:
    processor = CapturingTraceProcessor()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[
                            ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})
                        ],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Answer.", model="m"),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tracing={"workflow_name": "trace-test", "processors": [processor]},
        ),
    )

    topology = _trace_contract()["topology"]
    assert [span.name for span in processor.started] == topology["start_order"]
    assert [span.name for span in processor.ended] == topology["end_order"]
    run_span = processor.ended[-1]
    agent_span = processor.ended[-2]
    tool_span = processor.ended[0]
    assert run_span.trace_id == result.trace_id
    assert run_span.metadata["workflow_name"] == "trace-test"
    assert run_span.metadata["agent_name"] == "assistant"
    assert agent_span.parent_id == run_span.span_id
    assert tool_span.parent_id == agent_span.span_id
    assert tool_span.metadata["tool_name"] == TASK_FINISH_TOOL_NAME
    assert tool_span.ended_at is not None


def test_runner_closes_trace_spans_when_provider_resolution_fails(tmp_path: Path) -> None:
    processor = CapturingTraceProcessor()

    def failing_provider(_agent: Agent, _run_config: RunConfig):
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        Runner.run_sync(
            Agent(name="assistant", instructions="Answer.", model="m"),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=failing_provider,
                tracing={"processors": [processor]},
            ),
        )

    failure = _trace_contract()["failure_cleanup"]
    assert [span.name for span in processor.started] == failure["started"]
    assert [span.name for span in processor.ended] == failure["ended"]
    assert processor.ended[-1].metadata["status"] == failure["run_status"]
    assert processor.ended[-1].metadata["error"] == "provider unavailable"


def test_typed_output_failure_occurs_after_persistence_and_completed_event(tmp_path: Path) -> None:
    processor = CapturingTraceProcessor()
    event_store = RecordingRunEventStore()
    session = MemorySession("typed-output-session")

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[
                            ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "[]"})
                        ],
                    )
                ]
            ),
            _resolved(),
        )

    with pytest.raises(ValueError, match="Expected final output JSON object"):
        Runner.run_sync(
            Agent(name="assistant", instructions="Return JSON.", model="m", output_type=dict),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=model_provider,
                session=session,
                event_store=event_store,
                tracing={"processors": [processor]},
            ),
        )

    assert session.get_items()
    terminal_sequence = [
        event.type
        for event in event_store.events
        if isinstance(event, (SessionPersistedEvent, RunCompletedEvent))
    ]
    assert terminal_sequence == ["session_persisted", "run_completed"]
    completed = next(event for event in event_store.events if isinstance(event, RunCompletedEvent))
    assert completed.status == "completed"

    run_span = next(span for span in processor.ended if span.name == "run")
    assert run_span.metadata["status"] == "failed"
    assert "Expected final output JSON object" in run_span.metadata["error"]


def test_trace_processor_failures_are_isolated_from_the_run(tmp_path: Path) -> None:
    class BrokenProcessor:
        def on_span_start(self, _span: Span) -> None:
            raise RuntimeError("start down")

        def on_span_end(self, _span: Span) -> None:
            raise RuntimeError("end down")

        def flush(self) -> None:
            raise RuntimeError("flush down")

    with pytest.warns(RuntimeWarning) as warnings:
        result = Runner.run_sync(
            Agent(
                name="assistant",
                instructions="Answer.",
                model=ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="done",
                            tool_calls=[
                                ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})
                            ],
                        )
                    ]
                ),
            ),
            "go",
            run_config=RunConfig(workspace=tmp_path, tracing={"processors": [BrokenProcessor()]}),
        )

    assert result.status.value == _trace_contract()["sink_failure"]["run_status"]
    assert any("on_span_start failed" in str(warning.message) for warning in warnings)
    assert any("on_span_end failed" in str(warning.message) for warning in warnings)
    assert any("flush failed" in str(warning.message) for warning in warnings)


def test_jsonl_trace_exporter_uses_the_shared_span_wire(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    exporter = JsonlTraceExporter(path)
    span = Span(name="run", trace_id="trace_1", span_id="span_1", started_at=123.0)

    exporter.on_span_start(span)
    exporter.on_span_end(span.finish({"status": "completed"}))
    exporter.flush()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "span_start"
    assert records[0]["span"]["trace_id"] == "trace_1"
    assert records[1]["event"] == "span_end"
    assert records[1]["span"]["metadata"] == {"status": "completed"}
