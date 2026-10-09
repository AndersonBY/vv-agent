from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from support import FixedModelProvider

from vv_agent import (
    Agent,
    JsonlTraceExporter,
    RunCompletedEvent,
    RunConfig,
    RunEvent,
    RunEventReplayQuery,
    Runner,
    Span,
    TraceProcessor,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.model import ModelRef
from vv_agent.types import LLMResponse, ToolCall

TRACE_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "runner_trace_spans.json"


def _trace_contract() -> dict[str, Any]:
    return json.loads(TRACE_CONTRACT_PATH.read_bytes())


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

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Answer.", model="m"),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=FixedModelProvider(
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="update todos",
                            tool_calls=[ToolCall(id="todo", name="todo_write", arguments={"todos": []})],
                        ),
                        LLMResponse(content="ok"),
                    ]
                ),
                _resolved(),
            ),
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
    assert run_span.metadata["workflow_name"] is None
    assert run_span.metadata["agent_name"] == "assistant"
    assert agent_span.parent_id == run_span.span_id
    assert tool_span.parent_id == agent_span.span_id
    assert tool_span.metadata["tool_name"] == "todo_write"
    assert tool_span.ended_at is not None


def test_runner_closes_trace_spans_when_provider_resolution_fails(tmp_path: Path) -> None:
    processor = CapturingTraceProcessor()

    class FailingModelProvider(FixedModelProvider):
        def resolve(self, model: ModelRef) -> ResolvedModelConfig:
            del model
            raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        Runner.run_sync(
            Agent(name="assistant", instructions="Answer.", model="m"),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=FailingModelProvider(ScriptedLLM(steps=[]), _resolved()),
                tracing={"processors": [processor]},
            ),
        )

    assert processor.started == processor.ended == []


def test_typed_output_failure_is_a_retained_failed_terminal(tmp_path: Path) -> None:
    from support.kernel_runtime import start_runner

    from vv_agent.events import RunFailedEvent
    from vv_agent.session.surfaces import SessionDriver

    processor = CapturingTraceProcessor()
    event_store = RecordingRunEventStore()
    driver = SessionDriver()
    try:
        result = start_runner(
            driver,
            "typed-output-session",
            Agent("assistant", "Return JSON.", model="m", output_type=dict),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=FixedModelProvider(ScriptedLLM([LLMResponse("[]")]), _resolved()),
                event_store=event_store,
                tracing={"processors": [processor]},
            ),
        ).result()
        assert result.status.value == "failed"
        assert "Expected final output JSON object" in result.raw_result.error["message"]
        records = driver.store.read_state(result.raw_result.session_id)[1]
        assert [r.record.payload["status"] for r in records if r.record.kind == "turn_ended"] == ["failed"]
        assert any(isinstance(e, RunFailedEvent) for e in event_store.events)
        assert not any(isinstance(e, RunCompletedEvent) for e in event_store.events)
        assert processor.ended[-1].metadata["status"] == "failed"
    finally:
        driver.close()


def test_trace_processor_failures_are_isolated_from_the_run(tmp_path: Path) -> None:
    class BrokenProcessor:
        def on_span_start(self, _span: Span) -> None:
            raise RuntimeError("start down")

        def on_span_end(self, _span: Span) -> None:
            raise RuntimeError("end down")

        def flush(self) -> None:
            raise RuntimeError("flush down")

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Answer.",
            model="m",
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=FixedModelProvider(
                ScriptedLLM(steps=[LLMResponse(content="ok")]),
                _resolved(),
            ),
            tracing={"processors": [BrokenProcessor()]},
        ),
    )

    assert result.status.value == _trace_contract()["sink_failure"]["run_status"]


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
