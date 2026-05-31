from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, Span, TraceProcessor
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


class CapturingTraceProcessor(TraceProcessor):
    def __init__(self) -> None:
        self.started: list[Span] = []
        self.ended: list[Span] = []

    def on_span_start(self, span: Span) -> None:
        self.started.append(span)

    def on_span_end(self, span: Span) -> None:
        self.ended.append(span)


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

    assert [span.name for span in processor.started] == ["run", "tool"]
    assert [span.name for span in processor.ended] == ["tool", "run"]
    run_span = processor.ended[-1]
    tool_span = processor.ended[0]
    assert run_span.trace_id == result.trace_id
    assert run_span.metadata["workflow_name"] == "trace-test"
    assert run_span.metadata["agent_name"] == "assistant"
    assert tool_span.parent_id == run_span.span_id
    assert tool_span.metadata["tool_name"] == TASK_FINISH_TOOL_NAME
    assert tool_span.ended_at is not None
