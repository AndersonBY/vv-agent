from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vv_agent import (
    Agent,
    AgentStartedEvent,
    AssistantDeltaEvent,
    LLMStartedEvent,
    MemorySession,
    ModelSettings,
    RunCompletedEvent,
    RunConfig,
    Runner,
    RunResult,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall


def _fake_resolved(*, backend: str = "test", model: str = "m") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
    )


def test_runner_run_sync_executes_agent_with_model_provider(tmp_path: Path) -> None:
    seen_settings: list[ModelSettings] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        assert agent.name == "assistant"
        assert run_config.workspace == tmp_path
        seen_settings.append(agent.model_settings.resolve(run_config.model_settings))
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                    )
                ]
            ),
            _fake_resolved(model="override-model"),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Be concise.",
            model="base-model",
            model_settings=ModelSettings(temperature=0.1, max_tokens=100),
        ),
        "Say ok.",
        run_config=RunConfig(
            model="override-model",
            model_settings=ModelSettings(max_tokens=200),
            workspace=tmp_path,
            model_provider=model_provider,
        ),
    )

    assert isinstance(result, RunResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "ok"
    assert result.input == "Say ok."
    assert result.raw_result.final_answer == "ok"
    assert seen_settings == [ModelSettings(temperature=0.1, max_tokens=200)]


def test_runner_passes_resolved_model_settings_to_llm_complete(tmp_path: Path) -> None:
    seen_settings: list[ModelSettings | None] = []

    class CapturingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
        ) -> LLMResponse:
            del model, messages, tools, stream_callback
            seen_settings.append(model_settings)
            return LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return CapturingLLM(), _fake_resolved()

    Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use resolved settings.",
            model="m",
            model_settings=ModelSettings(temperature=0.2, max_tokens=100),
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            model_settings=ModelSettings(max_tokens=250, top_p=0.8),
        ),
    )

    assert seen_settings == [ModelSettings(temperature=0.2, top_p=0.8, max_tokens=250)]


def test_runner_stream_sync_yields_typed_events(tmp_path: Path) -> None:
    class StreamingLLM:
        def complete(self, *, model: str, messages: list[Message], tools: list[dict[str, object]], stream_callback=None):
            del model, messages, tools
            if stream_callback is not None:
                stream_callback({"event": "assistant_delta", "content_delta": "hel"})
                stream_callback({"event": "assistant_delta", "content_delta": "lo"})
            return LLMResponse(
                content="hello",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return StreamingLLM(), _fake_resolved()

    events = list(
        Runner.stream_sync(
            Agent(name="assistant", instructions="Stream.", model="m"),
            "go",
            run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
        )
    )

    assert [event.type for event in events] == [
        "run_started",
        "agent_started",
        "llm_started",
        "assistant_delta",
        "assistant_delta",
        "tool_started",
        "tool_finished",
        "run_completed",
    ]
    assert [event.delta for event in events if isinstance(event, AssistantDeltaEvent)] == ["hel", "lo"]
    completed = events[-1]
    assert isinstance(completed, RunCompletedEvent)
    assert completed.final_output == "done"
    assert completed.to_dict()["type"] == "run_completed"
    assert isinstance(events[1], AgentStartedEvent)
    assert isinstance(events[2], LLMStartedEvent)


def test_runner_appends_session_items_across_runs(tmp_path: Path) -> None:
    session = MemorySession("thread-1")
    calls: list[list[Message]] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config

        def respond(model: str, messages: list[Message]) -> LLMResponse:
            del model
            calls.append(list(messages))
            return LLMResponse(
                content="done",
                tool_calls=[ToolCall(id=f"c{len(calls)}", name=TASK_FINISH_TOOL_NAME, arguments={"message": f"ok-{len(calls)}"})],
            )

        return ScriptedLLM(steps=[respond]), _fake_resolved()

    agent = Agent(name="assistant", instructions="Remember context.", model="m")
    config = RunConfig(workspace=tmp_path, session=session, model_provider=model_provider)

    first = Runner.run_sync(agent, "first", run_config=config)
    second = Runner.run_sync(agent, "second", run_config=config)

    assert first.final_output == "ok-1"
    assert second.final_output == "ok-2"
    assert [item.content for item in session.get_items()] == ["first", "done", "second", "done"]
    assert [message.content for message in calls[1] if message.role == "user"] == ["first", "second"]


def test_runner_passes_tool_context_to_function_tool(tmp_path: Path) -> None:
    from vv_agent import ToolContext, function_tool

    @function_tool
    def report_context(context: ToolContext, value: str) -> str:
        """Report call context."""
        return f"{context.tool_name}:{context.tool_call_id}:{context.arguments['value']}:{value}"

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="call",
                        tool_calls=[ToolCall(id="tool-call-1", name="report_context", arguments={"value": "42"})],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use the tool.",
            model="m",
            tools=[report_context],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.final_output == "report_context:tool-call-1:42:42"


def test_runner_coerces_json_final_output_to_dataclass(tmp_path: Path) -> None:
    @dataclass
    class Summary:
        title: str
        count: int

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[
                            ToolCall(
                                id="finish",
                                name=TASK_FINISH_TOOL_NAME,
                                arguments={"message": '{"title": "orders", "count": 3}'},
                            )
                        ],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Return JSON.", model="m", output_type=Summary),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.final_output == Summary(title="orders", count=3)
