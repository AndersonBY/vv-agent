from __future__ import annotations

from threading import Event

from vv_agent import Agent, RunConfig, Runner, function_tool
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_runner_start_yields_tool_started_before_result_is_ready(tmp_path) -> None:
    gate = Event()

    @function_tool
    def slow_tool() -> str:
        gate.wait(timeout=2)
        return "slow done"

    agent = Agent(
        name="assistant",
        instructions="Use the tool.",
        model="test-model",
        tools=[slow_tool],
    )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="calling",
                tool_calls=[ToolCall(id="call_1", name="slow_tool", arguments={})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    def model_provider(_agent: Agent, _config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(model_provider=model_provider),
    )

    first_types: list[str] = []
    for event in handle.events():
        first_types.append(event.type)
        if event.type == "tool_call_started":
            assert not handle.done()
            gate.set()
        if event.type == "run_completed":
            break

    result = handle.result(timeout=2)
    assert result.final_output == "done"
    assert "tool_call_started" in first_types


def test_stream_sync_is_backed_by_live_handle() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(steps=[LLMResponse(content="hello", tool_calls=[])])

    def model_provider(_agent: Agent, _config: RunConfig):
        return llm, _resolved_model()

    events = list(
        Runner.stream_sync(
            agent,
            "say hi",
            run_config=RunConfig(model_provider=model_provider),
        )
    )

    assert events[0].type == "run_started"
    assert events[-1].type == "run_completed"
