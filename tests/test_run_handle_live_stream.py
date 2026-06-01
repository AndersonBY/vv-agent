from __future__ import annotations

from threading import Event, Thread

import pytest

from vv_agent import Agent, GuardrailResult, RunConfig, Runner, function_tool, input_guardrail
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def _finish_llm(message: str = "done") -> ScriptedLLM:
    return ScriptedLLM(
        steps=[
            LLMResponse(
                content=message,
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
            )
        ]
    )


def test_runner_start_yields_tool_started_and_result(tmp_path) -> None:
    gate = Event()

    @function_tool
    def slow_tool() -> str:
        gate.wait(timeout=5)
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
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(model_provider=model_provider),
    )

    first_types: list[str] = []
    for event in handle.events():
        first_types.append(event.type)
        if event.type == "tool_call_started" and getattr(event, "tool_name", "") == "slow_tool":
            try:
                assert not handle.done()
                with pytest.raises(TimeoutError):
                    handle.result(timeout=0.05)
            finally:
                gate.set()
        if event.type == "run_completed":
            break

    result = handle.result(timeout=2)
    assert result.final_output == "done"
    assert "tool_call_started" in first_types


def test_run_handle_state_reports_completed_result() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    def model_provider(agent: Agent, run_config: RunConfig):
        return _finish_llm("ok"), _resolved_model()

    handle = Runner.start(agent, "say hi", run_config=RunConfig(model_provider=model_provider))
    assert handle.result(timeout=2).status == AgentStatus.COMPLETED

    state = handle.state()
    assert state.status == "completed"
    assert state.done is True
    assert state.cancelled is False


def test_run_handle_state_reports_failed_result_from_guardrail() -> None:
    @input_guardrail
    def reject(_ctx, _input_text: str) -> GuardrailResult:
        return GuardrailResult.block("blocked")

    agent = Agent(
        name="assistant",
        instructions="Answer.",
        model="test-model",
        input_guardrails=[reject],
    )
    handle = Runner.start(agent, "say hi")
    assert handle.result(timeout=2).status == AgentStatus.FAILED

    state = handle.state()
    assert state.status == "failed"
    assert state.done is True
    assert state.cancelled is False


def test_runner_start_preserves_default_no_tool_continue_policy() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="first", tool_calls=[]),
            LLMResponse(content="second", tool_calls=[]),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(model_provider=model_provider, max_cycles=2),
    )
    result = handle.result(timeout=2)

    assert result.status == AgentStatus.MAX_CYCLES
    assert result.final_output == "Reached max cycles without finish signal."
    assert handle.state().status == "max_cycles"


def test_run_handle_state_keeps_result_status_when_cancel_was_requested() -> None:
    ready = Event()
    handle_ref = {}

    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    def model_provider(agent: Agent, run_config: RunConfig):
        ready.wait(timeout=2)
        return _finish_llm("ok"), _resolved_model()

    def stream(event) -> None:
        if event.type == "run_completed":
            assert handle_ref["handle"].cancel()

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(model_provider=model_provider, stream=stream),
    )
    handle_ref["handle"] = handle
    ready.set()

    assert handle.result(timeout=2).status == AgentStatus.COMPLETED
    state = handle.state()
    assert state.status == "completed"
    assert state.cancelled is True


def test_stream_sync_is_backed_by_live_handle() -> None:
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
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    stream = Runner.stream_sync(
        agent,
        "go",
        run_config=RunConfig(model_provider=model_provider),
    )
    seen_tool_started = Event()
    stream_finished = Event()
    events = []
    errors: list[BaseException] = []

    def consume_stream() -> None:
        try:
            for event in stream:
                events.append(event)
                if event.type == "tool_call_started":
                    seen_tool_started.set()
        except BaseException as exc:
            errors.append(exc)
        finally:
            stream_finished.set()

    consumer = Thread(target=consume_stream)
    consumer.start()
    try:
        assert seen_tool_started.wait(timeout=0.5)
        assert not stream_finished.is_set()
    finally:
        gate.set()
        consumer.join(timeout=2)

    assert events[0].type == "run_started"
    assert not consumer.is_alive()
    assert errors == []
    assert events[-1].type == "run_completed"


def test_stream_sync_raises_worker_exception_after_yielding_events() -> None:
    agent = Agent(name="assistant", instructions="Return JSON.", model="test-model", output_type=dict)

    def model_provider(agent: Agent, run_config: RunConfig):
        return _finish_llm("not json"), _resolved_model()

    events = []
    with pytest.raises(ValueError):
        for event in Runner.stream_sync(
            agent,
            "say hi",
            run_config=RunConfig(model_provider=model_provider),
        ):
            events.append(event)

    assert events[0].type == "run_started"
    assert events[-1].type == "run_completed"
