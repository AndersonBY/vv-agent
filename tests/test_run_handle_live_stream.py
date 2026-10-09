from __future__ import annotations

import json
import threading
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from support import FixedModelProvider

from vv_agent import Agent, GuardrailResult, RunConfig, Runner, function_tool, input_guardrail
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings
from vv_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall

RUN_HANDLE_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_handle.json"


def _run_handle_contract() -> dict[str, Any]:
    return json.loads(RUN_HANDLE_FIXTURE.read_bytes())


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
    return ScriptedLLM(steps=[LLMResponse(content=message)])


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
            LLMResponse(content="done"),
        ]
    )

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model())),
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

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(model_provider=FixedModelProvider(_finish_llm("ok"), _resolved_model())),
    )
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
    handle = Runner.start(
        agent, "say hi", run_config=RunConfig(model_provider=FixedModelProvider(_finish_llm(), _resolved_model()))
    )
    assert handle.result(timeout=2).status == AgentStatus.FAILED

    state = handle.state()
    assert state.status == "failed"
    assert state.done is True
    assert state.cancelled is False


def test_runner_start_preserves_default_no_tool_finish_policy() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="first", tool_calls=[]),
            LLMResponse(content="second", tool_calls=[]),
        ]
    )

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model()), max_cycles=2),
    )
    result = handle.result(timeout=2)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "first"
    assert len(result.result.cycles) == 1
    assert handle.state().status == "completed"


def test_completed_result_wins_over_late_cancel_request() -> None:
    ready = Event()
    handle_ref = {}

    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    def finish_when_ready(_request) -> LLMResponse:
        ready.wait(timeout=2)
        return LLMResponse(content="ok")

    def stream(event) -> None:
        if event.type == "run_completed":
            assert handle_ref["handle"].cancel() is False

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(
            model_provider=FixedModelProvider(ScriptedLLM(steps=[finish_when_ready]), _resolved_model()),
            stream=stream,
        ),
    )
    handle_ref["handle"] = handle
    ready.set()

    assert handle.result(timeout=2).status == AgentStatus.COMPLETED
    state = handle.state()
    assert state.status == "completed"
    assert state.cancelled is False


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
            LLMResponse(content="done"),
        ]
    )

    stream = Runner.stream_sync(
        agent,
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model())),
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


def test_stream_sync_yields_typed_output_failure_terminal():
    agent = Agent("assistant", "Return JSON.", model="test-model", output_type=dict)
    events = list(
        Runner.stream_sync(
            agent, "say hi", run_config=RunConfig(model_provider=FixedModelProvider(_finish_llm("not json"), _resolved_model()))
        )
    )
    assert events[0].type == "run_started"
    assert events[-1].type == "run_failed"
    assert not any(e.type == "run_completed" for e in events)


class _BurstStreamingLLM:
    def __init__(self, gate: Event, event_count: int) -> None:
        self.gate = gate
        self.event_count = event_count

    def complete(self, request: LlmRequest) -> LLMResponse:
        return self._respond(request, None)

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        return self._respond(request, stream_callback)

    def _respond(self, request: LlmRequest, stream_callback) -> LLMResponse:
        del request
        assert self.gate.wait(timeout=2)
        assert stream_callback is not None
        for index in range(self.event_count):
            stream_callback({"event": "assistant_delta", "content_delta": str(index)})
        return LLMResponse(content="done")


def test_run_handle_subscribers_are_independent_and_lossless_after_live_capacity() -> None:
    contract = _run_handle_contract()
    event_count = contract["subscribers"]["burst_event_count"]
    gate = Event()
    llm = _BurstStreamingLLM(gate, event_count)

    handle = Runner.start(
        Agent(name="burst", instructions="Finish.", model="test-model"),
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model())),
    )
    first = handle.events()
    second = handle.events()
    gate.set()
    result = handle.result(timeout=3)
    first_events = list(first)
    second_events = list(second)

    assert contract["subscribers"]["independent"] is True
    assert contract["subscribers"]["start_from_complete_backlog"] is True
    assert contract["subscribers"]["lossless_after_live_capacity"] is True
    assert [event.event_id for event in first_events] == [event.event_id for event in second_events]
    assert [
        event.event_id
        for event in first_events
        if event.type not in {"assistant_delta", "reasoning_delta", "model_tool_call_started", "model_tool_call_progress"}
    ] == [event.event_id for event in result.events]
    assert sum(event.type == "assistant_delta" for event in first_events) == event_count


class _BlockingCancellationLLM:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def complete(self, request: LlmRequest) -> LLMResponse:
        del request
        self.started.set()
        assert self.release.wait(timeout=3)
        return LLMResponse(content="should be cancelled")

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        del stream_callback
        return self.complete(request)


def test_run_handle_cancel_accepted_state_and_terminal_reason_match_fixture(tmp_path):
    contract = _run_handle_contract()["cancellation"]
    llm = _BlockingCancellationLLM()
    handle = Runner.start(
        Agent("cancel", "Wait.", model="test-model"),
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model()), workspace=tmp_path),
    )
    assert llm.started.wait(2)
    assert handle.cancel(contract["reason"]) is contract["accepted"]
    assert handle.state().status == "running" and not handle.state().done
    assert handle.cancel() is contract["repeated_request_accepted"]
    llm.release.set()
    result = handle.result(3)
    assert (
        result.status is AgentStatus.FAILED
        and result.completion_reason is not None
        and result.completion_reason.value == "cancelled"
    )
    assert handle.state().status == contract["terminal_status"]
    assert handle.state().done and handle.state().cancelled
    assert handle.cancel() is contract["late_request_accepted"]


class _AsyncChildStreamingLLM:
    def __init__(self) -> None:
        self.child_started = Event()
        self.release_child = Event()
        self.parent_calls = 0
        self.lock = threading.Lock()

    def complete(self, request: LlmRequest) -> LLMResponse:
        if request.messages and request.messages[0].role == "system" and request.messages[0].content == "Child prompt":
            self.child_started.set()
            assert self.release_child.wait(timeout=3)
            return LLMResponse(content="child done")
        with self.lock:
            self.parent_calls += 1
            parent_call = self.parent_calls
        if parent_call == 1:
            return LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="delegate",
                        name="create_sub_task",
                        arguments={
                            "agent_id": "researcher",
                            "task_description": "Finish after parent",
                            "wait_for_completion": False,
                        },
                    )
                ],
            )
        return LLMResponse(content="parent done")

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        del stream_callback
        return self.complete(request)


class _AsyncChildModelProvider:
    def __init__(self, llm: _AsyncChildStreamingLLM) -> None:
        self.llm = llm

    def resolve(self, model: ModelRef) -> ResolvedModelConfig:
        return _resolved_model(model.model())

    def client(self, resolved: ResolvedModelConfig) -> _AsyncChildStreamingLLM:
        del resolved
        return self.llm

    def default_settings(self, resolved: ResolvedModelConfig) -> ModelSettings:
        del resolved
        return ModelSettings()

    def default_model_ref(self) -> ModelRef:
        return ModelRef.named("test-model")


def test_run_handle_parent_terminal_is_retained_and_child_delivery_is_separate(tmp_path):
    from support.kernel_runtime import start_runner

    from vv_agent import ScriptedModelProvider
    from vv_agent.session.surfaces import SessionDriver

    contract = _run_handle_contract()["completion"]
    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "parent",
            Agent("parent", "Delegate.", model="m", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.from_steps(
                    "test",
                    "m",
                    [
                        LLMResponse(
                            "",
                            [
                                ToolCall(
                                    "delegate",
                                    "create_sub_task",
                                    {"agent_id": "worker", "task_description": "work", "wait_for_completion": False},
                                )
                            ],
                        ),
                        # Child completion can also require one more parent model call.
                        LLMResponse("done"),
                        LLMResponse("done"),
                        LLMResponse("done"),
                    ],
                ),
            ),
        )
        result = handle.result(3)
        assert result.status is AgentStatus.COMPLETED and handle.done()
        assert contract["parent_result_is_retained"] and contract["child_terminal_delivery_is_separate"]
        for child in driver.handles:
            child.join(3)
        assert handle.result().events == result.events
        assert handle.cancel() is False
    finally:
        driver.close()
