from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from support import FixedModelProvider

from vv_agent import Agent, GuardrailResult, RunConfig, Runner, function_tool, input_guardrail
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message, SubAgentConfig, ToolCall

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

    handle = Runner.start(
        agent,
        "say hi",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model()), max_cycles=2),
    )
    result = handle.result(timeout=2)

    assert result.status == AgentStatus.MAX_CYCLES
    assert result.final_output == "Reached max cycles without finish signal."
    assert handle.state().status == "max_cycles"


def test_completed_result_wins_over_late_cancel_request() -> None:
    ready = Event()
    handle_ref = {}

    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    def finish_when_ready(_request) -> LLMResponse:
        ready.wait(timeout=2)
        return LLMResponse(
            content="ok",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )

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
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
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


def test_stream_sync_raises_worker_exception_after_yielding_events() -> None:
    agent = Agent(name="assistant", instructions="Return JSON.", model="test-model", output_type=dict)

    events = []
    with pytest.raises(ValueError):
        for event in Runner.stream_sync(
            agent,
            "say hi",
            run_config=RunConfig(model_provider=FixedModelProvider(_finish_llm("not json"), _resolved_model())),
        ):
            events.append(event)

    assert events[0].type == "run_started"
    assert events[-1].type == "run_completed"


class _BurstStreamingLLM:
    def __init__(self, gate: Event, event_count: int) -> None:
        self.gate = gate
        self.event_count = event_count

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: Any = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, model_settings, request_metadata
        assert self.gate.wait(timeout=2)
        assert stream_callback is not None
        for index in range(self.event_count):
            stream_callback({"event": "assistant_delta", "content_delta": str(index)})
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )


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
    assert [event.event_id for event in first_events] == [event.event_id for event in result.events]
    assert sum(event.type == "assistant_delta" for event in first_events) == event_count


class _BlockingCancellationLLM:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: Any = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, messages, tools, stream_callback, model_settings, request_metadata
        self.started.set()
        assert self.release.wait(timeout=3)
        return LLMResponse(
            content="should be cancelled",
            tool_calls=[
                ToolCall(
                    id="cancel-finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "should be cancelled"},
                )
            ],
        )


def test_run_handle_cancel_accepted_state_and_terminal_reason_match_fixture(tmp_path: Path) -> None:
    contract = _run_handle_contract()["cancellation"]
    llm = _BlockingCancellationLLM()

    handle = Runner.start(
        Agent(name="cancel", instructions="Wait.", model="test-model"),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            workspace=tmp_path,
        ),
    )
    assert llm.started.wait(timeout=2)
    assert handle.cancel(contract["reason"]) is True

    accepted = handle.state()
    assert {
        "status": accepted.status,
        "done": accepted.done,
        "cancelled": accepted.cancelled,
    } == contract["accepted_state"]
    assert handle.cancel(contract["reason"]) is contract["repeated_request_accepted"]

    llm.release.set()
    result = handle.result(timeout=3)
    terminal = handle.state()
    assert result.status == AgentStatus.FAILED
    assert terminal.status == contract["terminal_status"]
    assert terminal.done is True
    assert terminal.cancelled is True
    assert terminal.error is not None and contract["reason"] in terminal.error
    assert handle.cancel(contract["reason"]) is contract["late_request_accepted"]


class _AsyncChildStreamingLLM:
    def __init__(self) -> None:
        self.child_started = Event()
        self.release_child = Event()
        self.parent_calls = 0
        self.lock = threading.Lock()

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: Any = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model, tools, stream_callback, model_settings, request_metadata
        if messages and messages[0].role == "system" and messages[0].content == "Child prompt":
            self.child_started.set()
            assert self.release_child.wait(timeout=3)
            return LLMResponse(
                content="child done",
                tool_calls=[ToolCall(id="child-finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "child done"})],
            )
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
        return LLMResponse(
            content="parent done",
            tool_calls=[ToolCall(id="parent-finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "parent done"})],
        )


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


def test_run_handle_events_wait_for_async_child_after_parent_result_and_allow_tail_cancel(tmp_path: Path) -> None:
    contract = _run_handle_contract()
    llm = _AsyncChildStreamingLLM()
    runtime_task = AgentTask(
        task_id="parent-task",
        model="test-model",
        system_prompt="Parent prompt",
        user_prompt="Delegate",
        max_cycles=3,
        sub_agents={
            "researcher": SubAgentConfig(
                model="test-model",
                description="Research",
                system_prompt="Child prompt",
                max_cycles=2,
            )
        },
    )

    handle = Runner._start_compiled(
        Agent(name="parent", instructions="Delegate.", model="test-model"),
        "go",
        task=runtime_task,
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=_AsyncChildModelProvider(llm),
        ),
    )
    consumed: list[Any] = []
    consumer = Thread(target=lambda: consumed.extend(handle.events()))
    consumer.start()
    assert llm.child_started.wait(timeout=2)
    result = handle.result(timeout=2)

    assert result.status == AgentStatus.COMPLETED
    assert contract["completion"]["result_may_precede_async_children"] is True
    assert consumer.is_alive()
    assert handle.done() is False
    assert handle.state().status == "running"
    assert handle.cancel(contract["cancellation"]["reason"]) is True
    accepted = handle.state()
    assert {
        "status": accepted.status,
        "done": accepted.done,
        "cancelled": accepted.cancelled,
    } == contract["cancellation"]["accepted_state"]
    assert handle.cancel(contract["cancellation"]["reason"]) is contract["cancellation"]["repeated_request_accepted"]
    llm.release_child.set()
    consumer.join(timeout=3)

    assert not consumer.is_alive()
    lifecycle = [event for event in consumed if event.type in {"sub_run_started", "sub_run_completed"}]
    assert [event.type for event in lifecycle] == ["sub_run_started", "sub_run_completed"]
    assert lifecycle[-1].status == AgentStatus.FAILED.value
    assert contract["cancellation"]["reason"] in (lifecycle[-1].error or "")
    assert contract["completion"]["events_wait_for_started_children"] is True
    terminal = handle.state()
    assert terminal.status == contract["cancellation"]["terminal_status"]
    assert terminal.done is True
    assert terminal.cancelled is True
    assert terminal.error is not None and contract["cancellation"]["reason"] in terminal.error
