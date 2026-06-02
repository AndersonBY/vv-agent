from __future__ import annotations

import threading
from typing import cast

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, AppServerErrorCode, ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.llm.scripted import ScriptStep
from vv_agent.types import LLMResponse, Message, ToolCall


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_processor_starts_thread_and_streams_turn_items() -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved_model()

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        run_config=RunConfig(model_provider=model_provider, max_cycles=1),
    )
    server = AppServer(transport=transport, host=host)

    for payload in [
        {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}},
        {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "/tmp/work"}},
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    ]:
        transport.send_inbound(payload)
        server.processor.process_next(transport)

    outbound: list[dict[str, object]] = []
    while True:
        message = transport.receive_outbound(timeout=10)
        outbound.append(message)
        if message.get("method") == "turn/completed":
            break

    response_ids = [message.get("id") for message in outbound if "result" in message]
    methods = [message.get("method") for message in outbound if "method" in message]

    assert response_ids[:3] == [0, 1, 2]
    assert "thread/started" in methods
    assert "turn/started" in methods
    assert "item/started" in methods
    assert "item/completed" in methods
    assert methods[-1] == "turn/completed"


def test_turn_start_and_completion_emit_thread_status_changes() -> None:
    server, transport = _server_with_scripted_steps(
        [
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )

    outbound = _drain_until_turn_completed(transport)
    status_changes = [
        message.get("params", {}).get("status")
        for message in outbound
        if message.get("method") == "thread/status/changed"
    ]

    assert status_changes == ["running", "idle"]


def test_turn_steer_injects_context_into_active_turn_next_cycle() -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()
    seen_user_messages: list[list[str]] = []

    def first_step(_model: str, _messages: list[Message]) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(content="continue", tool_calls=[])

    def second_step(_model: str, messages: list[Message]) -> LLMResponse:
        seen_user_messages.append([message.content for message in messages if message.role == "user"])
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    server, transport = _server_with_scripted_steps([first_step, second_step], max_cycles=3)
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    assert first_step_ready.wait(timeout=2)

    _send(
        transport,
        server,
        {
            "id": 3,
            "method": "turn/steer",
            "params": {
                "threadId": "thread_1",
                "expectedTurnId": "turn_1",
                "input": [{"type": "text", "text": "queued from app server"}],
            },
        },
    )
    first_step_can_finish.set()
    _drain_until_turn_completed(transport)

    assert seen_user_messages == [
        [
            "hello",
            "No tool call was produced. Continue the task and call `task_finish` when all todo items are done.",
            "queued from app server",
        ]
    ]


def test_turn_steer_rejects_turn_id_mismatch() -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()

    def first_step(_model: str, _messages: list[Message]) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    server, transport = _server_with_scripted_steps([first_step])
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    assert first_step_ready.wait(timeout=2)

    _send(
        transport,
        server,
        {
            "id": 3,
            "method": "turn/steer",
            "params": {"threadId": "thread_1", "expectedTurnId": "wrong", "input": [{"type": "text", "text": "ignored"}]},
        },
    )

    response = _receive_response(transport, 3)
    assert response == {"id": 3, "error": {"code": AppServerErrorCode.TURN_ID_MISMATCH, "message": "Turn id mismatch"}}
    first_step_can_finish.set()
    _drain_until_turn_completed(transport)


def test_turn_follow_up_starts_next_turn_after_active_turn_completes() -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()
    seen_user_messages: list[list[str]] = []

    def first_step(_model: str, _messages: list[Message]) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(
            content="first",
            tool_calls=[ToolCall(id="finish-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "first"})],
        )

    def second_step(_model: str, messages: list[Message]) -> LLMResponse:
        seen_user_messages.append([message.content for message in messages if message.role == "user"])
        return LLMResponse(
            content="second",
            tool_calls=[ToolCall(id="finish-2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "second"})],
        )

    server, transport = _server_with_scripted_steps([first_step, second_step])
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    assert first_step_ready.wait(timeout=2)
    _send(
        transport,
        server,
        {
            "id": 3,
            "method": "turn/followUp",
            "params": {
                "threadId": "thread_1",
                "expectedTurnId": "turn_1",
                "input": [{"type": "text", "text": "continue"}],
            },
        },
    )

    first_step_can_finish.set()
    outbound = _drain_until_turn_completed(transport, count=2)
    methods = [message.get("method") for message in outbound if "method" in message]

    assert methods.count("turn/started") == 2
    assert methods.count("turn/completed") == 2
    assert seen_user_messages == [["continue"]]


def test_turn_interrupt_cancels_active_turn() -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()

    def first_step(_model: str, _messages: list[Message]) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(content="continue", tool_calls=[])

    server, transport = _server_with_scripted_steps([first_step], max_cycles=2)
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    assert first_step_ready.wait(timeout=2)

    _send(
        transport,
        server,
        {"id": 3, "method": "turn/interrupt", "params": {"threadId": "thread_1", "expectedTurnId": "turn_1", "reason": "stop"}},
    )
    response = _receive_response(transport, 3)
    first_step_can_finish.set()
    outbound = _drain_until_turn_completed(transport)
    completed = [message for message in outbound if message.get("method") == "turn/completed"][-1]
    completed_params = completed.get("params")

    assert response == {"id": 3, "result": {"threadId": "thread_1", "turnId": "turn_1", "cancelled": True}}
    assert isinstance(completed_params, dict)
    assert cast(dict[str, object], completed_params)["status"] == "failed"


def _server_with_scripted_steps(steps: list[ScriptStep], *, max_cycles: int = 1) -> tuple[AppServer, ChannelTransport]:
    llm = ScriptedLLM(steps=steps)

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved_model()

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        run_config=RunConfig(model_provider=model_provider, max_cycles=max_cycles),
    )
    return AppServer(transport=transport, host=host), transport


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _receive_response(transport: ChannelTransport, response_id: int) -> dict[str, object]:
    while True:
        message = transport.receive_outbound(timeout=10)
        if message.get("id") == response_id:
            return message


def _drain_until_turn_completed(transport: ChannelTransport, *, count: int = 1) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    completed = 0
    while completed < count:
        message = transport.receive_outbound(timeout=10)
        outbound.append(message)
        if message.get("method") == "turn/completed":
            completed += 1
    return outbound
