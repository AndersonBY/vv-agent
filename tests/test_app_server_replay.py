from __future__ import annotations

import threading
from collections.abc import Callable
from typing import cast

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport
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


def test_thread_read_replays_emitted_items() -> None:
    server, transport = _server_with_steps([_finish_response("first")])
    _initialize_and_start_thread(server, transport)
    outbound = _start_turn_and_drain(server, transport, text="hello")
    emitted_items = [message["params"] for message in outbound if str(message.get("method", "")).startswith("item/")]

    _send(transport, server, {"id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    response = _receive_response(transport, 3)
    result = cast(dict[str, object], response["result"])
    replayed_items = cast(list[dict[str, object]], result["items"])

    assert replayed_items == emitted_items


def test_thread_resume_replays_timeline_and_subscribes_to_live_events() -> None:
    server, first = _server_with_steps([_finish_response("first"), _finish_response("second")])
    _initialize_and_start_thread(server, first)
    _start_turn_and_drain(server, first, text="hello")

    second = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(second)
    _send(second, server, {"id": 10, "method": "initialize", "params": {"clientInfo": {"name": "test-2"}}})
    _send(second, server, {"id": 11, "method": "thread/resume", "params": {"threadId": "thread_1"}})
    resume_response = _receive_response(second, 11)
    resume_result = cast(dict[str, object], resume_response["result"])
    assert cast(list[object], resume_result["items"])

    _start_turn_and_drain(server, first, text="second")
    live = _drain_until(second, lambda message: message.get("method") == "turn/completed")
    methods = [message.get("method") for message in live if "method" in message]

    assert "turn/started" in methods
    assert "item/completed" in methods
    assert methods[-1] == "turn/completed"


def test_resume_during_active_turn_subscribes_before_later_notifications() -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()

    def first_step(_model: str, _messages: list[Message]) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    server, first_transport = _server_with_steps([first_step])
    _send(first_transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "first"}}})
    _send(first_transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        first_transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    assert first_step_ready.wait(timeout=2)

    second_transport = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(second_transport)
    server.processor.process_message("conn_2", {"id": 10, "method": "initialize", "params": {"clientInfo": {"name": "second"}}})
    second_transport.receive_outbound(timeout=1)
    server.processor.process_message("conn_2", {"id": 11, "method": "thread/resume", "params": {"threadId": "thread_1"}})
    resume_response = second_transport.receive_outbound(timeout=1)

    first_step_can_finish.set()
    messages = []
    while True:
        message = second_transport.receive_outbound(timeout=10)
        messages.append(message)
        if message.get("method") == "turn/completed":
            break

    assert resume_response["id"] == 11
    assert resume_response["result"]["thread"]["threadId"] == "thread_1"
    assert any(message.get("method") == "turn/completed" for message in messages)


def _server_with_steps(steps: list[ScriptStep]) -> tuple[AppServer, ChannelTransport]:
    llm = ScriptedLLM(steps=steps)

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved_model()

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        run_config=RunConfig(model_provider=model_provider, max_cycles=1),
    )
    return AppServer(transport=transport, host=host), transport


def _finish_response(message: str) -> LLMResponse:
    return LLMResponse(
        content=message,
        tool_calls=[ToolCall(id=f"finish-{message}", name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


def _initialize_and_start_thread(server: AppServer, transport: ChannelTransport) -> None:
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})


def _start_turn_and_drain(server: AppServer, transport: ChannelTransport, *, text: str) -> list[dict[str, object]]:
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": text}]}},
    )
    return _drain_until(transport, lambda message: message.get("method") == "turn/completed")


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _receive_response(transport: ChannelTransport, response_id: int) -> dict[str, object]:
    while True:
        message = transport.receive_outbound(timeout=10)
        if message.get("id") == response_id:
            return message


def _drain_until(
    transport: ChannelTransport,
    predicate: Callable[[dict[str, object]], bool],
) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    while True:
        message = transport.receive_outbound(timeout=10)
        outbound.append(message)
        if predicate(message):
            return outbound
