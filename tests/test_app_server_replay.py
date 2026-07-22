from __future__ import annotations

import threading
from collections.abc import Callable
from typing import cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.llm.scripted import ScriptStep
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


def test_thread_read_replays_emitted_items() -> None:
    server, transport = _server_with_steps([_finish_response("first")])
    _initialize_and_start_thread(server, transport)
    outbound = _start_turn_and_drain(server, transport, text="hello")
    item_methods = {"item/started", "item/completed", "item/agentMessage/delta"}
    emitted_items = []
    for message in outbound:
        if message.get("method") not in item_methods:
            continue
        item = dict(cast(dict[str, object], message["params"]))
        item.pop("delta", None)
        emitted_items.append(item)

    _send(transport, server, {"jsonrpc": "2.0", "id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
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
    _send(second, server, {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {"clientInfo": {"name": "test-2"}}})
    _send(second, server, {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": "thread_1"}})
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

    def first_step(request: LlmRequest) -> LLMResponse:
        _model, _messages = request.model, request.messages
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=2)
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    server, first_transport = _server_with_steps([first_step])
    _send(
        first_transport, server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "first"}}}
    )
    _send(first_transport, server, {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        first_transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]},
        },
    )
    assert first_step_ready.wait(timeout=2)

    second_transport = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(second_transport)
    server.processor.process_message(
        "conn_2", {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {"clientInfo": {"name": "second"}}}
    )
    second_transport.receive_outbound(timeout=1)
    server.processor.process_message("conn_2", {"jsonrpc": "2.0", "method": "initialized"})
    server.processor.process_message(
        "conn_2", {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": "thread_1"}}
    )
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


def test_resume_subscription_and_reopen_are_installed_before_snapshot() -> None:
    state_manager = ThreadStateManager()
    state_manager.subscribe("thread_1", "old_connection")
    state_manager.unsubscribe("thread_1", "old_connection")
    assert state_manager.close_if_idle("thread_1") is True

    def snapshot() -> str:
        assert state_manager.subscribers("thread_1") == {"new_connection"}
        assert state_manager.status("thread_1") == "idle"
        return "snapshot"

    assert state_manager.subscribe_and_snapshot("thread_1", "new_connection", snapshot) == "snapshot"


def test_resume_snapshot_runtime_error_rolls_back_new_subscription_and_closed_state() -> None:
    state_manager = ThreadStateManager()
    state_manager.subscribe("thread_1", "old_connection")
    state_manager.unsubscribe("thread_1", "old_connection")
    assert state_manager.close_if_idle("thread_1") is True

    def fail_snapshot() -> None:
        raise RuntimeError("snapshot failed")

    with pytest.raises(RuntimeError, match="snapshot failed"):
        state_manager.subscribe_and_snapshot("thread_1", "new_connection", fail_snapshot)

    assert state_manager.subscribers("thread_1") == set()
    assert state_manager.status("thread_1") == "closed"


def _server_with_steps(steps: list[ScriptStep]) -> tuple[AppServer, ChannelTransport]:
    llm = ScriptedLLM(steps=steps)

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model()), max_cycles=1),
    )
    return AppServer(transport=transport, host=host), transport


def _finish_response(message: str) -> LLMResponse:
    return LLMResponse(
        content=message,
        tool_calls=[ToolCall(id=f"finish-{message}", name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


def _initialize_and_start_thread(server: AppServer, transport: ChannelTransport) -> None:
    _send(transport, server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})


def _start_turn_and_drain(server: AppServer, transport: ChannelTransport, *, text: str) -> list[dict[str, object]]:
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": "thread_1", "input": [{"type": "text", "text": text}]},
        },
    )
    return _drain_until(transport, lambda message: message.get("method") == "turn/completed")


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)
    if payload.get("method") == "initialize":
        transport.send_inbound({"jsonrpc": "2.0", "method": "initialized"})
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
