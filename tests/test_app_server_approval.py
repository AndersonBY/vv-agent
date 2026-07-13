from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import cast

import pytest

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.app_server import AppServer, ChannelTransport, MessageProcessor, OutgoingRouter, RequestId
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
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


def test_app_server_approval_allow_runs_tool_and_completes_turn() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    assert approval_request["method"] == "approval/request"
    assert approval_params["toolName"] == "dangerous_tool"
    assert calls == []

    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow", "message": "ok"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_allow_session_is_preserved_in_notification() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow_session", "message": "for this session"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert _approval_resolutions(outbound) == ["allow_session"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_deny_skips_tool_and_completes_turn() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "deny", "message": "no"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
    assert _approval_resolutions(outbound) == ["deny"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_timeout_denies_without_running_tool() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=0.01)

    approval_request = _start_and_wait_for_approval(server, transport)
    assert approval_request["method"] == "approval/request"
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
    assert _approval_resolutions(outbound) == ["timeout"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_disconnect_denies_without_running_tool() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    assert approval_request["method"] == "approval/request"
    server.router.unregister_transport("conn_1")

    deadline = time.time() + 2
    while time.time() < deadline:
        snapshot = server.store.read_thread("thread_1")
        if snapshot.turns and snapshot.turns[-1].status == "completed":
            break
        time.sleep(0.01)

    snapshot = server.store.read_thread("thread_1")
    assert calls == []
    assert snapshot.turns[-1].status == "completed"


def test_app_server_approval_is_owned_by_turn_starter_and_not_reassigned_on_disconnect() -> None:
    calls: list[str] = []
    server, owner = _server_with_approval_tool(calls)
    observer = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(observer)
    _initialize(server, owner, request_id=0, client_name="owner")
    thread_id = _start_thread(server, owner, request_id=1)
    _initialize(server, observer, request_id=10, client_name="observer")
    _send(
        observer,
        server,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    _drain_until(observer, lambda message: message.get("id") == 11)

    _send(
        owner,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": thread_id, "input": [{"type": "text", "text": "run tool"}]},
        },
    )
    approval_request = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    observer_before_disconnect = _drain_available(observer)
    assert any(message.get("method") == "turn/started" for message in observer_before_disconnect)
    assert not any(message.get("method") == "approval/request" for message in observer_before_disconnect)

    server.router.unregister_transport(owner.connection_id)
    observer_after_disconnect = _drain_until_turn_completed(observer, timeout=2)

    assert approval_request["method"] == "approval/request"
    assert not any(message.get("method") == "approval/request" for message in observer_after_disconnect)
    assert _approval_resolutions(observer_after_disconnect) == ["timeout"]
    assert calls == []


def test_app_server_follow_up_preserves_original_approval_owner() -> None:
    calls: list[str] = []
    server, owner = _server_with_approval_tool(calls)
    observer = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(observer)
    _initialize(server, owner, request_id=0, client_name="owner")
    thread_id = _start_thread(server, owner, request_id=1)
    _initialize(server, observer, request_id=10, client_name="observer")
    _send(
        observer,
        server,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    _drain_until(observer, lambda message: message.get("id") == 11)

    _send(
        owner,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": thread_id, "input": [{"type": "text", "text": "first"}]},
        },
    )
    first_approval = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    first_params = cast(dict[str, object], first_approval["params"])
    _send(
        observer,
        server,
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "turn/followUp",
            "params": {
                "threadId": thread_id,
                "expectedTurnId": first_params["turnId"],
                "input": [{"type": "text", "text": "second"}],
            },
        },
    )
    _drain_until(observer, lambda message: message.get("id") == 12)
    _send(
        owner,
        server,
        {"jsonrpc": "2.0", "id": first_approval["id"], "result": {"decision": "allow"}},
    )

    second_approval = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    observer_messages = _drain_available(observer)
    assert not any(message.get("method") == "approval/request" for message in observer_messages)
    _send(
        owner,
        server,
        {"jsonrpc": "2.0", "id": second_approval["id"], "result": {"decision": "allow"}},
    )
    _drain_until_turn_completed(owner)

    assert calls == ["ran", "ran"]


def test_app_server_approval_interrupt_releases_pending_request_without_waiting_for_timeout() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=None)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "turn/interrupt",
            "params": {
                "threadId": approval_params["threadId"],
                "expectedTurnId": approval_params["turnId"],
                "reason": "turn interrupted",
            },
        },
    )
    outbound = _drain_until_turn_completed(transport, timeout=2)

    assert calls == []
    assert server.router.pending_server_request_count() == 0
    assert any(
        message.get("id") == 3
        and message.get("result")
        == {
            "threadId": approval_params["threadId"],
            "turnId": approval_params["turnId"],
            "cancelled": True,
        }
        for message in outbound
    )
    assert _approval_resolutions(outbound) == ["timeout"]
    assert _completed_status(outbound) == "failed"


def test_app_server_approval_allow_session_can_be_resolved_by_client_request() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=1)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    requested = _drain_until(transport, lambda message: message.get("method") == "approval/requested")
    assert cast(dict[str, object], requested["params"])["requestId"] == approval_params["requestId"]

    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "approval/resolve",
            "params": {
                "threadId": approval_params["threadId"],
                "turnId": approval_params["turnId"],
                "requestId": approval_params["requestId"],
                "decision": "allow_session",
                "reason": "approved by owner",
                "metadata": {"ticket": 7},
            },
        },
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert any(message.get("id") == 3 and message.get("result") == {} for message in outbound)
    assert _approval_resolutions(outbound) == ["allow_session"]
    resolved = next(
        cast(dict[str, object], message["params"])
        for message in outbound
        if message.get("method") == "approval/resolved" and isinstance(message.get("params"), dict)
    )
    assert resolved["reason"] == "approved by owner"
    assert resolved["metadata"] == {"ticket": 7, "server_request_id": approval_params["requestId"]}


def test_server_request_response_is_bound_to_connection_thread_and_turn() -> None:
    owner = ChannelTransport(connection_id="owner")
    attacker = ChannelTransport(connection_id="attacker")
    router = OutgoingRouter()
    router.register_transport(owner)
    router.register_transport(attacker)
    processor = MessageProcessor(router=router)
    pending = router.send_server_request(
        "owner",
        "approval/request",
        {"threadId": "thread_1", "turnId": "turn_1"},
        request_id=RequestId("approval_1"),
    )

    processor.process_message(
        "attacker",
        {"jsonrpc": "2.0", "id": "approval_1", "result": {"decision": "allow"}},
    )
    with pytest.raises(TimeoutError):
        pending.result(timeout=0)

    assert not router.resolve_server_request(
        "owner",
        RequestId("approval_1"),
        {"decision": "allow"},
        method="approval/request",
        thread_id="thread_other",
        turn_id="turn_1",
    )
    assert router.resolve_server_request(
        "owner",
        RequestId("approval_1"),
        {"decision": "allow"},
        method="approval/request",
        thread_id="thread_1",
        turn_id="turn_1",
    )
    assert pending.result(timeout=0) == {"decision": "allow"}


def _server_with_approval_tool(
    calls: list[str],
    *,
    approval_timeout_seconds: float | None = None,
) -> tuple[AppServer, ChannelTransport]:
    @function_tool(needs_approval=True)
    def dangerous_tool() -> str:
        calls.append("ran")
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous_tool", arguments={})]),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
            LLMResponse(content="calling again", tool_calls=[ToolCall(id="call_2", name="dangerous_tool", arguments={})]),
            LLMResponse(
                content="done again",
                tool_calls=[ToolCall(id="finish_2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done again"})],
            ),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved_model()

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Use the tool.", model="test-model", tools=[dangerous_tool]),
        run_config=RunConfig(
            model_provider=model_provider,
            max_cycles=3,
            approval_timeout_seconds=approval_timeout_seconds,
        ),
    )
    return AppServer(transport=transport, host=host), transport


def _start_and_wait_for_approval(server: AppServer, transport: ChannelTransport) -> dict[str, object]:
    _initialize(server, transport, request_id=0, client_name="test")
    thread_id = _start_thread(server, transport, request_id=1)
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": thread_id, "input": [{"type": "text", "text": "run tool"}]},
        },
    )
    return _drain_until(transport, lambda message: message.get("method") == "approval/request")


def _initialize(
    server: AppServer,
    transport: ChannelTransport,
    *,
    request_id: int,
    client_name: str,
) -> None:
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {"clientInfo": {"name": client_name}},
        },
    )
    _drain_until(transport, lambda message: message.get("id") == request_id)
    _send(transport, server, {"jsonrpc": "2.0", "method": "initialized"})


def _start_thread(server: AppServer, transport: ChannelTransport, *, request_id: int) -> str:
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": request_id, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    response = _drain_until(transport, lambda message: message.get("id") == request_id)
    result = cast(dict[str, object], response["result"])
    return cast(str, result["threadId"])


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _drain_until(
    transport: ChannelTransport,
    predicate: Callable[[dict[str, object]], bool],
    *,
    timeout: float = 10,
) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = transport.receive_outbound(timeout=timeout)
        if predicate(message):
            return message
    raise AssertionError("Timed out waiting for App Server message")


def _drain_until_turn_completed(transport: ChannelTransport, *, timeout: float = 10) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout
    while True:
        message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        outbound.append(message)
        if message.get("method") == "turn/completed":
            return outbound


def _drain_available(transport: ChannelTransport) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    while True:
        try:
            messages.append(transport.receive_outbound(timeout=0))
        except queue.Empty:
            return messages


def _completed_status(outbound: list[dict[str, object]]) -> object:
    completed = [message for message in outbound if message.get("method") == "turn/completed"][-1]
    params = completed.get("params")
    assert isinstance(params, dict)
    return cast(dict[str, object], params)["status"]


def _approval_resolutions(outbound: list[dict[str, object]]) -> list[object]:
    return [
        cast(dict[str, object], message["params"])["decision"]
        for message in outbound
        if message.get("method") == "approval/resolved" and isinstance(message.get("params"), dict)
    ]
