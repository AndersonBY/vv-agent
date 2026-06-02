from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.app_server import AppServer, ChannelTransport
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
        {"id": approval_request["id"], "result": {"decision": "allow", "message": "ok"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_deny_skips_tool_and_completes_turn() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    _send(
        transport,
        server,
        {"id": approval_request["id"], "result": {"decision": "deny", "message": "no"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_timeout_denies_without_running_tool() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=0.01)

    approval_request = _start_and_wait_for_approval(server, transport)
    assert approval_request["method"] == "approval/request"
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
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
    _send(transport, server, {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    _send(transport, server, {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    _send(
        transport,
        server,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "run tool"}]}},
    )
    return _drain_until(transport, lambda message: message.get("method") == "approval/request")


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


def _drain_until_turn_completed(transport: ChannelTransport) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    while True:
        message = transport.receive_outbound(timeout=10)
        outbound.append(message)
        if message.get("method") == "turn/completed":
            return outbound


def _completed_status(outbound: list[dict[str, object]]) -> object:
    completed = [message for message in outbound if message.get("method") == "turn/completed"][-1]
    params = completed.get("params")
    assert isinstance(params, dict)
    return cast(dict[str, object], params)["status"]
