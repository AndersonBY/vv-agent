from __future__ import annotations

from typing import cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.app_server import (
    ApprovalDecision,
    ApprovalResolveParams,
    AppServerClient,
    AppServerClientError,
    AppServerErrorCode,
    ClientInfo,
    DefaultAppServerHost,
    ModelListRequest,
    ModelSummary,
    ThreadArchiveParams,
    ThreadListParams,
    ThreadReadParams,
    ThreadResumeParams,
    ThreadStartParams,
    ThreadUnsubscribeParams,
    TurnFollowUpParams,
    TurnInterruptParams,
    TurnStartParams,
    TurnSteerParams,
)
from vv_agent.app_server.schema import CLIENT_METHOD_SPECS
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _client(*, final_output: str = "done") -> AppServerClient:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="finish",
                        name=TASK_FINISH_TOOL_NAME,
                        arguments={"message": final_output},
                    )
                ],
            )
        ]
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[
            EndpointOption(
                endpoint=EndpointConfig(
                    endpoint_id="test",
                    api_key="secret",
                    api_base="https://example.invalid/v1",
                ),
                model_id="test-model",
            )
        ],
    )

    return AppServerClient.for_host(
        DefaultAppServerHost(
            agent=Agent(name="assistant", instructions="Finish.", model="test-model"),
            run_config=RunConfig(model_provider=FixedModelProvider(llm, resolved), max_cycles=2),
            models=[ModelSummary(id="test-model", provider="test", supports_tools=True)],
        )
    )


def _approval_client(calls: list[str]) -> AppServerClient:
    @function_tool(needs_approval=True)
    def dangerous_tool() -> str:
        calls.append("ran")
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="calling",
                tool_calls=[ToolCall(id="call_1", name="dangerous_tool", arguments={})],
            ),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[
            EndpointOption(
                endpoint=EndpointConfig(
                    endpoint_id="test",
                    api_key="secret",
                    api_base="https://example.invalid/v1",
                ),
                model_id="test-model",
            )
        ],
    )

    return AppServerClient.for_host(
        DefaultAppServerHost(
            agent=Agent(
                name="assistant",
                instructions="Use the tool.",
                model="test-model",
                tools=[dangerous_tool],
            ),
            run_config=RunConfig(model_provider=FixedModelProvider(llm, resolved), max_cycles=3),
        )
    )


def test_client_covers_thread_model_schema_and_close_lifecycle() -> None:
    client = _client()
    initialized = client.initialize(ClientInfo(name="test-client"))
    assert initialized["protocolVersion"] == "v1"

    thread = client.start_thread(ThreadStartParams(agent_key="default"))
    thread_id = str(thread["threadId"])
    assert client.resume_thread(ThreadResumeParams(thread_id=thread_id))["thread"]["threadId"] == thread_id
    assert client.read_thread(ThreadReadParams(thread_id=thread_id))["thread"]["threadId"] == thread_id
    assert client.list_threads(ThreadListParams())["threads"][0]["threadId"] == thread_id
    assert client.list_models(ModelListRequest(provider="test"))["models"][0]["id"] == "test-model"
    schema = client.export_schema()
    assert "jsonSchema" in schema
    assert "typescript" in schema

    unsubscribed = client.unsubscribe_thread(ThreadUnsubscribeParams(thread_id=thread_id))
    assert unsubscribed["subscribed"] is False
    assert unsubscribed["closed"] is True
    assert client.resume_thread(ThreadResumeParams(thread_id=thread_id))["thread"]["status"] == "idle"
    assert client.archive_thread(ThreadArchiveParams(thread_id=thread_id))["archived"] is True
    assert client.close() is True
    assert client.close() is False
    with pytest.raises(AppServerClientError, match="client is closed"):
        client.list_models()


def test_client_preserves_notifications_while_waiting_for_responses() -> None:
    client = _client(final_output="client result")
    client.initialize(ClientInfo(name="test-client"))
    thread = client.start_thread(ThreadStartParams(agent_key="default"))
    thread_id = str(thread["threadId"])

    turn = client.start_turn(
        TurnStartParams(
            thread_id=thread_id,
            input=[{"type": "text", "text": "finish"}],
        )
    )
    assert turn["threadId"] == thread_id
    assert client.list_models()["models"][0]["id"] == "test-model"

    methods: list[str] = []
    while "turn/completed" not in methods:
        message = client.next_message(timeout=5)
        method = message.get("method")
        if isinstance(method, str):
            methods.append(method)
    assert "thread/started" in methods
    assert "turn/started" in methods
    assert "turn/completed" in methods
    assert client.close() is True


def test_client_exposes_turn_control_methods_and_server_errors() -> None:
    client = _client()
    client.initialize(ClientInfo(name="test-client"))
    thread_id = str(client.start_thread(ThreadStartParams(agent_key="default"))["threadId"])

    for invoke in (
        lambda: client.steer_turn(TurnSteerParams(thread_id=thread_id, expected_turn_id="turn_1", input=[])),
        lambda: client.follow_up_turn(TurnFollowUpParams(thread_id=thread_id, expected_turn_id="turn_1", input=[])),
        lambda: client.interrupt_turn(TurnInterruptParams(thread_id=thread_id, expected_turn_id="turn_1")),
    ):
        with pytest.raises(AppServerClientError, match="Active turn not found") as caught:
            invoke()
        assert caught.value.code == AppServerErrorCode.ACTIVE_TURN_NOT_FOUND

    with pytest.raises(AppServerClientError, match="Method not found") as caught:
        client._send_request("missing/method", {})
    assert caught.value.code == AppServerErrorCode.METHOD_NOT_FOUND
    assert caught.value.data == {"method": "missing/method"}
    assert client.close() is True


def test_client_answers_server_requests_and_close_clears_pending() -> None:
    client = _client()
    client.initialize(ClientInfo(name="test-client"))
    router = client._server.router
    connection_id = client._transport.connection_id

    pending = router.send_server_request(
        connection_id,
        "approval/request",
        {"threadId": "thread_1", "turnId": "turn_1"},
    )
    request = client.next_message()
    client.resolve_approval(
        ApprovalResolveParams(
            request_id=str(request["id"]),
            thread_id="thread_1",
            turn_id="turn_1",
            decision=ApprovalDecision.ALLOW_SESSION,
            reason="approved by owner",
            metadata={"ticket": 7},
        )
    )
    assert pending.result(timeout=0) == {
        "decision": "allow_session",
        "reason": "approved by owner",
        "metadata": {"ticket": 7},
    }

    abandoned = router.send_server_request(
        connection_id,
        "approval/request",
        {"threadId": "thread_1", "turnId": "turn_2"},
    )
    assert router.pending_server_request_count() == 1
    assert client.close() is True
    assert router.pending_server_request_count() == 0
    with pytest.raises(RuntimeError, match="client_disconnected"):
        abandoned.result(timeout=0)
    with pytest.raises(AppServerClientError, match="client is closed"):
        client.next_message(timeout=0)


def test_client_resolves_approval_through_stable_request_method() -> None:
    calls: list[str] = []
    client = _approval_client(calls)
    client.initialize(ClientInfo(name="test-client"))
    thread_id = str(client.start_thread(ThreadStartParams(agent_key="default"))["threadId"])
    turn_id = str(
        client.start_turn(
            TurnStartParams(
                thread_id=thread_id,
                input=[{"type": "text", "text": "run approval tool"}],
            )
        )["turnId"]
    )

    while True:
        request = client.next_message(timeout=5)
        if request.get("method") == "approval/request":
            break
    client.resolve_approval_request(
        ApprovalResolveParams(
            request_id=str(request["id"]),
            thread_id=thread_id,
            turn_id=turn_id,
            decision=ApprovalDecision.ALLOW_SESSION,
            reason="approved by owner",
            metadata={"ticket": 7},
        )
    )

    resolutions: list[dict[str, object]] = []
    while True:
        message = client.next_message(timeout=5)
        if message.get("method") == "approval/resolved":
            params = message.get("params")
            assert isinstance(params, dict)
            resolutions.append(params)
        if message.get("method") == "turn/completed":
            break

    assert calls == ["ran"]
    assert len(resolutions) == 1
    assert resolutions[0]["decision"] == "allow_session"
    assert resolutions[0]["reason"] == "approved by owner"
    raw_metadata = resolutions[0]["metadata"]
    assert isinstance(raw_metadata, dict)
    metadata = cast(dict[str, object], raw_metadata)
    assert metadata["ticket"] == 7
    assert client.close() is True


def test_client_facade_covers_stable_method_inventory() -> None:
    facade_methods = {
        "initialize": "initialize",
        "initialized": "initialize",
        "model/list": "list_models",
        "thread/start": "start_thread",
        "thread/resume": "resume_thread",
        "thread/read": "read_thread",
        "thread/list": "list_threads",
        "thread/archive": "archive_thread",
        "thread/unsubscribe": "unsubscribe_thread",
        "turn/start": "start_turn",
        "turn/resume": "resume_turn",
        "turn/steer": "steer_turn",
        "turn/followUp": "follow_up_turn",
        "turn/interrupt": "interrupt_turn",
        "approval/resolve": "resolve_approval_request",
        "schema/export": "export_schema",
    }

    assert len(CLIENT_METHOD_SPECS) == 16
    assert set(facade_methods) == set(CLIENT_METHOD_SPECS)
    assert all(callable(getattr(AppServerClient, method)) for method in facade_methods.values())
    assert callable(AppServerClient.send_response)
    assert callable(AppServerClient.next_message)
    assert callable(AppServerClient.close)
