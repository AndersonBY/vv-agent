from __future__ import annotations

import pytest

from vv_agent.app_server.protocol import (
    ApprovalDecision,
    ApprovalRequestParams,
    ApprovalResolveParams,
    AppServerError,
    AppServerErrorCode,
    ClientInfo,
    InitializeParams,
    InitializeResponse,
    JsonRpcError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    RequestId,
    ThreadItem,
    ThreadStartParams,
    TurnStartParams,
)


def test_request_round_trips_with_jsonrpc_header() -> None:
    request = JsonRpcRequest(id=RequestId(1), method="initialize", params={"clientInfo": {"name": "test"}})

    payload = request.to_dict()

    assert payload == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "test"}},
    }
    assert JsonRpcMessage.from_dict(payload).message == request


def test_notification_round_trips() -> None:
    notification = JsonRpcNotification(method="initialized", params=None)

    payload = notification.to_dict()

    assert payload == {"jsonrpc": "2.0", "method": "initialized"}
    assert JsonRpcMessage.from_dict(payload).message == notification


def test_error_response_uses_stable_error_code() -> None:
    error = JsonRpcError(id=RequestId("req-1"), error=AppServerError.not_initialized())

    payload = error.to_dict()

    assert payload["id"] == "req-1"
    assert payload["error"]["code"] == AppServerErrorCode.NOT_INITIALIZED
    assert payload["error"]["message"] == "Not initialized"


def test_invalid_wire_message_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
        JsonRpcMessage.from_dict({"params": {}})

    with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
        JsonRpcMessage.from_dict({"jsonrpc": "2.0", "id": [], "method": "thread/start", "params": {}})

    with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
        JsonRpcMessage.from_dict({"jsonrpc": "1.0", "id": 1, "method": "thread/start"})

    for payload in (
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "extra": True},
        {"jsonrpc": "2.0", "method": "initialized", "extra": True},
        {"jsonrpc": "2.0", "id": 1, "result": {}, "extra": True},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "failed"}, "extra": True},
    ):
        with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
            JsonRpcMessage.from_dict(payload)


def test_method_not_found_includes_the_rejected_method() -> None:
    error = AppServerError.method_not_found("unknown/method").to_dict()

    assert error == {
        "code": AppServerErrorCode.METHOD_NOT_FOUND,
        "message": "Method not found: unknown/method",
        "data": {"method": "unknown/method"},
    }


def test_initialize_payload_uses_camel_case() -> None:
    params = InitializeParams(client_info=ClientInfo(name="v_claw", title="v-claw", version="0.2.1"))

    assert params.to_dict() == {
        "clientInfo": {"name": "v_claw", "title": "v-claw", "version": "0.2.1"},
        "capabilities": {"experimentalApi": False, "optOutNotificationMethods": []},
    }
    assert InitializeResponse(user_agent="v_claw/0.2.1", protocol_version="v1").to_dict()["protocolVersion"] == "v1"


def test_thread_and_turn_payloads_have_stable_ids() -> None:
    thread = ThreadStartParams(agent_key="default", cwd="/tmp/project")
    turn = TurnStartParams(thread_id="thread_1", input=[{"type": "text", "text": "run tests"}])

    assert thread.to_dict()["agentKey"] == "default"
    assert turn.to_dict()["threadId"] == "thread_1"
    assert turn.to_dict()["input"][0]["text"] == "run tests"


def test_thread_item_is_tagged_union_payload() -> None:
    item = ThreadItem(
        item_id="item_1",
        thread_id="thread_1",
        turn_id="turn_1",
        item_type="agentMessage",
        status="completed",
        payload={"text": "done"},
        created_at=1,
        updated_at=2,
    )

    assert item.to_dict() == {
        "itemId": "item_1",
        "threadId": "thread_1",
        "turnId": "turn_1",
        "type": "agentMessage",
        "status": "completed",
        "payload": {"text": "done"},
        "createdAt": 1,
        "updatedAt": 2,
    }


def test_approval_request_payload_is_client_ready() -> None:
    request = ApprovalRequestParams(
        request_id="req_1",
        thread_id="thread_1",
        turn_id="turn_1",
        tool_call_id="call_1",
        tool_name="bash",
        preview="Run pytest",
        arguments={"cmd": "pytest"},
    )

    assert request.to_dict()["toolName"] == "bash"
    assert request.to_dict()["arguments"] == {"cmd": "pytest"}


def test_approval_decision_round_trips_allow_session() -> None:
    params = ApprovalResolveParams(
        request_id="req_1",
        thread_id="thread_1",
        turn_id="turn_1",
        decision=ApprovalDecision.ALLOW_SESSION,
        reason="approved by owner",
        metadata={"ticket": 7},
    )

    assert params.to_dict()["decision"] == "allow_session"
    assert params.to_dict()["reason"] == "approved by owner"
    assert params.to_dict()["metadata"] == {"ticket": 7}
    assert ApprovalDecision.from_wire(params.to_dict()["decision"]) is ApprovalDecision.ALLOW_SESSION
