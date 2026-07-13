from __future__ import annotations

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter


def _processor_with_transport() -> tuple[MessageProcessor, ChannelTransport]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    return MessageProcessor(router=router), transport


def test_requests_before_initialize_are_rejected() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "model/list"})

    response = transport.receive_outbound(timeout=1)
    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": AppServerErrorCode.NOT_INITIALIZED, "message": "Not initialized"},
    }


def test_initialize_enables_dispatch() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 2, "method": "model/list"})

    initialize_response = transport.receive_outbound(timeout=1)
    model_response = transport.receive_outbound(timeout=1)
    assert initialize_response["result"]["protocolVersion"] == "v1"
    assert model_response == {"jsonrpc": "2.0", "id": 2, "result": {"models": []}}


def test_repeated_initialize_is_rejected() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )

    _ = transport.receive_outbound(timeout=1)
    response = transport.receive_outbound(timeout=1)
    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": AppServerErrorCode.ALREADY_INITIALIZED, "message": "Already initialized"},
    }


def test_initialize_rejects_blank_client_name_without_initializing_connection() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "   "}}}
    )

    response = transport.receive_outbound(timeout=1)
    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": AppServerErrorCode.INVALID_PARAMS, "message": "clientInfo.name is required"},
    }
    assert processor.connection_state("conn_1").initialized is False


def test_initialize_records_client_info_and_capabilities() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "v-claw", "title": "VClaw", "version": "1.2.3"},
                "capabilities": {
                    "experimentalApi": True,
                    "optOutNotificationMethods": ["item/agentMessage/delta", "thread/status/changed"],
                },
            },
        },
    )

    _ = transport.receive_outbound(timeout=1)
    state = processor.connection_state("conn_1")

    assert state.initialized is True
    assert state.client_name == "v-claw"
    assert state.client_title == "VClaw"
    assert state.client_version == "1.2.3"
    assert state.experimental_api is True
    assert state.opt_out_notification_methods == {"item/agentMessage/delta", "thread/status/changed"}


def test_initialize_enables_notifications_and_initialized_notification_is_idempotent() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    response = transport.receive_outbound(timeout=1)
    assert response["result"]["capabilities"] == {
        "modelList": True,
        "threadLifecycle": True,
        "notificationOptOut": True,
        "schemaExport": True,
        "approvalResolve": True,
    }

    processor._router.send_notification("conn_1", "turn/started", {"threadId": "thread_1", "turnId": "turn_1"})
    assert transport.receive_outbound(timeout=1)["method"] == "turn/started"

    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})
    processor._router.send_notification("conn_1", "turn/started", {"threadId": "thread_1", "turnId": "turn_1"})

    assert processor.connection_state("conn_1").ready_for_notifications is True
    assert transport.receive_outbound(timeout=1)["method"] == "turn/started"


def test_initialized_notification_before_initialize_does_not_bypass_initialize_gate() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "model/list"})

    response = transport.receive_outbound(timeout=1)
    assert response["error"]["code"] == AppServerErrorCode.NOT_INITIALIZED
    assert processor.connection_state("conn_1").ready_for_notifications is False


def test_initialize_rejects_non_string_opt_out_methods() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "test"},
                "capabilities": {"optOutNotificationMethods": ["turn/started", 42]},
            },
        },
    )

    response = transport.receive_outbound(timeout=1)
    assert response["id"] == 1
    assert response["error"]["code"] == AppServerErrorCode.INVALID_PARAMS
    assert response["error"]["message"] == "optOutNotificationMethods must be a list of strings"
