from __future__ import annotations

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter


def _processor_with_transport() -> tuple[MessageProcessor, ChannelTransport]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    return MessageProcessor(router=router), transport


def test_requests_before_initialize_are_rejected() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message("conn_1", {"id": 1, "method": "model/list"})

    response = transport.receive_outbound(timeout=1)
    assert response == {
        "id": 1,
        "error": {"code": AppServerErrorCode.NOT_INITIALIZED, "message": "Not initialized"},
    }


def test_initialize_enables_dispatch() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message("conn_1", {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    processor.process_message("conn_1", {"id": 2, "method": "model/list"})

    initialize_response = transport.receive_outbound(timeout=1)
    model_response = transport.receive_outbound(timeout=1)
    assert initialize_response["result"]["protocolVersion"] == "v1"
    assert model_response == {"id": 2, "result": {"models": []}}


def test_repeated_initialize_is_rejected() -> None:
    processor, transport = _processor_with_transport()

    processor.process_message("conn_1", {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    processor.process_message("conn_1", {"id": 2, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})

    _ = transport.receive_outbound(timeout=1)
    response = transport.receive_outbound(timeout=1)
    assert response == {
        "id": 2,
        "error": {"code": AppServerErrorCode.ALREADY_INITIALIZED, "message": "Already initialized"},
    }
