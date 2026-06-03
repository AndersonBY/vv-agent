from __future__ import annotations

import pytest

from vv_agent.app_server import ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.transport import AppServerOverloadedError


def test_channel_transport_processes_jsonrpc_request() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    processor = MessageProcessor(router=router)

    transport.send_inbound({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    processor.process_next(transport)

    response = transport.receive_outbound(timeout=1)
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == "v1"


def test_channel_transport_outbound_overflow_is_reported() -> None:
    transport = ChannelTransport(connection_id="conn_1", outbound_capacity=1)

    transport.write_outbound({"id": 1, "result": None})
    with pytest.raises(AppServerOverloadedError, match=r"Server overloaded; retry later\."):
        transport.write_outbound({"id": 2, "result": None})
