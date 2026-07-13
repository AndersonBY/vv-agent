from __future__ import annotations

from vv_agent.app_server import ChannelTransport, OutgoingRouter
from vv_agent.app_server.protocol import RequestId


def test_router_unregisters_transport_when_overloaded_error_cannot_be_written() -> None:
    transport = ChannelTransport(connection_id="conn_1", outbound_capacity=1)
    router = OutgoingRouter()
    router.register_transport(transport)
    pending = router.send_server_request("conn_1", "approval/request", {"threadId": "thread_1"})
    transport.receive_outbound(timeout=1)

    transport.write_outbound({"jsonrpc": "2.0", "method": "already-full"})
    router.send_notification("conn_1", "item/agentMessage/delta", {"text": "overflow"})

    assert router.cancel_server_request(RequestId(pending.request_id.to_wire())) is False
