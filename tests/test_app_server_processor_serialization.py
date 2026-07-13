from __future__ import annotations

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.request_serialization import RequestSerializationQueues


def test_thread_mutations_are_serialized_by_processor() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    queues = RequestSerializationQueues(max_queued_per_scope=1)
    processor = MessageProcessor(router=router, serialization_queues=queues)
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})

    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 2, "method": "thread/archive", "params": {"threadId": "thread_1"}}
    )

    response = transport.receive_outbound(timeout=1)
    assert response["id"] == 2
    assert response["result"]["archived"] is True


def test_processor_returns_overloaded_when_serialization_queue_is_full() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    queues = RequestSerializationQueues(max_queued_per_scope=0)
    processor = MessageProcessor(router=router, serialization_queues=queues)
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})

    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    response = transport.receive_outbound(timeout=1)

    assert response["id"] == 1
    assert response["error"]["code"] == AppServerErrorCode.SERVER_OVERLOADED
