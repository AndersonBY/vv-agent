from __future__ import annotations

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter


def _initialized_processor() -> tuple[MessageProcessor, ChannelTransport]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    processor = MessageProcessor(router=router)
    processor.process_message("conn_1", {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    transport.receive_outbound(timeout=1)
    return processor, transport


def test_thread_list_returns_active_threads() -> None:
    processor, transport = _initialized_processor()
    processor.process_message("conn_1", {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    processor.process_message("conn_1", {"id": 2, "method": "thread/list"})

    transport.receive_outbound(timeout=1)
    start_response = transport.receive_outbound(timeout=1)
    list_response = transport.receive_outbound(timeout=1)

    assert start_response["id"] == 1
    assert list_response["id"] == 2
    assert list_response["result"]["threads"][0]["threadId"] == "thread_1"
    assert list_response["result"]["threads"][0]["status"] == "idle"


def test_thread_archive_hides_thread_and_emits_notification() -> None:
    processor, transport = _initialized_processor()
    processor.process_message("conn_1", {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message("conn_1", {"id": 2, "method": "thread/archive", "params": {"threadId": "thread_1"}})
    archive_response = transport.receive_outbound(timeout=1)
    archived_notification = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"id": 3, "method": "thread/list"})
    list_response = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"id": 4, "method": "thread/list", "params": {"includeArchived": True}})
    archived_list_response = transport.receive_outbound(timeout=1)

    assert archive_response == {"id": 2, "result": {"threadId": "thread_1", "archived": True}}
    assert archived_notification["method"] == "thread/archived"
    assert list_response["result"]["threads"] == []
    assert archived_list_response["result"]["threads"][0]["status"] == "archived"


def test_turn_start_rejects_archived_thread() -> None:
    processor, transport = _initialized_processor()
    processor.process_message("conn_1", {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"id": 2, "method": "thread/archive", "params": {"threadId": "thread_1"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message(
        "conn_1",
        {"id": 3, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    response = transport.receive_outbound(timeout=1)

    assert response["id"] == 3
    assert response["error"]["code"] == AppServerErrorCode.THREAD_ARCHIVED


def test_thread_unsubscribe_closes_idle_thread() -> None:
    processor, transport = _initialized_processor()
    processor.process_message("conn_1", {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message("conn_1", {"id": 2, "method": "thread/unsubscribe", "params": {"threadId": "thread_1"}})
    response = transport.receive_outbound(timeout=1)
    closed_notification = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    read_response = transport.receive_outbound(timeout=1)

    assert response == {"id": 2, "result": {"threadId": "thread_1", "subscribed": False, "closed": True}}
    assert closed_notification == {"method": "thread/closed", "params": {"threadId": "thread_1"}}
    assert read_response["result"]["thread"]["status"] == "closed"
