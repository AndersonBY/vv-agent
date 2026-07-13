from __future__ import annotations

import queue

import pytest

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore


def _initialized_processor(
    *,
    store: ThreadStore | None = None,
) -> tuple[MessageProcessor, ChannelTransport, OutgoingRouter, ThreadStateManager]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    state_manager = ThreadStateManager()
    router.register_transport(transport)
    processor = MessageProcessor(router=router, state_manager=state_manager, store=store)
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
    )
    transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})
    return processor, transport, router, state_manager


def test_thread_list_returns_active_threads() -> None:
    processor, transport, _router, _state_manager = _initialized_processor()
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 2, "method": "thread/list"})

    start_response = transport.receive_outbound(timeout=1)
    started_notification = transport.receive_outbound(timeout=1)
    list_response = transport.receive_outbound(timeout=1)

    assert start_response["id"] == 1
    assert started_notification["method"] == "thread/started"
    assert list_response["id"] == 2
    assert list_response["result"]["threads"][0]["threadId"] == "thread_1"
    assert list_response["result"]["threads"][0]["status"] == "idle"


def test_thread_archive_hides_thread_and_emits_notification() -> None:
    processor, transport, _router, _state_manager = _initialized_processor()
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 2, "method": "thread/archive", "params": {"threadId": "thread_1"}}
    )
    archive_response = transport.receive_outbound(timeout=1)
    archived_notification = transport.receive_outbound(timeout=1)
    status_notification = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 3, "method": "thread/list"})
    list_response = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 4, "method": "thread/list", "params": {"includeArchived": True}})
    archived_list_response = transport.receive_outbound(timeout=1)

    assert archive_response == {"jsonrpc": "2.0", "id": 2, "result": {"threadId": "thread_1", "archived": True}}
    assert archived_notification["method"] == "thread/archived"
    assert status_notification == {
        "jsonrpc": "2.0",
        "method": "thread/status/changed",
        "params": {"threadId": "thread_1", "status": "archived"},
    }
    assert list_response["result"]["threads"] == []
    assert archived_list_response["result"]["threads"][0]["status"] == "archived"


def test_turn_start_rejects_archived_thread() -> None:
    processor, transport, _router, _state_manager = _initialized_processor()
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 2, "method": "thread/archive", "params": {"threadId": "thread_1"}}
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "turn/start",
            "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]},
        },
    )
    response = transport.receive_outbound(timeout=1)

    assert response["id"] == 3
    assert response["error"]["code"] == AppServerErrorCode.THREAD_ARCHIVED


def test_thread_unsubscribe_closes_idle_thread() -> None:
    processor, transport, _router, _state_manager = _initialized_processor()
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}})
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 2, "method": "thread/unsubscribe", "params": {"threadId": "thread_1"}}
    )
    response = transport.receive_outbound(timeout=1)
    closed_notification = transport.receive_outbound(timeout=1)
    status_notification = transport.receive_outbound(timeout=1)
    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    read_response = transport.receive_outbound(timeout=1)

    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"threadId": "thread_1", "subscribed": False, "closed": True},
    }
    assert closed_notification == {
        "jsonrpc": "2.0",
        "method": "thread/closed",
        "params": {"threadId": "thread_1"},
    }
    assert status_notification == {
        "jsonrpc": "2.0",
        "method": "thread/status/changed",
        "params": {"threadId": "thread_1", "status": "closed"},
    }
    assert read_response["result"]["thread"]["status"] == "closed"


@pytest.mark.parametrize("subscribe", [True, False])
def test_thread_resume_reopens_closed_thread(subscribe: bool) -> None:
    processor, transport, _router, state_manager = _initialized_processor()
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 2, "method": "thread/unsubscribe", "params": {"threadId": "thread_1"}},
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "thread/resume",
            "params": {"threadId": "thread_1", "subscribe": subscribe},
        },
    )
    response = transport.receive_outbound(timeout=1)

    assert response["result"]["thread"]["status"] == "idle"
    assert state_manager.status("thread_1") == "idle"
    assert state_manager.is_subscribed("thread_1", "conn_1") is subscribe


@pytest.mark.parametrize("subscribe", [True, False])
def test_thread_resume_reopens_persisted_closed_thread(subscribe: bool) -> None:
    store = ThreadStore()
    processor, transport, _router, _state_manager = _initialized_processor(store=store)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 2, "method": "thread/unsubscribe", "params": {"threadId": "thread_1"}},
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    store.set_active_turn("thread_1", None, "closed")

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "thread/resume",
            "params": {"threadId": "thread_1", "subscribe": subscribe},
        },
    )
    response = transport.receive_outbound(timeout=1)

    assert response["result"]["thread"]["status"] == "idle"
    assert store.read_thread("thread_1").thread.status == "idle"


def test_thread_archive_notifies_only_requester_without_subscribing_it() -> None:
    processor, first, router, state_manager = _initialized_processor()
    second = ChannelTransport(connection_id="conn_2")
    router.register_transport(second)
    processor.process_message(
        "conn_2",
        {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {"clientInfo": {"name": "second"}}},
    )
    second.receive_outbound(timeout=1)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    first.receive_outbound(timeout=1)
    first.receive_outbound(timeout=1)

    processor.process_message(
        "conn_2",
        {"jsonrpc": "2.0", "id": 11, "method": "thread/archive", "params": {"threadId": "thread_1"}},
    )

    assert second.receive_outbound(timeout=1)["id"] == 11
    assert second.receive_outbound(timeout=1)["method"] == "thread/archived"
    assert second.receive_outbound(timeout=1)["method"] == "thread/status/changed"
    with pytest.raises(queue.Empty):
        first.receive_outbound(timeout=0.01)
    assert state_manager.subscribers("thread_1") == {"conn_1"}


def test_active_turn_state_is_persisted_through_processor_owned_state_manager() -> None:
    store = ThreadStore()
    state_manager = ThreadStateManager()
    MessageProcessor(router=OutgoingRouter(), store=store, state_manager=state_manager)
    thread = store.create_thread(agent_key="default")

    state_manager.set_active_turn(thread_id=thread.thread_id, turn_id="turn_1", handle=object())
    running = store.read_thread(thread.thread_id).thread
    state_manager.clear_active_turn(thread.thread_id, "turn_1")
    idle = store.read_thread(thread.thread_id).thread

    assert (running.status, running.active_turn_id) == ("running", "turn_1")
    assert (idle.status, idle.active_turn_id) == ("idle", None)
