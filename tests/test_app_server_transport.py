from __future__ import annotations

import io
import threading
from typing import Any

import pytest

from vv_agent.app_server import AppServer, AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.transport import AppServerOverloadedError, StdioJsonlTransport


def test_channel_transport_processes_jsonrpc_request() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    processor = MessageProcessor(router=router)

    transport.send_inbound({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    processor.process_next(transport)

    response = transport.receive_outbound(timeout=1)
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == "v1"


def test_channel_transport_outbound_overflow_is_reported() -> None:
    transport = ChannelTransport(connection_id="conn_1", outbound_capacity=1)

    transport.write_outbound({"jsonrpc": "2.0", "id": 1, "result": None})
    with pytest.raises(AppServerOverloadedError, match=r"Server overloaded; retry later\."):
        transport.write_outbound({"jsonrpc": "2.0", "id": 2, "result": None})


def test_stdio_malformed_json_returns_parse_error_and_keeps_processing() -> None:
    output = io.StringIO()
    transport = StdioJsonlTransport(
        input_stream=io.StringIO(
            '{bad json}\n[]\n{"id":2,"method":"initialize","params":{}}\n'
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"test"}}}\n'
        ),
        output_stream=output,
    )

    AppServer(transport=transport).run_forever()

    messages = [__import__("json").loads(line) for line in output.getvalue().splitlines()]
    assert messages[0] == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": AppServerErrorCode.PARSE_ERROR, "message": "Parse error"},
    }
    assert messages[1] == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": AppServerErrorCode.INVALID_REQUEST, "message": "Invalid Request"},
    }
    assert messages[2] == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": AppServerErrorCode.INVALID_REQUEST, "message": "Invalid Request"},
    }
    assert messages[3]["id"] == 1
    assert messages[3]["result"]["protocolVersion"] == "v1"


def test_channel_transport_runs_generic_loop_and_disconnects_on_close() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    server = AppServer(transport=transport)
    worker = threading.Thread(target=server.run_forever)
    worker.start()

    transport.send_inbound({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
    assert transport.receive_outbound(timeout=1)["id"] == 1
    transport.close()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert server.processor.connection_state("conn_1").initialized is False
    assert server.router.is_registered("conn_1") is False


def test_generic_server_overload_disconnects_and_cleans_pending_and_subscription_state() -> None:
    transport = ChannelTransport(connection_id="conn_1", outbound_capacity=2)
    server = AppServer(transport=transport)
    server.processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}},
    )
    transport.receive_outbound(timeout=1)
    server.processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 2, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    transport.receive_outbound(timeout=1)
    transport.receive_outbound(timeout=1)
    pending = server.router.send_server_request("conn_1", "approval/request", {"threadId": "thread_1"})
    second_pending = server.router.send_server_request("conn_1", "approval/request", {"threadId": "thread_1"})

    worker = threading.Thread(target=server.run_forever)
    worker.start()
    transport.send_inbound({"jsonrpc": "2.0", "id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    worker.join(timeout=1)

    assert not worker.is_alive()
    with pytest.raises(RuntimeError, match="client_disconnected"):
        pending.result(timeout=0)
    with pytest.raises(RuntimeError, match="client_disconnected"):
        second_pending.result(timeout=0)
    assert server.router.pending_server_request_count() == 0
    assert server.state_manager.subscribers("thread_1") == set()
    assert server.processor.connection_state("conn_1").initialized is False


def test_outbound_failure_disconnects_only_failed_connection() -> None:
    failed = _ToggleFailTransport("failed")
    healthy = ChannelTransport(connection_id="healthy")
    router = OutgoingRouter()
    state_manager = ThreadStateManager()
    router.register_transport(failed)
    router.register_transport(healthy)
    processor = MessageProcessor(router=router, state_manager=state_manager)
    for connection_id in ("failed", "healthy"):
        processor.process_message(
            connection_id,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": connection_id}},
            },
        )
    failed.outbound.clear()
    healthy.receive_outbound(timeout=1)
    processor.process_message(
        "failed",
        {"jsonrpc": "2.0", "id": 2, "method": "thread/start", "params": {"agentKey": "default"}},
    )
    failed.outbound.clear()

    failed.fail_writes = True
    pending = router.send_server_request("failed", "approval/request", {"threadId": "thread_1"})

    with pytest.raises(RuntimeError, match="client_disconnected"):
        pending.result(timeout=0)
    assert processor.connection_state("failed").initialized is False
    assert state_manager.subscribers("thread_1") == set()
    assert router.is_registered("failed") is False
    assert router.is_registered("healthy") is True
    processor.process_message("healthy", {"jsonrpc": "2.0", "id": 3, "method": "model/list"})
    assert healthy.receive_outbound(timeout=1)["id"] == 3


class _ToggleFailTransport:
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.fail_writes = False
        self.outbound: list[dict[str, Any]] = []

    def write_outbound(self, payload: dict[str, Any]) -> None:
        if self.fail_writes:
            raise OSError("send failed")
        self.outbound.append(payload)
