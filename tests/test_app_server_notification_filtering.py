from __future__ import annotations

import queue

import pytest

from vv_agent.app_server import ChannelTransport, MessageProcessor, OutgoingRouter


def test_opted_out_connection_does_not_receive_exact_notification() -> None:
    first = ChannelTransport(connection_id="conn_1")
    second = ChannelTransport(connection_id="conn_2")
    router = OutgoingRouter()
    router.register_transport(first)
    router.register_transport(second)
    processor = MessageProcessor(router=router)

    processor.process_message(
        "conn_1",
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "first"},
                "capabilities": {"optOutNotificationMethods": ["item/agentMessage/delta"]},
            },
        },
    )
    processor.process_message("conn_2", {"id": 2, "method": "initialize", "params": {"clientInfo": {"name": "second"}}})
    first.receive_outbound(timeout=1)
    second.receive_outbound(timeout=1)

    router.send_notification("conn_1", "item/agentMessage/delta", {"threadId": "thread_1", "text": "hidden"})
    router.send_notification("conn_2", "item/agentMessage/delta", {"threadId": "thread_1", "text": "visible"})

    with pytest.raises(queue.Empty):
        first.receive_outbound(timeout=0.05)
    assert second.receive_outbound(timeout=1)["method"] == "item/agentMessage/delta"
