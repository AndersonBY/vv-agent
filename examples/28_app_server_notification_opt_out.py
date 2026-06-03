#!/usr/bin/env python3
"""Per-connection notification opt-out with two App Server clients."""

from __future__ import annotations

import json
import queue
from typing import Any

from vv_agent.app_server import ChannelTransport, MessageProcessor, OutgoingRouter


def _print(label: str, message: dict[str, Any]) -> None:
    print(f"{label}: {json.dumps(message, ensure_ascii=False, separators=(',', ':'))}")


def main() -> None:
    router = OutgoingRouter()
    first = ChannelTransport(connection_id="conn_1")
    second = ChannelTransport(connection_id="conn_2")
    router.register_transport(first)
    router.register_transport(second)
    processor = MessageProcessor(router=router)

    processor.process_message(
        "conn_1",
        {
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "timeline-without-deltas"},
                "capabilities": {"optOutNotificationMethods": ["item/agentMessage/delta"]},
            },
        },
    )
    processor.process_message("conn_2", {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "full-stream"}}})
    _print("conn_1", first.receive_outbound(timeout=1))
    _print("conn_2", second.receive_outbound(timeout=1))

    router.send_notification("conn_1", "item/agentMessage/delta", {"threadId": "thread_1", "text": "hidden"})
    router.send_notification("conn_2", "item/agentMessage/delta", {"threadId": "thread_1", "text": "visible"})

    try:
        _print("conn_1", first.receive_outbound(timeout=0.05))
    except queue.Empty:
        print("conn_1: opted out of item/agentMessage/delta")
    _print("conn_2", second.receive_outbound(timeout=1))

    router.broadcast_notification("thread/status/changed", {"threadId": "thread_1", "status": "running"})
    _print("conn_1", first.receive_outbound(timeout=1))
    _print("conn_2", second.receive_outbound(timeout=1))


if __name__ == "__main__":
    main()
