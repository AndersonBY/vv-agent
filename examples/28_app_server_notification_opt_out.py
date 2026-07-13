#!/usr/bin/env python3
"""Real App Server turn with per-connection notification opt-out."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vv_agent import Agent, RunConfig, ToolPolicy
from vv_agent.app_server import ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.server import AppServer
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.events import RunEvent


def print_event(event: RunEvent) -> None:
    if event.type in {"run_completed", "run_failed"}:
        print(f"runtime: {event.to_dict()}", flush=True)


def _host() -> DefaultAppServerHost:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"}
    max_cycles = int(os.getenv("V_AGENT_EXAMPLE_MAX_CYCLES", "3"))

    workspace.mkdir(parents=True, exist_ok=True)
    return DefaultAppServerHost(
        agent=Agent(
            name="notification-example-assistant",
            instructions=(
                "Answer the user directly without inspecting the workspace. "
                "When the short answer is ready, call task_finish with that answer."
            ),
            model=model,
        ),
        run_config=RunConfig(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            max_cycles=max(max_cycles, 1),
            stream=print_event if verbose else None,
            tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME]),
        ),
    )


def _print(label: str, message: dict[str, Any]) -> None:
    print(f"{label}: {json.dumps(message, ensure_ascii=False, separators=(',', ':'))}")


def _send(server: AppServer, transport: ChannelTransport, payload: dict[str, Any]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _drain_until(transport: ChannelTransport, label: str, method: str) -> None:
    while True:
        message = transport.receive_outbound(timeout=180)
        _print(label, message)
        if message.get("method") == method:
            return


def main() -> None:
    prompt = os.getenv(
        "V_AGENT_EXAMPLE_PROMPT",
        "请把这句话翻译成英文: notification opt-out 可以让客户端跳过不需要的状态通知。",
    )
    first = ChannelTransport(connection_id="conn_1")
    server = AppServer(transport=first, host=_host())
    second = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(second)

    server.processor.process_message(
        "conn_1",
        {
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "status-muted-client"},
                "capabilities": {"optOutNotificationMethods": ["thread/status/changed"]},
            },
        },
    )
    server.processor.process_message(
        "conn_2",
        {"id": 10, "method": "initialize", "params": {"clientInfo": {"name": "full-client"}}},
    )
    _print("conn_1", first.receive_outbound(timeout=1))
    _print("conn_2", second.receive_outbound(timeout=1))
    server.processor.process_message("conn_1", {"method": "initialized"})
    server.processor.process_message("conn_2", {"method": "initialized"})

    _send(server, first, {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "./workspace"}})
    _drain_until(first, "conn_1", "thread/started")
    _print("conn_1", first.receive_outbound(timeout=1))

    server.processor.process_message("conn_2", {"id": 11, "method": "thread/resume", "params": {"threadId": "thread_1"}})
    _print("conn_2", second.receive_outbound(timeout=1))

    _send(
        server,
        first,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": prompt}]}},
    )
    _drain_until(first, "conn_1", "turn/completed")
    _drain_until(second, "conn_2", "turn/completed")


if __name__ == "__main__":
    main()
