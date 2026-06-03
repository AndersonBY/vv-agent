#!/usr/bin/env python3
"""Real App Server lifecycle through ChannelTransport."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vv_agent import Agent, RunConfig, ToolPolicy
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.server import AppServer
from vv_agent.app_server.transport import ChannelTransport
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.events import RunEvent


def print_event(event: RunEvent) -> None:
    if event.type == "assistant_delta":
        print(event.to_dict().get("delta", ""), end="", flush=True)
    elif event.type in {"tool_call_started", "tool_call_completed", "run_completed", "run_failed"}:
        print(f"\n[runtime:{event.type}] {event.to_dict()}", flush=True)


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
            name="app-server-assistant",
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


def _send(server: AppServer, transport: ChannelTransport, payload: dict[str, Any]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _print_message(message: dict[str, Any]) -> None:
    print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))


def _drain_until(transport: ChannelTransport, *, response_id: int | None = None, method: str | None = None) -> None:
    while True:
        message = transport.receive_outbound(timeout=60)
        _print_message(message)
        if response_id is not None and message.get("id") == response_id:
            return
        if method is not None and message.get("method") == method:
            return


def main() -> None:
    prompt = os.getenv(
        "V_AGENT_EXAMPLE_PROMPT",
        "请把这句话翻译成英文: vv-agent App Server 正在通过真实模型处理 ChannelTransport 请求。",
    )
    transport = ChannelTransport(connection_id="example")
    server = AppServer(transport=transport, host=_host())

    _send(
        server,
        transport,
        {
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "channel-example"},
                "capabilities": {"optOutNotificationMethods": []},
            },
        },
    )
    _drain_until(transport, response_id=0)

    _send(server, transport, {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "./workspace"}})
    _drain_until(transport, response_id=1)

    _send(
        server,
        transport,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": prompt}]}},
    )
    _drain_until(transport, method="turn/completed")

    _send(server, transport, {"id": 3, "method": "thread/list"})
    _drain_until(transport, response_id=3)

    _send(server, transport, {"id": 4, "method": "thread/archive", "params": {"threadId": "thread_1"}})
    _drain_until(transport, method="thread/status/changed")


if __name__ == "__main__":
    main()
