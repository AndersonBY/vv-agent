#!/usr/bin/env python3
"""Real stdio JSONL client for an App Server process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

SERVER_CODE = r"""
from pathlib import Path
import os

from vv_agent import Agent, RunConfig, ToolPolicy
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.server import AppServer
from vv_agent.app_server.transport import StdioJsonlTransport
from vv_agent.constants import TASK_FINISH_TOOL_NAME

settings_file = Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py"))
backend = os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3")
workspace = Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
max_cycles = int(os.getenv("VV_AGENT_EXAMPLE_MAX_CYCLES", "3"))
workspace.mkdir(parents=True, exist_ok=True)

host = DefaultAppServerHost(
    agent=Agent(
        name="stdio-app-server-assistant",
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
        tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME]),
    ),
)
AppServer(transport=StdioJsonlTransport(), host=host).run_forever()
"""


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> None:
    prompt = os.getenv(
        "VV_AGENT_EXAMPLE_PROMPT",
        "请把这句话翻译成英文: stdio JSONL 客户端正在和真实 App Server 进程通信。",
    )
    process = subprocess.Popen(
        [sys.executable, "-c", SERVER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    process.stdin.write(_json_line({"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "stdio-example"}}}))
    process.stdin.flush()
    for line in process.stdout:
        stripped = line.rstrip()
        print(stripped)
        if '"id":0' in stripped:
            break

    for payload in [
        {"method": "initialized"},
        {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "./workspace"}},
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": prompt}]}},
    ]:
        process.stdin.write(_json_line(payload))
        process.stdin.flush()

    for line in process.stdout:
        stripped = line.rstrip()
        print(stripped)
        if '"method":"turn/completed"' in stripped:
            break

    process.stdin.close()
    return_code = process.wait(timeout=10)
    if return_code != 0:
        raise SystemExit(process.stderr.read())


if __name__ == "__main__":
    main()
