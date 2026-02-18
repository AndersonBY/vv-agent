#!/usr/bin/env python3
"""Session-first SDK example: multi-turn prompt, steer, follow-up, and event subscription."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)

client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
    ),
    agent=AgentDefinition(
        description="你是会话型任务助理, 能够持续维护上下文并根据插队消息调整执行策略.",
        model=model,
        backend=backend,
        max_cycles=24,
        enable_todo_management=True,
        use_workspace=True,
    ),
)

session = client.create_session()


def on_session_event(event: str, payload: dict[str, Any]) -> None:
    if verbose and event in {
        "session_run_start",
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "session_run_end",
        "session_steer_queued",
        "session_follow_up_queued",
    }:
        print(f"[{event}] {payload}", flush=True)


session.subscribe(on_session_event)

session.steer("如果读取到 README, 请优先总结 README.")
session.follow_up("上一轮完成后, 再给一个 3 条 bullet 的后续建议.")

run = session.prompt("请先快速分析 workspace 当前目录结构, 并给出执行建议.")
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
