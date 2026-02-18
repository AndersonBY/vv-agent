#!/usr/bin/env python3
"""Session resume example: handle WAIT_USER and continue with user reply."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import AgentStatus

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")
user_reply = os.getenv(
    "V_AGENT_EXAMPLE_USER_REPLY",
    "请使用正式风格, 输出到 artifacts/session_result.md, 长度控制在 5 条以内。",
)
verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

workspace.mkdir(parents=True, exist_ok=True)


def runtime_log(event: str, payload: dict[str, Any]) -> None:
    if not verbose:
        return
    if event in {
        "session_run_start",
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_wait_user",
        "run_completed",
        "session_run_end",
    }:
        print(f"[{event}] {payload}", flush=True)


client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
    ),
    agent=AgentDefinition(
        description=(
            "你是交互式写作 Agent. 在开始执行前, 必须先调用 `_ask_user` 收集关键偏好;"
            "拿到用户回复后再继续执行。"
        ),
        model=model,
        backend=backend,
        max_cycles=12,
        enable_todo_management=True,
    ),
)

session = client.create_session()
session.subscribe(runtime_log)

first_run = session.prompt(
    "请先询问我输出风格和目标文件, 然后再开始写作计划。",
    auto_follow_up=False,
)
print("[first_run]")
print(json.dumps(first_run.to_dict(), ensure_ascii=False, indent=2))

if first_run.result.status == AgentStatus.WAIT_USER:
    second_run = session.continue_run(user_reply)
    print("[second_run]")
    print(json.dumps(second_run.to_dict(), ensure_ascii=False, indent=2))
