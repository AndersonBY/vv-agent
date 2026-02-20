#!/usr/bin/env python3
"""Session resume example: handle WAIT_USER and continue with user reply."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.types import AgentStatus


def runtime_log(event: str, payload: dict[str, Any], verbose: bool = True) -> None:
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


def main() -> None:
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
    session.subscribe(lambda event, payload: runtime_log(event, payload, verbose))

    try:
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
    except Exception as e:
        print(f"Error running session: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
