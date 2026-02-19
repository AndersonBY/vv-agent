#!/usr/bin/env python3
"""SDK-style programmatic wrapper around v-agent runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import SubAgentConfig


def log_handler(event: str, payload: dict[str, Any]) -> None:
    if event in {
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_wait_user",
        "cycle_failed",
    }:
        print(f"[{event}] {payload}", flush=True)


default_agent = AgentDefinition(
    description="你是任务规划 Agent, 先拆任务, 再逐步执行并维护 todo.",
    model="kimi-k2.5",
    max_cycles=10,
    enable_todo_management=True,
)

agents = {
    "translator": AgentDefinition(
        description="你是专业翻译 Agent, 按段翻译并持续写入目标文件.",
        model="MiniMax-M2.5",
        backend="minimax",
        max_cycles=20,
        enable_todo_management=True,
    ),
    "orchestrator": AgentDefinition(
        description="你是主控 Agent, 负责把任务分派给已定义的子 Agent.",
        model="kimi-k2.5",
        enable_sub_agents=True,
        sub_agents={
            "research-sub": SubAgentConfig(
                model="kimi-k2.5",
                description="负责背景检索和资料整理.",
                max_cycles=8,
            ),
            "translate-sub": SubAgentConfig(
                model="MiniMax-M2.5",
                backend="minimax",
                description="负责分段翻译和术语一致性.",
                max_cycles=12,
            ),
        },
    ),
}


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}
    agent_name = os.getenv("V_AGENT_EXAMPLE_AGENT", "default")
    mode = os.getenv("V_AGENT_EXAMPLE_MODE", "run")
    prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "先拆分任务, 再逐步完成并汇报")

    workspace.mkdir(parents=True, exist_ok=True)

    if agent_name != "default" and agent_name not in agents:
        agent_name = "default"
    if mode not in {"run", "query"}:
        mode = "run"

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            log_handler=log_handler if verbose else None,
        ),
        agent=default_agent,
        agents=agents,
    )

    try:
        if mode == "query":
            if agent_name == "default":
                print(client.query(prompt=prompt, require_completed=False))
            else:
                print(client.query(prompt=prompt, agent=agent_name, require_completed=False))
        else:
            run = client.run(prompt=prompt) if agent_name == "default" else client.run(prompt=prompt, agent=agent_name)
            print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
