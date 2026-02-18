#!/usr/bin/env python3
"""SDK-style programmatic wrapper around v-agent runtime (non-CLI style)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import SubAgentConfig


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SETTINGS_FILE = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
BACKEND = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
WORKSPACE = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
VERBOSE = _bool_env("V_AGENT_EXAMPLE_VERBOSE", True)
AGENT_NAME = os.getenv("V_AGENT_EXAMPLE_AGENT", "planner")
MODE = os.getenv("V_AGENT_EXAMPLE_MODE", "run")
PROMPT = os.getenv("V_AGENT_EXAMPLE_PROMPT", "先拆分任务, 再逐步完成并汇报")


def build_example_client(*, settings_file: Path, backend: str, workspace: Path, verbose: bool) -> AgentSDKClient:
    def log_handler(event: str, payload: dict[str, Any]) -> None:
        if event in {"cycle_started", "cycle_llm_response", "tool_result", "run_completed", "run_wait_user", "cycle_failed"}:
            print(f"[{event}] {payload}", flush=True)

    options = AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=log_handler if verbose else None,
    )

    agents = {
        "planner": AgentDefinition(
            description="你是任务规划 Agent, 先拆任务, 再逐步执行并维护 todo.",
            model="kimi-k2.5",
            max_cycles=10,
            enable_todo_management=True,
        ),
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

    return AgentSDKClient(options=options, agents=agents)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    client = build_example_client(
        settings_file=SETTINGS_FILE,
        backend=BACKEND,
        workspace=WORKSPACE,
        verbose=VERBOSE,
    )

    selected_agent = AGENT_NAME if AGENT_NAME in {"planner", "translator", "orchestrator"} else "planner"
    selected_mode = MODE if MODE in {"run", "query"} else "run"

    if selected_mode == "query":
        text = client.query(agent_name=selected_agent, prompt=PROMPT, require_completed=False)
        print(text)
        return

    run = client.run_agent(agent_name=selected_agent, prompt=PROMPT)
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
