#!/usr/bin/env python3
"""SDK-style programmatic wrapper around v-agent runtime.

This follows the structure often seen in agent SDKs:
- AgentDefinition: reusable config unit
- AgentSDKOptions: shared environment and dependency wiring
- AgentSDKClient: named-agent execution entrypoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import SubAgentConfig


def build_example_client(*, settings_file: Path, backend: str, workspace: Path, verbose: bool) -> AgentSDKClient:
    def log_handler(event: str, payload: dict[str, Any]) -> None:
        if event in {"cycle_started", "cycle_llm_response", "tool_result", "run_completed", "run_wait_user", "cycle_failed"}:
            print(f"[{event}] {payload}")

    options = AgentSDKOptions(
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        log_handler=log_handler if verbose else None,
    )

    agents = {
        "planner": AgentDefinition(
            description="你是任务规划 Agent, 先拆任务, 再逐步执行并维护 todo。",
            model="kimi-k2.5",
            max_cycles=10,
            enable_todo_management=True,
        ),
        "translator": AgentDefinition(
            description="你是专业翻译 Agent, 按段翻译并持续写入目标文件。",
            model="MiniMax-M2.5",
            backend="minimax",
            max_cycles=20,
            enable_todo_management=True,
        ),
        "orchestrator": AgentDefinition(
            description="你是主控 Agent, 负责把任务分派给已定义的子 Agent。",
            model="kimi-k2.5",
            enable_sub_agents=True,
            sub_agents={
                "research-sub": SubAgentConfig(
                    model="kimi-k2.5",
                    description="负责背景检索和资料整理。",
                    max_cycles=8,
                ),
                "translate-sub": SubAgentConfig(
                    model="MiniMax-M2.5",
                    backend="minimax",
                    description="负责分段翻译和术语一致性。",
                    max_cycles=12,
                ),
            },
        ),
    }

    return AgentSDKClient(options=options, agents=agents)


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK-style v-agent example with named agents")
    parser.add_argument("--agent", required=True, choices=["planner", "translator", "orchestrator"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["run", "query"], default="run")
    parser.add_argument("--settings-file", default="local_settings.py")
    parser.add_argument("--backend", default="moonshot")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    client = build_example_client(
        settings_file=Path(args.settings_file),
        backend=args.backend,
        workspace=Path(args.workspace),
        verbose=args.verbose,
    )
    if args.mode == "query":
        text = client.query(agent_name=args.agent, prompt=args.prompt, require_completed=False)
        print(text)
        return

    run = client.run_agent(agent_name=args.agent, prompt=args.prompt)
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
