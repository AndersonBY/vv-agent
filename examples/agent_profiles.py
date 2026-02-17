#!/usr/bin/env python3
"""Programmatic profile examples built on v-agent SDK primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

PROFILES: dict[str, AgentDefinition] = {
    "researcher": AgentDefinition(
        description="你是研究助理, 先检索材料再输出结构化结论。",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=12,
        enable_todo_management=True,
    ),
    "translator": AgentDefinition(
        description="你是专业翻译助理, 按段翻译并持续写入目标文件。",
        backend="minimax",
        model="MiniMax-M2.5",
        max_cycles=20,
        enable_todo_management=True,
    ),
    "computer": AgentDefinition(
        description="你是桌面执行代理, 优先使用工具完成任务。",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=16,
        agent_type="computer",
        enable_document_tools=True,
    ),
}


def run_profile(
    *,
    profile_name: str,
    prompt: str,
    settings_file: Path,
    workspace: Path,
    default_backend: str = "moonshot",
) -> dict[str, object]:
    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=default_backend,
            workspace=workspace,
        ),
        agents=PROFILES,
    )
    run = client.run_agent(agent_name=profile_name, prompt=prompt)
    return run.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v-agent with reusable profile configs")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES.keys()))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--settings-file", default="local_settings.py")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--backend", default="moonshot")
    args = parser.parse_args()

    output = run_profile(
        profile_name=args.profile,
        prompt=args.prompt,
        settings_file=Path(args.settings_file),
        workspace=Path(args.workspace),
        default_backend=args.backend,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
