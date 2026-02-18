#!/usr/bin/env python3
"""Programmatic profile examples built on v-agent SDK primitives (non-CLI style)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

PROFILES: dict[str, AgentDefinition] = {
    "researcher": AgentDefinition(
        description="你是研究助理, 先检索材料再输出结构化结论.",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=12,
        enable_todo_management=True,
    ),
    "translator": AgentDefinition(
        description="你是专业翻译助理, 按段翻译并持续写入目标文件.",
        backend="minimax",
        model="MiniMax-M2.5",
        max_cycles=20,
        enable_todo_management=True,
    ),
    "computer": AgentDefinition(
        description="你是桌面执行代理, 优先使用工具完成任务.",
        backend="moonshot",
        model="kimi-k2.5",
        max_cycles=16,
        agent_type="computer",
    ),
}

SETTINGS_FILE = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
WORKSPACE = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
DEFAULT_BACKEND = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
PROFILE_NAME = os.getenv("V_AGENT_EXAMPLE_PROFILE", "researcher")
PROMPT = os.getenv(
    "V_AGENT_EXAMPLE_PROMPT",
    "分析 workspace 下这个文档的核心结论",
)


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
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    selected_profile = PROFILE_NAME if PROFILE_NAME in PROFILES else "researcher"
    output = run_profile(
        profile_name=selected_profile,
        prompt=PROMPT,
        settings_file=SETTINGS_FILE,
        workspace=WORKSPACE,
        default_backend=DEFAULT_BACKEND,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
