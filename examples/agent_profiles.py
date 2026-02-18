#!/usr/bin/env python3
"""Programmatic profile examples built on v-agent SDK primitives."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions

profiles: dict[str, AgentDefinition] = {
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

settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
default_backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
profile_name = os.getenv("V_AGENT_EXAMPLE_PROFILE", "researcher")
prompt = os.getenv("V_AGENT_EXAMPLE_PROMPT", "分析 workspace 下这个文档的核心结论")

workspace.mkdir(parents=True, exist_ok=True)

if profile_name not in profiles:
    profile_name = "researcher"

client = AgentSDKClient(
    options=AgentSDKOptions(
        settings_file=settings_file,
        default_backend=default_backend,
        workspace=workspace,
    ),
    agents=profiles,
)

run = client.run_agent(agent_name=profile_name, prompt=prompt)
print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
