# ruff: noqa: RUF001

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from vv_agent.skills import normalize_skill_list, render_skills_xml

TASK_FINISH_PROMPT = {
    "en-US": (
        "Use task_finish for an explicit final result. Natural completion is allowed when the configured "
        "no-tool policy permits it."
    ),
    "zh-CN": "可使用 task_finish 显式返回最终结果；若配置的 no-tool policy 允许，也可自然结束。",
}

ASK_USER_PROMPT = {
    "en-US": "Ask the user only for a required decision that cannot be resolved from context or available tools.",
    "zh-CN": "只有缺少无法从上下文或可用工具中获得的必要决策时才询问用户。",
}

WORKSPACE_PROMPT_TEMPLATE = {
    "en-US": "Prefer specialized workspace tools for direct file operations; use bash when they are insufficient.",
    "zh-CN": "直接操作文件时优先使用工作区专用工具；仅在专用工具不足时使用 bash。",
}

TODO_PROMPT = {
    "en-US": "For multi-step work, keep the TODO state current with at most one item in progress.",
    "zh-CN": "多步骤工作中，同一时间只保留一个进行中的 TODO。",
}


def _os_label() -> str:
    system = platform.system()
    if system == "Windows":
        return "Windows"
    if system == "Darwin":
        return "macOS"
    if system == "Linux":
        return "Linux"
    return system or "Unknown OS"


_COMPUTER_OS_LABEL = _os_label()


COMPUTER_AGENT_ENV_PROMPT = {
    "en-US": (f"You are running in a {_COMPUTER_OS_LABEL} workspace environment and can use tools to inspect and modify files."),
    "zh-CN": f"你运行在 {_COMPUTER_OS_LABEL} 工作区环境中, 可以用工具读取, 搜索, 修改文件.",
}

CURRENT_TIME_PROMPT = {
    "en-US": "Actual task start time (UTC):",
    "zh-CN": "任务开始时的真实 UTC 时间:",
}

SUB_AGENT_PROMPT = {
    "en-US": "Configured sub-agents:",
    "zh-CN": "已配置的子 Agent：",
}

SKILL_PROMPT_HEADER = {
    "en-US": "Available skills metadata (Agent Skills format):",
    "zh-CN": "可用技能元数据 (Agent Skills 标准格式):",
}


def render_workspace_tools(language: str) -> str:
    template = WORKSPACE_PROMPT_TEMPLATE.get(language, WORKSPACE_PROMPT_TEMPLATE["en-US"])
    return template


def render_sub_agents(language: str, available_sub_agents: dict[str, str]) -> str:
    header = SUB_AGENT_PROMPT.get(language, SUB_AGENT_PROMPT["en-US"])
    lines = [header]
    for name, description in sorted(available_sub_agents.items()):
        lines.append(f"- agent_id=`{name}`: {description}")
    return "\n".join(lines)


def render_available_skills(
    language: str,
    available_skills: list[dict[str, Any] | str],
    *,
    workspace: Path | None = None,
) -> str:
    header = SKILL_PROMPT_HEADER.get(language, SKILL_PROMPT_HEADER["en-US"])
    entries = normalize_skill_list(available_skills, workspace=workspace)
    if not entries:
        return ""
    return header + "\n" + render_skills_xml(entries)
