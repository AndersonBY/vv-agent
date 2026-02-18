from __future__ import annotations

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    FILE_LINE_REPLACE_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WORKSPACE_TOOLS,
    WRITE_FILE_TOOL_NAME,
)

TASK_FINISH_PROMPT = {
    "en-US": (
        f"When you confirm task completion, you must call `{TASK_FINISH_TOOL_NAME}` "
        "and put the final user-facing result in its `message` field."
    ),
    "zh-CN": (
        f"当你确认任务完成时, 必须调用 `{TASK_FINISH_TOOL_NAME}`, "
        "并在 `message` 字段里给出面向用户的最终结果."
    ),
}

ASK_USER_PROMPT = {
    "en-US": (
        f"When you need clarification or decision from the user, call `{ASK_USER_TOOL_NAME}`. "
        "You may provide options and set selection mode."
    ),
    "zh-CN": (
        f"当你需要用户补充信息或做出选择时, 调用 `{ASK_USER_TOOL_NAME}`. "
        "你可以提供 options 并设置选择模式."
    ),
}

WORKSPACE_PROMPT_TEMPLATE = {
    "en-US": "You can operate workspace files with tools: {tools}.",
    "zh-CN": "你可以使用这些工具操作工作区文件: {tools}.",
}

TODO_PROMPT = {
    "en-US": (
        f"Use `{TODO_WRITE_TOOL_NAME}` for multi-step tasks and keep progress updated. "
        "Only one item should be in progress at a time."
    ),
    "zh-CN": (
        f"多步骤任务请使用 `{TODO_WRITE_TOOL_NAME}` 管理任务清单并及时更新状态, "
        "任意时刻仅保留一个 in_progress."
    ),
}

TOOL_PRIORITY_PROMPT = {
    "en-US": (
        "Tool priority: prefer specialized tools over shell commands. "
        f"Read with `{READ_FILE_TOOL_NAME}`, write with `{WRITE_FILE_TOOL_NAME}`, "
        f"edit with `{FILE_STR_REPLACE_TOOL_NAME}`/`{FILE_LINE_REPLACE_TOOL_NAME}`, "
        f"search with `{WORKSPACE_GREP_TOOL_NAME}`. "
        f"Use `{BASH_TOOL_NAME}` only when specialized tools are insufficient."
    ),
    "zh-CN": (
        "工具优先级: 优先使用专用工具而不是 shell. "
        f"读取用 `{READ_FILE_TOOL_NAME}`, 写入用 `{WRITE_FILE_TOOL_NAME}`, "
        f"编辑用 `{FILE_STR_REPLACE_TOOL_NAME}`/`{FILE_LINE_REPLACE_TOOL_NAME}`, "
        f"搜索用 `{WORKSPACE_GREP_TOOL_NAME}`. "
        f"仅在专用工具不足时使用 `{BASH_TOOL_NAME}`."
    ),
}

COMPUTER_AGENT_ENV_PROMPT = {
    "en-US": "You are running in a Linux workspace environment and can use tools to inspect and modify files.",
    "zh-CN": "你运行在 Linux 工作区环境中, 可以用工具读取, 搜索, 修改文件.",
}

CURRENT_TIME_PROMPT = {
    "en-US": "Actual task start time (UTC):",
    "zh-CN": "任务开始时的真实 UTC 时间:",
}

SUB_AGENT_PROMPT = {
    "en-US": (
        f"If sub-agents are configured, delegate focused tasks with `{CREATE_SUB_TASK_TOOL_NAME}`. "
        f"For multiple independent tasks of the same agent, use `{BATCH_SUB_TASKS_TOOL_NAME}`."
    ),
    "zh-CN": (
        f"如果已配置子 Agent, 可使用 `{CREATE_SUB_TASK_TOOL_NAME}` 委派单个子任务; "
        f"同类型并行任务可使用 `{BATCH_SUB_TASKS_TOOL_NAME}` 批量委派。"
    ),
}


def render_workspace_tools(language: str) -> str:
    template = WORKSPACE_PROMPT_TEMPLATE.get(language, WORKSPACE_PROMPT_TEMPLATE["en-US"])
    return template.format(tools=", ".join(WORKSPACE_TOOLS))


def render_sub_agents(language: str, available_sub_agents: dict[str, str]) -> str:
    header = SUB_AGENT_PROMPT.get(language, SUB_AGENT_PROMPT["en-US"])
    lines = [header, "Available sub-agents:"]
    for name, description in sorted(available_sub_agents.items()):
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)
