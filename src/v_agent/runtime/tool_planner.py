from __future__ import annotations

from typing import Any

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKSPACE_TOOLS,
)
from v_agent.tools.registry import ToolRegistry
from v_agent.types import AgentTask


def plan_tool_names(task: AgentTask, *, memory_usage_percentage: int | None = None) -> list[str]:
    del memory_usage_percentage
    tool_names: list[str] = [TASK_FINISH_TOOL_NAME]

    if task.allow_interruption:
        tool_names.append(ASK_USER_TOOL_NAME)

    if task.use_workspace:
        tool_names.extend(WORKSPACE_TOOLS)

    if task.agent_type == "computer":
        tool_names.extend([BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME])

    if task.sub_agents_enabled:
        tool_names.extend([CREATE_SUB_TASK_TOOL_NAME, BATCH_SUB_TASKS_TOOL_NAME])

    if task.metadata.get("available_skills"):
        tool_names.append(ACTIVATE_SKILL_TOOL_NAME)

    if task.native_multimodal:
        tool_names.append(READ_IMAGE_TOOL_NAME)

    if task.extra_tool_names:
        tool_names.extend(task.extra_tool_names)

    if task.exclude_tools:
        excluded = set(task.exclude_tools)
        tool_names = [name for name in tool_names if name not in excluded]

    deduped: list[str] = []
    seen: set[str] = set()
    for tool_name in tool_names:
        if tool_name in seen:
            continue
        seen.add(tool_name)
        deduped.append(tool_name)
    return deduped


def plan_tool_schemas(
    *,
    registry: ToolRegistry,
    task: AgentTask,
    memory_usage_percentage: int | None = None,
) -> list[dict[str, Any]]:
    tool_names = plan_tool_names(task, memory_usage_percentage=memory_usage_percentage)
    available_names = [name for name in tool_names if registry.has_tool(name) and registry.has_schema(name)]
    return registry.list_openai_schemas(tool_names=available_names)
