from __future__ import annotations

from typing import Any

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    DOCUMENT_NAVIGATION_TOOLS,
    DOCUMENT_WRITE_TOOLS,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKFLOW_DESIGN_TOOLS,
    WORKSPACE_TOOLS,
)
from v_agent.tools.registry import ToolRegistry
from v_agent.types import AgentTask


def plan_tool_names(task: AgentTask, *, memory_usage_percentage: int | None = None) -> list[str]:
    tool_names: list[str] = [TASK_FINISH_TOOL_NAME]

    if task.allow_interruption:
        tool_names.append(ASK_USER_TOOL_NAME)

    if task.use_workspace:
        tool_names.extend(WORKSPACE_TOOLS)

    effective_memory_usage = task.metadata.get("memory_usage_percentage")
    memory_pct = effective_memory_usage if isinstance(effective_memory_usage, int) else 0
    if memory_usage_percentage is not None:
        memory_pct = memory_usage_percentage

    if memory_pct >= task.memory_threshold_percentage:
        tool_names.append(COMPRESS_MEMORY_TOOL_NAME)

    if task.agent_type == "computer":
        tool_names.extend([BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME])

    if task.has_sub_agents:
        tool_names.extend([CREATE_SUB_TASK_TOOL_NAME, BATCH_SUB_TASKS_TOOL_NAME])

    if task.metadata.get("available_skills"):
        tool_names.append(ACTIVATE_SKILL_TOOL_NAME)

    if task.enable_document_tools:
        tool_names.extend(DOCUMENT_NAVIGATION_TOOLS)
        if task.enable_document_write_tools:
            tool_names.extend(DOCUMENT_WRITE_TOOLS)

    if task.enable_workflow_tools:
        tool_names.extend(WORKFLOW_DESIGN_TOOLS)

    if task.native_multimodal:
        tool_names.append(READ_IMAGE_TOOL_NAME)

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
