"""Canonical tool selection from task values and read-only declarations."""

from __future__ import annotations

from typing import Any, Protocol

from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._tool_metadata import ToolMetadata, metadata_policy_denial_source
from vv_agent.kernel._tool_names import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    STOP_BACKGROUND_COMMAND_TOOL_NAME,
    SUB_TASK_STATUS_TOOL_NAME,
    WORKSPACE_TOOLS,
)


class ToolCatalog(Protocol):
    """Read-only declarations; signatures change with schema, exposure or metadata."""

    def planning_signature(self) -> tuple: ...
    def has_tool(self, name: str) -> bool: ...
    def has_schema(self, name: str) -> bool: ...
    def has_executor(self, name: str) -> bool: ...
    def tool_metadata(self, name: str) -> ToolMetadata | None: ...
    def list_openai_schemas(self, *, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return detached schemas in selected order, for model-visible tools only."""
        ...


def plan_tool_names(task: AgentTask, *, memory_usage_percentage: int | None = None) -> list[str]:
    del memory_usage_percentage
    tool_names: list[str] = []

    if task.allow_interruption:
        tool_names.append(ASK_USER_TOOL_NAME)

    if task.use_workspace:
        tool_names.extend(WORKSPACE_TOOLS)

    if task.agent_type == "computer":
        tool_names.extend([BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME, STOP_BACKGROUND_COMMAND_TOOL_NAME])

    if task.sub_agents_enabled:
        tool_names.extend([CREATE_SUB_TASK_TOOL_NAME, SUB_TASK_STATUS_TOOL_NAME])

    if task.metadata.get("available_skills"):
        tool_names.append(ACTIVATE_SKILL_TOOL_NAME)

    if task.native_multimodal:
        tool_names.append(READ_IMAGE_TOOL_NAME)

    if task.extra_tool_names:
        tool_names.extend(task.extra_tool_names)

    if task.exclude_tools:
        excluded = set(task.exclude_tools)
        tool_names = [name for name in tool_names if name not in excluded]

    allowed_tools = task.metadata.get("_vv_agent_allowed_tools")
    if isinstance(allowed_tools, list):
        allowed = {str(name) for name in allowed_tools}
        tool_names = [name for name in tool_names if name in allowed]

    disallowed_tools = task.metadata.get("_vv_agent_disallowed_tools")
    if isinstance(disallowed_tools, list):
        disallowed = {str(name) for name in disallowed_tools}
        tool_names = [name for name in tool_names if name not in disallowed]

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
    registry: ToolCatalog,
    task: AgentTask,
    memory_usage_percentage: int | None = None,
) -> list[dict[str, Any]]:
    tool_names = plan_tool_names(task, memory_usage_percentage=memory_usage_percentage)
    available_names = [
        name
        for name in tool_names
        if registry.has_tool(name)
        and registry.has_schema(name)
        and metadata_policy_denial_source(
            registry.tool_metadata(name),
            denied_side_effects=task.metadata.get("_vv_agent_denied_side_effects", []),
            denied_capability_tags=task.metadata.get(
                "_vv_agent_denied_capability_tags",
                [],
            ),
            deny_terminal_tools=task.metadata.get("_vv_agent_deny_terminal_tools", False),
            denied_cost_dimensions=task.metadata.get(
                "_vv_agent_denied_cost_dimensions",
                [],
            ),
        )
        is None
    ]
    return registry.list_openai_schemas(tool_names=available_names)
