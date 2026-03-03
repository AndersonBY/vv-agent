from __future__ import annotations

from typing import Any

from vv_agent.constants import (
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
from vv_agent.runtime.shell import resolve_shell_invocation
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask

_BASH_RUNTIME_HINT_METADATA_KEY = "_vv_agent_bash_runtime_hint"


def _normalize_shell_value(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _normalize_windows_shell_priority(raw: Any) -> tuple[list[str] | None, str | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "`windows_shell_priority` must be a list of shell names"

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized, None


def _build_bash_runtime_hint(task: AgentTask) -> str:
    shell = _normalize_shell_value(task.metadata.get("bash_shell"))
    windows_shell_priority, config_error = _normalize_windows_shell_priority(task.metadata.get("windows_shell_priority"))
    if config_error:
        return f"Runtime shell hint: invalid shell config. {config_error}."

    try:
        resolved = resolve_shell_invocation(
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )
    except ValueError as exc:
        return f"Runtime shell hint: unavailable on this host ({exc})."

    prefix = " ".join(resolved.prefix)
    return (
        "Runtime shell hint: "
        f"commands run via `{resolved.kind}` using prefix `{prefix}`."
    )


def _get_or_build_bash_runtime_hint(task: AgentTask) -> str:
    metadata = task.metadata if isinstance(task.metadata, dict) else None
    if metadata is not None:
        cached = metadata.get(_BASH_RUNTIME_HINT_METADATA_KEY)
        if isinstance(cached, str) and cached.strip():
            return cached

    hint = _build_bash_runtime_hint(task)
    if metadata is not None:
        metadata[_BASH_RUNTIME_HINT_METADATA_KEY] = hint
    return hint


def freeze_dynamic_tool_schema_hints(task: AgentTask) -> None:
    if task.agent_type == "computer" or BASH_TOOL_NAME in task.extra_tool_names:
        _get_or_build_bash_runtime_hint(task)


def _patch_dynamic_tool_schemas(
    *,
    task: AgentTask,
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bash_hint: str | None = None
    for schema in tool_schemas:
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != BASH_TOOL_NAME:
            continue
        if bash_hint is None:
            bash_hint = _get_or_build_bash_runtime_hint(task)
        description = function.get("description")
        base_description = description if isinstance(description, str) else ""
        function["description"] = f"{base_description.rstrip()}\n\n{bash_hint}".strip()
    return tool_schemas


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

    if task.metadata.get("available_skills") or task.metadata.get("bound_skills") or task.metadata.get("skill_directories"):
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
    schemas = registry.list_openai_schemas(tool_names=available_names)
    return _patch_dynamic_tool_schemas(task=task, tool_schemas=schemas)
