from __future__ import annotations

import hashlib
import json
from typing import Any

from vv_agent.constants import BASH_TOOL_NAME
from vv_agent.kernel._tool_planner import plan_tool_names as plan_tool_names
from vv_agent.kernel._tool_planner import plan_tool_schemas as _plan_tool_schemas
from vv_agent.runtime.shell import resolve_shell_invocation
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask

_BASH_RUNTIME_HINT_METADATA_KEY = "_vv_agent_bash_runtime_hint"


def _normalize_shell_value(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("`bash_shell` must be a string shell name")
    value = raw.strip()
    return value or None


def _invalid_shell_hint(error: object) -> str:
    message = str(error).rstrip(".")
    return f"Runtime shell hint: invalid shell config. {message}."


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
    try:
        shell = _normalize_shell_value(task.metadata.get("bash_shell"))
    except ValueError as exc:
        return _invalid_shell_hint(exc)
    windows_shell_priority, config_error = _normalize_windows_shell_priority(task.metadata.get("windows_shell_priority"))
    if config_error:
        return _invalid_shell_hint(config_error)

    try:
        resolved = resolve_shell_invocation(
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )
    except ValueError as exc:
        return _invalid_shell_hint(exc)

    prefix = " ".join(resolved.prefix)
    return f"Runtime shell hint: commands run via `{resolved.kind}` using prefix `{prefix}`."


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


def plan_tool_schemas(
    *,
    registry: ToolRegistry,
    task: AgentTask,
    memory_usage_percentage: int | None = None,
    include_dynamic_hints: bool = True,
) -> list[dict[str, Any]]:
    schemas = _plan_tool_schemas(registry=registry, task=task, memory_usage_percentage=memory_usage_percentage)
    if not include_dynamic_hints:
        return schemas
    return _patch_dynamic_tool_schemas(task=task, tool_schemas=schemas)


def toolset_schema_digest(
    registry: ToolRegistry,
    *,
    task: AgentTask | None = None,
) -> str:
    """Return the digest for a registry or its task-scoped schema projection.

    ``task`` uses vv-agent's canonical planner and deliberately excludes
    request-local dynamic hints.
    """

    schemas = (
        plan_tool_schemas(registry=registry, task=task, include_dynamic_hints=False)
        if task is not None
        else registry.list_openai_schemas()
    )
    canonical = json.dumps(schemas, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
