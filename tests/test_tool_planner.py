from __future__ import annotations

from v_agent.constants import ASK_USER_TOOL_NAME, COMPRESS_MEMORY_TOOL_NAME, TASK_FINISH_TOOL_NAME, WORKSPACE_TOOLS
from v_agent.runtime.tool_planner import plan_tool_names, plan_tool_schemas
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask


def _task(**overrides: object) -> AgentTask:
    task = AgentTask(
        task_id="task_planner",
        model="dummy",
        system_prompt="sys",
        user_prompt="user",
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def test_plan_tool_names_default_capabilities() -> None:
    names = plan_tool_names(_task())

    assert names[0] == TASK_FINISH_TOOL_NAME
    assert ASK_USER_TOOL_NAME in names
    for tool_name in WORKSPACE_TOOLS:
        assert tool_name in names


def test_plan_tool_names_respects_flags() -> None:
    names = plan_tool_names(
        _task(
            allow_interruption=False,
            use_workspace=False,
        )
    )

    assert names == [TASK_FINISH_TOOL_NAME]


def test_plan_tool_names_adds_compress_memory_on_threshold() -> None:
    names = plan_tool_names(
        _task(memory_threshold_percentage=80),
        memory_usage_percentage=95,
    )

    assert COMPRESS_MEMORY_TOOL_NAME in names


def test_plan_tool_schemas_only_returns_registered_tools() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(memory_threshold_percentage=80),
        memory_usage_percentage=95,
    )

    names = {schema["function"]["name"] for schema in schemas}
    assert COMPRESS_MEMORY_TOOL_NAME not in names
    assert TASK_FINISH_TOOL_NAME in names
