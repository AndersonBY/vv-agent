from __future__ import annotations

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKSPACE_TOOLS,
)
from v_agent.runtime.tool_planner import plan_tool_names, plan_tool_schemas
from v_agent.tools import build_default_registry
from v_agent.types import AgentTask, SubAgentConfig


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


def test_plan_tool_names_adds_computer_tools() -> None:
    names = plan_tool_names(_task(agent_type="computer"))

    assert BASH_TOOL_NAME in names
    assert CHECK_BACKGROUND_COMMAND_TOOL_NAME in names


def test_plan_tool_names_adds_sub_agent_tools_when_configured() -> None:
    names = plan_tool_names(
        _task(
            sub_agents={
                "research-sub": SubAgentConfig(model="kimi-k2.5", description="collect context"),
            }
        )
    )

    assert CREATE_SUB_TASK_TOOL_NAME in names
    assert BATCH_SUB_TASKS_TOOL_NAME in names


def test_plan_tool_schemas_adds_sub_agent_tools_when_configured() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(
            sub_agents={
                "research-sub": SubAgentConfig(model="kimi-k2.5", description="collect context"),
            }
        ),
    )
    names = {schema["function"]["name"] for schema in schemas}
    assert CREATE_SUB_TASK_TOOL_NAME in names
    assert BATCH_SUB_TASKS_TOOL_NAME in names


def test_plan_tool_names_includes_extra_tool_names() -> None:
    names = plan_tool_names(_task(extra_tool_names=["_custom_workflow_tool"]))
    assert "_custom_workflow_tool" in names


def test_plan_tool_names_includes_activate_skill_for_bound_skills() -> None:
    names = plan_tool_names(_task(metadata={"bound_skills": [{"name": "demo"}]}))

    assert ACTIVATE_SKILL_TOOL_NAME in names


def test_plan_tool_schemas_only_returns_registered_tools() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(),
    )

    names = {schema["function"]["name"] for schema in schemas}
    for tool_name in WORKSPACE_TOOLS:
        assert tool_name in names
    assert TASK_FINISH_TOOL_NAME in names
