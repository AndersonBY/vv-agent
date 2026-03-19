from __future__ import annotations

from types import SimpleNamespace

import vv_agent.runtime.tool_planner as tool_planner_module
from vv_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKSPACE_TOOLS,
)
from vv_agent.runtime.tool_planner import plan_tool_names, plan_tool_schemas
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask, SubAgentConfig


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


def test_plan_tool_names_includes_activate_skill_for_available_skills() -> None:
    names = plan_tool_names(_task(metadata={"available_skills": [{"name": "demo", "description": "Demo"}]}))

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


def test_plan_tool_names_includes_activate_skill_for_skill_dirs_in_available_skills() -> None:
    names = plan_tool_names(_task(metadata={"available_skills": ["skills"]}))

    assert ACTIVATE_SKILL_TOOL_NAME in names


def test_plan_tool_schemas_injects_runtime_shell_hint_for_bash(monkeypatch) -> None:
    registry = build_default_registry()

    def fake_resolve(*, shell: str | None = None, windows_shell_priority: list[str] | None = None):
        del shell, windows_shell_priority
        return SimpleNamespace(
            kind="powershell",
            prefix=["powershell", "-NoLogo", "-NoProfile", "-Command"],
        )

    monkeypatch.setattr(tool_planner_module, "resolve_shell_invocation", fake_resolve)

    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(
            agent_type="computer",
            metadata={"bash_shell": "powershell"},
        ),
    )

    bash_schema = next(schema for schema in schemas if schema["function"]["name"] == BASH_TOOL_NAME)
    description = bash_schema["function"]["description"]
    assert "Runtime shell hint" in description
    assert "powershell" in description
    assert "-NoProfile" in description


def test_plan_tool_schemas_reports_invalid_windows_shell_priority_config() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(
            agent_type="computer",
            metadata={"windows_shell_priority": "git-bash,powershell,cmd"},
        ),
    )

    bash_schema = next(schema for schema in schemas if schema["function"]["name"] == BASH_TOOL_NAME)
    description = bash_schema["function"]["description"]
    assert "Runtime shell hint" in description
    assert "invalid shell config" in description


def test_plan_tool_schemas_freezes_runtime_shell_hint_across_cycles(monkeypatch) -> None:
    registry = build_default_registry()
    task = _task(agent_type="computer")
    call_count = {"value": 0}

    def fake_resolve(*, shell: str | None = None, windows_shell_priority: list[str] | None = None):
        del shell, windows_shell_priority
        call_count["value"] += 1
        index = call_count["value"]
        return SimpleNamespace(
            kind=f"shell-{index}",
            prefix=[f"runner-{index}", "-Command"],
        )

    monkeypatch.setattr(tool_planner_module, "resolve_shell_invocation", fake_resolve)

    first = plan_tool_schemas(
        registry=registry,
        task=task,
    )
    second = plan_tool_schemas(
        registry=registry,
        task=task,
    )

    first_bash = next(schema for schema in first if schema["function"]["name"] == BASH_TOOL_NAME)
    second_bash = next(schema for schema in second if schema["function"]["name"] == BASH_TOOL_NAME)
    first_description = first_bash["function"]["description"]
    second_description = second_bash["function"]["description"]

    assert first_description == second_description
    assert "shell-1" in first_description
    assert call_count["value"] == 1
