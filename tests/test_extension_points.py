from __future__ import annotations

import json
from pathlib import Path

from v_agent.constants import ACTIVATE_SKILL_TOOL_NAME
from v_agent.runtime.tool_planner import plan_tool_schemas
from v_agent.tools import ToolContext, build_default_registry
from v_agent.types import AgentTask, ToolCall, ToolResultStatus


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, shared_state={"todo_list": []}, cycle_index=1)


def _task(**overrides: object) -> AgentTask:
    task = AgentTask(
        task_id="task_ext",
        model="dummy",
        system_prompt="sys",
        user_prompt="u",
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def test_skill_extension_handler_returns_not_enabled_error(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(id="c1", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert json.loads(result.content)["error_code"] == "skill_activation_not_enabled"


def test_planner_can_inject_skill_extension_tool_schema() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(metadata={"available_skills": ["demo"]}),
    )

    names = {schema["function"]["name"] for schema in schemas}
    assert ACTIVATE_SKILL_TOOL_NAME in names
