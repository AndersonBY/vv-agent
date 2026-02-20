from __future__ import annotations

import json
from pathlib import Path

from v_agent.constants import ACTIVATE_SKILL_TOOL_NAME
from v_agent.runtime.tool_planner import plan_tool_schemas
from v_agent.tools import ToolContext, build_default_registry
from v_agent.types import AgentTask, ToolCall, ToolResultStatus
from v_agent.workspace import LocalWorkspaceBackend


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path, shared_state={"todo_list": []},
        cycle_index=1, workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


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


def test_skill_extension_handler_requires_bound_skills(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(id="c1", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert json.loads(result.content)["error_code"] == "no_bound_skills_configured"


def test_skill_extension_handler_activates_inline_skill(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": [], "available_skills": [{"name": "demo", "instructions": "Do A then B"}]},
        cycle_index=3,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = registry.execute(
        ToolCall(id="c2", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo", "reason": "need workflow"}),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["status"] == "activated"
    assert payload["skill_name"] == "demo"
    assert payload["instructions"] == "Do A then B"
    assert context.shared_state["active_skills"] == ["demo"]
    assert context.shared_state["skill_activation_log"][0]["cycle_index"] == 3


def test_skill_extension_handler_loads_standard_skill_md_from_directory(tmp_path: Path) -> None:
    registry = build_default_registry()
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: Demo skill for tests
---
## Demo skill
Follow this guide.
""",
        encoding="utf-8",
    )
    context = ToolContext(
        workspace=tmp_path,
        shared_state={
            "todo_list": [],
            "available_skills": [{"name": "demo", "skill_directory": str(skill_dir)}],
        },
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = registry.execute(
        ToolCall(id="c3", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert "Demo skill" in payload["instructions"]
    assert payload["location"].endswith("/SKILL.md")
    assert payload["description"] == "Demo skill for tests"


def test_skill_extension_handler_rejects_invalid_standard_skill(tmp_path: Path) -> None:
    registry = build_default_registry()
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("## missing frontmatter", encoding="utf-8")
    context = ToolContext(
        workspace=tmp_path,
        shared_state={
            "todo_list": [],
            "available_skills": [{"name": "demo", "location": str(skill_dir)}],
        },
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = registry.execute(
        ToolCall(id="c4", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    payload = json.loads(result.content)
    assert payload["error_code"] == "skill_invalid"


def test_planner_can_inject_skill_extension_tool_schema() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(metadata={"available_skills": ["demo"]}),
    )

    names = {schema["function"]["name"] for schema in schemas}
    assert ACTIVATE_SKILL_TOOL_NAME in names


def test_skill_extension_handler_can_load_from_skill_root_directory(tmp_path: Path) -> None:
    registry = build_default_registry()
    root = tmp_path / "skills"
    skill_dir = root / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: Demo skill from root
---
## Demo skill
Follow this guide.
""",
        encoding="utf-8",
    )
    context = ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": [], "available_skills": ["skills"]},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )

    result = registry.execute(
        ToolCall(id="c5", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["skill_name"] == "demo"
