from __future__ import annotations

import json
from pathlib import Path

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    CREATE_WORKFLOW_TOOL_NAME,
    LIST_MOUNTED_DOCUMENTS_TOOL_NAME,
    RUN_WORKFLOW_TOOL_NAME,
)
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


def test_extension_handlers_return_not_enabled_errors(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    doc_result = registry.execute(
        ToolCall(id="c1", name=LIST_MOUNTED_DOCUMENTS_TOOL_NAME, arguments={}),
        context,
    )
    wf_result = registry.execute(
        ToolCall(id="c2", name=CREATE_WORKFLOW_TOOL_NAME, arguments={"json_file_path": "a.json", "title": "t"}),
        context,
    )
    skill_result = registry.execute(
        ToolCall(id="c3", name=ACTIVATE_SKILL_TOOL_NAME, arguments={"skill_name": "demo"}),
        context,
    )

    assert doc_result.status_code == ToolResultStatus.ERROR
    assert wf_result.status_code == ToolResultStatus.ERROR
    assert skill_result.status_code == ToolResultStatus.ERROR
    assert json.loads(doc_result.content)["error_code"] == "document_tools_not_enabled"
    assert json.loads(wf_result.content)["error_code"] == "workflow_tools_not_enabled"
    assert json.loads(skill_result.content)["error_code"] == "skill_activation_not_enabled"


def test_planner_can_inject_extension_tool_schemas() -> None:
    registry = build_default_registry()
    schemas = plan_tool_schemas(
        registry=registry,
        task=_task(
            enable_document_tools=True,
            enable_document_write_tools=True,
            enable_workflow_tools=True,
            metadata={"available_skills": ["demo"]},
        ),
    )

    names = {schema["function"]["name"] for schema in schemas}
    assert LIST_MOUNTED_DOCUMENTS_TOOL_NAME in names
    assert CREATE_WORKFLOW_TOOL_NAME in names
    assert RUN_WORKFLOW_TOOL_NAME in names
    assert ACTIVATE_SKILL_TOOL_NAME in names
