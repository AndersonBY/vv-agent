from __future__ import annotations

import json
from pathlib import Path

from vv_agent.constants import ASK_USER_TOOL_NAME, TASK_FINISH_TOOL_NAME, TODO_WRITE_TOOL_NAME
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.types import ToolCall, ToolDirective
from vv_agent.workspace import LocalWorkspaceBackend


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path, shared_state={"todo_list": []},
        cycle_index=1, workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_todo_write_enforces_single_in_progress(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(
            id="c1",
            name=TODO_WRITE_TOOL_NAME,
            arguments={
                "todos": [
                    {"title": "a", "status": "in_progress", "priority": "high"},
                    {"title": "b", "status": "in_progress", "priority": "medium"},
                ]
            },
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload["error_code"] == "multiple_in_progress_todos"


def test_ask_user_returns_structured_selection_metadata(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(
            id="c2",
            name=ASK_USER_TOOL_NAME,
            arguments={
                "question": "Choose",
                "options": ["A", "B", "B"],
                "selection_type": "multi",
                "allow_custom_options": True,
            },
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.directive == ToolDirective.WAIT_USER
    assert payload["selection_type"] == "multi"
    assert payload["allow_custom_options"] is True
    assert payload["options"] == ["A", "B"]


def test_task_finish_blocks_when_todo_incomplete(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    registry.execute(
        ToolCall(
            id="c3",
            name=TODO_WRITE_TOOL_NAME,
            arguments={"todos": [{"title": "step1", "status": "pending", "priority": "medium"}]},
        ),
        context,
    )

    result = registry.execute(
        ToolCall(
            id="c4",
            name=TASK_FINISH_TOOL_NAME,
            arguments={"message": "done"},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert payload["error_code"] == "todo_incomplete"
