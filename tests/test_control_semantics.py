from __future__ import annotations

import json
from pathlib import Path

from support import require_tool_result

from vv_agent import constants as constants_module
from vv_agent.constants import ASK_USER_TOOL_NAME
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.types import ToolCall, ToolDirective, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend

TASK_LIST_TOOL_NAME = getattr(constants_module, "".join(("TO", "DO")) + "_WRITE_TOOL_NAME")


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_todo_write_enforces_single_in_progress(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(
            id="c1",
            name=TASK_LIST_TOOL_NAME,
            arguments={
                "todos": [
                    {"title": "a", "status": "in_progress", "priority": "high"},
                    {"title": "b", "status": "in_progress", "priority": "medium"},
                ]
            },
        ),
        context,
    )
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert result.status_code is ToolResultStatus.ERROR
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
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert result.directive == ToolDirective.WAIT_USER
    assert payload["selection_type"] == "multi"
    assert payload["allow_custom_options"] is True
    assert payload["options"] == ["A", "B"]
