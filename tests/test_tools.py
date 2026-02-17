from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from v_agent.tools import ToolContext, build_default_registry
from v_agent.tools.registry import ToolNotFoundError
from v_agent.types import ToolCall, ToolDirective


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, shared_state={"todo_list": []}, cycle_index=1)


def test_workspace_write_and_read(registry, tool_context: ToolContext) -> None:
    write_call = ToolCall(id="call1", name=WRITE_FILE_TOOL_NAME, arguments={"path": "notes/test.txt", "content": "hello"})
    write_result = registry.execute(write_call, tool_context)
    assert write_result.status == "success"

    read_call = ToolCall(id="call2", name=READ_FILE_TOOL_NAME, arguments={"path": "notes/test.txt"})
    read_result = registry.execute(read_call, tool_context)
    payload = json.loads(read_result.content)
    assert payload["content"] == "hello"


def test_workspace_grep(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("hello world\nsecond line", encoding="utf-8")
    call = ToolCall(id="call1", name=WORKSPACE_GREP_TOOL_NAME, arguments={"pattern": "hello"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)
    assert payload["matches"][0]["line"] == 1


def test_todo_finish_guard(registry, tool_context: ToolContext) -> None:
    create_todo = ToolCall(
        id="call1",
        name=TODO_WRITE_TOOL_NAME,
        arguments={"todos": [{"title": "task 1", "status": "pending", "priority": "high"}]},
    )
    registry.execute(create_todo, tool_context)

    finish_call = ToolCall(id="call2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})
    finish_result = registry.execute(finish_call, tool_context)
    payload = json.loads(finish_result.content)

    assert finish_result.status == "error"
    assert finish_result.directive == ToolDirective.CONTINUE
    assert payload["error_code"] == "todo_incomplete"


def test_ask_user_sets_wait_directive(registry, tool_context: ToolContext) -> None:
    call = ToolCall(id="call1", name=ASK_USER_TOOL_NAME, arguments={"question": "Pick one", "options": ["A", "B"]})
    result = registry.execute(call, tool_context)
    assert result.directive == ToolDirective.WAIT_USER
    assert result.metadata["question"] == "Pick one"


def test_unknown_tool_raises(registry, tool_context: ToolContext) -> None:
    call = ToolCall(id="call1", name="missing", arguments={})
    with pytest.raises(ToolNotFoundError):
        registry.execute(call, tool_context)
