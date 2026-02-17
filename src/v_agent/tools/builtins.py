from __future__ import annotations

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_READ_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    get_default_tool_schemas,
)
from v_agent.tools.base import ToolSpec
from v_agent.tools.handlers import ask_user, list_files, read_file, task_finish, todo_read, todo_write, workspace_grep, write_file
from v_agent.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_schemas(get_default_tool_schemas())
    registry.register_many(
        [
            ToolSpec(name=TASK_FINISH_TOOL_NAME, handler=task_finish),
            ToolSpec(name=ASK_USER_TOOL_NAME, handler=ask_user),
            ToolSpec(name=TODO_WRITE_TOOL_NAME, handler=todo_write),
            ToolSpec(name=TODO_READ_TOOL_NAME, handler=todo_read),
            ToolSpec(name=LIST_FILES_TOOL_NAME, handler=list_files),
            ToolSpec(name=READ_FILE_TOOL_NAME, handler=read_file),
            ToolSpec(name=WRITE_FILE_TOOL_NAME, handler=write_file),
            ToolSpec(name=WORKSPACE_GREP_TOOL_NAME, handler=workspace_grep),
        ]
    )
    return registry
