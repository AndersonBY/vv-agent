from __future__ import annotations

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    get_default_tool_schemas,
)
from v_agent.tools.base import ToolSpec
from v_agent.tools.handlers import (
    activate_skill,
    ask_user,
    batch_sub_tasks,
    check_background_command,
    compress_memory,
    create_sub_task,
    file_info,
    file_str_replace,
    list_files,
    read_file,
    read_image,
    run_bash_command,
    task_finish,
    todo_write,
    workspace_grep,
    write_file,
)
from v_agent.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_schemas(get_default_tool_schemas())
    registry.register_many(
        [
            ToolSpec(name=TASK_FINISH_TOOL_NAME, handler=task_finish),
            ToolSpec(name=ASK_USER_TOOL_NAME, handler=ask_user),
            ToolSpec(name=ACTIVATE_SKILL_TOOL_NAME, handler=activate_skill),
            ToolSpec(name=TODO_WRITE_TOOL_NAME, handler=todo_write),
            ToolSpec(name=COMPRESS_MEMORY_TOOL_NAME, handler=compress_memory),
            ToolSpec(name=LIST_FILES_TOOL_NAME, handler=list_files),
            ToolSpec(name=FILE_INFO_TOOL_NAME, handler=file_info),
            ToolSpec(name=READ_FILE_TOOL_NAME, handler=read_file),
            ToolSpec(name=WRITE_FILE_TOOL_NAME, handler=write_file),
            ToolSpec(name=FILE_STR_REPLACE_TOOL_NAME, handler=file_str_replace),
            ToolSpec(name=WORKSPACE_GREP_TOOL_NAME, handler=workspace_grep),
            ToolSpec(name=BASH_TOOL_NAME, handler=run_bash_command),
            ToolSpec(name=CHECK_BACKGROUND_COMMAND_TOOL_NAME, handler=check_background_command),
            ToolSpec(name=CREATE_SUB_TASK_TOOL_NAME, handler=create_sub_task),
            ToolSpec(name=BATCH_SUB_TASKS_TOOL_NAME, handler=batch_sub_tasks),
            ToolSpec(name=READ_IMAGE_TOOL_NAME, handler=read_image),
        ]
    )
    return registry
