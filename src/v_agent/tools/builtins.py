from __future__ import annotations

from v_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    DOCUMENT_FIND_TOOL_NAME,
    DOCUMENT_GREP_TOOL_NAME,
    DOCUMENT_NAVIGATION_TOOLS_SCHEMAS,
    DOCUMENT_STR_REPLACE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    LIST_MOUNTED_DOCUMENTS_TOOL_NAME,
    READ_DOCUMENT_ABSTRACT_TOOL_NAME,
    READ_DOCUMENT_CONTENT_TOOL_NAME,
    READ_DOCUMENT_OVERVIEW_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    READ_FOLDER_ABSTRACT_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_READ_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_DOCUMENT_CONTENT_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    get_default_tool_schemas,
)
from v_agent.tools.base import ToolSpec
from v_agent.tools.handlers import (
    activate_skill,
    ask_user,
    check_background_command,
    document_abstract_read,
    document_find,
    document_grep,
    document_list_mounted,
    document_overview_read,
    document_read,
    document_str_replace,
    document_write,
    folder_abstract_read,
    list_files,
    read_file,
    read_image,
    run_bash_command,
    task_finish,
    todo_read,
    todo_write,
    workspace_grep,
    write_file,
)
from v_agent.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_schemas(get_default_tool_schemas())
    registry.register_schemas(DOCUMENT_NAVIGATION_TOOLS_SCHEMAS)
    registry.register_many(
        [
            ToolSpec(name=TASK_FINISH_TOOL_NAME, handler=task_finish),
            ToolSpec(name=ASK_USER_TOOL_NAME, handler=ask_user),
            ToolSpec(name=ACTIVATE_SKILL_TOOL_NAME, handler=activate_skill),
            ToolSpec(name=TODO_WRITE_TOOL_NAME, handler=todo_write),
            ToolSpec(name=TODO_READ_TOOL_NAME, handler=todo_read),
            ToolSpec(name=LIST_FILES_TOOL_NAME, handler=list_files),
            ToolSpec(name=READ_FILE_TOOL_NAME, handler=read_file),
            ToolSpec(name=WRITE_FILE_TOOL_NAME, handler=write_file),
            ToolSpec(name=WORKSPACE_GREP_TOOL_NAME, handler=workspace_grep),
            ToolSpec(name=BASH_TOOL_NAME, handler=run_bash_command),
            ToolSpec(name=CHECK_BACKGROUND_COMMAND_TOOL_NAME, handler=check_background_command),
            ToolSpec(name=READ_IMAGE_TOOL_NAME, handler=read_image),
            ToolSpec(name=LIST_MOUNTED_DOCUMENTS_TOOL_NAME, handler=document_list_mounted),
            ToolSpec(name=READ_DOCUMENT_CONTENT_TOOL_NAME, handler=document_read),
            ToolSpec(name=DOCUMENT_GREP_TOOL_NAME, handler=document_grep),
            ToolSpec(name=READ_DOCUMENT_ABSTRACT_TOOL_NAME, handler=document_abstract_read),
            ToolSpec(name=READ_DOCUMENT_OVERVIEW_TOOL_NAME, handler=document_overview_read),
            ToolSpec(name=READ_FOLDER_ABSTRACT_TOOL_NAME, handler=folder_abstract_read),
            ToolSpec(name=DOCUMENT_FIND_TOOL_NAME, handler=document_find),
            ToolSpec(name=WRITE_DOCUMENT_CONTENT_TOOL_NAME, handler=document_write),
            ToolSpec(name=DOCUMENT_STR_REPLACE_TOOL_NAME, handler=document_str_replace),
        ]
    )
    return registry
