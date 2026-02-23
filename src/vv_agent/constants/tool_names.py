from __future__ import annotations

DEEPSEEK_REASONING_MODELS = (
    "deepseek-reasoner",
    "deepseek-r1-tools",
)

MINIMAX_REASONING_MODELS = (
    "MiniMax-M2.1",
    "MiniMax-M2.1-lightning",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
)

MOONSHOT_REASONING_MODELS = (
    "kimi-k2-thinking",
    "kimi-k2.5",
)

TODO_INCOMPLETE_ERROR_CODE = "todo_incomplete"

ASK_USER_TOOL_NAME = "ask_user"
TASK_FINISH_TOOL_NAME = "task_finish"
READ_FILE_TOOL_NAME = "read_file"
WRITE_FILE_TOOL_NAME = "write_file"
LIST_FILES_TOOL_NAME = "list_files"
DELETE_FILE_TOOL_NAME = "delete_file"
FILE_STR_REPLACE_TOOL_NAME = "file_str_replace"
FILE_LINE_REPLACE_TOOL_NAME = "file_line_replace"
FILE_CONTENT_SEARCH_TOOL_NAME = "file_content_search"
WORKSPACE_GREP_TOOL_NAME = "workspace_grep"
FILE_APPLY_PATCH_TOOL_NAME = "file_apply_patch"
FILE_APPLY_DIFF_TOOL_NAME = "file_apply_diff"
GET_FILE_DOWNLOAD_URL_TOOL_NAME = "get_file_download_url"
DOWNLOAD_FILE_TOOL_NAME = "download_file"
RUN_PYTHON_FILE_TOOL_NAME = "run_python_file"
BASH_TOOL_NAME = "bash"
CHECK_BACKGROUND_COMMAND_TOOL_NAME = "check_background_command"
CREATE_SUB_TASK_TOOL_NAME = "create_sub_task"
BATCH_SUB_TASKS_TOOL_NAME = "batch_sub_tasks"
COMPRESS_MEMORY_TOOL_NAME = "compress_memory"
TODO_WRITE_TOOL_NAME = "todo_write"
TODO_READ_TOOL_NAME = "todo_read"
READ_IMAGE_TOOL_NAME = "read_image"
FILE_INFO_TOOL_NAME = "file_info"
ACTIVATE_SKILL_TOOL_NAME = "activate_skill"
