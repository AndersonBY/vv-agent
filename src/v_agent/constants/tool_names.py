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

ASK_USER_TOOL_NAME = "_ask_user"
TASK_FINISH_TOOL_NAME = "_task_finish"
READ_FILE_TOOL_NAME = "_read_file"
WRITE_FILE_TOOL_NAME = "_write_file"
LIST_FILES_TOOL_NAME = "_list_files"
DELETE_FILE_TOOL_NAME = "_delete_file"
FILE_STR_REPLACE_TOOL_NAME = "_file_str_replace"
FILE_LINE_REPLACE_TOOL_NAME = "_file_line_replace"
FILE_CONTENT_SEARCH_TOOL_NAME = "_file_content_search"
WORKSPACE_GREP_TOOL_NAME = "_workspace_grep"
FILE_APPLY_PATCH_TOOL_NAME = "_file_apply_patch"
FILE_APPLY_DIFF_TOOL_NAME = "_file_apply_diff"
GET_FILE_DOWNLOAD_URL_TOOL_NAME = "_get_file_download_url"
DOWNLOAD_FILE_TOOL_NAME = "_download_file"
RUN_PYTHON_FILE_TOOL_NAME = "_run_python_file"
BASH_TOOL_NAME = "_bash"
CHECK_BACKGROUND_COMMAND_TOOL_NAME = "_check_background_command"
CREATE_SUB_TASK_TOOL_NAME = "_create_sub_task"
BATCH_SUB_TASKS_TOOL_NAME = "_batch_sub_tasks"
COMPRESS_MEMORY_TOOL_NAME = "_compress_memory"
TODO_WRITE_TOOL_NAME = "_todo_write"
TODO_READ_TOOL_NAME = "_todo_read"
READ_IMAGE_TOOL_NAME = "_read_image"
FILE_INFO_TOOL_NAME = "_file_info"
ACTIVATE_SKILL_TOOL_NAME = "_activate_skill"

LIST_MOUNTED_DOCUMENTS_TOOL_NAME = "document_list_mounted"
DOCUMENT_GREP_TOOL_NAME = "document_grep"
READ_DOCUMENT_CONTENT_TOOL_NAME = "document_read"
DOCUMENT_FIND_TOOL_NAME = "document_find"
WRITE_DOCUMENT_CONTENT_TOOL_NAME = "document_write"
DOCUMENT_STR_REPLACE_TOOL_NAME = "document_str_replace"
READ_DOCUMENT_ABSTRACT_TOOL_NAME = "document_abstract_read"
READ_DOCUMENT_OVERVIEW_TOOL_NAME = "document_overview_read"
READ_FOLDER_ABSTRACT_TOOL_NAME = "folder_abstract_read"
