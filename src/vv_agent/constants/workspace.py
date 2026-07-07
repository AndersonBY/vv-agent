from __future__ import annotations

from copy import deepcopy
from typing import Any

from vv_agent.constants.tool_names import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    SUB_TASK_STATUS_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)

ToolSchema = dict[str, Any]

WORKSPACE_TOOLS = [
    FIND_FILES_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
]

WORKSPACE_TOOLS_SCHEMAS: dict[str, ToolSchema] = {
    READ_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_FILE_TOOL_NAME,
            "description": """Read file contents from workspace.

Supported behavior:
- Reads plain UTF-8 text files and returns a content slice.
- Uses 1-based line numbers for `start_line` and `end_line`.
- Can prepend line numbers with `show_line_numbers=true`.
- Enforces read limits per request: max 2000 lines or 50000 characters.
- Large reads return file info payload instead of full content.

Guidance:
- Prefer this tool instead of shell commands like cat/head/tail.
- For large files, read in chunks by line range.
- By default, paths are workspace-relative.
- If runtime metadata enables outside-workspace access, absolute local paths are allowed.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional starting line number (1-based).",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional ending line number (1-based, inclusive).",
                    },
                    "show_line_numbers": {
                        "type": "boolean",
                        "description": "When true, prefixes each output line with its source line number.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    WRITE_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": WRITE_FILE_TOOL_NAME,
            "description": """Write content to a file in workspace.

Use this for creating files, full-file overwrites, or appends.
For small edits inside an existing file, use `edit_file` when the runtime has a current read baseline.
Overwriting an existing file requires a full current baseline from `read_file`, a prior full `write_file`, or `edit_file`.

MODES:
- Overwrite (default): Replaces entire file content.
- Append: Adds to existing content (`append=true`).

WARNING:
- By default, this OVERWRITES the entire file.
- Use `append=true` to add content instead.

PARAMETERS:
- `path` (required): Workspace-relative path by default. Absolute path is allowed when outside-workspace access is enabled.
- `content` (required): Content to write.
- `append` (optional): Set true to append instead of overwrite.
- `leading_newline`/`trailing_newline` (optional): Add newlines when appending.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Set true to append instead of overwrite. Default is false (overwrite).",
                    },
                    "leading_newline": {
                        "type": "boolean",
                        "description": "Add a leading newline when appending. Default is false.",
                    },
                    "trailing_newline": {
                        "type": "boolean",
                        "description": "Add a trailing newline when appending. Default is false.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    FIND_FILES_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": FIND_FILES_TOOL_NAME,
            "description": (
                "Find files in workspace with optional path and glob filtering. "
                "Large results are truncated, and common dependency/cache directories "
                "(like node_modules/.venv) are summarized by default when listing from workspace root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root path. Use workspace-relative path by default; "
                            "absolute path is allowed when outside-workspace access is enabled. "
                            "Default '.'."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob pattern. Default **/*.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether hidden files are included. Default false.",
                    },
                    "include_ignored": {
                        "type": "boolean",
                        "description": (
                            "When listing workspace root, include files under common "
                            "dependency/cache directories. Default false."
                        ),
                    },
                    "include_sensitive": {
                        "type": "boolean",
                        "description": "Include sensitive paths such as .env and private keys. Default false.",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["modified_desc", "path_asc"],
                        "description": (
                            "Sort order. Default modified_desc for local files when mtimes are available, "
                            "otherwise path_asc."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Skip the first N files after filtering and sorting. Default 0.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of file paths returned in one call. "
                            "Default 100; larger values are capped."
                        ),
                    },
                    "scan_limit": {
                        "type": "integer",
                        "description": (
                            "Maximum files scanned before stopping early to keep listing fast. "
                            "If reached, response includes `count_is_estimate=true`."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    FILE_INFO_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": FILE_INFO_TOOL_NAME,
            "description": "Read file metadata in workspace, including size, modified time and type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    SEARCH_FILES_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": SEARCH_FILES_TOOL_NAME,
            "description": """Search workspace files with regex or literal text.

OUTPUT MODES:
- `files_with_matches` (default): show only file paths
- `content`: show matching lines (supports context and line numbers)
- `count`: show per-file match counts

FILTERS:
- `path` + `glob`: scope the search root and file pattern
- `type`: language/file-type shortcut (py/js/ts/md/json/...)
- `literal`: exact text search instead of regex
- default matching uses smart-case: all-lowercase patterns search case-insensitively
  and patterns containing uppercase stay case-sensitive
- `case_sensitive`: explicitly override smart-case behavior
- `multiline`: let `.` match newlines and allow multi-line patterns
- `include_hidden`: include hidden files/directories (default false)
- `include_ignored`: include common dependency/cache roots at workspace root (default false)
- `include_sensitive`: include sensitive files such as .env/private keys (default false)

CONTENT OPTIONS (only for `content` mode):
- `b`: lines before each match
- `a`: lines after each match
- `c`: lines before+after (overrides b/a)
- `n`: include line numbers (default true)

LIMITING:
- `head_limit`: return only first N output rows/entries

Guidance:
- Prefer this tool over ad-hoc shell grep for direct content search.
- Narrow broad searches with `path`/`glob`/`type` for better performance.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern or literal text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root or single file path. Use workspace-relative path by "
                            "default; absolute path is allowed when outside-workspace access is enabled. "
                            "Default '.'."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file glob filter. Default **/*.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether hidden files are included. Default false.",
                    },
                    "include_ignored": {
                        "type": "boolean",
                        "description": (
                            "When searching workspace root, include files under common "
                            "dependency/cache directories. Default false."
                        ),
                    },
                    "include_sensitive": {
                        "type": "boolean",
                        "description": "Include sensitive paths such as .env and private keys. Default false.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": "Search output mode. Default is 'files_with_matches'.",
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as exact text instead of regex. Default false.",
                    },
                    "b": {
                        "type": "integer",
                        "description": "Lines before each match. Only used in content mode.",
                    },
                    "a": {
                        "type": "integer",
                        "description": "Lines after each match. Only used in content mode.",
                    },
                    "c": {
                        "type": "integer",
                        "description": "Context lines before and after each match. Overrides b/a.",
                    },
                    "n": {
                        "type": "boolean",
                        "description": "Whether to include line numbers in content output. Default true.",
                    },
                    "type": {
                        "type": "string",
                        "description": "File type shortcut (e.g. py/js/ts/md/json).",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Skip the first N result rows or entries after sorting. Default 0.",
                    },
                    "head_limit": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Limit to first N output rows/entries. "
                            "Default 250; 0 means unlimited subject to hard caps."
                        ),
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Enable multiline regex mode.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Explicitly override smart-case behavior.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    EDIT_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": EDIT_FILE_TOOL_NAME,
            "description": """Safely edit an existing workspace file by replacing exact text.

When to use:
- Use this for small, local edits to an existing file.
- Call `read_file` first unless the file was just fully written with `write_file`
  or updated by a previous successful edit_file/write_file operation that preserved current context.
  A focused line-range read is enough when your `old_string` comes from that current read state;
  use a full read for broad or uncertain edits.
- Make `old_string` specific enough to match exactly one location.

Rules:
- `edit_file` requires a current read baseline. Appending to an unknown existing file does not count;
  call `read_file` before editing after that case.
- By default, `old_string` must match exactly one location.
- Set `replace_all=true` only when every occurrence should change.
- Do not use this to create files or overwrite entire files; use `write_file` for that.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact source text to replace. Must be non-empty and unique unless replace_all=true.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. May be empty.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string. Default false.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    COMPRESS_MEMORY_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": COMPRESS_MEMORY_TOOL_NAME,
            "description": "Store key summary notes to reduce future context load.",
            "parameters": {
                "type": "object",
                "properties": {
                    "core_information": {
                        "type": "string",
                        "description": "Key information that should be preserved after compression.",
                    },
                },
                "required": ["core_information"],
            },
        },
    },
    TODO_WRITE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": TODO_WRITE_TOOL_NAME,
            "description": """Create and manage structured TODO list for multi-step execution.

Protocol:
- Send the complete `todos` array each time.
- Existing items with matching `id` are updated.
- Items omitted from the new array are removed.
- Only one item may have `status=in_progress`.

Use this tool to keep task planning explicit and machine-readable.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Complete TODO list payload.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Existing TODO id for update; omit for new item.",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "TODO title.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "TODO status.",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                    "description": "TODO priority.",
                                },
                            },
                            "required": ["title", "status", "priority"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    BASH_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": BASH_TOOL_NAME,
            "description": """Execute bash command in workspace.

Guidelines:
- Prefer specialized read/write/search/edit tools when possible.
- Use this tool for command execution, package install, scripts, and piped workflows.
- For commands that may prompt for confirmation, pass `auto_confirm=true` or provide explicit `stdin`.
- Use `run_in_background=true` for long-running commands and poll with check tool.
- If a foreground command hits its timeout, it is automatically moved to a background
  session and returns a `session_id` for polling.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command string."},
                    "exec_dir": {
                        "type": "string",
                        "description": (
                            "Execution directory (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "timeout": {"type": "integer", "description": "Timeout seconds, default 300, max 600."},
                    "stdin": {"type": "string", "description": "Optional stdin content."},
                    "auto_confirm": {"type": "boolean", "description": "Pipe yes to command when true."},
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Run command in background and return session_id for polling.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    CHECK_BACKGROUND_COMMAND_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": CHECK_BACKGROUND_COMMAND_TOOL_NAME,
            "description": (
                "Check status/output for command launched in background mode, "
                "including sessions auto-detached after foreground timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Background session identifier."},
                },
                "required": ["session_id"],
            },
        },
    },
    CREATE_SUB_TASK_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": CREATE_SUB_TASK_TOOL_NAME,
            "description": """Create sub-tasks for a configured sub-agent.

Modes:
- Single task: provide `task_description` (+ optional `output_requirements`)
- Batch task: provide `tasks` array for multiple independent tasks of the same sub-agent

Execution:
- `wait_for_completion=true` (default): wait for result(s) and return final payload
- `wait_for_completion=false`: start background sub-task(s) and return `task_id` / `task_ids`

Use `sub_task_status` later to inspect progress, fetch results, or send follow-up messages.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Sub-agent identifier from configured sub_agents mapping.",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Single-task description. Mutually exclusive with `tasks`.",
                    },
                    "output_requirements": {
                        "type": "string",
                        "description": "Optional output constraints for single-task mode.",
                    },
                    "tasks": {
                        "type": "array",
                        "description": "Batch mode: multiple independent tasks for the same sub-agent.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Task description for one sub-task.",
                                },
                                "output_requirements": {
                                    "type": "string",
                                    "description": "Optional output constraints for one sub-task.",
                                },
                            },
                            "required": ["task_description"],
                        },
                    },
                    "include_main_summary": {
                        "type": "boolean",
                        "description": "Whether to include parent-task summary context. Default false.",
                    },
                    "exclude_files_pattern": {
                        "type": "string",
                        "description": "Optional regex for excluding files in shared context (reserved for compatibility).",
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": "Whether to wait for completion. Default true; false starts background execution.",
                    },
                },
                "required": ["agent_id"],
            },
        },
    },
    SUB_TASK_STATUS_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": SUB_TASK_STATUS_TOOL_NAME,
            "description": """Inspect sub-task status and optionally interact with a sub-task.

Capabilities:
- Query one or more sub-task ids
- Return lightweight snapshot progress (`detail_level=snapshot`)
- Send `message` to the first task id to steer a running task or continue a completed one
- Wait for long-running background sub-task completion without repeated polling (`wait_for_completion=true`)
- Optionally wait for the follow-up response with `wait_for_response=true`

Waiting:
- Use `wait_for_completion=true` when the parent Agent has no useful work until the background sub-task result is available.
- The runtime waits inside this tool call and returns when queried task(s) finish or `max_wait_seconds` is reached.
- Use `check_interval_seconds` as the suggested future re-check interval if the wait reaches its limit.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "description": (
                            "Sub-task ids to query. When `message` is provided, "
                            "only the first id is used as the target."
                        ),
                        "items": {"type": "string"},
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional follow-up or steering message for the first task id.",
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["basic", "snapshot"],
                        "description": (
                            "Status response detail level. `snapshot` includes recent activity, "
                            "latest tool call, and workspace files."
                        ),
                    },
                    "workspace_file_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of workspace files returned per task in snapshot mode. Default 20.",
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": (
                            "Optional. If true and any queried sub-task is still running, wait inside this tool call "
                            "until the task finishes or max_wait_seconds is reached. Use this for long-running "
                            "background sub-tasks when the parent Agent needs the result before continuing."
                        ),
                        "default": False,
                    },
                    "check_interval_seconds": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 1800,
                        "description": (
                            "Optional. Used with wait_for_completion=true. Suggested re-check interval in seconds "
                            "if max_wait_seconds is reached while tasks are still running. Default 300."
                        ),
                        "default": 300,
                    },
                    "max_wait_seconds": {
                        "type": ["integer", "null"],
                        "minimum": 60,
                        "maximum": 86400,
                        "description": (
                            "Optional. Used with wait_for_completion=true. Maximum total wait time before returning "
                            "the current still-running status to the Agent. Null or omitted uses the system default."
                        ),
                        "default": None,
                    },
                    "wait_for_response": {
                        "type": "boolean",
                        "description": "When `message` is provided, wait until the task finishes processing that message.",
                    },
                },
                "required": ["task_ids"],
            },
        },
    },
    READ_IMAGE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_IMAGE_TOOL_NAME,
            "description": (
                "Read image from workspace path or HTTP URL, then attach the image payload "
                "to the next LLM turn as multimodal content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Image path (workspace-relative by default; absolute path allowed when "
                            "outside-workspace access is enabled) or http(s) image URL."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
}

TASK_FINISH_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": TASK_FINISH_TOOL_NAME,
        "description": "When task goals are fully complete, call this tool to end the task and return final message.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final response shown to user.",
                },
                "exposed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional output file paths that should be exposed as final deliverables.",
                },
            },
            "required": [],
        },
    },
}

ASK_USER_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL_NAME,
        "description": "Pause execution and ask the user for required clarification or decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question text to ask the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional answer options shown to the user.",
                },
                "selection_type": {
                    "type": "string",
                    "enum": ["single", "multi"],
                    "description": "Single or multi-choice mode when options are provided.",
                },
                "allow_custom_options": {
                    "type": "boolean",
                    "description": "Whether users can add custom options.",
                },
            },
            "required": ["question"],
        },
    },
}

ACTIVATE_SKILL_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": ACTIVATE_SKILL_TOOL_NAME,
        "description": """Activate a skill from the current task's available skill list.

The skill metadata follows the Agent Skills specification (https://github.com/agentskills/agentskills):
- name/description are exposed in <available_skills>
- skill instructions are loaded from SKILL.md when location is provided

Use this tool only for skills explicitly listed in <available_skills>.""",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Skill identifier from available skill list.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for activating this skill.",
                },
            },
            "required": ["skill_name"],
        },
    },
}


def get_default_tool_schemas() -> dict[str, ToolSchema]:
    merged: dict[str, ToolSchema] = {
        TASK_FINISH_TOOL_NAME: deepcopy(TASK_FINISH_TOOL_SCHEMA),
        ASK_USER_TOOL_NAME: deepcopy(ASK_USER_TOOL_SCHEMA),
        ACTIVATE_SKILL_TOOL_NAME: deepcopy(ACTIVATE_SKILL_TOOL_SCHEMA),
    }
    for tool_name in WORKSPACE_TOOLS:
        merged[tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[tool_name])
    for extra_tool_name in (
        BASH_TOOL_NAME,
        CHECK_BACKGROUND_COMMAND_TOOL_NAME,
        CREATE_SUB_TASK_TOOL_NAME,
        SUB_TASK_STATUS_TOOL_NAME,
        READ_IMAGE_TOOL_NAME,
    ):
        merged[extra_tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[extra_tool_name])
    return merged
