from __future__ import annotations

from copy import deepcopy
from typing import Any

from v_agent.constants.tool_names import (
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
)

ToolSchema = dict[str, Any]

WORKSPACE_TOOLS = [
    LIST_FILES_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
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
- Paths are workspace-relative and cannot escape workspace root.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the target file.",
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

MODES:
- Overwrite (default): Replaces entire file content.
- Append: Adds to existing content (`append=true`).

WARNING:
- By default, this OVERWRITES the entire file.
- Use `append=true` to add content instead.

PARAMETERS:
- `path` (required): Relative path from workspace root.
- `content` (required): Content to write.
- `append` (optional): Set true to append instead of overwrite.
- `leading_newline`/`trailing_newline` (optional): Add newlines when appending.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative path of the file to write to from the workspace root.",
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
    LIST_FILES_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": LIST_FILES_TOOL_NAME,
            "description": "List files in workspace with optional path and glob filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional root path relative to workspace. Default ..",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob pattern. Default **/*.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether hidden files are included. Default false.",
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
                        "description": "Workspace-relative target file path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    WORKSPACE_GREP_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": WORKSPACE_GREP_TOOL_NAME,
            "description": """Search text across workspace files with regex.

Capabilities:
- Supports regex pattern matching.
- Supports path and glob filters.
- Returns file path, line number, and matching text.

Guidance:
- Prefer this tool over ad-hoc shell grep for direct search tasks.
- Limit broad searches with path/glob and max_results.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional search root relative to workspace. Default ..",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file glob filter. Default **/*.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case sensitive search when true.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum match rows to return. Default 50.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    FILE_STR_REPLACE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": FILE_STR_REPLACE_TOOL_NAME,
            "description": "Replace text in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The source text to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all matches when true. Default false.",
                    },
                    "max_replacements": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional cap when replace_all=false. Default 1.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
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
- Use `run_in_background=true` for long-running commands and poll with check tool.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command string."},
                    "exec_dir": {"type": "string", "description": "Workspace-relative execution directory."},
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
            "description": "Check status/output for command launched in background mode.",
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
            "description": """Create one sub-task and let a configured sub-agent execute it.

Use cases:
- Delegate a focused sub-problem to a specialized sub-agent.
- Keep parent context concise while getting targeted output.

Input rules:
- `agent_name` should match one key from configured `sub_agents`.
- `task_description` should be specific and actionable.
- `output_requirements` can constrain response format.
- `agent_id` is accepted as a compatibility alias for `agent_name`.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Sub-agent name from configured sub_agents mapping.",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Compatibility alias of agent_name.",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Task description passed to the sub-agent.",
                    },
                    "output_requirements": {
                        "type": "string",
                        "description": "Optional output format or content constraints.",
                    },
                    "include_main_summary": {
                        "type": "boolean",
                        "description": "Whether to include parent-task summary context. Default false.",
                    },
                    "exclude_files_pattern": {
                        "type": "string",
                        "description": "Optional regex for excluding files in shared context (reserved for compatibility).",
                    },
                },
                "required": ["task_description"],
            },
        },
    },
    BATCH_SUB_TASKS_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": BATCH_SUB_TASKS_TOOL_NAME,
            "description": """Create multiple sub-tasks for one sub-agent and aggregate results.

Use this when you have multiple independent tasks of the same skill type.
Each task can define its own output requirement. The tool returns a summary
with per-item execution result.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Sub-agent name from configured sub_agents mapping.",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Compatibility alias of agent_name.",
                    },
                    "tasks": {
                        "type": "array",
                        "description": "Batch sub-task definitions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Task description for one sub-task.",
                                },
                                "output_requirements": {
                                    "type": "string",
                                    "description": "Optional per-task output requirements.",
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
                },
                "required": ["tasks"],
            },
        },
    },
    READ_IMAGE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_IMAGE_TOOL_NAME,
            "description": "Read image from workspace path or HTTP URL and attach it to runtime context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative image path or http(s) image URL.",
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
        "description": "Activate a named skill and load its instructions into current task context.",
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
        BATCH_SUB_TASKS_TOOL_NAME,
        READ_IMAGE_TOOL_NAME,
    ):
        merged[extra_tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[extra_tool_name])
    return merged
