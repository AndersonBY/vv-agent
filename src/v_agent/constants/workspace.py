from __future__ import annotations

from copy import deepcopy
from typing import Any

from v_agent.constants.tool_names import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_READ_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)

ToolSchema = dict[str, Any]

WORKSPACE_TOOLS = [
    READ_FILE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    TODO_READ_TOOL_NAME,
]

WORKSPACE_TOOLS_SCHEMAS: dict[str, ToolSchema] = {
    READ_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_FILE_TOOL_NAME,
            "description": """Read file contents from workspace.

Supported behavior:
- Reads UTF-8 text files and returns a content slice.
- Uses 1-based line numbers for `start_line` and `end_line`.
- Returns a structured JSON payload with path and selected content.

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

Modes:
- Overwrite mode by default.
- Append mode when `append=true`.

Guidance:
- Prefer this tool instead of shell redirection.
- Use append for incremental logs or notes.
- Ensure content is final before overwrite.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Append instead of overwrite. Default false.",
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
    TODO_READ_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": TODO_READ_TOOL_NAME,
            "description": "Read current TODO list from runtime shared state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
    for extra_tool_name in (BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME, READ_IMAGE_TOOL_NAME):
        merged[extra_tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[extra_tool_name])
    return merged
