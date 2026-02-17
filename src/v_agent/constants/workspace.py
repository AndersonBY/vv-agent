from __future__ import annotations

from copy import deepcopy
from typing import Any

from v_agent.constants.tool_names import (
    ASK_USER_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
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
            "description": """Update runtime TODO list.

Current protocol (transitional):
- action=replace: replace the entire list with `items`
- action=append: append items to the end
- action=set_done: update one item completion flag

Use this tool for multi-step tasks to keep plan state explicit.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["replace", "append", "set_done"],
                        "description": "Mutation mode for TODO list.",
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "TODO title.",
                                },
                                "done": {
                                    "type": "boolean",
                                    "description": "Completion flag.",
                                },
                            },
                            "required": ["title"],
                        },
                    },
                    "index": {
                        "type": "integer",
                        "description": "Item index when action=set_done.",
                    },
                    "done": {
                        "type": "boolean",
                        "description": "Target completion flag when action=set_done.",
                    },
                },
                "required": ["action"],
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
                "require_all_todos_completed": {
                    "type": "boolean",
                    "description": "When true, reject finish if TODO list contains unfinished items.",
                },
            },
            "required": ["message"],
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


def get_default_tool_schemas() -> dict[str, ToolSchema]:
    merged: dict[str, ToolSchema] = {
        TASK_FINISH_TOOL_NAME: deepcopy(TASK_FINISH_TOOL_SCHEMA),
        ASK_USER_TOOL_NAME: deepcopy(ASK_USER_TOOL_SCHEMA),
    }
    for tool_name in WORKSPACE_TOOLS:
        merged[tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[tool_name])
    return merged
