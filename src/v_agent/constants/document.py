from __future__ import annotations

from typing import Any

from v_agent.constants.tool_names import (
    DOCUMENT_FIND_TOOL_NAME,
    DOCUMENT_GREP_TOOL_NAME,
    DOCUMENT_STR_REPLACE_TOOL_NAME,
    LIST_MOUNTED_DOCUMENTS_TOOL_NAME,
    READ_DOCUMENT_ABSTRACT_TOOL_NAME,
    READ_DOCUMENT_CONTENT_TOOL_NAME,
    READ_DOCUMENT_OVERVIEW_TOOL_NAME,
    READ_FOLDER_ABSTRACT_TOOL_NAME,
    WRITE_DOCUMENT_CONTENT_TOOL_NAME,
)

ToolSchema = dict[str, Any]

DOCUMENT_NAVIGATION_TOOLS = [
    LIST_MOUNTED_DOCUMENTS_TOOL_NAME,
    READ_DOCUMENT_CONTENT_TOOL_NAME,
    DOCUMENT_GREP_TOOL_NAME,
    READ_DOCUMENT_ABSTRACT_TOOL_NAME,
    READ_DOCUMENT_OVERVIEW_TOOL_NAME,
    READ_FOLDER_ABSTRACT_TOOL_NAME,
    DOCUMENT_FIND_TOOL_NAME,
]

DOCUMENT_WRITE_TOOLS = [
    WRITE_DOCUMENT_CONTENT_TOOL_NAME,
    DOCUMENT_STR_REPLACE_TOOL_NAME,
]

DOCUMENT_NAVIGATION_TOOLS_SCHEMAS: dict[str, ToolSchema] = {
    LIST_MOUNTED_DOCUMENTS_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": LIST_MOUNTED_DOCUMENTS_TOOL_NAME,
            "description": "List documents in mounted cloud storage paths with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_filter": {"type": "string"},
                    "file_type_filter": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    READ_DOCUMENT_CONTENT_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_DOCUMENT_CONTENT_TOOL_NAME,
            "description": "Read mounted document full text content (L2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        },
    },
    DOCUMENT_GREP_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": DOCUMENT_GREP_TOOL_NAME,
            "description": "Search patterns in mounted documents using regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path_filter": {"type": "string"},
                    "file_type_filter": {"type": "string"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                    "case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                    "context_lines": {"type": "integer"},
                    "show_line_numbers": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
    },
    READ_DOCUMENT_ABSTRACT_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_DOCUMENT_ABSTRACT_TOOL_NAME,
            "description": "Read document abstract (L0 concise summary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    READ_DOCUMENT_OVERVIEW_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_DOCUMENT_OVERVIEW_TOOL_NAME,
            "description": "Read document overview (L1 structure summary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    READ_FOLDER_ABSTRACT_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_FOLDER_ABSTRACT_TOOL_NAME,
            "description": "Read mounted folder abstract summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_path": {"type": "string"},
                },
                "required": ["folder_path"],
            },
        },
    },
    DOCUMENT_FIND_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": DOCUMENT_FIND_TOOL_NAME,
            "description": "Semantic retrieval across mounted documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "folder_path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    WRITE_DOCUMENT_CONTENT_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": WRITE_DOCUMENT_CONTENT_TOOL_NAME,
            "description": "Write or append mounted document content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                    "leading_newline": {"type": "boolean"},
                    "trailing_newline": {"type": "boolean"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    DOCUMENT_STR_REPLACE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": DOCUMENT_STR_REPLACE_TOOL_NAME,
            "description": "Replace string in mounted document content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["file_path", "old_str", "new_str"],
            },
        },
    },
}
