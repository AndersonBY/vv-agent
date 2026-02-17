from __future__ import annotations

from typing import Any

from v_agent.constants.tool_names import CREATE_WORKFLOW_TOOL_NAME, RUN_WORKFLOW_TOOL_NAME, WRITE_FILE_TOOL_NAME

ToolSchema = dict[str, Any]

WORKFLOW_DESIGN_TOOLS = [
    CREATE_WORKFLOW_TOOL_NAME,
    RUN_WORKFLOW_TOOL_NAME,
]

CREATE_WORKFLOW_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": CREATE_WORKFLOW_TOOL_NAME,
        "description": f"""Create a workflow from a JSON file in workspace.

Recommended flow:
1. Write workflow JSON file via {WRITE_FILE_TOOL_NAME}
2. Call this tool with json_file_path and title
3. Use {RUN_WORKFLOW_TOOL_NAME} for user validation""",
        "parameters": {
            "type": "object",
            "properties": {
                "json_file_path": {"type": "string"},
                "title": {"type": "string"},
                "brief": {"type": "string"},
            },
            "required": ["json_file_path", "title"],
        },
    },
}

RUN_WORKFLOW_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": RUN_WORKFLOW_TOOL_NAME,
        "description": "Run workflow for user verification and feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["workflow_id"],
        },
    },
}

WORKFLOW_DESIGN_TOOLS_SCHEMAS: dict[str, ToolSchema] = {
    CREATE_WORKFLOW_TOOL_NAME: CREATE_WORKFLOW_TOOL_SCHEMA,
    RUN_WORKFLOW_TOOL_NAME: RUN_WORKFLOW_TOOL_SCHEMA,
}
