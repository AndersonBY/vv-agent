from __future__ import annotations

from v_agent.constants import (
    BATCH_SUB_TASKS_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKSPACE_TOOLS,
)
from v_agent.tools import build_default_registry


def test_registry_exports_backend_style_tool_schemas() -> None:
    registry = build_default_registry()
    schemas = registry.list_openai_schemas()

    assert schemas
    first = schemas[0]
    assert first["type"] == "function"
    assert "function" in first

    names = {schema["function"]["name"] for schema in schemas}
    assert TASK_FINISH_TOOL_NAME in names
    assert READ_FILE_TOOL_NAME in names
    assert CREATE_SUB_TASK_TOOL_NAME in names
    assert BATCH_SUB_TASKS_TOOL_NAME in names
    for tool_name in WORKSPACE_TOOLS:
        assert tool_name in names


def test_schema_description_is_loaded_from_constants() -> None:
    registry = build_default_registry()
    read_schema = registry.get_schema(READ_FILE_TOOL_NAME)

    description = read_schema["function"]["description"]
    assert "workspace" in description.lower()
    assert "line" in description.lower()
