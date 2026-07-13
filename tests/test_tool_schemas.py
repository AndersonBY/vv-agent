from __future__ import annotations

from vv_agent.constants import (
    CREATE_SUB_TASK_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    SUB_TASK_STATUS_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WORKSPACE_TOOLS,
)
from vv_agent.tools import build_default_registry


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
    assert SUB_TASK_STATUS_TOOL_NAME in names
    for tool_name in WORKSPACE_TOOLS:
        assert tool_name in names


def test_schema_description_is_loaded_from_constants() -> None:
    registry = build_default_registry()
    read_schema = registry.get_schema(READ_FILE_TOOL_NAME)

    description = read_schema["function"]["description"]
    assert "workspace" in description.lower()
    assert "line" in description.lower()


def test_task_finish_schema_exposes_todo_completion_guard() -> None:
    registry = build_default_registry()
    schema = registry.get_schema(TASK_FINISH_TOOL_NAME)

    parameters = schema["function"]["parameters"]
    require_all_todos_completed = parameters["properties"]["require_all_todos_completed"]

    assert parameters["required"] == []
    assert require_all_todos_completed == {
        "type": "boolean",
        "description": (
            "Default true. When true, finish is rejected while TODO items remain pending or in_progress. "
            "Set false only when intentionally finishing with remaining TODOs, such as when the user "
            "explicitly accepts deferred work."
        ),
    }


def test_create_sub_task_schema_uses_agent_id_only() -> None:
    registry = build_default_registry()
    schema = registry.get_schema(CREATE_SUB_TASK_TOOL_NAME)

    parameters = schema["function"]["parameters"]
    properties = parameters["properties"]

    assert "agent_id" in properties
    assert "agent_name" not in properties
    assert "agent_id" in parameters["required"]


def test_search_and_find_tools_replace_old_workspace_search_names() -> None:
    registry = build_default_registry()
    names = {schema["function"]["name"] for schema in registry.list_openai_schemas()}

    assert SEARCH_FILES_TOOL_NAME in names
    assert FIND_FILES_TOOL_NAME in names
    assert "workspace_grep" not in names
    assert "list_files" not in names
    assert SEARCH_FILES_TOOL_NAME in WORKSPACE_TOOLS
    assert FIND_FILES_TOOL_NAME in WORKSPACE_TOOLS
    assert "workspace_grep" not in WORKSPACE_TOOLS
    assert "list_files" not in WORKSPACE_TOOLS


def test_search_files_schema_uses_clean_search_contract() -> None:
    registry = build_default_registry()
    schema = registry.get_schema(SEARCH_FILES_TOOL_NAME)

    parameters = schema["function"]["parameters"]
    properties = parameters["properties"]
    description = schema["function"]["description"]

    assert parameters["required"] == ["pattern"]
    assert set(properties) == {
        "pattern",
        "path",
        "glob",
        "include_hidden",
        "include_ignored",
        "include_sensitive",
        "output_mode",
        "literal",
        "b",
        "a",
        "c",
        "n",
        "type",
        "offset",
        "head_limit",
        "multiline",
        "case_sensitive",
    }
    assert properties["output_mode"]["enum"] == ["files_with_matches", "content", "count"]
    assert "Default is 'files_with_matches'" in properties["output_mode"]["description"]
    assert "literal" in properties
    assert "offset" in properties
    assert "include_sensitive" in properties
    assert "workspace_grep" not in description
    assert "max_results" not in properties
    assert "i" not in properties


def test_find_files_schema_uses_glob_only_contract() -> None:
    registry = build_default_registry()
    schema = registry.get_schema(FIND_FILES_TOOL_NAME)

    parameters = schema["function"]["parameters"]
    properties = parameters["properties"]

    assert parameters["required"] == []
    assert set(properties) == {
        "path",
        "glob",
        "include_hidden",
        "include_ignored",
        "include_sensitive",
        "sort",
        "offset",
        "max_results",
        "scan_limit",
    }
    assert "pattern" not in properties
    assert properties["sort"]["enum"] == ["modified_desc", "path_asc"]


def test_sub_task_status_schema_supports_long_wait_without_polling() -> None:
    registry = build_default_registry()
    schema = registry.get_schema(SUB_TASK_STATUS_TOOL_NAME)

    properties = schema["function"]["parameters"]["properties"]
    description = schema["function"]["description"]

    assert "wait_for_completion" in properties
    assert "check_interval_seconds" in properties
    assert "max_wait_seconds" in properties
    assert properties["wait_for_completion"]["type"] == "boolean"
    assert properties["check_interval_seconds"]["type"] == "integer"
    assert properties["max_wait_seconds"]["type"] == ["integer", "null"]
    assert "without repeated polling" in description
