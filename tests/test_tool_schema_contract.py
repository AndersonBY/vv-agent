from __future__ import annotations

import hashlib
import json
from typing import Any

from vv_agent.tools import ToolRegistry, build_default_registry

RUST_CANONICAL_SCHEMA_SHA256 = "24d8f7bde18b11374820f742cfa244c83666626a315e09d4b6e1b69e899a70aa"

CANONICAL_TOOL_NAMES = [
    "task_finish",
    "ask_user",
    "activate_skill",
    "todo_write",
    "compress_memory",
    "find_files",
    "file_info",
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "bash",
    "check_background_command",
    "create_sub_task",
    "sub_task_status",
    "read_image",
]

REQUIRED_FIELDS = {
    "activate_skill": ["skill_name"],
    "ask_user": ["question"],
    "bash": ["command"],
    "check_background_command": ["session_id"],
    "compress_memory": ["core_information"],
    "create_sub_task": ["agent_id"],
    "edit_file": ["path", "old_string", "new_string"],
    "file_info": ["path"],
    "find_files": [],
    "read_file": ["path"],
    "read_image": ["path"],
    "search_files": ["pattern"],
    "sub_task_status": ["task_ids"],
    "task_finish": [],
    "todo_write": ["todos"],
    "write_file": ["path", "content"],
}

PROPERTY_NAMES = {
    "activate_skill": {"skill_name", "reason"},
    "ask_user": {"question", "options", "selection_type", "allow_custom_options"},
    "bash": {"command", "exec_dir", "timeout", "stdin", "auto_confirm", "run_in_background"},
    "check_background_command": {"session_id"},
    "compress_memory": {"core_information"},
    "create_sub_task": {
        "agent_id",
        "task_description",
        "output_requirements",
        "tasks",
        "include_main_summary",
        "exclude_files_pattern",
        "wait_for_completion",
    },
    "edit_file": {"path", "old_string", "new_string", "replace_all"},
    "file_info": {"path"},
    "find_files": {
        "path",
        "glob",
        "include_hidden",
        "include_ignored",
        "include_sensitive",
        "sort",
        "offset",
        "max_results",
        "scan_limit",
    },
    "read_file": {"path", "start_line", "end_line", "show_line_numbers"},
    "read_image": {"path"},
    "search_files": {
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
    },
    "sub_task_status": {
        "task_ids",
        "message",
        "detail_level",
        "workspace_file_limit",
        "wait_for_completion",
        "check_interval_seconds",
        "max_wait_seconds",
        "wait_for_response",
    },
    "task_finish": {"message", "require_all_todos_completed", "exposed_files"},
    "todo_write": {"todos"},
    "write_file": {"path", "content", "append", "leading_newline", "trailing_newline"},
}

SCHEMA_CONSTRAINTS: dict[tuple[str, ...], Any] = {
    ("ask_user", "properties", "selection_type", "enum"): ["single", "multi"],
    ("find_files", "properties", "offset", "minimum"): 0,
    ("find_files", "properties", "sort", "enum"): ["modified_desc", "path_asc"],
    ("read_file", "properties", "end_line", "minimum"): 1,
    ("read_file", "properties", "start_line", "minimum"): 1,
    ("search_files", "properties", "head_limit", "minimum"): 0,
    ("search_files", "properties", "offset", "minimum"): 0,
    ("search_files", "properties", "output_mode", "enum"): ["files_with_matches", "content", "count"],
    ("sub_task_status", "properties", "check_interval_seconds", "default"): 300,
    ("sub_task_status", "properties", "check_interval_seconds", "maximum"): 1800,
    ("sub_task_status", "properties", "check_interval_seconds", "minimum"): 30,
    ("sub_task_status", "properties", "detail_level", "enum"): ["basic", "snapshot"],
    ("sub_task_status", "properties", "max_wait_seconds", "default"): None,
    ("sub_task_status", "properties", "max_wait_seconds", "maximum"): 86400,
    ("sub_task_status", "properties", "max_wait_seconds", "minimum"): 60,
    ("sub_task_status", "properties", "wait_for_completion", "default"): False,
    ("sub_task_status", "properties", "workspace_file_limit", "maximum"): 100,
    ("sub_task_status", "properties", "workspace_file_limit", "minimum"): 1,
    ("todo_write", "properties", "todos", "items", "properties", "priority", "enum"): [
        "low",
        "medium",
        "high",
    ],
    ("todo_write", "properties", "todos", "items", "properties", "status", "enum"): [
        "pending",
        "in_progress",
        "completed",
    ],
}


def _description(registry: ToolRegistry, tool_name: str) -> str:
    return str(registry.get_schema(tool_name)["function"]["description"])


def _property_description(registry: ToolRegistry, tool_name: str, property_name: str) -> str:
    properties = registry.get_schema(tool_name)["function"]["parameters"]["properties"]
    return str(properties[property_name]["description"])


def _collect_constraints(value: Any, path: tuple[str, ...], result: dict[tuple[str, ...], Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, key)
            if key in {"default", "enum", "maximum", "minimum"}:
                result[child_path] = child
            else:
                _collect_constraints(child, child_path, result)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_constraints(child, (*path, str(index)), result)


def test_runtime_schema_export_matches_rust_canonical_snapshot() -> None:
    schemas = build_default_registry().list_openai_schemas()
    canonical_json = json.dumps(schemas, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    assert hashlib.sha256(canonical_json.encode()).hexdigest() == RUST_CANONICAL_SCHEMA_SHA256


def test_builtin_tool_names_required_fields_and_properties_match_canonical_shape() -> None:
    registry = build_default_registry()
    schemas = registry.list_openai_schemas()

    assert [schema["function"]["name"] for schema in schemas] == CANONICAL_TOOL_NAMES

    for tool_name in CANONICAL_TOOL_NAMES:
        parameters = registry.get_schema(tool_name)["function"]["parameters"]
        assert parameters["required"] == REQUIRED_FIELDS[tool_name]
        assert set(parameters["properties"]) == PROPERTY_NAMES[tool_name]

    create_sub_task_items = registry.get_schema("create_sub_task")["function"]["parameters"]["properties"]["tasks"]["items"]
    assert create_sub_task_items["required"] == ["task_description"]
    assert set(create_sub_task_items["properties"]) == {"task_description", "output_requirements"}

    todo_items = registry.get_schema("todo_write")["function"]["parameters"]["properties"]["todos"]["items"]
    assert todo_items["required"] == ["title", "status", "priority"]
    assert set(todo_items["properties"]) == {"id", "title", "status", "priority"}

    max_wait_seconds = registry.get_schema("sub_task_status")["function"]["parameters"]["properties"]["max_wait_seconds"]
    assert max_wait_seconds["type"] == ["integer", "null"]


def test_builtin_tool_enum_default_and_bounds_match_canonical_shape() -> None:
    registry = build_default_registry()
    actual: dict[tuple[str, ...], Any] = {}

    for tool_name in CANONICAL_TOOL_NAMES:
        parameters = registry.get_schema(tool_name)["function"]["parameters"]
        _collect_constraints(parameters, (tool_name,), actual)

    assert actual == SCHEMA_CONSTRAINTS


def test_high_impact_tool_descriptions_keep_operational_guidance() -> None:
    registry = build_default_registry()
    operational_headings = ("When to use:", "Workflow:", "Protocol:", "Guidelines:", "Modes:", "Capabilities:")

    for tool_name in CANONICAL_TOOL_NAMES:
        description = _description(registry, tool_name)
        assert len(description) >= 280, f"{tool_name} description is too short: {description}"
        assert any(heading in description for heading in operational_headings), (
            f"{tool_name} description lacks operational guidance: {description}"
        )

    expected_fragments = {
        "task_finish": ["Only call this when", "unfinished TODO", "runtime rejects premature finish by default"],
        "ask_user": ["Do not use this for facts", "blocks the runtime", "include 2-3 options"],
        "activate_skill": ["Read the returned SKILL.md instructions", "Do not invent skill names"],
        "edit_file": ["exact `old_string`", "never guess whitespace", "fails if `old_string` is not found"],
        "compress_memory": ["durable memory note", "future compaction", "Do not store transient chatter"],
        "file_info": ["Use before reading large or binary files", "before deciding read ranges"],
        "read_image": ["Use this before reasoning about image content", "Supported formats", "5 MiB"],
        "check_background_command": ["Polling protocol:", "background_command_failed", "Returns:"],
    }
    for tool_name, fragments in expected_fragments.items():
        description = _description(registry, tool_name)
        for fragment in fragments:
            assert fragment in description, f"{tool_name} description is missing {fragment!r}"


def test_high_impact_parameters_keep_operational_guidance() -> None:
    registry = build_default_registry()
    expected_fragments = {
        ("task_finish", "exposed_files"): ["workspace-relative", "created or modified", "deliverables"],
        ("ask_user", "question"): ["smallest decision", "unblock progress"],
        ("bash", "stdin"): ["interactive", "confirmation", "heredoc"],
        ("bash", "auto_confirm"): ["non-interactive", "destructive", "authorized"],
        ("search_files", "pattern"): ["regex", "Escape", "literal"],
        ("search_files", "type"): ["shortcut", "unknown", "supported"],
        ("edit_file", "new_string"): ["Replacement", "line endings", "whitespace"],
        ("create_sub_task", "agent_id"): ["Exact", "configured `sub_agents`", "Do not pass"],
        ("create_sub_task", "output_requirements"): ["success criteria", "format", "deliverables"],
        ("create_sub_task", "exclude_files_pattern"): [
            "discovery only",
            "normalized workspace-relative",
            "direct known-path access",
            "not an access-control boundary or sandbox",
        ],
        ("sub_task_status", "workspace_file_limit"): ["snapshot", "noise", "progress"],
        ("read_image", "path"): ["PNG, JPEG, WEBP, or BMP", "HTTP URLs are passed through"],
    }

    for (tool_name, property_name), fragments in expected_fragments.items():
        description = _property_description(registry, tool_name, property_name)
        normalized = description.lower()
        for fragment in fragments:
            assert fragment.lower() in normalized, f"{tool_name}.{property_name} is missing {fragment!r}"

    todo_items = registry.get_schema("todo_write")["function"]["parameters"]["properties"]["todos"]["items"]
    for property_name, fragments in {
        "title": ["actionable", "observable"],
        "status": ["pending", "in_progress", "completed"],
        "priority": ["high", "medium", "low"],
    }.items():
        description = todo_items["properties"][property_name]["description"].lower()
        for fragment in fragments:
            assert fragment in description, f"todo_write.todos.items.{property_name} is missing {fragment!r}"


def test_model_visible_tool_schemas_stay_capability_focused() -> None:
    serialized = json.dumps(build_default_registry().list_openai_schemas(), ensure_ascii=True).lower()
    language = bytes([0x50, 0x79, 0x74, 0x68, 0x6F, 0x6E]).decode()
    joining = bytes([0x63, 0x6F, 0x6D, 0x70, 0x61, 0x74, 0x69, 0x62, 0x69, 0x6C, 0x69, 0x74, 0x79]).decode()
    transition = bytes([0x6D, 0x69, 0x67, 0x72, 0x61, 0x74, 0x69, 0x6F, 0x6E]).decode()
    equality = bytes([0x70, 0x61, 0x72, 0x69, 0x74, 0x79]).decode()
    source = bytes([0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6E, 0x63, 0x65]).decode()
    forbidden_terms = [
        language,
        f"{language} {joining}",
        f"{language}-{joining}",
        f"for {language}",
        f"{language} {source}",
        f"{language}-style",
        joining,
        transition,
        equality,
        f"{joining} alias",
        f"reserved for {joining}",
        "Scalar" + " values",
        "Numeric" + " strings",
        "converted" + " to text",
        "scalar" + " coercion",
    ]

    for forbidden in forbidden_terms:
        assert forbidden.lower() not in serialized, f"model-visible schema contains internal wording {forbidden!r}"
