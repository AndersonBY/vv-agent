from __future__ import annotations

from copy import deepcopy
from typing import Any

from vv_agent import _jsonschema
from vv_agent.kernel import _tool_schema
from vv_agent.kernel._tool_schema import INVALID_TOOL_ARGUMENTS_ERROR_CODE as INVALID_TOOL_ARGUMENTS_ERROR_CODE
from vv_agent.kernel._tool_schema import INVALID_TOOL_ARGUMENTS_MESSAGE as INVALID_TOOL_ARGUMENTS_MESSAGE
from vv_agent.types import ToolExecutionResult


def close_object_schemas(schema: dict[str, Any]) -> dict[str, Any]:
    closed = deepcopy(schema)
    _close_object_schemas_in_place(closed)
    return closed


def validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, str]]:
    validation = _tool_schema.validate_tool_arguments(schema, arguments)
    request = next(validation)
    issues = _jsonschema.schema_issues(request)
    try:
        validation.send(issues)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("schema check did not finish")


def invalid_tool_arguments_result(
    *,
    tool_call_id: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> ToolExecutionResult | None:
    issues = validate_tool_arguments(schema, arguments)
    return _tool_schema.invalid_tool_arguments_result(tool_call_id=tool_call_id, issues=issues)


def assert_valid_tool_schema(schema: dict[str, Any]) -> None:
    _jsonschema._validator_for(_tool_schema._canonical_schema(schema))


def _close_object_schemas_in_place(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _close_object_schemas_in_place(child)
    elif isinstance(value, list):
        for child in value:
            _close_object_schemas_in_place(child)
