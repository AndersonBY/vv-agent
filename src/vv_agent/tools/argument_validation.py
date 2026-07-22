from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

from vv_agent.types import ToolDirective, ToolExecutionResult, ToolResultStatus

INVALID_TOOL_ARGUMENTS_ERROR_CODE = "invalid_tool_arguments"
INVALID_TOOL_ARGUMENTS_MESSAGE = "Tool arguments do not match the declared schema"


def close_object_schemas(schema: dict[str, Any]) -> dict[str, Any]:
    closed = deepcopy(schema)
    _close_object_schemas_in_place(closed)
    return closed


def validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, str]]:
    validator = _validator_for(_canonical_schema(schema))
    issue_keys = {
        (
            _json_pointer(error.absolute_path),
            _json_pointer(error.absolute_schema_path),
            _validation_rule(error.validator, error.absolute_schema_path),
        )
        for error in validator.iter_errors(arguments)
    }
    return [
        {"instance_path": instance_path, "schema_path": schema_path, "rule": rule}
        for instance_path, schema_path, rule in sorted(issue_keys)
    ]


def invalid_tool_arguments_result(
    *,
    tool_call_id: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> ToolExecutionResult | None:
    issues = validate_tool_arguments(schema, arguments)
    if not issues:
        return None
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        status_code=ToolResultStatus.ERROR,
        directive=ToolDirective.CONTINUE,
        error_code=INVALID_TOOL_ARGUMENTS_ERROR_CODE,
        content=json.dumps(
            {
                "ok": False,
                "error": INVALID_TOOL_ARGUMENTS_MESSAGE,
                "error_code": INVALID_TOOL_ARGUMENTS_ERROR_CODE,
                "issues": issues,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        metadata={
            "error_code": INVALID_TOOL_ARGUMENTS_ERROR_CODE,
            "issue_count": len(issues),
        },
    )


def assert_valid_tool_schema(schema: dict[str, Any]) -> None:
    _validator_for(_canonical_schema(schema))


def _close_object_schemas_in_place(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _close_object_schemas_in_place(child)
    elif isinstance(value, list):
        for child in value:
            _close_object_schemas_in_place(child)


def _canonical_schema(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=256)
def _validator_for(serialized_schema: str) -> Validator:
    schema = json.loads(serialized_schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _json_pointer(path: Any) -> str:
    return "".join(f"/{str(segment).replace('~', '~0').replace('/', '~1')}" for segment in path)


def _validation_rule(validator: Any, schema_path: Any) -> str:
    if isinstance(validator, str) and validator:
        return validator
    path = list(schema_path)
    if path and isinstance(path[-1], str):
        return path[-1]
    return "falseSchema"
