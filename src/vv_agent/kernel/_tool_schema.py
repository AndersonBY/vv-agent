"""Tool-schema requests and issue/result formatting, independent of the host validator."""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from ._tool_values import ToolDirective, ToolExecutionResult, ToolResultStatus


@dataclass(frozen=True, slots=True)
class SchemaCheck:
    serialized_schema: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    instance_path: tuple[Any, ...]
    schema_path: tuple[Any, ...]
    validator: object


INVALID_TOOL_ARGUMENTS_ERROR_CODE = "invalid_tool_arguments"
INVALID_TOOL_ARGUMENTS_MESSAGE = "Tool arguments do not match the declared schema"


def validate_tool_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> Generator[SchemaCheck, tuple[SchemaIssue, ...], list[dict[str, str]]]:
    issues = yield SchemaCheck(_canonical_schema(schema), arguments)
    issue_keys = {
        (
            _json_pointer(error.instance_path),
            _json_pointer(error.schema_path),
            _validation_rule(error.validator, error.schema_path),
        )
        for error in issues
    }
    return [
        {"instance_path": instance_path, "schema_path": schema_path, "rule": rule}
        for instance_path, schema_path, rule in sorted(issue_keys)
    ]


def invalid_tool_arguments_result(
    *,
    tool_call_id: str,
    issues: list[dict[str, str]],
) -> ToolExecutionResult | None:
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


def _canonical_schema(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_pointer(path: Any) -> str:
    return "".join(f"/{str(segment).replace('~', '~0').replace('/', '~1')}" for segment in path)


def _validation_rule(validator: Any, schema_path: Any) -> str:
    if isinstance(validator, str) and validator:
        return validator
    path = list(schema_path)
    if path and isinstance(path[-1], str):
        return path[-1]
    return "falseSchema"
