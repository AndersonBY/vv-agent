"""Strict current Message value codec, independent of session storage."""

from __future__ import annotations

import json
import math
from typing import Any, cast

from vv_agent.types import Message, Role, ToolArtifactRef, validate_compaction_metadata

_SESSION_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
        "image_url",
        "metadata",
        "artifact_ref",
    }
)
_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "extra_content"})
_TOOL_FUNCTION_FIELDS = frozenset({"name", "arguments"})


def _decode_canonical_message(data: dict[str, Any]) -> Message:
    _reject_unknown_fields(data, _SESSION_MESSAGE_FIELDS, "Message")
    role = _required_string(data, "role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unknown message role: {role}")
    content = _required_string(data, "content")

    if "tool_calls" not in data:
        tool_calls: list[dict[str, Any]] = []
    else:
        raw_tool_calls = data["tool_calls"]
        if not isinstance(raw_tool_calls, list):
            raise ValueError('"tool_calls" must be an array')
        tool_calls = [_canonical_tool_call(value) for value in raw_tool_calls]

    if "metadata" not in data:
        metadata: dict[str, Any] = {}
    else:
        raw_metadata = data["metadata"]
        if not isinstance(raw_metadata, dict):
            raise ValueError('"metadata" must be an object')
        metadata = cast(dict[str, Any], _canonical_json(raw_metadata, field_name="metadata"))

    artifact_ref: ToolArtifactRef | None = None
    if "artifact_ref" in data:
        raw_artifact_ref = data["artifact_ref"]
        if not isinstance(raw_artifact_ref, dict):
            raise ValueError('"artifact_ref" must be an object')
        try:
            artifact_ref = ToolArtifactRef.from_dict(raw_artifact_ref)
        except (TypeError, ValueError) as exc:
            raise ValueError('"artifact_ref" is invalid') from exc

    validate_compaction_metadata(metadata)
    return Message(
        role=cast(Role, role),
        content=content,
        name=_optional_string(data, "name"),
        tool_call_id=_optional_string(data, "tool_call_id"),
        tool_calls=tool_calls or None,
        reasoning_content=_optional_string(data, "reasoning_content"),
        image_url=_optional_string(data, "image_url"),
        metadata=metadata,
        artifact_ref=artifact_ref,
    )


def _canonical_tool_call(value: Any) -> dict[str, Any]:
    data = _expect_object(value, "ToolCall")
    _reject_unknown_fields(data, _TOOL_CALL_FIELDS, "ToolCall")
    tool_call_id = _required_non_empty_string(data, "id")
    tool_call_type = _required_string(data, "type")
    if tool_call_type != "function":
        raise ValueError(f"unknown tool call type: {tool_call_type}")
    function = _expect_object(data.get("function"), "ToolCall.function")
    _reject_unknown_fields(function, _TOOL_FUNCTION_FIELDS, "ToolCall.function")
    name = _required_non_empty_string(function, "name")
    arguments = _canonical_tool_arguments(_required_string(function, "arguments"))

    canonical: dict[str, Any] = {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    }
    if "extra_content" in data:
        extra_content = data["extra_content"]
        if not isinstance(extra_content, dict):
            raise ValueError('"extra_content" must be an object')
        canonical["extra_content"] = _canonical_json(
            extra_content,
            field_name="extra_content",
        )
    return canonical


def _canonical_tool_arguments(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError('"arguments" must contain a JSON object') from exc
    if not isinstance(decoded, dict):
        raise ValueError('"arguments" must contain a JSON object')
    return cast(dict[str, Any], _canonical_json(decoded, field_name="arguments"))


def _canonical_json(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if value < -(1 << 63) or value > (1 << 64) - 1:
            raise ValueError(f'"{field_name}" contains an integer outside the JSON wire range')
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'"{field_name}" contains a non-finite number')
        return value
    if isinstance(value, list):
        return [_canonical_json(item, field_name=field_name) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f'"{field_name}" object keys must be strings')
        return {key: _canonical_json(value[key], field_name=field_name) for key in sorted(value)}
    raise ValueError(f'"{field_name}" contains a non-JSON value')


def _expect_object(value: Any, type_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{type_name} payload must be an object")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f'missing required string field "{key}"')
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f'field "{key}" must be a string')
    return value


def _required_non_empty_string(data: dict[str, Any], key: str) -> str:
    value = _required_string(data, key)
    if not value:
        raise ValueError(f'field "{key}" must be a non-empty string')
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f'field "{key}" must be a string')
    return value


def _reject_unknown_fields(data: dict[str, Any], allowed: frozenset[str], type_name: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ValueError(f"{type_name} contains unknown fields: {unknown!r}")
