"""Closed session schema checks without SDK or third-party dependencies."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def _compile_check(schema: dict[str, Any]) -> Callable[[Any], bool]:
    """Compile only the closed session schema vocabulary; reject new keywords at import."""
    if not schema:
        return lambda _: True
    if set(schema) == {"anyOf"}:
        checks = tuple(_compile_check(s) for s in schema["anyOf"])
        return lambda v: any(check(v) for check in checks)
    if set(schema) == {"oneOf"}:
        checks = tuple(_compile_check(s) for s in schema["oneOf"])
        return lambda v: sum(check(v) for check in checks) == 1
    if set(schema) == {"const"} and isinstance(schema["const"], str):
        return lambda v: isinstance(v, str) and v == schema["const"]
    kind = schema.get("type")
    if kind == "object":
        if set(schema) == {"type", "additionalProperties"}:
            check = _compile_check(schema["additionalProperties"])
            return lambda v: isinstance(v, dict) and all(check(item) for item in v.values())
        if set(schema) == {"type"}:
            return lambda v: isinstance(v, dict)
        if set(schema) == {"type", "properties", "required", "additionalProperties"}:
            props = schema["properties"]
            assert set(schema["required"]) <= set(props) and schema["additionalProperties"] is False
            required = frozenset(schema["required"])
            fields = tuple((k, _compile_check(s)) for k, s in props.items())
            if required == props.keys():
                return lambda v: isinstance(v, dict) and v.keys() == props.keys() and all(check(v[k]) for k, check in fields)
            return lambda v: (
                isinstance(v, dict)
                and required <= v.keys() <= props.keys()
                and all(k not in v or check(v[k]) for k, check in fields)
            )
    if kind == "array" and set(schema) == {"type", "items"}:
        check = _compile_check(schema["items"])
        return lambda v: isinstance(v, list) and all(check(item) for item in v)
    if kind == "number" and set(schema) <= {"type", "minimum", "maximum", "exclusiveMinimum"}:
        return lambda v: (
            type(v) in (int, float)
            and (
                not v < schema.get("minimum", -float("inf"))
                and not v > schema.get("maximum", float("inf"))
                and not v <= schema.get("exclusiveMinimum", -float("inf"))
            )
        )
    if kind == "integer":
        if set(schema) == {"type", "minimum", "maximum"}:
            return lambda v: type(v) is int and schema["minimum"] <= v <= schema["maximum"]
        if set(schema) == {"type", "minimum"}:
            return lambda v: type(v) is int and v >= schema["minimum"]
        if set(schema) == {"type", "const"}:
            return lambda v: type(v) is int and v == schema["const"]
    if kind == "string":
        if set(schema) == {"type"}:
            return lambda v: isinstance(v, str)
        if set(schema) == {"type", "minLength"}:
            return lambda v: isinstance(v, str) and len(v) >= schema["minLength"]
        if set(schema) == {"type", "enum"}:
            values = frozenset(schema["enum"])
            return lambda v: isinstance(v, str) and v in values
        if set(schema) == {"type", "pattern", "maxLength"}:
            pattern = re.compile(schema["pattern"])
            return lambda v: isinstance(v, str) and pattern.fullmatch(v) is not None
    if schema == {"type": "boolean"}:
        return lambda v: type(v) is bool
    if schema == {"type": "boolean", "const": True}:
        return lambda v: v is True
    if schema == {"type": "null"}:
        return lambda v: v is None
    raise ValueError(f"unsupported session schema: {schema}")
