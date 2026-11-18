"""Session-schema rejection and JSON object reading without outer dependencies."""

from __future__ import annotations

import json
from typing import Any

from ._canonical_json import canonical_json_bytes
from ._record_schema import _CHECKS


class RecordError(ValueError):
    pass


def validate(value: Any, schema: dict[str, Any], *, canonical: bool = True) -> None:
    try:
        if canonical:
            canonical_json_bytes(value)
        valid = _CHECKS[id(schema)](value)
    except (ValueError, TypeError) as exc:
        raise RecordError(str(exc)) from exc
    if not valid:
        raise RecordError("invalid session schema")


def _load(body: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecordError(f"duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(body, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise RecordError(str(exc)) from exc
    if not isinstance(value, dict):
        raise RecordError("expected object")
    return value
