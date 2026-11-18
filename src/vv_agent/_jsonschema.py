"""The real JSON Schema host primitive shared by SDK and core callers."""

from __future__ import annotations

import json
from functools import lru_cache

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

from vv_agent.kernel._tool_schema import SchemaCheck, SchemaIssue


@lru_cache(maxsize=256)
def _validator_for(serialized_schema: str) -> Validator:
    schema = json.loads(serialized_schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def schema_issues(request: SchemaCheck) -> tuple[SchemaIssue, ...]:
    validator = _validator_for(request.serialized_schema)
    return tuple(
        SchemaIssue(tuple(error.absolute_path), tuple(error.absolute_schema_path), error.validator)
        for error in validator.iter_errors(request.arguments)
    )
