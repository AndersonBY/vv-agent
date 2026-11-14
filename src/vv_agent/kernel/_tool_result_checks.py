"""Definitive tool-result and structured-body checks without record dependencies."""

from __future__ import annotations

import json
from typing import Any

from ._tool_values import ToolExecutionResult, ToolResultStatus


def _is_ambiguous_tool_error(result: Any) -> bool:
    """Return whether a tool error lacks an adapter-proven definitive outcome."""
    return (
        isinstance(result, ToolExecutionResult)
        and result.status_code is ToolResultStatus.ERROR
        and result.error_code
        in {
            "tool_timeout",
            "tool_cancelled",
            "tool_connection_lost",
            "tool_execution_failed",
            "tool_orchestrator_error",
        }
        and result.metadata.get("definitive_outcome") is not True
    )


class DefinitiveResultInvalid(ValueError):
    def __init__(
        self,
        message: str = "deferred resolution result is not definitive",
        *,
        code: str = "deferred_resolution_result_invalid",
    ) -> None:
        super().__init__(message)
        self.code = code


def validate_definitive_result(result: Any) -> None:
    if not isinstance(result, ToolExecutionResult) or result.status_code not in {
        ToolResultStatus.SUCCESS,
        ToolResultStatus.ERROR,
    }:
        raise DefinitiveResultInvalid()
    if result.status_code is ToolResultStatus.SUCCESS and result.error_code is not None:
        raise DefinitiveResultInvalid("tool_result_invalid", code="tool_result_invalid")
    if result.status_code is ToolResultStatus.ERROR and _is_ambiguous_tool_error(result):
        raise DefinitiveResultInvalid()


def result_kind(result):
    kind = result.metadata.get("output_type", "text")
    return kind if isinstance(kind, str) and kind in {"text", "json", "image", "file", "error"} else "text"


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant")


def validate_output_body(result):
    kind = result_kind(result)
    if kind == "text" or result.truncated:
        return
    body = json.loads(result.content, parse_constant=_reject_json_constant)
    if kind == "json":
        return
    if not isinstance(body, dict):
        raise ValueError("structured tool body must be an object")
    if kind in {"file", "image"}:
        fields = {"path", "mime_type"} | ({"url"} if kind == "image" else set())
        if set(body) != fields or any(value is not None and not isinstance(value, str) for value in body.values()):
            raise ValueError("invalid file/image body")
        if kind == "file" and not body["path"]:
            raise ValueError("file output needs a path")
    elif (
        set(body) != {"ok", "error", "error_code", "retryable"}
        or body["ok"] is not False
        or not isinstance(body["error"], str)
        or (body["error_code"] is not None and not isinstance(body["error_code"], str))
        or type(body["retryable"]) is not bool
    ):
        raise ValueError("invalid error body")


def provider_result_identity(raw):
    """Retention is kernel-owned enrichment, not part of the provider's original receipt."""
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        return raw
    result = dict(raw)
    metadata = {k: v for k, v in raw["metadata"].items() if k not in {"app_server_resources", "vv_artifact"}}
    if metadata:
        result["metadata"] = metadata
    else:
        result.pop("metadata", None)
    return result
