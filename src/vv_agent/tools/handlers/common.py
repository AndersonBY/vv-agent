from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeGuard

from vv_agent.tools.base import ToolContext
from vv_agent.types import ToolExecutionResult, ToolResultStatus


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def builtin_error(
    message: str,
    error_code: str,
    *,
    details: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    payload: dict[str, Any] = {
        "ok": False,
        "error": message,
        "error_code": error_code,
    }
    if details:
        payload.update(details)
    host_metadata: dict[str, Any] = {"error_code": error_code}
    if metadata:
        host_metadata.update(metadata)
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json(payload),
        metadata=host_metadata,
    )


def select_metadata(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload and payload[key] is not None}


def trim_portable_whitespace(value: str) -> str:
    return value.strip().strip("\x1c\x1d\x1e\x1f")


def is_string_keyed_dict(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def get_todo_list(shared_state: dict[str, Any]) -> list[dict[str, Any]]:
    todo = shared_state.get("todo_list")
    if isinstance(todo, list):
        return todo
    shared_state["todo_list"] = []
    return shared_state["todo_list"]


def normalize_todo_items(raw_items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return normalized

    for item in raw_items:
        if not is_string_keyed_dict(item):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        normalized.append({"title": title, "done": bool(item.get("done", False))})
    return normalized


def resolve_workspace_path(context: ToolContext, raw_path: str) -> Path:
    return context.resolve_workspace_path(raw_path)
