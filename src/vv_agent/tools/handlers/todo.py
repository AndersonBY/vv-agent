from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import get_todo_list, is_string_keyed_dict, to_json
from vv_agent.types import ToolExecutionResult

_ALLOWED_STATUS = {"pending", "in_progress", "completed"}
_ALLOWED_PRIORITY = {"low", "medium", "high"}


def _error(message: str, *, error_code: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        error_code=error_code,
        content=to_json({"error": message, "error_code": error_code}),
    )


def todo_write(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    todos = arguments.get("todos")
    if not isinstance(todos, list):
        return _error("`todos` must be an array", error_code="invalid_todos_payload")

    existing_todos = get_todo_list(context.shared_state)
    existing_map: dict[str, dict[str, Any]] = {}
    for item in existing_todos:
        if not is_string_keyed_dict(item):
            continue
        raw_item_id = item.get("id")
        item_id = str(raw_item_id).strip() if raw_item_id is not None else ""
        if item_id:
            existing_map[item_id] = item

    now = datetime.now(tz=UTC).isoformat()
    new_todo_list: list[dict[str, Any]] = []

    for index, raw_todo_item in enumerate(todos):
        if not is_string_keyed_dict(raw_todo_item):
            return _error(f"TODO item at index {index} must be an object", error_code="invalid_todo_item")

        title = str(raw_todo_item.get("title", "")).strip()
        if not title:
            return _error(f"TODO item at index {index} is missing `title`", error_code="todo_title_required")

        status = str(raw_todo_item.get("status", "pending")).lower()
        if status not in _ALLOWED_STATUS:
            return _error(
                f"TODO item {title} has invalid status {status}",
                error_code="invalid_todo_status",
            )

        priority = str(raw_todo_item.get("priority", "medium")).lower()
        if priority not in _ALLOWED_PRIORITY:
            return _error(
                f"TODO item {title} has invalid priority {priority}",
                error_code="invalid_todo_priority",
            )

        raw_id = raw_todo_item.get("id")
        item_id = str(raw_id).strip() if raw_id is not None else ""
        if not item_id:
            item_id = uuid.uuid4().hex[:8]

        if item_id in existing_map:
            previous = dict(existing_map[item_id])
            created_at = str(previous.get("created_at", now))
        else:
            created_at = now

        new_todo_list.append(
            {
                "id": item_id,
                "title": title,
                "status": status,
                "priority": priority,
                "created_at": created_at,
                "updated_at": now,
            }
        )

    in_progress_items = [item for item in new_todo_list if item.get("status") == "in_progress"]
    if len(in_progress_items) > 1:
        return _error(
            "Only one TODO item can be in_progress at a time",
            error_code="multiple_in_progress_todos",
        )

    context.shared_state["todo_list"] = new_todo_list

    result = {
        "action": "write",
        "todos": new_todo_list,
        "count": len(new_todo_list),
        "message": f"TODO list updated successfully with {len(new_todo_list)} items",
    }

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(result),
    )


def todo_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del arguments
    todos = get_todo_list(context.shared_state)
    result = {"action": "read", "todos": todos, "count": len(todos)}
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(result),
    )
