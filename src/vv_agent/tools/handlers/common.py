from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vv_agent.tools.base import ToolContext


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


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
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        normalized.append({"title": title, "done": bool(item.get("done", False))})
    return normalized


def resolve_workspace_path(context: ToolContext, raw_path: str) -> Path:
    return context.resolve_workspace_path(raw_path)
