from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from v_agent.tools.base import ToolContext, ToolSpec
from v_agent.tools.registry import ToolRegistry
from v_agent.types import ToolDirective, ToolExecutionResult


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _get_todo_list(shared_state: dict[str, Any]) -> list[dict[str, Any]]:
    todo = shared_state.get("todo_list")
    if isinstance(todo, list):
        return todo
    shared_state["todo_list"] = []
    return shared_state["todo_list"]


def _normalize_todo_items(raw_items: Any) -> list[dict[str, Any]]:
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


def _task_finish(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    todo = _get_todo_list(context.shared_state)
    require_all_done = bool(arguments.get("require_all_todos_completed", True))
    message = str(arguments.get("message", "Task completed"))

    incomplete = [
        {"index": index, "title": item.get("title", "")}
        for index, item in enumerate(todo)
        if not bool(item.get("done", False))
    ]

    if require_all_done and incomplete:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=_json(
                {
                    "ok": False,
                    "error": "todo_incomplete",
                    "message": "Cannot finish task while todo items are incomplete",
                    "incomplete": incomplete,
                }
            ),
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json({"ok": True, "message": message}),
        directive=ToolDirective.FINISH,
        metadata={"final_message": message},
    )


def _ask_user(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context
    question = str(arguments.get("question", "Need user input"))
    options = arguments.get("options", [])
    payload = {"question": question, "options": options}
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json(payload),
        directive=ToolDirective.WAIT_USER,
        metadata=payload,
    )


def _todo_write(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    todo = _get_todo_list(context.shared_state)
    action = str(arguments.get("action", "replace"))

    if action == "replace":
        context.shared_state["todo_list"] = _normalize_todo_items(arguments.get("items"))
    elif action == "append":
        todo.extend(_normalize_todo_items(arguments.get("items")))
    elif action == "set_done":
        index = int(arguments.get("index", -1))
        done = bool(arguments.get("done", True))
        if index < 0 or index >= len(todo):
            return ToolExecutionResult(
                tool_call_id="",
                status="error",
                content=_json({"ok": False, "error": f"todo index out of range: {index}"}),
            )
        todo[index]["done"] = done
    else:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=_json({"ok": False, "error": f"unsupported action: {action}"}),
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json({"ok": True, "todo_list": _get_todo_list(context.shared_state)}),
    )


def _todo_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del arguments
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json({"todo_list": _get_todo_list(context.shared_state)}),
    )


def _list_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    root = context.resolve_workspace_path(str(arguments.get("path", ".")))
    glob_pattern = str(arguments.get("glob", "**/*"))
    include_hidden = bool(arguments.get("include_hidden", False))

    files: list[str] = []
    for candidate in root.glob(glob_pattern):
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(context.workspace).as_posix()
        if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
            continue
        files.append(rel)

    files.sort()
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json({"files": files, "count": len(files)}),
    )


def _read_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = context.resolve_workspace_path(str(arguments["path"]))
    if not target.exists() or not target.is_file():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=_json({"error": f"file not found: {target.relative_to(context.workspace).as_posix()}"}),
        )

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start_line = int(arguments.get("start_line", 1))
    end_line = arguments.get("end_line")
    end_line_int = int(end_line) if end_line is not None else len(lines)

    start_idx = max(start_line - 1, 0)
    end_idx = max(end_line_int, start_idx)
    selected = lines[start_idx:end_idx]

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json(
            {
                "path": target.relative_to(context.workspace).as_posix(),
                "start_line": start_idx + 1,
                "end_line": start_idx + len(selected),
                "content": "\n".join(selected),
            }
        ),
    )


def _write_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = context.resolve_workspace_path(str(arguments["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)

    content = str(arguments.get("content", ""))
    append = bool(arguments.get("append", False))

    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as file_obj:
        file_obj.write(content)

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json(
            {
                "ok": True,
                "path": target.relative_to(context.workspace).as_posix(),
                "written_chars": len(content),
            }
        ),
    )


def _workspace_grep(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    root = context.resolve_workspace_path(str(arguments.get("path", ".")))
    glob_pattern = str(arguments.get("glob", "**/*"))
    pattern = str(arguments["pattern"])
    case_sensitive = bool(arguments.get("case_sensitive", False))
    max_results = int(arguments.get("max_results", 50))

    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)

    matches: list[dict[str, Any]] = []
    for candidate in root.glob(glob_pattern):
        if not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    {
                        "path": candidate.relative_to(context.workspace).as_posix(),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(matches) >= max_results:
                    return ToolExecutionResult(
                        tool_call_id="",
                        status="success",
                        content=_json({"matches": matches, "truncated": True}),
                    )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=_json({"matches": matches, "truncated": False}),
    )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(
        [
            ToolSpec(
                name="task_finish",
                description="Mark task as finished. Enforces todo completion by default.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "require_all_todos_completed": {"type": "boolean", "default": True},
                    },
                    "required": ["message"],
                },
                handler=_task_finish,
            ),
            ToolSpec(
                name="ask_user",
                description="Pause execution and request user input.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
                handler=_ask_user,
            ),
            ToolSpec(
                name="todo_write",
                description="Manage runtime todo list.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["replace", "append", "set_done"]},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "done": {"type": "boolean", "default": False},
                                },
                                "required": ["title"],
                            },
                        },
                        "index": {"type": "integer"},
                        "done": {"type": "boolean"},
                    },
                    "required": ["action"],
                },
                handler=_todo_write,
            ),
            ToolSpec(
                name="todo_read",
                description="Read runtime todo list.",
                input_schema={"type": "object", "properties": {}},
                handler=_todo_read,
            ),
            ToolSpec(
                name="list_files",
                description="List workspace files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "glob": {"type": "string", "default": "**/*"},
                        "include_hidden": {"type": "boolean", "default": False},
                    },
                },
                handler=_list_files,
            ),
            ToolSpec(
                name="read_file",
                description="Read file content with optional line range.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1, "default": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                },
                handler=_read_file,
            ),
            ToolSpec(
                name="write_file",
                description="Write file content into workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "content"],
                },
                handler=_write_file,
            ),
            ToolSpec(
                name="workspace_grep",
                description="Search text across workspace files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "glob": {"type": "string", "default": "**/*"},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 50, "minimum": 1},
                    },
                    "required": ["pattern"],
                },
                handler=_workspace_grep,
            ),
        ]
    )
    return registry
