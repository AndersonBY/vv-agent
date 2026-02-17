from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    LIST_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_INCOMPLETE_ERROR_CODE,
    TODO_READ_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    get_default_tool_schemas,
)
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
            error_code=TODO_INCOMPLETE_ERROR_CODE,
            content=_json(
                {
                    "ok": False,
                    "error": TODO_INCOMPLETE_ERROR_CODE,
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
    selection_type = str(arguments.get("selection_type", "single"))
    allow_custom_options = bool(arguments.get("allow_custom_options", False))

    options_raw = arguments.get("options")
    options: list[str] | None = None
    if isinstance(options_raw, list):
        normalized: list[str] = []
        seen: set[str] = set()
        for option in options_raw:
            option_text = str(option).strip()
            if option_text and option_text not in seen:
                seen.add(option_text)
                normalized.append(option_text)
        options = normalized or None

    if selection_type not in {"single", "multi"}:
        selection_type = "single"

    payload: dict[str, Any] = {
        "question": question,
        "selection_type": selection_type,
        "allow_custom_options": allow_custom_options,
    }
    if options is not None:
        payload["options"] = options

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
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=_json({"error": f"invalid regex pattern: {exc}"}),
        )

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
    registry.register_schemas(get_default_tool_schemas())
    registry.register_many(
        [
            ToolSpec(name=TASK_FINISH_TOOL_NAME, handler=_task_finish),
            ToolSpec(name=ASK_USER_TOOL_NAME, handler=_ask_user),
            ToolSpec(name=TODO_WRITE_TOOL_NAME, handler=_todo_write),
            ToolSpec(name=TODO_READ_TOOL_NAME, handler=_todo_read),
            ToolSpec(name=LIST_FILES_TOOL_NAME, handler=_list_files),
            ToolSpec(name=READ_FILE_TOOL_NAME, handler=_read_file),
            ToolSpec(name=WRITE_FILE_TOOL_NAME, handler=_write_file),
            ToolSpec(name=WORKSPACE_GREP_TOOL_NAME, handler=_workspace_grep),
        ]
    )
    return registry
