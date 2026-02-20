from __future__ import annotations

from typing import Any

from vv_agent.constants import TODO_INCOMPLETE_ERROR_CODE
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import get_todo_list, to_json
from vv_agent.types import ToolDirective, ToolExecutionResult


def task_finish(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    todo = get_todo_list(context.shared_state)
    require_all_done = bool(arguments.get("require_all_todos_completed", True))
    message = str(arguments.get("message", "Task completed"))
    exposed_files = arguments.get("exposed_files")

    incomplete_todos = []
    for item in todo:
        status = str(item.get("status", "")).lower()
        done_flag = bool(item.get("done", False))
        if status in {"completed", "done", "finished"} or done_flag:
            continue
        incomplete_todos.append(str(item.get("title", "Untitled TODO")))

    if require_all_done and incomplete_todos:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            error_code=TODO_INCOMPLETE_ERROR_CODE,
            content=to_json(
                {
                    "ok": False,
                    "error_code": TODO_INCOMPLETE_ERROR_CODE,
                    "error": "Cannot finish task while todo items are incomplete",
                    "incomplete_todos": incomplete_todos,
                }
            ),
        )

    metadata = {"final_message": message}
    if isinstance(exposed_files, list):
        metadata["exposed_files"] = [str(path) for path in exposed_files if str(path).strip()]

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json({"ok": True, "message": message}),
        directive=ToolDirective.FINISH,
        metadata=metadata,
    )


def ask_user(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
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
        content=to_json(payload),
        directive=ToolDirective.WAIT_USER,
        metadata=payload,
    )
