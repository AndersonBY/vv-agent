from __future__ import annotations

from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus


def compress_memory(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    core_information = str(arguments.get("core_information", "")).strip()
    if not core_information:
        return builtin_error(
            "`core_information` is required",
            "core_information_required",
        )

    notes = context.shared_state.get("memory_notes")
    if not isinstance(notes, list):
        notes = []
        context.shared_state["memory_notes"] = notes
    notes.append(
        {
            "cycle_index": context.cycle_index,
            "core_information": core_information,
        }
    )

    return ToolExecutionResult(
        tool_call_id="",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json({"ok": True, "saved_notes": len(notes)}),
        metadata={"saved_notes": len(notes)},
    )
