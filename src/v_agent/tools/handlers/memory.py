from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult


def compress_memory(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    core_information = str(arguments.get("core_information", "")).strip()
    if not core_information:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`core_information` is required"}),
            error_code="core_information_required",
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
        status="success",
        content=to_json({"ok": True, "saved_notes": len(notes)}),
        metadata={"saved_notes": len(notes)},
    )
