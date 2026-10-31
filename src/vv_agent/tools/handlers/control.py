from __future__ import annotations

from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, to_json, trim_portable_whitespace
from vv_agent.types import ToolDirective, ToolExecutionResult, ToolResultStatus


def ask_user(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context
    question = trim_portable_whitespace(str(arguments.get("question", "Need user input"))) or "Need user input"
    selection_type = trim_portable_whitespace(str(arguments.get("selection_type", "single")))
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

    from vv_agent.canonical_json import canonical_json_bytes

    if (
        len(question.encode("utf-8")) > 16_384
        or len(options or []) > 64
        or any(len(v.encode("utf-8")) > 1024 for v in options or [])
        or len(canonical_json_bytes(payload)) > 65_536
    ):
        return builtin_error("Interaction form exceeds the UTF-8 size or option limit.", "interaction_form_too_large")

    return ToolExecutionResult(
        tool_call_id="",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        directive=ToolDirective.WAIT_USER,
        metadata=payload,
    )
