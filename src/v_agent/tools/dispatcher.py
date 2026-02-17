from __future__ import annotations

import json
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.registry import ToolNotFoundError, ToolRegistry
from v_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus


def _error_result(tool_call_id: str, message: str, *, error_code: str | None = None) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=json.dumps({"ok": False, "error": message, "error_code": error_code}, ensure_ascii=False),
    )


def _parse_arguments(
    tool_call_id: str,
    raw_arguments: Any,
) -> tuple[dict[str, Any], ToolExecutionResult | None]:
    if raw_arguments is None:
        return {}, None

    if isinstance(raw_arguments, dict):
        return raw_arguments, None

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}, None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return {}, _error_result(tool_call_id, f"Invalid tool arguments JSON: {exc}", error_code="invalid_arguments_json")

        if isinstance(parsed, dict):
            return parsed, None
        return {}, _error_result(
            tool_call_id,
            "Tool arguments must decode to an object",
            error_code="invalid_arguments_payload",
        )

    return {}, _error_result(
        tool_call_id,
        f"Unsupported tool argument type: {type(raw_arguments).__name__}",
        error_code="invalid_arguments_type",
    )


def dispatch_tool_call(
    *,
    registry: ToolRegistry,
    context: ToolContext,
    call: ToolCall,
) -> ToolExecutionResult:
    arguments, parse_error = _parse_arguments(call.id, call.arguments)
    if parse_error is not None:
        return parse_error

    normalized_call = ToolCall(id=call.id, name=call.name, arguments=arguments)

    try:
        result = registry.execute(normalized_call, context)
    except ToolNotFoundError:
        return _error_result(call.id, f"Unknown tool: {call.name}", error_code="tool_not_found")
    except Exception as exc:
        return _error_result(call.id, f"Tool execution failed ({call.name}): {exc}", error_code="tool_execution_failed")

    if not result.tool_call_id or result.tool_call_id == "pending":
        result.tool_call_id = call.id

    if result.directive == ToolDirective.WAIT_USER and result.status_code == ToolResultStatus.SUCCESS:
        result.status_code = ToolResultStatus.WAIT_RESPONSE

    return result
