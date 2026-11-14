from __future__ import annotations

from vv_agent.kernel._tool_admission import _error_result, _parse_arguments
from vv_agent.tools.base import ToolContext
from vv_agent.tools.outcomes import HostToolOutcome
from vv_agent.tools.registry import ToolNotFoundError, ToolRegistry
from vv_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus


def _needs_tool_call_id(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return stripped == "" or stripped == "pending"


def dispatch_tool_call(
    *,
    registry: ToolRegistry,
    context: ToolContext,
    call: ToolCall,
) -> ToolExecutionResult | HostToolOutcome:
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

    if isinstance(result, HostToolOutcome):
        if result.kind == "completed" and isinstance(result.result, ToolExecutionResult):
            result = result.result
        else:
            return result

    if _needs_tool_call_id(result.tool_call_id):
        result.tool_call_id = call.id

    if result.directive == ToolDirective.WAIT_USER and result.status_code == ToolResultStatus.SUCCESS:
        result.status_code = ToolResultStatus.WAIT_RESPONSE

    return result
