from support.model_calls import model_call_context
from support.model_providers import FactoryModelProvider, FixedModelProvider, ModelMapProvider
from vv_agent.deferred import ToolCallOutcome
from vv_agent.types import ToolExecutionResult


def require_tool_result(value: ToolExecutionResult | ToolCallOutcome) -> ToolExecutionResult:
    """Narrow a real tool result while making deferred outcomes explicit in tests."""

    if isinstance(value, ToolExecutionResult):
        return value
    if value.is_deferred:
        raise AssertionError("expected a completed ToolExecutionResult, got a deferred outcome")
    if value.kind != "completed" or not isinstance(value.result, ToolExecutionResult):
        raise AssertionError("completed ToolCallOutcome did not contain a ToolExecutionResult")
    return value.result


__all__ = [
    "FactoryModelProvider",
    "FixedModelProvider",
    "ModelMapProvider",
    "model_call_context",
    "require_tool_result",
]
