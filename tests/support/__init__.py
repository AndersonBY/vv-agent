from support.model_calls import model_call_context
from support.model_providers import FactoryModelProvider, FixedModelProvider, ModelMapProvider
from vv_agent.tools.outcomes import HostToolOutcome
from vv_agent.types import ToolExecutionResult


def require_tool_result(value: ToolExecutionResult | HostToolOutcome) -> ToolExecutionResult:
    """Narrow a real tool result from host interaction outcomes."""

    if isinstance(value, ToolExecutionResult):
        return value
    if value.kind != "completed" or not isinstance(value.result, ToolExecutionResult):
        raise AssertionError("completed HostToolOutcome did not contain a ToolExecutionResult")
    return value.result


__all__ = [
    "FactoryModelProvider",
    "FixedModelProvider",
    "ModelMapProvider",
    "model_call_context",
    "require_tool_result",
]
