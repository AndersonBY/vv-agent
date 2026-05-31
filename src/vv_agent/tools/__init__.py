from vv_agent.tools.base import ToolContext, ToolHandler, ToolSpec
from vv_agent.tools.dispatcher import dispatch_tool_call
from vv_agent.tools.function import FunctionTool, Tool, function_tool
from vv_agent.tools.outputs import (
    ToolOutput,
    ToolOutputError,
    ToolOutputFile,
    ToolOutputImage,
    ToolOutputJson,
    ToolOutputText,
)
from vv_agent.tools.registry import ToolNotFoundError, ToolRegistry


def build_default_registry() -> ToolRegistry:
    from vv_agent.tools.builtins import build_default_registry as _build_default_registry

    return _build_default_registry()

__all__ = [
    "FunctionTool",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolNotFoundError",
    "ToolOutput",
    "ToolOutputError",
    "ToolOutputFile",
    "ToolOutputImage",
    "ToolOutputJson",
    "ToolOutputText",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
    "dispatch_tool_call",
    "function_tool",
]
