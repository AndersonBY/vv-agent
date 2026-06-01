from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.tools.base import ToolContext, ToolSpec
from vv_agent.types import ToolCall, ToolExecutionResult

if TYPE_CHECKING:
    from vv_agent.tools.function import ApprovalPredicate, FunctionTool, ToolErrorFormatter


class ToolExposure(StrEnum):
    DIRECT = "direct"
    DEFERRED = "deferred"
    DIRECT_MODEL_ONLY = "direct_model_only"
    HIDDEN = "hidden"


class ToolExecutor(Protocol):
    name: str
    description: str
    params_json_schema: dict[str, Any]
    strict_json_schema: bool
    exposure: ToolExposure
    needs_approval: bool | ApprovalPredicate
    timeout_seconds: float | None
    failure_error_function: ToolErrorFormatter | None
    metadata: dict[str, Any]

    def spec(self, context: ToolContext | None = None) -> ToolSpec: ...

    def openai_schema(self, context: ToolContext | None = None) -> dict[str, Any]: ...

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult: ...

    def requires_approval(self, context: ToolContext, arguments: dict[str, Any]) -> bool: ...


@dataclass(slots=True)
class FunctionToolExecutor:
    tool: FunctionTool
    exposure: ToolExposure = ToolExposure.DIRECT

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description

    @property
    def params_json_schema(self) -> dict[str, Any]:
        return dict(self.tool.params_json_schema)

    @property
    def strict_json_schema(self) -> bool:
        return self.tool.strict_json_schema

    @property
    def needs_approval(self) -> bool | ApprovalPredicate:
        return self.tool.needs_approval

    @property
    def timeout_seconds(self) -> float | None:
        return self.tool.timeout_seconds

    @property
    def failure_error_function(self) -> ToolErrorFormatter | None:
        return self.tool.failure_error_function

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.tool.metadata)

    def spec(self, context: ToolContext | None = None) -> ToolSpec:
        del context

        def handler(tool_context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
            call = ToolCall(id=tool_context.tool_call_id, name=self.name, arguments=dict(arguments))
            return self.execute(call, tool_context)

        return ToolSpec(name=self.name, handler=handler)

    def openai_schema(self, context: ToolContext | None = None) -> dict[str, Any]:
        del context
        return self.tool.to_openai_schema()

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult:
        output = self.tool.on_invoke(context, call.arguments)
        return self.tool.to_tool_execution_result(output, tool_call_id=call.id)

    def requires_approval(self, context: ToolContext, arguments: dict[str, Any]) -> bool:
        if callable(self.tool.needs_approval):
            approval_predicate = self.tool.needs_approval
            return bool(approval_predicate(context, arguments))
        return bool(self.tool.needs_approval)


@dataclass(slots=True)
class RegistryToolExecutor:
    name: str
    handler: Callable[[ToolContext, dict[str, Any]], ToolExecutionResult]
    schema: dict[str, Any] | None = None
    description: str = ""
    exposure: ToolExposure = ToolExposure.DIRECT
    needs_approval: bool | ApprovalPredicate = False
    strict_json_schema: bool = True
    timeout_seconds: float | None = None
    failure_error_function: ToolErrorFormatter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def params_json_schema(self) -> dict[str, Any]:
        if not self.schema:
            return {"type": "object", "properties": {}, "required": []}
        function_schema = self.schema.get("function")
        if isinstance(function_schema, dict):
            parameters = function_schema.get("parameters")
            if isinstance(parameters, dict):
                return dict(parameters)
        return {"type": "object", "properties": {}, "required": []}

    def spec(self, context: ToolContext | None = None) -> ToolSpec:
        del context
        return ToolSpec(name=self.name, handler=self.handler)

    def openai_schema(self, context: ToolContext | None = None) -> dict[str, Any]:
        del context
        if self.schema is not None:
            return dict(self.schema)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
            },
        }

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult:
        return self.handler(context, call.arguments)

    def requires_approval(self, context: ToolContext, arguments: dict[str, Any]) -> bool:
        if callable(self.needs_approval):
            return bool(self.needs_approval(context, arguments))
        del context, arguments
        return bool(self.needs_approval)
