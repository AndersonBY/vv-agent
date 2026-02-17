from __future__ import annotations

from dataclasses import dataclass, field

from v_agent.tools.base import ToolContext, ToolSpec
from v_agent.types import ToolCall, ToolExecutionResult


class ToolNotFoundError(KeyError):
    pass


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_openai_schemas(self) -> list[dict[str, object]]:
        schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            schemas.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                }
            )
        return schemas

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult:
        tool = self.get(call.name)
        return tool.handler(context, call.arguments)
