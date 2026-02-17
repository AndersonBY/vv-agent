from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from v_agent.tools.base import ToolContext, ToolSpec
from v_agent.types import ToolCall, ToolExecutionResult


class ToolNotFoundError(KeyError):
    pass


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    _schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def register_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        self._schemas[tool_name] = deepcopy(schema)

    def register_schemas(self, schemas: dict[str, dict[str, Any]]) -> None:
        for tool_name, schema in schemas.items():
            self.register_schema(tool_name, schema)

    def get(self, name: str) -> ToolSpec:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def has_schema(self, name: str) -> bool:
        return name in self._schemas

    def get_schema(self, name: str) -> dict[str, Any]:
        schema = self._schemas.get(name)
        if schema is None:
            raise KeyError(f"Schema not registered: {name}")
        return deepcopy(schema)

    def list_openai_schemas(self, *, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        ordered_names = tool_names if tool_names is not None else list(self._tools.keys())
        return [self.get_schema(name) for name in ordered_names]

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult:
        tool = self.get(call.name)
        return tool.handler(context, call.arguments)
