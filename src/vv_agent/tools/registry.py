from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from vv_agent.tools.base import ToolContext, ToolHandler, ToolSpec
from vv_agent.tools.executor import RegistryToolExecutor, ToolExecutor, ToolExposure
from vv_agent.types import ToolCall, ToolExecutionResult


class ToolNotFoundError(KeyError):
    pass


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    _schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    _executors: dict[str, ToolExecutor] = field(default_factory=dict)
    _planner_extra_tool_names: set[str] = field(default_factory=set)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        if spec.name not in self._executors:
            schema = self._schemas.get(spec.name)
            self._executors[spec.name] = RegistryToolExecutor(
                name=spec.name,
                handler=spec.handler,
                schema=deepcopy(schema) if schema else None,
            )

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def register_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        self._schemas[tool_name] = deepcopy(schema)
        executor = self._executors.get(tool_name)
        if isinstance(executor, RegistryToolExecutor):
            executor.schema = deepcopy(schema)
            executor.sync_description_from_schema()

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

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_planner_extra_tool_names(self) -> list[str]:
        return list(self._planner_extra_tool_names)

    def has_executor(self, name: str) -> bool:
        return name in self._executors

    def get_executor(self, name: str) -> ToolExecutor:
        executor = self._executors.get(name)
        if executor is None:
            raise ToolNotFoundError(name)
        return executor

    def mark_policy_managed_by_handler(self, name: str) -> None:
        executor = self.get_executor(name)
        if isinstance(executor, RegistryToolExecutor):
            executor.metadata["policy_managed_by_handler"] = True

    def register_executor(
        self,
        executor: ToolExecutor,
        *,
        expose_to_model: bool = True,
        planner_extra: bool = True,
    ) -> None:
        if executor.name in self._executors or executor.name in self._tools:
            raise ValueError(f"Tool already registered: {executor.name}")
        self._executors[executor.name] = executor
        is_model_visible = executor.exposure != ToolExposure.HIDDEN
        if expose_to_model and is_model_visible:
            self.register_schema(executor.name, executor.openai_schema(None))
        self._tools[executor.name] = executor.spec(None)
        if planner_extra and is_model_visible:
            self._planner_extra_tool_names.add(executor.name)

    def has_schema(self, name: str) -> bool:
        return name in self._schemas

    def get_schema(self, name: str) -> dict[str, Any]:
        schema = self._schemas.get(name)
        if schema is None:
            raise KeyError(f"Schema not registered: {name}")
        return deepcopy(schema)

    def list_openai_schemas(self, *, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        ordered_names = tool_names if tool_names is not None else list(self._tools.keys())
        return [self.get_schema(name) for name in ordered_names if self._is_model_visible(name)]

    def _is_model_visible(self, name: str) -> bool:
        executor = self._executors.get(name)
        return executor is None or executor.exposure != ToolExposure.HIDDEN

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom tool in one step (schema + handler)."""
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }
        self.register_schema(name, schema)
        self.register(ToolSpec(name=name, handler=handler))
        self._planner_extra_tool_names.add(name)

    def execute(self, call: ToolCall, context: ToolContext) -> ToolExecutionResult:
        tool = self.get(call.name)
        return tool.handler(context, call.arguments)
