from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from vv_agent.handoffs import Handoff, handoff
from vv_agent.model_settings import ModelSettings
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.outputs import ToolOutputText

ToolUseBehavior = Literal["run_llm_again", "stop_on_first_tool", "stop_at_tools", "tools_to_final_output"]


@dataclass(slots=True)
class RunContext[TContext]:
    context: TContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Agent[TContext]:
    name: str
    instructions: str | Callable[[RunContext[TContext], Agent[TContext]], str]
    model: str | Any | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[Any] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    input_guardrails: list[Any] = field(default_factory=list)
    output_guardrails: list[Any] = field(default_factory=list)
    output_type: type[Any] | Any | None = None
    hooks: Any | None = None
    memory_policy: Any | None = None
    tool_use_behavior: ToolUseBehavior = "run_llm_again"
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_instructions(self, run_context: RunContext[TContext] | None = None) -> str:
        if isinstance(self.instructions, str):
            return self.instructions
        return str(self.instructions(run_context or RunContext(), self))

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> FunctionTool:
        tool_name = name or self.name
        tool_description = description or f"Run the {self.name} agent."

        def invoke(_context: Any, arguments: dict[str, Any]) -> ToolOutputText:
            from vv_agent.runner import Runner

            prompt = str(arguments.get("input", ""))
            result = Runner.run_sync(self, prompt)
            return ToolOutputText(text=result.final_output or "")

        return FunctionTool(
            name=tool_name,
            description=tool_description,
            params_json_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
            on_invoke=invoke,
            metadata={"agent": self, "mode": "agent_as_tool"},
        )

    def as_background_task(self, *, name: str | None = None, description: str | None = None) -> FunctionTool:
        tool = self.as_tool(
            name=name or f"{self.name}_background",
            description=description or f"Start {self.name} in background.",
        )
        tool.metadata["mode"] = "background_task"
        return tool


__all__ = ["Agent", "RunContext", "ToolUseBehavior", "handoff"]
