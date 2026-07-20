from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vv_agent.background_task import BackgroundAgentTask
from vv_agent.handoffs import Handoff, handoff
from vv_agent.model_settings import ModelSettings
from vv_agent.output_validation import OutputRepair, OutputValidator
from vv_agent.run_config import _validate_bounded_int
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.outputs import ToolOutputText
from vv_agent.types import NoToolPolicy, SubAgentConfig, _trim_portable_whitespace, _validate_no_tool_policy

if TYPE_CHECKING:
    from vv_agent.run_config import ToolPolicy
    from vv_agent.runtime.hooks import RuntimeHook

ToolUseBehavior = Literal["run_llm_again", "stop_on_first_tool", "stop_at_tool_names"]


@dataclass(slots=True)
class RunContext[TContext]:
    context: TContext | None = None
    run_id: str = ""
    agent_name: str = ""
    model: str | Any | None = None
    workspace: str | Path | Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def app_state(self) -> TContext | None:
        return self.context


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
    hooks: list[RuntimeHook] = field(default_factory=list)
    max_cycles: int | None = None
    tool_policy: ToolPolicy | None = None
    tool_use_behavior: ToolUseBehavior = "run_llm_again"
    stop_at_tool_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    no_tool_policy: NoToolPolicy | None = None
    output_validation_enabled: bool = False
    output_validator: OutputValidator | None = None
    output_repair: OutputRepair | None = None
    output_validation_max_repairs: int = 1
    output_repair_model: Any | None = None
    output_repair_model_settings: Any | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("agent name cannot be empty")
        if isinstance(self.instructions, str) and not self.instructions.strip():
            raise ValueError("agent instructions cannot be empty")
        _validate_bounded_int(self.max_cycles, "max_cycles", minimum=1)
        _validate_no_tool_policy(self.no_tool_policy, "Agent.no_tool_policy")
        if not isinstance(self.output_validation_enabled, bool):
            raise TypeError("output_validation_enabled must be a boolean")
        if isinstance(self.output_validation_max_repairs, bool) or self.output_validation_max_repairs not in {0, 1}:
            raise ValueError("output_validation_max_repairs must be 0 or 1")
        if self.output_validation_enabled and self.output_validator is None:
            raise ValueError("enabled output validation requires an output_validator")
        if self.output_repair is not None and self.output_validator is None:
            raise ValueError("output_repair requires an output_validator")
        if self.output_validator is not None and not callable(self.output_validator):
            raise TypeError("output_validator must be callable")
        if self.output_repair is not None and not callable(self.output_repair):
            raise TypeError("output_repair must be callable")

        normalized_sub_agents: dict[str, SubAgentConfig] = {}
        for sub_agent_id, config in self.sub_agents.items():
            if not isinstance(sub_agent_id, str):
                raise TypeError("sub-agent id must be a string")
            normalized_id = _trim_portable_whitespace(sub_agent_id)
            if not normalized_id:
                raise ValueError("sub-agent id cannot be empty")
            if normalized_id in normalized_sub_agents:
                raise ValueError(f"duplicate sub-agent id after normalization: {normalized_id}")
            normalized_sub_agents[normalized_id] = deepcopy(config)
        self.sub_agents = normalized_sub_agents

    def resolve_instructions(self, run_context: RunContext[TContext] | None = None) -> str:
        if isinstance(self.instructions, str):
            return self.instructions
        return str(self.instructions(run_context or RunContext(), self))

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> FunctionTool:
        tool_name = name or self.name
        tool_description = description or f"Run the {self.name} agent."

        def invoke(_context: Any, arguments: dict[str, Any]) -> ToolOutputText:
            from vv_agent.runner import Runner

            prompt = Runner._child_agent_prompt(arguments=arguments, context=_context)
            result = Runner.run_sync(self, prompt)
            return ToolOutputText(text=result.final_output or "")

        return FunctionTool(
            name=tool_name,
            description=tool_description,
            params_json_schema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task for the delegated agent.",
                    },
                    "output_requirements": {
                        "type": "string",
                        "description": "Optional output requirements for the delegated agent.",
                    },
                    "include_main_summary": {
                        "type": "boolean",
                        "description": "Whether to include parent task summary.",
                    },
                },
                "required": ["task_description"],
                "additionalProperties": False,
            },
            on_invoke=invoke,
            metadata={"agent": self, "mode": "agent_as_tool"},
        )

    def as_background_task(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> BackgroundAgentTask:
        return BackgroundAgentTask(self, name=name, description=description)


__all__ = ["Agent", "RunContext", "ToolUseBehavior", "handoff"]
