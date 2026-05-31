from __future__ import annotations

import uuid

from vv_agent.agent import Agent, RunContext
from vv_agent.config import ResolvedModelConfig
from vv_agent.run_config import RunConfig
from vv_agent.tools.function import FunctionTool
from vv_agent.types import AgentTask

RuntimeTask = AgentTask


class AgentCompiler:
    def compile(
        self,
        *,
        agent: Agent,
        input: str,
        run_config: RunConfig,
        resolved: ResolvedModelConfig,
        trace_id: str,
    ) -> RuntimeTask:
        model = run_config.model or agent.model or resolved.selected_model
        metadata = dict(agent.metadata)
        metadata.update(run_config.metadata)
        metadata.setdefault("trace_id", trace_id)
        if run_config.tool_policy is not None:
            if run_config.tool_policy.allowed_tools is not None:
                metadata["_vv_agent_allowed_tools"] = list(run_config.tool_policy.allowed_tools)
            if run_config.tool_policy.disallowed_tools:
                metadata["_vv_agent_disallowed_tools"] = list(run_config.tool_policy.disallowed_tools)

        no_tool_policy = "continue"
        if agent.tool_use_behavior == "stop_on_first_tool" or run_config.approval_provider is not None:
            no_tool_policy = "finish"

        handoff_tool_names = [transfer.tool_name for transfer in agent.handoffs if transfer.tool_name]
        return RuntimeTask(
            task_id=f"{agent.name}_{uuid.uuid4().hex[:8]}",
            model=str(resolved.model_id or model),
            system_prompt=agent.resolve_instructions(RunContext(context=run_config.context, metadata=metadata)),
            user_prompt=input,
            max_cycles=max(int(run_config.max_cycles), 1),
            no_tool_policy=no_tool_policy,
            extra_tool_names=[
                *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool)],
                *handoff_tool_names,
            ],
            metadata=metadata,
            runtime_metadata={"trace_id": trace_id},
        )
