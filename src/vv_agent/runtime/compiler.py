from __future__ import annotations

import uuid
from copy import deepcopy

from vv_agent.agent import Agent, RunContext
from vv_agent.config import ResolvedModelConfig
from vv_agent.context_providers import (
    ContextFragment,
    ContextRequest,
    assemble_context_fragments,
    collect_context_fragments,
)
from vv_agent.prompt.templates import render_sub_agents
from vv_agent.run_config import RunConfig, _validate_bounded_int
from vv_agent.tools.executor import ToolExposure
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
        run_id: str = "",
    ) -> RuntimeTask:
        model = run_config.model or agent.model or resolved.selected_model
        metadata = dict(agent.metadata)
        metadata.update(run_config.metadata)
        metadata.pop("_vv_agent_allowed_tools", None)
        metadata.pop("_vv_agent_disallowed_tools", None)
        metadata.setdefault("trace_id", trace_id)
        if resolved.context_length is not None:
            metadata.setdefault("model_context_window", resolved.context_length)
        if resolved.max_output_tokens is not None:
            metadata.setdefault("reserved_output_tokens", resolved.max_output_tokens)
        if run_config.tool_policy is not None:
            if run_config.tool_policy.allowed_tools is not None:
                metadata["_vv_agent_allowed_tools"] = list(run_config.tool_policy.allowed_tools)
            if run_config.tool_policy.disallowed_tools:
                metadata["_vv_agent_disallowed_tools"] = list(run_config.tool_policy.disallowed_tools)

        no_tool_policy = run_config.no_tool_policy or agent.no_tool_policy or "continue"
        metadata["_vv_agent_tool_use_behavior"] = agent.tool_use_behavior
        if agent.stop_at_tool_names:
            metadata["_vv_agent_stop_at_tool_names"] = list(agent.stop_at_tool_names)

        handoff_tool_names = [transfer.tool_name for transfer in agent.handoffs if transfer.tool_name]
        resolved_instructions = agent.resolve_instructions(
            RunContext(
                context=run_config.context,
                run_id=run_id,
                agent_name=agent.name,
                model=str(resolved.model_id or model),
                workspace=run_config.workspace,
                metadata=metadata,
            )
        )
        request = ContextRequest(
            agent_name=agent.name,
            input=input,
            model=str(resolved.model_id or model),
            trace_id=trace_id,
            session=run_config.session,
            workspace=run_config.workspace,
            context=run_config.context,
            metadata=metadata,
            max_prompt_chars=run_config.max_context_chars,
        )
        fragments = [
            ContextFragment(
                id="agent_instructions",
                text=resolved_instructions,
                stable=True,
                priority=0,
                source="agent.instructions",
            )
        ]
        if agent.sub_agents:
            fragments.append(
                ContextFragment(
                    id="configured_sub_agents",
                    text=render_sub_agents(
                        "en-US",
                        {name: config.description for name, config in agent.sub_agents.items()},
                    ),
                    stable=True,
                    priority=10,
                    source="agent.sub_agents",
                )
            )
        if run_config.context_providers:
            fragments.extend(collect_context_fragments(request, run_config.context_providers))
        context_bundle = assemble_context_fragments(request, fragments)
        system_prompt = context_bundle.prompt
        if context_bundle.sections:
            metadata["system_prompt_sections"] = context_bundle.metadata_sections()
        if context_bundle.sources:
            metadata["system_prompt_sources"] = context_bundle.sources
        if context_bundle.omitted_section_ids:
            metadata["system_prompt_omitted_sections"] = list(context_bundle.omitted_section_ids)
        metadata["system_prompt_stable_hash"] = context_bundle.stable_hash

        max_cycles = _validate_bounded_int(
            run_config.max_cycles if run_config.max_cycles is not None else 10,
            "max_cycles",
            minimum=1,
        )
        assert max_cycles is not None
        return RuntimeTask(
            task_id=f"{agent.name}_{uuid.uuid4().hex[:8]}",
            model=str(resolved.model_id or model),
            system_prompt=system_prompt,
            user_prompt=input,
            max_cycles=max_cycles,
            no_tool_policy=no_tool_policy,
            has_sub_agents=False,
            sub_agents=deepcopy(agent.sub_agents),
            native_multimodal=resolved.native_multimodal,
            extra_tool_names=[
                *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool) and tool.exposure != ToolExposure.HIDDEN],
                *handoff_tool_names,
            ],
            model_settings=run_config.model_settings,
            initial_messages=list(run_config.initial_messages or []),
            initial_shared_state=dict(run_config.shared_state or {}),
            metadata=metadata,
        )
