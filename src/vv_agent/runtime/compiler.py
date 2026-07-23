from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.checkpoint import CheckpointError
from vv_agent.config import ResolvedModelConfig, project_resolved_model_limits
from vv_agent.constants import WORKSPACE_TOOLS
from vv_agent.context_providers import (
    ContextFragment,
    ContextRequest,
    assemble_context_fragments,
    collect_context_fragments,
)
from vv_agent.prompt.templates import render_sub_agents
from vv_agent.run_config import RunConfig, ToolPolicy, _validate_bounded_int
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolSideEffect
from vv_agent.types import AgentTask, Message, NoToolPolicy

_TASK_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)


def _apply_tool_policy_metadata(
    metadata: dict[str, Any],
    policy: ToolPolicy | None,
) -> None:
    for key in _TASK_TOOL_POLICY_METADATA_KEYS:
        metadata.pop(key, None)
    if policy is None:
        return
    if policy.allowed_tools is not None:
        metadata["_vv_agent_allowed_tools"] = list(policy.allowed_tools)
    if policy.disallowed_tools:
        metadata["_vv_agent_disallowed_tools"] = list(policy.disallowed_tools)
    if policy.denied_side_effects:
        metadata["_vv_agent_denied_side_effects"] = [ToolSideEffect(item).value for item in policy.denied_side_effects]
    if policy.denied_capability_tags:
        metadata["_vv_agent_denied_capability_tags"] = list(policy.denied_capability_tags)
    if policy.deny_terminal_tools:
        metadata["_vv_agent_deny_terminal_tools"] = True
    if policy.denied_cost_dimensions:
        metadata["_vv_agent_denied_cost_dimensions"] = list(policy.denied_cost_dimensions)


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
    ) -> AgentTask:
        model = run_config.model or agent.model or resolved.selected_model
        metadata = dict(agent.metadata)
        metadata.update(run_config.metadata)
        metadata["session_memory_enabled"] = run_config.session_memory_enabled
        _apply_tool_policy_metadata(metadata, run_config.tool_policy)
        metadata.setdefault("trace_id", trace_id)
        project_resolved_model_limits(
            metadata,
            context_length=resolved.context_length,
            max_output_tokens=resolved.max_output_tokens,
        )
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
        return AgentTask(
            task_id=f"{agent.name}_{uuid.uuid4().hex[:8]}",
            model=str(resolved.model_id or model),
            system_prompt=system_prompt,
            user_prompt=input,
            max_cycles=max_cycles,
            no_tool_policy=no_tool_policy,
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

    def compile_frozen_checkpoint(
        self,
        *,
        agent: Agent,
        run_config: RunConfig,
        resolved: ResolvedModelConfig,
        checkpoint: Any,
        trace_id: str,
    ) -> AgentTask:
        definition = getattr(checkpoint, "run_definition", None)
        if not isinstance(definition, dict):
            raise CheckpointError(
                "checkpoint is missing its embedded run definition",
                code="checkpoint_definition_invalid",
            )
        controls = definition.get("runtime_controls")
        model = definition.get("model")
        agent_definition = definition.get("agent")
        if not isinstance(controls, dict) or not isinstance(model, dict) or not isinstance(agent_definition, dict):
            raise CheckpointError(
                "checkpoint run definition has invalid runtime fields",
                code="checkpoint_definition_invalid",
            )
        messages = getattr(checkpoint, "messages", None)
        system_metadata: dict[str, object] = {}
        if isinstance(messages, list) and messages and isinstance(messages[0], Message) and messages[0].role == "system":
            system_metadata = deepcopy(messages[0].metadata)
        self._validate_frozen_static_prompt(agent, definition, system_metadata)

        metadata = dict(system_metadata)
        run_metadata = definition.get("run_metadata")
        if isinstance(run_metadata, dict):
            metadata.update(deepcopy(run_metadata))
        metadata["trace_id"] = trace_id
        project_resolved_model_limits(
            metadata,
            context_length=resolved.context_length,
            max_output_tokens=resolved.max_output_tokens,
        )
        metadata["_vv_agent_tool_use_behavior"] = controls["tool_use_behavior"]
        metadata["session_memory_enabled"] = controls["session_memory_enabled"]
        if controls["stop_at_tool_names"]:
            metadata["_vv_agent_stop_at_tool_names"] = list(controls["stop_at_tool_names"])
        _apply_tool_policy_metadata(metadata, run_config.tool_policy)

        initial_messages = [Message.from_dict(item) for item in definition["initial_messages"]]
        stored_tool_names = {
            str(function["name"])
            for item in definition["tools"]
            if isinstance(item, dict)
            and isinstance(item.get("schema"), dict)
            and isinstance((function := item["schema"].get("function")), dict)
            and isinstance(function.get("name"), str)
        }
        handoff_tool_names = [transfer.tool_name for transfer in agent.handoffs if transfer.tool_name]
        return AgentTask(
            task_id=str(checkpoint.task_id),
            model=str(model["model_id"]),
            system_prompt=str(definition["compiled_prompt"]),
            user_prompt=str(definition["root_input"]),
            max_cycles=int(controls["max_cycles"]),
            memory_compact_threshold=int(controls["memory_compact_threshold"]),
            memory_threshold_percentage=int(controls["memory_threshold_percentage"]),
            no_tool_policy=cast(NoToolPolicy, controls["no_tool_policy"]),
            allow_interruption=bool(controls["allow_interruption"]),
            use_workspace=bool(stored_tool_names.intersection(WORKSPACE_TOOLS)),
            sub_agents=deepcopy(agent.sub_agents),
            agent_type=agent_definition.get("type"),
            native_multimodal=bool(controls["native_multimodal"]),
            extra_tool_names=[
                *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool) and tool.exposure != ToolExposure.HIDDEN],
                *handoff_tool_names,
            ],
            model_settings=run_config.model_settings,
            initial_messages=initial_messages,
            initial_shared_state=deepcopy(definition["initial_shared_state"]),
            metadata=metadata,
        )

    @staticmethod
    def _validate_frozen_static_prompt(
        agent: Agent,
        definition: dict[str, object],
        system_metadata: dict[str, object],
    ) -> None:
        sections = system_metadata.get("system_prompt_sections")
        section_items: list[Any] = sections if isinstance(sections, list) else []
        section_map = {str(item.get("id")): str(item.get("text") or "") for item in section_items if isinstance(item, dict)}
        if isinstance(agent.instructions, str):
            expected = agent.instructions.strip()
            observed = section_map.get("agent_instructions")
            if observed is None:
                observed = str(definition["compiled_prompt"]).strip()
            if observed != expected and not str(definition["compiled_prompt"]).startswith(expected):
                raise CheckpointError(
                    "static agent instructions do not match the frozen checkpoint prompt",
                    code="checkpoint_definition_mismatch",
                )
        if agent.sub_agents:
            expected_sub_agents = render_sub_agents(
                "en-US",
                {name: config.description for name, config in agent.sub_agents.items()},
            ).strip()
            if section_map.get("configured_sub_agents", "").strip() != expected_sub_agents:
                raise CheckpointError(
                    "configured sub-agents do not match the frozen checkpoint prompt",
                    code="checkpoint_definition_mismatch",
                )
