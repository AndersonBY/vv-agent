from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from vv_agent.agent import Agent, RunContext
from vv_agent.config import ResolvedModelConfig, project_resolved_model_limits
from vv_agent.context_providers import (
    ContextFragment,
    ContextRequest,
    collect_context_fragments,
)
from vv_agent.memory.session_memory import load_session_memory_context
from vv_agent.prompt import PromptBundle, PromptSection
from vv_agent.prompt.builder import inject_session_memory_section
from vv_agent.prompt.templates import render_sub_agents
from vv_agent.run_config import RunConfig, ToolPolicy, _validate_bounded_int
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolSideEffect
from vv_agent.types import AgentTask

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
        task_id = f"{agent.name}_{uuid.uuid4().hex[:8]}"
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
        no_tool_policy = run_config.no_tool_policy or agent.no_tool_policy or "finish"
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
            session=None,
            workspace=run_config.workspace,
            context=run_config.context,
            metadata=metadata,
            max_prompt_chars=run_config.max_context_chars,
        )
        compiler_sections: list[PromptSection] = []
        if agent.sub_agents:
            compiler_sections.append(
                PromptSection(
                    id="configured_sub_agents",
                    text=render_sub_agents(
                        "en-US",
                        {name: config.description for name, config in agent.sub_agents.items()},
                    ),
                    stable=True,
                    source="agent.sub_agents",
                )
            )
        provider_fragments: list[ContextFragment] = []
        if run_config.context_providers:
            provider_fragments = collect_context_fragments(request, run_config.context_providers)
        prompt_bundle, omitted_section_ids = self._assemble_prompt_bundle(
            instructions=resolved_instructions,
            compiler_sections=compiler_sections,
            provider_fragments=provider_fragments,
            max_prompt_chars=run_config.max_context_chars,
        )
        prompt_bundle = self._inject_loaded_session_memory(
            prompt_bundle=prompt_bundle,
            metadata=metadata,
            task_id=task_id,
            workspace=run_config.workspace,
        )
        if omitted_section_ids:
            metadata["omitted_prompt_section_ids"] = omitted_section_ids

        max_cycles = _validate_bounded_int(
            run_config.max_cycles if run_config.max_cycles is not None else 10,
            "max_cycles",
            minimum=1,
        )
        assert max_cycles is not None
        return AgentTask(
            task_id=task_id,
            model=str(resolved.model_id or model),
            prompt_bundle=prompt_bundle,
            user_prompt=input,
            max_cycles=max_cycles,
            microcompaction_policy=run_config.microcompaction_policy,
            no_tool_policy=no_tool_policy,
            sub_agents=deepcopy(agent.sub_agents),
            native_multimodal=resolved.native_multimodal,
            extra_tool_names=[
                *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool) and tool.exposure == ToolExposure.DIRECT],
                *handoff_tool_names,
            ],
            model_settings=run_config.model_settings,
            initial_messages=list(run_config.initial_messages or []),
            initial_shared_state=dict(run_config.shared_state or {}),
            metadata=metadata,
        )

    @staticmethod
    def _inject_loaded_session_memory(
        *,
        prompt_bundle: PromptBundle,
        metadata: dict[str, Any],
        task_id: str,
        workspace: str | Path | None,
    ) -> PromptBundle:
        if metadata.get("session_memory_enabled") is not True or workspace is None:
            return prompt_bundle
        session_id = metadata.get("session_id")
        storage_scope = str(session_id).strip() if isinstance(session_id, str) and session_id.strip() else task_id
        context = load_session_memory_context(
            workspace=Path(workspace).resolve(),
            storage_scope=storage_scope,
            storage_dir=str(metadata.get("session_memory_storage_dir", ".memory/session")),
        )
        return inject_session_memory_section(prompt_bundle, context)

    @staticmethod
    def _assemble_prompt_bundle(
        *,
        instructions: str | PromptBundle,
        compiler_sections: list[PromptSection],
        provider_fragments: list[ContextFragment],
        max_prompt_chars: int | None,
    ) -> tuple[PromptBundle, list[str]]:
        if isinstance(instructions, PromptBundle):
            sections = list(instructions.sections)
        else:
            sections = [
                PromptSection(
                    id="agent_instructions",
                    text=instructions,
                    stable=True,
                    source="agent.instructions",
                )
            ]

        omitted_section_ids: list[str] = []

        def append_bounded(section: PromptSection) -> None:
            current_chars = sum(len(item.text) for item in sections) + max(0, len(sections) - 1) * 2
            next_chars = current_chars + (2 if sections else 0) + len(section.text)
            if max_prompt_chars is not None and next_chars > max(int(max_prompt_chars), 0):
                omitted_section_ids.append(section.id)
                return
            sections.append(section)

        for section in compiler_sections:
            append_bounded(section)
        for fragment in sorted(
            provider_fragments,
            key=lambda item: (
                int(item.priority),
                0 if item.stable else 1,
                str(item.id).encode("utf-16-be"),
            ),
        ):
            text = str(fragment.text or "").strip("\t\n\v\f\r ")
            if not text:
                continue
            append_bounded(
                PromptSection(
                    id=fragment.id,
                    text=text,
                    stable=fragment.stable,
                    source=fragment.source or None,
                    cache_hint=fragment.cache_hint,
                    metadata=dict(fragment.metadata),
                )
            )
        return PromptBundle(sections=tuple(sections)), omitted_section_ids


def tool_is_enabled(*, tool: FunctionTool, agent: Agent, run_config: RunConfig) -> bool:
    if callable(tool.is_enabled):
        run_context = RunContext(
            context=run_config.context,
            metadata={**agent.metadata, **run_config.metadata},
        )
        return bool(tool.is_enabled(run_context, agent))
    return bool(tool.is_enabled)
