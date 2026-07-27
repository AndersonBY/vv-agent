from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.checkpoint import CheckpointError
from vv_agent.config import ResolvedModelConfig, project_resolved_model_limits
from vv_agent.constants import WORKSPACE_TOOLS
from vv_agent.context_providers import (
    ContextFragment,
    ContextRequest,
    collect_context_fragments,
)
from vv_agent.memory.session_memory import load_session_memory_context
from vv_agent.microcompaction import normalize_microcompaction_policy
from vv_agent.prompt import PromptBundle, PromptSection
from vv_agent.prompt.builder import _trim_text, inject_session_memory_section
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
        try:
            prompt_bundle = PromptBundle.from_dict(cast(dict[str, Any], definition["prompt_bundle"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint run definition has an invalid prompt bundle",
                code="checkpoint_definition_invalid",
            ) from exc
        self._validate_frozen_static_prompt(agent, prompt_bundle)
        self._validate_frozen_checkpoint_messages(getattr(checkpoint, "messages", None), prompt_bundle)

        metadata: dict[str, Any] = {}
        run_metadata = definition.get("run_metadata")
        if isinstance(run_metadata, dict):
            metadata.update(deepcopy(run_metadata))
        try:
            microcompaction_policy = normalize_microcompaction_policy(controls["microcompaction_policy"])
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint run definition has an invalid microcompaction policy",
                code="checkpoint_definition_invalid",
            ) from exc
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
            prompt_bundle=prompt_bundle,
            user_prompt=str(definition["root_input"]),
            max_cycles=int(controls["max_cycles"]),
            memory_compact_threshold=int(controls["memory_compact_threshold"]),
            memory_threshold_percentage=int(controls["memory_threshold_percentage"]),
            microcompaction_policy=microcompaction_policy,
            no_tool_policy=cast(NoToolPolicy, controls["no_tool_policy"]),
            allow_interruption=bool(controls["allow_interruption"]),
            use_workspace=bool(stored_tool_names.intersection(WORKSPACE_TOOLS)),
            sub_agents=deepcopy(agent.sub_agents),
            agent_type=agent_definition.get("type"),
            native_multimodal=bool(controls["native_multimodal"]),
            extra_tool_names=[
                *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool) and tool.exposure == ToolExposure.DIRECT],
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
        prompt_bundle: PromptBundle,
    ) -> None:
        section_map = {section.id: section.text for section in prompt_bundle.sections}
        if isinstance(agent.instructions, str):
            expected = _trim_text(agent.instructions)
            observed = section_map.get("agent_instructions")
            if observed != expected:
                raise CheckpointError(
                    "static agent instructions do not match the frozen checkpoint prompt",
                    code="checkpoint_definition_mismatch",
                )
        elif isinstance(agent.instructions, PromptBundle):
            expected_sections = agent.instructions.sections
            observed_prefix = prompt_bundle.sections[: len(expected_sections)]
            if observed_prefix != expected_sections:
                raise CheckpointError(
                    "static agent prompt bundle does not match the frozen checkpoint prompt",
                    code="checkpoint_definition_mismatch",
                )
        if agent.sub_agents:
            expected_sub_agents = render_sub_agents(
                "en-US",
                {name: config.description for name, config in agent.sub_agents.items()},
            )
            if _trim_text(section_map.get("configured_sub_agents", "")) != _trim_text(expected_sub_agents):
                raise CheckpointError(
                    "configured sub-agents do not match the frozen checkpoint prompt",
                    code="checkpoint_definition_mismatch",
                )

    @staticmethod
    def _validate_frozen_checkpoint_messages(
        messages: Any,
        prompt_bundle: PromptBundle,
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise CheckpointError(
                "checkpoint is missing its frozen system message",
                code="checkpoint_definition_mismatch",
            )
        first = messages[0]
        if not isinstance(first, Message) or first.role != "system" or first.content != prompt_bundle.flatten():
            raise CheckpointError(
                "checkpoint system message does not match the frozen prompt bundle",
                code="checkpoint_definition_mismatch",
            )
        if any(isinstance(message, Message) and message.role == "system" for message in messages[1:]):
            raise CheckpointError(
                "checkpoint contains a non-canonical system message",
                code="checkpoint_definition_mismatch",
            )
