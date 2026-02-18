from __future__ import annotations

import uuid
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.prompt import build_system_prompt
from v_agent.runtime import AgentRuntime
from v_agent.sdk.types import AgentDefinition, AgentRun, AgentSDKOptions
from v_agent.tools import build_default_registry
from v_agent.types import AgentStatus, AgentTask


class AgentSDKClient:
    """Minimal SDK-style client for running named v-agent definitions."""

    def __init__(self, *, options: AgentSDKOptions, agents: dict[str, AgentDefinition] | None = None) -> None:
        self.options = options
        self._agents: dict[str, AgentDefinition] = dict(agents or {})

    def register_agent(self, name: str, definition: AgentDefinition) -> None:
        if not name:
            raise ValueError("Agent name cannot be empty")
        self._agents[name] = definition

    def register_agents(self, agents: dict[str, AgentDefinition]) -> None:
        for name, definition in agents.items():
            self.register_agent(name, definition)

    def list_agents(self) -> list[str]:
        return sorted(self._agents.keys())

    def prepare_task(
        self,
        *,
        agent_name: str,
        prompt: str,
        resolved_model_id: str,
    ) -> AgentTask:
        definition = self._get_agent(agent_name)
        metadata = dict(definition.metadata)
        metadata.setdefault("language", definition.language)
        if definition.sub_agents:
            metadata.setdefault("sub_agent_names", sorted(definition.sub_agents.keys()))

        if definition.system_prompt:
            system_prompt = definition.system_prompt
        else:
            system_prompt = build_system_prompt(
                definition.description,
                language=definition.language,
                allow_interruption=definition.allow_interruption,
                use_workspace=definition.use_workspace,
                enable_todo_management=definition.enable_todo_management,
                agent_type=definition.agent_type,
                available_sub_agents={
                    name: config.description for name, config in definition.sub_agents.items()
                }
                if definition.sub_agents
                else None,
            )

        return AgentTask(
            task_id=f"{agent_name}_{uuid.uuid4().hex[:8]}",
            model=resolved_model_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_cycles=max(definition.max_cycles, 1),
            no_tool_policy=definition.no_tool_policy,
            allow_interruption=definition.allow_interruption,
            use_workspace=definition.use_workspace,
            has_sub_agents=definition.enable_sub_agents,
            sub_agents=dict(definition.sub_agents),
            agent_type=definition.agent_type,
            enable_document_tools=definition.enable_document_tools,
            enable_document_write_tools=definition.enable_document_write_tools,
            native_multimodal=definition.native_multimodal,
            extra_tool_names=list(definition.extra_tool_names),
            exclude_tools=list(definition.exclude_tools),
            metadata=metadata,
        )

    def run_agent(
        self,
        *,
        agent_name: str,
        prompt: str,
        shared_state: dict[str, Any] | None = None,
    ) -> AgentRun:
        definition = self._get_agent(agent_name)
        backend = definition.backend or self.options.default_backend
        llm_builder = self.options.llm_builder or build_openai_llm_from_local_settings
        llm, resolved = llm_builder(
            self.options.settings_file,
            backend=backend,
            model=definition.model,
            timeout_seconds=self.options.timeout_seconds,
        )

        tool_registry_factory = self.options.tool_registry_factory or build_default_registry
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=tool_registry_factory(),
            default_workspace=self.options.workspace,
            log_handler=self.options.log_handler,
            settings_file=self.options.settings_file,
            default_backend=backend,
            llm_builder=llm_builder,
            tool_registry_factory=tool_registry_factory,
        )

        task = self.prepare_task(
            agent_name=agent_name,
            prompt=prompt,
            resolved_model_id=resolved.model_id,
        )
        result = runtime.run(task, shared_state=shared_state)
        return AgentRun(agent_name=agent_name, result=result, resolved=resolved)

    def query(
        self,
        *,
        agent_name: str,
        prompt: str,
        shared_state: dict[str, Any] | None = None,
        require_completed: bool = True,
    ) -> str:
        run = self.run_agent(agent_name=agent_name, prompt=prompt, shared_state=shared_state)
        if run.result.status == AgentStatus.COMPLETED:
            return run.result.final_answer or ""

        if require_completed:
            reason = (
                run.result.error
                or run.result.wait_reason
                or run.result.final_answer
                or "query did not complete successfully"
            )
            raise RuntimeError(f"Agent query failed with status={run.result.status.value}: {reason}")

        return run.result.final_answer or run.result.wait_reason or run.result.error or ""

    def _get_agent(self, agent_name: str) -> AgentDefinition:
        definition = self._agents.get(agent_name)
        if definition is None:
            available = ", ".join(sorted(self._agents))
            raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
        return definition
