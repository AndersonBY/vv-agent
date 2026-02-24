from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import BeforeCycleMessageProvider, InterruptionMessageProvider
from vv_agent.sdk.types import AgentDefinition, AgentRun, AgentSDKOptions, RuntimeLogHandler
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, Message


def _compose_log_handlers(*handlers: RuntimeLogHandler | None) -> RuntimeLogHandler | None:
    valid_handlers = [handler for handler in handlers if handler is not None]
    if not valid_handlers:
        return None

    def combined(event: str, payload: dict[str, Any]) -> None:
        for handler in valid_handlers:
            handler(event, payload)

    return combined


class AgentSDKClient:
    """SDK client with both simple single-agent mode and named profile mode."""

    def __init__(
        self,
        *,
        options: AgentSDKOptions,
        agent: AgentDefinition | None = None,
        agents: dict[str, AgentDefinition] | None = None,
    ) -> None:
        self.options = options
        self._default_agent = agent
        self._default_agent_name = "default"
        self._agents: dict[str, AgentDefinition] = dict(agents or {})
        self._prompt_templates: dict[str, str] = {}
        self._resource_skill_directories: list[str] = []
        self._resource_diagnostics: list[str] = []
        self._runtime_hooks = list(options.runtime_hooks)
        self._resource_loader = options.resource_loader

        if self._resource_loader is not None and self.options.auto_discover_resources:
            discovered = self._resource_loader.discover()
            for name, definition in discovered.agents.items():
                self._agents.setdefault(name, definition)
            self._prompt_templates.update(discovered.prompts)
            self._resource_skill_directories = list(discovered.skill_directories)
            self._runtime_hooks.extend(discovered.hooks)
            self._resource_diagnostics = list(discovered.diagnostics)

    @property
    def resource_diagnostics(self) -> list[str]:
        return list(self._resource_diagnostics)

    def register_agent(self, name: str, definition: AgentDefinition) -> None:
        if not name:
            raise ValueError("Agent name cannot be empty")
        self._agents[name] = definition

    def register_agents(self, agents: dict[str, AgentDefinition]) -> None:
        for name, definition in agents.items():
            self.register_agent(name, definition)

    def list_agents(self) -> list[str]:
        return sorted(self._agents.keys())

    def set_default_agent(self, definition: AgentDefinition) -> None:
        self._default_agent = definition

    def prepare_task(
        self,
        *,
        prompt: str,
        resolved_model_id: str,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
        task_name: str | None = None,
        workspace: str | Path | None = None,
    ) -> AgentTask:
        effective_workspace = self._resolve_workspace(workspace)
        resolved_name, raw_definition = self._resolve_agent(agent=agent, agent_name=agent_name)
        definition = self._effective_definition(raw_definition)
        effective_task_name = task_name or resolved_name
        metadata = dict(definition.metadata)
        metadata.setdefault("language", definition.language)
        if definition.sub_agents:
            metadata.setdefault("sub_agent_names", sorted(definition.sub_agents.keys()))

        raw_available_skills = metadata.get("available_skills")
        if not isinstance(raw_available_skills, list):
            raw_available_skills = metadata.get("bound_skills")

        if not isinstance(raw_available_skills, list):
            configured_skill_dirs: list[str] = []
            for item in definition.skill_directories:
                if isinstance(item, str) and item.strip():
                    configured_skill_dirs.append(item.strip())

            metadata_skill_dirs = metadata.get("skill_directories")
            if isinstance(metadata_skill_dirs, str) and metadata_skill_dirs.strip():
                configured_skill_dirs.append(metadata_skill_dirs.strip())
            elif isinstance(metadata_skill_dirs, list):
                for item in metadata_skill_dirs:
                    if isinstance(item, str) and item.strip():
                        configured_skill_dirs.append(item.strip())

            if configured_skill_dirs:
                raw_available_skills = configured_skill_dirs
                metadata.setdefault("available_skills", list(configured_skill_dirs))
            else:
                raw_available_skills = None

        available_skills: list[dict[str, Any] | str] | None = None
        if isinstance(raw_available_skills, list):
            normalized_skills: list[dict[str, Any] | str] = []
            for item in raw_available_skills:
                if isinstance(item, str | dict):
                    normalized_skills.append(item)
            if normalized_skills:
                available_skills = normalized_skills

        prompt_definition = definition.description
        if definition.system_prompt_template:
            template = self._prompt_templates.get(definition.system_prompt_template)
            if isinstance(template, str) and template.strip():
                prompt_definition = template

        if definition.system_prompt:
            system_prompt = definition.system_prompt
        else:
            system_prompt = build_system_prompt(
                prompt_definition,
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
                available_skills=available_skills,
                workspace=effective_workspace,
            )

        return AgentTask(
            task_id=f"{effective_task_name}_{uuid.uuid4().hex[:8]}",
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
            native_multimodal=definition.native_multimodal,
            extra_tool_names=list(definition.extra_tool_names),
            exclude_tools=list(definition.exclude_tools),
            metadata=metadata,
        )

    def run(
        self,
        *,
        prompt: str,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        log_handler: RuntimeLogHandler | None = None,
        initial_messages: list[Message] | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        task_name: str | None = None,
    ) -> AgentRun:
        resolved_name, definition = self._resolve_agent(agent=agent, agent_name=agent_name)
        return self._execute(
            prompt=prompt,
            resolved_name=resolved_name,
            definition=definition,
            workspace=workspace,
            shared_state=shared_state,
            log_handler=log_handler,
            initial_messages=initial_messages,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            task_name=task_name,
        )

    def run_agent(
        self,
        *,
        agent_name: str,
        prompt: str,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
    ) -> AgentRun:
        return self.run(prompt=prompt, agent=agent_name, workspace=workspace, shared_state=shared_state)

    def query(
        self,
        *,
        prompt: str,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        require_completed: bool = True,
    ) -> str:
        run = self.run(
            prompt=prompt,
            agent=agent,
            agent_name=agent_name,
            workspace=workspace,
            shared_state=shared_state,
        )
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

    def query_agent(
        self,
        *,
        agent_name: str,
        prompt: str,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        require_completed: bool = True,
    ) -> str:
        return self.query(
            prompt=prompt,
            agent=agent_name,
            workspace=workspace,
            shared_state=shared_state,
            require_completed=require_completed,
        )

    def create_session(
        self,
        *,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
    ):
        resolved_name, definition = self._resolve_agent(agent=agent, agent_name=agent_name)
        effective_workspace = self._resolve_workspace(workspace)

        from vv_agent.sdk.session import create_agent_session

        return create_agent_session(
            execute_run=self._execute,
            agent_name=resolved_name,
            definition=definition,
            workspace=effective_workspace,
            shared_state=shared_state,
        )

    def _execute(
        self,
        *,
        prompt: str,
        resolved_name: str | None = None,
        definition: AgentDefinition | None = None,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        log_handler: RuntimeLogHandler | None = None,
        initial_messages: list[Message] | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        task_name: str | None = None,
    ) -> AgentRun:
        if definition is None or resolved_name is None:
            resolved_name, definition = self._resolve_agent(agent=agent, agent_name=agent_name)
        definition = self._effective_definition(definition)
        effective_workspace = self._resolve_workspace(workspace)
        run_name = task_name or resolved_name
        backend = definition.backend or self.options.default_backend
        llm_builder = self.options.llm_builder or build_openai_llm_from_local_settings
        llm, resolved = llm_builder(
            self.options.settings_file,
            backend=backend,
            model=definition.model,
            timeout_seconds=self.options.timeout_seconds,
        )
        if self.options.debug_dump_dir and hasattr(llm, "debug_dump_dir"):
            cast(Any, llm).debug_dump_dir = self.options.debug_dump_dir

        tool_registry_factory = self.options.tool_registry_factory or build_default_registry
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=tool_registry_factory(),
            default_workspace=effective_workspace,
            log_handler=_compose_log_handlers(self.options.log_handler, log_handler),
            log_preview_chars=self.options.log_preview_chars,
            settings_file=self.options.settings_file,
            default_backend=backend,
            llm_builder=llm_builder,
            tool_registry_factory=tool_registry_factory,
            hooks=list(self._runtime_hooks),
            execution_backend=self.options.execution_backend,
        )

        task = self.prepare_task(
            prompt=prompt,
            resolved_model_id=resolved.model_id,
            agent=definition,
            task_name=run_name,
            workspace=effective_workspace,
        )

        ctx: ExecutionContext | None = None
        if self.options.stream_callback is not None:
            ctx = ExecutionContext(stream_callback=self.options.stream_callback)

        result = runtime.run(
            task,
            workspace=effective_workspace,
            shared_state=shared_state,
            initial_messages=initial_messages,
            user_message=prompt,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            ctx=ctx,
        )
        return AgentRun(agent_name=run_name, result=result, resolved=resolved)

    def _effective_definition(self, definition: AgentDefinition) -> AgentDefinition:
        effective = definition
        if not effective.skill_directories and self._resource_skill_directories:
            effective = replace(effective, skill_directories=list(self._resource_skill_directories))
        return effective

    def _resolve_agent(
        self,
        *,
        agent: str | AgentDefinition | None = None,
        agent_name: str | None = None,
    ) -> tuple[str, AgentDefinition]:
        if agent is not None and agent_name is not None:
            raise ValueError("Use either 'agent' or 'agent_name', not both.")

        if agent_name is not None:
            agent = agent_name

        if isinstance(agent, AgentDefinition):
            return "inline", agent

        if isinstance(agent, str):
            return agent, self._get_agent(agent)

        if self._default_agent is not None:
            return self._default_agent_name, self._default_agent

        if len(self._agents) == 1:
            only_name = next(iter(self._agents))
            return only_name, self._agents[only_name]

        if not self._agents:
            raise ValueError(
                "No agent configured. Pass agent=AgentDefinition(...) or register named agents first."
            )

        available = ", ".join(sorted(self._agents))
        raise ValueError(f"Multiple agents configured. Pass agent='name'. Available: {available}")

    def _get_agent(self, agent_name: str) -> AgentDefinition:
        definition = self._agents.get(agent_name)
        if definition is None:
            available = ", ".join(sorted(self._agents))
            raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
        return definition

    def _resolve_workspace(self, workspace: str | Path | None = None) -> Path:
        raw = workspace
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                raw = None
        target = Path(raw) if raw is not None else self.options.workspace
        return Path(target).expanduser().resolve()


def run(
    *,
    prompt: str,
    agent: AgentDefinition,
    options: AgentSDKOptions,
    workspace: str | Path | None = None,
    shared_state: dict[str, Any] | None = None,
) -> AgentRun:
    """One-shot helper aligned with SDK-style simple invocation."""

    client = AgentSDKClient(options=options, agent=agent)
    return client.run(prompt=prompt, workspace=workspace, shared_state=shared_state)


def query(
    *,
    prompt: str,
    agent: AgentDefinition,
    options: AgentSDKOptions,
    workspace: str | Path | None = None,
    shared_state: dict[str, Any] | None = None,
    require_completed: bool = True,
) -> str:
    """One-shot text helper aligned with SDK-style simple invocation."""

    client = AgentSDKClient(options=options, agent=agent)
    return client.query(
        prompt=prompt,
        workspace=workspace,
        shared_state=shared_state,
        require_completed=require_completed,
    )
