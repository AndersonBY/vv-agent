from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalProvider
from vv_agent.config import ResolvedModelConfig, build_openai_llm_from_local_settings
from vv_agent.context_providers import (
    ContextFragment,
    ContextProvider,
    ContextRequest,
    assemble_context_fragments,
    collect_context_fragments,
)
from vv_agent.llm.base import LLMClient
from vv_agent.memory.provider import MemoryProvider
from vv_agent.prompt import build_raw_system_prompt_sections, build_system_prompt_bundle
from vv_agent.run_config import RunConfig, ToolPolicy
from vv_agent.runner import Runner
from vv_agent.runtime import CancellationToken
from vv_agent.runtime.backends import ExecutionBackend
from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.engine import register_sub_agent_session, unregister_sub_agent_session
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools import ToolRegistry, build_default_registry
from vv_agent.types import AgentResult, AgentStatus, AgentTask, Message, NoToolPolicy, SubAgentConfig

LLMBuilder = Callable[..., tuple[LLMClient, ResolvedModelConfig]]
RuntimeLogHandler = Callable[[str, dict[str, Any]], None]
SessionEventHandler = Callable[[str, dict[str, Any]], None]
StreamCallback = Callable[[Any], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]


class _SupportsDebugDumpDir(Protocol):
    debug_dump_dir: str | None


@dataclass(slots=True)
class InteractiveAgentDefinition:
    description: str
    model: str
    backend: str | None = None
    language: str = "zh-CN"
    max_cycles: int = 10
    memory_compact_threshold: int = 128_000
    memory_threshold_percentage: int = 90
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    enable_todo_management: bool = True
    agent_type: str | None = None
    native_multimodal: bool = False
    enable_sub_agents: bool = False
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    skill_directories: list[str] = field(default_factory=list)
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    bash_shell: str | None = None
    windows_shell_priority: list[str] = field(default_factory=list)
    bash_env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    system_prompt_template: str | None = None
    context_providers: list[ContextProvider] = field(default_factory=list)
    memory_providers: list[MemoryProvider] = field(default_factory=list)


@dataclass(slots=True)
class AgentSessionOptions:
    settings_file: Path
    default_backend: str
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    timeout_seconds: float = 90.0
    log_preview_chars: int | None = None
    llm_builder: LLMBuilder | None = None
    tool_registry_factory: ToolRegistryFactory | None = None
    log_handler: RuntimeLogHandler | None = None
    runtime_hooks: list[RuntimeHook] = field(default_factory=list)
    execution_backend: ExecutionBackend | None = None
    stream_callback: StreamCallback | None = None
    debug_dump_dir: str | None = None
    approval_provider: ApprovalProvider | None = None
    approval_timeout_seconds: float | None = None
    tool_policy: ToolPolicy | None = None
    bash_shell: str | None = None
    windows_shell_priority: list[str] = field(default_factory=list)
    bash_env: dict[str, str] = field(default_factory=dict)
    context_providers: list[ContextProvider] = field(default_factory=list)
    memory_providers: list[MemoryProvider] = field(default_factory=list)


@dataclass(slots=True)
class AgentSessionRun:
    agent_name: str
    result: AgentResult
    resolved: ResolvedModelConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "status": self.result.status.value,
            "final_answer": self.result.final_answer,
            "wait_reason": self.result.wait_reason,
            "error": self.result.error,
            "cycles": len(self.result.cycles),
            "todo_list": self.result.todo_list,
            "token_usage": self.result.token_usage.to_dict(),
            "resolved": {
                "backend": self.resolved.backend,
                "selected_model": self.resolved.selected_model,
                "model_id": self.resolved.model_id,
                "endpoint": self.resolved.endpoint.endpoint_id,
            },
        }


@dataclass(slots=True)
class AgentSessionState:
    running: bool
    workspace: Path
    messages: list[Message] = field(default_factory=list)
    shared_state: dict[str, Any] = field(default_factory=dict)
    latest_run: AgentSessionRun | None = None


class AgentSession:
    """Stateful, interactive session wrapper for desktop/runtime integrations."""

    def __init__(
        self,
        *,
        execute_run: Callable[..., AgentSessionRun],
        session_id: str | None = None,
        agent_name: str,
        definition: InteractiveAgentDefinition,
        workspace: Path,
        shared_state: dict[str, Any] | None = None,
    ) -> None:
        self._execute_run = execute_run
        self.session_id = str(session_id or uuid.uuid4().hex[:12]).strip() or uuid.uuid4().hex[:12]
        self.agent_name = agent_name
        self.definition = definition
        self.workspace = Path(workspace).resolve()
        self._messages: list[Message] = []
        self._shared_state: dict[str, Any] = dict(shared_state or {})
        self._shared_state.setdefault("todo_list", [])
        self._latest_run: AgentSessionRun | None = None
        self._running = False
        self._listeners: list[SessionEventHandler] = []
        self._background_command_unsubscribers: dict[str, Callable[[], None]] = {}
        self._steering_queue: deque[str] = deque()
        self._follow_up_queue: deque[str] = deque()
        self._active_cancellation_token: CancellationToken | None = None
        self._active_run_handle: Any | None = None
        self._lock = RLock()

    @property
    def messages(self) -> list[Message]:
        with self._lock:
            return list(self._messages)

    @property
    def shared_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._shared_state)

    @property
    def latest_run(self) -> AgentSessionRun | None:
        with self._lock:
            return self._latest_run

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def active_run_handle(self) -> Any | None:
        with self._lock:
            return self._active_run_handle

    def subscribe(self, listener: SessionEventHandler) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe

    def steer(self, prompt: str) -> None:
        text = prompt.strip()
        if not text:
            raise ValueError("steer prompt cannot be empty")
        with self._lock:
            self._steering_queue.append(text)
        self._emit("session_steer_queued", prompt=text)

    def follow_up(self, prompt: str) -> None:
        text = prompt.strip()
        if not text:
            raise ValueError("follow_up prompt cannot be empty")
        with self._lock:
            self._follow_up_queue.append(text)
        self._emit("session_follow_up_queued", prompt=text)

    def clear_queues(self) -> None:
        with self._lock:
            self._steering_queue.clear()
            self._follow_up_queue.clear()
        self._emit("session_queues_cleared")

    def cancel(self) -> bool:
        with self._lock:
            if not self._running or self._active_cancellation_token is None:
                return False
            self._active_cancellation_token.cancel()
            self._steering_queue.clear()
            self._follow_up_queue.clear()
        self._emit("session_cancel_requested")
        return True

    def approve(self, request_id: str, decision: Any) -> None:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("approval request_id cannot be empty")
        with self._lock:
            handle = self._active_run_handle
        if handle is None:
            raise RuntimeError("No active run handle is available for approval.")
        handle.approve(normalized_request_id, decision)

    def prompt(self, prompt: str, *, auto_follow_up: bool = True) -> AgentSessionRun:
        text = prompt.strip()
        if not text:
            raise ValueError("prompt cannot be empty")

        run = self._run_once(text)
        if not auto_follow_up:
            return run

        while True:
            with self._lock:
                if run.result.status != AgentStatus.COMPLETED or not self._follow_up_queue:
                    break
                follow_up_prompt = self._follow_up_queue.popleft()
            self._emit("session_follow_up_dequeued", prompt=follow_up_prompt)
            run = self._run_once(follow_up_prompt)
        return run

    def continue_run(self, prompt: str | None = None) -> AgentSessionRun:
        if prompt is not None and prompt.strip():
            return self.prompt(prompt.strip(), auto_follow_up=False)

        queued_prompt = self._drain_next_queued_prompt()
        if queued_prompt is None:
            raise ValueError("No queued prompt available. Provide prompt or call steer()/follow_up() first.")
        return self.prompt(queued_prompt, auto_follow_up=False)

    def query(self, prompt: str, *, require_completed: bool = True) -> str:
        run = self.prompt(prompt)
        if run.result.status == AgentStatus.COMPLETED:
            return run.result.final_answer or ""
        if require_completed:
            reason = run.result.error or run.result.wait_reason or run.result.final_answer or "session query did not complete"
            raise RuntimeError(f"Session query failed with status={run.result.status.value}: {reason}")
        return run.result.final_answer or run.result.wait_reason or run.result.error or ""

    def state(self) -> AgentSessionState:
        with self._lock:
            return AgentSessionState(
                running=self._running,
                workspace=self.workspace,
                messages=list(self._messages),
                shared_state=dict(self._shared_state),
                latest_run=self._latest_run,
            )

    def replace_messages(self, messages: list[Message]) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot replace messages while session is running.")
            self._messages = list(messages)
        self._emit("session_messages_replaced", message_count=len(messages))

    def replace_shared_state(self, shared_state: dict[str, Any]) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot replace shared_state while session is running.")
            self._shared_state = dict(shared_state)
            self._shared_state.setdefault("todo_list", [])
        self._emit("session_shared_state_replaced")

    def _run_once(self, prompt: str) -> AgentSessionRun:
        with self._lock:
            if self._running:
                raise RuntimeError("Session is already running. Queue with steer()/follow_up() or wait for completion.")
            self._running = True
            self._active_cancellation_token = CancellationToken()
            initial_messages = list(self._messages)
            current_shared_state = dict(self._shared_state)

        self._emit("session_run_start", prompt=prompt, existing_messages=len(initial_messages))
        try:
            run_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "session_id": self.session_id,
                "agent": self.definition,
                "task_name": self.agent_name,
                "workspace": self.workspace,
                "shared_state": current_shared_state,
                "initial_messages": initial_messages,
                "before_cycle_messages": self._before_cycle_messages,
                "interruption_messages": self._interruption_messages,
                "log_handler": self._session_log_handler,
                "cancellation_token": self._active_cancellation_token,
                "active_handle_callback": self._set_active_run_handle,
            }
            run = self._execute_run(**run_kwargs)
        finally:
            self._set_active_run_handle(None)
            with self._lock:
                self._running = False
                self._active_cancellation_token = None

        with self._lock:
            self._messages = list(run.result.messages)
            self._shared_state = dict(run.result.shared_state)
            self._latest_run = run

        self._emit(
            "session_run_end",
            status=run.result.status.value,
            cycles=len(run.result.cycles),
            final_answer=run.result.final_answer,
            wait_reason=run.result.wait_reason,
            error=run.result.error,
        )
        return run

    def _set_active_run_handle(self, handle: Any | None) -> None:
        with self._lock:
            if self._active_run_handle is handle:
                return
            self._active_run_handle = handle
        self._emit("session_active_run_handle_changed", handle=handle)

    def _drain_next_queued_prompt(self) -> str | None:
        with self._lock:
            if self._steering_queue:
                return self._steering_queue.popleft()
            if self._follow_up_queue:
                return self._follow_up_queue.popleft()
        return None

    def _before_cycle_messages(self, cycle_index: int, _: list[Message], __: dict[str, Any]) -> list[Message]:
        del _, __
        with self._lock:
            if not self._steering_queue:
                return []
            prompt = self._steering_queue.popleft()
        self._emit("session_steer_dequeued", cycle=cycle_index, prompt=prompt)
        return [Message(role="user", content=prompt)]

    def _interruption_messages(self) -> list[Message]:
        with self._lock:
            if not self._steering_queue:
                return []
            prompt = self._steering_queue.popleft()
        self._emit("session_steer_interrupt", prompt=prompt)
        return [Message(role="user", content=prompt)]

    def _session_log_handler(self, event: str, payload: dict[str, Any]) -> None:
        self._sync_background_command_watchers(event, payload)
        self._emit(event, **payload)

    def _sync_background_command_watchers(self, event: str, payload: dict[str, Any]) -> None:
        if event != "tool_result" or not isinstance(payload, dict):
            return
        tool_name = str(payload.get("tool_name") or "").strip().lower()
        if tool_name not in {"bash", "check_background_command"}:
            return

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return

        background_session_id = str(metadata.get("session_id") or "").strip()
        if not background_session_id:
            return

        status = str(metadata.get("status") or payload.get("status") or "").strip().lower()
        if status == "running":
            self._subscribe_background_command(background_session_id)
            return
        if status in {"completed", "failed", "timeout", "missing"}:
            self._unsubscribe_background_command(background_session_id)

    def _subscribe_background_command(self, background_session_id: str) -> None:
        normalized_session_id = str(background_session_id or "").strip()
        if not normalized_session_id:
            return
        with self._lock:
            if normalized_session_id in self._background_command_unsubscribers:
                return
            self._background_command_unsubscribers[normalized_session_id] = lambda: None

        def _listener(payload: dict[str, Any]) -> None:
            self._handle_background_command_terminal(normalized_session_id, payload)

        unsubscribe = background_session_manager.subscribe(normalized_session_id, _listener)
        with self._lock:
            if normalized_session_id in self._background_command_unsubscribers:
                self._background_command_unsubscribers[normalized_session_id] = unsubscribe
            else:
                unsubscribe()

    def _unsubscribe_background_command(self, background_session_id: str) -> None:
        normalized_session_id = str(background_session_id or "").strip()
        if not normalized_session_id:
            return
        with self._lock:
            unsubscribe = self._background_command_unsubscribers.pop(normalized_session_id, None)
        if unsubscribe is not None:
            unsubscribe()

    def _handle_background_command_terminal(self, background_session_id: str, payload: dict[str, Any]) -> None:
        self._unsubscribe_background_command(background_session_id)
        notification_message = self._build_background_command_notification(payload)
        queued_to_running_session = False
        with self._lock:
            running = self._running
        if running:
            self.steer(notification_message)
            queued_to_running_session = True

        event_payload = dict(payload)
        event_payload["session_id"] = background_session_id
        event_payload["notification_message"] = notification_message
        event_payload["queued_to_running_session"] = queued_to_running_session

        status = str(payload.get("status") or "").strip().lower() or "terminal"
        self._emit(f"background_command_{status}", **event_payload)
        self._emit("background_command_terminal", **event_payload)

    @staticmethod
    def _build_background_command_notification(payload: dict[str, Any]) -> str:
        status = str(payload.get("status") or "").strip().lower()
        status_text = {
            "completed": "completed",
            "failed": "failed",
            "timeout": "timed out",
        }.get(status, status or "updated")
        background_session_id = str(payload.get("session_id") or "").strip()
        command = str(payload.get("command") or "").strip()
        output = str(payload.get("output") or "").strip()
        exit_code = payload.get("exit_code")
        summary = output or f"exit_code={exit_code}"
        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."

        lines = [f"System notification: background command {background_session_id} {status_text}."]
        if command:
            lines.append(f"Command: {command}")
        if summary:
            lines.append(f"Summary: {summary}")
        return "\n".join(lines)

    def _emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(event, payload)

class InteractiveAgentClient:
    """Session client backed by vv-agent runtime primitives."""

    def __init__(self, *, options: AgentSessionOptions) -> None:
        self.options = options

    def create_session(
        self,
        *,
        agent: InteractiveAgentDefinition,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        definition = self._apply_startup_shell_defaults(agent)
        effective_workspace = self._resolve_workspace(workspace)
        session_sub_task_manager = SubTaskManager(
            register_session=register_sub_agent_session,
            unregister_session=unregister_sub_agent_session,
        )

        def _execute_session_run(**kwargs: Any) -> AgentSessionRun:
            return self._execute(**kwargs, sub_task_manager=session_sub_task_manager)

        return AgentSession(
            execute_run=_execute_session_run,
            session_id=session_id,
            agent_name="inline",
            definition=definition,
            workspace=effective_workspace,
            shared_state=shared_state,
        )

    def prepare_task(
        self,
        *,
        prompt: str,
        resolved_model_id: str,
        agent: InteractiveAgentDefinition,
        task_name: str | None = None,
        workspace: str | Path | None = None,
        session_id: str | None = None,
    ) -> AgentTask:
        effective_workspace = self._resolve_workspace(workspace)
        definition = self._apply_startup_shell_defaults(agent)
        effective_task_name = task_name or "inline"
        metadata = dict(definition.metadata)
        normalized_session_id = str(session_id or "").strip()
        if normalized_session_id:
            metadata.setdefault("session_id", normalized_session_id)
        metadata.setdefault("language", definition.language)
        if definition.bash_shell:
            metadata.setdefault("bash_shell", definition.bash_shell)
        if definition.windows_shell_priority:
            metadata.setdefault("windows_shell_priority", list(definition.windows_shell_priority))
        if definition.bash_env:
            metadata.setdefault("bash_env", dict(definition.bash_env))
        if definition.sub_agents:
            metadata.setdefault("sub_agent_names", sorted(definition.sub_agents.keys()))

        available_skills: list[dict[str, Any] | str] | None = None
        if isinstance(metadata.get("available_skills"), list):
            available_skills = metadata["available_skills"]
        elif definition.skill_directories:
            available_skills = [
                directory.strip()
                for directory in definition.skill_directories
                if isinstance(directory, str) and directory.strip()
            ]
            if available_skills:
                metadata["available_skills"] = list(available_skills)
            else:
                available_skills = None

        if definition.system_prompt is not None:
            system_prompt = definition.system_prompt
            generated_sections = build_raw_system_prompt_sections(system_prompt)
        else:
            prompt_bundle = build_system_prompt_bundle(
                definition.description,
                language=definition.language,
                allow_interruption=definition.allow_interruption,
                use_workspace=definition.use_workspace,
                enable_todo_management=definition.enable_todo_management,
                agent_type=definition.agent_type,
                available_sub_agents={name: config.description for name, config in definition.sub_agents.items()}
                if definition.sub_agents
                else None,
                available_skills=available_skills,
                workspace=effective_workspace,
            )
            system_prompt = prompt_bundle.prompt
            generated_sections = prompt_bundle.sections
        if generated_sections:
            metadata.setdefault("system_prompt_sections", generated_sections)

        return AgentTask(
            task_id=f"{effective_task_name}_{uuid.uuid4().hex[:8]}",
            model=resolved_model_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_cycles=max(definition.max_cycles, 1),
            memory_compact_threshold=self._to_positive_int(
                definition.memory_compact_threshold,
                default=128_000,
            ),
            memory_threshold_percentage=self._to_percentage_int(
                definition.memory_threshold_percentage,
                default=90,
            ),
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

    def _execute(
        self,
        *,
        prompt: str,
        agent: InteractiveAgentDefinition,
        task_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        log_handler: RuntimeLogHandler | None = None,
        initial_messages: list[Message] | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        sub_task_manager: SubTaskManager | None = None,
        session_id: str | None = None,
        active_handle_callback: Callable[[Any | None], None] | None = None,
        **_: Any,
    ) -> AgentSessionRun:
        definition = self._apply_startup_shell_defaults(agent)
        effective_workspace = self._resolve_workspace(workspace)
        run_name = task_name or "inline"
        backend = definition.backend or self.options.default_backend
        llm_builder = self.options.llm_builder or build_openai_llm_from_local_settings
        llm, resolved = llm_builder(
            self.options.settings_file,
            backend=backend,
            model=definition.model,
            timeout_seconds=self.options.timeout_seconds,
        )
        if self.options.debug_dump_dir:
            cast(_SupportsDebugDumpDir, llm).debug_dump_dir = self.options.debug_dump_dir

        tool_registry_factory = self.options.tool_registry_factory or build_default_registry
        task = self.prepare_task(
            prompt=prompt,
            resolved_model_id=resolved.model_id,
            agent=definition,
            task_name=run_name,
            workspace=effective_workspace,
            session_id=session_id,
        )
        context_providers = [
            *self.options.context_providers,
            *definition.context_providers,
        ]
        memory_providers = [
            *self.options.memory_providers,
            *definition.memory_providers,
        ]
        if context_providers:
            self._apply_context_providers_to_task(
                task=task,
                input=prompt,
                model=str(resolved.model_id or definition.model),
                workspace=effective_workspace,
                context_providers=context_providers,
            )

        sdk_agent = Agent(
            name=run_name,
            instructions=task.system_prompt,
            model=definition.model,
            metadata=dict(task.metadata),
        )

        def model_provider(agent: Agent[Any], run_config: RunConfig) -> tuple[LLMClient, ResolvedModelConfig]:
            del agent, run_config
            return llm, resolved

        run_config = RunConfig(
            model_provider=model_provider,
            workspace=effective_workspace,
            max_cycles=task.max_cycles,
            tool_policy=self.options.tool_policy,
            execution_backend=self.options.execution_backend,
            cancellation_token=cancellation_token,
            approval_provider=self.options.approval_provider,
            approval_timeout_seconds=self.options.approval_timeout_seconds,
            tool_registry_factory=tool_registry_factory,
            runtime_hooks=list(self.options.runtime_hooks),
            log_preview_chars=self.options.log_preview_chars,
            debug_dump_dir=self.options.debug_dump_dir,
            settings_file=self.options.settings_file,
            default_backend=backend,
            llm_builder=llm_builder,
            timeout_seconds=self.options.timeout_seconds,
            context_providers=context_providers,
            memory_providers=memory_providers,
            metadata=dict(task.metadata),
            runtime_task=task,
            shared_state=shared_state,
            initial_messages=initial_messages,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            sub_task_manager=sub_task_manager,
            runtime_log_handler=self._compose_log_handlers(self.options.log_handler, log_handler),
            runtime_stream_callback=self.options.stream_callback,
        )
        handle = Runner.start(sdk_agent, prompt, run_config=run_config)
        if active_handle_callback is not None:
            active_handle_callback(handle)
        try:
            if log_handler is not None:
                for event in handle.events():
                    log_handler(event.type, event.to_dict())
            result = handle.result()
        finally:
            if active_handle_callback is not None:
                active_handle_callback(None)
        return AgentSessionRun(agent_name=run_name, result=result.raw_result, resolved=resolved)

    def _apply_context_providers_to_task(
        self,
        *,
        task: AgentTask,
        input: str,
        model: str,
        workspace: Path,
        context_providers: list[ContextProvider],
    ) -> None:
        request = ContextRequest(
            agent_name=task.task_id.rsplit("_", 1)[0],
            input=input,
            model=model,
            workspace=workspace,
            metadata=dict(task.metadata),
        )
        fragments = [
            ContextFragment(
                id="agent_instructions",
                text=task.system_prompt,
                stable=True,
                priority=0,
                source="agent.instructions",
            )
        ]
        fragments.extend(collect_context_fragments(request, context_providers))
        bundle = assemble_context_fragments(request, fragments)
        task.system_prompt = bundle.prompt
        if bundle.sections:
            task.metadata["system_prompt_sections"] = bundle.metadata_sections()
        if bundle.sources:
            task.metadata["system_prompt_sources"] = bundle.sources
        if bundle.omitted_section_ids:
            task.metadata["system_prompt_omitted_sections"] = list(bundle.omitted_section_ids)
        task.metadata["system_prompt_stable_hash"] = bundle.stable_hash

    def _apply_startup_shell_defaults(self, definition: InteractiveAgentDefinition) -> InteractiveAgentDefinition:
        effective_definition = definition
        if self.options.bash_shell and not effective_definition.bash_shell:
            effective_definition = replace(effective_definition, bash_shell=self.options.bash_shell)
        if self.options.windows_shell_priority and not effective_definition.windows_shell_priority:
            effective_definition = replace(
                effective_definition,
                windows_shell_priority=list(self.options.windows_shell_priority),
            )
        if self.options.bash_env:
            merged_bash_env = dict(self.options.bash_env)
            merged_bash_env.update(effective_definition.bash_env)
            effective_definition = replace(effective_definition, bash_env=merged_bash_env)
        return effective_definition

    def _resolve_workspace(self, workspace: str | Path | None = None) -> Path:
        raw = workspace
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                raw = None
        target = Path(raw) if raw is not None else self.options.workspace
        return Path(target).expanduser().resolve()

    @staticmethod
    def _compose_log_handlers(
        first: RuntimeLogHandler | None,
        second: RuntimeLogHandler | None,
    ) -> RuntimeLogHandler | None:
        handlers = [handler for handler in (first, second) if handler is not None]
        if not handlers:
            return None

        def _handler(event: str, payload: dict[str, Any]) -> None:
            for handler in handlers:
                handler(event, payload)

        return _handler

    @staticmethod
    def _to_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(parsed, 1)

    @staticmethod
    def _to_percentage_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, 100))


def create_agent_session(
    *,
    execute_run: Callable[..., AgentSessionRun],
    session_id: str | None = None,
    agent_name: str,
    definition: InteractiveAgentDefinition,
    workspace: Path,
    shared_state: dict[str, Any] | None = None,
) -> AgentSession:
    return AgentSession(
        execute_run=execute_run,
        session_id=session_id,
        agent_name=agent_name,
        definition=definition,
        workspace=workspace,
        shared_state=shared_state,
    )


__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "AgentSessionRun",
    "AgentSessionState",
    "InteractiveAgentClient",
    "InteractiveAgentDefinition",
    "create_agent_session",
]
