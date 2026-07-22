from __future__ import annotations

import logging
import time
import uuid
import weakref
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Condition, RLock
from typing import Any, Protocol, cast, overload

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalProvider
from vv_agent.config import ResolvedModelConfig, project_resolved_model_limits
from vv_agent.context_providers import (
    ContextFragment,
    ContextProvider,
    ContextRequest,
    assemble_context_fragments,
    collect_context_fragments,
)
from vv_agent.events import DiagnosticEvent, RunEvent
from vv_agent.memory.provider import MemoryProvider
from vv_agent.model import ModelProvider, ModelRef
from vv_agent.prompt import build_raw_system_prompt_sections, build_system_prompt_bundle
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig, ToolPolicy
from vv_agent.runner import Runner
from vv_agent.runtime import CancellationToken
from vv_agent.runtime.backends import ExecutionBackend
from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.engine import register_sub_agent_session, unregister_sub_agent_session
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.runtime.sub_task_manager import SubTaskManager, _SubTaskTurnSnapshot
from vv_agent.sessions import MemorySession, Session
from vv_agent.tools import ToolRegistry, build_default_registry
from vv_agent.types import AgentResult, AgentStatus, AgentTask, Message, NoToolPolicy, SubAgentConfig

RunEventObserver = Callable[[RunEvent], None]
SessionEventHandler = Callable[[str, dict[str, Any]], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentSessionEvent:
    event: str
    payload: dict[str, Any]


class AgentSessionEventGapError(RuntimeError):
    def __init__(self, missed: int) -> None:
        self.missed = max(int(missed), 1)
        super().__init__(f"interactive session event subscriber lagged by {self.missed} event(s)")


class AgentSessionEventStreamClosed(RuntimeError):
    pass


class AgentSessionEventSubscription:
    """Independent bounded pull subscription for interactive session events."""

    __slots__ = ("__weakref__", "_capacity", "_closed", "_condition", "_missed", "_queue")

    def __init__(self, capacity: int) -> None:
        self._capacity = max(int(capacity), 1)
        self._closed = False
        self._condition = Condition()
        self._missed = 0
        self._queue: deque[AgentSessionEvent] = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def recv(self, timeout: float | None = None) -> AgentSessionEvent:
        deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while not self._missed and not self._queue and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no interactive session event available")
                self._condition.wait(remaining)
            return self._recv_locked()

    def try_recv(self) -> AgentSessionEvent:
        with self._condition:
            if not self._missed and not self._queue and not self._closed:
                raise TimeoutError("no interactive session event available")
            return self._recv_locked()

    def close(self) -> None:
        self._close()

    def _publish(self, event: AgentSessionEvent) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._queue) >= self._capacity:
                self._queue.popleft()
                self._missed += 1
            self._queue.append(AgentSessionEvent(event=event.event, payload=dict(event.payload)))
            self._condition.notify_all()

    def _close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()

    def _recv_locked(self) -> AgentSessionEvent:
        if self._missed:
            missed = self._missed
            self._missed = 0
            raise AgentSessionEventGapError(missed)
        if self._queue:
            return self._queue.popleft()
        raise AgentSessionEventStreamClosed("interactive session event stream is closed")


class _SupportsDebugDumpDir(Protocol):
    debug_dump_dir: str | None


@dataclass(slots=True)
class InteractiveAgentDefinition:
    description: str
    model: str
    backend: str | None = None
    language: str = "zh-CN"
    max_cycles: int = 10
    memory_compact_threshold: int = 250_000
    memory_threshold_percentage: int = 90
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    enable_todo_management: bool = True
    agent_type: str | None = None
    native_multimodal: bool = False
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    skill_directories: list[str] = field(default_factory=list)
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    bash_shell: str | None = None
    windows_shell_priority: list[str] = field(default_factory=list)
    bash_env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    context_providers: list[ContextProvider] = field(default_factory=list)
    memory_providers: list[MemoryProvider] = field(default_factory=list)


@dataclass(slots=True)
class AgentSessionOptions:
    model_provider: ModelProvider
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    log_preview_chars: int | None = None
    tool_registry_factory: ToolRegistryFactory | None = None
    stream: RunEventObserver | None = None
    runtime_hooks: list[RuntimeHook] = field(default_factory=list)
    execution_backend: ExecutionBackend | None = None
    cancellation_token: CancellationToken | None = None
    debug_dump_dir: str | None = None
    approval_provider: ApprovalProvider | None = None
    approval_timeout_seconds: float | None = None
    approval_broker: ApprovalBroker | None = None
    tool_policy: ToolPolicy | None = None
    bash_shell: str | None = None
    windows_shell_priority: list[str] = field(default_factory=list)
    bash_env: dict[str, str] = field(default_factory=dict)
    context_providers: list[ContextProvider] = field(default_factory=list)
    memory_providers: list[MemoryProvider] = field(default_factory=list)
    session: Session | None = None
    event_buffer_capacity: int = 256


class AgentSessionRun(RunResult):
    """Interactive run result with durable-session persistence callbacks."""

    __slots__ = ("_after_persist", "_on_persist_failure")

    _after_persist: Callable[[], None] | None
    _on_persist_failure: Callable[[BaseException], None] | None

    def __init__(self, *, agent_name: str, result: AgentResult, resolved: ResolvedModelConfig) -> None:
        super().__init__(
            input="",
            new_items=[],
            final_output=result.final_answer or result.wait_reason or result.error,
            status=result.status,
            raw_result=result,
            token_usage=result.token_usage,
            agent_name=agent_name,
            resolved_model=resolved,
        )
        self._after_persist = None
        self._on_persist_failure = None

    def _set_persistence_callbacks(
        self,
        *,
        after_persist: Callable[[], None],
        on_persist_failure: Callable[[BaseException], None],
    ) -> None:
        self._after_persist = after_persist
        self._on_persist_failure = on_persist_failure

    def _notify_persisted(self) -> None:
        callback = self._after_persist
        self._after_persist = None
        self._on_persist_failure = None
        if callback is not None:
            callback()

    def _notify_persistence_failed(self, error: BaseException) -> None:
        callback = self._on_persist_failure
        self._after_persist = None
        self._on_persist_failure = None
        if callback is not None:
            callback(error)

    @classmethod
    def from_run_result(cls, result: RunResult) -> AgentSessionRun:
        instance = cls(
            agent_name=result.agent_name,
            result=result.raw_result,
            resolved=cast(ResolvedModelConfig, result.resolved_model),
        )
        instance.input = result.input
        instance.new_items = list(result.new_items)
        instance.final_output = result.final_output
        instance.status = result.status
        instance.events = list(result.events)
        instance.token_usage = result.token_usage
        instance.trace_id = result.trace_id
        instance.run_id = result.run_id
        instance.metadata = dict(result.metadata)
        instance._resume_context = result._resume_context
        return instance


@dataclass(slots=True)
class AgentSessionState:
    running: bool
    workspace: Path
    closed: bool = False
    messages: list[Message] = field(default_factory=list)
    shared_state: dict[str, Any] = field(default_factory=dict)
    latest_run: AgentSessionRun | None = None
    active_run_handle: Any | None = None
    pending_steering: int = 0
    pending_follow_ups: int = 0


class AgentSession:
    """Stateful, interactive session wrapper for desktop/runtime integrations."""

    def __init__(
        self,
        *,
        execute_run: Callable[..., AgentSessionRun],
        session_id: str | None = None,
        agent_name: str,
        definition: InteractiveAgentDefinition | None = None,
        agent: Agent | None = None,
        workspace: Path,
        shared_state: dict[str, Any] | None = None,
        session: Session | None = None,
        approval_broker: ApprovalBroker | None = None,
        parent_cancellation_token: CancellationToken | None = None,
        event_buffer_capacity: int = 256,
    ) -> None:
        self._execute_run = execute_run
        self._session = session or MemorySession(str(session_id or "").strip() or uuid.uuid4().hex[:12])
        self._owns_approval_broker = approval_broker is None
        self._approval_broker = approval_broker or ApprovalBroker()
        self._parent_cancellation_token = parent_cancellation_token
        self.session_id = self._resolve_session_id(session_id, self._session)
        if (definition is None) == (agent is None):
            raise ValueError("AgentSession requires exactly one of agent or definition")
        self.agent_name = agent_name
        self.agent = agent
        self.definition = definition
        self._agent_source: Agent | InteractiveAgentDefinition = (
            agent
            if agent is not None
            else cast(
                InteractiveAgentDefinition,
                definition,
            )
        )
        self.workspace = Path(workspace).resolve()
        self._messages = list(self._session.get_items())
        self._shared_state: dict[str, Any] = dict(shared_state or {})
        self._shared_state.setdefault("todo_list", [])
        self._latest_run: AgentSessionRun | None = None
        self._running = False
        self._closed = False
        self._event_buffer_capacity = max(int(event_buffer_capacity), 1)
        self._listeners: list[SessionEventHandler] = []
        self._event_subscriptions: weakref.WeakSet[AgentSessionEventSubscription] = weakref.WeakSet()
        self._background_command_unsubscribers: dict[str, Callable[[], None]] = {}
        self._steering_queue: deque[str] = deque()
        self._follow_up_queue: deque[str] = deque()
        self._active_cancellation_token: CancellationToken | None = None
        self._active_run_handle: Any | None = None
        self._active_approval_broker: ApprovalBroker | None = None
        self._lock = RLock()

    @property
    def messages(self) -> list[Message]:
        with self._lock:
            return list(self._messages)

    @property
    def session(self) -> Session:
        return self._session

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
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_run_handle(self) -> Any | None:
        with self._lock:
            return self._active_run_handle

    @overload
    def subscribe(self, listener: SessionEventHandler) -> Callable[[], None]: ...

    @overload
    def subscribe(
        self,
        listener: None = None,
        *,
        capacity: int | None = None,
    ) -> AgentSessionEventSubscription: ...

    def subscribe(
        self,
        listener: SessionEventHandler | None = None,
        *,
        capacity: int | None = None,
    ) -> Callable[[], None] | AgentSessionEventSubscription:
        if listener is None:
            subscription = AgentSessionEventSubscription(
                self._event_buffer_capacity if capacity is None else capacity,
            )
            with self._lock:
                if self._closed:
                    subscription._close()
                else:
                    self._event_subscriptions.add(subscription)
            return subscription
        if capacity is not None:
            raise ValueError("capacity is only valid for pull event subscriptions")
        with self._lock:
            self._ensure_open_locked()
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe

    def close(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            was_running = self._running
            token = self._active_cancellation_token
            handle = self._active_run_handle
            self._active_run_handle = None
            self._steering_queue.clear()
            self._follow_up_queue.clear()
            unsubscribers = list(self._background_command_unsubscribers.values())
            self._background_command_unsubscribers.clear()
            subscriptions = list(self._event_subscriptions)

        if token is not None:
            token.cancel("interactive session closed")
        if handle is not None:
            try:
                if hasattr(handle, "detach_controller"):
                    handle.detach_controller(self)
                handle.cancel("interactive session closed")
            except Exception:
                logger.exception("Interactive session active handle cleanup failed")
            self._emit("session_active_run_handle_changed", handle=None)
        self._emit("session_closed", aborted=was_running)
        for unsubscribe in unsubscribers:
            try:
                unsubscribe()
            except Exception:
                logger.exception("Interactive session background listener cleanup failed")
        for subscription in subscriptions:
            subscription._close()
        with self._lock:
            self._listeners.clear()
        return True

    def __enter__(self) -> AgentSession:
        with self._lock:
            self._ensure_open_locked()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def steer(self, prompt: str) -> None:
        text = prompt.strip()
        if not text:
            raise ValueError("steer prompt cannot be empty")
        with self._lock:
            self._ensure_open_locked()
            self._steering_queue.append(text)
        self._emit("session_steer_queued", prompt=text)

    def follow_up(self, prompt: str) -> None:
        text = prompt.strip()
        if not text:
            raise ValueError("follow_up prompt cannot be empty")
        with self._lock:
            self._ensure_open_locked()
            self._follow_up_queue.append(text)
        self._emit("session_follow_up_queued", prompt=text)

    def clear_queues(self) -> None:
        with self._lock:
            self._ensure_open_locked()
            self._steering_queue.clear()
            self._follow_up_queue.clear()
        self._emit("session_queues_cleared")

    def cancel(self) -> bool:
        with self._lock:
            if self._closed or not self._running or self._active_cancellation_token is None:
                return False
            self._active_cancellation_token.cancel()
            self._steering_queue.clear()
            self._follow_up_queue.clear()
        self._emit("session_cancel_requested")
        return True

    def approve(self, request_id: str, decision: ApprovalDecision | str) -> None:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("approval request_id cannot be empty")
        with self._lock:
            self._ensure_open_locked()
            approval_broker = self._active_approval_broker or self._approval_broker
        if not approval_broker.resolve(normalized_request_id, decision):
            raise KeyError(f"Unknown approval request: {normalized_request_id}")

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

    def _continue_run_with_snapshot(
        self,
        prompt: str,
        snapshot: _SubTaskTurnSnapshot,
    ) -> AgentSessionRun:
        text = prompt.strip()
        if not text:
            raise ValueError("prompt cannot be empty")
        return self._run_once(text, turn_snapshot=snapshot)

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
                closed=self._closed,
                messages=list(self._messages),
                shared_state=dict(self._shared_state),
                latest_run=self._latest_run,
                active_run_handle=self._active_run_handle,
                pending_steering=len(self._steering_queue),
                pending_follow_ups=len(self._follow_up_queue),
            )

    def replace_messages(self, messages: list[Message]) -> None:
        replacement = list(messages)
        with self._lock:
            self._ensure_open_locked()
            if self._running:
                raise RuntimeError("Cannot replace messages while session is running.")
            self._session.clear()
            if replacement:
                self._session.add_items(replacement)
            self._messages = replacement
        self._emit("session_messages_replaced", message_count=len(replacement))

    def replace_shared_state(self, shared_state: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_open_locked()
            if self._running:
                raise RuntimeError("Cannot replace shared_state while session is running.")
            self._shared_state = dict(shared_state)
            self._shared_state.setdefault("todo_list", [])
        self._emit("session_shared_state_replaced")

    def _run_once(
        self,
        prompt: str,
        *,
        turn_snapshot: _SubTaskTurnSnapshot | None = None,
    ) -> AgentSessionRun:
        with self._lock:
            self._ensure_open_locked()
            if self._running:
                raise RuntimeError("Session is already running. Queue with steer()/follow_up() or wait for completion.")
            self._running = True
            approval_broker = self._approval_broker
            if turn_snapshot is not None:
                current_broker = turn_snapshot.execution_metadata.get("_vv_agent_approval_broker")
                if isinstance(current_broker, ApprovalBroker):
                    approval_broker = current_broker
            if approval_broker is self._approval_broker and self._owns_approval_broker:
                approval_broker.reset_cancelled()
            self._active_approval_broker = approval_broker
            self._active_cancellation_token = self._new_run_cancellation_token()
            existing_message_count = len(self._messages)
            existing_messages = list(self._messages)
            current_shared_state = dict(self._shared_state)

        try:
            self._emit("session_run_start", prompt=prompt, existing_messages=existing_message_count)
            run_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "session_id": self.session_id,
                "agent": self._agent_source,
                "task_name": self.agent_name,
                "workspace": self.workspace,
                "shared_state": current_shared_state,
                "initial_messages": None,
                "before_cycle_messages": self._before_cycle_messages,
                "interruption_messages": self._interruption_messages,
                "event_handler": self._session_event_handler,
                "cancellation_token": self._active_cancellation_token,
                "approval_broker": approval_broker,
                "active_handle_callback": self._set_active_run_handle,
            }
            if turn_snapshot is not None:
                run_kwargs["_sub_task_turn_snapshot"] = turn_snapshot
            run_kwargs["session"] = self._session
            run = self._execute_run(**run_kwargs)
        finally:
            self._set_active_run_handle(None)
            with self._lock:
                self._running = False
                self._active_approval_broker = None
                self._active_cancellation_token = None

        try:
            self._persist_custom_run_delta(existing_messages, run)
        except BaseException as exc:
            try:
                run._notify_persistence_failed(exc)
            except BaseException:
                logger.exception("Agent session persistence failure callback failed")
            raise
        run._notify_persisted()
        with self._lock:
            self._messages = list(self._session.get_items())
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

    def _new_run_cancellation_token(self) -> CancellationToken:
        if self._parent_cancellation_token is not None:
            return self._parent_cancellation_token.child()
        return CancellationToken()

    def _persist_custom_run_delta(self, previous: list[Message], run: AgentSessionRun) -> None:
        persisted = list(self._session.get_items())
        if persisted != previous:
            return

        produced = list(run.result.messages)
        if produced and produced[: len(previous)] != previous:
            self._session.clear()
            self._session.add_items(produced)
            return

        delta = produced[len(previous) :] if produced else list(run.new_items)
        if delta:
            self._session.add_items(delta)

    @staticmethod
    def _resolve_session_id(requested_id: str | None, session: Session | None) -> str:
        normalized_requested_id = str(requested_id or "").strip()
        if session is None:
            return normalized_requested_id or uuid.uuid4().hex[:12]

        actual_id = str(session.session_id or "").strip()
        if not actual_id:
            raise ValueError("Session session_id cannot be empty.")
        if normalized_requested_id and normalized_requested_id != actual_id:
            raise ValueError(
                f"Requested session_id {normalized_requested_id!r} does not match backing Session session_id {actual_id!r}."
            )
        return normalized_requested_id or actual_id

    def _set_active_run_handle(self, handle: Any | None) -> None:
        with self._lock:
            rejected_handle = handle if self._closed and handle is not None else None
            if rejected_handle is not None:
                handle = None
            if self._active_run_handle is handle:
                previous_handle = None
                changed = False
            else:
                previous_handle = self._active_run_handle
                self._active_run_handle = handle
                changed = True
        if rejected_handle is not None:
            rejected_handle.cancel("interactive session closed")
        if not changed:
            return
        if previous_handle is not None and hasattr(previous_handle, "detach_controller"):
            previous_handle.detach_controller(self)
        if handle is not None and hasattr(handle, "attach_controller"):
            handle.attach_controller(self)
        self._emit("session_active_run_handle_changed", handle=handle)

    def _drain_next_queued_prompt(self) -> str | None:
        with self._lock:
            self._ensure_open_locked()
            if self._steering_queue:
                return self._steering_queue.popleft()
            if self._follow_up_queue:
                return self._follow_up_queue.popleft()
        return None

    def _before_cycle_messages(self, cycle_index: int, _: list[Message], __: dict[str, Any]) -> list[Message]:
        del _, __
        with self._lock:
            prompts = list(self._steering_queue)
            self._steering_queue.clear()
        for prompt in prompts:
            self._emit("session_steer_dequeued", cycle=cycle_index, prompt=prompt)
        return [Message(role="user", content=prompt) for prompt in prompts]

    def _interruption_messages(self) -> list[Message]:
        with self._lock:
            prompts = list(self._steering_queue)
            self._steering_queue.clear()
        for prompt in prompts:
            self._emit("session_steer_interrupt", prompt=prompt)
        return [Message(role="user", content=prompt) for prompt in prompts]

    def _session_event_handler(self, event: RunEvent) -> None:
        self._sync_background_command_watchers(event)
        self._emit(event.type, **event.to_dict())

    def _sync_background_command_watchers(self, event: RunEvent) -> None:
        if not isinstance(event, DiagnosticEvent) or event.code != "tool_result":
            return
        payload = event.details
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
            if self._closed:
                return
            if normalized_session_id in self._background_command_unsubscribers:
                return
            self._background_command_unsubscribers[normalized_session_id] = lambda: None

        weak_session = weakref.ref(self)

        def _listener(payload: dict[str, Any]) -> None:
            session = weak_session()
            if session is not None:
                session._handle_background_command_terminal(normalized_session_id, payload)

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
        with self._lock:
            if self._closed:
                return
            running = self._running
            if running:
                self._steering_queue.append(notification_message)
        if running:
            self._emit("session_steer_queued", prompt=notification_message)

        event_payload = dict(payload)
        event_payload["session_id"] = background_session_id
        event_payload["background_session_id"] = background_session_id
        event_payload["notification_message"] = notification_message
        event_payload["queued_to_session"] = running
        event_payload["queued_to_running_session"] = running

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
            subscriptions = list(self._event_subscriptions)
        session_event = AgentSessionEvent(event=event, payload=dict(payload))
        for subscription in subscriptions:
            subscription._publish(session_event)
        for listener in listeners:
            try:
                listener(event, payload)
            except Exception:
                logger.exception("Agent session listener failed for event %s", event)

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("Interactive session is closed.")


class InteractiveAgentClient:
    """Session client backed by vv-agent runtime primitives."""

    def __init__(self, *, options: AgentSessionOptions) -> None:
        self.options = options

    def create_session(
        self,
        *,
        agent: Agent | InteractiveAgentDefinition,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        session: Session | None = None,
    ) -> AgentSession:
        definition = self._apply_startup_shell_defaults(agent) if isinstance(agent, InteractiveAgentDefinition) else None
        sdk_agent = agent if isinstance(agent, Agent) else None
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
            agent_name=sdk_agent.name if sdk_agent is not None else "inline",
            definition=definition,
            agent=sdk_agent,
            workspace=effective_workspace,
            shared_state=shared_state,
            session=session if session is not None else self.options.session,
            approval_broker=self.options.approval_broker,
            parent_cancellation_token=self.options.cancellation_token,
            event_buffer_capacity=self.options.event_buffer_capacity,
        )

    def prepare_task(
        self,
        *,
        prompt: str,
        resolved_model_id: str,
        resolved_context_length: int | None = None,
        resolved_max_output_tokens: int | None = None,
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
        project_resolved_model_limits(
            metadata,
            context_length=resolved_context_length,
            max_output_tokens=resolved_max_output_tokens,
        )

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
            memory_compact_threshold=self._to_non_negative_int(
                definition.memory_compact_threshold,
                default=250_000,
            ),
            memory_threshold_percentage=self._to_percentage_int(
                definition.memory_threshold_percentage,
                default=90,
            ),
            no_tool_policy=definition.no_tool_policy,
            allow_interruption=definition.allow_interruption,
            use_workspace=definition.use_workspace,
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
        agent: Agent | InteractiveAgentDefinition,
        task_name: str | None = None,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        event_handler: RunEventObserver | None = None,
        initial_messages: list[Message] | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        sub_task_manager: SubTaskManager | None = None,
        session_id: str | None = None,
        session: Session | None = None,
        approval_broker: ApprovalBroker | None = None,
        active_handle_callback: Callable[[Any | None], None] | None = None,
        **_: Any,
    ) -> AgentSessionRun:
        if isinstance(agent, Agent):
            return self._execute_agent(
                prompt=prompt,
                agent=agent,
                workspace=workspace,
                shared_state=shared_state,
                event_handler=event_handler,
                initial_messages=initial_messages,
                before_cycle_messages=before_cycle_messages,
                interruption_messages=interruption_messages,
                cancellation_token=cancellation_token,
                sub_task_manager=sub_task_manager,
                session_id=session_id,
                session=session,
                approval_broker=approval_broker,
                active_handle_callback=active_handle_callback,
            )
        definition = self._apply_startup_shell_defaults(agent)
        effective_workspace = self._resolve_workspace(workspace)
        run_name = task_name or "inline"
        backend = str(definition.backend or "").strip()
        model_ref = ModelRef.backend(backend, definition.model) if backend else ModelRef.named(definition.model)
        resolved = self.options.model_provider.resolve(model_ref)
        llm = self.options.model_provider.client(resolved)
        if self.options.debug_dump_dir:
            cast(_SupportsDebugDumpDir, llm).debug_dump_dir = self.options.debug_dump_dir

        tool_registry_factory = self.options.tool_registry_factory or build_default_registry
        task = self.prepare_task(
            prompt=prompt,
            resolved_model_id=resolved.model_id,
            resolved_context_length=resolved.context_length,
            resolved_max_output_tokens=resolved.max_output_tokens,
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

        run_config = RunConfig(
            model=model_ref,
            model_provider=self.options.model_provider,
            workspace=effective_workspace,
            session=session,
            max_cycles=task.max_cycles,
            tool_policy=self.options.tool_policy,
            execution_backend=self.options.execution_backend,
            cancellation_token=cancellation_token,
            approval_provider=self.options.approval_provider,
            approval_timeout_seconds=self.options.approval_timeout_seconds,
            approval_broker=approval_broker,
            tool_registry_factory=tool_registry_factory,
            hooks=list(self.options.runtime_hooks),
            log_preview_chars=self.options.log_preview_chars,
            debug_dump_dir=self.options.debug_dump_dir,
            context_providers=context_providers,
            memory_providers=memory_providers,
            metadata=dict(task.metadata),
            shared_state=shared_state,
            initial_messages=initial_messages,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            sub_task_manager=sub_task_manager,
            stream=self._compose_event_handlers(self.options.stream, event_handler),
        )
        handle = Runner._start_compiled(sdk_agent, prompt, task=task, run_config=run_config)
        if active_handle_callback is not None:
            active_handle_callback(handle)
        try:
            result = handle.result()
        finally:
            if active_handle_callback is not None:
                active_handle_callback(None)
        return AgentSessionRun.from_run_result(result)

    def _execute_agent(
        self,
        *,
        prompt: str,
        agent: Agent,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        event_handler: RunEventObserver | None = None,
        initial_messages: list[Message] | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        sub_task_manager: SubTaskManager | None = None,
        session_id: str | None = None,
        session: Session | None = None,
        approval_broker: ApprovalBroker | None = None,
        active_handle_callback: Callable[[Any | None], None] | None = None,
    ) -> AgentSessionRun:
        effective_workspace = self._resolve_workspace(workspace)
        run_config = RunConfig(
            model_provider=self.options.model_provider,
            workspace=effective_workspace,
            session=session,
            tool_policy=self.options.tool_policy,
            execution_backend=self.options.execution_backend,
            cancellation_token=cancellation_token,
            approval_provider=self.options.approval_provider,
            approval_timeout_seconds=self.options.approval_timeout_seconds,
            approval_broker=approval_broker,
            tool_registry_factory=self.options.tool_registry_factory or build_default_registry,
            hooks=list(self.options.runtime_hooks),
            log_preview_chars=self.options.log_preview_chars,
            debug_dump_dir=self.options.debug_dump_dir,
            context_providers=list(self.options.context_providers),
            memory_providers=list(self.options.memory_providers),
            metadata={"session_id": session_id} if session_id else {},
            shared_state=shared_state,
            initial_messages=initial_messages,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            sub_task_manager=sub_task_manager,
            stream=self._compose_event_handlers(self.options.stream, event_handler),
        )
        handle = Runner.start(agent, prompt, run_config=run_config)
        if active_handle_callback is not None:
            active_handle_callback(handle)
        try:
            result = handle.result()
        finally:
            if active_handle_callback is not None:
                active_handle_callback(None)
        return AgentSessionRun.from_run_result(result)

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
    def _compose_event_handlers(
        first: RunEventObserver | None,
        second: RunEventObserver | None,
    ) -> RunEventObserver | None:
        handlers = [handler for handler in (first, second) if handler is not None]
        if not handlers:
            return None

        def _handler(event: RunEvent) -> None:
            for handler in handlers:
                try:
                    handler(event)
                except BaseException:
                    logger.exception("Interactive run event observer failed")

        return _handler

    @staticmethod
    def _to_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(parsed, 1)

    @staticmethod
    def _to_non_negative_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(parsed, 0)

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
    session: Session | None = None,
    approval_broker: ApprovalBroker | None = None,
    event_buffer_capacity: int = 256,
) -> AgentSession:
    return AgentSession(
        execute_run=execute_run,
        session_id=session_id,
        agent_name=agent_name,
        definition=definition,
        workspace=workspace,
        shared_state=shared_state,
        session=session,
        approval_broker=approval_broker,
        event_buffer_capacity=event_buffer_capacity,
    )


__all__ = [
    "AgentSession",
    "AgentSessionEvent",
    "AgentSessionEventGapError",
    "AgentSessionEventStreamClosed",
    "AgentSessionEventSubscription",
    "AgentSessionOptions",
    "AgentSessionRun",
    "AgentSessionState",
    "InteractiveAgentClient",
    "InteractiveAgentDefinition",
    "create_agent_session",
]
