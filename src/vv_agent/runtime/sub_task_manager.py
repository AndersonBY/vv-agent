from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.memory import sanitize_for_resume
from vv_agent.types import AgentStatus, Message, SubTaskOutcome

if TYPE_CHECKING:
    from vv_agent.interactive import AgentSessionRun
    from vv_agent.workspace.base import WorkspaceBackend

logger = logging.getLogger(__name__)

SessionRegistrar = Callable[[str, Any], None]
SessionUnregistrar = Callable[[str, Any | None], None]
SubTaskRunnerCallable = Callable[[], SubTaskOutcome]
SessionEventForwarder = Callable[[str, dict[str, Any]], None]

_TURN_LOG_HANDLER_METADATA_KEY = "_vv_agent_runtime_log_handler"
_TURN_EXECUTION_METADATA_ALLOWLIST = frozenset(
    {
        "_vv_agent_approval_provider",
        "_vv_agent_approval_broker",
        "_vv_agent_approval_timeout_seconds",
        "_vv_agent_memory_providers",
        "_vv_agent_model_provider",
        "_vv_agent_model_settings",
        "_vv_agent_trace_context",
        "trace_context",
    }
)


@dataclass(frozen=True, slots=True)
class _SubTaskTurnSnapshot:
    cancellation_token: Any | None = None
    stream_callback: Callable[[dict[str, Any]], None] | None = None
    event_sink: Callable[[Any], None] | None = None
    parent_log_handler: SessionEventForwarder | None = None
    state_store: Any | None = None
    run_context: Any | None = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] | None = None
    can_use_tool: Callable[[str, dict[str, Any]], bool] | None = None
    approval: str | None = None

    def tool_policy_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.allowed_tools is not None:
            metadata["_vv_agent_allowed_tools"] = list(self.allowed_tools)
        if self.disallowed_tools is not None:
            metadata["_vv_agent_disallowed_tools"] = list(self.disallowed_tools)
        if self.can_use_tool is not None:
            metadata["_vv_agent_tool_policy_can_use_tool"] = self.can_use_tool
        if self.approval is not None:
            metadata["_vv_agent_tool_policy_approval"] = self.approval
        return metadata


class SubTaskSession(Protocol):
    @property
    def messages(self) -> list[Message]: ...
    def replace_messages(self, messages: list[Message]) -> None: ...
    def subscribe(self, listener: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]: ...
    def continue_run(self, prompt: str) -> Any: ...


class ThreadHandle(Protocol):
    def is_alive(self) -> bool: ...
    def join(self, timeout: float | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class _ContinuationAdmissionState:
    task_title: str | None
    outcome: SubTaskOutcome | None
    recent_activity: str | None
    parent_run_id: str | None
    parent_tool_call_id: str | None
    thread: ThreadHandle | None
    active: bool
    execution_token: object | None
    updated_at: str | None
    use_turn_event_handler: bool
    turn_event_handler: SessionEventForwarder | None


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _is_blank_contract_string(value: str | None) -> bool:
    return not value or not value.strip().strip("\x1c\x1d\x1e\x1f")


def _preview_text(value: Any, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _message_snapshot(message: Any) -> Any:
    serializer = getattr(message, "to_dict", None)
    if callable(serializer):
        return serializer()
    return message


@dataclass(slots=True)
class ManagedSubTask:
    task_id: str
    session_id: str
    agent_name: str
    task_title: str | None = None
    workspace_backend: WorkspaceBackend | None = None
    session: SubTaskSession | None = None
    outcome: SubTaskOutcome | None = None
    resolved: dict[str, str] = field(default_factory=dict)
    current_cycle_index: int | None = None
    recent_activity: str | None = None
    latest_cycle: dict[str, Any] | None = None
    latest_tool_call: dict[str, Any] | None = None
    updated_at: str | None = None
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    initial_parent_run_id: str | None = None
    initial_parent_tool_call_id: str | None = None
    thread: ThreadHandle | None = None
    manager_listener_attached: bool = False
    forward_listener_attached: bool = False
    session_generation: int = 0
    manager_listener_generation: int | None = None
    forward_listener_generation: int | None = None
    initial_event_forwarder: SessionEventForwarder | None = None
    turn_event_handler: SessionEventForwarder | None = None
    use_turn_event_handler: bool = False
    active: bool = False
    execution_token: object | None = None
    generation_token: object = field(default_factory=object, repr=False)

    def is_running(self) -> bool:
        return self.active or (self.thread is not None and self.thread.is_alive())


class SubTaskManager:
    def __init__(
        self,
        *,
        register_session: SessionRegistrar,
        unregister_session: SessionUnregistrar,
    ) -> None:
        self._register_session = register_session
        self._unregister_session = unregister_session
        self._tasks: dict[str, ManagedSubTask] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_name: str,
        task_title: str,
        workspace_backend: WorkspaceBackend,
        runner: SubTaskRunnerCallable,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
    ) -> ManagedSubTask:
        with self._lock:
            previous_record = self._tasks.get(task_id)
            if previous_record is not None and previous_record.is_running():
                raise RuntimeError(f"Sub-task {task_id} is already running.")
            record = ManagedSubTask(
                task_id=task_id,
                session_id=session_id,
                agent_name=agent_name,
                task_title=task_title or None,
                workspace_backend=workspace_backend,
                parent_run_id=parent_run_id,
                parent_tool_call_id=parent_tool_call_id,
                active=True,
                updated_at=_now_iso(),
            )
            self._tasks[task_id] = record
            thread = threading.Thread(
                target=self._run_and_capture,
                kwargs={"task_id": task_id, "runner": runner},
                daemon=True,
                name=f"vv-agent-subtask-{task_id[:12]}",
            )
            record.thread = thread

        try:
            thread.start()
        except BaseException:
            with self._lock:
                if self._tasks.get(task_id) is record and record.thread is thread:
                    if previous_record is None:
                        self._tasks.pop(task_id, None)
                    else:
                        self._tasks[task_id] = previous_record
            raise
        return record

    def _begin_execution(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_name: str,
        task_title: str,
        workspace_backend: WorkspaceBackend,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
    ) -> object | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None and record.is_running():
                if record.thread is threading.current_thread():
                    return None
                raise RuntimeError(f"Sub-task {task_id} is already running.")
            if record is None:
                record = ManagedSubTask(
                    task_id=task_id,
                    session_id=session_id,
                    agent_name=agent_name,
                )
                self._tasks[task_id] = record

            execution_token = object()
            record.session_id = session_id
            record.agent_name = agent_name
            record.task_title = task_title or record.task_title
            record.workspace_backend = workspace_backend
            record.parent_run_id = parent_run_id
            record.parent_tool_call_id = parent_tool_call_id
            record.outcome = None
            record.thread = None
            record.active = True
            record.execution_token = execution_token
            record.updated_at = _now_iso()
            return execution_token

    def attach_session(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_name: str,
        task_title: str,
        workspace_backend: WorkspaceBackend,
        session: SubTaskSession,
        resolved: dict[str, str] | None = None,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        event_forwarder: SessionEventForwarder | None = None,
    ) -> ManagedSubTask:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                record = ManagedSubTask(
                    task_id=task_id,
                    session_id=session_id,
                    agent_name=agent_name,
                )
                self._tasks[task_id] = record

            first_session_attachment = record.session is None
            if record.session is not session:
                record.session_generation += 1
                record.manager_listener_attached = False
                record.forward_listener_attached = False
            elif record.session_generation == 0:
                record.session_generation = 1
            session_generation = record.session_generation

            record.session_id = session_id
            record.agent_name = agent_name
            record.task_title = task_title or record.task_title
            record.workspace_backend = workspace_backend
            record.session = session
            if first_session_attachment:
                record.initial_parent_run_id = parent_run_id or record.parent_run_id
                record.initial_parent_tool_call_id = parent_tool_call_id or record.parent_tool_call_id
            if parent_run_id is not None:
                record.parent_run_id = parent_run_id
            if parent_tool_call_id is not None:
                record.parent_tool_call_id = parent_tool_call_id
            if resolved:
                record.resolved = dict(resolved)
            record.updated_at = _now_iso()

            if record.manager_listener_generation != session_generation:
                manager_ref = weakref.ref(self)

                def manager_listener(
                    event: str,
                    payload: dict[str, Any],
                    *,
                    _task_id: str = task_id,
                    _session_generation: int = session_generation,
                    _generation_token: object = record.generation_token,
                ) -> None:
                    manager = manager_ref()
                    if manager is not None:
                        manager._handle_session_event(
                            task_id=_task_id,
                            session_generation=_session_generation,
                            generation_token=_generation_token,
                            event=event,
                            payload=payload,
                        )

                session.subscribe(
                    manager_listener
                )
                record.manager_listener_attached = True
                record.manager_listener_generation = session_generation

            if event_forwarder is not None:
                record.initial_event_forwarder = event_forwarder
                record.turn_event_handler = None
                record.use_turn_event_handler = False

            if record.forward_listener_generation != session_generation:
                manager_ref = weakref.ref(self)

                def forward_listener(
                    event: str,
                    payload: dict[str, Any],
                    *,
                    _task_id: str = task_id,
                    _session_generation: int = session_generation,
                    _generation_token: object = record.generation_token,
                ) -> None:
                    manager = manager_ref()
                    if manager is not None:
                        manager._forward_session_event(
                            task_id=_task_id,
                            session_generation=_session_generation,
                            generation_token=_generation_token,
                            event=event,
                            payload=payload,
                        )

                session.subscribe(
                    forward_listener
                )
                record.forward_listener_attached = True
                record.forward_listener_generation = session_generation

            return record

    def record_outcome(
        self,
        task_id: str,
        outcome: SubTaskOutcome,
        *,
        workspace_backend: WorkspaceBackend | None = None,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        execution_token: object | None = None,
    ) -> ManagedSubTask:
        if outcome.status == AgentStatus.FAILED and _is_blank_contract_string(outcome.error_code):
            outcome = replace(outcome, error_code="sub_task_failed")

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                record = ManagedSubTask(
                    task_id=task_id,
                    session_id=outcome.session_id or "",
                    agent_name=outcome.agent_name,
                )
                self._tasks[task_id] = record

            if workspace_backend is not None:
                record.workspace_backend = workspace_backend
            if parent_run_id is not None:
                record.parent_run_id = parent_run_id
            if parent_tool_call_id is not None:
                record.parent_tool_call_id = parent_tool_call_id
            record.session_id = outcome.session_id or record.session_id
            record.agent_name = outcome.agent_name
            if outcome.resolved:
                record.resolved = dict(outcome.resolved)
            record.updated_at = _now_iso()
            if execution_token is not None:
                if record.execution_token is not execution_token:
                    return record
                record.active = False
                record.execution_token = None
                self._apply_outcome(record, outcome)
            elif not record.is_running():
                self._apply_outcome(record, outcome)

            return record

    def continue_task(self, *, task_id: str, prompt: str) -> ManagedSubTask:
        return self._continue_task(task_id=task_id, prompt=prompt, snapshot=None)

    def _continue_task_with_context(
        self,
        *,
        task_id: str,
        prompt: str,
        context: Any,
    ) -> ManagedSubTask:
        return self._continue_task(
            task_id=task_id,
            prompt=prompt,
            snapshot=self._capture_turn_snapshot(context),
        )

    def _continue_task(
        self,
        *,
        task_id: str,
        prompt: str,
        snapshot: _SubTaskTurnSnapshot | None,
    ) -> ManagedSubTask:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise ValueError("Follow-up prompt cannot be empty.")

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(f"Sub-task {task_id} not found.")
            if record.is_running():
                raise RuntimeError(f"Sub-task {task_id} is already running.")
            if record.session is None:
                raise RuntimeError(f"Sub-task {task_id} session is not attached.")
            if record.outcome is not None and record.outcome.status == AgentStatus.MAX_CYCLES:
                raise RuntimeError(f"Sub-task {task_id} reached max cycles and cannot continue.")

            original_messages = list(record.session.messages)
            previous = self._admit_continuation(record=record, prompt=prompt_text, snapshot=snapshot)
            try:
                removed_messages = self._replace_session_messages_if_needed(record.session, original_messages)
                thread = threading.Thread(
                    target=self._continue_existing_session,
                    kwargs={"task_id": task_id, "prompt": prompt_text, "snapshot": snapshot},
                    daemon=True,
                    name=f"vv-agent-subtask-{task_id[:12]}-continue",
                )
                record.thread = thread
            except BaseException:
                self._rollback_failed_admission(
                    record=record,
                    previous=previous,
                    messages=original_messages,
                    task_id=task_id,
                )
                raise

            if removed_messages > 0:
                logger.info(f"Sanitized {removed_messages} stale message(s) before resuming sub-task {task_id}")

        try:
            thread.start()
        except BaseException:
            with self._lock:
                if record.thread is thread:
                    self._rollback_failed_admission(
                        record=record,
                        previous=previous,
                        messages=original_messages,
                        task_id=task_id,
                    )
            raise
        return record

    def get(self, task_id: str) -> ManagedSubTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def wait(self, task_id: str, timeout: float | None = None) -> ManagedSubTask | None:
        record = self.get(task_id)
        if record is None:
            return None
        thread = record.thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.get(task_id)

    def _run_and_capture(self, *, task_id: str, runner: SubTaskRunnerCallable) -> None:
        try:
            outcome = runner()
        except BaseException as exc:
            with self._lock:
                record = self._tasks.get(task_id)
                if record is None:
                    return
                outcome = SubTaskOutcome(
                    task_id=record.task_id,
                    session_id=record.session_id,
                    agent_name=record.agent_name,
                    status=AgentStatus.FAILED,
                    error=str(exc),
                    error_code="sub_task_failed",
                    resolved=dict(record.resolved),
                )
        self._finish_active_worker(task_id=task_id, outcome=outcome)

    def _continue_existing_session(
        self,
        *,
        task_id: str,
        prompt: str,
        snapshot: _SubTaskTurnSnapshot | None,
    ) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            if record.session is None:
                record.active = False
                return
            session = record.session
            session_id = record.session_id

        outcome: SubTaskOutcome | None = None
        try:
            try:
                self._register_session(session_id, session)
            except BaseException as exc:
                outcome = self._failed_continuation_outcome(task_id=task_id, error=exc)
            else:
                try:
                    continue_with_snapshot = getattr(session, "_continue_run_with_snapshot", None)
                    if snapshot is not None and callable(continue_with_snapshot):
                        run = continue_with_snapshot(prompt, snapshot)
                    else:
                        run = session.continue_run(prompt)
                    outcome = self._build_outcome_from_run(task_id=task_id, run=run)
                except BaseException as exc:
                    outcome = self._failed_continuation_outcome(task_id=task_id, error=exc)
        finally:
            try:
                self._unregister_session(session_id, session)
            except BaseException as exc:
                outcome = self._failed_continuation_outcome(task_id=task_id, error=exc)
            self._finish_active_worker(task_id=task_id, outcome=outcome)

    def _finish_active_worker(self, *, task_id: str, outcome: SubTaskOutcome | None) -> None:
        current_thread = threading.current_thread()
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None and record.thread is current_thread:
                record.active = False
                record.execution_token = None
                if outcome is not None:
                    if outcome.status == AgentStatus.FAILED and _is_blank_contract_string(outcome.error_code):
                        outcome = replace(outcome, error_code="sub_task_failed")
                    self._apply_outcome(record, outcome)

    @classmethod
    def _apply_outcome(cls, record: ManagedSubTask, outcome: SubTaskOutcome) -> None:
        record.session_id = outcome.session_id or record.session_id
        record.agent_name = outcome.agent_name
        record.outcome = outcome
        if outcome.resolved:
            record.resolved = dict(outcome.resolved)
        if outcome.cycles > 0:
            record.current_cycle_index = outcome.cycles
        record.updated_at = _now_iso()

        latest_cycle = dict(record.latest_cycle or {})
        latest_cycle["status"] = outcome.status.value
        if outcome.cycles > 0:
            latest_cycle["cycle_index"] = outcome.cycles
        record.latest_cycle = latest_cycle

        summary_text = (
            _preview_text(outcome.final_answer) or _preview_text(outcome.wait_reason) or _preview_text(outcome.error)
        )
        if summary_text:
            record.recent_activity = summary_text

        if record.session is not None:
            cls._release_event_forwarders(record)

    @staticmethod
    def _admit_continuation(
        *,
        record: ManagedSubTask,
        prompt: str,
        snapshot: _SubTaskTurnSnapshot | None,
    ) -> _ContinuationAdmissionState:
        previous = _ContinuationAdmissionState(
            task_title=record.task_title,
            outcome=record.outcome,
            recent_activity=record.recent_activity,
            parent_run_id=record.parent_run_id,
            parent_tool_call_id=record.parent_tool_call_id,
            thread=record.thread,
            active=record.active,
            execution_token=record.execution_token,
            updated_at=record.updated_at,
            use_turn_event_handler=record.use_turn_event_handler,
            turn_event_handler=record.turn_event_handler,
        )
        record.task_title = prompt
        record.outcome = None
        record.recent_activity = prompt
        record.active = True
        record.execution_token = None
        if snapshot is not None:
            record.parent_run_id = snapshot.parent_run_id
            record.parent_tool_call_id = snapshot.parent_tool_call_id
            record.initial_event_forwarder = None
            record.turn_event_handler = snapshot.parent_log_handler
            record.use_turn_event_handler = True
        else:
            record.parent_run_id = record.initial_parent_run_id
            record.parent_tool_call_id = record.initial_parent_tool_call_id
            record.turn_event_handler = None
            record.use_turn_event_handler = True
        record.updated_at = _now_iso()
        return previous

    @staticmethod
    def _rollback_continuation(
        *,
        record: ManagedSubTask,
        previous: _ContinuationAdmissionState,
    ) -> None:
        record.task_title = previous.task_title
        record.outcome = previous.outcome
        record.recent_activity = previous.recent_activity
        record.parent_run_id = previous.parent_run_id
        record.parent_tool_call_id = previous.parent_tool_call_id
        record.thread = previous.thread
        record.active = previous.active
        record.execution_token = previous.execution_token
        record.updated_at = previous.updated_at
        record.use_turn_event_handler = previous.use_turn_event_handler
        record.turn_event_handler = previous.turn_event_handler

    @staticmethod
    def _release_event_forwarders(record: ManagedSubTask) -> None:
        record.initial_event_forwarder = None
        record.turn_event_handler = None
        record.use_turn_event_handler = True

    @classmethod
    def _rollback_failed_admission(
        cls,
        *,
        record: ManagedSubTask,
        previous: _ContinuationAdmissionState,
        messages: list[Message],
        task_id: str,
    ) -> None:
        cls._rollback_continuation(record=record, previous=previous)
        if record.session is not None:
            cls._restore_session_messages(record.session, messages, task_id=task_id)
        # Standard AgentSession emits during replace_messages(), so restore record fields last too.
        cls._rollback_continuation(record=record, previous=previous)

    @staticmethod
    def _restore_session_messages(session: SubTaskSession, messages: list[Message], *, task_id: str) -> None:
        try:
            if [_message_snapshot(message) for message in session.messages] != [
                _message_snapshot(message) for message in messages
            ]:
                session.replace_messages(messages)
        except Exception:
            logger.exception("Failed to restore session messages after continuation admission failure for %s", task_id)

    def _failed_continuation_outcome(self, *, task_id: str, error: BaseException) -> SubTaskOutcome | None:
        with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                return None
            return SubTaskOutcome(
                task_id=current.task_id,
                session_id=current.session_id,
                agent_name=current.agent_name,
                status=AgentStatus.FAILED,
                error=str(error),
                error_code="sub_task_failed",
                resolved=dict(current.resolved),
            )

    @staticmethod
    def _capture_turn_snapshot(context: Any) -> _SubTaskTurnSnapshot:
        execution_context = getattr(context, "ctx", None)
        runtime_metadata = getattr(execution_context, "metadata", None)
        if not isinstance(runtime_metadata, dict):
            runtime_metadata = {}
        run_context = getattr(context, "run_context", None)
        run_context_metadata = getattr(run_context, "metadata", None)
        if not isinstance(run_context_metadata, dict):
            run_context_metadata = {}
        task_metadata = getattr(context, "task_metadata", None)
        if not isinstance(task_metadata, dict):
            task_metadata = {}
        tool_context_metadata = getattr(context, "metadata", None)
        if not isinstance(tool_context_metadata, dict):
            tool_context_metadata = {}

        def normalized_string(*values: Any) -> str | None:
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        allowed = runtime_metadata.get(
            "_vv_agent_allowed_tools",
            task_metadata.get("_vv_agent_allowed_tools"),
        )
        disallowed = runtime_metadata.get(
            "_vv_agent_disallowed_tools",
            task_metadata.get("_vv_agent_disallowed_tools"),
        )
        can_use_tool = runtime_metadata.get("_vv_agent_tool_policy_can_use_tool")
        approval = runtime_metadata.get("_vv_agent_tool_policy_approval")
        event_sink = runtime_metadata.get("_vv_agent_emit_event")
        parent_log_handler = tool_context_metadata.get(_TURN_LOG_HANDLER_METADATA_KEY)
        return _SubTaskTurnSnapshot(
            cancellation_token=getattr(execution_context, "cancellation_token", None),
            stream_callback=getattr(execution_context, "stream_callback", None),
            event_sink=event_sink if callable(event_sink) else None,
            parent_log_handler=parent_log_handler if callable(parent_log_handler) else None,
            state_store=getattr(execution_context, "state_store", None),
            run_context=run_context,
            execution_metadata={
                key: runtime_metadata[key]
                for key in _TURN_EXECUTION_METADATA_ALLOWLIST
                if key in runtime_metadata
            },
            trace_id=normalized_string(
                runtime_metadata.get("_vv_agent_trace_id"),
                runtime_metadata.get("trace_id"),
                run_context_metadata.get("_vv_agent_trace_id"),
                run_context_metadata.get("trace_id"),
                task_metadata.get("_vv_agent_trace_id"),
                task_metadata.get("trace_id"),
            ),
            parent_run_id=normalized_string(
                getattr(run_context, "run_id", None),
                runtime_metadata.get("_vv_agent_run_id"),
            ),
            parent_tool_call_id=normalized_string(getattr(context, "tool_call_id", None)),
            allowed_tools=(
                tuple(item for item in allowed if isinstance(item, str))
                if isinstance(allowed, list)
                else None
            ),
            disallowed_tools=(
                tuple(item for item in disallowed if isinstance(item, str))
                if isinstance(disallowed, list)
                else None
            ),
            can_use_tool=can_use_tool if callable(can_use_tool) else None,
            approval=approval if approval in {"always", "never", "on_request"} else None,
        )

    def _forward_session_event(
        self,
        *,
        task_id: str,
        session_generation: int,
        generation_token: object,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if (
                record is None
                or record.generation_token is not generation_token
                or record.session_generation != session_generation
            ):
                return
            if record.use_turn_event_handler:
                handler = record.turn_event_handler
                if handler is None:
                    return
                forwarded_event = f"sub_agent_{event}"
                forwarded_payload = dict(payload)
                forwarded_payload.setdefault("task_id", record.task_id)
                forwarded_payload.setdefault("session_id", record.session_id)
                forwarded_payload.setdefault("sub_agent_name", record.agent_name)
                if record.parent_run_id:
                    forwarded_payload.setdefault("parent_run_id", record.parent_run_id)
                if record.parent_tool_call_id:
                    forwarded_payload.setdefault("parent_tool_call_id", record.parent_tool_call_id)
            else:
                handler = record.initial_event_forwarder
                if handler is None:
                    return
                forwarded_event = event
                forwarded_payload = dict(payload)

        try:
            handler(forwarded_event, forwarded_payload)
        except Exception:
            logger.exception("Sub-task session event forwarder failed for %s", task_id)

    @staticmethod
    def _sanitize_resumable_session_messages(session: SubTaskSession) -> int:
        return SubTaskManager._replace_session_messages_if_needed(session, session.messages)

    @staticmethod
    def _replace_session_messages_if_needed(session: SubTaskSession, messages: list[Message]) -> int:
        original = list(messages)
        sanitized = sanitize_for_resume(original)
        if [_message_snapshot(message) for message in sanitized] == [_message_snapshot(message) for message in original]:
            return 0
        session.replace_messages(sanitized)
        return max(len(original) - len(sanitized), 0)

    def _build_outcome_from_run(self, *, task_id: str, run: AgentSessionRun) -> SubTaskOutcome:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(task_id)
            resolved = dict(record.resolved)
            session_id = record.session_id
            agent_name = record.agent_name

        return SubTaskOutcome(
            task_id=task_id,
            session_id=session_id,
            agent_name=agent_name,
            status=run.result.status,
            final_answer=run.result.final_answer,
            wait_reason=run.result.wait_reason,
            error=run.result.error,
            error_code=("sub_task_failed" if run.result.status == AgentStatus.FAILED else None),
            cycles=len(run.result.cycles),
            todo_list=run.result.todo_list,
            resolved=resolved,
        )

    def _handle_session_event(
        self,
        *,
        task_id: str,
        session_generation: int,
        generation_token: object,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if (
                record is None
                or record.generation_token is not generation_token
                or record.session_generation != session_generation
            ):
                return

            record.updated_at = _now_iso()

            if event == "session_run_start":
                prompt = _preview_text(payload.get("prompt"))
                if prompt:
                    record.task_title = prompt
                    record.recent_activity = prompt
                return

            if event == "cycle_started":
                cycle_index = payload.get("cycle")
                if isinstance(cycle_index, int):
                    record.current_cycle_index = cycle_index
                    record.latest_cycle = {
                        "cycle_index": cycle_index,
                        "status": "processing",
                    }
                return

            if event == "cycle_llm_response":
                cycle_index = payload.get("cycle")
                if isinstance(cycle_index, int):
                    record.current_cycle_index = cycle_index
                assistant_preview = _preview_text(payload.get("assistant_preview") or payload.get("assistant_message"))
                latest_cycle: dict[str, Any] = {
                    "status": "processing",
                }
                if isinstance(cycle_index, int):
                    latest_cycle["cycle_index"] = cycle_index
                if assistant_preview:
                    latest_cycle["assistant_preview"] = assistant_preview
                    record.recent_activity = assistant_preview
                record.latest_cycle = latest_cycle
                return

            if event == "tool_result":
                tool_status = _preview_text(payload.get("status"))
                record.latest_tool_call = {
                    "tool_call_id": payload.get("tool_call_id"),
                    "name": payload.get("tool_name"),
                    "status": tool_status,
                }
                if not record.recent_activity:
                    record.recent_activity = _preview_text(payload.get("tool_name"))
                return

            if event == "run_completed":
                self._mark_terminal_state(
                    record=record,
                    status=AgentStatus.COMPLETED.value,
                    detail=_preview_text(payload.get("final_answer")),
                )
                return

            if event == "run_wait_user":
                self._mark_terminal_state(
                    record=record,
                    status=AgentStatus.WAIT_USER.value,
                    detail=_preview_text(payload.get("wait_reason")),
                )
                return

            if event == "run_max_cycles":
                self._mark_terminal_state(
                    record=record,
                    status=AgentStatus.MAX_CYCLES.value,
                    detail=_preview_text(payload.get("final_answer") or payload.get("error")),
                )
                return

            if event == "cycle_failed":
                self._mark_terminal_state(
                    record=record,
                    status=AgentStatus.FAILED.value,
                    detail=_preview_text(payload.get("error")),
                )
                return

            if event == "session_run_end":
                status = _preview_text(payload.get("status"))
                if status:
                    latest_cycle = dict(record.latest_cycle or {})
                    latest_cycle["status"] = status
                    record.latest_cycle = latest_cycle
                detail = (
                    _preview_text(payload.get("final_answer"))
                    or _preview_text(payload.get("wait_reason"))
                    or _preview_text(payload.get("error"))
                )
                if detail:
                    record.recent_activity = detail

    @staticmethod
    def _mark_terminal_state(*, record: ManagedSubTask, status: str, detail: str | None) -> None:
        latest_cycle = dict(record.latest_cycle or {})
        latest_cycle["status"] = status
        if record.current_cycle_index is not None:
            latest_cycle.setdefault("cycle_index", record.current_cycle_index)
        record.latest_cycle = latest_cycle
        if detail:
            record.recent_activity = detail
