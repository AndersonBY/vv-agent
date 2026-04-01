from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.memory import sanitize_for_resume
from vv_agent.types import AgentStatus, SubTaskOutcome

if TYPE_CHECKING:
    from vv_agent.sdk.types import AgentRun
    from vv_agent.workspace.base import WorkspaceBackend

logger = logging.getLogger(__name__)

SessionRegistrar = Callable[[str, Any], None]
SessionUnregistrar = Callable[[str, Any | None], None]
SubTaskRunnerCallable = Callable[[], SubTaskOutcome]


class SubTaskSession(Protocol):
    def subscribe(self, listener: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]: ...
    def continue_run(self, prompt: str) -> Any: ...


class ThreadHandle(Protocol):
    def is_alive(self) -> bool: ...
    def join(self, timeout: float | None = None) -> None: ...


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


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
    thread: ThreadHandle | None = None
    manager_listener_attached: bool = False
    forward_listener_attached: bool = False

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


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
            if record.is_running():
                raise RuntimeError(f"Sub-task {task_id} is already running")

            record.session_id = session_id
            record.agent_name = agent_name
            record.task_title = task_title or record.task_title
            record.workspace_backend = workspace_backend
            record.outcome = None
            record.updated_at = _now_iso()
            thread = threading.Thread(
                target=self._run_and_capture,
                kwargs={"task_id": task_id, "runner": runner},
                daemon=True,
                name=f"vv-agent-subtask-{task_id[:12]}",
            )
            record.thread = thread

        thread.start()
        return record

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
        event_forwarder: Callable[[str, dict[str, Any]], None] | None = None,
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

            record.session_id = session_id
            record.agent_name = agent_name
            record.task_title = task_title or record.task_title
            record.workspace_backend = workspace_backend
            record.session = session
            if resolved:
                record.resolved = dict(resolved)
            record.updated_at = _now_iso()

            if not record.manager_listener_attached:
                session.subscribe(
                    lambda event, payload, *, _task_id=task_id: self._handle_session_event(
                        task_id=_task_id,
                        event=event,
                        payload=payload,
                    )
                )
                record.manager_listener_attached = True

            if event_forwarder is not None and not record.forward_listener_attached:
                session.subscribe(event_forwarder)
                record.forward_listener_attached = True

            return record

    def record_outcome(self, task_id: str, outcome: SubTaskOutcome) -> ManagedSubTask:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                record = ManagedSubTask(
                    task_id=outcome.task_id,
                    session_id=outcome.session_id or "",
                    agent_name=outcome.agent_name,
                )
                self._tasks[task_id] = record

            record.session_id = outcome.session_id or record.session_id
            record.agent_name = outcome.agent_name
            record.outcome = outcome
            if outcome.resolved:
                record.resolved = dict(outcome.resolved)
            if outcome.cycles > 0:
                record.current_cycle_index = outcome.cycles
            record.updated_at = _now_iso()

            status_value = outcome.status.value
            latest_cycle = dict(record.latest_cycle or {})
            latest_cycle["status"] = status_value
            if outcome.cycles > 0:
                latest_cycle["cycle_index"] = outcome.cycles
            if latest_cycle:
                record.latest_cycle = latest_cycle

            summary_text = (
                _preview_text(outcome.final_answer)
                or _preview_text(outcome.wait_reason)
                or _preview_text(outcome.error)
            )
            if summary_text:
                record.recent_activity = summary_text

            return record

    def continue_task(self, *, task_id: str, prompt: str) -> ManagedSubTask:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise ValueError("Follow-up prompt cannot be empty")

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(task_id)
            if record.is_running():
                raise RuntimeError(f"Sub-task {task_id} is already running")
            if record.session is None:
                raise RuntimeError(f"Sub-task {task_id} session is not available")
            if record.outcome is not None and record.outcome.status == AgentStatus.MAX_CYCLES:
                raise RuntimeError(f"Sub-task {task_id} reached max cycles and cannot continue")

            removed_messages = self._sanitize_resumable_session_messages(record.session)
            if removed_messages > 0:
                logger.info(
                    "Sanitized %s stale message(s) before resuming sub-task %s",
                    removed_messages,
                    task_id,
                )

            record.task_title = prompt_text
            record.outcome = None
            record.updated_at = _now_iso()
            thread = threading.Thread(
                target=self._continue_existing_session,
                kwargs={"task_id": task_id, "prompt": prompt_text},
                daemon=True,
                name=f"vv-agent-subtask-{task_id[:12]}-continue",
            )
            record.thread = thread

        thread.start()
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
        except Exception as exc:
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
                    resolved=dict(record.resolved),
                )
        self.record_outcome(task_id, outcome)

    def _continue_existing_session(self, *, task_id: str, prompt: str) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.session is None:
                return
            session = record.session
            session_id = record.session_id

        self._register_session(session_id, session)
        try:
            run = session.continue_run(prompt)
            outcome = self._build_outcome_from_run(task_id=task_id, run=run)
        except Exception as exc:
            with self._lock:
                current = self._tasks.get(task_id)
                if current is None:
                    return
                outcome = SubTaskOutcome(
                    task_id=current.task_id,
                    session_id=current.session_id,
                    agent_name=current.agent_name,
                    status=AgentStatus.FAILED,
                    error=str(exc),
                    resolved=dict(current.resolved),
                )
        finally:
            self._unregister_session(session_id, session)

        self.record_outcome(task_id, outcome)

    @staticmethod
    def _sanitize_resumable_session_messages(session: SubTaskSession) -> int:
        raw_messages = getattr(session, "_messages", None)
        if not isinstance(raw_messages, list):
            return 0

        session_lock = getattr(session, "_lock", None)
        if session_lock is not None and hasattr(session_lock, "__enter__") and hasattr(session_lock, "__exit__"):
            with session_lock:
                return SubTaskManager._replace_session_messages_if_needed(session, raw_messages)
        return SubTaskManager._replace_session_messages_if_needed(session, raw_messages)

    @staticmethod
    def _replace_session_messages_if_needed(session: SubTaskSession, messages: list[Any]) -> int:
        original = list(messages)
        sanitized = sanitize_for_resume(original)
        if [_message_snapshot(message) for message in sanitized] == [_message_snapshot(message) for message in original]:
            return 0
        session._messages = sanitized  # type: ignore[attr-defined]
        return max(len(original) - len(sanitized), 0)

    def _build_outcome_from_run(self, *, task_id: str, run: AgentRun) -> SubTaskOutcome:
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
            cycles=len(run.result.cycles),
            todo_list=run.result.todo_list,
            resolved=resolved,
        )

    def _handle_session_event(self, *, task_id: str, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
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
