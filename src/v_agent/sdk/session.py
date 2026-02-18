from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from v_agent.sdk.types import AgentDefinition, AgentRun
from v_agent.types import AgentStatus, Message

SessionEventHandler = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class AgentSessionState:
    running: bool
    messages: list[Message] = field(default_factory=list)
    shared_state: dict[str, Any] = field(default_factory=dict)
    latest_run: AgentRun | None = None


class AgentSession:
    """Session-first SDK abstraction for multi-turn, stateful task execution."""

    def __init__(
        self,
        *,
        execute_run: Callable[..., AgentRun],
        agent_name: str,
        definition: AgentDefinition,
        shared_state: dict[str, Any] | None = None,
    ) -> None:
        self._execute_run = execute_run
        self.agent_name = agent_name
        self.definition = definition
        self._messages: list[Message] = []
        self._shared_state: dict[str, Any] = dict(shared_state or {})
        self._shared_state.setdefault("todo_list", [])
        self._latest_run: AgentRun | None = None
        self._running = False
        self._listeners: list[SessionEventHandler] = []
        self._steering_queue: deque[str] = deque()
        self._follow_up_queue: deque[str] = deque()
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
    def latest_run(self) -> AgentRun | None:
        with self._lock:
            return self._latest_run

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

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

    def prompt(self, prompt: str, *, auto_follow_up: bool = True) -> AgentRun:
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

    def continue_run(self, prompt: str | None = None) -> AgentRun:
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
                messages=list(self._messages),
                shared_state=dict(self._shared_state),
                latest_run=self._latest_run,
            )

    def _run_once(self, prompt: str) -> AgentRun:
        with self._lock:
            if self._running:
                raise RuntimeError("Session is already running. Queue with steer()/follow_up() or wait for completion.")
            self._running = True
            initial_messages = list(self._messages)
            current_shared_state = dict(self._shared_state)

        self._emit("session_run_start", prompt=prompt, existing_messages=len(initial_messages))
        try:
            run = self._execute_run(
                prompt=prompt,
                agent=self.definition,
                task_name=self.agent_name,
                shared_state=current_shared_state,
                initial_messages=initial_messages,
                before_cycle_messages=self._before_cycle_messages,
                interruption_messages=self._interruption_messages,
                log_handler=self._session_log_handler,
            )
        finally:
            with self._lock:
                self._running = False

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
        self._emit(event, **payload)

    def _emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(event, payload)


def create_agent_session(
    *,
    execute_run: Callable[..., AgentRun],
    agent_name: str,
    definition: AgentDefinition,
    shared_state: dict[str, Any] | None = None,
) -> AgentSession:
    return AgentSession(
        execute_run=execute_run,
        agent_name=agent_name,
        definition=definition,
        shared_state=shared_state,
    )
