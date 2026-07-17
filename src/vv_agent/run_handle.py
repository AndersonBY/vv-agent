from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalBroker, ApprovalDecision
from vv_agent.events import RunEvent
from vv_agent.result import RunResult, RunState
from vv_agent.run_config import RunConfig
from vv_agent.runtime.cancellation import CancellationToken, CancelledError
from vv_agent.types import AgentStatus

ApprovalInput = ApprovalDecision | str
RunHandleStatus = Literal[
    "pending",
    "running",
    "reconciliation_required",
    "wait_user",
    "completed",
    "failed",
    "max_cycles",
    "cancelled",
]


class RunHandleController(Protocol):
    def steer(self, message: str) -> None: ...

    def follow_up(self, message: str) -> None: ...


class RunHandleRunner(Protocol):
    def _run(
        self,
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig,
        event_sink: Callable[[RunEvent], None] | None = None,
        _compiled_invocation: Any | None = None,
        _approval_invocation: Any | None = None,
    ) -> RunResult: ...

    def resume(self, state: RunState, *, input: str | None = None) -> RunResult: ...


@dataclass(frozen=True, slots=True)
class RunHandleState:
    status: RunHandleStatus
    done: bool
    cancelled: bool = False
    error: str | None = None


class RunHandle:
    def __init__(
        self,
        *,
        events: list[RunEvent],
        event_condition: threading.Condition,
        done_event: threading.Event,
        thread: threading.Thread,
        cancellation_token: CancellationToken,
        approval_broker: ApprovalBroker,
        runner: RunHandleRunner,
    ) -> None:
        self._events = events
        self._event_condition = event_condition
        self._done_event = done_event
        self._thread = thread
        self._cancellation_token = cancellation_token
        self._approval_broker = approval_broker
        self._runner = runner
        self._lock = threading.Lock()
        self._result: RunResult | None = None
        self._exception: BaseException | None = None
        self._cancel_requested = False
        self._terminal_event_seen = False
        self._terminal_seen_during_handoff = False
        self._active_handoffs = 0
        self._lifecycle_event_ids: set[str] = set()
        self._active_sub_run_ids: set[str] = set()
        self._controller: RunHandleController | None = None

    @classmethod
    def _start_worker(
        cls,
        *,
        agent: Agent,
        input: str,
        run_config: RunConfig,
        runner: RunHandleRunner,
        _compiled_invocation: Any | None = None,
    ) -> RunHandle:
        events: list[RunEvent] = []
        event_condition = threading.Condition()
        done_event = threading.Event()
        cancellation_token = run_config.cancellation_token or CancellationToken()
        approval_broker = run_config.approval_broker or ApprovalBroker()
        worker_config = run_config.with_cancellation_token(cancellation_token)
        worker_config.approval_broker = approval_broker

        handle = cls(
            events=events,
            event_condition=event_condition,
            done_event=done_event,
            thread=threading.Thread(),
            cancellation_token=cancellation_token,
            approval_broker=approval_broker,
            runner=runner,
        )

        original_stream = worker_config.stream

        def stream(event: RunEvent) -> None:
            handle._mark_terminal_event(event)
            if original_stream is not None:
                original_stream(event)

        worker_config = replace(worker_config, stream=stream)

        def event_sink(event: RunEvent) -> None:
            handle._mark_terminal_event(event)
            with event_condition:
                events.append(event)
                if event.type == "sub_run_started":
                    handle._active_sub_run_ids.add(event.run_id)
                elif event.type == "sub_run_completed":
                    handle._active_sub_run_ids.discard(event.run_id)
                event_condition.notify_all()

        def worker() -> None:
            try:
                result = runner._run(
                    agent,
                    input,
                    run_config=worker_config,
                    event_sink=event_sink,
                    _compiled_invocation=_compiled_invocation,
                )
            except BaseException as exc:
                with handle._lock:
                    handle._exception = exc
            else:
                with handle._lock:
                    handle._result = result
            finally:
                done_event.set()
                with event_condition:
                    event_condition.notify_all()

        handle._thread = threading.Thread(target=worker, name="vv-agent-run-handle", daemon=False)
        handle._thread.start()
        return handle

    def events(self) -> Iterator[RunEvent]:
        index = 0
        while True:
            with self._event_condition:
                while index >= len(self._events) and not self._event_stream_done():
                    self._event_condition.wait()
                if index >= len(self._events):
                    return
                event = self._events[index]
                index += 1
            yield event

    def result(self, timeout: float | None = None) -> RunResult:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("RunHandle result was not ready before timeout.")
        with self._lock:
            if self._exception is not None:
                raise self._exception
            if self._result is None:
                raise RuntimeError("RunHandle completed without a result.")
            return self._result

    def done(self) -> bool:
        with self._event_condition:
            return self._done_event.is_set() and not self._active_sub_run_ids

    def cancel(self, reason: str = "") -> bool:
        with self._lock:
            with self._event_condition:
                active_sub_runs = bool(self._active_sub_run_ids)
            if (
                (self._result is not None and not active_sub_runs)
                or (self._exception is not None and not active_sub_runs)
                or (self._terminal_event_seen and not active_sub_runs)
                or self._cancel_requested
                or (self.done() and not active_sub_runs)
            ):
                return False
            self._cancel_requested = True
        self._cancellation_token.cancel(reason or "Run was cancelled.")
        return True

    def attach_controller(self, controller: RunHandleController) -> None:
        with self._lock:
            self._controller = controller

    def detach_controller(self, controller: RunHandleController | None = None) -> None:
        with self._lock:
            if controller is not None and self._controller is not controller:
                return
            self._controller = None

    def steer(self, message: str) -> None:
        self._require_controller("steer").steer(message)

    def follow_up(self, message: str) -> None:
        self._require_controller("follow_up").follow_up(message)

    def approve(self, request_id: str, decision: ApprovalInput) -> None:
        if not self._approval_broker.resolve(request_id, decision):
            raise KeyError(f"Unknown approval request: {request_id}")

    def resume(
        self,
        state_or_token: RunState | str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunResult | None:
        if isinstance(state_or_token, str):
            controller = self._require_controller("resume")
            resume = getattr(controller, "resume", None)
            if callable(resume):
                resume(state_or_token, payload)
                return None
            raise NotImplementedError("The attached controller does not support RunHandle.resume().")

        state = state_or_token or self.result().into_state()
        resume_input = payload.get("input") if payload is not None else None
        return self._runner.resume(state, input=str(resume_input) if resume_input is not None else None)

    def state(self) -> RunHandleState:
        with self._lock:
            with self._event_condition:
                active_sub_runs = bool(self._active_sub_run_ids)
            if active_sub_runs:
                return RunHandleState(
                    status="running",
                    done=False,
                    cancelled=self._cancel_requested or self._cancellation_token.cancelled,
                )
            if self._exception is not None:
                if self._is_cancelled_error(self._exception):
                    return RunHandleState(status="cancelled", done=True, cancelled=True, error=str(self._exception))
                return RunHandleState(status="failed", done=True, error=str(self._exception))
            if self._cancel_requested and self._done_event.is_set():
                return RunHandleState(
                    status="cancelled",
                    done=True,
                    cancelled=True,
                    error=self._cancellation_token.reason,
                )
            if self._result is not None:
                if self._result_was_cancelled(self._result):
                    return RunHandleState(
                        status="cancelled",
                        done=True,
                        cancelled=True,
                        error=self._result.raw_result.error,
                    )
                return RunHandleState(
                    status=self._status_from_result(self._result.status),
                    done=True,
                    error=self._result.raw_result.error,
                )
        if not self.done():
            return RunHandleState(
                status="running",
                done=False,
                cancelled=self._cancel_requested or self._cancellation_token.cancelled,
            )
        if self._cancel_requested or self._cancellation_token.cancelled:
            return RunHandleState(
                status="cancelled",
                done=True,
                cancelled=True,
                error=self._cancellation_token.reason,
            )
        return RunHandleState(status="completed", done=True)

    def _is_cancelled_error(self, error: BaseException) -> bool:
        return isinstance(error, CancelledError) or (self._cancellation_token.cancelled and "cancel" in str(error).lower())

    def _result_was_cancelled(self, result: RunResult) -> bool:
        return (
            result.status == AgentStatus.FAILED
            and self._cancellation_token.cancelled
            and "cancel" in (result.raw_result.error or "").lower()
        )

    def _event_stream_done(self) -> bool:
        return self._done_event.is_set() and not self._active_sub_run_ids

    @staticmethod
    def _status_from_result(status: AgentStatus) -> RunHandleStatus:
        return status.value

    def _require_controller(self, method: str) -> RunHandleController:
        with self._lock:
            controller = self._controller
        if controller is None:
            raise NotImplementedError(
                f"RunHandle.{method}() is only available when the handle is attached to an interactive session."
            )
        return controller

    def _mark_terminal_event(self, event: RunEvent) -> None:
        if event.type not in {
            "handoff_started",
            "handoff_completed",
            "run_completed",
            "run_failed",
            "run_cancelled",
        }:
            return
        with self._lock:
            if event.event_id in self._lifecycle_event_ids:
                return
            self._lifecycle_event_ids.add(event.event_id)
            if event.type == "handoff_started":
                self._active_handoffs += 1
                self._terminal_event_seen = False
                self._terminal_seen_during_handoff = False
                return
            if event.type == "handoff_completed":
                self._active_handoffs = max(self._active_handoffs - 1, 0)
                if bool(event.metadata.get("chain_continues")):
                    self._terminal_event_seen = False
                    self._terminal_seen_during_handoff = False
                elif self._active_handoffs == 0 and self._terminal_seen_during_handoff:
                    self._terminal_event_seen = True
                    self._terminal_seen_during_handoff = False
                return
            if self._active_handoffs:
                self._terminal_seen_during_handoff = True
            else:
                self._terminal_event_seen = True
