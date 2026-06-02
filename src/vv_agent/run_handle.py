from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalBroker, ApprovalDecision
from vv_agent.events import RunEvent
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.types import AgentStatus

if TYPE_CHECKING:
    from vv_agent.runner import Runner


ApprovalInput = ApprovalDecision | str
RunHandleStatus = Literal["pending", "running", "wait_user", "completed", "failed", "max_cycles", "cancelled"]


class RunHandleController(Protocol):
    def steer(self, message: str) -> None:
        ...

    def follow_up(self, message: str) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RunHandleState:
    status: RunHandleStatus
    done: bool
    cancelled: bool = False
    error: str | None = None


class _Sentinel:
    pass


class RunHandle:
    def __init__(
        self,
        *,
        event_queue: queue.Queue[RunEvent | _Sentinel],
        done_event: threading.Event,
        thread: threading.Thread,
        cancellation_token: CancellationToken,
        approval_broker: ApprovalBroker,
    ) -> None:
        self._event_queue = event_queue
        self._done_event = done_event
        self._thread = thread
        self._cancellation_token = cancellation_token
        self._approval_broker = approval_broker
        self._lock = threading.Lock()
        self._result: RunResult | None = None
        self._exception: BaseException | None = None
        self._cancel_requested = False
        self._controller: RunHandleController | None = None

    @classmethod
    def start_worker(cls, *, agent: Agent, input: str, run_config: RunConfig, runner: type[Runner]) -> RunHandle:
        event_queue: queue.Queue[RunEvent | _Sentinel] = queue.Queue()
        done_event = threading.Event()
        cancellation_token = run_config.cancellation_token or CancellationToken()
        approval_broker = run_config.approval_broker or ApprovalBroker()
        worker_config = run_config.with_cancellation_token(cancellation_token)
        worker_config.approval_broker = approval_broker

        handle = cls(
            event_queue=event_queue,
            done_event=done_event,
            thread=threading.Thread(),
            cancellation_token=cancellation_token,
            approval_broker=approval_broker,
        )

        def event_sink(event: RunEvent) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                result = runner._run(agent, input, run_config=worker_config, event_sink=event_sink)
            except BaseException as exc:
                with handle._lock:
                    handle._exception = exc
            else:
                with handle._lock:
                    handle._result = result
            finally:
                done_event.set()
                event_queue.put(_Sentinel())

        handle._thread = threading.Thread(target=worker, name="vv-agent-run-handle", daemon=False)
        handle._thread.start()
        return handle

    def events(self) -> Iterator[RunEvent]:
        while True:
            item = self._event_queue.get()
            if isinstance(item, _Sentinel):
                break
            yield item

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
        return self._done_event.is_set()

    def cancel(self, reason: str = "") -> bool:
        if self.done():
            return False
        self._cancel_requested = True
        self._cancellation_token.cancel()
        self._approval_broker.cancel_pending(reason or "Run was cancelled.")
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

    def resume(self, resume_token: str, payload: dict[str, Any] | None = None) -> None:
        controller = self._require_controller("resume")
        resume = getattr(controller, "resume", None)
        if callable(resume):
            resume(resume_token, payload)
            return
        del resume_token, payload
        raise NotImplementedError("RunHandle.resume() is not supported by the synchronous runner yet.")

    def state(self) -> RunHandleState:
        with self._lock:
            if self._exception is not None:
                return RunHandleState(status="failed", done=True, cancelled=self._cancel_requested, error=str(self._exception))
            if self._result is not None:
                return RunHandleState(
                    status=self._status_from_result(self._result.status),
                    done=True,
                    cancelled=self._cancel_requested,
                    error=self._result.raw_result.error,
                )
        if not self.done():
            return RunHandleState(status="running", done=False, cancelled=self._cancel_requested)
        if self._cancel_requested:
            return RunHandleState(status="cancelled", done=True, cancelled=True)
        return RunHandleState(status="completed", done=True)

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
