from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from threading import Condition, Thread
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from vv_agent.events import RunEvent
from vv_agent.result import RunResult
from vv_agent.session.children import cancel_children, child_delivery, child_handles
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.records import InboxItem
from vv_agent.session.store import Conflict, LeaseLost, LeaseRetryExhausted
from vv_agent.tracing import trace_processors

if TYPE_CHECKING:
    from vv_agent.session.runtime import Runtime
    from vv_agent.session.surfaces import SessionDriver

RunHandleStatus = Literal[
    "pending",
    "running",
    "host_interaction",
    "suspended",
    "wait_user",
    "completed",
    "failed",
    "max_cycles",
    "cancelled",
    "reconciliation_required",
    "deferred",
]


@dataclass(frozen=True, slots=True)
class RunHandleState:
    status: RunHandleStatus
    done: bool
    cancelled: bool = False
    error: str | None = None


class RunHandleController(Protocol):
    def steer(self, message: str) -> None: ...

    def follow_up(self, message: str) -> None: ...


class RunHandle:
    def __init__(self, kernel: SessionDriver, sid: str, tid: str, runtime: Runtime, consumer: str | None) -> None:
        self.kernel, self.session_id, self.run_id, self.runtime, self.consumer = kernel, sid, tid, runtime, consumer
        self._condition = Condition()
        self._events: list[RunEvent] = []
        self._observer = runtime.config.stream
        runtime.config = replace(runtime.config, stream=self._emit)
        self._done = False
        self._cancel_bound = False
        self._error: BaseException | None = None
        self._deliver_to_parent = False
        self._thread = Thread(target=self._run, name="session-surface-driver")
        original_hook = runtime.hook

        def hook(point, record):
            original_hook(point, record)
            if point == "after_commit":
                self._bind_cancel()
                self._publish()
            if point in {"after_commit", "after_drive", "after_input"}:
                self._schedule_background()

        runtime.hook = hook

    def _bind_cancel(self) -> None:
        token = self.runtime.config.cancellation_token
        if self._cancel_bound or token is None:
            return
        state, _, _ = self.kernel.store.read_state(self.session_id)
        if self.run_id not in state.turns:
            return
        self._cancel_bound = True

        def cancel() -> None:
            self.cancel()

        token.on_cancel(cancel)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def _emit(self, event: RunEvent) -> None:
        if self.runtime.config.event_store:
            try:
                self.runtime.config.event_store.append(event)
            except Exception as exc:
                if self.runtime.config.event_store_fail_closed:
                    raise
                warnings.warn(f"Run event store append failed: {exc}", RuntimeWarning, stacklevel=2)
        if self._observer:
            with suppress(Exception):
                self._observer(event)
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def _publish(self) -> None:
        if self.consumer is None:
            return

        events = SessionRunEventStore(self.kernel.store, self.session_id, self.consumer)
        while events.consume(self._emit):
            pass
        from vv_agent.session.tracing import deliver_spans

        processors = trace_processors(self.runtime.config)
        if processors:
            deliver_spans(self.kernel.store, self.session_id, processors)

    def _schedule_background(self) -> None:
        parent_state, _, _ = self.kernel.store.read_state(self.session_id)
        for operation in parent_state.operations.values():
            for attempt in operation.attempts.values():
                if attempt.child_handle is None:
                    continue
                for child in child_handles(attempt.child_handle):
                    self._schedule_child(child)

    def _schedule_child(self, child) -> None:
        sid = child["session_id"]
        with self.kernel._background_lock:
            previous = self.kernel._background.get(sid)
            if previous is not None and not previous.done():
                return
            state, _, _ = self.kernel.store.read_state(sid)
            pending = self.kernel.store.peek_inbox(sid)
            if not child["background"] and not (
                state.active_turn_id is None and state.turns and any(item.item.kind == "follow_up" for item in pending)
            ):
                return
            if state.turns and state.active_turn_id is None and not pending:
                return
            handle = RunHandle(self.kernel, sid, child["turn_id"], self.runtime.child_runtime(self.kernel.store, sid), None)
            handle._deliver_to_parent = True
            self.kernel._background[sid] = handle
            self.kernel.handles.append(handle)
            handle.start()

    def _drive_tree(self, sid: str, runtime: Runtime, tid: str) -> None:
        from vv_agent.session.kernel import drive

        losses = 0
        while True:
            try:
                drive(self.kernel.store, sid, runtime=runtime, _one_turn=True, _wait_for_lease=True)
            except LeaseLost as exc:
                state = None
                with suppress(LeaseLost):
                    state, _, _ = self.kernel.store.read_state(sid)
                # A terminal receipt already committed by this drive must not admit another turn.
                if state is None or tid not in state.turns or not state.turns[tid].ended:
                    losses += 1
                    if losses >= runtime.lease_retry_attempts:
                        raise LeaseRetryExhausted(sid, tid, losses) from exc
                    delay = min(runtime.lease_retry_cap_seconds, runtime.lease_retry_base_seconds * 2 ** min(losses - 1, 30))
                    runtime.lease_retry_sleep(runtime.lease_retry_jitter(delay / 2, delay))
                    continue
            state, _, _ = self.kernel.store.read_state(sid)
            completed = False
            for (oid, _), wait in state.waits.items():
                handle = wait["handle"]
                if state.operations[oid].turn_id != state.active_turn_id or handle["kind"] != "child" or handle["background"]:
                    continue
                for child in child_handles(handle):
                    child_id = child["session_id"]
                    self._drive_tree(child_id, runtime.child_runtime(self.kernel.store, child_id), child["turn_id"])
                    with self.kernel.store.atomic() as tx:
                        delivered = child_delivery(self.kernel.store, tx, child_id)
                    child_state, _, _ = self.kernel.store.read_state(child_id)
                    completed |= bool(delivered and child_state.turns[child["turn_id"]].ended)
            if not completed:
                return

    def _run(self) -> None:
        try:
            self._bind_cancel()
            self._publish()
            self._schedule_background()
            self._drive_tree(self.session_id, self.runtime, self.run_id)
            if self._deliver_to_parent:
                with self.kernel.store.atomic() as tx:
                    child_delivery(self.kernel.store, tx, self.session_id)
            self._publish()
        except BaseException as exc:
            self._error = exc
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def events(self) -> Iterator[RunEvent]:
        position = 0
        while True:
            with self._condition:
                self._condition.wait_for(lambda position=position: position < len(self._events) or self._done)
                if position < len(self._events):
                    event = self._events[position]
                    position += 1
                else:
                    return
            yield event

    def result(self, timeout: float | None = None) -> RunResult:
        from vv_agent.session.result import project_result

        self.join(timeout)
        if not self._done:
            raise TimeoutError("session execution is still active")
        if self._error:
            raise self._error
        state, _, _ = self.kernel.store.read_state(self.session_id)
        tid = self.run_id if self.run_id in state.turns else state.active_turn_id or next(reversed(state.turns), "")
        if not tid:
            raise Conflict("session has no admitted turn")
        self.run_id = tid
        result = project_result(self.kernel.store, self.session_id, tid, runtime=self.runtime)
        waits = self.kernel.waits(self.session_id, self.runtime)
        if waits:
            result.metadata["session_waits"] = [{"session_id": sid, **wait} for sid, _, wait in waits]
            result.raw_result.wait_reason = waits[0][2].get("question", result.raw_result.wait_reason)
        result._session_driver = self.kernel
        return result

    def cancel(self, reason: str = "") -> bool:
        del reason
        state, _, _ = self.kernel.store.read_state(self.session_id)
        if state.active_turn_id != self.run_id:
            return False
        with self.kernel.store.atomic() as tx:
            state, _, _ = self.kernel.store.read_state(self.session_id)
            if state.active_turn_id != self.run_id:
                return False
            receipt = tx.push(self.session_id, InboxItem(f"cancel/{self.run_id}", "control", {"action": "cancel"}, self.run_id))
            cancel_children(self.kernel.store, tx, state, self.session_id, self.run_id)
        return not receipt.replayed

    def done(self) -> bool:
        return self._done

    def state(self) -> RunHandleState:
        if self._error:
            return RunHandleState("failed", True, error=str(self._error))
        if not self._done:
            return RunHandleState("running", False)
        result = self.result()
        cancelled = bool(result.completion_reason and result.completion_reason.value == "cancelled")
        return RunHandleState("cancelled" if cancelled else result.status.value, True, cancelled)

    def approve(self, request_id: str, decision) -> None:
        self.kernel.approve(self.session_id, self.runtime, request_id, decision, f"approval/{request_id}/answer")

    def steer(self, message: str) -> None:
        self.kernel.push(self.session_id, InboxItem(uuid4().hex, "steer", {"content": message}, self.run_id))

    def follow_up(self, message: str) -> None:
        self.kernel.push(self.session_id, InboxItem(uuid4().hex, "follow_up", {"content": message}))

    def resume(self, state_or_token=None, payload=None) -> RunResult:
        if state_or_token is not None or payload is not None:
            raise ValueError("resume uses the retained session and turn without new input")
        from vv_agent.session.surfaces import resume_turn

        return resume_turn(self.session_id, self.run_id)
