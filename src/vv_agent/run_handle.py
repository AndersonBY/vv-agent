from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from uuid import uuid4

from vv_agent.events import RunEvent
from vv_agent.result import RunResult
from vv_agent.session.callbacks import synchronous, validate
from vv_agent.session.children import cancel_children, child_delivery, child_handles
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.ownership import checkout, committed_guard
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
        committed_guard(kernel.store)
        self.kernel, self.session_id, self.run_id, self.runtime, self.consumer = kernel, sid, tid, runtime, consumer
        self._condition = Condition()
        self._events: list[RunEvent] = []
        self._observer = runtime.config.stream
        runtime.config = replace(runtime.config, stream=self._emit)
        self._source_store = kernel.store
        self._done = False
        self._cancel_bound = False
        self._error: BaseException | None = None
        self._deliver_to_parent = False
        self._background_child: tuple[RunHandle, dict[str, Any], bool] | None = None
        self._wait_for_lease = True
        self._lease_acquired = False
        self._lease_unavailable = False
        self._lease_unavailable_session: str | None = None
        self._lease_retry_allowed = True
        self._owner_wake_requested = False
        self._rerun_requested = False
        self._observed_sessions = {sid}
        self._drive_positions: dict[str, tuple[int, int]] = {}
        self._thread = Thread(target=self._run, name="session-surface-driver")
        original_hook = validate(runtime.hook, 2)

        def hook(point, record, *, tx):
            if point in {"after_commit", "after_drive"}:
                self._lease_acquired = True
            synchronous(original_hook(point, record, tx=tx))
            if point == "after_commit":
                self._bind_cancel()
                self._publish()
            if point in {"after_commit", "after_drive", "after_input"}:
                self._schedule_background()

        cast(Any, hook)._transaction_original = original_hook
        self._original_hook = original_hook
        self._owned_hook = hook
        runtime.hook = hook

    def _bind_cancel(self) -> None:
        token = self.runtime.config.cancellation_token
        if self._cancel_bound or token is None:
            return
        with checkout(self._source_store, role="projection") as projection:
            state, _, _ = projection.read_state(self.session_id)
        if self.run_id not in state.turns:
            return
        self._cancel_bound = True

        def cancel() -> None:
            self.cancel()

        token.on_cancel(cancel)

    def start(self) -> None:
        if self._done and self._thread.ident is None and self._error is not None:
            raise self._error
        try:
            self._thread.start()
        except BaseException as exc:
            self._fail_unstarted(exc)
            raise

    def _fail_unstarted(self, error: BaseException) -> bool:
        with self.kernel._background_lock, self._condition:
            if self._thread.ident is not None:
                return False
            self._error = error
            self._done = True
            self._condition.notify_all()
            return True

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

        with checkout(self._source_store, role="projection") as projection:
            events = SessionRunEventStore(projection, self.session_id, self.consumer)
            while events.consume(self._emit):
                pass
            from vv_agent.session.tracing import deliver_spans

            processors = trace_processors(self.runtime.config)
            if processors:
                deliver_spans(projection, self.session_id, processors)

    def _schedule_background(self) -> None:
        parent_state, _, _ = self._execution_store.read_state(self.session_id)
        for operation in parent_state.operations.values():
            for attempt in operation.attempts.values():
                if attempt.child_handle is None:
                    continue
                for child in child_handles(attempt.child_handle):
                    independent = attempt.yielded or bool(
                        attempt.wait and child["session_id"] in attempt.wait.get("yieldable_children", [])
                    )
                    self._schedule_child(child, independent=independent)

    def _schedule_child(self, child, *, independent: bool = False) -> None:
        sid = child["session_id"]
        self._observe_session(sid)
        # Delivery is recoverable even when the terminal child has no runnable work.
        while self._deliver_child(sid, self.runtime):
            pass
        with self.kernel._background_lock:
            previous = self.kernel._background.get(sid)
            if previous is not None and not previous.done():
                return
        state, _, _ = self._execution_store.read_state(sid)
        pending = self._execution_store.peek_inbox(sid)
        queued = next(
            (
                key
                for key, record in state.applied_inputs.items()
                if record._payload["disposition"] == "queued"
                and key not in state.admitted_inputs
                and record._payload["input"]["kind"] == "follow_up"
            ),
            None,
        ) or next((item.item.input_id for item in pending if item.item.kind == "follow_up"), None)
        if not queued and not self._execution_store.is_runnable(sid):
            return
        if not (child["background"] or independent) and not (state.active_turn_id is None and queued):
            return
        if state.turns and state.active_turn_id is None and not pending and not queued:
            return
        tid = state.active_turn_id or (f"{sid}/turn/{queued}" if queued else child["turn_id"])
        runtime = self.runtime.child_runtime(self._execution_store, sid)
        handle = RunHandle(self.kernel, sid, tid, runtime, None)
        handle._deliver_to_parent = True
        handle._background_child = self, child, independent
        with self.kernel._background_lock:
            # Another scheduler may have registered a worker while we read the store.
            if self.kernel._background.get(sid) is not previous:
                return
            self.kernel._background[sid] = handle
            self.kernel.handles.append(handle)
        handle.start()

    def _position(self, sid: str) -> tuple[int, int]:
        page = self._execution_store.read(sid, limit=1)
        return page.head_seq, page.inbox_seq

    def _observe_session(self, sid: str) -> None:
        with self.kernel._background_lock:
            self._observed_sessions.add(sid)

    def _wake(self) -> bool:
        # Caller holds B. Wake only marks an existing execution; its owner starts runs.
        if self._done:
            return False
        self._rerun_requested = True
        return True

    def _drive_tree(self, sid: str, runtime: Runtime, tid: str, *, ancestors: frozenset[str] = frozenset()) -> bool:
        from vv_agent.session.kernel import _drive_owned

        if sid in ancestors:
            raise Conflict(f"cyclic child ownership: {sid}")
        ancestors = ancestors | {sid}
        self._observe_session(sid)
        progressed = False

        def progress():
            nonlocal progressed
            progressed = True

        losses = 0
        while True:
            try:
                self._drive_positions[sid] = self._position(sid)
                acquired = _drive_owned(
                    self._execution_store,
                    sid,
                    runtime=runtime,
                    _one_turn=True,
                    _wait_for_lease=self._wait_for_lease,
                    _on_progress=progress,
                    _wake_on_exit=False,
                    _turn_id=tid,
                )
                if not acquired:
                    self._lease_unavailable = True
                    self._lease_unavailable_session = sid
                    return progressed
                if sid == self.session_id:
                    self._lease_acquired = True
            except LeaseLost as exc:
                state = None
                with suppress(LeaseLost):
                    state, _, _ = self._execution_store.read_state(sid)
                # A terminal receipt already committed by this drive must not admit another turn.
                if state is None or tid not in state.turns or not state.turns[tid].ended:
                    losses += 1
                    if losses >= runtime.lease_retry_attempts:
                        raise LeaseRetryExhausted(sid, tid, losses) from exc
                    delay = min(runtime.lease_retry_cap_seconds, runtime.lease_retry_base_seconds * 2 ** min(losses - 1, 30))
                    runtime.lease_retry_sleep(runtime.lease_retry_jitter(delay / 2, delay))
                    continue
            state, _, _ = self._execution_store.read_state(sid)
            if tid in state.turns and state.turns[tid].ended:
                # Parent closure enqueues descendant cancellation atomically. Drain those
                # controls even though terminal parents no longer have active child waits.
                for operation in state.operations.values():
                    if operation.turn_id != tid:
                        continue
                    for attempt in operation.attempts.values():
                        if attempt.child_handle is None:
                            continue
                        for child in child_handles(attempt.child_handle):
                            child_id = child["session_id"]
                            self._observe_session(child_id)
                            if any(i.item.kind == "control" for i in self._execution_store.peek_inbox(child_id)):
                                progressed |= self._drive_tree(
                                    child_id,
                                    runtime.child_runtime(self._execution_store, child_id),
                                    child["turn_id"],
                                    ancestors=ancestors,
                                )
                                progressed |= bool(self._deliver_child(child_id, runtime))
            completed = False
            workers = []
            foreground = set()
            blocked = False
            for (oid, _), wait in state.waits.items():
                handle = wait["handle"]
                if state.operations[oid].turn_id != state.active_turn_id:
                    continue
                if handle["kind"] != "child":
                    blocked = True
                    continue
                if handle["background"]:
                    continue
                for child in child_handles(handle):
                    child_id = child["session_id"]
                    foreground.add(child_id)
                    if child_id in wait.get("yieldable_children", []):
                        with self.kernel._background_lock:
                            worker = self.kernel._background.get(child_id)
                        if worker is not None and not worker.done():
                            workers.append(worker)
                            continue
                    progressed |= self._drive_tree(
                        child_id, runtime.child_runtime(self._execution_store, child_id), child["turn_id"], ancestors=ancestors
                    )
                    delivered = self._deliver_child(child_id, runtime)
                    progressed |= bool(delivered)
                    child_state, _, _ = self._execution_store.read_state(child_id)
                    # An unavailable child lease can leave its input queued before
                    # admission. It remains blocked until the ordinary lease retry.
                    child_turn = child_state.turns.get(child["turn_id"])
                    completed |= bool(delivered and child_turn is not None and child_turn.ended)
                    blocked |= child_turn is None or not child_turn.ended
            # Retained ownership outlives waits and parent turns; drive every remaining subtree.
            # An active worker owns its subtree; otherwise perform a real idle pass too.
            for operation in state.operations.values():
                for attempt in operation.attempts.values():
                    if attempt.child_handle is None:
                        continue
                    for child in child_handles(attempt.child_handle):
                        child_id = child["session_id"]
                        if child_id in foreground:
                            continue
                        with self.kernel._background_lock:
                            worker = self.kernel._background.get(child_id)
                        if worker is not None and not worker.done():
                            continue
                        self._observe_session(child_id)
                        child_state, _, _ = self._execution_store.read_state(child_id)
                        terminal_leaf = (
                            bool(child_state.turns)
                            and child_state.active_turn_id is None
                            and not any(a.child_handle for op in child_state.operations.values() for a in op.attempts.values())
                            and not any(
                                record._payload["disposition"] == "queued" and key not in child_state.admitted_inputs
                                for key, record in child_state.applied_inputs.items()
                            )
                        )
                        # A terminal leaf has only delivery work; no execution binding is needed.
                        if not terminal_leaf or self._execution_store.peek_inbox(child_id):
                            child_tid = child_state.active_turn_id or child["turn_id"]
                            progressed |= self._drive_tree(
                                child_id, runtime.child_runtime(self._execution_store, child_id), child_tid, ancestors=ancestors
                            )
                        delivered = self._deliver_child(child_id, runtime)
                        progressed |= bool(delivered)
                        completed |= bool(delivered and operation.turn_id == state.active_turn_id)
            if not completed and not blocked and workers:
                # A foreground observation must stay responsive while its child executes.
                # Once workers park for a person/provider, return the usual retained wait.
                while any(not worker.done() for worker in workers) and self._position(sid) == self._drive_positions[sid]:
                    with self._condition:
                        self._condition.wait(timeout=0.05)
                completed = self._position(sid) != self._drive_positions[sid]
            if not completed:
                return progressed or self._position(sid) != self._drive_positions[sid]

    def _deliver_child(self, sid: str, runtime: Runtime) -> int:
        pushed = False

        def hook(point):
            nonlocal pushed
            pushed |= point == "after_parent_inbox"

        with self._execution_store.atomic() as tx:
            delivered = child_delivery(self._execution_store, tx, sid, hook=hook)
        if pushed:
            created = self._execution_store.read(sid, limit=1).records[0].record
            parent_id = created._payload["parent_session_id"]
            self.kernel._wake(parent_id)
            runtime.wake(parent_id)
        return delivered

    def _drive_pass(self) -> bool:
        progressed = self._drive_tree(self.session_id, self.runtime, self.run_id)
        if self._deliver_to_parent:
            progressed |= bool(self._deliver_child(self.session_id, self.runtime))
        self._publish()
        return progressed

    def _run(self) -> None:
        try:
            first = True
            while True:
                with checkout(self._source_store) as execution:
                    self._execution_store = execution
                    if first:
                        self._bind_cancel()
                        self._publish()
                        self._schedule_background()
                        first = False
                    while True:
                        with self.kernel._background_lock:
                            self._rerun_requested = False
                        # A full idempotent tree pass, with no store I/O under B.
                        progressed = self._drive_pass()
                        with self.kernel._background_lock:
                            if self._rerun_requested or progressed:
                                continue
                            break
                # Release the connection before publishing done. A wake during
                # connection cleanup still belongs to this worker and gets a real
                # pass on a fresh connection, without creating a successor.
                with self.kernel._background_lock:
                    if self._rerun_requested:
                        continue
                    with self._condition:
                        self._done = True
                        self._condition.notify_all()
                    break
        except BaseException as exc:
            self._error = exc
        finally:
            if self.runtime.hook is self._owned_hook:
                self.runtime.hook = self._original_hook
            with self.kernel._background_lock, self._condition:
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

        committed_guard(self._source_store)
        self.join(timeout)
        if not self._done:
            raise TimeoutError("session execution is still active")
        if self._error:
            raise self._error
        with checkout(self._source_store, role="projection") as projection:
            snapshot = projection.read_state(self.session_id)
            state = snapshot[0]
            tid = self.run_id if self.run_id in state.turns else state.active_turn_id or next(reversed(state.turns), "")
            if not tid:
                raise Conflict("session has no admitted turn")
            self.run_id = tid
            result = project_result(projection, self.session_id, tid, runtime=self.runtime, snapshot=snapshot)
            waits = self.kernel._waits_on(projection, self.session_id, self.runtime, snapshot, frozenset())
        if waits:
            result.metadata["session_waits"] = [{"session_id": sid, **wait} for sid, _, wait in waits]
            result.raw_result.wait_reason = waits[0][2].get("question", result.raw_result.wait_reason)
        result._session_driver = self.kernel
        return result

    def cancel(self, reason: str = "") -> bool:
        del reason
        with checkout(self._source_store, role="projection") as admission, admission.atomic() as tx:
            state, _, _ = admission.read_state(self.session_id)
            if state.active_turn_id != self.run_id:
                return False
            receipt = tx.push(self.session_id, InboxItem(f"cancel/{self.run_id}", "control", {"action": "cancel"}, self.run_id))
            cancel_children(admission, tx, state, self.session_id, self.run_id)
        self.kernel._wake(self.session_id)
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
