from __future__ import annotations

from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4
from weakref import ReferenceType, ref

from vv_agent.events import RunEvent
from vv_agent.result import RunResult
from vv_agent.session.records import InboxItem

if TYPE_CHECKING:
    from vv_agent.session.runtime import Runtime
    from vv_agent.session.store import SessionStore
    from vv_agent.session.surfaces import SessionDriver, _ProjectionRuntime

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


@dataclass(frozen=True, slots=True)
class _StreamEvent:
    watermark: int
    ordinal: int
    event: RunEvent


class RunHandleController(Protocol):
    def steer(self, message: str) -> None: ...

    def follow_up(self, message: str) -> None: ...


class RunHandle:
    """An independent cursor over one durable session turn.

    The handle submits inputs and reads fresh client-store snapshots.  It never
    owns a drive store, a drive thread, or a worker-visible execution state.
    Provider deltas are retained as a temporary notification stream because
    they are deliberately absent from the durable ledger.
    """

    def __init__(
        self,
        kernel: SessionDriver,
        sid: str,
        tid: str,
        runtime: Runtime | None,
        consumer: str | None,
        *,
        handler_version: str | None = None,
    ) -> None:
        self.kernel, self.session_id, self.run_id, self.consumer = kernel, sid, tid, consumer
        self._notify = Event()
        self._stream_mailboxes: list[ReferenceType[SimpleQueue[_StreamEvent]]] = []
        self._stream_ordinal = 0
        self._pending_submission = None
        self._runtime_reservation: object | None = None
        self._owns_driver = False
        if runtime is not None:
            kernel._register_runtime(sid, runtime, tid=tid, armed=True, handler_version=handler_version)

    @property
    def runtime(self) -> Runtime | _ProjectionRuntime:
        return self.kernel._runtime_view(self.session_id)

    def start(self) -> None:
        """Arm, submit any held input, then wake the sole LocalWorker owner."""

        self.kernel._arm_handle(self)

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else monotonic() + timeout
        while not self.done():
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                return
            self._notify.wait(0.05 if remaining is None else min(0.05, remaining))
            self._notify.clear()

    def close(self) -> None:
        """Release an explicitly owned Runner/SessionDriver lifecycle."""

        self.notify()
        self.kernel._abandon_handle(self)
        if self._owns_driver:
            self.kernel.close()

    def __enter__(self) -> RunHandle:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def _add_stream_event(self, event: RunEvent, watermark: int) -> None:
        if event.run_id != self.run_id or event.session_id != self.session_id:
            return
        self._stream_ordinal += 1
        entry = _StreamEvent(watermark, self._stream_ordinal, event)
        for mailbox_ref in tuple(self._stream_mailboxes):
            mailbox = mailbox_ref()
            if mailbox is not None:
                mailbox.put(entry)
        self._notify.set()

    def notify(self) -> None:
        self._notify.set()

    def _durable_event_page(
        self,
        after_seq: int,
        *,
        store: SessionStore | None = None,
    ) -> tuple[str, list[RunEvent], int, bool, bool, BaseException | None]:
        return self.kernel._handle_event_page(
            self.session_id,
            self.run_id,
            after_seq=after_seq,
            store=store,
        )

    def events(self) -> Generator[RunEvent, None, None]:
        """Replay durable events plus the independent transient delta stream."""

        mailbox: SimpleQueue[_StreamEvent] = SimpleQueue()
        mailbox_ref = ref(mailbox)
        self._stream_mailboxes = [value for value in self._stream_mailboxes if value() is not None]
        self._stream_mailboxes.append(mailbox_ref)
        return self._iter_events(mailbox, mailbox_ref)

    def _iter_events(
        self,
        mailbox: SimpleQueue[_StreamEvent],
        mailbox_ref: ReferenceType[SimpleQueue[_StreamEvent]],
    ) -> Generator[RunEvent, None, None]:
        record_cursor = 0
        pending_stream: list[_StreamEvent] = []
        try:
            while True:
                self.run_id, durable, next_record_cursor, has_more, quiescent, error = self._durable_event_page(
                    record_cursor,
                    store=self.kernel.store,
                )
                record_cursor = next_record_cursor
                while True:
                    try:
                        pending_stream.append(mailbox.get_nowait())
                    except Empty:
                        break
                ready_stream = [entry for entry in pending_stream if entry.watermark <= record_cursor]
                if ready_stream:
                    ready_ids = {id(entry) for entry in ready_stream}
                    pending_stream[:] = [entry for entry in pending_stream if id(entry) not in ready_ids]
                merged: list[tuple[int, int, int, RunEvent, int | None]] = [
                    (event.metadata["session_seq"], 0, index, event, None) for index, event in enumerate(durable)
                ]
                merged.extend((entry.watermark, 1, entry.ordinal, entry.event, index) for index, entry in enumerate(ready_stream))
                emitted = False
                for _, _, _, event, _ in sorted(merged, key=lambda value: value[:3]):
                    emitted = True
                    yield event
                if quiescent and not has_more and not pending_stream and mailbox.empty():
                    if error is not None:
                        raise error
                    return
                if not emitted:
                    self._notify.wait(0.05)
                    self._notify.clear()
        finally:
            with suppress(ValueError):
                self._stream_mailboxes.remove(mailbox_ref)

    def result(self, timeout: float | None = None) -> RunResult:
        self.kernel._wait_for_handle(self, timeout)
        return self.kernel._project_handle_result(self)

    def cancel(self, reason: str = "") -> bool:
        del reason
        state = self.kernel._handle_snapshot(self.session_id, self.run_id)[0]
        if state is None or state.active_turn_id != self.run_id or self.kernel._is_closed:
            return False
        receipt = self.kernel.push(
            self.session_id,
            InboxItem(f"cancel/{self.run_id}", "control", {"action": "cancel"}, self.run_id),
            actor_subject_id="run-handle",
        )
        return not receipt.replayed

    def done(self) -> bool:
        return self.kernel._handle_done(self.session_id, self.run_id)

    def state(self) -> RunHandleState:
        if not self.done():
            return RunHandleState("running", False)
        try:
            result = self.result(timeout=0)
        except BaseException as exc:
            return RunHandleState("failed", True, error=str(exc))
        cancelled = bool(result.completion_reason and result.completion_reason.value == "cancelled")
        return RunHandleState("cancelled" if cancelled else result.status.value, True, cancelled)

    def approve(self, request_id: str, decision) -> None:
        self.kernel.approve(
            self.session_id,
            self.runtime,
            request_id,
            decision,
            f"approval/{request_id}/answer",
            actor=None,
        )

    def steer(self, message: str) -> None:
        generation = self.kernel._handle_steer_generation(self.session_id, self.run_id)
        self.kernel.push(
            self.session_id,
            InboxItem(uuid4().hex, "steer", {"content": message}, self.run_id, generation),
            actor_subject_id="run-handle",
        )

    def follow_up(self, message: str) -> None:
        self.kernel.push(
            self.session_id,
            InboxItem(uuid4().hex, "follow_up", {"content": message}),
            actor_subject_id="run-handle",
        )

    def resume(self, state_or_token=None, payload=None) -> RunResult:
        if state_or_token is not None or payload is not None:
            raise ValueError("resume uses the retained session and turn without new input")
        from vv_agent.session.surfaces import resume_turn

        return resume_turn(self.session_id, self.run_id, tenant_id=self.kernel.tenant_id)


__all__ = ["RunHandle", "RunHandleController", "RunHandleState"]
