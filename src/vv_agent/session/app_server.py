"""App Server adapter and disposable snapshots from kernel records; no thread ledger."""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.run_adapter import StartedTurn, TurnResumeError
from vv_agent.app_server.run_adapter import _RunAdapterFormatting as RunAdapter
from vv_agent.app_server.thread_store import ThreadRecord, ThreadSnapshot, TurnRecord
from vv_agent.app_server.usage_projection import task_token_usage_to_wire
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.events import HostInteractionRequestedEvent, RunStateChangedEvent
from vv_agent.interaction import derive_controller_command_id
from vv_agent.types import AgentStatus, Message

from .projection import project_records
from .records import InboxItem, SessionSpec
from .reducer import ExecutionState
from .result import project_result
from .store import Conflict
from .surfaces import SessionDriver


class ThreadStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PROVIDER_RUNNING = "provider_running"
    CHILD_RUNNING = "child_running"
    USER_WAIT = "user_wait"
    APPROVAL_WAIT = "approval_wait"
    RETRY_BACKOFF = "retry_backoff"
    SUSPENDED = "suspended"
    TERMINAL = "terminal"
    ARCHIVED = "archived"
    CLOSED = "closed"


def project_thread_status(state: ExecutionState, *, pending: bool = False, now_ms: int | None = None) -> ThreadStatus:
    if now_ms is None:
        import time

        now_ms = int(time.time() * 1000)
    if state.closed:
        return ThreadStatus.CLOSED
    if state.archived:
        return ThreadStatus.ARCHIVED
    tid = state.active_turn_id
    if not tid:
        return ThreadStatus.RUNNING if pending else ThreadStatus.TERMINAL if state.turns else ThreadStatus.IDLE
    if state.turns[tid].suspended:
        return ThreadStatus.SUSPENDED
    kinds = {w["handle"]["kind"] for (oid, _), w in state.waits.items() if state.operations[oid].turn_id == tid}
    for kind, status in (
        ("approval", ThreadStatus.APPROVAL_WAIT),
        ("user", ThreadStatus.USER_WAIT),
        ("child", ThreadStatus.CHILD_RUNNING),
        ("provider", ThreadStatus.PROVIDER_RUNNING),
    ):
        if kind in kinds or (kind == "user" and state.turns[tid].wait):
            return status
    for op in state.operations.values():
        if op.turn_id != tid or op.state == "completed":
            continue
        attempt = op.attempts[max(op.attempts)]
        retry_at = attempt.unknown._payload["retry_at_ms"] if attempt.unknown else attempt.plan._payload["not_before_ms"]
        if retry_at is not None and retry_at > now_ms:
            return ThreadStatus.RETRY_BACKOFF
        if op.kind == "model" and attempt.state == "started":
            return ThreadStatus.PROVIDER_RUNNING
    return ThreadStatus.RUNNING


def _content(
    adapter: RunAdapter, input: list[dict[str, Any]], metadata: dict[str, Any], kind: str, expected_turn_id: str | None
) -> dict[str, Any]:
    text = adapter._prompt_from_input(input)
    messages = [Message("user", text)]
    messages.extend(
        Message("user", "", image_url=item["url"])
        for item in input
        if item.get("type") == "image" and isinstance(item.get("url"), str)
    )
    return {
        "text": text,
        "messages": [m.to_dict() for m in messages],
        "app_server": {"input": input, "metadata": metadata, "kind": kind, "expected_turn_id": expected_turn_id},
    }


class SubmissionConflict(ValueError):
    pass


class TurnMismatch(ValueError):
    pass


@dataclass
class _NotificationPosition:
    seq: int
    due: float = field(default_factory=time.monotonic)
    delay: float = 0.05
    warning_at: float = 0
    lock: Any = field(default_factory=threading.RLock)


class _KernelThreadStore:
    def __init__(self, kernel: SessionDriver) -> None:
        self.kernel = kernel
        self.runtimes: dict[str, Any] = {}

    def create_thread_receipt(self, *, input_id, agent_key, cwd=None, metadata=None):
        with self.kernel.store.atomic() as tx:
            receipt = tx.create(
                SessionSpec(
                    input_id,
                    "local",
                    cwd or ".",
                    attributes={"app_server": {"agent_key": agent_key, "cwd": cwd, "metadata": metadata or {}}},
                ),
                consumers=("events", "app_server", "traces"),
            )
        return {
            "threadId": input_id,
            "agentKey": agent_key,
            "cwd": cwd,
            "status": self.read_thread(input_id).thread.status,
            "accepted": True,
            "replayed": receipt.replayed,
        }

    def _thread_metadata(self, thread_id, state, records):
        attributes = records[0].record.payload["attributes"].get("app_server")
        if attributes is None:
            raise KeyError(thread_id)
        pending = [
            x.item
            for x in self.kernel.store.peek_inbox(thread_id)
            if x.item.kind in {"user", "follow_up"} and x.item.target_turn_id is None
        ]
        pending.extend(
            InboxItem(**r._payload["input"])
            for key, r in state.applied_inputs.items()
            if r._payload["disposition"] == "queued" and key not in state.admitted_inputs
        )
        archived = next(
            (
                r.created_ms / 1000
                for r in records
                if r.record.kind == "input_applied"
                and r.record._payload["disposition"] == "applied"
                and r.record._payload["input"]["kind"] == "control"
                and r.record._payload["input"]["payload"]["action"] == "archive"
            ),
            None,
        )
        return ThreadRecord(
            thread_id,
            attributes["agent_key"],
            attributes["cwd"],
            records[0].created_ms / 1000,
            records[-1].created_ms / 1000,
            archived,
            project_thread_status(state, pending=bool(pending)).value,
            state.active_turn_id or (f"{thread_id}/turn/{pending[-1].input_id}" if pending else None),
            attributes["metadata"],
        )

    def read_thread(self, thread_id):
        try:
            state, records, _ = self.kernel.store.read_state(thread_id)
        except KeyError:
            raise KeyError(thread_id) from None
        thread = self._thread_metadata(thread_id, state, records)
        turns = []
        runtime = self.runtimes.get(thread_id)
        for tid, turn in state.turns.items():
            start = next(r for r in records if r.record == turn.start)
            initial = state.applied_inputs[turn.start._payload["input_ids"][0]]._payload["input"]
            content = initial["payload"]["content"]
            wire_input = content.get("app_server", {}).get("input", []) if isinstance(content, dict) else []
            terminal = next((r for r in records if r.record.kind == "turn_ended" and r.record.turn_id == tid), None)
            status = (
                "completed"
                if terminal and terminal.record._payload["status"] == "completed"
                else "failed"
                if terminal
                else ("interrupted" if state.turns[tid].wait or state.turns[tid].suspended or state.waits else "running")
            )
            projected = project_result(self.kernel.store, thread_id, tid, runtime=runtime, snapshot=(state, records, 0))
            result = _result_fields(projected)
            turns.append(
                TurnRecord(
                    tid,
                    thread_id,
                    tid,
                    status,
                    start.created_ms / 1000,
                    terminal.created_ms / 1000 if terminal else None,
                    wire_input,
                    result,
                )
            )
        pending = []
        for stored in self.kernel.store.peek_inbox(thread_id):
            if stored.item.kind in {"user", "follow_up"} and stored.item.target_turn_id is None:
                pending.append((stored.item, stored.received_ms))
        for key, r in state.applied_inputs.items():
            if r._payload["disposition"] == "queued" and key not in state.admitted_inputs:
                pending.append((InboxItem(**r._payload["input"]), records[-1].created_ms))
        for item, received in pending:
            tid = f"{thread_id}/turn/{item.input_id}"
            content = item.payload["content"]
            turns.append(
                TurnRecord(tid, thread_id, tid, input=content.get("app_server", {}).get("input", []), started_at=received / 1000)
            )
        items = []
        for event in project_records(records):
            projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
            if projection.item:
                items.append(projection.item)
        return ThreadSnapshot(thread, turns, items)

    def list_threads(self, *, include_archived=False, authorize=None, archived=None, offset=0, limit=None):
        candidates = []
        after = None
        while ids := self.kernel.store.list_sessions(after=after):
            for sid in ids:
                if authorize is not None and not authorize(sid):
                    continue
                page = self.kernel.store.read(sid, limit=1)
                if not page.records[0].record._payload["attributes"].get("app_server"):
                    continue
                is_archived = any(
                    r.record._payload["disposition"] == "applied" and r.record._payload["input"]["payload"]["action"] == "archive"
                    for r in self.kernel.store.control_records(sid, through_seq=page.head_seq)
                )
                if (include_archived or not is_archived) and (archived is None or archived is is_archived):
                    candidates.append((page.records[0].created_ms, sid))
            after = ids[-1]
        candidates.sort()
        end = None if limit is None else offset + limit
        threads = []
        for _, sid in candidates[offset:end]:
            state, records, _ = self.kernel.store.read_state(sid)
            threads.append(self._thread_metadata(sid, state, records))
        return threads

    def archive_thread(self, thread_id):
        self.read_thread(thread_id)
        runtime = self.runtimes.get(thread_id)
        self.kernel.control(thread_id, "archive", "app_server/archive", runtime=runtime)

    def append_item(self, item, *, run_event_id=None):
        del run_event_id
        if item not in self.read_thread(item.thread_id).items:
            raise ValueError("App Server items must project kernel records")
        return True

    def update_turn(self, turn_id, *, status, run_id=None, completed_at=None, result=None):
        del status, run_id, completed_at, result
        return next(
            t
            for thread in self.list_threads(include_archived=True)
            for t in self.read_thread(thread.thread_id).turns
            if t.turn_id == turn_id
        )

    def set_active_turn(self, thread_id, active_turn_id, status):
        del thread_id, active_turn_id, status


def _result_fields(result):
    fields: dict[str, Any] = {"tokenUsage": task_token_usage_to_wire(result.token_usage)}
    for name, value in (
        ("finalOutput", result.final_output),
        ("waitReason", "suspended" if result.status is AgentStatus.SUSPENDED else result.wait_reason),
        (
            "completionReason",
            result.completion_reason.value if result.completion_reason and result.status is not AgentStatus.SUSPENDED else None,
        ),
        ("completionToolName", result.completion_tool_name),
        ("partialOutput", result.partial_output),
    ):
        if value is not None:
            fields[name] = value
    if result.raw_result.error:
        fields["error"] = RunAdapter._result_error_text(result.raw_result.error)
    if result.budget_usage:
        fields["budgetUsage"] = result.budget_usage.to_dict()
    if result.budget_exhaustion:
        fields["budgetExhaustion"] = result.budget_exhaustion.to_dict()
    return fields


class _KernelRunAdapter(RunAdapter):
    _store: _KernelThreadStore

    def __init__(self, *, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.kernel.model_resolver = self._host.resolve_model
        self._pumps: list[threading.Thread] = []
        self._notification_lock = threading.RLock()
        self._notification_positions: dict[str, _NotificationPosition] = {}
        self._lease_retries: dict[str, tuple[float, str, int, int, str | None]] = {}
        self._notification_stop = threading.Event()
        self._notification_thread: threading.Thread | None = None
        self.kernel._on_close(self.close)

    def subscribe(self, thread_id, connection_id):
        state, records, _ = self.kernel.store.read_state(thread_id)
        start = state.turns[state.active_turn_id].start if state.active_turn_id else None
        position = next(r.seq for r in records if r.record == start) - 1 if start else records[-1].seq
        with self._notification_lock:
            if not self._router.is_registered(connection_id):
                return
            if thread_id not in self._notification_positions:
                self._notification_positions[thread_id] = _NotificationPosition(position)
            self._state_manager.subscribe(thread_id, connection_id)
            self._ensure_observer()

    def _ensure_observer(self):
        with self._notification_lock:
            if self._notification_thread is None:
                self._notification_thread = threading.Thread(
                    target=self._observe_turns, name="app-server-notifications", daemon=True
                )
                self._notification_thread.start()

    def unsubscribe(self, thread_id, connection_id):
        with self._notification_lock:
            self._state_manager.unsubscribe(thread_id, connection_id)
            if not self._state_manager.subscribers(thread_id):
                self._notification_positions.pop(thread_id, None)

    def unsubscribe_connection(self, connection_id):
        with self._notification_lock:
            self._state_manager.unsubscribe_connection(connection_id)
            for sid in list(self._notification_positions):
                if not self._state_manager.subscribers(sid):
                    del self._notification_positions[sid]

    def _observe_turns(self):
        while not self._notification_stop.wait(0.05):
            now = time.monotonic()
            with self._notification_lock:
                sessions = sorted(
                    ((sid, pos) for sid, pos in self._notification_positions.items() if pos.due <= now),
                    key=lambda entry: entry[1].due,
                )[:32]
                retries = [(sid, retry) for sid, retry in self._lease_retries.items() if retry[0] <= now][:32]
            for sid, retry in retries:
                self._retry_lease(sid, retry)
            for sid, position in sessions:
                changed = False
                try:
                    # Release the turn lock before any publication or transport I/O.
                    with self._state_manager.turn_lock(sid):
                        observe = self._state_manager.active_turn(sid) is None and self._state_manager.subscribers(sid)
                    if observe:
                        changed = self._publish_turns(sid)
                except Exception as exc:
                    if now >= position.warning_at:
                        position.warning_at = now + 1.0
                        self._notify_subscribers(sid, "error/warning", {"message": str(exc), "code": "notification_stream"})
                position.delay = 0.05 if changed else min(1.0, position.delay * 2)
                position.due = time.monotonic() + position.delay

    def _schedule_lease_retry(self, sid, started, connection_id):
        target = started.handle._lease_unavailable_session or sid
        if not started.handle._lease_retry_allowed:
            self._notify_subscribers(
                sid,
                "error/warning",
                {"message": "Execution lease still unavailable after expiry retry", "code": "lease_retry_exhausted"},
            )
            return
        head, epoch, remaining = self.kernel.store.lease_status(target)
        with self._notification_lock:
            self._lease_retries[sid] = (time.monotonic() + remaining / 1000, target, head, epoch, connection_id)
            self._ensure_observer()

    def _retry_lease(self, sid, retry):
        with self._notification_lock:
            if self._lease_retries.get(sid) != retry:
                return
            del self._lease_retries[sid]
        _, target, _, _, _ = retry
        try:
            new_head, new_epoch, remaining = self.kernel.store.lease_status(target)
            if remaining > 0:
                with self._notification_lock:
                    self._lease_retries[sid] = (time.monotonic() + remaining / 1000, target, new_head, new_epoch, None)
                return
            self.recover(self._subscriber(sid), sid, lease_retry=False)
        except Exception as exc:
            self._notify_subscribers(sid, "error/warning", {"message": str(exc), "code": "lease_retry_exhausted"})

    def _subscriber(self, sid):
        return next((cid for cid in self._state_manager.subscribers(sid) if self._router.is_registered(cid)), None)

    def wake(self, thread_id):
        self.recover(self._subscriber(thread_id), thread_id)

    def _publish_turns(self, sid, *, through_seq=None):
        """Each API node tails lifecycle records independently of execution consumers."""
        with self._notification_lock:
            position = self._notification_positions.get(sid)
        if position is None:
            return False
        # Lock order: turn (caller only), publication, then brief notification/store
        # locks separately. Never hold a store or notification lock during projection/I/O.
        with position.lock:
            with self._notification_lock:
                if self._notification_positions.get(sid) is not position:
                    return False
                previous = position.seq
            rows = self.kernel.store.read(sid, after_seq=previous, through_seq=through_seq, limit=256).records
            notifications = []
            snapshot = None
            for stored in rows:
                record = stored.record
                if record.kind == "turn_started":
                    notifications.append(("turn/started", {"threadId": sid, "turnId": record.turn_id}))
                elif record.kind == "turn_ended":
                    if snapshot is None:
                        snapshot = self.kernel.store.read_state(sid)
                    result = project_result(
                        self.kernel.store, sid, record.turn_id, runtime=self._store.runtimes.get(sid), snapshot=snapshot
                    )
                    notifications.append(
                        (
                            "turn/completed",
                            {
                                "threadId": sid,
                                "turnId": record.turn_id,
                                "status": "completed" if record._payload["status"] == "completed" else "failed",
                                **_result_fields(result),
                            },
                        )
                    )
            with self._notification_lock:
                if self._notification_positions.get(sid) is not position:
                    return False
                if rows:
                    position.seq = rows[-1].seq
            for method, params in notifications:
                self._notify_subscribers(sid, method, params)
            return position.seq != previous

    def close(self):
        self._notification_stop.set()
        if self._notification_thread is not None:
            self._notification_thread.join()

    def _binding(self, thread_id, content=None):
        thread = self._store.read_thread(thread_id).thread
        state, _records, _ = self.kernel.store.read_state(thread_id)
        if content is None:
            initial_id = state.turns[state.active_turn_id].start._payload["input_ids"][0] if state.active_turn_id else None
            queued = next(
                (
                    r._payload["input"]
                    for key, r in state.applied_inputs.items()
                    if r._payload["disposition"] == "queued"
                    and key not in state.admitted_inputs
                    and r._payload["input"]["kind"] in {"user", "follow_up"}
                ),
                None,
            )
            pending = next(
                (
                    x.item
                    for x in self.kernel.store.peek_inbox(thread_id)
                    if x.item.kind in {"user", "follow_up"} and x.item.target_turn_id is None
                ),
                None,
            )
            item = (
                self.kernel.store.read_input(thread_id, initial_id).item
                if initial_id
                else (InboxItem(**queued) if queued else pending)
            )
            if item is not None and isinstance(item.payload.get("content"), dict):
                content = item.payload["content"]
        metadata = thread.metadata | (content.get("app_server", {}).get("metadata", {}) if content else {})
        agent = self._host.resolve_agent(AgentResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        config = self._host.build_run_config(RunConfigResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        shared_state = config.shared_state
        if state.turns:
            prior = project_result(
                self.kernel.store, thread_id, next(reversed(state.turns)), runtime=self.kernel.runtime(agent, config)
            )
            shared_state = prior.raw_result.shared_state
        return (
            thread,
            agent,
            replace(config, shared_state=shared_state, initial_messages=None, metadata=config.metadata | metadata),
        )

    def submit_input(self, *, connection_id, thread_id, input_id, expected_turn_id, kind, input, metadata=None):
        content = _content(self, input, metadata or {}, kind, expected_turn_id)
        try:
            with self.kernel.store.atomic() as tx:
                tx.lock_session(thread_id)
                state, _, _ = self.kernel.store.read_state(thread_id)
                stored = self.kernel.store.read_input(thread_id, input_id)
                if stored:
                    retained = stored.item
                    if retained.kind != ("user" if kind == "start" else kind):
                        raise SubmissionConflict("submission_conflict")
                    original = retained.payload["content"]
                    if not isinstance(original, dict) or canonical_json_bytes(original.get("app_server")) != canonical_json_bytes(
                        content["app_server"]
                    ):
                        raise SubmissionConflict("submission_conflict")
                    item = retained
                    content = original
                else:
                    if state.closed or state.archived:
                        raise TurnMismatch("Thread is closed or archived")
                    latest = next(reversed(state.turns), None)
                    if expected_turn_id != (state.active_turn_id if kind == "steer" else latest):
                        raise TurnMismatch("turn_mismatch")
                    if kind == "start" and (
                        state.active_turn_id
                        or any(
                            r._payload["disposition"] == "queued"
                            and key not in state.admitted_inputs
                            and r._payload["input"]["kind"] in {"user", "follow_up"}
                            for key, r in state.applied_inputs.items()
                        )
                        or any(
                            x.item.kind in {"user", "follow_up"} and x.item.target_turn_id is None
                            for x in self.kernel.store.peek_inbox(thread_id)
                        )
                    ):
                        raise TurnMismatch("Thread already has an active turn")
                    if kind != "start" and expected_turn_id is None:
                        raise TurnMismatch("turn_mismatch")
                    item = InboxItem(
                        input_id,
                        "user" if kind == "start" else kind,
                        {"content": content},
                        expected_turn_id if kind == "steer" else None,
                        state.turns[expected_turn_id].start._payload["generation"] if kind == "steer" else None,
                    )
                receipt = self.kernel.push(thread_id, item)
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        tid = expected_turn_id if kind == "steer" else f"{thread_id}/turn/{input_id}"
        payload = {
            "threadId": thread_id,
            "turnId": tid,
            "inputId": input_id,
            "inputSeq": receipt.input_seq,
            "accepted": True,
            "replayed": receipt.replayed,
        }
        return payload, content

    def start_turn(self, *, connection_id, thread_id, input, input_id, expected_turn_id, metadata=None, request_id=None):
        payload, _content = self.submit_input(
            connection_id=connection_id,
            thread_id=thread_id,
            input_id=input_id,
            expected_turn_id=expected_turn_id,
            kind="start",
            input=input,
            metadata=metadata,
        )
        if request_id is not None:
            self._router.send_response(connection_id, request_id, payload)
        self.recover(connection_id, thread_id)
        return payload

    def _start(self, connection_id, thread_id, content, *, input_id=None, request_id=None, lease_retry=True):
        with self._state_manager.turn_lock(thread_id):
            active = self._state_manager.active_turn(thread_id)
            if active is None:
                return self._start_locked(
                    connection_id, thread_id, content, input_id=input_id, request_id=request_id, lease_retry=lease_retry
                )
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": active.turn_id, "runId": active.turn_id, "status": "running"},
                )
            return None

    def _start_locked(self, connection_id, thread_id, content, *, input_id=None, request_id=None, lease_retry=True):
        if connection_id is not None and self._router.is_registered(connection_id):
            self.subscribe(thread_id, connection_id)
        else:
            connection_id = self._subscriber(thread_id)
        thread, agent, config = self._binding(thread_id, content)
        with self.kernel.store.atomic():
            identities = [t.turn_id for t in self._store.read_thread(thread_id).turns]
            input_id = input_id or "resume"
            tid = f"{thread_id}/turn/{input_id}" if content is not None else thread.active_turn_id or identities[-1]
            config = self._with_app_server_controls(config, connection_id=connection_id, thread_id=thread_id, turn_id=tid)
            host_stream = config.stream

            def stream(event):
                if host_stream:
                    host_stream(event)
                if event.type == "run_started":
                    self._publish_turns(thread_id, through_seq=event.metadata["session_seq"])
                projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
                projection = self._hydrate_host_interaction_projection(thread_id, event, projection)
                if projection.notification_method:
                    self._notify_subscribers(thread_id, projection.notification_method, projection.notification_params)
                for method, params in projection.additional_notifications:
                    self._notify_subscribers(thread_id, method, params)

            # SessionRunEventStore commits its cursor inside consume(); pump delivery
            # keeps host callbacks and transport writes outside that transaction.
            config = replace(config, stream=None)
            handle = self.kernel.start(
                thread_id, agent, config, content, input_id=input_id, consumer="app_server", autostart=False
            )
            handle._wait_for_lease = False
            handle._lease_retry_allowed = lease_retry
            handle.runtime.wake = lambda _sid: self.wake(thread_id)
        self._store.runtimes[thread_id] = handle.runtime
        turn = next(t for t in self._store.read_thread(thread_id).turns if t.turn_id == handle.run_id)
        started = StartedTurn(thread, turn, handle)
        self._state_manager.set_active_turn(thread_id=thread_id, turn_id=turn.turn_id, handle=handle)
        status_payload = self.public_thread_status(thread_id)
        if request_id is not None:
            self._router.send_response(
                connection_id,
                request_id,
                {
                    "threadId": thread_id,
                    "turnId": turn.turn_id,
                    "status": "running",
                    **({"runId": turn.turn_id} if content is None else {}),
                },
            )
        self._notify_subscribers(thread_id, "thread/status/changed", status_payload)
        pump = threading.Thread(target=self._pump_events, args=(connection_id, started, stream), name="session-app-server-pump")
        try:
            handle.start()
            pump.start()
            self._pumps.append(pump)
        except Exception as exc:
            self._state_manager.clear_active_turn(thread_id, turn.turn_id, handle=handle)
            self._notify_subscribers(thread_id, "error/warning", {"message": str(exc), "code": "drive_start"})
        return started

    def recover(self, connection_id, thread_id, *, follow_up_only=False, lease_retry=True):
        try:
            with self._state_manager.turn_lock(thread_id):
                if self._state_manager.active_turn(thread_id) is None:
                    state, _, _ = self.kernel.store.read_state(thread_id)
                    kinds = {"follow_up"} if follow_up_only else {"user", "follow_up"}
                    queued = any(
                        r._payload["disposition"] == "queued"
                        and key not in state.admitted_inputs
                        and r._payload["input"]["kind"] in kinds
                        for key, r in state.applied_inputs.items()
                    )
                    if (
                        state.active_turn_id
                        or queued
                        or any(
                            x.item.kind in kinds and x.item.target_turn_id is None
                            for x in self.kernel.store.peek_inbox(thread_id)
                        )
                    ):
                        self._start(connection_id, thread_id, None, lease_retry=lease_retry)
        except Exception as exc:
            self._notify_subscribers(thread_id, "error/warning", {"message": str(exc), "code": "drive_start"})

    def _pump_events(self, connection_id, started, stream):
        result, error = None, None
        try:
            for _event in started.handle.events():
                with suppress(Exception):
                    stream(_event)
            result = started.handle.result(timeout=0)
        except BaseException as exc:
            error = exc
        self._complete_turn(connection_id, started, result=result, error=error)

    def resume_turn(self, *, connection_id, thread_id, turn_id, request_id=None):
        snapshot = self._store.read_thread(thread_id)
        if snapshot.thread.status == ThreadStatus.CLOSED:
            raise TurnResumeError("Thread is closed")
        turn = next((t for t in snapshot.turns if t.turn_id == turn_id), None)
        if turn is None:
            raise TurnResumeError("Turn does not belong to the requested thread")
        state, _, _ = self.kernel.store.read_state(thread_id)
        if state.turns[turn_id].ended:
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": turn.status, **turn.result},
                )
            return
        if state.active_turn_id != turn_id:
            raise TurnResumeError("Thread has a different active turn")
        if self._state_manager.active_turn(thread_id) is not None:
            if request_id is not None:
                self._router.send_response(
                    connection_id, request_id, {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": "running"}
                )
            return
        self._start(connection_id, thread_id, None, request_id=request_id)

    def public_thread_status(self, thread_id):
        import time

        state, records, _ = self.kernel.store.read_state(thread_id)
        pending = any(
            x.item.kind in {"user", "follow_up"} and x.item.target_turn_id is None
            for x in self.kernel.store.peek_inbox(thread_id)
        )
        pending = pending or any(
            r._payload["disposition"] == "queued" and key not in state.admitted_inputs for key, r in state.applied_inputs.items()
        )
        status = project_thread_status(state, pending=pending, now_ms=int(time.time() * 1000))
        wait_kind = {
            "provider_running": "provider",
            "child_running": "child",
            "user_wait": "user",
            "approval_wait": "approval",
            "retry_backoff": "retry",
        }.get(status.value)
        interaction = None
        if state.active_turn_id:
            waits = [{"session_id": sid, **wait} for sid, _, wait in self.kernel.waits(thread_id)]
            if waits:
                waits.sort(key=lambda w: "request_id" not in w)
                interaction = self._public_waits(waits)[0]
        return {
            "threadId": thread_id,
            "status": status.value,
            "phase": state.phase,
            "waitKind": wait_kind,
            "nextDriveAt": state.next_drive_ms,
            "pendingInteraction": interaction,
            "unconfirmedCount": sum(
                any(a.state != "completed" for a in op.attempts.values())
                for op in state.operations.values()
                if op.turn_id == (state.active_turn_id or next(reversed(state.turns), None))
            ),
            "projectionWatermark": records[-1].seq,
        }

    @staticmethod
    def _public_waits(waits):
        return [
            {
                "sessionId": wait["session_id"],
                "turnId": wait["turn_id"],
                **({"prompt": wait["question"]} if "question" in wait else {}),
                **({"interactionId": wait["interaction_id"]} if "interaction_id" in wait else {}),
                **({"requestId": wait["request_id"]} if "request_id" in wait else {}),
            }
            for wait in waits
        ]

    def select_model(self, *, thread_id, input_id, selection):
        self._store.read_thread(thread_id)
        try:
            self.kernel.push(thread_id, InboxItem(input_id, "model_selection", {"selection": selection}))
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        return {"threadId": thread_id, "inputId": input_id, "accepted": True}

    def controller_action(self, *, thread_id, turn_id, action_id, action, connection_id):
        from vv_agent.app_server.protocol.identity import client_id

        for identity, name in ((thread_id, "threadId"), (turn_id, "turnId"), (action_id, "actionId")):
            self._validate_public_controller_action_identity(identity, name)
        try:
            client_id(action_id, "actionId")
        except ValueError as exc:
            raise TurnResumeError(str(exc)) from exc
        self._validate_public_controller_action(action)
        input_id = "controller/" + derive_controller_command_id(thread_id, turn_id, action_id)
        previous = self.kernel._retained_reply(thread_id, input_id)
        if previous:
            retained = previous[1]
            original = (
                {"kind": "respond", "message": {"role": "user", "content": retained.payload["content"]["text"]}}
                if retained.kind == "user"
                else {"kind": retained.payload["action"]}
                if retained.kind == "control"
                else None
            )
            if canonical_json_bytes(original) != canonical_json_bytes(action):
                raise SubmissionConflict("submission_conflict")
            return {"threadId": thread_id, "turnId": turn_id, "actionId": action_id, "accepted": True, "status": "running"}
        state, _, _ = self.kernel.store.read_state(thread_id)
        if turn_id not in state.turns:
            raise TurnResumeError("Turn does not belong to the requested thread")
        if state.active_turn_id is not None and state.active_turn_id != turn_id:
            raise TurnResumeError("Thread has a different active turn")
        runtime = self._store.runtimes.get(thread_id)
        if runtime is None:
            _, agent, config = self._binding(thread_id)
            runtime = self.kernel.runtime(agent, config)
        try:
            if action["kind"] == "respond":
                receipt = self.kernel.answer(thread_id, runtime, action["message"]["content"], input_id)
            else:
                receipt = self.kernel.push(
                    thread_id,
                    InboxItem(
                        input_id,
                        "control",
                        {"action": action["kind"]},
                        turn_id,
                        state.turns[turn_id].start._payload["generation"],
                    ),
                )
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        except ValueError as exc:
            raise TurnResumeError(str(exc)) from exc
        active = self._state_manager.active_turn(thread_id)
        if active is None and not receipt.replayed and state.active_turn_id:
            self.recover(connection_id, thread_id)
        return {"threadId": thread_id, "turnId": turn_id, "actionId": action_id, "accepted": True, "status": "running"}

    def resolve_approval(self, *, connection_id, thread_id, turn_id, request_id, decision):
        from vv_agent.approval import ApprovalDecision

        state, _, _ = self.kernel.store.read_state(thread_id)
        previous = self.kernel._retained_reply(thread_id, f"approval/{request_id}/answer")
        if state.active_turn_id != turn_id and not previous:
            raise TurnResumeError("Turn does not belong to the active thread")
        _, agent, config = self._binding(thread_id)
        runtime = self._store.runtimes.get(thread_id) or self.kernel.runtime(agent, config)
        try:
            self.kernel.approve(thread_id, runtime, request_id, ApprovalDecision(**decision), f"approval/{request_id}/answer")
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        active = self._state_manager.active_turn(thread_id)
        if active is None:
            self.recover(connection_id, thread_id)

    def _hydrate_host_interaction_projection(self, started, event, projection):
        if isinstance(event, (HostInteractionRequestedEvent, RunStateChangedEvent)):
            return replace(
                projection,
                notification_method="thread/status/changed",
                notification_params=self.public_thread_status(started),
            )
        return projection

    def _complete_turn(self, connection_id, started, *, result, error):
        sid, tid = started.thread.thread_id, started.handle.run_id
        with self._state_manager.turn_lock(sid):
            if not self._state_manager.clear_active_turn(sid, started.turn.turn_id, handle=started.handle):
                return
            if started.handle._lease_unavailable:
                try:
                    self._publish_turns(sid)
                    self._schedule_lease_retry(sid, started, connection_id)
                except Exception as exc:
                    self._notify_subscribers(sid, "error/warning", {"message": str(exc), "code": "lease_retry_exhausted"})
                return
            if error is not None:
                from .store import LeaseRetryExhausted

                code = error.code if isinstance(error, LeaseRetryExhausted) else "event_stream"
                self._notify_subscribers(sid, "error/warning", {"message": str(error), "code": code})
                self._publish_turns(sid)
                state, _, _ = self.kernel.store.read_state(sid)
                if tid not in state.turns or not state.turns[tid].ended:
                    self._notify_subscribers(
                        sid, "turn/completed", {"threadId": sid, "turnId": tid, "status": "failed", "error": str(error)}
                    )
                return
            self._notify_subscribers(sid, "thread/status/changed", self.public_thread_status(sid))
            self._publish_turns(sid)
            state, _, _ = self.kernel.store.read_state(sid)
            incoming = [x.item for x in self.kernel.store.peek_inbox(sid)]
            if state.active_turn_id:
                if any(x.kind in {"control", "approval_answer", "user", "child_result"} for x in incoming):
                    self.recover(connection_id, sid)
                    return
                turn = next(t for t in self._store.read_thread(sid).turns if t.turn_id == tid)
                self._notify_subscribers(
                    sid, "turn/completed", {"threadId": sid, "turnId": tid, "status": turn.status, **turn.result}
                )
            else:
                self.recover(connection_id, sid, follow_up_only=True)

    def join(self):
        for pump in self._pumps:
            pump.join()
        for sid in list(self._notification_positions):
            while self._publish_turns(sid):
                pass
