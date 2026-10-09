"""App Server adapter and disposable snapshots from kernel records; no thread ledger."""

from __future__ import annotations

import threading
from dataclasses import replace
from enum import StrEnum
from typing import Any

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.run_adapter import StartedTurn, TurnResumeError
from vv_agent.app_server.run_adapter import _RunAdapterFormatting as RunAdapter
from vv_agent.app_server.thread_store import ThreadRecord, ThreadSnapshot, TurnRecord
from vv_agent.app_server.usage_projection import task_token_usage_to_wire
from vv_agent.events import HostInteractionRequestedEvent
from vv_agent.interaction import derive_controller_command_id
from vv_agent.types import AgentStatus, Message

from .projection import project_records
from .records import InboxItem
from .reducer import ExecutionState
from .result import project_result
from .store import Conflict
from .surfaces import SessionDriver


class ThreadStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    ARCHIVED = "archived"
    CLOSED = "closed"


def project_thread_status(state: ExecutionState, *, pending: bool = False) -> ThreadStatus:
    if state.closed:
        return ThreadStatus.CLOSED
    if state.archived:
        return ThreadStatus.ARCHIVED
    if state.active_turn_id and (
        state.turns[state.active_turn_id].suspended
        or state.turns[state.active_turn_id].wait is not None
        or any(state.operations[oid].turn_id == state.active_turn_id for oid, _ in state.waits)
    ):
        return ThreadStatus.INTERRUPTED
    return ThreadStatus.RUNNING if state.active_turn_id or pending else ThreadStatus.IDLE


def _content(adapter: RunAdapter, input: list[dict[str, Any]], metadata: dict[str, Any], owner: str) -> dict[str, Any]:
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
        "app_server": {"input": input, "metadata": metadata, "owner": owner},
    }


class _KernelThreadStore:
    def __init__(self, kernel: SessionDriver) -> None:
        self.kernel = kernel
        self.runtimes: dict[str, Any] = {}

    def create_thread(self, *, agent_key, cwd=None, metadata=None):
        with self.kernel.store.atomic():
            ids = self.kernel.store.list_sessions(limit=1_000_000)
            number = max((int(s.removeprefix("thread_")) for s in ids if s.startswith("thread_") and s[7:].isdigit()), default=0)
            sid = f"thread_{number + 1}"
            self.kernel.create(sid, cwd or ".", {"app_server": {"agent_key": agent_key, "cwd": cwd, "metadata": metadata or {}}})
        return self.read_thread(sid).thread

    def _events(self, sid):
        _, records, _ = self.kernel.store.read_state(sid)
        return project_records(records)

    def read_thread(self, thread_id):
        try:
            state, records, _ = self.kernel.store.read_state(thread_id)
        except KeyError:
            raise KeyError(thread_id) from None
        attributes = records[0].record.payload["attributes"].get("app_server")
        if attributes is None:
            raise KeyError(thread_id)
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
                else ("interrupted" if project_thread_status(state) == ThreadStatus.INTERRUPTED else "running")
            )
            projected = project_result(self.kernel.store, thread_id, tid, runtime=runtime)
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
        active = state.active_turn_id or (turns[-1].turn_id if pending else None)
        thread = ThreadRecord(
            thread_id,
            attributes["agent_key"],
            attributes["cwd"],
            records[0].created_ms / 1000,
            records[-1].created_ms / 1000,
            archived,
            project_thread_status(state, pending=bool(pending)).value,
            active,
            attributes["metadata"],
        )
        items = []
        for event in self._events(thread_id):
            projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
            if projection.item:
                items.append(projection.item)
        return ThreadSnapshot(thread, turns, items)

    def list_threads(self, *, include_archived=False):
        threads = []
        after = None
        while ids := self.kernel.store.list_sessions(after=after):
            for sid in ids:
                try:
                    thread = self.read_thread(sid).thread
                except KeyError:
                    continue
                if include_archived or thread.archived_at is None:
                    threads.append(thread)
            after = ids[-1]
        return sorted(threads, key=lambda t: (t.created_at, t.thread_id))

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
        self._pumps: list[threading.Thread] = []

    def _binding(self, thread_id, content=None):
        thread = self._store.read_thread(thread_id).thread
        metadata = thread.metadata | (content.get("app_server", {}).get("metadata", {}) if content else {})
        agent = self._host.resolve_agent(AgentResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        config = self._host.build_run_config(RunConfigResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        state, _records, _ = self.kernel.store.read_state(thread_id)
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

    def start_turn(self, *, connection_id, thread_id, input, metadata=None, request_id=None):
        content = _content(self, input, metadata or {}, connection_id)
        return self._start(connection_id, thread_id, content, request_id=request_id)

    def _start(self, connection_id, thread_id, content, *, request_id=None, _response_connection=None):
        thread, agent, config = self._binding(thread_id, content)
        with self.kernel.store.atomic():
            identities = [t.turn_id for t in self._store.read_thread(thread_id).turns]
            input_id = f"turn_{len(identities) + 1}"
            tid = f"{thread_id}/turn/{input_id}" if content is not None else thread.active_turn_id or identities[-1]
            config = self._with_app_server_controls(config, connection_id=connection_id, thread_id=thread_id, turn_id=tid)
            host_stream = config.stream

            def stream(event):
                if host_stream:
                    host_stream(event)
                projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
                projection = self._hydrate_host_interaction_projection(None, event, projection)
                if projection.notification_method:
                    self._notify_subscribers(thread_id, projection.notification_method, projection.notification_params)
                for method, params in projection.additional_notifications:
                    self._notify_subscribers(thread_id, method, params)

            config = replace(config, stream=stream)
            handle = self.kernel.start(
                thread_id, agent, config, content, input_id=input_id, consumer="app_server", autostart=False
            )
        self._store.runtimes[thread_id] = handle.runtime
        turn = next(t for t in self._store.read_thread(thread_id).turns if t.turn_id == handle.run_id)
        started = StartedTurn(thread, turn, handle)
        self._state_manager.set_active_turn(thread_id=thread_id, turn_id=turn.turn_id, handle=handle)
        if request_id is not None:
            self._router.send_response(
                _response_connection or connection_id,
                request_id,
                {
                    "threadId": thread_id,
                    "turnId": turn.turn_id,
                    "status": "running",
                    **({"runId": turn.turn_id} if content is None else {}),
                },
            )
        self._notify_subscribers(thread_id, "thread/status/changed", self.public_thread_status(thread_id))
        self._notify_subscribers(thread_id, "turn/started", {"threadId": thread_id, "turnId": turn.turn_id})
        pump = threading.Thread(target=self._pump_events, args=(connection_id, started), name="session-app-server-pump")
        self._pumps.append(pump)
        handle.start()
        pump.start()
        return started

    def recover(self, connection_id, thread_id):
        del connection_id
        if self._state_manager.active_turn(thread_id) is None:
            state, _, _ = self.kernel.store.read_state(thread_id)
            if state.active_turn_id:
                owner = self._owner(thread_id, state.active_turn_id)
                self._start(owner, thread_id, None)

    def _pump_events(self, connection_id, started):
        result, error = None, None
        try:
            for _event in started.handle.events():
                pass
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
        owner = self._owner(thread_id, turn_id)
        if not self._router.is_registered(owner):
            raise TurnResumeError("Retained turn owner is disconnected")
        self._start(owner, thread_id, None, request_id=request_id, _response_connection=connection_id)

    def public_thread_status(self, thread_id):
        thread = self._store.read_thread(thread_id).thread
        state, _, _ = self.kernel.store.read_state(thread_id)
        payload: dict[str, Any] = {
            "threadId": thread_id,
            "status": thread.status,
        }
        if state.phase == "suspended":
            return payload | {"waitReason": "suspended"}
        if state.active_turn_id:
            runtime = self._store.runtimes.get(thread_id)
            if runtime is None:
                _, agent, config = self._binding(thread_id)
                runtime = self.kernel.runtime(agent, config)
            waits = [{"session_id": sid, **wait} for sid, _, wait in self.kernel.waits(thread_id, runtime)]
            if waits:
                payload.update(waitReason="host_interaction", interactions=self._public_waits(waits))
                if "question" in waits[0]:
                    payload["prompt"] = waits[0]["question"]
        return payload

    @staticmethod
    def _public_waits(waits):
        return [
            {
                "sessionId": wait["session_id"],
                "turnId": wait["turn_id"],
                **({"prompt": wait["question"]} if "question" in wait else {}),
                **({"interactionId": wait["interaction_id"]} if "interaction_id" in wait else {}),
            }
            for wait in waits
        ]

    def controller_action(self, *, thread_id, turn_id, action_id, action):
        for identity, name in ((thread_id, "threadId"), (turn_id, "turnId"), (action_id, "actionId")):
            self._validate_public_controller_action_identity(identity, name)
        self._validate_public_controller_action(action)
        input_id = derive_controller_command_id(thread_id, turn_id, action_id)
        previous = self.kernel._retained_reply(thread_id, input_id)
        if previous and (previous[1].kind == "user") != (action["kind"] == "respond"):
            raise TurnResumeError("actionId was reused with a different action payload")
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
        except (Conflict, ValueError) as exc:
            raise TurnResumeError(str(exc)) from exc
        active = self._state_manager.active_turn(thread_id)
        if active is None and not receipt.replayed and state.active_turn_id:
            owner = self._owner(thread_id, turn_id)
            if self._router.is_registered(owner):
                self._start(owner, thread_id, None)
        return {"threadId": thread_id, "turnId": turn_id, "actionId": action_id, "accepted": True, "status": "running"}

    def queue_input(self, sid, tid, kind, input, connection_id, input_id):
        state, _, _ = self.kernel.store.read_state(sid)
        content = _content(self, input, {}, self._owner(sid, tid))
        self.kernel.push(
            sid,
            InboxItem(
                f"request/{tid}/{kind}/{connection_id}/{input_id}",
                kind,
                {"content": content},
                tid if kind == "steer" else None,
                state.turns[tid].start._payload["generation"] if kind == "steer" else None,
            ),
        )

    def _owner(self, sid, tid):
        state, _, _ = self.kernel.store.read_state(sid)
        initial = state.applied_inputs[state.turns[tid].start._payload["input_ids"][0]]._payload["input"]
        return initial["payload"]["content"]["app_server"]["owner"]

    def _hydrate_host_interaction_projection(self, started, event, projection):
        if isinstance(event, HostInteractionRequestedEvent):
            return replace(
                projection,
                notification_params=projection.notification_params
                | {
                    "prompt": event.prompt,
                    "interactionId": event.interaction_id,
                    "sessionId": event.session_id,
                    "childTurnId": event.run_id,
                },
            )
        return projection

    def _complete_turn(self, connection_id, started, *, result, error):
        if error is not None:
            from .store import LeaseRetryExhausted

            sid, tid = started.thread.thread_id, started.turn.turn_id
            self._state_manager.clear_active_turn(sid, tid)
            code = error.code if isinstance(error, LeaseRetryExhausted) else "event_stream"
            self._notify_subscribers(sid, "error/warning", {"message": str(error), "code": code})
            self._notify_subscribers(
                sid, "turn/completed", {"threadId": sid, "turnId": tid, "status": "failed", "error": str(error)}
            )
            return
        if result is not None and result.status is AgentStatus.SUSPENDED:
            result = replace(result, raw_result=replace(result.raw_result, wait_reason="suspended", completion_reason=None))
        sid, tid = started.thread.thread_id, started.turn.turn_id
        self._state_manager.clear_active_turn(sid, tid)
        self._notify_subscribers(sid, "thread/status/changed", self.public_thread_status(sid))
        snapshot = self._store.read_thread(sid)
        turn = next(t for t in snapshot.turns if t.turn_id == tid)
        self._notify_subscribers(sid, "turn/completed", {"threadId": sid, "turnId": tid, "status": turn.status, **turn.result})
        if result is not None and result.status.value == "completed":
            state, _, _ = self.kernel.store.read_state(started.thread.thread_id)
            if any(
                r._payload["disposition"] == "queued" and key not in state.admitted_inputs
                for key, r in state.applied_inputs.items()
            ):
                self._start(connection_id, started.thread.thread_id, None)

    def join(self):
        for pump in self._pumps:
            pump.join()
