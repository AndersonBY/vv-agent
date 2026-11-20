"""App Server adapter and disposable snapshots from kernel records; no thread ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.protocol import ApprovalRequestParams, RequestId
from vv_agent.app_server.run_adapter import TurnResumeError
from vv_agent.app_server.run_adapter import _RunAdapterFormatting as RunAdapter
from vv_agent.app_server.thread_store import ThreadRecord, ThreadSnapshot, TurnRecord
from vv_agent.app_server.usage_projection import task_token_usage_to_wire
from vv_agent.events import ApprovalRequestedEvent
from vv_agent.interaction import derive_controller_command_id
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import INPUT_ACTOR_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES, ContentRef
from vv_agent.kernel._content_publication import publish_session_created_content
from vv_agent.kernel._content_runtime import SessionContentWriter as CoreSessionContentWriter
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid
from vv_agent.kernel._requirements import SessionRequirements, normalize_requirements
from vv_agent.kernel._sql import UnsupportedFeatureLevel
from vv_agent.types import AgentStatus, Message

from .content_runtime import SessionContentWriter
from .context_view import context_view
from .projection import (
    SessionHydration,
    hydrate_input,
    hydrate_record_payload,
    hydration_for_store,
    project_records,
)
from .records import InboxItem, SessionSpec, digest
from .reducer import ExecutionState
from .result import project_result
from .store import Conflict
from .surfaces import PrincipalInput, SessionDriver, _principal_payload
from .timing import Clock, SystemClock


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
        now_ms = SystemClock().now_ms()
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
    input = cast(list[dict[str, Any]], json.loads(canonical_json_bytes(input)))
    metadata = cast(dict[str, Any], json.loads(canonical_json_bytes(metadata)))
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


def thread_status(kernel, thread_id, *, snapshot=None, clock: Clock | None = None, store=None):
    selected_store = store or kernel.store
    state, records, _ = snapshot or selected_store.read_state(thread_id)
    pending = any(
        x.item.kind in {"user", "follow_up"} and x.item.target_turn_id is None for x in selected_store.peek_inbox(thread_id)
    )
    pending = pending or any(
        r._payload["disposition"] == "queued" and key not in state.admitted_inputs for key, r in state.applied_inputs.items()
    )
    status = project_thread_status(state, pending=pending, now_ms=(clock if clock is not None else SystemClock()).now_ms())
    wait_kind = {
        "provider_running": "provider",
        "child_running": "child",
        "user_wait": "user",
        "approval_wait": "approval",
        "retry_backoff": "retry",
    }.get(status.value)
    interaction = None
    if state.active_turn_id:
        waits = [
            {"session_id": sid, **wait}
            for sid, _, wait in kernel.waits(
                thread_id,
                snapshot=(state, records, 0),
                store=selected_store,
            )
        ]
        if waits:
            waits.sort(key=lambda w: ("request_id" not in w, w["session_id"], w.get("operation_id", "")))
            interaction = (
                waits[0]["interaction"]
                if "interaction" in waits[0]
                else {
                    "kind": "approval",
                    "sessionId": waits[0]["session_id"],
                    "turnId": waits[0]["turn_id"],
                    "requestId": waits[0]["request_id"],
                }
            )
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


class SubmissionConflict(ValueError):
    pass


class ContextVersionMismatch(ValueError):
    pass


class TurnMismatch(ValueError):
    pass


class _JoinDriveState(StrEnum):
    LOCAL_LEASE = "local_lease"
    REMOTE_LEASE = "remote_lease"
    DUE = "due"
    FUTURE = "future"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class _JoinDrivePlan:
    state: _JoinDriveState
    head: int


def _database_now_ms(store) -> int | None:
    current = getattr(store, "_now", None)
    if not callable(current):
        return None
    now_ms = current()
    if isinstance(now_ms, bool) or not isinstance(now_ms, int):
        raise TypeError("store _now() must return integer milliseconds")
    return now_ms


@dataclass
class _NotificationPosition:
    seq: int
    due: float = 0.0
    delay: float = 0.05
    warning_at: float = 0.0
    replay_approvals: bool = False


class _KernelThreadStore:
    def __init__(self, kernel: SessionDriver, *, clock: Clock | None = None) -> None:
        self.clock = clock if clock is not None else SystemClock()
        self.kernel = kernel

    def _hydration(self, thread_id: str, store=None) -> SessionHydration:
        return hydration_for_store(
            store or self.kernel.store,
            tenant_id=self.kernel.tenant_id,
            session_id=thread_id,
        )

    def _runtime_view(self, thread_id: str, *, store=None):
        from .bindings import MissingHostBinding

        try:
            return self.kernel._projection_runtime_view(thread_id, store=store)
        except MissingHostBinding:
            return None

    def _thread_creation(self, thread_id: str) -> dict[str, Any]:
        try:
            page = self.kernel.store.read(thread_id, limit=1)
        except Conflict as exc:
            raise KeyError(thread_id) from exc
        if not page.records or page.records[0].record.kind != "session_created":
            raise SubmissionConflict("submission_conflict")
        creation = hydrate_record_payload(page.records[0], self._hydration(thread_id))
        if not isinstance(creation.get("attributes"), dict):
            raise SubmissionConflict("submission_conflict")
        return creation

    def _encoded_thread_principal(self, thread_id: str) -> dict[str, Any]:
        page = self.kernel.store.read(thread_id, limit=1)
        principal = page.records[0].record.payload.get("principal")
        if not isinstance(principal, dict):
            raise SubmissionConflict("submission_conflict")
        return principal

    def _thread_creation_matches(
        self,
        creation: dict[str, Any],
        *,
        thread_id: str,
        agent_key: str,
        cwd: str | None,
        metadata: dict[str, Any] | None,
        principal: PrincipalInput | None,
    ) -> bool:
        expected_principal = (
            {
                "tenant_id": self.kernel.tenant_id,
                "subject_id": "app-server",
                "subject_kind": "service",
                "attributes": {"role": "app-server"},
            }
            if principal is None
            else _principal_payload(principal)
        )
        app_server = creation["attributes"].get("app_server")
        principal_matches = (
            creation.get("principal") == expected_principal or self._encoded_thread_principal(thread_id) == expected_principal
        )
        return principal_matches and app_server == {
            "agent_key": agent_key,
            "cwd": cwd,
            "metadata": metadata or {},
        }

    def _thread_receipt(self, thread_id: str, *, agent_key: str, cwd: str | None, replayed: bool):
        state, _, _ = self.kernel.store.read_state(thread_id)
        return {
            "threadId": thread_id,
            "agentKey": agent_key,
            "cwd": cwd,
            "status": project_thread_status(state, now_ms=self.clock.now_ms()).value,
            "accepted": True,
            "replayed": replayed,
        }

    def create_thread_receipt(
        self,
        *,
        input_id,
        agent_key,
        cwd=None,
        metadata=None,
        context=None,
        principal: PrincipalInput | None = None,
        bootstrap: SessionRequirements | None = None,
    ):
        attributes = {"app_server": {"agent_key": agent_key, "cwd": cwd, "metadata": metadata or {}}}
        values = attributes | ({"app_server_context": context} if context is not None else {})
        try:
            existing = self._thread_creation(input_id)
        except KeyError:
            existing = None
        if existing is not None:
            if not self._thread_creation_matches(
                existing,
                thread_id=input_id,
                agent_key=agent_key,
                cwd=cwd,
                metadata=metadata,
                principal=principal,
            ):
                raise SubmissionConflict("submission_conflict")
            return self._thread_receipt(input_id, agent_key=agent_key, cwd=cwd, replayed=True)
        if bootstrap is not None:
            bootstrap = normalize_requirements(bootstrap)
            if bootstrap.level < 2:
                raise ValueError("App Server static bootstrap requires requirements.level >= 2")
            self.kernel._ensure_static_worker_support(
                level=bootstrap.level,
                capabilities=bootstrap.capabilities,
            )
        try:
            with self.kernel.store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=self.kernel.tenant_id, session_id=input_id)

                def session_spec(content: CoreSessionContentWriter) -> SessionSpec:
                    bound_principal = self.kernel._bound_principal(
                        principal,
                        subject_id="app-server",
                        writer=cast(SessionContentWriter, content),
                    )
                    created = publish_session_created_content(
                        content,
                        principal=bound_principal,
                        workspace=cwd or ".",
                        attributes=values,
                    )
                    return SessionSpec(
                        input_id,
                        created.principal,
                        created.workspace.to_dict(),
                        attributes=created.attributes.to_dict(),
                        feature_level=2 if bootstrap is not None else 1,
                        bootstrap=bootstrap,
                    )

                receipt = writer.create_session(
                    session_spec,
                    consumers=("events", "app_server", "traces"),
                )
            replayed = receipt.replayed
        except Conflict as create_error:
            try:
                existing = self._thread_creation(input_id)
            except KeyError:
                raise
            if not self._thread_creation_matches(
                existing,
                thread_id=input_id,
                agent_key=agent_key,
                cwd=cwd,
                metadata=metadata,
                principal=principal,
            ):
                raise SubmissionConflict("submission_conflict") from create_error
            replayed = True
        return self._thread_receipt(input_id, agent_key=agent_key, cwd=cwd, replayed=replayed)

    def _thread_metadata(self, thread_id, state, records):
        creation = hydrate_record_payload(records[0], self._hydration(thread_id))
        attributes = creation["attributes"].get("app_server")
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
            project_thread_status(state, pending=bool(pending), now_ms=self.clock.now_ms()).value,
            state.active_turn_id or (f"{thread_id}/turn/{pending[-1].input_id}" if pending else None),
            attributes["metadata"]
            | ({"lineage": attributes["app_server_fork"]["lineage"]} if "app_server_fork" in attributes else {}),
        )

    def _input_content(self, thread_id: str, item: InboxItem) -> Any:
        return hydrate_input(item, self._hydration(thread_id)).payload.get("content")

    def read_thread(self, thread_id):
        try:
            state, records, _ = self.kernel.store.read_state(thread_id)
        except KeyError:
            raise KeyError(thread_id) from None
        thread = self._thread_metadata(thread_id, state, records)
        hydration = self._hydration(thread_id)
        turns = []
        runtime = self._runtime_view(thread_id)
        for tid, turn in state.turns.items():
            start = next(r for r in records if r.record == turn.start)
            initial = InboxItem(**state.applied_inputs[turn.start._payload["input_ids"][0]]._payload["input"])
            content = self._input_content(thread_id, initial)
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
            content = self._input_content(thread_id, item)
            turns.append(
                TurnRecord(
                    tid,
                    thread_id,
                    tid,
                    input=content.get("app_server", {}).get("input", []) if isinstance(content, dict) else [],
                    started_at=received / 1000,
                )
            )
        items = []
        for event in project_records(records, hydration=hydration):
            projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
            if projection.item:
                items.append(projection.item)
        return ThreadSnapshot(
            thread, turns, items, thread_status(self.kernel, thread_id, snapshot=(state, records, 0), clock=self.clock)
        )

    def list_threads(self, *, include_archived=False, authorize=None, archived=None, offset=0, limit=None):
        candidates = []
        after = None
        while ids := self.kernel.store.list_sessions(after=after):
            for sid in ids:
                if authorize is not None and not authorize(sid):
                    continue
                page = self.kernel.store.read(sid, limit=1)
                attributes = hydrate_record_payload(page.records[0], self._hydration(sid))["attributes"]
                if not attributes.get("app_server"):
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

    def archive_thread(self, thread_id, *, actor: PrincipalInput | None = None):
        # Archiving is a durable control. Reading the full projection first can
        # surface an unfinished provider failure and prevent that control from
        # ever reaching the session driver.
        if not self._thread_creation(thread_id)["attributes"].get("app_server"):
            raise KeyError(thread_id)
        self.kernel.control(thread_id, "archive", "app_server/archive", actor=actor)

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
        self.clock = self._store.clock
        self.kernel.model_resolver = self._host.resolve_model
        self._notification_positions: dict[str, _NotificationPosition] = {}
        self._turn_connections: dict[tuple[str, str], str | None] = {}
        support = self.kernel._worker.support
        self._driver_observe = support.observe
        self._driver_projection_consumers = support.projection_consumers
        self.kernel._worker.support = replace(
            support,
            observe=self._observe_with_driver,
            projection_consumers=("child_delivery",) if support.project is not None else (),
        )
        self.kernel._on_close(self.close)

    def _hydration(self, thread_id: str, store=None) -> SessionHydration:
        return hydration_for_store(
            store or self.kernel.store,
            tenant_id=self.kernel.tenant_id,
            session_id=thread_id,
        )

    def _with_app_server_controls(self, run_config, *, connection_id: str | None, thread_id: str, turn_id: str):
        return replace(
            run_config,
            metadata={
                **run_config.metadata,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "session_id": thread_id,
            },
            approval_provider=run_config.approval_provider,
        )

    def _turn_runtime_config(self, run_config, *, thread_id: str, turn_id: str):
        host_stream = run_config.stream

        def stream(event):
            if host_stream:
                host_stream(event)
            projection = map_run_event(event, thread_id=thread_id, turn_id=event.run_id)
            if projection.item is None and projection.notification_method and projection.notification_method.endswith("/delta"):
                self._notify_subscribers(thread_id, projection.notification_method, projection.notification_params)
            for method, params in projection.additional_notifications:
                if method.endswith("/delta"):
                    self._notify_subscribers(thread_id, method, params)
            self.kernel._worker.wake(thread_id)

        return replace(run_config, stream=stream)

    def _prepare_turn_runtime(self, *, connection_id: str | None, thread_id: str, input_id: str, content):
        turn_id = f"{thread_id}/turn/{input_id}"
        _thread, agent, config = self._binding(thread_id, content)
        config = self._with_app_server_controls(config, connection_id=connection_id, thread_id=thread_id, turn_id=turn_id)
        runtime_config = self._turn_runtime_config(config, thread_id=thread_id, turn_id=turn_id)
        return self.kernel.runtime(agent, runtime_config, session_id=thread_id)

    def subscribe(self, thread_id, connection_id):
        state, records, _ = self.kernel.store.read_state(thread_id)
        start = state.turns[state.active_turn_id].start if state.active_turn_id else None
        position = next(r.seq for r in records if r.record == start) - 1 if start else records[-1].seq
        if not self._router.is_registered(connection_id):
            return
        local = self._state_manager.active_turn(thread_id)
        if state.active_turn_id is not None and local is not None and local.turn_id == state.active_turn_id:
            owner_key = (thread_id, state.active_turn_id)
            owner = self._turn_connections.get(owner_key)
            if owner_key not in self._turn_connections or owner is None or not self._router.is_registered(owner):
                self._turn_connections[owner_key] = connection_id
        elif state.active_turn_id is not None:
            self._claim_disconnected_turn_owner(connection_id, thread_id, state)
        self._notification_positions.setdefault(thread_id, _NotificationPosition(position, due=0.0))
        self._state_manager.subscribe(thread_id, connection_id)
        registration = self.kernel._registration_for_cursor(thread_id)
        pending_start = self._has_pending_start_input(thread_id, self.kernel.store, state)
        # Observe durable state without a local runtime, but leave worker scans
        # to the admission/recovery path when there is runnable turn work.
        if self.kernel._worker.support.drive is not None and (
            (pending_start and self._state_manager.active_turn(thread_id) is None)
            or (state.active_turn_id is not None and registration is None)
        ):
            return
        self.kernel._worker.post_observation(self._observe_with_driver)

    def unsubscribe(self, thread_id, connection_id):
        self._state_manager.unsubscribe(thread_id, connection_id)
        released = []
        for key, owner in tuple(self._turn_connections.items()):
            if key[0] == thread_id and owner == connection_id:
                self._turn_connections[key] = None
                released.append(key)
        for owned_thread, turn_id in released:
            self._router.cancel_matching_server_requests(thread_id=owned_thread, turn_id=turn_id)
        if not self._state_manager.subscribers(thread_id):
            self._notification_positions.pop(thread_id, None)
            state, _, _ = self.kernel.store.read_state(thread_id)
            self._settle_turn_connections(thread_id, state, self.kernel.store)

    def unsubscribe_connection(self, connection_id):
        self._state_manager.unsubscribe_connection(connection_id)
        owned = [key for key, owner in self._turn_connections.items() if owner == connection_id]
        for key in owned:
            self._turn_connections[key] = None
        for thread_id, turn_id in owned:
            self._router.cancel_matching_server_requests(thread_id=thread_id, turn_id=turn_id)
        for sid in list(self._notification_positions):
            if not self._state_manager.subscribers(sid):
                del self._notification_positions[sid]
                state, _, _ = self.kernel.store.read_state(sid)
                self._settle_turn_connections(sid, state, self.kernel.store)

    def _clear_turn_connections(self, thread_id: str) -> None:
        for key in tuple(self._turn_connections):
            if key[0] == thread_id:
                self._turn_connections.pop(key, None)

    def _subscriber(self, sid):
        return next((cid for cid in self._state_manager.subscribers(sid) if self._router.is_registered(cid)), None)

    def wake(self, thread_id):
        position = self._notification_positions.get(thread_id)
        if position is not None:
            position.due = 0.0
            position.delay = 0.05
            position.replay_approvals = True
        try:
            self.kernel._worker.post_observation(self._observe_with_driver)
        except RuntimeError as exc:
            if str(exc) not in {"LocalWorker is stopped", "LocalWorker cannot be restarted"}:
                raise

    def _observe_with_driver(self, store) -> None:
        if self._driver_observe is not None:
            self._driver_observe(store)
        self._observe(store)
        active_threads = {
            active.thread_id
            for handle in self.kernel.handles
            if handle.consumer == "app_server"
            for active in (self._state_manager.active_turn(handle.session_id),)
            if active is not None
        }
        for sid in active_threads:
            self._reconcile_local_turn(sid, store)

    def _observe(self, store) -> None:
        """Tail committed records and durable waits on the coordinator thread."""

        now = self.clock.monotonic()
        for sid in tuple(self._notification_positions):
            position = self._notification_positions.get(sid)
            if position is None:
                continue
            if not self._state_manager.subscribers(sid):
                self._notification_positions.pop(sid, None)
                state, _, _ = store.read_state(sid)
                self._settle_turn_connections(sid, state, store)
                continue
            try:
                operational = self._publish_operational_error(sid, store)
                if now < position.due:
                    if not store.read(sid, after_seq=position.seq, limit=1).records:
                        continue
                    # A committed record is progress even when its producer did
                    # not call wake(). Reset only this session's throttling.
                    position.due = 0.0
                    position.delay = 0.05
                previous_seq = position.seq
                approval_seq = self._next_approval_seq(sid, store, after_seq=previous_seq)
                if position.replay_approvals:
                    blocked = self._send_pending_approvals(
                        sid,
                        store,
                        after_seq=previous_seq,
                        through_seq=approval_seq,
                        replay_approvals=True,
                    )
                    if blocked is not None:
                        self._reconcile_local_turn(sid, store)
                        continue
                    position.replay_approvals = False
                if approval_seq is not None:
                    progressed = (
                        self._publish_turns(sid, store=store, through_seq=approval_seq - 1, include_items=True) or operational
                    )
                    blocked = self._send_pending_approvals(
                        sid,
                        store,
                        after_seq=previous_seq,
                        through_seq=approval_seq,
                    )
                    if blocked is not None:
                        self._reconcile_local_turn(sid, store)
                        continue
                    progressed = self._publish_turns(sid, store=store, include_items=True) or progressed
                else:
                    progressed = self._publish_turns(sid, store=store, include_items=True) or operational
                self._reconcile_local_turn(sid, store)
            except Exception as exc:
                position.delay = min(1.0, max(0.05, position.delay * 2))
                position.due = now + position.delay
                if now >= position.warning_at:
                    position.warning_at = now + 1.0
                    self._notify_subscribers(
                        sid,
                        "error/warning",
                        {"message": str(exc), "code": "notification_stream"},
                    )
            else:
                position.delay = 0.05 if progressed else min(1.0, max(0.05, position.delay * 2))
                position.due = now + position.delay

    def _publish_operational_error(self, sid: str, store) -> bool:
        failure = self.kernel._failure_after_progress(store, sid, self.kernel._errors.get(sid))
        if failure is None or not isinstance(failure.error, ProviderResultInvalid) or failure.warning_sent:
            return False
        if not self._state_manager.subscribers(sid):
            return False
        active = self._state_manager.active_turn(sid)
        self._notify_subscribers(
            sid,
            "error/warning",
            {"message": "provider_result_invalid", "code": "provider_result_invalid"},
        )
        failure.warning_sent = True
        if (
            active is not None
            and (failure.turn_id is None or failure.turn_id == active.turn_id)
            and self._state_manager.clear_active_turn(sid, active.turn_id, handle=active.handle)
        ):
            self.kernel._abandon_handle(active.handle)
        return True

    def _publish_turns(self, sid, *, store, through_seq=None, include_items=False):
        """Publish one committed record page exactly once for this node position."""

        position = self._notification_positions.get(sid)
        if position is None:
            return False
        previous = position.seq
        rows = store.read(sid, after_seq=previous, through_seq=through_seq, limit=256).records
        if not rows:
            return False
        notifications: list[tuple[int, str, dict[str, Any]]] = []
        state, all_records, _ = store.read_state(sid)
        runtime = self._store._runtime_view(sid, store=store)
        for stored in rows:
            record = stored.record
            if record.kind == "turn_started":
                self._ensure_turn_connection(sid, record.turn_id)
                notifications.append((stored.seq, "turn/started", {"threadId": sid, "turnId": record.turn_id}))
            elif record.kind == "turn_ended":
                result = project_result(store, sid, record.turn_id, runtime=runtime, snapshot=(state, all_records, 0))
                notifications.append(
                    (
                        stored.seq,
                        "thread/status/changed",
                        thread_status(
                            self.kernel,
                            sid,
                            snapshot=(state, all_records, 0),
                            clock=self.clock,
                            store=store,
                        ),
                    )
                )
                notifications.append(
                    (
                        stored.seq,
                        "turn/completed",
                        {
                            "threadId": sid,
                            "turnId": record.turn_id,
                            "status": "completed" if record._payload["status"] == "completed" else "failed",
                            **_result_fields(result),
                        },
                    )
                )
        sequences = {row.seq for row in rows}
        for event in project_records(store.projection_context(sid, rows), hydration=self._hydration(sid, store)):
            if event.metadata["session_seq"] not in sequences:
                continue
            projection = map_run_event(event, thread_id=sid, turn_id=event.run_id)
            if include_items and projection.item is not None and projection.notification_method is not None:
                notifications.append(
                    (event.metadata["session_seq"], projection.notification_method, projection.notification_params)
                )
            notifications.extend(
                (event.metadata["session_seq"], method, params)
                for method, params in projection.additional_notifications
                if method.startswith(("approval/", "contextCompaction/"))
            )
        position.seq = max(position.seq, rows[-1].seq)
        rank = {"turn/started": 0, "turn/completed": 2}
        for _, method, params in sorted(notifications, key=lambda entry: (entry[0], rank.get(entry[1], 1))):
            self._notify_subscribers(sid, method, params)
        for turn_id in {row.record.turn_id for row in rows if row.record.kind == "turn_ended"}:
            self._router.cancel_matching_server_requests(thread_id=sid, turn_id=turn_id)
        self._settle_turn_connections(sid, state, store)
        if state.active_turn_id is not None or self._has_pending_start_input(sid, store, state):
            self.kernel._worker.wake(sid)
        return True

    def _has_pending_start_input(self, thread_id: str, store, state) -> bool:
        if any(
            record._payload["disposition"] == "queued"
            and key not in state.admitted_inputs
            and record._payload["input"]["kind"] in {"user", "follow_up"}
            for key, record in state.applied_inputs.items()
        ):
            return True
        return any(
            entry.item.kind in {"user", "follow_up"} and entry.item.target_turn_id is None
            for entry in store.peek_inbox(thread_id)
        )

    def _next_turn_id(self, thread_id: str, store, state, *, input_id: str | None = None) -> str:
        if input_id is not None:
            return f"{thread_id}/turn/{input_id}"
        if state.active_turn_id is not None:
            return state.active_turn_id
        queued = next(
            (
                record._payload["input"]["input_id"]
                for key, record in state.applied_inputs.items()
                if record._payload["disposition"] == "queued"
                and key not in state.admitted_inputs
                and record._payload["input"]["kind"] in {"user", "follow_up"}
            ),
            None,
        )
        pending = next(
            (
                entry.item.input_id
                for entry in store.peek_inbox(thread_id)
                if entry.item.kind in {"user", "follow_up"} and entry.item.target_turn_id is None
            ),
            None,
        )
        identity = queued or pending
        return f"{thread_id}/turn/{identity}" if identity else next(reversed(state.turns), "")

    def _latest_turn_owner(self, thread_id: str, *, exclude: str | None = None) -> str | None:
        owner = None
        for (candidate_thread, turn_id), candidate in self._turn_connections.items():
            if candidate_thread != thread_id or turn_id == exclude or candidate is None:
                continue
            if self._router.is_registered(candidate):
                owner = candidate
        return owner

    def _claim_disconnected_turn_owner(self, connection_id, thread_id: str, state) -> bool:
        if state.active_turn_id is None or connection_id is None or not self._router.is_registered(connection_id):
            return False
        if self.kernel.store.lease_status(thread_id)[2] > 0:
            return False
        owner_key = (thread_id, state.active_turn_id)
        if owner_key not in self._turn_connections:
            return False
        owner = self._turn_connections[owner_key]
        if owner is not None and self._router.is_registered(owner):
            return False
        self._turn_connections[owner_key] = connection_id
        return True

    def _ensure_turn_connection(self, thread_id: str, turn_id: str) -> None:
        key = (thread_id, turn_id)
        if key in self._turn_connections:
            return
        owner = self._latest_turn_owner(thread_id, exclude=turn_id)
        if owner is not None:
            self._turn_connections[key] = owner

    def _settle_turn_connections(self, thread_id: str, state, store) -> None:
        active = state.active_turn_id
        pending = self._has_pending_start_input(thread_id, store, state)
        owner_turn = active or next(
            (turn_id for turn_id, turn in reversed(state.turns.items()) if not turn.ended),
            None,
        )
        if owner_turn is not None:
            self._ensure_turn_connection(thread_id, owner_turn)
            for key in tuple(self._turn_connections):
                if key[0] == thread_id and key[1] != owner_turn:
                    self._turn_connections.pop(key, None)
        elif pending:
            owners = [
                (key, owner)
                for key, owner in self._turn_connections.items()
                if key[0] == thread_id and owner is not None and self._router.is_registered(owner)
            ]
            self._clear_turn_connections(thread_id)
            if owners:
                self._turn_connections[owners[-1][0]] = owners[-1][1]
        else:
            self._clear_turn_connections(thread_id)

    def _reconcile_local_turn(self, sid: str, store) -> None:
        active = self._state_manager.active_turn(sid)
        if active is None:
            return
        state, _records, _ = store.read_state(sid)
        if store.lease_status(sid)[2] > 0:
            return
        turn = state.turns.get(active.turn_id)
        ended = turn is not None and turn.ended
        has_foreground_child_wait = any(
            wait.get("handle", {}).get("kind") == "child" and wait.get("handle", {}).get("background") is False
            for wait in state.waits.values()
        )
        durable_wait = (turn is not None and (turn.wait is not None or turn.suspended)) or bool(state.waits)
        parked = state.phase == "parked" and bool(state.waits)
        if not ended and not durable_wait and not parked:
            return
        if not ended and has_foreground_child_wait and self.kernel._pending_child_work(store, state):
            return
        if not self._state_manager.clear_active_turn(sid, active.turn_id, handle=active.handle):
            return
        self.kernel._abandon_handle(active.handle)
        self._settle_turn_connections(sid, state, store)
        if ended:
            self._router.cancel_matching_server_requests(thread_id=sid, turn_id=active.turn_id)

    def _approval_answered(self, store, thread_id: str, request_id: str) -> bool:
        input_id = f"approval/{request_id}/answer"
        state, _records, _ = store.read_state(thread_id)
        if input_id in state.applied_inputs:
            return True
        return any(entry.item.input_id == input_id for entry in store.peek_inbox(thread_id))

    def _next_approval_seq(self, sid: str, store, *, after_seq: int) -> int | None:
        cursor = after_seq
        while True:
            rows = store.read(sid, after_seq=cursor, limit=256).records
            if not rows:
                return None
            sequences = {row.seq for row in rows}
            for event in project_records(store.projection_context(sid, rows), hydration=self._hydration(sid, store)):
                seq = event.metadata["session_seq"]
                if isinstance(event, ApprovalRequestedEvent) and seq in sequences and seq > after_seq:
                    return seq
            cursor = rows[-1].seq

    def _send_pending_approvals(
        self,
        sid: str,
        store,
        *,
        after_seq: int,
        through_seq: int | None = None,
        replay_approvals: bool = False,
    ) -> int | None:
        registration = self.kernel._runtimes.get(sid)
        if registration is not None and registration.config.approval_provider is not None:
            return None
        state, records, inbox_seq = store.read_state(sid)
        events = {
            event.request_id: event
            for event in project_records(records, hydration=self._hydration(sid, store))
            if isinstance(event, ApprovalRequestedEvent)
        }
        waits = {
            (wait.get("turn_id"), wait.get("request_id")): wait
            for wait_sid, _, wait in self.kernel.waits(
                sid,
                snapshot=(state, records, inbox_seq),
                store=store,
            )
            if wait_sid == sid and wait.get("kind") == "approval"
        }
        now_ms = _database_now_ms(store)
        for request_id, event in events.items():
            event_seq = event.metadata["session_seq"]
            if (event_seq <= after_seq and not replay_approvals) or (through_seq is not None and event_seq > through_seq):
                continue
            turn_id = event.run_id
            wait = waits.get((turn_id, request_id))
            if replay_approvals and wait is None:
                continue
            if wait is not None:
                deadline_ms = wait.get("deadline_ms")
                if deadline_ms is None:
                    durable_wait = state.waits.get((wait.get("operation_id"), wait.get("attempt")))
                    deadline_ms = durable_wait.get("deadline_ms") if durable_wait is not None else None
                if deadline_ms is not None and now_ms is not None and now_ms >= deadline_ms:
                    continue
            if self._approval_answered(store, sid, request_id):
                continue
            owner = self._turn_connections.get((sid, turn_id))
            if owner is None or not self._router.is_registered(owner):
                continue
            if self._router.get_pending_server_request(RequestId(request_id)) is not None:
                continue
            if store.lease_status(sid)[2] > 0:
                return event_seq
            arguments: dict[str, Any] = {}
            if wait is not None:
                operation = state.operations.get(wait.get("operation_id"))
                attempt = operation.attempts.get(wait.get("attempt")) if operation is not None else None
                plan_request = attempt.plan._payload.get("request") if attempt is not None and attempt.plan is not None else None
                raw_arguments = plan_request.get("arguments") if isinstance(plan_request, dict) else None
                if isinstance(raw_arguments, dict):
                    arguments = dict(raw_arguments)
            try:
                self._router.send_server_request(
                    owner,
                    "approval/request",
                    ApprovalRequestParams(
                        request_id=request_id,
                        thread_id=sid,
                        turn_id=turn_id,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        preview=event.message or f"Approval required for tool {event.tool_name}.",
                        arguments=arguments,
                    ).to_dict(),
                    request_id=RequestId(request_id),
                )
            except ValueError as exc:
                if not str(exc).startswith("Duplicate server request id:"):
                    raise
            self.kernel._worker.wake(sid)
        return None

    def close(self):
        thread_ids = {key[0] for key in self._turn_connections} | set(self._notification_positions)
        subscribed = {connection_id for thread_id in thread_ids for connection_id in self._state_manager.subscribers(thread_id)}
        for thread_id, turn_id in tuple(self._turn_connections):
            self._router.cancel_matching_server_requests(thread_id=thread_id, turn_id=turn_id)
        for connection_id in subscribed:
            self._state_manager.unsubscribe_connection(connection_id)
        for handle in tuple(self.kernel.handles):
            if handle.consumer == "app_server":
                self.kernel._abandon_handle(handle)
        self._notification_positions.clear()
        self._turn_connections.clear()
        support = self.kernel._worker.support
        if support.observe == self._observe_with_driver:
            self.kernel._worker.support = replace(
                support,
                observe=self._driver_observe,
                projection_consumers=self._driver_projection_consumers,
            )

    def _binding(self, thread_id, content=None, *, turn_id=None):
        state, records, _ = self.kernel.store.read_state(thread_id)
        thread = self._store._thread_metadata(thread_id, state, records)
        if content is None:
            selected_turn = turn_id or state.active_turn_id
            initial_id = state.turns[selected_turn].start._payload["input_ids"][0] if selected_turn else None
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
            if item is not None:
                content = hydrate_input(item, self._hydration(thread_id)).payload.get("content")
        metadata = thread.metadata | (content.get("app_server", {}).get("metadata", {}) if content else {})
        agent = self._host.resolve_agent(AgentResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        config = self._host.build_run_config(RunConfigResolutionRequest(thread_id, thread.agent_key, thread.cwd, metadata))
        shared_state = config.shared_state
        if state.turns:
            prior = project_result(
                self.kernel.store,
                thread_id,
                turn_id or next(reversed(state.turns)),
                runtime=self._store._runtime_view(thread_id),
            )
            shared_state = prior.raw_result.shared_state
        return (
            thread,
            agent,
            replace(config, shared_state=shared_state, initial_messages=None, metadata=config.metadata | metadata),
        )

    def submit_input(
        self,
        *,
        connection_id,
        thread_id,
        input_id,
        expected_turn_id,
        kind,
        input,
        metadata=None,
        actor: PrincipalInput | None = None,
        _prepared_content=None,
        _start_runtime=None,
    ):
        content = (
            _prepared_content if _prepared_content is not None else _content(self, input, metadata or {}, kind, expected_turn_id)
        )
        previous_registration = None
        start_registration = None
        try:
            while True:
                state, records, _ = self.kernel.store.read_state(thread_id)
                try:
                    with self.kernel.store.atomic() as tx:
                        tx.lock_session(thread_id)
                        previous_registration = self.kernel._runtimes.get(thread_id)
                        if self.kernel.store.lease_status(thread_id)[0] != records[-1].seq:
                            continue
                        stored = self.kernel.store.read_input(thread_id, input_id)
                        if stored:
                            retained = stored.item
                            expected_kind = "user" if kind == "start" else kind
                            actor_payload = self._controller_actor_payload(thread_id, actor)
                            retained_content = hydrate_input(retained, self._hydration(thread_id)).payload.get("content")
                            if (
                                retained.kind != expected_kind
                                or not self._controller_actor_matches(thread_id, retained, actor_payload)
                                or canonical_json_bytes(retained_content) != canonical_json_bytes(content)
                            ):
                                raise SubmissionConflict("submission_conflict")
                            receipt = tx.push(thread_id, retained)
                            break
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
                        if kind == "start" and _start_runtime is not None:
                            start_registration = self.kernel._register_runtime(
                                thread_id,
                                _start_runtime,
                                armed=True,
                            )
                        receipt = self.kernel._push_in_transaction(
                            tx,
                            thread_id,
                            item,
                            actor=actor,
                            actor_subject_id="app-server",
                            registration=start_registration,
                        )
                        break
                except BaseException:
                    if start_registration is not None and self.kernel._runtimes.get(thread_id) is start_registration:
                        if previous_registration is None:
                            self.kernel._runtimes.pop(thread_id, None)
                        else:
                            self.kernel._runtimes[thread_id] = previous_registration
                    raise
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
        if kind in {"start", "follow_up"} and self._router.is_registered(connection_id):
            key = (thread_id, tid)
            owner = self._turn_connections.get(key)
            if kind == "start" or owner is None:
                inherited = None if kind == "start" else self._latest_turn_owner(thread_id, exclude=tid)
                self._turn_connections[key] = inherited or connection_id
        return payload, content

    def start_turn(
        self,
        *,
        connection_id,
        thread_id,
        input,
        input_id,
        expected_turn_id,
        metadata=None,
        request_id=None,
        actor: PrincipalInput | None = None,
    ):
        content = _content(self, input, metadata or {}, "start", expected_turn_id)
        start_runtime = None
        if self.kernel._worker.support.drive is not None:
            start_runtime = self._prepare_turn_runtime(
                connection_id=connection_id,
                thread_id=thread_id,
                input_id=input_id,
                content=content,
            )
        payload, _submitted_content = self.submit_input(
            connection_id=connection_id,
            thread_id=thread_id,
            input_id=input_id,
            expected_turn_id=expected_turn_id,
            kind="start",
            input=input,
            metadata=metadata,
            actor=actor,
            _prepared_content=content,
            _start_runtime=start_runtime,
        )
        self.subscribe(thread_id, connection_id)
        self.recover(connection_id, thread_id)
        if request_id is not None:
            self._router.send_response(connection_id, request_id, payload)
        return payload

    def _start(
        self,
        connection_id,
        thread_id,
        content,
        *,
        input_id=None,
        request_id=None,
        lease_retry=True,
        actor: PrincipalInput | None = None,
    ):
        with self._state_manager.turn_lock(thread_id):
            active = self._state_manager.active_turn(thread_id)
            if active is None:
                return self._start_locked(
                    connection_id,
                    thread_id,
                    content,
                    input_id=input_id,
                    request_id=request_id,
                    lease_retry=lease_retry,
                    actor=actor,
                )
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": active.turn_id, "runId": active.turn_id, "status": "running"},
                )
            return None

    def _start_locked(
        self,
        connection_id,
        thread_id,
        content,
        *,
        input_id=None,
        request_id=None,
        lease_retry=True,
        actor: PrincipalInput | None = None,
    ):
        if connection_id is not None and self._router.is_registered(connection_id):
            self.subscribe(thread_id, connection_id)
        else:
            connection_id = self._subscriber(thread_id)
        state, _records, _ = self.kernel.store.read_state(thread_id)
        tid = self._next_turn_id(
            thread_id,
            self.kernel.store,
            state,
            input_id=input_id if content is not None else None,
        )
        owner_key = (thread_id, tid)
        if (
            self._turn_connections.get(owner_key) is None
            and connection_id is not None
            and self._router.is_registered(connection_id)
        ):
            inherited_owner = self._latest_turn_owner(thread_id, exclude=tid)
            if inherited_owner is not None:
                self._turn_connections[owner_key] = inherited_owner
            elif self.kernel.store.lease_status(thread_id)[2] == 0:
                self._turn_connections[owner_key] = connection_id
        registration = self.kernel._runtimes.get(thread_id)
        if registration is not None and registration.config.metadata.get("turn_id") == tid:
            agent = registration.agent
            runtime_config = registration.config
        else:
            _thread, agent, config = self._binding(thread_id, content)
            config = self._with_app_server_controls(
                config,
                connection_id=connection_id,
                thread_id=thread_id,
                turn_id=tid,
            )
            runtime_config = self._turn_runtime_config(config, thread_id=thread_id, turn_id=tid)
            runtime = self.kernel.runtime(agent, runtime_config, session_id=thread_id)
            self.kernel._register_runtime(thread_id, runtime, armed=True)
        handle = self.kernel.start(
            thread_id,
            agent,
            runtime_config,
            content,
            input_id=input_id,
            consumer="app_server",
            autostart=False,
            actor=actor,
            actor_subject_id="app-server",
        )
        self._state_manager.set_active_turn(thread_id=thread_id, turn_id=handle.run_id, handle=handle)
        if request_id is not None:
            self._router.send_response(
                connection_id,
                request_id,
                {
                    "threadId": thread_id,
                    "turnId": handle.run_id,
                    "status": "running",
                    **({"runId": handle.run_id} if content is None else {}),
                },
            )
        self.kernel._worker.post_observation(self._observe_with_driver)
        try:
            handle.start()
        except Exception as exc:
            self._state_manager.clear_active_turn(thread_id, handle.run_id, handle=handle)
            self.kernel._abandon_handle(handle)
            self._notify_subscribers(thread_id, "error/warning", {"message": str(exc), "code": "drive_start"})
        if content is None and handle.done():
            state, _, _ = self.kernel.store.read_state(thread_id)
            if state.phase == "parked":
                self._state_manager.clear_active_turn(thread_id, handle.run_id, handle=handle)
                self.kernel._abandon_handle(handle)
        return handle

    def recover(self, connection_id, thread_id, *, follow_up_only=False, lease_retry=True):
        if self.kernel._worker.support.drive is None:
            self.kernel._worker.post_observation(self._observe_with_driver)
            return
        try:
            with self._state_manager.turn_lock(thread_id):
                if self._state_manager.active_turn(thread_id) is None:
                    state, _, _ = self.kernel.store.read_state(thread_id)
                    if state.active_turn_id is not None:
                        self._start(
                            connection_id,
                            thread_id,
                            None,
                            lease_retry=lease_retry,
                        )
                        return
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
                            (x.item.kind in kinds and x.item.target_turn_id is None)
                            or x.item.kind in {"control", "provider_result", "provider_evidence", "child_result"}
                            for x in self.kernel.store.peek_inbox(thread_id)
                        )
                    ):
                        self._start(
                            connection_id,
                            thread_id,
                            None,
                            lease_retry=lease_retry,
                        )
        except Exception as exc:
            self._notify_subscribers(thread_id, "error/warning", {"message": str(exc), "code": "drive_start"})

    def resume_turn(self, *, connection_id, thread_id, turn_id, request_id=None):
        snapshot = self._store.read_thread(thread_id)
        if snapshot.thread.status == ThreadStatus.CLOSED:
            raise TurnResumeError("Thread is closed")
        turn = next((t for t in snapshot.turns if t.turn_id == turn_id), None)
        if turn is None:
            raise TurnResumeError("Turn does not belong to the requested thread")
        state, _, _ = self.kernel.store.read_state(thread_id)
        durable_turn = state.turns.get(turn_id)
        if durable_turn is None:
            if state.active_turn_id is not None and state.active_turn_id != turn_id:
                raise TurnResumeError("Thread has a different active turn")
            self.recover(connection_id, thread_id)
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": turn.status},
                )
            return
        if durable_turn.ended:
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": turn.status, **turn.result},
                )
            return
        if state.active_turn_id != turn_id:
            raise TurnResumeError("Thread has a different active turn")
        if self.kernel._worker.support.drive is None:
            self._claim_disconnected_turn_owner(connection_id, thread_id, state)
            self.recover(connection_id, thread_id)
            if request_id is not None:
                self._router.send_response(
                    connection_id,
                    request_id,
                    {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": turn.status},
                )
            return
        self._clear_provider_error_for_retry(thread_id)
        if self._state_manager.active_turn(thread_id) is not None:
            if request_id is not None:
                self._router.send_response(
                    connection_id, request_id, {"threadId": thread_id, "turnId": turn_id, "runId": turn_id, "status": "running"}
                )
            return
        self._start(connection_id, thread_id, None, request_id=request_id)

    def _clear_provider_error_for_retry(self, thread_id: str) -> None:
        failure = self.kernel._errors.get(thread_id)
        if failure is not None and isinstance(failure.error, ProviderResultInvalid):
            self.kernel._errors.pop(thread_id, None)

    def public_thread_status(self, thread_id):
        return thread_status(self.kernel, thread_id, clock=self.clock)

    def _fork_principal_matches(self, thread_id: str, principal: PrincipalInput | None) -> bool:
        creation = self._store._thread_creation(thread_id)
        expected = (
            {
                "tenant_id": self.kernel.tenant_id,
                "subject_id": "app-server",
                "subject_kind": "service",
                "attributes": {"role": "app-server"},
            }
            if principal is None
            else _principal_payload(principal)
        )
        return creation.get("principal") == expected or self._store._encoded_thread_principal(thread_id) == expected

    def _normalize_fork_principal(
        self,
        source_thread_id: str,
        principal: PrincipalInput | None,
    ) -> PrincipalInput | None:
        if principal is None:
            return None
        payload = _principal_payload(principal)
        attributes = payload["attributes"]
        if not isinstance(attributes, ContentRef):
            return payload
        source_principal = self._store._encoded_thread_principal(source_thread_id)
        if attributes.to_dict() != source_principal.get("attributes"):
            raise PermissionError("persisted fork principal does not reference the source session principal")
        payload["attributes"] = self.kernel.read_record_content(
            source_thread_id,
            1,
            attributes,
            slot=SESSION_PRINCIPAL_ATTRIBUTES,
        )
        return payload

    def fork_thread(self, params, *, principal: PrincipalInput | None = None):
        from .fork import fork_thread

        principal = self._normalize_fork_principal(params["threadId"], principal)
        destination_id = "fork/" + digest([params["threadId"], params["forkId"]])
        try:
            self._store._thread_creation(destination_id)
        except KeyError:
            pass
        else:
            if not self._fork_principal_matches(destination_id, principal):
                raise SubmissionConflict("submission_conflict")

        result = fork_thread(self.kernel, self._host, params, principal=principal)
        if result.get("replayed") and not self._fork_principal_matches(destination_id, principal):
            raise SubmissionConflict("submission_conflict")
        return result

    def read_context(self, thread_id):
        return context_view(self.kernel.store, thread_id)

    def compact_thread(
        self,
        *,
        thread_id,
        action_id,
        expected_context_version=None,
        connection_id=None,
        actor: PrincipalInput | None = None,
    ):
        params = {"threadId": thread_id, "actionId": action_id}
        if expected_context_version is not None:
            params["expectedContextVersion"] = expected_context_version
        request_id = "compaction/" + digest([thread_id, action_id])
        content = {
            "text": "",
            "messages": [],
            "metadata": {"request_compaction": True},
            "app_server_compaction": params,
        }
        try:
            while True:
                state, records, watermark = self.kernel.store.read_state(thread_id)
                context_version = self.read_context(thread_id)["contextVersion"] if expected_context_version is not None else None
                head = records[-1].seq
                position = self.kernel.store.read(thread_id, after_seq=head, through_seq=head, limit=1)
                if (position.head_seq, position.inbox_seq) != (head, watermark):
                    continue
                with self.kernel.store.atomic() as tx:
                    tx.lock_session(thread_id)
                    position = self.kernel.store.read(thread_id, after_seq=head, through_seq=head, limit=1)
                    if (position.head_seq, position.inbox_seq) != (head, watermark):
                        continue
                    previous = self.kernel.store.read_input(thread_id, request_id)
                    if previous is None:
                        if state.closed or state.archived:
                            raise TurnResumeError("Thread is closed or archived")
                        if expected_context_version is not None and context_version != expected_context_version:
                            raise ContextVersionMismatch("context_version_mismatch")
                    item = InboxItem(request_id, "user", {"content": content})
                    receipt = self.kernel._push_in_transaction(
                        tx,
                        thread_id,
                        item,
                        actor=actor,
                        actor_subject_id="app-server",
                    )
                    break
        except Conflict as exc:
            if str(exc) == "session or cursor not found":
                raise KeyError(thread_id) from exc
            raise SubmissionConflict("submission_conflict") from exc
        result = {
            "threadId": thread_id,
            "actionId": action_id,
            "requestId": request_id,
            "inputSeq": receipt.input_seq,
            "accepted": True,
            "replayed": receipt.replayed,
        }
        if not receipt.replayed:
            self.recover(connection_id, thread_id)
        return result

    def select_model(self, *, thread_id, input_id, selection, actor: PrincipalInput | None = None):
        snapshot = self._store.read_thread(thread_id)
        if snapshot.thread.status in {ThreadStatus.CLOSED, ThreadStatus.ARCHIVED}:
            raise SubmissionConflict("submission_conflict")
        try:
            self.kernel.push(
                thread_id,
                InboxItem(
                    input_id,
                    "model_selection",
                    {"selection": selection},
                ),
                actor=actor,
                actor_subject_id="app-server",
            )
        except UnsupportedFeatureLevel:
            raise
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        return {"threadId": thread_id, "inputId": input_id, "accepted": True}

    def _controller_actor_payload(self, thread_id: str, actor: PrincipalInput | None) -> dict[str, Any]:
        if actor is None:
            principal = self.kernel._session_principal(thread_id)
            payload = principal.to_dict()
            payload["attributes"] = self.kernel.read_record_content(
                thread_id,
                1,
                principal.attributes,
                slot=SESSION_PRINCIPAL_ATTRIBUTES,
            )
        else:
            payload = _principal_payload(actor)
        if payload["tenant_id"] != self.kernel.tenant_id:
            raise ValueError("input actor must use the session tenant")
        return payload

    def _controller_actor_matches(
        self,
        thread_id: str,
        item: InboxItem,
        actor_payload: dict[str, Any],
    ) -> bool:
        retained = item.actor
        if retained is None:
            return False
        retained_payload = {key: retained[key] for key in ("tenant_id", "subject_id", "subject_kind")}
        retained_attributes = self.kernel.read_input_content(
            thread_id,
            item.input_id,
            retained["attributes"],
            slot=INPUT_ACTOR_ATTRIBUTES,
        )
        retained_payload["attributes"] = retained_attributes
        return canonical_json_bytes(retained_payload) == canonical_json_bytes(actor_payload)

    def controller_action(
        self,
        *,
        thread_id,
        turn_id,
        action_id,
        action,
        connection_id,
        actor: PrincipalInput | None = None,
    ):
        from vv_agent.app_server.protocol.identity import client_id

        for identity, name in ((thread_id, "threadId"), (turn_id, "turnId"), (action_id, "actionId")):
            self._validate_public_controller_action_identity(identity, name)
        try:
            client_id(action_id, "actionId")
        except ValueError as exc:
            raise TurnResumeError(str(exc)) from exc
        self._validate_public_controller_action(action)
        input_id = "controller/" + derive_controller_command_id(thread_id, turn_id, action_id)
        actor_payload = self._controller_actor_payload(thread_id, actor)
        previous = self.kernel._retained_reply(thread_id, input_id)
        if previous:
            retained = previous[1]
            if not self._controller_actor_matches(previous[0], retained, actor_payload):
                raise SubmissionConflict("submission_conflict")
            if retained.kind == "user":
                retained_content = hydrate_input(retained, self._hydration(previous[0])).payload.get("content")
                if not isinstance(retained_content, dict):
                    raise SubmissionConflict("submission_conflict")
                original = {"kind": "respond", **retained_content["metadata"]["interaction_response"]}
            elif retained.kind == "control":
                original = {"kind": retained.payload["action"]}
            else:
                original = None
            if canonical_json_bytes(original) != canonical_json_bytes(action):
                raise SubmissionConflict("submission_conflict")
            return {"threadId": thread_id, "turnId": turn_id, "actionId": action_id, "accepted": True, "status": "running"}
        from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget

        from .interactions import InteractionError
        from .tool_result_read import ToolResultReadError, validate_child_graph

        if action["kind"] == "respond":
            target = InteractionReplyTarget.from_dict(action["target"])
            if target.session_id != thread_id:
                if not self._host.authorize(
                    connection_id,
                    "turn/action",
                    {"threadId": target.session_id, "turnId": target.turn_id, "actionId": action_id, "action": action},
                ):
                    raise PermissionError("permission_denied")
                try:
                    _, root_turn = validate_child_graph(self.kernel.store, thread_id, target.session_id)
                except ToolResultReadError:
                    raise InteractionError("stale_interaction") from None
                if root_turn != turn_id:
                    raise TurnMismatch("Turn does not own the child interaction")
            target_state, _, _ = self.kernel.store.read_state(target.session_id)
            if target.turn_id not in target_state.turns:
                raise TurnMismatch("Turn does not belong to the target session")
        state, _, _ = self.kernel.store.read_state(thread_id)
        if turn_id not in state.turns:
            raise TurnMismatch("Turn does not belong to the requested thread")
        if state.active_turn_id is not None and state.active_turn_id != turn_id:
            raise TurnMismatch("Thread has a different active turn")
        runtime = self._store._runtime_view(thread_id)
        try:
            if action["kind"] == "respond":
                receipt = self.kernel.answer(
                    thread_id,
                    runtime,
                    target,
                    InteractionAnswer.from_dict(action["answer"]),
                    input_id,
                    actor=actor_payload,
                )
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
                    actor=actor_payload,
                    actor_subject_id="app-server",
                )
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        except InteractionError:
            raise
        except ValueError as exc:
            raise TurnResumeError(str(exc)) from exc
        active = self._state_manager.active_turn(thread_id)
        if active is None and not receipt.replayed and not state.turns[turn_id].ended:
            self.recover(connection_id, thread_id)
        return {"threadId": thread_id, "turnId": turn_id, "actionId": action_id, "accepted": True, "status": "running"}

    def resolve_approval(
        self,
        *,
        connection_id,
        thread_id,
        turn_id,
        request_id,
        decision,
        actor: PrincipalInput | None = None,
    ):
        from vv_agent.approval import ApprovalDecision

        state, _, _ = self.kernel.store.read_state(thread_id)
        previous = self.kernel._retained_reply(thread_id, f"approval/{request_id}/answer")
        if state.active_turn_id != turn_id and not previous:
            raise TurnResumeError("Turn does not belong to the active thread")
        runtime = self._store._runtime_view(thread_id)
        try:
            self.kernel.approve(
                thread_id,
                runtime,
                request_id,
                ApprovalDecision(**decision),
                f"approval/{request_id}/answer",
                actor=actor,
            )
        except Conflict as exc:
            raise SubmissionConflict("submission_conflict") from exc
        # The durable answer is authoritative. Release any legacy provider
        # waiter, then wake the sole local worker to consume the inbox answer.
        self._router.cancel_server_request(RequestId(request_id))
        self.kernel._worker.wake(thread_id)
        active = self._state_manager.active_turn(thread_id)
        if active is None or active.handle.done():
            self.recover(connection_id, thread_id)
        else:
            self.kernel._worker.post_observation(self._observe_with_driver)

    def _wait_for_local_claim_handoff(self, session_id: str) -> bool:
        worker = self.kernel._worker
        deadline = self.clock.monotonic() + max(5.0, worker.scan_interval * 4)
        # A completed previous drive can remain visible until the coordinator
        # reaps it. Drain that metadata before observing a fresh pass so a stale
        # local lease cannot be mistaken for this handoff.
        worker.wait_idle(max(0.0, deadline - self.clock.monotonic()))
        # The owner can release between two scans. Check two complete passes so
        # that release race cannot strand runnable work after ``join`` returns.
        for _ in range(2):
            previous = worker.coordinate_pass
            worker.wake(session_id)
            worker.wait_for_coordinate_pass(previous, max(0.0, deadline - self.clock.monotonic()))
            inflight = worker._inflight_lease
            if inflight is not None and inflight.session_id == session_id:
                return True
        return False

    def _join_drive_plan(self, session_id: str) -> _JoinDrivePlan:
        worker = self.kernel._worker
        head, _, lease_ms = self.kernel.store.lease_status(session_id)
        inflight = worker._inflight_lease
        if inflight is not None and inflight.session_id == session_id:
            return _JoinDrivePlan(_JoinDriveState.LOCAL_LEASE, head)
        if lease_ms > 0:
            # A wake that has claimed but not yet handed its lease to the
            # executor is local work. Without that worker marker the lease
            # belongs to another node and join must only observe it.
            state = _JoinDriveState.LOCAL_LEASE if session_id in worker.active_wake_sessions else _JoinDriveState.REMOTE_LEASE
            return _JoinDrivePlan(state, head)
        if self.kernel.store.is_runnable(session_id):
            return _JoinDrivePlan(_JoinDriveState.DUE, head)
        delay_ms = self.kernel.store.next_drive_delay_ms(session_id)
        if delay_ms is None:
            return _JoinDrivePlan(_JoinDriveState.NONE, head)
        if isinstance(delay_ms, bool) or not isinstance(delay_ms, int):
            raise TypeError("next_drive_delay_ms() must return integer milliseconds or None")
        state = _JoinDriveState.DUE if delay_ms == 0 else _JoinDriveState.FUTURE
        return _JoinDrivePlan(state, head)

    def _join_session(self, session_id: str) -> None:
        worker = self.kernel._worker
        while True:
            plan = self._join_drive_plan(session_id)
            if plan.state is _JoinDriveState.LOCAL_LEASE:
                inflight = worker._inflight_lease
                if (inflight is None or inflight.session_id != session_id) and not self._wait_for_local_claim_handoff(session_id):
                    return
                worker.wait_idle()
            elif plan.state is _JoinDriveState.DUE:
                if not self._wait_for_local_claim_handoff(session_id):
                    return
                worker.wait_idle()
            elif plan.state is _JoinDriveState.REMOTE_LEASE:
                # Observe the explicit join with one local coordination pass.
                # The remote owner itself is never awaited.
                previous = worker.coordinate_pass
                worker.wake(session_id)
                if worker.started:
                    worker.wait_for_coordinate_pass(previous, worker.scan_interval * 4)
                return
            else:
                return
            if self.kernel.store.lease_status(session_id)[0] == plan.head:
                return

    def join(self):
        """Wait for this node's accepted work without joining a remote lease."""

        handles = tuple(handle for handle in self.kernel.handles if handle.consumer == "app_server")
        wake_sessions = self.kernel._worker.active_wake_sessions
        if not self._notification_positions and not handles and not wake_sessions and not self.kernel._worker.started:
            return
        if self.kernel._worker.started and not self.kernel._worker.running:
            return
        self.kernel._worker.post_observation(self._observe_with_driver)
        self.kernel._worker.wait_idle()
        handles = tuple(handle for handle in self.kernel.handles if handle.consumer == "app_server")
        session_ids = dict.fromkeys(
            (
                *self._notification_positions,
                *self.kernel._worker.active_wake_sessions,
                *(handle.session_id for handle in handles),
            )
        )
        for sid in session_ids:
            self._join_session(sid)
        # Reconcile the terminal record, operational error, and released cursor.
        self.kernel._worker.post_observation(self._observe_with_driver)
        self.kernel._worker.wait_idle()
