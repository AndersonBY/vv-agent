"""Session host assembly and retained runtime bindings."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import RLock, get_ident
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalDecision
from vv_agent.memory import MemoryManager
from vv_agent.model import resolve_model_selection, resolve_run_model
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.run_handle import RunHandle
from vv_agent.types import AgentTask, Message

from .children import child_handles
from .context import project_context
from .kernel import drive
from .ownership import checkout
from .records import InboxItem, SessionSpec
from .result import project_result
from .runtime import Runtime
from .sql import SQLStore
from .sqlite import SQLiteStore
from .store import Conflict, SessionStore

_DRIVERS: WeakValueDictionary[str, SessionDriver] = WeakValueDictionary()
_STORE_CLOSERS: WeakKeyDictionary[SessionStore, list[Any]] = WeakKeyDictionary()


class SessionDriver:
    def __init__(self, path: str | Path = ":memory:", *, store: SessionStore | None = None, model_resolver=None) -> None:
        self.model_resolver = model_resolver
        self._owns_store = store is None
        self.store = store if store is not None else SQLiteStore(str(path))
        self.path = self.store.path if isinstance(self.store, SQLiteStore) else None
        # Workers, handles and resumable results retain this owner. Once none
        # remain, close only the connection this driver created itself.
        self._owned_connection = (
            finalize(self, self.store.connection.close) if self._owns_store and isinstance(self.store, SQLiteStore) else None
        )
        if isinstance(self.store, SQLiteStore) and not self.store.connection.execute("PRAGMA user_version").fetchone()[0]:
            self.store.install_schema()
        self.handles: list[RunHandle] = []
        self._background: dict[str, RunHandle] = {}
        self._background_lock = RLock()
        self._close_callbacks = []

    @contextmanager
    def _connection_scope(self, *, independent=True):
        if independent and isinstance(self.store, SQLStore):
            with self.store._connection_scope():
                yield
        else:
            yield

    def _projection_store(self):
        return checkout(self.store, role="projection")

    def _after_admission_commit(self, key, callback):
        if isinstance(self.store, SQLStore):
            transactions = tuple(self.store._atomic_stack)
            thread = get_ident()
            current = next((tx for tx in reversed(transactions) if tx._thread == thread and tx._scope_active), None)
            if current is not None:
                current._commit_callbacks[key] = callback
                return
            if not transactions:
                try:
                    self.store._transaction_id()
                except RuntimeError:
                    pass
                else:
                    # A native outer transaction is the host's on_commit responsibility.
                    return
        callback()

    def _wake_after_commit(self, sid):
        self._after_admission_commit(("wake", sid), lambda: self._wake(sid))

    def _control_runtime(self, sid: str) -> Runtime:
        from vv_agent.model import ScriptedModelProvider

        provider = ScriptedModelProvider.new("local", "control", [])
        return self.runtime(Agent("control", "Apply session controls.", model="control"), RunConfig(model_provider=provider))

    def close(self) -> None:
        for handle in self.handles:
            handle.join()
        callbacks = _STORE_CLOSERS.pop(self.store, []) if self._owns_store else list(self._close_callbacks)
        for callback in callbacks:
            callback()
            retained = _STORE_CLOSERS.get(self.store, [])
            if callback in retained:
                retained.remove(callback)
        self._close_callbacks.clear()
        if self._owned_connection is not None:
            with self.store._connection_lock:
                self._owned_connection()
                self.store._checkout_closed = True
            from .ownership import forget_host

            forget_host(self.store)

    def _on_close(self, callback):
        self._close_callbacks.append(callback)
        _STORE_CLOSERS.setdefault(self.store, []).append(callback)

    def create(self, sid: str, workspace: str, attributes: dict[str, Any] | None = None) -> None:
        _DRIVERS[sid] = self
        spec = SessionSpec(sid, "local", workspace, attributes=attributes)
        with self.store.atomic() as tx:
            tx.create(spec, consumers=("events", "app_server", "traces"))

    def runtime(self, agent: Agent, config: RunConfig, task: AgentTask | None = None) -> Runtime:
        llm, resolved = resolve_run_model(agent=agent, run_config=config)
        metadata = dict(agent.metadata) | config.metadata
        settings = config.model_provider.default_settings(resolved) if config.model_provider else agent.model_settings
        settings = settings.resolve(agent.model_settings).resolve(config.model_settings) if settings else config.model_settings
        reserved = settings.max_tokens if settings else None
        source = "model_settings" if reserved is not None else "framework_fallback"
        if reserved is None and isinstance(metadata.get("reserved_output_tokens"), int):
            reserved, source = metadata["reserved_output_tokens"], "task_metadata"
        if reserved is None:
            reserved = min(16_000, resolved.max_output_tokens) if resolved.max_output_tokens is not None else 16_000
            if reserved < 16_000:
                source = "framework_fallback_capped_by_model_capability"
        memory = MemoryManager(
            model_context_window=resolved.context_length or metadata.get("model_context_window") or 279_000,
            model_max_output_tokens=resolved.max_output_tokens,
            reserved_output_tokens=reserved,
            reserved_output_source=source,
            compact_threshold=task.memory_compact_threshold if task else 250_000,
            autocompact_buffer_tokens=metadata.get("autocompact_buffer_tokens", 13_000),
            keep_recent_messages=metadata.get("memory_keep_recent_messages", 10),
            microcompaction_policy=task.microcompaction_policy if task else config.microcompaction_policy,
        )
        runtime = Runtime(
            agent,
            config,
            resolved,
            llm,
            self._heartbeat_store,
            providers=dict(config.tool_providers),
            frozen_task=task,
            memory_manager=memory,
            model_resolver=self.model_resolver,
        )
        provider = config.model_provider
        if self.model_resolver is None and provider is not None:
            runtime.model_resolver = lambda request: resolve_model_selection(provider, request)
        return runtime

    def _heartbeat_store(self):
        return self.store.heartbeat_store()

    def start(
        self,
        sid: str,
        agent: Agent,
        config: RunConfig,
        content: Any | None,
        *,
        input_id: str | None = None,
        task: AgentTask | None = None,
        consumer: str = "events",
        autostart: bool = True,
    ) -> RunHandle:
        with self._projection_store() as projection:
            state, _, _ = projection.read_state(sid)
            pending_inputs = projection.peek_inbox(sid)
            if state.turns and config.shared_state is None:
                prior = project_result(projection, sid, next(reversed(state.turns)))
                config = replace(config, shared_state=prior.raw_result.shared_state)
        if task is not None and config.shared_state is not None:
            task = replace(task, initial_shared_state=dict(config.shared_state))
        runtime = self.runtime(agent, config, task)
        if state.closed or state.archived:
            raise RuntimeError("session is closed or archived")
        if content is not None:
            identity = input_id or uuid4().hex
            self.push(sid, InboxItem(identity, "user", {"content": content}))
            tid = f"{sid}/turn/{identity}"
        else:
            queued = next(
                (
                    r._payload["input"]
                    for key, r in state.applied_inputs.items()
                    if r._payload["disposition"] == "queued" and key not in state.admitted_inputs
                ),
                None,
            )
            pending = next(
                (s.item for s in pending_inputs if s.item.kind in {"user", "follow_up"} and s.item.target_turn_id is None),
                None,
            )
            tid = state.active_turn_id or (
                f"{sid}/turn/{queued['input_id']}"
                if queued
                else f"{sid}/turn/{pending.input_id}"
                if pending
                else next(reversed(state.turns), "")
            )
        handle = RunHandle(self, sid, tid, runtime, consumer)
        self.handles.append(handle)
        if autostart:
            handle.start()
        return handle

    def push(self, sid: str, item: InboxItem):
        with self.store.atomic() as tx:
            receipt = tx.push(sid, item)
        # Typed replies, controls and new turns have their own explicit drive owner.
        if item.kind in {"steer", "provider_result", "provider_evidence", "child_result", "control", "approval_answer"} or (
            item.kind == "user" and item.target_turn_id is not None
        ):
            self._wake_after_commit(sid)
        return receipt

    def _wake(self, sid: str) -> bool:
        if isinstance(self.store, SQLiteStore) and self.store._defer_memory_wake(sid, lambda: self._wake(sid)):
            return True
        with self._background_lock:
            for handle in reversed(self.handles):
                if sid in handle._observed_sessions and handle._wake():
                    return True
        return False

    def control(self, sid: str, action: str, input_id: str, *, runtime: Runtime | None = None):
        admission = self.store
        with admission.atomic() as tx:
            state, _, _ = admission.read_state(sid)
            tid = None if action in {"archive", "close"} else state.active_turn_id
            receipt = tx.push(sid, InboxItem(input_id, "control", {"action": action}, tid))
        self._wake_after_commit(sid)
        if runtime is None and state.active_turn_id is None:
            runtime = self._control_runtime(sid)
        if runtime is not None:

            def execute_control():
                drive(admission, sid, runtime=runtime, _one_turn=True)

            self._after_admission_commit(("control-drive", sid), execute_control)
        return receipt

    def messages(self, sid: str) -> list[Message]:
        with self._projection_store() as projection:
            state, records, _ = projection.read_state(sid)
            return [m for m in project_context(records, state) if m.role != "system"]

    def waits(
        self, sid: str, runtime: Runtime | None = None, *, snapshot=None
    ) -> list[tuple[str, Runtime | None, dict[str, Any]]]:
        with self._projection_store() as projection:
            return self._waits_on(projection, sid, runtime, snapshot, frozenset())

    def _waits_on(self, store, sid, runtime, snapshot, ancestors):
        if sid in ancestors:
            raise Conflict(f"cyclic child ownership: {sid}")
        if ancestors and runtime is not None:
            runtime = runtime.child_runtime(store, sid)
        ancestors = ancestors | {sid}
        state, _records, _ = snapshot or store.read_state(sid)
        from .interactions import user_interaction

        found = []
        tid = state.active_turn_id
        if tid is None:
            return found
        turn_wait = state.turns[tid].wait
        if turn_wait:
            found.append(
                (
                    sid,
                    runtime,
                    {
                        "interaction_id": turn_wait._payload["interaction_id"],
                        "question": turn_wait._payload["question"],
                        "turn_id": tid,
                        "interaction": user_interaction(
                            sid, tid, state.turns[tid].start._payload["generation"], turn_wait._payload
                        ),
                    },
                )
            )
        for (oid, number), wait in state.waits.items():
            if state.operations[oid].turn_id != tid:
                continue
            handle = wait["handle"]
            if handle["kind"] == "child":
                for child in child_handles(handle):
                    child_id = child["session_id"]
                    found.extend(self._waits_on(store, child_id, runtime, None, ancestors))
            elif handle["kind"] in {"user", "approval"}:
                public = {}
                if handle["kind"] == "user":
                    public["interaction"] = user_interaction(
                        sid, tid, state.turns[tid].start._payload["generation"], wait, operation_id=oid, attempt=number
                    )
                found.append((sid, runtime, handle | {"operation_id": oid, "attempt": number, "turn_id": tid} | public))
        # Background admissions retain a child handle after their parent tool completes.
        seen = {child_sid for child_sid, _, _ in found}
        for op in state.operations.values():
            if op.turn_id != tid:
                continue
            for admitted in op.attempts.values():
                if admitted.child_handle is None:
                    continue
                for child in child_handles(admitted.child_handle):
                    child_id = child["session_id"]
                    if (child["background"] or admitted.yielded) and child_id not in seen:
                        found.extend(self._waits_on(store, child_id, runtime, None, ancestors))
                        seen.add(child_id)
        return found

    def answer(self, sid: str, runtime: Runtime, target, answer, input_id: str):
        from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget

        from .interactions import InteractionError, reply_content, validate_reply
        from .tool_result_read import ToolResultReadError, validate_child_graph

        if not isinstance(target, InteractionReplyTarget) or not isinstance(answer, InteractionAnswer):
            raise TypeError("answer requires InteractionReplyTarget and InteractionAnswer")
        with self._projection_store() as projection:
            try:
                chain, root_turn = validate_child_graph(projection, sid, target.session_id)
            except ToolResultReadError:
                raise InteractionError("stale_interaction") from None
            previous = self._retained_reply_on(projection, sid, input_id)
            item = InboxItem(input_id, "user", {"content": reply_content(target, answer)}, target.turn_id, target.generation)
            if previous:
                if previous[0] != target.session_id or previous[1].encode() != item.encode():
                    raise Conflict("input_id reused with different bytes")
                replayed = True
            else:
                replayed = False
            if not replayed:
                captured = {identity: projection.read_state(identity) for identity in [sid, *chain]}
                try:
                    validate_child_graph(projection, sid, target.session_id, active=True)
                except ToolResultReadError:
                    raise InteractionError("stale_interaction") from None
                root_state, _, _ = captured[sid]
                if target.session_id != sid and root_turn != root_state.active_turn_id:
                    raise InteractionError("stale_interaction")
                state, _, _ = captured[target.session_id]
                validate_reply(state, target.session_id, target, answer)
        if replayed:
            return self.push(target.session_id, item)
        admission = self.store
        with admission.atomic() as tx:
            # Prepare outside row locks; recheck the captured heads before mutation.
            for identity in [sid, *chain]:
                tx.lock_session(identity)
                if admission.lease_status(identity)[0] != captured[identity][1][-1].seq:
                    raise InteractionError("stale_interaction")
            receipt = tx.push(target.session_id, item)
        self._wake_after_commit(target.session_id)
        return receipt

    def approve(self, sid: str, runtime: Runtime, request_id: str, decision: ApprovalDecision | str, input_id: str):
        if isinstance(decision, str):
            decision = ApprovalDecision.from_input(decision)
        previous = self._retained_reply(sid, input_id)
        if previous:
            target, item = previous
            if item.kind != "approval_answer":
                raise Conflict("input_id reused with different bytes")
            payload = item.payload | {
                "request_id": request_id,
                "decision": "approve" if decision.action == "allow" else decision.action,
                "reason": decision.reason,
                "metadata": decision.metadata,
            }
            return self.push(target, replace(item, payload=payload))
        waits = [w for w in self.waits(sid, runtime) if w[2].get("request_id") == request_id]
        if len(waits) != 1:
            raise KeyError(f"Unknown approval request: {request_id}")
        target, _, wait = waits[0]
        admission = self.store
        with admission.atomic() as tx:
            tx.lock_session(target)
            state, _, _ = admission.read_state(target)
            payload = {k: wait[k] for k in ("operation_id", "attempt", "request_id", "request_digest", "scope")}
            payload.update(
                decision="approve" if decision.action == "allow" else decision.action,
                reason=decision.reason,
                metadata=decision.metadata,
            )
            receipt = tx.push(
                target,
                InboxItem(
                    input_id,
                    "approval_answer",
                    payload,
                    wait["turn_id"],
                    state.turns[wait["turn_id"]].start._payload["generation"],
                ),
            )
        self._wake_after_commit(target)
        return receipt

    def _retained_reply(self, sid: str, input_id: str):
        with self._projection_store() as projection:
            return self._retained_reply_on(projection, sid, input_id)

    def _retained_reply_on(self, store, sid: str, input_id: str):
        state, records, _ = store.read_state(sid)
        if input_id in state.applied_inputs:
            return sid, InboxItem(**state.applied_inputs[input_id]._payload["input"])
        for pending in store.peek_inbox(sid):
            if pending.item.input_id == input_id:
                return sid, pending.item
        for stored in records:
            r = stored.record
            if r.kind == "op_parked" and r._payload["handle"]["kind"] == "child":
                for child in child_handles(r._payload["handle"]):
                    previous = self._retained_reply_on(store, child["session_id"], input_id)
                    if previous:
                        return previous
        return None


def resume_turn(session_id: str, turn_id: str) -> RunResult:
    from .bindings import MissingHostBinding

    if not isinstance(session_id, str) or not isinstance(turn_id, str):
        raise TypeError("resume requires session_id and turn_id strings")
    driver = _DRIVERS.get(session_id)
    if driver is None:
        raise MissingHostBinding("resume requires the original host runtime; use AgentSession for durable sessions")
    with driver._projection_store() as projection:
        state, _, _ = projection.read_state(session_id)
    if state.closed or state.archived:
        raise ValueError("session is closed or archived")
    if turn_id not in state.turns or state.active_turn_id not in {None, turn_id}:
        raise ValueError("turn_id does not identify the resumable turn")
    handle = next((h for h in reversed(driver.handles) if h.session_id == session_id and h.run_id == turn_id), None)
    if handle is None:
        raise MissingHostBinding("resume requires the original host runtime")
    if not handle.done():
        return handle.result()
    if state.turns[turn_id].ended:
        return handle.result()
    return driver.start(session_id, handle.runtime.agent, handle.runtime.config, None).result()
