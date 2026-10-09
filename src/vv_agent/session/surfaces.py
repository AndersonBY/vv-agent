"""Session host assembly and retained runtime bindings."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalDecision
from vv_agent.memory import MemoryManager
from vv_agent.model import resolve_run_model
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.run_handle import RunHandle
from vv_agent.types import AgentTask, Message

from .children import child_handles
from .context import project_context
from .kernel import drive
from .postgres import PostgresStore
from .records import InboxItem, SessionSpec
from .result import project_result
from .runtime import Runtime
from .sqlite import SQLiteStore
from .store import Conflict, SessionStore

_DRIVERS: WeakValueDictionary[str, SessionDriver] = WeakValueDictionary()


class SessionDriver:
    def __init__(self, path: str | Path = ":memory:", *, store: SessionStore | None = None) -> None:
        self._owns_store = store is None
        self.store = store if store is not None else SQLiteStore(str(path))
        self.path = self.store.path if isinstance(self.store, SQLiteStore) else None
        if isinstance(self.store, SQLiteStore) and not self.store.connection.execute("PRAGMA user_version").fetchone()[0]:
            self.store.install_schema()
        self.handles: list[RunHandle] = []
        self._background: dict[str, RunHandle] = {}
        self._background_lock = RLock()

    def _control_runtime(self, sid: str) -> Runtime:
        from vv_agent.model import ScriptedModelProvider

        provider = ScriptedModelProvider.new("local", "control", [])
        return self.runtime(Agent("control", "Apply session controls.", model="control"), RunConfig(model_provider=provider))

    def close(self) -> None:
        for handle in self.handles:
            handle.join()
        if self._owns_store and isinstance(self.store, SQLiteStore):
            self.store.connection.close()

    def create(self, sid: str, workspace: str, attributes: dict[str, Any] | None = None) -> None:
        _DRIVERS[sid] = self
        with self.store.atomic() as tx:
            tx.create(SessionSpec(sid, "local", workspace, attributes=attributes), consumers=("events", "app_server", "traces"))

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
        return Runtime(
            agent,
            config,
            resolved,
            llm,
            self._heartbeat_store,
            frozen_task=task,
            memory_manager=memory,
        )

    def _heartbeat_store(self):
        if isinstance(self.store, SQLiteStore) and self.store.path != ":memory:":
            return SQLiteStore.standalone(self.store.path)
        if isinstance(self.store, PostgresStore):
            return PostgresStore.standalone(self.store.connection.info.dsn)
        return nullcontext(self.store)

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
        state, _, _ = self.store.read_state(sid)
        if state.turns and config.shared_state is None:
            prior = project_result(self.store, sid, next(reversed(state.turns)))
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
            tid = state.active_turn_id or (f"{sid}/turn/{queued['input_id']}" if queued else next(reversed(state.turns), ""))
        handle = RunHandle(self, sid, tid, runtime, consumer)
        self.handles.append(handle)
        if autostart:
            handle.start()
        return handle

    def push(self, sid: str, item: InboxItem):
        with self.store.atomic() as tx:
            return tx.push(sid, item)

    def control(self, sid: str, action: str, input_id: str, *, runtime: Runtime | None = None):
        state, _, _ = self.store.read_state(sid)
        tid = None if action in {"archive", "close"} else state.active_turn_id
        receipt = self.push(sid, InboxItem(input_id, "control", {"action": action}, tid))
        if runtime is None and state.active_turn_id is None:
            runtime = self._control_runtime(sid)
        if runtime is not None:
            drive(self.store, sid, runtime=runtime, _one_turn=True)
        return receipt

    def messages(self, sid: str) -> list[Message]:
        state, records, _ = self.store.read_state(sid)
        return [m for m in project_context(records, state) if m.role != "system"]

    def waits(self, sid: str, runtime: Runtime) -> list[tuple[str, Runtime, dict[str, Any]]]:
        state, _records, _ = self.store.read_state(sid)
        found = []
        tid = state.active_turn_id
        turn_wait = state.turns[tid].wait if tid else None
        if turn_wait:
            found.append(
                (
                    sid,
                    runtime,
                    {
                        "interaction_id": turn_wait._payload["interaction_id"],
                        "question": turn_wait._payload["question"],
                        "turn_id": tid,
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
                    found.extend(self.waits(child_id, runtime.child_runtime(self.store, child_id)))
            elif handle["kind"] in {"user", "approval"}:
                found.append((sid, runtime, handle | {"operation_id": oid, "attempt": number, "turn_id": tid}))
        return found

    def answer(self, sid: str, runtime: Runtime, text: str, input_id: str):
        previous = self._retained_reply(sid, input_id)
        if previous:
            target, item = previous
            if item.kind != "user":
                raise Conflict("input_id reused with different bytes")
            content = item.payload["content"] | {"text": text}
            return self.push(target, replace(item, payload={"content": content}))
        waits = [w for w in self.waits(sid, runtime) if "interaction_id" in w[2]]
        if len(waits) != 1:
            raise ValueError("reply requires exactly one pending user interaction")
        target, _, wait = waits[0]
        state, _, _ = self.store.read_state(target)
        content = {"interaction_id": wait["interaction_id"], "text": text}
        if "operation_id" in wait:
            content["operation_id"] = wait["operation_id"]
        return self.push(
            target,
            InboxItem(
                input_id, "user", {"content": content}, wait["turn_id"], state.turns[wait["turn_id"]].start._payload["generation"]
            ),
        )

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
        state, _, _ = self.store.read_state(target)
        payload = {k: wait[k] for k in ("operation_id", "attempt", "request_id", "request_digest", "scope")}
        payload.update(
            decision="approve" if decision.action == "allow" else decision.action,
            reason=decision.reason,
            metadata=decision.metadata,
        )
        return self.push(
            target,
            InboxItem(
                input_id, "approval_answer", payload, wait["turn_id"], state.turns[wait["turn_id"]].start._payload["generation"]
            ),
        )

    def _retained_reply(self, sid: str, input_id: str):
        state, records, _ = self.store.read_state(sid)
        if input_id in state.applied_inputs:
            return sid, InboxItem(**state.applied_inputs[input_id]._payload["input"])
        for pending in self.store.peek_inbox(sid):
            if pending.item.input_id == input_id:
                return sid, pending.item
        for stored in records:
            r = stored.record
            if r.kind == "op_parked" and r._payload["handle"]["kind"] == "child":
                for child in child_handles(r._payload["handle"]):
                    previous = self._retained_reply(child["session_id"], input_id)
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
    state, _, _ = driver.store.read_state(session_id)
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
