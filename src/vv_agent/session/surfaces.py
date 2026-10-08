"""Private host assembly selector. No environment switch or public export."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import Any
from uuid import uuid4

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalDecision
from vv_agent.events import RunEvent
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.runner import Runner
from vv_agent.types import AgentTask, Message

from .children import child_delivery, child_handles
from .context import project_context
from .events import SessionRunEventStore
from .kernel import drive
from .records import InboxItem, SessionSpec
from .result import project_result
from .runtime import Runtime
from .sqlite import SQLiteStore
from .store import Conflict, LeaseLost


class _SessionKernel:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.store = SQLiteStore(self.path)
        if not self.store.connection.execute("PRAGMA user_version").fetchone()[0]:
            self.store.install_schema()
        self.handles: list[_KernelHandle] = []
        self._background: dict[str, _KernelHandle] = {}
        self._background_lock = RLock()

    def _control_runtime(self, sid: str) -> Runtime:
        from vv_agent.model import ScriptedModelProvider

        provider = ScriptedModelProvider.new("local", "control", [])
        return self.runtime(Agent("control", "Apply session controls.", model="control"), RunConfig(model_provider=provider))

    def close(self) -> None:
        for handle in self.handles:
            handle.join()
        self.store.connection.close()

    def create(self, sid: str, workspace: str, attributes: dict[str, Any] | None = None) -> None:
        with self.store.atomic() as tx:
            tx.create(SessionSpec(sid, "local", workspace, attributes=attributes), consumers=("events", "app_server", "traces"))

    def runtime(self, agent: Agent, config: RunConfig, task: AgentTask | None = None) -> Runtime:
        llm, resolved = Runner._resolve_model(agent=agent, run_config=config)
        return Runtime(
            agent,
            replace(config, session=None),
            resolved,
            llm,
            (lambda: nullcontext(self.store)) if self.path == ":memory:" else (lambda: SQLiteStore.standalone(self.path)),
            frozen_task=task,
        )

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
    ) -> _KernelHandle:
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
        handle = _KernelHandle(self, sid, tid, runtime, consumer)
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


class _KernelHandle:
    def __init__(self, kernel: _SessionKernel, sid: str, tid: str, runtime: Runtime, consumer: str | None) -> None:
        self.kernel, self.session_id, self.run_id, self.runtime, self.consumer = kernel, sid, tid, runtime, consumer
        self._condition = Condition()
        self._events: deque[RunEvent] = deque()
        self._done = False
        self._error: BaseException | None = None
        self._deliver_to_parent = False
        self._thread = Thread(target=self._run, name="session-surface-driver")
        original_hook = runtime.hook

        def hook(point, record):
            original_hook(point, record)
            if point == "after_commit":
                self._publish()
                self._schedule_background()

        runtime.hook = hook

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def _publish(self) -> None:
        if self.consumer is None:
            return

        def sink(event):
            if self.runtime.config.stream:
                self.runtime.config.stream(event)
            with self._condition:
                self._events.append(event)
                self._condition.notify_all()

        events = SessionRunEventStore(self.kernel.store, self.session_id, self.consumer)
        while events.consume(sink):
            pass

    def _schedule_background(self) -> None:
        _, records, _ = self.kernel.store.read_state(self.session_id)
        for stored in records:
            r = stored.record
            if r.kind != "op_parked" or r._payload["handle"]["kind"] != "child" or not r._payload["handle"]["background"]:
                continue
            for child in child_handles(r._payload["handle"]):
                sid = child["session_id"]
                with self.kernel._background_lock:
                    if sid in self.kernel._background:
                        continue
                    state, _, _ = self.kernel.store.read_state(sid)
                    if state.turns and state.active_turn_id is None:
                        continue
                    handle = _KernelHandle(
                        self.kernel, sid, child["turn_id"], self.runtime.child_runtime(self.kernel.store, sid), None
                    )
                    handle._deliver_to_parent = True
                    self.kernel._background[sid] = handle
                    self.kernel.handles.append(handle)
                    handle.start()

    def _drive_tree(self, sid: str, runtime: Runtime, tid: str) -> None:
        while True:
            try:
                drive(self.kernel.store, sid, runtime=runtime, _one_turn=True, _wait_for_lease=True)
            except LeaseLost:
                state, _, _ = self.kernel.store.read_state(sid)
                # A terminal receipt already committed by this drive must not admit another turn.
                if tid not in state.turns or not state.turns[tid].ended:
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
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._events or self._done)
                if self._events:
                    event = self._events.popleft()
                else:
                    return
            yield event

    def result(self, timeout: float | None = None) -> RunResult:
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
        return result

    def cancel(self, reason: str = "") -> bool:
        del reason
        state, _, _ = self.kernel.store.read_state(self.session_id)
        if state.active_turn_id is None:
            return False
        self.kernel.control(self.session_id, "cancel", uuid4().hex)
        return True
