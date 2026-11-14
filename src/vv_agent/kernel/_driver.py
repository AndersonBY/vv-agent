"""Shared driver state, record production and lease-fenced transaction algorithm."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Protocol

from vv_agent.kernel._lease_scope import Clock
from vv_agent.kernel._records import InboxItem, Record, copy_json, digest, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import Conflict, Lease, SessionStore, SessionTx, StoredRecord


class DriverLeaseScope(Protocol):
    lease: Lease
    turn_id: str | None
    generation: int | None

    def committing(self) -> AbstractContextManager[Lease]: ...


class CoreDriver:
    def __init__(
        self,
        store: SessionStore,
        sid: str,
        scope: DriverLeaseScope,
        *,
        fold_factory: Callable[[], Fold],
        clock: Clock,
        observe_cost: Callable[[], dict[str, Any] | None],
        hook: Callable[[str, Record | None], None],
        emit_phase: Callable[[float, dict[str, Any]], None],
    ) -> None:
        self.store, self.sid, self.scope = store, sid, scope
        self._fold_factory, self._clock = fold_factory, clock
        self._observe_cost, self._hook, self._emit_phase = observe_cost, hook, emit_phase
        self.load_state()
        self.polled: set[tuple[str, int]] = set()
        self.active_since = self.clock.monotonic_ns()
        self.active_tid: str | None = None

    @property
    def clock(self) -> Clock:
        return self._clock

    def load_state(self) -> None:
        self.state, self.records, self.watermark = self.store.read_state(self.sid)
        self.fold = self._fold_factory()
        self.fold.state = self.state
        self.fold.history = list(self.records)
        self.fold.seen = {r.record.record_id: r.record.encode() for r in self.records}
        self.fold.consumed = {key: InboxItem(**r._payload["input"]) for key, r in self.state.applied_inputs.items()}

    def refresh(self) -> None:
        page = self.store.read(self.sid, after_seq=self.records[-1].seq)
        head = page.head_seq
        if head < self.records[-1].seq:
            self.load_state()
        else:
            self.watermark = page.inbox_seq
            self.extend(page.records)
            while self.records[-1].seq < head:
                if not page.records:
                    raise Conflict("log sequence gap")
                page = self.store.read(self.sid, after_seq=self.records[-1].seq, through_seq=head)
                self.extend(page.records)
        self.bind_turn()

    def extend(self, records: tuple[StoredRecord, ...]) -> None:
        if not records:
            return
        if any(r.seq != self.records[-1].seq + index for index, r in enumerate(records, 1)):
            raise Conflict("log sequence gap")
        self.state = self.fold.extend(
            (r.record for r in records),
            consumed_inputs=(InboxItem(**r.record._payload["input"]) for r in records if r.record.kind == "input_applied"),
        )
        self.records += records

    def bind_turn(self) -> None:
        tid = self.state.active_turn_id
        if tid != self.active_tid:
            self.active_since = self.clock.monotonic_ns()
            self.active_tid = tid
        self.scope.turn_id = tid
        self.scope.generation = self.state.turns[tid].start._payload["generation"] if tid else None

    def record(self, kind: str, payload: dict[str, Any], plan: Record | None = None, *, tid: str | None = None) -> Record:
        return make_record(
            kind,
            session_id=self.sid,
            turn_id=plan.turn_id if plan else tid or self.state.active_turn_id,
            operation_id=plan.operation_id if plan else None,
            attempt=plan.attempt if plan else None,
            payload=payload,
        )

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[SessionTx], list[Record]] | None = None,
    ) -> None:
        tid = self.state.active_turn_id
        if tid is not None:
            meter = f"active/{tid}/{self.scope.lease.epoch}"
            cost = self._observe_cost()
            records = [
                self.record(
                    "usage_observed",
                    {
                        "meter_id": meter,
                        "observation": self.state.usage_observations.get(meter, 0) + 1,
                        "mode": "cumulative",
                        "source": "kernel-active-scope",
                        "usage": {
                            "elapsed_ms": (self.clock.monotonic_ns() - self.active_since) // 1000000,
                            "host_cost": cost,
                        },
                    },
                ),
                *records,
            ]
        timings: list[tuple[str, float, str, str | None]] = []
        atomic_start = self.clock.monotonic_ns()
        atomic_error = None

        @contextmanager
        def phase(name: str):
            begin = self.clock.monotonic_ns()
            error = None
            try:
                yield
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                timings.append((name, (self.clock.monotonic_ns() - begin) / 1_000_000, "error" if error else "ok", error))

        try:
            with ExitStack() as latch, self.store.atomic() as tx:
                if prepare is not None:
                    with phase("prepare"):
                        records.extend(prepare(tx))
                with phase("before_commit"):
                    self._hook("before_commit", records[-1] if records else None)
                # SQL append calls this only after encoding and immutable history
                # preparation. Keep the latch through the outer atomic exit.
                if isinstance(tx, SQLSessionTx):
                    tx._lease_guard = lambda: latch.enter_context(self.scope.committing())
                with phase("append"):
                    receipt = tx.append(
                        self.sid,
                        lease=self.scope.lease if isinstance(tx, SQLSessionTx) else latch.enter_context(self.scope.committing()),
                        expected_seq=self.records[-1].seq,
                        commit_id=digest({"record_ids": [r.record_id for r in records], "consumed": list(consumed)}),
                        records=tuple(records),
                        consume_input_ids=consumed,
                        expected_inbox_seq=self.watermark if guarded else None,
                    )
        except BaseException as exc:
            atomic_error = type(exc).__name__
            raise
        finally:
            atomic_ms = (self.clock.monotonic_ns() - atomic_start) / 1_000_000
            source = next((r for r in reversed(records) if r.operation_id), None)
            request_digest = source._payload.get("request_digest") if source else None
            if source and source.operation_id in self.state.operations:
                operation = self.state.operations[source.operation_id]
                attempt = operation.attempts.get(source.attempt or 1)
                if attempt is not None:
                    request_digest = attempt.execution_plan._payload["request_digest"]
            for name, duration, outcome, error in timings:
                self._emit_phase(
                    duration,
                    {
                        "session_id": self.sid,
                        "turn_id": tid,
                        "operation_id": source.operation_id if source else None,
                        "request_digest": request_digest,
                        "epoch": self.scope.lease.epoch,
                        "phase": name,
                        "callback_ms": duration if name != "append" else 0,
                        "append_ms": duration if name == "append" else 0,
                        "kernel_atomic_ms": atomic_ms,
                        "outcome": "error" if atomic_error else outcome,
                        "exception_type": error or atomic_error,
                    },
                )
        self._hook("after_commit", records[-1] if records else None)
        if receipt.replayed:
            self.refresh()
        else:
            self.extend(receipt.records)
            self.watermark = receipt.inbox_seq
            self.bind_turn()

    def boundary(self, stage: str, identity: str) -> Record | None:
        return self.state.boundaries.get((self.state.active_turn_id or "", stage, identity))

    def boundary_record(
        self,
        stage: str,
        identity: str,
        data: dict[str, Any],
        *,
        source_operation_id: str | None = None,
        source_digest: str | None = None,
    ) -> Record:
        return self.record(
            "boundary_recorded",
            {
                "stage": stage,
                "boundary_id": identity,
                "source_operation_id": source_operation_id,
                "source_digest": source_digest,
                "data": data,
            },
        )

    def durable_shared_state(self) -> dict[str, Any]:
        for stored in reversed(self.records):
            r = stored.record
            if r.turn_id != self.state.active_turn_id:
                continue
            if r.kind == "boundary_recorded" and "shared_state" in r._payload["data"]:
                return copy_json(r._payload["data"]["shared_state"])
            if r.kind == "op_prepared":
                return copy_json(r._payload["shared_state"])
            if r.kind == "op_completed" and r._payload["context"] == "normal":
                snapshot = r._payload["shared_state"]
                if snapshot is not None:
                    return copy_json(snapshot)
        assert self.state.active_turn_id is not None
        return copy_json(self.state.turns[self.state.active_turn_id].start._task().initial_shared_state)

    def pending_batch(self) -> bool:
        return any(
            op.turn_id == self.state.active_turn_id
            and (
                a.state in {"planned", "started", "parked"}
                or (a.state == "unknown" and a.unknown is not None and a.unknown._payload["retry"] == "retry")
            )
            for op in self.state.operations.values()
            for a in [op.attempts[max(op.attempts)]]
        )
