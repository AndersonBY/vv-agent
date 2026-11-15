"""Shared driver state, record production and lease-fenced transaction algorithm."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Protocol

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import INPUT_CONTENT, SESSION_ATTRIBUTES, ContentRef, ContentSessionWriter
from vv_agent.kernel._content_publication import declared_record_content, link_declared_record_content
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import ContentReader, MissingContentLink, SessionHydrator
from vv_agent.kernel._lease_scope import Clock
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import InboxItem, Record, copy_json, digest, make_record
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import Conflict, ContentSessionTx, Lease, SessionStore, SessionTx, StoredRecord


class DriverLeaseScope(Protocol):
    lease: Lease
    turn_id: str | None
    generation: int | None

    def committing(self) -> AbstractContextManager[Lease]: ...


class ActiveUsageContentUnavailable(RuntimeError):
    """The active usage producer requires the session content store."""

    code = "active_usage_content_unavailable"

    def __init__(self, meter_id: str) -> None:
        self.meter_id = meter_id
        super().__init__(
            f"S1 active usage meter {meter_id!r} requires the session content store; "
            "the commit was blocked before writing plaintext"
        )


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
        content_reader: ContentReader | None = None,
    ) -> None:
        self.store, self.sid, self.scope = store, sid, scope
        self._fold_factory, self._clock = fold_factory, clock
        self._observe_cost, self._hook, self._emit_phase = observe_cost, hook, emit_phase
        self.content_reader = content_reader
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

    def hydrator_for(self, record_owner: int | str) -> SessionHydrator:
        """Create the one owner-scoped hydrator over this driver's session reader."""

        reader, tenant_id = self._content_reader_binding()

        if type(record_owner) is int:
            if record_owner < 1:
                raise ValueError("record_owner must be a positive record sequence or input id")
            owner = f"record:{record_owner}"
        elif isinstance(record_owner, str) and record_owner:
            if record_owner == self.sid:
                raise ValueError("session_id cannot be used as a content owner")
            owner = record_owner if record_owner.startswith(("record:", "input:")) else f"input:{record_owner}"
        else:
            raise ValueError("record_owner must be a positive record sequence or input id")
        return SessionHydrator(reader, tenant_id=tenant_id, session_id=self.sid, owner=owner)

    def _content_reader_binding(self) -> tuple[ContentReader, str]:
        reader = self.content_reader
        if reader is None:
            raise RecordError("S1 content hydration is unavailable in the producer/codec phase")
        if self.store is None:
            raise RecordError("S1 content hydration requires a session store")

        binding = getattr(reader, "binding", None)
        store_tenant = getattr(self.store, "tenant_id", None)
        if binding is None:
            if not isinstance(store_tenant, str) or not store_tenant:
                raise RecordError("S1 content reader must have an explicit tenant and session binding")
            return reader, store_tenant
        if not isinstance(binding, tuple) or len(binding) != 2 or not all(isinstance(item, str) and item for item in binding):
            raise MissingContentLink("content reader binding is invalid")
        tenant_id, bound_session_id = binding
        if bound_session_id != self.sid or (store_tenant is not None and store_tenant != tenant_id):
            raise MissingContentLink("content reader does not match the driver store and session")
        return reader, tenant_id

    def read_content_json(self, owner: str, slot: str, ref: ContentRef | dict[str, Any]) -> Any:
        """Read one already-linked JSON object through the bound content reader."""

        parsed = ref if isinstance(ref, ContentRef) else ContentRef.from_dict(ref)
        reader, tenant_id = self._content_reader_binding()
        body = reader.read(tenant_id, self.sid, owner, slot, parsed)
        if type(body) is not bytes or len(body) != parsed.size:
            raise RecordError("S1 content reader returned bytes that do not match the reference")
        try:
            value = json.loads(body)
            if canonical_json_bytes(value) != body:
                raise ValueError("content is not canonical JSON")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecordError("S1 content must be canonical JSON") from exc
        return value

    def session_attributes(self) -> dict[str, Any]:
        """Hydrate the creation record's attributes or fail closed on a ref."""

        created = self.records[0]
        attributes = created.record._payload["attributes"]
        try:
            ref = ContentRef.from_dict(attributes)
        except (TypeError, ValueError):
            if type(attributes) is not dict:
                raise RecordError("session attributes require an object or ContentRef") from None
            return copy_json(attributes)
        value = self.read_content_json(f"record:{created.seq}", SESSION_ATTRIBUTES, ref)
        if type(value) is not dict:
            raise RecordError("session attributes must be a JSON object")
        return value

    def input_content(self, input_id: str, value: Any) -> Any:
        """Hydrate a session input only when its payload carries a ContentRef."""

        try:
            ref = ContentRef.from_dict(value)
        except (TypeError, ValueError):
            return copy_json(value)
        return self.read_content_json(f"input:{input_id}", INPUT_CONTENT, ref)

    def content_writer(self, tx: SessionTx | None = None) -> ContentSessionWriter:
        """Return a writer bound to the caller-owned transaction, when it carries content capability."""

        if not isinstance(tx, ContentSessionTx):
            raise ValueError("session content writer requires an outer transaction with ContentSessionTx capability")
        tenant_id = getattr(self.store, "tenant_id", None)
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("session content writer requires a store with a non-empty tenant binding")
        return tx.content_writer(tenant_id=tenant_id, session_id=self.sid)

    def _task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start._task()

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
            self._observe_cost()
            raise ActiveUsageContentUnavailable(meter)
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
                links = declared_record_content(records, receipt.record_sequences)
                if links:
                    with phase("finalize"):
                        writer = self.content_writer(tx)
                        link_declared_record_content(writer, links)
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
        return copy_json(self._task().initial_shared_state)

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
