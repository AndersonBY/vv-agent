"""Shared driver state, record production and lease-fenced transaction algorithm."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import replace
from typing import Any, Protocol

from vv_agent.kernel._boundary_hydration import validate_logical_preflight
from vv_agent.kernel._content import INPUT_CONTENT, SESSION_ATTRIBUTES, ContentRef, ContentSessionWriter, SessionHead
from vv_agent.kernel._content_publication import (
    has_record_content,
    has_record_content_payload,
    link_declared_record_content,
    make_retained_record_before_publication,
    publish_record_content,
    require_record_link_writer,
    resolve_record_content_links,
)
from vv_agent.kernel._execution_values import AgentTask, SubTaskOutcome
from vv_agent.kernel._hydration import (
    ContentReader,
    DefinitionContent,
    MissingContentLink,
    SessionHydrator,
)
from vv_agent.kernel._lease_scope import Clock
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import InboxItem, Record, copy_json, digest, make_record, record_identity
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._store import Conflict, ContentSessionTx, Lease, LeasePoll, SessionStore, SessionTx, StoredRecord
from vv_agent.kernel._tool_values import ToolExecutionResult

ConfiguredChildResultProjector = Callable[
    [Record, Sequence[SubTaskOutcome], Mapping[str, Any]],
    ToolExecutionResult,
]

_TURN_ENVIRONMENT_CONTENT_SLOTS = frozenset({"environment_descriptor", "environment_operational_policy"})

_RECORD_CONTENT_PATHS: dict[tuple[str, str], tuple[str, ...]] = {
    ("session_created", SESSION_ATTRIBUTES): ("attributes",),
    ("turn_started", "definition"): ("definition",),
    ("turn_started", "budget"): ("budget",),
    ("turn_started", "environment_descriptor"): ("environment", "descriptor"),
    ("turn_started", "environment_operational_policy"): ("environment", "operational_policy"),
    ("input_applied", "input_reason"): ("reason",),
    ("op_planned", "request"): ("request",),
    ("op_planned", "tool"): ("tool",),
    ("op_planned", "budget_admission"): ("budget_admission",),
    ("op_planned", "hook_result"): ("budget_admission", "hook_result"),
    ("op_planned", "shared_state"): ("budget_admission", "shared_state"),
    ("op_prepared", "request"): ("request",),
    ("op_prepared", "tool"): ("tool",),
    ("op_prepared", "hook_result"): ("hook_result",),
    ("op_prepared", "shared_state"): ("shared_state",),
    ("op_parked", "provider_handle_job_id"): ("handle", "job_id"),
    ("op_parked", "provider_handle_evidence"): ("handle", "evidence"),
    ("op_parked", "provider_handle_query_ref"): ("handle", "query_ref"),
    ("op_parked", "provider_handle_cancel_ref"): ("handle", "cancel_ref"),
    ("op_parked", "approval_scope"): ("handle", "scope"),
    ("op_parked", "user_question"): ("handle", "question"),
    ("op_parked", "interaction_result"): ("interaction_result",),
    ("op_parked", "delegation_agent_name"): ("delegation", "agent_name"),
    ("op_parked", "delegation_metadata"): ("delegation", "metadata"),
    ("op_completed", "result"): ("result",),
    ("op_completed", "usage"): ("usage",),
    ("op_completed", "shared_state"): ("shared_state",),
    ("op_completed", "evidence"): ("evidence",),
    ("op_unknown", "op_unknown_reason"): ("reason",),
    ("op_unknown", "op_unknown_dispatch_evidence"): ("dispatch_evidence",),
    ("op_unknown", "op_unknown_observation"): ("observation",),
    ("turn_ended", "turn_ended_reason"): ("reason",),
    ("turn_ended", "turn_ended_result"): ("result",),
    ("turn_ended", "turn_ended_budget"): ("budget",),
    ("usage_observed", "usage"): ("usage",),
    ("usage_observed", "usage_source"): ("source",),
    ("boundary_recorded", "budget_usage"): ("data", "usage"),
    ("boundary_recorded", "shared_state"): ("data", "shared_state"),
}
_PREPARED_PLAN_CONTENT_SLOTS = frozenset({"request", "tool", "hook_result", "shared_state"})


class DriverLeaseScope(Protocol):
    lease: Lease

    def confirmed_poll(self) -> LeasePoll: ...

    def bind_turn(self, turn_id: str | None, generation: int | None, applied_inputs: Mapping[str, object]) -> None: ...

    def reset_token(self) -> None: ...

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
        content_reader: ContentReader | None = None,
        expected_head: SessionHead | tuple[int, str] | None = None,
        project_configured_child_results: ConfiguredChildResultProjector | None = None,
    ) -> None:
        self.store, self.sid, self.scope = store, sid, scope
        self._fold_factory, self._clock = fold_factory, clock
        self._observe_cost, self._hook, self._emit_phase = observe_cost, hook, emit_phase
        self.content_reader = content_reader
        self._expected_head = expected_head
        self._project_configured_child_results = project_configured_child_results
        self._hydrators: dict[int, SessionHydrator] = {}
        self.load_state()
        self.polled: set[tuple[str, int]] = set()
        self.active_since = self.clock.monotonic_ns()
        self.active_tid: str | None = None

    @property
    def clock(self) -> Clock:
        return self._clock

    def load_state(self) -> None:
        self.state, self.records, self.watermark = self.store.read_state(self.sid, expected_head=self._expected_head)
        self._stored_by_id = {stored.record.record_id: stored for stored in self.records}
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
        self._stored_by_id.update((stored.record.record_id, stored) for stored in records)

    def bind_turn(self) -> None:
        tid = self.state.active_turn_id
        if tid != self.active_tid:
            self.active_since = self.clock.monotonic_ns()
            self.active_tid = tid
        generation = self.state.turns[tid].start._payload["generation"] if tid else None
        self.scope.bind_turn(tid, generation, self.state.applied_inputs)

    def record(
        self,
        kind: str,
        payload: dict[str, Any],
        plan: Record | None = None,
        *,
        tid: str | None = None,
        feature_level: int = 1,
    ) -> Record:
        payload = copy_json(payload)
        if kind == "turn_started":
            payload.setdefault("policy_digest", None)
            payload.setdefault(
                "requirements",
                {"level": 1, "handler_version": payload.get("handler_version"), "capabilities": []},
            )
            payload.setdefault("tokenizer_binding", None)
            payload.setdefault("environment", None)
        elif kind in {"op_planned", "op_prepared", "op_completed"}:
            payload.setdefault("policy_decision", None)
        turn_id = plan.turn_id if plan else tid or self.state.active_turn_id
        operation_id = plan.operation_id if plan else None
        attempt = plan.attempt if plan else None
        if not has_record_content_payload(kind, payload):
            return make_record(
                kind,
                session_id=self.sid,
                turn_id=turn_id,
                operation_id=operation_id,
                attempt=attempt,
                payload=payload,
                feature_level=feature_level,
            )
        return Record(
            record_identity(kind, self.sid, turn_id, operation_id, attempt, payload),
            kind,
            self.sid,
            turn_id,
            operation_id,
            attempt,
            payload,
            feature_level=feature_level,
        )

    def hydrator_for(self, record_owner: int | str) -> SessionHydrator:
        """Create the one owner-scoped hydrator over this driver's session reader."""

        if type(record_owner) is int:
            if record_owner < 1:
                raise ValueError("record_owner must be a positive record sequence or input id")
            owner = f"record:{record_owner}"
            cached = self._hydrators.get(record_owner)
            if cached is not None:
                return cached
        elif isinstance(record_owner, str) and record_owner:
            if record_owner == self.sid:
                raise ValueError("session_id cannot be used as a content owner")
            owner = record_owner if record_owner.startswith(("record:", "input:")) else f"input:{record_owner}"
        else:
            raise ValueError("record_owner must be a positive record sequence or input id")
        reader, tenant_id = self._content_reader_binding()
        hydrator = SessionHydrator(reader, tenant_id=tenant_id, session_id=self.sid, owner=owner)
        if type(record_owner) is int:
            self._hydrators[record_owner] = hydrator
        return hydrator

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

        if owner.startswith("record:") and owner.removeprefix("record:").isdecimal():
            record_owner: int | str = int(owner.removeprefix("record:"))
        elif owner.startswith("input:") and owner.removeprefix("input:"):
            record_owner = owner
        else:
            raise RecordError("content owner must use a record sequence or input id")
        self._content_reader_binding()
        hydrator = self.hydrator_for(record_owner)
        try:
            parsed = ref if isinstance(ref, ContentRef) else ContentRef.from_dict(ref)
        except (TypeError, ValueError) as exc:
            raise RecordError(f"S1 content slot {slot} requires a valid ContentRef") from exc
        return hydrator._read_link(slot, parsed)

    def stored_record(self, record: Record) -> StoredRecord:
        """Return the retained ledger row for one record identity."""

        if not isinstance(record, Record):
            raise TypeError("record must be a Record")
        if record.session_id != self.sid:
            raise RecordError("record belongs to another session")
        stored = self._stored_by_id.get(record.record_id)
        if stored is None:
            raise RecordError(f"record is outside the retained ledger: {record.record_id}")
        return stored

    @staticmethod
    def _content_field(record: Record, slot: str) -> Any:
        try:
            path = _RECORD_CONTENT_PATHS[(record.kind, slot)]
        except KeyError:
            raise RecordError(f"unknown content slot for {record.kind}: {slot}") from None
        if (
            record.kind == "turn_started"
            and slot in _TURN_ENVIRONMENT_CONTENT_SLOTS
            and record._payload.get("environment") is None
        ):
            return None
        value: Any = record._payload
        for part in path:
            if type(value) is not dict or part not in value:
                raise RecordError(f"missing content field for {record.kind}/{slot}: {'/'.join(path)}")
            value = value[part]
        return value

    def _prepared_source_record(self, record: Record, slot: str) -> StoredRecord | None:
        if record.kind != "op_planned" or slot not in _PREPARED_PLAN_CONTENT_SLOTS:
            return None
        if record.operation_id is None or record.attempt is None:
            return None
        operation = self.state.operations.get(record.operation_id)
        attempt = operation.attempts.get(record.attempt) if operation is not None else None
        if attempt is None or attempt.prepared is None:
            return None
        prepared = self.stored_record(attempt.prepared)
        try:
            value = self._content_field(attempt.prepared, slot)
        except RecordError:
            return None
        return prepared if value == self._content_field(record, slot) else None

    def record_content_ref(self, record: Record, slot: str) -> ContentRef | None:
        """Return the declared reference for one record slot, preserving nulls."""

        value = self._content_field(record, slot)
        if value is None:
            return None
        try:
            return ContentRef.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise RecordError(f"content field requires a valid ContentRef: {record.kind}/{slot}") from exc

    def hydrate_record_field(self, record: Record, slot: str, ref: Any) -> Any:
        """Hydrate one declared C field from the record that actually owns it."""

        value = self._content_field(record, slot)
        if value is None:
            if ref is not None:
                raise RecordError(f"content field disagrees with record payload: {record.kind}/{slot}")
            return None
        parsed = self.record_content_ref(record, slot)
        assert parsed is not None
        if isinstance(ref, ContentRef):
            supplied = ref
        else:
            try:
                supplied = ContentRef.from_dict(ref)
            except (TypeError, ValueError) as exc:
                raise RecordError(f"content field requires a valid ContentRef: {record.kind}/{slot}") from exc
        if supplied != parsed or value != parsed.to_dict():
            raise RecordError(f"content field disagrees with record payload: {record.kind}/{slot}")
        owner = self._prepared_source_record(record, slot) or self.stored_record(record)
        return self.read_content_json(f"record:{owner.seq}", slot, parsed)

    def _hydrate_record_nested_field(self, record: Record, slot: str, value: Any) -> Any:
        """Hydrate nested refs declared by the closed inventory for one record slot."""

        owner = self._prepared_source_record(record, slot) or self.stored_record(record)
        return self.hydrator_for(owner.seq).nested(owner.record.kind, owner.record._payload, slot, value)

    def hydrate_plan(self, record: Record) -> Record:
        """Hydrate one operation plan for dispatch without replacing retained state."""

        operation = self.state.operations.get(record.operation_id or "")
        attempt = operation.attempts.get(record.attempt) if operation is not None and record.attempt is not None else None
        if attempt is not None and attempt.prepared is not None:
            prepared = self.stored_record(attempt.prepared).record._payload
            record = replace(
                record,
                payload=record._payload
                | {
                    key: prepared[key]
                    for key in ("request", "request_digest", "tool", "op_kind", "provider_binding", "idempotency_key")
                }
                | {"budget_admission": {"hook_result": prepared["hook_result"], "shared_state": prepared["shared_state"]}},
            )
        value = record._payload
        payload = value | {
            "request": self.hydrate_record_field(record, "request", value["request"]),
            "tool": self.hydrate_record_field(record, "tool", value["tool"]),
        }
        payload["request"] = self._hydrate_record_nested_field(record, "request", payload["request"])
        if type(payload["request"]) is not dict or (payload["tool"] is not None and type(payload["tool"]) is not dict):
            raise RecordError("operation plan request/tool hydration has the wrong JSON shape")
        if attempt is not None and attempt.prepared is not None:
            admission = value["budget_admission"]
            if type(admission) is not dict:
                raise RecordError("prepared operation budget admission must be an object")
            admission = admission | {
                "hook_result": self.hydrate_record_field(record, "hook_result", admission["hook_result"]),
                "shared_state": self.hydrate_record_field(record, "shared_state", admission["shared_state"]),
            }
            admission["hook_result"] = self._hydrate_record_nested_field(record, "hook_result", admission["hook_result"])
            if admission["hook_result"] is not None and type(admission["hook_result"]) is not dict:
                raise RecordError("prepared hook result must be an object or null")
            if type(admission["shared_state"]) is not dict:
                raise RecordError("prepared shared state must be an object")
            payload["budget_admission"] = admission
        else:
            payload["budget_admission"] = self.hydrate_record_field(record, "budget_admission", value["budget_admission"])
            if type(payload["budget_admission"]) is not dict:
                raise RecordError("operation budget admission must be an object")
        return replace(record, payload=payload)

    def hydrate_parked_wait(self, operation_id: str, attempt: int, wait: dict[str, Any]) -> dict[str, Any]:
        """Hydrate the stored observation selected by the reducer for one attempt."""

        stored = next(
            (
                row
                for row in reversed(self.records)
                if row.record.kind == "op_parked"
                and row.record.operation_id == operation_id
                and row.record.attempt == attempt
                and row.record._payload == wait
            ),
            None,
        )
        if stored is None:
            raise RecordError("parked wait is outside the retained ledger")
        record = stored.record
        payload = wait | {}
        handle = payload.get("handle")
        if type(handle) is not dict:
            raise RecordError("parked wait handle must be an object")
        handle = handle | {}
        for key, slot in (
            ("job_id", "provider_handle_job_id"),
            ("evidence", "provider_handle_evidence"),
            ("query_ref", "provider_handle_query_ref"),
            ("cancel_ref", "provider_handle_cancel_ref"),
            ("scope", "approval_scope"),
            ("question", "user_question"),
        ):
            if key in handle:
                handle[key] = self.hydrate_record_field(record, slot, handle[key])
        payload["handle"] = handle
        if "interaction_result" in payload:
            payload["interaction_result"] = self.hydrate_record_field(record, "interaction_result", payload["interaction_result"])
        delegation = payload.get("delegation")
        if delegation is not None:
            if type(delegation) is not dict:
                raise RecordError("parked delegation must be an object")
            delegation = delegation | {}
            if "agent_name" in delegation:
                delegation["agent_name"] = self.hydrate_record_field(record, "delegation_agent_name", delegation["agent_name"])
            if "metadata" in delegation:
                delegation["metadata"] = self.hydrate_record_field(record, "delegation_metadata", delegation["metadata"])
            payload["delegation"] = delegation
        return copy_json(payload)

    def hydrated_definition(self, record: Record) -> DefinitionContent:
        """Hydrate a frozen turn definition through the typed definition parser."""

        stored = self.stored_record(record)
        ref = self.record_content_ref(record, "definition")
        if ref is None:
            raise RecordError("turn definition cannot be null")
        return self.hydrator_for(stored.seq).definition(ref, expected_reference_digest=record._payload["definition_digest"])

    def session_attributes(self) -> dict[str, Any]:
        """Hydrate the creation record's attributes."""

        created = self.records[0]
        value = self.hydrate_record_field(created.record, SESSION_ATTRIBUTES, created.record._payload["attributes"])
        if type(value) is not dict:
            raise RecordError("session attributes must be a JSON object")
        return self._hydrate_record_nested_field(created.record, SESSION_ATTRIBUTES, value)

    def input_content(self, input_id: str, value: Any) -> Any:
        """Hydrate one explicitly owned session input."""

        try:
            ref = ContentRef.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise RecordError("session input requires a ContentRef") from exc
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
        return self.hydrated_definition(self.state.turns[self.state.active_turn_id].start).task

    def _logical_preflight(
        self,
        records: list[Record],
        encoded_records: list[Record],
        *,
        base_fold: Fold | None = None,
    ) -> None:
        """Validate logical semantics inside the caller-owned transaction."""
        validate_logical_preflight(self, records, encoded_records, base_fold=base_fold)

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[SessionTx], list[Record]] | None = None,
        _logical_base: Fold | None = None,
    ) -> None:
        tid = self.state.active_turn_id
        prepared_records = list(records)
        commit_records = prepared_records
        content_writer = None
        declarations = ()
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
                        prepared_records.extend(prepare(tx))
                from ._driver_planning import resolve_command_records

                with phase("command_binding"):
                    prepared_records = resolve_command_records(self, tx, prepared_records)
                committed = {stored.record.record_id: stored.record for stored in self.records}
                record_ids = [record.record_id for record in prepared_records]
                if any(record_id in committed for record_id in record_ids):
                    if not all(record_id in committed for record_id in record_ids):
                        raise Conflict("commit mixes new records with already committed record ids")
                    prepared_records = [committed[record_id] for record_id in record_ids]
                    new_records = []
                else:
                    new_records = list(prepared_records)
                if new_records and tid is not None:
                    meter_id = f"active/{tid}/{self.scope.lease.epoch}"
                    usage = self.record(
                        "usage_observed",
                        {
                            "meter_id": meter_id,
                            "observation": self.state.usage_observations.get(meter_id, 0) + 1,
                            "mode": "cumulative",
                            "source": "kernel-active-scope",
                            "usage": {
                                "elapsed_ms": (self.clock.monotonic_ns() - self.active_since) // 1_000_000,
                                "host_cost": self._observe_cost(),
                            },
                        },
                    )
                    new_records.insert(0, usage)
                    prepared_records = new_records
                    commit_records = prepared_records
                if has_record_content(prepared_records):
                    with phase("publish"):
                        content_writer = require_record_link_writer(self.content_writer(tx))
                        publication = publish_record_content(
                            content_writer,
                            (
                                make_retained_record_before_publication(
                                    record.kind,
                                    session_id=record.session_id,
                                    turn_id=record.turn_id,
                                    operation_id=record.operation_id,
                                    attempt=record.attempt,
                                    payload=record._payload,
                                    feature_level=record.feature_level,
                                )
                                for record in prepared_records
                            ),
                        )
                        commit_records = list(publication.records)
                        declarations = publication.links
                with phase("before_commit"):
                    self._hook("before_commit", commit_records[-1] if commit_records else None)
                # SQL append evaluates the guard only after encoding and immutable
                # history preparation. Keep the late precheck through atomic exit.
                tx.bind_lease_guard(lambda: latch.enter_context(self.scope.committing()))
                with phase("append"):
                    receipt = tx.append(
                        self.sid,
                        lease=self.scope.lease,
                        expected_seq=self.records[-1].seq,
                        commit_id=digest({"record_ids": [r.record_id for r in commit_records], "consumed": list(consumed)}),
                        records=tuple(commit_records),
                        consume_input_ids=consumed,
                        expected_inbox_seq=self.watermark if guarded else None,
                        expected_head=self._expected_head,
                    )
                if new_records:
                    with phase("logical_preflight"):
                        self._logical_preflight(new_records, commit_records, base_fold=_logical_base)
                links = resolve_record_content_links(declarations, receipt.record_sequences)
                if links:
                    with phase("finalize"):
                        assert content_writer is not None
                        link_declared_record_content(content_writer, links)
        except BaseException as exc:
            atomic_error = type(exc).__name__
            raise
        finally:
            atomic_ms = (self.clock.monotonic_ns() - atomic_start) / 1_000_000
            source = next((r for r in reversed(commit_records) if r.operation_id), None)
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
        if self._expected_head is not None:
            self._expected_head = self.store.verify_ledger(self.sid)
        self._hook("after_commit", commit_records[-1] if commit_records else None)
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
                shared = self.hydrate_record_field(r, "shared_state", r._payload["data"]["shared_state"])
                if type(shared) is not dict:
                    raise RecordError("durable boundary shared state must be an object")
                return copy_json(shared)
            if r.kind == "op_prepared":
                shared = self.hydrate_record_field(r, "shared_state", r._payload["shared_state"])
                if type(shared) is not dict:
                    raise RecordError("prepared shared state must be an object")
                return copy_json(shared)
            if r.kind == "op_completed" and r._payload["context"] == "normal":
                snapshot = self.hydrate_record_field(r, "shared_state", r._payload["shared_state"])
                if snapshot is not None:
                    if type(snapshot) is not dict:
                        raise RecordError("completed shared state must be an object or null")
                    return copy_json(snapshot)
        assert self.state.active_turn_id is not None
        return copy_json(self._task().initial_shared_state)

    def pending_batch(self) -> bool:
        from ._driver_inputs import _pending_batch_in_state

        return _pending_batch_in_state(self.state)
