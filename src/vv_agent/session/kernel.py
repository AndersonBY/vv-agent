"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

from vv_agent.budget import BudgetEvaluator, BudgetExhaustion
from vv_agent.constants import SUB_TASK_STATUS_TOOL_NAME
from vv_agent.kernel import _driver_outcomes
from vv_agent.kernel._boundary_hydration import (
    HydratedFoldResult,
    SessionHydration,
    _lookup_input_owner,
    _session_content_frame,
    hydrate_and_fold_records,
)
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation, _BudgetStopIteration
from vv_agent.kernel._budget_values import BudgetUnavailableReason
from vv_agent.kernel._cancellation import Cancellation
from vv_agent.kernel._content import ContentRef, KeyUnavailable, SessionHead
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_planning import ProviderAvailable, _PlanningStopIteration
from vv_agent.kernel._execution_values import SubTaskOutcome
from vv_agent.kernel._hydration import DefinitionContent, HydrationError
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.kernel._reducer import logical_operation_payloads
from vv_agent.kernel._tool_values import ToolExecutionResult
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.types import AgentTask, Message

from . import _drive_phase, _driver_dispatch, _driver_inputs, _driver_planning, _driver_recovery
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _LeaseHeartbeat as _LeaseHeartbeat
from ._driver_scope import _Scope
from .bindings import MissingHostBinding
from .context import project_context
from .projection import (
    fold_hydrated_records,
    hydrate_input,
    hydrate_records,
    hydrate_stored_record,
    hydration_for_session_store,
)
from .providers import (
    Accepted,
    Definitive,
)
from .records import InboxItem, Record, copy_json
from .reducer import Attempt, ExecutionState, _fold_factory_for
from .runtime import Runtime
from .runtime import RuntimeNotReady as RuntimeNotReady
from .store import Lease, SessionStore, StoredRecord
from .timing import Clock, LeaseKeeper


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


logger = logging.getLogger(__name__)
_NOT_VERIFIED = object()


def _is_content_ref(value: Any) -> bool:
    if isinstance(value, ContentRef):
        return True
    if type(value) is not dict:
        return False
    try:
        ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return False
    return True


def _project_configured_child_results(
    plan: Record,
    outcomes: Sequence[SubTaskOutcome],
    delegation: Mapping[str, Any],
) -> ToolExecutionResult:
    del delegation
    values = iter(outcomes)
    context = SimpleNamespace(
        sub_task_runner=lambda _request: next(values),
        sub_task_manager=None,
        run_context=None,
        ctx=None,
        tool_call_id="",
        workspace_backend=None,
    )
    result = create_sub_task(
        cast(ToolContext, context),
        copy_json(plan._payload["request"]["arguments"]) | {"wait_for_completion": True},
    )
    result.tool_call_id = plan._payload["request"]["id"]
    return result


class _Driver(CoreDriver):
    scope: _Scope

    def __init__(
        self,
        store: SessionStore,
        sid: str,
        runtime: Runtime,
        scope: _Scope,
        *,
        one_turn: bool = False,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ):
        self.runtime, self.one_turn = runtime, one_turn
        self._load_state_version = 0
        self._logical_dirty = False
        self._logical_scope_depth = 0
        self._active_logical_fold: CoreFold | None = None
        self._logical_hydrated_records: tuple[StoredRecord, ...] = ()
        self._logical_projected_state: ExecutionState | None = None
        self._logical_retry_record_ids: set[str] = set()

        def observe_cost() -> dict[str, Any] | None:
            cost = None
            if runtime.config.host_cost_meter is not None:
                with suppress(Exception):
                    observed = runtime.config.host_cost_meter.read()
                    cost = observed.to_dict() if observed else None
            return cost

        def emit_phase(duration: float, fields: dict[str, Any]) -> None:
            logger.log(
                logging.WARNING if duration >= runtime.ttl_ms / 2 else logging.DEBUG,
                "session transaction phase",
                extra=fields,
            )

        super().__init__(
            store,
            sid,
            scope,
            fold_factory=lambda: _fold_factory_for(store)(),
            clock=runtime.clock,
            observe_cost=observe_cost,
            hook=lambda point, record: runtime.hook(point, record),
            emit_phase=emit_phase,
            content_reader=runtime.content_reader,
            expected_head=expected_head,
            project_configured_child_results=_project_configured_child_results,
        )

    def load_state(self) -> None:
        super().load_state()
        self._load_state_version += 1
        self._invalidate_logical_view()
        self._raw_prefix_changed()

    def extend(self, records: tuple[StoredRecord, ...]) -> None:
        super().extend(records)
        if records:
            if self._active_logical_fold is not None:
                try:
                    self._active_logical_fold = self._extend_logical_fold(records)
                except (HydrationError, KeyUnavailable):
                    self._invalidate_logical_view()
                    self._raw_prefix_changed()
                    return
            self._raw_prefix_changed()

    def _raw_prefix_changed(self) -> None:
        if self._logical_scope_depth:
            self._logical_dirty = True

    def bind_turn(self) -> None:
        state = self.fold.state if self._logical_dirty else self.state
        tid = state.active_turn_id
        if tid != self.active_tid:
            self.active_since = self.clock.monotonic_ns()
            self.active_tid = tid
        generation = state.turns[tid].start._payload["generation"] if tid else None
        self.scope.bind_turn(tid, generation, state.applied_inputs)

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[Any], list[Record]] | None = None,
        _logical_base: CoreFold | None = None,
    ) -> None:
        # A surrounding execution scope may hold logical state. Commits always advance the raw fold.
        logical_state = self.state if self.content_reader is not None and self.state is not self.fold.state else None
        load_state_version = self._load_state_version
        record_count = len(self.records)
        if logical_state is not None:
            self.state = self.fold.state
        logical_base = _logical_base
        if logical_base is None and logical_state is not None and self._active_logical_fold is not None:
            logical_base = self._active_logical_fold.fork()
        try:
            return super().commit(
                records,
                consumed,
                guarded=guarded,
                prepare=prepare,
                _logical_base=logical_base,
            )
        finally:
            if logical_state is not None and self._load_state_version == load_state_version:
                advanced = len(self.records) != record_count
                if advanced and self._active_logical_fold is not None:
                    assert self._logical_projected_state is not None
                    self._state = self._logical_projected_state
                    self._logical_dirty = False
                else:
                    self._state = logical_state
                    self._logical_dirty = advanced

    @property
    def state(self) -> ExecutionState:
        """Restore a committed logical view lazily when its execution scope still needs it."""

        if self._logical_dirty:
            result = self._logical_view()
            self._state = result.state
            self._logical_dirty = False
        return self._state

    @state.setter
    def state(self, value: ExecutionState) -> None:
        self._state = value
        self._logical_dirty = False

    def _logical_state(self) -> ExecutionState:
        if self.content_reader is None:
            return self._state
        if self._logical_scope_depth == 0:
            self._invalidate_logical_view()
            return self._logical_view().state
        if self._state is not self.fold.state and not self._logical_dirty:
            return self._state
        return self._logical_view().state

    def _invalidate_logical_view(self) -> None:
        self._active_logical_fold = None
        self._logical_hydrated_records = ()
        self._logical_projected_state = None
        self._logical_retry_record_ids.clear()

    def _logical_view(self) -> HydratedFoldResult:
        if self._active_logical_fold is not None and len(self._active_logical_fold.history) == len(self.records):
            assert self._logical_projected_state is not None
            result = HydratedFoldResult(self._active_logical_fold, self._logical_projected_state, ())
        else:
            input_ids = (input_id for stored in self.records if (input_id := _lookup_input_owner(stored.record)) is not None)
            with _session_content_frame(
                self,
                record_seqs=(stored.seq for stored in self.records),
                input_ids=input_ids,
            ) as hydration:
                hydrated_records = hydrate_records(self.records, hydration)
                retry_payloads = self._logical_retry_plan_payloads(hydration, hydrated_records=hydrated_records)
                self._logical_retry_record_ids = set(retry_payloads)
                with logical_operation_payloads(retry_payloads):
                    result = hydrate_and_fold_records(
                        self.records,
                        _fold_factory_for(self.store),
                        hydration,
                        hydrated_records=hydrated_records,
                    )
            self._active_logical_fold = result.fold
            self._logical_hydrated_records = result.hydrated_records
            self._logical_projected_state = self._logical_state_from_fold(result.fold)
            result = HydratedFoldResult(result.fold, self._logical_projected_state, result.hydrated_records)
        return result

    def _extend_logical_fold(self, records: tuple[StoredRecord, ...]) -> CoreFold:
        """Advance the retained logical prefix by only the newly committed records."""

        assert self._active_logical_fold is not None
        input_ids = (input_id for stored in records if (input_id := _lookup_input_owner(stored.record)) is not None)
        with _session_content_frame(
            self,
            record_seqs=(stored.seq for stored in records),
            input_ids=input_ids,
        ) as hydration:
            hydrated_records = hydrate_records(records, hydration)
            retry_payloads = self._logical_retry_plan_payloads_for(
                hydration,
                records,
                hydrated_records=hydrated_records,
                base_fold=self._active_logical_fold,
            )
            previous_state = self._active_logical_fold.state
            with logical_operation_payloads(retry_payloads):
                result = hydrate_and_fold_records(
                    records,
                    _fold_factory_for(self.store),
                    hydration,
                    hydrated_records=hydrated_records,
                    base_fold=self._active_logical_fold,
                    record_bodies=tuple(stored.record.encode() for stored in records),
                )
        self._logical_retry_record_ids.update(retry_payloads)
        self._logical_hydrated_records += result.hydrated_records
        if self._logical_projected_state is not None:
            self._logical_projected_state = self._advance_logical_state(previous_state, result.fold.state)
        return result.fold

    def _logical_state_from_fold(self, fold: CoreFold) -> ExecutionState:
        """Project the SDK-facing operation view from one captured logical fold."""

        state = fold.state
        previous_state, previous_dirty = self._state, self._logical_dirty
        self._state, self._logical_dirty = self.fold.state, False
        try:
            operations = {
                operation_id: self._logical_operation(operation) for operation_id, operation in state.operations.items()
            }
        finally:
            self._state, self._logical_dirty = previous_state, previous_dirty
        logical_state = copy(state)
        logical_state.operations = operations
        return logical_state

    def _advance_logical_state(self, previous: ExecutionState, current: ExecutionState) -> ExecutionState:
        """Replace only operations whose copy-on-write state changed in one appended batch."""

        assert self._logical_projected_state is not None
        operations = self._logical_projected_state.operations.copy()
        for operation_id, operation in current.operations.items():
            if previous.operations.get(operation_id) is not operation:
                operations[operation_id] = self._logical_operation(operation)
        for operation_id in previous.operations.keys() - current.operations.keys():
            operations.pop(operation_id, None)
        projected = copy(current)
        projected.operations = operations
        return projected

    def _logical_operation(self, operation) -> Any:
        # Interaction and retry plans remain logical for reply and retry identity;
        # other plans reopen through their raw owner records.
        return replace(
            operation,
            attempts={
                number: replace(
                    attempt,
                    plan=(
                        (
                            attempt.plan
                            if not _is_content_ref(attempt.plan._payload["request"])
                            else self.hydrate_plan(attempt.plan)
                        )
                        if operation.kind == "interaction" or attempt.plan.record_id in self._logical_retry_record_ids
                        else self.stored_record(attempt.plan).record
                    ),
                    prepared=(self.stored_record(attempt.prepared).record if attempt.prepared is not None else None),
                )
                for number, attempt in operation.attempts.items()
            },
        )

    def _logical_retry_plan_payloads_for(
        self,
        hydration: SessionHydration,
        records: tuple[StoredRecord, ...],
        *,
        hydrated_records: tuple[StoredRecord, ...] | None = None,
        base_fold: CoreFold | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve retry plans from either the retained prefix or the current batch."""

        seen_operations: set[str] = set()
        has_retry = False
        for stored in records:
            record = stored.record
            if record.kind != "op_planned" or record.operation_id is None:
                continue
            if (record.attempt or 0) > 1 and (base_fold is not None or record.operation_id in seen_operations):
                has_retry = True
                break
            seen_operations.add(record.operation_id)
        if not has_retry:
            return {}
        if hydrated_records is None:
            hydrated_records = hydrate_records(records, hydration)
        hydrated_by_identity = {(stored.record.record_id, stored.seq): stored for stored in hydrated_records}

        def hydrated(stored: StoredRecord) -> Record:
            found = hydrated_by_identity.get((stored.record.record_id, stored.seq))
            if found is not None:
                return found.record
            return hydrate_stored_record(stored, hydration, include_input=False).record

        def logical_previous(
            previous_stored: StoredRecord | None,
            previous_record: Record | None,
        ) -> Record | None:
            if previous_record is not None and not _is_content_ref(previous_record._payload["request"]):
                return previous_record
            if previous_stored is None and previous_record is not None:
                previous_stored = self._stored_by_id.get(previous_record.record_id)
            return hydrated(previous_stored) if previous_stored is not None else previous_record

        payloads: dict[str, dict[str, Any]] = {}
        previous_by_operation: dict[str, tuple[StoredRecord | None, Record | None]] = {}
        for stored in records:
            record = stored.record
            if record.kind != "op_planned" or record.operation_id is None:
                continue
            operation_id = record.operation_id
            attempt = record.attempt or 0
            previous = previous_by_operation.get(operation_id)
            if previous is None and attempt > 1 and base_fold is not None:
                operation = base_fold.state.operations.get(operation_id)
                previous_attempt = operation.attempts.get(attempt - 1) if operation is not None else None
                if previous_attempt is not None:
                    previous = (None, previous_attempt.plan)
            if previous is None:
                if attempt == 1:
                    previous_by_operation[operation_id] = (stored, None)
                continue

            current = hydrated(stored)
            if not _is_content_ref(record._payload["request"]):
                previous_by_operation[operation_id] = (stored, current)
                continue
            previous_stored, previous_record = previous
            previous_record = logical_previous(previous_stored, previous_record)
            if previous_record is None:
                previous_by_operation[operation_id] = (stored, current)
                continue
            payload = copy_json(current._payload)
            for slot in ("request", "tool", "budget_admission"):
                payload[slot] = copy_json(previous_record._payload[slot])
            payloads[record.record_id] = payload
            previous_by_operation[operation_id] = (stored, replace(current, payload=payload))
        return payloads

    def _logical_retry_plan_payloads(
        self,
        hydration: SessionHydration,
        *,
        hydrated_records: tuple[StoredRecord, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self._logical_retry_plan_payloads_for(
            hydration,
            self.records,
            hydrated_records=hydrated_records,
        )

    @contextmanager
    def _logical_state_scope(self) -> Iterator[ExecutionState]:
        """Expose a call-local logical view while the raw fold remains authoritative."""

        outer = self._logical_scope_depth == 0
        if self.content_reader is None:
            yield self.state
            return
        if outer:
            with _session_content_frame(
                self,
                record_seqs=(stored.seq for stored in self.records),
                input_ids=(
                    stored.record._payload["input"]["input_id"]
                    for stored in self.records
                    if stored.record.kind == "input_applied"
                ),
            ):
                result = self._logical_view()
                self.state = result.state
                self._active_logical_fold = result.fold
                self._logical_scope_depth += 1
                try:
                    yield self.state
                finally:
                    self._logical_scope_depth -= 1
                    self._invalidate_logical_view()
                    self.state = self.fold.state
            return
        self._logical_scope_depth += 1
        try:
            yield self.state
        finally:
            self._logical_scope_depth -= 1

    def _logical_input(self, item: InboxItem) -> InboxItem:
        if self.content_reader is None:
            return item
        return hydrate_input(item, hydration_for_session_store(self.store, self.sid))

    def live_budget(self) -> BudgetEvaluator | None:
        inputs = self._run_outcome_steps(_driver_outcomes.live_budget_inputs_steps(self))
        if inputs is None:
            return None
        limits, initial = inputs
        return BudgetEvaluator(
            limits,
            initial_usage=initial,
            host_cost_meter=self.runtime.config.host_cost_meter,
            clock_ns=self.clock.monotonic_ns,
        )

    def budget_record(
        self,
        identity: str,
        evaluator: BudgetEvaluator,
        exhaustion: BudgetExhaustion | None,
        names: list[str] | None = None,
        source: str | None = None,
    ) -> Record:
        return _driver_outcomes.budget_record(self, identity, evaluator, exhaustion, names, source)

    def task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        start = self.state.turns[self.state.active_turn_id].start
        if self.content_reader is None:
            return start.task()
        return deepcopy(self._hydrated_definition().task)

    def _task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        start = self.state.turns[self.state.active_turn_id].start
        if self.content_reader is None:
            return start._task()
        return self._hydrated_definition().task

    def _hydrated_definition(self) -> DefinitionContent:
        assert self.state.active_turn_id is not None
        start = self.state.turns[self.state.active_turn_id].start
        return self.hydrated_definition(start)

    def hydrated_definition(self, record: Record) -> DefinitionContent:
        return super().hydrated_definition(self.stored_record(record).record)

    def record_content_ref(self, record: Record, slot: str) -> ContentRef | None:
        stored = self._optional_stored_record(record)
        return super().record_content_ref(stored.record if stored is not None else record, slot)

    def _optional_stored_record(self, record: Record) -> StoredRecord | None:
        if not isinstance(record, Record):
            raise TypeError("record must be a Record")
        if record.session_id != self.sid:
            raise RecordError("record belongs to another session")
        return self._stored_by_id.get(record.record_id)

    def _verified_completed_result(self, record: Record, slot: str) -> Any:
        if (
            slot != "result"
            or record.kind != "op_completed"
            or self._logical_scope_depth == 0
            or self._logical_dirty
            or self._active_logical_fold is None
            or record.operation_id is None
            or record.attempt is None
        ):
            return _NOT_VERIFIED
        operation = self._active_logical_fold.state.operations.get(record.operation_id)
        attempt = operation.attempts.get(record.attempt) if operation is not None else None
        if attempt is None or attempt.result is not record:
            return _NOT_VERIFIED
        stored = self._stored_by_id.get(record.record_id)
        if stored is None or stored.record.kind != "op_completed" or record.encode() != stored.record.encode():
            return _NOT_VERIFIED
        return copy_json(record._payload["result"])

    def hydrate_record_field(self, record: Record, slot: str, ref: Any) -> Any:
        # A prepared overlay keeps the plan envelope, so resolve refs against the record that declared them.
        operation = self.state.operations.get(record.operation_id or "")
        attempt = operation.attempts.get(record.attempt) if operation is not None and record.attempt is not None else None
        if attempt is not None and attempt.prepared is not None:
            prepared = self.stored_record(attempt.prepared).record
            try:
                prepared_ref = self._content_field(prepared, slot)
            except RecordError:
                pass
            else:
                supplied_ref = ref.to_dict() if isinstance(ref, ContentRef) else ref
                if prepared_ref == supplied_ref:
                    return super().hydrate_record_field(prepared, slot, prepared_ref)
        verified = self._verified_completed_result(record, slot)
        if verified is not _NOT_VERIFIED:
            return verified
        stored = self._optional_stored_record(record)
        if stored is None:
            return super().hydrate_record_field(record, slot, ref)
        raw = stored.record
        return super().hydrate_record_field(raw, slot, self._content_field(raw, slot))

    def hydrate_plan(self, record: Record) -> Record:
        hydrated = super().hydrate_plan(self.stored_record(record).record)
        if not _is_content_ref(hydrated._payload["request"]):
            return hydrated
        previous = next(
            (
                stored.record
                for stored in reversed(self.records)
                if stored.record.kind == "op_planned"
                and stored.record.operation_id == hydrated.operation_id
                and (stored.record.attempt or 0) < (hydrated.attempt or 0)
            ),
            None,
        )
        if previous is None:
            return hydrated
        previous_plan = super().hydrate_plan(previous)
        if not _is_content_ref(previous_plan._payload["request"]):
            payload = copy_json(hydrated._payload)
            for slot in ("request", "tool", "budget_admission"):
                payload[slot] = copy_json(previous_plan._payload[slot])
            return replace(hydrated, payload=payload)
        return hydrated

    def hydrate_parked_wait(self, operation_id: str, attempt: int, wait: dict[str, Any]) -> dict[str, Any]:
        if self.state is self.fold.state:
            return super().hydrate_parked_wait(operation_id, attempt, wait)
        stored = next(
            (
                row.record
                for row in reversed(self.records)
                if row.record.kind == "op_parked" and row.record.operation_id == operation_id and row.record.attempt == attempt
            ),
            None,
        )
        return super().hydrate_parked_wait(operation_id, attempt, stored._payload if stored is not None else wait)

    def effective_model(self) -> dict[str, Any]:
        return _driver_planning.effective_model(self)

    def plan(
        self,
        oid: str,
        request: dict[str, Any],
        kind: str,
        *,
        number: int = 1,
        dependencies: list[str] | None = None,
        capability: dict[str, Any] | None = None,
        purpose: str = "primary",
    ) -> Record:
        return _driver_planning.plan(
            self, oid, request, kind, number=number, dependencies=dependencies, capability=capability, purpose=purpose
        )

    def _run_outcome_steps[T](
        self,
        steps: Generator[_driver_outcomes.OutcomeRequest, Any, T],
        *,
        prepare_resources: Callable[[Record, dict[str, Any], Any], dict[str, Any]] | None = None,
    ) -> T:
        response = None
        budget_meter = None
        try:
            while True:
                try:
                    request = steps.send(response)
                except StopIteration as done:
                    return done.value
                except _BudgetStopIteration as transport:
                    raise transport.original from transport.original.__cause__
                except (_driver_outcomes._OutcomeStopIteration, _PlanningStopIteration) as transport:
                    error = transport.original
                    original_context = error.__context__
                    try:
                        raise error
                    finally:
                        error.__context__ = original_context
                response = None
                if request in (
                    _driver_outcomes.OutcomeRead.CANCELLATION_REASON,
                    _driver_outcomes.OutcomeRead.SCOPE_CANCELLED,
                ):
                    token = self.scope.token
                    assert isinstance(token, Cancellation)
                    response = token.reason if request is _driver_outcomes.OutcomeRead.CANCELLATION_REASON else token.cancelled
                elif request is _driver_outcomes.OutcomeRead.HOST_CANCELLED:
                    response = (
                        self.runtime.config.cancellation_token is not None and self.runtime.config.cancellation_token.cancelled
                    )
                elif request is _driver_outcomes.OutcomeRead.APPROVAL_TIMEOUT:
                    response = self.runtime.config.approval_timeout_seconds
                elif request is _driver_outcomes.OutcomeRead.POLL_LEASE:
                    response = self.scope.confirmed_poll()
                elif request is _driver_outcomes.OutcomeRead.POLL_INTERVAL:
                    response = self.runtime.poll_ms
                elif request is _driver_outcomes.OutcomeRead.ACTIVE_INTERVAL_NS:
                    response = self.clock.monotonic_ns()
                elif request is _driver_outcomes.OutcomeRead.BIND_BUDGET_METER:
                    budget_meter = self.runtime.config.host_cost_meter
                elif request is BudgetObservation.CLOCK_NS:
                    response = self.clock.monotonic_ns()
                elif request is BudgetObservation.HOST_COST:
                    if budget_meter is None:
                        response = HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_MISSING)
                    else:
                        try:
                            reading = budget_meter.read()
                        except Exception:
                            response = HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_ERROR)
                        else:
                            response = HostCostObservation(reading=reading)
                elif isinstance(request, _driver_outcomes.PrepareToolResources):
                    assert prepare_resources is not None
                    response = prepare_resources(request.plan, request.result, self.runtime.workspace_backend)
                elif isinstance(request, _driver_outcomes.LookupYieldCallback):
                    binding = (
                        self.runtime.children.get(request.plan._payload["request"]["name"])
                        if request.child
                        else self.provider(request.plan)
                    )
                    response = getattr(binding, "on_yield", None)
                elif isinstance(request, _driver_outcomes.InvokeYieldCallback):
                    try:
                        response = request.callback(request.plan, request.handles, "steer")
                    except Exception as error:
                        response = _driver_outcomes.YieldCallbackError(error)
                elif isinstance(request, ProviderAvailable):
                    response = request.name in self.runtime.providers
                else:
                    raise AssertionError(f"Unknown receipt observation: {request!r}")
        finally:
            steps.close()

    def compaction_intent(self, plan: Record, result: dict[str, Any]) -> dict[str, Any]:
        return _driver_outcomes.compaction_intent(self, plan, result)

    def completed(self, plan: Record, outcome: Definitive) -> list[Record]:
        from .tool_results import prepare_tool_resources

        return self._run_outcome_steps(
            _driver_outcomes.completed_steps(self, plan, outcome), prepare_resources=prepare_tool_resources
        )

    def completion_effects(self, plan: Record, outcome: Definitive, context: str) -> list[Record]:
        return self._run_outcome_steps(_driver_outcomes.completion_effects_steps(self, plan, outcome, context))

    def unknown(self, plan: Record, reason: str) -> Record:
        return _driver_recovery.unknown(self, plan, reason)

    def provider(self, plan: Record):
        binding = plan._payload["provider_binding"]
        if binding == "function":
            return self.runtime.functions
        if binding not in self.runtime.providers:
            raise MissingHostBinding(f"Missing provider host binding: {binding}")
        provider = self.runtime.providers[binding]
        if binding == SUB_TASK_STATUS_TOOL_NAME:
            from .delegation import SubTaskStatusProvider

            if isinstance(provider, SubTaskStatusProvider):
                return provider.bind_confirmed_poll(lambda: self.scope.confirmed_poll().db_now_ms)
        return provider

    def check_provider_result(self, plan: Record, outcome: Definitive) -> None:
        return _driver_outcomes.check_provider_result(self, plan, outcome)

    def parked(self, plan: Record, handle: dict[str, Any], *, after: bool) -> Record:
        return self._run_outcome_steps(_driver_outcomes.parked_steps(self, plan, handle, after=after))

    def accepted(self, plan: Record, outcome: Accepted) -> Record:
        return self._run_outcome_steps(_driver_outcomes.accepted_steps(self, plan, outcome))

    def yield_results(self, attempts: list[Attempt], item: InboxItem) -> list[Record]:
        def prepare_resources(plan: Record, result: dict[str, Any], backend: Any) -> dict[str, Any]:
            from .tool_results import prepare_tool_resources

            return prepare_tool_resources(plan, result, backend)

        return self._run_outcome_steps(
            _driver_outcomes.yield_results_steps(self, attempts, item), prepare_resources=prepare_resources
        )

    def apply_input(self) -> bool:
        return _driver_inputs.apply_input(self)

    def apply_child(self, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
        return _driver_inputs.apply_child(self, item)

    def start_turn(self) -> bool:
        return _driver_inputs.start_turn(self)

    def transcript(self) -> list[Message]:
        if self.content_reader is None:
            return project_context(self.records, self.state, _fold_factory=_fold_factory_for(self.store))
        if self._active_logical_fold is not None and len(self._active_logical_fold.history) == len(self.records):
            return project_context(
                self._logical_hydrated_records,
                self._active_logical_fold.state,
                _fold_factory=_fold_factory_for(self.store),
            )
        hydration = hydration_for_session_store(self.store, self.sid)
        records = hydrate_records(self.records, hydration)
        state = fold_hydrated_records(records, self.records, _fold_factory_for(self.store), hydration)
        return project_context(records, state, _fold_factory=_fold_factory_for(self.store))

    def shared_state(self) -> dict[str, Any]:
        assert self.state.active_turn_id is not None
        return self.runtime.bind_state(self.durable_shared_state(), self._task())

    def model_plan(self) -> Record | None:
        return _driver_planning.model_plan(self)

    def close(self, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
        with self._logical_state_scope():
            return _driver_recovery.close(self, status, reason, result, transferred=transferred)

    def dispatch(self, attempt: Attempt) -> None:
        return _driver_dispatch.dispatch(self, attempt)

    def finalize_tool(self, plan: Record, context, outcome: Definitive) -> Definitive:
        return _driver_dispatch.finalize_tool(self, plan, context, outcome)

    def step(self) -> bool:
        if self.fold.state.active_turn_id is None:
            return _driver_recovery.step(self)
        with self._logical_state_scope():
            return _driver_recovery.step(self)


def drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    expected_head: SessionHead | tuple[int, str] | None = None,
    _one_turn: bool = False,
    _wait_for_lease: bool = False,
    _lease: Lease | None = None,
    _initial_polled: set[tuple[str, int]] | None = None,
) -> bool:
    """Run the session through the shared lease-fenced driver."""
    return _drive(
        store,
        session_id,
        runtime=runtime,
        failure_backoff_ms=failure_backoff_ms,
        expected_head=expected_head,
        _one_turn=_one_turn,
        _wait_for_lease=_wait_for_lease,
        _lease=_lease,
        _initial_polled=_initial_polled,
    )


def _drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    expected_head: SessionHead | tuple[int, str] | None = None,
    _one_turn: bool = False,
    _wait_for_lease: bool = False,
    _lease: Lease | None = None,
    _initial_polled: set[tuple[str, int]] | None = None,
    clock: Clock | None = None,
    lease_keeper: LeaseKeeper | None = None,
) -> bool:
    """Drive with runtime bindings and optional factory-admission seams."""
    return _drive_phase.drive_with_runtime(
        store,
        session_id,
        runtime=runtime,
        failure_backoff_ms=failure_backoff_ms,
        expected_head=expected_head,
        one_turn=_one_turn,
        wait_for_lease=_wait_for_lease,
        lease=_lease,
        initial_polled=_initial_polled,
        clock=clock,
        lease_keeper=lease_keeper,
    )
