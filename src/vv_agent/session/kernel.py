"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import replace
from typing import Any

from vv_agent.budget import BudgetEvaluator, BudgetExhaustion
from vv_agent.kernel import _driver_outcomes
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation, _BudgetStopIteration
from vv_agent.kernel._budget_values import BudgetUnavailableReason
from vv_agent.kernel._content import ContentRef, SessionHead
from vv_agent.kernel._content_publication import _payload_slots, declared_content_ref
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_planning import ProviderAvailable, _PlanningStopIteration
from vv_agent.kernel._hydration import DefinitionContent
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._reducer import (
    TransitionError,
    logical_boundary_payloads,
    logical_boundary_sources,
    logical_inputs,
    logical_operation_payloads,
    logical_operation_requests,
    logical_turn_wait_results,
    retry_request_identity,
)
from vv_agent.types import AgentTask, Message

from . import _drive_phase, _driver_dispatch, _driver_inputs, _driver_planning, _driver_recovery
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _LeaseHeartbeat as _LeaseHeartbeat
from ._driver_scope import _Scope
from .bindings import MissingHostBinding
from .context import project_context
from .projection import (
    _RetainedInboxItem,
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
from .store import SessionStore, StoredRecord
from .timing import Clock, LeaseKeeper


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


logger = logging.getLogger(__name__)

_LOGICAL_PREFLIGHT_KINDS = frozenset(
    {
        "op_planned",
        "op_parked",
        "op_completed",
        "op_unknown",
        "turn_parked",
        "turn_ended",
        "boundary_recorded",
        "context_compacted",
    }
)


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
        )

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[Any], list[Record]] | None = None,
    ) -> None:
        # A surrounding execution scope may hold logical state. Commits always advance the raw fold.
        logical_state = self.state if self.content_reader is not None and self.state is not self.fold.state else None
        record_count = len(self.records)
        if logical_state is not None:
            self.state = self.fold.state
        parked = [record for record in records if record.kind == "turn_parked"]
        applied = [record for record in records if record.kind == "input_applied"]
        operations = [record for record in records if record.kind in {"op_planned", "op_completed", "op_parked"}]
        boundaries = [
            record for record in records if record.kind == "boundary_recorded" and record._payload["source_digest"] is not None
        ]
        try:
            if (parked or applied or operations or boundaries) and self.content_reader is not None:
                hydration = hydration_for_session_store(self.store, self.sid)
                logical_input_items = {
                    record.record_id: hydrate_input(InboxItem(**record._payload["input"]), hydration) for record in applied
                }
                operation_payloads: dict[str, dict[str, Any]] = {record.record_id: record._payload for record in operations}
                operation_requests: dict[tuple[str, int], str] = {}
                if any(record.kind in {"op_planned", "op_parked"} for record in operations):
                    state = logical_state or self._logical_state()
                    for record in operations:
                        operation = state.operations.get(record.operation_id or "")
                        if operation is None or record.kind not in {"op_planned", "op_parked"}:
                            continue
                        attempt = operation.attempts.get(record.attempt or 0)
                        if attempt is None and record.kind == "op_planned":
                            attempt = operation.attempts[max(operation.attempts)]
                        if attempt is None:
                            continue
                        hydrated_plan = self.hydrate_plan(attempt.execution_plan)
                        plan_operation_id, plan_attempt = hydrated_plan.operation_id, hydrated_plan.attempt
                        assert plan_operation_id is not None and plan_attempt is not None
                        operation_payloads[hydrated_plan.record_id] = copy_json(hydrated_plan._payload)
                        operation_requests[(plan_operation_id, plan_attempt)] = retry_request_identity(
                            hydrated_plan._payload["request"]
                        )
                        if record.kind == "op_planned" and _is_content_ref(record._payload["request"]):
                            logical_plan = copy_json(record._payload)
                            for slot in ("request", "tool", "budget_admission"):
                                logical_plan[slot] = copy_json(hydrated_plan._payload[slot])
                            operation_payloads[record.record_id] = logical_plan
                boundary_sources = {}
                boundary_payloads = {}
                if boundaries:
                    state = logical_state or self._logical_state()
                    source = self._logical_boundary_source(state)
                    boundary_sources = {record.record_id: source for record in boundaries}
                    boundary_payloads = {record.record_id: record._payload for record in boundaries}
                logical_results = {}
                if parked:
                    state = logical_state or self._logical_state()
                    for record in parked:
                        operation = state.operations.get(record._payload["source_operation_id"])
                        attempt = operation.attempts.get(record._payload["source_attempt"]) if operation is not None else None
                        if attempt is not None and attempt.result is not None:
                            logical_results[record.record_id] = attempt.result._payload["result"]
                with (
                    logical_boundary_payloads(boundary_payloads),
                    logical_boundary_sources(boundary_sources),
                    logical_turn_wait_results(logical_results),
                    logical_inputs(logical_input_items),
                    logical_operation_payloads(operation_payloads),
                    logical_operation_requests(operation_requests),
                ):
                    result = super().commit(records, consumed, guarded=guarded, prepare=prepare)
            else:
                result = super().commit(records, consumed, guarded=guarded, prepare=prepare)
            return result
        finally:
            if logical_state is not None:
                self.state = logical_state if len(self.records) == record_count else self._logical_state()

    def _logical_preflight(self, records: list[Record], encoded_records: list[Record]) -> None:
        if self.content_reader is None or not any(record.kind in _LOGICAL_PREFLIGHT_KINDS for record in records):
            return

        for record in records:
            for path, _slot, value in _payload_slots(record.kind, record._payload):
                if record.kind == "input_applied" and path[:1] == ("input",):
                    continue
                if value is None:
                    continue
                try:
                    retained = declared_content_ref(value)
                except ValueError as exc:
                    raise TransitionError("logical preflight received a malformed ContentRef") from exc
                if retained is not None:
                    raise TransitionError("logical preflight requires plaintext producer values with a bound content owner")

        state = self.state
        snapshot = self._fold_factory()
        snapshot.state = state
        probe = self._fold_factory()
        probe.state = snapshot.snapshot()
        probe.history = list(self.records)
        probe.seen = {stored.record.record_id: stored.record.encode() for stored in self.records}
        probe.consumed = {
            input_id: InboxItem(**state.applied_inputs[input_id]._payload["input"]) for input_id in state.applied_inputs
        }
        hydration = hydration_for_session_store(self.store, self.sid)

        operation_payloads = {
            attempt.execution_plan.record_id: copy_json(self.hydrate_plan(attempt.execution_plan)._payload)
            for operation in state.operations.values()
            for attempt in operation.attempts.values()
        }
        logical_items: dict[str, InboxItem] = {}
        consumed: list[InboxItem] = []
        for record in records:
            if record.kind != "input_applied":
                continue
            retained = InboxItem(**record._payload["input"])
            item = hydrate_input(retained, hydration)
            logical_items[record.record_id] = item
            consumed.append(_RetainedInboxItem(item, retained.encode()))
        if state is self.fold.state:
            for turn in probe.state.turns.values():
                turn.start = hydrate_stored_record(self.stored_record(turn.start), hydration).record
            for operation_id, operation in probe.state.operations.items():
                for number, attempt in operation.attempts.items():
                    if attempt.wait is not None:
                        parked = next(
                            stored
                            for stored in reversed(self.records)
                            if stored.record.kind == "op_parked"
                            and stored.record.operation_id == operation_id
                            and stored.record.attempt == number
                        )
                        attempt.wait = copy_json(hydrate_stored_record(parked, hydration).record._payload)
                    if attempt.result is not None:
                        attempt.result = hydrate_stored_record(self.stored_record(attempt.result), hydration).record

        boundary_sources = {}
        if any(record.kind == "boundary_recorded" and record._payload["source_digest"] is not None for record in records):
            source = self._logical_boundary_source(state)
            boundary_sources = {
                record.record_id: source
                for record in records
                if record.kind == "boundary_recorded" and record._payload["source_digest"] is not None
            }

        with (
            logical_inputs(logical_items),
            logical_operation_payloads(operation_payloads),
            logical_boundary_sources(boundary_sources),
        ):
            probe.extend(records, consumed_inputs=consumed, bodies=(record.encode() for record in encoded_records))

    def _logical_boundary_source(self, state: ExecutionState) -> list[Message]:
        if state._protection is not None:
            return state._protection.messages
        hydration = hydration_for_session_store(self.store, self.sid)
        projection_kinds = {
            "session_created",
            "turn_started",
            "input_applied",
            "op_completed",
            "context_compacted",
            "boundary_recorded",
        }
        records = tuple(
            hydrate_stored_record(stored, hydration) for stored in self.records if stored.record.kind in projection_kinds
        )
        return project_context(records, state, _fold_factory=_fold_factory_for(self.store))

    def _logical_state(self) -> ExecutionState:
        if self.content_reader is None or self.state is not self.fold.state:
            return self.state
        hydration = hydration_for_session_store(self.store, self.sid)
        inputs = tuple(stored for stored in self.records if stored.record.kind == "input_applied")
        retry_payloads = self._logical_retry_plan_payloads(hydration)
        with logical_operation_payloads(retry_payloads):
            state = fold_hydrated_records(
                hydrate_records(inputs, hydration), self.records, _fold_factory_for(self.store), hydration
            )
        # Keep interaction plans logical for reply identity; planning and provider
        # callbacks reopen other plans through their raw owner records.
        state.operations = {
            operation_id: replace(
                operation,
                attempts={
                    number: replace(
                        attempt,
                        plan=(
                            self.hydrate_plan(attempt.plan)
                            if operation.kind == "interaction" or attempt.plan.record_id in retry_payloads
                            else self.stored_record(attempt.plan).record
                        ),
                        prepared=(self.stored_record(attempt.prepared).record if attempt.prepared is not None else None),
                    )
                    for number, attempt in operation.attempts.items()
                },
            )
            for operation_id, operation in state.operations.items()
        }
        return state

    def _logical_retry_plan_payloads(self, hydration) -> dict[str, dict[str, Any]]:
        plans = {}
        payloads = {}
        for stored in self.records:
            if stored.record.kind != "op_planned":
                continue
            record = hydrate_stored_record(stored, hydration, include_input=False).record
            previous = plans.get(record.operation_id)
            retained = _is_content_ref(record._payload["request"])
            if previous is not None and retained:
                payload = copy_json(record._payload)
                for slot in ("request", "tool", "budget_admission"):
                    payload[slot] = copy_json(previous._payload[slot])
                payloads[record.record_id] = payload
                record = replace(record, payload=payload)
            plans[record.operation_id] = record
        return payloads

    @contextmanager
    def _logical_state_scope(self) -> Iterator[ExecutionState]:
        """Expose a call-local logical view while the raw fold remains authoritative."""

        state = self.state
        nested = state is not self.fold.state
        if self.content_reader is None:
            yield state
            return
        self.state = self._logical_state()
        try:
            yield self.state
        finally:
            self.state = self._logical_state() if nested else self.fold.state

    def _logical_input(self, item: InboxItem) -> InboxItem:
        if self.content_reader is None:
            return item
        return hydrate_input(item, hydration_for_session_store(self.store, self.sid))

    def live_budget(self) -> BudgetEvaluator | None:
        with self._logical_state_scope():
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
        stored = next(
            (row for row in self.records if row.record is record or row.record.record_id == record.record_id),
            None,
        )
        return super().record_content_ref(stored.record if stored is not None else record, slot)

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
        stored = next(
            (row for row in self.records if row.record is record or row.record.record_id == record.record_id),
            None,
        )
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
                if request is _driver_outcomes.OutcomeRead.CANCELLATION_REASON:
                    response = self.scope.token.reason
                elif request is _driver_outcomes.OutcomeRead.SCOPE_CANCELLED:
                    response = self.scope.token.cancelled
                elif request is _driver_outcomes.OutcomeRead.HOST_CANCELLED:
                    response = (
                        self.runtime.config.cancellation_token is not None and self.runtime.config.cancellation_token.cancelled
                    )
                elif request is _driver_outcomes.OutcomeRead.APPROVAL_TIMEOUT:
                    response = self.runtime.config.approval_timeout_seconds
                elif request is _driver_outcomes.OutcomeRead.POLL_LEASE:
                    response = self.scope.poll(self.store)
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
        return self.runtime.providers[binding]

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
        clock=clock,
        lease_keeper=lease_keeper,
    )
