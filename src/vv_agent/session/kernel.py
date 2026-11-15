"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import suppress
from copy import deepcopy
from typing import Any

from vv_agent.budget import BudgetEvaluator, BudgetExhaustion
from vv_agent.kernel import _driver_outcomes
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation, _BudgetStopIteration
from vv_agent.kernel._budget_values import BudgetUnavailableReason
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_planning import ProviderAvailable, _PlanningStopIteration
from vv_agent.kernel._hydration import DefinitionContent, TypedSchemaFailure
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._reducer import logical_inputs, logical_operation_payloads, logical_turn_wait_results
from vv_agent.types import AgentTask, Message

from . import _drive_phase, _driver_dispatch, _driver_inputs, _driver_planning, _driver_recovery
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _LeaseHeartbeat as _LeaseHeartbeat
from ._driver_scope import _Scope
from .bindings import MissingHostBinding
from .context import project_context
from .projection import fold_hydrated_records, hydrate_input, hydrate_records, hydration_for_session_store
from .providers import (
    Accepted,
    Definitive,
)
from .records import InboxItem, Record
from .reducer import Attempt, ExecutionState, _fold_factory_for
from .runtime import Runtime
from .runtime import RuntimeNotReady as RuntimeNotReady
from .store import SessionStore, StoredRecord
from .timing import Clock, LeaseKeeper


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


logger = logging.getLogger(__name__)


class _Driver(CoreDriver):
    scope: _Scope

    def __init__(self, store: SessionStore, sid: str, runtime: Runtime, scope: _Scope, *, one_turn: bool = False):
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
        )

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[Any], list[Record]] | None = None,
    ) -> None:
        parked = [record for record in records if record.kind == "turn_parked"]
        applied = [record for record in records if record.kind == "input_applied"]
        if (parked or applied) and self.content_reader is not None:
            hydration = hydration_for_session_store(self.store, self.sid)
            logical_input_items = {
                record.record_id: hydrate_input(InboxItem(**record._payload["input"]), hydration) for record in applied
            }
            operation_payloads = {record.record_id: record._payload for record in records if record.kind == "op_completed"}
            logical_results = {}
            if parked:
                hydrated = hydrate_records(self.records, hydration)
                state = fold_hydrated_records(hydrated, self.records, _fold_factory_for(self.store), hydration)
                for record in parked:
                    operation = state.operations.get(record._payload["source_operation_id"])
                    attempt = operation.attempts.get(record._payload["source_attempt"]) if operation is not None else None
                    if attempt is not None and attempt.result is not None:
                        logical_results[record.record_id] = attempt.result._payload["result"]
            with (
                logical_turn_wait_results(logical_results),
                logical_inputs(logical_input_items),
                logical_operation_payloads(operation_payloads),
            ):
                return super().commit(records, consumed, guarded=guarded, prepare=prepare)
        return super().commit(records, consumed, guarded=guarded, prepare=prepare)

    def _logical_state(self) -> ExecutionState:
        if self.content_reader is None:
            return self.state
        hydration = hydration_for_session_store(self.store, self.sid)
        hydrated = hydrate_records(self.records, hydration)
        return fold_hydrated_records(hydrated, self.records, _fold_factory_for(self.store), hydration)

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
        stored = next(
            (row for row in self.records if row.record is start or row.record.record_id == start.record_id),
            None,
        )
        if stored is None:
            raise RecordError("turn_started record is outside the retained ledger")
        try:
            ref = ContentRef.from_dict(stored.record._payload["definition"])
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("turn_started.definition must be a ContentRef") from exc
        return self.hydrator_for(stored.seq).definition(ref)

    def hydrated_definition(self, record: Record) -> DefinitionContent:
        return super().hydrated_definition(self.stored_record(record).record)

    def record_content_ref(self, record: Record, slot: str) -> ContentRef | None:
        stored = next(
            (row for row in self.records if row.record is record or row.record.record_id == record.record_id),
            None,
        )
        return super().record_content_ref(stored.record if stored is not None else record, slot)

    def hydrate_record_field(self, record: Record, slot: str, ref: Any) -> Any:
        stored = next(
            (row for row in self.records if row.record is record or row.record.record_id == record.record_id),
            None,
        )
        if stored is None:
            return super().hydrate_record_field(record, slot, ref)
        raw = stored.record
        return super().hydrate_record_field(raw, slot, self._content_field(raw, slot))

    def hydrate_plan(self, record: Record) -> Record:
        return super().hydrate_plan(self.stored_record(record).record)

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
        return _driver_recovery.close(self, status, reason, result, transferred=transferred)

    def dispatch(self, attempt: Attempt) -> None:
        return _driver_dispatch.dispatch(self, attempt)

    def finalize_tool(self, plan: Record, context, outcome: Definitive) -> Definitive:
        return _driver_dispatch.finalize_tool(self, plan, context, outcome)

    def step(self) -> bool:
        return _driver_recovery.step(self)


def drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    _one_turn: bool = False,
    _wait_for_lease: bool = False,
) -> bool:
    """Run the session through the shared lease-fenced driver."""
    return _drive(
        store,
        session_id,
        runtime=runtime,
        failure_backoff_ms=failure_backoff_ms,
        _one_turn=_one_turn,
        _wait_for_lease=_wait_for_lease,
    )


def _drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
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
        one_turn=_one_turn,
        wait_for_lease=_wait_for_lease,
        clock=clock,
        lease_keeper=lease_keeper,
    )
