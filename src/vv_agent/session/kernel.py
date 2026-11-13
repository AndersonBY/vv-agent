"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import ExitStack, suppress
from dataclasses import replace
from typing import Any
from uuid import uuid4

from vv_agent.budget import _DEFAULT_BUDGET_CLOCK_NS, BudgetEvaluator, BudgetExhaustion
from vv_agent.kernel import _driver_outcomes
from vv_agent.kernel._budget import BudgetObservation, HostCostObservation, _BudgetStopIteration
from vv_agent.kernel._budget_values import BudgetUnavailableReason
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_planning import ProviderAvailable, _PlanningStopIteration
from vv_agent.kernel._sql import SQLStore
from vv_agent.tools.outcomes import validate_definitive_result
from vv_agent.types import AgentTask, Message, ToolDirective, ToolExecutionResult

from . import _driver_dispatch, _driver_inputs, _driver_planning, _driver_recovery
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _LeaseHeartbeat, _Scope
from .bindings import MissingHostBinding
from .children import child_handles
from .context import project_context
from .providers import (
    Accepted,
    Definitive,
)
from .records import InboxItem, Record, copy_json
from .reducer import Attempt, ExecutionState, _fold_factory_for
from .runtime import Runtime, RuntimeNotReady
from .store import Lease, LeaseLost, SequenceConflict, SessionStore, StoredRecord
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


FACTORY_LEASE_TTL_MS = 15000
FACTORY_HEARTBEAT_SECONDS = 0.25
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
        )

    @property
    def clock(self) -> Clock:
        return self.runtime.clock

    def live_budget(self) -> BudgetEvaluator | None:
        inputs = self._run_outcome_steps(_driver_outcomes.live_budget_inputs_steps(self))
        if inputs is None:
            return None
        limits, initial = inputs
        return BudgetEvaluator(limits, initial_usage=initial, host_cost_meter=self.runtime.config.host_cost_meter)

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
        return self.state.turns[self.state.active_turn_id].start.task()

    def _task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start._task()

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
                    response = self.runtime.clock.monotonic_ns()
                elif request is _driver_outcomes.OutcomeRead.BIND_BUDGET_METER:
                    budget_meter = self.runtime.config.host_cost_meter
                elif request is BudgetObservation.CLOCK_NS:
                    response = _DEFAULT_BUDGET_CLOCK_NS()
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
        records = []
        for a in attempts:
            plan = a.execution_plan
            assert plan.operation_id is not None and plan.attempt is not None
            assert a.wait is not None
            handle = a.wait["handle"]
            handles = child_handles(handle) if a.child_handle else handle
            binding = self.runtime.children.get(plan._payload["request"]["name"]) if a.child_handle else self.provider(plan)
            callback = getattr(binding, "on_yield", None)
            metadata: dict[str, Any] = {"yield_reason": "steer"}
            if a.child_handle:
                completed = self.state.child_completions or {}
                handles = [
                    h
                    | {
                        "status": (
                            completed[key]._payload["input"]["payload"]["status"]
                            if (key := (plan.operation_id, plan.attempt, h["session_id"])) in completed
                            else "running"
                        )
                    }
                    | (
                        {k: completed[key]._payload["input"]["payload"][k] for k in ("result", "terminal_seq", "terminal_digest")}
                        if key in completed
                        else {}
                    )
                    for h in handles
                ]
                metadata["children"] = [{"session_id": h["session_id"], "status": h["status"]} for h in handles]
                metadata["running_children"] = [
                    h["session_id"] for h in handles if (plan.operation_id, plan.attempt, h["session_id"]) not in completed
                ]
            result = ToolExecutionResult(
                plan._payload["request"]["id"],
                "Wait interrupted by new input; observed work has not been stopped.",
                metadata=metadata,
            )
            if a.child_handle:
                result.content += "\n" + canonical_json_bytes(
                    {"children": metadata["children"], "running_children": metadata["running_children"]}
                ).decode("utf-8")
            if callback is not None:
                try:
                    rendered = callback(plan, copy_json(handles), "steer")
                    validate_definitive_result(rendered)
                    from .tool_results import validate_output_body

                    validate_output_body(rendered)
                    if rendered.tool_call_id != result.tool_call_id or rendered.directive != ToolDirective.CONTINUE:
                        raise ValueError("invalid yield result")
                    canonical_json_bytes(rendered.to_dict())
                    result = rendered
                except Exception:
                    result.metadata["yield_callback_failed"] = True
            evidence = (f"steer/{item.input_id}",)
            if not a.child_handle:
                evidence += (handle["evidence"],)
            completed_records = self.completed(plan, Definitive(result.to_dict(), evidence))
            records.extend(
                replace(r, payload=r._payload | {"yielded_by": item.input_id}) if r.kind == "op_completed" else r
                for r in completed_records
            )
        return records

    def apply_input(self) -> bool:
        return _driver_inputs.apply_input(self)

    def apply_child(self, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
        return _driver_inputs.apply_child(self, item)

    def start_turn(self) -> bool:
        return _driver_inputs.start_turn(self)

    def transcript(self) -> list[Message]:
        return project_context(self.records, self.state, _fold_factory=_fold_factory_for(self.store))

    def shared_state(self) -> dict[str, Any]:
        assert self.state.active_turn_id is not None
        return self.runtime.bind_state(self.durable_shared_state(), self.state.turns[self.state.active_turn_id].start._task())

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


def _defer_after_error(store: SessionStore, lease: Lease, retry_after_ms: int, error: Exception) -> None:
    try:
        store.defer_drive(lease, retry_after_ms=retry_after_ms)
    except LeaseLost:
        pass
    except Exception as backoff_error:
        raise error from backoff_error


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
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    scheduled = not isinstance(runtime, Runtime)
    clock = clock if clock is not None else runtime.clock if isinstance(runtime, Runtime) else SystemClock()
    lease_keeper = (
        lease_keeper
        if lease_keeper is not None
        else runtime.lease_keeper
        if isinstance(runtime, Runtime)
        else ThreadLeaseKeeper()
    )
    ttl_ms = runtime.ttl_ms if isinstance(runtime, Runtime) else FACTORY_LEASE_TTL_MS
    lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    delay = 0.05
    while lease is None and _wait_for_lease:
        clock.sleep(delay)
        delay = min(1.0, delay * 2)
        lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    if lease is None:
        return False
    if not isinstance(runtime, Runtime):
        build = None
        # Resolve store setup on the driving thread (host connections can be thread-bound).
        try:
            heartbeat = store.heartbeat_store()
            build = _LeaseHeartbeat(
                lease,
                lambda: heartbeat,
                ttl_ms,
                FACTORY_HEARTBEAT_SECONDS,
                clock=clock,
                lease_keeper=lease_keeper,
                now_ms=store._now() if isinstance(store, SQLStore) else None,
            )
            build.start()
            try:
                if not build.ready.wait(ttl_ms / 1000):
                    raise LeaseLost("factory heartbeat did not become ready") from build.heartbeat_error
                if build.lost or not build.successful_ready:
                    raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
                runtime = runtime(session_id)
            finally:
                build.close()
                lease = build.lease
            if build.lost:
                raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
            lease = store.renew(lease, ttl_ms=runtime.ttl_ms).lease
        except RuntimeNotReady as exc:
            try:
                _defer_after_error(store, lease, exc.retry_after_ms, exc)
            finally:
                store.release(lease)
            return True
        except Exception as exc:
            if build is not None and build.lease_keeper.is_alive(build):
                raise
            try:
                _defer_after_error(store, lease, failure_backoff_ms, exc)
            finally:
                store.release(lease)
            raise
        except BaseException:
            if build is None or not build.lease_keeper.is_alive(build):
                with suppress(Exception):
                    store.release(lease)
            raise
    scope = _Scope(lease, runtime, now_ms=store._now() if isinstance(store, SQLStore) else None)
    scope.start()
    try:
        driver = _Driver(store, session_id, runtime, scope, one_turn=_one_turn)
        while not scope.lost:
            try:
                if not driver.step():
                    with ExitStack() as latch:
                        if isinstance(store, SQLStore):
                            store._defer_idle_drive(
                                scope.lease,
                                poll_ms=runtime.poll_ms,
                                lease_guard=lambda: latch.enter_context(scope.committing()),
                            )
                        else:
                            store.defer_idle_drive(latch.enter_context(scope.committing()), poll_ms=runtime.poll_ms)
                    break
                if _one_turn and driver.records[-1].record.kind == "turn_ended":
                    break
            except SequenceConflict:
                continue
        if scope.lost:
            raise LeaseLost("heartbeat lost ownership") from scope.heartbeat_error
    except Exception as exc:
        scope.close()
        if scheduled:
            _defer_after_error(store, scope.lease, failure_backoff_ms, exc)
        raise
    finally:
        scope.close()
        with suppress(Exception):
            store.release(scope.lease)
        with suppress(Exception):
            runtime.hook("after_drive", None)
        with suppress(Exception):
            if store.is_runnable(session_id):
                runtime.wake(session_id)
    return True
