"""One retained budget projection; evaluator decisions and usage conversion stay shared."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from vv_agent.kernel._budget import BudgetObservation, CoreBudgetEvaluator
from vv_agent.kernel._budget_values import (
    BudgetDimension,
    BudgetUnavailableDimension,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    RunBudgetLimits,
)
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._hydration import KeyUnavailable, TypedSchemaFailure, _parse_budget
from vv_agent.kernel._records import Record
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._usage import model_usage


class RetainedBudgetHydrationRequired(KeyUnavailable):
    """The budget was retained as a reference and needs a logical-state retry."""


class RetainedBudgetHydrator(Protocol):
    """Owner-bound hydration used only for content consumed by budget reconstruction."""

    def hydrate_record_field(self, record: Record, slot: str, ref: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class RetainedBudgetInputs:
    limits: RunBudgetLimits
    usage: BudgetUsageSnapshot
    host_cost: dict[str, Any] | None
    model_usages: tuple[dict[str, Any], ...]

    def account_models(self, evaluator: CoreBudgetEvaluator) -> None:
        for usage in self.model_usages:
            evaluator.observe_model_usage(model_usage(usage))


def _retained_value(record: Record, slot: str, value: Any, hydrator: RetainedBudgetHydrator | None) -> Any:
    try:
        retained = declared_content_ref(value)
    except ValueError as exc:
        raise TypedSchemaFailure(f"{record.kind}.{slot} contains a malformed ContentRef") from exc
    if retained is None:
        return value
    if hydrator is None:
        raise RetainedBudgetHydrationRequired(f"{record.kind}.{slot} requires hydrated logical state")
    return hydrator.hydrate_record_field(record, slot, retained)


def _retained_boundary_data(record: Record, hydrator: RetainedBudgetHydrator | None) -> dict[str, Any]:
    data = record._payload["data"]
    try:
        retained = declared_content_ref(data["usage"])
    except ValueError as exc:
        raise TypedSchemaFailure("boundary_recorded.budget_usage contains a malformed ContentRef") from exc
    if retained is None:
        return data
    if hydrator is None:
        raise RetainedBudgetHydrationRequired("boundary_recorded.budget_usage requires hydrated logical state")
    return data | {"usage": hydrator.hydrate_record_field(record, "budget_usage", retained)}


def _usage_object(record: Record, slot: str, value: Any, hydrator: RetainedBudgetHydrator | None) -> dict[str, Any]:
    usage = _retained_value(record, slot, value, hydrator)
    if type(usage) is not dict:
        raise TypedSchemaFailure(f"{record.kind}.{slot} must be an object")
    return usage


def _budget_snapshot(value: Any) -> BudgetUsageSnapshot:
    if type(value) is not dict:
        raise TypedSchemaFailure("boundary_recorded.budget_usage must be an object")
    try:
        snapshot = BudgetUsageSnapshot.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise TypedSchemaFailure("boundary_recorded.budget_usage does not match BudgetUsageSnapshot") from exc
    if snapshot.to_dict() != value:
        raise TypedSchemaFailure("boundary_recorded.budget_usage does not match BudgetUsageSnapshot")
    return snapshot


def _missing_active_interval(record: Record | None, hydrator: RetainedBudgetHydrator | None) -> bool:
    if record is None:
        return False
    observation = _retained_value(record, "op_unknown_observation", record._payload["observation"], hydrator)
    if type(observation) is not dict:
        raise TypedSchemaFailure("op_unknown.observation must be an object")
    return observation.get("active_interval_missing") is True


def retained_budget_inputs(
    state: ExecutionState,
    tid: str,
    *,
    hydrator: RetainedBudgetHydrator | None = None,
) -> RetainedBudgetInputs | None:
    start = state.turns[tid].start
    retained = _retained_value(start, "budget", start._payload["budget"], hydrator)
    parsed = _parse_budget(retained)
    if not parsed.has_limits:
        return None
    observation_records = [r for r in state.usage_values.values() if r.turn_id == tid]
    observations = [_usage_object(r, "usage", r._payload["usage"], hydrator) for r in observation_records]
    elapsed = sum(v.get("elapsed_ms", 0) for v in observations)
    boundaries = [
        _retained_boundary_data(r, hydrator)
        for (turn, stage, _), r in state.boundaries.items()
        if turn == tid and stage == "budget"
    ]
    previous = _budget_snapshot(boundaries[-1]["usage"]) if boundaries else BudgetUsageSnapshot()
    missing_interval = any(
        _missing_active_interval(a.unknown, hydrator)
        for op in state.operations.values()
        if op.turn_id == tid
        for a in op.attempts.values()
    )
    unavailable = list(previous.unavailable_dimensions)
    if missing_interval and not any(v.dimension == BudgetDimension.WALL_TIME for v in unavailable):
        unavailable.append(BudgetUnavailableDimension(BudgetDimension.WALL_TIME, BudgetUnavailableReason.ACCOUNTING_MISSING))
    started = [(op.kind, a) for op in state.operations.values() if op.turn_id == tid for a in op.attempts.values() if a.started]
    tool_counts = Counter(name for b in boundaries for name in b["tool_names"])
    # Host readings and unavailable classifications are retained at every active-scope commit.
    host = observations[-1].get("host_cost") if observations else None
    return RetainedBudgetInputs(
        limits=parsed,
        usage=BudgetUsageSnapshot(
            cycles=sum(kind == "model" and a.plan._payload["purpose"] == "primary" for kind, a in started),
            tool_calls=sum(tool_counts.values()),
            tool_calls_by_name=tool_counts,
            elapsed_ms=elapsed,
            host_cost=previous.host_cost,
            unavailable_dimensions=tuple(unavailable),
        ),
        host_cost=host,
        model_usages=tuple(
            _usage_object(attempt.result, "usage", attempt.result._payload["usage"], hydrator) if attempt.result else {}
            for kind, attempt in started
            if kind == "model" and (attempt.result is not None or attempt.unknown is not None)
        ),
    )


def retained_evaluator(inputs: RetainedBudgetInputs) -> CoreBudgetEvaluator:
    """Restore the real evaluator with the original deterministic replay clock."""
    steps = CoreBudgetEvaluator.create_steps(inputs.limits, initial_usage=inputs.usage)
    try:
        assert next(steps) is BudgetObservation.CLOCK_NS
        try:
            steps.send(0)
        except StopIteration as done:
            evaluator = done.value
        else:
            raise AssertionError("retained budget initialization requested another observation")
    finally:
        steps.close()
    inputs.account_models(evaluator)
    return evaluator
