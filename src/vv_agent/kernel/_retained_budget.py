"""One retained budget projection; evaluator decisions and usage conversion stay shared."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from vv_agent.kernel._budget import BudgetObservation, CoreBudgetEvaluator
from vv_agent.kernel._budget_values import (
    BudgetDimension,
    BudgetUnavailableDimension,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    RunBudgetLimits,
)
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._usage import model_usage


@dataclass(frozen=True, slots=True)
class RetainedBudgetInputs:
    limits: RunBudgetLimits
    usage: BudgetUsageSnapshot
    host_cost: dict[str, Any] | None
    model_usages: tuple[dict[str, Any], ...]

    def account_models(self, evaluator: CoreBudgetEvaluator) -> None:
        for usage in self.model_usages:
            evaluator.observe_model_usage(model_usage(usage))


def retained_budget_inputs(state: ExecutionState, tid: str) -> RetainedBudgetInputs | None:
    parsed = RunBudgetLimits.from_dict(state.turns[tid].start._payload["budget"])
    if not parsed.has_limits:
        return None
    observations = [r._payload["usage"] for r in state.usage_values.values() if r.turn_id == tid]
    elapsed = sum(v.get("elapsed_ms", 0) for v in observations)
    boundaries = [r._payload["data"] for (turn, stage, _), r in state.boundaries.items() if turn == tid and stage == "budget"]
    previous = BudgetUsageSnapshot.from_dict(boundaries[-1]["usage"]) if boundaries else BudgetUsageSnapshot()
    missing_interval = any(
        a.unknown and a.unknown._payload["observation"].get("active_interval_missing")
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
            attempt.result._payload["usage"] if attempt.result else {}
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
