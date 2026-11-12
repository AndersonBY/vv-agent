from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vv_agent.kernel._budget_values import (
    _DIMENSION_ORDER,
    _DIMENSION_PRECEDENCE,
    MAX_WIRE_INTEGER,
    BudgetDimension,
    BudgetEnforcementBoundary,
    BudgetExhaustion,
    BudgetExhaustionReason,
    BudgetUnavailableDimension,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    HostCost,
    HostCostMeter,
    RunBudgetLimits,
    UnavailableMetricPolicy,
)

if TYPE_CHECKING:
    from vv_agent.types import TokenUsage


class BudgetEvaluator:
    def __init__(
        self,
        limits: RunBudgetLimits,
        *,
        host_cost_meter: HostCostMeter | None = None,
        initial_usage: BudgetUsageSnapshot | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(limits, RunBudgetLimits) or not limits.has_limits:
            raise ValueError("BudgetEvaluator requires at least one configured limit")
        self.limits = limits
        self._host_cost_meter = host_cost_meter
        self._clock_ns = clock_ns
        self._started_ns = clock_ns()
        usage = initial_usage or BudgetUsageSnapshot()
        self._cycles = usage.cycles
        self._total_tokens = usage.total_tokens
        self._uncached_input_tokens = usage.uncached_input_tokens
        self._tool_calls = usage.tool_calls
        self._tool_calls_by_name = dict(usage.tool_calls_by_name)
        self._base_elapsed_ms = usage.elapsed_ms
        self._elapsed_ms = usage.elapsed_ms
        self._host_cost = usage.host_cost
        self._unavailable = {item.dimension: item for item in usage.unavailable_dimensions}

    @property
    def enabled(self) -> bool:
        return True

    def snapshot(self) -> BudgetUsageSnapshot:
        return BudgetUsageSnapshot(
            cycles=self._cycles,
            total_tokens=self._total_tokens,
            uncached_input_tokens=self._uncached_input_tokens,
            tool_calls=self._tool_calls,
            tool_calls_by_name=self._tool_calls_by_name,
            elapsed_ms=self._elapsed_ms,
            host_cost=self._host_cost,
            unavailable_dimensions=tuple(self._unavailable.values()),
        )

    def run_start(self) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.RUN_START
        self._observe_boundary(boundary)
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        return self._check_admission_limits(boundary, dimensions=(BudgetDimension.WALL_TIME, BudgetDimension.HOST_COST))

    def cycle_start(self) -> BudgetExhaustion | None:
        exhaustion = self.model_call_start()
        if exhaustion is None:
            self._cycles += 1
        return exhaustion

    def model_call_start(self) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.CYCLE_START
        self._observe_boundary(boundary)
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        exhaustion = self._check_admission_limits(
            boundary,
            dimensions=(
                BudgetDimension.WALL_TIME,
                BudgetDimension.TOTAL_TOKENS,
                BudgetDimension.UNCACHED_INPUT_TOKENS,
                BudgetDimension.HOST_COST,
            ),
        )
        return exhaustion

    def model_call_complete(self, token_usage: Any) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.MODEL_CALL_COMPLETE
        self.observe_model_usage(token_usage)
        self._observe_boundary(boundary)
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        return self._check_exceeded_limits(
            boundary,
            dimensions=(
                BudgetDimension.WALL_TIME,
                BudgetDimension.TOTAL_TOKENS,
                BudgetDimension.UNCACHED_INPUT_TOKENS,
                BudgetDimension.HOST_COST,
            ),
        )

    def preflight_tools(self, tool_names: list[str]) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.TOOL_BATCH_PREFLIGHT
        self._observe_boundary(boundary)
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        admission = self._check_admission_limits(
            boundary,
            dimensions=(BudgetDimension.WALL_TIME, BudgetDimension.HOST_COST),
        )
        if admission is not None:
            return admission

        batch_count = len(tool_names)
        if self.limits.max_tool_calls is not None:
            projected = self._tool_calls + batch_count
            if projected > self.limits.max_tool_calls:
                return self._count_preflight_exhaustion(
                    dimension=BudgetDimension.TOOL_CALLS,
                    limit=self.limits.max_tool_calls,
                    observed=self._tool_calls,
                    attempted_increment=batch_count,
                    boundary=boundary,
                )

        batch_by_name: dict[str, int] = {}
        for name in tool_names:
            batch_by_name[name] = batch_by_name.get(name, 0) + 1
        for name in sorted(self.limits.max_tool_calls_by_name):
            increment = batch_by_name.get(name, 0)
            if increment == 0:
                continue
            observed = self._tool_calls_by_name.get(name, 0)
            limit = self.limits.max_tool_calls_by_name[name]
            if observed + increment > limit:
                return self._count_preflight_exhaustion(
                    dimension=BudgetDimension.TOOL_CALLS_BY_NAME,
                    limit=limit,
                    observed=observed,
                    attempted_increment=increment,
                    boundary=boundary,
                    tool_name=name,
                )

        self._tool_calls += batch_count
        for name, increment in batch_by_name.items():
            self._tool_calls_by_name[name] = self._tool_calls_by_name.get(name, 0) + increment
        self._tool_calls_by_name = dict(sorted(self._tool_calls_by_name.items()))
        return None

    def tool_batch_complete(self, *, operation_failed: bool = False) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE
        self._observe_boundary(boundary)
        if operation_failed:
            return None
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        return self._check_exceeded_limits(
            boundary,
            dimensions=(BudgetDimension.WALL_TIME, BudgetDimension.HOST_COST),
        )

    def terminal(self) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.TERMINAL
        self._observe_boundary(boundary)
        unavailable = self._strict_unavailable(boundary)
        if unavailable is not None:
            return unavailable
        return self._check_exceeded_limits(boundary, dimensions=_DIMENSION_PRECEDENCE)

    def _observe_boundary(self, boundary: BudgetEnforcementBoundary) -> None:
        del boundary
        self._observe_elapsed()
        self._observe_host_cost()

    def _observe_elapsed(self) -> None:
        if BudgetDimension.WALL_TIME in self._unavailable:
            return
        now_ns = self._clock_ns()
        delta_ns = max(0, now_ns - self._started_ns)
        elapsed_ms = self._base_elapsed_ms + delta_ns // 1_000_000
        if elapsed_ms > MAX_WIRE_INTEGER:
            self._latch_unavailable(
                BudgetDimension.WALL_TIME,
                BudgetUnavailableReason.INTEGER_OVERFLOW,
                expected_unit="milliseconds",
            )
            return
        self._elapsed_ms = max(self._elapsed_ms, elapsed_ms)

    def _observe_host_cost(self) -> None:
        limit = self.limits.max_host_cost
        if limit is None or BudgetDimension.HOST_COST in self._unavailable:
            return
        if self._host_cost_meter is None:
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.METER_MISSING,
                expected_unit=limit.unit,
                expected_currency=limit.currency,
            )
            return
        try:
            reading = self._host_cost_meter.read()
        except Exception:
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.METER_ERROR,
                expected_unit=limit.unit,
                expected_currency=limit.currency,
            )
            return
        if reading is None:
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.METER_UNAVAILABLE,
                expected_unit=limit.unit,
                expected_currency=limit.currency,
            )
            return
        if not isinstance(reading, HostCost):
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.METER_ERROR,
                expected_unit=limit.unit,
                expected_currency=limit.currency,
            )
            return
        if reading.unit != limit.unit:
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.UNIT_MISMATCH,
                expected_unit=limit.unit,
                observed_unit=reading.unit,
                expected_currency=limit.currency,
                observed_currency=reading.currency,
            )
            return
        if reading.currency != limit.currency:
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.CURRENCY_MISMATCH,
                expected_unit=limit.unit,
                observed_unit=reading.unit,
                expected_currency=limit.currency,
                observed_currency=reading.currency,
            )
            return
        if self._host_cost is not None and reading.amount_microunits < self._host_cost.amount_microunits:
            self._host_cost = None
            self._latch_unavailable(
                BudgetDimension.HOST_COST,
                BudgetUnavailableReason.NON_MONOTONIC,
                expected_unit=limit.unit,
                observed_unit=reading.unit,
                expected_currency=limit.currency,
                observed_currency=reading.currency,
            )
            return
        self._host_cost = reading

    def observe_model_usage(self, usage: TokenUsage) -> None:
        usage_source = getattr(usage, "usage_source", None)
        source_value = getattr(usage_source, "value", usage_source)
        total = getattr(usage, "total_tokens", None)
        if source_value == "accounting_missing" or isinstance(total, bool) or not isinstance(total, int) or total < 0:
            self._total_tokens = None
            self._latch_unavailable(
                BudgetDimension.TOTAL_TOKENS,
                BudgetUnavailableReason.USAGE_MISSING,
                expected_unit="tokens",
            )
        elif self._total_tokens is not None:
            self._total_tokens = self._safe_add_or_latch(
                BudgetDimension.TOTAL_TOKENS,
                self._total_tokens,
                total,
                expected_unit="tokens",
            )

        cache_usage = getattr(usage, "cache_usage", None)
        uncached = getattr(cache_usage, "uncached_input_tokens", None)
        if isinstance(uncached, bool) or not isinstance(uncached, int) or uncached < 0:
            self._uncached_input_tokens = None
            self._latch_unavailable(
                BudgetDimension.UNCACHED_INPUT_TOKENS,
                BudgetUnavailableReason.USAGE_MISSING,
                expected_unit="tokens",
            )
        elif self._uncached_input_tokens is not None:
            self._uncached_input_tokens = self._safe_add_or_latch(
                BudgetDimension.UNCACHED_INPUT_TOKENS,
                self._uncached_input_tokens,
                uncached,
                expected_unit="tokens",
            )

    def _safe_add_or_latch(
        self,
        dimension: BudgetDimension,
        current: int,
        increment: int,
        *,
        expected_unit: str,
    ) -> int | None:
        total = current + increment
        if total <= MAX_WIRE_INTEGER:
            return total
        self._latch_unavailable(
            dimension,
            BudgetUnavailableReason.INTEGER_OVERFLOW,
            expected_unit=expected_unit,
        )
        if dimension is BudgetDimension.TOTAL_TOKENS:
            self._total_tokens = None
        elif dimension is BudgetDimension.UNCACHED_INPUT_TOKENS:
            self._uncached_input_tokens = None
        return None

    def _latch_unavailable(
        self,
        dimension: BudgetDimension,
        reason: BudgetUnavailableReason,
        *,
        expected_unit: str | None = None,
        observed_unit: str | None = None,
        expected_currency: str | None = None,
        observed_currency: str | None = None,
    ) -> None:
        if dimension in self._unavailable:
            return
        self._unavailable[dimension] = BudgetUnavailableDimension(
            dimension=dimension,
            reason=reason,
            expected_unit=expected_unit,
            observed_unit=observed_unit,
            expected_currency=expected_currency,
            observed_currency=observed_currency,
        )
        self._unavailable = dict(sorted(self._unavailable.items(), key=lambda item: _DIMENSION_ORDER[item[0]]))

    def _strict_unavailable(self, boundary: BudgetEnforcementBoundary) -> BudgetExhaustion | None:
        if self.limits.unavailable_metric_policy is not UnavailableMetricPolicy.STOP:
            return None
        for dimension in _DIMENSION_PRECEDENCE:
            unavailable = self._unavailable.get(dimension)
            if unavailable is None:
                continue
            limit, unit, currency = self._limit_descriptor(dimension)
            if limit is None:
                continue
            return BudgetExhaustion(
                dimension=dimension,
                reason=BudgetExhaustionReason.METRIC_UNAVAILABLE,
                limit=limit,
                observed=None,
                attempted_increment=None,
                overshoot=None,
                unit=unit,
                currency=currency,
                enforcement_boundary=boundary,
                unavailable_reason=unavailable.reason,
            )
        return None

    def _check_admission_limits(
        self,
        boundary: BudgetEnforcementBoundary,
        *,
        dimensions: tuple[BudgetDimension, ...],
    ) -> BudgetExhaustion | None:
        for dimension in _DIMENSION_PRECEDENCE:
            if dimension not in dimensions or dimension in self._unavailable:
                continue
            limit, unit, currency = self._limit_descriptor(dimension)
            observed = self._observed_value(dimension)
            if limit is None or observed is None or observed < limit:
                continue
            return BudgetExhaustion(
                dimension=dimension,
                reason=BudgetExhaustionReason.LIMIT_REACHED,
                limit=limit,
                observed=observed,
                attempted_increment=None,
                overshoot=max(0, observed - limit),
                unit=unit,
                currency=currency,
                enforcement_boundary=boundary,
            )
        return None

    def _check_exceeded_limits(
        self,
        boundary: BudgetEnforcementBoundary,
        *,
        dimensions: tuple[BudgetDimension, ...],
    ) -> BudgetExhaustion | None:
        for dimension in _DIMENSION_PRECEDENCE:
            if dimension not in dimensions or dimension in self._unavailable:
                continue
            limit, unit, currency = self._limit_descriptor(dimension)
            observed = self._observed_value(dimension)
            if limit is None or observed is None or observed <= limit:
                continue
            return BudgetExhaustion(
                dimension=dimension,
                reason=BudgetExhaustionReason.LIMIT_EXCEEDED,
                limit=limit,
                observed=observed,
                attempted_increment=None,
                overshoot=observed - limit,
                unit=unit,
                currency=currency,
                enforcement_boundary=boundary,
            )
        return None

    def _limit_descriptor(self, dimension: BudgetDimension) -> tuple[int | None, str, str | None]:
        if dimension is BudgetDimension.TOTAL_TOKENS:
            return self.limits.max_total_tokens, "tokens", None
        if dimension is BudgetDimension.UNCACHED_INPUT_TOKENS:
            return self.limits.max_uncached_input_tokens, "tokens", None
        if dimension is BudgetDimension.TOOL_CALLS:
            return self.limits.max_tool_calls, "calls", None
        if dimension is BudgetDimension.WALL_TIME:
            return self.limits.max_wall_time_ms, "milliseconds", None
        if dimension is BudgetDimension.HOST_COST:
            cost = self.limits.max_host_cost
            return (
                cost.amount_microunits if cost is not None else None,
                cost.unit if cost is not None else "host_cost",
                cost.currency if cost is not None else None,
            )
        return None, "calls", None

    def _observed_value(self, dimension: BudgetDimension) -> int | None:
        if dimension is BudgetDimension.TOTAL_TOKENS:
            return self._total_tokens
        if dimension is BudgetDimension.UNCACHED_INPUT_TOKENS:
            return self._uncached_input_tokens
        if dimension is BudgetDimension.TOOL_CALLS:
            return self._tool_calls
        if dimension is BudgetDimension.WALL_TIME:
            return self._elapsed_ms
        if dimension is BudgetDimension.HOST_COST:
            return self._host_cost.amount_microunits if self._host_cost is not None else None
        return None

    @staticmethod
    def _count_preflight_exhaustion(
        *,
        dimension: BudgetDimension,
        limit: int,
        observed: int,
        attempted_increment: int,
        boundary: BudgetEnforcementBoundary,
        tool_name: str | None = None,
    ) -> BudgetExhaustion:
        return BudgetExhaustion(
            dimension=dimension,
            tool_name=tool_name,
            reason=BudgetExhaustionReason.LIMIT_REACHED,
            limit=limit,
            observed=observed,
            attempted_increment=attempted_increment,
            overshoot=observed + attempted_increment - limit,
            unit="calls",
            enforcement_boundary=boundary,
        )


__all__ = [
    "MAX_WIRE_INTEGER",
    "BudgetDimension",
    "BudgetEnforcementBoundary",
    "BudgetEvaluator",
    "BudgetExhaustion",
    "BudgetExhaustionReason",
    "BudgetUnavailableDimension",
    "BudgetUnavailableReason",
    "BudgetUsageSnapshot",
    "HostCost",
    "HostCostMeter",
    "RunBudgetLimits",
    "UnavailableMetricPolicy",
]


class ModelCallBudgetExhausted(RuntimeError):
    vv_agent_control_flow = True

    def __init__(self, exhaustion: BudgetExhaustion) -> None:
        super().__init__("Run budget exhausted by a model call.")
        self.exhaustion = exhaustion
