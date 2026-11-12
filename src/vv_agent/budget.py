from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import Any

from vv_agent.kernel._budget import BudgetObservation, CoreBudgetEvaluator, HostCostObservation, _BudgetStopIteration
from vv_agent.kernel._budget_values import (
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


class BudgetEvaluator(CoreBudgetEvaluator):
    def __init__(
        self,
        limits: RunBudgetLimits,
        *,
        host_cost_meter: HostCostMeter | None = None,
        initial_usage: BudgetUsageSnapshot | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._host_cost_meter = host_cost_meter
        self._clock_ns = clock_ns
        self._drain(self._initialize_steps(limits, initial_usage=initial_usage))

    def _drain[T](self, steps: Generator[BudgetObservation, Any, T]) -> T:
        response = None
        try:
            while True:
                try:
                    request = steps.send(response)
                except StopIteration as done:
                    return done.value
                except _BudgetStopIteration as transport:
                    raise transport.original from transport.original.__cause__
                if request is BudgetObservation.CLOCK_NS:
                    response = self._clock_ns()
                elif self._host_cost_meter is None:
                    response = HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_MISSING)
                else:
                    try:
                        reading = self._host_cost_meter.read()
                    except Exception:
                        response = HostCostObservation(unavailable_reason=BudgetUnavailableReason.METER_ERROR)
                    else:
                        response = HostCostObservation(reading=reading)
        finally:
            steps.close()

    def run_start(self) -> BudgetExhaustion | None:
        return self._drain(self.run_start_steps())

    def cycle_start(self) -> BudgetExhaustion | None:
        return self._drain(self.cycle_start_steps())

    def model_call_start(self) -> BudgetExhaustion | None:
        return self._drain(self.model_call_start_steps())

    def model_call_complete(self, token_usage: Any) -> BudgetExhaustion | None:
        return self._drain(self.model_call_complete_steps(token_usage))

    def preflight_tools(self, tool_names: list[str]) -> BudgetExhaustion | None:
        return self._drain(self.preflight_tools_steps(tool_names))

    def tool_batch_complete(self, *, operation_failed: bool = False) -> BudgetExhaustion | None:
        return self._drain(self.tool_batch_complete_steps(operation_failed=operation_failed))

    def terminal(self) -> BudgetExhaustion | None:
        return self._drain(self.terminal_steps())


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
