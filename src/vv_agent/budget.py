from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, overload, runtime_checkable

MAX_WIRE_INTEGER = (1 << 53) - 1


class UnavailableMetricPolicy(StrEnum):
    CONTINUE_AND_MARK = "continue_and_mark"
    STOP = "stop"


class BudgetDimension(StrEnum):
    TOTAL_TOKENS = "total_tokens"
    UNCACHED_INPUT_TOKENS = "uncached_input_tokens"
    TOOL_CALLS = "tool_calls"
    TOOL_CALLS_BY_NAME = "tool_calls_by_name"
    WALL_TIME = "wall_time"
    HOST_COST = "host_cost"


class BudgetEnforcementBoundary(StrEnum):
    RUN_START = "run_start"
    CYCLE_START = "cycle_start"
    MODEL_CALL_COMPLETE = "model_call_complete"
    TOOL_BATCH_PREFLIGHT = "tool_batch_preflight"
    TOOL_BATCH_COMPLETE = "tool_batch_complete"
    TERMINAL = "terminal"


class BudgetExhaustionReason(StrEnum):
    LIMIT_REACHED = "limit_reached"
    LIMIT_EXCEEDED = "limit_exceeded"
    METRIC_UNAVAILABLE = "metric_unavailable"


class BudgetUnavailableReason(StrEnum):
    USAGE_MISSING = "usage_missing"
    METER_MISSING = "meter_missing"
    METER_UNAVAILABLE = "meter_unavailable"
    METER_ERROR = "meter_error"
    UNIT_MISMATCH = "unit_mismatch"
    CURRENCY_MISMATCH = "currency_mismatch"
    NON_MONOTONIC = "non_monotonic"
    INTEGER_OVERFLOW = "integer_overflow"


_DIMENSION_PRECEDENCE = (
    BudgetDimension.WALL_TIME,
    BudgetDimension.TOTAL_TOKENS,
    BudgetDimension.UNCACHED_INPUT_TOKENS,
    BudgetDimension.HOST_COST,
    BudgetDimension.TOOL_CALLS,
    BudgetDimension.TOOL_CALLS_BY_NAME,
)
_DIMENSION_ORDER = {dimension: index for index, dimension in enumerate(_DIMENSION_PRECEDENCE)}


@overload
def _wire_integer(value: Any, field_name: str, *, nullable: Literal[False] = False) -> int: ...


@overload
def _wire_integer(value: Any, field_name: str, *, nullable: Literal[True]) -> int | None: ...


def _wire_integer(value: Any, field_name: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WIRE_INTEGER:
        suffix = " or None" if nullable else ""
        raise ValueError(f"{field_name} must be between 0 and {MAX_WIRE_INTEGER}{suffix}")
    return value


@overload
def _non_empty_string(value: Any, field_name: str, *, nullable: Literal[False] = False) -> str: ...


@overload
def _non_empty_string(value: Any, field_name: str, *, nullable: Literal[True]) -> str | None: ...


def _non_empty_string(value: Any, field_name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if nullable else ""
        raise ValueError(f"{field_name} must be a non-empty string{suffix}")
    return value


@dataclass(frozen=True, slots=True)
class HostCost:
    unit: str
    amount_microunits: int
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "host cost unit"))
        object.__setattr__(
            self,
            "amount_microunits",
            _wire_integer(self.amount_microunits, "host cost amount_microunits"),
        )
        object.__setattr__(
            self,
            "currency",
            _non_empty_string(self.currency, "host cost currency", nullable=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "currency": self.currency,
            "amount_microunits": self.amount_microunits,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HostCost:
        if not isinstance(data, Mapping):
            raise ValueError("host cost must be an object")
        return cls(
            unit=_non_empty_string(data.get("unit"), "host cost unit"),
            currency=data.get("currency"),
            amount_microunits=_wire_integer(
                data.get("amount_microunits"),
                "host cost amount_microunits",
            ),
        )


@runtime_checkable
class HostCostMeter(Protocol):
    def read(self) -> HostCost | None: ...


@dataclass(frozen=True, slots=True)
class BudgetUnavailableDimension:
    dimension: BudgetDimension
    reason: BudgetUnavailableReason
    expected_unit: str | None = None
    observed_unit: str | None = None
    expected_currency: str | None = None
    observed_currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", BudgetDimension(self.dimension))
        object.__setattr__(self, "reason", BudgetUnavailableReason(self.reason))
        for field_name in (
            "expected_unit",
            "observed_unit",
            "expected_currency",
            "observed_currency",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty_string(getattr(self, field_name), field_name, nullable=True),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "reason": self.reason.value,
            "expected_unit": self.expected_unit,
            "observed_unit": self.observed_unit,
            "expected_currency": self.expected_currency,
            "observed_currency": self.observed_currency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BudgetUnavailableDimension:
        if not isinstance(data, Mapping):
            raise ValueError("budget unavailable dimension must be an object")
        return cls(
            dimension=BudgetDimension(data.get("dimension")),
            reason=BudgetUnavailableReason(data.get("reason")),
            expected_unit=data.get("expected_unit"),
            observed_unit=data.get("observed_unit"),
            expected_currency=data.get("expected_currency"),
            observed_currency=data.get("observed_currency"),
        )


@dataclass(frozen=True, slots=True)
class RunBudgetLimits:
    max_total_tokens: int | None = None
    max_uncached_input_tokens: int | None = None
    max_tool_calls: int | None = None
    max_tool_calls_by_name: Mapping[str, int] = field(default_factory=dict)
    max_wall_time_ms: int | None = None
    max_host_cost: HostCost | None = None
    unavailable_metric_policy: UnavailableMetricPolicy = UnavailableMetricPolicy.CONTINUE_AND_MARK

    def __post_init__(self) -> None:
        for field_name in (
            "max_total_tokens",
            "max_uncached_input_tokens",
            "max_tool_calls",
            "max_wall_time_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _wire_integer(getattr(self, field_name), field_name, nullable=True),
            )
        if not isinstance(self.max_tool_calls_by_name, Mapping):
            raise ValueError("max_tool_calls_by_name must be an object")
        named: dict[str, int] = {}
        for name, value in self.max_tool_calls_by_name.items():
            normalized_name = _non_empty_string(name, "named tool budget key")
            named[normalized_name] = _wire_integer(value, f"max_tool_calls_by_name[{normalized_name!r}]")
        object.__setattr__(self, "max_tool_calls_by_name", dict(sorted(named.items())))
        if self.max_host_cost is not None and not isinstance(self.max_host_cost, HostCost):
            if not isinstance(self.max_host_cost, Mapping):
                raise ValueError("max_host_cost must be a HostCost, object, or None")
            object.__setattr__(self, "max_host_cost", HostCost.from_dict(self.max_host_cost))
        object.__setattr__(
            self,
            "unavailable_metric_policy",
            UnavailableMetricPolicy(self.unavailable_metric_policy),
        )

    @property
    def has_limits(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_total_tokens,
                self.max_uncached_input_tokens,
                self.max_tool_calls,
                self.max_wall_time_ms,
                self.max_host_cost,
            )
        ) or bool(self.max_tool_calls_by_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_uncached_input_tokens": self.max_uncached_input_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_tool_calls_by_name": dict(self.max_tool_calls_by_name),
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_host_cost": self.max_host_cost.to_dict() if self.max_host_cost is not None else None,
            "unavailable_metric_policy": self.unavailable_metric_policy.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunBudgetLimits:
        if not isinstance(data, Mapping):
            raise ValueError("run budget limits must be an object")
        host_cost = data.get("max_host_cost")
        return cls(
            max_total_tokens=data.get("max_total_tokens"),
            max_uncached_input_tokens=data.get("max_uncached_input_tokens"),
            max_tool_calls=data.get("max_tool_calls"),
            max_tool_calls_by_name=data.get("max_tool_calls_by_name", {}),
            max_wall_time_ms=data.get("max_wall_time_ms"),
            max_host_cost=HostCost.from_dict(host_cost) if host_cost is not None else None,
            unavailable_metric_policy=data.get(
                "unavailable_metric_policy",
                UnavailableMetricPolicy.CONTINUE_AND_MARK.value,
            ),
        )


@dataclass(frozen=True, slots=True)
class BudgetUsageSnapshot:
    cycles: int = 0
    total_tokens: int | None = 0
    uncached_input_tokens: int | None = 0
    tool_calls: int = 0
    tool_calls_by_name: Mapping[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    host_cost: HostCost | None = None
    unavailable_dimensions: tuple[BudgetUnavailableDimension, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycles", _wire_integer(self.cycles, "budget usage cycles"))
        object.__setattr__(
            self,
            "total_tokens",
            _wire_integer(self.total_tokens, "budget usage total_tokens", nullable=True),
        )
        object.__setattr__(
            self,
            "uncached_input_tokens",
            _wire_integer(self.uncached_input_tokens, "budget usage uncached_input_tokens", nullable=True),
        )
        object.__setattr__(self, "tool_calls", _wire_integer(self.tool_calls, "budget usage tool_calls"))
        object.__setattr__(self, "elapsed_ms", _wire_integer(self.elapsed_ms, "budget usage elapsed_ms"))
        if not isinstance(self.tool_calls_by_name, Mapping):
            raise ValueError("budget usage tool_calls_by_name must be an object")
        named: dict[str, int] = {}
        for name, value in self.tool_calls_by_name.items():
            normalized_name = _non_empty_string(name, "budget usage tool name")
            named[normalized_name] = _wire_integer(value, f"budget usage tool_calls_by_name[{normalized_name!r}]")
        object.__setattr__(self, "tool_calls_by_name", dict(sorted(named.items())))
        if self.host_cost is not None and not isinstance(self.host_cost, HostCost):
            if not isinstance(self.host_cost, Mapping):
                raise ValueError("budget usage host_cost must be a HostCost, object, or None")
            object.__setattr__(self, "host_cost", HostCost.from_dict(self.host_cost))
        unavailable = tuple(
            item if isinstance(item, BudgetUnavailableDimension) else BudgetUnavailableDimension.from_dict(item)
            for item in self.unavailable_dimensions
        )
        dimensions = [item.dimension for item in unavailable]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("budget unavailable dimensions must be unique")
        object.__setattr__(
            self,
            "unavailable_dimensions",
            tuple(sorted(unavailable, key=lambda item: _DIMENSION_ORDER[item.dimension])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "total_tokens": self.total_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "elapsed_ms": self.elapsed_ms,
            "host_cost": self.host_cost.to_dict() if self.host_cost is not None else None,
            "unavailable_dimensions": [item.to_dict() for item in self.unavailable_dimensions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BudgetUsageSnapshot:
        if not isinstance(data, Mapping):
            raise ValueError("budget usage must be an object")
        unavailable = data.get("unavailable_dimensions", [])
        if not isinstance(unavailable, list):
            raise ValueError("budget unavailable_dimensions must be an array")
        tool_calls_by_name = data.get("tool_calls_by_name")
        if not isinstance(tool_calls_by_name, Mapping):
            raise ValueError("budget usage tool_calls_by_name must be an object")
        host_cost = data.get("host_cost")
        return cls(
            cycles=_wire_integer(data.get("cycles"), "budget usage cycles"),
            total_tokens=data.get("total_tokens"),
            uncached_input_tokens=data.get("uncached_input_tokens"),
            tool_calls=_wire_integer(data.get("tool_calls"), "budget usage tool_calls"),
            tool_calls_by_name=tool_calls_by_name,
            elapsed_ms=_wire_integer(data.get("elapsed_ms"), "budget usage elapsed_ms"),
            host_cost=HostCost.from_dict(host_cost) if host_cost is not None else None,
            unavailable_dimensions=tuple(BudgetUnavailableDimension.from_dict(item) for item in unavailable),
        )


@dataclass(frozen=True, slots=True)
class BudgetExhaustion:
    dimension: BudgetDimension
    reason: BudgetExhaustionReason
    limit: int
    observed: int | None
    attempted_increment: int | None
    overshoot: int | None
    unit: str
    enforcement_boundary: BudgetEnforcementBoundary
    tool_name: str | None = None
    currency: str | None = None
    unavailable_reason: BudgetUnavailableReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", BudgetDimension(self.dimension))
        object.__setattr__(self, "reason", BudgetExhaustionReason(self.reason))
        object.__setattr__(self, "limit", _wire_integer(self.limit, "budget exhaustion limit"))
        for field_name in ("observed", "attempted_increment", "overshoot"):
            object.__setattr__(
                self,
                field_name,
                _wire_integer(getattr(self, field_name), f"budget exhaustion {field_name}", nullable=True),
            )
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "budget exhaustion unit"))
        object.__setattr__(self, "tool_name", _non_empty_string(self.tool_name, "tool_name", nullable=True))
        object.__setattr__(self, "currency", _non_empty_string(self.currency, "currency", nullable=True))
        object.__setattr__(self, "enforcement_boundary", BudgetEnforcementBoundary(self.enforcement_boundary))
        if self.unavailable_reason is not None:
            object.__setattr__(self, "unavailable_reason", BudgetUnavailableReason(self.unavailable_reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "tool_name": self.tool_name,
            "reason": self.reason.value,
            "limit": self.limit,
            "observed": self.observed,
            "attempted_increment": self.attempted_increment,
            "overshoot": self.overshoot,
            "unit": self.unit,
            "currency": self.currency,
            "enforcement_boundary": self.enforcement_boundary.value,
            "unavailable_reason": self.unavailable_reason.value if self.unavailable_reason is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BudgetExhaustion:
        if not isinstance(data, Mapping):
            raise ValueError("budget exhaustion must be an object")
        unavailable_reason = data.get("unavailable_reason")
        return cls(
            dimension=BudgetDimension(data.get("dimension")),
            tool_name=data.get("tool_name"),
            reason=BudgetExhaustionReason(data.get("reason")),
            limit=_wire_integer(data.get("limit"), "budget exhaustion limit"),
            observed=data.get("observed"),
            attempted_increment=data.get("attempted_increment"),
            overshoot=data.get("overshoot"),
            unit=_non_empty_string(data.get("unit"), "budget exhaustion unit"),
            currency=data.get("currency"),
            enforcement_boundary=BudgetEnforcementBoundary(data.get("enforcement_boundary")),
            unavailable_reason=(BudgetUnavailableReason(unavailable_reason) if unavailable_reason is not None else None),
        )


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
        if exhaustion is None:
            self._cycles += 1
        return exhaustion

    def model_call_complete(self, token_usage: Any) -> BudgetExhaustion | None:
        boundary = BudgetEnforcementBoundary.MODEL_CALL_COMPLETE
        self._observe_token_usage(token_usage)
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

    def _observe_token_usage(self, usage: Any) -> None:
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
        self._unavailable = dict(
            sorted(self._unavailable.items(), key=lambda item: _DIMENSION_ORDER[item[0]])
        )

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
