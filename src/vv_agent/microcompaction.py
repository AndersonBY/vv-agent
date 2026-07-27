from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

_POLICY_FIELDS = frozenset(
    {
        "trigger_ratio",
        "target_ratio",
        "keep_recent_cycles",
        "min_result_chars",
    }
)
_MAX_U32 = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class MicrocompactionPolicy:
    trigger_ratio: float = 0.75
    target_ratio: float = 0.60
    keep_recent_cycles: int = 3
    min_result_chars: int = 500

    def __post_init__(self) -> None:
        trigger_ratio = _ratio(self.trigger_ratio, "trigger_ratio")
        target_ratio = _ratio(self.target_ratio, "target_ratio")
        if not target_ratio < trigger_ratio:
            raise ValueError("target_ratio must be less than trigger_ratio")
        object.__setattr__(self, "trigger_ratio", trigger_ratio)
        object.__setattr__(self, "target_ratio", target_ratio)
        object.__setattr__(
            self,
            "keep_recent_cycles",
            _bounded_integer(self.keep_recent_cycles, "keep_recent_cycles", minimum=0),
        )
        object.__setattr__(
            self,
            "min_result_chars",
            _bounded_integer(self.min_result_chars, "min_result_chars", minimum=1),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "trigger_ratio": self.trigger_ratio,
            "target_ratio": self.target_ratio,
            "keep_recent_cycles": self.keep_recent_cycles,
            "min_result_chars": self.min_result_chars,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MicrocompactionPolicy:
        if not isinstance(value, Mapping):
            raise TypeError("microcompaction policy must be an object")
        if set(value) != _POLICY_FIELDS:
            raise ValueError("microcompaction policy fields do not match the current shape")
        return cls(
            trigger_ratio=value["trigger_ratio"],
            target_ratio=value["target_ratio"],
            keep_recent_cycles=value["keep_recent_cycles"],
            min_result_chars=value["min_result_chars"],
        )


def normalize_microcompaction_policy(
    value: MicrocompactionPolicy | Mapping[str, Any] | None,
) -> MicrocompactionPolicy:
    if value is None:
        return MicrocompactionPolicy()
    if isinstance(value, MicrocompactionPolicy):
        return MicrocompactionPolicy.from_dict(value.to_dict())
    return MicrocompactionPolicy.from_dict(value)


def _ratio(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or not 0 < normalized <= 1:
        raise ValueError(f"{field_name} must be greater than 0 and at most 1")
    return normalized


def _bounded_integer(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if not minimum <= value <= _MAX_U32:
        raise ValueError(f"{field_name} must be between {minimum} and {_MAX_U32}")
    return value


__all__ = [
    "MicrocompactionPolicy",
    "normalize_microcompaction_policy",
]
