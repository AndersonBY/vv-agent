from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OUTPUT_VALIDATION_FAILED = "output_validation_failed"


@dataclass(frozen=True, slots=True)
class OutputValidationContext:
    """Task-neutral context supplied to a host-owned output validator."""

    run_id: str
    agent_name: str
    output_type: Any | None = None


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    """Typed result returned by an explicitly registered output validator."""

    valid: bool
    code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("OutputValidationResult.valid must be a boolean")
        if self.valid:
            if self.code is not None or self.message is not None:
                raise ValueError("a valid output result cannot contain an error")
            return
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("an invalid output result requires a non-empty code")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("OutputValidationResult.message must be a string or None")

    @classmethod
    def accept(cls) -> OutputValidationResult:
        return cls(valid=True)

    @classmethod
    def reject(cls, code: str, message: str | None = None) -> OutputValidationResult:
        return cls(valid=False, code=code, message=message)


@dataclass(frozen=True, slots=True)
class OutputRepairRequest:
    """Tools-free request supplied to a host-owned repair callback."""

    invalid_output: Any
    validation_code: str
    validation_message: str | None
    model: Any | None = None
    model_settings: Any | None = None
    tools: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.validation_code, str) or not self.validation_code.strip():
            raise ValueError("validation_code cannot be empty")
        if self.validation_message is not None and not isinstance(self.validation_message, str):
            raise TypeError("validation_message must be a string or None")
        if self.tools != ():
            raise ValueError("output repair requests cannot contain tools")


OutputValidator = Callable[[Any, OutputValidationContext], OutputValidationResult]
OutputRepair = Callable[[OutputRepairRequest], Any]


def output_validator(func: OutputValidator) -> OutputValidator:
    """Mark a host callback as an output validator without changing it."""

    return func


def output_repair(func: OutputRepair) -> OutputRepair:
    """Mark a host callback as an output repair provider without changing it."""

    return func


__all__ = [
    "OUTPUT_VALIDATION_FAILED",
    "OutputRepair",
    "OutputRepairRequest",
    "OutputValidationContext",
    "OutputValidationResult",
    "OutputValidator",
    "output_repair",
    "output_validator",
]
