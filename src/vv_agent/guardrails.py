from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from vv_agent.agent import RunContext

GuardrailOutcome = Literal["allow", "block", "rewrite", "require_approval"]


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    outcome: GuardrailOutcome
    message: str | None = None
    value: Any | None = None

    @classmethod
    def allow(cls) -> GuardrailResult:
        return cls(outcome="allow")

    @classmethod
    def block(cls, message: str) -> GuardrailResult:
        return cls(outcome="block", message=message)

    @classmethod
    def rewrite(cls, value: Any) -> GuardrailResult:
        return cls(outcome="rewrite", value=value)

    @classmethod
    def require_approval(cls, message: str) -> GuardrailResult:
        return cls(outcome="require_approval", message=message)


InputGuardrail = Callable[[RunContext[Any], str], GuardrailResult]
OutputGuardrail = Callable[[RunContext[Any], Any], GuardrailResult]


def input_guardrail(func: InputGuardrail) -> InputGuardrail:
    return func


def output_guardrail(func: OutputGuardrail) -> OutputGuardrail:
    return func
