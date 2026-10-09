from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vv_agent.agent import Agent

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


def apply_input_guardrails(*, agent: Agent, run_context: RunContext[Any], user_input: str) -> GuardrailResult:
    current_input = user_input
    for guardrail in agent.input_guardrails:
        result = guardrail(run_context, current_input)
        if result.outcome == "rewrite":
            current_input = str(result.value)
            continue
        if result.outcome != "allow":
            return result
    if current_input != user_input:
        return GuardrailResult.rewrite(current_input)
    return GuardrailResult.allow()


def apply_output_guardrails(*, agent: Agent, run_context: RunContext[Any], final_output: Any) -> GuardrailResult:
    current_output = final_output
    for guardrail in agent.output_guardrails:
        result = guardrail(run_context, current_output)
        if result.outcome == "rewrite":
            current_output = result.value
            continue
        if result.outcome != "allow":
            return result
    if current_output != final_output:
        return GuardrailResult.rewrite(current_output)
    return GuardrailResult.allow()
