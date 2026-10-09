"""Provider-owned evidence, independent of the session execution log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from vv_agent.runtime.cancellation import CancelledError
from vv_agent.tools.base import ToolContext
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.tools.outcomes import HostToolOutcome
from vv_agent.types import ToolCall, ToolExecutionResult, ToolResultStatus

from .records import InboxItem, Record


@dataclass(frozen=True)
class Definitive:
    result: dict[str, Any]
    evidence: tuple[str, ...] = ("synchronous-return",)
    usage: dict[str, Any] = field(default_factory=dict)
    shared_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Accepted:
    handle: dict[str, Any]


@dataclass(frozen=True)
class Unknown:
    reason: str


Outcome = Definitive | Accepted | Unknown


class Provider(Protocol):
    def submit(self, plan: Record, *, context: ToolContext) -> Outcome: ...
    def query(self, handle: dict[str, Any]) -> Outcome: ...
    def cancel(self, handle: dict[str, Any]) -> Outcome: ...
    def authenticate(self, item: InboxItem, plan: Record) -> bool: ...


class DispatchReady(BaseException):
    """Private preflight exit, deliberately outside the handler Exception catch."""


def _ready(_call: ToolCall) -> None:
    raise DispatchReady


class FunctionProvider:
    """The existing orchestrator owns validation, policy, approval and invocation."""

    def __init__(self, orchestrator: ToolOrchestrator):
        self.orchestrator = orchestrator

    def preflight(self, plan: Record, context: ToolContext) -> ToolExecutionResult | None:
        executor = self.orchestrator._resolve_executor(plan.payload["request"]["name"])
        if executor is not None and executor.metadata.get("policy_managed_by_handler"):
            return ToolExecutionResult(
                tool_call_id=plan.payload["request"]["id"],
                status_code=ToolResultStatus.ERROR,
                content="Handler-managed dispatch requires an explicit session provider adapter.",
                error_code="session_dispatch_boundary_required",
            )
        context.metadata["_vv_agent_tool_dispatch_callback"] = _ready
        try:
            result = self.orchestrator.run_one(
                ToolCall.from_dict(plan.payload["request"]),
                context=context,
                allowed_tool_names=context.metadata["session_tool_names"],
            )
        except DispatchReady:
            return None
        finally:
            context.metadata.pop("_vv_agent_tool_dispatch_callback", None)
        if not isinstance(result, ToolExecutionResult):
            raise ValueError("preflight handler bypassed the dispatch boundary")
        return result

    def submit(self, plan: Record, *, context: ToolContext) -> Outcome:
        def check(_call: ToolCall) -> None:
            assert context.ctx is not None
            context.ctx.check_cancelled()

        context.metadata["_vv_agent_tool_dispatch_callback"] = check
        try:
            result = self.orchestrator.run_one(
                ToolCall.from_dict(plan.payload["request"]),
                context=context,
                allowed_tool_names=context.metadata["session_tool_names"],
            )
        except CancelledError:
            return Definitive(
                ToolExecutionResult(
                    tool_call_id=plan.payload["request"]["id"],
                    content="Cooperative cancellation confirmed.",
                    status_code=ToolResultStatus.ERROR,
                    error_code="tool_cancelled",
                ).to_dict(),
                ("cooperative-stop",),
            )
        if isinstance(result, HostToolOutcome):
            return Unknown("host interaction outcome requires a session provider adapter")
        if result.error_code == "tool_timeout":
            return Unknown("handler timeout; thread may still be running")
        return Definitive(result.to_dict())

    def query(self, handle: dict[str, Any]) -> Outcome:
        return Unknown("ordinary synchronous handlers have no query handle")

    def cancel(self, handle: dict[str, Any]) -> Outcome:
        return Unknown("ordinary synchronous handlers have no provider cancellation handle")

    def authenticate(self, item: InboxItem, plan: Record) -> bool:
        return False  # Ordinary handlers cannot authenticate an external callback.
