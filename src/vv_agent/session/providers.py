"""Provider-owned evidence, independent of the session execution log."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.kernel import _driver_dispatch as dispatch
from vv_agent.kernel._provider_outcomes import Accepted as Accepted
from vv_agent.kernel._provider_outcomes import Definitive as Definitive
from vv_agent.kernel._provider_outcomes import Outcome as Outcome
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid as ProviderResultInvalid
from vv_agent.kernel._provider_outcomes import Unknown as Unknown
from vv_agent.kernel._provider_outcomes import validate_acceptance as validate_acceptance
from vv_agent.kernel._provider_outcomes import validate_lookup_provider_result
from vv_agent.kernel._provider_outcomes import validate_provider_result as validate_provider_result
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.tools.base import ToolContext
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.tools.outcomes import HostToolOutcome
from vv_agent.types import ToolCall, ToolExecutionResult, ToolResultStatus

from .records import InboxItem, Record

if TYPE_CHECKING:
    from .environments import LookupBinding, ProviderLookupResult


class Provider(Protocol):
    def submit(self, plan: Record, *, context: ToolContext) -> Outcome: ...
    def query(self, handle: dict[str, Any]) -> Outcome: ...
    def cancel(self, handle: dict[str, Any]) -> Outcome: ...
    def authenticate(self, item: InboxItem, plan: Record) -> bool: ...


class ProviderLookupPort(Protocol):
    """Authenticated recovery lookup for providers with durable executions."""

    def lookup(self, execution_id: str, binding: LookupBinding) -> ProviderLookupResult: ...


LookupProvider = ProviderLookupPort


def _validate_lookup_call(execution_id: str, binding: LookupBinding) -> None:
    from .environments import InvalidProviderLookupResult, LookupBinding

    if not isinstance(binding, LookupBinding):
        raise InvalidProviderLookupResult("lookup requires a complete LookupBinding")
    if not isinstance(execution_id, str) or execution_id != binding.execution_id:
        raise InvalidProviderLookupResult("lookup execution_id does not match the authenticated binding")


def lookup_provider(
    provider: object,
    execution_id: str,
    binding: LookupBinding,
    *,
    feature_level: int = 1,
) -> ProviderLookupResult:
    """Call a provider lookup capability and bind its evidence to the caller's lookup request."""
    from .environments import InvalidLookupBinding, InvalidProviderLookupResult, ProviderLookupResult

    _validate_lookup_call(execution_id, binding)
    lookup = getattr(provider, "lookup", None)
    if not callable(lookup):
        raise InvalidProviderLookupResult("provider does not support authenticated lookup")
    result = lookup(execution_id, binding)
    if not isinstance(result, ProviderLookupResult):
        raise InvalidProviderLookupResult("provider lookup must return ProviderLookupResult")
    try:
        result.assert_matches(binding)
    except InvalidLookupBinding as exc:
        raise InvalidProviderLookupResult("provider lookup result does not match the authenticated binding") from exc
    try:
        validate_lookup_provider_result(
            result.binding,
            result.status.value,
            result.outcome,
            result.evidence,
            feature_level=feature_level,
        )
    except ProviderResultInvalid as exc:
        raise InvalidProviderLookupResult("provider lookup result is invalid") from exc
    if result.feature_level != feature_level:
        raise InvalidProviderLookupResult("provider lookup feature level does not match the caller")
    return result


class DispatchReady(BaseException):
    """Private preflight exit, deliberately outside the handler Exception catch."""


def _ready(_call: ToolCall) -> None:
    raise DispatchReady


class FunctionProvider:
    """Core admission owns decisions; the orchestrator supplies concrete tool leaves."""

    def __init__(self, orchestrator: ToolOrchestrator):
        self.orchestrator = orchestrator

    def preflight(self, plan: Record, context: ToolContext) -> ToolExecutionResult | None:
        from ._dispatch_tools import PreflightHost

        return PreflightHost(self.orchestrator, context).run(dispatch.preflight(plan))

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
                    metadata={"definitive_outcome": True},
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

    def lookup(self, execution_id: str, binding: LookupBinding, *, feature_level: int = 1) -> ProviderLookupResult:
        from .environments import LookupStatus, ProviderLookupResult

        _validate_lookup_call(execution_id, binding)
        return ProviderLookupResult(
            binding,
            LookupStatus.UNKNOWN,
            Unknown("synchronous function provider has no persistent execution registry"),
            ("function-provider-no-registry",),
            feature_level=feature_level,
        )
