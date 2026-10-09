"""Closed tool-call outcomes returned by tool handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vv_agent.types import ToolDirective, ToolExecutionResult, ToolResultStatus

if TYPE_CHECKING:
    from vv_agent.interaction import HostInteractionRequest


def _is_ambiguous_tool_error(result: Any) -> bool:
    """Return whether a tool error lacks an adapter-proven definitive outcome."""
    return (
        isinstance(result, ToolExecutionResult)
        and result.status_code is ToolResultStatus.ERROR
        and result.error_code
        in {
            "tool_timeout",
            "tool_cancelled",
            "tool_connection_lost",
            "tool_execution_failed",
            "tool_orchestrator_error",
        }
        and result.metadata.get("definitive_outcome") is not True
    )


class DefinitiveResultInvalid(ValueError):
    def __init__(
        self,
        message: str = "deferred resolution result is not definitive",
        *,
        code: str = "deferred_resolution_result_invalid",
    ) -> None:
        super().__init__(message)
        self.code = code


def validate_definitive_result(result: Any) -> None:
    if not isinstance(result, ToolExecutionResult) or result.status_code not in {
        ToolResultStatus.SUCCESS,
        ToolResultStatus.ERROR,
    }:
        raise DefinitiveResultInvalid()
    if result.status_code is ToolResultStatus.SUCCESS and result.error_code is not None:
        raise DefinitiveResultInvalid("tool_result_invalid", code="tool_result_invalid")
    if result.status_code is ToolResultStatus.ERROR and _is_ambiguous_tool_error(result):
        raise DefinitiveResultInvalid()


TOOL_CALL_OUTCOME_SCHEMA = "vv-agent.tool-call-outcome.v3"


@dataclass(frozen=True, slots=True)
class HostToolOutcome:
    """Completed result or a validated request for host interaction."""

    kind: str
    result: ToolExecutionResult | None = None
    request: HostInteractionRequest | None = None
    schema_version: str = TOOL_CALL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CALL_OUTCOME_SCHEMA:
            raise ValueError("tool_call_outcome_invalid: unsupported schema_version")
        if self.kind == "completed":
            if self.result is None or self.request is not None:
                raise ValueError("tool_call_outcome_invalid: completed requires only result")
            from vv_agent.types import ToolExecutionResult, ToolResultStatus

            if not isinstance(self.result, ToolExecutionResult) or self.result.status_code not in {
                ToolResultStatus.SUCCESS,
                ToolResultStatus.ERROR,
                ToolResultStatus.WAIT_RESPONSE,
                ToolResultStatus.RUNNING,
                ToolResultStatus.PENDING_COMPRESS,
            }:
                raise ValueError("tool_call_outcome_invalid: completed result is invalid")
        elif self.kind == "host_interaction":
            from vv_agent.interaction import HostInteractionRequest

            if not isinstance(self.request, HostInteractionRequest):
                raise ValueError("tool_call_outcome_invalid: host interaction requires result and request")
            try:
                validate_definitive_result(self.result)
            except ValueError as exc:
                raise ValueError("tool_call_outcome_invalid: host interaction requires a definitive result") from exc
            assert self.result is not None
            if self.result.directive is not ToolDirective.CONTINUE or self.result.tool_call_id != self.request.tool_call_id:
                raise ValueError("tool_call_outcome_invalid: host interaction result identity or directive is invalid")
        else:
            raise ValueError("tool_call_outcome_invalid: unknown outcome kind")

    @classmethod
    def Completed(cls, result: ToolExecutionResult) -> HostToolOutcome:
        return cls(kind="completed", result=result)

    @classmethod
    def completed(cls, result: ToolExecutionResult) -> HostToolOutcome:
        return cls.Completed(result)

    @classmethod
    def HostInteraction(cls, result: ToolExecutionResult, request: HostInteractionRequest) -> HostToolOutcome:
        return cls(kind="host_interaction", result=result, request=request)

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "host_interaction":
            assert self.result is not None and self.request is not None
            return {
                "schema_version": TOOL_CALL_OUTCOME_SCHEMA,
                "kind": "host_interaction",
                "result": self.result.to_dict(),
                "request": self.request.to_dict(),
            }
        if self.kind == "completed":
            result = self.result
            assert result is not None
            return {"schema_version": TOOL_CALL_OUTCOME_SCHEMA, "kind": "completed", "result": result.to_dict()}
        raise AssertionError("validated host outcome kind")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HostToolOutcome:
        if not isinstance(payload, Mapping):
            raise ValueError("tool_call_outcome_invalid: payload must be an object")
        if payload.get("schema_version") != TOOL_CALL_OUTCOME_SCHEMA:
            raise ValueError("tool_call_outcome_invalid: unsupported schema_version")
        kind = payload.get("kind")
        if kind == "host_interaction":
            if set(payload) != {"schema_version", "kind", "result", "request"}:
                raise ValueError("tool_call_outcome_invalid: host interaction fields are not closed")
            from vv_agent.interaction import HostInteractionRequest

            return cls.HostInteraction(
                ToolExecutionResult.from_dict(payload["result"]),
                HostInteractionRequest.from_dict(payload["request"]),
            )
        if kind == "completed":
            if set(payload) != {"schema_version", "kind", "result"}:
                raise ValueError("tool_call_outcome_invalid: completed fields are not closed")
            return cls.Completed(ToolExecutionResult.from_dict(payload["result"]))
        raise ValueError("tool_call_outcome_invalid: unknown outcome kind")
