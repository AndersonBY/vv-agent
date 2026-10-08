"""Closed tool-call outcomes returned by tool handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from vv_agent.canonical_json import MAX_WIRE_INTEGER, canonical_json_sha256, validate_sha256
from vv_agent.types import ToolDirective, ToolExecutionResult, ToolResultStatus

if TYPE_CHECKING:
    from vv_agent.interaction import HostInteractionRequest

DEFERRED_HANDLE_SCHEMA = "vv-agent.deferred-tool-handle.v2"


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


class DeferredWireError(ValueError):
    """Base error carrying the stable contract error code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class DeferredHandleError(DeferredWireError):
    pass


class DeferredResolutionError(DeferredWireError):
    pass


class DeferredResolutionResultInvalid(DeferredResolutionError):
    def __init__(
        self,
        message: str = "deferred resolution result is not definitive",
        *,
        code: str = "deferred_resolution_result_invalid",
    ) -> None:
        super().__init__(message, code=code)


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeferredHandleError(f"{field} must be a non-empty string", code="deferred_handle_invalid")
    return value


@dataclass(frozen=True, slots=True)
class DeferredToolHandle:
    """Exact identity of one admitted deferred tool invocation."""

    checkpoint_key: str
    operation_id: str
    attempt: int
    request_digest: str
    schema_version: str = DEFERRED_HANDLE_SCHEMA

    SCHEMA_VERSION: ClassVar[str] = DEFERRED_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DEFERRED_HANDLE_SCHEMA:
            raise DeferredHandleError(
                f"unsupported deferred handle schema: {self.schema_version!r}",
                code="deferred_handle_schema_unsupported",
            )
        _non_empty(self.checkpoint_key, "checkpoint_key")
        _non_empty(self.operation_id, "operation_id")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or not 1 <= self.attempt <= MAX_WIRE_INTEGER:
            raise DeferredHandleError("attempt must be a positive JSON-safe integer", code="deferred_handle_invalid")
        try:
            validate_sha256(self.request_digest, "request_digest")
        except ValueError as exc:
            raise DeferredHandleError(
                "request_digest must be a lowercase SHA-256 digest",
                code="deferred_handle_invalid",
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_key": self.checkpoint_key,
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "request_digest": self.request_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeferredToolHandle:
        if not isinstance(payload, Mapping):
            raise DeferredHandleError("deferred handle must be an object", code="deferred_handle_invalid")
        expected = {"schema_version", "checkpoint_key", "operation_id", "attempt", "request_digest"}
        if "schema_version" not in payload or payload.get("schema_version") != DEFERRED_HANDLE_SCHEMA:
            raise DeferredHandleError(
                "deferred handle schema discriminator is unsupported",
                code="deferred_handle_schema_unsupported",
            )
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise DeferredHandleError(
                f"deferred handle contains unknown field(s): {sorted(unknown)}",
                code="deferred_handle_unknown_field",
            )
        if missing:
            raise DeferredHandleError(
                f"deferred handle is missing field(s): {sorted(missing)}",
                code="deferred_handle_invalid",
            )
        try:
            return cls(
                checkpoint_key=payload["checkpoint_key"],
                operation_id=payload["operation_id"],
                attempt=payload["attempt"],
                request_digest=payload["request_digest"],
                schema_version=payload["schema_version"],
            )
        except DeferredWireError:
            raise
        except (TypeError, ValueError) as exc:
            raise DeferredHandleError("deferred handle fields are invalid", code="deferred_handle_invalid") from exc

    @property
    def key(self) -> str:
        return canonical_json_sha256(self.to_dict(), "deferred handle")

    def __hash__(self) -> int:
        return hash(self.key)


def validate_definitive_result(result: Any) -> None:
    if not isinstance(result, ToolExecutionResult) or result.status_code not in {
        ToolResultStatus.SUCCESS,
        ToolResultStatus.ERROR,
    }:
        raise DeferredResolutionResultInvalid()
    if result.status_code is ToolResultStatus.SUCCESS and result.error_code is not None:
        raise DeferredResolutionResultInvalid("tool_result_invalid", code="tool_result_invalid")
    if result.status_code is ToolResultStatus.ERROR and _is_ambiguous_tool_error(result):
        raise DeferredResolutionResultInvalid()


TOOL_CALL_OUTCOME_SCHEMA = "vv-agent.tool-call-outcome.v3"


@dataclass(frozen=True, slots=True)
class ToolCallOutcome:
    """Closed completed, deferred, or host-interaction tool outcome."""

    kind: str
    result: ToolExecutionResult | None = None
    handle: DeferredToolHandle | None = None
    request: HostInteractionRequest | None = None
    schema_version: str = TOOL_CALL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CALL_OUTCOME_SCHEMA:
            raise ValueError("tool_call_outcome_invalid: unsupported schema_version")
        if self.kind == "completed":
            if self.result is None or self.handle is not None or self.request is not None:
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
        elif self.kind == "deferred":
            if self.result is not None or self.request is not None or not isinstance(self.handle, DeferredToolHandle):
                raise ValueError("tool_call_outcome_invalid: deferred requires only handle")
        elif self.kind == "host_interaction":
            from vv_agent.interaction import HostInteractionRequest

            if self.handle is not None or not isinstance(self.request, HostInteractionRequest):
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
    def Completed(cls, result: ToolExecutionResult) -> ToolCallOutcome:
        return cls(kind="completed", result=result)

    @classmethod
    def completed(cls, result: ToolExecutionResult) -> ToolCallOutcome:
        return cls.Completed(result)

    @classmethod
    def Deferred(cls, handle: DeferredToolHandle) -> ToolCallOutcome:
        return cls(kind="deferred", handle=handle)

    @classmethod
    def deferred(cls, handle: DeferredToolHandle) -> ToolCallOutcome:
        return cls.Deferred(handle)

    @classmethod
    def HostInteraction(cls, result: ToolExecutionResult, request: HostInteractionRequest) -> ToolCallOutcome:
        return cls(kind="host_interaction", result=result, request=request)

    @property
    def is_deferred(self) -> bool:
        return self.kind == "deferred"

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
        assert self.handle is not None
        return {"schema_version": TOOL_CALL_OUTCOME_SCHEMA, "kind": "deferred", "handle": self.handle.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToolCallOutcome:
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
        if kind == "deferred":
            if set(payload) != {"schema_version", "kind", "handle"}:
                raise ValueError("tool_call_outcome_invalid: deferred fields are not closed")
            return cls.Deferred(DeferredToolHandle.from_dict(payload["handle"]))
        raise ValueError("tool_call_outcome_invalid: unknown outcome kind")
