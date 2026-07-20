from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from vv_agent.budget import (
    BudgetDimension,
    BudgetEnforcementBoundary,
    BudgetExhaustion,
    BudgetExhaustionReason,
    BudgetUsageSnapshot,
)
from vv_agent.checkpoint import (
    OperationKind,
    OperationState,
    ReconciliationDecisionKind,
    ResumeObservation,
    ToolIdempotency,
)
from vv_agent.types import CompletionReason

if TYPE_CHECKING:
    from vv_agent.tools.metadata import ToolMetadata

RUN_EVENT_VERSION = "v1"
ApprovalAction = Literal["allow", "allow_session", "deny", "timeout"]
_APPROVAL_ACTIONS = frozenset({"allow", "allow_session", "deny", "timeout"})
_JSON_SAFE_INTEGER_MAX = (1 << 53) - 1
_TOOL_STATUS_VALUES = frozenset({"success", "error", "wait_response", "running", "pending_compress"})
_TOOL_DIRECTIVE_VALUES = frozenset({"continue", "finish", "wait_user"})
_TOOL_COMPLETED_ADDITIVE_FIELDS = frozenset(
    {"directive", "error_code", "execution_started", "duration_ms"}
)


class _MissingToolLifecycleField:
    __slots__ = ()


_MISSING_TOOL_LIFECYCLE_FIELD = _MissingToolLifecycleField()


class _StreamEventCommon(TypedDict):
    run_id: str
    trace_id: str
    agent_name: str | None
    session_id: str | None
    parent_run_id: str | None
    cycle_index: int
    metadata: dict[str, Any] | None


def _stream_delta(value: Any, field_name: str = "delta") -> str:
    if not isinstance(value, str):
        raise ValueError(f"Run event {field_name} must be a string")
    return value


def _optional_stream_counter(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _JSON_SAFE_INTEGER_MAX:
        raise ValueError(f"Run event {field_name} must be a non-negative JSON-safe integer or null")
    return value


def _completion_reason(value: Any) -> CompletionReason | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Run event completion_reason must be a string or null")
    try:
        return CompletionReason(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported run event completion_reason: {value!r}") from exc


def _completion_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Run event {field_name} must be a string or null")
    return value


def _budget_usage(value: Any) -> BudgetUsageSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Run event budget_usage must be an object or null")
    return BudgetUsageSnapshot.from_dict(value)


def _budget_exhaustion(value: Any) -> BudgetExhaustion | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Run event budget_exhaustion must be an object or null")
    return BudgetExhaustion.from_dict(value)


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def event_created_at() -> float:
    return time.time()


def _canonical_tool_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "success"
    if normalized not in _TOOL_STATUS_VALUES:
        raise ValueError(f"Unsupported tool event status: {status!r}")
    return normalized


def _wire_tool_status(value: Any) -> str:
    if not isinstance(value, str) or value not in _TOOL_STATUS_VALUES:
        raise ValueError(f"Unsupported tool event status: {value!r}")
    return value


def _tool_directive(value: Any) -> str:
    if not isinstance(value, str) or value not in _TOOL_DIRECTIVE_VALUES:
        raise ValueError(f"Unsupported tool event directive: {value!r}")
    return value


def _tool_error_code(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("Run event error_code must be a string or null")
    return value


def _tool_execution_started(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Run event execution_started must be a boolean")
    return value


def _tool_duration_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _JSON_SAFE_INTEGER_MAX:
        raise ValueError("Run event duration_ms must be a non-negative JSON-safe integer or null")
    return value


def _event_tool_metadata(value: ToolMetadata | dict[str, Any] | None) -> ToolMetadata | None:
    if value is None:
        return None
    from vv_agent.tools.metadata import ToolMetadata

    if isinstance(value, ToolMetadata):
        source = value.to_dict()
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError("Run event tool_metadata must be an object")
    try:
        return ToolMetadata.from_dict(source)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid run event tool_metadata: {exc}") from exc


def _canonical_approval_action(action: Any) -> ApprovalAction | None:
    if action is None:
        return None
    normalized = str(action).strip().lower()
    if normalized not in _APPROVAL_ACTIONS:
        raise ValueError(f"Unsupported approval action: {action!r}")
    return cast(ApprovalAction, normalized)


@dataclass(frozen=True, slots=True)
class RunEvent:
    type: str
    run_id: str
    trace_id: str
    version: str = RUN_EVENT_VERSION
    event_id: str = field(default_factory=new_event_id)
    session_id: str | None = None
    parent_event_id: str | None = None
    parent_run_id: str | None = None
    created_at: float = field(default_factory=event_created_at)
    cycle_index: int | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "type": self.type,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.parent_event_id:
            payload["parent_event_id"] = self.parent_event_id
        if self.parent_run_id:
            payload["parent_run_id"] = self.parent_run_id
        if self.cycle_index is not None:
            payload["cycle_index"] = self.cycle_index
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _set_run_event_fields(
    event: RunEvent,
    *,
    type: str,
    run_id: str,
    trace_id: str,
    cycle_index: int | None = None,
    agent_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    parent_event_id: str | None = None,
    parent_run_id: str | None = None,
    event_id: str | None = None,
    created_at: float | None = None,
) -> None:
    object.__setattr__(event, "type", type)
    object.__setattr__(event, "run_id", run_id)
    object.__setattr__(event, "trace_id", trace_id)
    object.__setattr__(event, "version", RUN_EVENT_VERSION)
    object.__setattr__(event, "event_id", event_id or new_event_id())
    object.__setattr__(event, "session_id", session_id)
    object.__setattr__(event, "parent_event_id", parent_event_id)
    object.__setattr__(event, "parent_run_id", parent_run_id)
    object.__setattr__(event, "created_at", event_created_at() if created_at is None else created_at)
    object.__setattr__(event, "cycle_index", cycle_index)
    object.__setattr__(event, "agent_name", agent_name)
    object.__setattr__(event, "metadata", dict(metadata or {}))


@dataclass(frozen=True, slots=True)
class RunStartedEvent(RunEvent):
    input: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        input: str,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="run_started",
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "input", input)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["input"] = self.input
        return payload


@dataclass(frozen=True, slots=True)
class AgentStartedEvent(RunEvent):
    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="agent_started",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class CycleStartedEvent(RunEvent):
    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="cycle_started",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class LLMStartedEvent(RunEvent):
    model: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        model: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="llm_started",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "model", model)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["model"] = self.model
        return payload


@dataclass(frozen=True, slots=True)
class RunStateChangedEvent(RunEvent):
    state: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        state: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="run_state_changed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["state"] = self.state
        return payload


@dataclass(frozen=True, slots=True)
class MemoryCompactedEvent(RunEvent):
    before_count: int | None = None
    after_count: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        before_count: int | None = None,
        after_count: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="memory_compacted",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "before_count", before_count)
        object.__setattr__(self, "after_count", after_count)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        if self.before_count is not None:
            payload["before_count"] = self.before_count
        if self.after_count is not None:
            payload["after_count"] = self.after_count
        return payload


@dataclass(frozen=True, slots=True)
class MemoryCompactStarted(RunEvent):
    message_count: int = 0
    estimated_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str = "",
        cycle_index: int | None = None,
        agent_name: str | None = None,
        message_count: int = 0,
        estimated_tokens: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="memory_compact_started",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "message_count", message_count)
        object.__setattr__(self, "estimated_tokens", estimated_tokens)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["message_count"] = self.message_count
        if self.estimated_tokens is not None:
            payload["estimated_tokens"] = self.estimated_tokens
        return payload


@dataclass(frozen=True, slots=True)
class MemoryCompactCompleted(RunEvent):
    before_count: int = 0
    after_count: int = 0
    summary_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str = "",
        cycle_index: int | None = None,
        agent_name: str | None = None,
        before_count: int = 0,
        after_count: int = 0,
        summary_tokens: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="memory_compact_completed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "before_count", before_count)
        object.__setattr__(self, "after_count", after_count)
        object.__setattr__(self, "summary_tokens", summary_tokens)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["before_count"] = self.before_count
        payload["after_count"] = self.after_count
        if self.summary_tokens is not None:
            payload["summary_tokens"] = self.summary_tokens
        return payload


@dataclass(frozen=True, slots=True)
class AssistantDeltaEvent(RunEvent):
    delta: str = ""
    content_chars: int | None = None
    estimated_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        delta: str,
        content_chars: int | None = None,
        estimated_tokens: int | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="assistant_delta",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "delta", _stream_delta(delta))
        object.__setattr__(self, "content_chars", _optional_stream_counter(content_chars, "content_chars"))
        object.__setattr__(self, "estimated_tokens", _optional_stream_counter(estimated_tokens, "estimated_tokens"))

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["delta"] = self.delta
        if self.content_chars is not None:
            payload["content_chars"] = self.content_chars
        if self.estimated_tokens is not None:
            payload["estimated_tokens"] = self.estimated_tokens
        return payload


@dataclass(frozen=True, slots=True)
class ReasoningDeltaEvent(RunEvent):
    delta: str = ""
    reasoning_chars: int | None = None
    estimated_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        delta: str,
        reasoning_chars: int | None = None,
        estimated_tokens: int | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="reasoning_delta",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "delta", _stream_delta(delta))
        object.__setattr__(self, "reasoning_chars", _optional_stream_counter(reasoning_chars, "reasoning_chars"))
        object.__setattr__(self, "estimated_tokens", _optional_stream_counter(estimated_tokens, "estimated_tokens"))

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["delta"] = self.delta
        if self.reasoning_chars is not None:
            payload["reasoning_chars"] = self.reasoning_chars
        if self.estimated_tokens is not None:
            payload["estimated_tokens"] = self.estimated_tokens
        return payload


def _set_model_tool_stream_fields(
    event: RunEvent,
    *,
    type: str,
    run_id: str,
    trace_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_call_index: int | None,
    arguments_chars: int | None,
    estimated_tokens: int | None,
    cycle_index: int | None,
    agent_name: str | None,
    session_id: str | None,
    parent_event_id: str | None,
    parent_run_id: str | None,
    event_id: str | None,
    created_at: float | None,
    metadata: dict[str, Any] | None,
) -> None:
    _set_run_event_fields(
        event,
        type=type,
        run_id=run_id,
        trace_id=trace_id,
        cycle_index=cycle_index,
        agent_name=agent_name,
        session_id=session_id,
        parent_event_id=parent_event_id,
        parent_run_id=parent_run_id,
        event_id=event_id,
        created_at=created_at,
        metadata=metadata,
    )
    object.__setattr__(event, "tool_call_id", _required_event_text(tool_call_id, "tool_call_id"))
    object.__setattr__(event, "tool_name", _required_event_text(tool_name, "tool_name"))
    object.__setattr__(event, "tool_call_index", _optional_stream_counter(tool_call_index, "tool_call_index"))
    object.__setattr__(event, "arguments_chars", _optional_stream_counter(arguments_chars, "arguments_chars"))
    object.__setattr__(event, "estimated_tokens", _optional_stream_counter(estimated_tokens, "estimated_tokens"))


def _model_tool_stream_dict(event: ModelToolCallStartedEvent | ModelToolCallProgressEvent) -> dict[str, Any]:
    payload = RunEvent.to_dict(event)
    payload["tool_call_id"] = event.tool_call_id
    if event.tool_call_index is not None:
        payload["tool_call_index"] = event.tool_call_index
    payload["tool_name"] = event.tool_name
    for field_name in ("arguments_chars", "estimated_tokens"):
        value = getattr(event, field_name)
        if value is not None:
            payload[field_name] = value
    return payload


@dataclass(frozen=True, slots=True)
class ModelToolCallStartedEvent(RunEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    tool_call_index: int | None = None
    arguments_chars: int | None = None
    estimated_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_call_index: int | None = None,
        arguments_chars: int | None = None,
        estimated_tokens: int | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_model_tool_stream_fields(
            self,
            type="model_tool_call_started",
            run_id=run_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_call_index=tool_call_index,
            arguments_chars=arguments_chars,
            estimated_tokens=estimated_tokens,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _model_tool_stream_dict(self)


@dataclass(frozen=True, slots=True)
class ModelToolCallProgressEvent(RunEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    tool_call_index: int | None = None
    arguments_chars: int | None = None
    estimated_tokens: int | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_call_index: int | None = None,
        arguments_chars: int | None = None,
        estimated_tokens: int | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_model_tool_stream_fields(
            self,
            type="model_tool_call_progress",
            run_id=run_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_call_index=tool_call_index,
            arguments_chars=arguments_chars,
            estimated_tokens=estimated_tokens,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _model_tool_stream_dict(self)


def _set_effectful_tool_call_event_fields(
    event: RunEvent,
    *,
    type: str,
    run_id: str,
    trace_id: str,
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, Any] | None,
    tool_metadata: ToolMetadata | dict[str, Any] | None,
    cycle_index: int | None,
    agent_name: str | None,
    session_id: str | None,
    parent_event_id: str | None,
    parent_run_id: str | None,
    event_id: str | None,
    created_at: float | None,
    metadata: dict[str, Any] | None,
) -> None:
    _set_run_event_fields(
        event,
        type=type,
        run_id=run_id,
        trace_id=trace_id,
        cycle_index=cycle_index,
        agent_name=agent_name,
        session_id=session_id,
        parent_event_id=parent_event_id,
        parent_run_id=parent_run_id,
        event_id=event_id,
        created_at=created_at,
        metadata=metadata,
    )
    object.__setattr__(event, "tool_name", _required_event_text(tool_name, "tool_name"))
    object.__setattr__(event, "tool_call_id", _required_event_text(tool_call_id, "tool_call_id"))
    metadata_arguments = metadata.get("arguments") if isinstance(metadata, dict) else None
    if not isinstance(metadata_arguments, dict) and isinstance(metadata, dict):
        metadata_arguments = metadata.get("tool_arguments")
    resolved_arguments = arguments if arguments is not None else metadata_arguments
    if resolved_arguments is not None and not isinstance(resolved_arguments, dict):
        raise ValueError("Run event tool arguments must be an object")
    object.__setattr__(event, "arguments", dict(resolved_arguments or {}))
    object.__setattr__(event, "tool_metadata", _event_tool_metadata(tool_metadata))


def _effectful_tool_call_event_dict(event: ToolCallPlannedEvent | ToolCallStartedEvent) -> dict[str, Any]:
    payload = RunEvent.to_dict(event)
    payload["tool_name"] = event.tool_name
    payload["tool_call_id"] = event.tool_call_id
    payload["arguments"] = dict(event.arguments)
    if event.tool_metadata is not None:
        payload["tool_metadata"] = event.tool_metadata.to_dict()
    return payload


@dataclass(frozen=True, slots=True)
class ToolCallPlannedEvent(RunEvent):
    tool_name: str = ""
    tool_call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_metadata: ToolMetadata | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any] | None = None,
        tool_metadata: ToolMetadata | dict[str, Any] | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_effectful_tool_call_event_fields(
            self,
            type="tool_call_planned",
            run_id=run_id,
            trace_id=trace_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            tool_metadata=tool_metadata,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _effectful_tool_call_event_dict(self)


@dataclass(frozen=True, slots=True)
class ToolCallStartedEvent(RunEvent):
    tool_name: str = ""
    tool_call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_metadata: ToolMetadata | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any] | None = None,
        tool_metadata: ToolMetadata | dict[str, Any] | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_effectful_tool_call_event_fields(
            self,
            type="tool_call_started",
            run_id=run_id,
            trace_id=trace_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            tool_metadata=tool_metadata,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _effectful_tool_call_event_dict(self)


@dataclass(frozen=True, slots=True)
class ToolCallCompletedEvent(RunEvent):
    tool_name: str = ""
    tool_call_id: str = ""
    status: str = ""
    directive: str | None = None
    error_code: str | None = None
    execution_started: bool | None = None
    duration_ms: int | None = None
    tool_metadata: ToolMetadata | None = None
    _present_additive_fields: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        status: str,
        directive: str | _MissingToolLifecycleField = _MISSING_TOOL_LIFECYCLE_FIELD,
        error_code: str | None | _MissingToolLifecycleField = _MISSING_TOOL_LIFECYCLE_FIELD,
        execution_started: bool | _MissingToolLifecycleField = _MISSING_TOOL_LIFECYCLE_FIELD,
        duration_ms: int | None | _MissingToolLifecycleField = _MISSING_TOOL_LIFECYCLE_FIELD,
        tool_metadata: ToolMetadata | dict[str, Any] | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="tool_call_completed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "tool_name", _required_event_text(tool_name, "tool_name"))
        object.__setattr__(self, "tool_call_id", _required_event_text(tool_call_id, "tool_call_id"))
        object.__setattr__(self, "status", _canonical_tool_status(status))
        present_fields: set[str] = set()
        if not isinstance(directive, _MissingToolLifecycleField):
            present_fields.add("directive")
            directive_value = _tool_directive(directive)
        else:
            directive_value = None
        if not isinstance(error_code, _MissingToolLifecycleField):
            present_fields.add("error_code")
            error_code_value = _tool_error_code(error_code)
        else:
            error_code_value = None
        if not isinstance(execution_started, _MissingToolLifecycleField):
            present_fields.add("execution_started")
            execution_started_value = _tool_execution_started(execution_started)
        else:
            execution_started_value = None
        if not isinstance(duration_ms, _MissingToolLifecycleField):
            present_fields.add("duration_ms")
            duration_ms_value = _tool_duration_ms(duration_ms)
        else:
            duration_ms_value = None
        if execution_started_value is False and duration_ms_value is not None:
            raise ValueError("Run event duration_ms must be null when execution_started is false")
        object.__setattr__(self, "directive", directive_value)
        object.__setattr__(self, "error_code", error_code_value)
        object.__setattr__(self, "execution_started", execution_started_value)
        object.__setattr__(self, "duration_ms", duration_ms_value)
        object.__setattr__(self, "tool_metadata", _event_tool_metadata(tool_metadata))
        object.__setattr__(self, "_present_additive_fields", frozenset(present_fields))

    def has_additive_field(self, field_name: str) -> bool:
        if field_name not in _TOOL_COMPLETED_ADDITIVE_FIELDS:
            raise ValueError(f"Unknown tool lifecycle field: {field_name}")
        return field_name in self._present_additive_fields

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        payload["status"] = self.status
        for field_name in ("directive", "error_code", "execution_started", "duration_ms"):
            if self.has_additive_field(field_name):
                payload[field_name] = getattr(self, field_name)
        if self.tool_metadata is not None:
            payload["tool_metadata"] = self.tool_metadata.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ApprovalRequestedEvent(RunEvent):
    request_id: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    message: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        message: str,
        request_id: str = "",
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="approval_requested",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "message", message)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["request_id"] = self.request_id
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        payload["message"] = self.message
        return payload


@dataclass(frozen=True, slots=True)
class ApprovalResolvedEvent(RunEvent):
    request_id: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    action: ApprovalAction | None = None
    approved: bool | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        action: ApprovalAction | str | None = None,
        approved: bool | None = None,
        request_id: str = "",
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="approval_resolved",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        metadata_action = metadata.get("action") if isinstance(metadata, dict) else None
        resolved_action = _canonical_approval_action(action if action is not None else metadata_action)
        resolved_approved = approved
        if resolved_action is None and approved is not None:
            resolved_action = "allow" if approved else "deny"
        if resolved_action is not None:
            action_approved = resolved_action in {"allow", "allow_session"}
            if approved is not None and approved is not action_approved:
                raise ValueError(f"Approval action {resolved_action!r} conflicts with approved={approved!r}")
            resolved_approved = action_approved
        object.__setattr__(self, "action", resolved_action)
        object.__setattr__(self, "approved", resolved_approved)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["request_id"] = self.request_id
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        if self.action is not None:
            payload["action"] = self.action
        if self.approved is not None:
            payload["approved"] = self.approved
        return payload


ToolStartedEvent = ToolCallStartedEvent
ToolFinishedEvent = ToolCallCompletedEvent
ToolApprovalRequestedEvent = ApprovalRequestedEvent


@dataclass(frozen=True, slots=True)
class HandoffEvent(RunEvent):
    source_agent: str = ""
    target_agent: str = ""
    tool_call_id: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        source_agent: str,
        target_agent: str,
        tool_call_id: str,
        cycle_index: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="handoff",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=source_agent,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "source_agent", source_agent)
        object.__setattr__(self, "target_agent", target_agent)
        object.__setattr__(self, "tool_call_id", tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["source_agent"] = self.source_agent
        payload["target_agent"] = self.target_agent
        payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class SubRunStartedEvent(RunEvent):
    parent_tool_call_id: str = ""
    child_session_id: str | None = None
    task_id: str | None = None
    status: str = "running"

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        parent_tool_call_id: str,
        agent_name: str | None = None,
        child_session_id: str | None = None,
        task_id: str | None = None,
        status: str = "running",
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="sub_run_started",
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "parent_tool_call_id", parent_tool_call_id)
        object.__setattr__(self, "child_session_id", child_session_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["parent_tool_call_id"] = self.parent_tool_call_id
        payload["status"] = self.status
        if self.child_session_id:
            payload["child_session_id"] = self.child_session_id
        if self.task_id:
            payload["task_id"] = self.task_id
        return payload


@dataclass(frozen=True, slots=True)
class SubRunCompletedEvent(RunEvent):
    parent_tool_call_id: str = ""
    child_session_id: str | None = None
    task_id: str | None = None
    status: str = ""
    final_output: str | None = None
    wait_reason: str | None = None
    error: str | None = None
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None
    token_usage: dict[str, Any] | None = None
    budget_usage: BudgetUsageSnapshot | None = None
    budget_exhaustion: BudgetExhaustion | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        parent_tool_call_id: str,
        status: str,
        agent_name: str | None = None,
        child_session_id: str | None = None,
        task_id: str | None = None,
        final_output: str | None = None,
        wait_reason: str | None = None,
        error: str | None = None,
        completion_reason: CompletionReason | None = None,
        completion_tool_name: str | None = None,
        partial_output: str | None = None,
        token_usage: dict[str, Any] | None = None,
        budget_usage: BudgetUsageSnapshot | None = None,
        budget_exhaustion: BudgetExhaustion | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="sub_run_completed",
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "parent_tool_call_id", parent_tool_call_id)
        object.__setattr__(self, "child_session_id", child_session_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "final_output", final_output)
        object.__setattr__(self, "wait_reason", wait_reason)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "completion_reason", completion_reason)
        object.__setattr__(self, "completion_tool_name", completion_tool_name)
        object.__setattr__(self, "partial_output", partial_output)
        object.__setattr__(self, "token_usage", dict(token_usage) if token_usage is not None else None)
        object.__setattr__(self, "budget_usage", budget_usage)
        object.__setattr__(self, "budget_exhaustion", budget_exhaustion)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["parent_tool_call_id"] = self.parent_tool_call_id
        payload["status"] = self.status
        if self.child_session_id:
            payload["child_session_id"] = self.child_session_id
        if self.task_id:
            payload["task_id"] = self.task_id
        if self.final_output is not None:
            payload["final_output"] = self.final_output
        if self.wait_reason is not None:
            payload["wait_reason"] = self.wait_reason
        if self.error is not None:
            payload["error"] = self.error
        if self.completion_reason is not None:
            payload["completion_reason"] = self.completion_reason.value
        if self.completion_tool_name is not None or self.budget_usage is not None:
            payload["completion_tool_name"] = self.completion_tool_name
        if self.partial_output is not None or self.budget_usage is not None:
            payload["partial_output"] = self.partial_output
        if self.token_usage is not None:
            payload["token_usage"] = dict(self.token_usage)
        if self.budget_usage is not None:
            payload["budget_usage"] = self.budget_usage.to_dict()
        if self.budget_exhaustion is not None:
            payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class HandoffStartedEvent(RunEvent):
    source_agent: str = ""
    target_agent: str = ""
    tool_call_id: str = ""
    status: str = "started"
    child_session_id: str | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        source_agent: str,
        target_agent: str,
        tool_call_id: str,
        status: str = "started",
        child_session_id: str | None = None,
        cycle_index: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="handoff_started",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=source_agent,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "source_agent", source_agent)
        object.__setattr__(self, "target_agent", target_agent)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "child_session_id", child_session_id)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["source_agent"] = self.source_agent
        payload["target_agent"] = self.target_agent
        payload["tool_call_id"] = self.tool_call_id
        payload["status"] = self.status
        if self.child_session_id:
            payload["child_session_id"] = self.child_session_id
        return payload


@dataclass(frozen=True, slots=True)
class HandoffCompletedEvent(RunEvent):
    source_agent: str = ""
    target_agent: str = ""
    tool_call_id: str = ""
    status: str = ""
    child_session_id: str | None = None
    child_run_id: str | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        source_agent: str,
        target_agent: str,
        tool_call_id: str,
        status: str,
        child_session_id: str | None = None,
        child_run_id: str | None = None,
        cycle_index: int | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="handoff_completed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=source_agent,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "source_agent", source_agent)
        object.__setattr__(self, "target_agent", target_agent)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "child_session_id", child_session_id)
        object.__setattr__(self, "child_run_id", child_run_id)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["source_agent"] = self.source_agent
        payload["target_agent"] = self.target_agent
        payload["tool_call_id"] = self.tool_call_id
        payload["status"] = self.status
        if self.child_session_id:
            payload["child_session_id"] = self.child_session_id
        if self.child_run_id:
            payload["child_run_id"] = self.child_run_id
        return payload


@dataclass(frozen=True, slots=True)
class SessionPersistedEvent(RunEvent):
    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        session_id: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="session_persisted",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshotEvent(RunEvent):
    enforcement_boundary: BudgetEnforcementBoundary = BudgetEnforcementBoundary.RUN_START
    budget_usage: BudgetUsageSnapshot = field(default_factory=BudgetUsageSnapshot)

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        enforcement_boundary: BudgetEnforcementBoundary,
        budget_usage: BudgetUsageSnapshot,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="budget_snapshot",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "enforcement_boundary", BudgetEnforcementBoundary(enforcement_boundary))
        if not isinstance(budget_usage, BudgetUsageSnapshot):
            raise TypeError("BudgetSnapshotEvent.budget_usage must be a BudgetUsageSnapshot")
        object.__setattr__(self, "budget_usage", budget_usage)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["enforcement_boundary"] = self.enforcement_boundary.value
        payload["budget_usage"] = self.budget_usage.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class BudgetExhaustedEvent(RunEvent):
    enforcement_boundary: BudgetEnforcementBoundary = BudgetEnforcementBoundary.RUN_START
    budget_usage: BudgetUsageSnapshot = field(default_factory=BudgetUsageSnapshot)
    budget_exhaustion: BudgetExhaustion = field(
        default_factory=lambda: BudgetExhaustion(
            dimension=BudgetDimension.WALL_TIME,
            reason=BudgetExhaustionReason.LIMIT_REACHED,
            limit=0,
            observed=0,
            attempted_increment=None,
            overshoot=0,
            unit="milliseconds",
            enforcement_boundary=BudgetEnforcementBoundary.RUN_START,
        )
    )

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        enforcement_boundary: BudgetEnforcementBoundary,
        budget_usage: BudgetUsageSnapshot,
        budget_exhaustion: BudgetExhaustion,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="budget_exhausted",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "enforcement_boundary", BudgetEnforcementBoundary(enforcement_boundary))
        if not isinstance(budget_usage, BudgetUsageSnapshot):
            raise TypeError("BudgetExhaustedEvent.budget_usage must be a BudgetUsageSnapshot")
        if not isinstance(budget_exhaustion, BudgetExhaustion):
            raise TypeError("BudgetExhaustedEvent.budget_exhaustion must be a BudgetExhaustion")
        if budget_exhaustion.enforcement_boundary is not self.enforcement_boundary:
            raise ValueError("BudgetExhaustedEvent boundaries must match")
        object.__setattr__(self, "budget_usage", budget_usage)
        object.__setattr__(self, "budget_exhaustion", budget_exhaustion)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["enforcement_boundary"] = self.enforcement_boundary.value
        payload["budget_usage"] = self.budget_usage.to_dict()
        payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RunCompletedEvent(RunEvent):
    final_output: str | None = None
    status: str = ""
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None
    budget_usage: BudgetUsageSnapshot | None = None
    budget_exhaustion: BudgetExhaustion | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        final_output: str | None,
        status: str,
        completion_reason: CompletionReason | None = None,
        completion_tool_name: str | None = None,
        partial_output: str | None = None,
        budget_usage: BudgetUsageSnapshot | None = None,
        budget_exhaustion: BudgetExhaustion | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="run_completed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "final_output", final_output)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "completion_reason", completion_reason)
        object.__setattr__(self, "completion_tool_name", completion_tool_name)
        object.__setattr__(self, "partial_output", partial_output)
        object.__setattr__(self, "budget_usage", budget_usage)
        object.__setattr__(self, "budget_exhaustion", budget_exhaustion)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["final_output"] = self.final_output
        payload["status"] = self.status
        if self.completion_reason is not None:
            payload["completion_reason"] = self.completion_reason.value
        if self.completion_tool_name is not None or self.budget_usage is not None:
            payload["completion_tool_name"] = self.completion_tool_name
        if self.partial_output is not None or self.budget_usage is not None:
            payload["partial_output"] = self.partial_output
        if self.budget_usage is not None:
            payload["budget_usage"] = self.budget_usage.to_dict()
        if self.budget_exhaustion is not None:
            payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RunFailedEvent(RunEvent):
    error: str = ""
    status: str | None = None
    completion_reason: CompletionReason | None = None
    completion_tool_name: str | None = None
    partial_output: str | None = None
    budget_usage: BudgetUsageSnapshot | None = None
    budget_exhaustion: BudgetExhaustion | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        error: str,
        status: str | None = None,
        completion_reason: CompletionReason | None = None,
        completion_tool_name: str | None = None,
        partial_output: str | None = None,
        budget_usage: BudgetUsageSnapshot | None = None,
        budget_exhaustion: BudgetExhaustion | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="run_failed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "completion_reason", completion_reason)
        object.__setattr__(self, "completion_tool_name", completion_tool_name)
        object.__setattr__(self, "partial_output", partial_output)
        object.__setattr__(self, "budget_usage", budget_usage)
        object.__setattr__(self, "budget_exhaustion", budget_exhaustion)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["error"] = self.error
        if self.status is not None:
            payload["status"] = self.status
        if self.completion_reason is not None:
            payload["completion_reason"] = self.completion_reason.value
        if self.completion_tool_name is not None or self.budget_usage is not None:
            payload["completion_tool_name"] = self.completion_tool_name
        if self.partial_output is not None or self.budget_usage is not None:
            payload["partial_output"] = self.partial_output
        if self.budget_usage is not None:
            payload["budget_usage"] = self.budget_usage.to_dict()
        if self.budget_exhaustion is not None:
            payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RunCancelledEvent(RunEvent):
    reason: str = ""
    completion_reason: CompletionReason | None = None
    partial_output: str | None = None
    budget_usage: BudgetUsageSnapshot | None = None
    budget_exhaustion: BudgetExhaustion | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        reason: str,
        completion_reason: CompletionReason | None = None,
        partial_output: str | None = None,
        budget_usage: BudgetUsageSnapshot | None = None,
        budget_exhaustion: BudgetExhaustion | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="run_cancelled",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "completion_reason", completion_reason)
        object.__setattr__(self, "partial_output", partial_output)
        object.__setattr__(self, "budget_usage", budget_usage)
        object.__setattr__(self, "budget_exhaustion", budget_exhaustion)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["reason"] = self.reason
        if self.completion_reason is not None:
            payload["completion_reason"] = self.completion_reason.value
        if self.partial_output is not None:
            payload["partial_output"] = self.partial_output
        if self.budget_usage is not None:
            payload["budget_usage"] = self.budget_usage.to_dict()
        if self.budget_exhaustion is not None:
            payload["budget_exhaustion"] = self.budget_exhaustion.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CheckpointCreatedEvent(RunEvent):
    checkpoint_key: str = ""
    resume_attempt: int = 1

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        resume_attempt: int,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="checkpoint_created",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "resume_attempt", _positive_event_integer(resume_attempt, "resume_attempt"))

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["checkpoint_key"] = self.checkpoint_key
        payload["resume_attempt"] = self.resume_attempt
        return payload


@dataclass(frozen=True, slots=True)
class CheckpointResumedEvent(CheckpointCreatedEvent):
    def __init__(self, **kwargs: Any) -> None:
        CheckpointCreatedEvent.__init__(self, **kwargs)
        object.__setattr__(self, "type", "checkpoint_resumed")


@dataclass(frozen=True, slots=True)
class OperationReplayedEvent(RunEvent):
    checkpoint_key: str = ""
    operation_id: str = ""
    operation_kind: OperationKind = OperationKind.MODEL
    receipt_state: OperationState = OperationState.SUCCEEDED

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        operation_id: str,
        operation_kind: OperationKind | str,
        receipt_state: OperationState | str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="operation_replayed",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        state = OperationState(receipt_state)
        if state not in {OperationState.SUCCEEDED, OperationState.FAILED}:
            raise ValueError("operation replay receipt_state must be succeeded or failed")
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "operation_id", _required_event_text(operation_id, "operation_id"))
        object.__setattr__(self, "operation_kind", OperationKind(operation_kind))
        object.__setattr__(self, "receipt_state", state)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload.update(
            checkpoint_key=self.checkpoint_key,
            operation_id=self.operation_id,
            operation_kind=self.operation_kind.value,
            receipt_state=self.receipt_state.value,
        )
        return payload


@dataclass(frozen=True, slots=True)
class OperationAmbiguousEvent(RunEvent):
    checkpoint_key: str = ""
    operation_id: str = ""
    operation_kind: OperationKind = OperationKind.MODEL
    risk: str = ""
    idempotency_support: ToolIdempotency | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        operation_id: str,
        operation_kind: OperationKind | str,
        risk: str,
        idempotency_support: ToolIdempotency | str | None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="operation_ambiguous",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        kind = OperationKind(operation_kind)
        support = ToolIdempotency(idempotency_support) if idempotency_support is not None else None
        if kind is OperationKind.TOOL and support is None:
            raise ValueError("ambiguous tool event requires idempotency_support")
        if kind is OperationKind.MODEL and support is not None:
            raise ValueError("ambiguous model event idempotency_support must be null")
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "operation_id", _required_event_text(operation_id, "operation_id"))
        object.__setattr__(self, "operation_kind", kind)
        object.__setattr__(self, "risk", _required_event_text(risk, "risk"))
        object.__setattr__(self, "idempotency_support", support)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload.update(
            checkpoint_key=self.checkpoint_key,
            operation_id=self.operation_id,
            operation_kind=self.operation_kind.value,
            risk=self.risk,
            idempotency_support=(self.idempotency_support.value if self.idempotency_support is not None else None),
        )
        return payload


@dataclass(frozen=True, slots=True)
class ReconciliationRequiredEvent(RunEvent):
    checkpoint_key: str = ""
    operation_id: str = ""
    operation_kind: OperationKind = OperationKind.MODEL
    interruption_reason: str = "resume_requires_reconciliation"
    resume_observation: ResumeObservation | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        operation_id: str,
        operation_kind: OperationKind | str,
        interruption_reason: str,
        resume_observation: ResumeObservation,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="reconciliation_required",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        kind = OperationKind(operation_kind)
        if not isinstance(resume_observation, ResumeObservation):
            raise TypeError("reconciliation event resume_observation must be ResumeObservation")
        if resume_observation.operation_id != operation_id or resume_observation.operation_kind is not kind:
            raise ValueError("reconciliation event operation must match resume_observation")
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "operation_id", _required_event_text(operation_id, "operation_id"))
        object.__setattr__(self, "operation_kind", kind)
        object.__setattr__(
            self,
            "interruption_reason",
            _required_event_text(interruption_reason, "interruption_reason"),
        )
        object.__setattr__(self, "resume_observation", resume_observation)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        assert self.resume_observation is not None
        payload.update(
            checkpoint_key=self.checkpoint_key,
            operation_id=self.operation_id,
            operation_kind=self.operation_kind.value,
            interruption_reason=self.interruption_reason,
            resume_observation=self.resume_observation.to_dict(),
        )
        return payload


@dataclass(frozen=True, slots=True)
class ModelRetryDuplicateRiskEvent(RunEvent):
    checkpoint_key: str = ""
    operation_id: str = ""
    operation_kind: OperationKind = OperationKind.MODEL
    risk: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        operation_id: str,
        operation_kind: OperationKind | str,
        risk: str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="model_retry_duplicate_risk",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        kind = OperationKind(operation_kind)
        if kind is not OperationKind.MODEL:
            raise ValueError("model retry duplicate risk event requires model operation_kind")
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "operation_id", _required_event_text(operation_id, "operation_id"))
        object.__setattr__(self, "operation_kind", kind)
        object.__setattr__(self, "risk", _required_event_text(risk, "risk"))

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload.update(
            checkpoint_key=self.checkpoint_key,
            operation_id=self.operation_id,
            operation_kind=self.operation_kind.value,
            risk=self.risk,
        )
        return payload


@dataclass(frozen=True, slots=True)
class ReconciliationResolvedEvent(RunEvent):
    checkpoint_key: str = ""
    operation_id: str = ""
    operation_kind: OperationKind = OperationKind.MODEL
    decision: ReconciliationDecisionKind = ReconciliationDecisionKind.DEFER

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        checkpoint_key: str,
        operation_id: str,
        operation_kind: OperationKind | str,
        decision: ReconciliationDecisionKind | str,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        parent_run_id: str | None = None,
        event_id: str | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _set_run_event_fields(
            self,
            type="reconciliation_resolved",
            run_id=run_id,
            trace_id=trace_id,
            cycle_index=cycle_index,
            agent_name=agent_name,
            session_id=session_id,
            parent_event_id=parent_event_id,
            parent_run_id=parent_run_id,
            event_id=event_id,
            created_at=created_at,
            metadata=metadata,
        )
        object.__setattr__(self, "checkpoint_key", _required_event_text(checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "operation_id", _required_event_text(operation_id, "operation_id"))
        object.__setattr__(self, "operation_kind", OperationKind(operation_kind))
        object.__setattr__(self, "decision", ReconciliationDecisionKind(decision))

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload.update(
            checkpoint_key=self.checkpoint_key,
            operation_id=self.operation_id,
            operation_kind=self.operation_kind.value,
            decision=self.decision.value,
        )
        return payload


def _required_event_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Run event {field_name} must be a non-empty string")
    return value


def _positive_event_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Run event {field_name} must be a positive integer")
    return value


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def _common_event_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    created_at = payload.get("created_at")
    if created_at is None:
        created_at_ms = payload.get("created_at_ms")
        if isinstance(created_at_ms, int | float):
            created_at = created_at_ms / 1000.0
    return {
        "run_id": payload["run_id"],
        "trace_id": payload["trace_id"],
        "session_id": payload.get("session_id"),
        "parent_event_id": payload.get("parent_event_id"),
        "parent_run_id": payload.get("parent_run_id"),
        "event_id": payload.get("event_id"),
        "created_at": created_at,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    }


def _with_cycle_and_agent(payload: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    return {
        **common,
        "cycle_index": payload.get("cycle_index") if isinstance(payload.get("cycle_index"), int) else None,
        "agent_name": payload.get("agent_name"),
    }


def _validate_event_wire(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Run event payload must be an object")
    if payload.get("version") != RUN_EVENT_VERSION:
        raise ValueError(f"Unsupported run event version: {payload.get('version')!r}")

    for field_name in ("type", "event_id", "run_id", "trace_id"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run event {field_name} must be a non-empty string")

    for field_name in ("session_id", "parent_event_id", "parent_run_id", "agent_name"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Run event {field_name} must be a string or null")

    created_at = payload.get("created_at")
    created_at_ms = payload.get("created_at_ms")
    timestamp = created_at if created_at is not None else created_at_ms
    if isinstance(timestamp, bool) or not isinstance(timestamp, int | float) or not isfinite(timestamp) or timestamp < 0:
        raise ValueError("Run event created_at must be a finite non-negative number")

    cycle_index = payload.get("cycle_index")
    if cycle_index is not None and (isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0):
        raise ValueError("Run event cycle_index must be a non-negative integer or null")

    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise ValueError("Run event metadata must be an object")
    tool_lifecycle_types = {"tool_call_planned", "tool_call_started", "tool_call_completed"}
    if payload["type"] in tool_lifecycle_types:
        _required_event_text(payload.get("tool_call_id"), "tool_call_id")
        _required_event_text(payload.get("tool_name"), "tool_name")
        if "tool_metadata" in payload:
            if not isinstance(payload["tool_metadata"], dict):
                raise ValueError("Run event tool_metadata must be an object")
            _event_tool_metadata(payload["tool_metadata"])
    if payload["type"] in {"tool_call_planned", "tool_call_started"} and not isinstance(
        payload.get("arguments"), dict
    ):
        raise ValueError("Run event tool arguments must be an object")
    if payload["type"] == "tool_call_completed":
        _wire_tool_status(payload.get("status"))
        if "directive" in payload:
            _tool_directive(payload["directive"])
        if "error_code" in payload:
            _tool_error_code(payload["error_code"])
        if "execution_started" in payload:
            _tool_execution_started(payload["execution_started"])
        if "duration_ms" in payload:
            _tool_duration_ms(payload["duration_ms"])
        if payload.get("execution_started") is False and payload.get("duration_ms") is not None:
            raise ValueError("Run event duration_ms must be null when execution_started is false")
    if payload["type"] in {"assistant_delta", "reasoning_delta"}:
        _stream_delta(payload.get("delta"))
    typed_stream_events = {
        "assistant_delta",
        "reasoning_delta",
        "model_tool_call_started",
        "model_tool_call_progress",
    }
    if payload["type"] in typed_stream_events and (
        isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 1
    ):
        raise ValueError("Run event stream variants require a positive cycle_index")
    stream_counter_fields = {
        "assistant_delta": ("content_chars", "estimated_tokens"),
        "reasoning_delta": ("reasoning_chars", "estimated_tokens"),
        "model_tool_call_started": ("tool_call_index", "arguments_chars", "estimated_tokens"),
        "model_tool_call_progress": ("tool_call_index", "arguments_chars", "estimated_tokens"),
    }
    for field_name in stream_counter_fields.get(payload["type"], ()):
        _optional_stream_counter(payload.get(field_name), field_name)
    if payload["type"] in {"model_tool_call_started", "model_tool_call_progress"}:
        _required_event_text(payload.get("tool_call_id"), "tool_call_id")
        _required_event_text(payload.get("tool_name"), "tool_name")

    _completion_reason(payload.get("completion_reason"))
    _completion_text(payload.get("completion_tool_name"), "completion_tool_name")
    _completion_text(payload.get("partial_output"), "partial_output")
    budget_usage = _budget_usage(payload.get("budget_usage"))
    budget_exhaustion = _budget_exhaustion(payload.get("budget_exhaustion"))
    if payload["type"] in {"budget_snapshot", "budget_exhausted"}:
        boundary_raw = payload.get("enforcement_boundary")
        try:
            boundary = BudgetEnforcementBoundary(boundary_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported budget enforcement boundary: {boundary_raw!r}") from exc
        if budget_usage is None:
            raise ValueError(f"Run event {payload['type']} requires budget_usage")
        if payload["type"] == "budget_exhausted":
            if budget_exhaustion is None:
                raise ValueError("Run event budget_exhausted requires budget_exhaustion")
            if budget_exhaustion.enforcement_boundary is not boundary:
                raise ValueError("Run event budget exhaustion boundaries must match")

    checkpoint_event_types = {
        "checkpoint_created",
        "checkpoint_resumed",
        "operation_replayed",
        "operation_ambiguous",
        "reconciliation_required",
        "model_retry_duplicate_risk",
        "reconciliation_resolved",
    }
    if payload["type"] in checkpoint_event_types:
        _required_event_text(payload.get("checkpoint_key"), "checkpoint_key")
        if not isinstance(payload.get("cycle_index"), int):
            raise ValueError("Run event cycle_index is required for checkpoint lifecycle events")
    if payload["type"] in {"checkpoint_created", "checkpoint_resumed"}:
        _positive_event_integer(payload.get("resume_attempt"), "resume_attempt")
    if payload["type"] in checkpoint_event_types - {"checkpoint_created", "checkpoint_resumed"}:
        _required_event_text(payload.get("operation_id"), "operation_id")
        OperationKind(payload.get("operation_kind"))
    if payload["type"] == "operation_replayed":
        receipt_state = OperationState(payload.get("receipt_state"))
        if receipt_state not in {OperationState.SUCCEEDED, OperationState.FAILED}:
            raise ValueError("operation replay receipt_state must be succeeded or failed")
    if payload["type"] in {"operation_ambiguous", "model_retry_duplicate_risk"}:
        _required_event_text(payload.get("risk"), "risk")
    if payload["type"] == "operation_ambiguous":
        operation_kind = OperationKind(payload.get("operation_kind"))
        support = payload.get("idempotency_support")
        if operation_kind is OperationKind.TOOL:
            if support is None:
                raise ValueError("ambiguous tool event requires idempotency_support")
            ToolIdempotency(support)
        elif support is not None:
            raise ValueError("ambiguous model event idempotency_support must be null")
    if payload["type"] == "reconciliation_required":
        _required_event_text(payload.get("interruption_reason"), "interruption_reason")
        observation = payload.get("resume_observation")
        if not isinstance(observation, dict):
            raise ValueError("Run event resume_observation must be an object")
        ResumeObservation.from_dict(observation)
    if payload["type"] == "model_retry_duplicate_risk" and payload.get("operation_kind") != "model":
        raise ValueError("model retry duplicate risk event requires model operation_kind")
    if payload["type"] == "reconciliation_resolved":
        ReconciliationDecisionKind(payload.get("decision"))


def event_from_dict(payload: dict[str, Any]) -> RunEvent:
    _validate_event_wire(payload)

    event_type = payload.get("type")
    common = _common_event_kwargs(payload)
    if event_type == "run_started":
        return RunStartedEvent(input=str(payload.get("input") or ""), agent_name=payload.get("agent_name"), **common)
    if event_type == "agent_started":
        return AgentStartedEvent(**_with_cycle_and_agent(payload, common))
    if event_type == "cycle_started":
        return CycleStartedEvent(**_with_cycle_and_agent(payload, common))
    if event_type == "llm_started":
        return LLMStartedEvent(model=str(payload.get("model") or ""), **_with_cycle_and_agent(payload, common))
    if event_type == "run_state_changed":
        return RunStateChangedEvent(
            state=str(payload.get("state") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "memory_compacted":
        return MemoryCompactedEvent(
            before_count=payload.get("before_count") if isinstance(payload.get("before_count"), int) else None,
            after_count=payload.get("after_count") if isinstance(payload.get("after_count"), int) else None,
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "memory_compact_started":
        message_count = payload.get("message_count")
        estimated_tokens = payload.get("estimated_tokens")
        return MemoryCompactStarted(
            message_count=message_count if isinstance(message_count, int) else 0,
            estimated_tokens=estimated_tokens if isinstance(estimated_tokens, int) else None,
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "memory_compact_completed":
        before_count = payload.get("before_count")
        after_count = payload.get("after_count")
        summary_tokens = payload.get("summary_tokens")
        return MemoryCompactCompleted(
            before_count=before_count if isinstance(before_count, int) else 0,
            after_count=after_count if isinstance(after_count, int) else 0,
            summary_tokens=summary_tokens if isinstance(summary_tokens, int) else None,
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "assistant_delta":
        return AssistantDeltaEvent(
            delta=payload["delta"],
            content_chars=payload.get("content_chars"),
            estimated_tokens=payload.get("estimated_tokens"),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "reasoning_delta":
        return ReasoningDeltaEvent(
            delta=payload["delta"],
            reasoning_chars=payload.get("reasoning_chars"),
            estimated_tokens=payload.get("estimated_tokens"),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type in {"model_tool_call_started", "model_tool_call_progress"}:
        event_class = ModelToolCallStartedEvent if event_type == "model_tool_call_started" else ModelToolCallProgressEvent
        return event_class(
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            tool_call_index=payload.get("tool_call_index"),
            arguments_chars=payload.get("arguments_chars"),
            estimated_tokens=payload.get("estimated_tokens"),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type in {"tool_call_planned", "tool_call_started"}:
        event_class = ToolCallPlannedEvent if event_type == "tool_call_planned" else ToolCallStartedEvent
        return event_class(
            tool_name=payload["tool_name"],
            tool_call_id=payload["tool_call_id"],
            arguments=payload["arguments"],
            tool_metadata=payload.get("tool_metadata"),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "tool_call_completed":
        additive_fields = {
            field_name: payload[field_name]
            for field_name in _TOOL_COMPLETED_ADDITIVE_FIELDS
            if field_name in payload
        }
        return ToolCallCompletedEvent(
            tool_name=payload["tool_name"],
            tool_call_id=payload["tool_call_id"],
            status=payload["status"],
            tool_metadata=payload.get("tool_metadata"),
            **additive_fields,
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "approval_requested":
        return ApprovalRequestedEvent(
            request_id=str(payload.get("request_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            message=str(payload.get("message") or payload.get("preview") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "approval_resolved":
        return ApprovalResolvedEvent(
            request_id=str(payload.get("request_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            action=payload.get("action") if isinstance(payload.get("action"), str) else None,
            approved=payload.get("approved") if isinstance(payload.get("approved"), bool) else None,
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "handoff":
        return HandoffEvent(
            source_agent=str(payload.get("source_agent") or payload.get("agent_name") or ""),
            target_agent=str(payload.get("target_agent") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            run_id=common["run_id"],
            trace_id=common["trace_id"],
            cycle_index=payload.get("cycle_index") if isinstance(payload.get("cycle_index"), int) else None,
            session_id=common["session_id"],
            parent_event_id=common["parent_event_id"],
            parent_run_id=common["parent_run_id"],
            event_id=common["event_id"],
            created_at=common["created_at"],
            metadata=common["metadata"],
        )
    if event_type == "sub_run_started":
        return SubRunStartedEvent(
            parent_tool_call_id=str(payload.get("parent_tool_call_id") or ""),
            agent_name=payload.get("agent_name"),
            child_session_id=payload.get("child_session_id"),
            task_id=payload.get("task_id"),
            status=str(payload.get("status") or "running"),
            **common,
        )
    if event_type == "sub_run_completed":
        token_usage = payload.get("token_usage")
        return SubRunCompletedEvent(
            parent_tool_call_id=str(payload.get("parent_tool_call_id") or ""),
            agent_name=payload.get("agent_name"),
            child_session_id=payload.get("child_session_id"),
            task_id=payload.get("task_id"),
            status=str(payload.get("status") or ""),
            final_output=payload.get("final_output") if payload.get("final_output") is not None else None,
            wait_reason=payload.get("wait_reason") if payload.get("wait_reason") is not None else None,
            error=payload.get("error") if payload.get("error") is not None else None,
            completion_reason=_completion_reason(payload.get("completion_reason")),
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            token_usage=token_usage if isinstance(token_usage, dict) else None,
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            **common,
        )
    if event_type == "handoff_started":
        return HandoffStartedEvent(
            source_agent=str(payload.get("source_agent") or payload.get("agent_name") or ""),
            target_agent=str(payload.get("target_agent") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or "started"),
            child_session_id=payload.get("child_session_id"),
            run_id=common["run_id"],
            trace_id=common["trace_id"],
            cycle_index=payload.get("cycle_index") if isinstance(payload.get("cycle_index"), int) else None,
            session_id=common["session_id"],
            parent_event_id=common["parent_event_id"],
            parent_run_id=common["parent_run_id"],
            event_id=common["event_id"],
            created_at=common["created_at"],
            metadata=common["metadata"],
        )
    if event_type == "handoff_completed":
        return HandoffCompletedEvent(
            source_agent=str(payload.get("source_agent") or payload.get("agent_name") or ""),
            target_agent=str(payload.get("target_agent") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or ""),
            child_session_id=payload.get("child_session_id"),
            child_run_id=payload.get("child_run_id"),
            run_id=common["run_id"],
            trace_id=common["trace_id"],
            cycle_index=payload.get("cycle_index") if isinstance(payload.get("cycle_index"), int) else None,
            session_id=common["session_id"],
            parent_event_id=common["parent_event_id"],
            parent_run_id=common["parent_run_id"],
            event_id=common["event_id"],
            created_at=common["created_at"],
            metadata=common["metadata"],
        )
    if event_type == "session_persisted":
        return SessionPersistedEvent(**_with_cycle_and_agent(payload, common))
    if event_type == "budget_snapshot":
        return BudgetSnapshotEvent(
            enforcement_boundary=BudgetEnforcementBoundary(payload.get("enforcement_boundary")),
            budget_usage=cast(BudgetUsageSnapshot, _budget_usage(payload.get("budget_usage"))),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "budget_exhausted":
        return BudgetExhaustedEvent(
            enforcement_boundary=BudgetEnforcementBoundary(payload.get("enforcement_boundary")),
            budget_usage=cast(BudgetUsageSnapshot, _budget_usage(payload.get("budget_usage"))),
            budget_exhaustion=cast(BudgetExhaustion, _budget_exhaustion(payload.get("budget_exhaustion"))),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "run_completed":
        final_output = payload.get("final_output")
        return RunCompletedEvent(
            final_output=str(final_output) if final_output is not None else None,
            status=str(payload.get("status") or ""),
            completion_reason=_completion_reason(payload.get("completion_reason")),
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "run_failed":
        return RunFailedEvent(
            error=str(payload.get("error") or ""),
            status=(str(payload["status"]) if payload.get("status") is not None else None),
            completion_reason=_completion_reason(payload.get("completion_reason")),
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "run_cancelled":
        return RunCancelledEvent(
            reason=str(payload.get("reason") or ""),
            completion_reason=_completion_reason(payload.get("completion_reason")),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "checkpoint_created":
        return CheckpointCreatedEvent(
            checkpoint_key=payload["checkpoint_key"],
            resume_attempt=payload["resume_attempt"],
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "checkpoint_resumed":
        return CheckpointResumedEvent(
            checkpoint_key=payload["checkpoint_key"],
            resume_attempt=payload["resume_attempt"],
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "operation_replayed":
        return OperationReplayedEvent(
            checkpoint_key=payload["checkpoint_key"],
            operation_id=payload["operation_id"],
            operation_kind=payload["operation_kind"],
            receipt_state=payload["receipt_state"],
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "operation_ambiguous":
        return OperationAmbiguousEvent(
            checkpoint_key=payload["checkpoint_key"],
            operation_id=payload["operation_id"],
            operation_kind=payload["operation_kind"],
            risk=payload["risk"],
            idempotency_support=payload.get("idempotency_support"),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "reconciliation_required":
        return ReconciliationRequiredEvent(
            checkpoint_key=payload["checkpoint_key"],
            operation_id=payload["operation_id"],
            operation_kind=payload["operation_kind"],
            interruption_reason=payload["interruption_reason"],
            resume_observation=ResumeObservation.from_dict(payload["resume_observation"]),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "model_retry_duplicate_risk":
        return ModelRetryDuplicateRiskEvent(
            checkpoint_key=payload["checkpoint_key"],
            operation_id=payload["operation_id"],
            operation_kind=payload["operation_kind"],
            risk=payload["risk"],
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "reconciliation_resolved":
        return ReconciliationResolvedEvent(
            checkpoint_key=payload["checkpoint_key"],
            operation_id=payload["operation_id"],
            operation_kind=payload["operation_kind"],
            decision=payload["decision"],
            **_with_cycle_and_agent(payload, common),
        )

    raise ValueError(f"Unsupported run event type: {event_type!r}")


def event_from_stream_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_name: str,
    session_id: str | None = None,
    parent_run_id: str | None = None,
    preserve_metadata: bool = False,
) -> RunEvent | None:
    raw_type = payload.get("event") or payload.get("type")
    cycle_index = payload.get("cycle_index") if preserve_metadata else payload.get("cycle")
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 1:
        return None
    metadata = dict(payload) if preserve_metadata else None
    common: _StreamEventCommon = {
        "run_id": run_id,
        "trace_id": trace_id,
        "agent_name": agent_name,
        "session_id": session_id,
        "parent_run_id": parent_run_id,
        "cycle_index": cycle_index,
        "metadata": metadata,
    }
    try:
        if raw_type == "assistant_delta":
            delta = payload.get("content_delta")
            if not isinstance(delta, str):
                delta = payload.get("delta")
            return AssistantDeltaEvent(
                delta=_stream_delta(delta, "content_delta or delta"),
                content_chars=_optional_stream_counter(payload.get("content_chars"), "content_chars"),
                estimated_tokens=_optional_stream_counter(payload.get("estimated_tokens"), "estimated_tokens"),
                **common,
            )
        if raw_type == "reasoning_delta":
            return ReasoningDeltaEvent(
                delta=_stream_delta(payload.get("reasoning_delta"), "reasoning_delta"),
                reasoning_chars=_optional_stream_counter(payload.get("reasoning_chars"), "reasoning_chars"),
                estimated_tokens=_optional_stream_counter(payload.get("estimated_tokens"), "estimated_tokens"),
                **common,
            )
        if raw_type in {"tool_call_started", "tool_call_progress"}:
            tool_call_id = _required_event_text(payload.get("tool_call_id"), "tool_call_id")
            tool_name = _required_event_text(payload.get("function_name"), "function_name")
            tool_call_index = _optional_stream_counter(payload.get("tool_call_index"), "tool_call_index")
            arguments_chars = _optional_stream_counter(payload.get("arguments_chars"), "arguments_chars")
            estimated_tokens = _optional_stream_counter(payload.get("estimated_tokens"), "estimated_tokens")
            if raw_type == "tool_call_started":
                return ModelToolCallStartedEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_call_index=tool_call_index,
                    arguments_chars=arguments_chars,
                    estimated_tokens=estimated_tokens,
                    **common,
                )
            return ModelToolCallProgressEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_call_index=tool_call_index,
                arguments_chars=arguments_chars,
                estimated_tokens=estimated_tokens,
                **common,
            )
    except ValueError:
        return None
    return None


def _runtime_session_id(payload: dict[str, Any], fallback: str | None) -> str | None:
    if fallback:
        return fallback
    payload_session_id = payload.get("session_id")
    if isinstance(payload_session_id, str) and payload_session_id:
        return payload_session_id
    return fallback


def event_from_runtime_log(
    event: str,
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_name: str,
    user_input: str,
    session_id: str | None = None,
) -> RunEvent | None:
    cycle_index = payload.get("cycle")
    if not isinstance(cycle_index, int):
        cycle_index = None
    session_id = _runtime_session_id(payload, session_id)

    if event == "run_started":
        return RunStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            input=user_input,
            metadata=dict(payload),
        )
    if event == "agent_started":
        return AgentStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            metadata=dict(payload),
        )
    if event == "cycle_started":
        return CycleStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            metadata=dict(payload),
        )
    if event == "llm_started":
        return LLMStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            model=str(payload.get("model") or ""),
            metadata=dict(payload),
        )
    if event == "cycle_llm_response":
        return None
    if event == "cycle_failed":
        return RunFailedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            error=str(payload.get("error") or "cycle failed"),
            metadata=dict(payload),
        )
    if event == "memory_compacted":
        before_count = payload.get("before_count")
        after_count = payload.get("after_count")
        return MemoryCompactedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            before_count=before_count if isinstance(before_count, int) else None,
            after_count=after_count if isinstance(after_count, int) else None,
            metadata=dict(payload),
        )
    if event == "memory_compact_started":
        message_count = payload.get("message_count")
        estimated_tokens = payload.get("estimated_tokens")
        return MemoryCompactStarted(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            message_count=message_count if isinstance(message_count, int) else 0,
            estimated_tokens=estimated_tokens if isinstance(estimated_tokens, int) else None,
            metadata=dict(payload),
        )
    if event == "memory_compact_completed":
        before_count = payload.get("before_count")
        after_count = payload.get("after_count")
        summary_tokens = payload.get("summary_tokens")
        return MemoryCompactCompleted(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            before_count=before_count if isinstance(before_count, int) else 0,
            after_count=after_count if isinstance(after_count, int) else 0,
            summary_tokens=summary_tokens if isinstance(summary_tokens, int) else None,
            metadata=dict(payload),
        )
    if event == "tool_call_planned":
        arguments = payload.get("arguments", payload.get("tool_arguments"))
        return ToolCallPlannedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            arguments=arguments if isinstance(arguments, dict) else None,
            tool_metadata=payload.get("tool_metadata"),
            metadata=dict(payload),
        )
    if event == "tool_result":
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("mode") == "approval_requested":
            return ApprovalRequestedEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                session_id=session_id,
                cycle_index=cycle_index,
                request_id=str(metadata.get("request_id") or ""),
                tool_name=str(metadata.get("tool_name") or payload.get("tool_name") or ""),
                tool_call_id=str(payload.get("tool_call_id") or ""),
                message=str(metadata.get("message") or payload.get("content") or ""),
                metadata=dict(payload),
            )
        if isinstance(metadata, dict) and metadata.get("mode") == "handoff":
            return HandoffEvent(
                run_id=run_id,
                trace_id=trace_id,
                source_agent=str(metadata.get("handoff_from") or agent_name),
                target_agent=str(metadata.get("handoff_to") or metadata.get("agent") or ""),
                tool_call_id=str(payload.get("tool_call_id") or ""),
                session_id=session_id,
                cycle_index=cycle_index,
                metadata=dict(metadata),
            )
        additive_fields = {
            field_name: payload[field_name]
            for field_name in _TOOL_COMPLETED_ADDITIVE_FIELDS
            if field_name in payload
        }
        return ToolCallCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or ""),
            tool_metadata=payload.get("tool_metadata"),
            **additive_fields,
            metadata=dict(payload),
        )
    if event in {"tool_started", "tool_call_started"}:
        arguments = payload.get("arguments", payload.get("tool_arguments"))
        return ToolCallStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            arguments=arguments if isinstance(arguments, dict) else None,
            tool_metadata=payload.get("tool_metadata"),
            metadata=dict(payload),
        )
    if event == "run_state_changed":
        return RunStateChangedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            state=str(payload.get("state") or ""),
            metadata=dict(payload),
        )
    if event == "session_persisted":
        return SessionPersistedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id or "",
            cycle_index=cycle_index,
            metadata=dict(payload),
        )
    if event == "budget_snapshot":
        usage = _budget_usage(payload.get("budget_usage"))
        if usage is None:
            raise ValueError("budget_snapshot runtime log requires budget_usage")
        return BudgetSnapshotEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            enforcement_boundary=BudgetEnforcementBoundary(payload.get("enforcement_boundary")),
            budget_usage=usage,
        )
    if event == "budget_exhausted":
        usage = _budget_usage(payload.get("budget_usage"))
        exhaustion = _budget_exhaustion(payload.get("budget_exhaustion"))
        if usage is None or exhaustion is None:
            raise ValueError("budget_exhausted runtime log requires budget usage and exhaustion")
        return BudgetExhaustedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            enforcement_boundary=BudgetEnforcementBoundary(payload.get("enforcement_boundary")),
            budget_usage=usage,
            budget_exhaustion=exhaustion,
        )
    if event == "run_completed":
        return RunCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            final_output=payload.get("final_output", payload.get("final_answer")),
            status="completed",
            completion_reason=_completion_reason(payload.get("completion_reason")),
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            metadata=dict(payload),
        )
    if event == "run_wait_user":
        return RunCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            final_output=payload.get("wait_reason"),
            status="wait_user",
            completion_reason=_completion_reason(payload.get("completion_reason")) or CompletionReason.WAIT_USER,
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            metadata=dict(payload),
        )
    if event in {"run_failed", "run_max_cycles"}:
        return RunFailedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            error=str(payload.get("error") or event),
            status=(str(payload["status"]) if payload.get("status") is not None else None),
            completion_reason=(
                _completion_reason(payload.get("completion_reason"))
                or (CompletionReason.MAX_CYCLES if event == "run_max_cycles" else CompletionReason.FAILED)
            ),
            completion_tool_name=_completion_text(payload.get("completion_tool_name"), "completion_tool_name"),
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            metadata=dict(payload),
        )
    if event == "run_cancelled":
        return RunCancelledEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            reason=str(payload.get("reason") or payload.get("error") or "run cancelled"),
            completion_reason=_completion_reason(payload.get("completion_reason")) or CompletionReason.CANCELLED,
            partial_output=_completion_text(payload.get("partial_output"), "partial_output"),
            budget_usage=_budget_usage(payload.get("budget_usage")),
            budget_exhaustion=_budget_exhaustion(payload.get("budget_exhaustion")),
            metadata=dict(payload),
        )
    return None
