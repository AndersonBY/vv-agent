from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

RUN_EVENT_VERSION = "v1"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def event_created_at() -> float:
    return time.time()


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

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        delta: str,
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
        object.__setattr__(self, "delta", delta)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["delta"] = self.delta
        return payload


@dataclass(frozen=True, slots=True)
class ToolCallStartedEvent(RunEvent):
    tool_name: str = ""
    tool_call_id: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
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
            type="tool_call_started",
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
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class ToolCallCompletedEvent(RunEvent):
    tool_name: str = ""
    tool_call_id: str = ""
    status: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
        status: str,
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
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        payload["status"] = self.status
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
    approved: bool | None = None

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        tool_name: str,
        tool_call_id: str,
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
        object.__setattr__(self, "approved", approved)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["request_id"] = self.request_id
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
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
    token_usage: dict[str, Any] | None = None

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
        token_usage: dict[str, Any] | None = None,
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
        object.__setattr__(self, "token_usage", dict(token_usage) if token_usage is not None else None)

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
        if self.token_usage is not None:
            payload["token_usage"] = dict(self.token_usage)
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
class RunCompletedEvent(RunEvent):
    final_output: str | None = None
    status: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        final_output: str | None,
        status: str,
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

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["final_output"] = self.final_output
        payload["status"] = self.status
        return payload


@dataclass(frozen=True, slots=True)
class RunFailedEvent(RunEvent):
    error: str = ""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        error: str,
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

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["error"] = self.error
        return payload


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def _common_event_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(payload["run_id"]),
        "trace_id": str(payload["trace_id"]),
        "session_id": payload.get("session_id"),
        "parent_event_id": payload.get("parent_event_id"),
        "parent_run_id": payload.get("parent_run_id"),
        "event_id": payload.get("event_id"),
        "created_at": payload.get("created_at"),
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    }


def _with_cycle_and_agent(payload: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    return {
        **common,
        "cycle_index": payload.get("cycle_index") if isinstance(payload.get("cycle_index"), int) else None,
        "agent_name": payload.get("agent_name"),
    }


def event_from_dict(payload: dict[str, Any]) -> RunEvent:
    if payload.get("version") != RUN_EVENT_VERSION:
        raise ValueError(f"Unsupported run event version: {payload.get('version')!r}")

    event_type = payload.get("type")
    common = _common_event_kwargs(payload)
    if event_type == "run_started":
        return RunStartedEvent(input=str(payload.get("input") or ""), agent_name=payload.get("agent_name"), **common)
    if event_type == "agent_started":
        return AgentStartedEvent(**_with_cycle_and_agent(payload, common))
    if event_type == "llm_started":
        return LLMStartedEvent(model=str(payload.get("model") or ""), **_with_cycle_and_agent(payload, common))
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
        return AssistantDeltaEvent(delta=str(payload.get("delta") or ""), **_with_cycle_and_agent(payload, common))
    if event_type == "tool_call_started":
        return ToolCallStartedEvent(
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "tool_call_completed":
        return ToolCallCompletedEvent(
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "approval_requested":
        return ApprovalRequestedEvent(
            request_id=str(payload.get("request_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            message=str(payload.get("message") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "approval_resolved":
        return ApprovalResolvedEvent(
            request_id=str(payload.get("request_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
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
            token_usage=token_usage if isinstance(token_usage, dict) else None,
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
    if event_type == "run_completed":
        final_output = payload.get("final_output")
        return RunCompletedEvent(
            final_output=str(final_output) if final_output is not None else None,
            status=str(payload.get("status") or ""),
            **_with_cycle_and_agent(payload, common),
        )
    if event_type == "run_failed":
        return RunFailedEvent(error=str(payload.get("error") or ""), **_with_cycle_and_agent(payload, common))

    return RunEvent(type=str(event_type or "run_event"), **_with_cycle_and_agent(payload, common))


def event_from_stream_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_name: str,
    session_id: str | None = None,
) -> RunEvent | None:
    raw_type = payload.get("type") or payload.get("event")
    if raw_type == "assistant_delta":
        delta = payload.get("delta", payload.get("content_delta", ""))
        return AssistantDeltaEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            delta=str(delta),
        )
    return RunEvent(
        type=str(raw_type or "stream_event"),
        run_id=run_id,
        trace_id=trace_id,
        agent_name=agent_name,
        session_id=session_id,
        metadata=dict(payload),
    )


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

    if event == "run_started":
        return RunStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            input=user_input,
            metadata=dict(payload),
        )
    if event == "cycle_started":
        return AgentStartedEvent(
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
                metadata=dict(payload),
            )
        return ToolCallCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or ""),
            metadata=dict(payload),
        )
    if event == "tool_started":
        return ToolCallStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            metadata=dict(payload),
        )
    if event == "run_completed":
        return RunCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            final_output=payload.get("final_answer"),
            status="completed",
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
            metadata=dict(payload),
        )
    return None
