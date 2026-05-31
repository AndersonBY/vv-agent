from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RunEvent:
    type: str
    run_id: str
    trace_id: str
    cycle_index: int | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
        }
        if self.cycle_index is not None:
            payload["cycle_index"] = self.cycle_index
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "run_started")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", None)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "agent_started")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))


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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "llm_started")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "memory_compacted")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "assistant_delta")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "delta", delta)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["delta"] = self.delta
        return payload


@dataclass(frozen=True, slots=True)
class ToolStartedEvent(RunEvent):
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "tool_started")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class ToolFinishedEvent(RunEvent):
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "tool_finished")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
class ToolApprovalRequestedEvent(RunEvent):
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
        cycle_index: int | None = None,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "tool_approval_requested")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "message", message)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["tool_name"] = self.tool_name
        payload["tool_call_id"] = self.tool_call_id
        payload["message"] = self.message
        return payload


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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "handoff")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", source_agent)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "run_completed")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "type", "run_failed")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "agent_name", agent_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "error", error)

    def to_dict(self) -> dict[str, Any]:
        payload = RunEvent.to_dict(self)
        payload["error"] = self.error
        return payload


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def event_from_stream_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_name: str,
) -> RunEvent | None:
    raw_type = payload.get("type") or payload.get("event")
    if raw_type == "assistant_delta":
        delta = payload.get("delta", payload.get("content_delta", ""))
        return AssistantDeltaEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            delta=str(delta),
        )
    return RunEvent(
        type=str(raw_type or "stream_event"),
        run_id=run_id,
        trace_id=trace_id,
        agent_name=agent_name,
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
) -> RunEvent | None:
    cycle_index = payload.get("cycle")
    if not isinstance(cycle_index, int):
        cycle_index = None

    if event == "run_started":
        return RunStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            input=user_input,
            metadata=dict(payload),
        )
    if event == "cycle_started":
        return AgentStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            cycle_index=cycle_index,
            metadata=dict(payload),
        )
    if event == "llm_started":
        return LLMStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
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
            cycle_index=cycle_index,
            before_count=before_count if isinstance(before_count, int) else None,
            after_count=after_count if isinstance(after_count, int) else None,
            metadata=dict(payload),
        )
    if event == "tool_result":
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("mode") == "approval_requested":
            return ToolApprovalRequestedEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                cycle_index=cycle_index,
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
                cycle_index=cycle_index,
                metadata=dict(payload),
            )
        return ToolFinishedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            cycle_index=cycle_index,
            tool_name=str(payload.get("tool_name") or ""),
            tool_call_id=str(payload.get("tool_call_id") or ""),
            status=str(payload.get("status") or ""),
            metadata=dict(payload),
        )
    if event == "tool_started":
        return ToolStartedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
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
            cycle_index=cycle_index,
            error=str(payload.get("error") or event),
            metadata=dict(payload),
        )
    return None
