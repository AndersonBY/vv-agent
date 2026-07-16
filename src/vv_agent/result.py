from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from threading import Lock
from typing import TYPE_CHECKING, Any

from vv_agent.budget import BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.events import RunEvent
from vv_agent.types import AgentResult, AgentStatus, AgentTask, CompletionReason, Message, TaskTokenUsage, ToolCall

if TYPE_CHECKING:
    from vv_agent.agent import Agent
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.run_config import RunConfig


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    interruption_id: str
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    cycle_index: int | None = None
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interruption_id": self.interruption_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "arguments": deepcopy(self.arguments),
            "message": self.message,
            "cycle_index": self.cycle_index,
            "approved": self.approved,
        }


@dataclass(frozen=True, slots=True)
class _PendingToolApproval:
    interruption_id: str
    call: ToolCall
    cycle_index: int
    context: Any
    allowed_tool_names: frozenset[str]
    orchestrator: Any
    task: AgentTask
    hook_manager: Any


@dataclass(frozen=True, slots=True)
class _RunResumeContext:
    agent: Agent
    input: str
    run_config: RunConfig
    runner: Any
    effective_run_config: RunConfig | None = None
    pending_tool_approval: _PendingToolApproval | None = None
    approval_consumption: _ApprovalConsumption = field(default_factory=lambda: _ApprovalConsumption(), compare=False)

    def claim_approval(self, interruption_id: str) -> bool:
        return self.approval_consumption.claim(interruption_id)


class _ApprovalConsumption:
    def __init__(self) -> None:
        self._lock = Lock()
        self._consumed: set[str] = set()

    def claim(self, interruption_id: str) -> bool:
        with self._lock:
            if interruption_id in self._consumed:
                return False
            self._consumed.add(interruption_id)
            return True


@dataclass(slots=True)
class RunResult:
    input: str
    new_items: list[Message]
    final_output: Any | None
    status: AgentStatus
    raw_result: AgentResult
    events: list[RunEvent] = field(default_factory=list)
    token_usage: TaskTokenUsage = field(default_factory=TaskTokenUsage)
    trace_id: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    resolved_model: ResolvedModelConfig | None = None
    _resume_context: _RunResumeContext | None = field(default=None, repr=False, compare=False)

    @property
    def result(self) -> AgentResult:
        return self.raw_result

    @property
    def resolved(self) -> ResolvedModelConfig | None:
        return self.resolved_model

    @property
    def raw_cycles(self) -> list[Any]:
        return list(self.raw_result.cycles)

    @property
    def completion_reason(self) -> CompletionReason | None:
        return self.raw_result.completion_reason

    @property
    def completion_tool_name(self) -> str | None:
        return self.raw_result.completion_tool_name

    @property
    def partial_output(self) -> str | None:
        return self.raw_result.partial_output

    @property
    def budget_usage(self) -> BudgetUsageSnapshot | None:
        return self.raw_result.budget_usage

    @property
    def budget_exhaustion(self) -> BudgetExhaustion | None:
        return self.raw_result.budget_exhaustion

    @property
    def approvals(self) -> tuple[ApprovalSnapshot, ...]:
        return self.approval_snapshot()

    def approval_snapshot(self) -> tuple[ApprovalSnapshot, ...]:
        return _approval_snapshots(self.raw_result)

    def into_state(self) -> RunState:
        return RunState.from_result(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "new_items": [item.to_dict() for item in self.new_items],
            "final_output": self._serializable_output(self.final_output),
            "status": self.status.value,
            "completion_reason": self.completion_reason.value if self.completion_reason is not None else None,
            "completion_tool_name": self.completion_tool_name,
            "partial_output": self.partial_output,
            "budget_usage": self.budget_usage.to_dict() if self.budget_usage is not None else None,
            "budget_exhaustion": self.budget_exhaustion.to_dict() if self.budget_exhaustion is not None else None,
            "events": [event.to_dict() for event in self.events],
            "token_usage": self.token_usage.to_dict(),
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
            "agent_name": self.agent_name,
            "resolved_model": self._resolved_model_dict(),
        }

    def _resolved_model_dict(self) -> dict[str, Any] | None:
        if self.resolved_model is None:
            return None
        endpoint = (
            self.resolved_model.endpoint_options[0].endpoint.endpoint_id
            if self.resolved_model.endpoint_options
            else None
        )
        return {
            "backend": self.resolved_model.backend,
            "requested_model": self.resolved_model.requested_model,
            "selected_model": self.resolved_model.selected_model,
            "model_id": self.resolved_model.model_id,
            "endpoint": endpoint,
        }

    @staticmethod
    def _serializable_output(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value


@dataclass(slots=True)
class RunState:
    result: RunResult
    _approved_interruption_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: RunResult) -> RunState:
        if result.status != AgentStatus.WAIT_USER:
            raise ValueError("only interrupted runs can be converted into RunState")
        return cls(result=result)

    @property
    def approvals(self) -> tuple[ApprovalSnapshot, ...]:
        return self.approval_snapshot()

    @property
    def approved_ids(self) -> tuple[str, ...]:
        return tuple(self._approved_interruption_ids)

    def approval_snapshot(self) -> tuple[ApprovalSnapshot, ...]:
        approved = set(self._approved_interruption_ids)
        return tuple(replace(snapshot, approved=snapshot.interruption_id in approved) for snapshot in self.result.approvals)

    def approve(self, interruption_id: str) -> None:
        if interruption_id not in self.pending_approval_ids():
            raise KeyError(f"Unknown approval interruption: {interruption_id}")
        if interruption_id not in self._approved_interruption_ids:
            self._approved_interruption_ids.append(interruption_id)

    def pending_approval_ids(self) -> list[str]:
        return [snapshot.interruption_id for snapshot in self.result.approvals]

    def approved_interruption_ids(self) -> tuple[str, ...]:
        return tuple(self._approved_interruption_ids)

    def _into_inner(self) -> tuple[RunResult, tuple[str, ...]]:
        return self.result, self.approved_interruption_ids()


def _approval_snapshots(result: AgentResult) -> tuple[ApprovalSnapshot, ...]:
    snapshots: list[ApprovalSnapshot] = []
    seen: set[str] = set()
    for cycle in result.cycles:
        for tool_result in cycle.tool_results:
            metadata = tool_result.metadata if isinstance(tool_result.metadata, dict) else {}
            interruption_id = metadata.get("approval_interruption_id") or metadata.get("request_id")
            mode = metadata.get("mode")
            if not isinstance(interruption_id, str) or not interruption_id:
                continue
            if mode != "approval_requested" and not metadata.get("approval_required"):
                continue
            if interruption_id in seen:
                continue
            arguments = metadata.get("arguments")
            snapshots.append(
                ApprovalSnapshot(
                    interruption_id=interruption_id,
                    tool_name=str(metadata.get("tool_name") or ""),
                    tool_call_id=tool_result.tool_call_id,
                    arguments=deepcopy(arguments) if isinstance(arguments, dict) else {},
                    message=str(metadata.get("message") or tool_result.content),
                    cycle_index=cycle.index,
                )
            )
            seen.add(interruption_id)
    return tuple(snapshots)


__all__ = ["ApprovalSnapshot", "RunResult", "RunState"]
