from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

from vv_agent.budget import BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.events import RunEvent
from vv_agent.types import AgentResult, AgentStatus, CompletionReason, Message, TaskTokenUsage

if TYPE_CHECKING:
    from vv_agent.config import ResolvedModelConfig


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
    _session_driver: Any = field(default=None, repr=False, compare=False)

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
    def wait_reason(self) -> str | None:
        return self.raw_result.wait_reason

    @property
    def budget_usage(self) -> BudgetUsageSnapshot | None:
        return self.raw_result.budget_usage

    @property
    def budget_exhaustion(self) -> BudgetExhaustion | None:
        return self.raw_result.budget_exhaustion

    @property
    def error_code(self) -> str | None:
        return self.raw_result.error_code

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        payload["session_id"] = self.raw_result.session_id
        payload["turn_id"] = self.raw_result.turn_id
        return payload

    def _resolved_model_dict(self) -> dict[str, Any] | None:
        if self.resolved_model is None:
            return None
        endpoint = self.resolved_model.endpoint_options[0].endpoint.endpoint_id if self.resolved_model.endpoint_options else None
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


__all__ = ["RunResult"]
