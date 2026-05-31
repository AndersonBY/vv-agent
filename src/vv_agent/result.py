from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from vv_agent.events import RunEvent
from vv_agent.types import AgentResult, AgentStatus, Message, TaskTokenUsage


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

    @property
    def raw_cycles(self) -> list[Any]:
        return list(self.raw_result.cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "new_items": [item.to_dict() for item in self.new_items],
            "final_output": self._serializable_output(self.final_output),
            "status": self.status.value,
            "events": [event.to_dict() for event in self.events],
            "token_usage": self.token_usage.to_dict(),
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _serializable_output(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value
