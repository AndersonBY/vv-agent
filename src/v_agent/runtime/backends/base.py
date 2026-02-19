from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from v_agent.runtime.context import ExecutionContext
from v_agent.types import AgentResult, AgentTask, CycleRecord, Message

CycleExecutor = Callable[
    [int, list[Message], list[CycleRecord], dict[str, Any], ExecutionContext | None],
    AgentResult | None,
]


@runtime_checkable
class ExecutionBackend(Protocol):
    def execute(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message],
        shared_state: dict[str, Any],
        cycle_executor: CycleExecutor,
        ctx: ExecutionContext | None,
        max_cycles: int,
    ) -> AgentResult: ...

    def parallel_map(self, fn: Callable[..., Any], items: list[Any]) -> list[Any]:
        """Execute fn over items using the backend's parallelism. Default: serial."""
        ...
