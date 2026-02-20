from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import AgentResult, AgentStatus, AgentTask, CycleRecord, Message


class InlineBackend:
    """Default synchronous backend that runs the cycle loop inline."""

    def execute(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message],
        shared_state: dict[str, Any],
        cycle_executor: CycleExecutor,
        ctx: ExecutionContext | None,
        max_cycles: int,
    ) -> AgentResult:
        messages = initial_messages
        cycles: list[CycleRecord] = []

        for cycle_index in range(1, max_cycles + 1):
            if ctx is not None:
                try:
                    ctx.check_cancelled()
                except Exception:
                    return AgentResult(
                        status=AgentStatus.FAILED,
                        messages=messages,
                        cycles=cycles,
                        error="Operation was cancelled",
                        shared_state=shared_state,
                        token_usage=summarize_task_token_usage(cycles),
                    )

            result = cycle_executor(cycle_index, messages, cycles, shared_state, ctx)
            if result is not None:
                return result

        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=messages,
            cycles=cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=shared_state,
            token_usage=summarize_task_token_usage(cycles),
        )

    def parallel_map(self, fn: Callable[..., Any], items: list[Any]) -> list[Any]:
        return [fn(item) for item in items]
