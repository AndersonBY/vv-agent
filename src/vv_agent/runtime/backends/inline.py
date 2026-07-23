from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    CycleRecord,
    Message,
    TaskTokenUsage,
    _last_assistant_output,
)


def _task_token_usage(ctx: ExecutionContext | None) -> TaskTokenUsage:
    return ctx.model_call_ledger.usage() if ctx is not None else TaskTokenUsage()


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
        start_cycle = 1
        checkpoint_controller = (
            ctx.metadata.get("_vv_agent_checkpoint_controller") if ctx is not None else None
        )
        if isinstance(checkpoint_controller, CheckpointResumeController):
            assert ctx is not None
            snapshot_provider = ctx.metadata.get("_vv_agent_checkpoint_budget_snapshot")
            messages, cycles, shared_state, start_cycle = checkpoint_controller.bind_runtime_state(
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
                budget_snapshot_provider=(snapshot_provider if callable(snapshot_provider) else None),
            )

        for cycle_index in range(start_cycle, max_cycles + 1):
            if ctx is not None:
                try:
                    ctx.check_cancelled()
                except Exception:
                    return AgentResult(
                        status=AgentStatus.FAILED,
                        completion_reason=CompletionReason.CANCELLED,
                        partial_output=_last_assistant_output(cycles),
                        messages=messages,
                        cycles=cycles,
                        error="Operation was cancelled",
                        shared_state=shared_state,
                        token_usage=_task_token_usage(ctx),
                    )

            result = cycle_executor(cycle_index, messages, cycles, shared_state, ctx)
            if (
                result is None
                and
                isinstance(checkpoint_controller, CheckpointResumeController)
                and cycles
                and cycles[-1].index == cycle_index
            ):
                checkpoint_controller.commit_cycle(
                    cycle_index=cycle_index,
                    messages=messages,
                    cycles=cycles,
                    shared_state=shared_state,
                )
            if result is not None:
                return result

        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            completion_reason=CompletionReason.MAX_CYCLES,
            partial_output=_last_assistant_output(cycles),
            messages=messages,
            cycles=cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=shared_state,
            token_usage=_task_token_usage(ctx),
        )

    def parallel_map(self, fn: Callable[..., Any], items: list[Any]) -> list[Any]:
        return [fn(item) for item in items]
