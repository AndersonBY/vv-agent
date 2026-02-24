"""CeleryBackend — optional Celery-based execution backend.

Supports two modes:
- **Distributed** (``runtime_recipe`` provided): each cycle is dispatched as
  an independent Celery task; the worker rebuilds the AgentRuntime from the
  recipe and executes a single cycle.
- **Inline fallback** (no recipe): cycles run synchronously in the calling
  process, identical to InlineBackend.

Requires ``celery`` to be installed.  Import will fail gracefully if celery
is not available.
"""
from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.state import Checkpoint, StateStore
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CycleRecord,
    Message,
)

_CELERY_AVAILABLE = importlib.util.find_spec("celery") is not None
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeRecipe:
    """Pure-data description of how to rebuild an AgentRuntime on a worker.

    Every field must be JSON-serialisable so it can travel through Celery's
    message broker.
    """

    settings_file: str
    backend: str
    model: str
    workspace: str
    timeout_seconds: float = 90.0
    hook_class_paths: list[str] = field(default_factory=list)
    log_preview_chars: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_file": self.settings_file,
            "backend": self.backend,
            "model": self.model,
            "workspace": self.workspace,
            "timeout_seconds": self.timeout_seconds,
            "hook_class_paths": list(self.hook_class_paths),
            "log_preview_chars": self.log_preview_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeRecipe:
        return cls(
            settings_file=data["settings_file"],
            backend=data["backend"],
            model=data["model"],
            workspace=data["workspace"],
            timeout_seconds=data.get("timeout_seconds", 90.0),
            hook_class_paths=list(data.get("hook_class_paths", [])),
            log_preview_chars=data.get("log_preview_chars"),
        )


class CeleryBackend:
    """Backend that dispatches cycles as Celery tasks with checkpoint persistence.

    Parameters
    ----------
    celery_app:
        A configured Celery application instance.
    state_store:
        Shared StateStore accessible by both the scheduler and workers.
    runtime_recipe:
        If provided, enables distributed mode where each cycle is dispatched
        as an independent Celery task.  If ``None``, falls back to inline
        synchronous execution.
    cycle_task_name:
        The registered Celery task name for single-cycle execution on workers.
    """

    def __init__(
        self,
        celery_app: Any,
        state_store: StateStore,
        *,
        runtime_recipe: RuntimeRecipe | None = None,
        cycle_task_name: str = "vv_agent.celery_tasks.run_single_cycle",
    ) -> None:
        if not _CELERY_AVAILABLE:
            raise ImportError(
                "celery is required for CeleryBackend. Install with: pip install celery"
            )
        self.celery_app = celery_app
        self.state_store = state_store
        self.runtime_recipe = runtime_recipe
        self.cycle_task_name = cycle_task_name

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
        if self.runtime_recipe is not None:
            return self._execute_distributed(
                task=task,
                initial_messages=initial_messages,
                shared_state=shared_state,
                ctx=ctx,
                max_cycles=max_cycles,
            )
        return self._execute_inline(
            task=task,
            initial_messages=initial_messages,
            shared_state=shared_state,
            cycle_executor=cycle_executor,
            ctx=ctx,
            max_cycles=max_cycles,
        )

    # ------------------------------------------------------------------
    # Inline fallback (backward-compatible, identical to InlineBackend)
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_inline(
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

            result = cycle_executor(
                cycle_index, messages, cycles, shared_state, ctx,
            )
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

    # ------------------------------------------------------------------
    # Distributed mode: each cycle → independent Celery task
    # ------------------------------------------------------------------

    def _execute_distributed(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message],
        shared_state: dict[str, Any],
        ctx: ExecutionContext | None,
        max_cycles: int,
    ) -> AgentResult:
        assert self.runtime_recipe is not None

        # Save initial checkpoint so the first worker can load it.
        self.state_store.save_checkpoint(
            Checkpoint(
                task_id=task.task_id,
                cycle_index=0,
                status=AgentStatus.RUNNING,
                messages=initial_messages,
                cycles=[],
                shared_state=shared_state,
            )
        )

        try:
            return self._distributed_loop(
                task=task,
                ctx=ctx,
                max_cycles=max_cycles,
            )
        finally:
            # Best-effort cleanup; checkpoint may already be deleted by
            # the last worker cycle on success.
            try:
                self.state_store.delete_checkpoint(task.task_id)
            except Exception:
                logger.debug(
                    "Failed to clean up checkpoint for %s", task.task_id,
                    exc_info=True,
                )

    def _distributed_loop(
        self,
        *,
        task: AgentTask,
        ctx: ExecutionContext | None,
        max_cycles: int,
    ) -> AgentResult:
        assert self.runtime_recipe is not None

        for cycle_index in range(1, max_cycles + 1):
            # Check cancellation before dispatching each cycle.
            if ctx is not None:
                try:
                    ctx.check_cancelled()
                except Exception:
                    cp = self.state_store.load_checkpoint(task.task_id)
                    return AgentResult(
                        status=AgentStatus.FAILED,
                        messages=cp.messages if cp else [],
                        cycles=cp.cycles if cp else [],
                        error="Operation was cancelled",
                        shared_state=cp.shared_state if cp else {},
                        token_usage=summarize_task_token_usage(
                            cp.cycles if cp else [],
                        ),
                    )

            # Dispatch a single cycle to a Celery worker.
            async_result = self.celery_app.send_task(
                self.cycle_task_name,
                kwargs={
                    "task_dict": task.to_dict(),
                    "recipe_dict": self.runtime_recipe.to_dict(),
                    "cycle_index": cycle_index,
                },
                serializer="json",
            )

            try:
                cycle_result: dict[str, Any] = async_result.get()
            except Exception as exc:
                cp = self.state_store.load_checkpoint(task.task_id)
                return AgentResult(
                    status=AgentStatus.FAILED,
                    messages=cp.messages if cp else [],
                    cycles=cp.cycles if cp else [],
                    error=f"Celery cycle {cycle_index} failed: {exc}",
                    shared_state=cp.shared_state if cp else {},
                    token_usage=summarize_task_token_usage(
                        cp.cycles if cp else [],
                    ),
                )

            if cycle_result.get("finished"):
                return AgentResult.from_dict(cycle_result["result"])

        # Exhausted all cycles without a terminal result.
        cp = self.state_store.load_checkpoint(task.task_id)
        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=cp.messages if cp else [],
            cycles=cp.cycles if cp else [],
            final_answer="Reached max cycles without finish signal.",
            shared_state=cp.shared_state if cp else {},
            token_usage=summarize_task_token_usage(cp.cycles if cp else []),
        )

    # ------------------------------------------------------------------
    # parallel_map (unchanged)
    # ------------------------------------------------------------------

    def parallel_map(
        self,
        fn: Callable[..., Any],
        items: list[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """Run *fn* over *items* in parallel using ``celery.group``.

        If *fn* looks like a Celery task (has a ``.s`` signature method), we
        build a :func:`celery.group` and execute it.  Otherwise we fall back
        to serial execution so plain callables still work.
        """
        if hasattr(fn, "s"):
            from celery import group

            job = group(fn.s(item) for item in items)  # type: ignore[attr-defined]
            result = job.apply_async()
            return result.get(timeout=timeout)
        return [fn(item) for item in items]


# ------------------------------------------------------------------
# Helper: register the worker-side cycle task on a Celery app
# ------------------------------------------------------------------

def register_cycle_task(
    celery_app: Any,
    *,
    task_name: str = "vv_agent.celery_tasks.run_single_cycle",
) -> Any:
    """Register the ``run_single_cycle`` task on *celery_app*.

    Returns the registered Celery task object.
    """
    from vv_agent.runtime.backends.celery_tasks import run_single_cycle

    return celery_app.task(name=task_name, bind=False)(run_single_cycle)
