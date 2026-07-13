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
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.backends.distributed import (
    DEFAULT_CYCLE_NAME,
    DEFAULT_LEASE_DURATION_MS,
    DistributedCapabilityRegistry,
    DistributedRunEnvelope,
    RuntimeRecipe,
)
from vv_agent.runtime.cancellation import CancelledError
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
_DISPATCH_POLL_SECONDS = 0.1


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
        dispatch_timeout_seconds: float = 10 * 60,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
    ) -> None:
        if not _CELERY_AVAILABLE:
            raise ImportError("celery is required for CeleryBackend. Install with: pip install celery")
        self.celery_app = celery_app
        self.state_store = state_store
        self.runtime_recipe = runtime_recipe
        self.cycle_task_name = cycle_task_name
        if dispatch_timeout_seconds <= 0:
            raise ValueError("dispatch_timeout_seconds must be positive")
        if isinstance(lease_duration_ms, bool) or not isinstance(lease_duration_ms, int) or lease_duration_ms <= 0:
            raise ValueError("lease_duration_ms must be a positive integer")
        self.dispatch_timeout_seconds = float(dispatch_timeout_seconds)
        self.lease_duration_ms = lease_duration_ms

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
                cycle_index,
                messages,
                cycles,
                shared_state,
                ctx,
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

        initial_checkpoint = Checkpoint(
            task_id=task.task_id,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=initial_messages,
            cycles=[],
            shared_state=shared_state,
        )
        try:
            store_spec = self.state_store.state_store_spec()
        except Exception as exc:
            return self._coordination_failure(
                initial_checkpoint,
                f"failed to describe the state store for task {task.task_id}: {exc}",
            )
        if store_spec is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                messages=initial_messages,
                cycles=[],
                error="Distributed backend requires a reconstructable SQLite or Redis state store",
                shared_state=shared_state,
                token_usage=summarize_task_token_usage([]),
            )
        recipe = RuntimeRecipe.from_dict(self.runtime_recipe.to_dict())
        recipe.state_store = store_spec

        try:
            created = self.state_store.create_checkpoint(initial_checkpoint)
        except Exception as exc:
            return self._coordination_failure(
                initial_checkpoint,
                f"failed to create the initial checkpoint for task {task.task_id}: {exc}",
            )

        checkpoint = initial_checkpoint
        if not created:
            checkpoint, failure = self._load_checkpoint(
                task.task_id,
                fallback=initial_checkpoint,
                operation="loading the existing checkpoint after create conflict",
                primary_context=f"checkpoint create conflict for task {task.task_id}",
            )
            if failure is not None:
                return failure
            if checkpoint is None:
                return self._coordination_failure(
                    initial_checkpoint,
                    f"checkpoint create conflict for task {task.task_id}, but the existing checkpoint disappeared",
                )
            if checkpoint.terminal_result is not None:
                return self._replay_terminal(checkpoint, context="checkpoint create conflict")
            if self._checkpoint_claim_is_active(checkpoint):
                return self._claimed_checkpoint_failure(
                    checkpoint,
                    context="checkpoint create conflict",
                )
            if checkpoint.status != AgentStatus.RUNNING:
                return self._coordination_failure(
                    checkpoint,
                    f"checkpoint create conflict found non-running checkpoint status {checkpoint.status.value!r}",
                )

        return self._distributed_loop(
            task=task,
            ctx=ctx,
            max_cycles=max_cycles,
            recipe=recipe,
            checkpoint=checkpoint,
        )

    def _distributed_loop(
        self,
        *,
        task: AgentTask,
        ctx: ExecutionContext | None,
        max_cycles: int,
        recipe: RuntimeRecipe,
        checkpoint: Checkpoint,
    ) -> AgentResult:
        for cycle_index in range(checkpoint.cycle_index + 1, max_cycles + 1):
            cancellation_reason = self._cancellation_reason(ctx)
            if cancellation_reason is not None:
                return self._persist_scheduler_result(
                    task.task_id,
                    fallback=checkpoint,
                    primary_context=cancellation_reason,
                    result_factory=lambda current, reason=cancellation_reason: self._failed_result(current, reason),
                )

            try:
                now_ms = time.time_ns() // 1_000_000
                envelope = DistributedRunEnvelope.for_cycle(
                    task=task,
                    recipe=recipe,
                    cycle_index=cycle_index,
                    cycle_name=DEFAULT_CYCLE_NAME,
                    deadline_unix_ms=now_ms + int(self.dispatch_timeout_seconds * 1000),
                    lease_duration_ms=self.lease_duration_ms,
                )
                async_result = self.celery_app.send_task(
                    self.cycle_task_name,
                    kwargs={"envelope_dict": envelope.to_dict()},
                    serializer="json",
                )
            except Exception as exc:
                return self._handle_dispatch_error(
                    task.task_id,
                    cycle_index=cycle_index,
                    fallback=checkpoint,
                    error=exc,
                )

            remaining_seconds = envelope.remaining_seconds(now_ms=now_ms)
            assert remaining_seconds is not None
            cycle_result, cancellation_reason, dispatch_error = self._wait_for_dispatch(
                async_result,
                ctx=ctx,
                timeout=remaining_seconds,
            )
            if cancellation_reason is not None:
                return self._persist_scheduler_result(
                    task.task_id,
                    fallback=checkpoint,
                    primary_context=cancellation_reason,
                    result_factory=lambda current, reason=cancellation_reason: self._failed_result(current, reason),
                )
            if dispatch_error is not None:
                return self._handle_dispatch_error(
                    task.task_id,
                    cycle_index=cycle_index,
                    fallback=checkpoint,
                    error=dispatch_error,
                )
            if not isinstance(cycle_result, Mapping):
                return self._handle_dispatch_error(
                    task.task_id,
                    cycle_index=cycle_index,
                    fallback=checkpoint,
                    error=TypeError("dispatcher returned a non-object payload"),
                )

            finished = cycle_result.get("finished")
            if not isinstance(finished, bool):
                return self._handle_dispatch_error(
                    task.task_id,
                    cycle_index=cycle_index,
                    fallback=checkpoint,
                    error=TypeError("dispatcher payload field 'finished' must be a boolean"),
                )
            if finished:
                return self._handle_finished_dispatch(
                    task.task_id,
                    cycle_index=cycle_index,
                    fallback=checkpoint,
                    payload=cycle_result,
                )
            if cycle_result.get("result") is not None:
                return self._coordination_failure(
                    checkpoint,
                    f"Celery cycle {cycle_index} returned an unfinished payload with a result",
                )

            verified_checkpoint, failure = self._verify_unfinished_dispatch(
                task.task_id,
                cycle_index=cycle_index,
                fallback=checkpoint,
            )
            if failure is not None:
                return failure
            assert verified_checkpoint is not None
            checkpoint = verified_checkpoint

        return self._persist_scheduler_result(
            task.task_id,
            fallback=checkpoint,
            primary_context=f"task {task.task_id} reached max cycles",
            result_factory=self._max_cycles_result,
        )

    def _wait_for_dispatch(
        self,
        async_result: Any,
        *,
        ctx: ExecutionContext | None,
        timeout: float,
    ) -> tuple[Any | None, str | None, Exception | None]:
        from celery.exceptions import TimeoutError as CeleryTimeoutError

        deadline = time.monotonic() + timeout
        while True:
            cancellation_reason = self._cancellation_reason(ctx)
            if cancellation_reason is not None:
                self._revoke_dispatch(async_result)
                return None, cancellation_reason, None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, None, TimeoutError(f"dispatch timed out after {timeout:g} seconds")
            try:
                return async_result.get(timeout=min(_DISPATCH_POLL_SECONDS, remaining)), None, None
            except CeleryTimeoutError:
                continue
            except Exception as exc:
                return None, None, exc

    @staticmethod
    def _revoke_dispatch(async_result: Any) -> None:
        revoke = getattr(async_result, "revoke", None)
        if not callable(revoke):
            return
        try:
            revoke(terminate=False)
        except TypeError:
            with suppress(Exception):
                revoke()
        except Exception:
            return

    @staticmethod
    def _cancellation_reason(ctx: ExecutionContext | None) -> str | None:
        if ctx is None:
            return None
        try:
            ctx.check_cancelled()
        except CancelledError as exc:
            return str(exc).strip() or "Operation was cancelled"
        return None

    def _handle_dispatch_error(
        self,
        task_id: str,
        *,
        cycle_index: int,
        fallback: Checkpoint,
        error: Exception,
    ) -> AgentResult:
        primary_context = f"Celery cycle {cycle_index} failed: {error}"
        return self._persist_scheduler_result(
            task_id,
            fallback=fallback,
            primary_context=primary_context,
            result_factory=lambda current: self._failed_result(current, primary_context),
        )

    def _handle_finished_dispatch(
        self,
        task_id: str,
        *,
        cycle_index: int,
        fallback: Checkpoint,
        payload: Mapping[str, Any],
    ) -> AgentResult:
        context = f"Celery cycle {cycle_index} returned a finished payload"
        checkpoint, failure = self._load_checkpoint(
            task_id,
            fallback=fallback,
            operation="loading the finished checkpoint",
            primary_context=context,
        )
        if failure is not None:
            return failure
        if checkpoint is None:
            return self._coordination_failure(fallback, f"{context}, but no durable checkpoint exists")

        revision = payload.get("checkpoint_revision")
        if revision is not None:
            if isinstance(revision, bool) or not isinstance(revision, int):
                return self._coordination_failure(
                    checkpoint,
                    f"{context} with an invalid checkpoint_revision",
                )
            if checkpoint.claim_token is not None:
                return self._claimed_checkpoint_failure(checkpoint, context=context)
            if checkpoint.terminal_result is None:
                return self._coordination_failure(
                    checkpoint,
                    f"{context} at revision {revision}, but the durable checkpoint is not terminal",
                )
            dispatched_result, parse_failure = self._parse_finished_result(payload, checkpoint, context=context)
            if parse_failure is not None:
                return parse_failure
            assert dispatched_result is not None
            durable_result = checkpoint.terminal_result
            if checkpoint.revision != revision:
                return self._coordination_failure(
                    checkpoint,
                    f"{context} revision mismatch: dispatcher={revision}, durable={checkpoint.revision}",
                )
            if not self._terminal_checkpoint_is_consistent(checkpoint):
                return self._coordination_failure(
                    checkpoint,
                    f"{context}, but the durable checkpoint fields do not match its terminal result",
                )
            if dispatched_result.to_dict() != durable_result.to_dict():
                return self._coordination_failure(
                    checkpoint,
                    f"{context}, but the dispatcher result does not match the durable terminal result",
                )
            return self._acknowledge_terminal(
                checkpoint,
                durable_result,
                expected_revision=revision,
                context=context,
            )

        if checkpoint.terminal_result is not None:
            return self._replay_terminal(checkpoint, context=context)
        if checkpoint.claim_token is not None:
            return self._claimed_checkpoint_failure(checkpoint, context=context)
        if checkpoint.status != AgentStatus.RUNNING:
            return self._coordination_failure(
                checkpoint,
                f"{context}, but the durable checkpoint status is {checkpoint.status.value!r}",
            )
        dispatched_result, parse_failure = self._parse_finished_result(payload, checkpoint, context=context)
        if parse_failure is not None:
            return parse_failure
        assert dispatched_result is not None
        return self._finalize_and_ack(
            checkpoint,
            dispatched_result,
            terminal_cycle_index=cycle_index,
            context=f"persisting compatibility finished payload for cycle {cycle_index}",
        )

    def _parse_finished_result(
        self,
        payload: Mapping[str, Any],
        checkpoint: Checkpoint,
        *,
        context: str,
    ) -> tuple[AgentResult | None, AgentResult | None]:
        raw_result = payload.get("result")
        if not isinstance(raw_result, dict):
            return None, self._coordination_failure(checkpoint, f"{context} without an object result payload")
        try:
            result = AgentResult.from_dict(raw_result)
        except Exception as exc:
            return None, self._coordination_failure(checkpoint, f"{context} with an invalid result payload: {exc}")
        if result.status in {AgentStatus.PENDING, AgentStatus.RUNNING}:
            return None, self._coordination_failure(
                checkpoint,
                f"{context} with non-terminal result status {result.status.value!r}",
            )
        return result, None

    def _verify_unfinished_dispatch(
        self,
        task_id: str,
        *,
        cycle_index: int,
        fallback: Checkpoint,
    ) -> tuple[Checkpoint | None, AgentResult | None]:
        context = f"Celery cycle {cycle_index} returned unfinished"
        checkpoint, failure = self._load_checkpoint(
            task_id,
            fallback=fallback,
            operation="loading the unfinished checkpoint",
            primary_context=context,
        )
        if failure is not None:
            return None, failure
        if checkpoint is None:
            return None, self._coordination_failure(fallback, f"{context}, but no durable checkpoint exists")
        if checkpoint.terminal_result is not None:
            return None, self._replay_terminal(checkpoint, context=context)
        if checkpoint.claim_token is not None:
            return None, self._claimed_checkpoint_failure(checkpoint, context=context)
        if checkpoint.status != AgentStatus.RUNNING:
            return None, self._coordination_failure(
                checkpoint,
                f"{context}, but the durable checkpoint status is {checkpoint.status.value!r}",
            )
        if checkpoint.cycle_index != cycle_index:
            return None, self._coordination_failure(
                checkpoint,
                f"{context} without durable progress: expected cycle_index {cycle_index}, "
                f"found {checkpoint.cycle_index}",
            )
        return checkpoint, None

    def _persist_scheduler_result(
        self,
        task_id: str,
        *,
        fallback: Checkpoint,
        primary_context: str,
        result_factory: Callable[[Checkpoint], AgentResult],
    ) -> AgentResult:
        checkpoint, failure = self._load_checkpoint(
            task_id,
            fallback=fallback,
            operation="loading checkpoint before scheduler finalization",
            primary_context=primary_context,
        )
        if failure is not None:
            return failure
        if checkpoint is None:
            return self._coordination_failure(
                fallback,
                f"{primary_context}; no durable checkpoint exists for scheduler finalization",
            )
        if checkpoint.terminal_result is not None:
            return self._replay_terminal(checkpoint, context=primary_context)
        if checkpoint.claim_token is not None:
            return self._claimed_checkpoint_failure(checkpoint, context=primary_context)
        if checkpoint.status != AgentStatus.RUNNING:
            return self._coordination_failure(
                checkpoint,
                f"{primary_context}; checkpoint status {checkpoint.status.value!r} cannot be finalized",
            )
        return self._finalize_and_ack(
            checkpoint,
            result_factory(checkpoint),
            context=primary_context,
        )

    def _finalize_and_ack(
        self,
        checkpoint: Checkpoint,
        result: AgentResult,
        *,
        context: str,
        terminal_cycle_index: int | None = None,
    ) -> AgentResult:
        if checkpoint.terminal_result is not None or checkpoint.claim_token is not None:
            return self._coordination_failure(
                checkpoint,
                f"{context}; checkpoint is not eligible for scheduler finalization",
            )
        terminal = replace(
            checkpoint,
            cycle_index=checkpoint.cycle_index if terminal_cycle_index is None else terminal_cycle_index,
            status=result.status,
            messages=result.messages,
            cycles=result.cycles,
            shared_state=result.shared_state,
            terminal_result=result,
        )
        expected_revision = checkpoint.revision
        try:
            finalized = self.state_store.finalize_checkpoint(terminal, expected_revision=expected_revision)
        except Exception as exc:
            return self._coordination_failure(
                checkpoint,
                f"{context}; checkpoint finalization raised an error: {exc}",
            )
        if not finalized:
            current, failure = self._load_checkpoint(
                checkpoint.task_id,
                fallback=checkpoint,
                operation="reloading checkpoint after finalization CAS returned false",
                primary_context=context,
            )
            if failure is not None:
                return failure
            snapshot = current or checkpoint
            return self._coordination_failure(
                snapshot,
                f"{context}; checkpoint finalization CAS returned false at revision {expected_revision}",
            )

        persisted = replace(terminal, revision=expected_revision + 1)
        return self._acknowledge_terminal(
            persisted,
            result,
            expected_revision=expected_revision + 1,
            context=context,
        )

    def _replay_terminal(self, checkpoint: Checkpoint, *, context: str) -> AgentResult:
        terminal_result = checkpoint.terminal_result
        if terminal_result is None:
            return self._coordination_failure(checkpoint, f"{context}; checkpoint is not terminal")
        if not self._terminal_checkpoint_is_consistent(checkpoint):
            return self._coordination_failure(
                checkpoint,
                f"{context}; durable checkpoint fields do not match its terminal result",
            )
        return self._acknowledge_terminal(
            checkpoint,
            terminal_result,
            expected_revision=checkpoint.revision,
            context=f"replaying terminal after {context}",
        )

    def _acknowledge_terminal(
        self,
        checkpoint: Checkpoint,
        result: AgentResult,
        *,
        expected_revision: int,
        context: str,
    ) -> AgentResult:
        try:
            acknowledged = self.state_store.acknowledge_terminal(
                checkpoint.task_id,
                expected_revision=expected_revision,
            )
        except Exception as exc:
            return self._coordination_failure(
                checkpoint,
                f"{context}; terminal acknowledgement raised an error: {exc}",
            )
        if acknowledged:
            return result

        current, failure = self._load_checkpoint(
            checkpoint.task_id,
            fallback=checkpoint,
            operation="reloading checkpoint after terminal acknowledgement returned false",
            primary_context=context,
        )
        if failure is not None:
            return failure
        if current is None:
            return result
        return self._coordination_failure(
            current,
            f"{context}; terminal acknowledgement returned false and checkpoint still exists "
            f"at revision {current.revision}",
        )

    def _load_checkpoint(
        self,
        task_id: str,
        *,
        fallback: Checkpoint,
        operation: str,
        primary_context: str,
    ) -> tuple[Checkpoint | None, AgentResult | None]:
        try:
            return self.state_store.load_checkpoint(task_id), None
        except Exception as exc:
            return None, self._coordination_failure(
                fallback,
                f"{primary_context}; {operation} failed for task {task_id}: {exc}",
            )

    @staticmethod
    def _terminal_checkpoint_is_consistent(checkpoint: Checkpoint) -> bool:
        result = checkpoint.terminal_result
        return bool(
            result is not None
            and result.status not in {AgentStatus.PENDING, AgentStatus.RUNNING}
            and checkpoint.status == result.status
            and checkpoint.messages == result.messages
            and checkpoint.cycles == result.cycles
            and checkpoint.shared_state == result.shared_state
        )

    @staticmethod
    def _failed_result(checkpoint: Checkpoint, error: str) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            messages=checkpoint.messages,
            cycles=checkpoint.cycles,
            error=error,
            shared_state=checkpoint.shared_state,
            token_usage=summarize_task_token_usage(checkpoint.cycles),
        )

    @staticmethod
    def _max_cycles_result(checkpoint: Checkpoint) -> AgentResult:
        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=checkpoint.messages,
            cycles=checkpoint.cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=checkpoint.shared_state,
            token_usage=summarize_task_token_usage(checkpoint.cycles),
        )

    @staticmethod
    def _coordination_failure(checkpoint: Checkpoint, detail: str) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            messages=checkpoint.messages,
            cycles=checkpoint.cycles,
            error=f"Distributed coordination failure: {detail}",
            shared_state=checkpoint.shared_state,
            token_usage=summarize_task_token_usage(checkpoint.cycles),
        )

    def _claimed_checkpoint_failure(self, checkpoint: Checkpoint, *, context: str) -> AgentResult:
        return self._coordination_failure(
            checkpoint,
            f"{context}; checkpoint is in progress under worker claim {checkpoint.claim_token!r}; "
            "the durable outcome is uncertain and the checkpoint was not overwritten",
        )

    @staticmethod
    def _checkpoint_claim_is_active(checkpoint: Checkpoint) -> bool:
        if checkpoint.claim_token is None:
            return False
        lease_expires_at_ms = checkpoint.lease_expires_at_ms
        return lease_expires_at_ms is None or lease_expires_at_ms > time.time_ns() // 1_000_000

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
        signature = getattr(fn, "s", None)
        if callable(signature):
            from celery import group

            job = group(signature(item) for item in items)
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
    capability_registry: DistributedCapabilityRegistry | None = None,
) -> Any:
    """Register the ``run_single_cycle`` task on *celery_app*.

    Returns the registered Celery task object.
    """
    from vv_agent.runtime.backends.celery_tasks import run_single_cycle

    def worker_task(*, envelope_dict: dict[str, Any]) -> dict[str, Any]:
        return run_single_cycle(envelope_dict=envelope_dict, capability_registry=capability_registry)

    return celery_app.task(name=task_name, bind=False)(worker_task)
