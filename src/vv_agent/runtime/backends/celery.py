"""Celery-backed distributed cycle execution."""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from typing import Any

from vv_agent.budget import RunBudgetLimits
from vv_agent.checkpoint import CheckpointError
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.backends.distributed import (
    DEFAULT_CYCLE_NAME,
    DEFAULT_LEASE_DURATION_MS,
    ClaimMode,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedCheckpointConfig,
    DistributedContractError,
    DistributedRunEnvelope,
    DistributedWorkerResponse,
    RuntimeRecipe,
)
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    CycleRecord,
    Message,
    _last_assistant_output,
)

_CELERY_AVAILABLE = importlib.util.find_spec("celery") is not None
_DISPATCH_POLL_SECONDS = 0.1


class CeleryBackend:
    """Backend that dispatches cycles as Celery tasks with checkpoint persistence.

    Parameters
    ----------
    celery_app:
        A configured Celery application instance.
    runtime_recipe:
        Worker reconstruction recipe. Durable dependencies are resolved from
        its capability references.
    cycle_task_name:
        The registered Celery task name for single-cycle execution on workers.
    """

    def __init__(
        self,
        celery_app: Any,
        *,
        runtime_recipe: RuntimeRecipe,
        cycle_task_name: str = "vv_agent.celery_tasks.run_single_cycle",
        dispatch_timeout_seconds: float = 10 * 60,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
    ) -> None:
        if not _CELERY_AVAILABLE:
            raise ImportError("celery is required for CeleryBackend. Install with: pip install celery")
        if not isinstance(runtime_recipe, RuntimeRecipe):
            raise TypeError("runtime_recipe must be a RuntimeRecipe")
        if runtime_recipe.capabilities.checkpoint_store_ref is None:
            raise DistributedContractError("CeleryBackend runtime_recipe requires checkpoint_store_ref")
        self.celery_app = celery_app
        self.runtime_recipe = runtime_recipe
        self.cycle_task_name = cycle_task_name
        if dispatch_timeout_seconds <= 0:
            raise ValueError("dispatch_timeout_seconds must be positive")
        if isinstance(lease_duration_ms, bool) or not isinstance(lease_duration_ms, int) or lease_duration_ms <= 0:
            raise ValueError("lease_duration_ms must be a positive integer")
        self.dispatch_timeout_seconds = float(dispatch_timeout_seconds)
        self.lease_duration_ms = lease_duration_ms

    @property
    def manages_run_budget(self) -> bool:
        return True

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
        del cycle_executor
        checkpoint_controller = ctx.metadata.get("_vv_agent_checkpoint_controller") if ctx is not None else None
        if not isinstance(checkpoint_controller, CheckpointResumeController):
            raise DistributedContractError("CeleryBackend requires RunConfig.checkpoint_config with a current checkpoint store")
        return self._execute_distributed(
            task=task,
            initial_messages=initial_messages,
            shared_state=shared_state,
            ctx=ctx,
            max_cycles=max_cycles,
            checkpoint_controller=checkpoint_controller,
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
        checkpoint_controller: CheckpointResumeController,
    ) -> AgentResult:
        assert ctx is not None
        budget_limits = ctx.metadata.get("_vv_agent_budget_limits")
        if not isinstance(budget_limits, RunBudgetLimits):
            budget_limits = None
        recipe = RuntimeRecipe.from_dict(self.runtime_recipe.to_dict())
        distributed_task = deepcopy(task)
        effective_model_settings = ctx.metadata.get("_vv_agent_model_settings")
        if isinstance(effective_model_settings, ModelSettings):
            distributed_task.model_settings = deepcopy(effective_model_settings)
        messages = initial_messages
        cycles: list[CycleRecord] = []
        snapshot_provider = ctx.metadata.get("_vv_agent_checkpoint_budget_snapshot")
        messages, cycles, shared_state, start_cycle = checkpoint_controller.bind_runtime_state(
            messages=messages,
            cycles=cycles,
            shared_state=shared_state,
            budget_snapshot_provider=(snapshot_provider if callable(snapshot_provider) else None),
        )
        first_claim_mode = checkpoint_controller.next_claim_mode

        for cycle_index in range(start_cycle, max_cycles + 1):
            cancellation_reason = self._cancellation_reason(ctx)
            if cancellation_reason is not None:
                current = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
                if current is not None and current.claim_token is not None:
                    raise CheckpointError(
                        "distributed cancellation observed while a worker still owns the checkpoint",
                        code="checkpoint_claim_active",
                    )
                return AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    partial_output=_last_assistant_output(cycles),
                    messages=messages,
                    cycles=cycles,
                    error=cancellation_reason,
                    shared_state=shared_state,
                    token_usage=summarize_task_token_usage(current.model_calls if current is not None else []),
                    budget_usage=(current.budget_usage if current is not None else None),
                )

            deadline_unix_ms = time.time_ns() // 1_000_000 + int(self.dispatch_timeout_seconds * 1000)
            response, cancellation_reason = self._dispatch_cycle(
                task=distributed_task,
                recipe=recipe,
                cycle_index=cycle_index,
                deadline_unix_ms=deadline_unix_ms,
                claim_mode=(first_claim_mode if cycle_index == start_cycle else "continue"),
                budget_limits=budget_limits,
                checkpoint_controller=checkpoint_controller,
                ctx=ctx,
            )
            if cancellation_reason is not None:
                current = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
                if current is not None and current.claim_token is not None:
                    raise CheckpointError(
                        "distributed cancellation left an active worker claim",
                        code="checkpoint_claim_active",
                    )
                return AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    partial_output=_last_assistant_output(cycles),
                    messages=messages,
                    cycles=cycles,
                    error=cancellation_reason,
                    shared_state=shared_state,
                    token_usage=summarize_task_token_usage(current.model_calls if current is not None else []),
                    budget_usage=(current.budget_usage if current is not None else None),
                )
            assert response is not None
            if response.is_terminal:
                return self._handle_terminal_response(
                    response=response,
                    cycle_index=cycle_index,
                    checkpoint_controller=checkpoint_controller,
                )
            if response.response_type != "committed":
                raise CheckpointError(
                    "distributed worker response did not commit durable progress",
                    code="checkpoint_store_conflict",
                )
            checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
            if checkpoint is None:
                raise CheckpointError(
                    "checkpoint disappeared after distributed cycle commit",
                    code="checkpoint_not_found",
                )
            if (
                checkpoint.terminal_result is not None
                or checkpoint.claim_token is not None
                or checkpoint.status is not AgentStatus.RUNNING
                or checkpoint.cycle_index != cycle_index
            ):
                raise CheckpointError(
                    "distributed worker unfinished payload does not match durable progress",
                    code="checkpoint_store_conflict",
                )
            if response.checkpoint_revision != checkpoint.revision or response.committed_cycle != checkpoint.cycle_index:
                raise CheckpointError(
                    "distributed worker progress revision or cycle does not match the checkpoint",
                    code="checkpoint_store_conflict",
                )
            messages[:] = deepcopy(checkpoint.messages)
            cycles[:] = deepcopy(checkpoint.cycles)
            shared_state.clear()
            shared_state.update(deepcopy(checkpoint.shared_state))
            checkpoint_controller.checkpoint = checkpoint
            ctx.model_call_ledger.replace(checkpoint.model_calls)
            checkpoint_controller.set_next_claim_mode("continue")

        current = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            completion_reason=CompletionReason.MAX_CYCLES,
            partial_output=_last_assistant_output(cycles),
            messages=messages,
            cycles=cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=shared_state,
            token_usage=summarize_task_token_usage(current.model_calls if current is not None else []),
            budget_usage=(current.budget_usage if current is not None else None),
        )

    def _dispatch_cycle(
        self,
        *,
        task: AgentTask,
        recipe: RuntimeRecipe,
        cycle_index: int,
        deadline_unix_ms: int,
        claim_mode: ClaimMode,
        budget_limits: RunBudgetLimits | None,
        checkpoint_controller: CheckpointResumeController,
        ctx: ExecutionContext,
    ) -> tuple[DistributedWorkerResponse | None, str | None]:
        last_error: Exception | None = None
        effective_claim_mode = claim_mode
        while True:
            now_ms = time.time_ns() // 1_000_000
            if now_ms >= deadline_unix_ms:
                detail = f": {last_error}" if last_error is not None else ""
                raise CheckpointError(
                    f"distributed cycle {cycle_index} exhausted its dispatch deadline{detail}",
                    code="checkpoint_dispatch_failed",
                )
            checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
            if checkpoint is None:
                raise CheckpointError(
                    "checkpoint disappeared before distributed dispatch",
                    code="checkpoint_not_found",
                )
            if checkpoint.terminal_result is not None:
                return DistributedWorkerResponse.terminal_replay(
                    checkpoint_revision=checkpoint.revision,
                    result=checkpoint.terminal_result,
                ), None
            if checkpoint.cycle_index >= cycle_index:
                if checkpoint.cycle_index != cycle_index or checkpoint.claim_token is not None:
                    raise CheckpointError(
                        "distributed checkpoint advanced beyond the dispatched cycle",
                        code="checkpoint_cycle_conflict",
                    )
                return DistributedWorkerResponse.committed(
                    checkpoint_revision=checkpoint.revision,
                    committed_cycle=checkpoint.cycle_index,
                ), None
            if checkpoint.claim_token is not None:
                if (checkpoint.lease_expires_at_ms or 0) > now_ms:
                    cancellation_reason = self._cancellation_reason(ctx)
                    if cancellation_reason is not None:
                        return None, cancellation_reason
                    time.sleep(
                        min(
                            _DISPATCH_POLL_SECONDS,
                            max(0.001, ((checkpoint.lease_expires_at_ms or now_ms) - now_ms) / 1000),
                        )
                    )
                    continue
                effective_claim_mode = "recovery"
            elif checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED or last_error is not None:
                effective_claim_mode = "recovery"

            config = DistributedCheckpointConfig.from_checkpoint_config(checkpoint_controller.config)
            envelope = DistributedRunEnvelope.for_cycle(
                task=task,
                recipe=recipe,
                cycle_index=cycle_index,
                root_run_id=checkpoint.root_run_id,
                trace_id=checkpoint.trace_id,
                run_definition_digest=checkpoint.run_definition_digest,
                claim_mode=effective_claim_mode,
                resume_attempt=checkpoint.resume_attempt,
                checkpoint_config=config,
                cycle_name=DEFAULT_CYCLE_NAME,
                run_id=checkpoint.root_run_id,
                deadline_unix_ms=deadline_unix_ms,
                lease_duration_ms=self.lease_duration_ms,
                budget_limits=budget_limits,
            )
            try:
                async_result = self.celery_app.send_task(
                    self.cycle_task_name,
                    kwargs={"envelope_dict": envelope.to_dict()},
                    serializer="json",
                    task_id=envelope.job_id,
                )
                remaining_seconds = envelope.remaining_seconds()
                assert remaining_seconds is not None
                result, cancellation_reason, dispatch_error = self._wait_for_dispatch(
                    async_result,
                    ctx=ctx,
                    timeout=remaining_seconds,
                )
            except Exception as exc:
                result = None
                cancellation_reason = None
                dispatch_error = exc
            if cancellation_reason is not None:
                return None, cancellation_reason
            if dispatch_error is None:
                try:
                    response = DistributedWorkerResponse.from_dict(result)
                except DistributedContractError as exc:
                    raise CheckpointError(
                        f"distributed dispatcher returned an invalid worker response: {exc}",
                        code="checkpoint_store_conflict",
                    ) from exc
                if response.response_type == "pending":
                    last_error = CheckpointError(
                        "distributed worker reported pending delivery without committed state",
                        code="checkpoint_store_conflict",
                    )
                    effective_claim_mode = "recovery"
                    time.sleep(0.001)
                    continue
                return response, None
            if not self._is_retryable_dispatch_error(dispatch_error):
                raise dispatch_error
            last_error = dispatch_error

    @staticmethod
    def _is_retryable_dispatch_error(error: Exception) -> bool:
        if isinstance(error, CheckpointError):
            return error.code in {
                "checkpoint_claim_active",
                "checkpoint_lease_lost",
                "checkpoint_store_conflict",
            }
        return not isinstance(
            error,
            (DistributedCapabilityError, DistributedContractError, TypeError, ValueError),
        )

    def _handle_terminal_response(
        self,
        *,
        response: DistributedWorkerResponse,
        cycle_index: int,
        checkpoint_controller: CheckpointResumeController,
    ) -> AgentResult:
        checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError(
                "checkpoint disappeared before terminal candidate verification",
                code="checkpoint_not_found",
            )
        result = response.result
        assert result is not None
        if response.response_type == "terminal_replay":
            if (
                checkpoint.terminal_result is None
                or response.checkpoint_revision != checkpoint.revision
                or checkpoint.terminal_result.to_dict() != result.to_dict()
            ):
                raise CheckpointError(
                    "distributed terminal replay does not match the durable checkpoint",
                    code="checkpoint_store_conflict",
                )
            checkpoint_controller.checkpoint = checkpoint
            return deepcopy(checkpoint.terminal_result)
        if response.response_type != "terminal_candidate":
            raise CheckpointError(
                "distributed worker returned a terminal without candidate semantics",
                code="checkpoint_store_conflict",
            )
        if response.checkpoint_revision != checkpoint.revision:
            raise CheckpointError(
                "distributed terminal candidate revision does not match the checkpoint",
                code="checkpoint_store_conflict",
            )
        if result.status is AgentStatus.RECONCILIATION_REQUIRED:
            if checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED or checkpoint.claim_token is not None:
                raise CheckpointError(
                    "distributed reconciliation candidate does not match durable state",
                    code="checkpoint_store_conflict",
                )
            checkpoint_controller.checkpoint = checkpoint
            return result
        if checkpoint.terminal_result is not None:
            raise CheckpointError(
                "distributed terminal candidate cannot replace a durable terminal",
                code="checkpoint_store_conflict",
            )
        if checkpoint.claim_token is not None and checkpoint.claimed_cycle != cycle_index:
            raise CheckpointError(
                "distributed terminal candidate belongs to a different claimed cycle",
                code="checkpoint_store_conflict",
            )
        if result.cycles and result.cycles[-1].index != cycle_index:
            raise CheckpointError(
                "distributed terminal candidate does not contain the dispatched cycle",
                code="checkpoint_cycle_conflict",
            )
        checkpoint_controller.checkpoint = checkpoint
        if checkpoint.claim_token is not None:
            checkpoint_controller.adopt_claim_for_terminal_finalize(
                claim_token=checkpoint.claim_token,
                lease_duration_ms=self.lease_duration_ms,
            )
        return result

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

    def worker_task(task: Any, *, envelope_dict: dict[str, Any]) -> dict[str, Any]:
        request = getattr(task, "request", None)
        delivery_info = getattr(request, "delivery_info", None)
        redelivered = bool(isinstance(delivery_info, Mapping) and delivery_info.get("redelivered") is True)
        retries = getattr(request, "retries", 0)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            retries = 0
        return run_single_cycle(
            envelope_dict=envelope_dict,
            capability_registry=capability_registry,
            transport_redelivered=redelivered,
            transport_retry_count=retries,
        )

    return celery_app.task(name=task_name, bind=True)(worker_task)
