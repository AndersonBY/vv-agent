"""Celery-backed distributed cycle execution."""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

from vv_agent.budget import RunBudgetLimits
from vv_agent.checkpoint import CheckpointError
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.backends.base import CycleExecutor
from vv_agent.runtime.backends.distributed import (
    DEFAULT_CYCLE_NAME,
    DEFAULT_LEASE_DURATION_MS,
    ClaimMode,
    DistributedAdvanceDecision,
    DistributedCapabilityError,
    DistributedCapabilityRegistry,
    DistributedCheckpointConfig,
    DistributedContractError,
    DistributedDeliveryOutcome,
    DistributedRunEnvelope,
    DistributedRunHandle,
    DistributedWaitReason,
    DistributedWorkerResponse,
    RuntimeRecipe,
    ToolsetRef,
    toolset_schema_digest,
)
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.dispatch_outbox import (
    DispatchOutboxClaim,
    DispatchOutboxRecord,
    DispatchOutboxStore,
    dispatch_envelope_digest,
)
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
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
    dispatch_outbox_store:
        Optional transport-owned receipt store. When omitted, enqueue uses a
        stable Celery task id and relies on the worker checkpoint claim/CAS
        boundary for duplicate-delivery safety; the core ``CheckpointStore``
        is never inspected for dispatch receipts.
    """

    def __init__(
        self,
        celery_app: Any,
        *,
        runtime_recipe: RuntimeRecipe,
        cycle_task_name: str = "vv_agent.celery_tasks.run_single_cycle",
        dispatch_timeout_seconds: float = 10 * 60,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
        capability_registry: DistributedCapabilityRegistry | None = None,
        dispatch_outbox_store: DispatchOutboxStore | None = None,
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
        if dispatch_outbox_store is not None and not isinstance(dispatch_outbox_store, DispatchOutboxStore):
            raise TypeError("dispatch_outbox_store must implement the transport receipt protocol")
        self.dispatch_timeout_seconds = float(dispatch_timeout_seconds)
        self.lease_duration_ms = lease_duration_ms
        self.capability_registry = capability_registry
        self.dispatch_outbox_store = dispatch_outbox_store

    @property
    def manages_run_budget(self) -> bool:
        return True

    def execute_local(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message],
        shared_state: dict[str, Any],
        cycle_executor: CycleExecutor,
        ctx: ExecutionContext | None,
        max_cycles: int,
    ) -> AgentResult:
        """Drive immediate or completed Celery responses through ``advance``."""
        del cycle_executor
        checkpoint_controller = ctx.metadata.get("_vv_agent_checkpoint_controller") if ctx is not None else None
        if not isinstance(checkpoint_controller, CheckpointResumeController):
            raise DistributedContractError("CeleryBackend requires a checkpoint controller for synchronous execution")
        budget_limits = self._budget_limits(ctx)
        recipe = RuntimeRecipe.from_dict(self.runtime_recipe.to_dict())
        distributed_task = self._distributed_task(task, ctx)
        _, _, _, cycle_index = checkpoint_controller.bind_runtime_state(
            messages=initial_messages,
            cycles=[],
            shared_state=shared_state,
            budget_snapshot_provider=None,
        )
        claim_mode = checkpoint_controller.next_claim_mode
        pending_envelope: DistributedRunEnvelope | None = None
        registry = self.capability_registry or getattr(self.celery_app, "capability_registry", None)
        registry = registry or getattr(self.celery_app, "registry", None)
        if not isinstance(registry, DistributedCapabilityRegistry):
            raise DistributedContractError("local Celery execution requires a capability registry")
        self.capability_registry = registry
        while True:
            try:
                if ctx is not None:
                    ctx.check_cancelled()
            except CancelledError as exc:
                return self._local_cancelled(checkpoint_controller, exc)
            checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
            if checkpoint is None:
                raise CheckpointError("checkpoint disappeared before distributed dispatch", code="checkpoint_not_found")
            if checkpoint.terminal_result is not None:
                return self._handle_terminal_response(
                    response=DistributedWorkerResponse.terminal_replay(
                        checkpoint_revision=checkpoint.revision,
                        result=checkpoint.terminal_result,
                    ),
                    cycle_index=checkpoint.cycle_index,
                    checkpoint_controller=checkpoint_controller,
                )
            if cycle_index > max_cycles:
                return AgentResult(
                    status=AgentStatus.MAX_CYCLES,
                    completion_reason=CompletionReason.MAX_CYCLES,
                    partial_output=_last_assistant_output(checkpoint.cycles),
                    messages=deepcopy(checkpoint.messages),
                    cycles=deepcopy(checkpoint.cycles),
                    final_answer="Reached max cycles without finish signal.",
                    shared_state=deepcopy(checkpoint.shared_state),
                    token_usage=summarize_task_token_usage(checkpoint.model_calls),
                    budget_usage=deepcopy(checkpoint.budget_usage),
                )
            envelope = pending_envelope
            pending_envelope = None
            if envelope is None:
                envelope = self._envelope_from_checkpoint(
                    task=distributed_task,
                    recipe=recipe,
                    checkpoint=checkpoint,
                    checkpoint_config=DistributedCheckpointConfig.from_checkpoint_config(checkpoint_controller.config),
                    cycle_index=cycle_index,
                    claim_mode=claim_mode,
                    budget_limits=budget_limits,
                )
            try:
                raw_response = self._enqueue_envelope(
                    envelope,
                    handle=self._handle_for_checkpoint(checkpoint),
                    continuation=None,
                )
            except Exception as exc:
                if isinstance(exc, CancelledError):
                    return self._local_cancelled(checkpoint_controller, exc)
                if isinstance(exc, (DistributedCapabilityError, DistributedContractError, TypeError, ValueError)):
                    raise
                outcome = exc
            else:
                try:
                    response_payload = self._wait_local_response(raw_response, ctx=ctx, envelope=envelope)
                except Exception as exc:
                    if isinstance(exc, CancelledError):
                        return self._local_cancelled(checkpoint_controller, exc)
                    outcome = exc
                else:
                    if isinstance(response_payload, BaseException):
                        outcome = response_payload
                    else:
                        if not isinstance(response_payload, dict):
                            raise DistributedContractError("local Celery execution requires a worker response mapping")
                        outcome = DistributedWorkerResponse.from_dict(response_payload)
            decision = self.advance(previous_envelope=envelope, outcome=outcome, enqueue=False)
            if decision.action == "terminal_replay":
                assert decision.result is not None
                return self._handle_terminal_response(
                    response=DistributedWorkerResponse.terminal_replay(
                        checkpoint_revision=decision.checkpoint_revision or 0,
                        result=decision.result,
                    ),
                    cycle_index=cycle_index,
                    checkpoint_controller=checkpoint_controller,
                )
            if decision.action == "finalize_required":
                if isinstance(outcome, DistributedWorkerResponse) and outcome.response_type == "terminal_candidate":
                    return self._handle_terminal_response(
                        response=outcome,
                        cycle_index=cycle_index,
                        checkpoint_controller=checkpoint_controller,
                    )
                assert decision.result is not None
                return decision.result
            if decision.action == "wait":
                current = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
                if current is None:
                    raise CheckpointError("checkpoint disappeared after a wait decision", code="checkpoint_not_found")
                wait_reason = {
                    DistributedWaitReason.DEFERRED_PENDING: "deferred_pending",
                    DistributedWaitReason.HOST_INTERACTION: "host_interaction",
                    DistributedWaitReason.SUSPENDED: "suspended",
                    DistributedWaitReason.RECONCILIATION_REQUIRED: "reconciliation_required",
                }.get(decision.reason or DistributedWaitReason.DEFERRED_PENDING, "deferred_pending")
                return AgentResult(
                    status=current.status,
                    messages=deepcopy(current.messages),
                    cycles=deepcopy(current.cycles),
                    shared_state=deepcopy(current.shared_state),
                    token_usage=summarize_task_token_usage(current.model_calls),
                    budget_usage=deepcopy(current.budget_usage),
                    wait_reason=wait_reason,
                    checkpoint_key=current.checkpoint_key,
                )
            if decision.envelope is None:
                raise CheckpointError("distributed advance returned no next envelope", code="checkpoint_store_conflict")
            next_envelope = decision.envelope
            retrying_delivery = isinstance(outcome, Exception) or (
                isinstance(outcome, DistributedWorkerResponse) and outcome.response_type == "pending"
            )
            if retrying_delivery and envelope.deadline_unix_ms is not None:
                if (time.time_ns() // 1_000_000) >= envelope.deadline_unix_ms:
                    return self._local_transport_timeout(checkpoint_controller, envelope)
                next_envelope = replace(next_envelope, deadline_unix_ms=envelope.deadline_unix_ms)
            if decision.not_before_unix_ms is not None:
                try:
                    self._wait_local_until(
                        decision.not_before_unix_ms,
                        ctx=ctx,
                        deadline_unix_ms=next_envelope.deadline_unix_ms,
                    )
                except CancelledError as exc:
                    return self._local_cancelled(checkpoint_controller, exc)
                except TimeoutError:
                    return self._local_transport_timeout(checkpoint_controller, next_envelope)
            pending_envelope = next_envelope
            cycle_index = next_envelope.cycle_index
            claim_mode = next_envelope.claim_mode or "continue"
            checkpoint_controller.set_next_claim_mode(claim_mode)

    @staticmethod
    def _wait_local_response(
        raw_response: Any,
        *,
        ctx: ExecutionContext | None,
        envelope: DistributedRunEnvelope,
    ) -> Any:
        if isinstance(raw_response, dict):
            return raw_response
        ready = getattr(raw_response, "ready", None)
        get = getattr(raw_response, "get", None)
        while True:
            if ctx is not None:
                ctx.check_cancelled()
            remaining = envelope.remaining_seconds()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"distributed delivery {envelope.job_id} timed out")
            if callable(ready) and ready():
                result = getattr(raw_response, "result", None)
                if callable(result):
                    result = result()
                if result is None:
                    result = getattr(raw_response, "value", None)
                if isinstance(result, BaseException):
                    return result
                return result
            if callable(get):
                timeout = _DISPATCH_POLL_SECONDS if remaining is None else min(_DISPATCH_POLL_SECONDS, remaining)
                try:
                    result = get(timeout=timeout)
                except Exception as exc:
                    if exc.__class__.__name__ == "TimeoutError":
                        continue
                    raise
                if isinstance(result, BaseException):
                    raise result
                return result
            result = getattr(raw_response, "result", None)
            if callable(result):
                result = result()
            if result is None:
                result = getattr(raw_response, "value", None)
            return result

    @staticmethod
    def _wait_local_until(
        target_unix_ms: int,
        *,
        ctx: ExecutionContext | None,
        deadline_unix_ms: int | None,
    ) -> None:
        while True:
            if ctx is not None:
                ctx.check_cancelled()
            now_ms = time.time_ns() // 1_000_000
            remaining_ms = target_unix_ms - now_ms
            if remaining_ms <= 0:
                return
            if deadline_unix_ms is not None:
                deadline_remaining_ms = deadline_unix_ms - now_ms
                if deadline_remaining_ms <= 0:
                    raise TimeoutError("distributed retry deadline expired")
                remaining_ms = min(remaining_ms, deadline_remaining_ms)
            time.sleep(min(remaining_ms / 1000, _DISPATCH_POLL_SECONDS))

    @staticmethod
    def _local_transport_timeout(
        checkpoint_controller: CheckpointResumeController,
        envelope: DistributedRunEnvelope,
    ) -> AgentResult:
        checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError("checkpoint disappeared after distributed timeout", code="checkpoint_not_found")
        reconciliation_required = (
            checkpoint.claim_token is not None
            and checkpoint.claim_token != checkpoint_controller._owned_claim_token
        )
        return AgentResult(
            status=AgentStatus.RECONCILIATION_REQUIRED if reconciliation_required else AgentStatus.FAILED,
            completion_reason=None if reconciliation_required else CompletionReason.FAILED,
            partial_output=None if reconciliation_required else _last_assistant_output(checkpoint.cycles),
            messages=deepcopy(checkpoint.messages),
            cycles=deepcopy(checkpoint.cycles),
            wait_reason="reconciliation_required" if reconciliation_required else None,
            error=None
            if reconciliation_required
            else {
                "code": "distributed_delivery_timeout",
                "message": f"distributed delivery {envelope.job_id} timed out",
                "retryable": True,
            },
            shared_state=deepcopy(checkpoint.shared_state),
            token_usage=summarize_task_token_usage(checkpoint.model_calls),
            budget_usage=deepcopy(checkpoint.budget_usage),
            checkpoint_key=checkpoint.checkpoint_key,
        )

    @staticmethod
    def _local_cancelled(
        checkpoint_controller: CheckpointResumeController,
        error: CancelledError,
    ) -> AgentResult:
        checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError("checkpoint disappeared during cancellation", code="checkpoint_not_found") from error
        reconciliation_required = (
            checkpoint.claim_token is not None
            and checkpoint.claim_token != checkpoint_controller._owned_claim_token
        )
        return AgentResult(
            status=AgentStatus.RECONCILIATION_REQUIRED if reconciliation_required else AgentStatus.FAILED,
            completion_reason=None if reconciliation_required else CompletionReason.CANCELLED,
            partial_output=None if reconciliation_required else _last_assistant_output(checkpoint.cycles),
            messages=deepcopy(checkpoint.messages),
            cycles=deepcopy(checkpoint.cycles),
            wait_reason="reconciliation_required" if reconciliation_required else None,
            error=None if reconciliation_required else {"code": "cancelled", "message": str(error), "retryable": False},
            shared_state=deepcopy(checkpoint.shared_state),
            token_usage=summarize_task_token_usage(checkpoint.model_calls),
            budget_usage=deepcopy(checkpoint.budget_usage),
            checkpoint_key=checkpoint.checkpoint_key,
        )

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
        """Reject synchronous execution; use the non-blocking distributed API."""
        del task, initial_messages, shared_state, cycle_executor, ctx, max_cycles
        raise DistributedContractError("CeleryBackend.execute is unavailable; use execute_local or start/advance")

    def start(
        self,
        *,
        task: AgentTask,
        checkpoint_controller: CheckpointResumeController,
        ctx: ExecutionContext | None = None,
        continuation: Callable[[DistributedRunHandle, DistributedRunEnvelope], Any] | Any | None = None,
        start_admission: Callable[[DistributedRunHandle, DistributedRunEnvelope], bool] | None = None,
    ) -> DistributedRunHandle:
        """Enqueue the first cycle of an admitted run and return immediately."""
        self._validate_nonblocking_recipe()
        assert self.capability_registry is not None
        self.capability_registry.validate(self.runtime_recipe.capabilities, task=task)
        checkpoint = checkpoint_controller.store.load_checkpoint(checkpoint_controller.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError("checkpoint disappeared before distributed start", code="checkpoint_not_found")
        checkpoint_ref = self.runtime_recipe.capabilities.checkpoint_store_ref
        assert checkpoint_ref is not None
        shared_store = self.capability_registry.resolve("checkpoint_store", checkpoint_ref)
        shared_checkpoint = (
            checkpoint
            if shared_store is checkpoint_controller.store
            else shared_store.load_checkpoint(checkpoint_controller.checkpoint_key)
        )
        if shared_checkpoint != checkpoint:
            raise CheckpointError(
                "distributed start checkpoint store does not match the runtime recipe fact source",
                code="checkpoint_store_conflict",
            )
        if checkpoint.terminal_result is not None:
            return self._handle_for_checkpoint(checkpoint)
        if checkpoint.claim_token is not None or checkpoint.cycle_index != 0:
            raise CheckpointError(
                "distributed start requires an unclaimed checkpoint before cycle 1",
                code="checkpoint_cycle_conflict",
            )
        distributed_task = self._distributed_task(task, ctx)
        budget_limits = self._budget_limits(ctx)
        tool_registry = self.capability_registry.resolve_toolset(
            self.runtime_recipe.capabilities.toolset_ref,
            task=distributed_task,
        )
        task_toolset_digest = toolset_schema_digest(tool_registry, task=distributed_task)
        recipe = RuntimeRecipe.from_dict(self.runtime_recipe.to_dict())
        if recipe.capabilities.toolset_ref.schema_digest != task_toolset_digest:
            recipe = replace(
                recipe,
                capabilities=replace(
                    recipe.capabilities,
                    toolset_ref=ToolsetRef(
                        id=recipe.capabilities.toolset_ref.id,
                        version=recipe.capabilities.toolset_ref.version,
                        schema_digest=task_toolset_digest,
                    ),
                ),
            )
        envelope = self._envelope_from_checkpoint(
            task=distributed_task,
            recipe=recipe,
            checkpoint=checkpoint,
            checkpoint_config=DistributedCheckpointConfig.from_checkpoint_config(checkpoint_controller.config),
            cycle_index=1,
            claim_mode=checkpoint_controller.next_claim_mode,
            budget_limits=budget_limits,
        )
        handle = self._handle_for_checkpoint(checkpoint)
        if start_admission is not None:
            if not callable(start_admission):
                raise DistributedContractError("distributed start admission must be callable")
            should_enqueue = start_admission(handle, envelope)
            if not isinstance(should_enqueue, bool):
                raise DistributedContractError("distributed start admission must return a boolean")
            if not should_enqueue:
                return handle
        self._enqueue_envelope(envelope, handle=handle, continuation=continuation)
        return handle

    def advance(
        self,
        *,
        previous_envelope: DistributedRunEnvelope | Mapping[str, Any],
        outcome: DistributedDeliveryOutcome | DistributedWorkerResponse | Mapping[str, Any] | BaseException | str,
        continuation: Callable[[DistributedRunHandle, DistributedRunEnvelope], Any] | Any | None = None,
        enqueue: bool = True,
        now_unix_ms: int | None = None,
    ) -> DistributedAdvanceDecision:
        """Reconcile one delivery and make one nonblocking scheduler decision."""
        self._validate_nonblocking_recipe()
        envelope = (
            previous_envelope
            if isinstance(previous_envelope, DistributedRunEnvelope)
            else DistributedRunEnvelope.from_dict(previous_envelope)
        )
        delivery = self._delivery_outcome(outcome)
        store = self._nonblocking_checkpoint_store(envelope)
        checkpoint = store.load_checkpoint(envelope.checkpoint_config.key)
        if checkpoint is None:
            raise CheckpointError("checkpoint disappeared before distributed advance", code="checkpoint_not_found")
        self._validate_advance_checkpoint(envelope, checkpoint)
        handle = self._handle_for_checkpoint(checkpoint)
        response = delivery.response

        if checkpoint.terminal_result is not None:
            if (
                response is not None
                and response.response_type == "terminal_replay"
                and (
                    response.checkpoint_revision != checkpoint.revision
                    or response.result is None
                    or response.result.to_dict() != checkpoint.terminal_result.to_dict()
                )
            ):
                raise CheckpointError(
                    "distributed terminal replay does not match the durable checkpoint",
                    code="checkpoint_store_conflict",
                )
            return DistributedAdvanceDecision(
                action="terminal_replay",
                handle=handle,
                checkpoint_revision=checkpoint.revision,
                result=deepcopy(checkpoint.terminal_result),
            )

        if response is not None and response.response_type == "terminal_replay":
            raise CheckpointError(
                "distributed terminal replay has no matching durable terminal",
                code="checkpoint_store_conflict",
            )

        # ``pending`` is the existing worker response for a successfully
        # admitted deferred batch.  The authoritative checkpoint barrier is
        # the observation; never claim it or dispatch another cycle here.
        if checkpoint.status is AgentStatus.DEFERRED:
            if checkpoint.claim_token is not None:
                raise CheckpointError(
                    "deferred checkpoint must be unclaimed",
                    code="checkpoint_store_conflict",
                )
            return DistributedAdvanceDecision(
                action="wait",
                handle=handle,
                reason=DistributedWaitReason.DEFERRED_PENDING,
            )

        if checkpoint.status is AgentStatus.HOST_INTERACTION:
            if checkpoint.claim_token is not None:
                raise CheckpointError(
                    "host interaction checkpoint must not retain a worker claim",
                    code="checkpoint_store_conflict",
                )
            return DistributedAdvanceDecision(
                action="wait",
                handle=handle,
                reason=DistributedWaitReason.HOST_INTERACTION,
            )

        if checkpoint.status is AgentStatus.SUSPENDED:
            if checkpoint.claim_token is not None:
                raise CheckpointError(
                    "suspended checkpoint must not retain a worker claim",
                    code="checkpoint_store_conflict",
                )
            return DistributedAdvanceDecision(
                action="wait",
                handle=handle,
                reason=DistributedWaitReason.SUSPENDED,
            )

        if response is not None and response.response_type == "terminal_candidate":
            result = response.result
            assert result is not None
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
                return DistributedAdvanceDecision(
                    action="wait",
                    handle=handle,
                    reason=DistributedWaitReason.RECONCILIATION_REQUIRED,
                )
            if checkpoint.claim_token is None or checkpoint.claimed_cycle != envelope.cycle_index:
                raise CheckpointError(
                    "distributed terminal candidate does not retain the dispatched cycle claim",
                    code="checkpoint_store_conflict",
                )
            if result.cycles and result.cycles[-1].index != envelope.cycle_index:
                raise CheckpointError(
                    "distributed terminal candidate does not contain the dispatched cycle",
                    code="checkpoint_cycle_conflict",
                )
            return DistributedAdvanceDecision(
                action="finalize_required",
                handle=handle,
                checkpoint_revision=checkpoint.revision,
                result=deepcopy(result),
            )

        if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED:
            return DistributedAdvanceDecision(
                action="wait",
                handle=handle,
                reason=DistributedWaitReason.RECONCILIATION_REQUIRED,
            )
        if delivery.transport_error is not None:
            self._requeue_dispatched(envelope, error=delivery.transport_error)
        elif response is not None and response.response_type == "pending":
            self._requeue_dispatched(
                envelope,
                error="worker returned pending before a durable checkpoint barrier",
            )
        if response is not None and response.response_type == "committed":
            assert response.committed_cycle is not None
            if response.committed_cycle != envelope.cycle_index:
                raise CheckpointError(
                    "distributed committed response does not match the dispatched cycle",
                    code="checkpoint_cycle_conflict",
                )
            if checkpoint.cycle_index < response.committed_cycle:
                raise CheckpointError(
                    "distributed committed response is ahead of the checkpoint",
                    code="checkpoint_store_conflict",
                )

        if checkpoint.cycle_index > envelope.cycle_index or (
            checkpoint.claimed_cycle is not None and checkpoint.claimed_cycle > envelope.cycle_index
        ):
            return DistributedAdvanceDecision(
                action="wait",
                handle=handle,
                reason=DistributedWaitReason.SUPERSEDED_DELIVERY,
            )
        if (
            response is not None
            and response.response_type == "committed"
            and checkpoint.claim_token is None
            and response.checkpoint_revision != checkpoint.revision
        ):
            raise CheckpointError(
                "distributed committed response revision does not match the checkpoint",
                code="checkpoint_store_conflict",
            )

        current_ms = time.time_ns() // 1_000_000 if now_unix_ms is None else now_unix_ms
        claim_mode: ClaimMode = "continue"
        not_before: int | None = None
        if checkpoint.claim_token is not None:
            cycle_index = checkpoint.claimed_cycle
            if cycle_index is None:
                raise CheckpointError("distributed checkpoint has a partial claim", code="checkpoint_store_conflict")
            claim_mode = "recovery"
            if (checkpoint.lease_expires_at_ms or 0) > current_ms:
                not_before = checkpoint.lease_expires_at_ms
        else:
            if checkpoint.cycle_index == envelope.cycle_index:
                cycle_index = checkpoint.cycle_index + 1
            elif checkpoint.cycle_index + 1 == envelope.cycle_index:
                cycle_index = envelope.cycle_index
                claim_mode = "recovery"
            else:
                raise CheckpointError(
                    "distributed delivery is out of order with the checkpoint",
                    code="checkpoint_cycle_conflict",
                )

        if cycle_index > envelope.task.max_cycles:
            result = AgentResult(
                status=AgentStatus.MAX_CYCLES,
                completion_reason=CompletionReason.MAX_CYCLES,
                partial_output=_last_assistant_output(checkpoint.cycles),
                messages=deepcopy(checkpoint.messages),
                cycles=deepcopy(checkpoint.cycles),
                final_answer="Reached max cycles without finish signal.",
                shared_state=deepcopy(checkpoint.shared_state),
                token_usage=summarize_task_token_usage(checkpoint.model_calls),
                budget_usage=deepcopy(checkpoint.budget_usage),
            )
            return DistributedAdvanceDecision(
                action="finalize_required",
                handle=handle,
                checkpoint_revision=checkpoint.revision,
                result=result,
            )

        next_envelope = self._envelope_from_checkpoint(
            task=envelope.task,
            recipe=envelope.recipe,
            checkpoint=checkpoint,
            checkpoint_config=envelope.checkpoint_config,
            cycle_index=cycle_index,
            claim_mode=claim_mode,
            budget_limits=envelope.budget_limits,
            deadline_base_unix_ms=max(current_ms, not_before if not_before is not None else current_ms),
        )
        action = "retry_at" if not_before is not None else "dispatch"
        decision = DistributedAdvanceDecision(
            action=action,
            handle=handle,
            envelope=next_envelope,
            not_before_unix_ms=not_before,
        )
        if enqueue:
            self._enqueue_envelope(
                next_envelope,
                handle=handle,
                continuation=continuation,
                not_before_unix_ms=not_before,
            )
        return decision

    # ------------------------------------------------------------------
    # Distributed mode: each cycle → independent Celery task
    # ------------------------------------------------------------------

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
            checkpoint_controller.preloaded_checkpoint = deepcopy(checkpoint)
            replayed = checkpoint_controller.admit()
            if replayed is None:
                raise CheckpointError(
                    "durable terminal checkpoint did not produce a replay",
                    code="checkpoint_store_conflict",
                )
            return replayed
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

    def _validate_nonblocking_recipe(self) -> None:
        if self.capability_registry is None:
            raise DistributedContractError("nonblocking CeleryBackend requires a scheduler DistributedCapabilityRegistry")
        capabilities = self.runtime_recipe.capabilities
        if capabilities.approval_provider_ref is not None or capabilities.approval_broker_ref is not None:
            raise DistributedContractError("nonblocking distributed runs do not support brokered approval waits")

    @staticmethod
    def _handle_for_checkpoint(checkpoint: Any) -> DistributedRunHandle:
        return DistributedRunHandle(
            checkpoint_key=checkpoint.checkpoint_key,
            run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
        )

    @staticmethod
    def _budget_limits(ctx: ExecutionContext | None) -> RunBudgetLimits | None:
        value = ctx.metadata.get("_vv_agent_budget_limits") if ctx is not None else None
        return value if isinstance(value, RunBudgetLimits) else None

    @staticmethod
    def _distributed_task(task: AgentTask, ctx: ExecutionContext | None) -> AgentTask:
        distributed_task = deepcopy(task)
        effective_model_settings = ctx.metadata.get("_vv_agent_model_settings") if ctx is not None else None
        if isinstance(effective_model_settings, ModelSettings):
            distributed_task.model_settings = deepcopy(effective_model_settings)
        return distributed_task

    def _nonblocking_checkpoint_store(self, envelope: DistributedRunEnvelope) -> Any:
        assert self.capability_registry is not None
        reference = envelope.recipe.capabilities.checkpoint_store_ref
        if reference is None:
            raise DistributedContractError("distributed run requires checkpoint_store_ref")
        return self.capability_registry.resolve("checkpoint_store", reference)

    @staticmethod
    def _delivery_outcome(
        outcome: DistributedDeliveryOutcome | DistributedWorkerResponse | Mapping[str, Any] | BaseException | str,
    ) -> DistributedDeliveryOutcome:
        if isinstance(outcome, DistributedDeliveryOutcome):
            return outcome
        if isinstance(outcome, (BaseException, str)):
            return DistributedDeliveryOutcome.transport_failure(outcome)
        if isinstance(outcome, DistributedWorkerResponse):
            return DistributedDeliveryOutcome.worker(outcome)
        return DistributedDeliveryOutcome.worker(outcome)

    @staticmethod
    def _validate_advance_checkpoint(envelope: DistributedRunEnvelope, checkpoint: Any) -> None:
        if (
            checkpoint.checkpoint_key != envelope.checkpoint_config.key
            or checkpoint.task_id != envelope.task.task_id
            or checkpoint.root_run_id != envelope.root_run_id
            or checkpoint.trace_id != envelope.trace_id
            or checkpoint.run_definition_digest != envelope.run_definition_digest
        ):
            raise CheckpointError(
                "distributed advance envelope does not match the authoritative checkpoint",
                code="checkpoint_definition_mismatch",
            )

    def _envelope_from_checkpoint(
        self,
        *,
        task: AgentTask,
        recipe: RuntimeRecipe,
        checkpoint: Any,
        checkpoint_config: DistributedCheckpointConfig,
        cycle_index: int,
        claim_mode: ClaimMode,
        budget_limits: RunBudgetLimits | None,
        deadline_base_unix_ms: int | None = None,
    ) -> DistributedRunEnvelope:
        deadline_base = time.time_ns() // 1_000_000 if deadline_base_unix_ms is None else deadline_base_unix_ms
        deadline_unix_ms = deadline_base + int(self.dispatch_timeout_seconds * 1000)
        return DistributedRunEnvelope.for_cycle(
            task=task,
            recipe=recipe,
            cycle_index=cycle_index,
            root_run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
            run_definition_digest=checkpoint.run_definition_digest,
            claim_mode=claim_mode,
            resume_attempt=checkpoint.resume_attempt,
            checkpoint_config=checkpoint_config,
            cycle_name=DEFAULT_CYCLE_NAME,
            run_id=checkpoint.root_run_id,
            deadline_unix_ms=deadline_unix_ms,
            lease_duration_ms=self.lease_duration_ms,
            budget_limits=budget_limits,
        )

    def _enqueue_envelope(
        self,
        envelope: DistributedRunEnvelope,
        *,
        handle: DistributedRunHandle,
        continuation: Callable[[DistributedRunHandle, DistributedRunEnvelope], Any] | Any | None,
        not_before_unix_ms: int | None = None,
    ) -> Any:
        dispatch_claim = self._claim_dispatch(envelope)
        if dispatch_claim is not None and not dispatch_claim.should_enqueue:
            return None
        options: dict[str, Any] = {
            "serializer": "json",
            "task_id": envelope.job_id,
        }
        try:
            if continuation is not None:
                options["link"] = continuation(handle, envelope) if callable(continuation) else continuation
            if not_before_unix_ms is not None:
                now_ms = time.time_ns() // 1_000_000
                options["countdown"] = max(0.0, (not_before_unix_ms - now_ms) / 1000)
            result = self.celery_app.send_task(
                self.cycle_task_name,
                kwargs={"envelope_dict": envelope.to_dict()},
                **options,
            )
        except BaseException as exc:
            if dispatch_claim is not None:
                self._complete_dispatch(envelope, dispatch_claim, outcome="ambiguous", error=str(exc))
            raise
        # Completion is deliberately outside the broker exception handler.
        # If an injected transport receipt is used, a process death after
        # send_task returns but before this CAS leaves a claimed receipt for
        # host-owned reaping. Without that adapter there is intentionally no
        # receipt to complete: the stable task id and worker checkpoint CAS
        # provide the at-least-once fallback boundary.
        if dispatch_claim is not None:
            self._complete_dispatch(envelope, dispatch_claim, outcome="delivered")
        return result

    def _claim_dispatch(self, envelope: DistributedRunEnvelope) -> DispatchOutboxClaim | None:
        store = self.dispatch_outbox_store
        if store is None:
            return None
        now_ms = time.time_ns() // 1_000_000
        lease_expires_at_ms = now_ms + max(self.lease_duration_ms, 1_000)
        claim = store.claim_distributed_dispatch(
            envelope.to_dict(),
            claim_token=uuid4().hex,
            lease_expires_at_ms=lease_expires_at_ms,
            now_ms=now_ms,
        )
        if not isinstance(claim, DispatchOutboxClaim):
            raise DistributedContractError("dispatch outbox store returned an invalid dispatch claim")
        return claim

    def _complete_dispatch(
        self,
        envelope: DistributedRunEnvelope,
        claim: DispatchOutboxClaim | None,
        *,
        outcome: str,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        if claim is None:
            return None
        store = self.dispatch_outbox_store
        if store is None:
            raise DistributedContractError("dispatch completion requires the injected dispatch outbox store")
        record = store.complete_distributed_dispatch(
            dispatch_id=envelope.job_id,
            envelope_digest=dispatch_envelope_digest(envelope.to_dict()),
            claim_token=claim.record.claim_token or "",
            attempt=claim.record.attempt,
            outcome=outcome,
            now_ms=time.time_ns() // 1_000_000,
            error=error,
        )
        if not isinstance(record, DispatchOutboxRecord):
            raise DistributedContractError("dispatch outbox store returned an invalid dispatch record")
        return record

    def _reconcile_dispatch(
        self,
        envelope: DistributedRunEnvelope,
        *,
        outcome: str,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        store = self.dispatch_outbox_store
        if store is None:
            return None
        record = store.reconcile_distributed_dispatch(
            dispatch_id=envelope.job_id,
            envelope_digest=dispatch_envelope_digest(envelope.to_dict()),
            outcome=outcome,
            now_ms=time.time_ns() // 1_000_000,
            error=error,
        )
        if not isinstance(record, DispatchOutboxRecord):
            raise DistributedContractError("dispatch outbox store returned an invalid dispatch record")
        return record

    def _requeue_dispatched(self, envelope: DistributedRunEnvelope, *, error: str) -> DispatchOutboxRecord | None:
        return self._reconcile_dispatch(envelope, outcome="retry", error=error)

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
            return result.join(timeout=timeout)
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

    return celery_app.task(
        name=task_name,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
    )(worker_task)
