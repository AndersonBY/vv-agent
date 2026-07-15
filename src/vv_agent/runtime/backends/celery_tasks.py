"""Worker-side Celery task for executing a single agent cycle.

This module provides ``run_single_cycle`` which:
1. Rebuilds an AgentRuntime from a ``RuntimeRecipe``
2. Loads the previous checkpoint from the shared StateStore
3. Executes exactly one cycle via the runtime's cycle executor
4. Saves the updated checkpoint (or returns the terminal result)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from vv_agent.agent import RunContext
from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.backends.distributed import (
    DEFAULT_CYCLE_NAME,
    DistributedCapabilityRegistry,
    DistributedRunEnvelope,
    RuntimeRecipe,
)
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import AgentRuntime, register_sub_agent_session, unregister_sub_agent_session
from vv_agent.runtime.state import CheckpointConflictError, _LeaseOperationClock, build_state_store
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.types import AgentResult, AgentStatus, AgentTask, CompletionReason
from vv_agent.workspace import LocalWorkspaceBackend

_MAX_U64 = (1 << 64) - 1


def _lease_expiry_at(*, now_ms: int, lease_duration_ms: int, deadline_unix_ms: int | None) -> int:
    if deadline_unix_ms is None:
        lease_expires_at_ms = now_ms + lease_duration_ms
    else:
        lease_expires_at_ms = now_ms + min(lease_duration_ms, deadline_unix_ms - now_ms)
    if lease_expires_at_ms > _MAX_U64:
        raise OverflowError("checkpoint lease overflow")
    return lease_expires_at_ms


def _lease_heartbeat_interval_seconds(lease_duration_ms: int) -> float:
    return min(lease_duration_ms / 3000, 30.0)


class _LeaseHeartbeat:
    def __init__(
        self,
        *,
        store: Any,
        envelope: DistributedRunEnvelope,
        claim_token: str,
        expected_revision: int,
    ) -> None:
        self._store = store
        self._envelope = envelope
        self._claim_token = claim_token
        self._expected_revision = expected_revision
        self._stopped = Event()
        self._state_lock = Lock()
        self._error: BaseException | None = None
        self._error_renew_started_during_commit = False
        self._commit_started = False
        self._commit_succeeded = False
        self._known_lease_expires_at_ms: int | None = None
        self._interval_seconds = _lease_heartbeat_interval_seconds(envelope.lease_duration_ms)
        self._thread = Thread(target=self._run, name=f"vv-agent-lease-{envelope.job_id}", daemon=True)

    def start(self) -> None:
        try:
            self._load_claim_lease_expiry()
            effective_lease_ms = self._renew_once()
        except BaseException as exc:
            self._record_error(exc, renew_started_during_commit=False)
            self._stopped.set()
            self.raise_if_failed()
            return
        self._interval_seconds = _lease_heartbeat_interval_seconds(effective_lease_ms)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread.ident is not None:
            self._thread.join()

    def begin_commit(self) -> None:
        with self._state_lock:
            error = self._error
            if error is None:
                self._commit_started = True
        if error is not None:
            raise CheckpointConflictError(f"checkpoint lease heartbeat failed: {error}") from error

    def mark_commit_succeeded(self) -> None:
        with self._state_lock:
            if not self._commit_started:
                raise RuntimeError("checkpoint commit phase has not started")
            self._commit_succeeded = True

    def raise_if_failed(self) -> None:
        with self._state_lock:
            error = self._error
            suppress_commit_rejection = bool(
                error is not None
                and self._commit_succeeded
                and self._error_renew_started_during_commit
                and isinstance(error, CheckpointConflictError)
                and str(error) == "claim is no longer active"
            )
        if error is not None and not suppress_commit_rejection:
            raise CheckpointConflictError(f"checkpoint lease heartbeat failed: {error}") from error

    def _record_error(self, error: BaseException, *, renew_started_during_commit: bool) -> None:
        with self._state_lock:
            if self._error is None:
                self._error = error
                self._error_renew_started_during_commit = renew_started_during_commit

    def _run(self) -> None:
        interval_seconds = self._interval_seconds
        while not self._stopped.wait(interval_seconds):
            try:
                effective_lease_ms = self._renew_once()
                interval_seconds = _lease_heartbeat_interval_seconds(effective_lease_ms)
                self._interval_seconds = interval_seconds
            except BaseException:  # the worker must observe heartbeat infrastructure failures
                self._stopped.set()
                return

    def _renew_once(self) -> int:
        renew_started_during_commit = False
        try:
            now_ms = time.time_ns() // 1_000_000
            clock = _LeaseOperationClock(now_ms)
            self._envelope.ensure_not_expired(now_ms=now_ms)
            lease_expires_at_ms = _lease_expiry_at(
                now_ms=now_ms,
                lease_duration_ms=self._envelope.lease_duration_ms,
                deadline_unix_ms=self._envelope.deadline_unix_ms,
            )
            with self._state_lock:
                renew_started_during_commit = self._commit_started
                known_lease_expires_at_ms = self._known_lease_expires_at_ms
            renewed = self._store.renew_checkpoint_claim(
                self._envelope.task.task_id,
                claim_token=self._claim_token,
                expected_revision=self._expected_revision,
                lease_expires_at_ms=lease_expires_at_ms,
                now_ms=now_ms,
            )
            observed_at_ms = max(clock.now_ms(), time.time_ns() // 1_000_000)
            if not renewed:
                if observed_at_ms >= lease_expires_at_ms or (
                    known_lease_expires_at_ms is not None and observed_at_ms >= known_lease_expires_at_ms
                ):
                    raise CheckpointConflictError("claim lease expired")
                raise CheckpointConflictError("claim is no longer active")
            if observed_at_ms >= lease_expires_at_ms or (
                known_lease_expires_at_ms is not None and observed_at_ms >= known_lease_expires_at_ms
            ):
                raise CheckpointConflictError("claim lease expired")
            with self._state_lock:
                self._known_lease_expires_at_ms = lease_expires_at_ms
            return lease_expires_at_ms - observed_at_ms
        except BaseException as exc:
            self._record_error(exc, renew_started_during_commit=renew_started_during_commit)
            raise

    def _load_claim_lease_expiry(self) -> None:
        checkpoint = self._store.load_checkpoint(self._envelope.task.task_id)
        if (
            checkpoint is None
            or checkpoint.revision != self._expected_revision
            or checkpoint.claim_token != self._claim_token
            or checkpoint.claimed_cycle != self._envelope.cycle_index
            or checkpoint.lease_expires_at_ms is None
        ):
            raise CheckpointConflictError("claim is no longer active")
        with self._state_lock:
            self._known_lease_expires_at_ms = checkpoint.lease_expires_at_ms


def _build_state_store(recipe: RuntimeRecipe) -> Any:
    """Rebuild the exact durable store selected by the scheduler."""
    if recipe.state_store is None:
        raise ValueError("distributed RuntimeRecipe is missing state_store")
    return build_state_store(recipe.state_store)


def _resolve_many(
    registry: DistributedCapabilityRegistry,
    kind: Any,
    references: tuple[Any, ...],
) -> list[Any]:
    return [registry.resolve(kind, reference) for reference in references]


def _rebuild_runtime(
    recipe: RuntimeRecipe,
    capability_registry: DistributedCapabilityRegistry,
) -> tuple[AgentRuntime, ExecutionContext, SubTaskManager, ToolPolicy]:
    """Reconstruct an AgentRuntime from a RuntimeRecipe on the worker."""
    capabilities = recipe.capabilities
    capability_registry.validate(capabilities)
    workspace = Path(recipe.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if capabilities.llm_client_ref is not None:
        llm = capability_registry.resolve("llm_client", capabilities.llm_client_ref)
    else:
        llm, _resolved = build_openai_llm_from_local_settings(
            recipe.settings_file,
            backend=recipe.backend,
            model=recipe.model,
            timeout_seconds=recipe.timeout_seconds,
        )
    tool_registry = capability_registry.resolve_toolset(capabilities.toolset_ref)
    tool_policy = capabilities.tool_policy.resolve(capability_registry)
    hooks = _resolve_many(capability_registry, "hook", capabilities.hook_refs)
    observers = _resolve_many(capability_registry, "observer", capabilities.observer_refs)
    event_sink = (
        capability_registry.resolve("event_sink", capabilities.event_sink_ref)
        if capabilities.event_sink_ref is not None
        else None
    )

    def log_handler(event: str, payload: dict[str, Any]) -> None:
        for observer in observers:
            observer(event, dict(payload))

    workspace_backend = (
        capability_registry.resolve("workspace_backend", capabilities.workspace_backend_ref)
        if capabilities.workspace_backend_ref is not None
        else LocalWorkspaceBackend(workspace)
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=tool_registry,
        default_workspace=workspace,
        log_handler=log_handler if observers else None,
        log_preview_chars=recipe.log_preview_chars,
        settings_file=recipe.settings_file,
        default_backend=recipe.backend,
        hooks=hooks,
        workspace_backend=workspace_backend,
    )
    metadata: dict[str, Any] = {
        "_vv_agent_run_id": "",
        "_vv_agent_allowed_tools": (list(tool_policy.allowed_tools) if tool_policy.allowed_tools is not None else None),
        "_vv_agent_disallowed_tools": list(tool_policy.disallowed_tools),
        "_vv_agent_tool_policy_approval": tool_policy.approval,
    }
    if tool_policy.can_use_tool is not None:
        metadata["_vv_agent_tool_policy_can_use_tool"] = tool_policy.can_use_tool
    if event_sink is not None:
        metadata["_vv_agent_emit_event"] = event_sink
    if capabilities.approval_provider_ref is not None:
        approval_broker_ref = capabilities.approval_broker_ref
        assert approval_broker_ref is not None
        metadata["_vv_agent_approval_provider"] = capability_registry.resolve(
            "approval_provider", capabilities.approval_provider_ref
        )
        metadata["_vv_agent_approval_broker"] = capability_registry.resolve("approval_broker", approval_broker_ref)
        metadata["_vv_agent_approval_timeout_seconds"] = capabilities.approval_timeout_seconds
    memory_providers = _resolve_many(
        capability_registry,
        "memory_provider",
        capabilities.memory_provider_refs,
    )
    if memory_providers:
        metadata["_vv_agent_memory_providers"] = memory_providers
    app_state = (
        capability_registry.resolve("app_state", capabilities.app_state_ref) if capabilities.app_state_ref is not None else None
    )
    metadata["_vv_agent_run_context"] = RunContext(
        context=app_state,
        model=recipe.model,
        workspace=workspace_backend,
    )
    cancellation_token = (
        capability_registry.resolve("cancellation", capabilities.cancellation_ref)
        if capabilities.cancellation_ref is not None
        else None
    )
    context = ExecutionContext(
        cancellation_token=cancellation_token,
        metadata=metadata,
    )
    sub_task_manager = (
        capability_registry.resolve("sub_task_manager", capabilities.sub_task_manager_ref)
        if capabilities.sub_task_manager_ref is not None
        else SubTaskManager(
            register_session=register_sub_agent_session,
            unregister_session=unregister_sub_agent_session,
        )
    )
    return runtime, context, sub_task_manager, tool_policy


def run_single_cycle(
    *,
    envelope_dict: dict[str, Any] | None = None,
    capability_registry: DistributedCapabilityRegistry | None = None,
    task_dict: dict[str, Any] | None = None,
    recipe_dict: dict[str, Any] | None = None,
    cycle_index: int | None = None,
) -> dict[str, Any]:
    """Execute a single agent cycle on a Celery worker.

    Returns a dict with:
    - ``finished``: bool — whether the agent reached a terminal state
    - ``result``: dict — serialised AgentResult (only when finished)
    """
    if envelope_dict is None:
        if task_dict is None or recipe_dict is None or cycle_index is None:
            raise TypeError("run_single_cycle requires envelope_dict")
        task = AgentTask.from_dict(task_dict)
        recipe = RuntimeRecipe.from_dict(recipe_dict)
        envelope = DistributedRunEnvelope.for_cycle(
            task=task,
            recipe=recipe,
            cycle_index=cycle_index,
            cycle_name=DEFAULT_CYCLE_NAME,
        )
    else:
        envelope = DistributedRunEnvelope.from_dict(envelope_dict)
        task = envelope.task
        recipe = envelope.recipe
        cycle_index = envelope.cycle_index

    # Load checkpoint saved by the scheduler (or previous cycle).
    store = _build_state_store(recipe)
    existing = store.load_checkpoint(task.task_id)
    if existing is not None and existing.terminal_result is not None:
        return {
            "finished": True,
            "result": existing.terminal_result.to_dict(),
            "checkpoint_revision": existing.revision,
        }
    envelope.ensure_not_expired()
    registry = capability_registry or DistributedCapabilityRegistry()
    runtime, ctx, sub_task_manager, tool_policy = _rebuild_runtime(recipe, registry)
    heartbeat_store = _build_state_store(recipe)
    if tool_policy.allowed_tools is None:
        task.metadata.pop("_vv_agent_allowed_tools", None)
    else:
        task.metadata["_vv_agent_allowed_tools"] = list(tool_policy.allowed_tools)
    if tool_policy.disallowed_tools:
        task.metadata["_vv_agent_disallowed_tools"] = list(tool_policy.disallowed_tools)
    else:
        task.metadata.pop("_vv_agent_disallowed_tools", None)
    ctx.state_store = store
    ctx.metadata["_vv_agent_run_id"] = envelope.run_id
    run_context = ctx.metadata.get("_vv_agent_run_context")
    if isinstance(run_context, RunContext):
        run_context.run_id = envelope.run_id

    now_ms = time.time_ns() // 1_000_000
    envelope.ensure_not_expired(now_ms=now_ms)
    lease_expires_at_ms = _lease_expiry_at(
        now_ms=now_ms,
        lease_duration_ms=envelope.lease_duration_ms,
        deadline_unix_ms=envelope.deadline_unix_ms,
    )
    claim_token = uuid.uuid4().hex
    try:
        checkpoint = store.claim_checkpoint(
            task.task_id,
            cycle_index,
            claim_token=claim_token,
            lease_expires_at_ms=lease_expires_at_ms,
            now_ms=now_ms,
        )
    except CheckpointConflictError as exc:
        raise CheckpointConflictError(f"retryable distributed delivery conflict: {exc}") from exc
    if checkpoint is None:
        return {
            "finished": True,
            "result": AgentResult(
                status=AgentStatus.FAILED,
                completion_reason=CompletionReason.FAILED,
                messages=[],
                cycles=[],
                error=f"No checkpoint found for task {task.task_id}",
                shared_state={},
            ).to_dict(),
        }

    # Build the cycle executor closure on the worker side.
    workspace_path = Path(recipe.workspace).resolve()
    memory_manager = runtime._build_memory_manager(
        task=task,
        workspace_path=workspace_path,
    )
    cycle_executor = runtime._build_cycle_executor(
        task=task,
        workspace_path=workspace_path,
        workspace_backend=runtime._workspace_backend or LocalWorkspaceBackend(workspace_path),
        memory_manager=memory_manager,
        before_cycle_messages=None,
        interruption_messages=None,
        sub_task_manager=sub_task_manager,
    )

    messages = checkpoint.messages
    cycles = checkpoint.cycles
    shared_state = checkpoint.shared_state

    # Execute exactly one cycle.
    heartbeat = _LeaseHeartbeat(
        store=heartbeat_store,
        envelope=envelope,
        claim_token=claim_token,
        expected_revision=checkpoint.revision,
    )
    try:
        heartbeat.start()
        result = cycle_executor(
            cycle_index,
            messages,
            cycles,
            shared_state,
            ctx,
        )
        if result is not None:
            checkpoint.cycle_index = cycle_index
            checkpoint.status = result.status
            checkpoint.messages = result.messages
            checkpoint.cycles = result.cycles
            checkpoint.shared_state = result.shared_state
            checkpoint.terminal_result = result
            expected_revision = checkpoint.revision
            heartbeat.begin_commit()
            if not store.commit_checkpoint(
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            ):
                raise CheckpointConflictError(
                    f"checkpoint changed while terminal cycle {cycle_index} was running for task {task.task_id}"
                )
            heartbeat.mark_commit_succeeded()
            outcome = {
                "finished": True,
                "result": result.to_dict(),
                "checkpoint_revision": expected_revision + 1,
            }
        else:
            checkpoint.cycle_index = cycle_index
            checkpoint.status = AgentStatus.RUNNING
            checkpoint.messages = messages
            checkpoint.cycles = cycles
            checkpoint.shared_state = shared_state
            expected_revision = checkpoint.revision
            heartbeat.begin_commit()
            if not store.commit_checkpoint(
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            ):
                raise CheckpointConflictError(f"checkpoint changed while cycle {cycle_index} was running for task {task.task_id}")
            heartbeat.mark_commit_succeeded()
            outcome = {"finished": False}
    finally:
        heartbeat.stop()
    heartbeat.raise_if_failed()
    return outcome
