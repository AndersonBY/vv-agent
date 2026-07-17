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
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from vv_agent.agent import RunContext
from vv_agent.budget import BudgetEvaluator, HostCostMeter
from vv_agent.checkpoint import CheckpointError, validate_checkpoint_extension
from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.backends.distributed import (
    DEFAULT_CYCLE_NAME,
    DISTRIBUTED_RUN_SCHEMA_VERSION_V2,
    ClaimMode,
    DistributedCapabilityRegistry,
    DistributedRunEnvelope,
    RuntimeRecipe,
)
from vv_agent.runtime.checkpoint_resume import (
    CheckpointReconciliationRequired,
    CheckpointResumeController,
)
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import (
    AgentRuntime,
    _RunBudgetController,
    register_sub_agent_session,
    unregister_sub_agent_session,
)
from vv_agent.runtime.run_definition import (
    _normalize_extra_headers,
    _redact_credential_slots,
    _tool_definitions,
)
from vv_agent.runtime.state import CheckpointConflictError, _LeaseOperationClock, build_state_store
from vv_agent.runtime.state_v2 import CheckpointV2
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


def _definition_mismatch(message: str) -> CheckpointError:
    return CheckpointError(message, code="checkpoint_definition_mismatch")


def _reference_payload(reference: Any | None) -> dict[str, str] | None:
    return reference.to_dict() if reference is not None else None


def _validate_reference(
    *,
    slot: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != _reference_payload(actual):
        raise _definition_mismatch(f"distributed capability {slot!r} does not match the run definition")


def _validate_v2_task_and_capabilities(
    *,
    envelope: DistributedRunEnvelope,
    checkpoint: CheckpointV2,
    registry: DistributedCapabilityRegistry,
    extensions: list[Any],
) -> None:
    config = envelope.checkpoint_config
    assert config is not None
    definition = checkpoint.run_definition
    if checkpoint.checkpoint_key != config.key:
        raise _definition_mismatch("distributed checkpoint key does not match the durable checkpoint")
    if checkpoint.task_id != envelope.task.task_id:
        raise _definition_mismatch("distributed task identity does not match the durable checkpoint")
    if checkpoint.root_run_id != envelope.root_run_id or checkpoint.trace_id != envelope.trace_id:
        raise _definition_mismatch("distributed run identity does not match the durable checkpoint")
    if checkpoint.run_definition_schema != envelope.run_definition_schema:
        raise CheckpointError(
            "distributed run definition schema is unsupported",
            code="checkpoint_definition_schema_unsupported",
        )
    if checkpoint.run_definition_digest != envelope.run_definition_digest:
        raise _definition_mismatch("distributed run definition digest does not match the durable checkpoint")
    if checkpoint.resume_attempt != envelope.resume_attempt:
        raise CheckpointError(
            "distributed resume_attempt does not match the durable checkpoint",
            code="checkpoint_resume_attempt_mismatch",
        )

    task = envelope.task
    controls = definition["runtime_controls"]
    expected_task_fields = {
        "compiled_prompt": task.system_prompt,
        "root_input": task.user_prompt,
        "model_id": task.model,
        "max_cycles": task.max_cycles,
        "no_tool_policy": task.no_tool_policy,
        "memory_compact_threshold": task.memory_compact_threshold,
        "memory_threshold_percentage": task.memory_threshold_percentage,
        "allow_interruption": task.allow_interruption,
        "native_multimodal": task.native_multimodal,
        "tool_use_behavior": task.metadata.get("_vv_agent_tool_use_behavior", "run_llm_again"),
        "stop_at_tool_names": list(task.metadata.get("_vv_agent_stop_at_tool_names") or []),
    }
    durable_task_fields = {
        "compiled_prompt": definition["compiled_prompt"],
        "root_input": definition["root_input"],
        "model_id": definition["model"]["model_id"],
        "max_cycles": controls["max_cycles"],
        "no_tool_policy": controls["no_tool_policy"],
        "memory_compact_threshold": controls["memory_compact_threshold"],
        "memory_threshold_percentage": controls["memory_threshold_percentage"],
        "allow_interruption": controls["allow_interruption"],
        "native_multimodal": controls["native_multimodal"],
        "tool_use_behavior": controls["tool_use_behavior"],
        "stop_at_tool_names": controls["stop_at_tool_names"],
    }
    if expected_task_fields != durable_task_fields:
        raise _definition_mismatch("distributed task does not match the embedded run definition")
    if task.initial_shared_state != definition["initial_shared_state"]:
        raise _definition_mismatch("distributed initial shared state does not match the run definition")
    if [message.to_dict() for message in task.initial_messages] != definition["initial_messages"]:
        raise _definition_mismatch("distributed initial messages do not match the run definition")
    if envelope.recipe.backend != definition["model"]["backend"]:
        raise _definition_mismatch("distributed model backend does not match the run definition")
    if envelope.recipe.model != definition["model"]["model_id"]:
        raise _definition_mismatch("distributed recipe model does not match the run definition")
    candidate_definition = deepcopy(definition)
    candidate_settings = task.model_settings.to_dict() if task.model_settings is not None else {}
    candidate_timeout = candidate_settings.pop("timeout_seconds", None)
    _normalize_extra_headers(candidate_settings)
    candidate_definition["model"]["settings"] = candidate_settings
    if candidate_timeout is not None:
        candidate_definition["model"]["transport_timeout_seconds"] = candidate_timeout
    _redact_credential_slots(
        candidate_definition,
        list(definition["credential_slots"]),
    )
    if candidate_definition["model"] != definition["model"]:
        raise _definition_mismatch("distributed model settings do not match the run definition")
    expected_budget = definition["budget_limits"]
    actual_budget = envelope.budget_limits.to_dict() if envelope.budget_limits is not None else None
    if actual_budget != expected_budget:
        raise _definition_mismatch("distributed budget limits do not match the run definition")

    checkpoint_policy = definition["checkpoint_policy"]
    if (
        checkpoint_policy["ambiguous_model_policy"] != config.ambiguous_model_policy.value
        or checkpoint_policy["ambiguous_tool_policy"] != config.ambiguous_tool_policy.value
        or checkpoint_policy["max_extension_state_bytes"] != config.max_extension_state_bytes
    ):
        raise _definition_mismatch("distributed checkpoint policy does not match the run definition")
    if not set(config.credential_slots).issubset(definition["credential_slots"]):
        raise _definition_mismatch("distributed credential slots do not match the run definition")

    capabilities = envelope.recipe.capabilities
    expected_policy = definition["tool_policy"]
    actual_policy = {
        "allowed_tools": (
            sorted(set(capabilities.tool_policy.allowed_tools)) if capabilities.tool_policy.allowed_tools is not None else None
        ),
        "disallowed_tools": sorted(set(capabilities.tool_policy.disallowed_tools)),
        "approval": capabilities.tool_policy.approval,
        "predicate_ref": _reference_payload(capabilities.tool_policy.predicate_ref),
        "approval_timeout_seconds": capabilities.approval_timeout_seconds,
    }
    if actual_policy != expected_policy:
        raise _definition_mismatch("distributed tool policy does not match the run definition")

    definition_refs = deepcopy(definition["capability_refs"])
    tool_registry = registry.resolve_toolset(capabilities.toolset_ref)
    actual_tools = _tool_definitions(
        registry=tool_registry,
        task=task,
        refs=definition_refs,
    )
    if actual_tools != definition["tools"]:
        raise _definition_mismatch("distributed tool schemas do not match the run definition")

    _validate_reference(
        slot="context",
        expected=definition["context_ref"],
        actual=capabilities.app_state_ref,
    )
    _validate_reference(
        slot="workspace",
        expected=definition["workspace_ref"],
        actual=capabilities.workspace_backend_ref,
    )
    _validate_reference(
        slot="tool_policy.predicate",
        expected=definition["tool_policy"]["predicate_ref"],
        actual=capabilities.tool_policy.predicate_ref,
    )
    remaining_refs = definition_refs
    for slot, reference in (
        ("approval_provider", capabilities.approval_provider_ref),
        ("host_cost_meter", capabilities.host_cost_meter_ref),
        ("reconciliation_provider", capabilities.reconciliation_provider_ref),
        ("sub_task_manager", capabilities.sub_task_manager_ref),
    ):
        if reference is not None or slot in remaining_refs:
            _validate_reference(slot=slot, expected=remaining_refs.pop(slot, None), actual=reference)
    for prefix, references in (
        ("memory_provider", capabilities.memory_provider_refs),
        ("runtime_hook", capabilities.hook_refs),
    ):
        for index, reference in enumerate(references):
            slot = f"{prefix}:{index}"
            _validate_reference(slot=slot, expected=remaining_refs.pop(slot, None), actual=reference)
        unexpected = sorted(key for key in remaining_refs if key.startswith(f"{prefix}:"))
        if unexpected:
            raise _definition_mismatch(f"distributed capabilities are missing run-definition refs: {', '.join(unexpected)}")

    extension_definitions = {entry["namespace"]: entry for entry in definition["extensions"]}
    resolved_extensions = {extension.namespace: extension for extension in extensions}
    declared_extensions = {reference.namespace: reference for reference in capabilities.checkpoint_extension_refs}
    if set(resolved_extensions) != set(declared_extensions):
        raise _definition_mismatch("distributed checkpoint extension capabilities are incomplete")
    for namespace, reference in declared_extensions.items():
        expected = extension_definitions.get(namespace)
        extension = resolved_extensions[namespace]
        if expected is None or expected != {
            "namespace": namespace,
            "version": extension.version,
            "required": reference.required,
        }:
            raise _definition_mismatch(f"distributed checkpoint extension {namespace!r} does not match the run definition")


def _resolve_v2_checkpoint_capabilities(
    envelope: DistributedRunEnvelope,
    registry: DistributedCapabilityRegistry,
) -> tuple[Any, Any | None, list[Any], Any | None, Any]:
    capabilities = envelope.recipe.capabilities
    registry.validate(capabilities, require_checkpoint_v2=True)
    assert capabilities.checkpoint_store_ref is not None
    store = registry.resolve("checkpoint_store", capabilities.checkpoint_store_ref)
    config = envelope.checkpoint_config
    assert config is not None
    config.to_runtime_config(store=store)

    event_store = None
    if capabilities.checkpoint_event_store_ref is not None:
        event_store = registry.resolve(
            "checkpoint_event_store",
            capabilities.checkpoint_event_store_ref,
        )
        if not callable(getattr(event_store, "append_once", None)):
            raise TypeError("distributed checkpoint_event_store capability must provide append_once()")

    extensions: list[Any] = []
    for reference in capabilities.checkpoint_extension_refs:
        extension = registry.resolve("checkpoint_extension", reference.reference)
        validate_checkpoint_extension(extension)
        if extension.namespace != reference.namespace:
            raise TypeError("distributed checkpoint_extension capability namespace does not match its reference")
        extensions.append(extension)

    reconciliation_provider = None
    if capabilities.reconciliation_provider_ref is not None:
        reconciliation_provider = registry.resolve(
            "reconciliation_provider",
            capabilities.reconciliation_provider_ref,
        )
        if not callable(getattr(reconciliation_provider, "reconcile", None)):
            raise TypeError("distributed reconciliation_provider capability must provide reconcile()")

    def event_sink(_event: Any) -> None:
        return None

    if capabilities.event_sink_ref is not None:
        event_sink = registry.resolve("event_sink", capabilities.event_sink_ref)
        if not callable(event_sink):
            raise TypeError("distributed event_sink capability must be callable")
    return store, event_store, extensions, reconciliation_provider, event_sink


def _effective_v2_claim_mode(
    *,
    envelope: DistributedRunEnvelope,
    checkpoint: CheckpointV2,
    transport_redelivered: bool,
    transport_retry_count: int,
    now_ms: int,
) -> ClaimMode:
    if transport_redelivered or transport_retry_count > 0:
        return "recovery"
    if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED:
        return "recovery"
    if checkpoint.claim_token is not None and (checkpoint.lease_expires_at_ms or 0) <= now_ms:
        return "recovery"
    assert envelope.claim_mode is not None
    return envelope.claim_mode


def _rebuild_runtime(
    recipe: RuntimeRecipe,
    capability_registry: DistributedCapabilityRegistry,
) -> tuple[AgentRuntime, ExecutionContext, SubTaskManager, ToolPolicy, HostCostMeter | None]:
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
    host_cost_meter = (
        capability_registry.resolve("host_cost_meter", capabilities.host_cost_meter_ref)
        if capabilities.host_cost_meter_ref is not None
        else None
    )
    if host_cost_meter is not None and not callable(getattr(host_cost_meter, "read", None)):
        raise TypeError("distributed host_cost_meter capability must provide read()")

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
    return runtime, context, sub_task_manager, tool_policy, host_cost_meter


def _run_single_cycle_v2(
    *,
    envelope: DistributedRunEnvelope,
    capability_registry: DistributedCapabilityRegistry,
    transport_redelivered: bool,
    transport_retry_count: int,
) -> dict[str, Any]:
    store, event_store, extensions, reconciliation_provider, event_sink = _resolve_v2_checkpoint_capabilities(
        envelope, capability_registry
    )
    config = envelope.checkpoint_config
    assert config is not None
    existing = store.load_checkpoint_v2(config.key)
    if existing is None:
        raise CheckpointError(
            f"checkpoint key {config.key!r} does not exist",
            code="checkpoint_not_found",
        )
    _validate_v2_task_and_capabilities(
        envelope=envelope,
        checkpoint=existing,
        registry=capability_registry,
        extensions=extensions,
    )
    if existing.terminal_result is not None:
        return {
            "finished": True,
            "result": existing.terminal_result.to_dict(),
            "checkpoint_revision": existing.revision,
            "terminal_replay": True,
        }
    if existing.cycle_index >= envelope.cycle_index:
        if existing.cycle_index != envelope.cycle_index or existing.claim_token is not None:
            raise CheckpointError(
                "distributed envelope cycle does not match the durable checkpoint",
                code="checkpoint_cycle_conflict",
            )
        return {
            "finished": False,
            "checkpoint_revision": existing.revision,
            "committed_cycle": existing.cycle_index,
        }
    expected_cycle = existing.claimed_cycle or (existing.cycle_index + 1)
    if envelope.cycle_index != expected_cycle:
        raise CheckpointError(
            "distributed envelope cycle does not match the durable checkpoint",
            code="checkpoint_cycle_conflict",
        )

    now_ms = time.time_ns() // 1_000_000
    envelope.ensure_not_expired(now_ms=now_ms)
    claim_mode = _effective_v2_claim_mode(
        envelope=envelope,
        checkpoint=existing,
        transport_redelivered=transport_redelivered,
        transport_retry_count=transport_retry_count,
        now_ms=now_ms,
    )
    runtime, ctx, sub_task_manager, tool_policy, host_cost_meter = _rebuild_runtime(
        envelope.recipe,
        capability_registry,
    )
    task = envelope.task
    if tool_policy.allowed_tools is None:
        task.metadata.pop("_vv_agent_allowed_tools", None)
    else:
        task.metadata["_vv_agent_allowed_tools"] = list(tool_policy.allowed_tools)
    if tool_policy.disallowed_tools:
        task.metadata["_vv_agent_disallowed_tools"] = list(tool_policy.disallowed_tools)
    else:
        task.metadata.pop("_vv_agent_disallowed_tools", None)
    ctx.state_store = store
    ctx.metadata["_vv_agent_run_id"] = envelope.root_run_id
    ctx.metadata["_vv_agent_trace_id"] = envelope.trace_id
    ctx.metadata["trace_id"] = envelope.trace_id
    run_context = ctx.metadata.get("_vv_agent_run_context")
    if isinstance(run_context, RunContext):
        run_context.run_id = envelope.root_run_id

    runtime_config = config.to_runtime_config(
        store=store,
        capability_refs=existing.run_definition["capability_refs"],
    )
    controller = CheckpointResumeController(
        config=runtime_config,
        task_id=task.task_id,
        run_id=str(envelope.root_run_id),
        trace_id=str(envelope.trace_id),
        run_definition=existing.run_definition,
        run_definition_digest=existing.run_definition_digest,
        initial_messages=existing.messages,
        initial_shared_state=existing.shared_state,
        initial_budget_usage=existing.budget_usage,
        extensions=extensions,
        reconciliation_provider=reconciliation_provider,
        event_sink=event_sink,
        event_store=event_store,
        lease_duration_ms=envelope.lease_duration_ms,
        preloaded_checkpoint=existing,
    )
    replayed = controller.admit()
    if replayed is not None:
        controller.close()
        retained = store.load_checkpoint_v2(config.key)
        return {
            "finished": True,
            "result": replayed.to_dict(),
            "checkpoint_revision": retained.revision if retained is not None else existing.revision,
            "terminal_replay": True,
        }
    controller.set_next_claim_mode(claim_mode)

    budget_controller: _RunBudgetController | None = None
    if envelope.budget_limits is not None and envelope.budget_limits.has_limits:
        budget_controller = _RunBudgetController(
            evaluator=BudgetEvaluator(
                envelope.budget_limits,
                host_cost_meter=host_cost_meter,
                initial_usage=controller.budget_usage,
            ),
            task=task,
            ctx=ctx,
            emit_log=runtime._emit_log,
        )
    ctx.metadata["_vv_agent_checkpoint_controller"] = controller
    ctx.metadata["_vv_agent_checkpoint_budget_snapshot"] = (
        (lambda: budget_controller.snapshot) if budget_controller is not None else (lambda: None)
    )
    messages: list[Any] = []
    cycles: list[Any] = []
    shared_state: dict[str, Any] = {}
    messages, cycles, shared_state, start_cycle = controller.bind_runtime_state(
        messages=messages,
        cycles=cycles,
        shared_state=shared_state,
        budget_snapshot_provider=ctx.metadata["_vv_agent_checkpoint_budget_snapshot"],
    )
    if start_cycle != envelope.cycle_index:
        controller.close()
        raise CheckpointError(
            "distributed envelope cycle changed before execution",
            code="checkpoint_cycle_conflict",
        )

    workspace_path = Path(envelope.recipe.workspace).resolve()
    memory_manager = runtime._build_memory_manager(
        task=task,
        workspace_path=workspace_path,
        ctx=ctx,
    )
    cycle_executor = runtime._build_cycle_executor(
        task=task,
        workspace_path=workspace_path,
        workspace_backend=runtime._workspace_backend or LocalWorkspaceBackend(workspace_path),
        memory_manager=memory_manager,
        before_cycle_messages=None,
        interruption_messages=None,
        sub_task_manager=sub_task_manager,
        budget_controller=budget_controller,
    )

    try:
        result: AgentResult | None = None
        if budget_controller is not None and existing.cycle_index == 0:
            cancelled = bool(ctx.cancellation_token is not None and ctx.cancellation_token.cancelled)
            if cancelled:
                result = AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    messages=messages,
                    cycles=cycles,
                    error=ctx.cancellation_token.reason or "Operation was cancelled",
                    shared_state=shared_state,
                    budget_usage=budget_controller.snapshot,
                )
            else:
                exhaustion = budget_controller.run_start()
                if exhaustion is not None:
                    result = runtime._budget_failure_result(
                        messages=messages,
                        cycles=cycles,
                        shared_state=shared_state,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
        if result is None:
            try:
                result = cycle_executor(
                    envelope.cycle_index,
                    messages,
                    cycles,
                    shared_state,
                    ctx,
                )
            except CheckpointReconciliationRequired as interruption:
                result = interruption.result
        if budget_controller is not None and result is not None and budget_controller.exhaustion is None:
            cancelled = bool(
                result.status == AgentStatus.FAILED and ctx.cancellation_token is not None and ctx.cancellation_token.cancelled
            )
            operation_failed = bool(
                result.status == AgentStatus.FAILED
                and result.completion_reason not in {CompletionReason.BUDGET_EXHAUSTED, CompletionReason.CANCELLED}
            )
            if result.status is not AgentStatus.RECONCILIATION_REQUIRED:
                exhaustion = budget_controller.terminal(
                    suppress_exhaustion=cancelled or operation_failed,
                )
                if exhaustion is not None and not cancelled and not operation_failed:
                    result = runtime._budget_failure_result(
                        messages=messages,
                        cycles=cycles,
                        shared_state=shared_state,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
        if budget_controller is not None and result is not None:
            result.budget_usage = budget_controller.snapshot
            result.budget_exhaustion = budget_controller.exhaustion

        if result is None:
            if not cycles or cycles[-1].index != envelope.cycle_index:
                raise CheckpointError(
                    "distributed cycle returned without a durable cycle record",
                    code="checkpoint_cycle_conflict",
                )
            controller.commit_cycle(
                cycle_index=envelope.cycle_index,
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
            )
            committed = store.load_checkpoint_v2(config.key)
            if committed is None:
                raise CheckpointError(
                    "checkpoint disappeared after distributed cycle commit",
                    code="checkpoint_not_found",
                )
            return {
                "finished": False,
                "checkpoint_revision": committed.revision,
                "committed_cycle": committed.cycle_index,
            }

        controller.assert_heartbeat_healthy()
        current = store.load_checkpoint_v2(config.key)
        if current is None:
            raise CheckpointError(
                "checkpoint disappeared before returning the terminal candidate",
                code="checkpoint_not_found",
            )
        if result.status is AgentStatus.RECONCILIATION_REQUIRED:
            if current.status is not AgentStatus.RECONCILIATION_REQUIRED or current.claim_token is not None:
                raise CheckpointError(
                    "reconciliation result does not match durable checkpoint state",
                    code="checkpoint_store_conflict",
                )
        elif current.terminal_result is not None:
            raise CheckpointError(
                "worker terminal candidate must not finalize the checkpoint",
                code="checkpoint_store_conflict",
            )
        return {
            "finished": True,
            "result": result.to_dict(),
            "checkpoint_revision": current.revision,
            "terminal_candidate": True,
        }
    finally:
        controller.close()


def run_single_cycle(
    *,
    envelope_dict: dict[str, Any] | None = None,
    capability_registry: DistributedCapabilityRegistry | None = None,
    task_dict: dict[str, Any] | None = None,
    recipe_dict: dict[str, Any] | None = None,
    cycle_index: int | None = None,
    transport_redelivered: bool = False,
    transport_retry_count: int = 0,
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

    if not isinstance(transport_redelivered, bool):
        raise TypeError("transport_redelivered must be a boolean")
    if isinstance(transport_retry_count, bool) or not isinstance(transport_retry_count, int) or transport_retry_count < 0:
        raise TypeError("transport_retry_count must be a non-negative integer")
    if envelope.schema_version == DISTRIBUTED_RUN_SCHEMA_VERSION_V2:
        return _run_single_cycle_v2(
            envelope=envelope,
            capability_registry=capability_registry or DistributedCapabilityRegistry(),
            transport_redelivered=transport_redelivered,
            transport_retry_count=transport_retry_count,
        )

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
    runtime, ctx, sub_task_manager, tool_policy, host_cost_meter = _rebuild_runtime(recipe, registry)
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

    budget_controller: _RunBudgetController | None = None
    if envelope.budget_limits is not None and envelope.budget_limits.has_limits:
        budget_controller = _RunBudgetController(
            evaluator=BudgetEvaluator(
                envelope.budget_limits,
                host_cost_meter=host_cost_meter,
                initial_usage=checkpoint.budget_usage,
            ),
            task=task,
            ctx=ctx,
            emit_log=runtime._emit_log,
        )

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
        budget_controller=budget_controller,
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
        result: AgentResult | None = None
        if budget_controller is not None and checkpoint.cycle_index == 0:
            cancelled = bool(ctx.cancellation_token is not None and ctx.cancellation_token.cancelled)
            if cancelled:
                result = AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    messages=messages,
                    cycles=cycles,
                    error=ctx.cancellation_token.reason or "Operation was cancelled",
                    shared_state=shared_state,
                    budget_usage=budget_controller.snapshot,
                )
            else:
                exhaustion = budget_controller.run_start()
                if exhaustion is not None:
                    result = runtime._budget_failure_result(
                        messages=messages,
                        cycles=cycles,
                        shared_state=shared_state,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
        if result is None:
            result = cycle_executor(
                cycle_index,
                messages,
                cycles,
                shared_state,
                ctx,
            )
        if budget_controller is not None and result is not None and budget_controller.exhaustion is None:
            cancelled = bool(
                result.status == AgentStatus.FAILED and ctx.cancellation_token is not None and ctx.cancellation_token.cancelled
            )
            operation_failed = bool(
                result.status == AgentStatus.FAILED
                and result.completion_reason not in {CompletionReason.BUDGET_EXHAUSTED, CompletionReason.CANCELLED}
            )
            exhaustion = budget_controller.terminal(
                suppress_exhaustion=cancelled or operation_failed,
            )
            if exhaustion is not None and not cancelled and not operation_failed:
                result = runtime._budget_failure_result(
                    messages=messages,
                    cycles=cycles,
                    shared_state=shared_state,
                    controller=budget_controller,
                    exhaustion=exhaustion,
                )
        if budget_controller is not None and result is not None:
            result.budget_usage = budget_controller.snapshot
            result.budget_exhaustion = budget_controller.exhaustion
        if result is not None:
            checkpoint.cycle_index = cycle_index
            checkpoint.status = result.status
            checkpoint.messages = result.messages
            checkpoint.cycles = result.cycles
            checkpoint.shared_state = result.shared_state
            checkpoint.terminal_result = result
            checkpoint.budget_usage = result.budget_usage
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
            checkpoint.budget_usage = budget_controller.snapshot if budget_controller is not None else None
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
