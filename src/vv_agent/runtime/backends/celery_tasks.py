"""Worker-side Celery task for executing a single agent cycle.

This module provides ``run_single_cycle`` which:
1. Rebuilds an AgentRuntime from a ``RuntimeRecipe``
2. Resolves and loads the checkpoint through the worker capability registry
3. Executes exactly one cycle via the runtime's cycle executor
4. Atomically commits progress or returns a terminal candidate
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from vv_agent.agent import RunContext
from vv_agent.budget import BudgetEvaluator, HostCostMeter
from vv_agent.checkpoint import (
    CheckpointError,
    compute_run_definition_digest,
    validate_checkpoint_extension,
)
from vv_agent.events import RunEvent
from vv_agent.model import ModelRef, VvLlmModelProvider
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.backends.distributed import (
    ClaimMode,
    DistributedCapabilityRegistry,
    DistributedRunEnvelope,
    DistributedWorkerResponse,
    RuntimeRecipe,
    _policy_with_task_metadata_denials,
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
from vv_agent.runtime.state import Checkpoint
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.types import AgentResult, AgentStatus, AgentTask, CompletionReason
from vv_agent.workspace import LocalWorkspaceBackend


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


def _validate_task_and_capabilities(
    *,
    envelope: DistributedRunEnvelope,
    checkpoint: Checkpoint,
    registry: DistributedCapabilityRegistry,
    extensions: list[Any],
) -> None:
    config = envelope.checkpoint_config
    assert config is not None
    stored_digest = compute_run_definition_digest(checkpoint.run_definition)
    if checkpoint.run_definition_digest != stored_digest:
        raise _definition_mismatch("distributed run definition digest does not match the embedded definition")
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
        "denied_side_effects": list(capabilities.tool_policy.denied_side_effects),
        "denied_capability_tags": list(capabilities.tool_policy.denied_capability_tags),
        "deny_terminal_tools": capabilities.tool_policy.deny_terminal_tools,
        "denied_cost_dimensions": list(capabilities.tool_policy.denied_cost_dimensions),
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
        ("after_cycle_hook", capabilities.after_cycle_hook_refs),
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


def _resolve_checkpoint_capabilities(
    envelope: DistributedRunEnvelope,
    registry: DistributedCapabilityRegistry,
) -> tuple[Any, Any | None, list[Any], Any | None, Any]:
    capabilities = envelope.recipe.capabilities
    registry.validate(capabilities)
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


def _effective_claim_mode(
    *,
    envelope: DistributedRunEnvelope,
    checkpoint: Checkpoint,
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
    *,
    task: AgentTask | None = None,
) -> tuple[AgentRuntime, ExecutionContext, SubTaskManager, ToolPolicy, HostCostMeter | None]:
    """Reconstruct an AgentRuntime from a RuntimeRecipe on the worker."""
    capabilities = recipe.capabilities
    capability_registry.validate(capabilities)
    workspace = Path(recipe.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if capabilities.llm_client_ref is not None:
        llm = capability_registry.resolve("llm_client", capabilities.llm_client_ref)
        model_provider = None
    else:
        model_provider = (
            VvLlmModelProvider.from_settings_file(recipe.settings_file)
            .with_default_backend(recipe.backend)
            .with_timeout_seconds(recipe.timeout_seconds)
        )
        resolved = model_provider.resolve(ModelRef.named(recipe.model))
        llm = model_provider.client(resolved)
    tool_registry = capability_registry.resolve_toolset(capabilities.toolset_ref)
    distributed_tool_policy = capabilities.tool_policy
    if task is not None:
        distributed_tool_policy = _policy_with_task_metadata_denials(distributed_tool_policy, task)
    tool_policy = distributed_tool_policy.resolve(capability_registry)
    hooks = _resolve_many(capability_registry, "hook", capabilities.hook_refs)
    after_cycle_hooks = _resolve_many(
        capability_registry,
        "after_cycle_hook",
        capabilities.after_cycle_hook_refs,
    )
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

    def event_handler(event: RunEvent) -> None:
        if event_sink is not None:
            event_sink(event)
        for observer in observers:
            try:
                observer(event)
            except BaseException:
                logging.getLogger(__name__).exception("Distributed run event observer failed")

    workspace_backend = (
        capability_registry.resolve("workspace_backend", capabilities.workspace_backend_ref)
        if capabilities.workspace_backend_ref is not None
        else LocalWorkspaceBackend(workspace)
    )
    runtime = AgentRuntime(
        llm_client=llm,
        model_provider=model_provider,
        tool_registry=tool_registry,
        default_workspace=workspace,
        event_handler=event_handler if event_sink is not None or observers else None,
        log_preview_chars=recipe.log_preview_chars,
        hooks=hooks,
        after_cycle_hooks=after_cycle_hooks,
        workspace_backend=workspace_backend,
    )
    metadata: dict[str, Any] = {
        "_vv_agent_run_id": "",
        "_vv_agent_allowed_tools": (list(tool_policy.allowed_tools) if tool_policy.allowed_tools is not None else None),
        "_vv_agent_disallowed_tools": list(tool_policy.disallowed_tools),
        "_vv_agent_tool_policy_approval": tool_policy.approval,
    }
    denied_side_effects = [getattr(value, "value", value) for value in getattr(tool_policy, "denied_side_effects", [])]
    if denied_side_effects:
        metadata["_vv_agent_denied_side_effects"] = denied_side_effects
    denied_capability_tags = list(getattr(tool_policy, "denied_capability_tags", []))
    if denied_capability_tags:
        metadata["_vv_agent_denied_capability_tags"] = denied_capability_tags
    if getattr(tool_policy, "deny_terminal_tools", False):
        metadata["_vv_agent_deny_terminal_tools"] = True
    denied_cost_dimensions = list(getattr(tool_policy, "denied_cost_dimensions", []))
    if denied_cost_dimensions:
        metadata["_vv_agent_denied_cost_dimensions"] = denied_cost_dimensions
    if tool_policy.can_use_tool is not None:
        metadata["_vv_agent_tool_policy_can_use_tool"] = tool_policy.can_use_tool
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
        event_handler=event_handler if event_sink is not None or observers else None,
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


def _run_single_cycle(
    *,
    envelope: DistributedRunEnvelope,
    capability_registry: DistributedCapabilityRegistry,
    transport_redelivered: bool,
    transport_retry_count: int,
) -> DistributedWorkerResponse:
    store, event_store, extensions, reconciliation_provider, event_sink = _resolve_checkpoint_capabilities(
        envelope, capability_registry
    )
    config = envelope.checkpoint_config
    assert config is not None
    existing = store.load_checkpoint(config.key)
    if existing is None:
        raise CheckpointError(
            f"checkpoint key {config.key!r} does not exist",
            code="checkpoint_not_found",
        )
    _validate_task_and_capabilities(
        envelope=envelope,
        checkpoint=existing,
        registry=capability_registry,
        extensions=extensions,
    )
    if existing.terminal_result is not None:
        return DistributedWorkerResponse.terminal_replay(
            checkpoint_revision=existing.revision,
            result=existing.terminal_result,
        )
    if existing.cycle_index >= envelope.cycle_index:
        if existing.cycle_index != envelope.cycle_index or existing.claim_token is not None:
            raise CheckpointError(
                "distributed envelope cycle does not match the durable checkpoint",
                code="checkpoint_cycle_conflict",
            )
        return DistributedWorkerResponse.committed(
            checkpoint_revision=existing.revision,
            committed_cycle=existing.cycle_index,
        )
    expected_cycle = existing.claimed_cycle or (existing.cycle_index + 1)
    if envelope.cycle_index != expected_cycle:
        raise CheckpointError(
            "distributed envelope cycle does not match the durable checkpoint",
            code="checkpoint_cycle_conflict",
        )

    now_ms = time.time_ns() // 1_000_000
    envelope.ensure_not_expired(now_ms=now_ms)
    claim_mode = _effective_claim_mode(
        envelope=envelope,
        checkpoint=existing,
        transport_redelivered=transport_redelivered,
        transport_retry_count=transport_retry_count,
        now_ms=now_ms,
    )
    runtime, ctx, sub_task_manager, tool_policy, host_cost_meter = _rebuild_runtime(
        envelope.recipe,
        capability_registry,
        task=envelope.task,
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
    ctx.checkpoint_store = store
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
        retained = store.load_checkpoint(config.key)
        return DistributedWorkerResponse.terminal_replay(
            checkpoint_revision=(retained.revision if retained is not None else existing.revision),
            result=replayed,
        )
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
            committed = store.load_checkpoint(config.key)
            if committed is None:
                raise CheckpointError(
                    "checkpoint disappeared after distributed cycle commit",
                    code="checkpoint_not_found",
                )
            return DistributedWorkerResponse.committed(
                checkpoint_revision=committed.revision,
                committed_cycle=committed.cycle_index,
            )

        controller.assert_heartbeat_healthy()
        current = store.load_checkpoint(config.key)
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
        return DistributedWorkerResponse.terminal_candidate(
            checkpoint_revision=current.revision,
            result=result,
        )
    finally:
        controller.close()


def run_single_cycle(
    *,
    envelope_dict: dict[str, Any],
    capability_registry: DistributedCapabilityRegistry | None = None,
    transport_redelivered: bool = False,
    transport_retry_count: int = 0,
) -> dict[str, Any]:
    """Execute a single agent cycle on a Celery worker.

    Returns one closed ``vv-agent.distributed-worker-response.v1`` payload.
    """
    envelope = DistributedRunEnvelope.from_dict(envelope_dict)

    if not isinstance(transport_redelivered, bool):
        raise TypeError("transport_redelivered must be a boolean")
    if isinstance(transport_retry_count, bool) or not isinstance(transport_retry_count, int) or transport_retry_count < 0:
        raise TypeError("transport_retry_count must be a non-negative integer")
    return _run_single_cycle(
        envelope=envelope,
        capability_registry=capability_registry or DistributedCapabilityRegistry(),
        transport_redelivered=transport_redelivered,
        transport_retry_count=transport_retry_count,
    ).to_dict()
