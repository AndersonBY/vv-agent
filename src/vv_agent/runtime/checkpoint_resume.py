from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import (
    AmbiguousModelPolicy,
    AmbiguousToolPolicy,
    CheckpointConfig,
    CheckpointError,
    EventCursor,
    OperationKind,
    OperationState,
    ReconciliationDecision,
    ReconciliationDecisionKind,
    ReconciliationProvider,
    ResumeObservation,
    ResumePolicy,
    ToolIdempotency,
    compute_operation_request_digest,
    compute_run_definition_digest,
)
from vv_agent.event_store import IdempotentRunEventStore, RunEventStore
from vv_agent.events import (
    CheckpointCreatedEvent,
    CheckpointResumedEvent,
    ModelRetryDuplicateRiskEvent,
    OperationAmbiguousEvent,
    OperationReplayedEvent,
    ReconciliationRequiredEvent,
    ReconciliationResolvedEvent,
    RunEvent,
    event_from_dict,
)
from vv_agent.llm.base import LlmRequest
from vv_agent.runtime.checkpoint_codec_v2 import run_definition_comparison_copy
from vv_agent.runtime.state_v2 import (
    CheckpointStoreV2,
    CheckpointV2,
    ClaimMode,
    EventOutboxEntry,
    ExtensionStateEntry,
    OperationError,
    OperationJournalEntry,
)
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    CycleRecord,
    LLMResponse,
    Message,
    ToolCall,
    ToolExecutionResult,
    ToolResultStatus,
)

DEFAULT_CHECKPOINT_LEASE_MS = 5 * 60 * 1000
_AMBIGUOUS_TOOL_ERROR_CODES = frozenset(
    {
        "tool_timeout",
        "tool_cancelled",
        "tool_connection_lost",
        "tool_execution_failed",
    }
)


@dataclass(frozen=True, slots=True)
class ToolOperationPlan:
    idempotency_key: str
    operation_id: str
    request_digest: str
    idempotency_support: ToolIdempotency
    replay_result: ToolExecutionResult | None = None


class CheckpointReconciliationRequired(RuntimeError):
    def __init__(self, result: AgentResult) -> None:
        super().__init__("resume_requires_reconciliation")
        self.result = result


class CheckpointResumeController:
    """Coordinates one opt-in checkpoint v2 root run against a durable store."""

    def __init__(
        self,
        *,
        config: CheckpointConfig,
        task_id: str,
        run_id: str,
        trace_id: str,
        run_definition: dict[str, Any],
        run_definition_digest: str,
        initial_messages: list[Message],
        initial_shared_state: dict[str, Any],
        initial_budget_usage: BudgetUsageSnapshot | None,
        extensions: list[Any],
        reconciliation_provider: ReconciliationProvider | None,
        event_sink: Callable[[RunEvent], None],
        event_store: RunEventStore | None = None,
        lease_duration_ms: int = DEFAULT_CHECKPOINT_LEASE_MS,
        preloaded_checkpoint: CheckpointV2 | None = None,
    ) -> None:
        if config.store is None:
            raise CheckpointError(
                "process-local checkpoint execution requires CheckpointConfig.store",
                code="checkpoint_store_unavailable",
            )
        self.config = replace(config)
        self.store: CheckpointStoreV2 = config.store
        self.task_id = task_id
        self.run_id = run_id
        self.trace_id = trace_id
        self.run_definition = deepcopy(run_definition)
        self.run_definition_digest = run_definition_digest
        self.initial_messages = deepcopy(initial_messages)
        self.initial_shared_state = deepcopy(initial_shared_state)
        self.initial_budget_usage = deepcopy(initial_budget_usage)
        self.extensions = {extension.namespace: extension for extension in extensions}
        self.reconciliation_provider = reconciliation_provider
        self.event_sink = event_sink
        self.event_store = event_store
        self.lease_duration_ms = lease_duration_ms
        self.preloaded_checkpoint = deepcopy(preloaded_checkpoint)
        self.checkpoint: CheckpointV2 | None = None
        self.terminal_replay: AgentResult | None = None
        self._created = False
        self._first_claim_is_recovery = False
        self._runtime_messages: list[Message] | None = None
        self._runtime_cycles: list[CycleRecord] | None = None
        self._runtime_shared_state: dict[str, Any] | None = None
        self._budget_snapshot_provider: Callable[[], BudgetUsageSnapshot | None] | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: CheckpointError | None = None
        self._owned_claim_token: str | None = None

    @property
    def checkpoint_key(self) -> str:
        checkpoint = self._require_checkpoint()
        return checkpoint.checkpoint_key

    @property
    def resume_attempt(self) -> int:
        return self._require_checkpoint().resume_attempt

    @property
    def budget_usage(self) -> BudgetUsageSnapshot | None:
        return deepcopy(self._require_checkpoint().budget_usage)

    @property
    def next_claim_mode(self) -> ClaimMode:
        return "recovery" if self._first_claim_is_recovery else "continue"

    def set_next_claim_mode(self, claim_mode: ClaimMode) -> None:
        if claim_mode not in {"continue", "recovery"}:
            raise CheckpointError(
                "checkpoint claim_mode must be continue or recovery",
                code="checkpoint_claim_mode_invalid",
            )
        self._first_claim_is_recovery = claim_mode == "recovery"

    def adopt_claim_for_terminal_finalize(
        self,
        *,
        claim_token: str,
        lease_duration_ms: int,
    ) -> None:
        checkpoint = self._require_checkpoint()
        if checkpoint.claim_token != claim_token:
            raise CheckpointError(
                "distributed terminal claim no longer matches the durable checkpoint",
                code="checkpoint_claim_active",
            )
        if isinstance(lease_duration_ms, bool) or not isinstance(lease_duration_ms, int) or lease_duration_ms <= 0:
            raise CheckpointError(
                "checkpoint lease duration must be a positive integer",
                code="checkpoint_config_invalid",
            )
        self._owned_claim_token = claim_token
        self.lease_duration_ms = lease_duration_ms
        self._renew_claim_before_dispatch()
        self._start_heartbeat()

    def assert_heartbeat_healthy(self) -> None:
        self._assert_heartbeat()

    @staticmethod
    def preload(config: CheckpointConfig | None) -> CheckpointV2 | None:
        if config is None or config.key is None or config.store is None:
            return None
        checkpoint = config.store.load_checkpoint_v2(config.key)
        return deepcopy(checkpoint)

    def admit(self) -> AgentResult | None:
        key = self.config.key
        if key is None:
            key = f"checkpoint_{uuid.uuid4().hex}"
            self.config.key = key

        existing = self.preloaded_checkpoint
        if existing is not None and existing.checkpoint_key != key:
            raise CheckpointError(
                "preloaded checkpoint key does not match CheckpointConfig.key",
                code="checkpoint_key_conflict",
            )
        if existing is None:
            existing = self.store.load_checkpoint_v2(key)
        if self.config.resume_policy is ResumePolicy.NEW:
            if existing is not None:
                raise CheckpointError(
                    f"checkpoint key {key!r} already exists",
                    code="checkpoint_key_conflict",
                )
            checkpoint = self._new_checkpoint(key)
            self._queue_initial_event(checkpoint, self._checkpoint_created_event(key))
            if not self.store.create_checkpoint_v2(checkpoint):
                raise CheckpointError(
                    f"checkpoint key {key!r} was created concurrently",
                    code="checkpoint_key_conflict",
                )
            self.checkpoint = checkpoint
            self._created = True
            self._deliver_pending_outbox()
            return None

        if existing is None:
            if self.config.resume_policy is ResumePolicy.REQUIRE_EXISTING:
                raise CheckpointError(
                    f"checkpoint key {key!r} does not exist",
                    code="checkpoint_not_found",
                )
            checkpoint = self._new_checkpoint(key)
            self._queue_initial_event(checkpoint, self._checkpoint_created_event(key))
            if not self.store.create_checkpoint_v2(checkpoint):
                existing = self.store.load_checkpoint_v2(key)
                if existing is None:
                    raise CheckpointError(
                        f"checkpoint key {key!r} disappeared after create conflict",
                        code="checkpoint_store_conflict",
                    )
            else:
                self.checkpoint = checkpoint
                self._created = True
                self._deliver_pending_outbox()
                return None

        assert existing is not None
        self._validate_existing_definition(existing)
        if existing.terminal_result is not None:
            replay = deepcopy(existing.terminal_result)
            replay.checkpoint_key = key
            self.checkpoint = existing
            self.terminal_replay = replay
            self._deliver_pending_outbox()
            self._acknowledge_terminal()
            return replay
        now_ms = self._now_ms()
        if existing.claim_token is not None and (existing.lease_expires_at_ms or 0) > now_ms:
            raise CheckpointError(
                f"checkpoint key {key!r} has a live claim",
                code="checkpoint_claim_active",
            )
        self.checkpoint = existing
        self._first_claim_is_recovery = True
        self._restore_extensions(existing)
        return None

    def close(self) -> None:
        self._stop_heartbeat()

    def bind_runtime_state(
        self,
        *,
        messages: list[Message],
        cycles: list[CycleRecord],
        shared_state: dict[str, Any],
        budget_snapshot_provider: Callable[[], BudgetUsageSnapshot | None] | None = None,
    ) -> tuple[list[Message], list[CycleRecord], dict[str, Any], int]:
        checkpoint = self._require_checkpoint()
        if not self._created:
            messages[:] = deepcopy(checkpoint.messages)
            cycles[:] = deepcopy(checkpoint.cycles)
            shared_state.clear()
            shared_state.update(deepcopy(checkpoint.shared_state))
        self._runtime_messages = messages
        self._runtime_cycles = cycles
        self._runtime_shared_state = shared_state
        self._budget_snapshot_provider = budget_snapshot_provider
        return messages, cycles, shared_state, checkpoint.cycle_index + 1

    def complete_model(
        self,
        *,
        cycle_index: int,
        operation_slot: str,
        request: LlmRequest,
        invoke: Callable[[], LLMResponse],
    ) -> LLMResponse:
        if not isinstance(operation_slot, str) or not operation_slot:
            raise CheckpointError(
                "model operation slot must be non-empty",
                code="checkpoint_journal_integrity_mismatch",
            )
        projection = self._model_request_projection(request)
        digest = compute_operation_request_digest(projection)
        operation_id = self._model_operation_id(cycle_index, operation_slot)
        self._ensure_claim(cycle_index)
        entry = self._find_operation(OperationKind.MODEL, operation_id=operation_id)
        if entry is not None and entry.request_digest != digest:
            raise CheckpointError(
                "model request does not match the durable operation slot",
                code="checkpoint_journal_integrity_mismatch",
            )
        if entry is not None and entry.state is OperationState.SUCCEEDED:
            assert entry.response is not None
            self._emit_operation_replayed(entry)
            return self._llm_response_from_dict(entry.response)
        if entry is not None and entry.state is OperationState.FAILED:
            error = entry.error or OperationError(
                code="model_request_failed",
                message="durable model operation failed",
            )
            raise RuntimeError(f"{error.code}: {error.message}")
        if entry is None:
            entry = OperationJournalEntry(
                kind=OperationKind.MODEL,
                operation_id=operation_id,
                cycle_index=cycle_index,
                attempt=1,
                state=OperationState.PLANNED,
                request_digest=digest,
                idempotency_key=None,
            )
            self._require_checkpoint().model_call_journal.append(entry)
            self._progress()
        if entry.state is not OperationState.PLANNED:
            raise CheckpointError(
                "model journal is not executable after recovery",
                code="checkpoint_journal_integrity_mismatch",
            )
        entry.state = OperationState.STARTED
        self._progress()
        self._renew_claim_before_dispatch()
        try:
            response = invoke()
        except BaseException as exc:
            if self._is_definitive_model_error(exc):
                entry.state = OperationState.FAILED
                entry.error = OperationError(
                    code=getattr(exc, "code", None) or "model_request_failed",
                    message=str(exc) or type(exc).__name__,
                    retryable=False,
                )
                self._progress()
                raise
            entry.state = OperationState.AMBIGUOUS
            self._progress()
            self._suspend_for(entry)
        entry.state = OperationState.SUCCEEDED
        entry.response = self._llm_response_to_dict(response)
        entry.error = None
        self._progress()
        return response

    def plan_tool(
        self,
        *,
        cycle_index: int,
        call: ToolCall,
        idempotency_support: ToolIdempotency,
        source_request_digest: str | None = None,
        source_idempotency_key: str | None = None,
    ) -> ToolOperationPlan:
        idempotency_key = source_idempotency_key or self._tool_idempotency_key(
            cycle_index,
            call.id,
        )
        projection = {
            "schema_version": "vv-agent.operation-request.v1",
            "kind": "tool",
            "request": {
                "tool_call_id": call.id,
                "tool_name": call.name,
                "arguments": deepcopy(call.arguments),
                "idempotency_key": idempotency_key,
            },
        }
        digest = compute_operation_request_digest(projection)
        if source_request_digest is not None and source_request_digest != digest:
            raise CheckpointError(
                "approval resume tool request does not match its source digest",
                code="checkpoint_journal_integrity_mismatch",
            )
        existing = self._find_tool_call(cycle_index=cycle_index, tool_call_id=call.id)
        if existing is not None:
            existing.verify_request(projection)
            if existing.idempotency_support is not idempotency_support:
                raise CheckpointError(
                    "tool idempotency declaration does not match the durable journal",
                    code="checkpoint_journal_integrity_mismatch",
                )
        self._ensure_claim(cycle_index)
        entry = self._find_tool_call(cycle_index=cycle_index, tool_call_id=call.id)
        if entry is not None and entry.state is OperationState.SUCCEEDED:
            assert entry.result is not None
            self._emit_operation_replayed(entry)
            return ToolOperationPlan(
                idempotency_key=idempotency_key,
                operation_id=entry.operation_id,
                request_digest=entry.request_digest,
                idempotency_support=idempotency_support,
                replay_result=ToolExecutionResult.from_dict(entry.result),
            )
        if entry is not None and entry.state is OperationState.FAILED:
            error = entry.error or OperationError(
                code="tool_operation_failed",
                message="durable tool operation failed",
            )
            self._emit_operation_replayed(entry)
            return ToolOperationPlan(
                idempotency_key=idempotency_key,
                operation_id=entry.operation_id,
                request_digest=entry.request_digest,
                idempotency_support=idempotency_support,
                replay_result=ToolExecutionResult(
                    tool_call_id=call.id,
                    content=error.message,
                    status="error",
                    status_code=ToolResultStatus.ERROR,
                    error_code=error.code,
                ),
            )
        if entry is None:
            entry = OperationJournalEntry(
                kind=OperationKind.TOOL,
                operation_id=f"op_tool_cycle_{cycle_index}_call_{len(self._require_checkpoint().tool_journal) + 1}",
                cycle_index=cycle_index,
                attempt=1,
                state=OperationState.PLANNED,
                request_digest=digest,
                idempotency_key=idempotency_key,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=deepcopy(call.arguments),
                idempotency_support=idempotency_support,
            )
            self._require_checkpoint().tool_journal.append(entry)
            self._progress()
        elif entry.state is not OperationState.PLANNED:
            raise CheckpointError(
                "tool journal is not executable after recovery",
                code="checkpoint_journal_integrity_mismatch",
            )
        return ToolOperationPlan(
            idempotency_key=idempotency_key,
            operation_id=entry.operation_id,
            request_digest=entry.request_digest,
            idempotency_support=idempotency_support,
        )

    def tool_started(self, *, cycle_index: int, call: ToolCall) -> None:
        entry = self._find_tool_call(cycle_index=cycle_index, tool_call_id=call.id)
        if entry is None:
            raise CheckpointError(
                f"tool call {call.id!r} was started without a durable plan",
                code="checkpoint_journal_integrity_mismatch",
            )
        if entry.state is OperationState.PLANNED:
            entry.state = OperationState.STARTED
            self._progress()
            self._renew_claim_before_dispatch()

    def finish_tool(
        self,
        *,
        cycle_index: int,
        call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        entry = self._find_tool_call(cycle_index=cycle_index, tool_call_id=call.id)
        if entry is None:
            return
        if entry.state is OperationState.PLANNED:
            if result.error_code == "tool_approval_required":
                return
            entry.state = OperationState.FAILED
            entry.error = OperationError(
                code=result.error_code or "tool_short_circuited",
                message=result.content or "tool invocation was short-circuited",
                retryable=False,
            )
            self._progress()
            return
        if entry.state is not OperationState.STARTED:
            return
        definitive_outcome = bool(result.metadata.get("definitive_outcome"))
        if result.error_code in _AMBIGUOUS_TOOL_ERROR_CODES and not definitive_outcome:
            entry.state = OperationState.AMBIGUOUS
            self._progress()
            self._suspend_for(entry)
        if result.status_code is ToolResultStatus.SUCCESS or result.status_code is ToolResultStatus.WAIT_RESPONSE:
            entry.state = OperationState.SUCCEEDED
            entry.result = result.to_dict()
            entry.error = None
        else:
            entry.state = OperationState.FAILED
            entry.result = None
            entry.error = OperationError(
                code=result.error_code or "tool_operation_failed",
                message=result.content or "tool operation failed",
                retryable=bool(result.metadata.get("retryable")),
            )
        self._progress()

    def commit_cycle(
        self,
        *,
        cycle_index: int,
        messages: list[Message],
        cycles: list[CycleRecord],
        shared_state: dict[str, Any],
    ) -> None:
        checkpoint = self._require_checkpoint()
        if checkpoint.claim_token is None:
            return
        if not cycles or cycles[-1].index != cycle_index:
            raise CheckpointError(
                "cannot commit a checkpoint without the completed cycle record",
                code="checkpoint_cycle_conflict",
            )
        self._refresh_snapshot(messages=messages, cycles=cycles, shared_state=shared_state)
        checkpoint.cycle_index = cycle_index
        checkpoint.status = AgentStatus.RUNNING
        checkpoint.event_outbox = [entry for entry in checkpoint.event_outbox if entry.state == "pending"]
        revision = checkpoint.revision
        claim_token = checkpoint.claim_token
        if not self.store.commit_checkpoint_v2(
            checkpoint,
            claim_token=claim_token,
            expected_revision=revision,
        ):
            raise CheckpointError(
                "checkpoint cycle commit lost its claim",
                code="checkpoint_store_conflict",
            )
        checkpoint.revision = revision + 1
        checkpoint.claim_token = None
        checkpoint.claimed_cycle = None
        checkpoint.lease_expires_at_ms = None
        checkpoint.model_call_journal = []
        checkpoint.tool_journal = []
        self._first_claim_is_recovery = False
        self._owned_claim_token = None
        self._stop_heartbeat()

    def finalize(self, result: AgentResult, *, terminal_event: RunEvent | None = None) -> AgentResult:
        if result.status is AgentStatus.RECONCILIATION_REQUIRED:
            result.checkpoint_key = self.checkpoint_key
            return result
        checkpoint = self.store.load_checkpoint_v2(self.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError(
                "checkpoint disappeared before terminal finalization",
                code="checkpoint_not_found",
            )
        if checkpoint.terminal_result is not None:
            self.checkpoint = checkpoint
            self._deliver_pending_outbox()
            self._acknowledge_terminal()
            return deepcopy(checkpoint.terminal_result)
        self.checkpoint = checkpoint
        unresolved = self._unresolved_operation(checkpoint)
        if unresolved is not None and not self._is_operator_abort_result(result):
            raise CheckpointError(
                "checkpoint terminal finalization has an unresolved operation",
                code="checkpoint_terminal_unresolved_operation",
            )
        terminal = deepcopy(result)
        terminal.checkpoint_key = checkpoint.checkpoint_key
        checkpoint.status = terminal.status
        checkpoint.terminal_result = terminal
        preserve_ambiguity = bool(
            self._is_operator_abort_result(terminal)
            and any(
                entry.state is OperationState.AMBIGUOUS for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
            )
        )
        if not preserve_ambiguity:
            checkpoint.model_call_journal = []
            checkpoint.tool_journal = []
        checkpoint.budget_usage = deepcopy(terminal.budget_usage)
        checkpoint.messages = deepcopy(terminal.messages)
        checkpoint.cycles = deepcopy(terminal.cycles)
        checkpoint.shared_state = deepcopy(terminal.shared_state)
        self._snapshot_extensions(checkpoint)
        checkpoint.event_outbox = [entry for entry in checkpoint.event_outbox if entry.state == "pending"]
        if terminal_event is not None:
            stable_terminal_event = self._with_stable_terminal_event_id(terminal_event)
            self._queue_outbox_event(checkpoint, stable_terminal_event)
        revision = checkpoint.revision
        claim_token = checkpoint.claim_token
        if claim_token is None:
            finalized = self.store.finalize_checkpoint_v2(
                checkpoint,
                expected_revision=revision,
            )
        else:
            finalized = self.store.finalize_claimed_checkpoint_v2(
                checkpoint,
                claim_token=claim_token,
                expected_revision=revision,
            )
        if not finalized:
            authoritative = self.store.load_checkpoint_v2(checkpoint.checkpoint_key)
            if authoritative is not None and authoritative.terminal_result is not None:
                self.checkpoint = authoritative
                self._deliver_pending_outbox()
                self._acknowledge_terminal()
                return deepcopy(authoritative.terminal_result)
            raise CheckpointError(
                "checkpoint terminal finalization lost its revision",
                code="checkpoint_store_conflict",
            )
        self._stop_heartbeat()
        authoritative = self.store.load_checkpoint_v2(checkpoint.checkpoint_key)
        if authoritative is None or authoritative.terminal_result is None:
            raise CheckpointError(
                "checkpoint terminal finalization was not retained",
                code="checkpoint_store_conflict",
            )
        self.checkpoint = authoritative
        self._deliver_pending_outbox()
        self._acknowledge_terminal()
        result.checkpoint_key = checkpoint.checkpoint_key
        return result

    def prepare_terminal(self, result: AgentResult) -> AgentResult:
        if result.status is AgentStatus.RECONCILIATION_REQUIRED or self._is_operator_abort_result(result):
            return result
        checkpoint = self.store.load_checkpoint_v2(self.checkpoint_key)
        if checkpoint is None:
            raise CheckpointError(
                "checkpoint disappeared before terminal preparation",
                code="checkpoint_not_found",
            )
        self.checkpoint = checkpoint
        if checkpoint.terminal_result is not None:
            return deepcopy(checkpoint.terminal_result)
        unresolved = self._unresolved_operation(checkpoint)
        if unresolved is None:
            return result
        if unresolved.state is OperationState.STARTED:
            unresolved.state = OperationState.AMBIGUOUS
            self._progress()
        self._suspend_for(unresolved)
        raise AssertionError("checkpoint suspension must transfer control")

    def persist_preterminal_event(self, event: RunEvent, *, identity: str) -> None:
        payload = event.to_dict()
        payload["event_id"] = self._stable_event_id(
            "preterminal",
            identity,
            str(event.cycle_index or 0),
        )
        self._emit(event_from_dict(payload))

    @staticmethod
    def _unresolved_operation(checkpoint: CheckpointV2) -> OperationJournalEntry | None:
        unresolved = next(
            (
                entry
                for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
                if entry.state in {OperationState.STARTED, OperationState.AMBIGUOUS}
            ),
            None,
        )
        return unresolved

    @staticmethod
    def _is_operator_abort_result(result: AgentResult) -> bool:
        return bool(
            result.status is AgentStatus.FAILED
            and result.error == "operator_abort_with_unknown_outcome"
            and result.resume_observation is not None
        )

    def _new_checkpoint(self, key: str) -> CheckpointV2:
        checkpoint = CheckpointV2(
            checkpoint_key=key,
            task_id=self.task_id,
            root_run_id=self.run_id,
            trace_id=self.trace_id,
            run_definition=deepcopy(self.run_definition),
            run_definition_digest=self.run_definition_digest,
            resume_attempt=1,
            cycle_index=0,
            status=AgentStatus.RUNNING,
            messages=deepcopy(self.initial_messages),
            cycles=[],
            shared_state=deepcopy(self.initial_shared_state),
            budget_usage=deepcopy(self.initial_budget_usage),
        )
        self._snapshot_extensions(checkpoint)
        return checkpoint

    def _ensure_claim(self, cycle_index: int) -> None:
        checkpoint = self._require_checkpoint()
        if checkpoint.claim_token is not None:
            if checkpoint.claim_token == self._owned_claim_token and checkpoint.claimed_cycle == cycle_index:
                return
            if not self._first_claim_is_recovery:
                raise CheckpointError(
                    "checkpoint claim is not owned by this execution",
                    code="checkpoint_claim_active",
                )
        now_ms = self._now_ms()
        claim_mode = "recovery" if self._first_claim_is_recovery else "continue"
        claim_token = f"claim_{uuid.uuid4().hex}"
        try:
            claimed = self.store.claim_checkpoint_v2(
                checkpoint.checkpoint_key,
                cycle_index,
                claim_token=claim_token,
                lease_expires_at_ms=now_ms + self.lease_duration_ms,
                now_ms=now_ms,
                claim_mode=claim_mode,
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or "checkpoint_store_conflict"
            raise CheckpointError(str(exc), code=code) from exc
        if claimed is None:
            raise CheckpointError(
                "checkpoint disappeared while claiming a cycle",
                code="checkpoint_not_found",
            )
        self.checkpoint = claimed
        self._owned_claim_token = claim_token
        self._start_heartbeat()
        if claim_mode == "recovery":
            self._emit(
                CheckpointResumedEvent(
                    run_id=self.run_id,
                    trace_id=self.trace_id,
                    cycle_index=checkpoint.cycle_index,
                    checkpoint_key=checkpoint.checkpoint_key,
                    resume_attempt=claimed.resume_attempt,
                    event_id=self._stable_event_id(
                        "checkpoint_resumed",
                        str(claimed.resume_attempt),
                    ),
                )
            )
            self._recover_ambiguous_operations()
        self._first_claim_is_recovery = False

    def _recover_ambiguous_operations(self) -> None:
        checkpoint = self._require_checkpoint()
        changed = False
        for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]:
            if entry.state is OperationState.STARTED:
                entry.state = OperationState.AMBIGUOUS
                changed = True
        if changed:
            self._progress()
        for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]:
            if entry.state is not OperationState.AMBIGUOUS:
                continue
            observation = self._observation(entry)
            self._emit_ambiguous(entry, observation)
            decision = self._reconciliation_decision(entry, observation)
            if decision.kind is ReconciliationDecisionKind.DEFER:
                self._suspend_for(
                    entry,
                    observation=observation,
                    ambiguity_emitted=True,
                )
            if decision.kind is ReconciliationDecisionKind.RETRY:
                entry.state = OperationState.PLANNED
                entry.attempt += 1
                entry.response = None
                entry.result = None
                entry.error = None
            elif decision.kind is ReconciliationDecisionKind.REPLAY_SUCCESS:
                entry.state = OperationState.SUCCEEDED
                entry.response = deepcopy(decision.response)
                entry.result = deepcopy(decision.result)
                entry.error = None
            elif decision.kind is ReconciliationDecisionKind.RECORD_FAILURE:
                assert decision.error is not None
                entry.state = OperationState.FAILED
                entry.error = OperationError(
                    code=decision.error.code,
                    message=decision.error.message,
                    retryable=decision.error.retryable,
                )
            elif decision.kind is ReconciliationDecisionKind.ABORT:
                self._abort_unknown_operation(entry, observation, decision)
            self._progress()
            self._emit(
                ReconciliationResolvedEvent(
                    run_id=self.run_id,
                    trace_id=self.trace_id,
                    cycle_index=entry.cycle_index,
                    checkpoint_key=checkpoint.checkpoint_key,
                    operation_id=entry.operation_id,
                    operation_kind=entry.kind,
                    decision=decision.kind,
                    event_id=self._stable_event_id(
                        "reconciliation_resolved",
                        entry.operation_id,
                        str(entry.attempt),
                    ),
                )
            )

    def _reconciliation_decision(
        self,
        entry: OperationJournalEntry,
        observation: ResumeObservation,
    ) -> ReconciliationDecision:
        if self.reconciliation_provider is not None:
            decision = self.reconciliation_provider.reconcile(observation)
            if not isinstance(decision, ReconciliationDecision):
                raise CheckpointError(
                    "reconciliation provider returned an invalid decision",
                    code="checkpoint_reconciliation_decision_invalid",
                )
            return decision
        if (
            entry.kind is OperationKind.MODEL
            and self.config.ambiguous_model_policy is AmbiguousModelPolicy.RETRY_WITH_DUPLICATE_RISK
        ):
            self._emit(
                ModelRetryDuplicateRiskEvent(
                    run_id=self.run_id,
                    trace_id=self.trace_id,
                    cycle_index=entry.cycle_index,
                    checkpoint_key=self.checkpoint_key,
                    operation_id=entry.operation_id,
                    operation_kind=OperationKind.MODEL,
                    risk="duplicate_model_request_and_cost",
                    event_id=self._stable_event_id(
                        "model_retry_duplicate_risk",
                        entry.operation_id,
                        str(entry.attempt + 1),
                    ),
                )
            )
            return ReconciliationDecision(ReconciliationDecisionKind.RETRY)
        if (
            entry.kind is OperationKind.TOOL
            and self.config.ambiguous_tool_policy is AmbiguousToolPolicy.RETRY_IDEMPOTENT_ONLY
            and entry.idempotency_support is ToolIdempotency.SUPPORTED
        ):
            return ReconciliationDecision(ReconciliationDecisionKind.RETRY)
        return ReconciliationDecision(ReconciliationDecisionKind.DEFER)

    def _suspend_for(
        self,
        entry: OperationJournalEntry,
        *,
        observation: ResumeObservation | None = None,
        ambiguity_emitted: bool = False,
    ) -> None:
        checkpoint = self._require_checkpoint()
        observation = observation or self._observation(entry)
        if not ambiguity_emitted:
            self._emit_ambiguous(entry, observation)
        self._emit(
            ReconciliationRequiredEvent(
                run_id=self.run_id,
                trace_id=self.trace_id,
                cycle_index=entry.cycle_index,
                checkpoint_key=checkpoint.checkpoint_key,
                operation_id=entry.operation_id,
                operation_kind=entry.kind,
                interruption_reason="resume_requires_reconciliation",
                resume_observation=observation,
                event_id=self._stable_event_id(
                    "reconciliation_required",
                    entry.operation_id,
                    str(entry.attempt),
                ),
            )
        )
        checkpoint.status = AgentStatus.RECONCILIATION_REQUIRED
        revision = checkpoint.revision
        claim_token = checkpoint.claim_token
        if claim_token is None or not self.store.suspend_checkpoint_v2(
            checkpoint,
            claim_token=claim_token,
            expected_revision=revision,
        ):
            raise CheckpointError(
                "failed to suspend checkpoint for reconciliation",
                code="checkpoint_store_conflict",
            )
        checkpoint.revision = revision + 1
        checkpoint.claim_token = None
        checkpoint.claimed_cycle = None
        checkpoint.lease_expires_at_ms = None
        self._owned_claim_token = None
        self._stop_heartbeat()
        result = AgentResult(
            status=AgentStatus.RECONCILIATION_REQUIRED,
            messages=deepcopy(checkpoint.messages),
            cycles=deepcopy(checkpoint.cycles),
            shared_state=deepcopy(checkpoint.shared_state),
            budget_usage=deepcopy(checkpoint.budget_usage),
            checkpoint_key=checkpoint.checkpoint_key,
            resume_observation=observation,
        )
        raise CheckpointReconciliationRequired(result)

    def _abort_unknown_operation(
        self,
        entry: OperationJournalEntry,
        observation: ResumeObservation,
        decision: ReconciliationDecision,
    ) -> None:
        checkpoint = self._require_checkpoint()
        terminal = AgentResult(
            status=AgentStatus.FAILED,
            messages=deepcopy(checkpoint.messages),
            cycles=deepcopy(checkpoint.cycles),
            error="operator_abort_with_unknown_outcome",
            shared_state=deepcopy(checkpoint.shared_state),
            budget_usage=deepcopy(checkpoint.budget_usage),
            checkpoint_key=checkpoint.checkpoint_key,
            resume_observation=observation,
        )
        if checkpoint.claim_token is None:
            raise CheckpointError(
                "operator abort requires an active recovery claim",
                code="checkpoint_claim_active",
            )
        raise CheckpointReconciliationRequired(terminal)

    def _progress(self) -> None:
        checkpoint = self._require_checkpoint()
        self._assert_heartbeat()
        claim_token = checkpoint.claim_token
        if claim_token is None:
            raise CheckpointError(
                "checkpoint progress requires an active claim",
                code="checkpoint_claim_active",
            )
        # In-flight transcript changes are reconstructed from operation receipts.
        # Only a completed cycle may advance durable messages and cycle records.
        self._refresh_snapshot(include_runtime_transcript=False)
        revision = checkpoint.revision
        if not self.store.progress_checkpoint_v2(
            checkpoint,
            claim_token=claim_token,
            expected_revision=revision,
        ):
            raise CheckpointError(
                "checkpoint progress lost its claim",
                code="checkpoint_store_conflict",
            )
        checkpoint.revision = revision + 1

    def _refresh_snapshot(
        self,
        *,
        messages: list[Message] | None = None,
        cycles: list[CycleRecord] | None = None,
        shared_state: dict[str, Any] | None = None,
        include_runtime_transcript: bool = True,
    ) -> None:
        checkpoint = self._require_checkpoint()
        source_messages = messages if messages is not None else (self._runtime_messages if include_runtime_transcript else None)
        source_cycles = cycles if cycles is not None else (self._runtime_cycles if include_runtime_transcript else None)
        source_shared = shared_state if shared_state is not None else self._runtime_shared_state
        if source_messages is not None:
            checkpoint.messages = deepcopy(source_messages)
        if source_cycles is not None:
            checkpoint.cycles = deepcopy(source_cycles)
        if source_shared is not None:
            checkpoint.shared_state = deepcopy(source_shared)
        if self._budget_snapshot_provider is not None:
            checkpoint.budget_usage = deepcopy(self._budget_snapshot_provider())
        self._snapshot_extensions(checkpoint)

    def _snapshot_extensions(self, checkpoint: CheckpointV2) -> None:
        snapshot: dict[str, ExtensionStateEntry] = {}
        for namespace, extension in self.extensions.items():
            snapshot[namespace] = ExtensionStateEntry(
                version=extension.version,
                required=bool(extension.required or namespace in self.config.required_extension_namespaces),
                state=deepcopy(extension.snapshot()),
            )
        for namespace, entry in checkpoint.extension_state.items():
            if namespace not in snapshot:
                snapshot[namespace] = deepcopy(entry)
        checkpoint.extension_state = snapshot

    def _restore_extensions(self, checkpoint: CheckpointV2) -> None:
        for namespace, entry in checkpoint.extension_state.items():
            extension = self.extensions.get(namespace)
            if extension is None:
                if entry.required:
                    raise CheckpointError(
                        f"required checkpoint extension {namespace!r} is unavailable",
                        code="checkpoint_extension_missing",
                    )
                continue
            if extension.version != entry.version:
                raise CheckpointError(
                    f"checkpoint extension {namespace!r} version mismatch",
                    code="checkpoint_extension_version_mismatch",
                )
            extension.restore(deepcopy(entry.state))
        for namespace in self.config.required_extension_namespaces:
            if namespace not in checkpoint.extension_state:
                raise CheckpointError(
                    f"required checkpoint extension {namespace!r} has no durable state",
                    code="checkpoint_extension_missing",
                )

    def _validate_existing_definition(self, checkpoint: CheckpointV2) -> None:
        stored_digest = compute_run_definition_digest(checkpoint.run_definition)
        if checkpoint.run_definition_digest != stored_digest:
            raise CheckpointError(
                "checkpoint run definition digest does not match its embedded definition",
                code="checkpoint_definition_mismatch",
            )
        current_digest = compute_run_definition_digest(self.run_definition)
        if self.run_definition_digest != current_digest:
            raise CheckpointError(
                "current run definition digest does not match its definition",
                code="checkpoint_definition_mismatch",
            )
        stored_comparison = run_definition_comparison_copy(checkpoint.run_definition)
        current_comparison = run_definition_comparison_copy(self.run_definition)
        if stored_comparison != current_comparison:
            raise CheckpointError(
                "checkpoint embedded run definition does not match this run",
                code="checkpoint_definition_mismatch",
            )

    def _model_request_projection(self, request: LlmRequest) -> dict[str, Any]:
        checkpoint = self._require_checkpoint()
        model_definition = checkpoint.run_definition["model"]
        model_override = request.metadata.get("_vv_agent_checkpoint_model")
        if isinstance(model_override, dict):
            backend = model_override.get("backend")
            model_id = model_override.get("model_id")
            if not isinstance(backend, str) or not isinstance(model_id, str):
                raise CheckpointError(
                    "checkpoint model override is invalid",
                    code="checkpoint_journal_integrity_mismatch",
                )
            effective_model = {"backend": backend, "model_id": model_id}
        else:
            effective_model = {
                "backend": model_definition["backend"],
                "model_id": model_definition["model_id"],
            }
        settings = request.model_settings.to_dict() if request.model_settings is not None else {}
        settings.pop("timeout_seconds", None)
        return {
            "schema_version": "vv-agent.operation-request.v1",
            "kind": "model",
            "request": {
                "model": {
                    **effective_model,
                },
                "messages": [message.to_openai_message() for message in request.messages],
                "tools": deepcopy(request.tools),
                "settings": settings,
                "output_schema": deepcopy(checkpoint.run_definition["output_schema"]),
                "idempotency_key": None,
            },
        }

    def _find_operation(
        self,
        kind: OperationKind,
        *,
        operation_id: str,
    ) -> OperationJournalEntry | None:
        checkpoint = self._require_checkpoint()
        journal = checkpoint.model_call_journal if kind is OperationKind.MODEL else checkpoint.tool_journal
        return next(
            (entry for entry in journal if entry.operation_id == operation_id),
            None,
        )

    @staticmethod
    def _model_operation_id(cycle_index: int, operation_slot: str) -> str:
        slot_digest = hashlib.sha256(operation_slot.encode("utf-8")).hexdigest()[:16]
        return f"op_model_cycle_{cycle_index}_{slot_digest}"

    def _find_tool_call(
        self,
        *,
        cycle_index: int,
        tool_call_id: str,
    ) -> OperationJournalEntry | None:
        return next(
            (
                entry
                for entry in self._require_checkpoint().tool_journal
                if entry.cycle_index == cycle_index and entry.tool_call_id == tool_call_id
            ),
            None,
        )

    def _observation(self, entry: OperationJournalEntry) -> ResumeObservation:
        if entry.kind is OperationKind.MODEL:
            risk = "duplicate_model_request_and_cost"
            support = None
        else:
            risk = "unknown_tool_side_effect"
            support = entry.idempotency_support
        return ResumeObservation(
            operation_id=entry.operation_id,
            operation_kind=entry.kind,
            cycle_index=entry.cycle_index,
            risk=risk,
            idempotency_support=support,
        )

    def _emit_ambiguous(
        self,
        entry: OperationJournalEntry,
        observation: ResumeObservation,
    ) -> None:
        self._emit(
            OperationAmbiguousEvent(
                run_id=self.run_id,
                trace_id=self.trace_id,
                cycle_index=entry.cycle_index,
                checkpoint_key=self.checkpoint_key,
                operation_id=entry.operation_id,
                operation_kind=entry.kind,
                risk=observation.risk,
                idempotency_support=observation.idempotency_support,
                event_id=self._stable_event_id(
                    "operation_ambiguous",
                    entry.operation_id,
                    str(entry.attempt),
                ),
            )
        )

    def _emit_operation_replayed(self, entry: OperationJournalEntry) -> None:
        self._emit(
            OperationReplayedEvent(
                run_id=self.run_id,
                trace_id=self.trace_id,
                cycle_index=entry.cycle_index,
                checkpoint_key=self.checkpoint_key,
                operation_id=entry.operation_id,
                operation_kind=entry.kind,
                receipt_state=entry.state,
                event_id=self._stable_event_id(
                    "operation_replayed",
                    entry.operation_id,
                    str(entry.attempt),
                ),
            )
        )

    def _emit(self, event: RunEvent) -> None:
        checkpoint = self._require_checkpoint()
        if checkpoint.claim_token is None:
            raise CheckpointError(
                "checkpoint event enqueue requires an active claim",
                code="checkpoint_claim_active",
            )
        self._queue_outbox_event(checkpoint, event)
        self._progress()
        self._deliver_pending_outbox()

    def _queue_initial_event(self, checkpoint: CheckpointV2, event: RunEvent) -> None:
        self._queue_outbox_event(checkpoint, event)

    @staticmethod
    def _queue_outbox_event(checkpoint: CheckpointV2, event: RunEvent) -> None:
        candidate = EventOutboxEntry.pending(event.event_id, event.to_dict())
        existing = next(
            (entry for entry in checkpoint.event_outbox if entry.event_id == event.event_id),
            None,
        )
        if existing is None:
            checkpoint.event_outbox.append(candidate)
            return
        existing.verify_payload()
        if existing.payload_digest != candidate.payload_digest:
            raise CheckpointError(
                f"checkpoint event id {event.event_id!r} has conflicting payload bytes",
                code="event_identity_conflict",
            )

    def _deliver_pending_outbox(self) -> None:
        checkpoint = self._require_checkpoint()
        for pending in [entry for entry in checkpoint.event_outbox if entry.state == "pending"]:
            pending.verify_payload()
            event = event_from_dict(pending.event)
            cursor = self._deliver_event(pending, event)
            expected_revision = checkpoint.revision
            recorded = self.store.record_event_delivery_v2(
                checkpoint.checkpoint_key,
                event_id=pending.event_id,
                payload_digest=pending.payload_digest,
                cursor=cursor,
                expected_revision=expected_revision,
                claim_token=checkpoint.claim_token,
            )
            if recorded:
                pending.state = "delivered"
                pending.cursor = deepcopy(cursor)
                checkpoint.event_cursor = deepcopy(cursor)
                checkpoint.revision = expected_revision + 1
                continue
            else:
                authoritative = self.store.load_checkpoint_v2(checkpoint.checkpoint_key)
                if authoritative is None:
                    raise CheckpointError(
                        "checkpoint disappeared while recording event delivery",
                        code="checkpoint_not_found",
                    )
                matching = next(
                    (entry for entry in authoritative.event_outbox if entry.event_id == pending.event_id),
                    None,
                )
                if (
                    matching is None
                    or matching.state != "delivered"
                    or matching.payload_digest != pending.payload_digest
                    or matching.cursor != cursor
                ):
                    raise CheckpointError(
                        "checkpoint event delivery lost its revision",
                        code="checkpoint_store_conflict",
                    )
                pending.state = "delivered"
                pending.cursor = deepcopy(cursor)
                checkpoint.event_cursor = deepcopy(cursor)
                checkpoint.revision = authoritative.revision

    def _deliver_event(self, pending: EventOutboxEntry, event: RunEvent) -> EventCursor:
        if isinstance(self.event_store, IdempotentRunEventStore):
            cursor = self.event_store.append_once(
                pending.event_id,
                pending.payload_digest,
                event,
            )
            if not isinstance(cursor, EventCursor):
                raise CheckpointError(
                    "idempotent event store returned an invalid cursor",
                    code="event_cursor_invalid",
                )
        else:
            if self.event_store is not None:
                self.event_store.append(event)
            cursor = EventCursor(
                store_ref={"id": "events.raw-sink", "version": "1"},
                value={"event_id": pending.event_id},
                last_event_id=pending.event_id,
            )
        self.event_sink(event)
        return cursor

    def _checkpoint_created_event(self, checkpoint_key: str) -> CheckpointCreatedEvent:
        return CheckpointCreatedEvent(
            run_id=self.run_id,
            trace_id=self.trace_id,
            cycle_index=0,
            checkpoint_key=checkpoint_key,
            resume_attempt=1,
            event_id=self._stable_event_id_for(
                checkpoint_key,
                "checkpoint_created",
                "0",
            ),
        )

    def _with_stable_terminal_event_id(self, event: RunEvent) -> RunEvent:
        payload = event.to_dict()
        payload["event_id"] = self._stable_event_id(
            "terminal",
            event.type,
            str(event.cycle_index or 0),
        )
        return event_from_dict(payload)

    def _acknowledge_terminal(self) -> None:
        checkpoint = self._require_checkpoint()
        if checkpoint.terminal_result is None or checkpoint.terminal_acknowledged:
            return
        if any(entry.state == "pending" for entry in checkpoint.event_outbox):
            return
        revision = checkpoint.revision
        if not self.store.acknowledge_terminal_v2(
            checkpoint.checkpoint_key,
            expected_revision=revision,
        ):
            authoritative = self.store.load_checkpoint_v2(checkpoint.checkpoint_key)
            if authoritative is None or not authoritative.terminal_acknowledged:
                raise CheckpointError(
                    "checkpoint terminal acknowledgement lost its revision",
                    code="checkpoint_store_conflict",
                )
        authoritative = self.store.load_checkpoint_v2(checkpoint.checkpoint_key)
        if authoritative is not None:
            self.checkpoint = authoritative

    def _start_heartbeat(self) -> None:
        checkpoint = self._require_checkpoint()
        claim_token = checkpoint.claim_token
        if claim_token is None:
            return
        self._stop_heartbeat()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_error = None
        interval_seconds = max(self.lease_duration_ms / 3 / 1000, 0.01)

        def heartbeat() -> None:
            while not self._heartbeat_stop.wait(interval_seconds):
                now_ms = self._now_ms()
                try:
                    renewed = self.store.renew_checkpoint_claim_v2(
                        checkpoint.checkpoint_key,
                        claim_token=claim_token,
                        lease_expires_at_ms=now_ms + self.lease_duration_ms,
                        now_ms=now_ms,
                    )
                except Exception as exc:
                    self._heartbeat_error = CheckpointError(
                        f"checkpoint lease renewal failed: {exc}",
                        code="checkpoint_lease_lost",
                    )
                    return
                if not renewed:
                    self._heartbeat_error = CheckpointError(
                        "checkpoint lease renewal lost its claim",
                        code="checkpoint_lease_lost",
                    )
                    return

        self._heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"vv-agent-checkpoint-{checkpoint.checkpoint_key[:32]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._heartbeat_thread = None

    def _renew_claim_before_dispatch(self) -> None:
        self._assert_heartbeat()
        checkpoint = self._require_checkpoint()
        claim_token = checkpoint.claim_token
        if claim_token is None:
            raise CheckpointError(
                "checkpoint external dispatch requires an active claim",
                code="checkpoint_claim_active",
            )
        now_ms = self._now_ms()
        try:
            renewed = self.store.renew_checkpoint_claim_v2(
                checkpoint.checkpoint_key,
                claim_token=claim_token,
                lease_expires_at_ms=now_ms + self.lease_duration_ms,
                now_ms=now_ms,
            )
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint lease renewal failed before external dispatch: {exc}",
                code="checkpoint_lease_lost",
            ) from exc
        if not renewed:
            raise CheckpointError(
                "checkpoint lease renewal lost its claim before external dispatch",
                code="checkpoint_lease_lost",
            )
        checkpoint.lease_expires_at_ms = now_ms + self.lease_duration_ms

    def _assert_heartbeat(self) -> None:
        if self._heartbeat_error is not None:
            raise self._heartbeat_error

    def _stable_event_id(self, event_type: str, *coordinates: str) -> str:
        return self._stable_event_id_for(self.checkpoint_key, event_type, *coordinates)

    @staticmethod
    def _stable_event_id_for(checkpoint_key: str, event_type: str, *coordinates: str) -> str:
        source = "\0".join((checkpoint_key, event_type, *coordinates))
        return f"evt_{hashlib.sha256(source.encode()).hexdigest()[:32]}"

    def _tool_idempotency_key(self, cycle_index: int, tool_call_id: str) -> str:
        source = f"{self.checkpoint_key}\0{cycle_index}\0{tool_call_id}"
        return f"idem_{hashlib.sha256(source.encode()).hexdigest()[:32]}"

    @staticmethod
    def _llm_response_to_dict(response: LLMResponse) -> dict[str, Any]:
        return {
            "content": response.content,
            "tool_calls": [call.to_dict() for call in response.tool_calls],
            "raw": deepcopy(response.raw),
        }

    @staticmethod
    def _llm_response_from_dict(payload: dict[str, Any]) -> LLMResponse:
        raw = payload.get("raw")
        raw_payload: dict[str, Any] = deepcopy(raw) if isinstance(raw, dict) else {}
        return LLMResponse(
            content=str(payload.get("content") or ""),
            tool_calls=[ToolCall.from_dict(item) for item in payload.get("tool_calls", []) if isinstance(item, dict)],
            raw=raw_payload,
        )

    @staticmethod
    def _is_definitive_model_error(error: BaseException) -> bool:
        if getattr(error, "definitive_outcome", False) is True:
            return True
        text = str(error).lower()
        return any(
            marker in text
            for marker in (
                "context length",
                "context_length_exceeded",
                "maximum context length",
                "prompt is too long",
                "request too large",
            )
        )

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _require_checkpoint(self) -> CheckpointV2:
        if self.checkpoint is None:
            raise RuntimeError("checkpoint controller has not been admitted")
        return self.checkpoint
