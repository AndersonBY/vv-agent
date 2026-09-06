from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Thread
from typing import Any

import pytest
from support import FactoryModelProvider, require_tool_result

from vv_agent import (
    Agent,
    AgentStatus,
    CheckpointConfig,
    RunConfig,
    Runner,
    ToolCallOutcome,
    ToolContext,
    ToolDirective,
    ToolExecutionResult,
    function_tool,
)
from vv_agent.checkpoint import (
    CheckpointError,
    ReconciliationDecision,
    ReconciliationDecisionKind,
    ResumePolicy,
    compute_event_payload_digest,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.deferred import (
    AcceptDeferredDecision,
    DeferredCheckpointClaimed,
    DeferredResolutionConflict,
    DeferredResolutionResultInvalid,
    DeferredResolveDecision,
    DeferredToolHandle,
    _is_ambiguous_tool_error,
    validate_definitive_result,
)
from vv_agent.events import ToolCallCompletedEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.checkpoint_codec import checkpoint_to_dict
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.state import compute_tool_identity_key
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.types import LLMResponse, ToolCall, ToolResultStatus
from vv_agent.workspace import MemoryWorkspaceBackend

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


def _provider(factory: Callable[[], ScriptedLLM]) -> FactoryModelProvider:
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        function_call_available=True,
    )
    return FactoryModelProvider(factory=factory, resolved=resolved)


def _context(*, metadata: dict[str, Any] | None = None) -> ToolContext:
    return ToolContext(
        workspace=Path.cwd(),
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
        tool_call_id="call-without-checkpoint",
        metadata=metadata or {},
    )


def _run_checkpointed_tools(
    store: InMemoryCheckpointStore, key: str, tools: list[Any], tool_calls: list[ToolCall]
) -> tuple[Any, Any]:
    result = Runner.run_sync(
        Agent(name="checkpointed-test-agent", instructions="Run the operation.", model="test-model", tools=tools),
        "run",
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="", tool_calls=tool_calls)])),
            max_cycles=1,
            no_tool_policy="finish",
            checkpoint_config=CheckpointConfig(key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT, store=store),
        ),
    )
    return result, store.load_checkpoint(key)


def test_function_tool_preserves_closed_outcome_and_fails_closed_without_checkpoint() -> None:
    @function_tool(name="defer", params_json_schema=_EMPTY_SCHEMA)
    def defer(context: ToolContext) -> ToolCallOutcome:
        return context.defer()

    outcome = defer.to_executor().execute(
        ToolCall(id="call-without-checkpoint", name="defer", arguments={}),
        _context(),
    )

    assert isinstance(outcome, ToolCallOutcome)
    assert outcome.kind == "completed"
    assert outcome.result is not None
    assert outcome.result.status_code is ToolResultStatus.ERROR
    assert outcome.result.error_code == "deferred_requires_checkpoint"


def test_require_tool_result_rejects_deferred_and_unwraps_completed() -> None:
    handle = DeferredToolHandle(
        checkpoint_key="require-tool-result",
        operation_id="op_tool_cycle_1_require_result",
        attempt=1,
        request_digest="a" * 64,
    )
    deferred = ToolCallOutcome.Deferred(handle)

    with pytest.raises(AssertionError, match="deferred outcome"):
        require_tool_result(deferred)

    completed = ToolExecutionResult(
        tool_call_id="require-result",
        content="done",
        status_code=ToolResultStatus.SUCCESS,
    )
    assert require_tool_result(completed) is completed
    assert require_tool_result(ToolCallOutcome.Completed(completed)) is completed


def test_checkpointed_non_definitive_tool_outcome_uses_normal_wait_user_lifecycle() -> None:
    store = InMemoryCheckpointStore()

    @function_tool(name="wait_for_user", params_json_schema=_EMPTY_SCHEMA)
    def wait_for_user(_context: ToolContext) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id="",
            content="Choose an option.",
            status_code=ToolResultStatus.SUCCESS,
            directive=ToolDirective.WAIT_USER,
        )

    def model() -> ScriptedLLM:
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-wait", name="wait_for_user", arguments={})],
                )
            ]
        )

    result = Runner.run_sync(
        Agent(
            name="checkpointed-wait-agent",
            instructions="Ask for the user choice.",
            model="test-model",
            tools=[wait_for_user],
        ),
        "wait",
        run_config=RunConfig(
            model_provider=_provider(model),
            max_cycles=1,
            no_tool_policy="finish",
            checkpoint_config=CheckpointConfig(
                key="checkpointed-wait",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            ),
        ),
    )

    assert result.status is AgentStatus.WAIT_USER
    checkpoint = store.load_checkpoint("checkpointed-wait")
    assert checkpoint is not None
    assert checkpoint.terminal_result is not None
    assert checkpoint.claim_token is None
    assert checkpoint.tool_journal == []
    assert sum(event.event.get("type") == "tool_call_completed" for event in checkpoint.event_outbox) == 1
    completed = [event for event in result.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].execution_started is True
    assert completed[0].status == "wait_response"
    assert completed[0].directive == "wait_user"


def test_incomplete_deferred_batch_reconciles_unclassified_wait_user() -> None:
    store = InMemoryCheckpointStore()
    handles: list[DeferredToolHandle] = []

    @function_tool(name="defer_first", params_json_schema=_EMPTY_SCHEMA)
    def defer_first(context: ToolContext) -> ToolCallOutcome:
        outcome = context.defer()
        assert outcome.handle is not None
        handles.append(outcome.handle)
        return outcome

    @function_tool(name="wait_for_user", params_json_schema=_EMPTY_SCHEMA)
    def wait_for_user(_context: ToolContext) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id="",
            content="Choose an option.",
            status_code=ToolResultStatus.SUCCESS,
            directive=ToolDirective.WAIT_USER,
        )

    result, checkpoint = _run_checkpointed_tools(
        store,
        "incomplete-deferred-wait-user",
        [defer_first, wait_for_user],
        [
            ToolCall(id="call-defer", name="defer_first", arguments={}),
            ToolCall(id="call-wait", name="wait_for_user", arguments={}),
        ],
    )

    assert result.status is AgentStatus.RECONCILIATION_REQUIRED
    assert checkpoint is not None
    assert checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED
    assert checkpoint.claim_token is None
    assert [entry.state.value for entry in checkpoint.tool_journal] == ["started", "ambiguous"]
    assert checkpoint.tool_journal[0].deferred_handle is None
    assert handles

    decision = store.resolve_deferred(
        handles[0],
        ToolExecutionResult(
            tool_call_id="call-defer",
            content="accepted",
            status_code=ToolResultStatus.SUCCESS,
        ),
    )
    assert decision.kind == "not_admitted"


@pytest.mark.parametrize(
    ("error_code", "metadata", "expected_status", "expected_states", "mixed", "has_handle"),
    [
        ("tool_timeout", {}, AgentStatus.RECONCILIATION_REQUIRED, ("ambiguous",), False, False),
        ("tool_execution_failed", {}, AgentStatus.RECONCILIATION_REQUIRED, ("ambiguous",), False, False),
        ("tool_orchestrator_error", {}, AgentStatus.RECONCILIATION_REQUIRED, ("ambiguous",), False, False),
        ("tool_execution_failed", {}, AgentStatus.RECONCILIATION_REQUIRED, ("started", "ambiguous"), True, False),
        ("tool_execution_failed", {"definitive_outcome": True}, AgentStatus.DEFERRED, ("deferred", "failed"), True, True),
    ],
)
def test_checkpointed_tool_failures_use_definitive_or_ambiguous_lifecycles(
    error_code: str,
    metadata: dict[str, Any],
    expected_status: AgentStatus,
    expected_states: tuple[str, ...],
    mixed: bool,
    has_handle: bool,
) -> None:
    @function_tool(name="defer_first", params_json_schema=_EMPTY_SCHEMA)
    def defer_first(context: ToolContext) -> ToolCallOutcome:
        return context.defer()

    @function_tool(name="uncertain_second", params_json_schema=_EMPTY_SCHEMA)
    def uncertain_second(_context: ToolContext) -> ToolExecutionResult:
        return ToolExecutionResult("", "unknown", ToolResultStatus.ERROR, error_code=error_code, metadata=metadata)

    tools: list[Any] = [uncertain_second]
    calls = [ToolCall(id="call-uncertain-second", name="uncertain_second", arguments={})]
    if mixed:
        tools.insert(0, defer_first)
        calls.insert(0, ToolCall(id="call-defer-first", name="defer_first", arguments={}))
    result, checkpoint = _run_checkpointed_tools(
        InMemoryCheckpointStore(),
        f"checkpointed-{error_code}-{mixed}",
        tools,
        calls,
    )
    assert result.status is expected_status
    assert checkpoint is not None
    assert (checkpoint.status, checkpoint.claim_token) == (expected_status, None)
    assert tuple(entry.state.value for entry in checkpoint.tool_journal) == expected_states
    if mixed:
        assert (checkpoint.tool_journal[0].deferred_handle is not None) is has_handle
        if not has_handle:
            lifecycle_types = {
                "tool_call_deferred",
                "tool_call_completed",
            }
            assert [event.type for event in result.events if event.type in lifecycle_types] == []
            assert all(entry.event["type"] not in lifecycle_types for entry in checkpoint.event_outbox)
            assert any(event.type == "operation_ambiguous" for event in result.events)
            assert any(event.type == "reconciliation_required" for event in result.events)
    if has_handle:
        assert checkpoint.tool_journal[1].error is not None and not checkpoint.tool_journal[1].error.retryable


def test_ambiguous_error_marker_is_strict_and_store_admission_is_fail_closed() -> None:
    result = ToolExecutionResult(
        "call-defer",
        "unknown",
        ToolResultStatus.ERROR,
        error_code="tool_execution_failed",
        metadata={"definitive_outcome": "false"},
    )
    with pytest.raises(DeferredResolutionResultInvalid):
        validate_definitive_result(result)

    success_with_error = ToolExecutionResult(
        "call-defer",
        "invalid success",
        ToolResultStatus.SUCCESS,
    )
    success_with_error.error_code = "tool_execution_failed"
    assert not _is_ambiguous_tool_error(success_with_error)
    with pytest.raises(DeferredResolutionResultInvalid) as caught:
        validate_definitive_result(success_with_error)
    assert caught.value.code == "tool_result_invalid"

    handle = DeferredToolHandle("ambiguous-admission", "op_tool_cycle_1_call_ambiguous", 1, "a" * 64)
    store = InMemoryCheckpointStore()
    before = _deferred_checkpoint(store, handle, admit=False)
    assert before is not None
    with pytest.raises(CheckpointError, match="deferred admission accepts deferred outcomes only") as caught:
        store.admit_deferred_batch(
            before,
            outcomes=[(ToolCall(id="call-defer", name="defer", arguments={}), ToolCallOutcome.Completed(result))],
            claim_token="claim",
            expected_revision=before.revision,
            claimed_cycle=1,
        )
    assert caught.value.code == "deferred_admission_completed_outcome_invalid"
    after = store.load_checkpoint(handle.checkpoint_key)
    assert after is not None and checkpoint_to_dict(after) == checkpoint_to_dict(before)

    with pytest.raises(CheckpointError, match="deferred admission accepts deferred outcomes only") as caught:
        store.admit_deferred_batch(
            before,
            outcomes=[
                (
                    ToolCall(id="call-defer", name="defer", arguments={}),
                    ToolCallOutcome.Completed(success_with_error),
                )
            ],
            claim_token="claim",
            expected_revision=before.revision,
            claimed_cycle=1,
        )
    assert caught.value.code == "deferred_admission_completed_outcome_invalid"
    after = store.load_checkpoint(handle.checkpoint_key)
    assert after is not None and checkpoint_to_dict(after) == checkpoint_to_dict(before)


@pytest.mark.parametrize(
    ("error_code", "metadata", "accepted"),
    [
        ("tool_timeout", {}, False),
        ("tool_cancelled", {}, False),
        ("tool_connection_lost", {}, False),
        ("tool_execution_failed", {}, False),
        ("tool_orchestrator_error", {}, False),
        ("tool_execution_failed", {"definitive_outcome": True}, False),
    ],
)
def test_mixed_deferred_and_ambiguous_admission_is_atomic(
    error_code: str,
    metadata: dict[str, Any],
    accepted: bool,
) -> None:
    deferred_handle = DeferredToolHandle(
        "mixed-admission",
        "op_tool_cycle_1_call_deferred",
        1,
        "a" * 64,
    )
    store = InMemoryCheckpointStore()
    before = _deferred_checkpoint(store, deferred_handle, admit=False)
    assert before is not None
    ambiguous_entry = deepcopy(before.tool_journal[0])
    ambiguous_entry.operation_id = "op_tool_cycle_1_call_ambiguous"
    ambiguous_entry.request_digest = "b" * 64
    ambiguous_entry.tool_call_id = "call-ambiguous"
    ambiguous_entry.tool_name = "ambiguous"
    ambiguous_entry.idempotency_key = "idem-ambiguous"
    before.tool_journal.append(ambiguous_entry)
    assert store.progress_checkpoint(before, claim_token="claim", expected_revision=before.revision)
    before.revision += 1
    before_wire = checkpoint_to_dict(before)
    result = ToolExecutionResult(
        "call-ambiguous",
        "unknown",
        ToolResultStatus.ERROR,
        error_code=error_code,
        metadata=metadata,
    )
    outcomes = [
        (
            ToolCall(id="call-defer", name="defer", arguments={}),
            ToolCallOutcome.Deferred(deferred_handle),
        ),
        (
            ToolCall(id="call-ambiguous", name="ambiguous", arguments={}),
            ToolCallOutcome.Completed(result),
        ),
    ]

    if accepted:
        assert store.admit_deferred_batch(
            before,
            outcomes=outcomes,
            claim_token="claim",
            expected_revision=before.revision,
            claimed_cycle=1,
        )
        after = store.load_checkpoint(deferred_handle.checkpoint_key)
        assert after is not None
        assert after.status is AgentStatus.DEFERRED
        assert after.claim_token is None
        assert after.revision == before.revision + 1
        assert [entry.state.value for entry in after.tool_journal] == ["deferred", "failed"]
        assert [entry.event["type"] for entry in after.event_outbox] == [
            "tool_call_deferred",
            "tool_call_completed",
        ]
        return

    with pytest.raises(CheckpointError, match="deferred admission accepts deferred outcomes only") as caught:
        store.admit_deferred_batch(
            before,
            outcomes=outcomes,
            claim_token="claim",
            expected_revision=before.revision,
            claimed_cycle=1,
        )
    assert caught.value.code == "deferred_admission_completed_outcome_invalid"
    after = store.load_checkpoint(deferred_handle.checkpoint_key)
    assert after is not None
    assert checkpoint_to_dict(after) == before_wire
    assert after.revision == before.revision
    assert after.claim_token == "claim"
    assert [entry.state.value for entry in after.tool_journal] == ["started", "started"]
    assert after.event_outbox == before.event_outbox


def test_completed_only_admission_keeps_claim_owner_until_cycle_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[str | None, str | None, bool, str | None]] = []
    original_commit = CheckpointResumeController.commit_cycle

    def capture_commit(self: CheckpointResumeController, **kwargs: Any) -> None:
        assert self.checkpoint is not None
        observed.append(
            (self._owned_claim_token, self._active_claim_mode, self._heartbeat_thread is not None, self.checkpoint.claim_token)
        )
        original_commit(self, **kwargs)

    monkeypatch.setattr(CheckpointResumeController, "commit_cycle", capture_commit)

    @function_tool(name="complete", params_json_schema=_EMPTY_SCHEMA)
    def complete(_context: ToolContext) -> str:
        return "accepted"

    result, _ = _run_checkpointed_tools(
        InMemoryCheckpointStore(), "completed-only", [complete], [ToolCall(id="call-complete", name="complete", arguments={})]
    )
    assert result.status is AgentStatus.MAX_CYCLES
    assert len(observed) == 1
    claim_token, claim_mode, heartbeat_active, checkpoint_claim = observed[0]
    assert claim_token is not None and (claim_mode, heartbeat_active, checkpoint_claim) == ("continue", True, claim_token)


class _CountingStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.admission_calls = 0

    def admit_deferred_batch(self, *args: Any, **kwargs: Any) -> bool:
        self.admission_calls += 1
        return super().admit_deferred_batch(*args, **kwargs)


class _RejectingOutboxPreflightStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.preflight_calls = 0

    def preflight_tool_batch(self, *args: Any, **kwargs: Any) -> bool:
        self.preflight_calls += 1
        return False


def test_outbox_preflight_rejection_happens_before_external_tool_effect() -> None:
    store = _RejectingOutboxPreflightStore()
    effects: list[str] = []

    @function_tool(name="provider_write", params_json_schema=_EMPTY_SCHEMA)
    def provider_write(_context: ToolContext) -> str:
        effects.append("provider")
        return "accepted"

    with pytest.raises(CheckpointError, match="outbox preflight"):
        Runner.run_sync(
            Agent(
                name="outbox-preflight-agent",
                instructions="Run the provider write.",
                model="test-model",
                tools=[provider_write],
            ),
            "run provider",
            run_config=RunConfig(
                model_provider=_provider(
                    lambda: ScriptedLLM(
                        steps=[
                            LLMResponse(
                                content="",
                                tool_calls=[ToolCall(id="call-provider", name="provider_write", arguments={})],
                            )
                        ]
                    )
                ),
                max_cycles=1,
                no_tool_policy="finish",
                checkpoint_config=CheckpointConfig(
                    key="outbox-preflight-rejected",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                ),
            ),
        )

    assert effects == []
    assert store.preflight_calls == 1


def test_store_acceptance_rejects_partial_ambiguity_batch() -> None:
    from vv_agent.checkpoint import OperationState

    store = InMemoryCheckpointStore()
    handle = DeferredToolHandle(
        checkpoint_key="acceptance-batch",
        operation_id="op_tool_cycle_1_call_1",
        attempt=1,
        request_digest="a" * 64,
    )
    checkpoint = _deferred_checkpoint(store, handle)
    assert checkpoint is not None
    with store._lock:
        current = store._store[checkpoint.checkpoint_key]
        current.status = AgentStatus.RECONCILIATION_REQUIRED
        current.claim_token = None
        current.claimed_cycle = None
        current.lease_expires_at_ms = None
        current.tool_journal[0].state = OperationState.AMBIGUOUS
        current.tool_journal[0].deferred_handle = None
        second = deepcopy(current.tool_journal[0])
        second.operation_id = "op_tool_cycle_1_call_2"
        second.tool_call_id = "call-2"
        second.request_digest = "b" * 64
        current.tool_journal.append(second)
        current.cycle_index = 0

    claimed = store.claim_checkpoint(
        "acceptance-batch",
        1,
        claim_token="recovery-claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="recovery",
    )
    assert claimed is not None
    assert not store.accept_deferred_batch(
        claimed,
        decisions=[AcceptDeferredDecision(handle=handle)],
        claim_token="recovery-claim",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    retained = store.load_checkpoint("acceptance-batch")
    assert retained is not None
    assert [entry.state for entry in retained.tool_journal] == [OperationState.AMBIGUOUS, OperationState.AMBIGUOUS]


def test_store_acceptance_rejects_model_and_tool_ambiguity_batch() -> None:
    from vv_agent.checkpoint import OperationKind, OperationState
    from vv_agent.events import ModelCallFailedEvent, ModelCallStartedEvent
    from vv_agent.runtime.state import EventOutboxEntry, OperationJournalEntry
    from vv_agent.types import ModelCallRecord, ModelCallStatus, TokenUsage

    store = InMemoryCheckpointStore()
    handle = DeferredToolHandle(
        checkpoint_key="acceptance-model-tool-batch",
        operation_id="op_tool_cycle_1_call_1",
        attempt=1,
        request_digest="a" * 64,
    )
    checkpoint = _deferred_checkpoint(store, handle)
    assert checkpoint is not None
    journal_fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "parity" / "operation_journal.json").read_text(encoding="utf-8")
    )
    model_payload = next(item["entry"] for item in journal_fixture["valid_entries"] if item["name"] == "model_ambiguous")
    model_entry = OperationJournalEntry.from_dict({**model_payload, "cycle_index": 1})
    assert model_entry.kind is OperationKind.MODEL
    assert model_entry.state is OperationState.AMBIGUOUS
    assert model_entry.model_operation is not None
    usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    record = ModelCallRecord(
        call_id=model_entry.call_id or "model-call",
        operation_id=model_entry.operation_id,
        attempt=model_entry.attempt,
        operation=model_entry.model_operation,
        cycle_index=model_entry.cycle_index,
        backend=model_entry.backend or "test",
        model=model_entry.model or "test-model",
        status=ModelCallStatus.AMBIGUOUS,
        usage=usage,
        error_code="model_outcome_ambiguous",
    )
    started_event = ModelCallStartedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        call_id=record.call_id,
        operation_id=record.operation_id,
        attempt=record.attempt,
        operation=record.operation,
        cycle_index=record.cycle_index,
        backend=record.backend,
        model=record.model,
        event_id="evt_model_ambiguous_started",
        created_at=0.0,
    ).to_dict()
    failed_event = ModelCallFailedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        call_id=record.call_id,
        operation_id=record.operation_id,
        attempt=record.attempt,
        operation=record.operation,
        cycle_index=record.cycle_index,
        backend=record.backend,
        model=record.model,
        outcome="ambiguous",
        usage=usage,
        error_code="model_outcome_ambiguous",
        event_id="evt_model_ambiguous_failed",
        created_at=0.0,
    ).to_dict()
    with store._lock:
        current = store._store[checkpoint.checkpoint_key]
        current.status = AgentStatus.RECONCILIATION_REQUIRED
        current.claim_token = None
        current.claimed_cycle = None
        current.lease_expires_at_ms = None
        current.model_call_journal = [model_entry]
        current.model_calls = [record]
        current.event_outbox.extend(
            [
                EventOutboxEntry.pending(started_event["event_id"], started_event),
                EventOutboxEntry.pending(failed_event["event_id"], failed_event),
            ]
        )
        current.tool_journal[0].state = OperationState.AMBIGUOUS
        current.tool_journal[0].deferred_handle = None

    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="recovery-model-tool",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="recovery",
    )
    assert claimed is not None
    assert not store.accept_deferred_batch(
        claimed,
        decisions=[AcceptDeferredDecision(handle=handle)],
        claim_token="recovery-model-tool",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.model_call_journal[0].state is OperationState.AMBIGUOUS
    assert retained.tool_journal[0].state is OperationState.AMBIGUOUS


def test_store_acceptance_replays_admitted_deferred_batch_without_claim_or_revision() -> None:
    store = InMemoryCheckpointStore()
    handle = DeferredToolHandle(
        checkpoint_key="acceptance-replay",
        operation_id="op_tool_cycle_1_call_1",
        attempt=1,
        request_digest="a" * 64,
    )
    checkpoint = _deferred_checkpoint(store, handle)
    assert checkpoint is not None
    revision = checkpoint.revision

    assert store.accept_deferred_batch(
        checkpoint,
        decisions=[AcceptDeferredDecision(handle=handle)],
        claim_token="",
        expected_revision=revision,
        claimed_cycle=1,
    )
    retained = store.load_checkpoint(handle.checkpoint_key)
    assert retained is not None
    assert retained.status is AgentStatus.DEFERRED
    assert retained.revision == revision


def test_runner_admits_mixed_batch_once_and_projects_no_deferred_tool_message() -> None:
    store = _CountingStore()
    effects: list[str] = []
    handles: list[DeferredToolHandle] = []

    @function_tool(
        name="defer_write",
        params_json_schema=_EMPTY_SCHEMA,
        tool_metadata={"idempotency": "supported"},
    )
    def defer_write(context: ToolContext) -> ToolCallOutcome:
        effects.append("defer")
        outcome = context.defer()
        if outcome.kind == "deferred":
            assert outcome.handle is not None
            handles.append(outcome.handle)
        return outcome

    @function_tool(
        name="complete_write",
        params_json_schema=_EMPTY_SCHEMA,
        tool_metadata={"idempotency": "supported"},
    )
    def complete_write(_context: ToolContext) -> str:
        effects.append("complete")
        return "accepted"

    def model() -> ScriptedLLM:
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call-defer", name="defer_write", arguments={}),
                        ToolCall(id="call-complete", name="complete_write", arguments={}),
                    ],
                )
            ]
        )

    agent = Agent(
        name="deferred-batch-agent",
        instructions="Run both operations.",
        model="test-model",
        tools=[defer_write, complete_write],
    )
    config = RunConfig(
        model_provider=_provider(model),
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="deferred-batch",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )

    result = Runner.run_sync(agent, "run both", run_config=config)
    checkpoint = store.load_checkpoint("deferred-batch")
    assert checkpoint is not None

    assert result.status is AgentStatus.DEFERRED
    assert result.raw_result.wait_reason == "deferred_pending"
    assert effects == ["defer", "complete"]
    assert len(handles) == 1
    assert store.admission_calls == 1
    assert checkpoint.status is AgentStatus.DEFERRED
    assert checkpoint.claim_token is None
    assert [entry.state.value for entry in checkpoint.tool_journal] == ["deferred", "succeeded"]
    assert [event.type for event in result.events if event.type.startswith("tool_call_")] == [
        "tool_call_planned",
        "tool_call_started",
        "tool_call_planned",
        "tool_call_started",
        "tool_call_completed",
        "tool_call_deferred",
    ]
    completed_event = next(event for event in checkpoint.event_outbox if event.event["type"] == "tool_call_completed")
    assert completed_event.event["checkpoint_key"] == checkpoint.checkpoint_key
    completed_event.verify_payload()
    assert all(message.tool_call_id != "call-defer" for message in result.raw_result.messages)


def test_deferred_resolution_is_receipt_first_and_concurrent_same_result_replays() -> None:
    handle = DeferredToolHandle(
        checkpoint_key="tenant/run",
        operation_id="op_tool_cycle_1_call_defer",
        attempt=1,
        request_digest="a" * 64,
    )
    result = ToolExecutionResult(
        tool_call_id="call-defer",
        content="accepted",
        status_code=ToolResultStatus.SUCCESS,
    )

    # Build a minimal admitted checkpoint through the normal producer path so
    # the callback test exercises the real receipt/journal identity checks.
    store = InMemoryCheckpointStore()
    checkpoint = _deferred_checkpoint(store, handle)
    assert checkpoint.status is AgentStatus.DEFERRED

    barrier = Barrier(2)
    decisions: list[str] = []

    def resolve() -> None:
        barrier.wait()
        decisions.append(store.resolve_deferred(handle, result).kind)

    workers = [Thread(target=resolve), Thread(target=resolve)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert sorted(decisions) == ["applied_ready", "replayed"]
    current = store.load_checkpoint(handle.checkpoint_key)
    assert current is not None
    assert current.status is AgentStatus.RUNNING
    revision = current.revision
    replay = store.resolve_deferred(handle, result)
    assert replay.kind == "replayed"
    replayed_checkpoint = store.load_checkpoint(handle.checkpoint_key)
    assert replayed_checkpoint is not None
    assert replayed_checkpoint.revision == revision
    receipt_event = next(event for event in replayed_checkpoint.event_outbox if event.event.get("type") == "tool_call_completed")
    identity_key = compute_tool_identity_key(
        handle.checkpoint_key,
        handle.operation_id,
        handle.attempt,
        "call-defer",
        handle.request_digest,
    )
    assert receipt_event.event_id == f"evt_receipt_{identity_key}"

    with pytest.raises(DeferredResolutionConflict):
        store.resolve_deferred(
            handle,
            ToolExecutionResult(
                tool_call_id="call-defer",
                content="different",
                status_code=ToolResultStatus.SUCCESS,
            ),
        )


def test_real_producer_claimed_checkpoint_resolution_is_typed_error_without_writes() -> None:
    handle = DeferredToolHandle(
        checkpoint_key="claimed-resolution",
        operation_id="op_tool_cycle_1_call_defer",
        attempt=1,
        request_digest="a" * 64,
    )
    result = ToolExecutionResult(
        tool_call_id="call-defer",
        content="accepted",
        status_code=ToolResultStatus.SUCCESS,
    )
    store = InMemoryCheckpointStore()
    admitted = _deferred_checkpoint(store, handle)
    assert admitted is not None

    # A live worker claim can race the callback after admission.  The
    # authoritative record is intentionally made running-with-barrier here;
    # no public claim path may acquire a deferred barrier, but the resolver
    # must still fail closed if a claimed record is observed.
    with store._lock:
        current = store._store[handle.checkpoint_key]
        current.status = AgentStatus.RUNNING
        current.claim_token = "owner-b"
        current.claimed_cycle = 1
        current.lease_expires_at_ms = 10_000
        current.revision += 1

    before = store.load_checkpoint(handle.checkpoint_key)
    assert before is not None
    before_wire = checkpoint_to_dict(before)
    with pytest.raises(DeferredCheckpointClaimed) as caught:
        store.resolve_deferred(handle, result)

    assert caught.value.code == "deferred_checkpoint_claimed"
    assert not isinstance(caught.value, DeferredResolveDecision)
    after = store.load_checkpoint(handle.checkpoint_key)
    assert after is not None
    assert checkpoint_to_dict(after) == before_wire
    with store._lock:
        assert store._deferred_receipts == {}


def test_resolution_producer_matches_contract_receipt_and_event_jcs_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = json.loads((Path(__file__).parent / "fixtures" / "parity" / "deferred_tool.json").read_text(encoding="utf-8"))
    receipt_contract = contract["resolution"]["receipt_index"]
    canonical = receipt_contract["canonical_entry"]
    event_golden = next(item for item in receipt_contract["golden_digest_vectors"] if item["name"] == "success_event_payload")
    handle = DeferredToolHandle.from_dict(canonical["handle"])
    result = ToolExecutionResult.from_dict(canonical["result"])
    store = InMemoryCheckpointStore()
    _deferred_checkpoint(
        store,
        handle,
        tool_call_id=result.tool_call_id,
        cycle_index=2,
        root_run_id="run_deferred",
        trace_id="trace_deferred",
        tool_name="remote_write",
    )

    # Freeze only the event clock; the event and receipt bytes still come from
    # the real store and typed event producers rather than a copied digest.
    monkeypatch.setattr("vv_agent.events.event_created_at", lambda: event_golden["value"]["created_at"])
    decision = store.resolve_deferred(handle, result)

    assert decision.kind == "applied_ready"
    assert decision.receipt is not None
    assert decision.receipt.handle_key == canonical["handle_key"]
    assert decision.receipt.result_digest == canonical["result_digest"]
    assert decision.receipt.event_id == canonical["event_id"]
    assert decision.receipt.event_payload_digest == canonical["event_payload_digest"]
    checkpoint = store.load_checkpoint(handle.checkpoint_key)
    assert checkpoint is not None
    completed_event = next(entry for entry in checkpoint.event_outbox if entry.event_id == canonical["event_id"])
    assert completed_event.event == event_golden["value"]
    assert compute_event_payload_digest(completed_event.event) == canonical["event_payload_digest"]
    completed_event.verify_payload()

    tampered = decision.receipt.to_dict()
    tampered["event_id"] = "evt_receipt_" + "0" * 64
    with pytest.raises(ValueError, match="deferred_receipt_identity_invalid"):
        type(decision.receipt).from_dict(tampered)


def test_recovery_accepts_deferred_batch_under_recovery_claim_without_model_dispatch() -> None:
    store = InMemoryCheckpointStore()
    handles: list[DeferredToolHandle] = []
    effects = 0
    model_calls = 0

    @function_tool(name="crash_after_acceptance", params_json_schema=_EMPTY_SCHEMA)
    def crash_after_acceptance(context: ToolContext) -> ToolCallOutcome:
        nonlocal effects
        effects += 1
        outcome = context.defer()
        assert outcome.handle is not None
        handles.append(outcome.handle)
        raise SystemExit("simulated crash after provider acceptance")

    def first_model() -> ScriptedLLM:
        nonlocal model_calls
        model_calls += 1
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-crash-defer", name="crash_after_acceptance", arguments={})],
                )
            ]
        )

    agent = Agent(
        name="deferred-recovery-agent",
        instructions="Accept the external operation and wait.",
        model="test-model",
        tools=[crash_after_acceptance],
    )
    initial_config = RunConfig(
        model_provider=_provider(first_model),
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="deferred-recovery",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            capability_refs={"reconciliation_provider": {"id": "test.reconciler", "version": "1"}},
        ),
    )
    with pytest.raises(SystemExit, match="simulated crash after provider acceptance"):
        Runner.run_sync(agent, "accept external operation", run_config=initial_config)
    assert effects == 1
    assert len(handles) == 1
    crashed = store.load_checkpoint("deferred-recovery")
    assert crashed is not None
    assert crashed.tool_journal[0].state.value == "started"
    with store._lock:
        store._store["deferred-recovery"].lease_expires_at_ms = 1

    class Provider:
        def reconcile(self, observation: Any) -> ReconciliationDecision:
            assert observation.operation_id == handles[0].operation_id
            return ReconciliationDecision(
                ReconciliationDecisionKind.ACCEPT_DEFERRED,
                handle=handles[0],
            )

    resumed = Runner.run_sync(
        agent,
        "accept external operation",
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[])),
            max_cycles=1,
            no_tool_policy="finish",
            checkpoint_config=initial_config.checkpoint_config,
            reconciliation_provider=Provider(),
        ),
    )

    assert resumed.status is AgentStatus.DEFERRED
    assert resumed.raw_result.wait_reason == "deferred_pending"
    assert model_calls == 1
    assert effects == 1
    retained = store.load_checkpoint("deferred-recovery")
    assert retained is not None
    assert retained.status is AgentStatus.DEFERRED
    assert retained.claim_token is None
    assert retained.tool_journal[0].state.value == "deferred"
    assert [event.type for event in resumed.events if event.type in {"reconciliation_resolved", "tool_call_deferred"}] == [
        "reconciliation_resolved",
        "tool_call_deferred",
    ]


def _deferred_checkpoint(
    store: InMemoryCheckpointStore,
    handle: DeferredToolHandle,
    *,
    tool_call_id: str = "call-defer",
    cycle_index: int = 1,
    root_run_id: str | None = None,
    trace_id: str | None = None,
    tool_name: str = "defer",
    admit: bool = True,
):
    from vv_agent.checkpoint import OperationKind, OperationState, ToolIdempotency
    from vv_agent.runtime.checkpoint_codec import checkpoint_from_dict
    from vv_agent.runtime.state import OperationJournalEntry

    fixture = json.loads((Path(__file__).parent / "fixtures" / "parity" / "checkpoint_codec.json").read_text(encoding="utf-8"))
    payload = deepcopy(next(case["payload"] for case in fixture["valid_cases"] if case["name"] == "minimal_running"))
    payload["checkpoint_key"] = handle.checkpoint_key
    if root_run_id is not None:
        payload["root_run_id"] = root_run_id
    if trace_id is not None:
        payload["trace_id"] = trace_id
    if cycle_index > 1:
        payload["cycle_index"] = cycle_index - 1
    checkpoint = checkpoint_from_dict(payload)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        handle.checkpoint_key,
        cycle_index,
        claim_token="claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.tool_journal.append(
        OperationJournalEntry(
            kind=OperationKind.TOOL,
            operation_id=handle.operation_id,
            cycle_index=cycle_index,
            attempt=handle.attempt,
            state=OperationState.STARTED,
            request_digest=handle.request_digest,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments={},
            idempotency_support=ToolIdempotency.SUPPORTED,
            idempotency_key="idem",
        )
    )
    assert store.progress_checkpoint(claimed, claim_token="claim", expected_revision=claimed.revision)
    claimed.revision += 1
    if not admit:
        return claimed
    assert store.admit_deferred_batch(
        claimed,
        outcomes=[(ToolCall(id=tool_call_id, name="defer", arguments={}), ToolCallOutcome.Deferred(handle))],
        claim_token="claim",
        expected_revision=claimed.revision,
        claimed_cycle=cycle_index,
    )
    return store.load_checkpoint(handle.checkpoint_key)
