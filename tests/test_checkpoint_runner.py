from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

import pytest
from support import FactoryModelProvider

from vv_agent import (
    Agent,
    CheckpointConfig,
    ContextFragment,
    MemorySession,
    OutputValidationContext,
    OutputValidationResult,
    RunConfig,
    Runner,
    ToolContext,
    function_tool,
)
from vv_agent.checkpoint import (
    AmbiguousModelPolicy,
    AmbiguousToolPolicy,
    CheckpointError,
    OperationKind,
    OperationState,
    ReconciliationDecision,
    ReconciliationDecisionKind,
    ReconciliationError,
    ResumeObservation,
    ResumePolicy,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import CheckpointResumedEvent, ToolCallCompletedEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import BaseRuntimeHook, BeforeToolCallEvent
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.checkpoint_codec import checkpoint_to_dict
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.state import CheckpointRenewal, RenewOutcome
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools.outputs import ToolOutputError
from vv_agent.types import (
    AgentStatus,
    CompletionReason,
    LLMResponse,
    Message,
    ToolCall,
    ToolExecutionResult,
    ToolResultStatus,
)


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        function_call_available=True,
    )


def _provider(factory: Callable[[], Any]) -> FactoryModelProvider:
    return FactoryModelProvider(factory=factory, resolved=_resolved())


def _config(
    store: InMemoryCheckpointStore,
    *,
    key: str,
    provider: FactoryModelProvider,
    max_cycles: int = 1,
    no_tool_policy: Literal["continue", "wait_user", "finish"] = "finish",
    session: MemorySession | None = None,
    capability_refs: dict[str, dict[str, str]] | None = None,
    ambiguous_tool_policy: AmbiguousToolPolicy | None = None,
) -> RunConfig:
    checkpoint_config = CheckpointConfig(
        key=key,
        resume_policy=ResumePolicy.RESUME_IF_PRESENT,
        store=store,
        capability_refs=capability_refs or {},
    )
    if ambiguous_tool_policy is not None:
        checkpoint_config.ambiguous_tool_policy = ambiguous_tool_policy
    return RunConfig(
        model_provider=provider,
        max_cycles=max_cycles,
        no_tool_policy=no_tool_policy,
        session=session,
        checkpoint_config=checkpoint_config,
    )


class _CrashBeforeTerminalFinalizeStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.crash_next_finalize = True

    def _finalize(self, finalize: Callable[..., bool], checkpoint: Any, **kwargs: Any) -> bool:
        if self.crash_next_finalize:
            self.crash_next_finalize = False
            raise SystemExit("crash before terminal finalize")
        return finalize(checkpoint, **kwargs)

    def finalize_checkpoint(
        self,
        checkpoint: Any,
        *,
        expected_revision: int,
    ) -> bool:
        return self._finalize(super().finalize_checkpoint, checkpoint, expected_revision=expected_revision)

    def finalize_claimed_checkpoint(self, checkpoint: Any, *, claim_token: str, expected_revision: int) -> bool:
        return self._finalize(
            super().finalize_claimed_checkpoint,
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )


class _CancelAfterProgressStore(InMemoryCheckpointStore):
    def progress_checkpoint(self, checkpoint: Any, *, claim_token: str, expected_revision: int) -> bool:
        written = super().progress_checkpoint(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if written:
            with self._lock:
                current = self._store[checkpoint.checkpoint_key]
                if current.tool_journal:
                    current.cancel_requested = True
        return written


class _CancelAtReceiptStore(InMemoryCheckpointStore):
    def record_tool_receipt(self, checkpoint: Any, **kwargs: Any) -> bool:
        with self._lock:
            current = self._store[checkpoint.checkpoint_key]
            current.cancel_requested = True
        return super().record_tool_receipt(checkpoint, **kwargs)


class _LeaseErrorAfterProgressStore(InMemoryCheckpointStore):
    def progress_checkpoint(self, checkpoint: Any, *, claim_token: str, expected_revision: int) -> bool:
        written = super().progress_checkpoint(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )
        if written:
            with self._lock:
                if self._store[checkpoint.checkpoint_key].tool_journal:
                    raise CheckpointError("checkpoint lease lost", code="checkpoint_lease_lost")
        return written


class _CancelBeforeCommitStore(InMemoryCheckpointStore):
    def commit_checkpoint(self, checkpoint: Any, *, claim_token: str, expected_revision: int) -> bool:
        with self._lock:
            self._store[checkpoint.checkpoint_key].cancel_requested = True
        return super().commit_checkpoint(
            checkpoint,
            claim_token=claim_token,
            expected_revision=expected_revision,
        )


class _ClaimLostOnRenewStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.renew_calls = 0
        self.checkpoint_at_loss: Any | None = None

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal:
        self.renew_calls += 1
        if self.renew_calls == 1:
            self.checkpoint_at_loss = self.load_checkpoint(checkpoint_key)
            assert self.checkpoint_at_loss is not None
            return CheckpointRenewal(
                outcome=RenewOutcome.CLAIM_LOST,
                revision=self.checkpoint_at_loss.revision,
            )
        return super().renew_checkpoint_claim(
            checkpoint_key,
            claim_token=claim_token,
            lease_expires_at_ms=lease_expires_at_ms,
            now_ms=now_ms,
        )


def test_runner_checkpoint_terminal_replay_skips_model_and_terminal_notification() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    def first_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="done")

        return ScriptedLLM(steps=[complete])

    agent = Agent(name="checkpoint-agent", instructions="Return the answer.", model="test-model")
    first = Runner.run_sync(
        agent,
        "process item 42",
        run_config=_config(store, key="terminal-replay", provider=_provider(first_model)),
    )

    assert first.status is AgentStatus.COMPLETED
    assert first.raw_result.checkpoint_key == "terminal-replay"
    assert model_calls == 1
    terminal = store.load_checkpoint("terminal-replay")
    assert terminal is not None
    assert terminal.terminal_result is not None
    assert terminal.terminal_acknowledged
    assert terminal.event_cursor is not None
    assert all(entry.state == "delivered" for entry in terminal.event_outbox)

    replay = Runner.run_sync(
        agent,
        "process item 42",
        run_config=_config(
            store,
            key="terminal-replay",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )

    assert replay.status is AgentStatus.COMPLETED
    assert replay.final_output == "done"
    assert model_calls == 1
    assert not any(event.type in {"run_completed", "run_failed"} for event in replay.events)


def test_runner_returns_authoritative_terminal_result_after_finalize_merge() -> None:
    class MergeTerminalObservationStore(InMemoryCheckpointStore):
        def _merge(self, checkpoint_key: str) -> None:
            with self._lock:
                terminal = self._store[checkpoint_key].terminal_result
                assert terminal is not None
                terminal.resume_observations = [
                    ResumeObservation(
                        operation_id="authoritative-operation",
                        operation_kind=OperationKind.TOOL,
                        cycle_index=1,
                        risk="unknown_tool_side_effect",
                        idempotency_support=None,
                    )
                ]

        def finalize_checkpoint(self, checkpoint: Any, *, expected_revision: int) -> bool:
            finalized = super().finalize_checkpoint(checkpoint, expected_revision=expected_revision)
            if finalized:
                self._merge(checkpoint.checkpoint_key)
            return finalized

        def finalize_claimed_checkpoint(
            self,
            checkpoint: Any,
            *,
            claim_token: str,
            expected_revision: int,
        ) -> bool:
            finalized = super().finalize_claimed_checkpoint(
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            )
            if finalized:
                self._merge(checkpoint.checkpoint_key)
            return finalized

    store = MergeTerminalObservationStore()
    result = Runner.run_sync(
        Agent(name="authoritative-terminal-agent", instructions="Answer.", model="test-model"),
        "run",
        run_config=_config(
            store,
            key="authoritative-terminal-result",
            provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="done")])),
        ),
    )

    assert result.raw_result.resume_observations == [
        ResumeObservation(
            operation_id="authoritative-operation",
            operation_kind=OperationKind.TOOL,
            cycle_index=1,
            risk="unknown_tool_side_effect",
            idempotency_support=None,
        )
    ]


def test_checkpoint_control_cancel_becomes_typed_claimed_terminal() -> None:
    store = _CancelAfterProgressStore()
    effects = 0

    @function_tool(name="should_not_run")
    def should_not_run() -> str:
        nonlocal effects
        effects += 1
        return "unexpected"

    agent = Agent(
        name="checkpoint-cancel-agent",
        instructions="Run the operation.",
        model="test-model",
        tools=[should_not_run],
    )
    result = Runner.run_sync(
        agent,
        "run",
        run_config=_config(
            store,
            key="checkpoint-control-cancel",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-cancel", name="should_not_run", arguments={})],
                        )
                    ]
                )
            ),
        ),
    )

    assert effects == 0
    assert result.status is AgentStatus.FAILED
    assert result.raw_result.error == {
        "code": "cancelled_with_unknown_outcome",
        "message": "Cancellation was accepted while the external outcome remained unknown.",
        "retryable": False,
    }
    assert result.raw_result.error_code is None
    checkpoint = store.load_checkpoint("checkpoint-control-cancel")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.tool_journal[0].error is not None
    assert checkpoint.tool_journal[0].error.code == "tool_cancelled"
    assert checkpoint.terminal_result.resume_observations
    aborted = [entry.event for entry in checkpoint.event_outbox if entry.event.get("type") == "cycle_aborted"]
    assert aborted and aborted[0]["logical_cycle"] == aborted[0]["cycle_index"] + 1
    terminal_types = [
        entry.event["type"]
        for entry in checkpoint.event_outbox
        if entry.event["type"] in {"cycle_aborted", "run_state_changed", "run_failed", "run_cancelled"}
    ]
    assert terminal_types[-2:] == ["cycle_aborted", "run_cancelled"]
    assert not any(entry.event["type"] == "run_failed" for entry in checkpoint.event_outbox)
    assert checkpoint.model_call_journal
    assert checkpoint.model_calls


def test_checkpoint_cancel_after_tool_receipt_keeps_definitive_receipt() -> None:
    store = _CancelAtReceiptStore()

    @function_tool(name="cancel_after_receipt")
    def cancel_after_receipt() -> str:
        return "written"

    result = Runner.run_sync(
        Agent(
            name="checkpoint-cancel-receipt-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[cancel_after_receipt],
            tool_use_behavior="stop_on_first_tool",
        ),
        "run",
        run_config=_config(
            store,
            key="checkpoint-cancel-after-receipt",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call-cancel-receipt",
                                    name="cancel_after_receipt",
                                    arguments={},
                                )
                            ],
                        )
                    ]
                )
            ),
        ),
    )

    assert result.status is AgentStatus.FAILED
    assert result.raw_result.completion_reason is CompletionReason.CANCELLED
    checkpoint = store.load_checkpoint("checkpoint-cancel-after-receipt")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert any(entry.event.get("type") == "tool_call_completed" for entry in checkpoint.event_outbox)
    assert checkpoint.model_call_journal
    terminal_types = [event.type for event in result.events if event.type in {"run_failed", "run_cancelled"}]
    assert terminal_types == ["run_cancelled"]


def test_checkpoint_cancel_after_started_tool_closes_ambiguous_claim() -> None:
    store = InMemoryCheckpointStore()
    token = CancellationToken()

    @function_tool(name="cancel_after_started")
    def cancel_after_started(context: ToolContext) -> str:
        assert context.ctx is not None
        assert context.ctx.cancellation_token is not None
        context.ctx.cancellation_token.cancel("cancel after durable start")
        return "external effect may have happened"

    config = _config(
        store,
        key="checkpoint-cancel-after-started",
        provider=_provider(
            lambda: ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-cancel-started", name="cancel_after_started", arguments={})],
                    )
                ]
            )
        ),
    )
    config.cancellation_token = token
    result = Runner.run_sync(
        Agent(
            name="checkpoint-cancel-started-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[cancel_after_started],
        ),
        "run",
        run_config=config,
    )

    assert result.status is AgentStatus.FAILED
    assert result.raw_result.error == {
        "code": "cancelled_with_unknown_outcome",
        "message": "Cancellation was accepted while the external outcome remained unknown.",
        "retryable": False,
    }
    checkpoint = store.load_checkpoint("checkpoint-cancel-after-started")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.claim_token is None
    assert checkpoint.tool_journal
    assert all(entry.state is OperationState.FAILED for entry in checkpoint.tool_journal)
    assert checkpoint.tool_journal[0].error is not None
    assert checkpoint.tool_journal[0].error.code == "tool_cancelled"
    assert checkpoint.terminal_result.resume_observations
    terminal_types = [
        entry.event["type"]
        for entry in checkpoint.event_outbox
        if entry.event["type"] in {"cycle_aborted", "run_state_changed", "run_cancelled"}
    ]
    assert terminal_types[-2:] == ["cycle_aborted", "run_cancelled"]


def test_checkpoint_control_lease_loss_becomes_typed_claimed_terminal() -> None:
    store = _LeaseErrorAfterProgressStore()

    @function_tool(name="lease_control")
    def lease_control() -> str:
        return "unexpected"

    result = Runner.run_sync(
        Agent(
            name="checkpoint-lease-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[lease_control],
        ),
        "run",
        run_config=_config(
            store,
            key="checkpoint-control-lease",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-lease", name="lease_control", arguments={})],
                        )
                    ]
                )
            ),
        ),
    )

    assert result.status is AgentStatus.FAILED
    assert result.raw_result.error == {
        "code": "lease_lost_with_unknown_outcome",
        "message": "Checkpoint lease was lost while the external outcome remained unknown.",
        "retryable": False,
    }
    assert result.raw_result.error_code is None
    checkpoint = store.load_checkpoint("checkpoint-control-lease")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.tool_journal[0].error is not None
    assert checkpoint.tool_journal[0].error.code == "tool_cancelled"


def test_runner_planned_short_circuit_receipt_commits_cycle() -> None:
    store = InMemoryCheckpointStore()
    executed: list[int] = []

    @function_tool(name="typed_short_circuit")
    def typed_short_circuit(value: int) -> str:
        executed.append(value)
        return str(value)

    result = Runner.run_sync(
        Agent(
            name="checkpoint-planned-short-circuit-agent",
            instructions="Return the answer after handling the tool result.",
            model="test-model",
            tools=[typed_short_circuit],
        ),
        "run",
        run_config=_config(
            store,
            key="checkpoint-planned-short-circuit",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call-invalid-arguments",
                                    name="typed_short_circuit",
                                    arguments={"value": "wrong"},
                                )
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                )
            ),
            max_cycles=2,
        ),
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "done"
    assert executed == []
    checkpoint = store.load_checkpoint("checkpoint-planned-short-circuit")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    completed = [event for event in result.events if isinstance(event, ToolCallCompletedEvent)]
    assert completed
    assert all(event.error_code == "invalid_tool_arguments" for event in completed)
    assert all(event.execution_started is False for event in completed)
    assert all(event.duration_ms is None for event in completed)
    assert checkpoint.tool_journal == []


def test_runner_planned_success_short_circuit_does_not_create_receipt() -> None:
    store = InMemoryCheckpointStore()
    executed = 0

    @function_tool(name="short_circuit_success")
    def short_circuit_success() -> str:
        nonlocal executed
        executed += 1
        return "must not execute"

    class SuccessHook(BaseRuntimeHook):
        def before_tool_call(self, event: BeforeToolCallEvent) -> ToolExecutionResult | None:
            if event.call.name != "short_circuit_success":
                return None
            return ToolExecutionResult(
                tool_call_id=event.call.id,
                content="handled before dispatch",
                status_code=ToolResultStatus.SUCCESS,
            )

    result = Runner.run_sync(
        Agent(
            name="checkpoint-planned-success-agent",
            instructions="Return the answer after handling the tool result.",
            model="test-model",
            tools=[short_circuit_success],
        ),
        "run",
        run_config=RunConfig(
            model_provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-short-success", name="short_circuit_success", arguments={})],
                        ),
                        LLMResponse(content="done"),
                    ]
                )
            ),
            max_cycles=2,
            no_tool_policy="finish",
            hooks=[SuccessHook()],
            checkpoint_config=CheckpointConfig(
                key="checkpoint-planned-success",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
                capability_refs={"runtime_hook:0": {"id": "test.success-hook", "version": "1"}},
            ),
        ),
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "done"
    assert executed == 0
    completed = [event for event in result.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].execution_started is False
    checkpoint = store.load_checkpoint("checkpoint-planned-success")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.tool_journal == []


def test_runner_claim_lost_renewal_returns_without_finalizing_and_recovers() -> None:
    store = _ClaimLostOnRenewStore()
    model_calls = 0

    def first_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="must not dispatch")

        return ScriptedLLM(steps=[complete])

    agent = Agent(name="checkpoint-renew-agent", instructions="Return the answer.", model="test-model")
    first = Runner.run_sync(
        agent,
        "run",
        run_config=_config(store, key="checkpoint-renew-claim-lost", provider=_provider(first_model)),
    )

    assert model_calls == 0
    assert first.status is AgentStatus.FAILED
    assert first.raw_result.error == {
        "code": "lease_lost_with_unknown_outcome",
        "message": "Checkpoint lease was lost while the external outcome remained unknown.",
        "retryable": False,
    }
    assert first.raw_result.checkpoint_key == "checkpoint-renew-claim-lost"
    assert store.checkpoint_at_loss is not None
    retained = store.load_checkpoint("checkpoint-renew-claim-lost")
    assert retained is not None
    assert checkpoint_to_dict(retained) == checkpoint_to_dict(store.checkpoint_at_loss)
    assert retained.terminal_result is None
    assert retained.claim_token is not None

    with store._lock:
        store._store["checkpoint-renew-claim-lost"].lease_expires_at_ms = 1
    recovered = Runner.run_sync(
        agent,
        "run",
        run_config=_config(
            store,
            key="checkpoint-renew-claim-lost",
            provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="recovered")])),
        ),
    )

    assert recovered.status is AgentStatus.COMPLETED
    assert recovered.final_output == "recovered"
    terminal = store.load_checkpoint("checkpoint-renew-claim-lost")
    assert terminal is not None and terminal.terminal_result is not None
    assert terminal.terminal_result.final_answer == "recovered"


def test_checkpoint_control_cancel_at_cycle_commit_becomes_typed_terminal() -> None:
    store = _CancelBeforeCommitStore()
    effects = 0

    @function_tool(name="commit_cancel_control")
    def commit_cancel_control() -> str:
        nonlocal effects
        effects += 1
        return "done"

    result = Runner.run_sync(
        Agent(
            name="checkpoint-commit-cancel-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[commit_cancel_control],
        ),
        "run",
        run_config=_config(
            store,
            key="checkpoint-commit-control-cancel",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-commit-cancel", name="commit_cancel_control", arguments={})],
                        )
                    ]
                )
            ),
        ),
    )

    assert effects == 1
    assert result.status is AgentStatus.FAILED
    assert result.raw_result.error == {
        "code": "cancelled_with_unknown_outcome",
        "message": "Cancellation was accepted while the external outcome remained unknown.",
        "retryable": False,
    }
    assert result.raw_result.error_code is None
    checkpoint = store.load_checkpoint("checkpoint-commit-control-cancel")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.claim_token is None


def test_runner_checkpoint_terminal_replay_repeats_typed_output_validation() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    def invalid_typed_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="not-json")

        return ScriptedLLM(steps=[complete])

    agent = Agent(
        name="typed-checkpoint-agent",
        instructions="Return a JSON object.",
        model="test-model",
        output_type=dict,
    )
    config = _config(
        store,
        key="typed-terminal-replay",
        provider=_provider(invalid_typed_model),
    )

    with pytest.raises(ValueError, match="failed to validate final output"):
        Runner.run_sync(agent, "return invalid typed output", run_config=config)
    assert model_calls == 1
    terminal = store.load_checkpoint("typed-terminal-replay")
    assert terminal is not None
    assert terminal.status is AgentStatus.COMPLETED
    assert terminal.terminal_result is not None

    with pytest.raises(ValueError, match="failed to validate final output"):
        Runner.run_sync(agent, "return invalid typed output", run_config=config)
    assert model_calls == 1


def test_runner_checkpoint_terminal_replay_reuses_validated_failure() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0
    validator_calls = 0

    def completed_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="invalid")

        return ScriptedLLM(steps=[complete])

    def validate(_output: Any, _context: OutputValidationContext) -> OutputValidationResult:
        nonlocal validator_calls
        validator_calls += 1
        return OutputValidationResult.reject("invalid")

    agent = Agent(
        name="validated-checkpoint-agent",
        instructions="Return the answer.",
        model="test-model",
        output_validation_enabled=True,
        output_validator=validate,
    )
    config = RunConfig(
        model_provider=_provider(completed_model),
        max_cycles=1,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key="validated-terminal-replay",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            capability_refs={
                "output_validator": {"id": "output.validation", "version": "1"},
            },
        ),
    )

    first = Runner.run_sync(agent, "return invalid output", run_config=config)
    assert first.status is AgentStatus.FAILED
    assert first.error_code == "output_validation_failed"
    assert model_calls == 1
    assert validator_calls == 1
    terminal = store.load_checkpoint("validated-terminal-replay")
    assert terminal is not None
    assert terminal.status is AgentStatus.FAILED
    assert terminal.terminal_result is not None
    assert terminal.terminal_result.error_code == "output_validation_failed"

    replay = Runner.run_sync(agent, "return invalid output", run_config=config)
    assert replay.status is AgentStatus.FAILED
    assert replay.error_code == "output_validation_failed"
    assert model_calls == 1
    assert validator_calls == 1
    assert not any(event.type in {"run_completed", "run_failed"} for event in replay.events)


@pytest.mark.parametrize("idempotency", ["supported", "unsupported", "unknown"])
def test_runner_injects_stable_tool_idempotency_key_and_replay_does_not_repeat_effect(idempotency: str) -> None:
    store = InMemoryCheckpointStore()
    effects: list[tuple[str | None, str]] = []

    @function_tool(name="write_record", tool_metadata={"idempotency": idempotency})
    def write_record(context: ToolContext, value: str) -> str:
        checkpoint = store.load_checkpoint("tool-replay")
        assert checkpoint is not None
        assert checkpoint.tool_journal[0].idempotency_key == context.idempotency_key
        effects.append((context.idempotency_key, value))
        return "written"

    def scripted() -> ScriptedLLM:
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-write-1",
                            name="write_record",
                            arguments={"value": "42"},
                        )
                    ],
                )
            ]
        )

    agent = Agent(
        name="checkpoint-tool-agent",
        instructions="Write the record.",
        model="test-model",
        tools=[write_record],
        tool_use_behavior="stop_on_first_tool",
    )
    config = _config(store, key="tool-replay", provider=_provider(scripted))
    first = Runner.run_sync(agent, "write 42", run_config=config)
    assert first.status is AgentStatus.COMPLETED
    assert len(effects) == 1
    key = effects[0][0]
    if idempotency == "unsupported":
        assert key is None
    else:
        assert key is not None and key.startswith("idem_")

    replay = Runner.run_sync(
        agent,
        "write 42",
        run_config=_config(
            store,
            key="tool-replay",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )
    assert replay.status is AgentStatus.COMPLETED
    assert len(effects) == 1


def test_runner_recovery_exposes_ambiguous_non_idempotent_tool_without_retry() -> None:
    store = InMemoryCheckpointStore()
    effects = 0

    @function_tool(name="unsafe_write", tool_metadata={"idempotency": "unknown"})
    def unsafe_write(value: str) -> str:
        nonlocal effects
        effects += 1
        raise SystemExit("simulated process crash after side effect")

    def scripted() -> ScriptedLLM:
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-unsafe-1",
                            name="unsafe_write",
                            arguments={"value": "42"},
                        )
                    ],
                )
            ]
        )

    agent = Agent(
        name="checkpoint-crash-agent",
        instructions="Write once.",
        model="test-model",
        tools=[unsafe_write],
    )
    with pytest.raises(SystemExit, match="simulated process crash"):
        Runner.run_sync(
            agent,
            "write 42",
            run_config=_config(
                store,
                key="ambiguous-tool",
                provider=_provider(scripted),
                ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
            ),
        )
    assert effects == 1

    with store._lock:
        crashed = store._store["ambiguous-tool"]
        assert crashed.tool_journal[0].state.value == "started"
        crashed.lease_expires_at_ms = 1

    resumed = Runner.run_sync(
        agent,
        "write 42",
        run_config=_config(
            store,
            key="ambiguous-tool",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
            ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
        ),
    )

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.completion_reason is None
    assert resumed.raw_result.resume_observations
    assert resumed.raw_result.resume_observations[0].risk == "unknown_tool_side_effect"
    assert effects == 1
    retained = store.load_checkpoint("ambiguous-tool")
    assert retained is not None
    assert retained.status is AgentStatus.RECONCILIATION_REQUIRED
    assert retained.tool_journal[0].state.value == "ambiguous"
    assert retained.claim_token is None

    resumed_again = Runner.run_sync(
        agent,
        "write 42",
        run_config=_config(
            store,
            key="ambiguous-tool",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
            ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
        ),
    )

    assert resumed_again.status is AgentStatus.RECONCILIATION_REQUIRED
    assert effects == 1
    retained_again = store.load_checkpoint("ambiguous-tool")
    assert retained_again is not None
    assert [entry.event["type"] for entry in retained_again.event_outbox].count("operation_ambiguous") == 1
    assert [entry.event["type"] for entry in retained_again.event_outbox].count("reconciliation_required") == 1


def test_checkpoint_defaults_retry_model_and_surface_unknown_tool_outcome() -> None:
    config = CheckpointConfig(store=InMemoryCheckpointStore())

    assert config.ambiguous_model_policy is AmbiguousModelPolicy.RETRY_WITH_DUPLICATE_RISK
    assert config.ambiguous_tool_policy is AmbiguousToolPolicy.SURFACE_TO_MODEL


def test_runner_default_surface_to_model_closes_unknown_tool_once() -> None:
    store = InMemoryCheckpointStore()
    provider_runs = 0
    effects = 0

    @function_tool(name="unsafe_surface_write", tool_metadata={"idempotency": "unknown"})
    def unsafe_surface_write() -> str:
        nonlocal effects
        effects += 1
        raise SystemExit("crash after unknown side effect")

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-surface", name="unsafe_surface_write", arguments={})],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="continued after unknown outcome")])

    agent = Agent(
        name="surface-unknown-agent",
        instructions="Run the operation and continue after an unknown outcome.",
        model="test-model",
        tools=[unsafe_surface_write],
    )
    with pytest.raises(SystemExit, match="unknown side effect"):
        Runner.run_sync(
            agent,
            "run once",
            run_config=_config(
                store,
                key="surface-unknown",
                provider=_provider(model_factory),
                max_cycles=2,
            ),
        )
    assert effects == 1

    with store._lock:
        store._store["surface-unknown"].lease_expires_at_ms = 1

    resumed = Runner.run_sync(
        agent,
        "run once",
        run_config=_config(
            store,
            key="surface-unknown",
            provider=_provider(model_factory),
            max_cycles=2,
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "continued after unknown outcome"
    assert effects == 1
    unknown_results = [
        result
        for cycle in resumed.raw_result.cycles
        for result in cycle.tool_results
        if result.error_code == "tool_outcome_unknown"
    ]
    assert len(unknown_results) == 1
    completed = [event for event in resumed.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].error_code == "tool_outcome_unknown"


def test_runner_reconciled_tool_receipt_owns_one_completed_event() -> None:
    store = InMemoryCheckpointStore()
    provider_runs = 0
    effects = 0

    @function_tool(name="reconciled_write", tool_metadata={"idempotency": "unknown"})
    def reconciled_write() -> str:
        nonlocal effects
        effects += 1
        raise SystemExit("crash before durable receipt")

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        return ScriptedLLM(
            steps=(
                [
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-reconciled", name="reconciled_write", arguments={})],
                    )
                ]
                if provider_runs == 1
                else [LLMResponse(content="done")]
            )
        )

    agent = Agent(
        name="reconciled-receipt-agent",
        instructions="Run the operation.",
        model="test-model",
        tools=[reconciled_write],
    )
    with pytest.raises(SystemExit, match="durable receipt"):
        Runner.run_sync(
            agent,
            "run once",
            run_config=_config(
                store,
                key="reconciled-receipt",
                provider=_provider(model_factory),
                max_cycles=2,
                capability_refs={"reconciliation_provider": {"id": "test.reconciler", "version": "1"}},
            ),
        )
    with store._lock:
        store._store["reconciled-receipt"].lease_expires_at_ms = 1

    class Provider:
        def reconcile(self, observation: Any) -> ReconciliationDecision:
            del observation
            return ReconciliationDecision(
                ReconciliationDecisionKind.REPLAY_SUCCESS,
                result=ToolExecutionResult(
                    tool_call_id="call-reconciled",
                    content="already written",
                    status_code=ToolResultStatus.SUCCESS,
                ).to_dict(),
            )

    resumed = Runner.run_sync(
        agent,
        "run once",
        run_config=RunConfig(
            model_provider=_provider(model_factory),
            max_cycles=2,
            no_tool_policy="finish",
            checkpoint_config=CheckpointConfig(
                key="reconciled-receipt",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
                capability_refs={"reconciliation_provider": {"id": "test.reconciler", "version": "1"}},
            ),
            reconciliation_provider=Provider(),
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert effects == 1
    completed = [event for event in resumed.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].execution_started is True


def test_runner_mixed_reconciliation_decisions_rebind_authoritative_entries() -> None:
    store = InMemoryCheckpointStore()
    provider_runs = 0

    @function_tool(name="mixed_reconcile_write", tool_metadata={"idempotency": "unknown"})
    def mixed_reconcile_write() -> str:
        raise SystemExit("crash before reconciliation")

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-mixed-1", name="mixed_reconcile_write", arguments={})],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="done")])

    agent = Agent(
        name="mixed-reconciliation-agent",
        instructions="Run the operation.",
        model="test-model",
        tools=[mixed_reconcile_write],
    )
    initial_config = _config(
        store,
        key="mixed-reconciliation",
        provider=_provider(model_factory),
        max_cycles=2,
        ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
        capability_refs={"reconciliation_provider": {"id": "test.mixed-reconciler", "version": "1"}},
    )
    with pytest.raises(SystemExit, match="reconciliation"):
        Runner.run_sync(agent, "run once", run_config=initial_config)

    with store._lock:
        crashed = store._store["mixed-reconciliation"]
        first = crashed.tool_journal[0]
        second = deepcopy(first)
        second.operation_id = f"{first.operation_id}-second"
        second.tool_call_id = "call-mixed-2"
        second.request_digest = "b" * 64
        second.identity_key = None
        second.result_digest = None
        second.result = None
        second.error = None
        second.resume_observation = None
        crashed.tool_journal.append(second)
        crashed.lease_expires_at_ms = 1
        first_operation_id = first.operation_id

    class Provider:
        def reconcile(self, observation: Any) -> ReconciliationDecision:
            if observation.operation_id == first_operation_id:
                return ReconciliationDecision(
                    ReconciliationDecisionKind.RECORD_FAILURE,
                    error=ReconciliationError(
                        code="tool_outcome_unknown",
                        message="The first outcome remains unknown.",
                    ),
                )
            return ReconciliationDecision(
                ReconciliationDecisionKind.REPLAY_SUCCESS,
                result=ToolExecutionResult(
                    tool_call_id="call-mixed-2",
                    content="already written",
                    status_code=ToolResultStatus.SUCCESS,
                ).to_dict(),
            )

    resumed = Runner.run_sync(
        agent,
        "run once",
        run_config=RunConfig(
            model_provider=_provider(model_factory),
            max_cycles=2,
            no_tool_policy="finish",
            checkpoint_config=CheckpointConfig(
                key="mixed-reconciliation",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
                store=store,
                capability_refs={"reconciliation_provider": {"id": "test.mixed-reconciler", "version": "1"}},
            ),
            reconciliation_provider=Provider(),
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    checkpoint = store.load_checkpoint("mixed-reconciliation")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert checkpoint.tool_journal == []
    completed = [
        event for event in resumed.events if isinstance(event, ToolCallCompletedEvent) and event.operation_id is not None
    ]
    assert {event.operation_id for event in completed} == {first_operation_id, f"{first_operation_id}-second"}
    by_operation = {event.operation_id: event for event in completed}
    assert by_operation[first_operation_id].error_code == "tool_outcome_unknown"
    assert by_operation[f"{first_operation_id}-second"].status == "success"


def test_runner_started_success_owns_one_completed_event() -> None:
    store = InMemoryCheckpointStore()

    @function_tool(name="durable_write")
    def durable_write() -> str:
        return "written"

    result = Runner.run_sync(
        Agent(
            name="durable-receipt-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[durable_write],
            tool_use_behavior="stop_on_first_tool",
        ),
        "run",
        run_config=_config(
            store,
            key="durable-receipt",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-durable", name="durable_write", arguments={})],
                        )
                    ]
                )
            ),
        ),
    )

    assert result.status is AgentStatus.COMPLETED
    completed = [event for event in result.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].execution_started is True


@pytest.mark.parametrize(
    "status",
    [ToolResultStatus.RUNNING, ToolResultStatus.PENDING_COMPRESS, ToolResultStatus.WAIT_RESPONSE],
)
def test_checkpointed_nondefinitive_tool_result_suspends_ambiguous(status: ToolResultStatus) -> None:
    store = InMemoryCheckpointStore()

    @function_tool(name="nondefinitive_tool")
    def nondefinitive_tool() -> ToolExecutionResult:
        return ToolExecutionResult(tool_call_id="", content="not final", status_code=status)

    result = Runner.run_sync(
        Agent(
            name="nondefinitive-checkpoint-agent",
            instructions="Run the operation.",
            model="test-model",
            tools=[nondefinitive_tool],
        ),
        "run",
        run_config=_config(
            store,
            key=f"nondefinitive-{status.value}",
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="call-nondefinitive", name="nondefinitive_tool", arguments={})],
                        )
                    ]
                )
            ),
            max_cycles=2,
        ),
    )

    assert result.status is AgentStatus.RECONCILIATION_REQUIRED
    checkpoint = store.load_checkpoint(f"nondefinitive-{status.value}")
    assert checkpoint is not None
    assert checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED
    assert checkpoint.claim_token is None
    assert checkpoint.tool_journal[0].state is OperationState.AMBIGUOUS
    assert checkpoint.tool_journal[0].resume_observation is None
    assert not any(entry.event.get("type") == "tool_call_completed" for entry in checkpoint.event_outbox)


def test_runner_resume_restores_frozen_metadata_when_system_metadata_is_empty() -> None:
    store = InMemoryCheckpointStore()
    metadata_snapshots: list[dict[str, Any]] = []

    @function_tool(name="retryable_metadata_write", tool_metadata={"idempotency": "supported"})
    def retryable_metadata_write(context: ToolContext) -> str:
        metadata_snapshots.append(dict(context.task_metadata))
        if len(metadata_snapshots) == 1:
            raise SystemExit("crash after retryable metadata write")
        return "written"

    def scripted() -> ScriptedLLM:
        return ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-metadata-1",
                            name="retryable_metadata_write",
                            arguments={},
                        )
                    ],
                )
            ]
        )

    agent = Agent(
        name="checkpoint-metadata-agent",
        instructions="Write once.",
        model="test-model",
        tools=[retryable_metadata_write],
        tool_use_behavior="stop_on_first_tool",
    )
    config = RunConfig(
        model_provider=_provider(scripted),
        max_cycles=1,
        metadata={
            "reserved_output_tokens": 4_096,
            "host_request_id": "request-42",
        },
        checkpoint_config=CheckpointConfig(
            key="frozen-run-metadata",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            ambiguous_tool_policy=AmbiguousToolPolicy.RETRY_IDEMPOTENT_ONLY,
            store=store,
            capability_refs={
                "behavior_affecting_run_metadata": {
                    "id": "metadata.request-42",
                    "version": "1",
                },
            },
        ),
    )

    with pytest.raises(SystemExit, match="crash after retryable metadata write"):
        Runner.run_sync(agent, "write with frozen metadata", run_config=config)

    with store._lock:
        crashed = store._store["frozen-run-metadata"]
        assert crashed.messages[0].role == "system"
        crashed.messages[0].metadata = {}
        crashed.lease_expires_at_ms = 1

    config.metadata = {
        "reserved_output_tokens": 1_024,
        "host_request_id": "stale-request",
    }
    resumed = Runner.run_sync(agent, "write with frozen metadata", run_config=config)

    assert resumed.status is AgentStatus.COMPLETED
    assert len(metadata_snapshots) == 2
    assert metadata_snapshots[1]["reserved_output_tokens"] == 4_096
    assert metadata_snapshots[1]["host_request_id"] == "request-42"


def test_runner_resume_freezes_prompt_session_and_identity_without_reinvoking_callbacks() -> None:
    store = InMemoryCheckpointStore()
    session = MemorySession("checkpoint-session")
    session.add_items([Message(role="user", content="history before run")])
    calls = {"instructions": 0, "context": 0, "guardrail": 0, "effects": 0}

    def instructions(_context: Any, _agent: Agent) -> str:
        calls["instructions"] += 1
        return "Use the frozen context."

    class Provider:
        def fragments(self, request: Any) -> list[ContextFragment]:
            del request
            calls["context"] += 1
            return [ContextFragment(id="tenant", text="Tenant context v1")]

    def guardrail(_context: Any, _input: str) -> GuardrailResult:
        calls["guardrail"] += 1
        return GuardrailResult.allow()

    @function_tool(name="unsafe_frozen_write", tool_metadata={"idempotency": "unknown"})
    def unsafe_frozen_write() -> str:
        calls["effects"] += 1
        raise SystemExit("crash after frozen side effect")

    agent = Agent(
        name="frozen-checkpoint-agent",
        instructions=instructions,
        model="test-model",
        tools=[unsafe_frozen_write],
        input_guardrails=[guardrail],
    )
    provider = Provider()
    config = RunConfig(
        model_provider=_provider(
            lambda: ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-frozen-1",
                                name="unsafe_frozen_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        ),
        max_cycles=2,
        session=session,
        context_providers=[provider],
        checkpoint_config=CheckpointConfig(
            key="frozen-resume",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
            capability_refs={
                "agent.instructions": {"id": "instructions.frozen", "version": "1"},
                "context_provider:0": {"id": "context.frozen", "version": "1"},
                "input_guardrail:0": {"id": "guardrail.frozen", "version": "1"},
                "session": {"id": "session.frozen", "version": "1"},
            },
        ),
    )

    with pytest.raises(SystemExit, match="crash after frozen side effect"):
        Runner.run_sync(agent, "write with frozen inputs", run_config=config)
    assert calls == {"instructions": 1, "context": 1, "guardrail": 1, "effects": 1}
    crashed = store.load_checkpoint("frozen-resume")
    assert crashed is not None
    with store._lock:
        store._store["frozen-resume"].lease_expires_at_ms = 1

    session.add_items([Message(role="user", content="history added after crash")])
    resumed = Runner.run_sync(agent, "write with frozen inputs", run_config=config)

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.run_id == crashed.root_run_id
    assert resumed.trace_id == crashed.trace_id
    assert calls == {"instructions": 1, "context": 1, "guardrail": 1, "effects": 1}


def test_runner_resume_rejects_changed_static_instructions_before_external_work() -> None:
    store = InMemoryCheckpointStore()
    first = Runner.run_sync(
        Agent(name="definition-agent", instructions="Version one.", model="test-model"),
        "answer once",
        run_config=_config(
            store,
            key="definition-change",
            provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="done")])),
        ),
    )
    assert first.status is AgentStatus.COMPLETED

    with pytest.raises(Exception) as captured:
        Runner.run_sync(
            Agent(name="definition-agent", instructions="Version two.", model="test-model"),
            "answer once",
            run_config=_config(
                store,
                key="definition-change",
                provider=_provider(lambda: ScriptedLLM(steps=[])),
            ),
        )

    assert getattr(captured.value, "code", None) == "checkpoint_definition_mismatch"


def test_approval_resume_uses_distinct_checkpoint_and_replays_same_tool_identity() -> None:
    store = InMemoryCheckpointStore()
    effects: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_write(context: ToolContext) -> str:
        assert context.idempotency_key is not None
        effects.append(context.idempotency_key)
        return "written"

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-approved-1",
                                name="approved_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="done after approval")])

    agent = Agent(
        name="approval-checkpoint-agent",
        instructions="Write only after approval.",
        model="test-model",
        tools=[approved_write],
    )
    source_config = _config(
        store,
        key="approval-source",
        provider=_provider(model_factory),
        max_cycles=1,
    )
    source = Runner.run_sync(agent, "perform approved write", run_config=source_config)
    assert source.status is AgentStatus.WAIT_USER
    source_checkpoint = store.load_checkpoint("approval-source")
    assert source_checkpoint is not None
    assert source_checkpoint.terminal_result is not None
    assert source_checkpoint.tool_journal == []

    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            )
        )
    )
    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "done after approval"
    assert len(effects) == 1
    target = store.load_checkpoint("approval-target")
    assert target is not None
    assert target.terminal_result is not None
    assert target.checkpoint_key != source_checkpoint.checkpoint_key

    replay = configured.resume(state)
    assert replay.status is AgentStatus.COMPLETED
    assert replay.run_id == resumed.run_id
    assert effects == [effects[0]]

    with pytest.raises(RuntimeError, match="approval_already_consumed"):
        Runner.configured(
            RunConfig(
                checkpoint_config=CheckpointConfig(
                    key="approval-other-target",
                    resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                    store=store,
                )
            )
        ).resume(state)


def test_approval_resume_emits_planned_and_completed_without_started_for_durable_tool_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCheckpointStore()
    effects: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_replay_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_replay_write(context: ToolContext) -> str:
        assert context.idempotency_key is not None
        effects.append(context.idempotency_key)
        return "written once"

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-approved-replay",
                                name="approved_replay_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="done after replay")])

    agent = Agent(
        name="approval-replay-agent",
        instructions="Write only after approval.",
        model="test-model",
        tools=[approved_replay_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-replay-source",
            provider=_provider(model_factory),
            max_cycles=1,
        ),
    )
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-replay-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            )
        )
    )
    original_execute = Runner._execute_checkpoint_approved_tool
    crash_once = True

    def crash_after_durable_result(*args: Any, **kwargs: Any) -> Any:
        nonlocal crash_once
        outcome = original_execute(*args, **kwargs)
        if crash_once:
            crash_once = False
            raise SystemExit("crash after durable tool result")
        return outcome

    monkeypatch.setattr(Runner, "_execute_checkpoint_approved_tool", staticmethod(crash_after_durable_result))

    with pytest.raises(SystemExit, match="durable tool result"):
        configured.resume(state)
    assert len(effects) == 1
    crashed = store.load_checkpoint("approval-replay-target")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.SUCCEEDED
    with store._lock:
        store._store["approval-replay-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "done after replay"
    assert len(effects) == 1
    replay_lifecycle = [
        event
        for event in resumed.events
        if getattr(event, "tool_call_id", None) == "call-approved-replay" and event.type.startswith("tool_call_")
    ]
    assert [event.type for event in replay_lifecycle] == [
        "tool_call_completed",
    ]
    completed = replay_lifecycle[-1]
    assert isinstance(completed, ToolCallCompletedEvent)
    assert completed.execution_started is True
    assert completed.duration_ms is None


def test_approval_resume_retains_journal_proven_failed_tool_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCheckpointStore()
    invocations: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_failed_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_failed_write(context: ToolContext) -> ToolOutputError:
        assert context.idempotency_key is not None
        invocations.append(context.idempotency_key)
        return ToolOutputError(
            message="permanent failure",
            error_code="permanent_error",
        )

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="approved-failed-call",
                                name="approved_failed_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        if provider_runs == 2:
            return ScriptedLLM(steps=[])

        def finish_after_failed_tool(request: Any) -> LLMResponse:
            assert any(
                message.role == "tool"
                and message.tool_call_id == "approved-failed-call"
                and message.content
                == '{"ok": false, "error": "permanent failure", "error_code": "permanent_error", "retryable": false}'
                for message in request.messages
            )
            return LLMResponse(content="finished after failed approval")

        return ScriptedLLM(steps=[finish_after_failed_tool])

    agent = Agent(
        name="approval-failed-replay-agent",
        instructions="Run the approved write and finish.",
        model="test-model",
        tools=[approved_failed_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-failed-source",
            provider=_provider(model_factory),
            max_cycles=1,
        ),
    )
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-failed-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            )
        )
    )
    original_execute = Runner._execute_checkpoint_approved_tool
    crash_once = True

    def crash_after_failed_receipt(*args: Any, **kwargs: Any) -> Any:
        nonlocal crash_once
        outcome = original_execute(*args, **kwargs)
        if crash_once:
            crash_once = False
            raise SystemExit("crash after failed tool receipt")
        return outcome

    monkeypatch.setattr(Runner, "_execute_checkpoint_approved_tool", staticmethod(crash_after_failed_receipt))

    with pytest.raises(SystemExit, match="failed tool receipt"):
        configured.resume(state)
    assert len(invocations) == 1
    crashed = store.load_checkpoint("approval-failed-target")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.FAILED
    assert crashed.tool_journal[0].result is not None
    assert ToolExecutionResult.from_dict(crashed.tool_journal[0].result).to_dict() == crashed.tool_journal[0].result
    assert crashed.tool_journal[0].result_digest is not None
    with store._lock:
        store._store["approval-failed-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "finished after failed approval"
    assert len(invocations) == 1


def test_approval_resume_failed_tool_session_commit_replays_identical_payload_after_crash() -> None:
    store = InMemoryCheckpointStore()
    invocations: list[str] = []
    provider_runs = 0

    class CrashAfterCommitSession(MemorySession):
        def __init__(self) -> None:
            super().__init__("approval-failed-session-crash")
            self.crash_next_commit = False
            self.commit_calls: list[tuple[str, str, list[Message], str]] = []

        def add_items_once(
            self,
            commit_id: str,
            payload_digest: str,
            items: list[Message],
        ) -> str:
            outcome = super().add_items_once(commit_id, payload_digest, items)
            self.commit_calls.append((commit_id, payload_digest, deepcopy(items), outcome))
            if self.crash_next_commit:
                self.crash_next_commit = False
                raise SystemExit("crash after failed approval session commit")
            return outcome

    session = CrashAfterCommitSession()
    session_ref = {"id": "session.approval-failed-crash", "version": "1"}

    @function_tool(
        name="approved_failed_session_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_failed_session_write(context: ToolContext) -> ToolOutputError:
        assert context.idempotency_key is not None
        invocations.append(context.idempotency_key)
        return ToolOutputError(message="permanent failure", error_code="permanent_error")

    failed_content = '{"ok": false, "error": "permanent failure", "error_code": "permanent_error", "retryable": false}'

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="approved-failed-session-call",
                                name="approved_failed_session_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )

        def finish_after_failed_tool(request: Any) -> LLMResponse:
            assert any(
                message.role == "tool"
                and message.tool_call_id == "approved-failed-session-call"
                and message.content == failed_content
                for message in request.messages
            )
            return LLMResponse(content="finished after failed approval")

        return ScriptedLLM(steps=[finish_after_failed_tool])

    agent = Agent(
        name="approval-failed-session-crash-agent",
        instructions="Run the approved write and finish.",
        model="test-model",
        tools=[approved_failed_session_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-failed-session-source",
            provider=_provider(model_factory),
            max_cycles=1,
            session=session,
            capability_refs={"session": session_ref},
        ),
    )
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    session.crash_next_commit = True
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-failed-session-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
                capability_refs={"session": session_ref},
            )
        )
    )

    with pytest.raises(SystemExit, match="failed approval session commit"):
        configured.resume(state)
    assert len(invocations) == 1
    assert len(session.commit_calls) == 2
    target_commit = session.commit_calls[-1]
    assert any(
        item.tool_call_id == "approved-failed-session-call" and item.content == failed_content for item in target_commit[2]
    )
    crashed = store.load_checkpoint("approval-failed-session-target")
    assert crashed is not None and crashed.terminal_result is None
    assert crashed.tool_journal[0].state is OperationState.FAILED
    assert crashed.tool_journal[0].result_digest is not None
    with store._lock:
        store._store["approval-failed-session-target"].lease_expires_at_ms = 1

    committed_items = session.get_items()
    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "finished after failed approval"
    assert len(invocations) == 1
    assert session.get_items() == committed_items
    target_commits = [entry for entry in session.commit_calls if entry[0] == target_commit[0]]
    assert [entry[3] for entry in target_commits] == ["committed", "replayed"]
    assert target_commits[0][1:3] == target_commits[1][1:3]
    retained = store.load_checkpoint("approval-failed-session-target")
    assert retained is not None and retained.terminal_result is not None
    session_events = [entry for entry in retained.event_outbox if entry.event.get("type") == "session_persisted"]
    assert len(session_events) == 1
    assert session_events[0].state == "delivered"


def test_ptl_retry_uses_distinct_model_operation_slots() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    def prompt_too_long(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        raise ValueError("maximum context length exceeded")

    def crash_after_second_dispatch(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        raise SystemExit("provider response was lost")

    result = Runner.run_sync(
        Agent(
            name="ptl-checkpoint-agent",
            instructions="Retry after compacting the prompt.",
            model="test-model",
        ),
        "process a long prompt",
        run_config=_config(
            store,
            key="ptl-operation-slots",
            provider=_provider(lambda: ScriptedLLM(steps=[prompt_too_long, crash_after_second_dispatch])),
            max_cycles=2,
        ),
    )

    assert result.status is AgentStatus.RECONCILIATION_REQUIRED
    assert model_calls == 2
    checkpoint = store.load_checkpoint("ptl-operation-slots")
    assert checkpoint is not None
    assert [entry.state for entry in checkpoint.model_call_journal] == [
        OperationState.FAILED,
        OperationState.AMBIGUOUS,
    ]
    assert len({entry.operation_id for entry in checkpoint.model_call_journal}) == 2


def test_session_commit_crash_replays_model_receipt_without_duplicate_append() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    class CrashAfterCommitSession:
        def __init__(self) -> None:
            self.session_id = "crash-session"
            self.inner = MemorySession(self.session_id)
            self.crash_next_commit = True

        def get_items(self, limit: int | None = None) -> list[Message]:
            return self.inner.get_items(limit)

        def add_items(self, items: list[Message]) -> None:
            self.inner.add_items(items)

        def add_items_once(
            self,
            commit_id: str,
            payload_digest: str,
            items: list[Message],
        ) -> str:
            outcome = self.inner.add_items_once(commit_id, payload_digest, items)
            if self.crash_next_commit:
                self.crash_next_commit = False
                raise SystemExit("crash after durable session commit")
            return outcome

    session = CrashAfterCommitSession()

    def completed_model() -> ScriptedLLM:
        def complete(_request: Any) -> LLMResponse:
            nonlocal model_calls
            model_calls += 1
            return LLMResponse(content="durable answer")

        return ScriptedLLM(steps=[complete])

    agent = Agent(
        name="session-crash-agent",
        instructions="Return one answer.",
        model="test-model",
    )
    config = RunConfig(
        model_provider=_provider(completed_model),
        max_cycles=1,
        no_tool_policy="finish",
        session=session,
        checkpoint_config=CheckpointConfig(
            key="session-commit-crash",
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            capability_refs={
                "session": {"id": "session.crash", "version": "1"},
            },
        ),
    )

    with pytest.raises(SystemExit, match="durable session commit"):
        Runner.run_sync(agent, "answer once", run_config=config)
    assert model_calls == 1
    assert [item.content for item in session.get_items()] == [
        "answer once",
        "durable answer",
    ]
    with store._lock:
        store._store["session-commit-crash"].lease_expires_at_ms = 1

    resumed = Runner.run_sync(
        agent,
        "answer once",
        run_config=RunConfig(
            model_provider=_provider(lambda: ScriptedLLM(steps=[])),
            max_cycles=1,
            no_tool_policy="finish",
            session=session,
            checkpoint_config=config.checkpoint_config,
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "durable answer"
    assert model_calls == 1
    assert [item.content for item in session.get_items()] == [
        "answer once",
        "durable answer",
    ]
    retained = store.load_checkpoint("session-commit-crash")
    assert retained is not None
    assert [record.operation_id for record in retained.model_calls] == ["op_model_cycle_1_main"]


def test_claimed_session_event_is_durable_before_terminal_finalize_crash() -> None:
    store = _CrashBeforeTerminalFinalizeStore()
    session = MemorySession("claimed-session")
    agent = Agent(name="claimed-session-agent", instructions="Return one answer.", model="test-model")
    config = _config(
        store,
        key="claimed-session",
        provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="durable answer")])),
        session=session,
        capability_refs={"session": {"id": "session.claimed", "version": "1"}},
    )

    with pytest.raises(SystemExit, match="crash before terminal finalize"):
        Runner.run_sync(agent, "answer once", run_config=config)

    crashed = store.load_checkpoint("claimed-session")
    assert crashed is not None and crashed.terminal_result is None and crashed.claim_token is not None
    session_events = [entry for entry in crashed.event_outbox if entry.event.get("type") == "session_persisted"]
    assert len(session_events) == 1
    assert session_events[0].state == "delivered"


def test_checkpoint_control_session_commit_is_replayed_without_duplicate_append() -> None:
    store = _CancelBeforeCommitStore()
    session = MemorySession("control-session")
    effects = 0

    @function_tool(name="control_session_tool")
    def control_session_tool() -> str:
        nonlocal effects
        effects += 1
        return "done"

    agent = Agent(
        name="control-session-agent",
        instructions="Run the operation.",
        model="test-model",
        tools=[control_session_tool],
    )
    config = _config(
        store,
        key="control-session",
        provider=_provider(
            lambda: ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="control-call", name="control_session_tool", arguments={})],
                    )
                ]
            )
        ),
        session=session,
        capability_refs={"session": {"id": "session.control", "version": "1"}},
    )

    first = Runner.run_sync(agent, "run once", run_config=config)
    assert first.status is AgentStatus.FAILED
    assert effects == 1
    committed_items = session.get_items()
    checkpoint = store.load_checkpoint("control-session")
    assert checkpoint is not None and checkpoint.terminal_result is not None
    assert [entry.event["type"] for entry in checkpoint.event_outbox if entry.event.get("type") == "session_persisted"] == [
        "session_persisted"
    ]

    replay = Runner.run_sync(
        agent,
        "run once",
        run_config=_config(
            store,
            key="control-session",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
            session=session,
            capability_refs={"session": {"id": "session.control", "version": "1"}},
        ),
    )

    assert replay.status is AgentStatus.FAILED
    assert effects == 1
    assert session.get_items() == committed_items
    retained = store.load_checkpoint("control-session")
    assert retained is not None
    assert len([entry for entry in retained.event_outbox if entry.event.get("type") == "session_persisted"]) == 1


def test_max_cycles_session_event_is_replayed_after_unclaimed_terminal_finalize_crash() -> None:
    store = _CrashBeforeTerminalFinalizeStore()
    session = MemorySession("max-cycles-session")
    agent = Agent(name="max-cycles-session-agent", instructions="Continue until the cycle limit.", model="test-model")
    first_config = _config(
        store,
        key="max-cycles-session",
        provider=_provider(lambda: ScriptedLLM(steps=[LLMResponse(content="partial answer")])),
        no_tool_policy="continue",
        session=session,
        capability_refs={"session": {"id": "session.max-cycles", "version": "1"}},
    )

    with pytest.raises(SystemExit, match="crash before terminal finalize"):
        Runner.run_sync(agent, "continue once", run_config=first_config)
    committed_items = session.get_items()
    crashed = store.load_checkpoint("max-cycles-session")
    assert crashed is not None and (crashed.cycle_index, crashed.claim_token, crashed.terminal_result) == (1, None, None)
    assert all(event.event.get("type") != "session_persisted" for event in crashed.event_outbox)

    resumed = Runner.run_sync(
        agent,
        "continue once",
        run_config=_config(
            store,
            key="max-cycles-session",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
            no_tool_policy="continue",
            session=session,
            capability_refs={"session": {"id": "session.max-cycles", "version": "1"}},
        ),
    )

    assert resumed.status is AgentStatus.MAX_CYCLES
    assert session.get_items() == committed_items
    assert [event.type for event in resumed.events].count("session_persisted") == 1
    retained = store.load_checkpoint("max-cycles-session")
    assert retained is not None and retained.terminal_result is not None
    assert retained.terminal_result.status is AgentStatus.MAX_CYCLES
    assert all(event.state == "delivered" for event in retained.event_outbox)


def test_approval_resume_crash_retries_same_idempotency_key_once() -> None:
    store = InMemoryCheckpointStore()
    effects: set[str] = set()
    invocations: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_idempotent_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_idempotent_write(context: ToolContext) -> str:
        assert context.idempotency_key is not None
        key = context.idempotency_key
        invocations.append(key)
        if key not in effects:
            effects.add(key)
            raise SystemExit("crash after idempotent side effect")
        return "already written"

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-approved-crash",
                                name="approved_idempotent_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="done after recovery")])

    agent = Agent(
        name="approval-crash-agent",
        instructions="Write only after approval.",
        model="test-model",
        tools=[approved_idempotent_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-crash-source",
            provider=_provider(model_factory),
            max_cycles=1,
        ),
    )
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-crash-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                ambiguous_tool_policy=AmbiguousToolPolicy.RETRY_IDEMPOTENT_ONLY,
                store=store,
            )
        )
    )

    with pytest.raises(SystemExit, match="idempotent side effect"):
        configured.resume(state)
    assert len(effects) == 1
    crashed = store.load_checkpoint("approval-crash-target")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.STARTED
    with store._lock:
        store._store["approval-crash-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "done after recovery"
    assert len(effects) == 1
    assert len(invocations) == 2
    assert invocations[0] == invocations[1]


def test_approval_resume_session_commit_replays_identical_payload_after_crash() -> None:
    store = InMemoryCheckpointStore()
    effects: list[str] = []
    provider_runs = 0

    class CrashAfterCommitSession(MemorySession):
        def __init__(self) -> None:
            super().__init__("approval-session-crash")
            self.crash_next_commit = False
            self.commit_calls: list[tuple[str, str, list[Message], str]] = []

        def add_items_once(
            self,
            commit_id: str,
            payload_digest: str,
            items: list[Message],
        ) -> str:
            outcome = super().add_items_once(commit_id, payload_digest, items)
            self.commit_calls.append((commit_id, payload_digest, deepcopy(items), outcome))
            if self.crash_next_commit:
                self.crash_next_commit = False
                raise SystemExit("crash after approval session commit")
            return outcome

    session = CrashAfterCommitSession()
    session_ref = {"id": "session.approval-crash", "version": "1"}

    @function_tool(
        name="approved_session_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_session_write(context: ToolContext) -> str:
        assert context.idempotency_key is not None
        effects.append(context.idempotency_key)
        return "written"

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="approved-session-call",
                                name="approved_session_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )

        def finish_after_approved(request: Any) -> LLMResponse:
            assert any(
                message.role == "tool" and message.tool_call_id == "approved-session-call" and message.content == "written"
                for message in request.messages
            )
            return LLMResponse(content="finished after approval")

        return ScriptedLLM(steps=[finish_after_approved])

    agent = Agent(
        name="approval-session-crash-agent",
        instructions="Write only after approval, then finish.",
        model="test-model",
        tools=[approved_session_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-session-source",
            provider=_provider(model_factory),
            max_cycles=1,
            session=session,
            capability_refs={"session": session_ref},
        ),
    )
    assert source.status is AgentStatus.WAIT_USER
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    session.crash_next_commit = True
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-session-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
                capability_refs={"session": session_ref},
            )
        )
    )

    with pytest.raises(SystemExit, match="approval session commit"):
        configured.resume(state)
    assert len(effects) == 1
    committed_items = session.get_items()
    assert len(session.commit_calls) == 2
    target_commit = session.commit_calls[-1]
    assert any(item.tool_call_id == "approved-session-call" for item in target_commit[2])
    crashed = store.load_checkpoint("approval-session-target")
    assert crashed is not None and crashed.terminal_result is None
    assert crashed.claim_token is not None
    assert crashed.claimed_cycle == 1
    assert crashed.lease_expires_at_ms is not None
    assert crashed.lease_expires_at_ms > time.time_ns() // 1_000_000
    assert crashed.resume_attempt == 1
    with store._lock:
        store._store["approval-session-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "finished after approval"
    assert len(effects) == 1
    assert session.get_items() == committed_items
    resumed_checkpoint_events = [event for event in resumed.events if isinstance(event, CheckpointResumedEvent)]
    assert len(resumed_checkpoint_events) == 1
    assert resumed_checkpoint_events[0].resume_attempt == 2
    target_commits = [entry for entry in session.commit_calls if entry[0] == target_commit[0]]
    assert [entry[3] for entry in target_commits] == ["committed", "replayed"]
    assert target_commits[0][1:3] == target_commits[1][1:3]
    retained = store.load_checkpoint("approval-session-target")
    assert retained is not None and retained.terminal_result is not None
    assert retained.resume_attempt == 2
    assert retained.claim_token is None
    assert len({entry.event_id for entry in retained.event_outbox}) == len(retained.event_outbox)
    actual_event_counts = {
        event_type: sum(entry.event.get("type") == event_type for entry in retained.event_outbox)
        for event_type in (
            "checkpoint_resumed",
            "operation_replayed",
            "tool_call_completed",
            "session_persisted",
            "run_completed",
        )
    }
    expected_event_counts = {
        "checkpoint_resumed": 1,
        "operation_replayed": 2,
        "tool_call_completed": 1,
        "session_persisted": 1,
        "run_completed": 1,
    }
    assert actual_event_counts == expected_event_counts
    replayed_operations = [
        entry.event["operation_id"] for entry in retained.event_outbox if entry.event.get("type") == "operation_replayed"
    ]
    assert len(replayed_operations) == len(set(replayed_operations)) == 2
    completed_tool_calls = [entry.event for entry in retained.event_outbox if entry.event.get("type") == "tool_call_completed"]
    assert {event["tool_call_id"] for event in completed_tool_calls} == {
        "approved-session-call",
    }
    assert len({event["operation_id"] for event in completed_tool_calls}) == 1
    tool_receipts = [
        entry.event
        for entry in retained.event_outbox
        if entry.event.get("type") == "tool_call_completed" and entry.event.get("tool_call_id") == "approved-session-call"
    ]
    assert len(tool_receipts) == 1
    assert tool_receipts[0]["operation_id"]
    assert tool_receipts[0]["operation_id"] in replayed_operations
    session_events = [entry for entry in retained.event_outbox if entry.event.get("type") == "session_persisted"]
    assert len(session_events) == 1
    assert session_events[0].state == "delivered"
    assert sum(event.type == "session_persisted" for event in resumed.events) == 1


def test_approval_resume_after_cycle_commit_continues_authoritative_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCheckpointStore()
    effects: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_cycle_write",
        needs_approval=True,
        tool_metadata={"idempotency": "supported"},
    )
    def approved_cycle_write(context: ToolContext) -> str:
        assert context.idempotency_key is not None
        effects.append(context.idempotency_key)
        return "written"

    def model_factory() -> ScriptedLLM:
        nonlocal provider_runs
        provider_runs += 1
        if provider_runs == 1:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="approved-cycle-call",
                                name="approved_cycle_write",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
        if provider_runs == 2:
            return ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="cycle committed",
                        tool_calls=[ToolCall(id="todo-cycle", name="todo_write", arguments={"todos": []})],
                    )
                ]
            )
        return ScriptedLLM(steps=[LLMResponse(content="finished after cycle recovery")])

    agent = Agent(
        name="approval-cycle-recovery-agent",
        instructions="Write only after approval, then continue.",
        model="test-model",
        tools=[approved_cycle_write],
    )
    source = Runner.run_sync(
        agent,
        "perform one approved write",
        run_config=_config(
            store,
            key="approval-cycle-source",
            provider=_provider(model_factory),
            max_cycles=3,
        ),
    )
    state = source.into_state()
    state.approve(state.pending_approval_ids()[0])
    configured = Runner.configured(
        RunConfig(
            checkpoint_config=CheckpointConfig(
                key="approval-cycle-target",
                resume_policy=ResumePolicy.RESUME_IF_PRESENT,
                store=store,
            )
        )
    )
    original_commit_cycle = CheckpointResumeController.commit_cycle
    crash_once = True

    def crash_after_target_cycle_commit(controller: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal crash_once
        result = original_commit_cycle(controller, *args, **kwargs)
        if crash_once and controller.checkpoint_key == "approval-cycle-target":
            crash_once = False
            raise SystemExit("crash after target cycle commit")
        return result

    monkeypatch.setattr(CheckpointResumeController, "commit_cycle", crash_after_target_cycle_commit)

    with pytest.raises(SystemExit, match="target cycle commit"):
        configured.resume(state)
    assert len(effects) == 1
    crashed = store.load_checkpoint("approval-cycle-target")
    assert crashed is not None
    assert crashed.cycle_index == 1
    assert crashed.terminal_result is None
    assert crashed.claim_token is None
    assert crashed.tool_journal == []
    assert crashed.resume_attempt == 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "finished after cycle recovery"
    assert len(effects) == 1
    assert provider_runs == 3
    assert sum(event.type == "checkpoint_resumed" for event in resumed.events) == 1
    assert not any(event.type == "operation_replayed" for event in resumed.events)
    retained = store.load_checkpoint("approval-cycle-target")
    assert retained is not None and retained.terminal_result is not None
    assert retained.terminal_result.status is AgentStatus.COMPLETED
    assert retained.resume_attempt == 2
