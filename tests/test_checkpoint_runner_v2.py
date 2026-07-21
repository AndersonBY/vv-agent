from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

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
    ToolIdempotency,
    function_tool,
)
from vv_agent.checkpoint import AmbiguousToolPolicy, OperationState, ResumePolicy
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import ToolCallCompletedEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.state import InMemoryStateStore
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall


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


def _provider(factory: Callable[[], Any]) -> Callable[[Agent, RunConfig], tuple[Any, ResolvedModelConfig]]:
    def resolve(agent: Agent, run_config: RunConfig) -> tuple[Any, ResolvedModelConfig]:
        del agent, run_config
        return factory(), _resolved()

    return resolve


def _config(
    store: InMemoryStateStore,
    *,
    key: str,
    provider: Callable[[Agent, RunConfig], tuple[Any, ResolvedModelConfig]],
    max_cycles: int = 1,
) -> RunConfig:
    return RunConfig(
        model_provider=provider,
        max_cycles=max_cycles,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(
            key=key,
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
        ),
    )


def test_runner_checkpoint_terminal_replay_skips_model_and_terminal_notification() -> None:
    store = InMemoryStateStore()
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
    terminal = store.load_checkpoint_v2("terminal-replay")
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


def test_runner_checkpoint_terminal_replay_repeats_typed_output_validation() -> None:
    store = InMemoryStateStore()
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
    terminal = store.load_checkpoint_v2("typed-terminal-replay")
    assert terminal is not None
    assert terminal.status is AgentStatus.COMPLETED
    assert terminal.terminal_result is not None

    with pytest.raises(ValueError, match="failed to validate final output"):
        Runner.run_sync(agent, "return invalid typed output", run_config=config)
    assert model_calls == 1


def test_runner_checkpoint_terminal_replay_reuses_validated_failure() -> None:
    store = InMemoryStateStore()
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
    terminal = store.load_checkpoint_v2("validated-terminal-replay")
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


def test_runner_injects_stable_tool_idempotency_key_and_replay_does_not_repeat_effect() -> None:
    store = InMemoryStateStore()
    effects: list[tuple[str, str]] = []

    @function_tool(name="write_record", idempotency=ToolIdempotency.SUPPORTED)
    def write_record(context: ToolContext, value: str) -> str:
        assert context.idempotency_key is not None
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
    assert effects[0][0].startswith("idem_")

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
    store = InMemoryStateStore()
    effects = 0

    @function_tool(name="unsafe_write", idempotency=ToolIdempotency.UNKNOWN)
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
            run_config=_config(store, key="ambiguous-tool", provider=_provider(scripted)),
        )
    assert effects == 1

    with store._lock:
        crashed = store._store_v2["ambiguous-tool"]
        assert crashed.tool_journal[0].state.value == "started"
        crashed.lease_expires_at_ms = 1

    resumed = Runner.run_sync(
        agent,
        "write 42",
        run_config=_config(
            store,
            key="ambiguous-tool",
            provider=_provider(lambda: ScriptedLLM(steps=[])),
        ),
    )

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.completion_reason is None
    assert resumed.raw_result.resume_observation is not None
    assert resumed.raw_result.resume_observation.risk == "unknown_tool_side_effect"
    assert effects == 1
    retained = store.load_checkpoint_v2("ambiguous-tool")
    assert retained is not None
    assert retained.status is AgentStatus.RECONCILIATION_REQUIRED
    assert retained.tool_journal[0].state.value == "ambiguous"
    assert retained.claim_token is None


def test_runner_resume_restores_frozen_metadata_when_system_metadata_is_empty() -> None:
    store = InMemoryStateStore()
    metadata_snapshots: list[dict[str, Any]] = []

    @function_tool(name="retryable_metadata_write", idempotency=ToolIdempotency.SUPPORTED)
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
        crashed = store._store_v2["frozen-run-metadata"]
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
    store = InMemoryStateStore()
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

    @function_tool(name="unsafe_frozen_write", idempotency=ToolIdempotency.UNKNOWN)
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
    crashed = store.load_checkpoint_v2("frozen-resume")
    assert crashed is not None
    with store._lock:
        store._store_v2["frozen-resume"].lease_expires_at_ms = 1

    session.add_items([Message(role="user", content="history added after crash")])
    resumed = Runner.run_sync(agent, "write with frozen inputs", run_config=config)

    assert resumed.status is AgentStatus.RECONCILIATION_REQUIRED
    assert resumed.run_id == crashed.root_run_id
    assert resumed.trace_id == crashed.trace_id
    assert calls == {"instructions": 1, "context": 1, "guardrail": 1, "effects": 1}


def test_runner_resume_rejects_changed_static_instructions_before_external_work() -> None:
    store = InMemoryStateStore()
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
    store = InMemoryStateStore()
    effects: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_write",
        needs_approval=True,
        idempotency=ToolIdempotency.SUPPORTED,
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
    source_checkpoint = store.load_checkpoint_v2("approval-source")
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
    target = store.load_checkpoint_v2("approval-target")
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
    store = InMemoryStateStore()
    effects: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_replay_write",
        needs_approval=True,
        idempotency=ToolIdempotency.SUPPORTED,
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
    crashed = store.load_checkpoint_v2("approval-replay-target")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.SUCCEEDED
    with store._lock:
        store._store_v2["approval-replay-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "done after replay"
    assert len(effects) == 1
    replay_lifecycle = [
        event
        for event in resumed.events
        if getattr(event, "tool_call_id", None) == "call-approved-replay"
        and event.type.startswith("tool_call_")
    ]
    assert [event.type for event in replay_lifecycle] == [
        "tool_call_planned",
        "tool_call_completed",
    ]
    completed = replay_lifecycle[-1]
    assert isinstance(completed, ToolCallCompletedEvent)
    assert completed.execution_started is False
    assert completed.duration_ms is None


def test_ptl_retry_uses_distinct_model_operation_slots() -> None:
    store = InMemoryStateStore()
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
            provider=_provider(
                lambda: ScriptedLLM(
                    steps=[prompt_too_long, crash_after_second_dispatch]
                )
            ),
            max_cycles=2,
        ),
    )

    assert result.status is AgentStatus.RECONCILIATION_REQUIRED
    assert model_calls == 2
    checkpoint = store.load_checkpoint_v2("ptl-operation-slots")
    assert checkpoint is not None
    assert [entry.state for entry in checkpoint.model_call_journal] == [
        OperationState.FAILED,
        OperationState.AMBIGUOUS,
    ]
    assert len({entry.operation_id for entry in checkpoint.model_call_journal}) == 2


def test_session_commit_crash_replays_model_receipt_without_duplicate_append() -> None:
    store = InMemoryStateStore()
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
        store._store_v2["session-commit-crash"].lease_expires_at_ms = 1

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


def test_approval_resume_crash_retries_same_idempotency_key_once() -> None:
    store = InMemoryStateStore()
    effects: set[str] = set()
    invocations: list[str] = []
    provider_runs = 0

    @function_tool(
        name="approved_idempotent_write",
        needs_approval=True,
        idempotency=ToolIdempotency.SUPPORTED,
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
    crashed = store.load_checkpoint_v2("approval-crash-target")
    assert crashed is not None
    assert crashed.tool_journal[0].state is OperationState.STARTED
    with store._lock:
        store._store_v2["approval-crash-target"].lease_expires_at_ms = 1

    resumed = configured.resume(state)

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_output == "done after recovery"
    assert len(effects) == 1
    assert len(invocations) == 2
    assert invocations[0] == invocations[1]
