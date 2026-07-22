from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from support import FixedModelProvider, ModelMapProvider

from vv_agent import Agent, RunConfig, Runner
from vv_agent.config import ResolvedModelConfig
from vv_agent.events import MemoryCompactCompleted, MemoryCompactStarted
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.memory import (
    CLEARED_MARKER,
    MemoryManager,
    MemoryProviderResult,
    MemorySaveResult,
    SessionMemory,
    SessionMemoryConfig,
    SessionMemoryEntry,
)
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "memory_lifecycle.json"
_CONTRACT = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _summary_payload() -> str:
    return json.dumps(
        {
            "summary_version": "2.0",
            "original_user_messages": ["original"],
            "user_constraints": [],
            "decisions": [],
            "files_examined_or_modified": [],
            "errors_and_fixes": [],
            "progress": ["done"],
            "key_facts": [],
            "open_issues": [],
            "current_work_state": "done",
            "next_steps": [],
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "case",
    _CONTRACT["capacity_contract"]["cases"],
    ids=[case["name"] for case in _CONTRACT["capacity_contract"]["cases"]],
)
def test_runtime_resolves_memory_capacity_from_contract_cases(
    case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = _CONTRACT["capacity_contract"]
    inputs = case["input"]
    expected = case["expected"]
    monkeypatch.setattr(
        "vv_agent.runtime.engine.resolve_model_token_limits",
        lambda _model: (None, None),
    )
    metadata: dict[str, Any] = {
        "model_context_window": inputs["model_context_window"],
        "model_max_output_tokens": inputs["model_max_output_tokens"],
        "autocompact_buffer_tokens": inputs["autocompact_buffer_tokens"],
        "session_memory_enabled": False,
    }
    if inputs["task_metadata_reserved_output_tokens"] is not None:
        metadata["reserved_output_tokens"] = inputs["task_metadata_reserved_output_tokens"]
    settings = (
        ModelSettings(max_tokens=inputs["effective_model_max_tokens"])
        if inputs["effective_model_max_tokens"] is not None
        else None
    )
    task = AgentTask(
        task_id=case["name"],
        model="capacity-model",
        system_prompt="system",
        user_prompt="run",
        memory_compact_threshold=inputs["configured_threshold"],
        model_settings=settings,
        metadata=metadata,
    )
    ctx = ExecutionContext(metadata={"_vv_agent_model_settings": settings} if settings is not None else {})
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path, ctx=ctx)

    assert manager.compact_threshold == inputs["configured_threshold"]
    assert manager.model_context_window == inputs["model_context_window"]
    assert manager.model_max_output_tokens == inputs["model_max_output_tokens"]
    assert manager.reserved_output_tokens == expected["reserved_output_tokens"]
    assert manager.reserved_output_source == expected["reserved_output_source"]
    assert manager.autocompact_buffer_tokens == capacity["autocompact_buffer_tokens_default"]
    assert manager.autocompact_threshold == expected["effective_threshold"]
    assert manager.microcompact_trigger_threshold == expected["microcompact_threshold"]


def test_omitted_memory_compact_threshold_defaults_match_contract() -> None:
    expected = _CONTRACT["capacity_contract"]["configured_default_threshold"]
    task = AgentTask(
        task_id="default-capacity",
        model="capacity-model",
        system_prompt="system",
        user_prompt="run",
    )
    restored = AgentTask.from_dict(
        {
            "task_id": "restored-default-capacity",
            "model": "capacity-model",
            "system_prompt": "system",
            "user_prompt": "run",
        }
    )

    assert task.memory_compact_threshold == expected
    assert restored.memory_compact_threshold == expected
    assert MemoryManager().compact_threshold == expected


@pytest.mark.parametrize(
    "case",
    _CONTRACT["capacity_contract"]["context_window_resolution"]["cases"],
    ids=[case["name"] for case in _CONTRACT["capacity_contract"]["context_window_resolution"]["cases"]],
)
def test_runtime_context_window_resolution_matches_contract(
    case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = case["input"]
    monkeypatch.setattr(
        "vv_agent.runtime.engine.resolve_model_token_limits",
        lambda _model: (inputs["resolved_model_context_window"], 32_000),
    )
    task = AgentTask(
        task_id=case["name"],
        model="capacity-model",
        system_prompt="system",
        user_prompt="run",
        metadata={
            "model_context_window": inputs["task_metadata_model_context_window"],
            "session_memory_enabled": False,
        },
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)

    assert manager.model_context_window == case["expected_model_context_window"]


def test_runtime_routes_summary_through_configured_backend_model_pair(tmp_path: Path) -> None:
    contract = _CONTRACT["summary_route"]
    requests: list[LlmRequest] = []

    def summarize(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content=_summary_payload())

    summary_llm = ScriptedLLM(steps=[summarize])
    resolved = ResolvedModelConfig(
        backend=contract["backend"],
        requested_model=contract["model"],
        selected_model=contract["model"],
        model_id=contract["model"],
        endpoint_options=[],
    )
    model_provider = ModelMapProvider(
        routes={contract["model"]: (summary_llm, resolved)},
        default_model=contract["model"],
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        model_provider=model_provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="memory_route",
        model="main-model",
        system_prompt="system",
        user_prompt="continue",
        initial_messages=[
            Message(role="system", content="system"),
            Message(role="user", content="u" * 160),
            Message(role="assistant", content="a" * 160),
            Message(role="user", content="c" * 160),
        ],
        max_cycles=1,
        no_tool_policy="finish",
        memory_compact_threshold=40,
        metadata={
            "memory_summary_backend": contract["backend"],
            "memory_summary_model": contract["model"],
            "model_context_window": 60,
            "reserved_output_tokens": 10,
            "autocompact_buffer_tokens": 10,
            "session_memory_enabled": False,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert model_provider.resolved_models == [contract["model"]]
    assert len(model_provider.resolved_models) == contract["resolution_count"]
    assert [request.model for request in requests] == [contract["request_model"]]


def test_runtime_routes_session_extraction_through_its_own_backend_model_pair(
    tmp_path: Path,
) -> None:
    contract = _CONTRACT["session_extraction_route"]
    extraction_requests: list[LlmRequest] = []

    def extract(request: LlmRequest) -> LLMResponse:
        extraction_requests.append(request)
        return LLMResponse(content='[{"category":"decision","content":"route extraction separately","importance":8}]')

    extraction_llm = ScriptedLLM(steps=[extract])
    resolved = ResolvedModelConfig(
        backend=contract["backend"],
        requested_model=contract["model"],
        selected_model=contract["model"],
        model_id=contract["model"],
        endpoint_options=[],
    )
    model_provider = ModelMapProvider(
        routes={contract["model"]: (extraction_llm, resolved)},
        default_model=contract["model"],
    )

    def assert_injected(request: LlmRequest) -> LLMResponse:
        assert "route extraction separately" in request.messages[0].content
        return LLMResponse(content="done")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[assert_injected]),
        model_provider=model_provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="session_memory_route",
        model="main-model",
        system_prompt="system",
        user_prompt="remember this decision",
        max_cycles=1,
        no_tool_policy="finish",
        memory_compact_threshold=10_000,
        metadata={
            "session_memory_enabled": True,
            "session_memory_min_tokens": 1,
            "session_memory_min_text_messages": 1,
            "session_memory_extraction_backend": contract["backend"],
            "session_memory_extraction_model": contract["model"],
            "model_context_window": 20_000,
            "reserved_output_tokens": 0,
            "autocompact_buffer_tokens": 0,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert model_provider.resolved_models == [contract["model"]]
    assert len(model_provider.resolved_models) == contract["resolution_count"]
    assert [request.model for request in extraction_requests] == [contract["request_model"]]


class RecordingMemoryProvider:
    def __init__(self, *, fail_before: bool = False, fail_after: bool = False) -> None:
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.started: list[Any] = []
        self.completed: list[Any] = []

    def search(self, request):
        del request
        return []

    def save(self, request):
        del request
        return MemorySaveResult()

    def before_compact(self, event):
        self.started.append(event)
        if self.fail_before:
            raise RuntimeError("before exploded")
        return MemoryProviderResult(metadata={"phase": "before"})

    def after_compact(self, event):
        self.completed.append(event)
        if self.fail_after:
            raise RuntimeError("after exploded")


def _ptl_memory_manager() -> MemoryManager:
    return MemoryManager(
        compact_threshold=10_000,
        model="main-model",
        model_context_window=20_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
    )


def _ptl_task() -> AgentTask:
    return AgentTask(
        task_id="memory_ptl",
        model="main-model",
        system_prompt="system",
        user_prompt="continue",
        no_tool_policy="finish",
    )


def test_ptl_forced_and_emergency_attempts_notify_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _CONTRACT["provider_attempts"]
    emitted: list[Any] = []
    provider = RecordingMemoryProvider()
    emergency_drop_ratios: list[float] = []
    original_emergency_compact = MemoryManager.emergency_compact

    def tracking_emergency_compact(self, messages, *, cycle_index=None, drop_ratio=0.2):
        emergency_drop_ratios.append(drop_ratio)
        return original_emergency_compact(
            self,
            messages,
            cycle_index=cycle_index,
            drop_ratio=drop_ratio,
        )

    monkeypatch.setattr(MemoryManager, "emergency_compact", tracking_emergency_compact)

    def prompt_too_long(_request: LlmRequest) -> LLMResponse:
        raise RuntimeError("Prompt is too long for this model")

    runner = CycleRunner(
        llm_client=ScriptedLLM(
            steps=[
                prompt_too_long,
                prompt_too_long,
                LLMResponse(content="done"),
            ]
        ),
        tool_registry=build_default_registry(),
    )
    next_messages, cycle = runner.run_cycle(
        task=_ptl_task(),
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content="first"),
            Message(role="assistant", content="working"),
            Message(role="user", content="continue"),
        ],
        cycle_index=2,
        memory_manager=_ptl_memory_manager(),
        ctx=ExecutionContext(
            event_handler=emitted.append,
            metadata={
                "_vv_agent_memory_providers": [provider],
                "_vv_agent_run_id": "run_memory",
                "_vv_agent_trace_id": "trace_memory",
                "_vv_agent_agent_name": "assistant",
            },
        ),
    )

    assert cycle.memory_compacted is True
    assert next_messages[-1].content == "done"
    assert len(provider.started) == contract["started_count"]
    assert len(provider.completed) == contract["completed_count"]
    assert [event.type for event in emitted] == [
        "memory_compact_started",
        "memory_compact_completed",
        "memory_compact_started",
        "memory_compact_completed",
    ]
    assert emitted[0].metadata["memory_provider_results"]["RecordingMemoryProvider"] == contract["result_metadata"]
    assert emergency_drop_ratios == [contract["strategies"][1]["drop_ratio"]]
    assert [event.trigger for event in provider.started] == ["prompt_too_long", "prompt_too_long"]
    assert [event.mode for event in provider.completed] == ["summary", "none"]
    assert [event.changed for event in provider.completed] == [True, False]


def _microcompact_messages(recent_tool_chars: int = 800) -> list[Message]:
    return [
        Message(role="system", content="system"),
        Message(role="user", content="start"),
        Message(
            role="assistant",
            content="old tool call",
            tool_calls=[
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="x" * 800, tool_call_id="call_old"),
        Message(role="assistant", content="cycle two"),
        Message(role="user", content="continue"),
        Message(
            role="assistant",
            content="recent tool call",
            tool_calls=[
                {
                    "id": "call_recent",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="y" * recent_tool_chars, tool_call_id="call_recent"),
    ]


@pytest.mark.parametrize(
    ("messages", "expected_mode", "expected_changed"),
    [
        (_microcompact_messages(), "micro", True),
        (
            [
                Message(role="system", content="system"),
                Message(role="user", content="continue"),
            ],
            "none",
            False,
        ),
    ],
    ids=["same-count-content-change", "trigger-without-change"],
)
def test_preemptive_compaction_producer_reports_content_aware_outcome(
    messages: list[Message],
    expected_mode: str,
    expected_changed: bool,
) -> None:
    emitted: list[Any] = []
    manager = MemoryManager(
        compact_threshold=4_000,
        model="main-model",
        model_context_window=4_000,
        model_max_output_tokens=1_000,
        reserved_output_tokens=0,
        reserved_output_source="task_metadata",
        autocompact_buffer_tokens=0,
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=500,
    )
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    next_messages, _ = runner.run_cycle(
        task=_ptl_task(),
        messages=messages,
        cycle_index=4,
        memory_manager=manager,
        previous_prompt_tokens=3_500,
        ctx=ExecutionContext(event_handler=emitted.append),
    )

    started, completed = emitted
    assert started.trigger == "micro_threshold"
    assert started.effective_threshold == 4_000
    assert started.microcompact_threshold == 3_000
    assert completed.mode == expected_mode
    assert completed.changed is expected_changed
    assert completed.before_count == completed.after_count
    if expected_changed:
        assert any(message.content == CLEARED_MARKER for message in next_messages)


def test_preemptive_microcompact_runs_before_optional_warning() -> None:
    contract = _CONTRACT["compaction_events"]["simultaneous_warning_and_microcompact"]
    emitted: list[Any] = []
    manager = MemoryManager(
        compact_threshold=4_000,
        model="main-model",
        model_context_window=4_000,
        reserved_output_tokens=0,
        reserved_output_source="task_metadata",
        autocompact_buffer_tokens=0,
        include_memory_warning=True,
        language="en-US",
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=500,
    )
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    next_messages, _ = runner.run_cycle(
        task=_ptl_task(),
        messages=_microcompact_messages(),
        cycle_index=4,
        memory_manager=manager,
        previous_prompt_tokens=3_800,
        ctx=ExecutionContext(event_handler=emitted.append),
    )

    assert contract["order"] == [
        "microcompact_eligible_old_tool_results",
        "recalculate_effective_length",
        "append_memory_warning_only_if_post_microcompact_length_remains_eligible",
    ]
    assert any(message.content == CLEARED_MARKER for message in next_messages)
    assert all("current memory usage has exceeded" not in message.content for message in next_messages)
    completed = emitted[-1]
    assert completed.mode == "micro"
    assert completed.changed is True


@pytest.mark.parametrize("initial_usage", [3_800, 4_200])
def test_public_memory_manager_microcompacts_before_warning_on_both_threshold_paths(
    initial_usage: int,
) -> None:
    manager = MemoryManager(
        compact_threshold=4_000,
        model="main-model",
        model_context_window=4_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        include_memory_warning=True,
        language="en-US",
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=500,
    )

    result = manager.compact_with_result(
        _microcompact_messages(),
        cycle_index=4,
        total_tokens=initial_usage,
    )

    assert result.changed is True
    assert result.mode == "micro"
    assert any(message.content == CLEARED_MARKER for message in result.messages)
    assert all("current memory usage has exceeded" not in message.content for message in result.messages)


def test_public_memory_manager_reports_structural_as_stronger_than_microcompact() -> None:
    messages = _microcompact_messages()
    messages.insert(2, Message(role="assistant", content=""))
    manager = MemoryManager(
        compact_threshold=4_000,
        model="main-model",
        model_context_window=4_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=500,
    )

    result = manager.compact_with_result(
        messages,
        cycle_index=4,
        total_tokens=3_800,
    )

    assert result.changed is True
    assert result.mode == "structural"
    assert any(message.content == CLEARED_MARKER for message in result.messages)
    assert all(message.role != "assistant" or message.content for message in result.messages)


def test_public_memory_manager_keeps_warning_when_post_microcompact_usage_requires_it() -> None:
    messages = _microcompact_messages(recent_tool_chars=8_000)
    manager = MemoryManager(
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        include_memory_warning=True,
        language="en-US",
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=500,
    )
    post_microcompact, cleared = manager.microcompact_messages(messages, cycle_index=4)
    assert cleared == 1
    post_tokens = manager._calculate_message_length(post_microcompact)
    assert post_tokens > 900
    full_threshold = post_tokens + 100
    manager.compact_threshold = full_threshold
    manager.model_context_window = full_threshold

    result = manager.compact_with_result(
        messages,
        cycle_index=4,
        total_tokens=full_threshold,
    )

    assert result.changed is True
    assert result.mode == "structural"
    assert any(message.content == CLEARED_MARKER for message in result.messages)
    assert any("current memory usage has exceeded" in message.content for message in result.messages)


def test_runner_memory_provider_runtime_payload_and_journal_share_identity(
    tmp_path: Path,
) -> None:
    requests: list[LlmRequest] = []

    def summarize(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content=_summary_payload())

    def finish(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content="done")

    llm = ScriptedLLM(steps=[summarize, finish])

    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="capacity-model",
        selected_model="capacity-model",
        model_id="capacity-model",
        endpoint_options=[],
        context_length=64_000,
        max_output_tokens=8_192,
    )
    model_provider = FixedModelProvider(llm, resolved)

    provider = RecordingMemoryProvider()
    observed_events: list[Any] = []
    observer_order: list[str] = []

    def typed_observer(event: Any) -> None:
        if isinstance(event, (MemoryCompactStarted, MemoryCompactCompleted)):
            observed_events.append(event)
            observer_order.append(event.type)
        if isinstance(event, MemoryCompactStarted):
            raise RuntimeError("typed memory observer failed")

    with pytest.warns(RuntimeWarning, match="Run event stream observer failed: typed memory observer failed"):
        result = Runner.run_sync(
            Agent(
                name="capacity-agent",
                instructions="Finish.",
                model="capacity-model",
                metadata={"model_context_window": 0},
            ),
            "finish",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=model_provider,
                no_tool_policy="finish",
                initial_messages=[
                    Message(role="user", content="token " * 50_000),
                    Message(role="assistant", content="working"),
                ],
                memory_providers=[provider],
                stream=typed_observer,
            ),
        )

    assert result.status == AgentStatus.COMPLETED
    assert requests[-1].metadata["model_context_window"] == 64_000
    assert provider.started[0].model_context_window == 64_000
    for provider_events, event_type in (
        (provider.started, MemoryCompactStarted),
        (provider.completed, MemoryCompactCompleted),
    ):
        observed = [event for event in observed_events if isinstance(event, event_type)]
        journal_events = [event for event in result.events if isinstance(event, event_type)]
        assert len(observed) == len(provider_events) == len(journal_events) == 1
        provider_event = provider_events[0]
        assert observed[0].event_id == provider_event.event_id
        assert observed[0].created_at == provider_event.created_at
        assert journal_events[0].event_id == provider_event.event_id
        assert journal_events[0].created_at == provider_event.created_at
    assert observer_order == [
        "memory_compact_started",
        "memory_compact_completed",
    ]


def test_direct_runtime_memory_logs_are_emitted_and_observer_failures_are_isolated(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime_logs: list[str] = []

    def runtime_observer(event: Any) -> None:
        runtime_logs.append(event.type)
        if isinstance(event, MemoryCompactStarted):
            raise RuntimeError("direct memory observer failed")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(content=_summary_payload()),
                LLMResponse(content="done"),
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        event_handler=runtime_observer,
    )
    task = AgentTask(
        task_id="direct-memory-observer",
        model="capacity-model",
        system_prompt="system",
        user_prompt="finish",
        initial_messages=[
            Message(role="user", content="u" * 120),
            Message(role="assistant", content="a" * 120),
            Message(role="user", content="c" * 120),
        ],
        max_cycles=1,
        no_tool_policy="finish",
        memory_compact_threshold=40,
        metadata={
            "model_context_window": 60,
            "reserved_output_tokens": 10,
            "autocompact_buffer_tokens": 10,
            "session_memory_enabled": False,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert runtime_logs.count("memory_compact_started") == 1
    assert runtime_logs.count("memory_compact_completed") == 1
    assert "Runtime event observer failed" in caplog.text


def test_memory_provider_attempt_errors_are_fail_open() -> None:
    contract = _CONTRACT["provider_attempts"]
    emitted: list[Any] = []
    provider = RecordingMemoryProvider(fail_before=True, fail_after=True)
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    with pytest.warns(RuntimeWarning) as warnings:
        next_messages, cycle = runner.run_cycle(
            task=_ptl_task(),
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content="u" * 120),
                Message(role="assistant", content="a" * 120),
                Message(role="user", content="c" * 120),
            ],
            cycle_index=2,
            memory_manager=MemoryManager(
                compact_threshold=40,
                model="main-model",
                model_context_window=60,
                reserved_output_tokens=10,
                autocompact_buffer_tokens=10,
                summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
            ),
            previous_prompt_tokens=160,
            ctx=ExecutionContext(
                event_handler=emitted.append,
                metadata={
                    "_vv_agent_memory_providers": [provider],
                },
            ),
        )

    assert len(warnings) == 2
    assert cycle.memory_compacted is True
    assert next_messages[-1].content == "done"
    before_error = emitted[0].metadata["memory_provider_errors"][0]
    after_error = emitted[1].metadata["memory_provider_errors"][0]
    assert before_error["stage"] == contract["before_error"]["stage"]
    assert before_error["error"] == contract["before_error"]["error"]
    assert after_error["stage"] == contract["after_error"]["stage"]
    assert after_error["error"] == contract["after_error"]["error"]


def test_session_memory_refreshes_in_place_and_resets_token_baseline() -> None:
    contract = _CONTRACT["session_memory"]
    session_memory = SessionMemory(SessionMemoryConfig())
    session_memory.state.entries = [SessionMemoryEntry("decision", contract["stale_fact"], source_cycle=1)]
    manager = MemoryManager(
        compact_threshold=40,
        model="main-model",
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        base_system_prompt="system",
        session_memory=session_memory,
        summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
    )
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="u" * 120),
        Message(role="assistant", content="a" * 120),
        Message(role="user", content="c" * 120),
    ]
    stale = manager.apply_session_memory_context(messages)
    session_memory.state.entries = [SessionMemoryEntry("decision", contract["fresh_fact"], source_cycle=2)]

    refreshed = manager.apply_session_memory_context(stale)

    assert contract["stale_fact"] not in refreshed[0].content
    assert contract["fresh_fact"] in refreshed[0].content
    assert refreshed[0].content.count("<Session Memory>") == contract["block_count"]

    compacted, changed = manager.compact(refreshed, cycle_index=3, force=True)
    expected_baseline = manager._calculate_message_length(manager.apply_session_memory_context(compacted))

    assert changed is True
    assert session_memory.state.initialized is contract["initialized_after_compaction"]
    assert session_memory.state.tokens_at_last_extraction == expected_baseline
    assert expected_baseline > manager._calculate_message_length(compacted)


def _artifact_path(message: Message) -> str:
    for line in message.content.splitlines():
        if line.startswith("artifact_path:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"artifact path missing from {message.content!r}")


def test_artifact_fallbacks_are_unique_and_fail_open_at_workspace_boundary(tmp_path: Path) -> None:
    contract = _CONTRACT["artifacts"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = MemoryManager(
        workspace=workspace,
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    messages = [
        Message(role="tool", content="first artifact payload", tool_call_id="/"),
        Message(role="tool", content="second artifact payload", tool_call_id="/"),
    ]

    compacted, changed = manager._compact_tool_messages(messages, cycle_index=4)
    paths = [_artifact_path(message) for message in compacted]

    assert changed is True
    assert len(paths) == contract["fallback_count"]
    assert len(set(paths)) == len(paths)
    pattern = re.compile(contract["fallback_pattern"])
    assert all(pattern.fullmatch(PurePosixPath(path).name) for path in paths)
    assert all((workspace / path).is_file() for path in paths)

    blocked = workspace / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    failing = MemoryManager(
        workspace=workspace,
        tool_result_artifact_dir="blocked/nested",
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    failed_messages, failed_changed = failing._compact_tool_messages(
        [Message(role="tool", content="write failure payload", tool_call_id="call")],
        cycle_index=4,
    )
    assert failed_changed is True
    assert _artifact_path(failed_messages[0]) == contract["write_failure_path"]

    escaping = MemoryManager(
        workspace=workspace,
        tool_result_artifact_dir="../outside",
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    escaped_messages, escaped_changed = escaping._compact_tool_messages(
        [Message(role="tool", content="escape payload", tool_call_id="call")],
        cycle_index=4,
    )
    assert escaped_changed is True
    assert _artifact_path(escaped_messages[0]) == contract["escape_path"]
    assert not (tmp_path / "outside").exists()
