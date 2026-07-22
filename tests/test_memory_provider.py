from __future__ import annotations

from typing import Any

import pytest

from vv_agent.events import event_from_dict
from vv_agent.llm import ScriptedLLM
from vv_agent.memory import MemoryManager
from vv_agent.memory.provider import (
    MemoryCompactCompleted,
    MemoryCompactStarted,
    MemoryProviderResult,
    MemorySaveRequest,
    MemorySaveResult,
    MemorySearchRequest,
    MemorySearchResult,
)
from vv_agent.run_config import RunConfig
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask, LLMResponse, Message


def test_memory_compact_events_are_serializable() -> None:
    started = MemoryCompactStarted(
        run_id="run_1",
        trace_id="trace_1",
        session_id="session_1",
        cycle_index=3,
        message_count=12,
        estimated_tokens=9000,
        trigger="full_threshold",
        configured_threshold=250_000,
        effective_threshold=250_000,
        microcompact_threshold=187_500,
        model_context_window=1_000_000,
        model_max_output_tokens=384_000,
        reserved_output_tokens=16_000,
        reserved_output_source="framework_fallback",
        autocompact_buffer_tokens=13_000,
    )
    completed = MemoryCompactCompleted(
        run_id="run_1",
        trace_id="trace_1",
        session_id="session_1",
        cycle_index=3,
        before_count=12,
        after_count=5,
        summary_tokens=500,
        mode="summary",
        changed=True,
    )

    assert started.to_dict()["estimated_tokens"] == 9000
    assert started.to_dict()["model_max_output_tokens"] == 384_000
    assert completed.to_dict()["after_count"] == 5
    assert completed.to_dict()["mode"] == "summary"
    assert isinstance(event_from_dict(started.to_dict()), MemoryCompactStarted)
    assert isinstance(event_from_dict(completed.to_dict()), MemoryCompactCompleted)


def test_run_config_memory_providers_default_is_isolated() -> None:
    first = RunConfig()
    second = RunConfig()

    first.memory_providers.append(RecordingMemoryProvider())

    assert second.memory_providers == []


class RecordingMemoryProvider:
    def __init__(self) -> None:
        self.started: list[MemoryCompactStarted] = []
        self.completed: list[MemoryCompactCompleted] = []

    def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        del request
        return []

    def save(self, request: MemorySaveRequest) -> MemorySaveResult:
        del request
        return MemorySaveResult()

    def before_compact(self, event: MemoryCompactStarted) -> MemoryProviderResult:
        self.started.append(event)
        return MemoryProviderResult(metadata={"phase": "before"})

    def after_compact(self, event: MemoryCompactCompleted) -> None:
        self.completed.append(event)


class ThrowingMemoryProvider(RecordingMemoryProvider):
    def __init__(self, *, fail_before: bool = False, fail_after: bool = False) -> None:
        super().__init__()
        self.fail_before = fail_before
        self.fail_after = fail_after

    def before_compact(self, event: MemoryCompactStarted) -> MemoryProviderResult:
        if self.fail_before:
            raise RuntimeError("before exploded")
        return super().before_compact(event)

    def after_compact(self, event: MemoryCompactCompleted) -> None:
        if self.fail_after:
            raise RuntimeError("after exploded")
        super().after_compact(event)


def _build_runner() -> CycleRunner:
    return CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )


def _build_memory_manager() -> MemoryManager:
    return MemoryManager(
        model="gpt-5.4",
        model_context_window=60,
        model_max_output_tokens=32,
        reserved_output_tokens=10,
        reserved_output_source="task_metadata",
        autocompact_buffer_tokens=10,
        summary_callback=lambda _prompt, _backend, _model: (
            '{"summary_version":"2.0","original_user_messages":["original"],'
            '"user_constraints":[],"decisions":[],"files_examined_or_modified":[],'
            '"errors_and_fixes":[],"progress":["done"],"key_facts":[],"open_issues":[],'
            '"current_work_state":"done","next_steps":[]}'
        ),
    )


def _run_compacting_cycle(provider: RecordingMemoryProvider, emitted: list[Any]) -> tuple[list[Message], Any]:
    return _build_runner().run_cycle(
        task=AgentTask(
            task_id="task_1",
            model="gpt-5.4",
            system_prompt="sys",
            user_prompt="start",
            max_cycles=1,
        ),
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="u" * 80),
            Message(role="assistant", content="a" * 80),
            Message(role="user", content="c" * 80),
        ],
        cycle_index=3,
        memory_manager=_build_memory_manager(),
        previous_prompt_tokens=160,
        ctx=ExecutionContext(
            event_handler=emitted.append,
            metadata={
                "_vv_agent_memory_providers": [provider],
                "_vv_agent_run_id": "run_1",
                "_vv_agent_trace_id": "trace_1",
                "_vv_agent_agent_name": "assistant",
                "_vv_agent_session_id": "session_1",
            },
        ),
    )


def test_cycle_runner_calls_memory_providers_and_emits_compact_events() -> None:
    emitted: list[Any] = []
    provider = RecordingMemoryProvider()

    _, cycle_record = _run_compacting_cycle(provider, emitted)

    assert cycle_record.memory_compacted is True
    assert provider.started[0].to_dict()["estimated_tokens"] == 160
    assert provider.started[0].message_count == 4
    assert provider.started[0].metadata["messages"][1].content == "u" * 80
    assert provider.started[0].trigger == "full_threshold"
    assert provider.started[0].configured_threshold == 250_000
    assert provider.started[0].effective_threshold == 40
    assert provider.started[0].microcompact_threshold == 30
    assert provider.started[0].model_context_window == 60
    assert provider.started[0].model_max_output_tokens == 32
    assert provider.started[0].reserved_output_tokens == 10
    assert provider.started[0].reserved_output_source == "task_metadata"
    assert provider.started[0].autocompact_buffer_tokens == 10
    assert provider.completed[0].before_count == 4
    assert provider.completed[0].after_count < 4
    assert provider.completed[0].mode == "summary"
    assert provider.completed[0].changed is True
    assert [event.type for event in emitted] == ["memory_compact_started", "memory_compact_completed"]
    assert "messages" not in emitted[0].metadata
    assert emitted[0].metadata["memory_provider_results"]["RecordingMemoryProvider"]["phase"] == "before"


def test_cycle_runner_fails_open_when_before_memory_provider_raises() -> None:
    emitted: list[Any] = []
    provider = ThrowingMemoryProvider(fail_before=True)

    with pytest.warns(RuntimeWarning, match="Memory provider ThrowingMemoryProvider before_compact failed"):
        next_messages, cycle_record = _run_compacting_cycle(provider, emitted)

    assert cycle_record.memory_compacted is True
    assert next_messages[-1].content == "done"
    assert emitted[0].metadata["memory_provider_errors"][0]["provider"] == "ThrowingMemoryProvider"
    assert emitted[0].metadata["memory_provider_errors"][0]["stage"] == "before_compact"
    assert emitted[0].metadata["memory_provider_errors"][0]["error"] == "before exploded"


def test_cycle_runner_fails_open_when_after_memory_provider_raises() -> None:
    emitted: list[Any] = []
    provider = ThrowingMemoryProvider(fail_after=True)

    with pytest.warns(RuntimeWarning, match="Memory provider ThrowingMemoryProvider after_compact failed"):
        next_messages, cycle_record = _run_compacting_cycle(provider, emitted)

    assert cycle_record.memory_compacted is True
    assert next_messages[-1].content == "done"
    assert emitted[1].metadata["memory_provider_errors"][0]["provider"] == "ThrowingMemoryProvider"
    assert emitted[1].metadata["memory_provider_errors"][0]["stage"] == "after_compact"
    assert emitted[1].metadata["memory_provider_errors"][0]["error"] == "after exploded"
