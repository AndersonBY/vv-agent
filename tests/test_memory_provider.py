from __future__ import annotations

from typing import Any

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
        session_id="session_1",
        cycle_index=3,
        message_count=12,
        estimated_tokens=9000,
    )
    completed = MemoryCompactCompleted(
        run_id="run_1",
        session_id="session_1",
        cycle_index=3,
        before_count=12,
        after_count=5,
        summary_tokens=500,
    )

    assert started.to_dict()["estimated_tokens"] == 9000
    assert completed.to_dict()["after_count"] == 5
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
        return MemoryProviderResult()

    def after_compact(self, event: MemoryCompactCompleted) -> None:
        self.completed.append(event)


def test_cycle_runner_calls_memory_providers_and_emits_compact_events() -> None:
    emitted: list[Any] = []
    provider = RecordingMemoryProvider()
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )
    memory_manager = MemoryManager(
        model="gpt-5.4",
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        summary_callback=lambda _prompt, _backend, _model: (
            '{"summary_version":"2.0","original_user_messages":["original"],'
            '"user_constraints":[],"decisions":[],"files_examined_or_modified":[],'
            '"errors_and_fixes":[],"progress":["done"],"key_facts":[],"open_issues":[],'
            '"current_work_state":"done","next_steps":[]}'
        ),
    )

    _, cycle_record = runner.run_cycle(
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
        memory_manager=memory_manager,
        previous_prompt_tokens=160,
        ctx=ExecutionContext(
            metadata={
                "_vv_agent_memory_providers": [provider],
                "_vv_agent_emit_event": emitted.append,
                "_vv_agent_run_id": "run_1",
                "_vv_agent_trace_id": "trace_1",
                "_vv_agent_agent_name": "assistant",
                "_vv_agent_session_id": "session_1",
            }
        ),
    )

    assert cycle_record.memory_compacted is True
    assert provider.started[0].to_dict()["estimated_tokens"] == 160
    assert provider.started[0].message_count == 4
    assert provider.completed[0].before_count == 4
    assert provider.completed[0].after_count < 4
    assert [event.type for event in emitted] == ["memory_compact_started", "memory_compact_completed"]
