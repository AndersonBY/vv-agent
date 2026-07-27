from __future__ import annotations

import pytest
from support import model_call_context

from vv_agent.constants import READ_FILE_TOOL_NAME
from vv_agent.events import event_from_dict
from vv_agent.llm import ScriptedLLM
from vv_agent.memory import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask, LLMResponse, Message
from vv_agent.workspace import MemoryWorkspaceBackend


def _task() -> AgentTask:
    return AgentTask(
        task_id="micro-events",
        model="unknown-provider-model",
        prompt_bundle=build_raw_system_prompt_bundle("system"),
        user_prompt="continue",
        no_tool_policy="finish",
    )


def _manager() -> MemoryManager:
    return MemoryManager(
        compact_threshold=1_000,
        model="unknown-provider-model",
        model_context_window=1_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompaction_policy=MicrocompactionPolicy(
            trigger_ratio=0.75,
            target_ratio=0.60,
            keep_recent_cycles=1,
            min_result_chars=500,
        ),
        workspace_backend=MemoryWorkspaceBackend(),
        artifact_scope="micro-events",
    )


def _messages(result_chars: int) -> list[Message]:
    return [
        Message(role="system", content="system"),
        Message(
            role="assistant",
            content="search",
            tool_calls=[
                {
                    "id": "call-search",
                    "type": "function",
                    "function": {"name": "custom_search", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="x" * result_chars, tool_call_id="call-search"),
        Message(role="assistant", content="recent"),
    ]


def test_microcompact_events_include_candidate_and_archive_statistics() -> None:
    emitted = []
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    runner.run_cycle(
        task=_task(),
        messages=_messages(2_000),
        cycle_index=3,
        memory_manager=_manager(),
        previous_prompt_tokens=900,
        ctx=model_call_context(
            event_handler=emitted.append,
            metadata={
                "_vv_agent_run_id": "run-micro-events",
                "_vv_agent_trace_id": "trace-micro-events",
            },
        ),
    )

    memory_events = [event for event in emitted if event.type.startswith("memory_compact_")]
    assert [event.type for event in memory_events] == [
        "memory_compact_started",
        "memory_compact_completed",
    ]
    started, completed = memory_events
    assert started.microcompact_threshold == 750
    assert started.microcompact_target == 600
    assert started.candidate_count == 1
    assert started.estimated_reclaimable_tokens > 0
    assert completed.mode == "micro"
    assert completed.changed is True
    assert completed.archived_count == 1
    assert completed.reclaimed_tokens > 0
    assert completed.artifact_failure_count == 0
    assert event_from_dict(started.to_dict()).to_dict() == started.to_dict()
    assert event_from_dict(completed.to_dict()).to_dict() == completed.to_dict()


def test_microcompact_threshold_without_candidate_emits_no_memory_event() -> None:
    emitted = []
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    runner.run_cycle(
        task=_task(),
        messages=_messages(500),
        cycle_index=3,
        memory_manager=_manager(),
        previous_prompt_tokens=900,
        ctx=model_call_context(event_handler=emitted.append),
    )

    assert [event for event in emitted if event.type.startswith("memory_compact_")] == []


@pytest.mark.parametrize("configuration", ["workspace_disabled", "read_file_excluded"])
def test_microcompact_does_not_archive_when_read_file_is_not_model_visible(configuration: str) -> None:
    emitted = []
    task = _task()
    if configuration == "workspace_disabled":
        task.use_workspace = False
    else:
        task.exclude_tools = [READ_FILE_TOOL_NAME]
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )
    manager = _manager()
    messages = _messages(2_000)

    next_messages, _cycle = runner.run_cycle(
        task=task,
        messages=messages,
        cycle_index=3,
        memory_manager=manager,
        previous_prompt_tokens=900,
        ctx=model_call_context(event_handler=emitted.append),
    )

    assert next_messages[2].content == messages[2].content
    assert [event for event in emitted if event.type.startswith("memory_compact_")] == []
