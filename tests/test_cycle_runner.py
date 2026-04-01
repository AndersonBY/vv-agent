from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from vv_agent.llm import ScriptedLLM
from vv_agent.memory import (
    CompactionExhaustedError,
    MemoryManager,
    SessionMemory,
    SessionMemoryConfig,
)
from vv_agent.memory.microcompact import CLEARED_MARKER
from vv_agent.runtime.cycle_runner import MAX_PTL_RETRIES, CycleRunner
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask, LLMResponse, Message


def _fake_summary(_prompt: str, _backend: str | None, _model: str | None) -> str:
    return json.dumps(
        {
            "summary_version": 1,
            "user_constraints": [],
            "decisions": [],
            "progress": ["done"],
            "key_facts": [],
            "open_issues": [],
            "next_steps": [],
        },
        ensure_ascii=False,
    )


def _fake_session_memory_extract(_prompt: str, _backend: str | None, _model: str | None) -> str:
    return json.dumps(
        [{"category": "key_fact", "content": "session memory survives", "importance": 9}],
        ensure_ascii=False,
    )


def _build_task() -> AgentTask:
    return AgentTask(
        task_id="task_cycle_runner",
        model="gpt-5.4",
        system_prompt="sys",
        user_prompt="start",
        max_cycles=3,
    )


def _build_memory_manager(**overrides: Any) -> MemoryManager:
    params: dict[str, Any] = {
        "model": "gpt-5.4",
        "model_context_window": 60,
        "reserved_output_tokens": 10,
        "autocompact_buffer_tokens": 10,
        "summary_callback": _fake_summary,
    }
    params.update(overrides)
    return MemoryManager(**params)


def test_cycle_runner_retries_prompt_too_long_with_forced_compaction() -> None:
    sent_messages: list[list[Message]] = []

    def raise_ptl(_model: str, _messages: list[Message]) -> LLMResponse:
        raise RuntimeError("Prompt is too long for this model")

    def succeed_after_compact(_model: str, messages: list[Message]) -> LLMResponse:
        sent_messages.append(messages)
        return LLMResponse(content="done", raw={"usage": {"prompt_tokens": 12, "completion_tokens": 4}})

    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[raise_ptl, succeed_after_compact]),
        tool_registry=build_default_registry(),
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 40),
        Message(role="assistant", content="a" * 40),
        Message(role="user", content="c" * 40),
    ]

    next_messages, cycle_record = runner.run_cycle(
        task=_build_task(),
        messages=messages,
        cycle_index=1,
        memory_manager=_build_memory_manager(),
    )

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert any("<Compressed Agent Memory>" in message.content for message in sent_messages[0] if message.role == "user")
    assert next_messages[-1].content == "done"


def test_cycle_runner_retries_prompt_too_long_then_emergency_compact(monkeypatch) -> None:
    sent_messages: list[list[Message]] = []
    emergency_calls: list[float] = []

    def raise_ptl(_model: str, _messages: list[Message]) -> LLMResponse:
        raise RuntimeError("Prompt is too long for this model")

    def succeed_after_retry(_model: str, messages: list[Message]) -> LLMResponse:
        sent_messages.append(messages)
        return LLMResponse(content="done")

    memory_manager = _build_memory_manager()
    original_emergency_compact = memory_manager.emergency_compact

    def tracking_emergency_compact(
        self: MemoryManager,
        messages: list[Message],
        *,
        cycle_index: int | None = None,
        drop_ratio: float = 0.2,
    ) -> list[Message]:
        emergency_calls.append(drop_ratio)
        return original_emergency_compact(messages, cycle_index=cycle_index, drop_ratio=drop_ratio)

    monkeypatch.setattr(MemoryManager, "emergency_compact", tracking_emergency_compact)
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[raise_ptl, raise_ptl, succeed_after_retry]),
        tool_registry=build_default_registry(),
    )

    next_messages, cycle_record = runner.run_cycle(
        task=_build_task(),
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="u" * 40),
            Message(role="assistant", content="a" * 40),
            Message(role="user", content="c" * 40),
        ],
        cycle_index=1,
        memory_manager=memory_manager,
    )

    assert cycle_record.memory_compacted is True
    assert emergency_calls == [0.4]
    assert sent_messages
    assert next_messages[-1].content == "done"


def test_cycle_runner_raises_compaction_exhausted_after_max_ptl_retries() -> None:
    def raise_ptl(_model: str, _messages: list[Message]) -> LLMResponse:
        raise RuntimeError("context_length_exceeded")

    ptl_step = cast(Callable[[str, list[Message]], LLMResponse], raise_ptl)
    ptl_steps: list[LLMResponse | Callable[[str, list[Message]], LLMResponse]] = [
        cast(LLMResponse | Callable[[str, list[Message]], LLMResponse], ptl_step)
        for _ in range(MAX_PTL_RETRIES + 1)
    ]

    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=ptl_steps),
        tool_registry=build_default_registry(),
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 40),
        Message(role="assistant", content="a" * 40),
        Message(role="user", content="c" * 40),
    ]

    with pytest.raises(CompactionExhaustedError) as exc_info:
        runner.run_cycle(
            task=_build_task(),
            messages=messages,
            cycle_index=1,
            memory_manager=_build_memory_manager(),
        )

    assert exc_info.value.attempts == MAX_PTL_RETRIES + 1
    assert "context_length_exceeded" in str(exc_info.value.last_error)


def test_cycle_runner_does_not_swallow_non_ptl_errors() -> None:
    def raise_other(_model: str, _messages: list[Message]) -> LLMResponse:
        raise RuntimeError("network down")

    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[raise_other]),
        tool_registry=build_default_registry(),
    )

    with pytest.raises(RuntimeError, match="network down"):
        runner.run_cycle(
            task=_build_task(),
            messages=[Message(role="system", content="sys"), Message(role="user", content="hello")],
            cycle_index=1,
            memory_manager=_build_memory_manager(),
        )


def test_cycle_runner_recognizes_prompt_too_long_patterns() -> None:
    assert CycleRunner._is_prompt_too_long_error(RuntimeError("maximum context length exceeded")) is True
    assert CycleRunner._is_prompt_too_long_error(RuntimeError("request too large")) is True
    assert CycleRunner._is_prompt_too_long_error(RuntimeError("network down")) is False


def test_cycle_runner_recognizes_prompt_too_long_in_exception_chain() -> None:
    inner = RuntimeError("prompt is too long")
    outer = ValueError("API call failed")
    outer.__cause__ = inner

    assert CycleRunner._is_prompt_too_long_error(outer) is True


def test_cycle_runner_preemptively_microcompacts_before_threshold() -> None:
    sent_messages: list[list[Message]] = []

    def capture(_model: str, messages: list[Message]) -> LLMResponse:
        sent_messages.append(messages)
        return LLMResponse(content="done")

    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[capture]),
        tool_registry=build_default_registry(),
    )
    memory_manager = _build_memory_manager(
        model_context_window=240,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        microcompact_trigger_ratio=0.6,
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=200,
        tool_result_compact_threshold=2_000,
    )
    messages = [
        Message(role="system", content="sys"),
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
        Message(role="tool", content="x" * 600, tool_call_id="call_old"),
        Message(role="assistant", content="recent reply"),
        Message(role="user", content="latest ask"),
    ]

    _, cycle_record = runner.run_cycle(
        task=_build_task(),
        messages=messages,
        cycle_index=3,
        memory_manager=memory_manager,
    )

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert any(message.role == "tool" and message.content == CLEARED_MARKER for message in sent_messages[0])
    assert all("<Compressed Agent Memory>" not in message.content for message in sent_messages[0])


def test_cycle_runner_reapplies_session_memory_after_full_compaction() -> None:
    sent_messages: list[list[Message]] = []

    def capture(_model: str, messages: list[Message]) -> LLMResponse:
        sent_messages.append(messages)
        return LLMResponse(content="done")

    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[capture]),
        tool_registry=build_default_registry(),
    )
    memory_manager = _build_memory_manager(
        model_context_window=70,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        base_system_prompt="sys",
        session_memory=SessionMemory(
            SessionMemoryConfig(
                min_tokens_before_extraction=50,
                min_text_messages=2,
                extraction_callback=_fake_session_memory_extract,
                token_model="gpt-5.4",
            )
        ),
    )

    _, cycle_record = runner.run_cycle(
        task=_build_task(),
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="u" * 40),
            Message(role="assistant", content="a" * 40),
            Message(role="user", content="c" * 40),
        ],
        cycle_index=2,
        memory_manager=memory_manager,
        previous_prompt_tokens=150,
    )

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert "<Session Memory>" in sent_messages[0][0].content
    assert "session memory survives" in sent_messages[0][0].content
