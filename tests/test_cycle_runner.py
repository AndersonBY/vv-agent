from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from support import model_call_context
from support.compaction import run_model_turn

from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.llm.errors import MAX_PTL_RETRIES, is_prompt_too_long_error
from vv_agent.memory import (
    MemoryManager,
    SessionMemory,
    SessionMemoryConfig,
)
from vv_agent.memory.microcompact import COMPACT_MARKER_OPENING
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask, LLMResponse, Message
from vv_agent.workspace import MemoryWorkspaceBackend


def _fake_summary(_prompt: str, _backend: str | None, _model: str | None) -> str:
    from test_memory_lifecycle_contract import _summary_payload

    return _summary_payload()


def _fake_session_memory_extract(_prompt: str, _backend: str | None, _model: str | None) -> str:
    return json.dumps(
        [{"category": "key_fact", "content": "session memory survives", "importance": 9}],
        ensure_ascii=False,
    )


def _build_task() -> AgentTask:
    return AgentTask(
        task_id="task_cycle_runner",
        model="gpt-5.4",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="start",
        max_cycles=3,
    )


def _build_memory_manager(**overrides: Any) -> MemoryManager:
    params: dict[str, Any] = {
        "model": "gpt-5.4",
        "model_context_window": 4000,
        "compact_threshold": 3000,
        "keep_recent_messages": 1,
        "reserved_output_tokens": 10,
        "autocompact_buffer_tokens": 10,
        "summary_callback": _fake_summary,
    }
    params.update(overrides)
    return MemoryManager(**params)


def test_cycle_runner_retries_prompt_too_long_with_forced_compaction() -> None:
    sent_messages: list[list[Message]] = []

    def raise_ptl(request: LlmRequest) -> LLMResponse:
        _model, _messages = request.model, request.messages
        raise RuntimeError("Prompt is too long for this model")

    def succeed_after_compact(request: LlmRequest) -> LLMResponse:
        _model, messages = request.model, request.messages
        sent_messages.append(messages)
        return LLMResponse(content="done", raw={"usage": {"prompt_tokens": 12, "completion_tokens": 4}})

    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[raise_ptl, succeed_after_compact]),
        tool_registry=build_default_registry(),
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u " * 800),
        Message(role="assistant", content="a " * 800),
        Message(role="user", content="c" * 40),
    ]

    result = runner(
        task=_build_task(),
        messages=messages,
        memory_manager=_build_memory_manager(),
        ctx=model_call_context(),
    )
    next_messages = result.messages
    cycle_record = result.cycles[0]

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert any("<Compressed Agent Memory>" in message.content for message in sent_messages[0] if message.role == "user")
    assert next_messages[-1].content == "done"


def test_cycle_runner_retries_prompt_too_long_then_emergency_compact(monkeypatch) -> None:
    sent_messages: list[list[Message]] = []
    emergency_calls: list[float] = []

    def raise_ptl(request: LlmRequest) -> LLMResponse:
        _model, _messages = request.model, request.messages
        raise RuntimeError("Prompt is too long for this model")

    def succeed_after_retry(request: LlmRequest) -> LLMResponse:
        _model, messages = request.model, request.messages
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
    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[raise_ptl, raise_ptl, succeed_after_retry]),
        tool_registry=build_default_registry(),
    )

    result = runner(
        task=_build_task(),
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="u " * 800),
            Message(role="assistant", content="a " * 800),
            Message(role="user", content="c" * 40),
        ],
        memory_manager=memory_manager,
        ctx=model_call_context(),
    )
    next_messages = result.messages
    cycle_record = result.cycles[0]

    assert cycle_record.memory_compacted is True
    assert emergency_calls == []  # Emergency is a logged summary plan, not an in-memory pruner.
    assert sent_messages
    assert next_messages[-1].content == "done"


def test_cycle_runner_raises_compaction_exhausted_after_max_ptl_retries() -> None:
    def raise_ptl(request: LlmRequest) -> LLMResponse:
        _model, _messages = request.model, request.messages
        raise RuntimeError("context_length_exceeded")

    ptl_step = cast(Callable[[LlmRequest], LLMResponse], raise_ptl)
    ptl_steps: list[LLMResponse | Callable[[LlmRequest], LLMResponse]] = [
        cast(LLMResponse | Callable[[LlmRequest], LLMResponse], ptl_step) for _ in range(MAX_PTL_RETRIES + 1)
    ]

    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=ptl_steps),
        tool_registry=build_default_registry(),
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u " * 800),
        Message(role="assistant", content="a " * 800),
        Message(role="user", content="c" * 40),
    ]

    result = runner(
        task=_build_task(),
        messages=messages,
        memory_manager=_build_memory_manager(),
        ctx=model_call_context(),
    )

    assert result.status.value == "failed"
    assert "CompactionExhaustedError" in str(result.error)
    assert len([c for c in result.token_usage.model_calls if c.operation.value == "agent_cycle"]) == MAX_PTL_RETRIES + 1


def test_cycle_runner_does_not_swallow_non_ptl_errors() -> None:
    def raise_other(request: LlmRequest) -> LLMResponse:
        _model, _messages = request.model, request.messages
        raise RuntimeError("network down")

    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[raise_other]),
        tool_registry=build_default_registry(),
    )

    result = runner(
        task=_build_task(),
        messages=[Message(role="system", content="sys"), Message(role="user", content="hello")],
        memory_manager=_build_memory_manager(),
        ctx=model_call_context(),
    )
    assert result.status.value == "failed"
    assert result.error_code == "model_outcome_unknown"


def test_cycle_runner_recognizes_prompt_too_long_patterns() -> None:
    assert is_prompt_too_long_error(RuntimeError("maximum context length exceeded")) is True
    assert is_prompt_too_long_error(RuntimeError("request too large")) is True
    assert is_prompt_too_long_error(RuntimeError("network down")) is False


def test_cycle_runner_recognizes_prompt_too_long_in_exception_chain() -> None:
    inner = RuntimeError("prompt is too long")
    outer = ValueError("API call failed")
    outer.__cause__ = inner

    assert is_prompt_too_long_error(outer) is True

    argument_wrapper = RuntimeError("API call failed", inner)
    assert is_prompt_too_long_error(argument_wrapper) is True

    cycle = RuntimeError("network down")
    cycle.__cause__ = cycle
    cycle.__context__ = outer
    assert is_prompt_too_long_error(cycle) is True
    cycle.__context__ = None
    assert is_prompt_too_long_error(cycle) is False


def test_cycle_runner_preemptively_microcompacts_before_threshold() -> None:
    sent_messages: list[list[Message]] = []

    def capture(request: LlmRequest) -> LLMResponse:
        _model, messages = request.model, request.messages
        sent_messages.append(messages)
        return LLMResponse(content="done")

    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[capture]),
        tool_registry=build_default_registry(),
    )
    memory_manager = _build_memory_manager(
        model_context_window=240,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        microcompaction_policy=MicrocompactionPolicy(
            trigger_ratio=0.6,
            target_ratio=0.5,
            keep_recent_cycles=1,
            min_result_chars=200,
        ),
        workspace_backend=MemoryWorkspaceBackend(),
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
        Message(role="tool", content="large result " * 600, tool_call_id="call_old"),
        Message(role="assistant", content="recent reply"),
        Message(role="user", content="latest ask"),
    ]

    result = runner(
        task=_build_task(),
        messages=messages,
        memory_manager=memory_manager,
        ctx=model_call_context(),
    )
    cycle_record = result.cycles[0]

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert any(message.role == "tool" and message.content.startswith(COMPACT_MARKER_OPENING) for message in sent_messages[0])
    assert all("<Compressed Agent Memory>" not in message.content for message in sent_messages[0])


def test_cycle_runner_keeps_the_frozen_prompt_after_session_memory_extraction() -> None:
    sent_messages: list[list[Message]] = []

    def capture(request: LlmRequest) -> LLMResponse:
        _model, messages = request.model, request.messages
        sent_messages.append(messages)
        return LLMResponse(content="done")

    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[capture]),
        tool_registry=build_default_registry(),
    )
    memory_manager = _build_memory_manager(
        model_context_window=4000,
        compact_threshold=100,
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

    result = runner(
        task=_build_task(),
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="user evidence " * 40),
            Message(role="assistant", content="assistant analysis " * 40),
            Message(role="user", content="current request " * 40),
        ],
        memory_manager=memory_manager,
        ctx=model_call_context(),
    )
    cycle_record = result.cycles[0]

    assert cycle_record.memory_compacted is True
    assert sent_messages
    assert sent_messages[0][0].content == "sys"
    assert "<Session Memory>" not in sent_messages[0][0].content
    assert memory_manager.session_memory is not None
    assert any(call.operation.value == "session_memory" for call in result.token_usage.model_calls)
