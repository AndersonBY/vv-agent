from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from support.compaction import fixture, section
from support.compaction import messages as current_messages

from vv_agent.memory import MemoryManager, SessionMemory, SessionMemoryConfig, SessionMemoryEntry
from vv_agent.memory.microcompact import COMPACT_MARKER_OPENING
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.types import Message
from vv_agent.workspace import MemoryWorkspaceBackend


def _fake_summary(_prompt: str, _backend: str | None, _model: str | None) -> str:
    return json.dumps(
        {
            "summary_version": "2.0",
            "original_user_messages": ["original user request"],
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


def _fake_session_memory_extract(_prompt: str, _backend: str | None, _model: str | None) -> str:
    return json.dumps(
        [{"category": "key_fact", "content": "preserve prior decisions", "importance": 9}],
        ensure_ascii=False,
    )


def _build_manager(**overrides: Any) -> MemoryManager:
    params: dict[str, Any] = {
        "model": "gpt-5.4",
        "model_context_window": 80,
        "reserved_output_tokens": 10,
        "autocompact_buffer_tokens": 10,
    }
    params.update(overrides)
    return MemoryManager(**params)


def test_memory_compress_prompt_includes_original_user_messages_and_file_fields() -> None:
    manager = _build_manager(summary_event_limit=7)
    prompt = manager._build_compress_memory_prompt(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="Preserve my exact words."),
            Message(role="assistant", content="Working on it."),
        ]
    )

    assert '"original_user_messages"' in prompt
    assert '"files_examined_or_modified"' in prompt
    assert '"errors_and_fixes"' in prompt
    assert "7" in prompt


def test_memory_file_action_summary_uses_edit_file_for_modified_files() -> None:
    manager = MemoryManager()
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_edit",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({"path": "src/app.py"}),
                    },
                }
            ],
        )
    ]

    actions = manager._collect_file_actions(messages)

    assert actions == [{"path": "src/app.py", "action": "modified", "summary": "Modified src/app.py"}]


def test_memory_does_not_compact_when_small() -> None:
    manager = _build_manager(
        model_context_window=500,
        reserved_output_tokens=50,
        autocompact_buffer_tokens=50,
        keep_recent_messages=4,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is False
    assert compacted == messages


def test_memory_thresholds_respect_configured_ceiling() -> None:
    manager = MemoryManager(
        model_context_window=200_000,
        reserved_output_tokens=16_000,
        autocompact_buffer_tokens=13_000,
        warning_threshold_percentage=90,
    )

    assert manager.effective_context_window == 184_000
    assert manager.autocompact_threshold == 171_000
    assert manager.warning_threshold == 153_900


def test_memory_thresholds_fall_back_to_model_limit_when_smaller() -> None:
    manager = MemoryManager(
        compact_threshold=128_000,
        model_context_window=64_000,
        reserved_output_tokens=8_000,
        autocompact_buffer_tokens=13_000,
    )

    assert manager.autocompact_threshold == 43_000


def test_memory_compaction_persists_session_memory_without_rewriting_current_prompt() -> None:
    observed_prompts: list[str] = []

    def tracking_summary(prompt: str, backend: str | None, model: str | None) -> str:
        observed_prompts.append(prompt)
        return _fake_summary(prompt, backend, model)

    session_memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=50,
            min_text_messages=2,
            extraction_callback=_fake_session_memory_extract,
            token_model="gpt-5.4",
        )
    )
    manager = _build_manager(
        model_context_window=4000,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=2,
        summary_callback=tracking_summary,
        base_system_prompt="sys",
        session_memory=session_memory,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u " * 1000),
        Message(role="assistant", content="a " * 1000),
        Message(role="user", content="c" * 40),
    ]

    compacted, changed = manager.compact(messages, cycle_index=2, total_tokens=150, force=True)

    assert changed is True
    assert compacted[-2:] == messages[-2:]
    assert session_memory.state.entries
    assert session_memory.state.last_extracted_message_index == -1
    assert observed_prompts
    assert "<Session Memory>" not in observed_prompts[0]

    assert all("<Session Memory>" not in message.content for message in compacted)
    assert "preserve prior decisions" not in compacted[0].content


def test_session_memory_extraction_does_not_mutate_current_prompt() -> None:
    session_memory = SessionMemory(SessionMemoryConfig(token_model="gpt-5.4"))
    session_memory.state.entries = [
        SessionMemoryEntry(
            "decision",
            "keep the Python API small",
            source_cycle=2,
            importance=9,
        )
    ]
    manager = _build_manager(
        compact_threshold=10_000,
        model_context_window=20_000,
        reserved_output_tokens=100,
        autocompact_buffer_tokens=0,
        session_memory=session_memory,
    )
    messages = [Message(role="system", content="sys"), Message(role="user", content="small")]

    updated, changed = manager.compact(messages)

    assert changed is False
    assert updated == messages
    assert "<Session Memory>" not in updated[0].content
    assert "keep the Python API small" not in updated[0].content


def test_memory_compact_uses_microcompact_before_full_summary() -> None:
    manager = _build_manager(
        compact_threshold=1_000,
        model_context_window=1_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompaction_policy=MicrocompactionPolicy(
            keep_recent_cycles=1,
            min_result_chars=200,
        ),
        workspace_backend=MemoryWorkspaceBackend(),
        recovery_tool_available=True,
        summary_callback=_fake_summary,
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

    compacted, changed = manager.compact(messages, cycle_index=3, total_tokens=900)

    assert changed is True
    assert any(message.role == "tool" and message.content.startswith(COMPACT_MARKER_OPENING) for message in compacted)
    assert all("<Compressed Agent Memory>" not in message.content for message in compacted)


def test_below_threshold_return_preserves_all_tool_pairs_without_summary() -> None:
    case = fixture("memory_local")["summary_compaction"]["cases"][0]
    original = current_messages(case["input"]["messages"])
    manager = MemoryManager(compact_threshold=1, reserved_output_tokens=0, autocompact_buffer_tokens=0)
    result, changed = manager.compact(original, cycle_index=1000)
    assert result == original
    assert changed is False


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_structural_return_never_discards_image_without_summary(role: Literal["user", "assistant"]) -> None:
    original = [
        Message(role="system", content="sys"),
        Message(role=role, content="", image_url="data:image/png;base64,AAAA"),
        Message(role="assistant", content="image processed"),
    ]
    result, changed = MemoryManager(compact_threshold=1).compact(original, force=True)
    assert result == original
    assert changed is False


@pytest.mark.parametrize("force", [True, False])
def test_full_compaction_does_not_run_a_second_pruner(force: bool) -> None:
    captured = []
    backend = MemoryWorkspaceBackend()
    original = [Message(role="user", content="request")]
    for index in range(6):
        original.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[{"id": str(index), "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
                ),
                Message(role="tool", tool_call_id=str(index), content=f"result {index} " * 1000),
            ]
        )
    manager = MemoryManager(
        compact_threshold=1,
        model_context_window=100000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        keep_recent_messages=2,
        workspace_backend=backend,
        recovery_tool_available=True,
        tool_result_retentions={"lookup": ToolResultRetention.PRESERVE},
        summary_callback=lambda prompt, *_: captured.append(prompt) or '{"progress":["done"]}',
    )
    result = manager.compact_with_result(original, cycle_index=1000, force=force)
    assert result.mode == "summary"
    assert result.archived_count == 0
    prefix = section(captured[0], "Conversation Prefix")
    assert [item["content"] for item in prefix if item["role"] == "tool"] == [
        m.content for m in original[:-2] if m.role == "tool"
    ]
    assert result.messages[-2:] == original[-2:]


def test_microcompact_plan_indices_match_applied_messages() -> None:
    original = [
        Message(role="system", content="sys"),
        Message(role="assistant", content=""),
        Message(role="system", name="memory_summary", content="keep old summary"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "old", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        ),
        Message(role="tool", content="result " * 1500, tool_call_id="old"),
        Message(role="assistant", content="recent"),
    ]
    manager = MemoryManager(
        compact_threshold=10000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        workspace_backend=MemoryWorkspaceBackend(),
        recovery_tool_available=True,
        microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1),
    )
    plan = manager.plan_microcompaction(original, cycle_index=1000, current_tokens=8000)
    assert plan and [c.message_index for c in plan.candidates] == [4]
    result = manager.compact_with_result(
        original, cycle_index=1000, total_tokens=8000, microcompact_plan=plan, microcompact_planned=True
    )
    assert result.archived_count == 1
    assert next(m for m in result.messages if m.role == "tool").content.startswith(COMPACT_MARKER_OPENING)
    assert original[2] in result.messages
    assert original[4].content == "result " * 1500


@pytest.mark.parametrize(
    "variant", fixture("memory_local")["summary_compaction"]["control_failure_case"]["variants"], ids=lambda v: v["name"]
)
def test_summary_control_failures_propagate(variant: dict[str, Any]) -> None:
    from vv_agent.checkpoint import CheckpointError
    from vv_agent.runtime.cancellation import CancelledError

    inputs = fixture("memory_local")["summary_compaction"]["control_failure_case"]["input"]
    from vv_agent.budget import BudgetDimension, BudgetEnforcementBoundary, BudgetExhaustion, BudgetExhaustionReason
    from vv_agent.runtime.model_calls import ModelCallBudgetExhausted

    error = CancelledError("cancelled") if variant["name"] == "cancellation" else CheckpointError("control", code=variant["name"])
    if variant["name"] == "budget_exhaustion":
        error = ModelCallBudgetExhausted(
            BudgetExhaustion(
                dimension=BudgetDimension.TOTAL_TOKENS,
                reason=BudgetExhaustionReason.LIMIT_EXCEEDED,
                limit=10,
                observed=11,
                attempted_increment=None,
                overshoot=1,
                unit="tokens",
                enforcement_boundary=BudgetEnforcementBoundary.MODEL_CALL_COMPLETE,
            )
        )

    def callback(*_args):
        raise error

    manager = MemoryManager(keep_recent_messages=2, recovery_tool_available=True, summary_callback=callback)
    original = current_messages(inputs["messages"])
    with pytest.raises(type(error)):
        manager.compact(original, force=True)
    assert [m.to_dict() for m in original] == inputs["messages"]


def test_prompt_substitution_does_not_reinterpret_history() -> None:
    manager = MemoryManager(language="en-US", summary_event_limit=0)
    message = Message(role="user", content="{event_limit} {previous_summary_jcs} {conversation_prefix_jcs}")
    prompt = manager._build_compress_memory_prompt([message])
    assert section(prompt, "Conversation Prefix")[0]["content"] == message.content
    assert "Preserve up to 1 critical events" in prompt


def test_summary_raw_tail_stays_unpruned_after_empty_assistant_filtering() -> None:
    original = [Message(role="user", content="history " * 500)]
    for index in range(2):
        original.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[{"id": str(index), "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
                ),
                Message(role="tool", content="evidence " * 1000, tool_call_id=str(index)),
            ]
        )
    original.extend([Message(role="assistant", content=""), Message(role="assistant", content="")])
    manager = MemoryManager(
        compact_threshold=100,
        model_context_window=100000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        keep_recent_messages=2,
        workspace_backend=MemoryWorkspaceBackend(),
        recovery_tool_available=True,
        microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=0),
        summary_callback=lambda *_: '{"progress":["done"]}',
    )
    result = manager.compact_with_result(original, cycle_index=1000)
    assert result.mode == "summary"
    assert result.messages[-2:] == original[3:5]
    assert result.archived_count == 1


@pytest.mark.parametrize("stage", ["archive", "session_memory"])
def test_compaction_stages_propagate_cancellation(stage: str) -> None:
    from vv_agent.runtime.cancellation import CancelledError

    class CancelledBackend(MemoryWorkspaceBackend):
        def write_text_exclusive(self, path: str, content: str) -> int:
            raise CancelledError("cancelled archive")

    def cancelled_extract(*_args):
        raise CancelledError("cancelled extraction")

    session = (
        SessionMemory(
            SessionMemoryConfig(min_tokens_before_extraction=1, min_text_messages=1, extraction_callback=cancelled_extract)
        )
        if stage == "session_memory"
        else None
    )
    manager = MemoryManager(
        compact_threshold=1000,
        keep_recent_messages=1,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        workspace_backend=CancelledBackend(),
        recovery_tool_available=True,
        microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=0),
        session_memory=session,
    )
    original = [
        Message(role="user", content="request"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "old", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        ),
        Message(role="tool", content="evidence " * 1000, tool_call_id="old"),
        Message(role="assistant", content="recent"),
    ]
    with pytest.raises(CancelledError):
        manager.compact(original, cycle_index=1000)


@pytest.mark.parametrize("accepted", [True, False])
def test_prefix_images_only_leave_after_accepted_summary(accepted: bool) -> None:
    captured = []
    original = [
        Message(role="system", content="sys"),
        Message(role="user", content="go"),
        Message(role="user", content="screenshot", image_url="data:image/png;base64,PREFIX"),
        Message(role="assistant", content="UI state " * 2000),
        Message(role="user", content="", image_url="data:image/png;base64,TAIL"),
    ]
    manager = MemoryManager(
        keep_recent_messages=1,
        summary_callback=lambda prompt, *_: (
            captured.append(prompt) or ('{"current_work_state":"UI inspected."}' if accepted else "")
        ),
    )
    output, changed = manager.compact(original, force=True)
    assert changed is accepted
    assert captured
    prefix = section(captured[0], "Conversation Prefix")
    assert len(prefix) == 3
    assert prefix[1] == {"role": "user", "content": "[image omitted from summary input: screenshot]"}
    assert "data:image/" not in captured[0]
    assert output[-1] == original[-1]
    if not accepted:
        assert output == original


def test_trailing_partial_tool_block_stays_in_tail() -> None:
    case = next(
        c
        for c in fixture("memory_local")["summary_compaction"]["cases"]
        if c["name"] == "trailing_incomplete_block_stays_in_tail"
    )
    original = current_messages(case["input"]["messages"])
    original.append(Message(role="tool", content="first result", tool_call_id="p1"))
    captured = []
    manager = MemoryManager(
        keep_recent_messages=1, summary_callback=lambda prompt, *_: captured.append(prompt) or case["input"]["summary_response"]
    )
    output, changed = manager.compact(original, force=True)
    assert changed
    assert output[-2:] == original[-2:]
    prefix = section(captured[0], "Conversation Prefix")
    assert prefix == [message.to_dict() for message in original[1:-2]]


@pytest.mark.parametrize(
    "language, expected", [("en-US", "Preserve up to 1 critical events"), ("zh-CN", "最多保留 1 条关键进展")]
)
def test_summary_event_limit_bounds_instruction_not_input(language: str, expected: str) -> None:
    original = [Message(role="system", content="sys")]
    original.extend(Message(role="user", content=f"message {i} " * 200) for i in range(6))
    original.append(Message(role="assistant", content="tail"))
    captured = []
    manager = MemoryManager(
        keep_recent_messages=1,
        language=language,
        summary_event_limit=1,
        summary_callback=lambda prompt, *_: captured.append(prompt) or '{"progress":["one","two","three"]}',
    )
    output, changed = manager.compact(original, force=True)
    assert changed
    assert expected in captured[0]
    assert section(captured[0], "Conversation Prefix") == [m.to_dict() for m in original[1:-1]]
    assert section(output[1].content, "Compressed Agent Memory")["progress"] == ["one", "two", "three"]
