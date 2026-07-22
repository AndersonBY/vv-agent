from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent.memory.message_sanitizer import sanitize_for_resume
from vv_agent.types import Message


def _configured_sub_agent_contract() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assistant_reasoning_contract() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "parity" / "assistant_reasoning_history.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assistant_reasoning_case(name: str) -> dict[str, Any]:
    return next(case for case in _assistant_reasoning_contract()["cases"] if case["name"] == name)


def test_sanitize_for_resume_drops_blank_assistant_messages() -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="   "),
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == [Message(role="user", content="hello")]


def test_sanitize_for_resume_preserves_reasoning_only_messages() -> None:
    case = _assistant_reasoning_case("reasoning_only_assistant_is_preserved")
    assistant = Message.from_dict(case["message"])
    messages = [
        assistant,
        Message(role="user", content="continue"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert case["expected"]["retain_in_resumable_history"] is True
    assert sanitized == messages
    assert assistant.content == case["expected"]["visible_content"]
    assert assistant.reasoning_content == case["expected"]["reasoning_content"]
    assert assistant.to_openai_message() == case["expected"]["openai_compatible_projection"]


def test_sanitize_for_resume_drops_fully_empty_assistant_from_contract() -> None:
    case = _assistant_reasoning_case("fully_empty_assistant_is_removed")

    sanitized = sanitize_for_resume([Message.from_dict(case["message"])])

    assert case["expected"]["retain_in_resumable_history"] is False
    assert sanitized == []


def test_sanitize_for_resume_drops_orphan_tool_results() -> None:
    messages = [
        Message(role="assistant", content="done"),
        Message(role="tool", content="result", tool_call_id="orphan-call"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == [Message(role="assistant", content="done")]


def test_sanitize_for_resume_drops_unresolved_tail_tool_use() -> None:
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}}],
        )
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == []


def test_sanitize_for_resume_trims_only_unresolved_tool_calls() -> None:
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[
                {"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}},
                {"id": "call-2", "name": "write_file", "arguments": {"path": "notes.md"}},
            ],
        ),
        Message(role="tool", content="README", tool_call_id="call-1"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert len(sanitized) == 2
    assert sanitized[0].tool_calls == [{"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}}]


@pytest.mark.parametrize("tool_call_id", [None, "", " \n "])
def test_sanitize_for_resume_drops_tool_results_with_empty_ids(tool_call_id: str | None) -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(role="tool", content="unlinked result", tool_call_id=tool_call_id),
    ]

    assert sanitize_for_resume(messages) == [Message(role="user", content="hello")]


@pytest.mark.parametrize("tool_call_id", [None, "", " \n "])
def test_sanitize_for_resume_drops_unresolved_tool_calls_with_empty_ids(tool_call_id: str | None) -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": tool_call_id, "name": "read_file", "arguments": {"path": "README.md"}}],
        ),
    ]

    assert sanitize_for_resume(messages) == [Message(role="user", content="hello")]


def test_sanitize_for_resume_removes_empty_call_from_mixed_tool_calls() -> None:
    assert _configured_sub_agent_contract()["continuation"]["empty_tool_call_id_policy"] == "drop_incomplete_turn"
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[
                {"id": "", "name": "read_file", "arguments": {"path": "missing.md"}},
                {"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}},
            ],
        ),
        Message(role="tool", content="README", tool_call_id="call-1"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert len(sanitized) == 2
    assert sanitized[0].tool_calls == [{"id": "call-1", "name": "read_file", "arguments": {"path": "README.md"}}]


def test_sanitize_for_resume_matches_tool_call_ids_after_trimming() -> None:
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[{"id": " call-1 ", "name": "read_file", "arguments": {"path": "README.md"}}],
        ),
        Message(role="tool", content="README", tool_call_id="call-1"),
    ]

    assert sanitize_for_resume(messages) == messages


def test_sanitize_for_resume_does_not_reuse_an_earlier_result_for_a_later_call() -> None:
    assert _configured_sub_agent_contract()["continuation"]["tool_result_pairing"] == "immediately_following_assistant_turn"
    completed_assistant = Message(
        role="assistant",
        content="first",
        tool_calls=[{"id": "reused", "name": "read_file", "arguments": {"path": "first.md"}}],
    )
    completed_result = Message(role="tool", content="first result", tool_call_id="reused")
    messages = [
        completed_assistant,
        completed_result,
        Message(role="user", content="next"),
        Message(
            role="assistant",
            content="second",
            tool_calls=[{"id": "reused", "name": "read_file", "arguments": {"path": "second.md"}}],
        ),
        Message(role="user", content="resume"),
    ]

    assert sanitize_for_resume(messages) == [
        completed_assistant,
        completed_result,
        Message(role="user", content="next"),
        Message(role="user", content="resume"),
    ]


def test_sanitize_for_resume_drops_ambiguous_duplicate_ids_and_results() -> None:
    assert _configured_sub_agent_contract()["continuation"]["duplicate_tool_call_id_policy"] == "drop_ambiguous_call_and_results"
    messages = [
        Message(role="user", content="before"),
        Message(
            role="assistant",
            content="ambiguous",
            tool_calls=[
                {"id": "duplicate", "name": "read_file", "arguments": {"path": "a.md"}},
                {"id": "duplicate", "name": "read_file", "arguments": {"path": "b.md"}},
            ],
        ),
        Message(role="tool", content="which call?", tool_call_id="duplicate"),
        Message(role="user", content="after"),
    ]

    assert sanitize_for_resume(messages) == [
        Message(role="user", content="before"),
        Message(role="user", content="after"),
    ]


def test_sanitize_for_resume_drops_out_of_order_results_and_unresolved_calls() -> None:
    assert _configured_sub_agent_contract()["continuation"]["out_of_order_tool_result_policy"] == "drop_orphan_result"
    messages = [
        Message(role="tool", content="too early", tool_call_id="late"),
        Message(role="user", content="boundary"),
        Message(
            role="assistant",
            content="late call",
            tool_calls=[{"id": "late", "name": "read_file", "arguments": {"path": "late.md"}}],
        ),
        Message(role="user", content="resume"),
    ]

    assert sanitize_for_resume(messages) == [
        Message(role="user", content="boundary"),
        Message(role="user", content="resume"),
    ]


def test_sanitize_for_resume_requires_results_in_tool_call_order() -> None:
    assert _configured_sub_agent_contract()["continuation"]["tool_result_order"] == "same_as_tool_calls"
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[
                {"id": "call-a", "name": "read_file", "arguments": {"path": "a.md"}},
                {"id": "call-b", "name": "read_file", "arguments": {"path": "b.md"}},
            ],
        ),
        Message(role="tool", content="B", tool_call_id="call-b"),
        Message(role="tool", content="A", tool_call_id="call-a"),
        Message(role="user", content="resume"),
    ]

    assert sanitize_for_resume(messages) == [Message(role="user", content="resume")]


def test_sanitize_for_resume_keeps_only_ordered_result_prefix() -> None:
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[
                {"id": "call-a", "name": "read_file", "arguments": {"path": "a.md"}},
                {"id": "call-b", "name": "read_file", "arguments": {"path": "b.md"}},
                {"id": "call-c", "name": "read_file", "arguments": {"path": "c.md"}},
            ],
        ),
        Message(role="tool", content="A", tool_call_id="call-a"),
        Message(role="tool", content="C", tool_call_id="call-c"),
        Message(role="tool", content="B", tool_call_id="call-b"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert len(sanitized) == 2
    assert sanitized[0].tool_calls == [
        {"id": "call-a", "name": "read_file", "arguments": {"path": "a.md"}},
    ]
    assert sanitized[1].tool_call_id == "call-a"


@pytest.mark.parametrize(
    ("call_ids", "result_ids"),
    [
        (["first", "tail"], ["first", "first", "tail"]),
        (["duplicate", "duplicate", "tail"], ["duplicate", "tail"]),
    ],
)
def test_sanitize_for_resume_does_not_skip_ambiguity_to_keep_a_later_pair(
    call_ids: list[str],
    result_ids: list[str],
) -> None:
    assert _configured_sub_agent_contract()["continuation"]["mismatched_tool_result_policy"] == "retain_ordered_prefix"
    messages = [
        Message(
            role="assistant",
            content="Working",
            tool_calls=[{"id": call_id, "name": "read_file", "arguments": {"path": f"{call_id}.md"}} for call_id in call_ids],
        ),
        *[Message(role="tool", content=f"result for {result_id}", tool_call_id=result_id) for result_id in result_ids],
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == []
    assert all(message.role != "tool" for message in sanitized)
