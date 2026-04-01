from __future__ import annotations

from vv_agent.memory.message_sanitizer import sanitize_for_resume
from vv_agent.types import Message


def test_sanitize_for_resume_drops_blank_assistant_messages() -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="   "),
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == [Message(role="user", content="hello")]


def test_sanitize_for_resume_drops_thinking_only_messages() -> None:
    messages = [
        Message(role="assistant", content="", reasoning_content="thinking"),
        Message(role="user", content="continue"),
    ]

    sanitized = sanitize_for_resume(messages)

    assert sanitized == [Message(role="user", content="continue")]


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
