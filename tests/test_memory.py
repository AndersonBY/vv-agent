from __future__ import annotations

from v_agent.memory import MemoryManager
from v_agent.types import Message


def test_memory_compacts_when_threshold_exceeded() -> None:
    manager = MemoryManager(threshold_chars=120, keep_recent_messages=4)
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="u" * 40),
        Message(role="assistant", content="a" * 40),
        Message(role="tool", content="t" * 40, tool_call_id="call1"),
        Message(role="assistant", content="b" * 40),
        Message(role="user", content="c" * 40),
        Message(role="assistant", content="d" * 40),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is True
    assert compacted[1].name == "memory_summary"
    assert compacted[-1].content == "d" * 40


def test_memory_does_not_compact_when_small() -> None:
    manager = MemoryManager(threshold_chars=1_000, keep_recent_messages=4)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is False
    assert compacted == messages


def test_memory_replaces_previous_summary() -> None:
    manager = MemoryManager(threshold_chars=10, keep_recent_messages=2)
    messages = [
        Message(role="system", content="sys"),
        Message(role="system", name="memory_summary", content="old summary"),
        Message(role="user", content="x" * 20),
        Message(role="assistant", content="y" * 20),
        Message(role="user", content="z" * 20),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is True
    summaries = [msg for msg in compacted if msg.name == "memory_summary"]
    assert len(summaries) == 1
