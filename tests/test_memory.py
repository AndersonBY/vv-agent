from __future__ import annotations

from pathlib import Path

from v_agent.memory import MemoryManager
from v_agent.types import Message


def test_memory_compacts_when_threshold_exceeded() -> None:
    manager = MemoryManager(threshold_chars=60, keep_recent_messages=3)
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
    assert all(not (msg.role == "tool" and msg.tool_call_id == "call1") for msg in compacted)
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
    assert len(summaries) <= 1


def test_memory_compaction_keeps_tool_boundary_consistent() -> None:
    manager = MemoryManager(threshold_chars=10, keep_recent_messages=2)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 30),
        Message(
            role="assistant",
            content="plan tools",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "_read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="tool result 1", tool_call_id="call_1"),
        Message(role="tool", content="tool result 2", tool_call_id="call_1"),
        Message(role="assistant", content="next step"),
        Message(role="user", content="continue"),
        Message(role="assistant", content="done"),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is True
    summary_index = next((idx for idx, msg in enumerate(compacted) if msg.name == "memory_summary"), None)
    assert summary_index is not None
    next_index = summary_index + 1
    if next_index < len(compacted):
        assert compacted[next_index].role != "tool"


def test_memory_compacts_large_tool_result_to_workspace_artifact(tmp_path: Path) -> None:
    manager = MemoryManager(
        threshold_chars=80,
        keep_recent_messages=2,
        workspace=tmp_path,
        tool_result_compact_threshold=30,
        tool_result_keep_last=0,
    )
    large_tool_result = "x" * 200
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="read file"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "_read_file", "arguments": "{}"}}],
        ),
        Message(role="tool", content=large_tool_result, tool_call_id="call_1"),
        Message(role="assistant", content="continue"),
    ]

    compacted, changed = manager.compact(messages, cycle_index=3)
    assert changed is True
    tool_messages = [msg for msg in compacted if msg.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content.startswith("<Tool Result Compact>")
    artifact_file = tmp_path / ".memory" / "tool_results" / "cycle_3" / "call_1.txt"
    assert artifact_file.exists()
    assert artifact_file.read_text(encoding="utf-8") == large_tool_result


def test_memory_compacts_processed_image_payload() -> None:
    manager = MemoryManager(threshold_chars=80, keep_recent_messages=2)
    image_payload = "data:image/png;base64," + ("a" * 400)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="original request"),
        Message(role="user", content="[Image loaded] img.png", image_url=image_payload),
        Message(role="assistant", content="image parsed"),
        Message(role="assistant", content="next"),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is True
    image_messages = [msg for msg in compacted if msg.content.startswith("[Image loaded]")]
    assert image_messages
    assert image_messages[0].image_url is None
    assert "image payload compacted" in image_messages[0].content
