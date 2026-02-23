from __future__ import annotations

import json
from pathlib import Path

from vv_agent.memory import MemoryManager
from vv_agent.types import Message


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


def test_memory_compacts_when_threshold_exceeded_to_summary_block() -> None:
    manager = MemoryManager(compact_threshold=60, keep_recent_messages=3, summary_callback=_fake_summary)
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
    assert len(compacted) == 2
    assert compacted[0].role == "system"
    assert compacted[1].role == "user"
    assert "<Compressed Agent Memory>" in compacted[1].content


def test_memory_does_not_compact_when_small() -> None:
    manager = MemoryManager(compact_threshold=1_000, keep_recent_messages=4)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is False
    assert compacted == messages


def test_memory_replaces_previous_summary_with_compressed_block() -> None:
    manager = MemoryManager(compact_threshold=10, keep_recent_messages=2, summary_callback=_fake_summary)
    messages = [
        Message(role="system", content="sys"),
        Message(role="system", name="memory_summary", content="old summary"),
        Message(role="user", content="x" * 20),
        Message(role="assistant", content="y" * 20),
        Message(role="user", content="z" * 20),
    ]

    compacted, changed = manager.compact(messages)
    assert changed is True
    assert len(compacted) == 2
    assert all(msg.name != "memory_summary" for msg in compacted)
    assert "<Compressed Agent Memory>" in compacted[1].content


def test_memory_compaction_keeps_tool_boundary_consistent() -> None:
    manager = MemoryManager(compact_threshold=10, keep_recent_messages=2, summary_callback=_fake_summary)
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
                    "function": {"name": "read_file", "arguments": "{}"},
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
    assert len(compacted) == 2
    assert all(msg.role != "tool" for msg in compacted)


def test_memory_compacts_large_tool_result_to_workspace_artifact(tmp_path: Path) -> None:
    manager = MemoryManager(
        compact_threshold=80,
        keep_recent_messages=2,
        workspace=tmp_path,
        tool_result_compact_threshold=30,
        tool_result_keep_last=0,
        summary_callback=_fake_summary,
    )
    large_tool_result = "x" * 200
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="read file"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        ),
        Message(role="tool", content=large_tool_result, tool_call_id="call_1"),
        Message(role="assistant", content="continue"),
    ]

    compacted, changed = manager.compact(messages, cycle_index=3)
    assert changed is True
    assert len(compacted) == 2
    artifact_file = tmp_path / ".memory" / "tool_results" / "cycle_3" / "call_1.txt"
    assert artifact_file.exists()
    assert artifact_file.read_text(encoding="utf-8") == large_tool_result
    assert "<Persisted Artifacts>" in compacted[1].content
    assert "call_1.txt" in compacted[1].content
    assert "tool: read_file" in compacted[1].content


def test_memory_compacts_processed_image_payload() -> None:
    manager = MemoryManager(compact_threshold=80, keep_recent_messages=2)
    image_payload = "data:image/png;base64," + ("a" * 400)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="original request"),
        Message(role="user", content="[Image loaded] img.png", image_url=image_payload),
        Message(role="assistant", content="image parsed"),
        Message(role="assistant", content="next"),
    ]

    compacted, changed = manager._compact_processed_image_messages(messages)
    assert changed is True
    image_messages = [msg for msg in compacted if msg.content.startswith("[Image loaded]")]
    assert image_messages
    assert image_messages[0].image_url is None
    assert "image payload compacted" in image_messages[0].content


def test_memory_uses_token_based_length_with_recent_tool_ids() -> None:
    manager = MemoryManager(compact_threshold=120)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="plan",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="x" * 200, tool_call_id="call_1"),
    ]

    compacted, changed = manager.compact(messages, total_tokens=20, recent_tool_call_ids=set())
    assert changed is False
    assert compacted == messages

    compacted2, changed2 = manager.compact(messages, total_tokens=200, recent_tool_call_ids=set())
    assert changed2 is True
    assert len(compacted2) == 2
