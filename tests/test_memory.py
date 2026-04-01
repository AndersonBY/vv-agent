from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vv_agent.memory import MemoryManager, SessionMemory, SessionMemoryConfig
from vv_agent.memory.microcompact import CLEARED_MARKER
from vv_agent.types import Message


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


def test_memory_compacts_when_threshold_exceeded_to_summary_block() -> None:
    manager = _build_manager(
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=3,
        summary_callback=_fake_summary,
    )
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


def test_memory_replaces_previous_summary_with_compressed_block() -> None:
    manager = _build_manager(
        model_context_window=40,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=2,
        summary_callback=_fake_summary,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="system", name="memory_summary", content="old summary"),
        Message(role="user", content="x" * 20),
        Message(role="assistant", content="y" * 20),
        Message(role="user", content="z" * 20),
    ]

    compacted, changed = manager.compact(messages, force=True)
    assert changed is True
    assert len(compacted) == 2
    assert all(msg.name != "memory_summary" for msg in compacted)
    assert "<Compressed Agent Memory>" in compacted[1].content


def test_memory_compaction_keeps_tool_boundary_consistent() -> None:
    manager = _build_manager(keep_recent_messages=2, summary_callback=_fake_summary)
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

    compacted, changed = manager.compact(messages, force=True)
    assert changed is True
    assert len(compacted) == 2
    assert all(msg.role != "tool" for msg in compacted)


def test_memory_compacts_large_tool_result_to_workspace_artifact(tmp_path: Path) -> None:
    manager = _build_manager(
        model_context_window=50,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
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
    manager = _build_manager(keep_recent_messages=2)
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
    manager = _build_manager(
        model_context_window=120,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        tool_result_compact_threshold=20,
        tool_result_keep_last=0,
    )
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
        Message(role="tool", content="x" * 400, tool_call_id="call_1"),
    ]

    compacted, changed = manager.compact(messages, total_tokens=100, recent_tool_call_ids=set())
    assert changed is False
    assert compacted == messages

    compacted2, changed2 = manager.compact(messages, total_tokens=100, recent_tool_call_ids={"call_1"})
    assert changed2 is True
    assert len(compacted2) == 2


def test_memory_thresholds_respect_configured_ceiling() -> None:
    manager = MemoryManager(
        model_context_window=200_000,
        reserved_output_tokens=16_000,
        autocompact_buffer_tokens=13_000,
        warning_threshold_percentage=90,
    )

    assert manager.effective_context_window == 184_000
    assert manager.autocompact_threshold == 128_000
    assert manager.warning_threshold == 115_200


def test_memory_thresholds_fall_back_to_model_limit_when_smaller() -> None:
    manager = MemoryManager(
        compact_threshold=128_000,
        model_context_window=64_000,
        reserved_output_tokens=8_000,
        autocompact_buffer_tokens=13_000,
    )

    assert manager.autocompact_threshold == 43_000


def test_memory_recomputes_compacted_length_without_stale_total_tokens() -> None:
    manager = _build_manager(
        model_context_window=160,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        tool_result_compact_threshold=20,
        tool_result_keep_last=0,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="read file"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        ),
        Message(role="tool", content="x" * 400, tool_call_id="call_1"),
        Message(role="assistant", content="continue"),
    ]

    compacted, changed = manager.compact(messages, total_tokens=500, recent_tool_call_ids=set())

    assert changed is True
    assert len(compacted) > 2
    assert all("<Compressed Agent Memory>" not in message.content for message in compacted)


def test_memory_compaction_preserves_session_memory_and_excludes_it_from_summary_prompt() -> None:
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
        model_context_window=70,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=2,
        summary_callback=tracking_summary,
        base_system_prompt="sys",
        session_memory=session_memory,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 40),
        Message(role="assistant", content="a" * 40),
        Message(role="user", content="c" * 40),
    ]

    compacted, changed = manager.compact(messages, cycle_index=2, total_tokens=150)

    assert changed is True
    assert len(compacted) == 2
    assert session_memory.state.entries
    assert session_memory.state.last_extracted_message_index == -1
    assert observed_prompts
    assert "<Session Memory>" not in observed_prompts[0]

    request_messages = manager.apply_session_memory_context(compacted)
    assert "<Session Memory>" in request_messages[0].content
    assert "preserve prior decisions" in request_messages[0].content


def test_memory_compaction_strips_analysis_and_restores_key_files(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.py"
    file_path.write_text("print('restored')\n", encoding="utf-8")

    def summary_with_analysis(_prompt: str, _backend: str | None, _model: str | None) -> str:
        return (
            "<analysis>drafting scratchpad</analysis>\n"
            '{'
            '"summary_version":"2.0",'
            '"original_user_messages":["please update demo.py"],'
            '"user_constraints":["keep behavior"],'
            '"decisions":["edit demo.py"],'
            '"files_examined_or_modified":[{"path":"demo.py","action":"modified","summary":"updated demo"}],'
            '"errors_and_fixes":[],'
            '"progress":["edited file"],'
            '"key_facts":[],'
            '"open_issues":[],'
            '"current_work_state":"waiting for verification",'
            '"next_steps":["run tests"]'
            '}'
        )

    manager = _build_manager(
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=2,
        workspace=tmp_path,
        summary_callback=summary_with_analysis,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="please update demo.py"),
        Message(role="assistant", content="reviewing"),
        Message(role="user", content="keep behavior"),
        Message(role="assistant", content="editing"),
    ]

    compacted, changed = manager.compact(messages, force=True)

    assert changed is True
    assert "<analysis>" not in compacted[1].content
    assert '"original_user_messages":["please update demo.py"]' in compacted[1].content
    assert "<Post-Compaction File Context>" in compacted[1].content
    assert 'path="demo.py"' in compacted[1].content
    assert "print('restored')" in compacted[1].content


def test_memory_second_compaction_preserves_original_user_messages() -> None:
    manager = _build_manager(
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        keep_recent_messages=2,
        summary_callback=None,
    )
    first_messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="please preserve this exact request"),
        Message(role="assistant", content="working"),
    ]

    first_compacted, first_changed = manager.compact(first_messages, force=True)
    assert first_changed is True

    second_messages = [
        first_compacted[0],
        first_compacted[1],
        Message(role="assistant", content="made progress"),
        Message(role="user", content="and keep this follow-up too"),
    ]

    second_compacted, second_changed = manager.compact(second_messages, force=True)

    assert second_changed is True
    assert '"original_user_messages": ["please preserve this exact request", "and keep this follow-up too"]' in (
        second_compacted[1].content
    )


def test_memory_force_compact_bypasses_threshold() -> None:
    manager = _build_manager(
        model_context_window=400,
        reserved_output_tokens=50,
        autocompact_buffer_tokens=50,
        summary_callback=_fake_summary,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
        Message(role="user", content="continue"),
    ]

    compacted, changed = manager.compact(messages, force=True)

    assert changed is True
    assert len(compacted) == 2
    assert "<Compressed Agent Memory>" in compacted[1].content


def test_memory_emergency_compact_preserves_recent_tool_context() -> None:
    manager = _build_manager(keep_recent_messages=2)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="old request"),
        Message(
            role="assistant",
            content="call tool",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="tool result", tool_call_id="call_1"),
        Message(role="assistant", content="recent analysis"),
        Message(role="user", content="latest ask"),
    ]

    compacted = manager.emergency_compact(messages, drop_ratio=0.5)

    assert compacted[0].role == "system"
    assert all(message.content != "old request" for message in compacted)
    assert any(message.role == "assistant" and message.tool_calls for message in compacted)
    assert any(message.role == "tool" and message.tool_call_id == "call_1" for message in compacted)


def test_memory_compact_uses_microcompact_before_full_summary() -> None:
    manager = _build_manager(
        model_context_window=150,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        microcompact_keep_recent_cycles=1,
        microcompact_min_result_length=200,
        tool_result_compact_threshold=2_000,
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
        Message(role="tool", content="x" * 600, tool_call_id="call_old"),
        Message(role="assistant", content="recent reply"),
        Message(role="user", content="latest ask"),
    ]

    compacted, changed = manager.compact(messages, cycle_index=3)

    assert changed is True
    assert any(message.role == "tool" and message.content == CLEARED_MARKER for message in compacted)
    assert all("<Compressed Agent Memory>" not in message.content for message in compacted)


def test_memory_normalize_orphan_tool_messages_respects_message_order_with_reused_call_id() -> None:
    manager = MemoryManager(
        tool_calls_keep_last=1,
        assistant_no_tool_keep_last=10,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(
            role="assistant",
            content="first capture request",
            tool_calls=[
                {
                    "id": "screen_capture:4",
                    "type": "function",
                    "function": {"name": "screen_capture", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="first result", tool_call_id="screen_capture:4"),
        Message(role="assistant", content="narration"),
        Message(
            role="assistant",
            content="second capture request",
            tool_calls=[
                {
                    "id": "screen_capture:4",
                    "type": "function",
                    "function": {"name": "screen_capture", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="second result", tool_call_id="screen_capture:4"),
    ]

    compacted, changed = manager._compact_messages(messages, cycle_index=1)
    assert changed is True

    assistant_tool_call_indices = [idx for idx, msg in enumerate(compacted) if msg.role == "assistant" and msg.tool_calls]
    tool_indices = [
        idx
        for idx, msg in enumerate(compacted)
        if msg.role == "tool" and msg.tool_call_id == "screen_capture:4"
    ]
    assert len(assistant_tool_call_indices) == 1
    assert len(tool_indices) == 1
    assert tool_indices[0] > assistant_tool_call_indices[0]
    assert compacted[tool_indices[0]].content == "second result"


def test_memory_normalize_orphan_tool_messages_drops_excess_tool_results_per_call_id() -> None:
    manager = MemoryManager()
    messages = [
        Message(
            role="assistant",
            content="call",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="first", tool_call_id="call_1"),
        Message(role="tool", content="second", tool_call_id="call_1"),
    ]

    normalized, changed = manager._normalize_orphan_tool_messages(messages)
    assert changed is True
    tool_messages = [msg for msg in normalized if msg.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "first"
