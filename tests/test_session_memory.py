from __future__ import annotations

import json

from vv_agent.memory import SessionMemory, SessionMemoryConfig, SessionMemoryEntry
from vv_agent.types import Message


def test_session_memory_should_extract_respects_init_and_growth_thresholds() -> None:
    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=100,
            min_text_messages=3,
            extraction_callback=lambda _prompt, _backend, _model: "[]",
        )
    )

    assert memory.should_extract(99, 3) is False
    assert memory.should_extract(100, 2) is False
    assert memory.should_extract(100, 3) is True

    memory.state.initialized = True
    memory.state.tokens_at_last_extraction = 120

    assert memory.should_extract(169, 4) is False
    assert memory.should_extract(170, 4) is True


def test_session_memory_should_extract_handles_negative_growth() -> None:
    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=100,
            min_text_messages=1,
            extraction_callback=lambda _prompt, _backend, _model: "[]",
        )
    )
    memory.state.initialized = True
    memory.state.tokens_at_last_extraction = 500

    assert memory.should_extract(40, 2) is False
    assert memory.should_extract(120, 2) is True


def test_session_memory_extracts_only_new_messages() -> None:
    prompts: list[str] = []

    def extraction_callback(prompt: str, _backend: str | None, _model: str | None) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            payload = [{"category": "decision", "content": "keep tests green", "importance": 8}]
        else:
            payload = [{"category": "file_change", "content": "updated manager.py", "importance": 7}]
        return json.dumps(payload, ensure_ascii=False)

    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=50,
            min_text_messages=1,
            extraction_callback=extraction_callback,
        )
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="alpha"),
        Message(role="assistant", content="beta"),
    ]

    merged_one = memory.extract(messages, current_cycle=4, current_tokens=80)
    assert merged_one == 1
    assert len(memory.state.entries) == 1
    assert "alpha" in prompts[0]
    assert "beta" in prompts[0]

    updated_messages = [*messages, Message(role="user", content="gamma")]
    merged_two = memory.extract(updated_messages, current_cycle=5, current_tokens=140)

    assert merged_two == 1
    assert len(memory.state.entries) == 2
    assert "gamma" in prompts[1]
    assert "alpha" not in prompts[1]


def test_session_memory_skips_compacted_summary_messages() -> None:
    prompts: list[str] = []

    def extraction_callback(prompt: str, _backend: str | None, _model: str | None) -> str:
        prompts.append(prompt)
        return json.dumps(
            [{"category": "key_fact", "content": "new follow-up", "importance": 6}],
            ensure_ascii=False,
        )

    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=50,
            min_text_messages=1,
            extraction_callback=extraction_callback,
        )
    )
    messages = [
        Message(role="system", content="sys"),
        Message(
            role="user",
            content="<Original User Request>\nstart\n</Original User Request>\n\n"
            "<Compressed Agent Memory>\nsummary\n</Compressed Agent Memory>",
        ),
        Message(role="assistant", content="new answer"),
    ]

    merged = memory.extract(messages, current_cycle=7, current_tokens=120)

    assert merged == 1
    assert prompts
    assert "<Compressed Agent Memory>" not in prompts[0]
    assert "new answer" in prompts[0]


def test_session_memory_extract_handles_callback_exception() -> None:
    def extraction_callback(_prompt: str, _backend: str | None, _model: str | None) -> str:
        raise RuntimeError("boom")

    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=50,
            min_text_messages=1,
            extraction_callback=extraction_callback,
        )
    )

    merged = memory.extract(
        [Message(role="system", content="sys"), Message(role="user", content="alpha")],
        current_cycle=1,
        current_tokens=80,
    )

    assert merged == 0
    assert memory.state.entries == []
    assert memory.state.initialized is False


def test_session_memory_render_groups_entries_by_category() -> None:
    memory = SessionMemory(SessionMemoryConfig())
    memory.state.entries = [
        SessionMemoryEntry(category="decision", content="use session memory", source_cycle=2, importance=8),
        SessionMemoryEntry(category="key_fact", content="workspace is tmp", source_cycle=3, importance=6),
    ]

    rendered = memory.render_as_system_context()

    assert rendered.startswith("<Session Memory>")
    assert "## decision" in rendered
    assert "- use session memory" in rendered
    assert "## key_fact" in rendered
    assert rendered.endswith("</Session Memory>")


def test_session_memory_prunes_low_importance_entries_first() -> None:
    memory = SessionMemory(SessionMemoryConfig(max_tokens=80, token_model="gpt-5.4"))
    memory.state.entries = [
        SessionMemoryEntry(category="key_fact", content="a" * 180, source_cycle=1, importance=9),
        SessionMemoryEntry(category="key_fact", content="b" * 180, source_cycle=2, importance=2),
        SessionMemoryEntry(category="key_fact", content="c" * 180, source_cycle=3, importance=5),
    ]

    memory._prune_to_budget()

    remaining_contents = {entry.content for entry in memory.state.entries}
    assert "a" * 180 in remaining_contents
    assert "b" * 180 not in remaining_contents


def test_session_memory_can_persist_load_and_reset_on_compaction(tmp_path) -> None:
    memory = SessionMemory(
        SessionMemoryConfig(storage_dir=".memory/session"),
        workspace=tmp_path,
        storage_scope="task-a",
    )
    memory.state.entries = [
        SessionMemoryEntry(category="user_intent", content="finish phase 4", source_cycle=9, importance=10)
    ]
    memory.state.last_extracted_message_index = 12
    memory.state.tokens_at_last_extraction = 320
    memory.state.initialized = True
    memory._save()

    loaded = SessionMemory(
        SessionMemoryConfig(storage_dir=".memory/session"),
        workspace=tmp_path,
        storage_scope="task-a",
    )
    loaded.load()

    assert len(loaded.state.entries) == 1
    assert loaded.state.entries[0].content == "finish phase 4"
    assert loaded.state.last_extracted_message_index == 12

    loaded.on_compaction(current_tokens=33)

    assert loaded.state.last_extracted_message_index == -1
    assert loaded.state.tokens_at_last_extraction == 33
    assert loaded.state.entries[0].content == "finish phase 4"


def test_session_memory_storage_scope_isolates_new_tasks(tmp_path) -> None:
    scoped = SessionMemory(
        SessionMemoryConfig(storage_dir=".memory/session"),
        workspace=tmp_path,
        storage_scope="session-one",
    )
    scoped.state.entries = [
        SessionMemoryEntry(category="key_fact", content="first session fact", source_cycle=2, importance=8)
    ]
    scoped._save()

    isolated = SessionMemory(
        SessionMemoryConfig(storage_dir=".memory/session"),
        workspace=tmp_path,
        storage_scope="session-two",
    )
    isolated.load()

    assert isolated.state.entries == []


def test_session_memory_rejects_storage_dir_path_traversal(tmp_path) -> None:
    memory = SessionMemory(SessionMemoryConfig(storage_dir="../../outside"), workspace=tmp_path)

    assert memory._storage_path() is None


def test_session_memory_parse_handles_non_array_and_greedy_noise() -> None:
    memory = SessionMemory(SessionMemoryConfig())

    assert memory._parse_extraction_result('{"category":"key_fact"}', cycle=1) == []

    parsed = memory._parse_extraction_result(
        'prefix [{"category":"key_fact","content":"ok","importance":6}] trailing ] noise',
        cycle=2,
    )

    assert len(parsed) == 1
    assert parsed[0].content == "ok"
