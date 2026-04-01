from __future__ import annotations

from vv_agent.prompt import CacheBreakTracker, hash_system_prompt_sections, hash_tool_payload


def test_hash_system_prompt_sections_ignores_invalid_entries() -> None:
    hash_value = hash_system_prompt_sections(
        [
            {"id": "core", "text": "stable", "stable": True},
            {"id": "blank", "text": "   ", "stable": False},
            "invalid",
        ]
    )

    assert hash_value


def test_hash_tool_payload_is_stable_for_sorted_content() -> None:
    left = hash_tool_payload([{"name": "read_file", "input_schema": {"type": "object", "a": 1}}])
    right = hash_tool_payload([{"input_schema": {"a": 1, "type": "object"}, "name": "read_file"}])

    assert left == right


def test_cache_break_tracker_detects_system_and_tool_changes() -> None:
    tracker = CacheBreakTracker()

    assert tracker.check(system_hash="system-1", tool_hash="tool-1") == []
    assert tracker.check(system_hash="system-1", tool_hash="tool-1") == []
    assert tracker.check(system_hash="system-2", tool_hash="tool-1") == ["system_prompt_changed"]
    assert tracker.check(system_hash="system-2", tool_hash="tool-2") == ["tool_schemas_changed"]
    assert tracker.total_requests == 4
    assert tracker.cache_breaks == 2
    assert tracker.break_reasons == ["system_prompt_changed", "tool_schemas_changed"]
    assert tracker.cache_hit_rate == 0.5
