from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vv_agent.memory.manager import MemoryManager
from vv_agent.memory.microcompact import (
    COMPACT_MARKER_CLOSING,
    COMPACT_MARKER_OPENING,
    build_compacted_tool_content,
    replace_with_compacted_marker,
)
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.types import Message, ToolArtifactRef
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import persist_text_artifact


def _tool_turn(call_id: str, tool_name: str, content: str) -> list[Message]:
    return [
        Message(
            role="assistant",
            content=f"call {tool_name}",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        ),
        Message(
            role="tool",
            content=content,
            tool_call_id=call_id,
        ),
    ]


def _messages() -> list[Message]:
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="start"),
        *_tool_turn("call_first", "custom_search", "a" * 8_000),
        *_tool_turn("call_second", "custom_fetch", "b" * 4_000),
        Message(role="assistant", content="recent response"),
    ]


def _manager(
    backend: MemoryWorkspaceBackend | None,
    *,
    result_retentions: dict[str, ToolResultRetention] | None = None,
) -> MemoryManager:
    return MemoryManager(
        compact_threshold=1_000,
        model="unknown-provider-model",
        model_context_window=1_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompaction_policy=MicrocompactionPolicy(
            trigger_ratio=0.75,
            target_ratio=0.60,
            keep_recent_cycles=1,
            min_result_chars=500,
        ),
        tool_result_retentions=dict(result_retentions or {}),
        workspace_backend=backend,
        recovery_tool_available=True,
        artifact_scope="run-7",
    )


def test_microcompact_archives_custom_tool_oldest_first_until_target() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    messages = _messages()

    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1_000)
    assert plan is not None
    assert [candidate.tool_name for candidate in plan.candidates] == ["custom_search", "custom_fetch"]

    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 1
    assert result.artifact_failure_count == 0
    assert result.reclaimed_tokens >= 400
    first = result.messages[3]
    second = result.messages[5]
    assert first.content.startswith(COMPACT_MARKER_OPENING)
    assert first.content.endswith(COMPACT_MARKER_CLOSING)
    assert second.content == "b" * 4_000
    assert result.reclaimed_tokens == max(
        manager._estimate_message_tokens(messages[3]) - manager._estimate_message_tokens(first),
        0,
    )
    assert first.artifact_ref is not None
    artifact_path = first.artifact_ref.path
    assert artifact_path.startswith(".vv-agent/artifacts/run-7/")
    assert backend.read_text(artifact_path) == "a" * 8_000

    marker_fields = {line.split(":", 1)[0] for line in first.content.splitlines()[1:5] if ":" in line}
    assert marker_fields == {"tool_name", "artifact_path", "retrieval_hint", "excerpt"}
    for forbidden in (
        "original_bytes",
        "visible_bytes",
        "size_bytes",
        "sha256",
        "total_chars",
        "truncated_chars",
    ):
        assert f"{forbidden}:" not in first.content


def test_microcompact_uses_actual_replacement_tokens_to_reach_target() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    messages = _messages()
    artifacts = [
        persist_text_artifact(backend, "run-7", "existing-first", messages[3].content),
        persist_text_artifact(backend, "run-7", "existing-second", messages[5].content),
    ]
    messages[3] = replace(messages[3], artifact_ref=artifacts[0])
    messages[5] = replace(messages[5], artifact_ref=artifacts[1])

    current_tokens = 10_000
    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=current_tokens)
    assert plan is not None
    assert len(plan.candidates) == 2
    first = plan.candidates[0]
    first_marker = build_compacted_tool_content(
        messages[first.message_index].content,
        artifact_path=artifacts[0].path,
        tool_name=first.tool_name,
    )
    first_replacement = replace_with_compacted_marker(
        messages[first.message_index],
        first,
        artifact=artifacts[0],
        marker=first_marker,
    )
    first_actual_reclaim = max(
        manager._estimate_message_tokens(messages[first.message_index]) - manager._estimate_message_tokens(first_replacement),
        0,
    )
    target_tokens = current_tokens - first_actual_reclaim - 1
    misleading_first = replace(
        first,
        estimated_reclaimable_tokens=current_tokens - target_tokens,
    )
    plan = replace(
        plan,
        candidates=(misleading_first, *plan.candidates[1:]),
        target_tokens=target_tokens,
    )

    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 2
    assert result.reclaimed_tokens > first_actual_reclaim


def test_microcompact_skips_candidate_when_marker_would_not_reduce_tokens() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    messages = [
        Message(role="system", content="sys"),
        *_tool_turn("call", "custom_search", " " * 501),
        Message(role="assistant", content="recent"),
    ]

    plan = manager.plan_microcompaction(messages, cycle_index=3, current_tokens=1_000)

    assert plan is not None
    assert plan.candidates == ()
    assert backend.list_files(".vv-agent/artifacts", "**/*") == []


def test_microcompact_uses_history_age_when_new_run_starts_from_session_messages() -> None:
    manager = _manager(MemoryWorkspaceBackend())
    manager.microcompaction_policy = MicrocompactionPolicy(
        trigger_ratio=0.75,
        target_ratio=0.60,
        keep_recent_cycles=3,
        min_result_chars=500,
    )
    messages = [Message(role="system", content="sys"), Message(role="user", content="start")]
    for index in range(1, 6):
        messages.extend(_tool_turn(f"call_{index}", "custom_search", str(index) * 2_000))

    plan = manager.plan_microcompaction(messages, cycle_index=1, current_tokens=1_000)

    assert plan is not None
    assert [candidate.tool_call_id for candidate in plan.candidates] == ["call_1", "call_2"]


def test_microcompact_preserve_retention_excludes_only_that_candidate() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(
        backend,
        result_retentions={"custom_search": ToolResultRetention.PRESERVE},
    )
    messages = _messages()

    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1_000)

    assert plan is not None
    assert [candidate.tool_name for candidate in plan.candidates] == ["custom_fetch"]
    result = manager.apply_microcompaction(messages, plan=plan)
    assert result.messages[3].content == "a" * 8_000
    assert result.messages[5].content.startswith(COMPACT_MARKER_OPENING)


class FailingArtifactBackend(MemoryWorkspaceBackend):
    def write_text_exclusive(self, path: str, content: str) -> int:
        del path, content
        raise OSError("storage unavailable")


def test_microcompact_archive_failure_preserves_original_text() -> None:
    manager = _manager(FailingArtifactBackend())
    messages = _messages()
    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1_000)
    assert plan is not None

    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 0
    assert result.artifact_failure_count == 2
    assert result.messages == messages


def test_microcompact_reuses_existing_artifact() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    messages = _messages()
    artifact = persist_text_artifact(backend, "run-7", "existing", "complete prior output")
    messages[3] = replace(
        messages[3],
        artifact_ref=artifact,
    )

    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1_000)
    assert plan is not None
    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 1
    assert result.messages[3].artifact_ref == artifact
    assert backend.read_text(artifact.path) == "complete prior output"


@pytest.mark.parametrize("invalid_field", ["missing", "size", "sha256", "utf8"])
def test_microcompact_invalid_existing_artifact_preserves_original_without_rearchiving(
    invalid_field: str,
) -> None:
    class InvalidReadBackend(MemoryWorkspaceBackend):
        def __init__(self) -> None:
            super().__init__()
            self.exclusive_writes = 0

        def read_bytes(self, path: str) -> bytes:
            if invalid_field == "utf8":
                return b"\xff"
            return super().read_bytes(path)

        def write_text_exclusive(self, path: str, content: str) -> int:
            self.exclusive_writes += 1
            return super().write_text_exclusive(path, content)

    backend = InvalidReadBackend()
    manager = _manager(backend)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="start"),
        *_tool_turn("call_first", "custom_search", "a" * 8_000),
        Message(role="assistant", content="recent response"),
    ]
    artifact = persist_text_artifact(backend, "run-7", "existing", "complete prior output")
    if invalid_field == "missing":
        artifact = ToolArtifactRef(
            path=".vv-agent/artifacts/run-7/missing.txt",
            media_type="text/plain",
            encoding="utf-8",
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
    elif invalid_field == "size":
        artifact = replace(artifact, size_bytes=artifact.size_bytes + 1)
    elif invalid_field == "sha256":
        artifact = replace(artifact, sha256="0" * 64)
    messages[3] = replace(messages[3], artifact_ref=artifact)
    writes_before_compaction = backend.exclusive_writes

    plan = manager.plan_microcompaction(messages, cycle_index=3, current_tokens=1_000)
    assert plan is not None
    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 0
    assert result.artifact_failure_count == 1
    assert result.messages[3] == messages[3]
    assert backend.exclusive_writes == writes_before_compaction


def test_microcompact_recovery_envelope_without_typed_artifact_is_not_archived() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    recovery = json.dumps(
        {
            "vv_agent_recovery": {
                "truncated": True,
                "truncation_reason": "read_limit",
                "original_bytes": 20_000,
                "visible_bytes": 8_000,
                "cursor": {
                    "kind": "read_file",
                    "path": "logs/output.txt",
                    "offset_chars": 8_000,
                    "sha256": "a" * 64,
                },
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="start"),
        *_tool_turn(
            "call_cursor",
            "read_file",
            f"{'preview ' * 1_000}\n{recovery}",
        ),
        Message(role="assistant", content="recent response"),
    ]

    plan = manager.plan_microcompaction(messages, cycle_index=3, current_tokens=1_000)

    assert plan is not None
    assert plan.candidates == ()
    assert manager.apply_microcompaction(messages, plan=plan).messages == messages


def test_compact_marker_excerpt_omits_recovery_bookkeeping() -> None:
    recovery = json.dumps(
        {
            "vv_agent_recovery": {
                "original_bytes": 20_000,
                "visible_bytes": 8_000,
                "sha256": "a" * 64,
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    marker = build_compacted_tool_content(
        f"clean excerpt\n{recovery}",
        artifact_path=".vv-agent/artifacts/run/call.txt",
        tool_name="custom_search",
    )

    assert "clean excerpt" in marker
    assert "vv_agent_recovery" not in marker
    assert "original_bytes" not in marker
    assert "sha256" not in marker


class ShortWriteBackend(MemoryWorkspaceBackend):
    def write_text_exclusive(self, path: str, content: str) -> int:
        del path
        return max(len(content.encode("utf-8")) - 1, 0)


def test_microcompact_short_artifact_write_preserves_original_text() -> None:
    manager = _manager(ShortWriteBackend())
    messages = _messages()
    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1_000)
    assert plan is not None

    result = manager.apply_microcompaction(messages, plan=plan)

    assert result.archived_count == 0
    assert result.artifact_failure_count == 2
    assert result.messages == messages


def test_microcompact_same_task_and_call_ids_allocate_distinct_artifacts() -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    first_messages = _messages()
    first_plan = manager.plan_microcompaction(first_messages, cycle_index=4, current_tokens=1_000)
    assert first_plan is not None
    first_result = manager.apply_microcompaction(first_messages, plan=first_plan)
    assert first_result.messages[3].artifact_ref is not None
    first_path = first_result.messages[3].artifact_ref.path

    second_messages = _messages()
    second_messages[3] = replace(second_messages[3], content="changed" * 2_000)
    second_plan = manager.plan_microcompaction(second_messages, cycle_index=4, current_tokens=1_000)
    assert second_plan is not None
    second_result = manager.apply_microcompaction(second_messages, plan=second_plan)
    assert second_result.messages[3].artifact_ref is not None
    second_path = second_result.messages[3].artifact_ref.path

    assert first_path != second_path
    assert backend.read_text(first_path) == "a" * 8_000
    assert backend.read_text(second_path) == "changed" * 2_000


@pytest.mark.parametrize("length", [499, 500])
def test_microcompact_minimum_boundary_has_no_candidate(length: int) -> None:
    backend = MemoryWorkspaceBackend()
    manager = _manager(backend)
    messages = [
        Message(role="system", content="sys"),
        *_tool_turn("call", "custom_search", "x" * length),
        Message(role="assistant", content="recent"),
    ]

    plan = manager.plan_microcompaction(messages, cycle_index=3, current_tokens=1_000)

    assert plan is not None
    assert plan.candidate_count == 0
