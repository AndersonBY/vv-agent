from __future__ import annotations

import json
import time
from pathlib import Path

from vv_agent.constants import (
    EDIT_FILE_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.types import ToolCall, ToolDirective, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def _tool_runtime(workspace: Path):
    return build_default_registry(), ToolContext(
        workspace=workspace,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(workspace),
    )


def _execute(registry, context: ToolContext, name: str, arguments: dict):
    return registry.execute(ToolCall(id=f"call_{name}", name=name, arguments=arguments), context)


def test_read_file_counts_unicode_characters_and_preserves_result_contract(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    content = "中" * 10_000
    (tmp_path / "cjk.txt").write_text(content, encoding="utf-8")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "cjk.txt"})

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata == {}
    assert result.content == content
    assert result.truncated is None
    assert result.cursor is None


def test_read_file_too_large_counts_unicode_characters(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    content = "中" * 12_001
    (tmp_path / "large-cjk.txt").write_text(content, encoding="utf-8")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "large-cjk.txt"})

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata == {}
    assert result.truncated is True
    assert result.content == "中" * 12_000
    assert result.original_bytes == len(content.encode())
    assert result.visible_bytes == len(result.content.encode())
    assert result.cursor is not None
    assert result.cursor.path == "large-cjk.txt"
    assert result.cursor.offset_chars == 12_000

    continued = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "large-cjk.txt", "cursor": result.cursor.to_dict()},
    )
    assert continued.content == "中"
    assert continued.truncated is None
    assert continued.cursor is None


def test_read_file_line_limit_returns_recoverable_cursor(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    (tmp_path / "large-lines.txt").write_text("x\n" * 2_001, encoding="utf-8")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "large-lines.txt"})

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.truncated is True
    assert result.content == "x\n" * 2_000
    assert result.cursor is not None
    assert result.cursor.offset_chars == 4_000

    continued = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "large-lines.txt", "cursor": result.cursor.to_dict()},
    )
    assert continued.content == "x\n"
    assert continued.cursor is None


def test_read_file_line_number_previews_preserve_rendered_output_across_cursors(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    source = "x\n" * 2_001
    expected = "".join(f"{line_number}: x\n" for line_number in range(1, 2_002))
    (tmp_path / "numbered-lines.txt").write_text(source, encoding="utf-8")

    arguments = {"path": "numbered-lines.txt", "show_line_numbers": True}
    chunks: list[str] = []
    for _ in range(10):
        result = _execute(registry, context, READ_FILE_TOOL_NAME, arguments)
        assert result.status_code is ToolResultStatus.SUCCESS
        assert len(result.content) <= 12_000
        chunks.append(result.content)
        if result.cursor is None:
            break
        assert result.truncated is True
        assert result.visible_bytes == len(result.content.encode("utf-8"))
        assert result.original_bytes is not None
        assert result.original_bytes >= result.visible_bytes
        arguments = {
            "path": "numbered-lines.txt",
            "show_line_numbers": True,
            "cursor": result.cursor.to_dict(),
        }
    else:
        raise AssertionError("read_file cursor did not reach EOF")

    assert "".join(chunks) == expected


def test_read_file_cursor_rejects_range_path_mismatch_stale_source_and_invalid_offset(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    target = tmp_path / "chars.txt"
    target.write_text("a" * 12_001, encoding="utf-8")
    initial = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "chars.txt"})
    assert initial.cursor is not None
    cursor = initial.cursor.to_dict()

    incompatible = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "chars.txt", "cursor": cursor, "start_line": 1},
    )
    assert incompatible.error_code == "invalid_arguments"

    mismatch = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "missing.txt", "cursor": cursor},
    )
    assert mismatch.error_code == "cursor_path_mismatch"

    target.write_bytes(b"\xff")
    stale = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "chars.txt", "cursor": cursor},
    )
    assert stale.error_code == "stale_cursor"

    target.write_text("a" * 12_001 + "b", encoding="utf-8")
    current = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "chars.txt"})
    assert current.cursor is not None
    invalid_cursor = current.cursor.to_dict()
    invalid_cursor["offset_chars"] = 9_007_199_254_740_991
    invalid_offset = _execute(
        registry,
        context,
        READ_FILE_TOOL_NAME,
        {"path": "chars.txt", "cursor": invalid_cursor},
    )
    assert invalid_offset.error_code == "cursor_offset_invalid"


def test_read_file_recovers_reserved_artifact_through_normal_tool_path(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    artifact_path = ".vv-agent/artifacts/task-7/call-7.txt"
    context.workspace_backend.write_text_exclusive(artifact_path, "complete artifact output")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": artifact_path})

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.content == "complete artifact output"
    assert result.truncated is None


def test_read_file_validation_and_not_found_errors_are_structured(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)

    missing_path = _execute(registry, context, READ_FILE_TOOL_NAME, {})
    missing_payload = json.loads(missing_path.content)
    assert missing_path.status_code is ToolResultStatus.ERROR
    assert missing_path.directive == ToolDirective.CONTINUE
    assert missing_path.error_code == "invalid_tool_arguments"
    assert missing_path.metadata == {
        "error_code": "invalid_tool_arguments",
        "issue_count": 1,
    }
    assert missing_payload == {
        "ok": False,
        "error": "Tool arguments do not match the declared schema",
        "error_code": "invalid_tool_arguments",
        "issues": [
            {
                "instance_path": "",
                "rule": "required",
                "schema_path": "/required",
            }
        ],
    }

    not_found = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "missing.txt"})
    not_found_payload = json.loads(not_found.content)
    assert not_found.status_code is ToolResultStatus.ERROR
    assert not_found.directive == ToolDirective.CONTINUE
    assert not_found.error_code == "file_not_found"
    assert not_found.metadata == {"error_code": "file_not_found", "path": "missing.txt"}
    assert not_found_payload["ok"] is False
    assert not_found_payload["error_code"] == "file_not_found"
    assert not_found_payload["path"] == "missing.txt"


def test_write_file_reports_utf8_bytes_and_compatible_unicode_chars(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)

    result = _execute(
        registry,
        context,
        WRITE_FILE_TOOL_NAME,
        {"path": "written.txt", "content": "中文"},
    )
    payload = json.loads(result.content)

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert payload["written_bytes"] == 6
    assert payload["written_chars"] == 2
    assert result.metadata == {
        "changed_files": ["written.txt"],
        "operation": "write_file",
        "append": False,
    }


def test_edit_file_large_document_consecutive_edits_keep_compact_receipts_and_current_baseline(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    target = tmp_path / "report.md"
    before = "重复正文 references / 中文来源\n" * 4_000
    before += "\n".join(f"source{i}/target" for i in range(7))
    target.write_text(before, encoding="utf-8")
    read = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "report.md", "start_line": 1, "end_line": 1})
    assert read.status_code is ToolResultStatus.SUCCESS

    expected = before
    started = time.monotonic()
    for i in range(7):
        result = _execute(
            registry,
            context,
            EDIT_FILE_TOOL_NAME,
            {"path": "report.md", "old_string": f"source{i}/target", "new_string": f"source{i} / target"},
        )
        assert result.status_code is ToolResultStatus.SUCCESS
        assert json.loads(result.content) == {"ok": True, "path": "report.md", "replaced_count": 1}
        assert result.metadata == {"changed_files": ["report.md"], "operation": "edit_file", "line_ending": "lf"}
        expected = expected.replace(f"source{i}/target", f"source{i} / target")
    assert time.monotonic() - started < 5
    assert target.read_text(encoding="utf-8") == expected

    target.write_text(expected + "\nexternal change", encoding="utf-8")
    rejected = _execute(
        registry,
        context,
        EDIT_FILE_TOOL_NAME,
        {"path": "report.md", "old_string": "source0", "new_string": "updated"},
    )
    assert rejected.error_code == "file_changed_since_read"
    assert target.read_text(encoding="utf-8") == expected + "\nexternal change"


def test_read_and_edit_preserve_utf8_bom_and_crlf(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    target = tmp_path / "bom-crlf.txt"
    target.write_bytes(b"\xef\xbb\xbffirst\r\n" + "第二行\r\n".encode())

    read_result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "bom-crlf.txt"})
    assert read_result.content == "first\r\n第二行\r\n"

    edit_result = _execute(
        registry,
        context,
        EDIT_FILE_TOOL_NAME,
        {"path": "bom-crlf.txt", "old_string": "第二行", "new_string": "更新行"},
    )

    assert edit_result.status_code is ToolResultStatus.SUCCESS
    assert edit_result.metadata["line_ending"] == "crlf"
    assert target.read_bytes() == b"\xef\xbb\xbffirst\r\n" + "更新行\r\n".encode()


def test_search_files_uses_unicode_output_budget_and_omits_zero_sensitive_count(
    tmp_path: Path,
) -> None:
    registry, context = _tool_runtime(tmp_path)
    (tmp_path / "search.txt").write_text("token " + "中" * 40_000, encoding="utf-8")

    result = _execute(
        registry,
        context,
        SEARCH_FILES_TOOL_NAME,
        {"pattern": "token", "output_mode": "content"},
    )

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata["content_truncated"] is True
    assert result.content.startswith("Found 1 matches in 1 files for pattern 'token'")
    assert "Shown: 3 lines, 30000 characters" in result.content
    assert "sensitive_files_omitted" not in result.metadata
