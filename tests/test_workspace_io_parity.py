from __future__ import annotations

import json
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
    content = "中" * 20_000
    (tmp_path / "cjk.txt").write_text(content, encoding="utf-8")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "cjk.txt"})
    payload = json.loads(result.content)

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata == {}
    assert payload == {
        "path": "cjk.txt",
        "start_line": 1,
        "end_line": 1,
        "show_line_numbers": False,
        "content": content,
    }


def test_read_file_too_large_counts_unicode_characters(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    (tmp_path / "large-cjk.txt").write_text("中" * 50_001, encoding="utf-8")

    result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "large-cjk.txt"})
    payload = json.loads(result.content)

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata == {}
    assert payload["content"] is None
    assert payload["file_info"] == {"total_lines": 1, "total_chars": 50_001}
    assert payload["requested"] == {"line_count": 1, "char_count": 50_001}
    assert payload["limits"] == {"max_lines": 2_000, "max_chars": 50_000}


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


def test_edit_file_returns_real_unified_diff(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    (tmp_path / "diff.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "diff.txt"})

    result = _execute(
        registry,
        context,
        EDIT_FILE_TOOL_NAME,
        {"path": "diff.txt", "old_string": "beta", "new_string": "BETTA"},
    )

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.directive == ToolDirective.CONTINUE
    assert result.error_code is None
    assert result.metadata["diff"] == ("--- diff.txt\n+++ diff.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETTA\n gamma\n")
    assert result.metadata["diff_truncated"] is False
    assert result.metadata["additions"] == 1
    assert result.metadata["deletions"] == 1


def test_edit_file_truncates_large_cjk_diff_at_unicode_boundary(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    before = "旧" * 6_100
    after = "新" * 6_100
    (tmp_path / "large-diff.txt").write_text(before, encoding="utf-8")
    _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "large-diff.txt"})

    result = _execute(
        registry,
        context,
        EDIT_FILE_TOOL_NAME,
        {"path": "large-diff.txt", "old_string": before, "new_string": after},
    )
    diff = result.metadata["diff"]

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.metadata["diff_truncated"] is True
    assert len(diff) == 12_000
    assert len(diff.encode("utf-8")) > 12_000
    assert diff.startswith("--- large-diff.txt\n+++ large-diff.txt\n@@ -1 +1 @@\n-")
    assert (tmp_path / "large-diff.txt").read_text(encoding="utf-8") == after


def test_read_and_edit_preserve_utf8_bom_and_crlf(tmp_path: Path) -> None:
    registry, context = _tool_runtime(tmp_path)
    target = tmp_path / "bom-crlf.txt"
    target.write_bytes(b"\xef\xbb\xbffirst\r\n" + "第二行\r\n".encode())

    read_result = _execute(registry, context, READ_FILE_TOOL_NAME, {"path": "bom-crlf.txt"})
    read_payload = json.loads(read_result.content)
    assert read_payload["content"] == "first\n第二行"

    edit_result = _execute(
        registry,
        context,
        EDIT_FILE_TOOL_NAME,
        {"path": "bom-crlf.txt", "old_string": "第二行", "new_string": "更新行"},
    )

    assert edit_result.status_code is ToolResultStatus.SUCCESS
    assert edit_result.metadata["line_ending"] == "crlf"
    assert "\ufeff" not in edit_result.metadata["diff"]
    assert "\r" not in edit_result.metadata["diff"]
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
