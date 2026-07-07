from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from vv_agent import constants as constants_module
from vv_agent.constants import (
    ASK_USER_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.handlers import search as search_handler
from vv_agent.tools.handlers import workspace_io
from vv_agent.tools.registry import ToolNotFoundError
from vv_agent.types import ToolCall, ToolDirective
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

TASK_LIST_TOOL_NAME = getattr(constants_module, "".join(("TO", "DO")) + "_WRITE_TOOL_NAME")


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_edit_file_is_registered_in_default_tools(registry) -> None:
    tool_names = registry.list_tool_names()
    old_tool_name = "file" + "_str_replace"

    assert EDIT_FILE_TOOL_NAME in tool_names
    assert old_tool_name not in tool_names

    edit_schema = registry.get_schema(EDIT_FILE_TOOL_NAME)
    edit_description = edit_schema["function"]["description"]
    parameters = edit_schema["function"]["parameters"]
    properties = parameters["properties"]

    assert edit_schema["function"]["name"] == "edit_file"
    assert "read_file" in edit_description
    assert "write_file" in edit_description
    assert "previous successful edit_file" in edit_description
    assert parameters["required"] == ["path", "old_string", "new_string"]
    assert set(properties) == {"path", "old_string", "new_string", "replace_all"}


def test_legacy_edit_tool_name_is_removed(registry, tool_context: ToolContext) -> None:
    old_tool_name = "file" + "_str_replace"
    legacy_args = {"path": "edit.txt", "old" + "_str": "a", "new" + "_str": "b"}
    call = ToolCall(id="call_removed_replace", name=old_tool_name, arguments=legacy_args)

    with pytest.raises(ToolNotFoundError):
        registry.execute(call, tool_context)


def test_old_search_tool_names_are_not_registered(registry, tool_context: ToolContext) -> None:
    with pytest.raises(ToolNotFoundError):
        registry.execute(
            ToolCall(id="old_grep", name="workspace_grep", arguments={"pattern": "token"}),
            tool_context,
        )

    with pytest.raises(ToolNotFoundError):
        registry.execute(
            ToolCall(id="old_list", name="list_files", arguments={}),
            tool_context,
        )


def test_edit_file_rejects_legacy_argument_names(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "legacy.txt"
    target.write_text("hello", encoding="utf-8")
    registry.execute(
        ToolCall(id="read_legacy", name=READ_FILE_TOOL_NAME, arguments={"path": "legacy.txt"}),
        tool_context,
    )

    old_key = "old" + "_str"
    new_key = "new" + "_str"
    result = registry.execute(
        ToolCall(
            id="edit_legacy",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "legacy.txt", old_key: "hello", new_key: "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert payload["error_code"] == "invalid_arguments"
    assert "old_string" in payload["message"]


def test_edit_file_rejects_missing_path(registry, tool_context: ToolContext) -> None:
    result = registry.execute(
        ToolCall(
            id="edit_missing_path",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"old_string": "hello", "new_string": "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert payload["error_code"] == "invalid_arguments"
    assert "path" in payload["message"]
    assert "old_string" in payload["message"]
    assert "new_string" in payload["message"]


def test_edit_file_rejects_empty_old_string(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "empty_old.txt"
    target.write_text("hello", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="edit_empty_old",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "empty_old.txt", "old_string": "", "new_string": "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "old_string_empty"
    assert payload["error_code"] == "old_string_empty"
    assert "old_string" in payload["message"]
    assert target.read_text(encoding="utf-8") == "hello"


def test_edit_file_requires_unique_old_string_by_default(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "duplicate.txt"
    target.write_text("hello world\nhello agent", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_duplicate", name=READ_FILE_TOOL_NAME, arguments={"path": "duplicate.txt"}),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="edit_duplicate",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "duplicate.txt", "old_string": "hello", "new_string": "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "old_string_not_unique"
    assert payload["error_code"] == "old_string_not_unique"
    assert payload["match_count"] == 2
    assert result.metadata["match_count"] == 2
    assert target.read_text(encoding="utf-8") == "hello world\nhello agent"


def test_edit_file_rejects_missing_new_string(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "missing_new.txt"
    target.write_text("hello", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="edit_missing_new",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "missing_new.txt", "old_string": "hello"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert payload["error_code"] == "invalid_arguments"
    assert "old_string" in payload["message"]
    assert "new_string" in payload["message"]
    assert target.read_text(encoding="utf-8") == "hello"


def test_workspace_write_and_read(registry, tool_context: ToolContext) -> None:
    write_call = ToolCall(id="call1", name=WRITE_FILE_TOOL_NAME, arguments={"path": "notes/test.txt", "content": "hello"})
    write_result = registry.execute(write_call, tool_context)
    assert write_result.status == "success"

    read_call = ToolCall(id="call2", name=READ_FILE_TOOL_NAME, arguments={"path": "notes/test.txt"})
    read_result = registry.execute(read_call, tool_context)
    payload = json.loads(read_result.content)
    assert payload["content"] == "hello"


def test_read_file_allows_absolute_path_when_enabled(registry, tool_context: ToolContext) -> None:
    outside = (tool_context.workspace.parent / f"{tool_context.workspace.name}_outside_read.txt").resolve()
    outside.write_text("outside", encoding="utf-8")
    tool_context.task_metadata = {"allow_outside_workspace_paths": True}
    tool_context.workspace_backend = LocalWorkspaceBackend(
        tool_context.workspace,
        allow_outside_root=True,
    )

    read_call = ToolCall(id="call_abs_read", name=READ_FILE_TOOL_NAME, arguments={"path": str(outside)})
    read_result = registry.execute(read_call, tool_context)
    payload = json.loads(read_result.content)

    assert read_result.status == "success"
    assert payload["content"] == "outside"


def test_find_files_allows_absolute_path_when_enabled(registry, tool_context: ToolContext) -> None:
    outside_dir = (tool_context.workspace.parent / f"{tool_context.workspace.name}_outside_list").resolve()
    outside_dir.mkdir(parents=True, exist_ok=True)
    target = (outside_dir / "a.txt").resolve()
    target.write_text("a", encoding="utf-8")
    tool_context.task_metadata = {"allow_outside_workspace_paths": True}
    tool_context.workspace_backend = LocalWorkspaceBackend(
        tool_context.workspace,
        allow_outside_root=True,
    )

    list_call = ToolCall(
        id="call_abs_list",
        name=FIND_FILES_TOOL_NAME,
        arguments={"path": str(outside_dir)},
    )
    list_result = registry.execute(list_call, tool_context)
    payload = json.loads(list_result.content)

    assert list_result.status == "success"
    assert str(target) in payload["files"]


def test_write_file_append_with_optional_newlines(registry, tool_context: ToolContext) -> None:
    base_call = ToolCall(id="call_write_base", name=WRITE_FILE_TOOL_NAME, arguments={"path": "notes/log.txt", "content": "line1"})
    base_result = registry.execute(base_call, tool_context)
    assert base_result.status == "success"

    append_call = ToolCall(
        id="call_write_append",
        name=WRITE_FILE_TOOL_NAME,
        arguments={
            "path": "notes/log.txt",
            "content": "line2",
            "append": True,
            "leading_newline": True,
            "trailing_newline": True,
        },
    )
    append_result = registry.execute(append_call, tool_context)
    append_payload = json.loads(append_result.content)

    assert append_payload["append"] is True
    assert append_payload["leading_newline"] is True
    assert append_payload["trailing_newline"] is True
    assert append_payload["written_chars"] == 7
    assert (tool_context.workspace / "notes/log.txt").read_text(encoding="utf-8") == "line1\nline2\n"


def test_write_file_ignores_newline_flags_when_overwriting(registry, tool_context: ToolContext) -> None:
    write_call = ToolCall(
        id="call_write",
        name=WRITE_FILE_TOOL_NAME,
        arguments={
            "path": "notes/overwrite.txt",
            "content": "final",
            "leading_newline": True,
            "trailing_newline": True,
        },
    )
    write_result = registry.execute(write_call, tool_context)
    payload = json.loads(write_result.content)

    assert payload["append"] is False
    assert payload["leading_newline"] is False
    assert payload["trailing_newline"] is False
    assert payload["written_chars"] == 5
    assert (tool_context.workspace / "notes/overwrite.txt").read_text(encoding="utf-8") == "final"


def test_find_files_truncates_large_response(registry, tool_context: ToolContext) -> None:
    for idx in range(620):
        (tool_context.workspace / f"f_{idx:04d}.txt").write_text("x", encoding="utf-8")

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["count"] == 620
    assert payload["truncated"] is True
    assert payload["max_results"] == 100
    assert payload["returned_count"] == 100
    assert len(payload["files"]) == 100
    assert payload["remaining_count"] == 520


def test_find_files_summarizes_common_roots_by_default(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "src").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "src" / "main.ts").write_text("export {}", encoding="utf-8")
    (tool_context.workspace / "node_modules" / "pkg" / "a.js").write_text("a", encoding="utf-8")
    (tool_context.workspace / "node_modules" / "pkg" / "b.js").write_text("b", encoding="utf-8")

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={"path": "."})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["src/main.ts"]
    ignored = payload.get("ignored_roots") or []
    assert ignored == [{"path": "node_modules"}]
    assert "summarized" in str(payload.get("message", "")).lower()


def test_find_files_include_ignored_roots_when_requested(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "node_modules" / "pkg" / "a.js").write_text("a", encoding="utf-8")

    call = ToolCall(
        id="call_list",
        name=FIND_FILES_TOOL_NAME,
        arguments={"path": ".", "include_ignored": True},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["node_modules/pkg/a.js"]
    assert payload.get("ignored_roots") is None


def test_find_files_reports_estimated_count_when_scan_limit_reached(registry, tool_context: ToolContext) -> None:
    for idx in range(40):
        (tool_context.workspace / f"scan_{idx:03d}.txt").write_text("x", encoding="utf-8")

    call = ToolCall(
        id="call_list",
        name=FIND_FILES_TOOL_NAME,
        arguments={"max_results": 10, "scan_limit": 12},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["returned_count"] == 10
    assert payload["truncated"] is True
    assert payload["count_is_estimate"] is True
    assert payload["scan_limit"] == 12


def test_find_files_can_list_inside_ignored_root(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "node_modules" / "pkg" / "a.js").write_text("a", encoding="utf-8")

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={"path": "node_modules"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["node_modules/pkg/a.js"]
    assert payload.get("ignored_roots") is None


def test_find_files_prefers_ripgrep_when_available(registry, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"sub/b.txt\x00a.txt\x00")
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    def _fake_popen(*args, **kwargs):
        assert args[0][0] == "rg"
        return _FakeProcess()

    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(workspace_io.subprocess, "Popen", _fake_popen)

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={"path": "."})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["a.txt", "sub/b.txt"]
    assert payload["count"] == 2
    assert payload["truncated"] is False


def test_find_files_glob_matches_rg_dot_slash_paths(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"./doc.md\x00./nested/inner.md\x00")
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(workspace_io.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={"path": ".", "glob": "*.md"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["doc.md"]
    assert payload["count"] == 1
    assert payload["truncated"] is False


def test_find_files_falls_back_when_ripgrep_errors(registry, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    (tool_context.workspace / "fallback.txt").write_text("x", encoding="utf-8")

    class _FailingProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"permission denied")
            self.returncode = 2

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(workspace_io.subprocess, "Popen", lambda *args, **kwargs: _FailingProcess())

    call = ToolCall(id="call_list", name=FIND_FILES_TOOL_NAME, arguments={"path": "."})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["fallback.txt"]
    assert payload["count"] == 1


def test_find_files_rejects_pattern_argument(registry, tool_context: ToolContext) -> None:
    result = registry.execute(
        ToolCall(id="find_pattern_rejected", name=FIND_FILES_TOOL_NAME, arguments={"pattern": "*.py"}),
        tool_context,
    )
    payload = json.loads(result.content)

    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert payload["error_code"] == "invalid_arguments"
    assert "pattern" in payload["message"]


def test_find_files_supports_offset_sort_and_sensitive_filter(registry, tool_context: ToolContext) -> None:
    first = tool_context.workspace / "first.txt"
    second = tool_context.workspace / "second.txt"
    secret = tool_context.workspace / ".env"
    first.write_text("1", encoding="utf-8")
    second.write_text("2", encoding="utf-8")
    secret.write_text("SECRET=1", encoding="utf-8")
    os.utime(first, (1_700_000_000, 1_700_000_000))
    os.utime(second, (1_700_000_010, 1_700_000_010))

    result = registry.execute(
        ToolCall(
            id="find_page",
            name=FIND_FILES_TOOL_NAME,
            arguments={"glob": "*.txt", "sort": "modified_desc", "offset": 0, "max_results": 1},
        ),
        tool_context,
    )
    payload = json.loads(result.content)
    assert payload["files"] == ["second.txt"]
    assert payload["sort"] == "modified_desc"
    assert payload["offset"] == 0
    assert payload["remaining_count"] == 1

    hidden_default = registry.execute(
        ToolCall(
            id="find_sensitive_default",
            name=FIND_FILES_TOOL_NAME,
            arguments={"glob": "**/*", "include_hidden": True, "sort": "path_asc"},
        ),
        tool_context,
    )
    hidden_payload = json.loads(hidden_default.content)
    assert ".env" not in hidden_payload["files"]
    assert hidden_payload["sensitive_files_omitted"] == 1

    hidden_included = registry.execute(
        ToolCall(
            id="find_sensitive_included",
            name=FIND_FILES_TOOL_NAME,
            arguments={"glob": "**/*", "include_hidden": True, "include_sensitive": True, "sort": "path_asc"},
        ),
        tool_context,
    )
    included_payload = json.loads(hidden_included.content)
    assert ".env" in included_payload["files"]


def test_find_files_non_local_backend_honors_scan_limit(registry, tmp_path: Path) -> None:
    backend = MemoryWorkspaceBackend()
    for index in range(5):
        backend.write_text(f"file_{index}.txt", "x")
    context = ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=backend,
    )

    result = registry.execute(
        ToolCall(
            id="find_memory_scan_limit",
            name=FIND_FILES_TOOL_NAME,
            arguments={"max_results": 2, "scan_limit": 2, "sort": "path_asc"},
        ),
        context,
    )
    payload = json.loads(result.content)

    assert payload["files"] == ["file_0.txt", "file_1.txt"]
    assert payload["count"] == 2
    assert payload["count_is_estimate"] is True
    assert payload["scan_limit"] == 2


def test_read_file_can_show_line_numbers(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "notes.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    call = ToolCall(
        id="call_read",
        name=READ_FILE_TOOL_NAME,
        arguments={"path": "notes.txt", "start_line": 2, "show_line_numbers": True},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["show_line_numbers"] is True
    assert payload["content"] == "2: beta\n3: gamma"


def test_read_file_returns_file_info_when_line_limit_exceeded(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "long.txt"
    target.write_text("\n".join(f"line-{index}" for index in range(1, 2002)), encoding="utf-8")

    call = ToolCall(id="call_read", name=READ_FILE_TOOL_NAME, arguments={"path": "long.txt"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["content"] is None
    assert payload["limits"] == {"max_lines": 2000, "max_chars": 50000}
    assert payload["file_info"]["total_lines"] == 2001
    assert payload["requested"]["line_count"] == 2001


def test_read_file_returns_file_info_when_char_limit_exceeded(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "chars.txt"
    target.write_text("a" * 50001, encoding="utf-8")

    call = ToolCall(id="call_read", name=READ_FILE_TOOL_NAME, arguments={"path": "chars.txt"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["content"] is None
    assert payload["requested"]["char_count"] > 50000
    assert payload["file_info"]["total_chars"] == 50001


def test_read_file_returns_file_info_when_requested_range_exceeds_limit(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "ranged.txt"
    target.write_text("\n".join(f"row-{index}" for index in range(1, 3001)), encoding="utf-8")

    call = ToolCall(
        id="call_read",
        name=READ_FILE_TOOL_NAME,
        arguments={"path": "ranged.txt", "start_line": 1, "end_line": 2501},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["content"] is None
    assert payload["requested"]["line_count"] == 2501
    assert payload["suggested_range"] == {"start_line": 1, "end_line": 2000}


def test_search_files(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("hello world\nsecond line", encoding="utf-8")
    call = ToolCall(id="call1", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "hello", "output_mode": "content"})
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert result.content.startswith("Found 1 matches in 1 files")
    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["line"] == 1


def test_search_files_defaults_to_files_with_matches(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("token one", encoding="utf-8")
    (tool_context.workspace / "b.txt").write_text("token two", encoding="utf-8")

    result = registry.execute(
        ToolCall(id="search_default", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "token"}),
        tool_context,
    )

    assert result.status == "success"
    assert result.metadata["output_mode"] == "files_with_matches"
    assert result.metadata["files"] == ["a.txt", "b.txt"]
    assert "matches" not in result.metadata


def test_search_files_literal_offset_and_unlimited_head_limit(registry, tool_context: ToolContext) -> None:
    for index in range(4):
        (tool_context.workspace / f"file_{index}.txt").write_text("a.b token", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="search_literal_page",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "a.b", "literal": True, "offset": 1, "head_limit": 2},
        ),
        tool_context,
    )

    assert result.metadata["files"] == ["file_1.txt", "file_2.txt"]
    assert result.metadata["offset"] == 1
    assert result.metadata["head_limit"] == 2
    assert result.metadata["total_result_items"] == 4
    assert result.metadata["returned_count"] == 2

    unlimited = registry.execute(
        ToolCall(
            id="search_literal_unlimited",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "a.b", "literal": True, "head_limit": 0},
        ),
        tool_context,
    )
    assert unlimited.metadata["returned_count"] == 4


def test_search_files_omits_sensitive_paths_by_default(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tool_context.workspace / "visible.txt").write_text("TOKEN=public", encoding="utf-8")

    default_result = registry.execute(
        ToolCall(id="search_sensitive_default", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "TOKEN"}),
        tool_context,
    )
    assert default_result.metadata["files"] == ["visible.txt"]
    assert default_result.metadata["sensitive_files_omitted"] == 1

    included = registry.execute(
        ToolCall(
            id="search_sensitive_included",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "TOKEN", "include_hidden": True, "include_sensitive": True},
        ),
        tool_context,
    )
    assert included.metadata["files"] == [".env", "visible.txt"]


def test_search_files_rg_excludes_sensitive_globs_before_scanning(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "visible.txt").write_text("TOKEN=public", encoding="utf-8")
    (tool_context.workspace / "private.pem").write_text("TOKEN=secret", encoding="utf-8")

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        '{"type":"begin","data":{"path":{"text":"visible.txt"}}}',
                        (
                            '{"type":"match","data":{"path":{"text":"visible.txt"},'
                            '"lines":{"text":"TOKEN=public\\n"},"line_number":1,'
                            '"submatches":[{"start":0,"end":5}]}}'
                        ),
                        '{"type":"summary","data":{"stats":{"searches":1}}}',
                    ]
                )
                + "\n"
            )
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def _fake_popen(*args, **kwargs):
        command = args[0]
        assert "--glob" in command
        assert "!**/*.pem" in command
        return _FakeProcess()

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", _fake_popen)

    result = registry.execute(
        ToolCall(id="search_sensitive_rg", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "TOKEN"}),
        tool_context,
    )

    assert result.metadata["files"] == ["visible.txt"]
    assert result.metadata["summary"]["files_searched"] == 1
    assert result.metadata["summary"]["total_matches"] == 1
    assert result.metadata["sensitive_files_omitted"] == 1


def test_search_files_hidden_config_sensitive_path_uses_fallback_not_rg(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / ".config").mkdir()
    (tool_context.workspace / ".config" / "service_token.json").write_text("TOKEN=secret", encoding="utf-8")
    (tool_context.workspace / "visible.txt").write_text("TOKEN=public", encoding="utf-8")

    def _fail_if_rg_runs(*args, **kwargs):
        raise AssertionError("rg must not scan uncovered .config sensitive paths")

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", _fail_if_rg_runs)

    result = registry.execute(
        ToolCall(
            id="search_config_sensitive",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "TOKEN", "include_hidden": True},
        ),
        tool_context,
    )

    assert result.metadata["files"] == ["visible.txt"]
    assert result.metadata["summary"]["total_matches"] == 1
    assert result.metadata["sensitive_files_omitted"] == 1


def test_search_files_uppercase_sensitive_suffix_uses_fallback_not_rg(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "keys").mkdir()
    (tool_context.workspace / "keys" / "AuthKey_ABC123.P8").write_text("TOKEN=secret", encoding="utf-8")
    (tool_context.workspace / "visible.txt").write_text("TOKEN=public", encoding="utf-8")

    def _fail_if_rg_runs(*args, **kwargs):
        raise AssertionError("rg must not scan case-variant sensitive suffixes")

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", _fail_if_rg_runs)

    result = registry.execute(
        ToolCall(id="search_uppercase_sensitive", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "TOKEN"}),
        tool_context,
    )

    assert result.metadata["files"] == ["visible.txt"]
    assert result.metadata["summary"]["total_matches"] == 1
    assert result.metadata["sensitive_files_omitted"] == 1


def test_search_files_uses_smart_case_for_lowercase_patterns(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("update lower\nUpdate upper", encoding="utf-8")

    call = ToolCall(
        id="call_smart_case_lower",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "update", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 2
    assert [row["text"] for row in payload["matches"]] == ["update lower", "Update upper"]


def test_search_files_uses_case_sensitive_default_when_pattern_has_uppercase(
    registry,
    tool_context: ToolContext,
) -> None:
    (tool_context.workspace / "a.txt").write_text("update lower\nUpdate upper", encoding="utf-8")

    call = ToolCall(
        id="call_smart_case_upper",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Update", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 1
    assert [row["text"] for row in payload["matches"]] == ["Update upper"]


def test_search_files_explicit_case_flags_override_smart_case(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("update lower\nUpdate upper", encoding="utf-8")

    case_sensitive_call = ToolCall(
        id="call_smart_case_override_lower",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "update", "output_mode": "content", "case_sensitive": True},
    )
    case_sensitive_result = registry.execute(case_sensitive_call, tool_context)
    case_sensitive_payload = case_sensitive_result.metadata

    assert case_sensitive_payload["summary"]["total_matches"] == 1
    assert [row["text"] for row in case_sensitive_payload["matches"]] == ["update lower"]

    case_insensitive_call = ToolCall(
        id="call_smart_case_override_upper",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Update", "output_mode": "content", "case_sensitive": False},
    )
    case_insensitive_result = registry.execute(case_insensitive_call, tool_context)
    case_insensitive_payload = case_insensitive_result.metadata

    assert case_insensitive_payload["summary"]["total_matches"] == 2
    assert [row["text"] for row in case_insensitive_payload["matches"]] == ["update lower", "Update upper"]


def test_search_files_allows_absolute_path_when_enabled(registry, tool_context: ToolContext) -> None:
    outside_dir = (tool_context.workspace.parent / f"{tool_context.workspace.name}_outside_grep").resolve()
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "a.txt").write_text("hello outside", encoding="utf-8")
    tool_context.task_metadata = {"allow_outside_workspace_paths": True}
    tool_context.workspace_backend = LocalWorkspaceBackend(
        tool_context.workspace,
        allow_outside_root=True,
    )

    call = ToolCall(
        id="call_abs_grep",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "hello", "output_mode": "content", "path": str(outside_dir)},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["path"] == str((outside_dir / "a.txt").resolve())


def test_search_files_supports_files_with_matches_mode(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.py").write_text("TOKEN = 1", encoding="utf-8")
    (tool_context.workspace / "b.py").write_text("token = 2", encoding="utf-8")
    (tool_context.workspace / "c.md").write_text("no hit", encoding="utf-8")

    call = ToolCall(
        id="call_files",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "token", "output_mode": "files_with_matches", "case_sensitive": False, "type": "py"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["files"] == ["a.py", "b.py"]
    assert payload["summary"]["files_with_matches"] == 2
    assert payload["summary"]["total_matches"] == 2


def test_search_files_ignores_removed_max_results_alias(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("hit one\nhit two", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="call_removed_max_results",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "hit", "output_mode": "content", "max_results": 1},
        ),
        tool_context,
    )

    assert len(result.metadata["matches"]) == 2
    assert result.metadata["head_limit"] == 250
    assert "max_results" not in result.metadata


def test_search_files_supports_count_mode(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "logs").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "logs" / "x.log").write_text("err\nok\nerr", encoding="utf-8")
    (tool_context.workspace / "logs" / "y.log").write_text("err", encoding="utf-8")

    call = ToolCall(
        id="call_count",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "err", "output_mode": "count", "path": "logs", "type": "log"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["file_counts"] == {"logs/x.log": 2, "logs/y.log": 1}
    assert payload["summary"]["total_matches"] == 3


def test_search_files_supports_context_lines(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "ctx.txt").write_text("line1\nhit\nline3", encoding="utf-8")
    call = ToolCall(
        id="call_ctx",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "hit", "output_mode": "content", "c": 1, "n": True},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert [row["line"] for row in payload["matches"]] == [1, 2, 3]
    assert payload["matches"][0]["is_match"] is False
    assert payload["matches"][1]["is_match"] is True


def test_search_files_supports_multiline_and_head_limit(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "multi.txt").write_text("start\nalpha\nbeta\nend", encoding="utf-8")
    call = ToolCall(
        id="call_multi",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "alpha\\nbeta", "output_mode": "content", "multiline": True, "head_limit": 1},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert len(payload["matches"]) == 1
    assert payload["head_limit"] == 1
    assert payload["head_limited"] is False
    assert payload["summary"]["total_matches"] == 1


def test_search_files_caps_structured_payload_without_duplication(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(3):
        (tool_context.workspace / f"match_{index}.txt").write_text("token\n", encoding="utf-8")

    monkeypatch.setattr(search_handler, "_MAX_STRUCTURED_ITEMS", 2)
    monkeypatch.setattr(search_handler, "_MAX_STRUCTURED_CHARS", 10_000)

    call = ToolCall(
        id="call_structured_cap",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "token", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["total_result_items"] == 3
    assert payload["returned_count"] == 2
    assert payload["structured_truncated"] is True
    assert payload["truncated"] is True
    assert len(payload["matches"]) == 2
    assert "Showing first 2 rows." in result.content
    assert "\"matches\"" not in result.content


def test_search_files_excludes_hidden_by_default(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / ".hidden.txt").write_text("secret Agent marker", encoding="utf-8")

    call = ToolCall(
        id="call_hidden_default",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 0


def test_search_files_can_include_hidden_files(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / ".hidden.txt").write_text("secret Agent marker", encoding="utf-8")

    call = ToolCall(
        id="call_hidden_include",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "output_mode": "content", "include_hidden": True},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["path"] == ".hidden.txt"


def test_search_files_skips_common_ignored_roots_by_default(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "node_modules" / "pkg" / "x.js").write_text("Agent token", encoding="utf-8")

    call = ToolCall(
        id="call_ignored_default",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 0


def test_search_files_can_include_common_ignored_roots(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "node_modules" / "pkg" / "x.js").write_text("Agent token", encoding="utf-8")

    call = ToolCall(
        id="call_ignored_include",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "output_mode": "content", "include_ignored": True},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["path"] == "node_modules/pkg/x.js"


def test_search_files_supports_file_path_target(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "articles").mkdir(parents=True, exist_ok=True)
    file_path = "articles/essay.md"
    (tool_context.workspace / file_path).write_text("intro\nabout Agent design\noutro", encoding="utf-8")

    call = ToolCall(
        id="call_file_target",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "path": file_path, "output_mode": "content", "c": 1},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["files_searched"] == 1
    assert payload["summary"]["files_with_matches"] == 1
    assert payload["summary"]["total_matches"] == 1
    assert any(row["path"] == file_path and row["is_match"] for row in payload["matches"])


def test_search_files_file_path_works_inside_ignored_root(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    file_path = "node_modules/pkg/x.js"
    (tool_context.workspace / file_path).write_text("const token = 'Agent';", encoding="utf-8")

    call = ToolCall(
        id="call_file_ignored",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "Agent", "path": file_path, "output_mode": "files_with_matches"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["files"] == [file_path]
    assert payload["summary"]["total_matches"] == 1


def test_search_files_prefers_ripgrep_when_available(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "a.py").write_text("token = 1\n", encoding="utf-8")
    (tool_context.workspace / "b.py").write_text("token = 2\n", encoding="utf-8")

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        '{"type":"begin","data":{"path":{"text":"a.py"}}}',
                        (
                            '{"type":"match","data":{"path":{"text":"a.py"},'
                            '"lines":{"text":"token = 1\\n"},"line_number":1,'
                            '"submatches":[{"start":0,"end":5}]}}'
                        ),
                        '{"type":"end","data":{"path":{"text":"a.py"}}}',
                        '{"type":"begin","data":{"path":{"text":"b.py"}}}',
                        (
                            '{"type":"match","data":{"path":{"text":"b.py"},'
                            '"lines":{"text":"token = 2\\n"},"line_number":1,'
                            '"submatches":[{"start":0,"end":5}]}}'
                        ),
                        '{"type":"end","data":{"path":{"text":"b.py"}}}',
                        '{"type":"summary","data":{}}',
                    ]
                )
                + "\n"
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def _fake_popen(*args, **kwargs):
        assert args[0][0] == "rg"
        assert "--json" in args[0]
        return _FakeProcess()

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", _fake_popen)

    call = ToolCall(
        id="call_rg_files",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "token", "output_mode": "files_with_matches", "type": "py"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["files"] == ["a.py", "b.py"]
    assert payload["summary"]["total_matches"] == 2
    assert payload["summary"]["files_with_matches"] == 2


def test_search_files_rg_files_searched_uses_summary_searches_like_fallback(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "hit.txt").write_text("token\n", encoding="utf-8")
    (tool_context.workspace / "miss.txt").write_text("nothing\n", encoding="utf-8")

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: None)
    fallback = registry.execute(
        ToolCall(
            id="search_fallback_count",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "token", "output_mode": "content"},
        ),
        tool_context,
    )

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        '{"type":"begin","data":{"path":{"text":"hit.txt"}}}',
                        (
                            '{"type":"match","data":{"path":{"text":"hit.txt"},'
                            '"lines":{"text":"token\\n"},"line_number":1,'
                            '"submatches":[{"start":0,"end":5}]}}'
                        ),
                        '{"type":"summary","data":{"stats":{"searches":2}}}',
                    ]
                )
                + "\n"
            )
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    rg_result = registry.execute(
        ToolCall(
            id="search_rg_count",
            name=SEARCH_FILES_TOOL_NAME,
            arguments={"pattern": "token", "output_mode": "content"},
        ),
        tool_context,
    )

    assert fallback.metadata["summary"]["files_searched"] == 2
    assert rg_result.metadata["summary"]["files_searched"] == fallback.metadata["summary"]["files_searched"]


def test_search_files_accepts_ripgrep_returncode_2_with_results(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "a.py").write_text("no token here", encoding="utf-8")

    class _PartialErrorProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        '{"type":"begin","data":{"path":{"text":"a.py"}}}',
                        (
                            '{"type":"match","data":{"path":{"text":"a.py"},'
                            '"lines":{"text":"Agent from rg\\n"},"line_number":1,'
                            '"submatches":[{"start":0,"end":5}]}}'
                        ),
                        '{"type":"summary","data":{}}',
                    ]
                )
                + "\n"
            )
            self.returncode = 2

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", lambda *args, **kwargs: _PartialErrorProcess())

    call = ToolCall(id="call_rg_partial_error", name=SEARCH_FILES_TOOL_NAME, arguments={"pattern": "Agent"})
    result = registry.execute(call, tool_context)
    payload = result.metadata

    # If returncode=2 forced fallback, this would be 0 because source file does not contain "Agent".
    assert payload["summary"]["total_matches"] == 1


def test_search_files_falls_back_when_ripgrep_errors(
    registry,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tool_context.workspace / "fallback.txt").write_text("hello fallback", encoding="utf-8")

    class _FailingProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("ripgrep failed")
            self.returncode = 3

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler.subprocess, "Popen", lambda *args, **kwargs: _FailingProcess())

    call = ToolCall(
        id="call_rg_fallback",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "hello", "output_mode": "content"},
    )
    result = registry.execute(call, tool_context)
    payload = result.metadata

    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["path"] == "fallback.txt"


def test_search_files_rejects_unknown_file_type(registry, tool_context: ToolContext) -> None:
    call = ToolCall(
        id="call_invalid_type",
        name=SEARCH_FILES_TOOL_NAME,
        arguments={"pattern": "x", "type": "unknown"},
    )
    result = registry.execute(call, tool_context)

    assert result.status == "error"
    assert "Unsupported file type" in result.content
    assert "Unsupported file type" in result.metadata["error"]


def test_file_info_reports_file_metadata(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "edit.txt"
    target.write_text("hello world\nhello agent", encoding="utf-8")

    info_call = ToolCall(id="call_info", name=FILE_INFO_TOOL_NAME, arguments={"path": "edit.txt"})
    info_result = registry.execute(info_call, tool_context)
    info_payload = json.loads(info_result.content)

    assert info_payload["is_file"] is True
    assert info_payload["size"] > 0


def test_edit_file_requires_full_read_before_edit(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "edit.txt"
    target.write_text("hello world", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="edit_without_read",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "edit.txt", "old_string": "hello", "new_string": "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "file_not_read"
    assert payload["error_code"] == "file_not_read"
    assert target.read_text(encoding="utf-8") == "hello world"


def test_edit_file_rejects_partial_read_baseline(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "partial.txt"
    target.write_text("line1\nline2\nline3", encoding="utf-8")

    registry.execute(
        ToolCall(
            id="read_partial",
            name=READ_FILE_TOOL_NAME,
            arguments={"path": "partial.txt", "start_line": 1, "end_line": 1},
        ),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="edit_partial",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "partial.txt", "old_string": "line2", "new_string": "changed"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "file_not_read"
    assert payload["error_code"] == "file_not_read"
    assert target.read_text(encoding="utf-8") == "line1\nline2\nline3"


def test_edit_file_rejects_file_changed_since_read(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "changed.txt"
    target.write_text("hello world", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_changed", name=READ_FILE_TOOL_NAME, arguments={"path": "changed.txt"}),
        tool_context,
    )
    target.write_text("hello user", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="edit_changed",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "changed.txt", "old_string": "hello", "new_string": "hi"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "file_changed_since_read"
    assert payload["error_code"] == "file_changed_since_read"
    assert target.read_text(encoding="utf-8") == "hello user"


def test_edit_file_allows_consecutive_edits_after_full_read(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "consecutive.txt"
    target.write_text("alpha beta gamma", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_consecutive", name=READ_FILE_TOOL_NAME, arguments={"path": "consecutive.txt"}),
        tool_context,
    )

    first = registry.execute(
        ToolCall(
            id="edit_consecutive_first",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "consecutive.txt", "old_string": "alpha", "new_string": "one"},
        ),
        tool_context,
    )
    second = registry.execute(
        ToolCall(
            id="edit_consecutive_second",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "consecutive.txt", "old_string": "beta", "new_string": "two"},
        ),
        tool_context,
    )

    assert first.status == "success"
    assert second.status == "success"
    assert target.read_text(encoding="utf-8") == "one two gamma"


def test_edit_file_accepts_full_write_file_baseline(registry, tool_context: ToolContext) -> None:
    write_result = registry.execute(
        ToolCall(
            id="write_before_edit",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "write_only.txt", "content": "created by write_file"},
        ),
        tool_context,
    )
    assert write_result.status == "success"

    result = registry.execute(
        ToolCall(
            id="edit_after_write_only",
            name=EDIT_FILE_TOOL_NAME,
            arguments={
                "path": "write_only.txt",
                "old_string": "write_file",
                "new_string": "read_file",
            },
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "success"
    assert payload["ok"] is True
    assert (tool_context.workspace / "write_only.txt").read_text(encoding="utf-8") == "created by read_file"


def test_edit_file_rejects_append_to_unknown_existing_file_baseline(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "append_unknown.txt"
    target.write_text("known? ", encoding="utf-8")

    append_result = registry.execute(
        ToolCall(
            id="append_unknown",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "append_unknown.txt", "content": "append", "append": True},
        ),
        tool_context,
    )
    assert append_result.status == "success"

    result = registry.execute(
        ToolCall(
            id="edit_after_unknown_append",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "append_unknown.txt", "old_string": "append", "new_string": "changed"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "file_not_read"
    assert payload["error_code"] == "file_not_read"
    assert target.read_text(encoding="utf-8") == "known? append"


def test_edit_file_accepts_append_to_known_existing_file_baseline(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "append_known.txt"
    target.write_text("before ", encoding="utf-8")
    registry.execute(
        ToolCall(id="read_before_append", name=READ_FILE_TOOL_NAME, arguments={"path": "append_known.txt"}),
        tool_context,
    )

    append_result = registry.execute(
        ToolCall(
            id="append_known",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "append_known.txt", "content": "after", "append": True},
        ),
        tool_context,
    )
    assert append_result.status == "success"

    result = registry.execute(
        ToolCall(
            id="edit_after_known_append",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "append_known.txt", "old_string": "after", "new_string": "changed"},
        ),
        tool_context,
    )

    assert result.status == "success"
    assert target.read_text(encoding="utf-8") == "before changed"


def test_edit_file_replace_all_replaces_every_match(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "replace_all.txt"
    target.write_text("hello world\nhello agent", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_replace_all", name=READ_FILE_TOOL_NAME, arguments={"path": "replace_all.txt"}),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="edit_replace_all",
            name=EDIT_FILE_TOOL_NAME,
            arguments={
                "path": "replace_all.txt",
                "old_string": "hello",
                "new_string": "hi",
                "replace_all": True,
            },
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "success"
    assert payload["replaced_count"] == 2
    assert target.read_text(encoding="utf-8") == "hi world\nhi agent"


def test_edit_file_success_returns_changed_files_and_diff_metadata(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "diff.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_diff", name=READ_FILE_TOOL_NAME, arguments={"path": "diff.txt"}),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="edit_diff",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "diff.txt", "old_string": "beta", "new_string": "BETTA"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert payload["replaced_count"] == 1
    assert result.metadata["changed_files"] == ["diff.txt"]
    assert result.metadata["operation"] == "edit_file"
    assert "-beta" in result.metadata["diff"]
    assert "+BETTA" in result.metadata["diff"]
    assert result.metadata["additions"] == 1
    assert result.metadata["deletions"] == 1


def test_edit_file_preserves_crlf_when_old_string_uses_lf(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "crlf.txt"
    target.write_bytes(b"first\r\nsecond\r\nthird\r\n")

    registry.execute(
        ToolCall(id="read_crlf", name=READ_FILE_TOOL_NAME, arguments={"path": "crlf.txt"}),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="edit_crlf",
            name=EDIT_FILE_TOOL_NAME,
            arguments={"path": "crlf.txt", "old_string": "second\nthird", "new_string": "SECOND\nTHIRD"},
        ),
        tool_context,
    )

    assert result.status == "success"
    assert target.read_bytes() == b"first\r\nSECOND\r\nTHIRD\r\n"
    assert result.metadata["line_ending"] == "crlf"


def test_write_file_create_new_file_without_read(registry, tool_context: ToolContext) -> None:
    result = registry.execute(
        ToolCall(
            id="write_new",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "new.txt", "content": "created"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "success"
    assert payload["written_chars"] == 7
    assert result.metadata["changed_files"] == ["new.txt"]
    assert (tool_context.workspace / "new.txt").read_text(encoding="utf-8") == "created"


def test_write_file_overwrite_existing_requires_read_baseline(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "existing.txt"
    target.write_text("original", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="overwrite_without_read",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "existing.txt", "content": "changed"},
        ),
        tool_context,
    )

    payload = json.loads(result.content)
    assert result.status == "error"
    assert result.error_code == "file_not_read"
    assert payload["error_code"] == "file_not_read"
    assert target.read_text(encoding="utf-8") == "original"


def test_write_file_overwrite_existing_after_full_read_succeeds(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "overwrite_after_read.txt"
    target.write_text("original", encoding="utf-8")

    registry.execute(
        ToolCall(id="read_overwrite", name=READ_FILE_TOOL_NAME, arguments={"path": "overwrite_after_read.txt"}),
        tool_context,
    )
    result = registry.execute(
        ToolCall(
            id="overwrite_after_read",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "overwrite_after_read.txt", "content": "changed"},
        ),
        tool_context,
    )

    assert result.status == "success"
    assert result.metadata["changed_files"] == ["overwrite_after_read.txt"]
    assert target.read_text(encoding="utf-8") == "changed"


def test_write_file_append_returns_changed_files_metadata(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "append.txt"
    target.write_text("a", encoding="utf-8")

    result = registry.execute(
        ToolCall(
            id="append_without_read",
            name=WRITE_FILE_TOOL_NAME,
            arguments={"path": "append.txt", "content": "b", "append": True},
        ),
        tool_context,
    )

    assert result.status == "success"
    assert result.metadata["changed_files"] == ["append.txt"]
    assert result.metadata["operation"] == "write_file"
    assert result.metadata["append"] is True
    assert target.read_text(encoding="utf-8") == "ab"


def test_compress_memory_writes_note(registry, tool_context: ToolContext) -> None:
    call = ToolCall(
        id="call_mem",
        name=COMPRESS_MEMORY_TOOL_NAME,
        arguments={"core_information": "current decision and progress"},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)
    assert payload["ok"] is True
    assert payload["saved_notes"] == 1
    assert tool_context.shared_state["memory_notes"][0]["core_information"] == "current decision and progress"


def test_todo_finish_guard(registry, tool_context: ToolContext) -> None:
    create_todo = ToolCall(
        id="call1",
        name=TASK_LIST_TOOL_NAME,
        arguments={"todos": [{"title": "task 1", "status": "pending", "priority": "high"}]},
    )
    registry.execute(create_todo, tool_context)

    finish_call = ToolCall(id="call2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})
    finish_result = registry.execute(finish_call, tool_context)
    payload = json.loads(finish_result.content)

    assert finish_result.status == "error"
    assert finish_result.directive == ToolDirective.CONTINUE
    assert payload["error_code"] == "todo_incomplete"


def test_ask_user_sets_wait_directive(registry, tool_context: ToolContext) -> None:
    call = ToolCall(id="call1", name=ASK_USER_TOOL_NAME, arguments={"question": "Pick one", "options": ["A", "B"]})
    result = registry.execute(call, tool_context)
    assert result.directive == ToolDirective.WAIT_USER
    assert result.metadata["question"] == "Pick one"


def test_unknown_tool_raises(registry, tool_context: ToolContext) -> None:
    call = ToolCall(id="call1", name="missing", arguments={})
    with pytest.raises(ToolNotFoundError):
        registry.execute(call, tool_context)
