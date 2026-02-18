from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FILE_STR_REPLACE_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WORKSPACE_GREP_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from v_agent.tools import ToolContext, build_default_registry
from v_agent.tools.registry import ToolNotFoundError
from v_agent.types import ToolCall, ToolDirective


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, shared_state={"todo_list": []}, cycle_index=1)


def test_workspace_write_and_read(registry, tool_context: ToolContext) -> None:
    write_call = ToolCall(id="call1", name=WRITE_FILE_TOOL_NAME, arguments={"path": "notes/test.txt", "content": "hello"})
    write_result = registry.execute(write_call, tool_context)
    assert write_result.status == "success"

    read_call = ToolCall(id="call2", name=READ_FILE_TOOL_NAME, arguments={"path": "notes/test.txt"})
    read_result = registry.execute(read_call, tool_context)
    payload = json.loads(read_result.content)
    assert payload["content"] == "hello"


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


def test_workspace_grep(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.txt").write_text("hello world\nsecond line", encoding="utf-8")
    call = ToolCall(id="call1", name=WORKSPACE_GREP_TOOL_NAME, arguments={"pattern": "hello", "output_mode": "content"})
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)
    assert payload["summary"]["total_matches"] == 1
    assert payload["matches"][0]["line"] == 1


def test_workspace_grep_supports_files_with_matches_mode(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "a.py").write_text("TOKEN = 1", encoding="utf-8")
    (tool_context.workspace / "b.py").write_text("token = 2", encoding="utf-8")
    (tool_context.workspace / "c.md").write_text("no hit", encoding="utf-8")

    call = ToolCall(
        id="call_files",
        name=WORKSPACE_GREP_TOOL_NAME,
        arguments={"pattern": "token", "output_mode": "files_with_matches", "i": True, "type": "py"},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["files"] == ["a.py", "b.py"]
    assert payload["summary"]["files_with_matches"] == 2
    assert payload["summary"]["total_matches"] == 2


def test_workspace_grep_supports_count_mode(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "logs").mkdir(parents=True, exist_ok=True)
    (tool_context.workspace / "logs" / "x.log").write_text("err\nok\nerr", encoding="utf-8")
    (tool_context.workspace / "logs" / "y.log").write_text("err", encoding="utf-8")

    call = ToolCall(
        id="call_count",
        name=WORKSPACE_GREP_TOOL_NAME,
        arguments={"pattern": "err", "output_mode": "count", "path": "logs", "type": "log"},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert payload["file_counts"] == {"logs/x.log": 2, "logs/y.log": 1}
    assert payload["summary"]["total_matches"] == 3


def test_workspace_grep_supports_context_lines(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "ctx.txt").write_text("line1\nhit\nline3", encoding="utf-8")
    call = ToolCall(
        id="call_ctx",
        name=WORKSPACE_GREP_TOOL_NAME,
        arguments={"pattern": "hit", "output_mode": "content", "c": 1, "n": True},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert [row["line"] for row in payload["matches"]] == [1, 2, 3]
    assert payload["matches"][0]["is_match"] is False
    assert payload["matches"][1]["is_match"] is True


def test_workspace_grep_supports_multiline_and_head_limit(registry, tool_context: ToolContext) -> None:
    (tool_context.workspace / "multi.txt").write_text("start\nalpha\nbeta\nend", encoding="utf-8")
    call = ToolCall(
        id="call_multi",
        name=WORKSPACE_GREP_TOOL_NAME,
        arguments={"pattern": "alpha\\nbeta", "multiline": True, "head_limit": 1},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert len(payload["matches"]) == 1
    assert payload["head_limit"] == 1
    assert payload["summary"]["total_matches"] == 1


def test_workspace_grep_rejects_unknown_file_type(registry, tool_context: ToolContext) -> None:
    call = ToolCall(
        id="call_invalid_type",
        name=WORKSPACE_GREP_TOOL_NAME,
        arguments={"pattern": "x", "type": "unknown"},
    )
    result = registry.execute(call, tool_context)
    payload = json.loads(result.content)

    assert result.status == "error"
    assert "Unsupported file type" in payload["error"]


def test_file_info_and_string_replace(registry, tool_context: ToolContext) -> None:
    target = tool_context.workspace / "edit.txt"
    target.write_text("hello world\nhello agent", encoding="utf-8")

    info_call = ToolCall(id="call_info", name=FILE_INFO_TOOL_NAME, arguments={"path": "edit.txt"})
    info_result = registry.execute(info_call, tool_context)
    info_payload = json.loads(info_result.content)
    assert info_payload["is_file"] is True
    assert info_payload["size"] > 0

    replace_call = ToolCall(
        id="call_replace",
        name=FILE_STR_REPLACE_TOOL_NAME,
        arguments={"path": "edit.txt", "old_str": "hello", "new_str": "hi", "replace_all": True},
    )
    replace_result = registry.execute(replace_call, tool_context)
    replace_payload = json.loads(replace_result.content)
    assert replace_payload["replaced_count"] == 2
    assert target.read_text(encoding="utf-8") == "hi world\nhi agent"


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
        name=TODO_WRITE_TOOL_NAME,
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
