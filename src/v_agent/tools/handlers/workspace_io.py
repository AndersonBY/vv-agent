from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import resolve_workspace_path, to_json
from v_agent.types import ToolExecutionResult

READ_FILE_MAX_LINES = 2_000
READ_FILE_MAX_CHARS = 50_000


def list_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    root = resolve_workspace_path(context, str(arguments.get("path", ".")))
    glob_pattern = str(arguments.get("glob", "**/*"))
    include_hidden = bool(arguments.get("include_hidden", False))

    files: list[str] = []
    for candidate in root.glob(glob_pattern):
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(context.workspace).as_posix()
        if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
            continue
        files.append(rel)

    files.sort()
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json({"files": files, "count": len(files)}),
    )


def read_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = resolve_workspace_path(context, str(arguments["path"]))
    if not target.exists() or not target.is_file():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"file not found: {target.relative_to(context.workspace).as_posix()}"}),
        )

    try:
        start_line = max(int(arguments.get("start_line", 1)), 1)
        end_line_raw = arguments.get("end_line")
        end_line_int = int(end_line_raw) if end_line_raw is not None else None
    except (TypeError, ValueError):
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`start_line`/`end_line` must be integers"}),
        )

    if end_line_int is not None:
        end_line_int = max(end_line_int, start_line)

    show_line_numbers = bool(arguments.get("show_line_numbers", False))

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start_idx = max(start_line - 1, 0)
    end_idx = len(lines) if end_line_int is None else max(end_line_int, start_idx)
    selected = lines[start_idx:end_idx]
    selected_line_count = len(selected)
    actual_start_line = start_idx + 1
    actual_end_line = start_idx + selected_line_count

    rendered_lines = selected
    if show_line_numbers:
        rendered_lines = [f"{start_idx + offset + 1}: {line}" for offset, line in enumerate(selected)]
    content = "\n".join(rendered_lines)

    if selected_line_count > READ_FILE_MAX_LINES or len(content) > READ_FILE_MAX_CHARS:
        total_lines = len(lines)
        total_chars = len(text)
        suggested_start = min(start_line, total_lines)
        suggested_end = min(suggested_start + READ_FILE_MAX_LINES - 1, total_lines)
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            content=to_json(
                {
                    "path": target.relative_to(context.workspace).as_posix(),
                    "start_line": actual_start_line,
                    "end_line": actual_end_line,
                    "show_line_numbers": show_line_numbers,
                    "content": None,
                    "file_info": {
                        "total_lines": total_lines,
                        "total_chars": total_chars,
                    },
                    "requested": {
                        "line_count": selected_line_count,
                        "char_count": len(content),
                    },
                    "limits": {
                        "max_lines": READ_FILE_MAX_LINES,
                        "max_chars": READ_FILE_MAX_CHARS,
                    },
                    "suggested_range": {
                        "start_line": suggested_start,
                        "end_line": suggested_end,
                    },
                    "message": "Requested read exceeds limits. Use start_line/end_line for a smaller range.",
                }
            ),
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "path": target.relative_to(context.workspace).as_posix(),
                "start_line": actual_start_line,
                "end_line": actual_end_line,
                "show_line_numbers": show_line_numbers,
                "content": content,
            }
        ),
    )


def write_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = resolve_workspace_path(context, str(arguments["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)

    content = str(arguments.get("content", ""))
    append = bool(arguments.get("append", False))
    leading_newline = bool(arguments.get("leading_newline", False))
    trailing_newline = bool(arguments.get("trailing_newline", False))

    write_content = content
    if append:
        prefix = "\n" if leading_newline else ""
        suffix = "\n" if trailing_newline else ""
        write_content = f"{prefix}{content}{suffix}"

    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as file_obj:
        file_obj.write(write_content)

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "ok": True,
                "path": target.relative_to(context.workspace).as_posix(),
                "append": append,
                "leading_newline": leading_newline if append else False,
                "trailing_newline": trailing_newline if append else False,
                "written_chars": len(write_content),
            }
        ),
    )


def file_info(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = resolve_workspace_path(context, str(arguments["path"]))
    if not target.exists():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"path not found: {target.relative_to(context.workspace).as_posix()}"}),
        )

    stat = target.stat()
    payload: dict[str, Any] = {
        "path": target.relative_to(context.workspace).as_posix(),
        "exists": True,
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }
    if target.is_file():
        payload["suffix"] = target.suffix
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(payload),
    )


def file_str_replace(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = resolve_workspace_path(context, str(arguments["path"]))
    if not target.exists() or not target.is_file():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"file not found: {target.relative_to(context.workspace).as_posix()}"}),
        )

    old_str = str(arguments.get("old_str", ""))
    if not old_str:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`old_str` cannot be empty"}),
        )
    new_str = str(arguments.get("new_str", ""))
    replace_all = bool(arguments.get("replace_all", False))
    max_replacements = int(arguments.get("max_replacements", 1))
    max_replacements = max(max_replacements, 1)

    text = target.read_text(encoding="utf-8", errors="replace")
    occurrence_count = text.count(old_str)
    if occurrence_count == 0:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`old_str` not found in file"}),
        )

    if replace_all:
        replaced_text = text.replace(old_str, new_str)
        replaced_count = occurrence_count
    else:
        replaced_text = text.replace(old_str, new_str, max_replacements)
        replaced_count = min(occurrence_count, max_replacements)

    target.write_text(replaced_text, encoding="utf-8")

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "ok": True,
                "path": target.relative_to(context.workspace).as_posix(),
                "replaced_count": replaced_count,
            }
        ),
    )
