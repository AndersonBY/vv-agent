from __future__ import annotations

from pathlib import Path
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import ToolExecutionResult

READ_FILE_MAX_LINES = 2_000
READ_FILE_MAX_CHARS = 50_000
LIST_FILES_DEFAULT_MAX_RESULTS = 500
LIST_FILES_HARD_MAX_RESULTS = 5_000
LIST_FILES_IGNORED_ROOTS = frozenset(
    {
        ".venv",
        "venv",
        "node_modules",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".cache",
        "target",
        "vendor",
    }
)


def _is_workspace_root(path: str) -> bool:
    normalized = path.strip()
    if not normalized:
        return True
    return Path(normalized).as_posix() in {".", ""}


def _top_level_segment(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return ""
    return str(parts[0]).strip().lower()


def list_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments.get("path", "."))
    glob_pattern = str(arguments.get("glob", "**/*"))
    include_hidden = bool(arguments.get("include_hidden", False))
    include_ignored = bool(arguments.get("include_ignored", False))
    max_results_raw = arguments.get("max_results", LIST_FILES_DEFAULT_MAX_RESULTS)
    try:
        max_results = int(max_results_raw)
    except (TypeError, ValueError):
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`max_results` must be an integer"}),
        )
    max_results = min(max(max_results, 1), LIST_FILES_HARD_MAX_RESULTS)

    all_files = backend.list_files(path, glob_pattern)
    if not include_hidden:
        all_files = [
            f for f in all_files
            if not any(part.startswith(".") for part in Path(f).parts)
        ]

    ignored_roots_summary: list[dict[str, Any]] = []
    if _is_workspace_root(path) and not include_ignored:
        visible_files: list[str] = []
        ignored_counts: dict[str, int] = {}
        for rel_path in all_files:
            top = _top_level_segment(rel_path)
            if top in LIST_FILES_IGNORED_ROOTS:
                ignored_counts[top] = ignored_counts.get(top, 0) + 1
                continue
            visible_files.append(rel_path)
        if ignored_counts:
            ignored_roots_summary = [
                {"path": key, "count": ignored_counts[key]}
                for key in sorted(ignored_counts.keys())
            ]
        all_files = visible_files

    total_count = len(all_files)
    files = all_files[:max_results]
    payload: dict[str, Any] = {
        "files": files,
        "count": total_count,
        "returned_count": len(files),
        "truncated": total_count > len(files),
        "max_results": max_results,
    }
    if total_count > len(files):
        payload["remaining_count"] = total_count - len(files)
    if ignored_roots_summary:
        payload["ignored_roots"] = ignored_roots_summary
        payload["message"] = (
            "Common dependency/cache directories are summarized by default. "
            "List those directories explicitly when needed."
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(payload),
    )


def read_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments["path"])

    if not backend.is_file(path):
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"file not found: {path}"}),
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

    text = backend.read_text(path)
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
                    "path": path,
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
                "path": path,
                "start_line": actual_start_line,
                "end_line": actual_end_line,
                "show_line_numbers": show_line_numbers,
                "content": content,
            }
        ),
    )


def write_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments["path"])

    content = str(arguments.get("content", ""))
    append = bool(arguments.get("append", False))
    leading_newline = bool(arguments.get("leading_newline", False))
    trailing_newline = bool(arguments.get("trailing_newline", False))

    write_content = content
    if append:
        prefix = "\n" if leading_newline else ""
        suffix = "\n" if trailing_newline else ""
        write_content = f"{prefix}{content}{suffix}"

    backend.write_text(path, write_content, append=append)

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "ok": True,
                "path": path,
                "append": append,
                "leading_newline": leading_newline if append else False,
                "trailing_newline": trailing_newline if append else False,
                "written_chars": len(write_content),
            }
        ),
    )


def file_info(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments["path"])
    info = backend.file_info(path)

    if info is None:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"path not found: {path}"}),
        )

    payload: dict[str, Any] = {
        "path": info.path,
        "exists": True,
        "is_file": info.is_file,
        "is_dir": info.is_dir,
        "size": info.size,
        "modified_at": info.modified_at,
    }
    if info.is_file:
        payload["suffix"] = info.suffix
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(payload),
    )


def file_str_replace(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments["path"])

    if not backend.is_file(path):
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"file not found: {path}"}),
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

    text = backend.read_text(path)
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

    backend.write_text(path, replaced_text)

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "ok": True,
                "path": path,
                "replaced_count": replaced_count,
            }
        ),
    )
