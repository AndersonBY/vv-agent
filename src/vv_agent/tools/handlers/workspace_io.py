from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import ToolExecutionResult

READ_FILE_MAX_LINES = 2_000
READ_FILE_MAX_CHARS = 50_000
LIST_FILES_DEFAULT_MAX_RESULTS = 500
LIST_FILES_HARD_MAX_RESULTS = 5_000
LIST_FILES_DEFAULT_SCAN_LIMIT = 50_000
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


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob pattern supporting ``**``."""
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i:i + 3] == "**/":
            parts.append("(?:.+/)?")
            i += 3
        elif pattern[i:i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    regex = "^" + "".join(parts) + "$"
    return re.match(regex, path) is not None


def _list_files_local_fast(
    context: ToolContext,
    *,
    path: str,
    glob_pattern: str,
    include_hidden: bool,
    include_ignored: bool,
    max_results: int,
    scan_limit: int,
) -> tuple[list[str], int, bool, bool, list[dict[str, Any]]]:
    workspace_root = context.workspace.resolve()
    base_path = context.resolve_workspace_path(path)
    if not base_path.exists():
        raise ValueError(f"path not found: {path}")
    if not base_path.is_dir():
        raise ValueError(f"not a directory: {path}")

    root_listing = _is_workspace_root(path)
    ignored_roots_summary: list[dict[str, Any]] = []
    ignored_roots_set: set[str] = set()
    if root_listing and not include_ignored:
        try:
            with os.scandir(base_path) as iterator:
                for entry in iterator:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    token = entry.name.lower()
                    if token in LIST_FILES_IGNORED_ROOTS:
                        ignored_roots_set.add(token)
                        ignored_roots_summary.append({"path": entry.name})
        except OSError:
            # Ignore listing errors from inaccessible roots.
            pass

    matched_files: list[str] = []
    matched_count = 0
    scanned_count = 0
    scan_limited = False
    for current_root, dirs, filenames in os.walk(
        base_path,
        topdown=True,
        onerror=lambda _e: None,
        followlinks=False,
    ):
        dirs.sort(key=str.lower)
        filenames.sort(key=str.lower)

        if not include_hidden:
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            filenames = [name for name in filenames if not name.startswith(".")]

        current_path = Path(current_root)
        if root_listing and not include_ignored and current_path == base_path:
            dirs[:] = [name for name in dirs if name.lower() not in ignored_roots_set]

        rel_dir = current_path.relative_to(base_path).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        for filename in filenames:
            scanned_count += 1
            if scanned_count > scan_limit:
                scan_limited = True
                break

            rel_from_base = f"{rel_dir}/{filename}" if rel_dir else filename
            if not _glob_match(rel_from_base, glob_pattern):
                continue

            try:
                rel_workspace = (current_path / filename).relative_to(workspace_root).as_posix()
            except (OSError, ValueError):
                continue

            matched_count += 1
            if len(matched_files) < max_results:
                matched_files.append(rel_workspace)

        if scan_limited:
            break

    matched_files.sort()
    truncated = matched_count > len(matched_files) or scan_limited
    return matched_files, matched_count, truncated, scan_limited, ignored_roots_summary


def list_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    backend = context.workspace_backend
    path = str(arguments.get("path", "."))
    glob_pattern = str(arguments.get("glob", "**/*"))
    include_hidden = bool(arguments.get("include_hidden", False))
    include_ignored = bool(arguments.get("include_ignored", False))
    max_results_raw = arguments.get("max_results", LIST_FILES_DEFAULT_MAX_RESULTS)
    scan_limit_raw = arguments.get("scan_limit", LIST_FILES_DEFAULT_SCAN_LIMIT)
    try:
        max_results = int(max_results_raw)
        scan_limit = int(scan_limit_raw)
    except (TypeError, ValueError):
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": "`max_results` and `scan_limit` must be integers"}),
        )
    max_results = min(max(max_results, 1), LIST_FILES_HARD_MAX_RESULTS)
    scan_limit = max(scan_limit, max_results)

    ignored_roots_summary: list[dict[str, Any]] = []
    scan_limited = False
    local_root = getattr(backend, "root", None)
    if isinstance(local_root, Path):
        files, total_count, truncated, scan_limited, ignored_roots_summary = _list_files_local_fast(
            context,
            path=path,
            glob_pattern=glob_pattern,
            include_hidden=include_hidden,
            include_ignored=include_ignored,
            max_results=max_results,
            scan_limit=scan_limit,
        )
    else:
        all_files = backend.list_files(path, glob_pattern)
        if not include_hidden:
            all_files = [
                f for f in all_files
                if not any(part.startswith(".") for part in Path(f).parts)
            ]
        total_count = len(all_files)
        files = all_files[:max_results]
        truncated = total_count > len(files)

    payload: dict[str, Any] = {
        "files": files,
        "count": total_count,
        "returned_count": len(files),
        "truncated": truncated,
        "max_results": max_results,
    }
    if total_count > len(files):
        payload["remaining_count"] = total_count - len(files)
    if scan_limited:
        payload["count_is_estimate"] = True
        payload["scan_limit"] = scan_limit
        payload["message"] = (
            "Listing stopped early due to scan limit. Narrow `path`/`glob` "
            "or increase `scan_limit` for more complete results."
        )
    if ignored_roots_summary:
        payload["ignored_roots"] = ignored_roots_summary
        ignored_message = (
            "Common dependency/cache directories are summarized by default. "
            "List those directories explicitly when needed."
        )
        if payload.get("message"):
            payload["message"] = f"{payload['message']} {ignored_message}"
        else:
            payload["message"] = ignored_message

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
