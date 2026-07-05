from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.tools.handlers.sensitive_paths import is_sensitive_path
from vv_agent.types import ToolExecutionResult

READ_FILE_MAX_LINES = 2_000
READ_FILE_MAX_CHARS = 50_000
FILE_BASELINES_STATE_KEY = "_workspace_file_baselines"
EDIT_DIFF_MAX_CHARS = 12_000
UTF8_BOM = b"\xef\xbb\xbf"
LIST_FILES_DEFAULT_MAX_RESULTS = 100
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
_RG_EXECUTABLE_CACHE: str | None | bool = None


def _is_workspace_root(path: str) -> bool:
    normalized = path.strip()
    if not normalized:
        return True
    return Path(normalized).as_posix() in {".", ""}


def _compile_glob_pattern(pattern: str) -> re.Pattern[str]:
    """Compile glob once; callers can reuse the regex across many paths."""
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
    return re.compile(regex)


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized == ".":
        return ""
    return normalized


def _format_output_path(context: ToolContext, candidate_path: Path) -> str:
    resolved = candidate_path.resolve()
    workspace_root = context.workspace.resolve()
    try:
        rel = resolved.relative_to(workspace_root).as_posix()
        return rel or "."
    except ValueError:
        return str(resolved)


def _baseline_key(path: str) -> str:
    return _normalize_relative_path(path)


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _line_ending_label(text: str) -> str:
    return "crlf" if "\r\n" in text else "lf"


def _get_file_baselines(context: ToolContext) -> dict[str, dict[str, Any]]:
    raw = context.shared_state.setdefault(FILE_BASELINES_STATE_KEY, {})
    if not isinstance(raw, dict):
        raw = {}
        context.shared_state[FILE_BASELINES_STATE_KEY] = raw
    return raw


def _decode_workspace_text(raw: bytes) -> tuple[str, bool]:
    has_bom = raw.startswith(UTF8_BOM)
    payload = raw[len(UTF8_BOM):] if has_bom else raw
    try:
        return payload.decode("utf-8"), has_bom
    except UnicodeDecodeError as exc:
        raise ValueError("unsupported_encoding") from exc


def _encode_workspace_text(text: str, *, has_bom: bool) -> bytes:
    payload = text.encode("utf-8")
    return UTF8_BOM + payload if has_bom else payload


def _record_file_baseline(
    context: ToolContext,
    *,
    path: str,
    raw: bytes,
    text: str,
    is_partial: bool,
) -> None:
    baselines = _get_file_baselines(context)
    baselines[_baseline_key(path)] = {
        "hash": _content_hash(raw),
        "size": len(raw),
        "line_ending": _line_ending_label(text),
        "is_partial": bool(is_partial),
    }


def _baseline_error(context: ToolContext, *, path: str, current_raw: bytes) -> str | None:
    baseline = _get_file_baselines(context).get(_baseline_key(path))
    if not baseline or baseline.get("is_partial"):
        return "file_not_read"
    if baseline.get("hash") != _content_hash(current_raw):
        return "file_changed_since_read"
    return None


def _workspace_error(message: str, *, error_code: str, **details: Any) -> ToolExecutionResult:
    payload: dict[str, Any] = {"error": message, "error_code": error_code, "message": message}
    payload.update(details)
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        error_code=error_code,
        content=to_json(payload),
        metadata={"error_code": error_code, **details},
    )


def _bounded_unified_diff(path: str, before: str, after: str) -> tuple[str, bool, int, int]:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            lineterm="",
        )
    )
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    diff_text = "".join(diff_lines)
    if len(diff_text) <= EDIT_DIFF_MAX_CHARS:
        return diff_text, False, additions, deletions
    return diff_text[:EDIT_DIFF_MAX_CHARS], True, additions, deletions


def _resolve_rg_executable() -> str | None:
    """Locate ripgrep executable shipped by environment or available in PATH."""
    global _RG_EXECUTABLE_CACHE

    cached = _RG_EXECUTABLE_CACHE
    if isinstance(cached, str):
        return cached
    if cached is False:
        return None

    script_names = ("rg.exe", "rg.cmd", "rg") if os.name == "nt" else ("rg",)
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        scripts_root = Path(scripts_dir)
        for script_name in script_names:
            candidate = scripts_root / script_name
            if candidate.exists():
                _RG_EXECUTABLE_CACHE = str(candidate)
                return _RG_EXECUTABLE_CACHE

    resolved = shutil.which("rg")
    if resolved:
        _RG_EXECUTABLE_CACHE = resolved
        return resolved

    _RG_EXECUTABLE_CACHE = False
    return None


def _list_files_local_rg(
    *,
    context: ToolContext,
    base_path: Path,
    base_is_workspace_root: bool,
    glob_pattern: str,
    include_hidden: bool,
    include_ignored: bool,
    ignored_root_names: list[str],
    max_results: int,
    scan_limit: int,
    glob_regex: re.Pattern[str],
) -> tuple[list[str], int, bool, bool] | None:
    rg_executable = _resolve_rg_executable()
    if not rg_executable:
        return None

    command = [
        rg_executable,
        "--files",
        "--null",
        "--no-messages",
        "--no-ignore",
        "--no-ignore-vcs",
    ]
    if include_hidden:
        command.append("--hidden")
    if glob_pattern and glob_pattern != "**/*":
        command.extend(["--glob", glob_pattern])
    if base_is_workspace_root and not include_ignored:
        for ignored_name in ignored_root_names:
            command.extend(["--glob", f"!{ignored_name}/**"])
    command.append(".")

    try:
        process = subprocess.Popen(
            command,
            cwd=base_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None

    assert process.stdout is not None

    matched_files: list[str] = []
    matched_count = 0
    scanned_count = 0
    scan_limited = False
    should_stop = False
    remainder = b""

    while True:
        chunk = process.stdout.read(64 * 1024)
        if not chunk:
            break

        payload = remainder + chunk
        entries = payload.split(b"\x00")
        remainder = entries.pop()

        for raw_entry in entries:
            if not raw_entry:
                continue

            scanned_count += 1
            if scanned_count > scan_limit:
                scan_limited = True
                should_stop = True
                break

            rel_from_base = _normalize_relative_path(raw_entry.decode("utf-8", errors="replace"))
            if not rel_from_base:
                continue
            if not glob_regex.match(rel_from_base):
                continue

            candidate_path = base_path / rel_from_base
            rel_workspace = _format_output_path(context, candidate_path)

            matched_count += 1
            if len(matched_files) < max_results:
                matched_files.append(rel_workspace)

        if should_stop:
            break

    if should_stop:
        process.terminate()

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)

    if not scan_limited and process.returncode not in (0, 1):
        return None

    if remainder and not scan_limited:
        rel_from_base = _normalize_relative_path(remainder.decode("utf-8", errors="replace"))
        scanned_count += 1
        if scanned_count <= scan_limit and rel_from_base and glob_regex.match(rel_from_base):
            candidate_path = base_path / rel_from_base
            rel_workspace = _format_output_path(context, candidate_path)
            matched_count += 1
            if len(matched_files) < max_results:
                matched_files.append(rel_workspace)
        elif scanned_count > scan_limit:
            scan_limited = True

    matched_files.sort()
    truncated = matched_count > len(matched_files) or scan_limited
    return matched_files, matched_count, truncated, scan_limited


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
    base_path = context.resolve_workspace_path(path)
    if not base_path.exists():
        raise ValueError(f"path not found: {path}")
    if not base_path.is_dir():
        raise ValueError(f"not a directory: {path}")

    root_listing = _is_workspace_root(path)
    ignored_roots_summary: list[dict[str, Any]] = []
    ignored_roots_set: set[str] = set()
    ignored_root_names: list[str] = []
    if root_listing and not include_ignored:
        try:
            with os.scandir(base_path) as iterator:
                for entry in iterator:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    token = entry.name.lower()
                    if token in LIST_FILES_IGNORED_ROOTS:
                        ignored_roots_set.add(token)
                        ignored_root_names.append(entry.name)
                        ignored_roots_summary.append({"path": entry.name})
        except OSError:
            # Ignore listing errors from inaccessible roots.
            pass

    glob_regex = _compile_glob_pattern(glob_pattern)
    rg_result = _list_files_local_rg(
        context=context,
        base_path=base_path,
        base_is_workspace_root=root_listing,
        glob_pattern=glob_pattern,
        include_hidden=include_hidden,
        include_ignored=include_ignored,
        ignored_root_names=ignored_root_names,
        max_results=max_results,
        scan_limit=scan_limit,
        glob_regex=glob_regex,
    )
    if rg_result is not None:
        files, total_count, truncated, scan_limited = rg_result
        return files, total_count, truncated, scan_limited, ignored_roots_summary

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
            if not glob_regex.match(rel_from_base):
                continue

            rel_workspace = _format_output_path(context, current_path / filename)

            matched_count += 1
            if len(matched_files) < max_results:
                matched_files.append(rel_workspace)

        if scan_limited:
            break

    matched_files.sort()
    truncated = matched_count > len(matched_files) or scan_limited
    return matched_files, matched_count, truncated, scan_limited, ignored_roots_summary


def _sort_find_files(context: ToolContext, files: list[str], sort: str) -> list[str]:
    if sort == "path_asc":
        return sorted(files)

    def key(file_path: str) -> tuple[float, str]:
        try:
            stat = context.resolve_workspace_path(file_path).stat()
            return (-stat.st_mtime, file_path)
        except OSError:
            return (0.0, file_path)

    return sorted(files, key=key)


def find_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    if "pattern" in arguments:
        return _workspace_error(
            "`glob` is required for file patterns; `pattern` is not supported",
            error_code="invalid_arguments",
        )

    backend = context.workspace_backend
    path = str(arguments.get("path", "."))
    glob_pattern = str(arguments.get("glob", "**/*"))
    include_hidden = bool(arguments.get("include_hidden", False))
    include_ignored = bool(arguments.get("include_ignored", False))
    include_sensitive = bool(arguments.get("include_sensitive", False))
    sort = str(arguments.get("sort", "modified_desc"))
    max_results_raw = arguments.get("max_results", LIST_FILES_DEFAULT_MAX_RESULTS)
    offset_raw = arguments.get("offset", 0)
    scan_limit_raw = arguments.get("scan_limit", LIST_FILES_DEFAULT_SCAN_LIMIT)
    if sort not in {"modified_desc", "path_asc"}:
        return _workspace_error(
            "`sort` must be modified_desc or path_asc",
            error_code="invalid_arguments",
        )
    try:
        max_results = int(max_results_raw)
        offset = int(offset_raw)
        scan_limit = int(scan_limit_raw)
    except (TypeError, ValueError):
        return _workspace_error(
            "`max_results`, `offset`, and `scan_limit` must be integers",
            error_code="invalid_arguments",
        )
    max_results = min(max(max_results, 1), LIST_FILES_HARD_MAX_RESULTS)
    offset = max(offset, 0)
    scan_limit = max(scan_limit, max_results + offset)

    ignored_roots_summary: list[dict[str, Any]] = []
    scan_limited = False
    effective_sort = sort
    local_root = getattr(backend, "root", None)
    if isinstance(local_root, Path):
        files, total_count, truncated, scan_limited, ignored_roots_summary = _list_files_local_fast(
            context,
            path=path,
            glob_pattern=glob_pattern,
            include_hidden=include_hidden,
            include_ignored=include_ignored,
            max_results=scan_limit,
            scan_limit=scan_limit,
        )
    else:
        all_files = backend.list_files(path, glob_pattern)
        if not include_hidden:
            all_files = [
                f for f in all_files
                if not any(part.startswith(".") for part in Path(f).parts)
            ]
        all_files = sorted(all_files)
        scan_limited = len(all_files) > scan_limit
        files = all_files[:scan_limit] if scan_limited else all_files
        total_count = len(files)
        truncated = scan_limited
        effective_sort = "path_asc"

    sensitive_files_omitted = 0
    if not include_sensitive:
        kept_files: list[str] = []
        for file_path in files:
            if is_sensitive_path(file_path):
                sensitive_files_omitted += 1
            else:
                kept_files.append(file_path)
        files = kept_files
        total_count = max(0, total_count - sensitive_files_omitted)

    files = _sort_find_files(context, files, effective_sort) if isinstance(local_root, Path) else sorted(files)
    visible_files = files[offset:offset + max_results]
    truncated = offset + len(visible_files) < total_count or scan_limited

    payload: dict[str, Any] = {
        "files": visible_files,
        "count": total_count,
        "returned_count": len(visible_files),
        "truncated": truncated,
        "max_results": max_results,
        "offset": offset,
        "sort": effective_sort,
    }
    if total_count > offset + len(visible_files):
        payload["remaining_count"] = total_count - offset - len(visible_files)
    if sensitive_files_omitted:
        payload["sensitive_files_omitted"] = sensitive_files_omitted
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
        raw = backend.read_bytes(path)
        baseline_text = raw.decode("utf-8", errors="replace")
        _record_file_baseline(context, path=path, raw=raw, text=baseline_text, is_partial=True)
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

    raw = backend.read_bytes(path)
    baseline_text = raw.decode("utf-8", errors="replace")
    is_partial = "start_line" in arguments or "end_line" in arguments
    _record_file_baseline(context, path=path, raw=raw, text=baseline_text, is_partial=is_partial)

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

    exists_before = backend.exists(path)
    is_existing_file = exists_before and backend.is_file(path)
    if is_existing_file and not append:
        try:
            current_raw = backend.read_bytes(path)
        except Exception:
            current_raw = backend.read_text(path).encode("utf-8", errors="replace")
        baseline_issue = _baseline_error(context, path=path, current_raw=current_raw)
        if baseline_issue:
            message = "Read the full file with read_file before overwriting."
            if baseline_issue == "file_changed_since_read":
                message = "File changed since it was last read. Re-read it before overwriting."
            return _workspace_error(message, error_code=baseline_issue, path=path)

    backend.write_text(path, write_content, append=append)

    try:
        updated_raw = backend.read_bytes(path)
        updated_text, _has_bom = _decode_workspace_text(updated_raw)
    except Exception:
        updated_text = backend.read_text(path)
        updated_raw = updated_text.encode("utf-8", errors="replace")
    _record_file_baseline(context, path=path, raw=updated_raw, text=updated_text, is_partial=False)

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
        metadata={
            "changed_files": [path],
            "operation": "write_file",
            "append": append,
        },
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


def edit_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    required_fields = ("path", "old_string", "new_string")
    missing_fields = [field for field in required_fields if field not in arguments]
    if missing_fields:
        message = "`path`, `old_string`, and `new_string` are required."
        return _workspace_error(message, error_code="invalid_arguments", missing_arguments=missing_fields)

    backend = context.workspace_backend
    path = str(arguments["path"])
    if not backend.is_file(path):
        return _workspace_error(f"file not found: {path}", error_code="file_not_found", path=path)

    old_string = str(arguments["old_string"])
    if not old_string:
        return _workspace_error("`old_string` cannot be empty.", error_code="old_string_empty", path=path)

    new_string = str(arguments["new_string"])
    if old_string == new_string:
        return _workspace_error(
            "No changes: old_string and new_string are identical.",
            error_code="no_changes",
            path=path,
        )
    replace_all = bool(arguments.get("replace_all", False))

    raw = backend.read_bytes(path)
    try:
        text, has_bom = _decode_workspace_text(raw)
    except ValueError:
        return _workspace_error("Unsupported file encoding for edit_file.", error_code="unsupported_encoding", path=path)

    baseline_issue = _baseline_error(context, path=path, current_raw=raw)
    if baseline_issue:
        message = "Read the full file with read_file before editing."
        if baseline_issue == "file_changed_since_read":
            message = "File changed since it was last read. Re-read it before editing."
        return _workspace_error(message, error_code=baseline_issue, path=path)

    actual_old = old_string
    actual_new = new_string
    occurrence_count = text.count(actual_old)
    line_ending = _line_ending_label(text)
    if occurrence_count == 0 and line_ending == "crlf" and "\r\n" not in old_string:
        actual_old = old_string.replace("\n", "\r\n")
        actual_new = new_string.replace("\n", "\r\n")
        occurrence_count = text.count(actual_old)

    if occurrence_count == 0:
        return _workspace_error("`old_string` not found in file.", error_code="old_string_not_found", path=path)
    if occurrence_count > 1 and not replace_all:
        return _workspace_error(
            "`old_string` matched multiple locations; make it unique or set replace_all=true.",
            error_code="old_string_not_unique",
            path=path,
            match_count=occurrence_count,
        )

    if replace_all:
        updated = text.replace(actual_old, actual_new)
        replaced_count = occurrence_count
    else:
        updated = text.replace(actual_old, actual_new, 1)
        replaced_count = 1

    backend.write_text(path, "\ufeff" + updated if has_bom else updated)
    updated_raw = _encode_workspace_text(updated, has_bom=has_bom)
    _record_file_baseline(context, path=path, raw=updated_raw, text=updated, is_partial=False)

    diff, diff_truncated, additions, deletions = _bounded_unified_diff(path, text, updated)
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json({"ok": True, "path": path, "replaced_count": replaced_count}),
        metadata={
            "changed_files": [path],
            "diff": diff,
            "diff_truncated": diff_truncated,
            "additions": additions,
            "deletions": deletions,
            "operation": "edit_file",
            "line_ending": line_ending,
        },
    )
