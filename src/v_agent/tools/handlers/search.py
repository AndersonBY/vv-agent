from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult

FILE_TYPE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "py": (".py", ".pyw", ".pyi"),
    "js": (".js", ".jsx", ".mjs"),
    "ts": (".ts", ".tsx"),
    "html": (".html", ".htm", ".xhtml"),
    "css": (".css", ".scss", ".sass", ".less"),
    "java": (".java",),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"),
    "rust": (".rs",),
    "go": (".go",),
    "php": (".php", ".php3", ".php4", ".php5"),
    "rb": (".rb", ".rbx", ".rhtml", ".ruby"),
    "sh": (".sh", ".bash", ".zsh", ".fish"),
    "sql": (".sql",),
    "json": (".json",),
    "xml": (".xml", ".xsl", ".xsd"),
    "yaml": (".yaml", ".yml"),
    "md": (".md", ".markdown", ".mdown", ".mkd"),
    "txt": (".txt",),
    "log": (".log",),
    "ini": (".ini", ".cfg", ".conf"),
    "dockerfile": ("dockerfile",),
    "makefile": ("makefile", "gnumakefile"),
}

_BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".mp3",
    ".wav",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
}

_OUTPUT_MODES = {"content", "files_with_matches", "count"}
_MAX_RESULT_LINES = 500
_MAX_RESULT_CHARS = 30_000


def _to_int(value: Any, *, name: str, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be an integer") from exc
    return max(parsed, min_value)


def _result_error(message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        content=to_json({"error": message}),
    )


def _matches_file_type(rel_path: str, file_type: str | None) -> bool:
    p = PurePosixPath(rel_path)
    suffix = p.suffix.lower()
    if file_type is None:
        return suffix not in _BINARY_SUFFIXES

    normalized = file_type.strip().lower()
    if normalized not in FILE_TYPE_EXTENSIONS:
        return False

    filename = p.name.lower()
    allowed = FILE_TYPE_EXTENSIONS[normalized]
    return filename in allowed or suffix in allowed


def _truncate_result_text(result_text: str, *, total_matches: int, files_with_matches: int) -> tuple[str, bool]:
    lines = result_text.splitlines()
    if len(lines) <= _MAX_RESULT_LINES and len(result_text) <= _MAX_RESULT_CHARS:
        return result_text, False

    truncated = result_text
    if len(result_text) > _MAX_RESULT_CHARS:
        truncated = result_text[:_MAX_RESULT_CHARS]
        last_newline = truncated.rfind("\n")
        if last_newline > _MAX_RESULT_CHARS * 0.8:
            truncated = truncated[:last_newline]
    else:
        truncated = "\n".join(lines[:_MAX_RESULT_LINES])

    shown_lines = len(truncated.splitlines())
    truncated_info = (
        "\n\n--- TRUNCATED ---\n"
        f"Shown: {shown_lines} lines, {len(truncated)} characters\n"
        f"Total matches: {total_matches} in {files_with_matches} files\n"
        "Use a narrower pattern/path/glob/type/head_limit for more focused output."
    )
    return f"{truncated}{truncated_info}", True


def workspace_grep(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(arguments.get("pattern", "")).strip()
    if not pattern:
        return _result_error("Search pattern is required")

    output_mode = str(arguments.get("output_mode", "content"))
    if output_mode not in _OUTPUT_MODES:
        return _result_error(
            f"Invalid `output_mode`: {output_mode}. Supported: {', '.join(sorted(_OUTPUT_MODES))}"
        )

    try:
        head_limit_raw = arguments.get("head_limit", arguments.get("max_results"))
        head_limit = _to_int(head_limit_raw, name="head_limit", min_value=1) if head_limit_raw is not None else None
        lines_before = _to_int(arguments["b"], name="b") if "b" in arguments else None
        lines_after = _to_int(arguments["a"], name="a") if "a" in arguments else None
        context_lines = _to_int(arguments["c"], name="c") if "c" in arguments else None
    except ValueError as exc:
        return _result_error(str(exc))

    file_type = arguments.get("type")
    if file_type is not None:
        file_type = str(file_type).strip().lower()
        if file_type not in FILE_TYPE_EXTENSIONS:
            supported = ", ".join(sorted(FILE_TYPE_EXTENSIONS.keys()))
            return _result_error(f"Unsupported file type: {file_type}. Supported types: {supported}")

    path = str(arguments.get("path", "."))
    glob_pattern = str(arguments.get("glob", "**/*"))
    backend = context.workspace_backend

    case_insensitive = bool(arguments.get("i", False))
    if "case_sensitive" in arguments:
        case_insensitive = not bool(arguments.get("case_sensitive"))

    multiline_mode = bool(arguments.get("multiline", False))
    show_line_numbers = bool(arguments.get("n", True))

    regex_flags = re.IGNORECASE if case_insensitive else 0
    if multiline_mode:
        regex_flags |= re.MULTILINE | re.DOTALL
    try:
        regex = re.compile(pattern, regex_flags)
    except re.error as exc:
        return _result_error(f"Invalid regular expression: {exc}")

    context_before = context_lines if context_lines is not None else (lines_before or 0)
    context_after = context_lines if context_lines is not None else (lines_after or 0)

    files_searched = 0
    total_matches = 0
    files_with_matches: list[str] = []
    file_counts: dict[str, int] = {}
    content_rows: list[dict[str, Any]] = []

    for rel_path in backend.list_files(path, glob_pattern):
        if not _matches_file_type(rel_path, file_type):
            continue

        try:
            text = backend.read_text(rel_path)
        except OSError:
            continue

        files_searched += 1

        if multiline_mode:
            matches = list(regex.finditer(text))
            if not matches:
                continue

            file_match_count = len(matches)
            total_matches += file_match_count
            files_with_matches.append(rel_path)
            file_counts[rel_path] = file_match_count

            if output_mode == "content":
                for match in matches:
                    line_no = text.count("\n", 0, match.start()) + 1
                    content_rows.append(
                        {
                            "path": rel_path,
                            "line": line_no,
                            "text": match.group(0),
                            "is_match": True,
                        }
                    )
            continue

        lines = text.splitlines()
        matched_line_numbers: list[int] = []
        file_match_count = 0

        for line_number, line in enumerate(lines, start=1):
            line_matches = list(regex.finditer(line))
            if not line_matches:
                continue
            file_match_count += len(line_matches)
            matched_line_numbers.append(line_number)

        if file_match_count == 0:
            continue

        total_matches += file_match_count
        files_with_matches.append(rel_path)
        file_counts[rel_path] = file_match_count

        if output_mode != "content":
            continue

        matched_set = set(matched_line_numbers)
        if context_before > 0 or context_after > 0:
            selected_lines: set[int] = set()
            for line_number in matched_set:
                start_line = max(1, line_number - context_before)
                end_line = min(len(lines), line_number + context_after)
                selected_lines.update(range(start_line, end_line + 1))
            line_numbers = sorted(selected_lines)
        else:
            line_numbers = sorted(matched_set)

        for line_number in line_numbers:
            content_rows.append(
                {
                    "path": rel_path,
                    "line": line_number,
                    "text": lines[line_number - 1],
                    "is_match": line_number in matched_set,
                }
            )

    files_with_matches.sort()

    if output_mode == "files_with_matches":
        visible_files = files_with_matches[:head_limit] if head_limit is not None else files_with_matches
        result_lines = [f"Found {len(files_with_matches)} files matching pattern {pattern!r}"]
        if visible_files:
            result_lines.extend(visible_files)
        else:
            result_lines.append("No matches found.")
        result_text = "\n".join(result_lines)
        payload: dict[str, Any] = {"files": visible_files}
    elif output_mode == "count":
        ordered_file_counts = dict(sorted(file_counts.items()))
        count_items = list(ordered_file_counts.items())
        visible_items = count_items[:head_limit] if head_limit is not None else count_items
        result_lines = [f"Match counts for pattern {pattern!r}"]
        for file_path, count in visible_items:
            result_lines.append(f"{file_path}: {count}")
        result_lines.append(f"Total: {total_matches} matches in {len(files_with_matches)} files")
        result_text = "\n".join(result_lines)
        payload = {"file_counts": {file_path: count for file_path, count in visible_items}}
    else:
        visible_rows = content_rows[:head_limit] if head_limit is not None else content_rows
        result_lines = [f"Found {total_matches} matches in {len(files_with_matches)} files for pattern {pattern!r}"]
        if not visible_rows:
            result_lines.append("No matches found.")
        else:
            current_file: str | None = None
            for row in visible_rows:
                row_path = str(row["path"])
                row_line = int(row["line"])
                row_text = str(row["text"])
                row_is_match = bool(row["is_match"])

                if current_file != row_path:
                    result_lines.append(f"File: {row_path}")
                    current_file = row_path

                marker = "" if row_is_match else "-"
                if show_line_numbers:
                    result_lines.append(f"  {marker}{row_line}: {row_text}")
                else:
                    result_lines.append(f"  {marker}{row_text}")

        result_text = "\n".join(result_lines)
        payload = {"matches": visible_rows}

    truncated_text, text_truncated = _truncate_result_text(
        result_text,
        total_matches=total_matches,
        files_with_matches=len(files_with_matches),
    )

    payload.update(
        {
            "result": truncated_text,
            "output_mode": output_mode,
            "truncated": text_truncated,
            "summary": {
                "files_searched": files_searched,
                "files_with_matches": len(files_with_matches),
                "total_matches": total_matches,
            },
        }
    )

    if head_limit is not None:
        payload["head_limit"] = head_limit

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(payload),
    )
