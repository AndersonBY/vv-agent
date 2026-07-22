from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sysconfig
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sensitive_paths import is_sensitive_path
from vv_agent.types import ToolExecutionResult, ToolResultStatus

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
_MAX_STRUCTURED_ITEMS = 200
_MAX_STRUCTURED_CHARS = 20_000
_RG_EXECUTABLE_CACHE: str | None | bool = None
_COMMON_IGNORED_ROOTS = frozenset(
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
_SENSITIVE_RG_EXCLUDE_GLOBS = (
    "!**/.env",
    "!**/.npmrc",
    "!**/.pypirc",
    "!**/.netrc",
    "!**/credentials",
    "!**/id_rsa",
    "!**/id_dsa",
    "!**/id_ecdsa",
    "!**/id_ed25519",
    "!**/*.key",
    "!**/*.pem",
    "!**/*.p8",
    "!**/*.p12",
    "!**/*.pfx",
    "!**/secret.*",
    "!**/secrets.*",
    "!**/config/**/*token*",
    "!**/config/**/*credential*",
    "!**/config/**/*secret*",
    "!**/config/**/*private_key*",
    "!**/configs/**/*token*",
    "!**/configs/**/*credential*",
    "!**/configs/**/*secret*",
    "!**/configs/**/*private_key*",
    "!**/keys/**/*token*",
    "!**/keys/**/*credential*",
    "!**/keys/**/*secret*",
    "!**/keys/**/*private_key*",
    "!**/secrets/**/*token*",
    "!**/secrets/**/*credential*",
    "!**/secrets/**/*secret*",
    "!**/secrets/**/*private_key*",
    "!**/.ssh/**/*token*",
    "!**/.ssh/**/*credential*",
    "!**/.ssh/**/*secret*",
    "!**/.ssh/**/*private_key*",
    "!**/.aws/**/*token*",
    "!**/.aws/**/*credential*",
    "!**/.aws/**/*secret*",
    "!**/.aws/**/*private_key*",
    "!**/.gcp/**/*token*",
    "!**/.gcp/**/*credential*",
    "!**/.gcp/**/*secret*",
    "!**/.gcp/**/*private_key*",
)
_SENSITIVE_RG_COVERED_EXACT_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
_SENSITIVE_RG_COVERED_SUFFIXES = {".key", ".pem", ".p8", ".p12", ".pfx"}
_SENSITIVE_RG_COVERED_CONFIG_DIRS = {"config", "configs", "keys", "secrets", ".ssh", ".aws", ".gcp"}
_SENSITIVE_RG_COVERED_NAME_TOKENS = ("credential", "credentials", "secret", "secrets", "token", "private_key")


def _to_int(value: Any, *, name: str, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be an integer") from exc
    return max(parsed, min_value)


def _smart_case_defaults_to_case_insensitive(pattern: str) -> bool:
    return not any(char.isupper() for char in pattern)


def _result_error(message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status_code=ToolResultStatus.ERROR,
        content=message,
        metadata={"error": message},
    )


def _slice_results(items: list[Any], *, offset: int, head_limit: int) -> tuple[list[Any], bool]:
    if offset >= len(items):
        return [], False
    sliced = items[offset:]
    if head_limit == 0:
        return sliced, False
    return sliced[:head_limit], len(sliced) > head_limit


def _sensitive_candidate_paths(raw_paths: list[str], file_type: str | None) -> list[str]:
    return [
        raw_path.replace("\\", "/")
        for raw_path in raw_paths
        if _matches_file_type(raw_path.replace("\\", "/"), file_type) and is_sensitive_path(raw_path.replace("\\", "/"))
    ]


def _sensitive_path_is_covered_by_rg_excludes(path: str) -> bool:
    parts = PurePosixPath(path.replace("\\", "/").strip("/")).parts
    if not parts:
        return False
    name = parts[-1]
    if name in _SENSITIVE_RG_COVERED_EXACT_NAMES:
        return True
    if name.startswith(("secret.", "secrets.")):
        return True
    if any(name.endswith(suffix) for suffix in _SENSITIVE_RG_COVERED_SUFFIXES):
        return True
    if any(token in name for token in _SENSITIVE_RG_COVERED_NAME_TOKENS):
        return any(part in _SENSITIVE_RG_COVERED_CONFIG_DIRS for part in parts[:-1])
    return False


def _resolve_rg_executable() -> str | None:
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


def _is_workspace_root(path: str) -> bool:
    normalized = path.strip()
    if not normalized:
        return True
    return Path(normalized).as_posix() in {".", ""}


def _collect_ignored_root_names(base_path: Path) -> list[str]:
    ignored_root_names: list[str] = []
    try:
        with os.scandir(base_path) as iterator:
            for entry in iterator:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name.lower() in _COMMON_IGNORED_ROOTS:
                    ignored_root_names.append(entry.name)
    except OSError:
        pass
    return ignored_root_names


def _is_hidden_path(rel_path: str) -> bool:
    return any(part.startswith(".") for part in PurePosixPath(rel_path).parts)


def _is_in_common_ignored_root(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    return bool(parts) and parts[0].lower() in _COMMON_IGNORED_ROOTS


def _decode_rg_field(field: Any) -> str:
    if not isinstance(field, dict):
        return ""
    text = field.get("text")
    if isinstance(text, str):
        return text
    raw = field.get("bytes")
    if not isinstance(raw, str):
        return ""
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""


def _format_output_path(context: ToolContext, candidate_path: Path) -> str:
    resolved = candidate_path.resolve()
    workspace_root = context.workspace.resolve()
    try:
        rel = resolved.relative_to(workspace_root).as_posix()
        return rel or "."
    except ValueError:
        return str(resolved)


def _search_files_local_rg(
    context: ToolContext,
    *,
    path: str,
    glob_pattern: str,
    pattern: str,
    output_mode: str,
    file_type: str | None,
    case_insensitive: bool,
    multiline_mode: bool,
    context_before: int,
    context_after: int,
    include_hidden: bool,
    include_ignored: bool,
    include_sensitive: bool,
    literal: bool,
) -> tuple[int, int, list[str], dict[str, int], list[dict[str, Any]]] | None:
    rg_executable = _resolve_rg_executable()
    if not rg_executable:
        return None

    base_path = context.resolve_workspace_path(path)
    if not base_path.exists() or not base_path.is_dir():
        return None
    base_is_workspace_root = _is_workspace_root(path)
    ignored_root_names = _collect_ignored_root_names(base_path) if base_is_workspace_root and not include_ignored else []

    command = [
        rg_executable,
        "--json",
        "--line-number",
        "--color",
        "never",
        "--no-messages",
    ]
    if include_hidden:
        command.append("--hidden")
    if include_ignored:
        command.extend(["--no-ignore", "--no-ignore-vcs"])
    if case_insensitive:
        command.append("-i")
    if multiline_mode:
        command.extend(["--multiline", "--multiline-dotall"])
    if literal:
        command.append("--fixed-strings")
    if context_before > 0:
        command.extend(["--before-context", str(context_before)])
    if context_after > 0:
        command.extend(["--after-context", str(context_after)])
    if glob_pattern and glob_pattern != "**/*":
        command.extend(["--glob", glob_pattern])
    if not include_sensitive:
        for sensitive_glob in _SENSITIVE_RG_EXCLUDE_GLOBS:
            command.extend(["--glob", sensitive_glob])
    if base_is_workspace_root and not include_ignored:
        for ignored_name in ignored_root_names:
            command.extend(["--glob", f"!{ignored_name}/**"])
    if file_type:
        for token in FILE_TYPE_EXTENSIONS[file_type]:
            if token.startswith("."):
                command.extend(["--iglob", f"**/*{token}"])
            else:
                command.extend(["--iglob", f"**/{token}"])
    command.extend(["--regexp", pattern, "."])

    try:
        process = subprocess.Popen(
            command,
            cwd=base_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    assert process.stdout is not None

    searched_files: set[str] = set()
    files_with_matches_set: set[str] = set()
    file_counts: dict[str, int] = {}
    line_rows: dict[tuple[str, int], dict[str, Any]] = {}
    content_rows: list[dict[str, Any]] = []
    summary_searches: int | None = None

    try:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None

            event_type = event.get("type")
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if event_type == "summary":
                stats = data.get("stats")
                if isinstance(stats, dict):
                    searches = stats.get("searches")
                    if isinstance(searches, int):
                        summary_searches = searches
                continue

            rel_from_base = _decode_rg_field(data.get("path"))
            if not rel_from_base:
                continue
            normalized = rel_from_base.replace("\\", "/")
            rel_workspace = _format_output_path(context, base_path / normalized)
            if file_type and not _matches_file_type(rel_workspace, file_type):
                continue

            if event_type == "begin":
                searched_files.add(rel_workspace)
                continue

            if event_type == "match":
                searched_files.add(rel_workspace)
                submatches = data.get("submatches")
                increment = len(submatches) if isinstance(submatches, list) and submatches else 1
                file_counts[rel_workspace] = file_counts.get(rel_workspace, 0) + increment
                files_with_matches_set.add(rel_workspace)

                if output_mode != "content":
                    continue

                line_number = data.get("line_number")
                if not isinstance(line_number, int):
                    line_number = 1
                matched_lines = _decode_rg_field(data.get("lines"))

                if multiline_mode:
                    if isinstance(submatches, list) and submatches:
                        for sub in submatches:
                            if not isinstance(sub, dict):
                                continue
                            start = sub.get("start")
                            end = sub.get("end")
                            if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(matched_lines):
                                snippet = matched_lines[start:end]
                            else:
                                snippet = _decode_rg_field(sub.get("match")) or matched_lines
                            content_rows.append(
                                {
                                    "path": rel_workspace,
                                    "line": line_number,
                                    "text": snippet,
                                    "is_match": True,
                                }
                            )
                    else:
                        content_rows.append(
                            {
                                "path": rel_workspace,
                                "line": line_number,
                                "text": matched_lines,
                                "is_match": True,
                            }
                        )
                else:
                    row_key = (rel_workspace, line_number)
                    row_text = matched_lines.rstrip("\n")
                    existing = line_rows.get(row_key)
                    if existing is None:
                        line_rows[row_key] = {
                            "path": rel_workspace,
                            "line": line_number,
                            "text": row_text,
                            "is_match": True,
                        }
                    else:
                        existing["is_match"] = True
                        existing["text"] = row_text
                continue

            if event_type == "context" and output_mode == "content" and not multiline_mode:
                searched_files.add(rel_workspace)
                line_number = data.get("line_number")
                if not isinstance(line_number, int):
                    continue
                row_key = (rel_workspace, line_number)
                if row_key in line_rows:
                    continue
                line_rows[row_key] = {
                    "path": rel_workspace,
                    "line": line_number,
                    "text": _decode_rg_field(data.get("lines")).rstrip("\n"),
                    "is_match": False,
                }
    finally:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    if process.returncode not in (0, 1, 2):
        return None

    if output_mode == "content" and not multiline_mode:
        content_rows = sorted(line_rows.values(), key=lambda row: (str(row["path"]).lower(), int(row["line"])))
    elif output_mode == "content":
        content_rows.sort(key=lambda row: (str(row["path"]).lower(), int(row["line"])))

    files_with_matches = sorted(files_with_matches_set)
    total_matches = sum(file_counts.values())
    if summary_searches is not None:
        files_searched = summary_searches
    elif searched_files:
        files_searched = len(searched_files)
    else:
        files_searched = len(files_with_matches)
    return files_searched, total_matches, files_with_matches, file_counts, content_rows


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


def _estimate_match_row_size(row: dict[str, Any]) -> int:
    return len(str(row.get("path") or "")) + len(str(row.get("line") or "")) + len(str(row.get("text") or "")) + 32


def _estimate_file_path_size(path: str) -> int:
    return len(path) + 4


def _estimate_file_count_size(item: tuple[str, int]) -> int:
    file_path, count = item
    return len(file_path) + len(str(count)) + 8


def _cap_structured_items(
    items: list[Any],
    *,
    estimator: Callable[[Any], int],
) -> tuple[list[Any], bool]:
    capped: list[Any] = []
    used_chars = 0

    for item in items:
        item_size = max(int(estimator(item)), 1)
        if capped and (len(capped) >= _MAX_STRUCTURED_ITEMS or used_chars + item_size > _MAX_STRUCTURED_CHARS):
            return capped, True
        capped.append(item)
        used_chars += item_size

    return capped, False


def search_files(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(arguments.get("pattern", "")).strip()
    if not pattern:
        return _result_error("Search pattern is required")

    output_mode = str(arguments.get("output_mode", "files_with_matches"))
    if output_mode not in _OUTPUT_MODES:
        return _result_error(f"Invalid `output_mode`: {output_mode}. Supported: {', '.join(sorted(_OUTPUT_MODES))}")

    try:
        head_limit = _to_int(arguments.get("head_limit", 250), name="head_limit", min_value=0)
        offset = _to_int(arguments.get("offset", 0), name="offset", min_value=0)
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
    include_hidden = bool(arguments.get("include_hidden", False))
    include_ignored = bool(arguments.get("include_ignored", False))
    include_sensitive = bool(arguments.get("include_sensitive", False))
    literal = bool(arguments.get("literal", False))
    root_listing = _is_workspace_root(path)
    explicit_file_target = backend.is_file(path)

    if "case_sensitive" in arguments:
        case_insensitive = not bool(arguments.get("case_sensitive"))
    else:
        case_insensitive = _smart_case_defaults_to_case_insensitive(pattern)

    multiline_mode = bool(arguments.get("multiline", False))
    show_line_numbers = bool(arguments.get("n", True))

    regex_flags = re.IGNORECASE if case_insensitive else 0
    if multiline_mode:
        regex_flags |= re.MULTILINE | re.DOTALL
    regex_pattern = re.escape(pattern) if literal else pattern
    try:
        regex = re.compile(regex_pattern, regex_flags)
    except re.error as exc:
        return _result_error(f"Invalid regular expression: {exc}")

    context_before = context_lines if context_lines is not None else (lines_before or 0)
    context_after = context_lines if context_lines is not None else (lines_after or 0)

    files_searched = 0
    total_matches = 0
    files_with_matches: list[str] = []
    file_counts: dict[str, int] = {}
    content_rows: list[dict[str, Any]] = []
    sensitive_files_omitted = 0
    sensitive_candidates: list[str] = []

    if not include_sensitive:
        raw_sensitive_candidates = [path] if explicit_file_target else backend.list_files(path, glob_pattern)
        sensitive_candidates = _sensitive_candidate_paths(raw_sensitive_candidates, file_type)
        sensitive_files_omitted = len(set(sensitive_candidates))

    rg_result: tuple[int, int, list[str], dict[str, int], list[dict[str, Any]]] | None = None
    local_root = getattr(backend, "root", None)
    can_exclude_sensitive_in_rg = include_sensitive or all(
        _sensitive_path_is_covered_by_rg_excludes(candidate) for candidate in sensitive_candidates
    )
    if isinstance(local_root, Path) and not explicit_file_target and can_exclude_sensitive_in_rg:
        rg_result = _search_files_local_rg(
            context,
            path=path,
            glob_pattern=glob_pattern,
            pattern=pattern if literal else regex_pattern,
            output_mode=output_mode,
            file_type=file_type,
            case_insensitive=case_insensitive,
            multiline_mode=multiline_mode,
            context_before=context_before,
            context_after=context_after,
            include_hidden=include_hidden,
            include_ignored=include_ignored,
            include_sensitive=include_sensitive,
            literal=literal,
        )

    if rg_result is not None:
        files_searched, total_matches, files_with_matches, file_counts, content_rows = rg_result
        if not include_sensitive:
            sensitive_paths = {file_path for file_path in files_with_matches if is_sensitive_path(file_path)}
            files_with_matches = [file_path for file_path in files_with_matches if file_path not in sensitive_paths]
            file_counts = {file_path: count for file_path, count in file_counts.items() if file_path not in sensitive_paths}
            content_rows = [row for row in content_rows if row["path"] not in sensitive_paths]
            total_matches = sum(file_counts.values())
    else:
        raw_paths = [path] if explicit_file_target else backend.list_files(path, glob_pattern)
        for raw_rel_path in raw_paths:
            rel_path = raw_rel_path.replace("\\", "/")
            if not _matches_file_type(rel_path, file_type):
                continue
            if not include_sensitive and is_sensitive_path(rel_path):
                continue
            if not explicit_file_target and not include_hidden and _is_hidden_path(rel_path):
                continue
            if not explicit_file_target and root_listing and not include_ignored and _is_in_common_ignored_root(rel_path):
                continue

            try:
                text = backend.read_text(raw_rel_path)
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

    metadata: dict[str, Any] = {
        "output_mode": output_mode,
        "pattern": pattern,
        "path": path,
        "glob": glob_pattern,
        "type": file_type,
        "literal": literal,
        "offset": offset,
        "head_limit": head_limit,
        "summary": {
            "files_searched": files_searched,
            "files_with_matches": len(files_with_matches),
            "total_matches": total_matches,
        },
    }
    if sensitive_files_omitted:
        metadata["sensitive_files_omitted"] = sensitive_files_omitted

    total_result_items = 0
    returned_count = 0
    head_limited = False
    structured_capped = False

    if output_mode == "files_with_matches":
        total_result_items = len(files_with_matches)
        visible_files, head_limited = _slice_results(files_with_matches, offset=offset, head_limit=head_limit)
        visible_files, structured_capped = _cap_structured_items(
            visible_files,
            estimator=lambda item: _estimate_file_path_size(str(item)),
        )
        returned_count = len(visible_files)
        result_lines = [f"Found {len(files_with_matches)} files matching pattern {pattern!r}"]
        if visible_files:
            if head_limited or structured_capped:
                result_lines.append(f"Showing first {len(visible_files)} files.")
            result_lines.extend(visible_files)
        else:
            result_lines.append("No matches found.")
        result_text = "\n".join(result_lines)
        metadata["files"] = visible_files
    elif output_mode == "count":
        count_items = sorted(file_counts.items())
        total_result_items = len(count_items)
        visible_items, head_limited = _slice_results(count_items, offset=offset, head_limit=head_limit)
        visible_items, structured_capped = _cap_structured_items(
            visible_items,
            estimator=lambda item: _estimate_file_count_size((str(item[0]), int(item[1]))),
        )
        returned_count = len(visible_items)
        result_lines = [f"Match counts for pattern {pattern!r}"]
        if visible_items and (head_limited or structured_capped):
            result_lines.append(f"Showing first {len(visible_items)} files.")
        for file_path, count in visible_items:
            result_lines.append(f"{file_path}: {count}")
        result_lines.append(f"Total: {total_matches} matches in {len(files_with_matches)} files")
        result_text = "\n".join(result_lines)
        metadata["file_counts"] = {file_path: count for file_path, count in visible_items}
    else:
        total_result_items = len(content_rows)
        visible_rows, head_limited = _slice_results(content_rows, offset=offset, head_limit=head_limit)
        visible_rows, structured_capped = _cap_structured_items(
            visible_rows,
            estimator=lambda item: _estimate_match_row_size(item if isinstance(item, dict) else {}),
        )
        returned_count = len(visible_rows)
        result_lines = [f"Found {total_matches} matches in {len(files_with_matches)} files for pattern {pattern!r}"]
        if not visible_rows:
            result_lines.append("No matches found.")
        else:
            if head_limited or structured_capped:
                result_lines.append(f"Showing first {len(visible_rows)} rows.")
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
        metadata["matches"] = visible_rows

    truncated_text, text_truncated = _truncate_result_text(
        result_text,
        total_matches=total_matches,
        files_with_matches=len(files_with_matches),
    )

    structured_truncated = head_limited or structured_capped
    metadata.update(
        {
            "total_result_items": total_result_items,
            "returned_count": returned_count,
            "head_limited": head_limited,
            "content_truncated": text_truncated,
            "structured_truncated": structured_truncated,
            "truncated": text_truncated or structured_truncated,
        }
    )
    if structured_capped:
        metadata["structured_item_limit"] = _MAX_STRUCTURED_ITEMS
        metadata["structured_char_limit"] = _MAX_STRUCTURED_CHARS

    return ToolExecutionResult(
        tool_call_id="",
        status_code=ToolResultStatus.SUCCESS,
        content=truncated_text,
        metadata=metadata,
    )
