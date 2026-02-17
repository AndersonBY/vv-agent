from __future__ import annotations

from pathlib import Path
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import resolve_workspace_path, to_json
from v_agent.types import ToolExecutionResult


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

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start_line = int(arguments.get("start_line", 1))
    end_line = arguments.get("end_line")
    end_line_int = int(end_line) if end_line is not None else len(lines)

    start_idx = max(start_line - 1, 0)
    end_idx = max(end_line_int, start_idx)
    selected = lines[start_idx:end_idx]

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "path": target.relative_to(context.workspace).as_posix(),
                "start_line": start_idx + 1,
                "end_line": start_idx + len(selected),
                "content": "\n".join(selected),
            }
        ),
    )


def write_file(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    target = resolve_workspace_path(context, str(arguments["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)

    content = str(arguments.get("content", ""))
    append = bool(arguments.get("append", False))

    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as file_obj:
        file_obj.write(content)

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json(
            {
                "ok": True,
                "path": target.relative_to(context.workspace).as_posix(),
                "written_chars": len(content),
            }
        ),
    )
