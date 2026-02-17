from __future__ import annotations

import re
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import resolve_workspace_path, to_json
from v_agent.types import ToolExecutionResult


def workspace_grep(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    root = resolve_workspace_path(context, str(arguments.get("path", ".")))
    glob_pattern = str(arguments.get("glob", "**/*"))
    pattern = str(arguments["pattern"])
    case_sensitive = bool(arguments.get("case_sensitive", False))
    max_results = int(arguments.get("max_results", 50))

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            content=to_json({"error": f"invalid regex pattern: {exc}"}),
        )

    matches: list[dict[str, Any]] = []
    for candidate in root.glob(glob_pattern):
        if not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    {
                        "path": candidate.relative_to(context.workspace).as_posix(),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(matches) >= max_results:
                    return ToolExecutionResult(
                        tool_call_id="",
                        status="success",
                        content=to_json({"matches": matches, "truncated": True}),
                    )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        content=to_json({"matches": matches, "truncated": False}),
    )
