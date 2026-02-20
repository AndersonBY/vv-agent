from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus

_DANGEROUS_SNIPPETS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=/dev/zero of=/dev/",
)

_OUTPUT_LIMIT = 50_000


def run_bash_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    command = str(arguments.get("command", "")).strip()
    if not command:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="command_required",
            content=to_json({"error": "`command` is required"}),
        )

    lowered = command.lower()
    for snippet in _DANGEROUS_SNIPPETS:
        if snippet in lowered:
            return ToolExecutionResult(
                tool_call_id="",
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code="dangerous_command",
                content=to_json({"error": f"dangerous command blocked: {snippet}"}),
            )

    timeout = int(arguments.get("timeout", 300))
    timeout = max(1, min(timeout, 600))

    exec_dir_raw = str(arguments.get("exec_dir", "."))
    exec_dir = context.resolve_workspace_path(exec_dir_raw)
    if not exec_dir.exists() or not exec_dir.is_dir():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="invalid_exec_dir",
            content=to_json({"error": f"exec_dir not found: {exec_dir_raw}"}),
        )

    stdin_data = arguments.get("stdin")
    stdin_text = str(stdin_data) if stdin_data is not None else None
    auto_confirm = bool(arguments.get("auto_confirm", False))
    run_in_background = bool(arguments.get("run_in_background", False))

    if run_in_background:
        session_id = background_session_manager.start(
            command=command,
            cwd=exec_dir,
            timeout_seconds=timeout,
            stdin=stdin_text,
            auto_confirm=auto_confirm,
        )
        payload = {
            "status": "running",
            "session_id": session_id,
            "command": command,
        }
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(payload),
            metadata=payload,
        )

    wrapped_command = command
    if auto_confirm:
        wrapped_command = f"yes | ({command})"

    try:
        completed = subprocess.run(
            ["bash", "-lc", wrapped_command],
            cwd=str(exec_dir),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="command_timeout",
            content=to_json({"error": f"command timed out after {timeout} seconds"}),
        )

    combined_output = f"{completed.stdout}{completed.stderr}"[:_OUTPUT_LIMIT]
    payload = {
        "command": command,
        "cwd": Path(exec_dir).relative_to(context.workspace).as_posix() if exec_dir != context.workspace else ".",
        "exit_code": completed.returncode,
        "output": combined_output,
    }

    if completed.returncode != 0:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="command_failed",
            content=to_json(payload),
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
    )
