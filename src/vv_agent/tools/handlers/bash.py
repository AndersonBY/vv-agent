from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.shell import prepare_shell_execution
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


def _normalize_shell_value(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _normalize_windows_shell_priority(raw: Any, *, strict: bool) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        if strict:
            raise ValueError("`windows_shell_priority` must be a list of shell names")
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _normalize_bash_env(raw: Any, *, strict: bool) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        if strict:
            raise ValueError("`bash_env` must be an object mapping env names to values")
        return None

    normalized: dict[str, str] = {}
    for key, value in raw.items():
        env_name = str(key).strip()
        if not env_name:
            if strict:
                raise ValueError("`bash_env` contains empty env variable name")
            continue
        normalized[env_name] = "" if value is None else str(value)
    return normalized


def _read_shell_defaults(context: ToolContext) -> tuple[str | None, list[str] | None, dict[str, str] | None]:
    default_shell: str | None = None
    default_priority: list[str] | None = None
    default_bash_env: dict[str, str] | None = None

    metadata_sources: list[dict[str, Any]] = []
    runtime_metadata = getattr(context.ctx, "metadata", None)
    if isinstance(runtime_metadata, dict):
        metadata_sources.append(runtime_metadata)

    task_metadata = getattr(context, "task_metadata", None)
    if isinstance(task_metadata, dict):
        metadata_sources.append(task_metadata)

    for metadata in metadata_sources:
        if default_shell is None:
            default_shell = _normalize_shell_value(metadata.get("bash_shell"))
        if default_priority is None:
            default_priority = _normalize_windows_shell_priority(
                metadata.get("windows_shell_priority"),
                strict=True,
            )
        if default_bash_env is None:
            default_bash_env = _normalize_bash_env(
                metadata.get("bash_env"),
                strict=True,
            )

    return default_shell, default_priority, default_bash_env


def _build_process_env(extra_env: dict[str, str] | None) -> dict[str, str] | None:
    if not extra_env:
        return None
    env = dict(os.environ)
    env.update(extra_env)
    return env


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
    try:
        shell, windows_shell_priority, bash_env = _read_shell_defaults(context)
    except ValueError as exc:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="invalid_shell_config",
            content=to_json({"error": str(exc)}),
        )
    process_env = _build_process_env(bash_env)

    if run_in_background:
        try:
            session_id = background_session_manager.start(
                command=command,
                cwd=exec_dir,
                timeout_seconds=timeout,
                stdin=stdin_text,
                auto_confirm=auto_confirm,
                shell=shell,
                windows_shell_priority=windows_shell_priority,
                env=process_env,
            )
        except ValueError as exc:
            return ToolExecutionResult(
                tool_call_id="",
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code="shell_unavailable",
                content=to_json({"error": str(exc)}),
            )
        payload = {
            "status": "running",
            "session_id": session_id,
            "command": command,
        }
        if shell:
            payload["shell"] = shell
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(payload),
            metadata=payload,
        )

    try:
        shell_command, prepared_stdin = prepare_shell_execution(
            command,
            auto_confirm=auto_confirm,
            stdin=stdin_text,
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )
    except ValueError as exc:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="shell_unavailable",
            content=to_json({"error": str(exc)}),
        )

    try:
        completed = subprocess.run(
            shell_command,
            cwd=str(exec_dir),
            input=prepared_stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=process_env,
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
    if shell:
        payload["shell"] = shell

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
