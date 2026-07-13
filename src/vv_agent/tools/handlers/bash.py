from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.processes import (
    read_captured_output,
    remove_captured_output,
    start_captured_process,
)
from vv_agent.runtime.shell import prepare_shell_execution
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, select_metadata, to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus

_DANGEROUS_SNIPPETS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=/dev/zero of=/dev/",
)

_OUTPUT_LIMIT = 50_000
_WINDOWS_PYTHON_ENV_DEFAULTS = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _normalize_shell_value(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("`bash_shell` must be a string shell name")
    value = raw.strip()
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
    if not extra_env and os.name != "nt":
        return None
    env = dict(os.environ)
    if os.name == "nt":
        for key, value in _WINDOWS_PYTHON_ENV_DEFAULTS.items():
            env.setdefault(key, value)
    if extra_env:
        env.update(extra_env)
    return env


def _parse_timeout_seconds(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("`timeout` must be an integer")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError("`timeout` must be an integer") from exc
    else:
        raise ValueError("`timeout` must be an integer")
    return max(1, min(value, 600))


def run_bash_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    command = str(arguments.get("command", "")).strip()
    if not command:
        return builtin_error("`command` is required", "command_required")

    lowered = command.lower()
    for snippet in _DANGEROUS_SNIPPETS:
        if snippet in lowered:
            return builtin_error(
                f"dangerous command blocked: {snippet}",
                "dangerous_command",
            )

    try:
        timeout = _parse_timeout_seconds(arguments.get("timeout", 300))
    except ValueError as exc:
        return builtin_error(str(exc), "invalid_timeout")

    exec_dir_raw = str(arguments.get("exec_dir", "."))
    try:
        exec_dir = context.resolve_workspace_path(exec_dir_raw)
    except ValueError as exc:
        return builtin_error(str(exc), "path_escapes_workspace")
    if not exec_dir.exists() or not exec_dir.is_dir():
        return builtin_error(
            f"exec_dir not found: {exec_dir_raw}",
            "invalid_exec_dir",
        )

    stdin_data = arguments.get("stdin")
    stdin_text = str(stdin_data) if stdin_data is not None else None
    auto_confirm = bool(arguments.get("auto_confirm", False))
    run_in_background = bool(arguments.get("run_in_background", False))
    try:
        shell, windows_shell_priority, bash_env = _read_shell_defaults(context)
    except ValueError as exc:
        return builtin_error(str(exc), "invalid_shell_config")
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
            return builtin_error(
                str(exc),
                "invalid_shell_config",
            )
        except OSError as exc:
            selected_shell = shell or "shell"
            return builtin_error(
                f"Failed to start {selected_shell}: {exc}",
                "command_failed",
            )
        payload = {
            "status": "running",
            "session_id": session_id,
        }
        if shell:
            payload["shell"] = shell
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(payload),
            metadata=select_metadata(payload, "status", "session_id", "shell"),
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
        return builtin_error(str(exc), "invalid_shell_config")

    try:
        started_process = start_captured_process(
            shell_command,
            cwd=exec_dir,
            stdin_text=prepared_stdin,
            env=process_env,
        )
    except OSError as exc:
        selected_shell = shell or "shell"
        return builtin_error(
            f"Failed to start {selected_shell}: {exc}",
            "command_failed",
        )

    try:
        completed_exit_code = started_process.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        background_session_id = background_session_manager.adopt_running_process(
            command=command,
            cwd=exec_dir,
            timeout_seconds=timeout,
            process=started_process.process,
            output_path=started_process.output_path,
            shell=shell,
        )
        payload = {
            "status": "running",
            "session_id": background_session_id,
            "cwd": exec_dir_raw,
            "message": (
                f"command exceeded foreground timeout after {timeout} seconds and "
                "continues in background; use `check_background_command` with this "
                "session_id to inspect progress"
            ),
            "output": read_captured_output(started_process.output_path, limit_chars=_OUTPUT_LIMIT),
            "transitioned_to_background": True,
        }
        if shell:
            payload["shell"] = shell
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.RUNNING,
            content=to_json(payload),
            metadata=select_metadata(
                payload,
                "status",
                "session_id",
                "cwd",
                "shell",
                "transitioned_to_background",
            ),
        )

    combined_output = read_captured_output(started_process.output_path, limit_chars=_OUTPUT_LIMIT)
    remove_captured_output(started_process.output_path)
    workspace_root = context.workspace.resolve()
    resolved_exec_dir = Path(exec_dir).resolve()
    if resolved_exec_dir == workspace_root:
        cwd = "."
    else:
        try:
            cwd = resolved_exec_dir.relative_to(workspace_root).as_posix()
        except ValueError:
            cwd = str(resolved_exec_dir)

    payload = {
        "cwd": cwd,
        "exit_code": completed_exit_code,
        "output": combined_output,
    }
    if shell:
        payload["shell"] = shell

    metadata = select_metadata(payload, "cwd", "exit_code", "shell")

    if completed_exit_code != 0:
        return builtin_error(
            f"command exited with code {completed_exit_code}",
            "command_failed",
            details=payload,
            metadata=metadata,
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        metadata=metadata,
    )
