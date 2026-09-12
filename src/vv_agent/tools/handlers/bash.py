from __future__ import annotations

import os
from typing import Any

from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.processes import start_captured_process
from vv_agent.runtime.shell import prepare_shell_execution
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.background import background_command_result
from vv_agent.tools.handlers.common import builtin_error
from vv_agent.types import ToolExecutionResult

_DANGEROUS_SNIPPETS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=/dev/zero of=/dev/",
)
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
            default_priority = _normalize_windows_shell_priority(metadata.get("windows_shell_priority"), strict=True)
        if default_bash_env is None:
            default_bash_env = _normalize_bash_env(metadata.get("bash_env"), strict=True)
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


def _bounded_integer(raw: Any, name: str, minimum: int, maximum: int) -> int:
    if type(raw) is not int or not minimum <= raw <= maximum:
        raise ValueError(f"`{name}` must be an integer from {minimum} through {maximum}")
    return raw


def run_bash_command(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    command = str(arguments.get("command", "")).strip()
    if not command:
        return builtin_error("`command` is required", "command_required")
    lowered = command.lower()
    for snippet in _DANGEROUS_SNIPPETS:
        if snippet in lowered:
            return builtin_error(f"dangerous command blocked: {snippet}", "dangerous_command")
    try:
        yield_time_ms = _bounded_integer(arguments.get("yield_time_ms", 1000), "yield_time_ms", 0, 10000)
        timeout_seconds = (
            _bounded_integer(arguments["timeout_seconds"], "timeout_seconds", 1, 86400)
            if "timeout_seconds" in arguments
            else None
        )
    except ValueError as exc:
        return builtin_error(str(exc), "invalid_tool_arguments")
    exec_dir_raw = str(arguments.get("exec_dir", "."))
    try:
        exec_dir = context.resolve_workspace_path(exec_dir_raw)
    except ValueError as exc:
        return builtin_error(str(exc), "path_escapes_workspace")
    if not exec_dir.is_dir():
        return builtin_error(f"exec_dir not found: {exec_dir_raw}", "invalid_exec_dir")
    stdin_data = arguments.get("stdin")
    stdin_text = str(stdin_data) if stdin_data is not None else None
    try:
        shell, windows_shell_priority, bash_env = _read_shell_defaults(context)
        shell_command, prepared_stdin = prepare_shell_execution(
            command,
            auto_confirm=bool(arguments.get("auto_confirm", False)),
            stdin=stdin_text,
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )
    except ValueError as exc:
        return builtin_error(str(exc), "invalid_shell_config")
    try:
        started = start_captured_process(
            shell_command,
            cwd=exec_dir,
            stdin_text=prepared_stdin,
            env=_build_process_env(bash_env),
        )
    except OSError as exc:
        return builtin_error(f"Failed to start {shell or 'shell'}: {exc}", "command_failed")
    session_id = background_session_manager.adopt_running_process(
        command=command,
        cwd=exec_dir,
        timeout_seconds=timeout_seconds,
        process=started.process,
        output_path=started.output_path,
        shell=shell,
        started_at=started.started_at,
        owner_task_id=context.task_id,
        owner_workspace=context.workspace,
        artifact_backend=context.workspace_backend,
        artifact_task_id=context.task_id,
        artifact_tool_call_id=context.tool_call_id,
    )
    background_session_manager.wait(session_id, yield_time_ms)
    payload = background_session_manager.check_for_tool(
        session_id,
        context.workspace_backend,
        context.task_id,
        context.tool_call_id,
        workspace=context.workspace,
    )
    try:
        cwd = exec_dir.relative_to(context.workspace.resolve()).as_posix()
    except ValueError:
        cwd = str(exec_dir)
    return background_command_result(payload, foreground_cwd=cwd, force_handle=yield_time_ms == 0)
