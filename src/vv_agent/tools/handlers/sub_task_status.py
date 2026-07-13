from __future__ import annotations

import time
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json, trim_portable_whitespace
from vv_agent.types import AgentStatus, ToolExecutionResult, ToolResultStatus

DEFAULT_SUB_TASK_SNAPSHOT_FILE_LIMIT = 20
MAX_SUB_TASK_SNAPSHOT_FILE_LIMIT = 100
DEFAULT_SUB_TASK_WAIT_INTERVAL_SECONDS = 300
MIN_SUB_TASK_WAIT_INTERVAL_SECONDS = 30
MAX_SUB_TASK_WAIT_INTERVAL_SECONDS = 1800
DEFAULT_SUB_TASK_MAX_WAIT_SECONDS = 3600
MIN_SUB_TASK_MAX_WAIT_SECONDS = 60
MAX_SUB_TASK_MAX_WAIT_SECONDS = 24 * 60 * 60
LOCAL_SUB_TASK_WAIT_POLL_SECONDS = 0.1
MIN_I64 = -(2**63)
MAX_I64 = 2**63 - 1
RUNNING_SUB_TASK_STATUSES = {
    AgentStatus.PENDING.value,
    AgentStatus.RUNNING.value,
    "pending",
    "running",
}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, str):
        normalized = trim_portable_whitespace(value).lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_detail_level(detail_level: str | None) -> str:
    normalized = trim_portable_whitespace(detail_level or "basic").lower()
    return normalized if normalized in {"basic", "snapshot"} else "basic"


def _parse_integer_arg(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if MIN_I64 <= value <= MAX_I64 else None
    if isinstance(value, str):
        normalized = trim_portable_whitespace(value)
        if not normalized or (normalized[0] in "+-" and len(normalized) == 1):
            return None
        digits = normalized[1:] if normalized[0] in "+-" else normalized
        if not digits.isascii() or not digits.isdigit():
            return None
        try:
            parsed = int(normalized)
        except ValueError:
            return None
        return parsed if MIN_I64 <= parsed <= MAX_I64 else None
    return None


def _normalize_workspace_file_limit(workspace_file_limit: Any) -> int:
    limit = _parse_integer_arg(workspace_file_limit)
    if limit is None:
        limit = DEFAULT_SUB_TASK_SNAPSHOT_FILE_LIMIT
    return max(1, min(limit, MAX_SUB_TASK_SNAPSHOT_FILE_LIMIT))


def _normalize_wait_interval_seconds(check_interval_seconds: Any) -> int:
    seconds = _parse_integer_arg(check_interval_seconds)
    if seconds is None:
        seconds = DEFAULT_SUB_TASK_WAIT_INTERVAL_SECONDS
    return max(MIN_SUB_TASK_WAIT_INTERVAL_SECONDS, min(seconds, MAX_SUB_TASK_WAIT_INTERVAL_SECONDS))


def _normalize_max_wait_seconds(max_wait_seconds: Any) -> int:
    if max_wait_seconds is None:
        return DEFAULT_SUB_TASK_MAX_WAIT_SECONDS
    seconds = _parse_integer_arg(max_wait_seconds)
    if seconds is None:
        return DEFAULT_SUB_TASK_MAX_WAIT_SECONDS
    return max(MIN_SUB_TASK_MAX_WAIT_SECONDS, min(seconds, MAX_SUB_TASK_MAX_WAIT_SECONDS))


def _error(message: str, *, error_code: str, details: dict[str, Any] | None = None) -> ToolExecutionResult:
    payload: dict[str, Any] = {"ok": False, "error": message, "error_code": error_code}
    if details:
        payload["details"] = details
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json(payload),
        metadata=payload,
    )


def _success(payload: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        metadata=payload,
    )


def _is_internal_workspace_file(path: str) -> bool:
    normalized = str(path or "").strip().strip("/")
    if not normalized:
        return True
    return any(part.startswith(".") for part in normalized.split("/"))


def _build_workspace_snapshot(record: Any, workspace_file_limit: int) -> dict[str, Any]:
    workspace_backend = getattr(record, "workspace_backend", None)
    if workspace_backend is None:
        return {
            "workspace_files": [],
            "workspace_file_count": 0,
            "workspace_files_truncated": False,
        }

    try:
        raw_files = workspace_backend.list_files(".", "**/*")
    except Exception:
        return {
            "workspace_files": [],
            "workspace_file_count": 0,
            "workspace_files_truncated": False,
        }

    visible_files = [path for path in raw_files if isinstance(path, str) and not _is_internal_workspace_file(path)]
    return {
        "workspace_files": visible_files[:workspace_file_limit],
        "workspace_file_count": len(visible_files),
        "workspace_files_truncated": len(visible_files) > workspace_file_limit,
    }


def _build_snapshot(record: Any, *, workspace_file_limit: int) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "current_cycle_index": getattr(record, "current_cycle_index", None),
    }
    if getattr(record, "updated_at", None):
        snapshot["updated_at"] = record.updated_at
    if getattr(record, "task_title", None):
        snapshot["task_title"] = record.task_title
    if getattr(record, "recent_activity", None):
        snapshot["recent_activity"] = record.recent_activity
    if getattr(record, "latest_cycle", None):
        snapshot["latest_cycle"] = dict(record.latest_cycle)
    if getattr(record, "latest_tool_call", None):
        snapshot["latest_tool_call"] = dict(record.latest_tool_call)
    snapshot.update(_build_workspace_snapshot(record, workspace_file_limit))
    return snapshot


def _build_status_entry(
    *,
    task_id: str,
    record: Any,
    detail_level: str,
    workspace_file_limit: int,
) -> dict[str, Any]:
    if record is None:
        return {
            "task_id": task_id,
            "status": "missing",
            "error": f"Sub-task {task_id} not found.",
        }

    running = record.is_running()
    status = AgentStatus.RUNNING.value if running else AgentStatus.PENDING.value
    outcome = getattr(record, "outcome", None)
    if outcome is not None and not running:
        status = outcome.status.value

    entry: dict[str, Any] = {
        "task_id": record.task_id,
        "session_id": record.session_id,
        "agent_name": record.agent_name,
        "status": status,
    }
    if record.task_title:
        entry["task_description"] = record.task_title
    if getattr(record, "parent_run_id", None):
        entry["parent_run_id"] = record.parent_run_id
    if getattr(record, "parent_tool_call_id", None):
        entry["parent_tool_call_id"] = record.parent_tool_call_id

    if outcome is not None and not running:
        for key in ("final_answer", "wait_reason", "error", "error_code"):
            value = getattr(outcome, key, None)
            if value:
                entry[key] = value
        if outcome.cycles:
            entry["cycles"] = outcome.cycles
        if outcome.todo_list:
            entry["todo_list"] = outcome.todo_list
        if outcome.resolved:
            entry["resolved"] = outcome.resolved

    if detail_level == "snapshot":
        entry["snapshot"] = _build_snapshot(record, workspace_file_limit=workspace_file_limit)
    return entry


def _running_task_ids(tasks: list[dict[str, Any]]) -> list[str]:
    return [
        str(entry.get("task_id"))
        for entry in tasks
        if entry.get("status") in RUNNING_SUB_TASK_STATUSES and entry.get("task_id")
    ]


def _build_status_entries(
    *,
    manager: Any,
    task_ids: list[str],
    detail_level: str,
    workspace_file_limit: int,
) -> list[dict[str, Any]]:
    return [
        _build_status_entry(
            task_id=task_id,
            record=manager.get(task_id),
            detail_level=detail_level,
            workspace_file_limit=workspace_file_limit,
        )
        for task_id in task_ids
    ]


def _wait_for_sub_task_completion(
    *,
    manager: Any,
    task_ids: list[str],
    detail_level: str,
    workspace_file_limit: int,
    max_wait_seconds: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    deadline = time.monotonic() + max_wait_seconds
    tasks = _build_status_entries(
        manager=manager,
        task_ids=task_ids,
        detail_level=detail_level,
        workspace_file_limit=workspace_file_limit,
    )
    running_task_ids = _running_task_ids(tasks)
    wait_exceeded = False

    while running_task_ids:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            wait_exceeded = True
            break

        progressed = False
        wait_slice = min(LOCAL_SUB_TASK_WAIT_POLL_SECONDS, remaining_seconds)
        for task_id in list(running_task_ids):
            record = manager.wait(task_id, timeout=wait_slice)
            if record is not None and not record.is_running():
                progressed = True
                break
            if time.monotonic() >= deadline:
                break

        tasks = _build_status_entries(
            manager=manager,
            task_ids=task_ids,
            detail_level=detail_level,
            workspace_file_limit=workspace_file_limit,
        )
        next_running_task_ids = _running_task_ids(tasks)
        if not next_running_task_ids:
            running_task_ids = []
            break
        if time.monotonic() >= deadline:
            running_task_ids = next_running_task_ids
            wait_exceeded = True
            break
        if progressed or next_running_task_ids != running_task_ids:
            running_task_ids = next_running_task_ids
            continue
        running_task_ids = next_running_task_ids

    return tasks, running_task_ids, wait_exceeded


def _add_wait_metadata(
    payload: dict[str, Any],
    *,
    wait_for_completion: bool,
    check_interval_seconds: int,
    max_wait_seconds: int,
    running_task_ids: list[str],
    wait_exceeded: bool,
) -> None:
    if not wait_for_completion:
        if running_task_ids:
            payload["running_task_ids"] = running_task_ids
        payload["suggested_next_check_after_seconds"] = check_interval_seconds
        return

    payload.update(
        {
            "wait_for_completion": True,
            "wait_exceeded": wait_exceeded,
            "running_task_ids": running_task_ids,
            "suggested_next_check_after_seconds": check_interval_seconds,
            "max_wait_seconds": max_wait_seconds,
        }
    )
    if wait_exceeded:
        payload["message"] = (
            "Sub-task(s) are still running after the maximum wait. "
            "Call sub_task_status again later instead of tight polling."
        )


def sub_task_status(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    manager = context.sub_task_manager
    if manager is None:
        return _error("Sub-task manager is not available for this task", error_code="sub_task_manager_unavailable")

    raw_task_ids = arguments.get("task_ids")
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        return _error("`task_ids` must be a non-empty array", error_code="invalid_task_ids")

    task_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_task_ids:
        if not isinstance(item, str):
            return _error("`task_ids` must contain only strings", error_code="invalid_task_ids")
        task_id = trim_portable_whitespace(item)
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        task_ids.append(task_id)
    if not task_ids:
        return _error("`task_ids` must include at least one valid task id", error_code="invalid_task_ids")

    raw_detail_level = arguments.get("detail_level")
    if "detail_level" in arguments and not isinstance(raw_detail_level, str):
        return _error("`detail_level` must be a string", error_code="invalid_detail_level")
    detail_level = _normalize_detail_level(raw_detail_level)
    workspace_file_limit = _normalize_workspace_file_limit(arguments.get("workspace_file_limit"))
    raw_message = arguments.get("message")
    if "message" in arguments and not isinstance(raw_message, str):
        return _error("`message` must be a string", error_code="invalid_sub_task_message")
    message = trim_portable_whitespace(raw_message) if raw_message is not None else ""
    wait_for_response = _coerce_bool(arguments.get("wait_for_response"), default=False)
    wait_for_completion = _coerce_bool(arguments.get("wait_for_completion"), default=False)
    check_interval_seconds = _normalize_wait_interval_seconds(arguments.get("check_interval_seconds"))
    max_wait_seconds = _normalize_max_wait_seconds(arguments.get("max_wait_seconds"))

    interaction: dict[str, Any] | None = None
    if message:
        from vv_agent.runtime.engine import steer_sub_agent_session

        target_id = task_ids[0]
        record = manager.get(target_id)
        if record is None:
            return _error(
                f"Sub-task {target_id} not found.",
                error_code="sub_task_not_found",
                details={"task_id": target_id},
            )

        previous_status = (
            record.outcome.status.value
            if record.outcome is not None
            else AgentStatus.RUNNING.value
            if record.is_running()
            else AgentStatus.PENDING.value
        )
        if record.is_running():
            if record.session is None:
                return _error(
                    f"Sub-task {target_id} session is not ready yet.",
                    error_code="sub_task_session_not_ready",
                    details={"task_id": target_id},
                )
            if not steer_sub_agent_session(session_id=record.session_id, prompt=message):
                return _error(
                    f"Failed to queue message for running sub-task {target_id}.",
                    error_code="sub_task_message_queue_failed",
                    details={"task_id": target_id},
                )
            interaction = {
                "task_id": target_id,
                "action": "message_queued",
                "previous_status": previous_status,
            }
        else:
            if record.outcome is not None and record.outcome.status == AgentStatus.MAX_CYCLES:
                return _error(
                    f"Sub-task {target_id} reached max cycles and cannot continue.",
                    error_code="sub_task_max_cycles_reached",
                    details={"task_id": target_id},
                )
            try:
                manager._continue_task_with_context(
                    task_id=target_id,
                    prompt=message,
                    context=context,
                )
            except KeyError:
                return _error(
                    f"Sub-task {target_id} not found.",
                    error_code="sub_task_not_found",
                    details={"task_id": target_id},
                )
            except (RuntimeError, ValueError) as exc:
                return _error(
                    str(exc),
                    error_code="sub_task_continue_failed",
                    details={"task_id": target_id},
                )
            interaction = {
                "task_id": target_id,
                "action": "continued",
                "previous_status": previous_status,
            }

        if wait_for_response:
            manager.wait(target_id)

    if wait_for_completion:
        tasks, running_task_ids, wait_exceeded = _wait_for_sub_task_completion(
            manager=manager,
            task_ids=task_ids,
            detail_level=detail_level,
            workspace_file_limit=workspace_file_limit,
            max_wait_seconds=max_wait_seconds,
        )
    else:
        tasks = _build_status_entries(
            manager=manager,
            task_ids=task_ids,
            detail_level=detail_level,
            workspace_file_limit=workspace_file_limit,
        )
        running_task_ids = _running_task_ids(tasks)
        wait_exceeded = False

    payload: dict[str, Any] = {
        "tasks": tasks,
        "detail_level": detail_level,
    }
    _add_wait_metadata(
        payload,
        wait_for_completion=wait_for_completion,
        check_interval_seconds=check_interval_seconds,
        max_wait_seconds=max_wait_seconds,
        running_task_ids=running_task_ids,
        wait_exceeded=wait_exceeded,
    )
    if interaction is not None:
        payload["interaction"] = interaction
    return _success(payload)
