from __future__ import annotations

from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import AgentStatus, ToolExecutionResult, ToolResultStatus

DEFAULT_SUB_TASK_SNAPSHOT_FILE_LIMIT = 20
MAX_SUB_TASK_SNAPSHOT_FILE_LIMIT = 100


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_detail_level(detail_level: Any) -> str:
    normalized = str(detail_level or "basic").strip().lower()
    return normalized if normalized in {"basic", "snapshot"} else "basic"


def _normalize_workspace_file_limit(workspace_file_limit: Any) -> int:
    try:
        limit = int(workspace_file_limit or DEFAULT_SUB_TASK_SNAPSHOT_FILE_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_SUB_TASK_SNAPSHOT_FILE_LIMIT
    return max(1, min(limit, MAX_SUB_TASK_SNAPSHOT_FILE_LIMIT))


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

    status = AgentStatus.RUNNING.value if record.is_running() else AgentStatus.PENDING.value
    outcome = getattr(record, "outcome", None)
    if outcome is not None:
        status = outcome.status.value

    entry: dict[str, Any] = {
        "task_id": record.task_id,
        "session_id": record.session_id,
        "agent_name": record.agent_name,
        "status": status,
    }
    if record.task_title:
        entry["task_description"] = record.task_title

    if outcome is not None:
        for key in ("final_answer", "wait_reason", "error"):
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
        task_id = str(item or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        task_ids.append(task_id)
    if not task_ids:
        return _error("`task_ids` must include at least one valid task id", error_code="invalid_task_ids")

    detail_level = _normalize_detail_level(arguments.get("detail_level"))
    workspace_file_limit = _normalize_workspace_file_limit(arguments.get("workspace_file_limit"))
    message = str(arguments.get("message", "")).strip() if arguments.get("message") is not None else ""
    wait_for_response = _coerce_bool(arguments.get("wait_for_response"), default=False)

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

        previous_status = record.outcome.status.value if record.outcome is not None else AgentStatus.RUNNING.value
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
                manager.continue_task(task_id=target_id, prompt=message)
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

    tasks = [
        _build_status_entry(
            task_id=task_id,
            record=manager.get(task_id),
            detail_level=detail_level,
            workspace_file_limit=workspace_file_limit,
        )
        for task_id in task_ids
    ]

    payload: dict[str, Any] = {
        "tasks": tasks,
        "detail_level": detail_level,
    }
    if interaction is not None:
        payload["interaction"] = interaction
    return _success(payload)
