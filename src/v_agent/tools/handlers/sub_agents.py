from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import AgentStatus, SubTaskRequest, ToolExecutionResult, ToolResultStatus


def _resolve_agent_name(arguments: dict[str, Any]) -> str:
    for key in ("agent_name", "agent_id"):
        raw = arguments.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return ""


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


def create_sub_task(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    if context.sub_task_runner is None:
        return _error("Sub-agent runtime is not available for this task", error_code="sub_agents_not_enabled")

    agent_name = _resolve_agent_name(arguments)
    task_description = str(arguments.get("task_description", "")).strip()
    if not agent_name:
        return _error("`agent_name` is required", error_code="agent_name_required")
    if not task_description:
        return _error("`task_description` is required", error_code="task_description_required")

    request = SubTaskRequest(
        agent_name=agent_name,
        task_description=task_description,
        output_requirements=str(arguments.get("output_requirements", "")).strip(),
        include_main_summary=bool(arguments.get("include_main_summary", False)),
        exclude_files_pattern=(
            str(arguments.get("exclude_files_pattern")).strip() if arguments.get("exclude_files_pattern") is not None else None
        ),
    )
    outcome = context.sub_task_runner(request)
    payload = outcome.to_dict()

    if outcome.status == AgentStatus.COMPLETED:
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.SUCCESS,
            content=to_json(payload),
            metadata=payload,
        )

    error_code = "sub_task_wait_user" if outcome.status == AgentStatus.WAIT_USER else "sub_task_failed"
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json(payload),
        metadata=payload,
    )


def batch_sub_tasks(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    if context.sub_task_runner is None:
        return _error("Sub-agent runtime is not available for this task", error_code="sub_agents_not_enabled")

    agent_name = _resolve_agent_name(arguments)
    raw_tasks = arguments.get("tasks")
    if not agent_name:
        return _error("`agent_name` is required", error_code="agent_name_required")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return _error("`tasks` must be a non-empty array", error_code="invalid_tasks_payload")

    include_main_summary = bool(arguments.get("include_main_summary", False))
    exclude_files_pattern = (
        str(arguments.get("exclude_files_pattern")).strip() if arguments.get("exclude_files_pattern") is not None else None
    )

    results: list[dict[str, Any]] = []
    completed = 0
    failed = 0

    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            failed += 1
            results.append(
                {
                    "index": index,
                    "status": AgentStatus.FAILED.value,
                    "error": "Task item must be an object",
                }
            )
            continue

        task_description = str(item.get("task_description", "")).strip()
        if not task_description:
            failed += 1
            results.append(
                {
                    "index": index,
                    "status": AgentStatus.FAILED.value,
                    "error": "`task_description` is required",
                }
            )
            continue

        request = SubTaskRequest(
            agent_name=agent_name,
            task_description=task_description,
            output_requirements=str(item.get("output_requirements", "")).strip(),
            include_main_summary=include_main_summary,
            exclude_files_pattern=exclude_files_pattern,
            metadata={"batch_index": index},
        )
        outcome = context.sub_task_runner(request)
        if outcome.status == AgentStatus.COMPLETED:
            completed += 1
        else:
            failed += 1

        item_payload = outcome.to_dict()
        item_payload["index"] = index
        results.append(item_payload)

    payload = {
        "summary": {
            "total": len(raw_tasks),
            "completed": completed,
            "failed": failed,
        },
        "results": results,
    }
    if completed == 0:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="batch_sub_tasks_failed",
            content=to_json(payload),
            metadata=payload,
        )

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        metadata=payload,
    )
