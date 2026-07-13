from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from vv_agent.runtime.sub_task_identity import assigned_sub_task_identity, normalize_identity_string
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import is_string_keyed_dict, to_json, trim_portable_whitespace
from vv_agent.types import AgentStatus, SubTaskRequest, ToolExecutionResult, ToolResultStatus
from vv_agent.workspace import (
    INVALID_EXCLUDE_FILES_PATTERN_CODE,
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    InvalidPortableRegexError,
    WorkspaceBackend,
)


def _resolve_agent_name(arguments: dict[str, Any]) -> tuple[str | None, ToolExecutionResult | None]:
    if "agent_id" not in arguments:
        return "", None
    raw = arguments["agent_id"]
    if not isinstance(raw, str):
        return None, _error("`agent_id` must be a string", error_code="invalid_agent_id")
    return trim_portable_whitespace(raw), None


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


def _build_async_identity(context: ToolContext, agent_name: str) -> tuple[str, str]:
    parent_task_id = str(context.task_id or "task").strip() or "task"
    task_id = f"{parent_task_id}_sub_{agent_name}_{uuid.uuid4().hex[:8]}"
    return task_id, task_id


def _run_with_assigned_identity(
    sub_task_runner: Any,
    request: SubTaskRequest,
    task_id: str,
    session_id: str,
) -> Any:
    with assigned_sub_task_identity(task_id, session_id):
        outcome = sub_task_runner(request)
    return replace(outcome, task_id=task_id, session_id=session_id)


def _extract_shared_flags(
    arguments: dict[str, Any],
) -> tuple[bool, str | None, ToolExecutionResult | None]:
    include_main_summary = _coerce_bool(arguments.get("include_main_summary"), default=False)
    exclude_files_pattern = None
    if "exclude_files_pattern" in arguments:
        raw_exclude_files_pattern = arguments["exclude_files_pattern"]
        if not isinstance(raw_exclude_files_pattern, str):
            return False, None, _error(
                "`exclude_files_pattern` must be a string",
                error_code="invalid_exclude_files_pattern",
            )
        exclude_files_pattern = trim_portable_whitespace(raw_exclude_files_pattern) or None
    return include_main_summary, exclude_files_pattern, None


def _optional_string(
    arguments: dict[str, Any],
    key: str,
    *,
    error_code: str,
) -> tuple[str | None, ToolExecutionResult | None]:
    if key not in arguments:
        return None, None
    raw = arguments[key]
    if not isinstance(raw, str):
        return None, _error(f"`{key}` must be a string", error_code=error_code)
    return trim_portable_whitespace(raw), None


def _build_single_request(
    *,
    agent_name: str,
    task_description: str,
    output_requirements: str,
    include_main_summary: bool,
    exclude_files_pattern: str | None,
    metadata: dict[str, Any] | None = None,
) -> SubTaskRequest:
    return SubTaskRequest(
        agent_name=agent_name,
        task_description=task_description,
        output_requirements=output_requirements,
        include_main_summary=include_main_summary,
        exclude_files_pattern=exclude_files_pattern,
        metadata=dict(metadata or {}),
    )


def _parent_lineage_metadata(context: ToolContext) -> dict[str, str]:
    metadata: dict[str, str] = {}
    parent_tool_call_id = normalize_identity_string(context.tool_call_id)
    if parent_tool_call_id:
        metadata["parent_tool_call_id"] = parent_tool_call_id
    parent_run_id = normalize_identity_string(getattr(context.run_context, "run_id", None))
    runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
    if not parent_run_id:
        parent_run_id = normalize_identity_string(runtime_metadata.get("_vv_agent_run_id"))
    if parent_run_id:
        metadata["parent_run_id"] = parent_run_id
    return metadata


def _run_requests_in_parallel_if_possible(
    *,
    context: ToolContext,
    requests: list[tuple[int, SubTaskRequest]],
) -> dict[int, Any]:
    sub_task_runner = context.sub_task_runner
    if sub_task_runner is None:
        raise RuntimeError("Sub-agent runtime is not available for this task")

    execution_backend = None
    if context.ctx is not None:
        execution_backend = context.ctx.metadata.get("execution_backend")
    if execution_backend is not None and hasattr(execution_backend, "parallel_map") and requests:
        outcomes = execution_backend.parallel_map(
            lambda item: (item[0], sub_task_runner(item[1])),
            requests,
        )
        return dict(outcomes)

    outcome_map: dict[int, Any] = {}
    for index, request in requests:
        outcome_map[index] = sub_task_runner(request)
    return outcome_map


def _format_single_sync_result(outcome: Any) -> ToolExecutionResult:
    payload = outcome.to_dict()
    if outcome.status == AgentStatus.COMPLETED:
        return _success(payload)

    error_code = (
        outcome.error_code
        if isinstance(outcome.error_code, str) and trim_portable_whitespace(outcome.error_code)
        else None
    ) or (
        "sub_task_wait_user" if outcome.status == AgentStatus.WAIT_USER else "sub_task_failed"
    )
    payload["error_code"] = error_code
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json(payload),
        metadata=payload,
    )


def _manager_workspace_backend(
    workspace_backend: WorkspaceBackend,
    exclude_files_pattern: str | None,
) -> WorkspaceBackend:
    if exclude_files_pattern is None or not exclude_files_pattern.strip():
        return workspace_backend
    return DiscoveryFilteredWorkspaceBackend(workspace_backend, exclude_files_pattern)


def create_sub_task(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    sub_task_runner = context.sub_task_runner
    if sub_task_runner is None:
        return _error("Sub-agent runtime is not available for this task", error_code="sub_agents_not_enabled")

    agent_name, argument_error = _resolve_agent_name(arguments)
    if argument_error is not None:
        return argument_error
    if not agent_name:
        return _error("`agent_id` is required", error_code="agent_id_required")

    include_main_summary, exclude_files_pattern, argument_error = _extract_shared_flags(arguments)
    if argument_error is not None:
        return argument_error

    task_description, argument_error = _optional_string(
        arguments,
        "task_description",
        error_code="invalid_tasks_payload",
    )
    if argument_error is not None:
        return argument_error
    wait_for_completion = _coerce_bool(arguments.get("wait_for_completion"), default=True)
    parent_lineage = _parent_lineage_metadata(context)

    task_description = task_description or ""
    raw_tasks = arguments.get("tasks")
    has_single = bool(task_description)
    has_batch = "tasks" in arguments

    if has_single and has_batch:
        return _error(
            "`task_description` and `tasks` are mutually exclusive",
            error_code="sub_task_payload_conflict",
        )
    if not has_single and not has_batch:
        return _error(
            "Provide either `task_description` or `tasks`",
            error_code="sub_task_payload_missing",
        )

    if has_batch and (not isinstance(raw_tasks, list) or not raw_tasks):
        return _error("`tasks` must be a non-empty array", error_code="invalid_tasks_payload")

    if has_single:
        output_requirements, argument_error = _optional_string(
            arguments,
            "output_requirements",
            error_code="invalid_tasks_payload",
        )
        if argument_error is not None:
            return argument_error
        try:
            manager_workspace_backend = _manager_workspace_backend(context.workspace_backend, exclude_files_pattern)
        except InvalidPortableRegexError:
            return _error(
                INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
                error_code=INVALID_EXCLUDE_FILES_PATTERN_CODE,
            )
        single_request = _build_single_request(
            agent_name=agent_name,
            task_description=task_description,
            output_requirements=output_requirements or "",
            include_main_summary=include_main_summary,
            exclude_files_pattern=exclude_files_pattern,
            metadata=parent_lineage,
        )
        if wait_for_completion:
            outcome = sub_task_runner(single_request)
            return _format_single_sync_result(outcome)

        if context.sub_task_manager is None:
            return _error("Sub-task manager is not available for async mode", error_code="sub_task_manager_unavailable")
        task_id, session_id = _build_async_identity(context, agent_name)

        def run_single_async() -> Any:
            return _run_with_assigned_identity(
                sub_task_runner,
                single_request,
                task_id,
                session_id,
            )

        try:
            context.sub_task_manager.submit(
                task_id=task_id,
                session_id=session_id,
                agent_name=agent_name,
                task_title=single_request.task_description,
                workspace_backend=manager_workspace_backend,
                parent_run_id=parent_lineage.get("parent_run_id"),
                parent_tool_call_id=parent_lineage.get("parent_tool_call_id"),
                runner=run_single_async,
            )
        except Exception as exc:
            return _error(str(exc), error_code="sub_task_submit_failed")
        return _success(
            {
                "task_id": task_id,
                "session_id": session_id,
                "agent_name": agent_name,
                "status": AgentStatus.RUNNING.value,
                "task_description": single_request.task_description,
                "wait_for_completion": False,
            }
        )

    assert isinstance(raw_tasks, list)

    requests: list[tuple[int, SubTaskRequest | None, str | None]] = []
    for index, raw_item in enumerate(raw_tasks):
        if not is_string_keyed_dict(raw_item):
            requests.append((index, None, "Task item must be an object"))
            continue
        raw_item_description = raw_item.get("task_description")
        if "task_description" in raw_item and not isinstance(raw_item_description, str):
            requests.append((index, None, "`task_description` must be a string"))
            continue
        item_description = trim_portable_whitespace(raw_item_description or "")
        if not item_description:
            requests.append((index, None, "`task_description` is required"))
            continue
        raw_output_requirements = raw_item.get("output_requirements")
        if "output_requirements" in raw_item and not isinstance(raw_output_requirements, str):
            requests.append((index, None, "`output_requirements` must be a string"))
            continue
        requests.append(
            (
                index,
                _build_single_request(
                    agent_name=agent_name,
                    task_description=item_description,
                    output_requirements=trim_portable_whitespace(raw_output_requirements or ""),
                    include_main_summary=include_main_summary,
                    exclude_files_pattern=exclude_files_pattern,
                    metadata={"batch_index": index, **parent_lineage},
                ),
                None,
            )
        )

    valid_requests = [(index, request) for index, request, error in requests if request is not None and error is None]
    if not valid_requests:
        payload = {
            "summary": {"total": len(raw_tasks), "accepted": 0, "failed": len(raw_tasks)},
            "results": [{"index": index, "status": AgentStatus.FAILED.value, "error": error} for index, _, error in requests],
            "task_ids": [],
            "wait_for_completion": wait_for_completion,
        }
        return _error("No valid sub-tasks were provided", error_code="invalid_tasks_payload", details=payload)

    try:
        manager_workspace_backend = _manager_workspace_backend(context.workspace_backend, exclude_files_pattern)
    except InvalidPortableRegexError:
        return _error(
            INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
            error_code=INVALID_EXCLUDE_FILES_PATTERN_CODE,
        )

    if wait_for_completion:
        outcome_map = _run_requests_in_parallel_if_possible(context=context, requests=valid_requests)
        results: list[dict[str, Any]] = []
        completed = 0
        failed = 0

        for index, _, error in requests:
            if error is not None:
                failed += 1
                results.append({"index": index, "status": AgentStatus.FAILED.value, "error": error})
                continue

            outcome = outcome_map[index]
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
            "wait_for_completion": True,
        }
        if completed == 0:
            return _error("All batch sub-tasks failed", error_code="create_sub_task_batch_failed", details=payload)
        return _success(payload)

    started = 0
    failed = 0
    task_ids: list[str] = []
    results: list[dict[str, Any]] = []

    if context.sub_task_manager is None:
        return _error("Sub-task manager is not available for async mode", error_code="sub_task_manager_unavailable")

    for index, request, error in requests:
        if error is not None or request is None:
            failed += 1
            results.append({"index": index, "status": AgentStatus.FAILED.value, "error": error})
            continue

        task_id, session_id = _build_async_identity(context, agent_name)

        def run_batch_async(
            _request: SubTaskRequest = request,
            _task_id: str = task_id,
            _session_id: str = session_id,
        ) -> Any:
            return _run_with_assigned_identity(
                sub_task_runner,
                _request,
                _task_id,
                _session_id,
            )

        try:
            context.sub_task_manager.submit(
                task_id=task_id,
                session_id=session_id,
                agent_name=agent_name,
                task_title=request.task_description,
                workspace_backend=manager_workspace_backend,
                parent_run_id=parent_lineage.get("parent_run_id"),
                parent_tool_call_id=parent_lineage.get("parent_tool_call_id"),
                runner=run_batch_async,
            )
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "index": index,
                    "task_id": task_id,
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "status": AgentStatus.FAILED.value,
                    "error": str(exc),
                    "error_code": "sub_task_submit_failed",
                }
            )
            continue
        started += 1
        task_ids.append(task_id)
        results.append(
            {
                "index": index,
                "task_id": task_id,
                "session_id": session_id,
                "agent_name": agent_name,
                "status": AgentStatus.RUNNING.value,
                "task_description": request.task_description,
            }
        )

    payload = {
        "summary": {
            "total": len(raw_tasks),
            "accepted": started,
            "failed": failed,
        },
        "task_ids": task_ids,
        "results": results,
        "wait_for_completion": False,
    }
    if started == 0:
        return _error("All batch sub-tasks failed", error_code="create_sub_task_batch_failed", details=payload)
    return _success(payload)
