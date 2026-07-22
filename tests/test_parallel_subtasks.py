from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vv_agent.agent import RunContext
from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.types import AgentStatus, SubTaskOutcome, SubTaskRequest, ToolResultStatus
from vv_agent.workspace import INVALID_EXCLUDE_FILES_PATTERN_CODE, INVALID_EXCLUDE_FILES_PATTERN_MESSAGE, LocalWorkspaceBackend


def _build_manager() -> SubTaskManager:
    return SubTaskManager(
        register_session=lambda *_args, **_kwargs: None,
        unregister_session=lambda *_args, **_kwargs: None,
    )


class TestParallelSubTasks:
    def test_create_sub_task_lineage_prefers_run_context_without_task_id_fallback(self, tmp_path: Path) -> None:
        captured: list[SubTaskRequest] = []
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=lambda request: captured.append(request) or _completed_outcome(request),
            task_id="parent-task",
            ctx=ExecutionContext(metadata={"_vv_agent_run_id": "execution-run"}),
            run_context=RunContext(run_id="public-run"),
            tool_call_id="delegate",
        )

        result = create_sub_task(
            context,
            {"agent_id": "worker", "task_description": "inspect lineage"},
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert captured[0].metadata["parent_run_id"] == "public-run"
        assert captured[0].metadata["parent_tool_call_id"] == "delegate"

        captured.clear()
        context.run_context = None
        context.ctx = None
        result = create_sub_task(
            context,
            {"agent_id": "worker", "task_description": "inspect missing lineage"},
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert "parent_run_id" not in captured[0].metadata

    @pytest.mark.parametrize("pattern", [r"(?=secret)", r"(a)\1", r"\p{Greek}"])
    def test_create_sub_task_rejects_non_portable_regex_before_start(
        self,
        tmp_path: Path,
        pattern: str,
    ) -> None:
        calls: list[SubTaskRequest] = []
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=lambda request: calls.append(request) or _completed_outcome(request),
            sub_task_manager=_build_manager(),
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {
                "agent_id": "worker",
                "task_description": "inspect files",
                "exclude_files_pattern": pattern,
            },
        )

        assert result.status_code == ToolResultStatus.ERROR
        assert result.error_code == INVALID_EXCLUDE_FILES_PATTERN_CODE
        assert result.metadata == {
            "ok": False,
            "error": INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
            "error_code": INVALID_EXCLUDE_FILES_PATTERN_CODE,
        }
        assert calls == []

    def test_create_sub_task_accepts_portable_non_capturing_group(self, tmp_path: Path) -> None:
        calls: list[SubTaskRequest] = []
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=lambda request: calls.append(request) or _completed_outcome(request),
            sub_task_manager=_build_manager(),
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {
                "agent_id": "worker",
                "task_description": "inspect files",
                "exclude_files_pattern": r"^(?:generated|logs)/",
            },
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert [request.exclude_files_pattern for request in calls] == [r"^(?:generated|logs)/"]

    def test_create_sub_task_batch_uses_parallel_map(self, tmp_path: Path):
        call_log: list[str] = []
        lock = threading.Lock()

        def mock_runner(request: SubTaskRequest) -> SubTaskOutcome:
            with lock:
                call_log.append(request.task_description)
            return SubTaskOutcome(
                task_id=f"sub_{request.task_description}",
                agent_name=request.agent_name,
                status=AgentStatus.COMPLETED,
                final_answer=f"done: {request.task_description}",
            )

        backend = ThreadBackend(max_workers=4)
        ctx = ExecutionContext(metadata={"execution_backend": backend})
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=mock_runner,
            sub_task_manager=_build_manager(),
            ctx=ctx,
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {
                "agent_id": "worker",
                "tasks": [
                    {"task_description": "task A"},
                    {"task_description": "task B"},
                    {"task_description": "task C"},
                ],
            },
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert set(call_log) == {"task A", "task B", "task C"}

    def test_create_sub_task_batch_fallback_serial(self, tmp_path: Path):
        call_order: list[str] = []

        def mock_runner(request: SubTaskRequest) -> SubTaskOutcome:
            call_order.append(request.task_description)
            return SubTaskOutcome(
                task_id=f"sub_{request.task_description}",
                agent_name=request.agent_name,
                status=AgentStatus.COMPLETED,
                final_answer=f"done: {request.task_description}",
            )

        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=mock_runner,
            sub_task_manager=_build_manager(),
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {
                "agent_id": "worker",
                "tasks": [
                    {"task_description": "first"},
                    {"task_description": "second"},
                ],
            },
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert call_order == ["first", "second"]

    def test_create_sub_task_batch_async_returns_task_ids(self, tmp_path: Path):
        def mock_runner(request: SubTaskRequest) -> SubTaskOutcome:
            time.sleep(0.05)
            return SubTaskOutcome(
                task_id=str(request.metadata.get("task_id")),
                session_id=str(request.metadata.get("session_id")),
                agent_name=request.agent_name,
                status=AgentStatus.COMPLETED,
                final_answer=f"done: {request.task_description}",
            )

        manager = _build_manager()
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=mock_runner,
            sub_task_manager=manager,
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {
                "agent_id": "worker",
                "tasks": [
                    {"task_description": "first"},
                    {"task_description": "second"},
                ],
                "wait_for_completion": False,
            },
        )

        assert result.status_code == ToolResultStatus.SUCCESS
        assert len(result.metadata["task_ids"]) == 2
        assert all(manager.wait(task_id) is not None for task_id in result.metadata["task_ids"])

    def test_create_sub_task_requires_agent_id(self, tmp_path: Path):
        context = ToolContext(
            workspace=tmp_path,
            shared_state={},
            cycle_index=1,
            workspace_backend=LocalWorkspaceBackend(tmp_path),
            sub_task_runner=lambda request: SubTaskOutcome(
                task_id="unused",
                agent_name=request.agent_name,
                status=AgentStatus.COMPLETED,
            ),
            sub_task_manager=_build_manager(),
            task_id="parent_task",
        )

        result = create_sub_task(
            context,
            {"task_description": "missing required agent id"},
        )

        assert result.status_code == ToolResultStatus.ERROR
        assert result.metadata["error_code"] == "agent_id_required"

    def test_thread_backend_parallel_map_concurrency(self):
        backend = ThreadBackend(max_workers=4)
        thread_ids: list[int] = []
        lock = threading.Lock()

        def worker(x: int) -> int:
            with lock:
                thread_ids.append(threading.current_thread().ident or 0)
            time.sleep(0.05)
            return x * 2

        results = backend.parallel_map(worker, [1, 2, 3, 4])
        assert results == [2, 4, 6, 8]
        unique_threads = set(thread_ids)
        assert len(unique_threads) >= 2, f"Expected parallel execution, got threads: {unique_threads}"


def _completed_outcome(request: SubTaskRequest) -> SubTaskOutcome:
    return SubTaskOutcome(
        task_id="sub-task",
        session_id="sub-session",
        agent_name=request.agent_name,
        status=AgentStatus.COMPLETED,
        final_answer="done",
    )
