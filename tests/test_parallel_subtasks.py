from __future__ import annotations

import threading
import time
from pathlib import Path

from vv_agent.runtime.backends.thread import ThreadBackend
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.types import AgentStatus, SubTaskOutcome, SubTaskRequest, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def _build_manager() -> SubTaskManager:
    return SubTaskManager(
        register_session=lambda *_args, **_kwargs: None,
        unregister_session=lambda *_args, **_kwargs: None,
    )


class TestParallelSubTasks:
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
                "agent_name": "worker",
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
                "agent_name": "worker",
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
                "agent_name": "worker",
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
