from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from v_agent.runtime.backends.thread import ThreadBackend
from v_agent.runtime.context import ExecutionContext
from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.sub_agents import batch_sub_tasks
from v_agent.types import AgentStatus, SubTaskOutcome, SubTaskRequest, ToolResultStatus


class TestParallelSubTasks:
    def test_batch_sub_tasks_uses_parallel_map(self, tmp_path: Path):
        """batch_sub_tasks uses execution_backend.parallel_map when available via ctx."""
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
            sub_task_runner=mock_runner,
            ctx=ctx,
        )

        result = batch_sub_tasks(context, {
            "agent_name": "worker",
            "tasks": [
                {"task_description": "task A"},
                {"task_description": "task B"},
                {"task_description": "task C"},
            ],
        })

        assert result.status_code == ToolResultStatus.SUCCESS
        assert set(call_log) == {"task A", "task B", "task C"}

    def test_batch_sub_tasks_fallback_serial(self, tmp_path: Path):
        """batch_sub_tasks falls back to serial when no backend in ctx."""
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
            sub_task_runner=mock_runner,
        )

        result = batch_sub_tasks(context, {
            "agent_name": "worker",
            "tasks": [
                {"task_description": "first"},
                {"task_description": "second"},
            ],
        })

        assert result.status_code == ToolResultStatus.SUCCESS
        assert call_order == ["first", "second"]

    def test_thread_backend_parallel_map_concurrency(self):
        """Verify ThreadBackend.parallel_map runs items concurrently."""
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
