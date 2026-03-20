from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vv_agent.runtime.engine import _register_sub_agent_session, _unregister_sub_agent_session
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_task_status import sub_task_status
from vv_agent.types import AgentResult, AgentStatus, Message, SubTaskOutcome, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


class _DummySession:
    def __init__(self) -> None:
        self._listeners = []
        self.queued_messages: list[str] = []

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: None

    def emit(self, event: str, payload: dict[str, object]) -> None:
        for listener in list(self._listeners):
            listener(event, payload)

    def steer(self, prompt: str) -> None:
        self.queued_messages.append(prompt)

    def continue_run(self, prompt: str):
        self.emit("session_run_start", {"prompt": prompt})
        self.emit("cycle_started", {"cycle": 2})
        self.emit("cycle_llm_response", {"cycle": 2, "assistant_preview": "Finishing the follow-up"})
        self.emit("tool_result", {"tool_name": "write_file", "tool_call_id": "tool-2", "status": "SUCCESS"})
        self.emit("run_completed", {"final_answer": "follow-up done"})
        self.emit(
            "session_run_end",
            {
                "status": AgentStatus.COMPLETED.value,
                "cycles": 2,
                "final_answer": "follow-up done",
            },
        )
        return SimpleNamespace(
            result=AgentResult(
                status=AgentStatus.COMPLETED,
                messages=[Message(role="assistant", content="done")],
                cycles=[],
                final_answer="follow-up done",
                shared_state={"todo_list": []},
            )
        )


class _AliveThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        del timeout


def _build_manager() -> SubTaskManager:
    return SubTaskManager(
        register_session=lambda *_args, **_kwargs: None,
        unregister_session=lambda *_args, **_kwargs: None,
    )


def _build_context(tmp_path: Path, manager: SubTaskManager) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        sub_task_manager=manager,
        task_id="parent",
    )


def test_sub_task_status_snapshot_exposes_recent_activity_and_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    hidden_dir = tmp_path / ".internal"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("ignore", encoding="utf-8")

    manager = _build_manager()
    session = _DummySession()
    record = manager.attach_session(
        task_id="sub-1",
        session_id="sub-1",
        agent_name="research-sub",
        task_title="Inspect docs",
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        session=session,
        resolved={"backend": "moonshot"},
        event_forwarder=lambda *_args, **_kwargs: None,
    )
    record.thread = _AliveThread()
    session.emit("session_run_start", {"prompt": "Inspect docs"})
    session.emit("cycle_started", {"cycle": 1})
    session.emit("cycle_llm_response", {"cycle": 1, "assistant_preview": "Reading the workspace files"})
    session.emit("tool_result", {"tool_name": "read_file", "tool_call_id": "tool-1", "status": "SUCCESS"})

    result = sub_task_status(
        _build_context(tmp_path, manager),
        {
            "task_ids": ["sub-1"],
            "detail_level": "snapshot",
        },
    )

    assert result.status_code == ToolResultStatus.SUCCESS
    task_entry = result.metadata["tasks"][0]
    assert task_entry["status"] == AgentStatus.RUNNING.value
    assert task_entry["snapshot"]["recent_activity"] == "Reading the workspace files"
    assert task_entry["snapshot"]["latest_tool_call"]["name"] == "read_file"
    assert task_entry["snapshot"]["workspace_files"] == ["notes.md"]


def test_sub_task_status_can_queue_message_for_running_task(tmp_path: Path) -> None:
    manager = _build_manager()
    session = _DummySession()
    record = manager.attach_session(
        task_id="sub-running",
        session_id="sub-running",
        agent_name="research-sub",
        task_title="Long running task",
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        session=session,
        resolved={},
        event_forwarder=lambda *_args, **_kwargs: None,
    )
    record.thread = _AliveThread()

    _register_sub_agent_session("sub-running", session)
    try:
        result = sub_task_status(
            _build_context(tmp_path, manager),
            {
                "task_ids": ["sub-running"],
                "message": "Focus on the README first",
            },
        )
    finally:
        _unregister_sub_agent_session("sub-running", session)

    assert result.status_code == ToolResultStatus.SUCCESS
    assert result.metadata["interaction"]["action"] == "message_queued"
    assert session.queued_messages == ["Focus on the README first"]


def test_sub_task_status_can_continue_completed_task(tmp_path: Path) -> None:
    manager = _build_manager()
    session = _DummySession()
    manager.attach_session(
        task_id="sub-completed",
        session_id="sub-completed",
        agent_name="research-sub",
        task_title="Initial task",
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        session=session,
        resolved={"backend": "moonshot"},
        event_forwarder=lambda *_args, **_kwargs: None,
    )
    manager.record_outcome(
        "sub-completed",
        SubTaskOutcome(
            task_id="sub-completed",
            session_id="sub-completed",
            agent_name="research-sub",
            status=AgentStatus.COMPLETED,
            final_answer="initial done",
            resolved={"backend": "moonshot"},
        ),
    )

    result = sub_task_status(
        _build_context(tmp_path, manager),
        {
            "task_ids": ["sub-completed"],
            "message": "Add a short appendix",
            "wait_for_response": True,
            "detail_level": "snapshot",
        },
    )

    assert result.status_code == ToolResultStatus.SUCCESS
    task_entry = result.metadata["tasks"][0]
    assert result.metadata["interaction"]["action"] == "continued"
    assert task_entry["status"] == AgentStatus.COMPLETED.value
    assert task_entry["final_answer"] == "follow-up done"
    assert task_entry["snapshot"]["latest_tool_call"]["name"] == "write_file"
