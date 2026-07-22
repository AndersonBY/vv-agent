from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vv_agent.runtime.engine import _register_sub_agent_session, _unregister_sub_agent_session
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_task_status import sub_task_status
from vv_agent.types import AgentResult, AgentStatus, Message, SubTaskOutcome, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend

ASSISTANT_REASONING_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "assistant_reasoning_history.json"


def _assistant_reasoning_case(name: str) -> dict[str, Any]:
    contract = json.loads(ASSISTANT_REASONING_FIXTURE_PATH.read_text(encoding="utf-8"))
    return next(case for case in contract["cases"] if case["name"] == name)


class _DummySession:
    def __init__(self) -> None:
        self._listeners = []
        self._stored_messages: list[Message] = []
        self.queued_messages: list[str] = []
        self.continue_messages_snapshot: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._stored_messages)

    def replace_messages(self, messages: list[Message]) -> None:
        self._stored_messages = list(messages)

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: None

    def emit(self, event: str, payload: dict[str, object]) -> None:
        for listener in list(self._listeners):
            listener(event, payload)

    def steer(self, prompt: str) -> None:
        self.queued_messages.append(prompt)

    def continue_run(self, prompt: str):
        self.continue_messages_snapshot = self.messages
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
        run = SimpleNamespace(
            result=AgentResult(
                status=AgentStatus.COMPLETED,
                messages=[Message(role="assistant", content="done")],
                cycles=[],
                final_answer="follow-up done",
                shared_state={"todo_list": []},
            )
        )
        self.replace_messages(list(run.result.messages))
        return run


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


def test_sub_task_status_can_wait_for_background_task_completion(tmp_path: Path) -> None:
    manager = _build_manager()
    manager.submit(
        task_id="sub-wait",
        session_id="sub-wait",
        agent_name="research-sub",
        task_title="Wait for background completion",
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        runner=lambda: (
            time.sleep(0.2)
            or SubTaskOutcome(
                task_id="sub-wait",
                session_id="sub-wait",
                agent_name="research-sub",
                status=AgentStatus.COMPLETED,
                final_answer="waited done",
                cycles=1,
            )
        ),
    )

    started_at = time.monotonic()
    result = sub_task_status(
        _build_context(tmp_path, manager),
        {
            "task_ids": ["sub-wait"],
            "wait_for_completion": True,
            "check_interval_seconds": 300,
            "max_wait_seconds": 3600,
        },
    )
    elapsed = time.monotonic() - started_at

    assert result.status_code == ToolResultStatus.SUCCESS
    assert elapsed < 1.0
    assert result.metadata["wait_for_completion"] is True
    assert result.metadata["wait_exceeded"] is False
    assert result.metadata["running_task_ids"] == []
    assert result.metadata["suggested_next_check_after_seconds"] == 300
    task_entry = result.metadata["tasks"][0]
    assert task_entry["status"] == AgentStatus.COMPLETED.value
    assert task_entry["final_answer"] == "waited done"


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


def test_sub_task_manager_sanitizes_session_messages_before_continue(tmp_path: Path) -> None:
    manager = _build_manager()
    session = _DummySession()
    reasoning_case = _assistant_reasoning_case("reasoning_only_assistant_is_preserved")
    empty_case = _assistant_reasoning_case("fully_empty_assistant_is_removed")
    reasoning_message = Message.from_dict(reasoning_case["message"])
    session.replace_messages(
        [
            Message(role="system", content="sys"),
            reasoning_message,
            Message.from_dict(empty_case["message"]),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "tool-1", "name": "read_file", "arguments": {"path": "README.md"}}],
            ),
        ]
    )
    manager.attach_session(
        task_id="sub-sanitize",
        session_id="sub-sanitize",
        agent_name="research-sub",
        task_title="Initial task",
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        session=session,
        resolved={},
        event_forwarder=lambda *_args, **_kwargs: None,
    )
    manager.record_outcome(
        "sub-sanitize",
        SubTaskOutcome(
            task_id="sub-sanitize",
            session_id="sub-sanitize",
            agent_name="research-sub",
            status=AgentStatus.COMPLETED,
            final_answer="initial done",
        ),
    )

    manager.continue_task(task_id="sub-sanitize", prompt="resume")
    manager.wait("sub-sanitize", timeout=5.0)

    assert reasoning_case["expected"]["retain_in_resumable_history"] is True
    assert empty_case["expected"]["retain_in_resumable_history"] is False
    assert session.continue_messages_snapshot == [Message(role="system", content="sys"), reasoning_message]
    assert session.continue_messages_snapshot[1].content == reasoning_case["expected"]["visible_content"]
    assert session.continue_messages_snapshot[1].reasoning_content == reasoning_case["expected"]["reasoning_content"]
    assert session.messages == [Message(role="assistant", content="done")]
