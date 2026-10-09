"""Sub-task status reads retained child sessions, not a Python session registry."""

from pathlib import Path
from threading import Event, Thread

from support.kernel_runtime import start_runner

from vv_agent import Agent, RunConfig, ScriptedModelProvider, SubAgentConfig
from vv_agent.session.children import child_handles
from vv_agent.session.delegation import status_with_store
from vv_agent.session.kernel import drive
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.base import ToolContext
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def _completed_child(driver, workspace, continuation):
    steps = [
        LLMResponse("", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "first"})]),
        LLMResponse("child done"),
        LLMResponse("parent done"),
        continuation,
    ]
    parent = start_runner(
        driver,
        "parent",
        Agent("parent", "Delegate.", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
        "go",
        run_config=RunConfig(workspace=workspace, model_provider=ScriptedModelProvider.from_steps("test", "m", steps)),
    )
    assert parent.result().status is AgentStatus.COMPLETED
    parent.runtime.wake = lambda _sid: None
    rows = driver.store.read_state(parent.session_id)[1]
    parked = next(
        row.record for row in rows if row.record.kind == "op_parked" and row.record.payload["handle"]["kind"] == "child"
    )
    child = child_handles(parked.payload["handle"])[0]["session_id"]
    manager = parent.runtime.child_tasks(driver.store, parent.session_id)
    context = ToolContext(
        workspace=workspace,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(workspace),
        sub_task_manager=manager,
        idempotency_key="status",
    )
    return parent, child, manager, context


def test_sub_task_status_snapshot_exposes_retained_cycle_and_visible_workspace_files(tmp_path: Path):
    (tmp_path / "notes.md").write_text("# Notes\n")
    (tmp_path / ".internal").mkdir()
    (tmp_path / ".internal" / "secret.txt").write_text("hidden")
    driver = SessionDriver()
    try:
        _, child, _, context = _completed_child(driver, tmp_path, LLMResponse("next"))
        result = status_with_store(context, {"task_ids": [child], "detail_level": "snapshot"})
        assert result.status_code is ToolResultStatus.SUCCESS
        entry = result.metadata["tasks"][0]
        assert entry["status"] == "completed" and entry["final_answer"] == "child done"
        assert entry["snapshot"]["latest_cycle"]["cycle_index"] == 1
        assert entry["snapshot"]["workspace_files"] == ["notes.md"]
    finally:
        driver.close()


def test_sub_task_status_queues_continuation_without_replacing_history(tmp_path):
    requests = []
    driver = SessionDriver()
    try:
        parent, child, manager, context = _completed_child(
            driver, tmp_path, lambda request: requests.append(request) or LLMResponse("continued")
        )
        original = driver.store.read_state(child)[1]
        result = status_with_store(context, {"task_ids": [child], "message": "appendix"})
        assert result.metadata["interaction"]["action"] == "continued"
        assert result.metadata["tasks"][0]["status"] == "running"
        drive(driver.store, child, runtime=parent.runtime.child_runtime(driver.store, child))
        assert manager.get(child).outcome.final_answer == "continued"
        assert driver.store.read_state(child)[1][: len(original)] == original
        assert any(message.content == "child done" for message in requests[0].messages)
        assert any(message.content == "appendix" for message in requests[0].messages)
    finally:
        driver.close()


def test_sub_task_status_waits_for_the_retained_continuation(tmp_path):
    ready, finished = Event(), Event()
    errors = []
    driver = SessionDriver()
    try:
        parent, child, manager, context = _completed_child(driver, tmp_path, LLMResponse("continued"))
        original_message = manager.message

        def message(*args):
            result = original_message(*args)
            ready.set()
            return result

        manager.message = message
        results = []

        def status():
            try:
                results.append(
                    status_with_store(context, {"task_ids": [child], "message": "appendix", "wait_for_response": True})
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = Thread(target=status)
        worker.start()
        assert ready.wait(2) and not finished.is_set()
        drive(driver.store, child, runtime=parent.runtime.child_runtime(driver.store, child))
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert results[0].metadata["tasks"][0]["final_answer"] == "continued"
        assert results[0].metadata["tasks"][0]["status"] == "completed"
    finally:
        driver.close()
