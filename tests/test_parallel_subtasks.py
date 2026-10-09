"""Configured batches use retained kernel child admissions and results."""

import json
from threading import Event

import pytest
from session.test_delegation_parity import routed
from support.kernel_runtime import start_runner

from vv_agent import Agent, RunConfig, SubAgentConfig
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolResultStatus
from vv_agent.workspace import INVALID_EXCLUDE_FILES_PATTERN_CODE, INVALID_EXCLUDE_FILES_PATTERN_MESSAGE


@pytest.fixture
def driver():
    value = SessionDriver()
    try:
        yield value
    finally:
        value.close()


def start(driver, workspace, arguments, children):
    return start_runner(
        driver,
        "parent",
        Agent("parent", "Delegate.", model="parent", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
        "go",
        run_config=RunConfig(
            workspace=workspace,
            model_provider=routed(
                [LLMResponse("", [ToolCall("delegate", "create_sub_task", arguments)]), LLMResponse("parent done")],
                children,
            ),
        ),
    )


def tool_result(handle):
    result = handle.result()
    assert result.status is AgentStatus.COMPLETED
    return result.raw_result.cycles[0].tool_results[0]


def test_create_sub_task_lineage_is_frozen_from_parent_turn(driver, tmp_path):
    handle = start(driver, tmp_path, {"agent_id": "worker", "task_description": "work"}, [LLMResponse("done")])
    assert tool_result(handle).status_code is ToolResultStatus.SUCCESS
    child = next(sid for sid in driver.store.list_sessions() if sid != "parent")
    created = driver.store.read(child, limit=1).records[0].record
    metadata = created.payload["attributes"]["child_admission"]["definition"]["task"]["metadata"]
    assert metadata["parent_run_id"] == handle.run_id
    assert metadata["parent_tool_call_id"] == "delegate"
    assert metadata["session_id"] == child


@pytest.mark.parametrize("pattern", [r"(?=secret)", r"(a)\1", r"\p{Greek}"])
def test_create_sub_task_rejects_non_portable_regex_before_admission(driver, tmp_path, pattern):
    handle = start(driver, tmp_path, {"agent_id": "worker", "task_description": "work", "exclude_files_pattern": pattern}, [])
    result = tool_result(handle)
    assert result.status_code is ToolResultStatus.ERROR
    assert result.error_code == INVALID_EXCLUDE_FILES_PATTERN_CODE
    assert result.metadata == {
        "ok": False,
        "error": INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
        "error_code": INVALID_EXCLUDE_FILES_PATTERN_CODE,
    }
    assert driver.store.list_sessions() == ("parent",)


def test_create_sub_task_accepts_portable_non_capturing_group(driver, tmp_path):
    pattern = r"^(?:generated|logs)/"
    handle = start(
        driver,
        tmp_path,
        {"agent_id": "worker", "task_description": "work", "exclude_files_pattern": pattern},
        [LLMResponse("done")],
    )
    assert tool_result(handle).status_code is ToolResultStatus.SUCCESS
    child = next(sid for sid in driver.store.list_sessions() if sid != "parent")
    assert (
        driver.store.read(child, limit=1).records[0].record.payload["attributes"]["child_admission"]["exclude_files_pattern"]
        == pattern
    )


@pytest.mark.parametrize("wait", [False, True])
def test_create_sub_task_batch_has_durable_identities_and_terminal_results(driver, tmp_path, wait):
    entered, release = Event(), Event()

    def blocked(_request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("child done")

    args = {
        "agent_id": "worker",
        "tasks": [{"task_description": "first"}, {"task_description": "second"}],
        "wait_for_completion": wait,
    }
    handle = start(driver, tmp_path, args, [LLMResponse("child done")] * 2 if wait else [blocked] * 2)
    try:
        result = tool_result(handle)
        if not wait:
            assert entered.wait(2)
    finally:
        release.set()
    for child_handle in driver.handles:
        child_handle.join(5)
        assert not child_handle._thread.is_alive()
    payload = json.loads(result.content)
    ids = [entry["task_id"] for entry in payload["results"]]
    assert len(set(ids)) == 2 and payload["summary"]["total"] == 2
    if not wait:
        assert payload["task_ids"] == ids
        assert all(entry["status"] == "running" for entry in payload["results"])
    else:
        assert all(entry["status"] == "completed" and entry["final_answer"] == "child done" for entry in payload["results"])
    manager = handle.runtime.child_tasks(driver.store, "parent")
    for sid in ids:
        entry = manager.get(sid)
        assert entry is not None and entry.outcome.status is AgentStatus.COMPLETED
        assert entry.outcome.final_answer == "child done"
        assert len(driver.store.read_state(sid)[0].turns) == 1


def test_create_sub_task_requires_agent_id_before_admission(driver, tmp_path):
    handle = start(driver, tmp_path, {"task_description": "work"}, [])
    result = tool_result(handle)
    assert result.status_code is ToolResultStatus.ERROR
    assert result.error_code == "invalid_tool_arguments"
    assert driver.store.list_sessions() == ("parent",)
