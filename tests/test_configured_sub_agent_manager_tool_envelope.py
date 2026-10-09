from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from session.test_delegation_parity import routed
from support import require_tool_result
from support.kernel_runtime import start_runner

from vv_agent import Agent, RunConfig, ScriptedModelProvider, SubAgentConfig
from vv_agent.session.delegation import status_with_store
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.outcomes import HostToolOutcome
from vv_agent.types import (
    AgentStatus,
    CompletionReason,
    LLMResponse,
    SubTaskOutcome,
    ToolCall,
    ToolExecutionResult,
    ToolResultStatus,
)
from vv_agent.workspace import MemoryWorkspaceBackend

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "manager_tool_envelope.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_bytes())


@pytest.fixture
def manager_driver():
    driver = SessionDriver()
    try:
        yield driver
    finally:
        driver.close()


def _manager(driver, workspace):
    driver.create("parent", str(workspace))
    runtime = driver.runtime(Agent("parent", "Delegate."), RunConfig(model_provider=ScriptedModelProvider.new("test", "m", [])))
    return runtime.child_tasks(driver.store, "parent")


def _child(driver, workspace, response):
    parent = start_runner(
        driver,
        "parent",
        Agent("parent", "Delegate.", model="parent", sub_agents={"researcher": SubAgentConfig(model="m", description="Work.")}),
        "go",
        run_config=RunConfig(
            workspace=workspace,
            model_provider=routed(
                [
                    LLMResponse(
                        "", [ToolCall("delegate", "create_sub_task", {"agent_id": "researcher", "task_description": "work"})]
                    ),
                    LLMResponse("parent done"),
                ],
                [response, response] if callable(response) else [response],
            ),
        ),
    )
    assert parent.result().status is AgentStatus.COMPLETED
    sid = next(sid for sid in driver.store.list_sessions() if sid != "parent")
    manager = parent.runtime.child_tasks(driver.store, "parent")
    context = _context(workspace)
    context.sub_task_manager = manager
    return parent, sid, manager, context


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
    )


def _assert_error_metadata_matches_content(result: ToolExecutionResult | HostToolOutcome) -> None:
    result = require_tool_result(result)
    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.ERROR
    assert result.metadata == payload


def _assert_schema_error_metadata(result: ToolExecutionResult | HostToolOutcome) -> None:
    result = require_tool_result(result)
    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.ERROR
    assert payload["error_code"] == "invalid_tool_arguments"
    assert result.metadata == {
        "error_code": "invalid_tool_arguments",
        "issue_count": len(payload["issues"]),
    }


@pytest.mark.parametrize("case", _fixture()["create_error_cases"], ids=lambda case: case["name"])
def test_create_sub_task_error_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any]) -> None:
    context = _context(tmp_path)
    context.sub_task_runner = lambda request: SubTaskOutcome(
        task_id="child",
        agent_name=request.agent_name,
        status=AgentStatus.COMPLETED,
    )

    result = build_default_registry().execute(
        ToolCall(id=case["name"], name="create_sub_task", arguments=case["arguments"]),
        context,
    )
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert payload == case["expected"]
    assert result.error_code == case["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


@pytest.mark.parametrize("case", _fixture()["status_error_cases"], ids=lambda case: case["name"])
def test_sub_task_status_error_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any], manager_driver) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager(manager_driver, tmp_path)

    result = build_default_registry().execute(
        ToolCall(id=case["name"], name="sub_task_status", arguments=case["arguments"]),
        context,
    )
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert payload == case["expected"]
    assert result.error_code == case["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


@pytest.mark.parametrize("case", _fixture()["status_success_cases"], ids=lambda case: case["name"])
def test_sub_task_status_success_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any], manager_driver) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager(manager_driver, tmp_path)

    result = build_default_registry().execute(
        ToolCall(id=case["name"], name="sub_task_status", arguments=case["arguments"]),
        context,
    )
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert payload == case["expected"]
    assert result.status_code == ToolResultStatus.SUCCESS
    assert result.error_code is None
    assert result.metadata == payload


def test_sync_failed_outcome_normalizes_blank_error_code(tmp_path: Path) -> None:
    contract = _fixture()["sync_failed_outcome"]
    context = _context(tmp_path)
    context.sub_task_runner = lambda _request: SubTaskOutcome(
        task_id="failed-child",
        agent_name="researcher",
        status=AgentStatus.FAILED,
        completion_reason=CompletionReason.FAILED,
        partial_output="last child draft",
        error="child failed",
        error_code=contract["input_error_code"],
    )

    result = (
        build_default_registry()
        .get("create_sub_task")
        .handler(
            context,
            {"agent_id": "researcher", "task_description": "fail"},
        )
    )
    result = require_tool_result(result)

    payload = json.loads(result.content)
    assert payload == contract["expected"]
    assert result.error_code == contract["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


def test_sync_wait_stays_on_child_until_terminal_delivery():
    from support.kernel_runtime import start_runner

    from vv_agent import Agent, RunConfig, Runner, ScriptedModelProvider, SubAgentConfig, function_tool
    from vv_agent.session.children import child_handles
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse, ToolDirective

    @function_tool
    def wait_tool():
        return ToolExecutionResult("", "Choose", directive=ToolDirective.WAIT_USER)

    contract = _fixture()["sync_wait_outcome"]
    driver = SessionDriver()
    try:
        parent = start_runner(
            driver,
            "parent",
            Agent(
                "parent", "Delegate.", tools=[wait_tool], sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}
            ),
            "go",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_steps(
                    "scripted",
                    "m",
                    [
                        LLMResponse(
                            "", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]
                        ),
                        LLMResponse("", [ToolCall("ask", "wait_tool", {})]),
                        LLMResponse("child done"),
                        LLMResponse("parent done"),
                    ],
                )
            ),
        )
        assert parent.result().status is AgentStatus.WAIT_USER
        rows = driver.store.read_state("parent")[1]
        parked = next(r.record for r in rows if r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "child")
        child = child_handles(parked.payload["handle"])[0]
        assert not contract["parent_adopts_intermediate_wait"]
        assert not any(r.record.kind == "child_terminal" for r in rows)
        parent.runtime.wake = lambda _sid: None
        parent.runtime.child_tasks(driver.store, "parent").message(child["session_id"], "answer", "choice")
        assert Runner.resume(parent.session_id, parent.run_id).final_output == "parent done"
        state = driver.store.read_state(child["session_id"])[0]
        assert list(state.turns) == [child["turn_id"]] and contract["same_turn_reply"]
    finally:
        driver.close()


@pytest.mark.parametrize("failed", [False, True])
def test_retained_child_identity_error_and_unicode_status_match_contract(manager_driver, tmp_path, failed):
    contract = _fixture()["manager_outcome"]
    preview = contract["unicode_preview"]["text"] * contract["unicode_preview"]["repeat"]

    def fail(_request):
        raise RuntimeError("child failed")

    _, sid, manager, context = _child(manager_driver, tmp_path, fail if failed else LLMResponse(preview))
    entry = manager.get(sid)
    assert entry is not None and entry.task_id == entry.session_id == sid
    result = status_with_store(context, {"task_ids": [sid]})
    assert result.status_code is ToolResultStatus.SUCCESS
    status = json.loads(result.content)["tasks"][0]
    assert status["task_id"] == status["session_id"] == sid
    if failed:
        assert status["status"] == contract["status_entry"]["status"]
        assert status["error_code"] == contract["status_entry"]["error_code"]
        assert status["error"] == entry.outcome.error == "model_outcome_unknown"
    else:
        assert status["status"] == "completed" and status["final_answer"] == preview


def test_early_errors_mirror_content_into_metadata(tmp_path: Path) -> None:
    registry = build_default_registry()
    create_result = registry.get("create_sub_task").handler(
        _context(tmp_path),
        {"agent_id": "researcher", "task_description": "Research"},
    )
    status_result = registry.get("sub_task_status").handler(
        _context(tmp_path),
        {"task_ids": ["task"]},
    )

    _assert_error_metadata_matches_content(create_result)
    _assert_error_metadata_matches_content(status_result)
    assert _fixture()["early_error_metadata_matches_content"] is True


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        (
            {
                "agent_id": "researcher",
                "task_description": "single",
                "tasks": [{"task_description": "batch"}],
                "exclude_files_pattern": r"(?=secret)",
            },
            "sub_task_payload_conflict",
        ),
        (
            {
                "agent_id": "researcher",
                "tasks": "not an array",
                "exclude_files_pattern": r"(?=secret)",
            },
            "invalid_tool_arguments",
        ),
        (
            {
                "agent_id": "researcher",
                "tasks": [],
                "exclude_files_pattern": r"(?=secret)",
            },
            "invalid_tool_arguments",
        ),
        (
            {
                "agent_id": "researcher",
                "tasks": [42],
                "exclude_files_pattern": r"(?=secret)",
            },
            "invalid_tool_arguments",
        ),
    ],
)
def test_schema_and_payload_mode_validation_precede_exclude_pattern(
    tmp_path: Path,
    arguments: dict[str, Any],
    expected_code: str,
) -> None:
    context = _context(tmp_path)
    context.sub_task_runner = lambda request: SubTaskOutcome(
        task_id="child",
        agent_name=request.agent_name,
        status=AgentStatus.COMPLETED,
    )

    result = build_default_registry().execute(
        ToolCall(id="payload_priority", name="create_sub_task", arguments=arguments),
        context,
    )
    result = require_tool_result(result)

    assert result.error_code == expected_code
    if expected_code == "invalid_tool_arguments":
        _assert_schema_error_metadata(result)
    else:
        _assert_error_metadata_matches_content(result)
    assert _fixture()["validation"]["payload_mode_validation_precedes_exclude_pattern"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"agent_id": ["researcher"], "task_description": "Research"},
        {"agent_id": "researcher", "task_description": {"task": "Research"}},
        {"agent_id": "researcher", "task_description": "Research", "output_requirements": ["json"]},
        {"agent_id": "researcher", "task_description": "Research", "exclude_files_pattern": 42},
        {"agent_id": "researcher", "tasks": [{"task_description": ["Research"]}]},
        {
            "agent_id": "researcher",
            "tasks": [{"task_description": "Research", "output_requirements": {"format": "json"}}],
        },
    ],
)
def test_create_sub_task_rejects_non_string_schema_values(
    tmp_path: Path,
    arguments: dict[str, Any],
) -> None:
    calls = 0

    def run(request: Any) -> SubTaskOutcome:
        nonlocal calls
        calls += 1
        return SubTaskOutcome(
            task_id="child",
            agent_name=request.agent_name,
            status=AgentStatus.COMPLETED,
        )

    context = _context(tmp_path)
    context.sub_task_runner = run

    result = build_default_registry().execute(
        ToolCall(id="invalid_create_arguments", name="create_sub_task", arguments=arguments),
        context,
    )
    result = require_tool_result(result)

    _assert_schema_error_metadata(result)
    assert result.error_code == "invalid_tool_arguments"
    assert calls == 0
    assert _fixture()["validation"]["handler_receives_schema_valid_arguments"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"task_ids": [42]},
        {"task_ids": ["unknown"], "message": {"prompt": "continue"}},
        {"task_ids": ["unknown"], "detail_level": ["snapshot"]},
    ],
)
def test_sub_task_status_rejects_non_string_schema_values(
    tmp_path: Path,
    manager_driver,
    arguments: dict[str, Any],
) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager(manager_driver, tmp_path)

    result = build_default_registry().execute(
        ToolCall(id="invalid_status_arguments", name="sub_task_status", arguments=arguments),
        context,
    )
    result = require_tool_result(result)

    _assert_schema_error_metadata(result)
    assert result.error_code == "invalid_tool_arguments"
    assert _fixture()["validation"]["handler_receives_schema_valid_arguments"] is True


def test_status_envelope_preserves_lineage_and_omits_unknown_activity(manager_driver, tmp_path):
    parent, sid, _, context = _child(manager_driver, tmp_path, LLMResponse("child done"))
    result = status_with_store(context, {"task_ids": [sid], "detail_level": "snapshot"})
    entry = json.loads(result.content)["tasks"][0]
    contract = _fixture()["status_envelope"]
    assert all(field in entry for field in contract["lineage_fields"])
    assert entry["parent_run_id"] == parent.run_id
    assert entry["parent_tool_call_id"] == "delegate"
    assert "recent_activity" not in entry["snapshot"]
    assert contract["recent_activity_when_unavailable"] == "omitted"
