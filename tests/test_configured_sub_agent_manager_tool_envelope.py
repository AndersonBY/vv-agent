from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import SubRunCompletedEvent
from vv_agent.runtime import SubTaskManager
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.types import AgentStatus, CompletionReason, SubTaskOutcome, ToolResultStatus
from vv_agent.workspace import MemoryWorkspaceBackend

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "manager_tool_envelope_v1.json"
FIXTURE_SHA256 = "2f1dfc343b9c1800b95de8b21e3afa9cdfab7514071c221b6465188441221f02"


def _fixture() -> dict[str, Any]:
    raw = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def _manager() -> SubTaskManager:
    return SubTaskManager(
        register_session=lambda _session_id, _session: None,
        unregister_session=lambda _session_id, _session=None: None,
    )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=MemoryWorkspaceBackend(),
    )


def _assert_error_metadata_matches_content(result: Any) -> None:
    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.ERROR
    assert result.metadata == payload


@pytest.mark.parametrize("case", _fixture()["create_error_cases"], ids=lambda case: case["name"])
def test_create_sub_task_error_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any]) -> None:
    context = _context(tmp_path)
    context.sub_task_runner = lambda request: SubTaskOutcome(
        task_id="child",
        agent_name=request.agent_name,
        status=AgentStatus.COMPLETED,
    )

    result = build_default_registry().get("create_sub_task").handler(context, case["arguments"])

    payload = json.loads(result.content)
    assert payload == case["expected"]
    assert result.error_code == case["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


@pytest.mark.parametrize("case", _fixture()["status_error_cases"], ids=lambda case: case["name"])
def test_sub_task_status_error_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any]) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager()

    result = build_default_registry().get("sub_task_status").handler(context, case["arguments"])

    payload = json.loads(result.content)
    assert payload == case["expected"]
    assert result.error_code == case["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


@pytest.mark.parametrize("case", _fixture()["status_success_cases"], ids=lambda case: case["name"])
def test_sub_task_status_success_corpus_matches_full_envelope(tmp_path: Path, case: dict[str, Any]) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager()

    result = build_default_registry().get("sub_task_status").handler(context, case["arguments"])

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

    payload = json.loads(result.content)
    assert payload == contract["expected"]
    assert result.error_code == contract["expected"]["error_code"]
    _assert_error_metadata_matches_content(result)


def test_sync_wait_outcome_preserves_completion_observation(tmp_path: Path) -> None:
    contract = _fixture()["sync_wait_outcome"]
    context = _context(tmp_path)
    outcome = SubTaskOutcome(
        task_id="wait-child",
        agent_name="researcher",
        status=AgentStatus.WAIT_USER,
        session_id="wait-session",
        wait_reason="Approve dangerous.",
        completion_reason=CompletionReason.WAIT_USER,
        completion_tool_name="dangerous",
        partial_output="proposed change",
        cycles=1,
    )
    assert outcome.error_code == contract["internal_error_code"]
    assert contract["manager_status_error_code_field"] == "omitted"
    assert "error_code" not in outcome.to_dict()
    context.sub_task_runner = lambda _request: outcome

    result = (
        build_default_registry()
        .get("create_sub_task")
        .handler(
            context,
            {"agent_id": "researcher", "task_description": "wait"},
        )
    )

    payload = json.loads(result.content)
    assert payload == contract["expected"]
    assert result.error_code == contract["sync_single_tool_envelope_error_code"]
    _assert_error_metadata_matches_content(result)

    manager = _manager()
    manager.record_outcome("wait-child", outcome)
    status_context = _context(tmp_path)
    status_context.sub_task_manager = manager
    status_result = (
        build_default_registry()
        .get("sub_task_status")
        .handler(
            status_context,
            {"task_ids": ["wait-child"]},
        )
    )
    status_entry = json.loads(status_result.content)["tasks"][0]
    assert "error_code" not in status_entry

    sub_run_event = SubRunCompletedEvent(
        run_id="child-run",
        trace_id="trace",
        parent_tool_call_id="parent-tool",
        status=AgentStatus.WAIT_USER.value,
        wait_reason=outcome.wait_reason,
        completion_reason=outcome.completion_reason,
        metadata={"cycles": outcome.cycles},
    ).to_dict()
    assert contract["sub_run_event_error_code_field"] == "omitted"
    assert "error_code" not in sub_run_event["metadata"]


def test_manager_outcome_identity_blank_code_and_unicode_preview_match_contract() -> None:
    contract = _fixture()["manager_outcome"]
    assert _fixture()["listener_identity"] == {
        "stale_session_events_ignored": True,
        "subscribe_failure_retry": True,
    }
    assert _fixture()["worker_visibility"] == {
        "running_status_authoritative": True,
        "terminal_fields_hidden_until_worker_exit": True,
    }
    manager = _manager()
    manager.record_outcome(
        contract["lookup_task_id"],
        SubTaskOutcome(
            task_id=contract["outcome_task_id"],
            session_id="wire-session",
            agent_name="researcher",
            status=AgentStatus.FAILED,
            error="child failed",
            error_code=" ",
        ),
    )

    entry = manager.get(contract["lookup_task_id"])
    assert entry is not None
    assert entry.task_id == contract["lookup_task_id"]
    result = (
        build_default_registry()
        .get("sub_task_status")
        .handler(
            ToolContext(
                workspace=Path.cwd(),
                shared_state={},
                cycle_index=1,
                workspace_backend=MemoryWorkspaceBackend(),
                sub_task_manager=manager,
            ),
            {"task_ids": [contract["lookup_task_id"]]},
        )
    )
    assert json.loads(result.content)["tasks"][0] == contract["status_entry"]

    preview = contract["unicode_preview"]["text"] * contract["unicode_preview"]["repeat"]
    manager.record_outcome(
        "unicode-preview",
        SubTaskOutcome(
            task_id="unicode-preview-wire",
            session_id="unicode-preview-session",
            agent_name="researcher",
            status=AgentStatus.COMPLETED,
            final_answer=preview,
        ),
    )
    preview_entry = manager.get("unicode-preview")
    assert preview_entry is not None
    assert preview_entry.recent_activity == preview


class _PendingRecord:
    task_id = "pending-task"
    session_id = "pending-session"
    agent_name = "researcher"
    task_title = "pending"
    parent_run_id = None
    parent_tool_call_id = None
    outcome = None
    session = object()

    @staticmethod
    def is_running() -> bool:
        return False


class _PendingManager:
    def __init__(self) -> None:
        self.record = _PendingRecord()

    def get(self, _task_id: str) -> _PendingRecord:
        return self.record

    def _continue_task_with_context(self, *, task_id: str, prompt: str, context: Any) -> None:
        del task_id, prompt, context


def test_pending_interaction_previous_status_matches_contract(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = cast(Any, _PendingManager())

    result = (
        build_default_registry()
        .get("sub_task_status")
        .handler(
            context,
            {"task_ids": ["pending-task"], "message": "continue"},
        )
    )

    payload = json.loads(result.content)
    assert payload["interaction"]["previous_status"] == _fixture()["pending_interaction_previous_status"]


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
            "invalid_tasks_payload",
        ),
        (
            {
                "agent_id": "researcher",
                "tasks": [],
                "exclude_files_pattern": r"(?=secret)",
            },
            "invalid_tasks_payload",
        ),
        (
            {
                "agent_id": "researcher",
                "tasks": [42],
                "exclude_files_pattern": r"(?=secret)",
            },
            "invalid_tasks_payload",
        ),
    ],
)
def test_payload_validation_precedes_exclude_pattern(
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

    result = build_default_registry().get("create_sub_task").handler(context, arguments)

    assert result.error_code == expected_code
    _assert_error_metadata_matches_content(result)
    assert _fixture()["validation"]["payload_validation_precedes_exclude_pattern"] is True


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

    result = build_default_registry().get("create_sub_task").handler(context, arguments)

    _assert_error_metadata_matches_content(result)
    assert calls == 0
    assert _fixture()["validation"]["non_string_schema_values"] == "reject"


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
    arguments: dict[str, Any],
) -> None:
    context = _context(tmp_path)
    context.sub_task_manager = _manager()

    result = build_default_registry().get("sub_task_status").handler(context, arguments)

    _assert_error_metadata_matches_content(result)
    assert _fixture()["validation"]["non_string_schema_values"] == "reject"


def test_status_envelope_preserves_lineage_and_omits_unknown_activity(tmp_path: Path) -> None:
    manager = _manager()
    manager.submit(
        task_id="status-task",
        session_id="status-session",
        agent_name="researcher",
        task_title="Inspect status",
        workspace_backend=MemoryWorkspaceBackend(),
        parent_run_id="parent-run",
        parent_tool_call_id="delegate",
        runner=lambda: SubTaskOutcome(
            task_id="status-task",
            session_id="status-session",
            agent_name="researcher",
            status=AgentStatus.COMPLETED,
        ),
    )
    manager.wait("status-task", timeout=2)
    record = manager.get("status-task")
    assert record is not None
    record.recent_activity = None
    context = _context(tmp_path)
    context.sub_task_manager = manager

    result = (
        build_default_registry()
        .get("sub_task_status")
        .handler(
            context,
            {"task_ids": ["status-task"], "detail_level": "snapshot"},
        )
    )
    payload = json.loads(result.content)
    entry = payload["tasks"][0]
    fixture = _fixture()["status_envelope"]

    assert all(field in entry for field in fixture["lineage_fields"])
    assert entry["parent_run_id"] == "parent-run"
    assert entry["parent_tool_call_id"] == "delegate"
    assert "recent_activity" not in entry["snapshot"]
    assert fixture["recent_activity_when_unavailable"] == "omitted"
