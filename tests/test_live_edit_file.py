from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.constants import (
    BASH_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.prompt import build_system_prompt_bundle
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentResult, AgentStatus, AgentTask, ToolCall, ToolExecutionResult

pytestmark = pytest.mark.live


@dataclass(slots=True)
class ToolEvent:
    name: str
    call: ToolCall
    result: ToolExecutionResult


def _live_settings_file() -> Path:
    settings_file = Path(
        os.getenv(
            "V_AGENT_LOCAL_SETTINGS",
            Path(__file__).resolve().parents[1] / "local_settings.py",
        )
    )
    if os.getenv("V_AGENT_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set V_AGENT_RUN_LIVE_TESTS=1 to run live integration tests")
    if not settings_file.exists():
        pytest.skip(f"Live settings file not found: {settings_file}")
    return settings_file


def _build_live_runtime(
    workspace: Path,
    *,
    log_handler: Callable[[str, dict[str, object]], None] | None = None,
) -> tuple[AgentRuntime, str]:
    settings_file = _live_settings_file()
    backend = os.getenv("V_AGENT_LIVE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_LIVE_MODEL", "kimi-k2.6")
    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_handler=log_handler,
        settings_file=settings_file,
        default_backend=backend,
        tool_registry_factory=build_default_registry,
    )
    return runtime, resolved.model_id


def _build_live_task(*, model_id: str, workspace: Path, task_id: str, user_prompt: str) -> AgentTask:
    prompt_bundle = build_system_prompt_bundle(
        (
            "You are Vector Vein agent runtime live validation. Follow the user's exact test sequence, "
            "use workspace tools, and finish briefly only after the requested edit behavior is verified."
        ),
        language="en-US",
        allow_interruption=True,
        use_workspace=True,
        enable_todo_management=False,
        workspace=workspace,
    )
    return AgentTask(
        task_id=task_id,
        model=model_id,
        system_prompt=prompt_bundle.prompt,
        user_prompt=user_prompt,
        max_cycles=10,
        metadata={"language": "en-US"},
    )


def _tool_events(result: AgentResult) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    for cycle in result.cycles:
        results_by_id = {tool_result.tool_call_id: tool_result for tool_result in cycle.tool_results}
        for call in cycle.tool_calls:
            tool_result = results_by_id.get(call.id)
            if tool_result is not None:
                events.append(ToolEvent(name=call.name, call=call, result=tool_result))
    return events


def _event_names(events: list[ToolEvent]) -> list[str]:
    return [event.name for event in events]


def test_live_edit_file_feedback_recovers_when_model_edits_before_read(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nneedle = original no-read\nomega\n", encoding="utf-8")
    runtime, model_id = _build_live_runtime(tmp_path)
    task = _build_live_task(
        model_id=model_id,
        workspace=tmp_path,
        task_id="live_edit_file_requires_read_recovery",
        user_prompt=(
            "This validates edit_file preconditions. First, intentionally call edit_file immediately without "
            "using read_file. Try to replace old_string 'needle = original no-read' with new_string "
            "'needle = recovered after read' in target.txt. If edit_file returns an error saying the full file "
            "must be read before editing, obey that feedback: call read_file for the full target.txt, then call "
            "edit_file again with the same replacement. Do not use bash or write_file for editing. Finish only "
            "after the file is changed."
        ),
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED, result.error
    assert target.read_text(encoding="utf-8") == "alpha\nneedle = recovered after read\nomega\n"
    events = _tool_events(result)
    event_names = _event_names(events)
    assert BASH_TOOL_NAME not in event_names
    assert WRITE_FILE_TOOL_NAME not in event_names
    assert event_names[:3] == [EDIT_FILE_TOOL_NAME, READ_FILE_TOOL_NAME, EDIT_FILE_TOOL_NAME], event_names
    assert events[0].result.error_code == "file_not_read"
    assert events[0].result.status == "error"
    assert events[2].result.status == "success"
    assert event_names[-1] == TASK_FINISH_TOOL_NAME


def test_live_edit_file_feedback_recovers_when_file_changes_after_read(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("alpha\nneedle = stale original\nomega\n", encoding="utf-8")
    state = {"externally_modified": False}

    def log_handler(event: str, payload: dict[str, object]) -> None:
        if event != "tool_result":
            return
        if payload.get("tool_name") != READ_FILE_TOOL_NAME or state["externally_modified"]:
            return
        target.write_text("alpha\nneedle = externally changed\nomega\n", encoding="utf-8")
        state["externally_modified"] = True

    runtime, model_id = _build_live_runtime(tmp_path, log_handler=log_handler)
    task = _build_live_task(
        model_id=model_id,
        workspace=tmp_path,
        task_id="live_edit_file_stale_baseline_recovery",
        user_prompt=(
            "Use workspace tools only. First call read_file for the full target.txt. Then call edit_file to "
            "replace old_string 'needle = stale original' with new_string 'needle = stale recovery works'. If "
            "edit_file returns file_changed_since_read or says the file changed after it was read, obey the tool "
            "feedback: call read_file for the full target.txt again, then call edit_file using the current line "
            "content as old_string and 'needle = stale recovery works' as new_string. Do not use bash or "
            "write_file for editing. Finish only after the edit succeeds."
        ),
    )

    result = runtime.run(task)

    assert state["externally_modified"], "test harness did not modify target.txt after read_file"
    assert result.status == AgentStatus.COMPLETED, result.error
    assert target.read_text(encoding="utf-8") == "alpha\nneedle = stale recovery works\nomega\n"
    events = _tool_events(result)
    event_names = _event_names(events)
    assert BASH_TOOL_NAME not in event_names
    assert WRITE_FILE_TOOL_NAME not in event_names
    assert event_names.count(READ_FILE_TOOL_NAME) >= 2, event_names
    assert any(
        event.name == EDIT_FILE_TOOL_NAME and event.result.error_code == "file_changed_since_read"
        for event in events
    ), event_names
    assert any(
        event.name == EDIT_FILE_TOOL_NAME and event.result.status == "success"
        for event in events
    ), event_names
    assert event_names[-1] == TASK_FINISH_TOOL_NAME
