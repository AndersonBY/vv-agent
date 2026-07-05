from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from vv_agent.config import build_openai_llm_from_local_settings
from vv_agent.constants import FIND_FILES_TOOL_NAME, SEARCH_FILES_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.model_settings import ModelSettings
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


def _build_live_runtime(workspace: Path) -> tuple[AgentRuntime, str]:
    settings_file = _live_settings_file()
    backend = os.getenv("V_AGENT_LIVE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_LIVE_MODEL", "kimi-k2.6")
    llm, resolved = build_openai_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        settings_file=settings_file,
        default_backend=backend,
        tool_registry_factory=build_default_registry,
    )
    return runtime, resolved.model_id


def _build_live_task(*, model_id: str, workspace: Path) -> AgentTask:
    prompt_bundle = build_system_prompt_bundle(
        (
            "You are Vector Vein agent runtime live validation. Follow the user's exact tool order. "
            "Use only the available workspace search tools and finish with task_finish."
        ),
        language="en-US",
        allow_interruption=False,
        use_workspace=True,
        enable_todo_management=False,
        workspace=workspace,
    )
    return AgentTask(
        task_id="live_search_tools_find_then_search",
        model=model_id,
        system_prompt=prompt_bundle.prompt,
        user_prompt=(
            "Validate the renamed search tools against the workspace. "
            "Step 1: call find_files with path '.', glob '**/*.txt', sort 'path_asc', and max_results 10. "
            "Step 2: call search_files with pattern 'CALYPSO_NEEDLE_7421', glob '**/*.txt', "
            "output_mode 'content', and n true. "
            "Step 3: call task_finish with message exactly 'LIVE_SEARCH_TOOLS_OK'. "
            "Do not call read_file, bash, workspace_grep, or list_files."
        ),
        max_cycles=8,
        allow_interruption=False,
        use_workspace=True,
        metadata={
            "language": "en-US",
            "_vv_agent_allowed_tools": [
                FIND_FILES_TOOL_NAME,
                SEARCH_FILES_TOOL_NAME,
                TASK_FINISH_TOOL_NAME,
            ],
            "_vv_agent_model_settings": ModelSettings(
                temperature=0.0,
                max_tokens=2048,
                tool_choice="required",
                parallel_tool_calls=False,
                timeout_seconds=180.0,
            ),
        },
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


def test_live_model_uses_find_files_then_search_files(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "alpha.txt").write_text("alpha only\n", encoding="utf-8")
    (notes / "target.txt").write_text("line one\nCALYPSO_NEEDLE_7421 lives here\nline three\n", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("CALYPSO_NEEDLE_7421 in markdown should not match txt glob\n", encoding="utf-8")

    runtime, model_id = _build_live_runtime(tmp_path)
    task = _build_live_task(model_id=model_id, workspace=tmp_path)

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED, result.error
    events = _tool_events(result)
    event_names = [event.name for event in events]

    assert "workspace_grep" not in event_names
    assert "list_files" not in event_names
    assert event_names[:2] == [FIND_FILES_TOOL_NAME, SEARCH_FILES_TOOL_NAME], event_names
    assert event_names[-1] == TASK_FINISH_TOOL_NAME

    find_event = events[0]
    assert find_event.result.status == "success"
    find_payload = json.loads(find_event.result.content)
    assert "notes/target.txt" in find_payload["files"]

    search_event = events[1]
    assert search_event.result.status == "success"
    assert search_event.call.arguments.get("output_mode") == "content"
    assert search_event.result.metadata["summary"]["total_matches"] == 1
    assert search_event.result.metadata["matches"][0]["path"] == "notes/target.txt"
    assert "CALYPSO_NEEDLE_7421 lives here" in search_event.result.metadata["matches"][0]["text"]
