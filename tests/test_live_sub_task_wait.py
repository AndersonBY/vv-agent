from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vv_agent.config import build_vv_llm_from_local_settings
from vv_agent.model import VvLlmModelProvider
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, SubAgentConfig

pytestmark = pytest.mark.live


def test_live_agent_waits_for_background_sub_task_completion(tmp_path: Path) -> None:
    if os.getenv("VV_AGENT_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set VV_AGENT_RUN_LIVE_TESTS=1 to run live integration tests")

    settings_file = Path(
        os.getenv(
            "VV_AGENT_LOCAL_SETTINGS",
            Path(__file__).resolve().parents[1] / "local_settings.py",
        )
    )
    if not settings_file.exists():
        pytest.skip(f"Live settings file not found: {settings_file}")

    backend = os.getenv("VV_AGENT_LIVE_BACKEND", "moonshot")
    model = os.getenv("VV_AGENT_LIVE_MODEL", "kimi-k3")
    llm, resolved = build_vv_llm_from_local_settings(settings_file, backend=backend, model=model)
    provider = VvLlmModelProvider(
        settings_file=settings_file,
        default_backend=backend,
        timeout_seconds=90.0,
    )
    runtime = AgentRuntime(
        llm_client=llm,
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
    )

    parent_prompt = build_system_prompt(
        (
            "You are testing sub-agent orchestration. Follow these exact steps and do not answer directly. "
            "Step 1: call create_sub_task with agent_id='slow-researcher', task_description asking the child "
            "to sleep briefly then return the token WAIT-SUBTASK-OK, and wait_for_completion=false. "
            "Step 2: after you receive the task_id, call sub_task_status with that task_id, wait_for_completion=true, "
            "check_interval_seconds=300, max_wait_seconds=120, and detail_level='snapshot'. "
            "Step 3: after the status result is completed, call task_finish with a message containing WAIT-SUBTASK-OK."
        ),
        language="en-US",
        available_sub_agents={"slow-researcher": "Sleeps briefly, then returns the requested token."},
        workspace=tmp_path,
    )
    child_prompt = (
        "You are the slow-researcher sub-agent. First call bash with command 'sleep 2' and timeout 10. "
        "After bash succeeds, call task_finish with exactly: WAIT-SUBTASK-OK."
    )
    task = AgentTask(
        task_id="live_sub_task_wait",
        model=resolved.model_id,
        system_prompt=parent_prompt,
        user_prompt="Run the exact live sub-task wait verification now.",
        max_cycles=8,
        sub_agents={
            "slow-researcher": SubAgentConfig(
                model=model,
                backend=backend,
                description="Sleeps briefly and returns WAIT-SUBTASK-OK.",
                system_prompt=child_prompt,
                max_cycles=4,
            )
        },
        metadata={"language": "en-US"},
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED, result.error
    tool_calls = [call for cycle in result.cycles for call in cycle.tool_calls]
    create_calls = [call for call in tool_calls if call.name == "create_sub_task"]
    status_calls = [call for call in tool_calls if call.name == "sub_task_status"]
    assert create_calls, "live parent did not call create_sub_task"
    assert status_calls, "live parent did not call sub_task_status"
    assert create_calls[0].arguments.get("wait_for_completion") is False
    assert any(call.arguments.get("wait_for_completion") is True for call in status_calls)

    status_payloads = [
        json.loads(result.content)
        for cycle in result.cycles
        for result in cycle.tool_results
        if result.tool_call_id in {call.id for call in status_calls}
    ]
    assert status_payloads, "live run did not return sub_task_status payload"
    completed_status = next(
        (
            payload
            for payload in status_payloads
            if payload.get("wait_for_completion") is True
            and payload.get("wait_exceeded") is False
            and payload.get("tasks", [{}])[0].get("status") == "completed"
        ),
        None,
    )
    assert completed_status is not None, status_payloads
    assert completed_status["running_task_ids"] == []
    assert completed_status["suggested_next_check_after_seconds"] == 300
    assert "WAIT-SUBTASK-OK" in json.dumps(completed_status, ensure_ascii=False)
    assert "WAIT-SUBTASK-OK" in (result.final_answer or "")
