"""Resume addresses retained session/turn identities."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vv_agent import Agent, ApprovalDecision, RunConfig, Runner, ScriptedModelProvider, function_tool
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall


def test_resume_keeps_original_owner_and_turn(tmp_path: Path):
    provider = ScriptedModelProvider.from_steps(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})]),
            LLMResponse("blue"),
        ],
    )
    origin = Runner.configured(RunConfig(model_provider=provider, workspace=tmp_path))
    handle = origin.start(Agent("assistant", "Ask then answer."), "choose")
    parked = handle.result(2)
    sid, tid = parked.raw_result.session_id, parked.raw_result.turn_id
    assert sid is not None and tid is not None
    wait = parked.metadata["session_waits"][0]
    handle.kernel.push(
        sid,
        InboxItem(
            "reply",
            "user",
            {
                "content": {
                    "interaction_id": wait["interaction_id"],
                    "operation_id": wait["operation_id"],
                    "text": "blue",
                }
            },
            tid,
        ),
    )
    receiving = Runner.configured(RunConfig(model_provider=ScriptedModelProvider.new("other", "other", [])))
    result = receiving.resume(sid, tid)
    assert result.final_output == "blue"
    assert result.raw_result.session_id == sid and result.raw_result.turn_id == tid
    assert len([r for r in handle.kernel.store.read_state(sid)[1] if r.record.kind == "turn_ended"]) == 1
    assert Runner.resume(sid, tid).to_dict() == result.to_dict()
    handle.kernel.close()


def test_handle_approval_and_resume_do_not_replay_tool(tmp_path: Path):
    executions = []

    @function_tool(needs_approval=True)
    def write(value: str) -> str:
        executions.append(value)
        return value

    handle = Runner.start(
        Agent("writer", "Write.", tools=[write], tool_use_behavior="stop_on_first_tool"),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=ScriptedModelProvider.from_steps(
                "test",
                "m",
                [
                    LLMResponse("", [ToolCall("write", "write", {"value": "done"})]),
                ],
            ),
        ),
    )
    parked = handle.result(2)
    assert not executions
    handle.approve(parked.metadata["session_waits"][0]["request_id"], ApprovalDecision.allow())
    result = handle.resume()
    assert result.final_output == "done" and executions == ["done"]
    assert handle.resume().final_output == "done" and executions == ["done"]
    with pytest.raises(ValueError, match="retained session"):
        handle.resume({"version": "v23"})
    handle.kernel.close()


def test_resume_public_signature_has_only_explicit_identities():
    assert list(inspect.signature(Runner.resume).parameters) == ["session_id", "turn_id"]
