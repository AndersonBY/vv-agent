from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from vv_agent import (
    Agent,
    MicrocompactionPolicy,
    RunConfig,
    ToolMetadata,
    ToolResultRetention,
)
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.types import AgentTask


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_ratio": 0},
        {"target_ratio": 0.75},
        {"target_ratio": 0.8},
        {"trigger_ratio": 1.01},
        {"trigger_ratio": True},
        {"target_ratio": float("nan")},
        {"target_ratio": float("inf")},
        {"keep_recent_cycles": -1},
        {"keep_recent_cycles": True},
        {"keep_recent_cycles": 1.5},
        {"keep_recent_cycles": 1 << 32},
        {"min_result_chars": 0},
        {"min_result_chars": True},
        {"min_result_chars": 1.5},
        {"min_result_chars": 1 << 32},
    ],
)
def test_microcompaction_policy_rejects_invalid_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "trigger_ratio": 0.75,
        "target_ratio": 0.60,
        "keep_recent_cycles": 3,
        "min_result_chars": 500,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        MicrocompactionPolicy(**values)


def test_run_config_and_agent_task_freeze_typed_policy_in_explicit_wire() -> None:
    policy = MicrocompactionPolicy(
        trigger_ratio=0.8,
        target_ratio=0.5,
        keep_recent_cycles=4,
        min_result_chars=700,
    )
    config = RunConfig(microcompaction_policy=policy)
    task = AgentTask(
        task_id="policy",
        model="model",
        prompt_bundle=build_raw_system_prompt_bundle("system"),
        user_prompt="run",
        microcompaction_policy=config.microcompaction_policy,
    )

    payload = task.to_dict()
    restored = AgentTask.from_dict(payload)

    assert payload["microcompaction_policy"] == policy.to_dict()
    assert payload["metadata"] == {}
    assert restored.microcompaction_policy == policy
    assert task.metadata == {}
    assert restored.metadata == {}


def test_session_definition_freezes_policy_and_restores_current_task():
    from vv_agent import ScriptedModelProvider
    from vv_agent.session.surfaces import SessionDriver

    policy = MicrocompactionPolicy(trigger_ratio=0.85, target_ratio=0.55, keep_recent_cycles=2, min_result_chars=900)
    driver = SessionDriver()
    try:
        runtime = driver.runtime(
            Agent("policy", "Work."),
            RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []), microcompaction_policy=policy),
        )
        task = runtime.compile("go", "policy/turn/initial")
        definition = runtime.definition(task)
        assert definition["task"]["microcompaction_policy"] == policy.to_dict()
        assert definition["memory_settings"]["microcompaction_policy"] == policy.to_dict()
        assert "microcompaction_policy" not in definition["task"]["metadata"]
        assert AgentTask.from_dict(definition["task"]).microcompaction_policy == policy
    finally:
        driver.close()


def test_current_task_rejects_invalid_microcompaction_policy_shapes():
    fixture = json.loads((Path(__file__).parent / "fixtures/parity/run_definition.json").read_text())
    source = fixture["golden_cases"][0]["definition"]["task"]
    assert AgentTask.from_dict(source).microcompaction_policy == MicrocompactionPolicy()
    for invalid in ({"future_behavior": True}, {"target_ratio": 0.75}, {"keep_recent_cycles": True}):
        payload = deepcopy(source)
        payload["microcompaction_policy"].update(invalid)
        with pytest.raises((TypeError, ValueError)):
            AgentTask.from_dict(payload)


def test_tool_result_retention_defaults_to_archive_and_is_strict() -> None:
    assert ToolMetadata().result_retention is ToolResultRetention.ARCHIVE
    assert ToolMetadata(result_retention=ToolResultRetention.PRESERVE).to_dict()["result_retention"] == "preserve"
    with pytest.raises(ValueError, match="Unsupported tool result retention"):
        ToolMetadata.from_dict({"result_retention": "drop"})
