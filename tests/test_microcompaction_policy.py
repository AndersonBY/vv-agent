from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vv_agent import (
    Agent,
    MicrocompactionPolicy,
    ModelSettings,
    RunConfig,
    ToolMetadata,
    ToolResultRetention,
)
from vv_agent.checkpoint import (
    RUN_DEFINITION_SCHEMA,
    CheckpointConfig,
    CheckpointError,
    validate_run_definition,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.run_definition import _behavior_metadata, build_run_definition
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools import ToolRegistry
from vv_agent.types import AgentTask, Message


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
    assert (
        _behavior_metadata(
            agent=Agent(name="assistant", instructions="system"),
            run_config=config,
        )
        == {}
    )


def test_frozen_compile_restores_policy_from_runtime_controls_v4() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "run_definition.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    definition = deepcopy(next(case["definition"] for case in fixture["golden_cases"] if case["name"] == "minimal"))
    policy = MicrocompactionPolicy(
        trigger_ratio=0.85,
        target_ratio=0.55,
        keep_recent_cycles=2,
        min_result_chars=900,
    )
    definition["schema_version"] = RUN_DEFINITION_SCHEMA
    definition["root_input"] = "go"
    definition["prompt_bundle"] = build_raw_system_prompt_bundle("system").to_dict()
    definition["runtime_controls"]["microcompaction_policy"] = policy.to_dict()
    definition["model"]["model_id"] = "model-id"
    checkpoint = SimpleNamespace(
        run_definition=definition,
        messages=[Message(role="system", content="system")],
        task_id="frozen-policy",
    )
    endpoint = EndpointConfig(
        endpoint_id="fake",
        api_key="key",
        api_base="https://example.invalid/v1",
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="model-id",
        selected_model="model-id",
        model_id="model-id",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="model-id")],
    )

    task = AgentCompiler().compile_frozen_checkpoint(
        agent=Agent(name="checkpoint-agent", instructions="system", model="model-id"),
        run_config=RunConfig(),
        resolved=resolved,
        checkpoint=checkpoint,
        trace_id="trace",
    )

    assert task.microcompaction_policy == policy


def test_run_definition_v4_rejects_invalid_microcompaction_policy_shapes() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "run_definition.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    definition = deepcopy(fixture["golden_cases"][0]["definition"])
    definition["schema_version"] = RUN_DEFINITION_SCHEMA
    definition["runtime_controls"]["microcompaction_policy"] = MicrocompactionPolicy().to_dict()

    validate_run_definition(definition)

    invalid_cases = []
    missing = deepcopy(definition)
    del missing["runtime_controls"]["microcompaction_policy"]
    invalid_cases.append(missing)
    unknown = deepcopy(definition)
    unknown["runtime_controls"]["microcompaction_policy"]["future_behavior"] = True
    invalid_cases.append(unknown)
    invalid_ratio = deepcopy(definition)
    invalid_ratio["runtime_controls"]["microcompaction_policy"]["target_ratio"] = 0.75
    invalid_cases.append(invalid_ratio)

    for payload in invalid_cases:
        with pytest.raises(CheckpointError) as raised:
            validate_run_definition(payload)
        assert raised.value.code == "checkpoint_definition_invalid"


def test_default_checkpoint_policy_does_not_require_behavior_metadata_ref() -> None:
    agent = Agent(name="assistant", instructions="system", model="model-id")
    config = RunConfig(
        model="model-id",
        model_settings=ModelSettings(),
        max_handoffs=10,
        checkpoint_config=CheckpointConfig(store=InMemoryCheckpointStore()),
    )
    task = AgentTask(
        task_id="default-checkpoint-policy",
        model="model-id",
        prompt_bundle=build_raw_system_prompt_bundle("system"),
        user_prompt="run",
        microcompaction_policy=config.microcompaction_policy,
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="model-id",
        selected_model="model-id",
        model_id="model-id",
        endpoint_options=[],
    )

    definition, _digest = build_run_definition(
        agent=agent,
        root_input="run",
        run_config=config,
        resolved=resolved,
        model_settings=ModelSettings(),
        task=task,
        registry=ToolRegistry(),
        initial_messages=[],
    )

    assert definition["runtime_controls"]["microcompaction_policy"] == MicrocompactionPolicy().to_dict()
    assert definition["run_metadata"] == {}
    assert "behavior_affecting_run_metadata" not in definition["capability_refs"]


def test_tool_result_retention_defaults_to_archive_and_is_strict() -> None:
    assert ToolMetadata().result_retention is ToolResultRetention.ARCHIVE
    assert ToolMetadata(result_retention=ToolResultRetention.PRESERVE).to_dict()["result_retention"] == "preserve"
    with pytest.raises(ValueError, match="Unsupported tool result retention"):
        ToolMetadata.from_dict({"result_retention": "drop"})
