from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent import Agent, ModelSettings, RunConfig
from vv_agent.checkpoint import CheckpointConfig, CheckpointError
from vv_agent.config import ResolvedModelConfig
from vv_agent.runtime.run_definition import build_run_definition
from vv_agent.runtime.state import InMemoryStateStore
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "run_definition_v1.json"


def _minimal_inputs() -> tuple[Agent, RunConfig, ResolvedModelConfig, AgentTask]:
    agent = Agent(
        name="checkpoint-agent",
        instructions="You are a careful assistant.",
        model="test-model",
    )
    config = RunConfig(
        model="test-model",
        model_settings=ModelSettings(),
        max_cycles=10,
        max_handoffs=10,
        no_tool_policy="continue",
        timeout_seconds=90,
        checkpoint_config=CheckpointConfig(store=InMemoryStateStore()),
    )
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[],
    )
    task = AgentTask(
        task_id="checkpoint-agent_fixture",
        model="test-model",
        system_prompt="You are a careful assistant.",
        user_prompt="Summarize the status.",
        max_cycles=10,
    )
    return agent, config, resolved, task


def test_minimal_run_definition_matches_canonical_golden_vector() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    golden = next(case for case in fixture["golden_cases"] if case["name"] == "minimal")
    agent, config, resolved, task = _minimal_inputs()

    definition, digest = build_run_definition(
        agent=agent,
        root_input="Summarize the status.",
        run_config=config,
        resolved=resolved,
        model_settings=ModelSettings(),
        task=task,
        registry=ToolRegistry(),
        initial_messages=[],
    )

    assert definition == golden["definition"]
    assert digest == golden["sha256"]


def test_behavior_callbacks_require_explicit_stable_refs() -> None:
    agent, config, resolved, task = _minimal_inputs()
    config.metadata["tenant_mode"] = "strict"

    with pytest.raises(CheckpointError) as error:
        build_run_definition(
            agent=agent,
            root_input="Summarize the status.",
            run_config=config,
            resolved=resolved,
            model_settings=ModelSettings(),
            task=task,
            registry=ToolRegistry(),
            initial_messages=[],
        )

    assert error.value.code == "checkpoint_definition_unstable"
