from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent import (
    AfterCycleDecision,
    AfterCycleSnapshot,
    Agent,
    ModelSettings,
    RunConfig,
    ToolMetadata,
    ToolPolicy,
    ToolSideEffect,
)
from vv_agent.checkpoint import CheckpointConfig, CheckpointError, ToolIdempotency
from vv_agent.config import ResolvedModelConfig
from vv_agent.runtime.checkpoint_codec_v2 import (
    _strict_json_loads,
    run_definition_comparison_copy,
)
from vv_agent.runtime.run_definition import build_run_definition
from vv_agent.runtime.state import InMemoryStateStore
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, ToolExecutionResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "run_definition_v1.json"
LEGACY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "run_definition_legacy_v1.json"
TOOL_METADATA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "tool_metadata_v1.json"


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
    fixture = _strict_json_loads(FIXTURE_PATH.read_text(encoding="utf-8"))
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


def test_run_definition_freezes_typed_tool_metadata_and_policy_denials() -> None:
    metadata_contract = _strict_json_loads(TOOL_METADATA_FIXTURE_PATH.read_text(encoding="utf-8"))
    checkpoint_contract = metadata_contract["checkpoint_v2"]
    agent, config, resolved, task = _minimal_inputs()
    registry = ToolRegistry()

    def inspect(_context: object, _arguments: dict[str, object]) -> ToolExecutionResult:
        return ToolExecutionResult(tool_call_id="", content="ok")

    registry.register_tool(
        "inspect",
        inspect,
        "Inspect one source.",
        idempotency=ToolIdempotency.SUPPORTED,
        tool_metadata=ToolMetadata(
            side_effect=ToolSideEffect.READ,
            idempotency=ToolIdempotency.SUPPORTED,
            capability_tags=[" source.inspect ", "source.inspect"],
            cost_dimensions=["workspace.bytes_read"],
        ),
    )
    registry.register_tool(
        "legacy_inspect",
        inspect,
        "Inspect without a typed declaration.",
        idempotency=ToolIdempotency.SUPPORTED,
    )
    task.extra_tool_names = ["inspect", "legacy_inspect"]
    config.tool_policy = ToolPolicy(
        denied_side_effects=["execute"],
        denied_capability_tags=["filesystem.delete"],
        deny_terminal_tools=True,
        denied_cost_dimensions=["gpu.second"],
    )

    definition, _digest = build_run_definition(
        agent=agent,
        root_input="Summarize the status.",
        run_config=config,
        resolved=resolved,
        model_settings=ModelSettings(),
        task=task,
        registry=registry,
        initial_messages=[],
    )

    metadata_field = checkpoint_contract["run_definition_tool_field"]
    tool = next(item for item in definition["tools"] if item["schema"]["function"]["name"] == "inspect")
    legacy_tool = next(
        item for item in definition["tools"] if item["schema"]["function"]["name"] == "legacy_inspect"
    )
    assert tool["idempotency"] == "supported"
    assert tool[metadata_field] == {
        "side_effect": "read",
        "idempotency": "supported",
        "terminal": False,
        "capability_tags": ["source.inspect"],
        "cost_dimensions": ["workspace.bytes_read"],
    }
    assert legacy_tool[metadata_field] == checkpoint_contract["missing_value"]
    assert checkpoint_contract["policy_fields_are_frozen"] is True
    assert checkpoint_contract["generic_metadata_is_not_promoted"] is True
    assert checkpoint_contract["resume_must_match_original_declaration"] is True
    assert definition["tool_policy"] == {
        "allowed_tools": None,
        "disallowed_tools": [],
        "approval": "default",
        "predicate_ref": None,
        "approval_timeout_seconds": None,
        "denied_side_effects": ["execute"],
        "denied_capability_tags": ["filesystem.delete"],
        "deny_terminal_tools": True,
        "denied_cost_dimensions": ["gpu.second"],
    }


def test_legacy_run_definition_defaults_only_the_comparison_copy() -> None:
    current_fixture = _strict_json_loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    legacy_fixture = _strict_json_loads(LEGACY_FIXTURE_PATH.read_text(encoding="utf-8"))
    current_cases = {case["name"]: case for case in current_fixture["golden_cases"]}
    current_names = {
        "minimal": "minimal",
        "full_unicode_float_and_capabilities": "full_legacy_shape_with_additive_defaults",
    }

    for legacy_case in legacy_fixture["cases"]:
        stored_definition = legacy_case["definition"]
        original = json.loads(json.dumps(stored_definition))
        original_digest = legacy_case["sha256"]
        current = current_cases[current_names[legacy_case["name"]]]["definition"]

        assert run_definition_comparison_copy(stored_definition) == current
        assert stored_definition == original
        assert legacy_case["sha256"] == original_digest


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


def test_after_cycle_hooks_are_pinned_as_behavior_capability_refs() -> None:
    class ObserverHook:
        def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
            del snapshot
            return AfterCycleDecision.continue_run()

    agent, config, resolved, task = _minimal_inputs()
    config.after_cycle_hooks = [ObserverHook()]

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
    assert "after_cycle_hook:0" in str(error.value)

    assert config.checkpoint_config is not None
    config.checkpoint_config.capability_refs["after_cycle_hook:0"] = {
        "id": "lifecycle.observer",
        "version": "1",
    }
    definition, _digest = build_run_definition(
        agent=agent,
        root_input="Summarize the status.",
        run_config=config,
        resolved=resolved,
        model_settings=ModelSettings(),
        task=task,
        registry=ToolRegistry(),
        initial_messages=[],
    )

    assert definition["capability_refs"]["after_cycle_hook:0"] == {
        "id": "lifecycle.observer",
        "version": "1",
    }
