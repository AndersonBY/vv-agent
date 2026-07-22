from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

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
from vv_agent.checkpoint import (
    CheckpointConfig,
    CheckpointError,
    ToolIdempotency,
    validate_run_definition,
)
from vv_agent.config import ResolvedModelConfig
from vv_agent.runtime.checkpoint_codec import _strict_json_loads
from vv_agent.runtime.run_definition import build_run_definition
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, ToolExecutionResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "run_definition.json"
TOOL_METADATA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "tool_metadata.json"


def _run_definition_golden(name: str = "full_unicode_float_and_capabilities") -> dict[str, Any]:
    fixture = _strict_json_loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return deepcopy(next(case["definition"] for case in fixture["golden_cases"] if case["name"] == name))


def _json_pointer_target(document: Any, pointer: str) -> Any:
    current = document
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _model_settings() -> ModelSettings:
    return ModelSettings(timeout_seconds=90.0)


def _minimal_inputs() -> tuple[Agent, RunConfig, ResolvedModelConfig, AgentTask]:
    agent = Agent(
        name="checkpoint-agent",
        instructions="You are a careful assistant.",
        model="test-model",
    )
    config = RunConfig(
        model="test-model",
        model_settings=_model_settings(),
        max_cycles=10,
        max_handoffs=10,
        no_tool_policy="continue",
        checkpoint_config=CheckpointConfig(store=InMemoryCheckpointStore()),
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
    task.memory_compact_threshold = golden["definition"]["runtime_controls"]["memory_compact_threshold"]

    definition, digest = build_run_definition(
        agent=agent,
        root_input="Summarize the status.",
        run_config=config,
        resolved=resolved,
        model_settings=_model_settings(),
        task=task,
        registry=ToolRegistry(),
        initial_messages=[],
    )

    assert definition == golden["definition"]
    assert digest == golden["sha256"]


@pytest.mark.parametrize(
    "object_pointer",
    [
        "/agent",
        "/initial_messages/0",
        "/model",
        "/model/settings",
        "/model/settings/retry",
        "/runtime_controls",
        "/tools/0",
        "/tools/0/schema",
        "/tools/0/schema/function",
        "/tools/0/tool_metadata",
        "/tools/0/approval",
        "/tool_policy",
        "/checkpoint_policy",
        "/budget_limits",
        "/budget_limits/max_host_cost",
        "/context_ref",
        "/workspace_ref",
        "/session_ref",
        "/extensions/0",
        "/capability_refs/context_provider.primary",
    ],
)
def test_run_definition_rejects_unknown_fields_in_every_closed_object(
    object_pointer: str,
) -> None:
    definition = _run_definition_golden()
    target = _json_pointer_target(definition, object_pointer)
    assert isinstance(target, dict)
    target["future_behavior"] = True

    with pytest.raises(CheckpointError) as error:
        validate_run_definition(definition)

    assert error.value.code == "checkpoint_definition_invalid"


@pytest.mark.parametrize(
    ("object_pointer", "required_field"),
    [
        ("/agent", "name"),
        ("/initial_messages/0", "content"),
        ("/model", "backend"),
        ("/runtime_controls", "max_cycles"),
        ("/tools/0", "approval"),
        ("/tools/0/schema", "function"),
        ("/tools/0/schema/function", "description"),
        ("/tools/0/tool_metadata", "terminal"),
        ("/tools/0/approval", "required"),
        ("/tool_policy", "allowed_tools"),
        ("/checkpoint_policy", "ambiguous_model_policy"),
        ("/budget_limits", "max_total_tokens"),
        ("/budget_limits/max_host_cost", "currency"),
        ("/context_ref", "version"),
        ("/workspace_ref", "version"),
        ("/session_ref", "version"),
        ("/extensions/0", "version"),
        ("/capability_refs/context_provider.primary", "version"),
    ],
)
def test_run_definition_rejects_missing_fields_in_every_closed_object(
    object_pointer: str,
    required_field: str,
) -> None:
    definition = _run_definition_golden()
    target = _json_pointer_target(definition, object_pointer)
    assert isinstance(target, dict)
    del target[required_field]

    with pytest.raises(CheckpointError) as error:
        validate_run_definition(definition)

    assert error.value.code == "checkpoint_definition_invalid"


def test_run_definition_rejects_unknown_fields_in_conditional_closed_objects() -> None:
    mutations = (
        ("tool_choice", {"type": "function", "function": {"name": "write_record"}, "future_behavior": True}),
        (
            "tool_choice",
            {"type": "function", "function": {"name": "write_record", "future_behavior": True}},
        ),
        ("response_format", {"type": "text", "future_behavior": True}),
    )
    for field_name, value in mutations:
        definition = _run_definition_golden()
        definition["model"]["settings"][field_name] = value
        with pytest.raises(CheckpointError) as error:
            validate_run_definition(definition)
        assert error.value.code == "checkpoint_definition_invalid"


def test_run_definition_rejects_unknown_fields_in_message_tool_call_wire() -> None:
    definition = _run_definition_golden()
    definition["initial_messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_record",
                        "arguments": '{"record_id":"42","value":"approved"}',
                    },
                }
            ],
        }
    ]
    for object_pointer in (
        "/initial_messages/0/tool_calls/0",
        "/initial_messages/0/tool_calls/0/function",
    ):
        mutated = deepcopy(definition)
        target = _json_pointer_target(mutated, object_pointer)
        target["future_behavior"] = True
        with pytest.raises(CheckpointError) as error:
            validate_run_definition(mutated)
        assert error.value.code == "checkpoint_definition_invalid"


def test_run_definition_keeps_only_declared_extension_maps_open() -> None:
    definition = _run_definition_golden()
    definition["initial_shared_state"]["application_state"] = {"future": True}
    definition["run_metadata"]["application_metadata"] = {"future": True}
    definition["initial_messages"][0]["metadata"] = {"application": {"future": True}}
    definition["model"]["settings"]["reasoning"] = {"provider_mode": "future"}
    definition["model"]["settings"]["extra_headers"]["x-provider-future"] = "enabled"
    definition["model"]["settings"]["extra_body"]["provider_future"] = {"enabled": True}
    definition["model"]["settings"]["extra_args"] = {"provider_future": {"enabled": True}}
    definition["tools"][0]["schema"]["function"]["parameters"]["x-json-schema-future"] = True
    definition["output_schema"] = {"type": "object", "x-json-schema-future": True}
    definition["capability_refs"]["provider.future"] = {"id": "provider.future", "version": "1"}

    assert validate_run_definition(definition) == definition


def test_run_definition_freezes_typed_tool_metadata_and_policy_denials() -> None:
    metadata_contract = _strict_json_loads(TOOL_METADATA_FIXTURE_PATH.read_text(encoding="utf-8"))
    checkpoint_contract = metadata_contract["checkpoint"]
    agent, config, resolved, task = _minimal_inputs()
    registry = ToolRegistry()

    def inspect(_context: object, _arguments: dict[str, object]) -> ToolExecutionResult:
        return ToolExecutionResult(tool_call_id="", content="ok")

    registry.register_tool(
        "inspect",
        inspect,
        "Inspect one source.",
        tool_metadata=ToolMetadata(
            side_effect=ToolSideEffect.READ,
            idempotency=ToolIdempotency.SUPPORTED,
            capability_tags=[" source.inspect ", "source.inspect"],
            cost_dimensions=["workspace.bytes_read"],
        ),
    )
    task.extra_tool_names = ["inspect"]
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
        model_settings=_model_settings(),
        task=task,
        registry=registry,
        initial_messages=[],
    )

    metadata_field = checkpoint_contract["run_definition_tool_field"]
    tool = next(item for item in definition["tools"] if item["schema"]["function"]["name"] == "inspect")
    assert tool[metadata_field] == {
        "side_effect": "read",
        "idempotency": "supported",
        "terminal": False,
        "capability_tags": ["source.inspect"],
        "cost_dimensions": ["workspace.bytes_read"],
    }
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


def test_behavior_callbacks_require_explicit_stable_refs() -> None:
    agent, config, resolved, task = _minimal_inputs()
    config.metadata["tenant_mode"] = "strict"

    with pytest.raises(CheckpointError) as error:
        build_run_definition(
            agent=agent,
            root_input="Summarize the status.",
            run_config=config,
            resolved=resolved,
            model_settings=_model_settings(),
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
            model_settings=_model_settings(),
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
        model_settings=_model_settings(),
        task=task,
        registry=ToolRegistry(),
        initial_messages=[],
    )

    assert definition["capability_refs"]["after_cycle_hook:0"] == {
        "id": "lifecycle.observer",
        "version": "1",
    }
