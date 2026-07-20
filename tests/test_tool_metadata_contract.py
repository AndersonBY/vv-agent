from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import (
    Agent,
    FunctionTool,
    RunConfig,
    Runner,
    ToolContext,
    ToolIdempotency,
    ToolMetadata,
    ToolOutputText,
    ToolPolicy,
    function_tool,
)
from vv_agent.llm import ScriptedLLM
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import LLMResponse, ToolCall, ToolDirective, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "tool_metadata_v1.json"
with _FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
    _CONTRACT: dict[str, Any] = json.load(fixture_file)

_TOOL_NAME = "fixture_tool"


def _implementation() -> str:
    return "ok"


def _build_tool(
    *,
    tool_metadata: ToolMetadata | dict[str, Any] | None = None,
    idempotency: ToolIdempotency = ToolIdempotency.UNKNOWN,
) -> FunctionTool:
    return function_tool(
        _implementation,
        name=_TOOL_NAME,
        description="Fixture-backed tool metadata producer.",
        idempotency=idempotency,
        tool_metadata=tool_metadata,
    )


def _tool_metadata_dict(tool: FunctionTool) -> dict[str, Any] | None:
    return tool.tool_metadata.to_dict() if tool.tool_metadata is not None else None


def _generated_value(generator: dict[str, Any]) -> list[str]:
    if "count" in generator:
        return [f"{generator['value_prefix']}{index}" for index in range(int(generator["count"]))]
    return [str(generator["value"]) * int(generator["code_points"])]


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _tool_context(tmp_path: Path, policy: ToolPolicy | None = None) -> ToolContext:
    metadata: dict[str, Any] = {}
    if policy is not None:
        metadata = {
            "_vv_agent_denied_side_effects": [getattr(item, "value", item) for item in policy.denied_side_effects],
            "_vv_agent_denied_capability_tags": list(policy.denied_capability_tags),
            "_vv_agent_deny_terminal_tools": policy.deny_terminal_tools,
            "_vv_agent_denied_cost_dimensions": list(policy.denied_cost_dimensions),
        }
    return ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        metadata=metadata,
    )


def _event_projection(event: Any, fields: list[str]) -> dict[str, Any]:
    payload = event.to_dict()
    projected = {"type": payload["type"]}
    for field_name in fields:
        if field_name in payload:
            projected[field_name] = payload[field_name]
    return projected


@pytest.mark.parametrize(
    "case",
    _CONTRACT["normalization_cases"],
    ids=lambda case: case["name"],
)
def test_public_tool_metadata_normalization_cases(case: dict[str, Any]) -> None:
    metadata = ToolMetadata.from_dict(deepcopy(case["input"]))

    @function_tool(
        name=_TOOL_NAME,
        description="Fixture-backed tool metadata producer.",
        tool_metadata=metadata,
    )
    def produced_tool() -> str:
        return "ok"

    assert produced_tool.tool_metadata is not metadata
    assert _tool_metadata_dict(produced_tool) == case["expected"]


@pytest.mark.parametrize(
    "case",
    _CONTRACT["invalid_cases"],
    ids=lambda case: case["name"],
)
def test_public_function_tool_rejects_invalid_metadata_cases(case: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _build_tool(tool_metadata=deepcopy(case["input"]))


@pytest.mark.parametrize(
    "case",
    _CONTRACT["generated_invalid_cases"],
    ids=lambda case: case["name"],
)
def test_public_producers_reject_generated_invalid_cases(case: dict[str, Any]) -> None:
    generator = case["generator"]
    field_name = str(generator["field"])
    value = _generated_value(generator)

    with pytest.raises((TypeError, ValueError)):
        if field_name == "denied_capability_tags":
            ToolPolicy(denied_capability_tags=value)
        elif field_name == "denied_cost_dimensions":
            ToolPolicy(denied_cost_dimensions=value)
        else:
            _build_tool(tool_metadata={field_name: value})


@pytest.mark.parametrize(
    "case",
    _CONTRACT["legacy_idempotency"]["cases"],
    ids=lambda case: case["name"],
)
def test_function_tool_consumes_legacy_idempotency_cases(case: dict[str, Any]) -> None:
    legacy = ToolIdempotency(case["legacy"])
    typed = deepcopy(case["typed"])

    if not case["valid"]:
        with pytest.raises(ValueError, match=str(case["error_code"])):
            _build_tool(tool_metadata=typed, idempotency=legacy)
        return

    tool = _build_tool(tool_metadata=typed, idempotency=legacy)

    assert tool.idempotency.value == case["expected_effective"]
    assert tool.idempotency.value == case["expected_run_definition_idempotency"]
    assert _tool_metadata_dict(tool) == case["expected_run_definition_tool_metadata"]

    normalized_again = _build_tool(
        tool_metadata=tool.tool_metadata,
        idempotency=tool.idempotency,
    )
    assert normalized_again.idempotency == tool.idempotency
    assert _tool_metadata_dict(normalized_again) == _tool_metadata_dict(tool)


@pytest.mark.parametrize(
    "case",
    _CONTRACT["policy_cases"],
    ids=lambda case: case["name"],
)
def test_runner_consumes_tool_policy_cases(case: dict[str, Any], tmp_path: Path) -> None:
    invocations: list[str] = []

    def implementation() -> str:
        invocations.append(_TOOL_NAME)
        return "ok"

    tool = function_tool(
        implementation,
        name=_TOOL_NAME,
        description="Fixture-backed tool policy producer.",
        tool_metadata=deepcopy(case["metadata"]),
    )
    policy_fields = deepcopy(case["policy"])
    if case.get("existing_name_policy_allows") is False:
        policy_fields["disallowed_tools"] = [_TOOL_NAME]
    policy = ToolPolicy(**policy_fields)
    agent = Agent(
        name="fixture-agent",
        instructions="Call the fixture tool.",
        model=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="fixture-call", name=_TOOL_NAME, arguments={})],
                )
            ]
        ),
        tools=[tool],
        tool_use_behavior="stop_on_first_tool",
    )

    result = Runner.run_sync(
        agent,
        "run the fixture case",
        run_config=RunConfig(
            workspace=tmp_path,
            max_cycles=1,
            tool_policy=policy,
        ),
    )
    tool_result = result.raw_result.cycles[0].tool_results[0]

    if case["allowed"]:
        assert invocations == [_TOOL_NAME]
        assert tool_result.status_code == ToolResultStatus.SUCCESS
        assert tool_result.metadata.get("policy_source") is None
    else:
        assert invocations == []
        assert tool_result.error_code == "tool_not_allowed"
        assert tool_result.metadata["policy_source"] == case["policy_source"]


def test_generic_metadata_is_not_promoted_and_tool_metadata_is_not_model_visible() -> None:
    assert _CONTRACT["metadata_contract"]["generic_metadata_is_not_a_declaration"] is True
    assert _CONTRACT["metadata_contract"]["model_visible"] is False
    assert _CONTRACT["public_construction"]["generic_metadata_remains_separate"] is True

    declaration = deepcopy(_CONTRACT["normalization_cases"][0]["expected"])
    generic_metadata = {**declaration, "tool_metadata": deepcopy(declaration)}

    def invoke(_context: ToolContext | None, _arguments: dict[str, Any]) -> ToolOutputText:
        return ToolOutputText(text="ok")

    generic_tool = FunctionTool(
        name="generic_fixture_tool",
        description="Generic metadata remains separate.",
        params_json_schema={"type": "object", "properties": {}, "required": []},
        on_invoke=invoke,
        metadata=generic_metadata,
    )
    typed_tool = _build_tool(tool_metadata=ToolMetadata.from_dict(declaration))

    assert generic_tool.metadata == generic_metadata
    assert generic_tool.tool_metadata is None
    assert _tool_metadata_dict(typed_tool) == declaration

    forbidden_schema_keys = {
        "tool_metadata",
        *_CONTRACT["metadata_contract"]["closed_fields"],
    }
    assert _nested_keys(generic_tool.to_openai_schema()).isdisjoint(forbidden_schema_keys)
    assert _nested_keys(typed_tool.to_openai_schema()).isdisjoint(forbidden_schema_keys)


def test_telemetry_contract_matches_public_status_and_directive_values() -> None:
    telemetry = _CONTRACT["telemetry_contract"]

    assert telemetry["event_types"] == [
        "tool_call_planned",
        "tool_call_started",
        "tool_call_completed",
    ]
    assert telemetry["tool_status_values"] == [status.value.lower() for status in ToolResultStatus]
    assert set(telemetry["directive_values"]) == {directive.value for directive in ToolDirective}
    assert telemetry["parse_failure_before_planning_has_no_tool_lifecycle"] is True
    assert telemetry["missing_metadata_field"] == "omit"
    assert telemetry["telemetry_changes_runtime_decisions"] is False


@pytest.mark.parametrize(
    "case",
    _CONTRACT["producer_cases"],
    ids=lambda case: case["name"],
)
def test_real_orchestrator_consumes_canonical_producer_cases(
    case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _CONTRACT["telemetry_contract"]
    expected_events = case.get("expected_events")
    first_expected = expected_events[0] if expected_events else None
    tool_name = str(first_expected["tool_name"] if first_expected else _TOOL_NAME)
    tool_call_id = str(first_expected["tool_call_id"] if first_expected else f"call_{case['name']}")
    arguments = deepcopy(first_expected["arguments"] if first_expected else {})
    invocations: list[dict[str, Any]] = []

    def implementation(path: str = "") -> str:
        invocations.append({"path": path} if path else {})
        return "ok"

    tool = function_tool(
        implementation,
        name=tool_name,
        description="Canonical telemetry producer.",
        tool_metadata=deepcopy(case["tool_metadata"]),
    )
    policy = ToolPolicy(**deepcopy(case.get("policy", {})))
    events: list[Any] = []

    if expected_events:
        ticks = iter((1_000_000_000, 1_007_000_000))
        monkeypatch.setattr("vv_agent.tools.orchestrator.time.monotonic_ns", lambda: next(ticks))

    result = ToolOrchestrator.from_tools([tool]).run_one(
        ToolCall(id=tool_call_id, name=tool_name, arguments=arguments),
        context=_tool_context(tmp_path, policy),
        event_sink=events.append,
    )
    event_types = [event.type for event in events]

    if expected_events:
        field_map = {
            "tool_call_planned": telemetry["planned_fields"],
            "tool_call_started": telemetry["started_fields"],
            "tool_call_completed": telemetry["completed_fields"],
        }
        assert [_event_projection(event, field_map[event.type]) for event in events] == expected_events
        assert invocations == [arguments]
        assert result.status_code == ToolResultStatus.SUCCESS
        return

    assert event_types == case["expected_event_types"]
    if case["name"] == "metadata_policy_denial_has_no_execution_start":
        completed = _event_projection(events[-1], telemetry["completed_fields"])
        assert {key: completed[key] for key in case["expected_completed"]} == case["expected_completed"]
        assert invocations == []
        assert result.error_code == "tool_not_allowed"
        return

    assert case["typed_metadata_field_present"] is False
    assert all("tool_metadata" not in event.to_dict() for event in events)
    assert invocations == [arguments]
    assert result.status_code == ToolResultStatus.SUCCESS


def test_parse_failure_boundary_is_driven_by_canonical_telemetry_contract(tmp_path: Path) -> None:
    events: list[Any] = []

    result = ToolOrchestrator.from_tools([]).run_one(
        ToolCall(id="call_invalid", name="missing", arguments=cast(Any, "{")),
        context=_tool_context(tmp_path),
        event_sink=events.append,
    )

    assert _CONTRACT["telemetry_contract"]["parse_failure_before_planning_has_no_tool_lifecycle"] is True
    assert result.error_code == "invalid_arguments_json"
    assert events == []
