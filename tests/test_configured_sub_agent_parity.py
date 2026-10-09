from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from support.kernel_runtime import KernelRuntime as AgentRuntime

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.llm import ScriptedLLM
from vv_agent.model_settings import ModelSettings
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    SubAgentConfig,
    SubTaskOutcome,
    ToolCall,
)
from vv_agent.workspace import (
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    MemoryWorkspaceBackend,
    compile_portable_workspace_regex,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent.json"
EVENT_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent_events.jsonl"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _event_contract() -> list[dict[str, Any]]:
    return [json.loads(line) for line in EVENT_CONTRACT_PATH.read_text(encoding="utf-8").splitlines() if line]


def test_configured_sub_agent_shared_fixtures_use_one_current_version() -> None:
    assert _contract()["version"] == "v2"
    # The configured-sub-agent wire has its own v2 schema; emitted RunEvent
    # payloads follow the current shared event v6 discriminator.
    assert all(event["version"] == "v6" for event in _event_contract())


def test_portable_workspace_regex_cases_match_shared_contract() -> None:
    cases = _contract()["workspace_filter"]["portable_cases"]

    for case in cases["accepted"]:
        regex = compile_portable_workspace_regex(case["pattern"])
        assert all(regex.search(path) is not None for path in case["matches"])
        assert all(regex.search(path) is None for path in case["misses"])

    for pattern in cases["rejected"]:
        with pytest.raises(ValueError, match=INVALID_EXCLUDE_FILES_PATTERN_MESSAGE):
            compile_portable_workspace_regex(pattern)

    for positive_pattern, negative_pattern in ((r"^\w$", r"^\W$"), (r"^\s$", r"^\S$")):
        case = next(item for item in cases["accepted"] if item["pattern"] == positive_pattern)
        negative_regex = compile_portable_workspace_regex(negative_pattern)
        assert all(negative_regex.search(value) is None for value in case["matches"])
        assert all(negative_regex.search(value) is not None for value in case["misses"])


class _RawPathWorkspaceBackend(MemoryWorkspaceBackend):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self._raw_paths = list(paths)

    def list_files(self, base: str, glob: str) -> list[str]:
        del base, glob
        return list(self._raw_paths)


def test_workspace_filter_normalizes_for_matching_but_preserves_custom_backend_raw_paths() -> None:
    fixture = _contract()["workspace_filter"]["custom_backend_path_normalization"]
    backend = _RawPathWorkspaceBackend(fixture["raw_paths"])
    filtered = DiscoveryFilteredWorkspaceBackend(backend, fixture["pattern"])

    visible = filtered.list_files(".", "**/*")

    assert visible == fixture["visible_paths"]
    assert (visible[0] == fixture["raw_paths"][1]) is fixture["preserve_non_matching_raw_paths"]


def _parent_task() -> AgentTask:
    return AgentTask(
        task_id="parent-task",
        model="parent-model",
        prompt_bundle=build_raw_system_prompt_bundle("Parent prompt"),
        user_prompt="Parent task",
        max_cycles=6,
        memory_compact_threshold=250_000,
        memory_threshold_percentage=80,
        use_workspace=True,
        agent_type="computer",
        extra_tool_names=["custom_tool"],
        exclude_tools=["parent_excluded"],
        model_settings=ModelSettings(temperature=0.25),
        metadata={
            "language": "en-US",
            "available_skills": [{"name": "review"}],
            "active_skills": ["review"],
            "bash_shell": "bash",
        },
    )


def test_agent_task_wire_round_trips_model_settings_messages_and_state() -> None:
    task = _parent_task()
    task.initial_messages = [Message(role="user", content="persisted")]
    task.initial_shared_state = {"scope": "child"}

    restored = AgentTask.from_dict(task.to_dict())

    assert restored.model_settings == task.model_settings
    assert restored.initial_messages == task.initial_messages
    assert restored.initial_shared_state == task.initial_shared_state


def test_sub_agent_model_normalization_and_outcome_wire_match_contract() -> None:
    validation = _contract()["validation"]
    config = SubAgentConfig(
        model=validation["normalized_model_input"],
        description="Research",
    )
    outcome = SubTaskOutcome(
        task_id="child-task",
        agent_name="researcher",
        status=AgentStatus.FAILED,
        error="failed",
    )

    assert config.model == validation["normalized_model_value"]
    assert "error_code" not in outcome.to_dict()


def test_sub_agent_config_uses_shared_portable_whitespace_contract() -> None:
    portable = _contract()["validation"]["portable_whitespace"]

    config = SubAgentConfig(model=portable["model_input"], description="Research")

    assert config.model == portable["model_value"]
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_model_message"]):
        SubAgentConfig(model=portable["blank_model_input"], description="Research")
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_system_prompt_message"]):
        SubAgentConfig(
            model="child-model",
            description="Research",
            system_prompt=portable["blank_system_prompt_input"],
        )

    mutated = SubAgentConfig(model="child-model", description="Research", system_prompt="Child prompt")
    mutated.system_prompt = portable["blank_system_prompt_input"]
    with pytest.raises(ValueError, match=_contract()["validation"]["empty_system_prompt_message"]):
        AgentRuntime._validate_sub_agent_config(mutated)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "", "description": "Research"}, "sub-agent model cannot be empty"),
        (
            {"model": "shared-model", "description": "Research", "system_prompt": ""},
            "sub-agent system_prompt cannot be empty when provided",
        ),
    ],
)
def test_sub_agent_config_constructor_validation_matches_contract(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as error:
        SubAgentConfig(**kwargs)

    assert str(error.value) == message


def test_sub_agent_config_from_wire_normalizes_model() -> None:
    fixture = _contract()["validation"]

    restored = SubAgentConfig.from_dict(
        {
            "model": fixture["normalized_model_input"],
        }
    )

    assert restored.model == fixture["normalized_model_value"]
    assert {
        "description": restored.description,
        "backend": restored.backend,
        "system_prompt": restored.system_prompt,
        "max_cycles": restored.max_cycles,
        "session_memory_enabled": restored.session_memory_enabled,
        "exclude_tools": restored.exclude_tools,
        "denied_side_effects": restored.denied_side_effects,
        "denied_capability_tags": restored.denied_capability_tags,
        "deny_terminal_tools": restored.deny_terminal_tools,
        "denied_cost_dimensions": restored.denied_cost_dimensions,
        "metadata": restored.metadata,
    } == fixture["wire_defaults"]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "  "},
        {"model": "child-model", "system_prompt": " \n "},
    ],
)
def test_sub_agent_config_from_wire_rejects_invalid_values(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SubAgentConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "fixture_key"),
    [
        ("backend", "backend_non_string"),
        ("max_cycles", "max_cycles_negative"),
        ("denied_side_effects", "denied_side_effects_non_array"),
        ("denied_capability_tags", "denied_capability_tags_non_array"),
        ("deny_terminal_tools", "deny_terminal_tools_non_boolean"),
        ("denied_cost_dimensions", "denied_cost_dimensions_non_array"),
    ],
)
def test_sub_agent_config_from_wire_rejects_shared_type_and_range_corpus(
    field: str,
    fixture_key: str,
) -> None:
    payload = {"model": "child-model", field: _contract()["validation"]["wire_rejections"][fixture_key]}

    with pytest.raises((TypeError, ValueError)):
        SubAgentConfig.from_dict(payload)


def test_configured_sub_agent_wire_rejects_unknown_top_level_fields() -> None:
    assert _contract()["validation"]["unknown_top_level_fields"] == "reject"
    with pytest.raises(ValueError, match="unknown fields: backned"):
        SubAgentConfig.from_dict({"model": "child-model", "backned": "invalid"})
    with pytest.raises(ValueError, match="unknown fields: runtime_metadata"):
        AgentTask.from_dict(
            {
                "task_id": "task",
                "model": "model",
                "system_prompt": "system",
                "user_prompt": "user",
                "runtime_metadata": {"trace_id": "invalid"},
            }
        )


def test_real_sub_run_events_normalize_line_by_line_to_shared_fixture(tmp_path: Path) -> None:
    from datetime import datetime
    from unittest.mock import patch

    from support.kernel_runtime import start_runner

    from vv_agent.tools import ToolRegistry, build_default_registry

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(1000, UTC)

    def registry():
        result = ToolRegistry()
        result.register_executor(build_default_registry().get_executor("create_sub_task"))
        return result

    from vv_agent.events import SubRunCompletedEvent, SubRunStartedEvent
    from vv_agent.session.surfaces import SessionDriver

    driver = SessionDriver()
    try:
        agent = Agent("parent", "Be precise.", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")})
        config = RunConfig(
            workspace="/fixture",
            tool_registry_factory=registry,
            model_provider=ScriptedModelProvider(
                "scripted",
                "m",
                ScriptedLLM(
                    [
                        LLMResponse(
                            "",
                            [
                                ToolCall(
                                    "children",
                                    "create_sub_task",
                                    {
                                        "agent_id": "worker",
                                        "tasks": [{"task_description": "first"}, {"task_description": "second"}],
                                    },
                                )
                            ],
                        ),
                        LLMResponse("first done"),
                        LLMResponse("second done"),
                        LLMResponse("parent done"),
                    ]
                ),
                context_length=None,
                max_output_tokens=None,
            ),
        )
        with patch("vv_agent.prompt.builder.datetime", FixedDatetime):
            result = start_runner(driver, "children", agent, "go", run_config=config).result()
        rows = driver.store.read_state("children")[1]
        case = _contract()["producer_cases"][0]
        parked = next(r.record for r in rows if r.record.record_id == case["record_id"])
        assert parked.payload == case["admission"]
        created_child = driver.store.read_state(_contract()["identity"]["session_id"])[1]
        task = next(r.record.payload["definition"]["task"] for r in created_child if r.record.kind == "turn_started")
        assert task == _contract()["task_projection"], (task["metadata"], _contract()["task_projection"]["metadata"])
        assert task["metadata"] == _contract()["metadata_projection"]
        events = [e.to_dict() for e in result.events if isinstance(e, (SubRunStartedEvent, SubRunCompletedEvent))]

        def stable(event):
            return {k: v for k, v in event.items() if k not in {"created_at", "metadata"}}

        assert [stable(e) for e in events if e["child_session_id"] == _contract()["identity"]["session_id"]] == [
            stable(e) for e in _event_contract()
        ]
        assert [e["type"] for e in events] == _contract()["lifecycle"]["event_sequence"]
        assert result.final_output == "parent done"
        assert {
            r.record.payload["parent_session_id"]
            for sid in driver.store.list_sessions()
            if sid != "children"
            for r in driver.store.read_state(sid)[1][:1]
        } == {"children"}
    finally:
        driver.close()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_configured_sub_agent_uses_atomic_frozen_child_definitions(tmp_path: Path, batch: bool, background: bool) -> None:
    from support.kernel_runtime import start_runner

    from vv_agent.session.children import child_handles
    from vv_agent.session.surfaces import SessionDriver

    args: dict[str, Any] = {"agent_id": "worker", "wait_for_completion": not background}
    args.update(
        {"tasks": [{"task_description": "first"}, {"task_description": "second"}]} if batch else {"task_description": "first"}
    )
    agent = Agent("parent", "Parent.", sub_agents={"worker": SubAgentConfig(model="m", description="Work.", max_cycles=3)})
    steps = [LLMResponse("", [ToolCall("delegate", "create_sub_task", args)]), *[LLMResponse("done") for _ in range(5)]]
    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "configured",
            agent,
            "go",
            run_config=RunConfig(workspace=tmp_path, model_provider=ScriptedModelProvider.new("scripted", "m", steps)),
        )
        result = handle.result()
        for child in driver.handles:
            child.join(2)
        rows = driver.store.read_state("configured")[1]
        parked = next(r for r in rows if r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "child")
        started = next(r for r in rows if r.record.kind == "op_started" and r.record.operation_id == parked.record.operation_id)
        assert started.commit_id == parked.commit_id
        handles = child_handles(parked.record.payload["handle"])
        assert len(handles) == (2 if batch else 1)
        for child in handles:
            created = driver.store.read(child["session_id"], limit=1).records[0].record
            admission = created.payload["attributes"]["child_admission"]
            assert created.payload["parent_session_id"] == "configured"
            assert admission["definition"]["task"]["max_cycles"] == 3
            assert admission["definition"]["task"]["model"] == "m"
            assert child["background"] is background
            assert driver.store.read_state(child["session_id"])[0].active_turn_id is None
        assert result.status is AgentStatus.COMPLETED
    finally:
        driver.close()


@pytest.mark.parametrize("mode", ["sync", "async", "batch"])
def test_parent_cancellation_reaches_configured_children(tmp_path: Path, mode: str) -> None:
    from threading import Event

    from support.kernel_runtime import start_runner

    from vv_agent import CompletionReason
    from vv_agent.session.surfaces import SessionDriver

    entered, release = Event(), Event()

    def child_step(_request):
        entered.set()
        release.wait(2)
        return LLMResponse("done")

    args: dict[str, Any] = {"agent_id": "worker", "wait_for_completion": mode != "async"}
    args.update(
        {"tasks": [{"task_description": "first"}, {"task_description": "second"}]}
        if mode == "batch"
        else {"task_description": "work"}
    )
    provider = ScriptedModelProvider.from_steps(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("delegate", "create_sub_task", args)]), child_step, child_step, LLMResponse("parent")],
    )
    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "cancel-parent",
            Agent("parent", "Delegate.", sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
            "go",
            run_config=RunConfig(workspace=tmp_path, model_provider=provider),
        )
        assert entered.wait(2)
        handle.cancel()
        release.set()
        result = handle.result(2)
        for h in driver.handles:
            h.join(2)
        assert result.completion_reason is CompletionReason.CANCELLED
        assert all(driver.store.read_state(sid)[0].active_turn_id is None for sid in driver.store.list_sessions())
    finally:
        release.set()
        driver.close()
