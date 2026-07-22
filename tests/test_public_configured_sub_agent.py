from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest

import vv_agent.runtime.compiler as compiler_module
from vv_agent import Agent, RunConfig, Runner, ScriptedModelProvider
from vv_agent.config import ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.context_providers import ContextBundle, ContextFragment, ContextRequest
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.tool_planner import plan_tool_names
from vv_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall

_CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "public_configured_sub_agent.json"


def _contract() -> dict[str, Any]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sub_agents(entries: list[dict[str, Any]]) -> dict[str, SubAgentConfig]:
    return {cast(str, entry["id"]): SubAgentConfig.from_dict(cast(dict[str, Any], entry["config"])) for entry in entries}


def _resolved(model: str) -> ResolvedModelConfig:
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[],
    )


def _agent_from_contract(contract: dict[str, Any]) -> Agent[Any]:
    normalization = cast(dict[str, Any], contract["normalization"])
    projection = cast(dict[str, Any], contract["projection"])
    public_runner = cast(dict[str, Any], contract["public_runner"])
    sections = cast(list[dict[str, Any]], projection["sections"])
    return Agent(
        name="coordinator",
        instructions=cast(str, sections[0]["text"]),
        model=cast(str, public_runner["parent_model"]),
        sub_agents=_sub_agents(cast(list[dict[str, Any]], normalization["raw_entries"])),
    )


def _request_tool_names(request: LlmRequest) -> list[str]:
    names: list[str] = []
    for schema in request.tools:
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        name = cast(dict[str, Any], function).get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def test_agent_normalizes_sub_agent_ids_and_deep_copies_configs() -> None:
    contract = _contract()
    normalization = cast(dict[str, Any], contract["normalization"])
    raw_entries = cast(list[dict[str, Any]], normalization["raw_entries"])
    source = _sub_agents(raw_entries)
    raw_researcher_id = cast(str, raw_entries[0]["id"])
    source_researcher = source[raw_researcher_id]

    agent = Agent(name="coordinator", instructions="Coordinate.", sub_agents=source)

    assert list(agent.sub_agents) == normalization["normalized_ids"]
    assert agent.sub_agents["researcher"] is not source_researcher

    source_researcher.exclude_tools.append("workspace_read_file")
    cast(dict[str, Any], source_researcher.metadata["nested"])["scope"] = "mutated"
    source.clear()

    assert agent.sub_agents["researcher"].to_dict() == normalization["retained_researcher_config"]
    assert normalization["copy_on_agent_construction"] is True


def test_agent_rejects_empty_and_colliding_normalized_sub_agent_ids() -> None:
    contract = _contract()
    normalization = cast(dict[str, Any], contract["normalization"])
    raw_entries = cast(list[dict[str, Any]], normalization["raw_entries"])
    config = SubAgentConfig.from_dict(cast(dict[str, Any], raw_entries[0]["config"]))

    with pytest.raises(ValueError) as empty_error:
        Agent(name="coordinator", instructions="Coordinate.", sub_agents={" \x1c\x1f ": config})
    assert str(empty_error.value) == normalization["empty_id_error"]

    with pytest.raises(ValueError) as collision_error:
        Agent(
            name="coordinator",
            instructions="Coordinate.",
            sub_agents={
                cast(str, raw_entries[0]["id"]): config,
                "researcher": config,
            },
        )
    assert str(collision_error.value) == normalization["collision_error"]


def test_compiler_projects_configured_sub_agents_and_prompt_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    normalization = cast(dict[str, Any], contract["normalization"])
    projection = cast(dict[str, Any], contract["projection"])
    public_runner = cast(dict[str, Any], contract["public_runner"])
    expected_sections = cast(list[dict[str, Any]], projection["sections"])
    captured_fragments: list[ContextFragment] = []
    original_assemble = compiler_module.assemble_context_fragments

    def capture_fragments(
        request: ContextRequest,
        fragments: Iterable[ContextFragment],
    ) -> ContextBundle:
        materialized = list(fragments)
        captured_fragments.extend(materialized)
        return original_assemble(request, materialized)

    monkeypatch.setattr(compiler_module, "assemble_context_fragments", capture_fragments)
    agent = _agent_from_contract(contract)
    task = AgentCompiler().compile(
        agent=agent,
        input="Delegate the research.",
        run_config=RunConfig(),
        resolved=_resolved(cast(str, public_runner["parent_model"])),
        trace_id="trace-public-configured-sub-agent",
    )

    assert task.sub_agents_enabled is projection["sub_agents_enabled"]
    assert task.sub_agents is not agent.sub_agents
    assert {name: config.to_dict() for name, config in task.sub_agents.items()} == dict(
        zip(
            cast(list[str], normalization["normalized_ids"]),
            [entry["config"] for entry in cast(list[dict[str, Any]], normalization["raw_entries"])],
            strict=True,
        )
    )

    configured_names = cast(list[str], projection["configured_tool_names"])
    assert [name for name in plan_tool_names(task) if name in configured_names] == configured_names

    fragment_contract = cast(dict[str, Any], projection["fragment"])
    fragment = next(item for item in captured_fragments if item.id == fragment_contract["id"])
    assert fragment.text == fragment_contract["text"]
    assert fragment.stable is fragment_contract["stable"]
    assert fragment.priority == fragment_contract["priority"]
    assert fragment.source == fragment_contract["source"]

    assert task.system_prompt == projection["prompt"]
    assert task.metadata["system_prompt_sections"] == [
        {key: value for key, value in section.items() if key != "priority"} for section in expected_sections
    ]
    assert task.metadata["system_prompt_sources"] == projection["sources"]
    assert task.metadata["system_prompt_stable_hash"] == projection["stable_hash"]
    assert len(task.system_prompt) == projection["total_chars"]


def test_configured_sub_agent_fragment_participates_in_context_budget() -> None:
    contract = _contract()
    projection = cast(dict[str, Any], contract["projection"])
    public_runner = cast(dict[str, Any], contract["public_runner"])
    instruction = cast(str, cast(list[dict[str, Any]], projection["sections"])[0]["text"])

    task = AgentCompiler().compile(
        agent=_agent_from_contract(contract),
        input="Delegate the research.",
        run_config=RunConfig(max_context_chars=len(instruction)),
        resolved=_resolved(cast(str, public_runner["parent_model"])),
        trace_id="trace-budgeted-public-configured-sub-agent",
    )

    assert task.system_prompt == instruction
    assert task.metadata["system_prompt_omitted_sections"] == ["configured_sub_agents"]
    assert task.metadata["system_prompt_sources"] == {"agent_instructions": "agent.instructions"}
    assert task.metadata["system_prompt_stable_hash"] == hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def test_public_runner_executes_configured_child_without_runtime_task(tmp_path: Path) -> None:
    contract = _contract()
    projection = cast(dict[str, Any], contract["projection"])
    public_runner = cast(dict[str, Any], contract["public_runner"])
    configured_names = cast(list[str], projection["configured_tool_names"])
    observed_models: list[str] = []

    def delegate(request: LlmRequest) -> LLMResponse:
        observed_models.append(request.model)
        assert [name for name in _request_tool_names(request) if name in configured_names] == configured_names
        return LLMResponse(
            content="delegate",
            tool_calls=[
                ToolCall(
                    id="delegate-research",
                    name=CREATE_SUB_TASK_TOOL_NAME,
                    arguments={
                        "agent_id": public_runner["delegated_agent_id"],
                        "task_description": "Research the requested facts.",
                    },
                )
            ],
        )

    def finish_child(request: LlmRequest) -> LLMResponse:
        observed_models.append(request.model)
        assert set(_request_tool_names(request)).isdisjoint(configured_names)
        return LLMResponse(
            content="finish child",
            tool_calls=[
                ToolCall(
                    id="finish-child",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": public_runner["child_final_output"]},
                )
            ],
        )

    def finish_parent(request: LlmRequest) -> LLMResponse:
        observed_models.append(request.model)
        assert [name for name in _request_tool_names(request) if name in configured_names] == configured_names
        return LLMResponse(
            content="finish parent",
            tool_calls=[
                ToolCall(
                    id="finish-parent",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": public_runner["parent_final_output"]},
                )
            ],
        )

    llm = ScriptedLLM(steps=[delegate, finish_child, finish_parent])
    provider = ScriptedModelProvider(
        backend="test",
        default_model=cast(str, public_runner["parent_model"]),
        llm=llm,
        context_length=None,
        max_output_tokens=None,
    )
    run_config = RunConfig(model_provider=provider, workspace=tmp_path)
    assert not hasattr(run_config, "runtime_task")

    result = Runner.run_sync(
        _agent_from_contract(contract),
        "Coordinate the work.",
        run_config=run_config,
    )

    delegation_result = json.loads(result.raw_result.cycles[0].tool_results[0].content)
    assert delegation_result["final_answer"] == public_runner["child_final_output"]
    assert result.final_output == public_runner["parent_final_output"]
    assert result.status == AgentStatus(public_runner["terminal_status"])
    assert observed_models == [
        public_runner["parent_model"],
        public_runner["child_model"],
        public_runner["parent_model"],
    ]
    assert llm.steps == []
    assert public_runner["constructs_agent_task"] is False
