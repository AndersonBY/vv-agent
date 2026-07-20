from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from vv_agent import Agent, HandoffEvent, RunConfig, Runner, ToolPolicy, function_tool, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.events import HandoffCompletedEvent, HandoffStartedEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.tools import ToolContext
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "handoff_contract_v1.json"
CONTRACT_SHA256 = "c35c2335bd4a79626afca8459eb2966722f0539e1a0efc8014bd14b132100a74"


def _contract() -> dict[str, Any]:
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _resolved(agent_name: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=agent_name,
        selected_model=agent_name,
        model_id=agent_name,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=agent_name)],
    )


def test_handoff_transfers_control_and_finishes_with_target_output(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write the answer.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Transfer writing tasks.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.", metadata={"routing_group": "writing"})],
    )
    provider_calls: list[str] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        provider_calls.append(agent.name)
        if agent.name == "writer":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="writer done",
                            tool_calls=[
                                ToolCall(
                                    id="writer-finish",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "written by target"},
                                )
                            ],
                        )
                    ]
                ),
                _resolved(agent.name),
            )
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-call",
                                name="transfer_to_writer",
                                arguments={"input": "write this"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    result = Runner.run_sync(triage, "please write", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "written by target"
    assert result.agent_name == "writer"
    assert provider_calls == ["triage", "writer"]
    handoff_events = [event for event in result.events if isinstance(event, HandoffEvent)]
    assert len(handoff_events) == 1
    assert handoff_events[0].source_agent == "triage"
    assert handoff_events[0].target_agent == "writer"


def test_handoff_run_emits_lifecycle_events_and_legacy_event(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write the answer.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Transfer writing tasks.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.", metadata={"routing_group": "writing"})],
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        if agent.name == "writer":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="writer done",
                            tool_calls=[
                                ToolCall(
                                    id="writer-finish",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "written by target"},
                                )
                            ],
                        )
                    ]
                ),
                _resolved(agent.name),
            )
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-call",
                                name="transfer_to_writer",
                                arguments={"input": "write this"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    result = Runner.run_sync(triage, "please write", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    started = [event for event in result.events if isinstance(event, HandoffStartedEvent)]
    completed = [event for event in result.events if isinstance(event, HandoffCompletedEvent)]
    legacy = [event for event in result.events if isinstance(event, HandoffEvent)]

    assert result.status == AgentStatus.COMPLETED
    assert len(started) == 1
    assert len(completed) == 1
    assert len(legacy) == 1
    assert started[0].source_agent == "triage"
    assert started[0].target_agent == "writer"
    assert started[0].tool_call_id == "handoff-call"
    assert started[0].status == "started"
    assert completed[0].source_agent == "triage"
    assert completed[0].target_agent == "writer"
    assert completed[0].tool_call_id == "handoff-call"
    assert completed[0].status == AgentStatus.COMPLETED.value
    assert completed[0].child_run_id
    assert started[0].metadata["routing_group"] == "writing"
    assert completed[0].metadata["routing_group"] == "writing"

    target_started = next(event for event in result.events if event.type == "run_started" and event.agent_name == "writer")
    source_terminal = next(
        event
        for event in result.events
        if event.run_id == started[0].run_id and event.type in {"run_completed", "run_failed", "run_cancelled"}
    )
    target_terminal = next(
        event
        for event in result.events
        if event.run_id == target_started.run_id and event.type in {"run_completed", "run_failed", "run_cancelled"}
    )
    indices = {id(event): index for index, event in enumerate(result.events)}
    assert [
        legacy[0].type,
        "source_run_terminal",
        started[0].type,
        "target_run_started",
        "target_run_terminal",
        completed[0].type,
    ] == _contract()["lifecycle_order"]
    assert indices[id(legacy[0])] < indices[id(source_terminal)]
    assert indices[id(source_terminal)] < indices[id(started[0])]
    assert indices[id(started[0])] < indices[id(target_started)]
    assert indices[id(target_started)] < indices[id(target_terminal)]
    assert indices[id(target_terminal)] < indices[id(completed[0])]
    assert completed[0].child_run_id == target_started.run_id


def test_handoff_tool_schema_and_marker_match_shared_contract(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.", metadata={"routing_group": "writing"})],
    )
    config = Runner._effective_run_config(triage, RunConfig(workspace=tmp_path))
    registry = Runner._build_tool_registry(agent=triage, run_config=config)
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        tool_call_id="handoff-call",
        tool_name="transfer_to_writer",
        arguments={"input": "write this"},
    )

    assert registry.get_schema("transfer_to_writer") == _contract()["tool_schema"]
    result = registry.execute(
        ToolCall(id="handoff-call", name="transfer_to_writer", arguments={"input": "write this"}),
        context,
    )
    assert json.loads(result.content) == _contract()["tool_result"]["content"]
    assert result.metadata == _contract()["tool_result"]["metadata"]

    invalid = registry.execute(
        ToolCall(id="invalid", name="transfer_to_writer", arguments={"input": "   "}),
        context,
    )
    assert invalid.status_code == ToolResultStatus.ERROR
    assert invalid.error_code == "invalid_handoff_arguments"


def test_handoff_target_guardrail_failure_is_the_final_result(tmp_path: Path) -> None:
    writer = Agent(
        name="writer",
        instructions="Write.",
        model="writer",
        input_guardrails=[lambda _context, _input: GuardrailResult.block("writer blocked")],
    )
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer)],
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        assert agent.name == "triage"
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-call",
                                name="transfer_to_writer",
                                arguments={"input": "write this"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    result = Runner.run_sync(
        triage,
        "please write",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.agent_name == "writer"
    assert result.status == AgentStatus.FAILED
    assert result.final_output == "writer blocked"
    completed = next(event for event in result.events if isinstance(event, HandoffCompletedEvent))
    assert completed.status == AgentStatus.FAILED.value
    assert completed.child_run_id == result.run_id


def test_handoff_chain_enforces_independent_max_handoffs(tmp_path: Path) -> None:
    final = Agent(name="final", instructions="Finish.", model="final")
    middle = Agent(
        name="middle",
        instructions="Route again.",
        model="middle",
        handoffs=[handoff(agent=final)],
    )
    first = Agent(
        name="first",
        instructions="Route.",
        model="first",
        handoffs=[handoff(agent=middle)],
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        target = "middle" if agent.name == "first" else "final"
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id=f"handoff-{agent.name}",
                                name=f"transfer_to_{target}",
                                arguments={"input": f"to {target}"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    with pytest.raises(RuntimeError, match="maximum handoff depth exceeded"):
        Runner.run_sync(
            first,
            "start",
            run_config=RunConfig(workspace=tmp_path, model_provider=model_provider, max_handoffs=1),
        )


def test_handoff_preserves_mutated_shared_state_for_target_tools(tmp_path: Path) -> None:
    @function_tool
    def set_handoff_state(context: ToolContext) -> str:
        context.shared_state["handoff_value"] = "preserved"
        return "set"

    @function_tool
    def read_handoff_state(context: ToolContext) -> str:
        return str(context.shared_state.get("handoff_value"))

    writer = Agent(
        name="writer",
        instructions="Read state.",
        model="writer",
        tools=[read_handoff_state],
        tool_use_behavior="stop_on_first_tool",
    )
    triage = Agent(
        name="triage",
        instructions="Set state and route.",
        model="triage",
        tools=[set_handoff_state],
        handoffs=[handoff(agent=writer)],
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        if agent.name == "writer":
            calls = [ToolCall(id="read", name="read_handoff_state", arguments={})]
        else:
            calls = [
                ToolCall(id="set", name="set_handoff_state", arguments={}),
                ToolCall(
                    id="handoff",
                    name="transfer_to_writer",
                    arguments={"input": "read the state"},
                ),
            ]
        return ScriptedLLM(steps=[LLMResponse(content="", tool_calls=calls)]), _resolved(agent.name)

    result = Runner.run_sync(
        triage,
        "start",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            shared_state={"initial": True},
        ),
    )

    assert result.agent_name == "writer"
    assert result.final_output == "preserved"
    assert result.raw_result.shared_state["initial"] is True
    assert result.raw_result.shared_state["handoff_value"] == "preserved"


def test_run_handle_can_cancel_while_handoff_target_is_running(tmp_path: Path) -> None:
    target_release = Event()
    writer = Agent(name="writer", instructions="Write.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer)],
    )

    def target_step(_request: Any) -> LLMResponse:
        target_release.wait(timeout=2)
        return LLMResponse(
            content="done",
            tool_calls=[
                ToolCall(
                    id="finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "done"},
                )
            ],
        )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        if agent.name == "writer":
            return ScriptedLLM(steps=[target_step]), _resolved(agent.name)
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="handoff",
                                name="transfer_to_writer",
                                arguments={"input": "write"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    handle = Runner.start(
        triage,
        "start",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )
    saw_target = False
    for event in handle.events():
        if event.type == "run_started" and event.agent_name == "writer":
            saw_target = True
            assert handle.cancel("stop target") is True
            target_release.set()

    assert saw_target
    result = handle.result(timeout=2)
    assert result.agent_name == "writer"
    assert result.status == AgentStatus.FAILED


def test_approved_handoff_resume_switches_to_target_agent(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer)],
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        if agent.name == "writer":
            calls = [
                ToolCall(
                    id="writer-finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "written"},
                )
            ]
        else:
            calls = [
                ToolCall(
                    id="handoff",
                    name="transfer_to_writer",
                    arguments={"input": "write"},
                )
            ]
        return ScriptedLLM(steps=[LLMResponse(content="", tool_calls=calls)]), _resolved(agent.name)

    runner = Runner.configured(
        RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(approval="always"),
        )
    )
    interrupted = runner.run_sync(triage, "start")
    assert interrupted.status == AgentStatus.WAIT_USER
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = runner.resume(state)

    assert resumed.agent_name == "writer"
    assert resumed.status == AgentStatus.WAIT_USER
    completed = next(event for event in resumed.events if isinstance(event, HandoffCompletedEvent))
    assert completed.child_run_id == resumed.run_id
