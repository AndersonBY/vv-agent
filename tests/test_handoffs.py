from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any

from support import ModelMapProvider

from vv_agent import Agent, RunConfig, Runner, ToolPolicy, function_tool, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import HandoffCompletedEvent, HandoffStartedEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import ScriptedLLM
from vv_agent.tools import ToolContext
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolResultStatus

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "parity" / "handoff_contract.json"


def _contract() -> dict[str, Any]:
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


def _provider(routes: dict[str, ScriptedLLM]) -> ModelMapProvider:
    return ModelMapProvider(
        routes={name: (llm, _resolved(name)) for name, llm in routes.items()},
        default_model=next(iter(routes)),
    )


def test_handoff_transfers_control_and_finishes_with_target_output(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write the answer.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Transfer writing tasks.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.", metadata={"routing_group": "writing"})],
    )
    model_provider = _provider(
        {
            "writer": ScriptedLLM(steps=[LLMResponse(content="written by target")]),
            "triage": ScriptedLLM(
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
        }
    )

    result = Runner.run_sync(triage, "please write", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "written by target"
    assert result.agent_name == "writer"
    assert set(model_provider.resolved_models) == {"triage", "writer"}


def test_handoff_run_emits_lifecycle_events(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write the answer.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Transfer writing tasks.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.", metadata={"routing_group": "writing"})],
    )

    model_provider = _provider(
        {
            "writer": ScriptedLLM(steps=[LLMResponse(content="written by target")]),
            "triage": ScriptedLLM(
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
        }
    )

    result = Runner.run_sync(triage, "please write", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    started = [event for event in result.events if isinstance(event, HandoffStartedEvent)]
    completed = [event for event in result.events if isinstance(event, HandoffCompletedEvent)]

    assert result.status == AgentStatus.COMPLETED
    assert len(started) == 1
    assert len(completed) == 1
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

    driver = result._session_driver
    child_id = next(sid for sid in driver.store.list_sessions() if sid != result.raw_result.session_id)
    child_rows = driver.store.read_state(child_id)[1]
    child_terminal = next(r for r in child_rows if r.record.kind == "turn_ended")
    assert child_terminal.record.payload["status"] == "completed"
    assert completed[0].child_run_id == child_terminal.record.turn_id
    assert result.events.index(started[0]) < result.events.index(completed[0])
    assert result.events[-1].type == _contract()["lifecycle_order"][-1]


def test_handoff_tool_schema_and_invalid_input_match_shared_contract(tmp_path: Path) -> None:
    from vv_agent.session.surfaces import SessionDriver

    writer = Agent("writer", "Write.", model="writer")
    triage = Agent("triage", "Route.", model="triage", handoffs=[handoff(agent=writer, description="Use for writing.")])
    provider = _provider(
        {
            "triage": ScriptedLLM(
                [LLMResponse("", [ToolCall("invalid", "transfer_to_writer", {"input": "   "})]), LLMResponse("done")]
            ),
            "writer": ScriptedLLM([]),
        }
    )
    driver = SessionDriver()
    try:
        rt = driver.runtime(triage, RunConfig(workspace=tmp_path, model_provider=provider))
        schema = rt.registry.get_schema("transfer_to_writer")
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": False,
        }
        driver.create("invalid-handoff", str(tmp_path))
        result = driver.start("invalid-handoff", triage, RunConfig(workspace=tmp_path, model_provider=provider), "go").result()
        rejected = result.raw_result.cycles[0].tool_results[0]
        assert rejected.status_code == ToolResultStatus.ERROR
        assert rejected.error_code == "invalid_handoff_arguments"
        assert driver.store.list_sessions() == ("invalid-handoff",)
    finally:
        driver.close()


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

    model_provider = _provider(
        {
            "writer": ScriptedLLM([]),
            "triage": ScriptedLLM(
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
        }
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
    assert completed.child_run_id != result.run_id
    assert completed.run_id == result.run_id


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

    model_provider = _provider(
        {
            "first": ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-first",
                                name="transfer_to_middle",
                                arguments={"input": "to middle"},
                            )
                        ],
                    )
                ]
            ),
            "middle": ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-middle",
                                name="transfer_to_final",
                                arguments={"input": "to final"},
                            )
                        ],
                    )
                ]
            ),
        }
    )

    result = Runner.run_sync(
        first, "start", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider, max_handoffs=1)
    )
    assert result.status == AgentStatus.FAILED
    assert result.raw_result.error_code == "maximum_handoffs_exceeded"


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

    model_provider = _provider(
        {
            "writer": ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="read", name="read_handoff_state", arguments={})],
                    )
                ]
            ),
            "triage": ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="set", name="set_handoff_state", arguments={}),
                            ToolCall(
                                id="handoff",
                                name="transfer_to_writer",
                                arguments={"input": "read the state"},
                            ),
                        ],
                    )
                ]
            ),
        }
    )

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
    target_entered = Event()
    writer = Agent(name="writer", instructions="Write.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer)],
    )

    def target_step(_request: Any) -> LLMResponse:
        target_entered.set()
        target_release.wait(timeout=2)
        return LLMResponse(content="done")

    model_provider = _provider(
        {
            "writer": ScriptedLLM(steps=[target_step]),
            "triage": ScriptedLLM(
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
        }
    )

    handle = Runner.start(
        triage,
        "start",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )
    assert target_entered.wait(timeout=2)
    assert handle.cancel("stop target") is True
    target_release.set()
    result = handle.result(timeout=2)
    for child in handle.kernel.handles:
        child.join(2)
    assert result.status == AgentStatus.FAILED
    assert result.completion_reason is not None and result.completion_reason.value == "cancelled"
    assert any(
        handle.kernel.store.read_state(sid)[0].active_turn_id is None
        for sid in handle.kernel.store.list_sessions()
        if sid != handle.session_id
    )


def test_approved_handoff_resume_switches_to_target_agent(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Route.",
        model="triage",
        handoffs=[handoff(agent=writer)],
    )

    model_provider = _provider(
        {
            "writer": ScriptedLLM(steps=[LLMResponse(content="written")]),
            "triage": ScriptedLLM(
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
        }
    )

    runner = Runner.configured(
        RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(approval="always"),
        )
    )
    interrupted = runner.run_sync(triage, "start")
    assert interrupted.status == AgentStatus.WAIT_USER
    sid, tid = interrupted.raw_result.session_id, interrupted.raw_result.turn_id
    assert sid is not None and tid is not None
    owner = interrupted._session_driver
    owner.approve(sid, owner.handles[-1].runtime, interrupted.metadata["session_waits"][0]["request_id"], "approve", "approval")
    resumed = runner.resume(sid, tid)

    assert resumed.agent_name == "writer"
    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "written"
    completed = next(event for event in resumed.events if isinstance(event, HandoffCompletedEvent))
    assert completed.child_run_id != resumed.run_id
    assert completed.run_id == resumed.run_id
