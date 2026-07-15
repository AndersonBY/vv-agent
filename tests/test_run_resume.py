from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    ApprovalRequestedEvent,
    GuardrailResult,
    MemorySession,
    RunConfig,
    Runner,
    ToolContext,
    ToolPolicy,
    function_tool,
    output_guardrail,
)
from vv_agent.agent import ToolUseBehavior
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.event_store import JsonlRunEventStore
from vv_agent.llm import ScriptedLLM
from vv_agent.result import RunState
from vv_agent.runtime import CancellationToken, InlineBackend
from vv_agent.types import AgentStatus, CompletionReason, LLMResponse, ToolCall


def _approval_resume_case(name: str) -> dict[str, Any]:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "parity" / "completion_policy_v1.json").read_text(encoding="utf-8")
    )
    return next(case for case in fixture["approval_resume"]["cases"] if case["name"] == name)


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="approval-model",
        selected_model="approval-model",
        model_id="approval-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="approval-model")],
    )


def test_interrupted_result_snapshot_and_runner_resume_execute_approved_call_once(tmp_path: Path) -> None:
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def delete_file(path: str) -> str:
        executions.append(path)
        return f"deleted {path}"

    llm_calls = 0

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        nonlocal llm_calls
        llm_calls += 1
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="delete",
                        tool_calls=[ToolCall(id="delete-call", name="delete_file", arguments={"path": "danger.txt"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Delete after approval.",
            model="approval-model",
            tools=[delete_file],
            tool_use_behavior="stop_on_first_tool",
        ),
        "delete danger.txt",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.WAIT_USER
    assert executions == []
    assert len(result.approvals) == 1
    approval = result.approvals[0]
    assert approval.tool_name == "delete_file"
    assert approval.tool_call_id == "delete-call"
    assert approval.arguments == {"path": "danger.txt"}
    requested = next(event for event in result.events if isinstance(event, ApprovalRequestedEvent))
    assert requested.request_id == approval.interruption_id

    state = result.into_state()
    assert isinstance(state, RunState)
    state.approve(approval.interruption_id)
    assert state.approvals[0].approved is True
    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "deleted danger.txt"
    assert executions == ["danger.txt"]
    assert llm_calls == 1


def test_approved_continue_tool_returns_to_the_model_loop(tmp_path: Path) -> None:
    contract = _approval_resume_case("approved_continue_uses_full_fresh_cycle_budget")
    expected = contract["expected"]
    assert isinstance(expected, dict)
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def lookup(value: str) -> str:
        executions.append(value)
        return f"approved:{value}"

    def finish_after_tool(request):
        assert any(
            message.role == "tool" and message.tool_call_id == "lookup-call" and message.content == "approved:item"
            for message in request.messages
        )
        return LLMResponse(
            content="ready to finish",
            tool_calls=[ToolCall(id="finish-call", name="task_finish", arguments={"message": "finished after approval"})],
        )

    model = ScriptedLLM(
        steps=[
            LLMResponse(
                content="checking",
                tool_calls=[ToolCall(id="lookup-call", name="lookup", arguments={"value": "item"})],
            ),
            finish_after_tool,
        ]
    )
    interrupted = Runner.run_sync(
        Agent(name="approver", instructions="Continue after the approved lookup.", model=model, tools=[lookup]),
        "look up item",
        run_config=RunConfig(workspace=tmp_path, max_cycles=int(contract["configured_max_cycles"])),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "finished after approval"
    assert resumed.completion_reason == CompletionReason(str(expected["completion_reason"]))
    assert resumed.run_id != interrupted.run_id
    assert resumed.trace_id == interrupted.trace_id
    assert len(resumed.raw_result.cycles) == expected["cycles"]
    assert executions == ["item"]
    assert model.steps == []


@pytest.mark.parametrize(
    ("tool_use_behavior", "stop_at_tool_names", "expected_reason"),
    [
        ("stop_on_first_tool", [], CompletionReason.STOP_ON_FIRST_TOOL),
        ("stop_at_tool_names", ["lookup"], CompletionReason.STOP_AT_TOOL_NAME),
    ],
)
def test_approved_continue_tool_honors_tool_stop_policy(
    tmp_path: Path,
    tool_use_behavior: ToolUseBehavior,
    stop_at_tool_names: list[str],
    expected_reason: CompletionReason,
) -> None:
    @function_tool(needs_approval=True)
    def lookup() -> str:
        return "approved result"

    event_store = JsonlRunEventStore(tmp_path / f"{expected_reason.value}.jsonl")
    interrupted = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Stop according to the declared tool policy.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="lookup draft",
                        tool_calls=[ToolCall(id="lookup-call", name="lookup", arguments={})],
                    )
                ]
            ),
            tools=[lookup],
            tool_use_behavior=tool_use_behavior,
            stop_at_tool_names=stop_at_tool_names,
        ),
        "look up",
        run_config=RunConfig(workspace=tmp_path, event_store=event_store),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.run_id != interrupted.run_id
    assert resumed.final_output == "approved result"
    assert resumed.completion_reason == expected_reason
    assert resumed.completion_tool_name == "lookup"
    assert resumed.events[-1].to_dict()["completion_reason"] == expected_reason.value
    result_terminals = [event for event in resumed.events if event.type in {"run_completed", "run_failed", "run_cancelled"}]
    assert {event.run_id for event in result_terminals} == {interrupted.run_id, resumed.run_id}
    assert len([event for event in result_terminals if event.run_id == resumed.run_id]) == 1
    replayed = list(event_store.replay(run_id=resumed.run_id))
    assert len([event for event in replayed if event.type in {"run_completed", "run_failed", "run_cancelled"}]) == 1
    assert replayed[-1].to_dict()["completion_reason"] == expected_reason.value


def test_approved_terminal_tool_applies_output_guardrails_and_updates_terminal_event(tmp_path: Path) -> None:
    @function_tool(needs_approval=True)
    def unsafe_action() -> str:
        return "unsafe tool result"

    @output_guardrail
    def block_after_approval(_context, output):
        if isinstance(output, str) and output.startswith("Approval required"):
            return GuardrailResult.allow()
        return GuardrailResult.block("blocked approved output")

    event_store = JsonlRunEventStore(tmp_path / "approval-guardrail.jsonl")
    interrupted = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run only after approval.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="assistant draft before unsafe tool",
                        tool_calls=[ToolCall(id="unsafe-call", name="unsafe_action", arguments={})],
                    )
                ]
            ),
            tools=[unsafe_action],
            output_guardrails=[block_after_approval],
            tool_use_behavior="stop_on_first_tool",
        ),
        "run unsafe action",
        run_config=RunConfig(workspace=tmp_path, event_store=event_store),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.status == AgentStatus.FAILED
    assert resumed.run_id != interrupted.run_id
    assert resumed.completion_reason == CompletionReason.FAILED
    assert resumed.completion_tool_name is None
    assert resumed.partial_output == "assistant draft before unsafe tool"
    assert resumed.final_output == "blocked approved output"
    assert resumed.events[-1].type == "run_failed"
    assert resumed.events[-1].to_dict()["completion_reason"] == CompletionReason.FAILED.value
    assert (
        len(
            [
                event
                for event in resumed.events
                if event.run_id == resumed.run_id and event.type in {"run_completed", "run_failed", "run_cancelled"}
            ]
        )
        == 1
    )
    replayed = list(event_store.replay(run_id=resumed.run_id))
    assert len([event for event in replayed if event.type in {"run_completed", "run_failed", "run_cancelled"}]) == 1
    assert replayed[-1].type == "run_failed"
    assert replayed[-1].to_dict()["completion_reason"] == CompletionReason.FAILED.value


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_status", "expected_reason", "expected_output"),
    [
        ("ask_user", {"question": "Choose one"}, AgentStatus.WAIT_USER, CompletionReason.WAIT_USER, "Choose one"),
        (
            "task_finish",
            {"message": "approved finish"},
            AgentStatus.COMPLETED,
            CompletionReason.TOOL_FINISH,
            "approved finish",
        ),
    ],
)
def test_approved_explicit_directive_preserves_wait_or_finish_semantics(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, str],
    expected_status: AgentStatus,
    expected_reason: CompletionReason,
    expected_output: str,
) -> None:
    interrupted = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Execute the explicitly controlled tool.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="assistant draft before approval",
                        tool_calls=[ToolCall(id="controlled-call", name=tool_name, arguments=arguments)],
                    )
                ]
            ),
        ),
        "run controlled tool",
        run_config=RunConfig(workspace=tmp_path, tool_policy=ToolPolicy(approval="always")),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.status == expected_status
    assert resumed.run_id != interrupted.run_id
    assert resumed.completion_reason == expected_reason
    assert resumed.completion_tool_name == tool_name
    assert resumed.final_output == expected_output
    assert resumed.partial_output == ("assistant draft before approval" if expected_status == AgentStatus.WAIT_USER else None)
    assert resumed.events[-1].to_dict()["status"] == expected_status.value
    assert resumed.events[-1].to_dict()["completion_reason"] == expected_reason.value


def test_run_handle_resume_uses_interrupted_result_state(tmp_path: Path) -> None:
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def guarded_write(value: str) -> str:
        executions.append(value)
        return value

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="write",
                        tool_calls=[ToolCall(id="write-call", name="guarded_write", arguments={"value": "approved"})],
                    )
                ]
            ),
            _resolved(),
        )

    handle = Runner.start(
        Agent(
            name="writer",
            instructions="Write after approval.",
            model="approval-model",
            tools=[guarded_write],
            tool_use_behavior="stop_on_first_tool",
        ),
        "write",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )
    interrupted = handle.result(timeout=2)
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = handle.resume(state)

    assert resumed is not None
    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "approved"
    assert executions == ["approved"]


def test_manual_approval_resume_rechecks_current_policy_and_preserves_tool_context(tmp_path: Path) -> None:
    executions: list[str] = []
    allowed = True
    manager = object()
    backend = InlineBackend()

    @function_tool(needs_approval=True)
    def guarded_write(context: ToolContext, value: str) -> str:
        executions.append(value)
        assert context.sub_task_manager is manager
        assert context.ctx is not None
        assert context.ctx.metadata["execution_backend"] is backend
        assert context.run_context is not None
        context.shared_state["resumed"] = value
        return value

    def approval_call() -> LLMResponse:
        return LLMResponse(
            content="write",
            tool_calls=[ToolCall(id="guarded-call", name="guarded_write", arguments={"value": "approved"})],
        )

    def finish_after_denial(request) -> LLMResponse:
        assert any(
            message.role == "tool" and message.tool_call_id == "guarded-call" and "not allowed" in message.content.lower()
            for message in request.messages
        )
        return LLMResponse(
            content="handled denied tool",
            tool_calls=[ToolCall(id="denied-finish", name="task_finish", arguments={"message": "denial handled"})],
        )

    active_model = ScriptedLLM(steps=[approval_call(), finish_after_denial])

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return active_model, _resolved()

    result = Runner.run_sync(
        Agent(
            name="writer",
            instructions="Write after approval.",
            model="approval-model",
            tools=[guarded_write],
            tool_use_behavior="stop_on_first_tool",
        ),
        "write",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            sub_task_manager=manager,
            execution_backend=backend,
            shared_state={"original": True},
            tool_policy=ToolPolicy(can_use_tool=lambda name, _arguments: allowed or name != "guarded_write"),
        ),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])
    allowed = False

    denied = Runner.resume(state)

    assert denied.status == AgentStatus.COMPLETED
    assert denied.final_output == "denial handled"
    assert executions == []

    allowed = True
    active_model = ScriptedLLM(steps=[approval_call()])
    result = Runner.run_sync(
        Agent(
            name="writer",
            instructions="Write after approval.",
            model="approval-model",
            tools=[guarded_write],
            tool_use_behavior="stop_on_first_tool",
        ),
        "write",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            sub_task_manager=manager,
            execution_backend=backend,
            shared_state={"original": True},
            tool_policy=ToolPolicy(can_use_tool=lambda name, _arguments: allowed or name != "guarded_write"),
        ),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = Runner.resume(state)

    assert resumed.raw_result.shared_state == {"original": True, "todo_list": [], "resumed": "approved"}
    assert executions == ["approved"]


def test_only_wait_user_results_convert_to_run_state() -> None:
    result = Runner.run_sync(
        Agent(
            name="done",
            instructions="Finish.",
            model=ScriptedLLM(
                steps=[LLMResponse(content="done", tool_calls=[ToolCall(id="finish", name="task_finish", arguments={})])]
            ),
        ),
        "go",
    )

    with pytest.raises(ValueError, match="only interrupted runs"):
        result.into_state()


def test_approval_snapshot_nested_arguments_are_isolated_and_resume_is_once_only(tmp_path: Path) -> None:
    executions: list[dict[str, object]] = []
    session = MemorySession("approval-once")

    @function_tool(needs_approval=True)
    def guarded(payload: dict[str, object]) -> str:
        executions.append(payload)
        return str(payload["nested"])

    arguments = {"payload": {"nested": {"value": "original"}}}
    result = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run once.",
            model=ScriptedLLM(
                steps=[LLMResponse(content="guard", tool_calls=[ToolCall(id="guard", name="guarded", arguments=arguments)])]
            ),
            tools=[guarded],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, session=session),
    )
    state = result.into_state()
    approval = state.approvals[0]
    arguments["payload"]["nested"]["value"] = "mutated"  # type: ignore[index]
    approval.arguments["payload"]["nested"]["value"] = "also mutated"  # type: ignore[index]
    state.approve(approval.interruption_id)

    resumed = Runner.resume(state)

    assert executions == [{"nested": {"value": "original"}}]
    with pytest.raises(RuntimeError, match="approval_already_consumed"):
        Runner.resume(state)
    session_tool_results = [item for item in session.get_items() if item.role == "tool" and item.tool_call_id == "guard"]
    assert [item.content for item in session_tool_results] == [resumed.final_output]
    resumed_cycle = resumed.raw_result.cycles[0]
    assert len([item for item in resumed_cycle.tool_results if item.tool_call_id == "guard"]) == 1


def test_approval_resume_claim_is_shared_across_concurrent_state_uses(tmp_path: Path) -> None:
    executions = 0
    execution_lock = threading.Lock()

    @function_tool(needs_approval=True)
    def guarded(value: str) -> str:
        nonlocal executions
        with execution_lock:
            executions += 1
        return value

    result = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run once.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="guard",
                        tool_calls=[ToolCall(id="guard", name="guarded", arguments={"value": "ok"})],
                    )
                ]
            ),
            tools=[guarded],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path),
    )
    state = result.into_state()
    state.approve(state.pending_approval_ids()[0])
    barrier = threading.Barrier(2)

    def resume() -> str:
        barrier.wait(timeout=2)
        try:
            return Runner.resume(state).final_output or ""
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: resume(), range(2)))

    assert executions == 1
    assert sorted(outcomes) == ["approval_already_consumed", "ok"]


def test_approved_resume_rejects_input_without_consuming_shared_claim(tmp_path: Path) -> None:
    contract = _approval_resume_case("approved_resume_rejects_input_before_claim")
    expected = contract["expected"]
    assert isinstance(expected, dict)
    executions: list[str] = []

    @function_tool(needs_approval=True)
    def guarded(value: str) -> str:
        executions.append(value)
        return value

    interrupted = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run once.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="guard",
                        tool_calls=[ToolCall(id="guard", name="guarded", arguments={"value": "ok"})],
                    )
                ]
            ),
            tools=[guarded],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path),
    )
    state = interrupted.into_state()
    interruption_id = state.pending_approval_ids()[0]
    state.approve(interruption_id)
    cloned_state = RunState(result=state.result)
    cloned_state.approve(interruption_id)

    with pytest.raises(ValueError, match=str(expected["error"])):
        Runner.resume(state, input=str(contract["resume_input"]))

    assert len(executions) == expected["tool_execution_count"]
    resumed = Runner.resume(cloned_state)
    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "ok"
    assert executions == ["ok"]
    with pytest.raises(RuntimeError, match="approval_already_consumed"):
        Runner.resume(state)


def test_pre_cancelled_approved_resume_with_input_rejects_before_cancellation(tmp_path: Path) -> None:
    contract = _approval_resume_case("pre_cancelled_approved_resume_with_input_rejects_before_cancellation")
    expected = contract["expected"]
    assert isinstance(expected, dict)
    executions: list[str] = []
    guardrail_calls = 0
    cancellation = CancellationToken()
    event_store = JsonlRunEventStore(tmp_path / "cancelled-approval-input.jsonl")

    @function_tool(needs_approval=True)
    def guarded() -> str:
        executions.append("executed")
        return "executed"

    @output_guardrail
    def observe_guardrail(_context, _output):
        nonlocal guardrail_calls
        guardrail_calls += 1
        return GuardrailResult.allow()

    interrupted = Runner.run_sync(
        Agent(
            name="approver",
            instructions="Run after approval.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="guard",
                        tool_calls=[ToolCall(id="guard", name="guarded", arguments={})],
                    )
                ]
            ),
            tools=[guarded],
            output_guardrails=[observe_guardrail],
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, cancellation_token=cancellation, event_store=event_store),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])
    guardrail_calls_before_resume = guardrail_calls
    cancellation.cancel(str(contract["cancellation_reason"]))

    with pytest.raises(ValueError, match=str(expected["error"])):
        Runner.resume(state, input=str(contract["resume_input"]))

    assert len(executions) == expected["tool_execution_count"]
    assert guardrail_calls - guardrail_calls_before_resume == expected["output_guardrail_count"]
    stored_payloads = [json.loads(line) for line in event_store.path.read_text(encoding="utf-8").splitlines()]
    fresh_terminals = [
        payload
        for payload in stored_payloads
        if payload["run_id"] != interrupted.run_id and payload["type"] in {"run_completed", "run_failed", "run_cancelled"}
    ]
    assert len(fresh_terminals) == expected["terminal_count"]


def test_pre_cancelled_approved_resume_has_no_side_effect_or_guardrail_and_one_fresh_terminal(
    tmp_path: Path,
) -> None:
    contract = _approval_resume_case("pre_cancelled_approved_resume_has_no_side_effects")
    expected = contract["expected"]
    assert isinstance(expected, dict)
    executions: list[str] = []
    guardrail_calls = 0
    cancellation = CancellationToken()
    event_store = JsonlRunEventStore(tmp_path / "cancelled-approval-resume.jsonl")

    @function_tool(needs_approval=True)
    def guarded(value: str) -> str:
        executions.append(value)
        return value

    @output_guardrail
    def observe_guardrail(_context, _output):
        nonlocal guardrail_calls
        guardrail_calls += 1
        return GuardrailResult.allow()

    runner = Runner.configured(RunConfig(cancellation_token=cancellation))
    interrupted = runner.run_sync(
        Agent(
            name="approver",
            instructions="Run only after approval.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="assistant draft before approval",
                        tool_calls=[ToolCall(id="guard", name="guarded", arguments={"value": "unsafe"})],
                    )
                ]
            ),
            tools=[guarded],
            output_guardrails=[observe_guardrail],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, event_store=event_store),
    )
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])
    guardrail_calls_before_resume = guardrail_calls
    assert guardrail_calls_before_resume == 1
    cancellation.cancel(str(contract["cancellation_reason"]))

    resumed = runner.resume(state)

    assert resumed.status == AgentStatus(str(expected["status"]))
    assert resumed.completion_reason == CompletionReason(str(expected["completion_reason"]))
    assert resumed.completion_tool_name == expected["completion_tool_name"]
    assert resumed.partial_output == expected["partial_output"]
    assert resumed.final_output == expected["final_output"]
    assert resumed.run_id != interrupted.run_id
    assert resumed.trace_id == interrupted.trace_id
    assert len(executions) == expected["tool_execution_count"]
    assert guardrail_calls - guardrail_calls_before_resume == expected["output_guardrail_count"]
    terminals = [
        event
        for event in resumed.events
        if event.run_id == resumed.run_id and event.type in {"run_completed", "run_failed", "run_cancelled"}
    ]
    assert len(terminals) == expected["terminal_count"]
    assert terminals[0].type == expected["terminal_event"]
    assert terminals[0].to_dict()["completion_reason"] == expected["completion_reason"]
    replayed = list(event_store.replay(run_id=resumed.run_id))
    replayed_terminals = [event for event in replayed if event.type in {"run_completed", "run_failed", "run_cancelled"}]
    assert len(replayed_terminals) == expected["terminal_count"]
    assert replayed_terminals[0].type == expected["terminal_event"]


def test_approval_typed_output_error_follows_fresh_terminal(tmp_path: Path) -> None:
    contract = _approval_resume_case("approval_typed_output_error_follows_fresh_terminal")
    expected = contract["expected"]
    assert isinstance(expected, dict)
    event_store = JsonlRunEventStore(tmp_path / "approval-typed-output.jsonl")
    interrupted = Runner.run_sync(
        Agent(
            name="typed-approver",
            instructions="Return typed output.",
            model=ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="typed candidate",
                        tool_calls=[ToolCall(id="finish", name="task_finish", arguments={"message": "not-json"})],
                    )
                ]
            ),
            output_type=dict,
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            event_store=event_store,
            tool_policy=ToolPolicy(approval="always"),
        ),
    )
    state = interrupted.into_state()
    interruption_id = state.pending_approval_ids()[0]
    state.approve(interruption_id)
    retry = RunState(result=state.result)
    retry.approve(interruption_id)

    with pytest.raises(ValueError, match=str(expected["error_contains"])):
        Runner.resume(state)

    stored_payloads = [json.loads(line) for line in event_store.path.read_text(encoding="utf-8").splitlines()]
    fresh_terminals = [
        payload
        for payload in stored_payloads
        if payload["run_id"] != interrupted.run_id and payload["type"] in {"run_completed", "run_failed", "run_cancelled"}
    ]
    assert len(fresh_terminals) == 1
    assert (fresh_terminals[0]["run_id"] != interrupted.run_id) is expected["fresh_run_id"]
    with pytest.raises(RuntimeError, match="approval_already_consumed"):
        Runner.resume(retry)
