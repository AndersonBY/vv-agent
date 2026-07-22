from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import pytest
from support import FixedModelProvider

from vv_agent import Agent, AgentStatus, RunConfig, Runner, ToolPolicy, function_tool
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import ApprovalRequestedEvent, ApprovalResolvedEvent, RunEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools import ToolContext
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult
from vv_agent.workspace import LocalWorkspaceBackend

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "approval_tool_policy.json"
_CONTRACT: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_ExecutionPath = Literal["runner", "orchestrator"]


class _DecisionProvider:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    def should_request(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        assert request is self.requests[-1]
        return self.decision


class _FailingDecisionProvider:
    def __init__(self, message: str) -> None:
        self.message = message
        self.request_id = ""

    def should_request(self, request: ApprovalRequest) -> bool:
        self.request_id = request.request_id
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError(self.message)


class _RecordingBroker(ApprovalBroker):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_request: ApprovalRequest | None = None

    def register(self, request: ApprovalRequest) -> None:
        super().register(request)
        self.recorded_request = request


def _resolved_model() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def _dangerous_tool(*, needs_approval: bool, hidden: bool, calls: list[dict[str, str]]) -> FunctionTool:
    def dangerous(path: str) -> str:
        calls.append({"path": path})
        return "ran"

    return function_tool(
        dangerous,
        name="dangerous",
        needs_approval=needs_approval,
        exposure=ToolExposure.HIDDEN if hidden else ToolExposure.DIRECT,
    )


def _runner_tool_result(
    tmp_path: Path,
    *,
    tool: FunctionTool,
    tool_policy: ToolPolicy | None = None,
    approval_provider: _DecisionProvider | None = None,
    approval_broker: ApprovalBroker | None = None,
) -> tuple[ToolExecutionResult, list[RunEvent]]:
    call = _CONTRACT["tool_call"]
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="calling",
                tool_calls=[ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"]))],
            )
        ]
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use the requested tool.", model="m", tools=[tool]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=FixedModelProvider(llm, _resolved_model()),
            max_cycles=1,
            tool_policy=tool_policy,
            approval_provider=approval_provider,
            approval_broker=approval_broker,
        ),
    )
    return result.raw_result.cycles[0].tool_results[0], result.events


def _orchestrator_tool_result(
    tmp_path: Path,
    *,
    tool: FunctionTool,
    planned_tools: list[str],
    runtime_metadata: dict[str, Any],
) -> tuple[ToolExecutionResult, list[RunEvent]]:
    call = _CONTRACT["tool_call"]
    events: list[RunEvent] = []
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        ctx=ExecutionContext(
            event_handler=events.append,
            metadata={
                "_vv_agent_run_id": "run-contract",
                "_vv_agent_trace_id": "trace-contract",
                "_vv_agent_agent_name": "assistant",
                **runtime_metadata,
            },
        ),
    )
    result = ToolOrchestrator.from_tools([tool.to_executor()]).run_one(
        ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"])),
        context=context,
        allowed_tool_names=planned_tools,
    )
    return result, events


def _assert_result_shape(result: ToolExecutionResult, shape: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(result.content)
    assert result.status_code.value == shape["status_code"]
    assert result.directive.value == shape["directive"]
    assert set(payload) == set(shape["content_keys"])
    assert set(result.metadata) == set(shape["metadata_keys"])
    assert result.metadata["mode"] == shape["mode"]
    return payload


def test_approval_tool_policy_fixture_is_canonical() -> None:
    assert _CONTRACT["contract"] == "approval_tool_policy"
    assert _CONTRACT["policy"]["precedence"] == [
        "allowed_tools",
        "disallowed_tools",
        "can_use_tool",
        "planned_name",
    ]


@pytest.mark.parametrize("execution_path", ["runner", "orchestrator"])
@pytest.mark.parametrize("decision_case", _CONTRACT["approval"]["decisions"], ids=lambda case: case["action"])
def test_approval_decision_contract(
    tmp_path: Path,
    execution_path: _ExecutionPath,
    decision_case: dict[str, Any],
) -> None:
    calls: list[dict[str, str]] = []
    tool = _dangerous_tool(needs_approval=True, hidden=False, calls=calls)
    decision = ApprovalDecision(
        action=decision_case["action"],
        reason=decision_case["reason"],
        metadata=dict(decision_case["metadata"]),
    )
    provider = _DecisionProvider(decision)
    broker = _RecordingBroker()

    if execution_path == "runner":
        result, events = _runner_tool_result(
            tmp_path,
            tool=tool,
            approval_provider=provider,
            approval_broker=broker,
        )
    else:
        result, events = _orchestrator_tool_result(
            tmp_path,
            tool=tool,
            planned_tools=[tool.name],
            runtime_metadata={
                "_vv_agent_approval_provider": provider,
                "_vv_agent_approval_broker": broker,
            },
        )

    requested_events = [event for event in events if isinstance(event, ApprovalRequestedEvent)]
    resolved_events = [event for event in events if isinstance(event, ApprovalResolvedEvent)]
    assert len(provider.requests) == 1
    assert broker.recorded_request is not None
    assert len(requested_events) == 1
    assert len(resolved_events) == 1

    request_id = provider.requests[0].request_id
    same_request_ids = {
        request_id,
        broker.recorded_request.request_id,
        requested_events[0].request_id,
        resolved_events[0].request_id,
        result.metadata["request_id"],
    }
    assert len(same_request_ids) == 1
    assert re.fullmatch(_CONTRACT["request_id"]["regex"], request_id)
    assert requested_events[0].message == _CONTRACT["approval"]["required_message"]
    assert set(requested_events[0].metadata) == set(_CONTRACT["approval"]["requested_event_metadata_keys"])
    assert requested_events[0].metadata == {
        "arguments": _CONTRACT["tool_call"]["arguments"],
        "tool_name": tool.name,
    }
    assert resolved_events[0].action == decision_case["action"]
    assert set(resolved_events[0].metadata) == set(_CONTRACT["approval"]["resolved_event_metadata_keys"])
    assert resolved_events[0].metadata == {
        "reason": decision_case["reason"],
        "decision_metadata": decision_case["metadata"],
    }

    shape = _CONTRACT["approval"]["result_shape"]
    payload = _assert_result_shape(result, shape)
    assert result.error_code == decision_case["error_code"]
    assert payload == {
        "ok": False,
        "error": decision_case["message"],
        "error_code": decision_case["error_code"],
        "tool_name": tool.name,
    }
    assert result.metadata == {
        "mode": shape["mode"],
        "request_id": request_id,
        "tool_name": tool.name,
        "arguments": _CONTRACT["tool_call"]["arguments"],
        "action": decision_case["action"],
        "message": decision_case["message"],
    }
    assert calls == []


def test_approval_provider_failure_contract(tmp_path: Path) -> None:
    failure = _CONTRACT["approval"]["provider_failure"]
    calls: list[dict[str, str]] = []
    tool = _dangerous_tool(needs_approval=True, hidden=False, calls=calls)
    provider = _FailingDecisionProvider(failure["message"])
    broker = ApprovalBroker()
    call = _CONTRACT["tool_call"]
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="calling",
                tool_calls=[ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"]))],
            )
        ]
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use the requested tool.", model="m", tools=[tool]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=FixedModelProvider(llm, _resolved_model()),
            max_cycles=1,
            approval_provider=provider,
            approval_broker=broker,
        ),
    )

    assert result.status == AgentStatus(failure["status"])
    assert result.raw_result.error == failure["message"]
    assert bool(calls) is failure["tool_executes"]
    assert provider.request_id
    assert (broker.pending_request(provider.request_id) is not None) is failure["broker_retains_request"]
    assert [
        event.type for event in result.events if event.type in {"approval_requested", "approval_resolved", "run_failed"}
    ] == failure["events"]


@pytest.mark.parametrize("execution_path", ["runner", "orchestrator"])
@pytest.mark.parametrize("policy_case", _CONTRACT["policy"]["cases"], ids=lambda case: case["id"])
def test_tool_policy_precedence_contract(
    tmp_path: Path,
    execution_path: _ExecutionPath,
    policy_case: dict[str, Any],
) -> None:
    calls: list[dict[str, str]] = []
    predicate_calls: list[tuple[str, dict[str, Any]]] = []
    tool = _dangerous_tool(needs_approval=False, hidden=True, calls=calls)

    def can_use_tool(tool_name: str, arguments: dict[str, Any]) -> bool:
        predicate_calls.append((tool_name, dict(arguments)))
        return bool(policy_case["can_use_tool"])

    if execution_path == "runner":
        result, _ = _runner_tool_result(
            tmp_path,
            tool=tool,
            tool_policy=ToolPolicy(
                allowed_tools=list(policy_case["allowed_tools"]),
                disallowed_tools=list(policy_case["disallowed_tools"]),
                can_use_tool=can_use_tool,
            ),
        )
    else:
        result, _ = _orchestrator_tool_result(
            tmp_path,
            tool=tool,
            planned_tools=list(policy_case["planned_tools"]),
            runtime_metadata={
                "_vv_agent_allowed_tools": list(policy_case["allowed_tools"]),
                "_vv_agent_disallowed_tools": list(policy_case["disallowed_tools"]),
                "_vv_agent_tool_policy_can_use_tool": can_use_tool,
            },
        )

    policy = _CONTRACT["policy"]
    shape = policy["result_shape"]
    payload = _assert_result_shape(result, shape)
    assert result.error_code == policy["error_code"]
    assert payload == {
        "ok": False,
        "error": policy["message"],
        "error_code": policy["error_code"],
        "tool_name": tool.name,
    }
    assert result.metadata == {
        "mode": shape["mode"],
        "policy_source": policy_case["policy_source"],
        "tool_name": tool.name,
        "arguments": _CONTRACT["tool_call"]["arguments"],
        "message": policy["message"],
    }
    expected_predicate_calls = 0 if policy_case["policy_source"] in {"allowed_tools", "disallowed_tools"} else 1
    assert len(predicate_calls) == expected_predicate_calls
    assert calls == []
