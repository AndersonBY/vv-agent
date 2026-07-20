from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from vv_agent import (
    AfterCycleDecision,
    AfterCycleSnapshot,
    NativeCycleOutcomeKind,
    RunBudgetLimits,
)
from vv_agent.constants import ASK_USER_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.lifecycle import AFTER_CYCLE_CONTROL_STATE_KEY
from vv_agent.tools import build_default_registry
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    CompletionReason,
    LLMResponse,
    ToolCall,
    ToolExecutionResult,
)


@dataclass
class RecordingHook:
    decisions: list[AfterCycleDecision]
    snapshots: list[AfterCycleSnapshot] = field(default_factory=list)

    def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
        self.snapshots.append(snapshot)
        return self.decisions.pop(0)


def _tool_names(request: LlmRequest) -> set[str]:
    names: set[str] = set()
    for schema in request.tools:
        function = schema.get("function")
        if isinstance(function, dict):
            name = cast(dict[str, Any], function).get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def test_after_cycle_steer_defers_no_tool_completion(tmp_path: Path) -> None:
    requests: list[LlmRequest] = []

    def checked_answer(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content="checked answer")

    hook = RecordingHook(
        decisions=[
            AfterCycleDecision.steer(["Check the output once more."]),
            AfterCycleDecision.continue_run(),
        ]
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                lambda request: requests.append(request) or LLMResponse(content="first answer"),
                checked_answer,
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[hook],
    )
    result = runtime.run(
        AgentTask(
            task_id="after-cycle-no-tool-steer",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=3,
            no_tool_policy="finish",
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.completion_reason is CompletionReason.NO_TOOL_FINISH
    assert result.final_answer == "checked answer"
    assert [snapshot.native_outcome.kind for snapshot in hook.snapshots] == [
        NativeCycleOutcomeKind.COMPLETED,
        NativeCycleOutcomeKind.COMPLETED,
    ]
    assert hook.snapshots[0].remaining_cycles == 2
    assert requests[1].messages[-1].role == "user"
    assert requests[1].messages[-1].content == "Check the output once more."
    assert all(message.content != "Continue working on the task." for message in requests[1].messages)
    assert AFTER_CYCLE_CONTROL_STATE_KEY not in result.shared_state


def test_after_cycle_steer_defers_tool_finish(tmp_path: Path) -> None:
    hook = RecordingHook(
        decisions=[
            AfterCycleDecision.steer(["Verify before finalizing."]),
            AfterCycleDecision.continue_run(),
        ]
    )
    requests: list[LlmRequest] = []
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="first",
                    tool_calls=[
                        ToolCall(
                            id="finish-1",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": "first answer"},
                        )
                    ],
                ),
                lambda request: requests.append(request)
                or LLMResponse(
                    content="second",
                    tool_calls=[
                        ToolCall(
                            id="finish-2",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": "verified answer"},
                        )
                    ],
                ),
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[hook],
    )
    result = runtime.run(
        AgentTask(
            task_id="after-cycle-tool-steer",
            model="test-model",
            system_prompt="system",
            user_prompt="finish",
            max_cycles=3,
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.completion_reason is CompletionReason.TOOL_FINISH
    assert result.final_answer == "verified answer"
    assert requests[0].messages[-1].content == "Verify before finalizing."
    assert [snapshot.native_outcome.kind for snapshot in hook.snapshots] == [
        NativeCycleOutcomeKind.COMPLETED,
        NativeCycleOutcomeKind.COMPLETED,
    ]


def test_after_cycle_permission_narrowing_hides_schema_and_blocks_dispatch(
    tmp_path: Path,
) -> None:
    executions: list[dict[str, Any]] = []
    requests: list[LlmRequest] = []
    registry = build_default_registry()

    def restricted_tool(_context: Any, arguments: dict[str, Any]) -> ToolExecutionResult:
        executions.append(arguments)
        return ToolExecutionResult(tool_call_id="", content="executed")

    registry.register_tool("restricted_tool", restricted_tool, "A restricted test tool.")

    def forbidden_call(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        assert "restricted_tool" not in _tool_names(request)
        return LLMResponse(
            content="try anyway",
            tool_calls=[
                ToolCall(
                    id="restricted-1",
                    name="restricted_tool",
                    arguments={"value": "forbidden"},
                )
            ],
        )

    hook = RecordingHook(
        decisions=[
            AfterCycleDecision.continue_run(disallow_tools=["restricted_tool"]),
            AfterCycleDecision.continue_run(),
        ]
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                lambda request: requests.append(request) or LLMResponse(content="first cycle"),
                forbidden_call,
            ]
        ),
        tool_registry=registry,
        default_workspace=tmp_path,
        after_cycle_hooks=[hook],
    )
    task = AgentTask(
        task_id="after-cycle-deny",
        model="test-model",
        system_prompt="system",
        user_prompt="run",
        max_cycles=2,
        no_tool_policy="continue",
        extra_tool_names=["restricted_tool"],
    )
    result = runtime.run(task)

    assert result.status is AgentStatus.MAX_CYCLES
    assert "restricted_tool" in _tool_names(requests[0])
    assert executions == []
    assert result.cycles[1].tool_results[0].error_code == "tool_not_allowed"
    assert result.shared_state[AFTER_CYCLE_CONTROL_STATE_KEY] == {
        "schema_version": "vv-agent.after-cycle-control.v1",
        "disallowed_tools": ["restricted_tool"],
    }
    assert "_vv_agent_disallowed_tools" not in task.metadata


def test_after_cycle_stop_is_always_non_success(tmp_path: Path) -> None:
    events: list[str] = []
    hook = RecordingHook(
        decisions=[
            AfterCycleDecision.stop_non_success(
                code="host.policy_stop",
                message="Host policy stopped this run.",
            )
        ]
    )
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="candidate answer")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[hook],
        log_handler=lambda event, _payload: events.append(event),
    )
    result = runtime.run(
        AgentTask(
            task_id="after-cycle-stop",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=3,
            no_tool_policy="finish",
        )
    )

    assert result.status is AgentStatus.FAILED
    assert result.completion_reason is CompletionReason.FAILED
    assert result.final_answer is None
    assert result.error == "host.policy_stop: Host policy stopped this run."
    assert "after_cycle_stopped" in events
    assert "run_completed" not in events


def test_after_cycle_steer_cannot_cross_wait_or_max_cycle(tmp_path: Path) -> None:
    wait_runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="need input",
                    tool_calls=[
                        ToolCall(
                            id="ask-1",
                            name=ASK_USER_TOOL_NAME,
                            arguments={"question": "Confirm?"},
                        )
                    ],
                )
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[
            RecordingHook(decisions=[AfterCycleDecision.steer(["Do not wait."])])
        ],
    )
    wait_result = wait_runtime.run(
        AgentTask(
            task_id="after-cycle-wait-boundary",
            model="test-model",
            system_prompt="system",
            user_prompt="ask",
            max_cycles=3,
        )
    )

    max_runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="candidate")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[
            RecordingHook(decisions=[AfterCycleDecision.steer(["Try again."])])
        ],
    )
    max_result = max_runtime.run(
        AgentTask(
            task_id="after-cycle-max-boundary",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=1,
            no_tool_policy="finish",
        )
    )

    for result in (wait_result, max_result):
        assert result.status is AgentStatus.FAILED
        assert result.completion_reason is CompletionReason.FAILED
        assert result.error is not None
        assert result.error.startswith("after_cycle_steer_unavailable:")


def test_after_cycle_invalid_durable_control_state_fails_before_model(
    tmp_path: Path,
) -> None:
    model_calls = 0

    def complete(_request: LlmRequest) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(content="must not run")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[complete]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    result = runtime.run(
        AgentTask(
            task_id="after-cycle-invalid-state",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            initial_shared_state={
                AFTER_CYCLE_CONTROL_STATE_KEY: {
                    "schema_version": "vv-agent.after-cycle-control.v1",
                    "disallowed_tools": ["z", "a"],
                }
            },
        )
    )

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert result.error.startswith("after_cycle_control_state_invalid:")
    assert model_calls == 0


def test_after_cycle_hook_is_not_called_when_budget_stops_after_model(
    tmp_path: Path,
) -> None:
    hook = RecordingHook(decisions=[AfterCycleDecision.continue_run()])
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="over budget",
                    raw={
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        }
                    },
                )
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[hook],
    )
    result = runtime.run(
        AgentTask(
            task_id="after-cycle-budget-boundary",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=3,
            no_tool_policy="finish",
        ),
        budget_limits=RunBudgetLimits(max_total_tokens=1),
    )

    assert result.status is AgentStatus.FAILED
    assert result.completion_reason is CompletionReason.BUDGET_EXHAUSTED
    assert hook.snapshots == []


def test_after_cycle_snapshot_is_detached_and_composition_overflow_is_typed(
    tmp_path: Path,
) -> None:
    class MutatingHook:
        def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
            nested = snapshot.shared_state["nested"]
            assert isinstance(nested, dict)
            nested["values"].append("hook")
            snapshot.cycle.assistant_message = "mutated copy"
            return AfterCycleDecision.continue_run()

    detached = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="original")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[MutatingHook()],
    ).run(
        AgentTask(
            task_id="after-cycle-copy",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=1,
            no_tool_policy="finish",
            initial_shared_state={"nested": {"values": []}},
        )
    )
    assert detached.final_answer == "original"
    assert detached.cycles[0].assistant_message == "original"
    assert detached.shared_state["nested"] == {"values": []}

    steering = [f"message-{index}" for index in range(20)]
    overflow = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="continue")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        after_cycle_hooks=[
            RecordingHook(decisions=[AfterCycleDecision.steer(steering)]),
            RecordingHook(decisions=[AfterCycleDecision.steer(steering)]),
        ],
    ).run(
        AgentTask(
            task_id="after-cycle-overflow",
            model="test-model",
            system_prompt="system",
            user_prompt="answer",
            max_cycles=2,
            no_tool_policy="continue",
        )
    )
    assert overflow.status is AgentStatus.FAILED
    assert overflow.error is not None
    assert overflow.error.startswith("after_cycle_decision_invalid:")
