from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from vv_agent import (
    AfterCycleDecision,
    AfterCycleSnapshot,
    Agent,
    ModelRef,
    ModelSettings,
    RunConfig,
    Runner,
    ScriptedModelProvider,
    ToolContext,
    function_tool,
)
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _finish_response(message: str = "done") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="finish",
                name=TASK_FINISH_TOOL_NAME,
                arguments={"message": message},
            )
        ],
    )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", (1 << 32)])
def test_run_config_rejects_invalid_max_cycles(value: object) -> None:
    with pytest.raises(ValueError, match="max_cycles must be between 1 and 4294967295"):
        RunConfig(max_cycles=cast(int, value))


def test_agent_rejects_invalid_max_cycles() -> None:
    with pytest.raises(ValueError, match="max_cycles must be between 1 and 4294967295"):
        Agent(name="invalid", instructions="Invalid.", max_cycles=0)


def test_run_config_max_handoffs_accepts_zero_but_rejects_invalid_types() -> None:
    assert RunConfig(max_handoffs=0).max_handoffs == 0
    for value in (True, -1, 1.5, "1", 1 << 32):
        with pytest.raises(ValueError, match="max_handoffs must be between 0 and 4294967295"):
            RunConfig(max_handoffs=cast(int, value))


def test_configured_runner_uses_shared_provider_model_and_settings_precedence(tmp_path: Path) -> None:
    requests: list[LlmRequest] = []

    def respond(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return _finish_response()

    provider = ScriptedModelProvider.from_callback("scripted", "provider-model", respond).with_default_settings(
        ModelSettings(
            temperature=0.1,
            top_p=0.1,
            max_tokens=100,
            parallel_tool_calls=False,
            extra_body={"winner": "provider", "provider_only": True},
        )
    )
    runner = Runner.configured(
        RunConfig(
            model_provider=provider,
            model=ModelRef.named("runner-model"),
            model_settings=ModelSettings(
                temperature=0.2,
                top_p=0.2,
                max_tokens=200,
                extra_body={"winner": "runner", "runner_only": True},
            ),
            workspace=tmp_path,
        )
    )
    agent = Agent(
        name="assistant",
        instructions="Finish.",
        model=ModelRef.named("agent-model"),
        model_settings=ModelSettings(
            temperature=0.3,
            top_p=0.3,
            extra_body={"winner": "agent", "agent_only": True},
        ),
    )

    result = runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model=ModelRef.named("run-model"),
            model_settings=ModelSettings(
                temperature=0.4,
                extra_body={"winner": "run", "run_only": True},
            ),
        ),
    )

    assert result.final_output == "done"
    assert requests[0].model == "run-model"
    assert requests[0].model_settings == ModelSettings(
        temperature=0.4,
        top_p=0.3,
        max_tokens=200,
        parallel_tool_calls=False,
        extra_body={
            "winner": "run",
            "provider_only": True,
            "runner_only": True,
            "agent_only": True,
            "run_only": True,
        },
    )


def test_configured_runner_model_fallback_order(tmp_path: Path) -> None:
    models: list[str] = []

    def respond(request: LlmRequest) -> LLMResponse:
        models.append(request.model)
        return _finish_response()

    provider = ScriptedModelProvider.from_callback("scripted", "provider-model", respond)
    runner = Runner.configured(
        RunConfig(
            model_provider=provider,
            model=ModelRef.named("runner-model"),
            workspace=tmp_path,
        )
    )
    agent = Agent(name="assistant", instructions="Finish.", model=ModelRef.named("agent-model"))

    runner.run_sync(agent, "run", run_config=RunConfig(model=ModelRef.named("run-model")))
    runner.run_sync(agent, "agent")
    runner.run_sync(Agent(name="runner", instructions="Finish."), "runner")
    Runner.configured(RunConfig(model_provider=provider, workspace=tmp_path)).run_sync(
        Agent(name="provider", instructions="Finish."),
        "provider",
    )

    assert models == ["run-model", "agent-model", "runner-model", "provider-model"]


def test_per_run_provider_override_does_not_reuse_runner_backend_model(tmp_path: Path) -> None:
    runner_provider = ScriptedModelProvider.new("runner", "runner-provider-default", [])
    override_requests: list[LlmRequest] = []
    override_provider = ScriptedModelProvider.from_callback(
        "override",
        "override-provider-default",
        lambda request: override_requests.append(request) or _finish_response(),
    )
    runner = Runner.configured(
        RunConfig(
            model_provider=runner_provider,
            model=ModelRef.backend("runner", "runner-model"),
            workspace=tmp_path,
        )
    )

    result = runner.run_sync(
        Agent(name="assistant", instructions="Finish."),
        "go",
        run_config=RunConfig(model_provider=override_provider),
    )

    assert result.resolved is not None
    assert result.resolved.backend == "override"
    assert result.resolved.model_id == "override-provider-default"
    assert [request.model for request in override_requests] == ["override-provider-default"]


def test_configured_runner_start_and_resume_preserve_runner_defaults(tmp_path: Path) -> None:
    executions: list[str] = []
    runner_metadata: list[object] = []

    @function_tool(needs_approval=True)
    def guarded_write(context: ToolContext, value: str) -> str:
        executions.append(value)
        runner_metadata.append(context.task_metadata.get("runner_default"))
        return value

    provider = ScriptedModelProvider.from_callback(
        "scripted",
        "provider-model",
        lambda _request: LLMResponse(
            content="write",
            tool_calls=[ToolCall(id="write", name="guarded_write", arguments={"value": "approved"})],
        ),
    )
    runner = Runner.configured(
        RunConfig(
            model_provider=provider,
            model=ModelRef.named("runner-model"),
            workspace=tmp_path,
            metadata={"runner_default": True},
        )
    )
    agent = Agent(
        name="writer",
        instructions="Write after approval.",
        tools=[guarded_write],
        tool_use_behavior="stop_on_first_tool",
    )

    handle = runner.start(agent, "write")
    interrupted = handle.result(timeout=2)
    assert interrupted.status == AgentStatus.WAIT_USER
    state = interrupted.into_state()
    state.approve(state.pending_approval_ids()[0])

    resumed = handle.resume(state)

    assert resumed is not None
    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "approved"
    assert executions == ["approved"]
    assert runner_metadata == [True]


def test_explicit_framework_values_override_configured_runner_defaults() -> None:
    agent = Agent(name="assistant", instructions="Finish.")
    runner_defaults = RunConfig(settings_file="runner_settings.py", timeout_seconds=12.0)

    inherited = Runner._effective_run_config(agent, RunConfig(), runner_defaults=runner_defaults)
    overridden = Runner._effective_run_config(
        agent,
        RunConfig(settings_file="local_settings.py", timeout_seconds=90.0),
        runner_defaults=runner_defaults,
    )

    assert inherited.settings_file == "runner_settings.py"
    assert inherited.timeout_seconds == 12.0
    assert overridden.settings_file == "local_settings.py"
    assert overridden.timeout_seconds == 90.0


def test_configured_runner_runs_default_after_cycle_hooks_before_per_run_hooks(
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class OrderedHook:
        def __init__(self, name: str) -> None:
            self.name = name

        def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
            del snapshot
            order.append(self.name)
            return AfterCycleDecision.continue_run()

    provider = ScriptedModelProvider.from_steps(
        "scripted",
        "test-model",
        [LLMResponse(content="done")],
    )
    runner = Runner.configured(
        RunConfig(
            model_provider=provider,
            workspace=tmp_path,
            no_tool_policy="finish",
            after_cycle_hooks=[OrderedHook("runner")],
        )
    )
    result = runner.run_sync(
        Agent(name="ordered-hooks", instructions="Answer.", model="test-model"),
        "answer",
        run_config=RunConfig(after_cycle_hooks=[OrderedHook("run")]),
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.final_output == "done"
    assert order == ["runner", "run"]


def test_resume_uses_the_runner_that_created_the_state(tmp_path: Path) -> None:
    origin_requests: list[LlmRequest] = []
    origin_provider = ScriptedModelProvider.from_steps(
        "origin",
        "origin-model",
        [
            lambda request: (
                origin_requests.append(request)
                or LLMResponse(
                    content="need input",
                    tool_calls=[ToolCall(id="ask", name="ask_user", arguments={"question": "Which color?"})],
                )
            ),
            lambda request: origin_requests.append(request) or _finish_response("selected blue"),
        ],
    )
    receiving_provider = ScriptedModelProvider.new("receiving", "receiving-model", [])
    origin = Runner.configured(RunConfig(model_provider=origin_provider, workspace=tmp_path))
    receiving = Runner.configured(RunConfig(model_provider=receiving_provider, workspace=tmp_path))
    agent = Agent(name="assistant", instructions="Ask once, then finish.")

    interrupted = origin.run_sync(agent, "choose")
    assert interrupted.status == AgentStatus.WAIT_USER

    resumed = receiving.resume(interrupted.into_state(), input="blue")

    assert resumed.status == AgentStatus.COMPLETED
    assert resumed.final_output == "selected blue"
    assert [request.model for request in origin_requests] == ["origin-model", "origin-model"]
