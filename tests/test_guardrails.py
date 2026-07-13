from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, GuardrailResult, RunConfig, Runner, input_guardrail, output_guardrail
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def test_input_guardrail_can_block_before_model_call(tmp_path: Path) -> None:
    provider_called = False

    @input_guardrail
    def reject_empty(ctx, input_text: str) -> GuardrailResult:
        del ctx
        if not input_text.strip():
            return GuardrailResult.block("input is required")
        return GuardrailResult.allow()

    def model_provider(agent: Agent, run_config: RunConfig):
        nonlocal provider_called
        del agent, run_config
        provider_called = True
        return ScriptedLLM(), _resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Answer.", model="m", input_guardrails=[reject_empty]),
        "   ",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert provider_called is False
    assert result.status == AgentStatus.FAILED
    assert result.final_output == "input is required"
    assert result.resolved_model is None
    assert result.events[-1].type == "run_failed"


def test_input_guardrail_can_rewrite_input(tmp_path: Path) -> None:
    seen_user_messages: list[str] = []

    @input_guardrail
    def normalize(ctx, input_text: str) -> GuardrailResult:
        del ctx, input_text
        return GuardrailResult.rewrite("rewritten prompt")

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config

        def respond(request: LlmRequest) -> LLMResponse:
            model, messages = request.model, request.messages
            del model
            seen_user_messages.extend(message.content for message in messages if message.role == "user")
            return LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            )

        return ScriptedLLM(steps=[respond]), _resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Answer.", model="m", input_guardrails=[normalize]),
        "original",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.input == "rewritten prompt"
    assert result.final_output == "ok"
    assert seen_user_messages == ["rewritten prompt"]


def test_output_guardrail_can_rewrite_final_output(tmp_path: Path) -> None:
    @output_guardrail
    def redact(ctx, output: str | None) -> GuardrailResult:
        del ctx
        return GuardrailResult.rewrite(f"redacted: {output}")

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "secret"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Answer.", model="m", output_guardrails=[redact]),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "redacted: secret"
