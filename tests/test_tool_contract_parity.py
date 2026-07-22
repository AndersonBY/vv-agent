from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from vv_agent import Agent, RunConfig, Runner, ToolContext, ToolOutputText
from vv_agent.model import ScriptedModelProvider
from vv_agent.model_settings import ModelSettings
from vv_agent.tools import RegistryToolExecutor, ToolExposure, ToolOrchestrator
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall, ToolExecutionResult
from vv_agent.workspace import MemoryWorkspaceBackend


class ProtocolEchoTool:
    name = "protocol_echo"
    description = "Echo a value through the public Tool protocol."
    params_json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    strict_json_schema = True
    is_enabled = True
    needs_approval = False
    exposure = ToolExposure.DIRECT

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def invoke(self, context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutputText:
        assert context is not None
        value = str(arguments["value"])
        self.calls.append((context.tool_call_id, value))
        return ToolOutputText(text=f"echo:{value}")


class CapturingToolLLM:
    model_id = "tool-contract"

    def __init__(self) -> None:
        self.tool_names: list[str] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback=None,
        model_settings: ModelSettings | None = None,
        request_metadata=None,
    ) -> LLMResponse:
        del model, messages, stream_callback, model_settings, request_metadata
        self.tool_names = [str(cast(dict[str, object], schema["function"])["name"]) for schema in tools]
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="protocol-call", name="protocol_echo", arguments={"value": "ok"})],
        )


def test_runner_adapts_public_tool_protocol_instead_of_silently_skipping_it(tmp_path: Path) -> None:
    tool = ProtocolEchoTool()
    llm = CapturingToolLLM()

    result = Runner.run_sync(
        Agent(
            name="protocol-agent",
            instructions="Use the protocol tool.",
            model="tool-contract",
            tools=[tool],
            tool_use_behavior="stop_on_first_tool",
        ),
        "echo",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=ScriptedModelProvider(
                backend="test",
                default_model="tool-contract",
                llm=llm,
            ),
        ),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "echo:ok"
    assert "protocol_echo" in llm.tool_names
    assert tool.calls == [("protocol-call", "ok")]


def test_runner_rejects_invalid_agent_tool_with_clear_type_error() -> None:
    with pytest.raises(TypeError, match="implement the Tool protocol"):
        llm = CapturingToolLLM()
        Runner.run_sync(
            Agent(
                name="invalid-tool-agent",
                instructions="Finish.",
                model="tool-contract",
                tools=[object()],
            ),
            "go",
            run_config=RunConfig(
                model_provider=ScriptedModelProvider(
                    backend="test",
                    default_model="tool-contract",
                    llm=llm,
                )
            ),
        )


def test_orchestrator_applies_registry_executor_failure_formatter(tmp_path: Path) -> None:
    def fail(_context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        raise RuntimeError("upstream unavailable")

    executor = RegistryToolExecutor(
        name="registry_failure",
        handler=fail,
        description="Always fail.",
        failure_error_function=lambda exc: f"mapped: {exc}",
    )
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=MemoryWorkspaceBackend(),
    )

    result = ToolOrchestrator.from_tools([executor]).run_one(
        ToolCall(id="registry-call", name="registry_failure", arguments={}),
        context=context,
    )

    assert result.error_code == "tool_execution_failed"
    assert json.loads(result.content)["error"] == "mapped: upstream unavailable"
