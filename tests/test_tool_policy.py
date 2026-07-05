from __future__ import annotations

from pathlib import Path
from typing import cast

from vv_agent import Agent, RunConfig, Runner, ToolPolicy, build_default_registry, function_tool
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import (
    BASH_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.types import LLMResponse, Message, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


class CapturingLLM:
    def __init__(self) -> None:
        self.tool_names: list[str] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback=None,
        model_settings=None,
    ) -> LLMResponse:
        del model, messages, stream_callback, model_settings
        names: list[str] = []
        for tool in tools:
            function = cast(dict[str, object], tool["function"])
            assert isinstance(function, dict)
            names.append(str(function["name"]))
        self.tool_names = names
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )


def test_tool_policy_allowed_tools_filters_default_and_custom_tool_schemas(tmp_path: Path) -> None:
    @function_tool
    def allowed() -> str:
        """Allowed tool."""
        return "allowed"

    @function_tool
    def secret() -> str:
        """Secret tool."""
        return "secret"

    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m", tools=[allowed, secret]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "allowed"]),
        ),
    )

    assert result.final_output == "ok"
    assert llm.tool_names == [TASK_FINISH_TOOL_NAME, "allowed"]


def test_tool_policy_disallowed_tools_filters_custom_tool_schema(tmp_path: Path) -> None:
    @function_tool
    def blocked() -> str:
        """Blocked tool."""
        return "blocked"

    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m", tools=[blocked]),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(disallowed_tools=["blocked"]),
        ),
    )

    assert "blocked" not in llm.tool_names


def test_default_registry_factory_does_not_expose_gated_tools_for_plain_agent(tmp_path: Path) -> None:
    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m"),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_registry_factory=build_default_registry,
        ),
    )

    assert BASH_TOOL_NAME not in llm.tool_names
    assert CREATE_SUB_TASK_TOOL_NAME not in llm.tool_names
    assert READ_IMAGE_TOOL_NAME not in llm.tool_names


def test_tool_policy_disallowed_default_tool_cannot_execute_even_if_model_calls_it(tmp_path: Path) -> None:
    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            CapturingLLMWithToolCall(
                WRITE_FILE_TOOL_NAME,
                {"path": "secret.txt", "content": "leak"},
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m"),
        "write",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(disallowed_tools=[WRITE_FILE_TOOL_NAME]),
        ),
    )

    assert not (tmp_path / "secret.txt").exists()
    assert result.raw_result.cycles[0].tool_results[0].error_code == "tool_not_allowed"


def test_tool_policy_allowed_tools_can_expose_edit_file_schema(tmp_path: Path) -> None:
    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m"),
        "edit",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, EDIT_FILE_TOOL_NAME]),
        ),
    )

    assert result.final_output == "ok"
    assert set(llm.tool_names) == {EDIT_FILE_TOOL_NAME, TASK_FINISH_TOOL_NAME}


def test_tool_policy_disallowed_tools_can_hide_edit_file_schema(tmp_path: Path) -> None:
    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m"),
        "edit",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(disallowed_tools=[EDIT_FILE_TOOL_NAME]),
        ),
    )

    assert EDIT_FILE_TOOL_NAME not in llm.tool_names


def test_tool_policy_can_use_tool_blocks_executor_registered_tool(tmp_path: Path) -> None:
    calls: list[str] = []

    @function_tool
    def registered_executor(path: str) -> str:
        calls.append(path)
        return "ran"

    def registry_factory():
        registry = build_default_registry()
        registry.register_executor(registered_executor.to_executor())
        return registry

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            CapturingLLMWithToolCall("registered_executor", {"path": "secret.txt"}),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m"),
        "run",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_registry_factory=registry_factory,
            tool_policy=ToolPolicy(can_use_tool=lambda name, args: False),
        ),
    )

    assert calls == []
    assert result.raw_result.cycles[0].tool_results[0].error_code == "tool_not_allowed"


def test_function_tool_is_enabled_false_hides_schema(tmp_path: Path) -> None:
    @function_tool(is_enabled=False)
    def disabled() -> str:
        """Disabled tool."""
        return "disabled"

    llm = CapturingLLM()

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _resolved()

    Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m", tools=[disabled]),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert "disabled" not in llm.tool_names


def test_tool_policy_can_use_tool_blocks_invocation_with_arguments(tmp_path: Path) -> None:
    invoked = False

    @function_tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        nonlocal invoked
        invoked = True
        return f"deleted {path}"

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            CapturingLLMWithToolCall("delete_file", {"path": "secrets.txt"}),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use tools.",
            model="m",
            tools=[delete_file],
            tool_use_behavior="stop_on_first_tool",
        ),
        "delete",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(
                can_use_tool=lambda tool_name, arguments: tool_name != "delete_file"
                or arguments.get("path") != "secrets.txt"
            ),
        ),
    )

    assert invoked is False
    assert result.final_output is not None
    assert "not allowed" in result.final_output


class CapturingLLMWithToolCall:
    def __init__(self, tool_name: str, arguments: dict[str, object]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback=None,
        model_settings=None,
    ) -> LLMResponse:
        del model, messages, tools, stream_callback, model_settings
        return LLMResponse(
            content="call tool",
            tool_calls=[ToolCall(id="policy-call", name=self.tool_name, arguments=self.arguments)],
        )
