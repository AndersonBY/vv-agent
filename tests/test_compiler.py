from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from vv_agent import Agent, RunConfig, ToolPolicy, function_tool, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.types import AgentTask, Message


def _resolved(
    *,
    context_length: int | None = None,
    max_output_tokens: int | None = None,
) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="requested",
        selected_model="selected",
        model_id="model-id",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="model-id")],
        context_length=context_length,
        max_output_tokens=max_output_tokens,
    )


def test_agent_compiler_builds_runtime_task_from_public_contract() -> None:
    @function_tool
    def lookup(order_id: str) -> str:
        """Lookup order."""
        return order_id

    writer = Agent(name="writer", instructions="Write.", model="writer")
    agent = Agent(
        name="ops",
        instructions="Check facts.",
        model="agent-model",
        tools=[lookup],
        handoffs=[handoff(agent=writer, description="Transfer to writer.")],
        metadata={"team": "ops"},
    )

    task = AgentCompiler().compile(
        agent=agent,
        input="analyze order",
        run_config=RunConfig(
            model="override-model",
            max_cycles=12,
            tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "lookup", "transfer_to_writer"]),
            metadata={"request_id": "r1"},
        ),
        resolved=_resolved(),
        trace_id="trace-1",
    )

    assert isinstance(task, AgentTask)
    assert task.task_id.startswith("ops_")
    assert task.model == "model-id"
    assert task.system_prompt == "Check facts."
    assert task.user_prompt == "analyze order"
    assert task.max_cycles == 12
    assert task.extra_tool_names == ["lookup", "transfer_to_writer"]
    assert task.metadata["team"] == "ops"
    assert task.metadata["request_id"] == "r1"
    assert task.metadata["_vv_agent_allowed_tools"] == [TASK_FINISH_TOOL_NAME, "lookup", "transfer_to_writer"]
    assert task.metadata["trace_id"] == "trace-1"
    assert not hasattr(task, "runtime_metadata")


def test_agent_compiler_records_model_output_capability_without_fabricating_reserve() -> None:
    task = AgentCompiler().compile(
        agent=Agent(name="assistant", instructions="Answer.", model="model-id"),
        input="go",
        run_config=RunConfig(),
        resolved=_resolved(context_length=1_048_576, max_output_tokens=1_048_576),
        trace_id="trace-capacity",
    )

    assert task.metadata["model_context_window"] == 1_048_576
    assert task.metadata["model_max_output_tokens"] == 1_048_576
    assert "reserved_output_tokens" not in task.metadata


def test_agent_compiler_treats_non_positive_context_metadata_as_absent() -> None:
    task = AgentCompiler().compile(
        agent=Agent(
            name="assistant",
            instructions="Answer.",
            model="model-id",
            metadata={"model_context_window": 0},
        ),
        input="go",
        run_config=RunConfig(),
        resolved=_resolved(context_length=64_000, max_output_tokens=8_192),
        trace_id="trace-positive-context",
    )

    assert task.metadata["model_context_window"] == 64_000
    assert task.metadata["model_max_output_tokens"] == 8_192


def test_frozen_checkpoint_uses_current_threshold_and_metadata_without_rewriting_record() -> None:
    definition = {
        "compiled_prompt": "Answer.",
        "root_input": "go",
        "initial_messages": [],
        "initial_shared_state": {},
        "run_metadata": {},
        "model": {"model_id": "model-id"},
        "agent": {"type": None},
        "tools": [],
        "runtime_controls": {
            "max_cycles": 8,
            "memory_compact_threshold": 128_000,
            "memory_threshold_percentage": 90,
            "no_tool_policy": "continue",
            "allow_interruption": True,
            "native_multimodal": False,
            "tool_use_behavior": "run_llm_again",
            "stop_at_tool_names": [],
        },
    }
    system_message = Message(
        role="system",
        content="Answer.",
        metadata={"reserved_output_tokens": 4_096},
    )
    checkpoint = SimpleNamespace(
        run_definition=definition,
        messages=[system_message],
        task_id="checkpoint-task",
    )
    original_definition = deepcopy(definition)
    original_metadata = dict(system_message.metadata)

    task = AgentCompiler().compile_frozen_checkpoint(
        agent=Agent(name="assistant", instructions="Answer.", model="model-id"),
        run_config=RunConfig(),
        resolved=_resolved(context_length=32_000, max_output_tokens=8_192),
        checkpoint=checkpoint,
        trace_id="trace-resume-capacity",
    )

    assert task.memory_compact_threshold == 128_000
    assert task.metadata["model_context_window"] == 32_000
    assert task.metadata["model_max_output_tokens"] == 8_192
    assert task.metadata["reserved_output_tokens"] == 4_096
    assert definition == original_definition
    assert system_message.metadata == original_metadata


def test_frozen_checkpoint_restores_run_metadata_when_system_metadata_is_empty() -> None:
    definition = {
        "compiled_prompt": "Answer.",
        "root_input": "go",
        "initial_messages": [],
        "initial_shared_state": {},
        "run_metadata": {
            "reserved_output_tokens": 4_096,
            "host_request_id": "request-42",
            "model_context_window": 48_000,
        },
        "model": {"model_id": "model-id"},
        "agent": {"type": None},
        "tools": [],
        "runtime_controls": {
            "max_cycles": 8,
            "memory_compact_threshold": 128_000,
            "memory_threshold_percentage": 90,
            "no_tool_policy": "continue",
            "allow_interruption": True,
            "native_multimodal": False,
            "tool_use_behavior": "run_llm_again",
            "stop_at_tool_names": [],
        },
    }
    system_message = Message(role="system", content="Answer.", metadata={})
    checkpoint = SimpleNamespace(
        run_definition=definition,
        messages=[system_message],
        task_id="frozen-metadata-task",
    )
    original_definition = deepcopy(definition)

    task = AgentCompiler().compile_frozen_checkpoint(
        agent=Agent(name="assistant", instructions="Answer.", model="model-id"),
        run_config=RunConfig(),
        resolved=_resolved(context_length=32_000, max_output_tokens=8_192),
        checkpoint=checkpoint,
        trace_id="trace-frozen-metadata",
    )

    assert task.metadata["reserved_output_tokens"] == 4_096
    assert task.metadata["host_request_id"] == "request-42"
    assert task.metadata["model_context_window"] == 48_000
    assert task.metadata["model_max_output_tokens"] == 8_192
    assert task.metadata["trace_id"] == "trace-frozen-metadata"
    assert definition == original_definition
    assert system_message.metadata == {}

    stale_system_message = Message(
        role="system",
        content="Answer.",
        metadata={
            "reserved_output_tokens": 1_024,
            "host_request_id": "stale-request",
            "model_context_window": 16_000,
        },
    )
    precedence_task = AgentCompiler().compile_frozen_checkpoint(
        agent=Agent(name="assistant", instructions="Answer.", model="model-id"),
        run_config=RunConfig(),
        resolved=_resolved(context_length=32_000, max_output_tokens=8_192),
        checkpoint=SimpleNamespace(
            run_definition=definition,
            messages=[stale_system_message],
            task_id="frozen-metadata-precedence-task",
        ),
        trace_id="trace-frozen-metadata-precedence",
    )

    assert precedence_task.metadata["reserved_output_tokens"] == 4_096
    assert precedence_task.metadata["host_request_id"] == "request-42"
    assert precedence_task.metadata["model_context_window"] == 48_000
    assert stale_system_message.metadata["host_request_id"] == "stale-request"
