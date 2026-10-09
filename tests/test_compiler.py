from __future__ import annotations

import json
from pathlib import Path

from vv_agent import Agent, RunConfig, ToolPolicy, function_tool, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.types import AgentTask


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
            tool_policy=ToolPolicy(allowed_tools=["lookup", "transfer_to_writer"]),
            metadata={"request_id": "r1"},
        ),
        resolved=_resolved(),
        trace_id="trace-1",
    )

    assert isinstance(task, AgentTask)
    assert task.task_id.startswith("ops_")
    assert task.model == "model-id"
    assert task.prompt_bundle.flatten() == "Check facts."
    assert task.prompt_bundle.sections[0].id == "agent_instructions"
    assert task.user_prompt == "analyze order"
    assert task.max_cycles == 12
    assert task.extra_tool_names == ["lookup", "transfer_to_writer"]
    assert task.metadata["team"] == "ops"
    assert task.metadata["request_id"] == "r1"
    assert task.metadata["_vv_agent_allowed_tools"] == ["lookup", "transfer_to_writer"]
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


def test_agent_compiler_freezes_loaded_session_memory_into_a_new_run_bundle(tmp_path: Path) -> None:
    storage = tmp_path / ".memory" / "session" / "shared-session" / "session_memory.json"
    storage.parent.mkdir(parents=True)
    storage.write_text(
        json.dumps(
            {
                "entries": [
                    {"category": "decision", "content": "reuse the reviewed evidence", "source_cycle": 4, "importance": 9}
                ],
                "last_extracted_message_index": 3,
                "tokens_at_last_extraction": 100,
                "initialized": True,
            }
        ),
        encoding="utf-8",
    )

    task = AgentCompiler().compile(
        agent=Agent(name="assistant", instructions="Answer.", model="model-id"),
        input="continue",
        run_config=RunConfig(
            workspace=tmp_path,
            session_memory_enabled=True,
            metadata={"session_id": "shared-session"},
        ),
        resolved=_resolved(),
        trace_id="trace-session-memory",
    )

    assert [section.id for section in task.prompt_bundle.sections] == ["agent_instructions", "session_memory"]
    assert "reuse the reviewed evidence" in task.prompt_bundle.flatten()
