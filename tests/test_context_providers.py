from __future__ import annotations

from vv_agent import Agent, RunConfig, collect_context_fragments
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.context_providers import ContextFragment, ContextRequest, assemble_context_fragments
from vv_agent.runtime.compiler import AgentCompiler


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="requested",
        selected_model="selected",
        model_id="model-id",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="model-id")],
    )


def test_context_fragments_are_ordered_and_source_tagged() -> None:
    fragments = [
        ContextFragment(id="volatile", text="current", stable=False, priority=50, source="test"),
        ContextFragment(id="stable", text="durable", stable=True, priority=10, source="test"),
    ]

    bundle = assemble_context_fragments(ContextRequest(agent_name="assistant"), fragments)

    assert [section.id for section in bundle.sections] == ["stable", "volatile"]
    assert "durable" in bundle.prompt
    assert bundle.sources["stable"] == "test"


def test_context_assembly_skips_empty_fragments_and_reports_metadata() -> None:
    fragments = [
        ContextFragment(id="empty", text="  ", stable=True, priority=1, source="empty"),
        ContextFragment(id="volatile", text="current", stable=False, priority=50, source="test", cache_hint="ephemeral"),
        ContextFragment(id="stable", text="durable", stable=True, priority=10, source="test", cache_hint="cache"),
    ]

    bundle = assemble_context_fragments(ContextRequest(agent_name="assistant", max_prompt_chars=20), fragments)

    assert [section.id for section in bundle.sections] == ["stable", "volatile"]
    assert bundle.prompt == "durable\n\ncurrent"
    assert bundle.total_chars == len(bundle.prompt)
    assert bundle.sources == {"stable": "test", "volatile": "test"}
    assert bundle.metadata_sections() == [
        {"id": "stable", "text": "durable", "stable": True, "source": "test", "cache_hint": "cache"},
        {"id": "volatile", "text": "current", "stable": False, "source": "test", "cache_hint": "ephemeral"},
    ]


def test_context_budget_counts_unicode_characters() -> None:
    bundle = assemble_context_fragments(
        ContextRequest(agent_name="assistant", max_prompt_chars=4),
        [ContextFragment(id="unicode", text="你好世界")],
    )

    assert bundle.prompt == "你好世界"
    assert bundle.total_chars == 4
    assert bundle.omitted_section_ids == []


def test_collect_context_fragments_is_exported_from_package_root() -> None:
    expected = ContextFragment(id="runtime", text="Current runtime context")

    class StaticProvider:
        def fragments(self, request: ContextRequest) -> list[ContextFragment]:
            del request
            return [expected]

    assert collect_context_fragments(
        ContextRequest(agent_name="assistant"),
        [StaticProvider()],
    ) == [expected]


def test_compiler_assembles_agent_instructions_with_context_providers() -> None:
    class StaticProvider:
        def fragments(self, request: ContextRequest) -> list[ContextFragment]:
            assert request.agent_name == "ops"
            assert request.input == "analyze order"
            return [
                ContextFragment(id="runtime_context", text="Current order status.", stable=False, priority=-10, source="test")
            ]

    task = AgentCompiler().compile(
        agent=Agent(name="ops", instructions="Check facts.", model="agent-model"),
        input="analyze order",
        run_config=RunConfig(context_providers=[StaticProvider()], metadata={"request_id": "r1"}),
        resolved=_resolved(),
        trace_id="trace-1",
    )

    assert task.system_prompt == "Current order status.\n\nCheck facts."
    assert task.metadata["system_prompt_sources"] == {
        "agent_instructions": "agent.instructions",
        "runtime_context": "test",
    }
    assert task.metadata["system_prompt_sections"] == [
        {
            "id": "runtime_context",
            "text": "Current order status.",
            "stable": False,
            "source": "test",
        },
        {
            "id": "agent_instructions",
            "text": "Check facts.",
            "stable": True,
            "source": "agent.instructions",
        },
    ]


def test_compiler_reports_agent_instructions_as_builtin_context_section() -> None:
    task = AgentCompiler().compile(
        agent=Agent(name="ops", instructions="Check facts.", model="agent-model"),
        input="analyze order",
        run_config=RunConfig(metadata={"request_id": "r1"}),
        resolved=_resolved(),
        trace_id="trace-1",
    )

    assert task.system_prompt == "Check facts."
    assert task.metadata["system_prompt_sections"] == [
        {
            "id": "agent_instructions",
            "text": "Check facts.",
            "stable": True,
            "source": "agent.instructions",
        }
    ]
