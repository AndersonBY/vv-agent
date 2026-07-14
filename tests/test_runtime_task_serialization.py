from __future__ import annotations

from vv_agent.types import (
    AgentTask,
    CacheUsage,
    CacheUsageStatus,
    CycleRecord,
    SubAgentConfig,
    TaskTokenUsage,
    TokenUsage,
    UsageSource,
)


def test_agent_task_round_trips_sub_agents_and_metadata() -> None:
    task = AgentTask(
        task_id="parent",
        model="m",
        system_prompt="sys",
        user_prompt="user",
        sub_agents={"research": SubAgentConfig(model="m2", description="Research facts.", backend="b")},
        metadata={"legacy": True, "trace_id": "trace-1"},
    )

    payload = task.to_dict()
    restored = AgentTask.from_dict(payload)

    assert restored.sub_agents_enabled is True
    assert restored.sub_agents["research"].model == "m2"
    assert restored.sub_agents["research"].description == "Research facts."
    assert restored.sub_agents["research"].backend == "b"
    assert restored.metadata == {"legacy": True, "trace_id": "trace-1"}
    assert "runtime_metadata" not in payload


def test_cycle_record_round_trips_token_usage() -> None:
    cycle = CycleRecord(
        index=1,
        assistant_message="answer",
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=3,
            reasoning_tokens=2,
            input_tokens=11,
            output_tokens=4,
            cache_creation_tokens=1,
            usage_source=UsageSource.PROVIDER_REPORTED,
            cache_usage=CacheUsage(
                status=CacheUsageStatus.PROVIDER_REPORTED,
                read_tokens=3,
                write_tokens=1,
                uncached_input_tokens=7,
                source="provider_usage",
            ),
            raw={"provider": "raw"},
        ),
    )

    restored = CycleRecord.from_dict(cycle.to_dict())

    assert restored.token_usage.prompt_tokens == 10
    assert restored.token_usage.completion_tokens == 5
    assert restored.token_usage.total_tokens == 15
    assert restored.token_usage.cached_tokens == 3
    assert restored.token_usage.reasoning_tokens == 2
    assert restored.token_usage.input_tokens == 11
    assert restored.token_usage.output_tokens == 4
    assert restored.token_usage.cache_creation_tokens == 1
    assert restored.token_usage.usage_source is UsageSource.PROVIDER_REPORTED
    assert restored.token_usage.cache_usage.read_tokens == 3
    assert restored.token_usage.cache_usage.uncached_input_tokens == 7
    assert restored.token_usage.raw == {"provider": "raw"}


def test_task_token_usage_round_trips_cycle_breakdown() -> None:
    usage = TaskTokenUsage()
    usage.add_cycle(1, TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    usage.add_cycle(2, TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15))

    restored = TaskTokenUsage.from_dict(usage.to_dict())

    assert len(restored.cycles) == 2
    assert restored.cycles[0].cycle_index == 1
    assert restored.cycles[0].usage.prompt_tokens == 10
    assert restored.cycles[1].cycle_index == 2
    assert restored.cycles[1].usage.output_tokens == 7
