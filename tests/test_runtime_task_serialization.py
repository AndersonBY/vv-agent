from __future__ import annotations

from vv_agent.types import (
    AgentTask,
    CacheUsage,
    CacheUsageStatus,
    CycleRecord,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
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
        metadata={"scope": "research", "trace_id": "trace-1"},
    )

    payload = task.to_dict()
    restored = AgentTask.from_dict(payload)

    assert restored.sub_agents_enabled is True
    assert restored.sub_agents["research"].model == "m2"
    assert restored.sub_agents["research"].description == "Research facts."
    assert restored.sub_agents["research"].backend == "b"
    assert restored.metadata == {"scope": "research", "trace_id": "trace-1"}


def test_cycle_record_round_trips_without_model_usage() -> None:
    cycle = CycleRecord(
        index=1,
        assistant_message="answer",
    )

    payload = cycle.to_dict()
    restored = CycleRecord.from_dict(payload)

    assert "token_usage" not in payload
    assert restored.index == 1
    assert restored.assistant_message == "answer"


def test_task_token_usage_round_trips_model_call_breakdown() -> None:
    usage = TaskTokenUsage()
    usage.add_model_call(
        ModelCallRecord(
            call_id="op_model_cycle_1_main:attempt:1",
            operation_id="op_model_cycle_1_main",
            attempt=1,
            operation=ModelCallOperation.AGENT_CYCLE,
            cycle_index=1,
            backend="test",
            model="model-a",
            status=ModelCallStatus.COMPLETED,
            usage=TokenUsage(
                input_tokens=11,
                output_tokens=4,
                total_tokens=15,
                reasoning_tokens=2,
                usage_source=UsageSource.PROVIDER_REPORTED,
                cache_usage=CacheUsage(
                    status=CacheUsageStatus.PROVIDER_REPORTED,
                    read_input_tokens=3,
                    write_input_tokens=1,
                    uncached_input_tokens=7,
                    source="provider_usage",
                ),
                provider_usage={"provider": "raw"},
            ),
        )
    )
    usage.add_model_call(
        ModelCallRecord(
            call_id="op_model_cycle_2_main:attempt:1",
            operation_id="op_model_cycle_2_main",
            attempt=1,
            operation=ModelCallOperation.AGENT_CYCLE,
            cycle_index=2,
            backend="test",
            model="model-a",
            status=ModelCallStatus.COMPLETED,
            usage=TokenUsage(input_tokens=8, output_tokens=7, total_tokens=15),
        )
    )

    restored = TaskTokenUsage.from_dict(usage.to_dict())

    assert len(restored.model_calls) == 2
    assert restored.model_calls[0].cycle_index == 1
    assert restored.model_calls[0].usage.input_tokens == 11
    assert restored.model_calls[0].usage.reasoning_tokens == 2
    assert restored.model_calls[0].usage.usage_source is UsageSource.PROVIDER_REPORTED
    assert restored.model_calls[0].usage.cache_usage.read_input_tokens == 3
    assert restored.model_calls[0].usage.provider_usage == {"provider": "raw"}
    assert restored.model_calls[1].cycle_index == 2
    assert restored.model_calls[1].usage.output_tokens == 7
