from __future__ import annotations

from typing import Any

import pytest

from vv_agent.types import AgentTask, SubAgentConfig

INVALID_SUB_AGENT_VALUES: list[tuple[str, Any, type[Exception]]] = [
    ("model", 123, TypeError),
    ("description", None, TypeError),
    ("backend", 123, TypeError),
    ("system_prompt", 123, TypeError),
    ("max_cycles", True, TypeError),
    ("max_cycles", 1.5, TypeError),
    ("max_cycles", -1, ValueError),
    ("max_cycles", 1 << 32, ValueError),
    ("exclude_tools", "read_file", TypeError),
    ("exclude_tools", ["read_file", 123], TypeError),
    ("metadata", [], TypeError),
    ("metadata", {1: "value"}, TypeError),
]

INVALID_AGENT_TASK_VALUES: list[tuple[str, Any]] = [
    ("max_cycles", True),
    ("max_cycles", -1),
    ("max_cycles", 1 << 32),
    ("max_cycles", 1.5),
    ("memory_compact_threshold", True),
    ("memory_compact_threshold", -1),
    ("memory_compact_threshold", 1 << 64),
    ("memory_compact_threshold", 1.5),
    ("memory_threshold_percentage", True),
    ("memory_threshold_percentage", -1),
    ("memory_threshold_percentage", 256),
    ("memory_threshold_percentage", 1.5),
    ("no_tool_policy", 1),
    ("no_tool_policy", "invalid"),
    ("allow_interruption", 1),
    ("use_workspace", 1),
    ("has_sub_agents", 1),
    ("sub_agents", []),
    ("sub_agents", {"research": "not-an-object"}),
    ("agent_type", 1),
    ("native_multimodal", 1),
    ("extra_tool_names", "read_file"),
    ("extra_tool_names", ["read_file", 1]),
    ("exclude_tools", "write_file"),
    ("exclude_tools", ["write_file", 1]),
    ("model_settings", []),
    ("initial_messages", {}),
    ("initial_messages", [1]),
    ("initial_shared_state", []),
    ("metadata", []),
]


def _sub_agent_values() -> dict[str, Any]:
    return {
        "model": "child-model",
        "description": "Research",
        "backend": "openai",
        "system_prompt": "Child prompt",
        "max_cycles": 8,
        "exclude_tools": ["write_file"],
        "metadata": {"scope": "research"},
    }


def _agent_task_values() -> dict[str, Any]:
    return {
        "task_id": "task-1",
        "model": "model-1",
        "system_prompt": "system",
        "user_prompt": "user",
    }


@pytest.mark.parametrize(("field_name", "invalid_value", "error_type"), INVALID_SUB_AGENT_VALUES)
def test_sub_agent_config_constructor_rejects_non_rust_values(
    field_name: str,
    invalid_value: Any,
    error_type: type[Exception],
) -> None:
    values = _sub_agent_values()
    values[field_name] = invalid_value

    with pytest.raises(error_type):
        SubAgentConfig(**values)


@pytest.mark.parametrize(("field_name", "invalid_value", "error_type"), INVALID_SUB_AGENT_VALUES)
def test_sub_agent_config_from_dict_rejects_non_rust_values(
    field_name: str,
    invalid_value: Any,
    error_type: type[Exception],
) -> None:
    payload = _sub_agent_values()
    payload[field_name] = invalid_value

    with pytest.raises(error_type):
        SubAgentConfig.from_dict(payload)


def test_sub_agent_config_accepts_full_u32_range_without_clamping_zero() -> None:
    zero_cycles = SubAgentConfig(model="child-model", description="Research", max_cycles=0)
    max_cycles = SubAgentConfig.from_dict({"model": "child-model", "description": "Research", "max_cycles": (1 << 32) - 1})

    assert zero_cycles.max_cycles == 0
    assert max_cycles.max_cycles == (1 << 32) - 1


def test_agent_task_from_dict_uses_public_dict_defaults_for_sparse_payload() -> None:
    task = AgentTask.from_dict(_agent_task_values())

    assert task.to_dict() == {
        **_agent_task_values(),
        "max_cycles": 8,
        "memory_compact_threshold": 128_000,
        "memory_threshold_percentage": 90,
        "no_tool_policy": "continue",
        "allow_interruption": True,
        "use_workspace": True,
        "has_sub_agents": False,
        "agent_type": None,
        "native_multimodal": False,
        "sub_agents": {},
        "extra_tool_names": [],
        "exclude_tools": [],
        "model_settings": None,
        "initial_messages": [],
        "initial_shared_state": {},
        "metadata": {},
    }


@pytest.mark.parametrize("field_name", ["task_id", "model", "system_prompt", "user_prompt"])
def test_agent_task_from_dict_requires_all_core_fields(field_name: str) -> None:
    payload = _agent_task_values()
    del payload[field_name]

    with pytest.raises(KeyError, match=field_name):
        AgentTask.from_dict(payload)


@pytest.mark.parametrize("field_name", ["task_id", "model", "system_prompt", "user_prompt"])
def test_agent_task_from_dict_requires_core_string_fields(field_name: str) -> None:
    payload = _agent_task_values()
    payload[field_name] = 123

    with pytest.raises(TypeError, match=field_name):
        AgentTask.from_dict(payload)


@pytest.mark.parametrize(("field_name", "invalid_value"), INVALID_AGENT_TASK_VALUES)
def test_agent_task_from_dict_rejects_invalid_optional_field_values(field_name: str, invalid_value: Any) -> None:
    payload = _agent_task_values()
    payload[field_name] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        AgentTask.from_dict(payload)


def test_agent_task_from_dict_accepts_unsigned_wire_boundaries() -> None:
    payload = _agent_task_values()
    payload.update(
        {
            "max_cycles": (1 << 32) - 1,
            "memory_compact_threshold": (1 << 64) - 1,
            "memory_threshold_percentage": 255,
        }
    )

    task = AgentTask.from_dict(payload)

    assert task.max_cycles == (1 << 32) - 1
    assert task.memory_compact_threshold == (1 << 64) - 1
    assert task.memory_threshold_percentage == 255
