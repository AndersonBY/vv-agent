from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.backends.distributed import toolset_schema_digest
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools import ToolContext
from vv_agent.tools.executor import RegistryToolExecutor
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, ToolExecutionResult

_EXTRA_TOOL_NAMES = ("_tool_alpha", "_tool_beta", "_tool_gamma", "_tool_delta")
_SUBPROCESS_SCRIPT = """
import json

from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.backends.distributed import toolset_schema_digest
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools.executor import RegistryToolExecutor
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, ToolExecutionResult


def handler(_context, _arguments):
    return ToolExecutionResult(tool_call_id="", content="ok")


registry = ToolRegistry()
for name in ("_tool_alpha", "_tool_beta", "_tool_gamma", "_tool_delta"):
    registry.register_executor(RegistryToolExecutor(name=name, handler=handler))

task = AgentTask(
    task_id="deterministic-tools",
    model="m",
    prompt_bundle=build_raw_system_prompt_bundle("sys"),
    user_prompt="u",
    allow_interruption=False,
    use_workspace=False,
    extra_tool_names=registry.list_planner_extra_tool_names(),
)
serialized = json.loads(json.dumps(task.to_dict(), ensure_ascii=True, sort_keys=True))
restored = AgentTask.from_dict(serialized)
print(
    json.dumps(
        {
            "planner_extra_tool_names": registry.list_planner_extra_tool_names(),
            "task_extra_tool_names": restored.extra_tool_names,
            "schemas": plan_tool_schemas(registry=registry, task=restored, include_dynamic_hints=False),
            "digest": toolset_schema_digest(registry, task=restored),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


def _handler(_context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(tool_call_id="", content="ok")


def _executor(name: str) -> RegistryToolExecutor:
    return RegistryToolExecutor(name=name, handler=_handler)


def _registry(names: tuple[str, ...] = _EXTRA_TOOL_NAMES) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register_executor(_executor(name))
    return registry


def _task(*, extra_tool_names: list[str]) -> AgentTask:
    return AgentTask(
        task_id="deterministic-tools",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="u",
        allow_interruption=False,
        use_workspace=False,
        extra_tool_names=extra_tool_names,
    )


def test_planner_extra_names_preserve_registration_order() -> None:
    registry = _registry()

    assert registry.list_planner_extra_tool_names() == list(_EXTRA_TOOL_NAMES)


def test_register_tool_preserves_planner_extra_registration_order() -> None:
    registry = ToolRegistry()

    registry.register_tool("_tool_alpha", _handler, "alpha")
    registry.register_tool("_tool_beta", _handler, "beta")

    assert registry.list_planner_extra_tool_names() == ["_tool_alpha", "_tool_beta"]


def test_repeated_planner_extra_marking_is_idempotent() -> None:
    registry = ToolRegistry()

    registry._add_planner_extra_tool_name("_tool_alpha")
    registry._add_planner_extra_tool_name("_tool_beta")
    registry._add_planner_extra_tool_name("_tool_alpha")

    assert registry.list_planner_extra_tool_names() == ["_tool_alpha", "_tool_beta"]


def test_duplicate_executor_registration_does_not_duplicate_planner_extra() -> None:
    registry = _registry(("_tool_alpha",))

    with pytest.raises(ValueError, match="Tool already registered: _tool_alpha"):
        registry.register_executor(_executor("_tool_alpha"))

    assert registry.list_planner_extra_tool_names() == ["_tool_alpha"]


def test_task_serialization_preserves_tool_definitions_and_digest() -> None:
    registry = _registry()
    task = _task(extra_tool_names=registry.list_planner_extra_tool_names())
    before_schemas = plan_tool_schemas(registry=registry, task=task, include_dynamic_hints=False)
    before_digest = toolset_schema_digest(registry, task=task)

    serialized = json.loads(json.dumps(task.to_dict(), ensure_ascii=True, sort_keys=True))
    restored = AgentTask.from_dict(serialized)

    assert restored.extra_tool_names == list(_EXTRA_TOOL_NAMES)
    assert plan_tool_schemas(registry=registry, task=restored, include_dynamic_hints=False) == before_schemas
    assert toolset_schema_digest(registry, task=restored) == before_digest


def test_planner_tool_definitions_and_digest_are_stable_across_hash_seeds() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    outputs: list[dict[str, Any]] = []
    for seed in (1, 2, 3, 4):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        source_path = str(repository_root / "src")
        environment["PYTHONPATH"] = os.pathsep.join(path for path in (source_path, environment.get("PYTHONPATH")) if path)
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT],
            cwd=repository_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs
    assert all(output == outputs[0] for output in outputs[1:])
    assert outputs[0]["planner_extra_tool_names"] == list(_EXTRA_TOOL_NAMES)
    assert outputs[0]["task_extra_tool_names"] == list(_EXTRA_TOOL_NAMES)
    assert [schema["function"]["name"] for schema in outputs[0]["schemas"]] == list(_EXTRA_TOOL_NAMES)
