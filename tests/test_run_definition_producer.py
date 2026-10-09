"""Current run definitions produced by the sole runtime assembly."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent import Agent, AgentTask, RunConfig, ScriptedModelProvider
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.memory.manager import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolMetadata
from vv_agent.tools.outputs import ToolOutputText
from vv_agent.tools.registry import ToolRegistry

FIXTURE = json.loads((Path(__file__).parent / "fixtures/parity/run_definition.json").read_text())


def produce(source):
    task = AgentTask.from_dict(deepcopy(source["task"]))
    tools = []
    for entry in source["tools"]:
        schema = entry["function"]
        tools.append(
            FunctionTool(
                schema["name"],
                schema["description"],
                schema["parameters"],
                lambda _ctx, _args: ToolOutputText("ok"),
                tool_metadata=ToolMetadata.from_dict(source["capabilities"][schema["name"]])
                if schema["name"] in source["capabilities"]
                else None,
            )
        )
    binding = source["model_binding"]
    agent = Agent(source["agent_name"], task.prompt_bundle, model=binding["model"], tools=tools)
    memory = deepcopy(source["memory_settings"])
    memory["microcompaction_policy"] = MicrocompactionPolicy.from_dict(memory["microcompaction_policy"])
    driver = SessionDriver()
    try:
        runtime = driver.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new(binding["backend"], binding["model"], []),
                workspace="/fixture",
                tool_registry_factory=ToolRegistry,
            ),
        )
        runtime.memory_manager = MemoryManager(**memory)
        value = runtime.definition(task)
        return value, runtime.definition_digest(task)
    finally:
        driver.close()


@pytest.mark.parametrize("case", FIXTURE["golden_cases"][:3] + FIXTURE["producer_cases"], ids=lambda case: case["name"])
def test_current_definition_real_producer(case):
    definition, digest = produce(case["definition"])
    assert definition == case["definition"]
    raw = canonical_json_bytes(definition)
    assert raw == base64.b64decode(case["bytes_base64"])
    assert sha256(raw).hexdigest() == digest == case["definition_digest"]


def test_definition_detaches_host_mutation():
    source = FIXTURE["golden_cases"][0]["definition"]
    first, digest = produce(source)
    first["task"]["metadata"]["host_mutation"] = True
    second, second_digest = produce(source)
    assert second == source
    assert second_digest == digest
