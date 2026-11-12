"""Record values and private fragments have one stdlib-only implementation."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import vv_agent
from vv_agent.kernel import _records as core
from vv_agent.kernel._execution_values import AgentTask, SubAgentConfig
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._model_settings import ModelSettings
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
from vv_agent.session import records


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_records_import_and_task_decode_are_core_only(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {"vv_agent", "vv_agent.kernel"}
                or fullname.startswith("vv_agent.kernel.")
                or fullname.split(".")[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel import _records as r
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
spec = r.SessionSpec("s", "p", "w").record()
assert r.Record.parse(spec.encode()).digest == spec.digest
assert r.InboxItem.parse(r.InboxItem("i", "user", {"content": "go"}).encode()).input_id == "i"
task = AgentTask("t", "m", PromptBundle((PromptSection("system", "Answer."),)), "go")
definition = {"task": task.to_dict()}
record = r.make_record("turn_started", session_id="s", turn_id="t", payload={
    "input_ids": [], "definition": definition, "definition_digest": r.digest(definition),
    "handler_version": "1", "budget": {}, "binding": "task", "generation": 1,
})
assert type(record._task()) is AgentTask
assert record._task() is record._task()
assert record.task().to_dict() == task.to_dict()
assert all(name in {"vv_agent", "vv_agent.kernel"} or name.startswith("vv_agent.kernel.")
           for name in sys.modules if name.startswith("vv_agent"))
"""
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_exports_are_the_actual_core_objects():
    for name in ("Record", "InboxItem", "SessionSpec"):
        assert getattr(vv_agent, name) is getattr(records, name) is getattr(core, name)
        assert getattr(core, name).__module__ == "vv_agent.kernel._records"
    for name in ("_MessageCache", "_RetainedTools", "copy_json", "digest", "record_identity", "make_record"):
        assert getattr(records, name) is getattr(core, name)


def test_private_keys_keep_numeric_null_and_string_types_distinct():
    cache = core._MessageCache()
    values = [None, float("nan"), float("inf"), True, 1, 1.0, "1", 0, -0.0, False]
    wires = [{"role": "user", "content": "same", "metadata": {"value": value}} for value in values]
    shared = [cache.share(wire) for wire in wires]
    assert len({id(wire) for wire in shared}) == len(wires)
    assert math.isnan(shared[1]["metadata"]["value"])
    for wire, retained in zip(wires, shared, strict=True):
        assert cache.share(dict(reversed(list(wire.items())))) is retained


@pytest.mark.parametrize(
    "field,valid,invalid",
    [("tool_calls", [], ()), ("metadata", {"1": "x"}, {1: "x"})],
)
def test_private_keys_do_not_coerce_containers_or_keys(field, valid, invalid):
    cache = core._MessageCache()
    good = {"role": "assistant", "content": "same", field: valid}
    bad = good | {field: invalid}
    assert cache.share(good) is good
    retained = cache.share(bad)
    assert retained is bad
    assert retained[field] is invalid


def test_private_keys_defer_message_argument_validation():
    cache = core._MessageCache()
    wire = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call", "type": "function", "function": {"name": "read", "arguments": '{"x":"\\ud800"}'}}],
    }
    shared = cache.share(wire)
    assert shared is wire
    assert json.loads(shared["tool_calls"][0]["function"]["arguments"]) == {"x": "\ud800"}
    message = cache.parse(shared)
    with pytest.raises(ValueError, match="I-JSON"):
        cache.source_digest([message])


def test_record_freezes_but_inbox_reencodes_mutable_payload():
    source = {"nested": [1]}
    created = core.SessionSpec("s", "p", "w").record()
    record = replace(created, payload=created.payload | {"attributes": source})
    source["nested"].append(2)
    body = record.encode()
    source["nested"].append(3)
    record.payload["attributes"]["nested"].clear()
    assert record.encode() is body
    assert record.payload["attributes"] == {"nested": [1, 2]}
    assert deepcopy(record) is record
    item = core.InboxItem("i", "user", {"content": {"nested": [1]}})
    first = item.encode()
    item.payload["content"]["nested"].append(2)
    assert item.encode() != first
    assert core.InboxItem.parse(item.encode()).payload == item.payload


def test_cached_core_task_remains_detached_at_every_mutable_boundary(monkeypatch):
    task = AgentTask(
        "t",
        "m",
        PromptBundle((PromptSection("system", "Answer."),)),
        "go",
        metadata={"nested": [1]},
        initial_shared_state={"nested": [1]},
        initial_messages=[Message("user", "before", metadata={"nested": [1]})],
        sub_agents={"child": SubAgentConfig("m", "child", metadata={"nested": [1]})},
        model_settings=ModelSettings(extra_body={"nested": [1]}),
        extra_tool_names=["tool"],
        exclude_tools=["excluded"],
    )
    definition = {"task": task.to_dict()}
    record = core.make_record(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload={
            "input_ids": [],
            "definition": definition,
            "definition_digest": core.digest(definition),
            "handler_version": "1",
            "budget": {},
            "binding": "task",
            "generation": 1,
        },
    )
    cached = record._task()
    assert type(cached) is AgentTask and record._task() is cached
    expected = record.task().to_dict()
    detached = record.task()
    detached.metadata["nested"].clear()
    detached.initial_shared_state["nested"].clear()
    detached.initial_messages[0].metadata["nested"].clear()
    detached.sub_agents["child"].metadata["nested"].clear()
    assert detached.model_settings is not None and detached.model_settings.extra_body is not None
    detached.model_settings.extra_body["nested"].clear()
    detached.extra_tool_names.clear()
    detached.exclude_tools.clear()

    def unexpected(*args, **kwargs):
        raise AssertionError("cached core task must not be decoded again")

    monkeypatch.setattr(AgentTask, "from_dict", unexpected)
    assert record.task().to_dict() == expected
