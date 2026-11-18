"""The SDK and a stdlib-only caller execute one definition/planning algorithm."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from kernel_definition_probe import inputs
from test_run_definition_producer import FIXTURE, produce

from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _definition as owner
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_stored_record
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, ContentSessionWriter, SourceKind
from vv_agent.kernel._content_publication import make_record_before_publication, publish_record_content
from vv_agent.kernel._definition import DefinitionCache, model_identity
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._records import copy_json, digest
from vv_agent.kernel._store import StoredRecord
from vv_agent.kernel._tool_metadata import ToolResultRetention

ROOT = Path(__file__).resolve().parents[1]


class _MemoryContentPort(ContentSessionWriter):
    tenant_id = "tenant"
    session_id = "s"

    def __init__(self) -> None:
        self._bodies: dict[ContentRef, bytes] = {}
        self._next_id = 0

    def put_json(self, value: Any, *, labels: Any = ...) -> ContentRef:
        self._next_id += 1
        body = canonical_json_bytes(value)
        ref = ContentRef(
            content_id=f"content/retained/{self._next_id}",
            mac=digest(value),
            size=len(body),
            key_id="key/session",
            labels=ContentLabels((SourceKind.INTERNAL,), Classification.INTERNAL),
        )
        self._bodies[ref] = body
        return ref

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        assert (tenant_id, session_id, owner) == ("tenant", "s", "record:1")
        return self._bodies[ref]


def retained(value):
    contents = _MemoryContentPort()
    source = make_record_before_publication(
        "turn_started",
        session_id="s",
        turn_id="t",
        payload={
            "input_ids": [],
            "definition": value,
            "handler_version": "1",
            "budget": RunBudgetLimits().to_dict(),
            "binding": "task",
            "generation": 1,
        },
    )
    published = publish_record_content(contents, (source,)).records[0]
    published.encode()
    stored = StoredRecord(published, 1, "commit/1", 1, 1)
    hydrated = hydrate_session_stored_record(
        stored,
        SessionHydration(contents, tenant_id="tenant", session_id="s"),
    )
    assert hydrated.record._task_value is not None
    return hydrated.record


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_definition_producer_without_sdk(tmp_path, site_flags):
    script = """
import importlib.abc
import runpy
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'}
                or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
probe = runpy.run_path(str(root / 'tests/kernel_definition_probe.py'))
probe['check_fixtures'](root)
assert all(n in {'vv_agent', 'vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", FIXTURE["golden_cases"][:3] + FIXTURE["producer_cases"], ids=lambda c: c["name"])
def test_sdk_and_core_produce_the_same_frozen_bytes(case):
    task, bindings, catalog = inputs(case["definition"])
    cache = DefinitionCache()
    value, identity = produce(case["definition"])
    assert cache.definition(task, bindings, catalog) == value
    assert cache.definition_digest(task, bindings, catalog) == identity


def test_retained_fast_path_rechecks_bindings_without_schema_or_task_rebuild(monkeypatch):
    task, bindings, catalog = inputs(FIXTURE["golden_cases"][0]["definition"])
    cache = DefinitionCache()
    frozen = retained(cache.definition(task, bindings, catalog))
    identity = cache.definition_digest(frozen, bindings, catalog)
    assert catalog.schema_reads == 1
    original = owner._fingerprint

    def fingerprint(value):
        assert value is not frozen._payload["definition"]["task"], "retained task serialized again"
        return original(value)

    def unexpected(*args, **kwargs):
        raise AssertionError("retained task must not be serialized")

    monkeypatch.setattr(owner, "_fingerprint", fingerprint)
    monkeypatch.setattr(AgentTask, "to_dict", unexpected)
    assert cache.definition_digest(frozen, bindings, catalog) == identity
    changed = replace(bindings, child_tools=["new-child"])
    assert cache.definition_digest(frozen, changed, catalog) != identity
    assert catalog.schema_reads == 1


def test_identity_and_cached_copies_distinguish_boolean_integer_and_host_mutation():
    task, bindings, catalog = inputs(FIXTURE["golden_cases"][0]["definition"])
    cache = DefinitionCache()
    task.metadata["typed"] = True
    value = cache.definition(task, bindings, catalog)
    before = cache.definition_digest(task, bindings, catalog)
    value["task"]["metadata"]["typed"] = "caller mutation"
    bindings.model_binding["internal"] = {"compaction": {"backend": "scripted", "model": "summary"}}
    current = cache._definition(task, bindings, catalog)
    bindings.model_binding["internal"]["compaction"]["model"] = "changed"
    assert current["model_binding"]["internal"]["compaction"]["model"] == "summary"
    bindings.model_binding.pop("internal")
    assert cache.definition_digest(task, bindings, catalog) == before
    task.metadata["typed"] = 1
    assert cache.definition_digest(task, bindings, catalog) != before
    assert cache.definition(task, bindings, catalog)["task"]["metadata"]["typed"] == 1


def test_mutable_catalog_signature_invalidates_selected_schemas():
    source = next(c["definition"] for c in FIXTURE["producer_cases"] if c["definition"]["tools"])
    task, bindings, catalog = inputs(source)
    cache = DefinitionCache()
    before = cache.definition_digest(task, bindings, catalog)
    name = source["tools"][0]["function"]["name"]
    catalog.schemas[name]["function"]["description"] = "changed declaration"
    catalog.signature[0] += 1
    after = cache.definition(task, bindings, catalog)
    assert after["tools"][0]["function"]["description"] == "changed declaration"
    assert cache.definition_digest(task, bindings, catalog) != before
    assert catalog.schema_reads == 2


def test_retention_preserve_wins_and_input_settings_remain_unmodified():
    source = next(c["definition"] for c in FIXTURE["producer_cases"] if c["definition"]["capabilities"])
    task, bindings, catalog = inputs(source)
    name = next(iter(catalog.metadata))
    for host, declaration in [("preserve", "archive"), ("archive", "preserve")]:
        bindings.memory_settings["tool_result_retentions"][name] = host
        original = copy_json(bindings.memory_settings)
        catalog.metadata[name].result_retention = ToolResultRetention(declaration)
        value = DefinitionCache().definition(task, bindings, catalog)
        assert value["memory_settings"]["tool_result_retentions"][name] == "preserve"
        assert bindings.memory_settings == original


def test_planner_facade_and_workspace_names_share_the_core_objects():
    from vv_agent.constants import WORKSPACE_TOOLS
    from vv_agent.kernel._tool_names import WORKSPACE_TOOLS as core_names
    from vv_agent.kernel._tool_planner import plan_tool_names
    from vv_agent.runtime.tool_planner import plan_tool_names as facade
    from vv_agent.session.runtime import _model_identity

    assert facade is plan_tool_names
    assert _model_identity is model_identity
    assert type(core_names) is list
    assert core_names is WORKSPACE_TOOLS
