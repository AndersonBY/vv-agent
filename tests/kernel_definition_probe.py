"""Declarative core caller: no SDK compiler, registry, definition builder or defaults."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._definition import DefinitionBindings, DefinitionCache
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._records import copy_json
from vv_agent.kernel._tool_metadata import ToolMetadata


class DeclarationCatalog:
    """Only fixture declarations and catalog reads; production code makes all decisions."""

    def __init__(self, source):
        self.schemas = {s["function"]["name"]: copy_json(s) for s in source["tools"]}
        self.metadata = {n: ToolMetadata.from_dict(v) for n, v in source["capabilities"].items()}
        self.signature = [0]
        self.schema_reads = 0

    def planning_signature(self):
        return (self.signature, [(n, m.to_dict()) for n, m in self.metadata.items()])

    def has_tool(self, name):
        return name in self.schemas

    def has_schema(self, name):
        return name in self.schemas

    def has_executor(self, name):
        return name in self.schemas

    def tool_metadata(self, name):
        return self.metadata.get(name)

    def list_openai_schemas(self, *, tool_names=None):
        self.schema_reads += 1
        return [copy_json(self.schemas[n]) for n in (self.schemas if tool_names is None else tool_names)]


def inputs(source):
    return (
        AgentTask.from_dict(copy_json(source["task"])),
        DefinitionBindings(
            agent_name=source["agent_name"],
            memory_settings=copy_json(source["memory_settings"]),
            child_tools=source["child_tools"][:],
            model_binding=copy_json(source["model_binding"]),
        ),
        DeclarationCatalog(source),
    )


def check_fixtures(root: Path):
    fixture = json.loads((root / "tests/fixtures/parity/run_definition.json").read_text())
    for case in fixture["golden_cases"][:3] + fixture["producer_cases"]:
        task, bindings, catalog = inputs(case["definition"])
        cache = DefinitionCache()
        result = cache.definition(task, bindings, catalog)
        assert result == case["definition"], case["name"]
        raw = canonical_json_bytes(result)
        assert raw == base64.b64decode(case["bytes_base64"]), case["name"]
        assert cache.definition_digest(task, bindings, catalog) == sha256(raw).hexdigest() == case["definition_digest"]
        assert catalog.schema_reads == 1
