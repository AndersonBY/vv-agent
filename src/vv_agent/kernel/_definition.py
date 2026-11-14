"""Frozen run definitions and their single canonical planning/identity cache."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._records import Record, copy_json
from vv_agent.kernel._tool_metadata import ToolResultRetention
from vv_agent.kernel._tool_planner import ToolCatalog, plan_tool_names, plan_tool_schemas


def _fingerprint(value: Any) -> str | bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return canonical_json_bytes(value)


def model_identity(binding: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in binding.items() if key != "endpoints"}


@dataclass(frozen=True)
class DefinitionBindings:
    """Already resolved facts; memory defaults and model routes belong to the host."""

    agent_name: str
    memory_settings: dict[str, Any]
    child_tools: Sequence[str]
    model_binding: dict[str, Any]


class DefinitionCache:
    def __init__(self) -> None:
        self._definition_key: tuple[str | bytes, str | bytes] | None = None
        self._definition_value: dict[str, Any] = {}
        self._definition_digest = ""
        self._retained_task_key: tuple[Record, str | bytes] | None = None
        self._definition_binding_key: tuple | None = None
        self._definition_binding: dict[str, Any] = {}
        self._definition_prefix = b""
        self._definition_suffix = b""
        self._schema_key: str | bytes | None = None
        self._schemas: list[dict[str, Any]] = []

    def definition(self, task: AgentTask | Record, bindings: DefinitionBindings, catalog: ToolCatalog) -> dict[str, Any]:
        return copy_json(self._definition(task, bindings, catalog))

    def definition_digest(self, task: AgentTask | Record, bindings: DefinitionBindings, catalog: ToolCatalog) -> str:
        current = self._definition(task, bindings, catalog)
        if isinstance(task, Record) and "endpoints" in task._payload["definition"]["model_binding"]:
            # Reproduce stored bytes without treating their historical routes as identity.
            current = copy_json(current)
            retained = task._payload["definition"]["model_binding"]
            for binding, old in [
                (current["model_binding"], retained),
                *[
                    (value, retained.get("internal", {}).get(purpose, {}))
                    for purpose, value in current["model_binding"].get("internal", {}).items()
                ],
            ]:
                if "endpoints" in old:
                    binding["endpoints"] = old["endpoints"]
                    for limit in ("context_length", "max_output_tokens"):
                        if limit not in old:
                            binding.pop(limit, None)
            return sha256(canonical_json_bytes(current)).hexdigest()
        return self._definition_digest

    def _definition(self, task: AgentTask | Record, bindings: DefinitionBindings, catalog: ToolCatalog) -> dict[str, Any]:
        # Retained tasks are immutable; mutable runtime bindings are checked on every step.
        retained = task if isinstance(task, Record) else None
        if retained is not None:
            task_value = retained._payload["definition"]["task"]
            task = retained._task()
        else:
            assert isinstance(task, AgentTask)
            task_value = task.to_dict()
        if retained is not None and self._retained_task_key is not None and self._retained_task_key[0] is retained:
            task_key = self._retained_task_key[1]
        else:
            task_key = _fingerprint(task_value)
            if retained is not None:
                self._retained_task_key = (retained, task_key)
        memory_settings = bindings.memory_settings
        model_binding = model_identity(bindings.model_binding)
        value: dict[str, Any] = {
            "agent_name": bindings.agent_name,
            "memory_settings": memory_settings,
            "child_tools": sorted(bindings.child_tools),
            "model_binding": model_binding,
        }
        signature = catalog.planning_signature()
        # JSON fingerprints distinguish True from 1 and detach mutable cache inputs.
        binding_key = _fingerprint([value, signature])
        key = (task_key, binding_key)
        if key == self._definition_key:
            return self._definition_value
        schema_key = _fingerprint(
            [
                plan_tool_names(task),
                signature,
                [
                    task.metadata.get(control, [])
                    for control in (
                        "_vv_agent_denied_side_effects",
                        "_vv_agent_denied_capability_tags",
                        "_vv_agent_denied_cost_dimensions",
                    )
                ],
                task.metadata.get("_vv_agent_deny_terminal_tools", False),
            ]
        )
        if schema_key != self._schema_key:
            self._schemas = plan_tool_schemas(registry=catalog, task=task)
            self._schema_key = schema_key
        if (binding_key, schema_key) != self._definition_binding_key:
            memory_settings = copy_json(memory_settings)
            value["memory_settings"] = memory_settings
            schemas = self._schemas
            names = [s["function"]["name"] for s in schemas]
            retentions = dict(memory_settings["tool_result_retentions"])
            for name in names:
                metadata = catalog.tool_metadata(name)
                if metadata is not None:
                    retentions[name] = (
                        ToolResultRetention.PRESERVE
                        if ToolResultRetention.PRESERVE in (retentions.get(name), metadata.result_retention)
                        else metadata.result_retention
                    )
            memory_settings["tool_result_retentions"] = retentions
            value["capabilities"] = {
                n: metadata.to_dict()
                for n in names
                if catalog.has_executor(n) and (metadata := catalog.tool_metadata(n)) is not None
            }
            # These closed keys precede task, and tools follows it in JCS order.
            prefix = canonical_json_bytes(value)[:-1] + b',"task":'
            suffix = b',"tools":' + canonical_json_bytes(schemas) + b"}"
            self._definition_binding = copy_json(value) | {"tools": schemas}
            self._definition_prefix, self._definition_suffix = prefix, suffix
            self._definition_binding_key = (binding_key, schema_key)
        task_bytes = canonical_json_bytes(task_value)
        value = self._definition_binding | {"task": copy_json(task_value)}
        self._definition_digest = sha256(self._definition_prefix + task_bytes + self._definition_suffix).hexdigest()
        self._definition_value = value
        self._definition_key = key
        return value
