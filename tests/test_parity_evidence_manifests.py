from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import platform
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vv_agent.context_providers import ContextFragment
from vv_agent.prompt import PromptBundle, PromptSection, SystemPromptBuilder, build_system_prompt_bundle
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.tools import ToolExposure, build_default_registry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
CANONICAL_FIXTURES = ("public_api.json", "prompt_bundle.json", "builtin_tools.json")
EXPECTED_DOMAINS = (
    "agent",
    "runner",
    "run_config",
    "result",
    "run_handle",
    "interactive",
    "app_server",
    "tools",
    "workspace",
    "memory",
    "skills",
    "tracing",
    "llm_bridge",
    "runtime_backend",
    "session",
    "exports",
)
EXPECTED_RUNNER_OPERATIONS = (
    "run",
    "start",
    "stream",
    "resume",
    "configured",
)
EXPECTED_RUN_HANDLE_OPERATIONS = (
    "cancel",
    "events",
    "result",
    "state",
    "approve",
    "steer",
    "follow_up",
    "resume",
)
EXPECTED_APP_SERVER_PROTOCOL_OPERATIONS = (
    "initialize",
    "thread/start",
    "thread/resume",
    "thread/read",
    "thread/list",
    "thread/archive",
    "thread/unsubscribe",
    "turn/start",
    "turn/interrupt",
    "turn/resume",
    "turn/steer",
    "turn/followUp",
    "approval/resolve",
    "model/list",
    "schema/export",
    "initialized",
)


PROMPT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "en-US-full",
        "normalizations": ["computer_os"],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": "computer",
            "allow_interruption": True,
            "available_skills": [
                {
                    "allowed_tools": "read_file search_files",
                    "compatibility": "vv-agent >= 1",
                    "description": "Review code against the requested contract.",
                    "location": "skills/review-code/SKILL.md",
                    "name": "review-code",
                }
            ],
            "available_sub_agents": {
                "researcher": "Finds source evidence.",
                "writer": "Writes the final report.",
            },
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": True,
            "language": "en-US",
            "original_system_prompt": "You are a careful coding agent.",
            "session_memory_enabled": True,
            "session_memory_context": "<Session Memory>\n- Keep source evidence.\n</Session Memory>",
            "use_workspace": True,
        },
    },
    {
        "id": "zh-CN-full",
        "normalizations": ["computer_os"],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": "computer",
            "allow_interruption": True,
            "available_skills": [
                {
                    "allowed_tools": "read_file search_files",
                    "compatibility": "vv-agent >= 1",
                    "description": "核对代码与契约。",
                    "location": "skills/review-code/SKILL.md",
                    "name": "review-code",
                }
            ],
            "available_sub_agents": {
                "researcher": "查找源码证据。",
                "writer": "整理最终报告。",
            },
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": True,
            "language": "zh-CN",
            "original_system_prompt": "你是一个严谨的编码 Agent。",
            "session_memory_enabled": True,
            "session_memory_context": "<Session Memory>\n- 保留源码证据。\n</Session Memory>",
            "use_workspace": True,
        },
    },
    {
        "id": "en-US-minimal",
        "normalizations": [],
        "producer": "build_system_prompt_bundle",
        "input": {
            "agent_type": None,
            "allow_interruption": False,
            "available_skills": [],
            "available_sub_agents": {},
            "current_time_utc": "2026-07-10T08:30:00Z",
            "enable_todo_management": False,
            "language": "en-US",
            "original_system_prompt": "Return only verified facts.",
            "session_memory_enabled": False,
            "session_memory_context": "<Session Memory>\n- MUST NOT RENDER.\n</Session Memory>",
            "use_workspace": False,
        },
    },
    {
        "id": "custom-section-metadata",
        "normalizations": [],
        "producer": "SystemPromptBuilder",
        "input": {
            "sections": [
                {
                    "cache_hint": "ephemeral",
                    "id": "policy",
                    "metadata": {"owner": "parity", "priority": 1},
                    "source": "contract://policy",
                    "stable": True,
                    "text": "Use verified evidence.",
                },
                {
                    "id": "runtime",
                    "source": "runtime://request",
                    "stable": False,
                    "text": "request_id=fixture-1",
                },
                {
                    "cache_hint": "ignored-empty",
                    "id": "empty",
                    "stable": True,
                    "text": "   ",
                },
            ]
        },
    },
)


def _resolve_python_target(path: str) -> Any:
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as error:
        if error.name != path:
            raise
    return _resolve_python_export(path)


def _signature_projection(value: Any) -> dict[str, Any]:
    signature = inspect.signature(value)
    projection = {
        "async": inspect.iscoroutinefunction(value),
        "parameters": [
            {
                "kind": parameter.kind.name.lower(),
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
            }
            for parameter in signature.parameters.values()
            if not parameter.name.startswith("_")
        ],
    }
    return projection


def _field_declaration(target: Any, name: str) -> str:
    if is_dataclass(target) and name in {field.name for field in fields(target)}:
        return "dataclass_field"
    if inspect.isclass(target):
        for base in target.__mro__:
            if name in getattr(base, "__annotations__", {}):
                return "annotation"
    raise AssertionError(f"{target!r}.{name} is not a declared public field")


def _normalize_computer_os(value: str) -> str:
    labels = {"Windows", "macOS", "Linux", platform.system() or "Unknown OS"}
    for label in labels:
        value = value.replace(label, "<OS>")
    return value


def _project_prompt_output(bundle: Any, normalizations: list[str]) -> dict[str, Any]:
    sections = list(bundle.sections)
    if "computer_os" in normalizations:
        sections = [
            PromptSection(
                id=section.id,
                text=_normalize_computer_os(section.text),
                stable=section.stable,
                source=section.source,
                cache_hint=section.cache_hint,
                metadata=dict(section.metadata),
            )
            for section in sections
        ]
        bundle = PromptBundle(sections=tuple(sections))
    return {
        "sections": [section.to_dict() for section in bundle.sections],
        "flat_prompt": bundle.flatten(),
        "stable_hash": bundle.stable_hash,
    }


def _render_prompt_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    inputs = scenario["input"]
    producer = scenario["producer"]
    if producer == "build_system_prompt_bundle":
        current_time = datetime.fromisoformat(str(inputs["current_time_utc"]).replace("Z", "+00:00"))
        bundle = build_system_prompt_bundle(
            str(inputs["original_system_prompt"]),
            language=str(inputs["language"]),
            allow_interruption=bool(inputs["allow_interruption"]),
            use_workspace=bool(inputs["use_workspace"]),
            enable_todo_management=bool(inputs["enable_todo_management"]),
            agent_type=inputs["agent_type"],
            available_sub_agents=deepcopy(inputs["available_sub_agents"]),
            available_skills=deepcopy(inputs["available_skills"]),
            current_time_utc=current_time,
            session_memory_enabled=bool(inputs.get("session_memory_enabled", False)),
            session_memory_context=str(inputs["session_memory_context"]),
        )
    elif producer == "SystemPromptBuilder":
        builder = SystemPromptBuilder()
        for raw in inputs["sections"]:
            text = str(raw["text"])
            if not text.strip():
                continue
            section = PromptSection(
                id=str(raw["id"]),
                text=text,
                stable=bool(raw["stable"]),
                source=raw.get("source"),
                cache_hint=raw.get("cache_hint"),
                metadata=deepcopy(raw.get("metadata", {})),
            )
            builder.add_section(section)
        bundle = builder.build_result()
    elif producer == "PromptBundle":
        bundle = PromptBundle(
            sections=tuple(PromptSection.from_dict(raw) for raw in inputs["sections"]),
        )
    elif producer == "AgentCompiler":
        instruction_bundle = PromptBundle(
            sections=tuple(PromptSection.from_dict(raw) for raw in inputs["instruction_bundle"]["sections"])
        )
        compiler_sections = [PromptSection.from_dict(raw) for raw in inputs["compiler_owned_sections"]]
        provider_fragments = [
            ContextFragment(
                id=raw["id"],
                text=raw["text"],
                stable=raw["stable"],
                priority=raw["priority"],
                source=raw.get("source", ""),
                cache_hint=raw.get("cache_hint"),
                metadata=deepcopy(raw.get("metadata", {})),
            )
            for raw in inputs["provider_fragments"]
        ]
        bundle, omitted = AgentCompiler._assemble_prompt_bundle(
            instructions=instruction_bundle,
            compiler_sections=compiler_sections,
            provider_fragments=provider_fragments,
            max_prompt_chars=None,
        )
        assert omitted == []
        return {
            "section_ids": [section.id for section in bundle.sections],
            "flat_prompt": bundle.flatten(),
            "stable_hash": bundle.stable_hash,
        }
    else:
        raise AssertionError(f"unknown prompt producer: {producer}")
    return _project_prompt_output(bundle, list(scenario.get("normalizations", [])))


def _build_prompt_bundle_manifest() -> dict[str, Any]:
    fixture = _load_fixture("prompt_bundle.json")
    scenarios = []
    for source in fixture["scenarios"]:
        scenario = deepcopy(source)
        scenario["output"] = _render_prompt_scenario(scenario)
        scenarios.append(scenario)
    return {key: deepcopy(value) for key, value in fixture.items() if key != "scenarios"} | {
        "scenarios": scenarios,
    }


def _approval_name(needs_approval: Any) -> str:
    if callable(needs_approval):
        return "dynamic"
    return "required" if bool(needs_approval) else "not_required"


def _build_builtin_tools_manifest() -> dict[str, Any]:
    registry = build_default_registry()
    tools = []
    for name in registry.list_tool_names():
        executor = registry.get_executor(name)
        schema = executor.openai_schema(None)
        function = schema["function"]
        assert function["name"] == name
        assert function["description"] == executor.description
        tools.append(
            {
                "approval": _approval_name(executor.needs_approval),
                "description": executor.description,
                "exposure": executor.exposure.value,
                "kind": "function",
                "metadata": deepcopy(executor.metadata),
                "model_visible": executor.exposure == ToolExposure.DIRECT,
                "name": name,
                "parameters": deepcopy(function["parameters"]),
                "strict": executor.strict_json_schema,
                "timeout_seconds": executor.timeout_seconds,
                "type": schema["type"],
            }
        )
    return {
        "contract": "vv-agent-builtin-tools-v4",
        "schema_version": 4,
        "exposure_contract": {
            "allowed_values": ["direct", "hidden"],
            "model_visible_values": ["direct"],
            "host_only_values": ["hidden"],
            "unknown_values": "reject",
        },
        "tools": tools,
    }


def _fixture_payloads() -> dict[str, dict[str, Any]]:
    return {
        "builtin_tools.json": _build_builtin_tools_manifest(),
        "prompt_bundle.json": _build_prompt_bundle_manifest(),
    }


def _load_fixture(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _resolve_python_export(path: str) -> Any:
    if path.startswith("list[") and path.endswith("]"):
        inner = _resolve_python_export(path[5:-1])
        return list[inner]
    parts = path.split(".")
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        attributes = parts[split_index:]
        public_names = getattr(module, "__all__", None)
        if public_names is not None:
            assert attributes[0] in public_names, f"{path} exists but is absent from {module_name}.__all__"
        value: Any = module
        for attribute in attributes:
            value = getattr(value, attribute)
        return value
    raise ModuleNotFoundError(f"No importable module prefix for {path}")


def _verify_fixture_python_member(surface: dict[str, Any], member: dict[str, Any]) -> None:
    python = member["python"]
    target = _resolve_python_target(str(python.get("target", surface["python_target"])))
    name = str(python["name"])
    kind = str(python["kind"])
    if kind in {"method", "function"}:
        value = getattr(target, name)
        assert callable(value), f"{target!r}.{name} is not callable"
        if "signature" in python:
            assert python["signature"] == _signature_projection(value)
        return
    if kind == "property":
        value = inspect.getattr_static(target, name)
        assert isinstance(value, property), f"{target!r}.{name} is not a property"
        assert value.fget is not None
        assert python["declaration"] == "property"
        assert python["signature"] == _signature_projection(value.fget)
        return
    if kind == "field":
        assert python["declaration"] == _field_declaration(target, name)
        return
    raise AssertionError(f"unsupported fixture Python member kind: {kind}")


def test_public_api_manifest_resolves_real_python_exports() -> None:
    fixture = _load_fixture("public_api.json")
    assert tuple(domain["id"] for domain in fixture["domains"]) == EXPECTED_DOMAINS

    capability_ids: set[str] = set()
    for domain in fixture["domains"]:
        assert domain["capabilities"], domain["id"]
        for capability in domain["capabilities"]:
            assert capability["id"] not in capability_ids
            capability_ids.add(capability["id"])
            assert _resolve_python_export(capability["python"]) is not None
    assert len(capability_ids) == 239

    surfaces = {surface["id"]: surface for surface in fixture["surfaces"]}
    assert len(surfaces) == len(fixture["surfaces"])
    assert (
        sum(
            len(surface.get(group, []))
            for surface in fixture["surfaces"]
            for group in ("members", "protocol_operations", "supporting_operations")
        )
        == 281
    )
    assert tuple(member["id"] for member in surfaces["runner"]["members"]) == EXPECTED_RUNNER_OPERATIONS
    assert tuple(member["id"] for member in surfaces["run_handle"]["members"]) == EXPECTED_RUN_HANDLE_OPERATIONS
    assert (
        tuple(member["id"] for member in surfaces["app_server_client"]["protocol_operations"])
        == EXPECTED_APP_SERVER_PROTOCOL_OPERATIONS
    )

    for surface in surfaces.values():
        for group in ("members", "protocol_operations", "supporting_operations"):
            for member in surface.get(group, []):
                _verify_fixture_python_member(surface, member)


def test_prompt_bundle_manifest_uses_real_prompt_producers() -> None:
    fixture = _load_fixture("prompt_bundle.json")
    assert fixture == _build_prompt_bundle_manifest()
    assert {scenario["producer"] for scenario in fixture["scenarios"]} == {
        "AgentCompiler",
        "PromptBundle",
        "SystemPromptBuilder",
        "build_system_prompt_bundle",
    }


def test_prompt_bundle_manifest_enforces_session_memory_gate() -> None:
    fixture = _load_fixture("prompt_bundle.json")
    scenarios = {scenario["id"]: scenario for scenario in fixture["scenarios"]}
    gate = fixture["session_memory_gate"]
    assert gate["control"] == "session_memory_enabled"
    assert gate["default"] is False
    assert gate["enabled_value"] is True
    assert gate["context_presence_does_not_enable_session_memory"] is True

    for probe in gate["probe_cases"]:
        scenario = deepcopy(scenarios[probe["base_scenario"]])
        mutation = probe.get("input_mutation")
        if isinstance(mutation, dict) and "remove" in mutation:
            scenario["input"].pop(str(mutation["remove"]), None)
        output = _render_prompt_scenario(scenario)
        base_output = scenarios[probe["base_scenario"]]["output"]
        assert (
            sum(section["id"] == "session_memory" for section in output["sections"])
            == probe["expected_session_memory_section_count"]
        )
        assert (output["flat_prompt"] == base_output["flat_prompt"]) is probe["expected_prompt_equals_base_output"]


def test_builtin_tools_manifest_uses_real_default_registry() -> None:
    fixture = _load_fixture("builtin_tools.json")
    assert fixture == _build_builtin_tools_manifest()
    assert len(fixture["tools"]) == 15
    assert all(tool["model_visible"] for tool in fixture["tools"])


def test_evidence_json_is_utf8_and_newline_terminated() -> None:
    for name in CANONICAL_FIXTURES:
        path = FIXTURE_DIR / name
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        assert text.endswith("\n"), name
        assert json.loads(text) == _load_fixture(name)


def test_sha256sums_covers_every_parity_fixture() -> None:
    checksum_path = FIXTURE_DIR / "SHA256SUMS"
    entries: dict[str, str] = {}
    listed_names: list[str] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        assert name not in entries
        entries[name] = digest
        listed_names.append(name)

    fixture_names = sorted(path.name for path in FIXTURE_DIR.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    assert listed_names == fixture_names
    for name in fixture_names:
        assert hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest() == entries[name]
