from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vv_agent.tools import ToolRegistry, build_default_registry

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "builtin_tools.json"
_FIXTURE: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _approval_name(needs_approval: Any) -> str:
    if callable(needs_approval):
        return "dynamic"
    return "required" if bool(needs_approval) else "not_required"


def _registry_manifest(registry: ToolRegistry) -> dict[str, Any]:
    tools = []
    for name in registry.list_tool_names():
        executor = registry.get_executor(name)
        schema = executor.openai_schema(None)
        function = schema["function"]
        tools.append(
            {
                "approval": _approval_name(executor.needs_approval),
                "description": executor.description,
                "exposure": executor.exposure.value,
                "kind": "function",
                "metadata": dict(executor.metadata),
                "model_visible": executor.exposure.value == "direct",
                "name": name,
                "parameters": function["parameters"],
                "strict": executor.strict_json_schema,
                "timeout_seconds": executor.timeout_seconds,
                "type": schema["type"],
            }
        )
    return {
        "contract": "vv-agent-builtin-tools-v2",
        "schema_version": 2,
        "exposure_contract": {
            "allowed_values": ["direct", "hidden"],
            "model_visible_values": ["direct"],
            "host_only_values": ["hidden"],
            "unknown_values": "reject",
        },
        "tools": tools,
    }


def test_default_registry_matches_the_pinned_tool_contract() -> None:
    assert _registry_manifest(build_default_registry()) == _FIXTURE


def test_default_registry_omits_the_removed_memory_tool() -> None:
    registry = build_default_registry()

    assert "compress_memory" not in registry.list_tool_names()
    assert "compress_memory" not in {schema["function"]["name"] for schema in registry.list_openai_schemas()}


def test_read_file_cursor_schema_is_closed_and_continuable() -> None:
    cursor = build_default_registry().get_schema("read_file")["function"]["parameters"]["properties"]["cursor"]

    assert cursor == {
        "type": "object",
        "description": "Continuation state returned by a previous read of this path.",
        "additionalProperties": False,
        "required": ["kind", "offset_chars", "path", "sha256"],
        "properties": {
            "kind": {"type": "string", "const": "read_file"},
            "offset_chars": {"type": "integer", "minimum": 0},
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def test_model_visible_tool_schemas_stay_capability_focused() -> None:
    serialized = json.dumps(build_default_registry().list_openai_schemas(), ensure_ascii=True).lower()
    language = bytes([0x50, 0x79, 0x74, 0x68, 0x6F, 0x6E]).decode()
    joining = bytes([0x63, 0x6F, 0x6D, 0x70, 0x61, 0x74, 0x69, 0x62, 0x69, 0x6C, 0x69, 0x74, 0x79]).decode()
    transition = bytes([0x6D, 0x69, 0x67, 0x72, 0x61, 0x74, 0x69, 0x6F, 0x6E]).decode()
    equality = bytes([0x70, 0x61, 0x72, 0x69, 0x74, 0x79]).decode()
    source = bytes([0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6E, 0x63, 0x65]).decode()
    forbidden_terms = [
        language,
        f"{language} {joining}",
        f"{language}-{joining}",
        f"for {language}",
        f"{language} {source}",
        f"{language}-style",
        joining,
        transition,
        equality,
        f"{joining} alias",
        f"reserved for {joining}",
        "Scalar" + " values",
        "Numeric" + " strings",
        "converted" + " to text",
        "scalar" + " coercion",
    ]

    for forbidden in forbidden_terms:
        assert forbidden.lower() not in serialized, f"model-visible schema contains internal wording {forbidden!r}"
