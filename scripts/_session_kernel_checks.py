"""Check the complete authoring inventory against current producers and Keep files."""

from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from dataclasses import fields
from pathlib import Path
from urllib.parse import unquote

from vv_agent import events, types
from vv_agent.app_server.schema import export_schema_bundles
from vv_agent.interaction import CONTROLLER_COMMAND_ID_SCHEMA, HOST_OUTCOME_SCHEMA, HOST_REQUEST_SCHEMA
from vv_agent.prompt import PromptSection
from vv_agent.runtime.lifecycle import AFTER_CYCLE_CONTROL_SCHEMA
from vv_agent.session.records import INPUT_SCHEMA, RECORD_SCHEMA

DELETE = frozenset(
    (
        "checkpoint_codec.json",
        "checkpoint_config.json",
        "checkpoint_resume.json",
        "checkpoint_sqlite_canonical.sql",
        "checkpoint_store.json",
        "controller_command.json",
        "deferred_tool.json",
        "distributed_run_driver.json",
        "distributed_run_envelope.json",
        "distributed_worker_response.json",
        "operation_journal.json",
        "resume_events.jsonl",
        "session_sqlite_canonical.sql",
    )
)
SCHEMA_TOKEN = re.compile(r"vv-agent(?:\.[A-Za-z0-9_-]+)+\.v[0-9]+|vv-agent-public-api-v[0-9]+")
REFERENCE = re.compile(r"(?:fixtures/([A-Za-z0-9_./-]+\.(?:jsonl?|sql))(?:#([^\s\"']*))?|([A-Za-z0-9_.-]+\.jsonl?)#([^\s\"']*))")


def walk(value, path=""):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + "/" + str(index))


def resolve_pointer(value, fragment):
    pointer = unquote(fragment)
    if not pointer:
        return value
    assert pointer.startswith("/"), f"not a JSON pointer: {fragment}"
    for part in pointer[1:].split("/"):
        assert not re.search(r"~(?![01])", part), f"invalid pointer escape: {fragment}"
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            assert re.fullmatch(r"0|[1-9][0-9]*", key), f"invalid array index: {fragment}"
            value = value[int(key)]
        else:
            value = value[key]
    return value


def literal_fields(decoder, variable):
    """The strict decoder's field set is authoritative, rather than a fixture copy."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(decoder)))
    return next(
        set(ast.literal_eval(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)
    )


def validate_outputs(values, base: Path, keep, replacements):
    kept = {name: json.loads((base / name).read_text()) for name in keep}
    schemas = {name: json.loads(source) for name, source in export_schema_bundles()["jsonSchema"].items()}
    baseline = {name: json.loads((base / name).read_text()) for name in replacements if name.endswith(".json")}
    # Inventory versions belong to the author; wire versions come from live producer constants/schemas.
    inventory_versions = {name: value.get("version") for name, value in baseline.items() if "version" in value}
    inventory_versions.update(
        {
            "run_definition.json": 6,
            "result_public.json": 7,
            "app_server_observable.json": 4,
            "runner_terminal.json": "v2",
            "runner_trace_spans.json": "v2",
            "run_handle.json": "v2",
        }
    )
    current = {HOST_REQUEST_SCHEMA, HOST_OUTCOME_SCHEMA, CONTROLLER_COMMAND_ID_SCHEMA, AFTER_CYCLE_CONTROL_SCHEMA}
    current.update(
        value["schema_version"]
        for name, value in baseline.items()
        if isinstance(value.get("schema_version"), str) and name != "run_definition.json"
    )
    current.add("vv-agent-public-api-v8")
    current.add(baseline["memory_local.json"]["microcompact"]["schema_version"])
    for schema in schemas.values():
        current.update(
            node["const"]
            for _, node in walk(schema)
            if isinstance(node, dict) and isinstance(node.get("const"), str) and SCHEMA_TOKEN.fullmatch(node["const"])
        )
    current.update({types.TOKEN_USAGE_SCHEMA_VERSION, types.MODEL_CALL_SCHEMA_VERSION, types.TASK_TOKEN_USAGE_SCHEMA_VERSION})
    event_version = events.RUN_EVENT_VERSION
    protocol_version = schemas["InitializeResponse"]["properties"]["protocolVersion"]["const"]
    obsolete = set()
    for name in (*replacements, *DELETE):
        if (base / name).exists():
            obsolete.update(SCHEMA_TOKEN.findall((base / name).read_text()))
    obsolete -= current
    terminal = schemas["ServerNotification"]["$defs"]["TurnCompletedParams"]
    assert terminal["additionalProperties"] is False
    terminal_optional = set(terminal["properties"]) - set(terminal["required"])
    owners = {
        ("app_server_observable.json", "/terminal"): terminal_optional,
        ("session_codec.json", "/message_contract"): types._MESSAGE_FIELDS - {"role", "content"},
        ("result_public.json", "/agent_result_wire/message_wire"): types._MESSAGE_FIELDS - {"role", "content"},
        ("bounded_tool_result.json", "/result_contract"): literal_fields(types.ToolExecutionResult.from_dict, "allowed")
        - {"tool_call_id", "content", "status_code", "directive"},
        ("bounded_tool_result.json", "/result_contract/canonical_writer_normalization"): literal_fields(
            types.ToolExecutionResult.from_dict, "allowed"
        )
        - {"tool_call_id", "content", "status_code", "directive"},
        ("prompt_bundle.json", "/section_contract"): {f.name for f in fields(PromptSection)} - {"id", "kind", "text", "stable"},
        ("result_public.json", "/agent_result_wire"): literal_fields(types.AgentResult.from_dict, "optional_fields"),
    }
    references, optional_lists, rejected_versions = 0, 0, []
    for name, value in values.items():
        for path, node in walk(value):
            location = name + "#" + path
            assert not any(retired in path for retired in DELETE), f"{location}: deleted fixture"
            if isinstance(node, str):
                assert not any(retired in node for retired in DELETE), f"{location}: deleted fixture"
                stale = set(SCHEMA_TOKEN.findall(node)) & obsolete
                assert not stale, f"{location}: superseded discriminator {sorted(stale)}"
                unknown = set(SCHEMA_TOKEN.findall(node)) - current
                assert not unknown, f"{location}: unknown discriminator {sorted(unknown)}"
                for match in REFERENCE.finditer(node):
                    target, fragment = (match[1], match[2]) if match[1] else (match[3], match[4])
                    assert target in values or target in kept, f"{location}: missing fixture {target}"
                    if fragment is not None:
                        try:
                            resolve_pointer(values[target] if target in values else kept[target], fragment)
                        except (AssertionError, KeyError, IndexError, TypeError) as exc:
                            raise AssertionError(f"{location}: unresolved {target}#{fragment}") from exc
                    references += 1
                # Schema/export retains JSON and TypeScript strings, so check their versions too.
                for version in re.findall(r'protocolVersion\s*[:=]\s*["\'](v[0-9]+)', node):
                    assert version == protocol_version, f"{location}: protocol {version}"
                for version in re.findall(r"RunEvent\s+(v[0-9]+)", node):
                    assert version == event_version, f"{location}: RunEvent {version}"
                if node.startswith("{"):
                    try:
                        nested = json.loads(node)
                    except ValueError:
                        continue
                    for nested_path, item in walk(nested):
                        if isinstance(item, dict) and "protocolVersion" in item.get("properties", {}):
                            assert item["properties"]["protocolVersion"]["const"] == protocol_version, location + nested_path
            if not isinstance(node, dict):
                continue
            for key, item in node.items():
                normalized = key.replace("_", "").lower()
                if normalized in {
                    "schemaversion",
                    "sourceschemaversion",
                    "version",
                    "wireversion",
                    "protocolversion",
                } and isinstance(item, (str, int)):
                    expected = None
                    if "/invalid" in path:
                        rejected_versions.append({"location": location + "/" + key, "value": item})
                        continue
                    if normalized in {"schemaversion", "sourceschemaversion"}:
                        expected = (
                            current
                            if isinstance(item, str)
                            else {
                                8
                                if name == "public_api.json" and not path
                                else RECORD_SCHEMA["properties"]["schema_version"]["const"]
                            }
                        )
                    elif normalized == "protocolversion":
                        expected = {protocol_version}
                    elif normalized == "wireversion" or "event_id" in node:
                        expected = {event_version}
                    elif not path and name in inventory_versions:
                        expected = {inventory_versions[name]}
                    elif name == "app_server_observable.json" and path == "/jsonRpc":
                        expected = {"2.0"}
                    assert expected is not None and item in expected, f"{location}/{key}: unexpected discriminator {item}"
                if re.search(r"optional|omit.*absent|omit.*optional", key, re.I) and isinstance(item, list):
                    owner = owners.get((name, path))
                    if key == "optional_source_fields":
                        wire_type = node["wire_type"]
                        owner = events._EVENT_FIELDS[wire_type] - events._EVENT_REQUIRED_FIELDS[wire_type]
                    assert owner is not None, f"{location}/{key}: missing closed-schema owner"
                    assert set(item) <= owner, f"{location}/{key}: unknown optional fields {set(item) - owner}"
                    if (name, path) == ("app_server_observable.json", "/terminal"):
                        assert set(item) == terminal_optional, f"{location}: incomplete terminal optional fields"
                    optional_lists += 1
                if key.endswith("_omitted_when_absent") and key != "optional_fields_omitted_when_absent":
                    field = key.removesuffix("_omitted_when_absent")
                    assert field in owners.get((name, path), ()), f"{location}/{key}: unknown omitted field {field}"
    assert INPUT_SCHEMA["properties"]["schema_version"]["const"] == RECORD_SCHEMA["properties"]["schema_version"]["const"]
    return {
        "references": references,
        "optional_lists": optional_lists,
        "rejected_versions": rejected_versions,
        "keep_superseded_discriminators": {
            name: sorted(set(SCHEMA_TOKEN.findall(json.dumps(value))) & obsolete)
            for name, value in kept.items()
            if set(SCHEMA_TOKEN.findall(json.dumps(value))) & obsolete
        },
    }
