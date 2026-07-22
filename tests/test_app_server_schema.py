from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import vv_agent.app_server.schema as schema_module
from vv_agent.app_server import ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.processor import CLIENT_METHODS
from vv_agent.app_server.schema import (
    _schema_bundle,
    generate_json_schema,
    generate_typescript,
    json_schema_bundle,
    typescript_schema_bundle,
)


def _observable_contract() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "parity" / "app_server_observable.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_generate_json_schema_writes_expected_files(tmp_path) -> None:
    generate_json_schema(tmp_path)

    assert (tmp_path / "json" / "ClientRequest.json").is_file()
    assert (tmp_path / "json" / "ServerNotification.json").is_file()
    assert (tmp_path / "json" / "ServerRequest.json").is_file()
    bundled = tmp_path / "json" / "vv_agent_app_server.schemas.json"
    assert bundled.is_file()

    data = json.loads(bundled.read_text())
    methods = json.dumps(data, sort_keys=True)
    for method in [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
        "turn/resume",
        "turn/steer",
        "approval/resolve",
        "schema/export",
        "approval/request",
        "approval/requested",
        "approval/resolved",
        "item/toolCall/delta",
        "error/warning",
    ]:
        assert method in methods
    assert "tool/requestUserInput" not in methods


def test_schema_client_methods_match_processor_registry() -> None:
    schema = _schema_bundle()["ClientRequest"]
    schema_methods = {variant["properties"]["method"]["const"] for variant in schema["oneOf"]}

    assert schema_methods == set(CLIENT_METHODS)


def test_client_request_schema_uses_typed_params_and_bundle_has_typed_results() -> None:
    bundle = _schema_bundle()
    client_request = bundle["ClientRequest"]
    turn_start = next(variant for variant in client_request["oneOf"] if variant["properties"]["method"]["const"] == "turn/start")

    assert turn_start["properties"]["params"] == {"$ref": "#/$defs/TurnStartParams"}
    assert bundle["TurnStartResponse"]["properties"]["turnId"]["type"] == "string"
    assert bundle["SchemaExportResponse"]["required"] == ["jsonSchema", "typescript"]
    approval_request = client_request["$defs"]["ApprovalRequestParams"]
    approval_resolve = client_request["$defs"]["ApprovalResolveParams"]
    assert "choices" not in approval_request["properties"]
    assert approval_resolve["properties"]["decision"]["enum"] == ["allow", "allow_session", "deny", "timeout"]
    assert "message" not in approval_resolve["properties"]
    assert approval_resolve["properties"]["reason"] == {"type": "string"}
    assert approval_resolve["properties"]["metadata"]["type"] == "object"
    assert "reason" not in approval_resolve["required"]
    assert "metadata" not in approval_resolve["required"]
    assert client_request["$defs"]["ServerCapabilities"]["required"] == [
        "modelList",
        "threadLifecycle",
        "notificationOptOut",
        "schemaExport",
        "approvalResolve",
    ]


def test_typescript_generation_is_self_contained(tmp_path) -> None:
    schema_module.generate_typescript(tmp_path)

    generated = tmp_path / "ClientRequest.ts"
    source = generated.read_text(encoding="utf-8")
    assert "export type ClientRequest" in source
    assert "export interface TurnStartParams" in source
    assert 'export type ApprovalDecision = "allow" | "allow_session" | "deny" | "timeout";' in source
    assert "import " not in source


def test_schema_bundle_file_sets_and_sources_match_shared_fixture(tmp_path) -> None:
    contract = _observable_contract()
    schema_contract = contract["schema"]
    assert isinstance(schema_contract, dict)
    expected_json_files = schema_contract["json"]
    expected_typescript_files = schema_contract["typescript"]
    assert isinstance(expected_json_files, list)
    assert isinstance(expected_typescript_files, list)

    json_bundle = json_schema_bundle()
    typescript_bundle = typescript_schema_bundle()
    assert len(json_bundle) == 19
    assert len(typescript_bundle) == 18
    assert [f"{name}.json" for name in json_bundle] == expected_json_files
    assert list(typescript_bundle) == expected_typescript_files

    generate_json_schema(tmp_path)
    generate_typescript(tmp_path / "typescript")
    generated_json_files = sorted(
        path.name for path in (tmp_path / "json").glob("*.json") if path.name != "vv_agent_app_server.schemas.json"
    )
    generated_typescript_files = sorted(path.name for path in (tmp_path / "typescript").glob("*.ts"))
    assert generated_json_files == expected_json_files
    assert generated_typescript_files == expected_typescript_files
    for filename in expected_json_files:
        assert (tmp_path / "json" / filename).read_text(encoding="utf-8") == json_bundle[filename.removesuffix(".json")]

    sources = list(typescript_bundle.values())
    assert len(set(sources)) == 1
    for filename in expected_typescript_files:
        source = (tmp_path / "typescript" / filename).read_text(encoding="utf-8")
        assert source == typescript_bundle[filename]
        assert "export type JsonRpcMessage" in source
        assert "import " not in source
        assert "bigint" not in source

    terminal = _schema_bundle()["ServerNotification"]["$defs"]["TurnCompletedParams"]
    required = set(terminal["required"])
    optional_fields = contract["terminal"]["optionalFieldsOmittedWhenAbsent"]
    assert isinstance(optional_fields, list)
    assert required.isdisjoint(optional_fields)


def test_every_internal_schema_reference_resolves() -> None:
    for name, schema in _schema_bundle().items():
        definitions = schema.get("$defs", {})
        for reference in _internal_refs(schema):
            assert reference.removeprefix("#/$defs/") in definitions, f"{name}: {reference}"


def test_schema_export_request_returns_json_and_typescript_bundles() -> None:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    processor = MessageProcessor(router=router)
    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"clientInfo": {"name": "test"}},
        },
    )
    _ = transport.receive_outbound(timeout=1)

    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 2, "method": "schema/export", "params": {}})

    result = transport.receive_outbound(timeout=1)["result"]
    assert json.loads(result["jsonSchema"]["ClientRequest"])["title"] == "ClientRequest"
    assert "export type ClientRequest" in result["typescript"]["ClientRequest.ts"]
    assert len(result["jsonSchema"]) == 19
    assert len(result["typescript"]) == 18

    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 3, "method": "schema/export", "params": {"unexpected": True}},
    )
    error = transport.receive_outbound(timeout=1)["error"]
    assert error["code"] == -32602


def _internal_refs(value):
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            yield reference
        for nested in value.values():
            yield from _internal_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _internal_refs(nested)
