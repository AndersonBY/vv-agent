from __future__ import annotations

import json

from vv_agent.app_server.processor import CLIENT_METHODS
from vv_agent.app_server.schema import _schema_bundle, generate_json_schema


def test_generate_json_schema_writes_expected_files(tmp_path) -> None:
    generate_json_schema(tmp_path)

    assert (tmp_path / "json" / "ClientRequest.json").is_file()
    assert (tmp_path / "json" / "ServerNotification.json").is_file()
    assert (tmp_path / "json" / "ServerRequest.json").is_file()
    bundled = tmp_path / "json" / "vv_agent_app_server.schemas.json"
    assert bundled.is_file()

    data = json.loads(bundled.read_text())
    methods = json.dumps(data, sort_keys=True)
    for method in ["initialize", "thread/start", "turn/start", "turn/steer", "approval/request"]:
        assert method in methods


def test_schema_client_methods_match_processor_registry() -> None:
    schema = _schema_bundle()["ClientRequest"]
    schema_methods = set(schema["properties"]["method"]["enum"])

    assert schema_methods == set(CLIENT_METHODS)
