from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_json_schema(out_dir: str | Path) -> None:
    root = Path(out_dir)
    json_dir = root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    schemas = _schema_bundle()
    for name in ["ClientRequest", "ServerNotification", "ServerRequest"]:
        (json_dir / f"{name}.json").write_text(json.dumps(schemas[name], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (json_dir / "vv_agent_app_server.schemas.json").write_text(
        json.dumps(schemas, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _schema_bundle() -> dict[str, Any]:
    client_methods = [
        "initialize",
        "thread/start",
        "thread/resume",
        "thread/read",
        "thread/list",
        "thread/archive",
        "thread/unsubscribe",
        "turn/start",
        "turn/steer",
        "turn/followUp",
        "turn/interrupt",
        "model/list",
    ]
    server_notifications = [
        "thread/started",
        "thread/status/changed",
        "turn/started",
        "item/started",
        "item/agentMessage/delta",
        "item/completed",
        "turn/completed",
    ]
    server_requests = ["approval/request", "tool/requestUserInput"]
    return {
        "ClientRequest": _method_schema("ClientRequest", client_methods),
        "ServerNotification": _method_schema("ServerNotification", server_notifications),
        "ServerRequest": _method_schema("ServerRequest", server_requests),
    }


def _method_schema(title: str, methods: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": ["method"],
        "properties": {
            "id": {"type": ["string", "integer"]},
            "method": {"enum": sorted(methods)},
            "params": {"type": ["object", "array", "string", "number", "boolean", "null"]},
        },
        "additionalProperties": False,
    }
