#!/usr/bin/env python3
"""Schema export and overload handling for App Server clients."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vv_agent.app_server import AppServerErrorCode, ChannelTransport, MessageProcessor, OutgoingRouter
from vv_agent.app_server.request_serialization import RequestSerializationQueues
from vv_agent.app_server.schema import generate_json_schema, generate_typescript
from vv_agent.app_server.transport import AppServerOverloadedError


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        generate_json_schema(temp_dir)
        generate_typescript(Path(temp_dir) / "typescript")
        schema_dir = Path(temp_dir) / "json"
        print("schemas:", ", ".join(sorted(path.name for path in schema_dir.glob("*.json"))))
        print("typescript:", ", ".join(sorted(path.name for path in (Path(temp_dir) / "typescript").glob("*.ts"))))

    schema_transport = ChannelTransport(connection_id="schema-client")
    schema_router = OutgoingRouter()
    schema_router.register_transport(schema_transport)
    schema_processor = MessageProcessor(router=schema_router)
    schema_processor.process_message(
        "schema-client",
        {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "schema-example"}}},
    )
    print(json.dumps(schema_transport.receive_outbound(timeout=1), ensure_ascii=False, separators=(",", ":")))
    schema_processor.process_message("schema-client", {"method": "initialized"})
    schema_processor.process_message("schema-client", {"id": 1, "method": "schema/export", "params": {}})
    exported = schema_transport.receive_outbound(timeout=1)["result"]
    print("schema/export:", sorted(exported))

    bounded = ChannelTransport(connection_id="slow-client", outbound_capacity=1)
    bounded.write_outbound({"id": 0, "result": {}})
    try:
        bounded.write_outbound({"id": 1, "result": {}})
    except AppServerOverloadedError as exc:
        print("transport overload:", str(exc))

    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    processor = MessageProcessor(router=router, serialization_queues=RequestSerializationQueues(max_queued_per_scope=0))
    processor.process_message("conn_1", {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "overload-example"}}})
    print(json.dumps(transport.receive_outbound(timeout=1), ensure_ascii=False, separators=(",", ":")))
    processor.process_message("conn_1", {"method": "initialized"})

    processor.process_message("conn_1", {"id": 1, "method": "model/list", "params": {}})
    response = transport.receive_outbound(timeout=1)
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    assert response["error"]["code"] == AppServerErrorCode.SERVER_OVERLOADED


if __name__ == "__main__":
    main()
