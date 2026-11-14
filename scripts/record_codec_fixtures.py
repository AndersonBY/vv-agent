"""Capture session-schema diagnostics from real codecs, runtime and App Server."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Accepted, ProviderResultInvalid, validate_acceptance
from vv_agent.session.records import InboxItem, Record, RecordError, SessionSpec, digest, make_record
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.workspace import MemoryWorkspaceBackend


def rejected(callback) -> dict[str, str]:
    try:
        callback()
    except RecordError as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("invalid value was accepted")


def app_server_error() -> dict[str, Any]:
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        transport = ChannelTransport(connection_id="codec-owner", outbound_capacity=100)
        host = DefaultAppServerHost(
            agent=Agent("codec", "Answer."),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.new("test", "m", []), workspace_backend=MemoryWorkspaceBackend()
            ),
        )
        server = AppServer(transport=transport, host=host, store=store)
        request_id = 0

        def rpc(method, params):
            nonlocal request_id
            request_id += 1
            server.processor.process_message(
                transport.connection_id, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            while True:
                wire = transport.receive_outbound(timeout=10)
                if wire.get("id") == request_id:
                    return wire

        try:
            assert "result" in rpc("initialize", {"clientInfo": {"name": "codec-producer"}})
            server.processor.process_message(transport.connection_id, {"jsonrpc": "2.0", "method": "initialized"})
            assert "result" in rpc("thread/start", {"inputId": "codec-thread"})
            value = store.read("codec-thread", limit=1).records[0].record.to_dict()
            value["payload"]["workspace"] = ["CUSTOMER_CONTENT"]
            body = canonical_json_bytes(value)
            with store.atomic():
                store._rows(
                    "UPDATE sk_record SET body=%s,digest=%s WHERE session_id=%s AND seq=1",
                    (body, sha256(body).hexdigest(), "codec-thread"),
                )
            reply = rpc("thread/read", {"threadId": "codec-thread"})
            return reply["error"]
        finally:
            server.run_adapter.close()
            server.kernel.close()


def model_selection_reason() -> str:
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        client = ScriptedLLM([])
        resolved = ResolvedModelConfig("scripted", "m", "m", "m", [])
        runtime = Runtime(
            agent=Agent("codec", "Answer."),
            config=RunConfig(workspace="/fixture", session_memory_enabled=False),
            resolved=resolved,
            llm=client,
            heartbeat_store=lambda: nullcontext(store),
            model_resolver=lambda _request: (client, replace(resolved, backend="")),
        )
        with store.atomic() as tx:
            tx.create(SessionSpec("selection", "p", "/fixture"), consumers=())
            tx.push("selection", InboxItem("select", "model_selection", {"selection": {"model": "m", "route": {}}}))
        drive(store, "selection", runtime=runtime)
        state, rows, _ = store.read_state("selection")
        applied = next(row.record for row in rows if row.record.kind == "input_applied")
        assert applied.payload["disposition"] == "rejected"
        assert state.model_selection is None and not state.turns and not client.steps
        return applied.payload["reason"]


def provider_reason() -> str:
    request = {"messages": []}
    plan = make_record(
        "op_planned",
        session_id="s",
        turn_id="t",
        operation_id="op",
        attempt=1,
        payload={
            "op_kind": "model",
            "purpose": "primary",
            "request": request,
            "request_digest": digest(request),
            "context_version": "v1",
            "dependencies": [],
            "tool": None,
            "idempotency_key": "key",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "provider",
            "consumed_unknowns": [],
        },
    )
    handle = {
        "kind": "provider",
        "provider": "provider",
        "job_id": "job",
        "operation_id": "op",
        "attempt": 1,
        "request_digest": digest(request),
        "evidence": "accepted",
        "query_ref": None,
        "cancel_ref": None,
        "unexpected": "CUSTOMER_CONTENT",
    }
    try:
        validate_acceptance(plan, Accepted(handle))
    except ProviderResultInvalid as error:
        return str(error)
    raise AssertionError("invalid provider acceptance was accepted")


def produce() -> dict[str, Any]:
    created = SessionSpec("s", "p", "/fixture").record()
    body = created.encode()
    wrong_payload = created.to_dict()
    wrong_payload["payload"]["workspace"] = ["CUSTOMER_CONTENT"]
    wrong_version = created.to_dict() | {"schema_version": True}
    wrong_id = created.to_dict() | {"record_id": "wrong"}
    item = InboxItem("i", "user", {"content": "go"})
    selection = {"model": "m", "route": {}, "model_settings": {"tool_choice": {"type": "function", "function": {"name": " "}}}}
    value: dict[str, Any] = {
        "version": 1,
        "schema_diagnostics": [
            {"case": "record_version", "error": rejected(lambda: Record.parse(json.dumps(wrong_version).encode()))},
            {"case": "record_payload", "error": rejected(lambda: Record.parse(json.dumps(wrong_payload).encode()))},
            {"case": "inbox_envelope", "error": rejected(lambda: replace(item, available_ms=True).encode())},
            {
                "case": "reserved_metadata",
                "error": rejected(
                    lambda: SessionSpec(
                        "s", "p", "/fixture", attributes={"app_server": {"unexpected": "CUSTOMER_CONTENT"}}
                    ).record()
                ),
            },
        ],
        "preserved_diagnostics": [
            {"case": "record_identity", "error": rejected(lambda: Record.parse(json.dumps(wrong_id).encode()))},
            {
                "case": "embedded_digest",
                "error": rejected(
                    lambda: make_record(
                        "input_applied",
                        session_id="s",
                        payload={
                            "input": item.to_dict(),
                            "input_digest": "0" * 64,
                            "disposition": "applied",
                            "reason": None,
                            "target_operation_id": None,
                            "target_wait_id": None,
                            "position": "before_dispatch",
                        },
                    )
                ),
            },
            {"case": "duplicate_field", "error": rejected(lambda: Record.parse(b'{"a":1,"\\u0061":2}'))},
            {"case": "nonobject", "error": rejected(lambda: Record.parse(b"[]"))},
            {"case": "malformed_json", "error": rejected(lambda: Record.parse(b"{"))},
            {
                "case": "model_settings",
                "error": rejected(lambda: InboxItem("i", "model_selection", {"selection": selection}).encode()),
            },
        ],
        "app_server_error": app_server_error(),
        "provider_reason": provider_reason(),
        "model_selection_reason": model_selection_reason(),
        "canonical": {
            "record_utf8": body.decode(),
            "record_sha256": created.digest,
            "inbox_utf8": item.encode().decode(),
            "inbox_sha256": item.digest,
        },
    }
    assert all(
        case["error"] == {"type": "RecordError", "message": "invalid session schema"} for case in value["schema_diagnostics"]
    )
    assert value["app_server_error"] == {"code": -32603, "message": "invalid session schema"}
    assert value["provider_reason"] == "provider_result_invalid"
    assert value["model_selection_reason"] == "model_selection_unresolvable"
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(produce(), ensure_ascii=True, sort_keys=True, indent=2) + "\n")
