"""Capture session-schema diagnostics from real codecs, runtime and App Server."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from scripts.s1_fixture_values import (
        content_ref,
        create_fixture_session,
        fixture_content_reader,
        principal,
        push_fixture_input,
        wrap_fixture_store,
    )
except ModuleNotFoundError:
    from s1_fixture_values import (
        content_ref,
        create_fixture_session,
        fixture_content_reader,
        principal,
        push_fixture_input,
        wrap_fixture_store,
    )

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import declare_input_content, publish_record_content
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.kernel import drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
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
        store = wrap_fixture_store(store)
        client = ScriptedLLM([])
        resolved = ResolvedModelConfig("scripted", "m", "m", "m", [])
        runtime = Runtime(
            agent=Agent("codec", "Answer."),
            config=RunConfig(workspace="/fixture", session_memory_enabled=False),
            resolved=resolved,
            llm=client,
            heartbeat_store=lambda: nullcontext(store),
            model_resolver=lambda _request: (client, replace(resolved, backend="")),
            content_reader=fixture_content_reader(store, "selection"),
        )
        create_fixture_session(store, "selection", workspace="/fixture", consumers=())
        push_fixture_input(
            store,
            "selection",
            InboxItem("select", "model_selection", {"selection": {"model": "m", "route": {}}}),
        )
        drive(store, "selection", runtime=runtime)
        state, rows, _ = store.read_state("selection")
        applied = next(row for row in rows if row.record.kind == "input_applied")
        logical = hydrate_record_payload(applied, hydration_for_session_store(store, "selection"))
        assert logical["disposition"] == "rejected"
        assert state.model_selection is None and not state.turns and not client.steps
        return logical["reason"]


def provider_reason() -> str:
    request_ref = content_ref("provider/request")
    budget_ref = content_ref("provider/budget")
    plan = make_record(
        "op_planned",
        session_id="s",
        turn_id="t",
        operation_id="op",
        attempt=1,
        payload={
            "op_kind": "model",
            "purpose": "primary",
            "request": request_ref,
            "request_digest": digest(request_ref),
            "context_version": "v1",
            "dependencies": [],
            "tool": None,
            "idempotency_key": "key",
            "budget_admission": budget_ref,
            "not_before_ms": None,
            "provider_binding": "provider",
            "consumed_unknowns": [],
        },
    )
    handle = {
        "kind": "provider",
        "provider": "provider",
        "job_id": content_ref("provider/job"),
        "operation_id": "op",
        "attempt": 1,
        "request_digest": digest(request_ref),
        "evidence": content_ref("provider/evidence"),
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
    created = SessionSpec("s", principal(), content_ref("workspace"), attributes=content_ref("attributes")).record()
    body = created.encode()
    wrong_payload = created.to_dict()
    wrong_payload["payload"]["workspace"] = ["CUSTOMER_CONTENT"]
    wrong_version = created.to_dict() | {"schema_version": True}
    wrong_id = created.to_dict() | {"record_id": "wrong"}
    item = InboxItem(
        "i",
        "user",
        {"content": content_ref("go")},
        actor=principal("fixture/user"),
    )
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
                        "s",
                        principal(),
                        content_ref("workspace"),
                        attributes={"app_server": {"unexpected": "CUSTOMER_CONTENT"}},
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
    assert all(case["error"]["type"] == "RecordError" for case in value["schema_diagnostics"])
    assert value["app_server_error"] == {"code": -32603, "message": "stored record chain digest mismatch"}
    assert value["provider_reason"] == "provider_result_invalid"
    assert value["model_selection_reason"] == "model_selection_unresolvable"
    return value


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def _source_tree_digest(repo: Path) -> str:
    paths = ("src/vv_agent", "scripts/record_codec_fixtures.py", "scripts/s1_fixture_values.py")
    digest_input = bytearray(_git(repo, "diff", "--binary", "HEAD", "--", *paths))
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", *paths).decode().splitlines()
    for relative in sorted(untracked):
        path = repo / relative
        if path.is_file():
            digest_input.extend(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return sha256(digest_input).hexdigest()


def _content_ref_paths(value: Any, path: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if set(value) == {"content_id", "mac", "size", "key_id", "labels"}:
            ContentRef.from_dict(value)
            return [{"path": list(path), **value}]
        for key, child in value.items():
            refs.extend(_content_ref_paths(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            refs.extend(_content_ref_paths(child, (*path, index)))
    return sorted(refs, key=lambda ref: json.dumps(ref["path"], ensure_ascii=True, separators=(",", ":")))


def _capture_value(record: Record | InboxItem, producer: str) -> dict[str, Any]:
    body = record.encode()
    expected = record.to_dict()
    parsed = (Record if isinstance(record, Record) else InboxItem).parse(body)
    assert parsed.to_dict() == expected
    assert parsed.encode() == body
    encoded = base64.b64encode(body).decode()
    return {
        "name": producer,
        "producer": f"vv_agent.session.records.{type(record).__name__}.encode",
        "bytes_base64": encoded,
        "canonical_json_base64": encoded,
        "canonical_json_utf8_bytes": len(body),
        "sha256": sha256(body).hexdigest(),
        "content_refs": _content_ref_paths(expected),
        "wire": expected,
    }


def _trusted_storage_capture_impl() -> dict[str, Any]:
    """Capture contract 40 S1 records, inboxes and content references from Python producers."""

    repo = Path(__file__).resolve().parents[1]
    revision = _git(repo, "rev-parse", "HEAD").decode().strip()
    dirty = bool(_git(repo, "status", "--porcelain=v1", "--untracked-files=normal"))

    with SQLiteStore.standalone(":memory:") as raw_store:
        raw_store.install_schema()
        store = wrap_fixture_store(raw_store)
        create_fixture_session(store, "trusted", workspace={"cwd": "/fixture"}, consumers=())
        created = store.read("trusted", limit=1).records[0].record
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="trusted")
            user_attributes = writer.put_json({"role": "fixture/user"})
            user_actor = principal("trusted/user")
            user_actor["attributes"] = user_attributes.to_dict()

            declared_user = declare_input_content(
                writer,
                InboxItem("trusted-input", "user", {"content": {"text": "trusted input"}}, actor=user_actor),
            ).item
            signed_user = replace(
                declared_user,
                admission_proof={
                    "admission_key_id": "ledger/key-1",
                    "admission_signature": "AAECAwQ=",
                    "admission_signed_ms": 1800000000000,
                },
            )

            definition = writer.put_json({"task": {"name": "trusted-fixture"}})
            budget = writer.put_json({"tokens": 1000, "tool_calls": 4})
            operational_policy = writer.put_json({"network_policy_digest": "3" * 64, "isolation_policy_digest": "4" * 64})
            environment_descriptor = writer.put_json({"kind": "deterministic-local"})
            turn_started = make_record(
                "turn_started",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                payload={
                    "input_ids": ["trusted-input"],
                    "definition": definition.to_dict(),
                    "definition_digest": digest(definition.to_dict()),
                    "handler_version": "fixture-v1",
                    "budget": budget.to_dict(),
                    "binding": None,
                    "generation": 0,
                    "policy_digest": "1" * 64,
                    "requirements": {"level": 1, "handler_version": "fixture-v1", "capabilities": []},
                    "tokenizer_binding": {
                        "model_identity": "model-binding/trusted-fixture",
                        "counting_method": "tokenizer",
                        "implementation_id": "tokenizer/trusted-fixture",
                        "asset_identity": "asset/trusted-fixture-v1",
                        "normalization_id": "normalization/none",
                    },
                    "environment": {
                        "descriptor": environment_descriptor.to_dict(),
                        "environment_capabilities": ["command.execute", "network.deny"],
                        "capabilities_digest": "2" * 64,
                        "network_policy_ref_digest": "3" * 64,
                        "isolation_policy_ref_digest": "4" * 64,
                        "operational_policy": operational_policy.to_dict(),
                    },
                },
            )
            input_applied = make_record(
                "input_applied",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                payload={
                    "input": signed_user.to_dict(),
                    "input_digest": signed_user.digest,
                    "input_proof": {
                        "input_seq": 1,
                        "received_ms": 1800000000001,
                        "key_id": "ledger/key-1",
                        "signature": "AAECAwQ=",
                        "signed_ms": 1800000000000,
                    },
                    "disposition": "applied",
                    "reason": None,
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "before_dispatch",
                },
            )

            request = writer.put_json({"messages": [{"role": "user", "content": "trusted input"}]})
            model_tool = writer.put_json({"name": "fixture-tool"})
            budget_admission = writer.put_json({"tokens": 1000})
            shared_state = writer.put_json({"phase": "prepared"})
            op_planned = make_record(
                "op_planned",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                operation_id="trusted/turn/initial/model/primary/1",
                attempt=1,
                payload={
                    "op_kind": "model",
                    "purpose": "primary",
                    "request": request.to_dict(),
                    "request_digest": digest(request.to_dict()),
                    "context_version": "v1",
                    "dependencies": [],
                    "tool": None,
                    "idempotency_key": "trusted/model/1",
                    "budget_admission": budget_admission.to_dict(),
                    "not_before_ms": None,
                    "provider_binding": "model",
                    "consumed_unknowns": [],
                    "policy_decision": {
                        "policy_digest": "1" * 64,
                        "rule_id": "rule/allow-fixture",
                        "effect": "allow",
                    },
                },
            )
            prepared_request = writer.put_json({"name": "fixture-tool", "arguments": {}})
            op_prepared = make_record(
                "op_prepared",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                operation_id="trusted/turn/initial/model/primary/1/tool/0",
                attempt=1,
                payload={
                    "request": prepared_request.to_dict(),
                    "request_digest": digest(prepared_request.to_dict()),
                    "op_kind": "tool",
                    "tool": model_tool.to_dict(),
                    "provider_binding": "function",
                    "idempotency_key": "trusted/tool/1",
                    "hook_result": None,
                    "shared_state": shared_state.to_dict(),
                    "policy_decision": {
                        "policy_digest": "1" * 64,
                        "rule_id": "rule/prepare-fixture",
                        "effect": "require_approval",
                    },
                },
            )
            result = writer.put_json({"content": "trusted result"})
            usage = writer.put_json({"input_tokens": 3, "output_tokens": 2})
            evidence = writer.put_json({"provider": "scripted", "status": "succeeded"})
            op_completed = make_record(
                "op_completed",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                operation_id="trusted/turn/initial/model/primary/1",
                attempt=1,
                payload={
                    "result": result.to_dict(),
                    "result_digest": digest(result.to_dict()),
                    "usage": usage.to_dict(),
                    "usage_counts": {
                        "tokens": {
                            "input_tokens": 3,
                            "output_tokens": 2,
                            "total_tokens": 5,
                            "reasoning_tokens": None,
                            "usage_source": "provider_reported",
                            "cache_usage": {
                                "status": "provider_reported",
                                "read_input_tokens": 1,
                                "write_input_tokens": 0,
                                "uncached_input_tokens": 2,
                            },
                        },
                        "budget": {
                            "cycles": 1,
                            "tool_calls": 0,
                            "elapsed_ms": 10,
                            "total_tokens": 5,
                            "uncached_input_tokens": 2,
                            "tool_calls_by_name": {},
                            "host_cost": None,
                            "unavailable_dimensions": [],
                        },
                    },
                    "shared_state": None,
                    "evidence": evidence.to_dict(),
                    "execution_started": True,
                    "context": "normal",
                    "request_digest": digest(request.to_dict()),
                    "provider_binding": "model",
                    "policy_decision": {
                        "policy_digest": "1" * 64,
                        "rule_id": "rule/complete-fixture",
                        "effect": "allow",
                    },
                },
            )

            approval_scope = writer.put_json({"tool": "fixture-tool"})
            approval_reason = writer.put_json({"reason": "fixture approval"})
            approval_metadata = writer.put_json({"source": "fixture"})
            approval = declare_input_content(
                writer,
                InboxItem(
                    "trusted-approval",
                    "approval_answer",
                    {
                        "operation_id": "trusted/turn/initial/model/primary/1/tool/0",
                        "attempt": 1,
                        "request_digest": digest(prepared_request.to_dict()),
                        "request_id": "approval/trusted/1",
                        "decision": "allow_session",
                        "scope": approval_scope.to_dict(),
                        "reason": approval_reason.to_dict(),
                        "metadata": approval_metadata.to_dict(),
                        "policy_digest": "1" * 64,
                        "rule_id": "rule/approval-fixture",
                        "expires_ms": 1800000060000,
                    },
                    target_turn_id="trusted/turn/initial",
                    generation=0,
                    actor=user_actor,
                ),
            ).item
            provider_actor = principal("fixture/provider", "service")
            provider_actor["attributes"] = writer.put_json({"role": "fixture/provider"}).to_dict()
            job_id = writer.put_json({"job": "trusted-1"})
            provider_evidence_value = writer.put_json({"provider_result": {"status": "unknown"}})
            query_ref = writer.put_json({"query": "trusted-1"})
            cancel_ref = writer.put_json({"cancel": "trusted-1"})
            provider_input = declare_input_content(
                writer,
                InboxItem(
                    "trusted-evidence",
                    "provider_evidence",
                    {
                        "operation_id": "trusted/turn/initial/model/primary/1",
                        "attempt": 1,
                        "request_digest": digest(request.to_dict()),
                        "handle": {
                            "kind": "provider",
                            "provider": "scripted",
                            "job_id": job_id.to_dict(),
                            "operation_id": "trusted/turn/initial/model/primary/1",
                            "attempt": 1,
                            "request_digest": digest(request.to_dict()),
                            "evidence": provider_evidence_value.to_dict(),
                            "query_ref": query_ref.to_dict(),
                            "cancel_ref": cancel_ref.to_dict(),
                        },
                    },
                    target_turn_id="trusted/turn/initial",
                    generation=0,
                    actor=provider_actor,
                ),
            ).item
            boundary_event = writer.put_json({"event": "trusted-memory-started"})
            boundary = make_record(
                "boundary_recorded",
                session_id="trusted",
                turn_id="trusted/turn/initial",
                payload={
                    "boundary_id": "trusted-boundary",
                    "stage": "memory_started",
                    "source_operation_id": None,
                    "source_digest": None,
                    "data": {"event": boundary_event.to_dict()},
                },
            )

            publication = publish_record_content(
                writer,
                (turn_started, input_applied, op_planned, op_prepared, op_completed, boundary),
            )
            published = {
                record.kind + ":" + record.operation_id if record.operation_id else record.kind: record
                for record in publication.records
            }
            turn_started = published["turn_started"]
            input_applied = published["input_applied"]
            op_planned = published["op_planned:trusted/turn/initial/model/primary/1"]
            op_prepared = published["op_prepared:trusted/turn/initial/model/primary/1/tool/0"]
            op_completed = published["op_completed:trusted/turn/initial/model/primary/1"]
            boundary = published["boundary_recorded"]

    accepted = {
        "session_created_principal_and_content_ref": _capture_value(created, "session_created_principal_and_content_ref"),
        "turn_started_requirements_policy_and_environment": _capture_value(
            turn_started, "turn_started_requirements_policy_and_environment"
        ),
        "signed_user_inbox_admission_proof": _capture_value(signed_user, "signed_user_inbox_admission_proof"),
        "input_applied_copied_input_proof": _capture_value(input_applied, "input_applied_copied_input_proof"),
        "policy_usage_and_tokenizer_projection": _capture_value(op_completed, "policy_usage_and_tokenizer_projection"),
        "op_planned_policy_projection": _capture_value(op_planned, "op_planned_policy_projection"),
        "op_prepared_policy_projection": _capture_value(op_prepared, "op_prepared_policy_projection"),
        "approval_answer_policy_projection": _capture_value(approval, "approval_answer_policy_projection"),
        "provider_evidence_content_ref": _capture_value(provider_input, "provider_evidence_content_ref"),
        "boundary_event_content_ref": _capture_value(boundary, "boundary_event_content_ref"),
    }
    missing = object()

    def reject(
        name: str,
        base: str,
        path: list[str],
        reason: str,
        *,
        value: Any = missing,
        delete: bool = False,
    ) -> dict[str, Any]:
        operation = {"path": path}
        if delete:
            operation["delete"] = True
        else:
            assert value is not missing
            operation["set"] = value
        return {"name": name, "base": base, "patch": [operation], "reason": reason}

    session_base = "session_created_principal_and_content_ref"
    turn_base = "turn_started_requirements_policy_and_environment"
    inbox_base = "signed_user_inbox_admission_proof"
    usage_base = "policy_usage_and_tokenizer_projection"
    rejected = [
        reject("legacy_record_discriminator", session_base, ["schema_version"], "record schema_version must be S1", value=1),
        reject("legacy_inbox_discriminator", inbox_base, ["schema_version"], "inbox schema_version must be S1", value=3),
        reject("unsupported_feature_level", session_base, ["feature_level"], "unknown feature levels reject", value=2),
        reject(
            "logical_chain_digest_rejected",
            session_base,
            ["chain_digest"],
            "chain fields belong to the storage envelope",
            value="0" * 64,
        ),
        reject(
            "principal_unknown_field",
            session_base,
            ["payload", "principal", "extra"],
            "principal is closed",
            value=True,
        ),
        reject(
            "invalid_subject_kind",
            inbox_base,
            ["actor", "subject_kind"],
            "subject kinds are closed",
            value="system",
        ),
        reject(
            "raw_workspace_text_rejected",
            session_base,
            ["payload", "workspace"],
            "workspace is a ContentRef",
            value="/plaintext",
        ),
        reject(
            "raw_inbox_content_rejected",
            inbox_base,
            ["payload", "content"],
            "inbox content is a ContentRef",
            value="go",
        ),
        reject(
            "operational_policy_digests_are_encrypted",
            turn_base,
            ["payload", "environment", "operational_policy"],
            "operational_policy is a ContentRef",
            value={"network_policy_digest": "1" * 64, "isolation_policy_digest": "2" * 64},
        ),
        reject(
            "tokenizer_binding_unknown_field",
            turn_base,
            ["payload", "tokenizer_binding", "extra"],
            "tokenizer binding is closed",
            value=True,
        ),
        reject(
            "policy_decision_invalid_effect",
            usage_base,
            ["payload", "policy_decision", "effect"],
            "policy decision effects are closed",
            value="maybe",
        ),
        reject(
            "usage_counts_unknown_field",
            usage_base,
            ["payload", "usage_counts", "extra"],
            "usage projection is closed",
            value=1,
        ),
        reject(
            "content_ref_unknown_field",
            session_base,
            ["payload", "workspace", "extra"],
            "ContentRef is closed",
            value=True,
        ),
        reject(
            "content_label_source_unsorted",
            turn_base,
            ["payload", "environment", "descriptor", "labels", "source"],
            "ContentLabels.source is in source-kind order",
            value=["untrusted", "user"],
        ),
        reject(
            "content_label_source_duplicate",
            turn_base,
            ["payload", "environment", "descriptor", "labels", "source"],
            "ContentLabels.source is duplicate-free",
            value=["user", "user"],
        ),
        reject(
            "requirements_capabilities_unsorted",
            turn_base,
            ["payload", "requirements", "capabilities"],
            "requirements capabilities are sorted",
            value=["z", "a"],
        ),
        reject(
            "requirements_capabilities_duplicate",
            turn_base,
            ["payload", "requirements", "capabilities"],
            "requirements capabilities are duplicate-free",
            value=["a", "a"],
        ),
        reject(
            "admission_proof_partial",
            inbox_base,
            ["admission_proof", "admission_signed_ms"],
            "admission proof fields are all present",
            delete=True,
        ),
        reject(
            "admission_proof_unknown_field",
            inbox_base,
            ["admission_proof", "extra"],
            "admission proof is closed",
            value=True,
        ),
        reject(
            "admission_signature_noncanonical_base64",
            inbox_base,
            ["admission_proof", "admission_signature"],
            "signature bytes use canonical base64",
            value="AAECAwQ",
        ),
        reject(
            "raw_request_text_rejected",
            "op_planned_policy_projection",
            ["payload", "request"],
            "request is a ContentRef",
            value={"model": "plaintext"},
        ),
        reject(
            "raw_result_text_rejected",
            usage_base,
            ["payload", "result"],
            "result is a ContentRef",
            value={"content": "plaintext"},
        ),
        reject(
            "raw_usage_text_rejected",
            usage_base,
            ["payload", "usage"],
            "usage is a ContentRef",
            value={"prompt_tokens": 1},
        ),
        reject(
            "raw_approval_scope_rejected",
            "approval_answer_policy_projection",
            ["payload", "scope"],
            "approval scope is a ContentRef",
            value={"tool": "echo"},
        ),
        reject(
            "raw_provider_evidence_rejected",
            "provider_evidence_content_ref",
            ["payload", "handle", "evidence"],
            "provider evidence is a ContentRef",
            value={"raw": True},
        ),
        reject(
            "raw_boundary_event_rejected",
            "boundary_event_content_ref",
            ["payload", "data", "event"],
            "boundary event is a ContentRef",
            value={"raw": True},
        ),
    ]
    bases = {case["name"]: case["wire"] for case in accepted.values()}
    for case in rejected:
        wire = deepcopy(bases[case["base"]])
        for operation in case["patch"]:
            target: Any = wire
            for part in operation["path"][:-1]:
                target = target[part]
            if operation.get("delete"):
                del target[operation["path"][-1]]
            else:
                target[operation["path"][-1]] = operation["set"]

    for case in accepted.values():
        body = base64.b64decode(case["bytes_base64"], validate=True)
        wire = json.loads(body)
        assert body == canonical_json_bytes(wire)
        assert sha256(body).hexdigest() == case["sha256"]
        actual_refs = _content_ref_paths(wire)
        if case["content_refs"] != actual_refs:
            raise AssertionError(
                json.dumps(
                    {"case": case["name"], "expected": case["content_refs"], "actual": actual_refs},
                    indent=2,
                    sort_keys=True,
                )
            )
        assert case["content_refs"], case["name"]

    return {
        "schema_version": "vv-agent.trusted-storage-s1.v1",
        "status": "producer-captured",
        "storage_major": "S1",
        "feature_level": 1,
        "revision": revision,
        "record_schema_version": "S1",
        "inbox_schema_version": "S1",
        "provenance": {
            "producer_capture": True,
            "producer_repository": "https://github.com/AndersonBY/vv-agent",
            "capture_command": (
                "uv run python scripts/record_codec_fixtures.py "
                "--output <diagnostics.json> --trusted-output <trusted_storage_s1.json>"
            ),
            "source_revision": revision,
            "source_worktree_dirty": dirty,
            "source_tree_sha256": _source_tree_digest(repo),
            "note": (
                "Captured from the current Python S1 constructors, content publication writer "
                "and canonical encoder. Signed proof fields are deterministic valid wire values; "
                "this capture does not claim a cryptographic signer was executed."
            ),
            "historical_capture": "fixtures/session_record_diagnostics.json",
        },
        "field_dispositions": {
            "session_record.envelope": {"schema_version": "const S1", "feature_level": "const 1", "closed": True},
            "session_inbox.envelope": {
                "schema_version": "const S1",
                "feature_level": "const 1",
                "actor": "Principal",
                "admission_proof": "closed|null",
                "closed": True,
            },
            "session_created.payload.principal": {
                "shape": "{tenant_id,subject_id,subject_kind,attributes:ContentRef}",
                "closed": True,
            },
            "session_created.payload.workspace": {"shape": "ContentRef", "closed": True},
            "session_created.payload.attributes": {"shape": "ContentRef", "closed": True},
            "turn_started.payload.definition": {"shape": "ContentRef", "closed": True},
            "turn_started.payload.budget": {"shape": "ContentRef", "closed": True},
            "turn_started.payload.policy_digest": {"shape": "H|null"},
            "turn_started.payload.requirements": {"shape": "{level,handler_version,capabilities}", "closed": True},
            "turn_started.payload.tokenizer_binding": {"shape": "closed|null"},
            "turn_started.payload.environment": {"shape": "closed|null", "operational_policy": "ContentRef|null"},
            "turn_started.payload.environment.operational_policy": {
                "shape": "ContentRef|null",
                "hydrated": "{network_policy_digest,isolation_policy_digest}",
                "closed": True,
            },
            "input_applied.payload.input_proof": {
                "shape": "null|{input_seq,received_ms,key_id,signature,signed_ms}",
                "closed": True,
            },
            "op_planned.payload.request_tool_budget_admission": {
                "shape": "ContentRef|ContentRef|null|ContentRef",
                "closed": True,
            },
            "op_prepared.payload.request_tool_hook_shared_state": {
                "shape": "ContentRef|ContentRef|ContentRef|null|ContentRef",
                "closed": True,
            },
            "op_completed.payload.result_usage_evidence": {"shape": "ContentRef", "closed": True},
            "op.policy_decision": {"shape": "null|{policy_digest,rule_id,effect}", "closed": True},
            "op_completed.payload.usage_counts": {"shape": "closed usage projection", "closed": True},
            "approval_answer.payload.scope_reason_metadata": {
                "shape": "ContentRef|ContentRef|null|ContentRef|null",
                "closed": True,
            },
            "approval_answer.policy_projection": {
                "shape": "policy_digest:H|null,rule_id:T|null,expires_ms:N|null",
                "closed": True,
            },
            "provider_handles.payload_fields": {"shape": "ContentRef|ContentRef|null", "closed": True},
            "boundary_recorded.payload.data": {"shape": "stage-specific ContentRef leaves", "closed": True},
            "session_storage_envelope": {
                "fixture": "fixtures/session_storage_envelope.json",
                "chain_excluded_from_record_identity": True,
            },
            "session_storage_lifecycle": {
                "fixture": "fixtures/session_storage_lifecycle.json",
                "key_destroy_forget_export": True,
            },
            "ContentRef": {"shape": "{content_id,mac,size,key_id,labels}", "closed": True},
            "ContentLabels": {"P2_default": "source=[] and classification=internal", "closed": True},
        },
        "accepted_cases": list(accepted.values()),
        "rejected_cases": rejected,
    }


class _DeterministicSecrets:
    def __init__(self) -> None:
        self._counter = 0

    def token_bytes(self, count: int) -> bytes:
        output = bytearray()
        while len(output) < count:
            self._counter += 1
            output.extend(sha256(f"trusted-storage-capture:{self._counter}".encode()).digest())
        return bytes(output[:count])

    def token_urlsafe(self, count: int) -> str:
        return base64.urlsafe_b64encode(self.token_bytes(count)).decode().rstrip("=")


def trusted_storage_capture() -> dict[str, Any]:
    from vv_agent.session import content_store as content_store_module

    with patch.object(content_store_module, "secrets", _DeterministicSecrets()):
        return _trusted_storage_capture_impl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(produce(), ensure_ascii=True, sort_keys=True, indent=2) + "\n")
    if args.trusted_output is not None:
        args.trusted_output.write_text(json.dumps(trusted_storage_capture(), ensure_ascii=True, sort_keys=True, indent=2) + "\n")
