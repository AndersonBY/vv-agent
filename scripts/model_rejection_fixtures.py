"""Generate HTTP rejection evidence with the real kernel and deterministic model doubles."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from scripts.s1_fixture_values import (
    create_fixture_session,
    fixture_content_reader,
    push_fixture_input,
    wrap_fixture_store,
)
from tests.model_context_cases import CONTEXT_QUOTA_CASES
from tests.session.test_compaction import SUMMARY, configured, history
from tests.session.test_endpoint_routing import host
from tests.session.test_model_http_observation import local_vendor_transport
from tests.session.test_model_rejection import adapter_error, context_http_error, fail, http_error
from tests.session.test_model_rejection_retry_policy import retry_transport
from tests.session.test_model_response_completion import local_response_transport
from tests.session.test_recovery_matrix import records_of, runtime
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.llm.errors import ModelRequestRejected, ModelStreamInterrupted
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.output_validation import OutputValidationResult
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store, project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse


def _historical_wire(wire: dict[str, Any]) -> dict[str, Any]:
    """Project current S1 records onto the Contract 39 rejection wire."""
    for receipt in wire["receipts"]:
        result = receipt.get("result")
        if isinstance(result, dict) and isinstance(result.get("raw"), dict):
            result["raw"].pop("requested_model_id", None)
    for event in wire["events"]:
        if event.get("type") != "run_failed":
            continue
        metadata = event.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("session_seq"), int):
            metadata["session_seq"] += 1
    return wire


def produce() -> dict:
    vectors = []
    cases = [
        ("bad_request", http_error(400), False, False),
        ("rate_limit_success", http_error(429), True, False),
        ("rate_limit_exhausted", http_error(429), False, False),
        ("temporary_rejection", ModelRequestRejected(409, "Busy", retryable=True), True, False),
        ("provider_internal", http_error(500), False, False),
        ("request_timeout", http_error(408), False, False),
        ("timeout", TimeoutError("timeout"), False, False),
        ("connection_lost", ConnectionError("lost"), False, False),
        ("ordinary_exception", RuntimeError("ordinary"), False, False),
        ("summary_rejected", http_error(400), True, True),
        ("tool_string_rejected", context_http_error("tool_string"), False, False),
        ("context_openai", context_http_error("openai"), True, False),
        ("context_openai_wrapped", context_http_error("openai", wrapped=True), True, False),
        ("context_typed", context_http_error("typed"), True, False),
        ("context_anthropic", context_http_error("anthropic"), True, False),
        ("context_moonshot", context_http_error("moonshot"), True, False),
        ("context_payload", context_http_error("payload"), True, False),
        ("context_content_string", context_http_error("content_string"), True, False),
    ]
    routes: dict[str, list[str]] = {}
    quota_transports: dict[str, tuple[int, dict[str, Any]]] = {}
    quota_expected: dict[str, tuple[int, bool]] = {}
    for label, status, detail, too_long in CONTEXT_QUOTA_CASES:
        name = f"context_quota_{label}" if too_long else f"quota_rejected_{label}"
        error = http_error(status, detail.get("message", "Rejected"))
        error.body = {"error": detail}
        cases.append((name, error, too_long, False))
        quota_expected[name] = (status, too_long)
        if not too_long:
            quota_transports[name] = (status, detail)
    for status in (400, 401, 403, 404, 429):
        for mode in ("single", "success", "exhausted"):
            name = f"endpoint_{status}_{mode}"
            routes[name] = ["a"] if mode == "single" else ["a", "b", "c"]
            cases.append((name, http_error(status), mode == "success", False))
    for status, retryable, endpoints in ((401, False, ["a", "b"]), (401, True, ["a"]), (429, False, ["a", "b"])):
        name = f"endpoint_override_{status}_{retryable}"
        routes[name] = endpoints
        cases.append((name, ModelRequestRejected(status, "Denied", retryable=retryable), True, False))
    for status in (408, 500):
        for explicit in (False, True):
            outer = ModelRequestRejected(400, "Rejected") if explicit else http_error(400)
            outer.__cause__ = http_error(status)
            cases.append((f"mixed_{status}_{explicit}", outer, False, False))
    for kind, uncertain in (
        ("timeout", TimeoutError("lost")),
        ("transport", ConnectionError("lost")),
        ("stream", ModelStreamInterrupted("partial")),
    ):
        outer = ModelRequestRejected(400, "Rejected")
        outer.__cause__ = uncertain
        cases.append((f"mixed_{kind}", outer, False, False))
    cases.append(("unobservable_http", http_error(400, observed=False), False, False))
    cases.append(("unobservable_http_context", http_error(400, "maximum context length exceeded", observed=False), False, False))
    for provider in ("openai", "anthropic"):
        for exception_name in ("APIConnectionError", "APITimeoutError"):
            for mode in ("explicit", "automatic", "context"):
                outer = (
                    RuntimeError("maximum context length exceeded")
                    if mode == "context"
                    else ModelRequestRejected(400, "Rejected")
                    if mode == "explicit"
                    else http_error(400)
                )
                outer.__cause__ = getattr(import_module(provider), exception_name)(
                    request=httpx2.Request("POST", "https://provider.invalid")
                )
                cases.append((f"native_{provider}_{exception_name}_{mode}", outer, False, False))
    cases.append(
        (
            "private_diagnostic",
            ModelRequestRejected(
                400,
                "x" * 480 + "API key provided: SYNTHETIC_CREDENTIAL messages=[SYNTHETIC_INPUT]",
                provider_code="api_key:SYNTHETIC_PRIVATE",
            ),
            False,
            False,
        )
    )
    cases.append(
        (
            "repair_private_diagnostic",
            http_error(400, "Authorization: Bearer SYNTHETIC_CREDENTIAL messages=[SYNTHETIC_INPUT]"),
            False,
            False,
        )
    )
    transport_cases: dict[str, tuple[str, str]] = {}
    completion_cases: dict[str, tuple[str, str, int]] = {}
    retry_cases: dict[str, tuple[str, str, int, dict[str, str]]] = {}
    for provider in ("openai", "anthropic"):
        for case in ("timeout_then_400", "500_then_400", "always_429", "single_400"):
            name = f"transport_{provider}_{case}"
            transport_cases[name] = (provider, case)
            cases.append((name, None, False, False))
        for label, case, status, headers in [
            ("conflict", "complete", 409, {}),
            ("retry_after", "complete", 429, {"retry-after": "7"}),
            ("retry_after_timeout", "timeout", 429, {"retry-after": "7"}),
            ("retry_false_timeout", "timeout", 429, {"retry-after": "7", "x-should-retry": "false"}),
            ("retry_true_400", "complete", 400, {"x-should-retry": "true"}),
            ("oversized", "large", 400, {}),
        ]:
            name = f"retry_{provider}_{label}"
            retry_cases[name] = (provider, case, status, headers)
            cases.append((name, None, False, False))
        for case, status in [
            ("pre_raw", 413),
            ("flush", 413),
            ("mid_body", 413),
            ("closed_unread", 413),
            ("gzip_complete", 400),
            ("gzip_complete", 429),
            (f"success_{provider}", 200),
        ]:
            name = f"body_{provider}_{case}_{status}"
            completion_cases[name] = (provider, case, status)
            cases.append((name, None, False, False))
    for name, error, succeeds, compaction in cases:
        with (
            TemporaryDirectory(prefix="model-rejection-") as directory,
            SQLiteStore.standalone(f"{directory}/session.sqlite") as store,
            patch("vv_agent.kernel._driver_outcomes.MODEL_REJECTION_BACKOFF_SECONDS", 0),
        ):
            uses_frozen_history = name.startswith("context_") or compaction
            store.install_schema()
            database = Path(store.path)
            calls = []
            endpoint_calls = []
            rejected = fail(error, calls)
            store = wrap_fixture_store(store)
            create_fixture_session(
                store,
                "s",
                attributes={
                    "seed": {
                        "messages": [message.to_dict() for message in history()] if uses_frozen_history else [],
                        "shared_state": {},
                        "unseen_input_protection": not uses_frozen_history,
                        "response_rejection": True,
                    }
                },
            )
            push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
            if name.startswith("context_"):
                rt = configured(database, [rejected, LLMResponse(SUMMARY), LLMResponse("done")])
                rt.memory_manager.compact_threshold = 999999
            else:
                rt = (configured if uses_frozen_history else runtime)(
                    database, [rejected, LLMResponse("done") if succeeds else rejected]
                )
            rt.content_reader = fixture_content_reader(store, "s")
            if name == "repair_private_diagnostic":
                rt = runtime(database, [LLMResponse("invalid")])
                rt.content_reader = fixture_content_reader(store, "s")
                rt.agent.output_validation_enabled = True
                rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid")
                rt.agent.output_repair = rejected
            http_requests = None
            retry_sleeps = None
            if name in transport_cases or name in completion_cases or name in retry_cases or name in quota_transports:
                if name in quota_transports:
                    provider = "openai"
                    transport = local_vendor_transport("quota", rejection=quota_transports[name])
                elif name in transport_cases:
                    provider, transport_case = transport_cases[name]
                    transport = local_vendor_transport(transport_case)
                elif name in retry_cases:
                    provider, retry_case, status, headers = retry_cases[name]
                    transport = retry_transport(provider, retry_case, status, headers)
                else:
                    provider, completion_case, status = completion_cases[name]
                    transport = local_response_transport(completion_case, status)
                rt = runtime(database, [])
                rt.content_reader = fixture_content_reader(store, "s")
                rt.llm = VvLlmClient([EndpointTarget("a", "synthetic-unused", "https://provider.invalid")], backend=provider)
                with transport as evidence:
                    requests, clients = (evidence[0], evidence[3] if name in retry_cases else evidence[-1])
                    if name in retry_cases:
                        retry_sleeps = evidence[2]
                    drive(store, "s", runtime=rt)
                assert all(client.is_closed for client in clients)
                http_requests = len(requests)
            elif name in routes:
                rt = host(database, {"scripted": routes[name]})
                rt.content_reader = fixture_content_reader(store, "s")

                def create(*, endpoint_id, endpoint_calls=endpoint_calls, succeeds=succeeds, calls=calls, error=error, **_kwargs):
                    def complete(**_payload):
                        endpoint_calls.append(endpoint_id)
                        if len(endpoint_calls) == 1 or not succeeds:
                            calls.append(endpoint_id)
                            raise adapter_error(
                                http_error(
                                    getattr(error, "status_code", 400),
                                    f"Denied at {endpoint_id}. Authorization: Bearer sk-secret",
                                )
                                if not isinstance(error, ModelRequestRejected)
                                else error
                            )
                        return [SimpleNamespace(content="done", tool_calls=[], usage={})]

                    return SimpleNamespace(create_completion=complete)

                with patch("vv_agent.llm.vv_llm_client.create_chat_client", create):
                    drive(store, "s", runtime=rt)
            else:
                drive(store, "s", runtime=rt)
            stored_records = read_state(store, "s")[1]
            hydrated_payloads = {
                row.record.record_id: hydrate_record_payload(
                    row,
                    hydration_for_session_store(store, "s"),
                )
                for row in stored_records
            }
            wire: dict[str, Any] = {
                "id": name,
                "rejected_calls": len(calls),
                "plans": [
                    {
                        "attempt": r.attempt,
                        "purpose": hydrated_payloads[r.record_id]["purpose"],
                        "not_before_ms": hydrated_payloads[r.record_id]["not_before_ms"],
                    }
                    for r in records_of(store, "op_planned")
                ],
                "receipts": [
                    {
                        "attempt": r.attempt,
                        "result": hydrated_payloads[r.record_id]["result"],
                        "execution_started": hydrated_payloads[r.record_id]["execution_started"],
                    }
                    for r in records_of(store, "op_completed")
                ],
                "unknowns": [hydrated_payloads[r.record_id] for r in records_of(store, "op_unknown")],
                "terminal": hydrated_payloads[records_of(store, "turn_ended")[-1].record_id],
                "context_compacted": len(records_of(store, "context_compacted")),
                "events": [
                    {k: v for k, v in e.to_dict().items() if k in {"type", "error", "error_code", "outcome", "metadata"}}
                    for e in project_records(
                        stored_records,
                        hydration=hydration_for_session_store(store, "s"),
                    )
                    if e.type in {"model_call_failed", "model_retry_duplicate_risk", "run_failed"}
                ],
            }
            if http_requests is not None:
                wire["physical_http_requests"] = http_requests
            if retry_sleeps is not None:
                wire["vendor_retry_sleeps"] = [{"status_code": status, "delay_seconds": delay} for status, delay in retry_sleeps]
            if name in routes:
                wire |= {"endpoint_order": routes[name], "endpoint_calls": endpoint_calls}
            if name in quota_expected:
                status, too_long = quota_expected[name]
                receipts = wire["receipts"]
                if too_long:
                    assert receipts[0]["result"]["error_code"] == "prompt_too_long"
                    assert wire["context_compacted"] == 1
                    assert [p["purpose"] for p in wire["plans"]] == ["primary", "compaction", "primary"]
                    assert wire["terminal"]["status"] == "completed"
                else:
                    attempts = 2 if status == 429 else 1
                    assert [r["result"]["error_code"] for r in receipts] == ["model_request_rejected"] * attempts
                    assert wire["physical_http_requests"] == (6 if status == 429 else 1)
                    assert wire["context_compacted"] == 0
                    assert [p["attempt"] for p in wire["plans"]] == list(range(1, attempts + 1))
                    assert all(p["purpose"] == "primary" for p in wire["plans"])
                    assert wire["terminal"]["reason"] == "model_request_rejected"
                assert not wire["unknowns"]
            wire = _historical_wire(wire)
            encoded = canonical_json_bytes(wire)
            vectors.append(
                {"wire": wire, "bytes_base64": base64.b64encode(encoded).decode(), "sha256": sha256(encoded).hexdigest()}
            )
    return {
        "historical_contract_version": "39.0.0",
        "historical_record_schema_version": 1,
        "historical_inbox_schema_version": 3,
        "schema_version": 1,
        "record_schema_version": 1,
        "inbox_schema_version": 3,
        "vectors": vectors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(produce(), indent=2, ensure_ascii=False) + "\n")
