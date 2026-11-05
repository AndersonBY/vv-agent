"""Generate HTTP rejection evidence with the real kernel and deterministic model doubles."""

from __future__ import annotations

import argparse
import base64
import json
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx2

from tests.session.test_compaction import SUMMARY, configured
from tests.session.test_endpoint_routing import host
from tests.session.test_model_http_observation import local_vendor_transport
from tests.session.test_model_rejection import adapter_error, context_http_error, fail, http_error
from tests.session.test_model_rejection_retry_policy import retry_transport
from tests.session.test_model_response_completion import local_response_transport
from tests.session.test_recovery_matrix import records_of, runtime, start
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.llm.errors import ModelRequestRejected, ModelStreamInterrupted
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.output_validation import OutputValidationResult
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse


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
    routes = {}
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
    transport_cases = {}
    completion_cases = {}
    retry_cases = {}
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
        with SQLiteStore.standalone(":memory:") as store, patch("vv_agent.session.kernel.MODEL_REJECTION_BACKOFF_SECONDS", 0):
            store.install_schema()
            calls = []
            endpoint_calls = []
            rejected = fail(error, calls)
            start(store)
            if name.startswith("context_"):
                rt = configured(store, [rejected, LLMResponse(SUMMARY), LLMResponse("done")])
                rt.memory_manager.compact_threshold = 999999
            else:
                rt = (configured if compaction else runtime)(store, [rejected, LLMResponse("done") if succeeds else rejected])
            if name == "repair_private_diagnostic":
                rt = runtime(store, [LLMResponse("invalid")])
                rt.agent.output_validation_enabled = True
                rt.agent.output_validator = lambda value, context: OutputValidationResult.reject("invalid")
                rt.agent.output_repair = rejected
            http_requests = None
            retry_sleeps = None
            if name in transport_cases or name in completion_cases or name in retry_cases:
                if name in transport_cases:
                    provider, transport_case = transport_cases[name]
                    transport = local_vendor_transport(transport_case)
                elif name in retry_cases:
                    provider, retry_case, status, headers = retry_cases[name]
                    transport = retry_transport(provider, retry_case, status, headers)
                else:
                    provider, completion_case, status = completion_cases[name]
                    transport = local_response_transport(completion_case, status)
                rt = runtime(store, [])
                rt.llm = VvLlmClient([EndpointTarget("a", "synthetic-unused", "https://provider.invalid")], backend=provider)
                with transport as evidence:
                    requests, clients = (evidence[0], evidence[3] if name in retry_cases else evidence[-1])
                    if name in retry_cases:
                        retry_sleeps = evidence[2]
                    drive(store, "s", runtime=rt)
                assert all(client.is_closed for client in clients)
                http_requests = len(requests)
            elif name in routes:
                rt = host(store, {"scripted": routes[name]})

                def create(*, endpoint_id, endpoint_calls=endpoint_calls, succeeds=succeeds, calls=calls, error=error, **_kwargs):
                    def complete(**_payload):
                        endpoint_calls.append(endpoint_id)
                        if len(endpoint_calls) == 1 or not succeeds:
                            calls.append(endpoint_id)
                            raise adapter_error(
                                http_error(error.status_code, f"Denied at {endpoint_id}. Authorization: Bearer sk-secret")
                                if not isinstance(error, ModelRequestRejected)
                                else error
                            )
                        return [SimpleNamespace(content="done", tool_calls=[], usage={})]

                    return SimpleNamespace(create_completion=complete)

                with patch("vv_agent.llm.vv_llm_client.create_chat_client", create):
                    drive(store, "s", runtime=rt)
            else:
                drive(store, "s", runtime=rt)
            wire = {
                "id": name,
                "rejected_calls": len(calls),
                "plans": [
                    {"attempt": r.attempt, "purpose": r.payload["purpose"], "not_before_ms": r.payload["not_before_ms"]}
                    for r in records_of(store, "op_planned")
                ],
                "receipts": [
                    {"attempt": r.attempt, "result": r.payload["result"], "execution_started": r.payload["execution_started"]}
                    for r in records_of(store, "op_completed")
                ],
                "unknowns": [r.payload for r in records_of(store, "op_unknown")],
                "terminal": records_of(store, "turn_ended")[-1].payload,
                "context_compacted": len(records_of(store, "context_compacted")),
                "events": [
                    {k: v for k, v in e.to_dict().items() if k in {"type", "error", "error_code", "outcome", "metadata"}}
                    for e in project_records(read_state(store, "s")[1])
                    if e.type in {"model_call_failed", "model_retry_duplicate_risk", "run_failed"}
                ],
            }
            if http_requests is not None:
                wire["physical_http_requests"] = http_requests
            if retry_sleeps is not None:
                wire["vendor_retry_sleeps"] = [{"status_code": status, "delay_seconds": delay} for status, delay in retry_sleeps]
            if name in routes:
                wire |= {"endpoint_order": routes[name], "endpoint_calls": endpoint_calls}
            encoded = canonical_json_bytes(wire)
            vectors.append(
                {"wire": wire, "bytes_base64": base64.b64encode(encoded).decode(), "sha256": sha256(encoded).hexdigest()}
            )
    return {"schema_version": 1, "record_schema_version": 1, "inbox_schema_version": 3, "vectors": vectors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(produce(), indent=2, ensure_ascii=False) + "\n")
