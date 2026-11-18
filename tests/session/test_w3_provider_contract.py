"""Contract 41 provider lookup keeps ordinary handles closed and binds W3 identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent.kernel._provider_outcomes import (
    Accepted,
    Definitive,
    ProviderResultInvalid,
    validate_provider_handle,
    validate_provider_result,
    validate_w3_provider_handle,
)
from vv_agent.session.environments import (
    InvalidProviderLookupResult,
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import lookup_provider
from vv_agent.types import ToolExecutionResult

REQUEST_DIGEST = "5" * 64
EXECUTION_ID = "1" * 64
COMMAND_DIGEST = "3" * 64


def _plan(*, feature_level: int) -> Any:
    payload: dict[str, Any] = {
        "provider_binding": "provider-remote-1",
        "request_digest": REQUEST_DIGEST,
        "op_kind": "tool",
        "request": {"id": "call-1"},
    }
    if feature_level == 3:
        payload.update(
            execution_identity={
                "operation_key": "sess-1/op-bg-1",
                "tenant_id": "tenant-a",
                "environment_id": "env-remote-1",
                "session_id": "sess-1",
                "operation_id": "op-bg-1",
                "execution_id": EXECUTION_ID,
            },
            command_digest=COMMAND_DIGEST,
        )
    return SimpleNamespace(
        feature_level=feature_level,
        session_id="sess-1",
        operation_id="op-bg-1",
        attempt=1,
        _payload=payload,
    )


def _handle() -> dict[str, Any]:
    return {
        "kind": "provider",
        "provider": "provider-remote-1",
        "job_id": "job-remote-1",
        "operation_id": "op-bg-1",
        "attempt": 1,
        "request_digest": REQUEST_DIGEST,
        "evidence": ["provider-registry:running"],
        "query_ref": "query-remote-1",
        "cancel_ref": "cancel-remote-1",
    }


def _binding() -> LookupBinding:
    operation_key = stable_operation_key("sess-1", "op-bg-1")
    return LookupBinding(
        tenant_id="tenant-a",
        session_id="sess-1",
        operation_id="op-bg-1",
        attempt=1,
        request_digest=REQUEST_DIGEST,
        idempotency_key=operation_key,
        execution_id=execution_id_for("tenant-a", "env-remote-1", operation_key),
        command_digest=COMMAND_DIGEST,
        environment_id="env-remote-1",
        generation=7,
        provider_id="provider-remote-1",
    )


def _w3_result(name: str) -> dict[str, Any]:
    source = os.environ.get("VV_AGENT_CONTRACT_SOURCE")
    if not source:
        pytest.skip("VV_AGENT_CONTRACT_SOURCE is not configured")
    fixture = Path(source) / "fixtures/execution_worker_w3.json"
    vectors = json.loads(fixture.read_text(encoding="utf-8"))["lookup_results"]
    return next(vector["result"] for vector in vectors if vector["name"] == name)


def test_contract_40_handle_stays_closed_and_level_three_is_explicitly_gated() -> None:
    ordinary = _handle()
    validate_provider_handle(_plan(feature_level=1), ordinary)
    validate_provider_handle(_plan(feature_level=3), ordinary)

    extended = ordinary | {
        "execution_id": EXECUTION_ID,
        "command_digest": COMMAND_DIGEST,
        "environment_id": "env-remote-1",
        "generation": 7,
    }
    with pytest.raises(ProviderResultInvalid):
        validate_provider_handle(_plan(feature_level=1), extended)

    with pytest.raises(ProviderResultInvalid):
        validate_provider_handle(_plan(feature_level=3), extended)


def test_canonical_level_three_binding_and_handle_validate_together() -> None:
    vector = _w3_result("running_reuses_original_execution")
    binding = LookupBinding.from_dict(vector["binding"])
    handle = vector["outcome"]["handle"]

    validate_w3_provider_handle(binding, handle)
    result = ProviderLookupResult(
        binding,
        LookupStatus.RUNNING,
        Accepted(handle, vector["outcome"]["yield_on_steer"]),
        tuple(vector["evidence"]),
        feature_level=3,
    )
    assert (
        lookup_provider(
            SimpleNamespace(lookup=lambda execution_id, authenticated: result),
            binding.execution_id,
            binding,
            feature_level=3,
        )
        is result
    )


def test_cross_execution_handle_is_rejected_at_the_lookup_boundary() -> None:
    vector = _w3_result("running_reuses_original_execution")
    binding = LookupBinding.from_dict(vector["binding"])
    handle = dict(vector["outcome"]["handle"])
    result = ProviderLookupResult(
        binding,
        LookupStatus.RUNNING,
        Accepted(handle, vector["outcome"]["yield_on_steer"]),
        tuple(vector["evidence"]),
        feature_level=3,
    )
    result.outcome.handle["execution_id"] = "f" * 64

    with pytest.raises(InvalidProviderLookupResult, match="provider lookup result is invalid") as caught:
        lookup_provider(
            SimpleNamespace(lookup=lambda execution_id, authenticated: result),
            binding.execution_id,
            binding,
            feature_level=3,
        )

    assert isinstance(caught.value.__cause__, ProviderResultInvalid)


def test_terminal_result_must_keep_acceptance_evidence_and_tool_result_shape() -> None:
    handle = _handle()
    result = ToolExecutionResult("call-1", "terminal").to_dict()

    with pytest.raises(ProviderResultInvalid) as caught:
        validate_provider_result(
            _plan(feature_level=1),
            Definitive(result, ("provider-process:exited",), usage={}),
            handle,
        )
    assert str(caught.value.__cause__) == "provider result lost acceptance evidence"

    binding = _binding()
    with pytest.raises(InvalidProviderLookupResult, match="tool_result_invalid") as malformed:
        ProviderLookupResult(
            binding,
            LookupStatus.EXITED,
            Definitive({"session_id": "bg_1", "status": "completed"}, ("terminal",), usage={}),
            ("terminal",),
        )
    assert isinstance(malformed.value.__cause__, ProviderResultInvalid)
