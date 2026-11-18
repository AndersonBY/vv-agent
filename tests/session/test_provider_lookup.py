"""Authenticated provider lookup is a capability separate from execution polling."""

from __future__ import annotations

from typing import Any, cast

import pytest

from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.session.environments import (
    InvalidLookupBinding,
    InvalidProviderLookupResult,
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import (
    Accepted,
    Definitive,
    FunctionProvider,
    Unknown,
    lookup_provider,
)
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import ToolExecutionResult

TENANT = "tenant-a"
SESSION = "session/a"
OPERATION = "operation/start"
ENVIRONMENT = "environment/a"
PROVIDER = "provider/local-test"


def _binding(
    *,
    tenant_id: str = TENANT,
    environment_id: str = ENVIRONMENT,
) -> LookupBinding:
    operation_key = stable_operation_key(SESSION, OPERATION)
    return LookupBinding(
        tenant_id=tenant_id,
        session_id=SESSION,
        operation_id=OPERATION,
        attempt=1,
        request_digest=canonical_json_sha256({"request": 1}),
        idempotency_key=operation_key,
        execution_id=execution_id_for(tenant_id, environment_id, operation_key),
        command_digest="3" * 64,
        environment_id=environment_id,
        generation=0,
        provider_id=PROVIDER,
    )


def _result(binding: LookupBinding, status: LookupStatus) -> ProviderLookupResult:
    handle = {
        "kind": "provider",
        "provider": binding.provider_id,
        "job_id": f"job:{binding.operation_id}",
        "operation_id": binding.operation_id,
        "attempt": binding.attempt,
        "request_digest": binding.request_digest,
        "evidence": ["registry:present"],
        "query_ref": f"query:{binding.operation_id}",
        "cancel_ref": f"cancel:{binding.operation_id}",
    }
    outcomes: dict[LookupStatus, tuple[Any, tuple[str, ...]]] = {
        LookupStatus.NOT_STARTED: (None, ("registry:absent",)),
        LookupStatus.RUNNING: (
            Accepted(handle),
            ("registry:present",),
        ),
        LookupStatus.EXITED: (
            Definitive(result=ToolExecutionResult("call-1", "terminal").to_dict(), evidence=("registry:exited",), usage={}),
            ("registry:exited",),
        ),
        LookupStatus.UNKNOWN: (Unknown("registry unavailable"), ("registry:unavailable",)),
    }
    outcome, evidence = outcomes[status]
    return ProviderLookupResult(binding, status, outcome, evidence)


def test_function_provider_lookup_is_explicitly_unknown() -> None:
    binding = _binding()
    provider = FunctionProvider(ToolOrchestrator.from_tools([]))

    result = lookup_provider(provider, binding.execution_id, binding)

    assert result.status is LookupStatus.UNKNOWN
    assert isinstance(result.outcome, Unknown)
    assert "no persistent execution registry" in result.outcome.reason
    assert result.binding == binding


@pytest.mark.parametrize("status", list(LookupStatus))
def test_lookup_provider_preserves_each_existing_lookup_state(status: LookupStatus) -> None:
    binding = _binding()
    expected = _result(binding, status)

    class LookupProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, LookupBinding]] = []

        def lookup(self, execution_id: str, authenticated: LookupBinding) -> ProviderLookupResult:
            self.calls.append((execution_id, authenticated))
            return expected

    provider = LookupProvider()
    result = lookup_provider(provider, binding.execution_id, binding)

    assert result is expected
    assert result.status is status
    assert provider.calls == [(binding.execution_id, binding)]


def test_lookup_provider_rejects_a_result_from_another_binding() -> None:
    binding = _binding()
    other_binding = _binding(tenant_id="tenant-b")

    class CrossBindingProvider:
        def lookup(self, execution_id: str, authenticated: LookupBinding) -> ProviderLookupResult:
            assert execution_id == authenticated.execution_id
            return _result(other_binding, LookupStatus.UNKNOWN)

    with pytest.raises(InvalidProviderLookupResult, match="does not match") as caught:
        lookup_provider(CrossBindingProvider(), binding.execution_id, binding)

    assert isinstance(caught.value.__cause__, InvalidLookupBinding)


def test_lookup_provider_requires_execution_id_and_complete_binding() -> None:
    binding = _binding()

    class LookupProvider:
        def __init__(self) -> None:
            self.calls = 0

        def lookup(self, execution_id: str, authenticated: LookupBinding) -> ProviderLookupResult:
            self.calls += 1
            return _result(authenticated, LookupStatus.RUNNING)

    provider = LookupProvider()
    untyped_lookup = cast(Any, lookup_provider)
    with pytest.raises(TypeError):
        untyped_lookup(provider, binding.execution_id)
    with pytest.raises(InvalidProviderLookupResult, match="requires a complete LookupBinding"):
        untyped_lookup(provider, binding.execution_id, binding.to_dict())
    with pytest.raises(InvalidProviderLookupResult, match="execution_id"):
        lookup_provider(provider, "f" * 64, binding)
    assert provider.calls == 0


def test_lookup_provider_rejects_missing_capability_and_wrong_result_type() -> None:
    binding = _binding()

    class BareIdProvider:
        def lookup(self, execution_id: str, authenticated: LookupBinding) -> str:
            return execution_id

    with pytest.raises(InvalidProviderLookupResult, match="does not support"):
        lookup_provider(object(), binding.execution_id, binding)
    with pytest.raises(InvalidProviderLookupResult, match="must return ProviderLookupResult"):
        lookup_provider(BareIdProvider(), binding.execution_id, binding)


def test_lookup_provider_does_not_swallow_provider_exceptions() -> None:
    binding = _binding()
    failure = RuntimeError("registry transport failed")

    class FailingProvider:
        def lookup(self, execution_id: str, authenticated: LookupBinding) -> ProviderLookupResult:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        lookup_provider(FailingProvider(), binding.execution_id, binding)
    assert caught.value is failure
