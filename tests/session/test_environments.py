"""W3 environment values bind stable execution identity without executing work."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.session.environments import (
    EnvironmentCapabilities,
    InvalidEnvironmentCapabilities,
    InvalidLookupBinding,
    InvalidProviderLookupResult,
    IsolationMode,
    LookupBinding,
    LookupStatus,
    NetworkMode,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import Accepted, Definitive, Unknown
from vv_agent.types import ToolExecutionResult

TENANT = "tenant-a"
SESSION = "session/a"
OPERATION = "operation/start"
ENVIRONMENT = "environment/a"
PROVIDER = "provider/local-test"
NETWORK_REF = "1" * 64
ISOLATION_REF = "2" * 64


def _environment(
    *,
    environment_id: str = ENVIRONMENT,
    generation: int = 0,
    isolation: IsolationMode = IsolationMode.CONTAINER,
    network: NetworkMode = NetworkMode.ALLOWLIST,
) -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id=environment_id,
        generation=generation,
        isolation=isolation,
        network=network,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("filesystem.read", "network.allowlist"),
        network_policy_ref_digest=NETWORK_REF,
        isolation_policy_ref_digest=ISOLATION_REF,
    )


def _binding(
    *,
    tenant_id: str = TENANT,
    session_id: str = SESSION,
    operation_id: str = OPERATION,
    attempt: int = 1,
    request: Any = 1,
    environment_id: str = ENVIRONMENT,
    generation: int = 0,
    execution_id: str | None = None,
    provider_id: str = PROVIDER,
) -> LookupBinding:
    operation_key = stable_operation_key(session_id, operation_id)
    return LookupBinding(
        tenant_id=tenant_id,
        session_id=session_id,
        operation_id=operation_id,
        attempt=attempt,
        request_digest=canonical_json_sha256(request),
        idempotency_key=operation_key,
        execution_id=execution_id or execution_id_for(tenant_id, environment_id, operation_key),
        command_digest="3" * 64,
        environment_id=environment_id,
        generation=generation,
        provider_id=provider_id,
    )


def _provider_handle(binding: LookupBinding) -> dict[str, Any]:
    return {
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


def test_environment_capabilities_are_deterministic_closed_and_digest_checked() -> None:
    first = _environment()
    second = _environment()

    assert first == second
    assert first.capabilities_digest == second.capabilities_digest
    assert first.environment_capabilities == ("filesystem.read", "network.allowlist")
    assert EnvironmentCapabilities.from_dict(first.to_dict()) == first
    assert first.to_dict()["capabilities_digest"] == first.capabilities_digest

    payload = first.to_dict()
    payload["generation"] = 1
    with pytest.raises(InvalidEnvironmentCapabilities, match="capabilities_digest"):
        EnvironmentCapabilities.from_dict(payload)

    payload = first.to_dict() | {"unknown": True}
    with pytest.raises(InvalidEnvironmentCapabilities, match="closed object"):
        EnvironmentCapabilities.from_dict(payload)

    with pytest.raises(InvalidEnvironmentCapabilities, match="sorted unique"):
        _environment_with_capabilities(("network.allowlist", "filesystem.read"))
    with pytest.raises(InvalidEnvironmentCapabilities, match="sorted unique"):
        _environment_with_capabilities(("filesystem.read", "filesystem.read"))


def _environment_with_capabilities(capabilities: tuple[str, ...]) -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id=ENVIRONMENT,
        generation=0,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.ALLOWLIST,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=capabilities,
        network_policy_ref_digest=NETWORK_REF,
        isolation_policy_ref_digest=ISOLATION_REF,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", True),
        ("isolation", "host"),
        ("network", "egress"),
        ("enforces_network_policy", 1),
        ("persistent", "true"),
        ("environment_capabilities", "filesystem.read"),
        ("network_policy_ref_digest", "A" * 64),
    ],
)
def test_environment_capabilities_reject_wrong_field_domains(field: str, value: Any) -> None:
    payload = _environment().to_dict()
    payload[field] = value

    with pytest.raises(InvalidEnvironmentCapabilities):
        EnvironmentCapabilities.from_dict(payload)


def test_environment_capabilities_reject_unenforced_network_restriction() -> None:
    payload = _environment().to_dict()
    payload["enforces_network_policy"] = False
    payload["capabilities_digest"] = EnvironmentCapabilities.compute_digest(
        environment_id=payload["environment_id"],
        generation=payload["generation"],
        isolation=payload["isolation"],
        network=payload["network"],
        enforces_network_policy=False,
        enforces_resource_limits=payload["enforces_resource_limits"],
        persistent=payload["persistent"],
        environment_capabilities=payload["environment_capabilities"],
        network_policy_ref_digest=payload["network_policy_ref_digest"],
        isolation_policy_ref_digest=payload["isolation_policy_ref_digest"],
    )

    with pytest.raises(InvalidEnvironmentCapabilities, match="requires enforced network policy"):
        EnvironmentCapabilities.from_dict(payload)


def test_environment_capabilities_accept_open_network_and_null_policy_hashes() -> None:
    environment = EnvironmentCapabilities.create(
        environment_id=ENVIRONMENT,
        generation=0,
        isolation=IsolationMode.NONE,
        network=NetworkMode.OPEN,
        enforces_network_policy=False,
        enforces_resource_limits=False,
        persistent=False,
        environment_capabilities=("filesystem.read",),
        network_policy_ref_digest=None,
        isolation_policy_ref_digest=None,
    )

    assert environment.network_policy_ref_digest is None
    assert environment.isolation_policy_ref_digest is None
    assert environment.capabilities_digest is not None
    assert EnvironmentCapabilities.from_dict(environment.to_dict()) == environment

    payload = environment.to_dict()
    payload["capabilities_digest"] = None
    unchecked = EnvironmentCapabilities.from_dict(payload)
    assert unchecked.capabilities_digest is None

    payload["environment_id"] = "environment/without-digest"
    assert EnvironmentCapabilities.from_dict(payload).environment_id == "environment/without-digest"


@pytest.mark.parametrize("network", [NetworkMode.ALLOWLIST, NetworkMode.NONE])
def test_restricted_network_modes_require_enforcement(network: NetworkMode) -> None:
    with pytest.raises(InvalidEnvironmentCapabilities, match="requires enforced network policy"):
        EnvironmentCapabilities(
            environment_id=ENVIRONMENT,
            generation=0,
            isolation=IsolationMode.CONTAINER,
            network=network,
            enforces_network_policy=False,
            enforces_resource_limits=True,
            persistent=False,
            environment_capabilities=(),
            network_policy_ref_digest=None,
            isolation_policy_ref_digest=None,
            capabilities_digest=None,
        )


def test_execution_id_is_deterministic_and_uses_only_authenticated_stable_inputs() -> None:
    operation_key = stable_operation_key(SESSION, OPERATION)
    first = execution_id_for(TENANT, ENVIRONMENT, operation_key)

    assert first == execution_id_for(TENANT, ENVIRONMENT, operation_key)
    assert first != execution_id_for("tenant-b", ENVIRONMENT, operation_key)
    assert first != execution_id_for(TENANT, "environment/b", operation_key)
    assert first != execution_id_for(TENANT, ENVIRONMENT, stable_operation_key(SESSION, "operation/other"))
    assert _binding().execution_id == first


def test_attempt_and_request_do_not_change_stable_execution_id() -> None:
    first = _binding(attempt=1, request={"request": 1})
    retry = _binding(attempt=2, request={"request": 2})

    assert first.attempt != retry.attempt
    assert first.request_digest != retry.request_digest
    assert first.execution_id == retry.execution_id
    assert first.idempotency_key == retry.idempotency_key


def test_lookup_binding_rejects_cross_tenant_environment_and_generation_bindings() -> None:
    committed = _binding()
    committed.assert_matches(_binding())

    with pytest.raises(InvalidLookupBinding, match="does not match"):
        _binding(tenant_id="tenant-b").assert_matches(committed)
    with pytest.raises(InvalidLookupBinding, match="does not match"):
        _binding(environment_id="environment/b").assert_matches(committed)
    with pytest.raises(InvalidLookupBinding, match="does not match"):
        _binding(generation=1).assert_matches(committed)


def test_lookup_binding_rejects_random_wrong_and_override_values() -> None:
    with pytest.raises(InvalidLookupBinding, match="lowercase SHA-256"):
        _binding(execution_id=uuid4().hex)

    with pytest.raises(InvalidLookupBinding, match="does not match authenticated"):
        _binding(execution_id="f" * 64)

    payload = _binding().to_dict()
    payload["idempotency_key"] = "caller-selected"
    with pytest.raises(InvalidLookupBinding, match="stable session/operation key"):
        LookupBinding.from_dict(payload)

    payload = _binding().to_dict()
    payload["request_digest"] = "NOT-A-DIGEST"
    with pytest.raises(InvalidLookupBinding, match="lowercase SHA-256"):
        LookupBinding.from_dict(payload)


def test_provider_lookup_reuses_provider_outcomes_and_validates_state() -> None:
    binding = _binding()
    not_started = ProviderLookupResult(binding, LookupStatus.NOT_STARTED, None, ("registry:absent",))
    running = ProviderLookupResult(
        binding,
        LookupStatus.RUNNING,
        Accepted(_provider_handle(binding)),
        ("registry:present",),
    )
    exited = ProviderLookupResult(
        binding,
        LookupStatus.EXITED,
        Definitive(ToolExecutionResult("call-1", "terminal").to_dict(), evidence=("registry:exited",), usage={}),
        ("registry:exited",),
    )
    unknown = ProviderLookupResult(
        binding,
        LookupStatus.UNKNOWN,
        Unknown("registry unavailable"),
        ("registry:unavailable",),
    )

    assert not_started.outcome is None
    assert isinstance(running.outcome, Accepted)
    assert isinstance(exited.outcome, Definitive)
    assert isinstance(unknown.outcome, Unknown)
    for result in (not_started, running, exited, unknown):
        result.assert_matches(binding)


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (LookupStatus.NOT_STARTED, Unknown("not absent")),
        (LookupStatus.RUNNING, Unknown("not running")),
        (LookupStatus.EXITED, Accepted(handle={})),
        (LookupStatus.UNKNOWN, Definitive(result={}, evidence=("terminal",))),
        ("started", None),
    ],
)
def test_provider_lookup_rejects_status_outcome_mismatch(status: Any, outcome: Any) -> None:
    with pytest.raises(InvalidProviderLookupResult):
        ProviderLookupResult(_binding(), status, outcome, ("evidence",))


def test_provider_lookup_rejects_handle_attempt_substitution() -> None:
    binding = _binding()
    handle = _provider_handle(binding)
    handle["attempt"] = binding.attempt + 1
    with pytest.raises(InvalidProviderLookupResult, match="provider_result_invalid"):
        ProviderLookupResult(
            binding,
            LookupStatus.RUNNING,
            Accepted(handle),
            ("registry:present",),
        )
