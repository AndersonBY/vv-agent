from __future__ import annotations

import ast
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest

import vv_agent.runtime.execution_environment as execution_environment_module
from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.runtime.container_environment import ContainerEnvironment
from vv_agent.runtime.execution_environment import ExecutionEnvironment
from vv_agent.runtime.local_environment import LocalEnvironment
from vv_agent.runtime.remote_environment import RemoteBindingError, RemoteEnvironment
from vv_agent.session.environments import (
    CommandSpec,
    EnvironmentCapabilities,
    IsolationMode,
    LookupBinding,
    LookupStatus,
    NetworkMode,
    ProviderLookupResult,
    StdinMode,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import Unknown

ENVIRONMENT = "environment/remote-protocol"
PROVIDER = "provider/remote-protocol"
SESSION = "session/protocol"
OPERATION = "operation/protocol"
NETWORK_REF = "1" * 64
ISOLATION_REF = "2" * 64
NETWORK_POLICY = "3" * 64
ISOLATION_POLICY = "4" * 64


class _Transport:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, path, deepcopy(payload)))
        return deepcopy(self.responses.pop(0))


def _capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id=ENVIRONMENT,
        generation=6,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.NONE,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("command.execute", "filesystem.workspace"),
        network_policy_ref_digest=NETWORK_REF,
        isolation_policy_ref_digest=ISOLATION_REF,
    )


def _policy() -> dict[str, str]:
    return {
        "network_policy_digest": NETWORK_POLICY,
        "isolation_policy_digest": ISOLATION_POLICY,
    }


def _binding() -> LookupBinding:
    operation_key = stable_operation_key(SESSION, OPERATION)
    return LookupBinding(
        tenant_id="tenant/protocol",
        session_id=SESSION,
        operation_id=OPERATION,
        attempt=1,
        request_digest=canonical_json_sha256({"request": 1}),
        idempotency_key=operation_key,
        execution_id=execution_id_for("tenant/protocol", ENVIRONMENT, operation_key),
        command_digest="5" * 64,
        environment_id=ENVIRONMENT,
        generation=6,
        provider_id=PROVIDER,
    )


def _command() -> CommandSpec:
    return CommandSpec(
        command_text="printf protocol",
        argv=(),
        cwd="/workspace",
        stdin_base64=None,
        stdin_mode=StdinMode.NONE,
        secret_ref_names=(),
        wall_time_ms=30_000,
        output_bytes=4096,
    )


def _environment(transport: _Transport) -> RemoteEnvironment:
    return RemoteEnvironment(
        transport,
        _capabilities(),
        _policy(),
        provider_id=PROVIDER,
        persistence_verified=True,
    )


def _unknown_response(binding: LookupBinding) -> dict[str, Any]:
    return {
        "schema_version": "vv-agent.provider-lookup.v1",
        "status": "unknown",
        "binding": binding.to_dict(),
        "outcome": {"kind": "Unknown", "reason": "registry unavailable"},
        "evidence": ["remote:lookup-registry-unavailable"],
    }


def test_remote_environment_explicitly_implements_the_protocol() -> None:
    environment = _environment(_Transport())

    assert ExecutionEnvironment in RemoteEnvironment.__bases__
    assert isinstance(environment, ExecutionEnvironment)


def test_protocol_reuses_the_canonical_command_and_lookup_values() -> None:
    start_types = get_type_hints(ExecutionEnvironment.start)
    lookup_types = get_type_hints(ExecutionEnvironment.lookup)
    poll_types = get_type_hints(ExecutionEnvironment.poll)

    assert start_types["command"] is CommandSpec
    assert start_types["binding"] is LookupBinding
    assert EnvironmentCapabilities in get_args(start_types["expected_capabilities"])
    assert start_types["return"] is ProviderLookupResult
    assert lookup_types["binding"] is LookupBinding
    assert lookup_types["return"] is ProviderLookupResult
    assert poll_types["return"] is ProviderLookupResult

    module_tree = ast.parse(inspect.getsource(execution_environment_module))
    declared_classes = {node.name for node in module_tree.body if isinstance(node, ast.ClassDef)}
    duplicate_value_types = {"CommandSpec", "EnvironmentCapabilities", "LookupBinding", "ProviderLookupResult"}
    assert declared_classes.isdisjoint(duplicate_value_types)
    assert execution_environment_module.CommandSpec is CommandSpec
    assert execution_environment_module.EnvironmentCapabilities is EnvironmentCapabilities
    assert execution_environment_module.LookupBinding is LookupBinding
    assert execution_environment_module.ProviderLookupResult is ProviderLookupResult


def test_local_and_container_are_not_provider_backed_execution_environments(tmp_path: Path) -> None:
    local = LocalEnvironment()
    container = ContainerEnvironment(
        engine="docker",
        image="runtime:test",
        environment_id="environment/container",
        provider_id="provider/container",
        workspace_root=tmp_path,
        container_workspace="/workspace",
        network=NetworkMode.NONE,
        network_policy_ref_digest=NETWORK_REF,
        runtime_ready=True,
        executable_finder=lambda _engine: "/usr/bin/docker",
    )

    assert not isinstance(local, ExecutionEnvironment)
    assert not isinstance(container, ExecutionEnvironment)


def test_remote_lookup_rejects_execution_identity_substitution_before_transport() -> None:
    transport = _Transport()
    environment = _environment(transport)

    with pytest.raises(RemoteBindingError, match="execution_id"):
        environment.lookup("0" * 64, _binding())

    assert transport.calls == []


def test_remote_unknown_lookup_stays_closed_to_a_new_start() -> None:
    binding = _binding()
    transport = _Transport(_unknown_response(binding))
    environment = _environment(transport)

    lookup = environment.lookup(binding.execution_id, binding)
    blocked_start = environment.start(
        _command(),
        binding,
        expected_capabilities=_capabilities(),
        expected_operational_policy=_policy(),
    )

    assert isinstance(lookup, ProviderLookupResult)
    assert lookup.status is LookupStatus.UNKNOWN
    assert isinstance(lookup.outcome, Unknown)
    assert blocked_start.status is LookupStatus.UNKNOWN
    assert isinstance(blocked_start.outcome, Unknown)
    assert blocked_start.evidence == ("remote:start-uncertain",)
    assert [call[1] for call in transport.calls] == ["/lookup"]
