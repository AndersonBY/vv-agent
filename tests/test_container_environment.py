from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.runtime.container_environment import (
    ContainerEnvironment,
    ContainerEnvironmentCapabilityError,
    ContainerEnvironmentPolicyError,
    ContainerEnvironmentUnavailable,
)
from vv_agent.runtime.local_environment import ExecutionEnvironmentPort
from vv_agent.runtime.processes import CapturedProcess
from vv_agent.session.environments import IsolationMode, NetworkMode

NETWORK_REF = "1" * 64
ISOLATION_REF = "2" * 64


def _environment(
    tmp_path: Path,
    *,
    network: NetworkMode = NetworkMode.NONE,
    memory_limit: str | None = None,
    resource_limits_enforced: bool = False,
    process_starter: Any = None,
) -> ContainerEnvironment:
    return ContainerEnvironment(
        engine="docker",
        image="registry.example/runtime@sha256:" + "a" * 64,
        environment_id="environment/container-a",
        provider_id="provider/container-a",
        workspace_root=tmp_path,
        container_workspace="/workspace",
        generation=3,
        network=network,
        network_policy_ref_digest=NETWORK_REF if network is NetworkMode.NONE else None,
        isolation_policy_ref_digest=ISOLATION_REF,
        memory_limit=memory_limit,
        resource_limits_enforced=resource_limits_enforced,
        runtime_ready=True,
        process_starter=process_starter or (lambda *args, **kwargs: cast(CapturedProcess, SimpleNamespace())),
        executable_finder=lambda engine: f"/usr/bin/{engine}",
    )


def test_container_rejects_unavailable_or_unverified_runtime(tmp_path: Path) -> None:
    with pytest.raises(ContainerEnvironmentUnavailable, match="readiness"):
        ContainerEnvironment(
            engine="docker",
            image="runtime:test",
            environment_id="environment/container-a",
            provider_id="provider/container-a",
            workspace_root=tmp_path,
            container_workspace="/workspace",
            runtime_ready=False,
            executable_finder=lambda _engine: "/usr/bin/docker",
        )

    with pytest.raises(ContainerEnvironmentUnavailable, match="not available"):
        ContainerEnvironment(
            engine="podman",
            image="runtime:test",
            environment_id="environment/container-a",
            provider_id="provider/container-a",
            workspace_root=tmp_path,
            container_workspace="/workspace",
            runtime_ready=True,
            executable_finder=lambda _engine: None,
        )


def test_container_reports_only_engine_flags_and_explicit_resource_guarantees(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    assert isinstance(environment, ExecutionEnvironmentPort)
    assert environment.capabilities.isolation is IsolationMode.CONTAINER
    assert environment.capabilities.network is NetworkMode.NONE
    assert environment.capabilities.enforces_network_policy is True
    assert environment.capabilities.enforces_resource_limits is False
    assert environment.capabilities.persistent is False
    assert "search_files" not in environment.capabilities.environment_capabilities
    with pytest.raises(ContainerEnvironmentPolicyError, match="not proven"):
        environment.validate_binding(require_enforced_resource_limits=True)


def test_container_resource_capability_requires_a_concrete_limit(tmp_path: Path) -> None:
    with pytest.raises(ContainerEnvironmentPolicyError, match="at least one"):
        _environment(tmp_path, resource_limits_enforced=True)

    environment = _environment(
        tmp_path,
        memory_limit="512m",
        resource_limits_enforced=True,
    )

    assert environment.capabilities.enforces_resource_limits is True
    environment.validate_binding(require_enforced_resource_limits=True)


def test_container_does_not_claim_allowlist_network_enforcement(tmp_path: Path) -> None:
    with pytest.raises(ContainerEnvironmentPolicyError, match="allowlist"):
        _environment(tmp_path, network=NetworkMode.ALLOWLIST)


def test_container_starts_via_engine_and_rejects_unadvertised_search(tmp_path: Path) -> None:
    workdir = tmp_path / "project"
    workdir.mkdir()
    captured: dict[str, Any] = {}

    def start(command: list[str], **kwargs: Any):
        captured.update(command=command, **kwargs)
        return cast(CapturedProcess, SimpleNamespace())

    environment = _environment(
        tmp_path,
        memory_limit="256m",
        resource_limits_enforced=True,
        process_starter=start,
    )
    environment.start_process(["printf", "ok"], cwd=workdir)

    assert captured["command"][:4] == ["/usr/bin/docker", "run", "--rm", "--network"]
    assert captured["command"][4] == "none"
    assert "--memory" in captured["command"]
    assert captured["command"][-3:] == ["registry.example/runtime@sha256:" + "a" * 64, "printf", "ok"]
    with pytest.raises(ContainerEnvironmentCapabilityError, match="search"):
        environment.start_search(["rg", "needle"], cwd=workdir, text=True, capture_stderr=False)
