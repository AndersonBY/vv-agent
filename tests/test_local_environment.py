from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent.runtime.local_environment import (
    LocalEnvironment,
    LocalEnvironmentBindingError,
    LocalEnvironmentPolicyError,
)
from vv_agent.session.environments import EnvironmentCapabilities, IsolationMode, NetworkMode


def test_local_environment_declares_only_non_enforcing_host_capabilities() -> None:
    capabilities = LocalEnvironment().capabilities

    assert capabilities.isolation is IsolationMode.NONE
    assert capabilities.network is NetworkMode.OPEN
    assert capabilities.enforces_network_policy is False
    assert capabilities.enforces_resource_limits is False
    assert capabilities.persistent is False
    assert capabilities.capabilities_digest is not None


def test_local_environment_rejects_unsupported_policy_and_binding_requests() -> None:
    environment = LocalEnvironment(environment_id="local-test", generation=4)

    with pytest.raises(LocalEnvironmentPolicyError, match="does not enforce network policy"):
        environment.validate_binding(require_enforced_network_policy=True)
    with pytest.raises(LocalEnvironmentPolicyError, match="cannot enforce network policy 'none'"):
        environment.validate_binding(required_network_policy=NetworkMode.NONE)
    with pytest.raises(LocalEnvironmentBindingError, match="environment_id"):
        environment.validate_binding(expected_environment_id="other")
    with pytest.raises(LocalEnvironmentBindingError, match="generation"):
        environment.validate_binding(expected_generation=5)

    different_generation = EnvironmentCapabilities.create(
        environment_id="local-test",
        generation=5,
        isolation=IsolationMode.NONE,
        network=NetworkMode.OPEN,
        enforces_network_policy=False,
        enforces_resource_limits=False,
        persistent=False,
        environment_capabilities=environment.capabilities.environment_capabilities,
        network_policy_ref_digest=None,
        isolation_policy_ref_digest=None,
    )
    with pytest.raises(LocalEnvironmentBindingError, match="capabilities"):
        environment.validate_binding(expected_capabilities=different_generation)

    environment.validate_binding(
        expected_capabilities=environment.capabilities,
        expected_environment_id="local-test",
        expected_generation=4,
        required_network_policy=NetworkMode.OPEN,
    )


def test_local_environment_preserves_explicit_process_environment() -> None:
    captured: dict[str, Any] = {}

    def start(command: list[str], **kwargs: Any):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(process=object(), output_path=Path("capture.log"), started_at=1.0)

    explicit_env = {"CALLER_VALUE": "unchanged"}
    environment = LocalEnvironment(process_starter=start)

    environment.start_process(["echo", "ok"], cwd=Path.cwd(), stdin_text="input", env=explicit_env)

    assert captured["command"] == ["echo", "ok"]
    assert captured["stdin_text"] == "input"
    assert captured["env"] is explicit_env
