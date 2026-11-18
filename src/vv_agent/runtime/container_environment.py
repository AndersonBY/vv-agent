"""Container-backed execution environment using an injected process provider."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from vv_agent.runtime.local_environment import (
    SearchProcess,
    StartProcess,
)
from vv_agent.runtime.processes import CapturedProcess, start_captured_process
from vv_agent.session.environments import (
    EnvironmentCapabilities,
    IsolationMode,
    NetworkMode,
)

_CONTAINER_ENVIRONMENT_CAPABILITIES = ("command", "filesystem.workspace")


class ContainerEnvironmentError(ValueError):
    """The container environment cannot provide the requested execution boundary."""

    code = "container_environment_rejected"


class ContainerEnvironmentUnavailable(ContainerEnvironmentError):
    """The configured container engine is not ready for trusted execution."""

    code = "container_environment_unavailable"


class ContainerEnvironmentPolicyError(ContainerEnvironmentError):
    """The configured container engine cannot enforce the requested policy."""

    code = "container_environment_policy_unsupported"


class ContainerEnvironmentCapabilityError(ContainerEnvironmentError):
    """The container adapter does not implement the requested operation."""

    code = "container_environment_capability_unsupported"


class ContainerEnvironmentBindingError(ContainerEnvironmentError):
    """The request does not match the attached container environment."""

    code = "environment_binding_mismatch"


class ContainerEnvironment:
    """Run an isolated command through an explicitly configured Docker/Podman CLI.

    The host supplies verified runtime facts. This adapter reports network and
    resource enforcement only when concrete engine flags and an explicit host
    declaration back the corresponding capability.
    """

    def __init__(
        self,
        *,
        engine: str,
        image: str,
        environment_id: str,
        provider_id: str,
        workspace_root: Path,
        container_workspace: str,
        generation: int = 0,
        network: NetworkMode | str = NetworkMode.NONE,
        network_policy_ref_digest: str | None = None,
        isolation_policy_ref_digest: str | None = None,
        memory_limit: str | None = None,
        cpu_limit: str | None = None,
        pids_limit: int | None = None,
        resource_limits_enforced: bool = False,
        runtime_ready: bool = False,
        process_starter: StartProcess = start_captured_process,
        executable_finder: Callable[[str], str | None] = shutil.which,
    ) -> None:
        if engine not in {"docker", "podman"}:
            raise ContainerEnvironmentUnavailable("engine must be explicitly configured as docker or podman")
        if type(runtime_ready) is not bool or not runtime_ready:
            raise ContainerEnvironmentUnavailable("container runtime readiness must be verified by the host")
        executable = executable_finder(engine)
        if not executable:
            raise ContainerEnvironmentUnavailable(f"{engine} executable is not available")
        if not isinstance(image, str) or not image.strip() or any(character.isspace() for character in image):
            raise ValueError("image must be one non-empty container image reference")
        if not isinstance(workspace_root, Path):
            raise TypeError("workspace_root must be a Path")
        if not isinstance(container_workspace, str) or not container_workspace.startswith("/") or "\x00" in container_workspace:
            raise ValueError("container_workspace must be an absolute POSIX path")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        try:
            network_mode = NetworkMode(network)
        except (TypeError, ValueError) as exc:
            raise ContainerEnvironmentPolicyError("network must be open, allowlist or none") from exc
        if network_mode is NetworkMode.ALLOWLIST:
            raise ContainerEnvironmentPolicyError("Docker/Podman flags in this adapter cannot prove allowlist enforcement")
        if network_mode is NetworkMode.NONE and network_policy_ref_digest is None:
            raise ContainerEnvironmentPolicyError("network policy 'none' requires a frozen policy reference digest")
        if type(resource_limits_enforced) is not bool:
            raise TypeError("resource_limits_enforced must be a boolean")
        if pids_limit is not None and (type(pids_limit) is not int or pids_limit <= 0):
            raise ValueError("pids_limit must be a positive integer")
        if memory_limit is not None and (not isinstance(memory_limit, str) or not memory_limit.strip()):
            raise ValueError("memory_limit must be a non-empty engine value")
        if cpu_limit is not None and (not isinstance(cpu_limit, str) or not cpu_limit.strip()):
            raise ValueError("cpu_limit must be a non-empty engine value")
        if resource_limits_enforced and not any((memory_limit, cpu_limit, pids_limit)):
            raise ContainerEnvironmentPolicyError("resource enforcement requires at least one concrete limit")

        self.provider_id = provider_id
        self.capabilities = EnvironmentCapabilities.create(
            environment_id=environment_id,
            generation=generation,
            isolation=IsolationMode.CONTAINER,
            network=network_mode,
            enforces_network_policy=network_mode is not NetworkMode.OPEN,
            enforces_resource_limits=resource_limits_enforced,
            persistent=False,
            environment_capabilities=_CONTAINER_ENVIRONMENT_CAPABILITIES,
            network_policy_ref_digest=network_policy_ref_digest,
            isolation_policy_ref_digest=isolation_policy_ref_digest,
        )
        self._executable = executable
        self._image = image
        self._workspace_root = workspace_root.resolve()
        self._container_workspace = PurePosixPath(container_workspace)
        self._network_name = "none" if network_mode is NetworkMode.NONE else "bridge"
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._pids_limit = pids_limit
        self._process_starter = process_starter

    def validate_binding(
        self,
        *,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any] | None = None,
        expected_environment_id: str | None = None,
        expected_generation: int | None = None,
        required_network_policy: NetworkMode | str | None = None,
        require_enforced_network_policy: bool = False,
        require_enforced_resource_limits: bool = False,
    ) -> None:
        """Reject every requested guarantee not proven by this runtime."""
        if require_enforced_network_policy and not self.capabilities.enforces_network_policy:
            raise ContainerEnvironmentPolicyError("container network policy is not enforced")
        if require_enforced_resource_limits and not self.capabilities.enforces_resource_limits:
            raise ContainerEnvironmentPolicyError("container resource limits are not proven")
        if required_network_policy is not None:
            try:
                required = NetworkMode(required_network_policy)
            except (TypeError, ValueError) as exc:
                raise ContainerEnvironmentBindingError("required_network_policy is not recognized") from exc
            if required is not self.capabilities.network:
                raise ContainerEnvironmentPolicyError(
                    f"container network is {self.capabilities.network.value!r}, not {required.value!r}"
                )
        if expected_environment_id is not None and expected_environment_id != self.capabilities.environment_id:
            raise ContainerEnvironmentBindingError("expected environment_id does not match the container")
        if expected_generation is not None and expected_generation != self.capabilities.generation:
            raise ContainerEnvironmentBindingError("expected generation does not match the container")
        if expected_capabilities is None:
            return
        if isinstance(expected_capabilities, Mapping):
            try:
                expected_capabilities = EnvironmentCapabilities.from_dict(cast(Mapping[str, Any], expected_capabilities))
            except ValueError as exc:
                raise ContainerEnvironmentBindingError(str(exc)) from exc
        if not isinstance(expected_capabilities, EnvironmentCapabilities):
            raise ContainerEnvironmentBindingError("expected_capabilities must be EnvironmentCapabilities")
        if expected_capabilities != self.capabilities:
            raise ContainerEnvironmentBindingError("expected capabilities do not match the container")

    def _container_cwd(self, cwd: Path) -> PurePosixPath:
        resolved = cwd.resolve()
        try:
            relative = resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ContainerEnvironmentBindingError("cwd is outside the configured container workspace") from exc
        return self._container_workspace / relative

    def _run_command(self, command: list[str], *, cwd: Path) -> tuple[list[str], Path]:
        if not command or any(not isinstance(part, str) or not part or "\x00" in part for part in command):
            raise ContainerEnvironmentError("command must be a non-empty argv vector")
        container_cwd = self._container_cwd(cwd)
        engine_command = [
            self._executable,
            "run",
            "--rm",
            "--network",
            self._network_name,
            "--mount",
            f"type=bind,src={self._workspace_root},dst={self._container_workspace}",
            "--workdir",
            container_cwd.as_posix(),
        ]
        if self._memory_limit is not None:
            engine_command.extend(("--memory", self._memory_limit))
        if self._cpu_limit is not None:
            engine_command.extend(("--cpus", self._cpu_limit))
        if self._pids_limit is not None:
            engine_command.extend(("--pids-limit", str(self._pids_limit)))
        engine_command.extend((self._image, *command))
        return engine_command, self._workspace_root

    def start_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        stdin_text: str | None = None,
        env: Mapping[str, str] | None = None,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any] | None = None,
        expected_environment_id: str | None = None,
        expected_generation: int | None = None,
        required_network_policy: NetworkMode | str | None = None,
        require_enforced_network_policy: bool = False,
        require_enforced_resource_limits: bool = False,
    ) -> CapturedProcess:
        self.validate_binding(
            expected_capabilities=expected_capabilities,
            expected_environment_id=expected_environment_id,
            expected_generation=expected_generation,
            required_network_policy=required_network_policy,
            require_enforced_network_policy=require_enforced_network_policy,
            require_enforced_resource_limits=require_enforced_resource_limits,
        )
        engine_command, host_cwd = self._run_command(command, cwd=cwd)
        return self._process_starter(engine_command, cwd=host_cwd, stdin_text=stdin_text, env=env)

    def start_search(
        self,
        command: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_stderr: bool,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any] | None = None,
        expected_environment_id: str | None = None,
        expected_generation: int | None = None,
        required_network_policy: NetworkMode | str | None = None,
        require_enforced_network_policy: bool = False,
        require_enforced_resource_limits: bool = False,
    ) -> SearchProcess:
        del (
            command,
            cwd,
            text,
            capture_stderr,
            expected_capabilities,
            expected_environment_id,
            expected_generation,
            required_network_policy,
            require_enforced_network_policy,
            require_enforced_resource_limits,
        )
        raise ContainerEnvironmentCapabilityError("container adapter does not expose streaming search")

    def search(
        self,
        command: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_stderr: bool,
    ) -> SearchProcess:
        return self.start_search(command, cwd=cwd, text=text, capture_stderr=capture_stderr)


__all__ = [
    "ContainerEnvironment",
    "ContainerEnvironmentBindingError",
    "ContainerEnvironmentCapabilityError",
    "ContainerEnvironmentError",
    "ContainerEnvironmentPolicyError",
    "ContainerEnvironmentUnavailable",
]
