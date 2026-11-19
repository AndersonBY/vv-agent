"""Trusted local execution boundary for process and search starts.

The local adapter deliberately advertises only capabilities this process can
guarantee. It does not isolate commands or enforce network/resource policy.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from vv_agent.runtime.processes import CapturedProcess, start_captured_process
from vv_agent.session.environments import (
    EnvironmentCapabilities,
    IsolationMode,
    NetworkMode,
)

LOCAL_ENVIRONMENT_METADATA_KEY = "local_environment"
LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY = "local_environment_capabilities"

_LOCAL_ENVIRONMENT_ID = "local"
_LOCAL_ENVIRONMENT_PROVIDER_ID = "provider/local-test"
_LOCAL_ENVIRONMENT_GENERATION = 0
_LOCAL_ENVIRONMENT_CAPABILITIES = ("command", "find_files", "search_files")


class LocalEnvironmentError(ValueError):
    """The local environment cannot satisfy the requested execution binding."""

    code = "local_environment_rejected"


class LocalEnvironmentPolicyError(LocalEnvironmentError):
    """The request requires a policy the local environment does not enforce."""

    code = "local_environment_policy_unsupported"


class LocalEnvironmentBindingError(LocalEnvironmentError):
    """The caller's frozen environment binding differs from this adapter."""

    code = "environment_binding_mismatch"


class SearchProcess:
    """Small search-process handle that keeps subprocess details in this module."""

    __slots__ = ("_process",)

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self._process = process

    @property
    def stdout(self) -> Any:
        return self._process.stdout

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("search process did not exit before its deadline") from exc

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


@runtime_checkable
class ExecutionEnvironmentPort(Protocol):
    """Narrow W3 environment surface used by built-in local tools."""

    capabilities: EnvironmentCapabilities
    provider_id: str

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
    ) -> CapturedProcess: ...

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
    ) -> SearchProcess: ...


LocalEnvironmentPort = ExecutionEnvironmentPort
StartProcess = Callable[..., CapturedProcess]


class LocalEnvironment:
    """Adapter over the existing process supervisor and host filesystem."""

    def __init__(
        self,
        *,
        environment_id: str = _LOCAL_ENVIRONMENT_ID,
        provider_id: str = _LOCAL_ENVIRONMENT_PROVIDER_ID,
        generation: int = _LOCAL_ENVIRONMENT_GENERATION,
        process_starter: StartProcess = start_captured_process,
    ) -> None:
        self.provider_id = provider_id
        self.capabilities = EnvironmentCapabilities.create(
            environment_id=environment_id,
            generation=generation,
            isolation=IsolationMode.NONE,
            network=NetworkMode.OPEN,
            enforces_network_policy=False,
            enforces_resource_limits=False,
            persistent=False,
            environment_capabilities=_LOCAL_ENVIRONMENT_CAPABILITIES,
            network_policy_ref_digest=None,
            isolation_policy_ref_digest=None,
        )
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
        """Reject policy guarantees and frozen bindings this adapter cannot meet."""
        if require_enforced_network_policy:
            raise LocalEnvironmentPolicyError("LocalEnvironment does not enforce network policy")
        if require_enforced_resource_limits:
            raise LocalEnvironmentPolicyError("LocalEnvironment does not enforce resource limits")

        if required_network_policy is not None:
            try:
                network_policy = NetworkMode(required_network_policy)
            except (TypeError, ValueError) as exc:
                raise LocalEnvironmentBindingError("required_network_policy is not recognized") from exc
            if network_policy is not NetworkMode.OPEN:
                raise LocalEnvironmentPolicyError(f"LocalEnvironment cannot enforce network policy {network_policy.value!r}")

        if expected_environment_id is not None and expected_environment_id != self.capabilities.environment_id:
            raise LocalEnvironmentBindingError("expected environment_id does not match LocalEnvironment")
        if expected_generation is not None and expected_generation != self.capabilities.generation:
            raise LocalEnvironmentBindingError("expected generation does not match LocalEnvironment")

        if expected_capabilities is None:
            return
        if isinstance(expected_capabilities, Mapping):
            expected_capabilities = EnvironmentCapabilities.from_dict(cast(Mapping[str, Any], expected_capabilities))
        if not isinstance(expected_capabilities, EnvironmentCapabilities):
            raise LocalEnvironmentBindingError("expected_capabilities must be EnvironmentCapabilities")
        if expected_capabilities.to_dict() != self.capabilities.to_dict():
            raise LocalEnvironmentBindingError("expected capabilities do not match LocalEnvironment")

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
        return self._process_starter(command, cwd=cwd, stdin_text=stdin_text, env=env)

    def start_captured_process(
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
        return self.start_process(
            command,
            cwd=cwd,
            stdin_text=stdin_text,
            env=env,
            expected_capabilities=expected_capabilities,
            expected_environment_id=expected_environment_id,
            expected_generation=expected_generation,
            required_network_policy=required_network_policy,
            require_enforced_network_policy=require_enforced_network_policy,
            require_enforced_resource_limits=require_enforced_resource_limits,
        )

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
        self.validate_binding(
            expected_capabilities=expected_capabilities,
            expected_environment_id=expected_environment_id,
            expected_generation=expected_generation,
            required_network_policy=required_network_policy,
            require_enforced_network_policy=require_enforced_network_policy,
            require_enforced_resource_limits=require_enforced_resource_limits,
        )
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        }
        if text:
            kwargs.update(text=True, encoding="utf-8", errors="replace")
        return SearchProcess(subprocess.Popen(command, **kwargs))

    def search(
        self,
        command: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_stderr: bool,
    ) -> SearchProcess:
        return self.start_search(command, cwd=cwd, text=text, capture_stderr=capture_stderr)


_DEFAULT_LOCAL_ENVIRONMENT = LocalEnvironment()


def _metadata_sources(context: Any) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for source in (
        getattr(context, "metadata", None),
        getattr(getattr(context, "ctx", None), "metadata", None),
        getattr(context, "task_metadata", None),
    ):
        if isinstance(source, Mapping) and source not in sources:
            sources.append(source)
    return sources


def local_environment_for(
    context: Any = None,
    *,
    default: ExecutionEnvironmentPort | None = None,
) -> ExecutionEnvironmentPort:
    """Resolve a host-provided environment, falling back to the local adapter."""
    for source in _metadata_sources(context):
        candidate = source.get(LOCAL_ENVIRONMENT_METADATA_KEY)
        if candidate is None:
            continue
        if not isinstance(candidate, ExecutionEnvironmentPort):
            raise TypeError("local_environment must implement ExecutionEnvironmentPort")
        return candidate
    return default or _DEFAULT_LOCAL_ENVIRONMENT


def expected_capabilities_for(context: Any) -> EnvironmentCapabilities | None:
    for source in _metadata_sources(context):
        value = source.get(LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY)
        if value is None:
            continue
        if isinstance(value, EnvironmentCapabilities):
            return value
        if isinstance(value, Mapping):
            return EnvironmentCapabilities.from_dict(value)
        raise TypeError("local_environment_capabilities must be EnvironmentCapabilities")
    return None
