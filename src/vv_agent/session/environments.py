"""Closed W3 values for execution environments and stable provider identity.

This module deliberately contains no execution transport. It validates the
values that an environment/provider adapter must freeze before it starts work.
The consuming adapter must resolve ``CommandSpec.cwd`` against the selected
environment, enforce that environment's workspace authority, and launch exactly
the declared command: the exact ``argv`` vector when non-empty, otherwise the
declared shell ``command_text``. It must not add a hidden environment or inject
credentials. ``secret_ref_names`` contains broker reference names only; their
authorization and materialization remain adapter concerns.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vv_agent.kernel._canonical_json import (
    MAX_WIRE_INTEGER,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_sha256,
)
from vv_agent.kernel._execution_values import execution_id_for_values
from vv_agent.kernel._provider_outcomes import (
    ProviderResultInvalid,
    validate_lookup_provider_result,
)

from .providers import Outcome


class EnvironmentContractError(ValueError):
    """A rejected environment/provider value."""

    code = "invalid_environment_contract_value"


class InvalidEnvironmentCapabilities(EnvironmentContractError):
    code = "invalid_environment_capabilities"


class InvalidCommandSpec(EnvironmentContractError):
    code = "invalid_command_spec"


class InvalidLookupBinding(EnvironmentContractError):
    code = "invalid_lookup_binding"


class InvalidProviderLookupResult(EnvironmentContractError):
    code = "invalid_provider_lookup_result"


class IsolationMode(StrEnum):
    NONE = "none"
    CONTAINER = "container"
    MICROVM = "microvm"


class NetworkMode(StrEnum):
    OPEN = "open"
    ALLOWLIST = "allowlist"
    NONE = "none"


class StdinMode(StrEnum):
    NONE = "none"
    PROVIDED = "provided"
    PIPE = "pipe"


class LookupStatus(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    EXITED = "exited"
    UNKNOWN = "unknown"


def _closed_mapping(
    value: Any,
    fields: set[str],
    error: type[EnvironmentContractError],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise error(f"{label} must be a closed object")
    return value


def _identifier(value: Any, field_name: str, error: type[EnvironmentContractError]) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise error(f"{field_name} must be a non-empty identifier without surrounding whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise error(f"{field_name} must not contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise error(f"{field_name} must be valid UTF-8 text") from exc
    return value


def _digest(value: Any, field_name: str, error: type[EnvironmentContractError]) -> str:
    try:
        return validate_sha256(value, field_name)
    except ValueError as exc:
        raise error(str(exc)) from exc


def _optional_digest(
    value: Any,
    field_name: str,
    error: type[EnvironmentContractError],
) -> str | None:
    return None if value is None else _digest(value, field_name, error)


def _integer(
    value: Any,
    field_name: str,
    error: type[EnvironmentContractError],
    *,
    minimum: int,
    maximum: int = MAX_WIRE_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise error(f"{field_name} must be an integer between {minimum} and {maximum}")
    return value


def _boolean(value: Any, field_name: str, error: type[EnvironmentContractError]) -> bool:
    if type(value) is not bool:
        raise error(f"{field_name} must be a boolean")
    return value


def _capability_names(
    value: Any,
    field_name: str,
    error: type[EnvironmentContractError],
) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise error(f"{field_name} must be a sorted unique string array")
    names = tuple(_identifier(item, field_name, error) for item in value)
    if len(names) != len(set(names)) or names != tuple(sorted(names)):
        raise error(f"{field_name} must be a sorted unique string array")
    return names


def _evidence(
    value: Any,
    field_name: str,
    error: type[EnvironmentContractError],
) -> tuple[str, ...]:
    if not isinstance(value, tuple | list) or not value:
        raise error(f"{field_name} must be a non-empty string array")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise error(f"{field_name} must contain non-empty strings")
    try:
        canonical_json_bytes(items, field_name)
    except (TypeError, ValueError) as exc:
        raise error(f"{field_name} must contain canonical JSON strings") from exc
    return items


def _canonical_json(value: Any, field_name: str, error: type[EnvironmentContractError]) -> None:
    try:
        canonical_json_bytes(value, field_name)
    except (TypeError, ValueError) as exc:
        raise error(f"{field_name} must be canonical JSON data") from exc


_COMMAND_SPEC_FIELDS = {
    "command_text",
    "argv",
    "cwd",
    "stdin_base64",
    "stdin_mode",
    "secret_ref_names",
    "limits",
}
_COMMAND_LIMIT_FIELDS = {"wall_time_ms", "output_bytes"}


def _command_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidCommandSpec(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise InvalidCommandSpec(f"{field_name} must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidCommandSpec(f"{field_name} must be valid UTF-8 text") from exc
    return value


def _command_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise InvalidCommandSpec(f"{field_name} must be an array of non-empty strings")
    strings = tuple(_command_text(item, field_name) for item in value)
    return strings


def _secret_ref_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise InvalidCommandSpec("secret_ref_names must be a unique array of non-empty strings")
    names = tuple(_identifier(item, "secret_ref_names", InvalidCommandSpec) for item in value)
    if len(names) != len(set(names)):
        raise InvalidCommandSpec("secret_ref_names must be a unique array of non-empty strings")
    return names


def _stdin_base64(value: Any, mode: StdinMode) -> str | None:
    if mode is StdinMode.PROVIDED:
        if not isinstance(value, str):
            raise InvalidCommandSpec("stdin_mode provided requires stdin_base64")
    else:
        if value is not None:
            raise InvalidCommandSpec(f"stdin_mode {mode.value} requires stdin_base64 to be null")
        return None
    try:
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise InvalidCommandSpec("stdin_base64 must be canonical base64") from exc
    if base64.b64encode(decoded) != encoded:
        raise InvalidCommandSpec("stdin_base64 must be canonical base64")
    return value


def stable_operation_key(session_id: str, operation_id: str) -> str:
    """Return the existing literal operation key used across attempts."""
    session_id = _identifier(session_id, "session_id", InvalidLookupBinding)
    operation_id = _identifier(operation_id, "operation_id", InvalidLookupBinding)
    return f"{session_id}/{operation_id}"


def execution_id_for(
    tenant_id: str,
    environment_id: str,
    operation_key: str,
) -> str:
    """Derive the stable, domain-framed execution identity.

    ``attempt``, request/call/dispatch identifiers, clocks and command text are
    intentionally absent from this framing.
    """
    tenant_id = _identifier(tenant_id, "tenant_id", InvalidLookupBinding)
    environment_id = _identifier(environment_id, "environment_id", InvalidLookupBinding)
    operation_key = _identifier(operation_key, "operation_key", InvalidLookupBinding)
    return execution_id_for_values(tenant_id, environment_id, operation_key)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command_text: str
    argv: tuple[str, ...]
    cwd: str
    stdin_base64: str | None
    stdin_mode: StdinMode
    secret_ref_names: tuple[str, ...]
    wall_time_ms: int
    output_bytes: int

    def __post_init__(self) -> None:
        try:
            mode = StdinMode(self.stdin_mode)
        except (TypeError, ValueError) as exc:
            raise InvalidCommandSpec("stdin_mode must be none, provided or pipe") from exc
        object.__setattr__(self, "command_text", _command_text(self.command_text, "command_text"))
        object.__setattr__(self, "argv", _command_strings(self.argv, "argv"))
        object.__setattr__(self, "cwd", _command_text(self.cwd, "cwd"))
        object.__setattr__(self, "stdin_base64", _stdin_base64(self.stdin_base64, mode))
        object.__setattr__(self, "stdin_mode", mode)
        object.__setattr__(self, "secret_ref_names", _secret_ref_names(self.secret_ref_names))
        object.__setattr__(
            self,
            "wall_time_ms",
            _integer(self.wall_time_ms, "limits.wall_time_ms", InvalidCommandSpec, minimum=1),
        )
        object.__setattr__(
            self,
            "output_bytes",
            _integer(self.output_bytes, "limits.output_bytes", InvalidCommandSpec, minimum=1),
        )

    @property
    def launch_command(self) -> str | tuple[str, ...]:
        """Return the exact argv or shell text selected for process launch."""
        return self.argv if self.argv else self.command_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_text": self.command_text,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "stdin_base64": self.stdin_base64,
            "stdin_mode": self.stdin_mode.value,
            "secret_ref_names": list(self.secret_ref_names),
            "limits": {
                "wall_time_ms": self.wall_time_ms,
                "output_bytes": self.output_bytes,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommandSpec:
        payload = _closed_mapping(value, _COMMAND_SPEC_FIELDS, InvalidCommandSpec, "CommandSpec")
        limits = _closed_mapping(payload["limits"], _COMMAND_LIMIT_FIELDS, InvalidCommandSpec, "limits")
        return cls(
            command_text=payload["command_text"],
            argv=payload["argv"],
            cwd=payload["cwd"],
            stdin_base64=payload["stdin_base64"],
            stdin_mode=payload["stdin_mode"],
            secret_ref_names=payload["secret_ref_names"],
            wall_time_ms=limits["wall_time_ms"],
            output_bytes=limits["output_bytes"],
        )


_ENVIRONMENT_FIELDS = {
    "environment_id",
    "generation",
    "isolation",
    "network",
    "enforces_network_policy",
    "enforces_resource_limits",
    "persistent",
    "environment_capabilities",
    "network_policy_ref_digest",
    "isolation_policy_ref_digest",
    "capabilities_digest",
}


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilities:
    environment_id: str
    generation: int
    isolation: IsolationMode
    network: NetworkMode
    enforces_network_policy: bool
    enforces_resource_limits: bool
    persistent: bool
    environment_capabilities: tuple[str, ...]
    network_policy_ref_digest: str | None
    isolation_policy_ref_digest: str | None
    capabilities_digest: str | None

    def __post_init__(self) -> None:
        environment_id = _identifier(self.environment_id, "environment_id", InvalidEnvironmentCapabilities)
        generation = _integer(self.generation, "generation", InvalidEnvironmentCapabilities, minimum=0)
        try:
            isolation = IsolationMode(self.isolation)
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("isolation must be none, container or microvm") from exc
        try:
            network = NetworkMode(self.network)
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("network must be open, allowlist or none") from exc
        enforces_network_policy = _boolean(
            self.enforces_network_policy,
            "enforces_network_policy",
            InvalidEnvironmentCapabilities,
        )
        enforces_resource_limits = _boolean(
            self.enforces_resource_limits,
            "enforces_resource_limits",
            InvalidEnvironmentCapabilities,
        )
        if network is not NetworkMode.OPEN and not enforces_network_policy:
            raise InvalidEnvironmentCapabilities("allowlist or none network mode requires enforced network policy")
        persistent = _boolean(self.persistent, "persistent", InvalidEnvironmentCapabilities)
        environment_capabilities = _capability_names(
            self.environment_capabilities,
            "environment_capabilities",
            InvalidEnvironmentCapabilities,
        )
        network_policy_ref_digest = _optional_digest(
            self.network_policy_ref_digest,
            "network_policy_ref_digest",
            InvalidEnvironmentCapabilities,
        )
        isolation_policy_ref_digest = _optional_digest(
            self.isolation_policy_ref_digest,
            "isolation_policy_ref_digest",
            InvalidEnvironmentCapabilities,
        )
        capabilities_digest = _optional_digest(
            self.capabilities_digest,
            "capabilities_digest",
            InvalidEnvironmentCapabilities,
        )
        if capabilities_digest is not None:
            expected_digest = self.compute_digest(
                environment_id=environment_id,
                generation=generation,
                isolation=isolation,
                network=network,
                enforces_network_policy=enforces_network_policy,
                enforces_resource_limits=enforces_resource_limits,
                persistent=persistent,
                environment_capabilities=environment_capabilities,
                network_policy_ref_digest=network_policy_ref_digest,
                isolation_policy_ref_digest=isolation_policy_ref_digest,
            )
            if capabilities_digest != expected_digest:
                raise InvalidEnvironmentCapabilities("capabilities_digest does not match the closed environment binding")
        object.__setattr__(self, "isolation", isolation)
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "environment_capabilities", environment_capabilities)

    @classmethod
    def compute_digest(
        cls,
        *,
        environment_id: str,
        generation: int,
        isolation: IsolationMode | str,
        network: NetworkMode | str,
        enforces_network_policy: bool,
        enforces_resource_limits: bool,
        persistent: bool,
        environment_capabilities: Sequence[str],
        network_policy_ref_digest: str | None,
        isolation_policy_ref_digest: str | None,
    ) -> str:
        capabilities = _capability_names(
            environment_capabilities,
            "environment_capabilities",
            InvalidEnvironmentCapabilities,
        )
        try:
            isolation_value = IsolationMode(isolation).value
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("isolation must be none, container or microvm") from exc
        try:
            network_value = NetworkMode(network).value
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("network must be open, allowlist or none") from exc
        return canonical_json_sha256(
            {
                "environment_id": _identifier(environment_id, "environment_id", InvalidEnvironmentCapabilities),
                "generation": _integer(
                    generation,
                    "generation",
                    InvalidEnvironmentCapabilities,
                    minimum=0,
                ),
                "isolation": isolation_value,
                "network": network_value,
                "enforces_network_policy": _boolean(
                    enforces_network_policy,
                    "enforces_network_policy",
                    InvalidEnvironmentCapabilities,
                ),
                "enforces_resource_limits": _boolean(
                    enforces_resource_limits,
                    "enforces_resource_limits",
                    InvalidEnvironmentCapabilities,
                ),
                "persistent": _boolean(persistent, "persistent", InvalidEnvironmentCapabilities),
                "environment_capabilities": list(capabilities),
                "network_policy_ref_digest": _optional_digest(
                    network_policy_ref_digest,
                    "network_policy_ref_digest",
                    InvalidEnvironmentCapabilities,
                ),
                "isolation_policy_ref_digest": _optional_digest(
                    isolation_policy_ref_digest,
                    "isolation_policy_ref_digest",
                    InvalidEnvironmentCapabilities,
                ),
            },
            "environment capabilities",
        )

    @classmethod
    def create(
        cls,
        *,
        environment_id: str,
        generation: int,
        isolation: IsolationMode | str,
        network: NetworkMode | str,
        enforces_network_policy: bool,
        enforces_resource_limits: bool,
        persistent: bool,
        environment_capabilities: Sequence[str],
        network_policy_ref_digest: str | None,
        isolation_policy_ref_digest: str | None,
    ) -> EnvironmentCapabilities:
        capabilities = _capability_names(
            environment_capabilities,
            "environment_capabilities",
            InvalidEnvironmentCapabilities,
        )
        try:
            isolation_mode = IsolationMode(isolation)
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("isolation must be none, container or microvm") from exc
        try:
            network_mode = NetworkMode(network)
        except (TypeError, ValueError) as exc:
            raise InvalidEnvironmentCapabilities("network must be open, allowlist or none") from exc
        digest = cls.compute_digest(
            environment_id=environment_id,
            generation=generation,
            isolation=isolation_mode,
            network=network_mode,
            enforces_network_policy=enforces_network_policy,
            enforces_resource_limits=enforces_resource_limits,
            persistent=persistent,
            environment_capabilities=capabilities,
            network_policy_ref_digest=network_policy_ref_digest,
            isolation_policy_ref_digest=isolation_policy_ref_digest,
        )
        return cls(
            environment_id=environment_id,
            generation=generation,
            isolation=isolation_mode,
            network=network_mode,
            enforces_network_policy=enforces_network_policy,
            enforces_resource_limits=enforces_resource_limits,
            persistent=persistent,
            environment_capabilities=capabilities,
            network_policy_ref_digest=network_policy_ref_digest,
            isolation_policy_ref_digest=isolation_policy_ref_digest,
            capabilities_digest=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "generation": self.generation,
            "isolation": self.isolation.value,
            "network": self.network.value,
            "enforces_network_policy": self.enforces_network_policy,
            "enforces_resource_limits": self.enforces_resource_limits,
            "persistent": self.persistent,
            "environment_capabilities": list(self.environment_capabilities),
            "network_policy_ref_digest": self.network_policy_ref_digest,
            "isolation_policy_ref_digest": self.isolation_policy_ref_digest,
            "capabilities_digest": self.capabilities_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvironmentCapabilities:
        payload = _closed_mapping(
            value,
            _ENVIRONMENT_FIELDS,
            InvalidEnvironmentCapabilities,
            "EnvironmentCapabilities",
        )
        environment_capabilities = _capability_names(
            payload["environment_capabilities"],
            "environment_capabilities",
            InvalidEnvironmentCapabilities,
        )
        return cls(
            environment_id=payload["environment_id"],
            generation=payload["generation"],
            isolation=payload["isolation"],
            network=payload["network"],
            enforces_network_policy=payload["enforces_network_policy"],
            enforces_resource_limits=payload["enforces_resource_limits"],
            persistent=payload["persistent"],
            environment_capabilities=environment_capabilities,
            network_policy_ref_digest=payload["network_policy_ref_digest"],
            isolation_policy_ref_digest=payload["isolation_policy_ref_digest"],
            capabilities_digest=payload["capabilities_digest"],
        )


_LOOKUP_BINDING_FIELDS = {
    "tenant_id",
    "session_id",
    "operation_id",
    "attempt",
    "request_digest",
    "idempotency_key",
    "execution_id",
    "command_digest",
    "environment_id",
    "generation",
    "provider_id",
}


@dataclass(frozen=True, slots=True)
class LookupBinding:
    tenant_id: str
    session_id: str
    operation_id: str
    attempt: int
    request_digest: str
    idempotency_key: str
    execution_id: str
    command_digest: str
    environment_id: str
    generation: int
    provider_id: str

    def __post_init__(self) -> None:
        tenant_id = _identifier(self.tenant_id, "tenant_id", InvalidLookupBinding)
        session_id = _identifier(self.session_id, "session_id", InvalidLookupBinding)
        operation_id = _identifier(self.operation_id, "operation_id", InvalidLookupBinding)
        _integer(self.attempt, "attempt", InvalidLookupBinding, minimum=1)
        _digest(self.request_digest, "request_digest", InvalidLookupBinding)
        idempotency_key = _identifier(self.idempotency_key, "idempotency_key", InvalidLookupBinding)
        expected_idempotency_key = stable_operation_key(session_id, operation_id)
        if idempotency_key != expected_idempotency_key:
            raise InvalidLookupBinding("idempotency_key must equal the stable session/operation key")
        execution_id = _digest(self.execution_id, "execution_id", InvalidLookupBinding)
        expected_execution_id = execution_id_for(tenant_id, self.environment_id, idempotency_key)
        if execution_id != expected_execution_id:
            raise InvalidLookupBinding("execution_id does not match authenticated tenant, environment and operation key")
        _digest(self.command_digest, "command_digest", InvalidLookupBinding)
        _identifier(self.environment_id, "environment_id", InvalidLookupBinding)
        _integer(self.generation, "generation", InvalidLookupBinding, minimum=0)
        _identifier(self.provider_id, "provider_id", InvalidLookupBinding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "request_digest": self.request_digest,
            "idempotency_key": self.idempotency_key,
            "execution_id": self.execution_id,
            "command_digest": self.command_digest,
            "environment_id": self.environment_id,
            "generation": self.generation,
            "provider_id": self.provider_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LookupBinding:
        payload = _closed_mapping(value, _LOOKUP_BINDING_FIELDS, InvalidLookupBinding, "LookupBinding")
        return cls(**payload)

    def assert_matches(self, expected: LookupBinding) -> None:
        """Reject any caller-selected tenant/environment/generation substitution."""
        if not isinstance(expected, LookupBinding) or self != expected:
            raise InvalidLookupBinding("lookup binding does not match the committed operation binding")


@dataclass(frozen=True, slots=True)
class ProviderLookupResult:
    binding: LookupBinding
    status: LookupStatus
    outcome: Outcome | None
    evidence: tuple[str, ...]
    feature_level: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.binding, LookupBinding):
            raise InvalidProviderLookupResult("binding must be LookupBinding")
        try:
            status = LookupStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise InvalidProviderLookupResult("lookup status is not recognized") from exc
        evidence = _evidence(self.evidence, "lookup evidence", InvalidProviderLookupResult)
        if type(self.feature_level) is not int or self.feature_level not in {1, 2, 3}:
            raise InvalidProviderLookupResult("lookup feature level is not supported")
        try:
            validate_lookup_provider_result(
                self.binding,
                status.value,
                self.outcome,
                evidence,
                feature_level=self.feature_level,
            )
        except ProviderResultInvalid as exc:
            detail = str(exc.__cause__) if exc.__cause__ is not None else str(exc)
            raise InvalidProviderLookupResult(detail) from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", evidence)

    def assert_matches(self, expected: LookupBinding) -> None:
        """Validate the result against the caller's authenticated lookup binding."""
        self.binding.assert_matches(expected)


__all__ = [
    "CommandSpec",
    "EnvironmentCapabilities",
    "EnvironmentContractError",
    "InvalidCommandSpec",
    "InvalidEnvironmentCapabilities",
    "InvalidLookupBinding",
    "InvalidProviderLookupResult",
    "IsolationMode",
    "LookupBinding",
    "LookupStatus",
    "NetworkMode",
    "ProviderLookupResult",
    "StdinMode",
    "execution_id_for",
    "stable_operation_key",
]
