"""Minimal in-process broker for mediated credential use.

The broker never adds credentials to a model request, command environment or
filesystem. A trusted environment adapter can register raw material, receive a
process-local :class:`SecretReference`, and later pass that reference to
:meth:`InMemorySecretBroker.use` with a callback that performs the authorized
operation. Credential material is only supplied to that callback.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import NoReturn, Protocol, SupportsIndex, TypeVar, runtime_checkable

from vv_agent.kernel._content import Principal

SecretResult = TypeVar("SecretResult")

_SECRET_NAME_PATTERN = re.compile(r"secret://[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*\Z")
_MAX_SECRET_NAME_LENGTH = 255
_MIN_TTL_SECONDS = 0.001


class SecretBrokerError(RuntimeError):
    """Base class for safe, classified secret-broker rejections."""

    code = "secret_broker_error"


class InvalidSecretRequest(SecretBrokerError):
    """A secret name, binding, callback or lifetime is invalid."""

    code = "invalid_secret_request"


class SecretDenied(SecretBrokerError):
    """The requested secret is unknown for this tenant or action."""

    code = "secret_denied"


class SecretBindingMismatch(SecretBrokerError):
    """A reference is being used outside its issued binding."""

    code = "secret_binding_mismatch"


class SecretAlreadyUsed(SecretBrokerError):
    """A one-shot reference has already authorized a mediated operation."""

    code = "secret_already_used"


class SecretExpired(SecretBrokerError):
    """A reference has reached its expiry deadline."""

    code = "secret_expired"


class SecretRevoked(SecretBrokerError):
    """A reference was explicitly revoked."""

    code = "secret_revoked"


class UnsupportedSecretMediation(SecretBrokerError):
    """No trusted mediated operation is registered for the service and action."""

    code = "unsupported_secret_mediation"


class SecretReference:
    """A process-local capability that cannot reveal or serialize its binding."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<opaque>)"

    __str__ = __repr__

    def __copy__(self) -> SecretReference:
        raise TypeError("SecretReference cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> SecretReference:
        del memo
        raise TypeError("SecretReference cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("SecretReference cannot be serialized")


@dataclass(frozen=True, slots=True)
class SecretBinding:
    """The complete authority checked whenever a reference is used."""

    principal: Principal
    session_id: str
    environment_id: str
    generation: int
    operation_id: str
    service: str
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal, Principal):
            raise InvalidSecretRequest("principal must be a Principal")
        _identifier(self.principal.tenant_id, "tenant_id")
        _identifier(self.principal.subject_id, "principal")
        _identifier(self.session_id, "session_id")
        _identifier(self.environment_id, "environment_id")
        _generation(self.generation)
        _identifier(self.operation_id, "operation_id")
        _identifier(self.service, "service")
        _identifier(self.action, "action")


@runtime_checkable
class SecretBroker(Protocol):
    """Host-only contract for opaque references and mediated credential use."""

    def register(
        self,
        secret_name: str,
        material: str | bytes,
        *,
        tenant_id: str,
        allowed_actions: Mapping[str, Sequence[str]],
    ) -> None: ...

    def issue(
        self,
        secret_name: str,
        *,
        principal: Principal,
        session_id: str,
        environment_id: str,
        generation: int,
        operation_id: str,
        service: str,
        action: str,
        ttl_seconds: float,
    ) -> SecretReference: ...

    def use(
        self,
        reference: SecretReference,
        *,
        principal: Principal,
        session_id: str,
        environment_id: str,
        generation: int,
        operation_id: str,
        service: str,
        action: str,
        mediated_operation: Callable[[bytes], SecretResult],
    ) -> SecretResult: ...

    def revoke(self, reference: SecretReference) -> None: ...


class _RegisteredSecret:
    __slots__ = ("allowed_actions", "material")

    def __init__(self, material: bytes, allowed_actions: dict[str, frozenset[str]]) -> None:
        self.material = material
        self.allowed_actions = allowed_actions

    def __repr__(self) -> str:
        return "_RegisteredSecret(<redacted>)"


class _IssuedSecret:
    __slots__ = ("binding", "expires_at", "material", "revoked", "used")

    def __init__(self, binding: SecretBinding, expires_at: float, material: bytes) -> None:
        self.binding = binding
        self.expires_at = expires_at
        self.material = material
        self.revoked = False
        self.used = False

    def __repr__(self) -> str:
        return "_IssuedSecret(<redacted>)"


class InMemorySecretBroker:
    """Trusted host broker with one-shot, expiring opaque references.

    ``use`` verifies the complete binding before consuming the reference. The
    callback runs outside the broker lock and intentionally cannot be reused
    through the same reference if it raises. Callback code must therefore belong
    to a trusted adapter; this module provides no raw-secret getter or generic
    credentialed request API.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._registered: dict[tuple[str, str], _RegisteredSecret] = {}
        self._issued: dict[SecretReference, _IssuedSecret] = {}
        self._lock = RLock()

    def register(
        self,
        secret_name: str,
        material: str | bytes,
        *,
        tenant_id: str,
        allowed_actions: Mapping[str, Sequence[str]],
    ) -> None:
        """Register trusted material without making the name an authorization."""

        _secret_name(secret_name)
        tenant_id = _identifier(tenant_id, "tenant_id")
        material_bytes = _material(material)
        normalized_actions = _allowed_actions(allowed_actions)
        key = (tenant_id, secret_name)
        with self._lock:
            if key in self._registered:
                raise InvalidSecretRequest("secret is already registered for this tenant")
            self._registered[key] = _RegisteredSecret(material_bytes, normalized_actions)

    def issue(
        self,
        secret_name: str,
        *,
        principal: Principal,
        session_id: str,
        environment_id: str,
        generation: int,
        operation_id: str,
        service: str,
        action: str,
        ttl_seconds: float,
    ) -> SecretReference:
        """Issue a one-shot capability for exactly one complete binding."""

        _secret_name(secret_name)
        binding = SecretBinding(
            principal=principal,
            session_id=session_id,
            environment_id=environment_id,
            generation=generation,
            operation_id=operation_id,
            service=service,
            action=action,
        )
        ttl = _ttl(ttl_seconds)
        with self._lock:
            registered = self._registered.get((principal.tenant_id, secret_name))
            if registered is None:
                raise SecretDenied("secret request is not authorized")
            if action not in registered.allowed_actions.get(service, frozenset()):
                raise UnsupportedSecretMediation("secret service/action has no trusted mediation")
            now = _now(self._clock)
            expires_at = now + ttl
            if not math.isfinite(expires_at):
                raise InvalidSecretRequest("secret reference expiry is out of range")
            reference = SecretReference()
            self._issued[reference] = _IssuedSecret(binding, expires_at, registered.material)
            return reference

    def use(
        self,
        reference: SecretReference,
        *,
        principal: Principal,
        session_id: str,
        environment_id: str,
        generation: int,
        operation_id: str,
        service: str,
        action: str,
        mediated_operation: Callable[[bytes], SecretResult],
    ) -> SecretResult:
        """Recheck the binding, consume the reference, then mediate one operation."""

        if not isinstance(reference, SecretReference):
            raise InvalidSecretRequest("reference must be a SecretReference")
        if not callable(mediated_operation):
            raise InvalidSecretRequest("mediated_operation must be callable")
        expected = SecretBinding(
            principal=principal,
            session_id=session_id,
            environment_id=environment_id,
            generation=generation,
            operation_id=operation_id,
            service=service,
            action=action,
        )
        with self._lock:
            issued = self._issued.get(reference)
            if issued is None:
                raise SecretDenied("secret reference is not active")
            if issued.binding != expected:
                raise SecretBindingMismatch("secret reference binding does not match")
            if issued.revoked:
                raise SecretRevoked("secret reference was revoked")
            if issued.used:
                raise SecretAlreadyUsed("secret reference has already been used")
            if _now(self._clock) >= issued.expires_at:
                raise SecretExpired("secret reference has expired")
            issued.used = True
            material = issued.material
        return mediated_operation(material)

    def revoke(self, reference: SecretReference) -> None:
        """Revoke an unused or consumed reference; unknown references are rejected."""

        if not isinstance(reference, SecretReference):
            raise InvalidSecretRequest("reference must be a SecretReference")
        with self._lock:
            issued = self._issued.get(reference)
            if issued is None:
                raise SecretDenied("secret reference is not active")
            issued.revoked = True


def _secret_name(value: object) -> str:
    if not isinstance(value, str) or len(value) > _MAX_SECRET_NAME_LENGTH or _SECRET_NAME_PATTERN.fullmatch(value) is None:
        raise InvalidSecretRequest("secret name must use secret:// with path-safe segments")
    return value


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidSecretRequest(f"{field_name} must be a non-empty identifier")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise InvalidSecretRequest(f"{field_name} must not contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidSecretRequest(f"{field_name} must be valid UTF-8") from exc
    return value


def _generation(value: object) -> int:
    if type(value) is not int or value < 0:
        raise InvalidSecretRequest("generation must be a nonnegative integer")
    return value


def _material(value: object) -> bytes:
    if isinstance(value, str):
        material = value.encode("utf-8")
    elif isinstance(value, bytes):
        material = bytes(value)
    else:
        raise InvalidSecretRequest("secret material must be str or bytes")
    if not material:
        raise InvalidSecretRequest("secret material must not be empty")
    return material


def _allowed_actions(value: object) -> dict[str, frozenset[str]]:
    if not isinstance(value, Mapping) or not value:
        raise InvalidSecretRequest("allowed_actions must contain at least one service")
    allowed: dict[str, frozenset[str]] = {}
    for service, actions in value.items():
        service = _identifier(service, "service")
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence) or not actions:
            raise InvalidSecretRequest("each service must contain at least one action")
        normalized = frozenset(_identifier(action, "action") for action in actions)
        if not normalized:
            raise InvalidSecretRequest("each service must contain at least one action")
        allowed[service] = normalized
    return allowed


def _ttl(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSecretRequest("ttl_seconds must be a finite positive number")
    ttl = float(value)
    if not math.isfinite(ttl) or ttl < _MIN_TTL_SECONDS:
        raise InvalidSecretRequest("ttl_seconds must be a finite positive number")
    return ttl


def _now(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSecretRequest("secret broker clock returned an invalid value")
    now = float(value)
    if not math.isfinite(now):
        raise InvalidSecretRequest("secret broker clock returned an invalid value")
    return now
