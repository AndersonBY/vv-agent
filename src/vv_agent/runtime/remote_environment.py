"""Minimal authenticated remote execution environment over HTTP.

The adapter binds every command lifecycle request to the W3 ``LookupBinding``.
The remote service owns durable deduplication and process supervision; this
module only provides a small standard-library transport seam and strict result
validation.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Mapping
from typing import Any, NoReturn, Protocol, cast, runtime_checkable
from urllib import error, parse, request

from vv_agent.kernel._canonical_json import validate_sha256
from vv_agent.kernel._provider_outcomes import (
    Accepted,
    Definitive,
    ProviderResultInvalid,
    Unknown,
    validate_lookup_provider_handle,
)
from vv_agent.runtime.execution_environment import ExecutionEnvironment
from vv_agent.session.environments import (
    CommandSpec,
    EnvironmentCapabilities,
    InvalidProviderLookupResult,
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
)

_SCHEMA_VERSION = "vv-agent.remote-environment.v1"
_LOOKUP_SCHEMA_VERSION = "vv-agent.provider-lookup.v1"
_MAX_RESPONSE_BYTES = 1024 * 1024
_POLICY_FIELDS = {"network_policy_digest", "isolation_policy_digest"}
_LOOKUP_FIELDS = {"schema_version", "status", "binding", "outcome", "evidence"}
_ACCEPTED_FIELDS = {"kind", "handle", "yield_on_steer"}
_DEFINITIVE_FIELDS = {"kind", "result", "usage", "shared_state", "evidence"}
_UNKNOWN_FIELDS = {"kind", "reason"}
_UNSUPPORTED_REMOTE_CAPABILITIES = {"search", "search_files"}


class RemoteEnvironmentError(RuntimeError):
    """Base error for a remote protocol or configuration failure."""

    code = "remote_environment_error"


class RemoteBindingError(ValueError):
    """The frozen request binding does not match the attached environment."""

    code = "environment_binding_mismatch"


class RemoteProtocolError(RemoteEnvironmentError):
    """The remote service returned an invalid or ambiguously bound response."""

    code = "remote_protocol_invalid"


class RemoteUnsupportedOperation(RemoteEnvironmentError):
    """The remote protocol does not implement the requested operation."""

    code = "remote_capability_unsupported"


class RemoteRequestRejected(RemoteEnvironmentError):
    """The remote service explicitly rejected an authenticated request."""

    code = "remote_request_rejected"


class RemoteTransportUnavailable(RemoteEnvironmentError):
    """No trustworthy response was available from the remote service."""

    code = "remote_transport_unavailable"


@runtime_checkable
class RemoteTransport(Protocol):
    """One JSON request/response seam used by ``RemoteEnvironment``."""

    def request(self, method: str, path: str, payload: dict[str, Any]) -> Mapping[str, Any]: ...


class UrllibRemoteTransport:
    """Small standard-library HTTP implementation of ``RemoteTransport``."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be greater than 0 and at most 300")
        parsed = parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an absolute http or https URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not contain credentials")
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._headers = dict(headers or {})

    def request(self, method: str, path: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        if method != "POST" or not path.startswith("/") or ".." in path.split("/"):
            raise ValueError("remote requests require POST and a fixed absolute path")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json", **self._headers}
        http_request = request.Request(
            f"{self._endpoint}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            if exc.code in {408, 425, 429} or exc.code >= 500:
                raise RemoteTransportUnavailable(f"remote HTTP status {exc.code} is ambiguous") from exc
            raise RemoteRequestRejected(f"remote service rejected the request with HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise RemoteTransportUnavailable("remote transport was unavailable") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise RemoteTransportUnavailable("remote response exceeded the size limit")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteProtocolError("remote response is not valid JSON") from exc
        if type(decoded) is not dict:
            raise RemoteProtocolError("remote response must be a JSON object")
        return decoded


def _policy(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
        raise RemoteBindingError("operational_policy must contain the two frozen policy digests")
    result: dict[str, str] = {}
    for field_name in sorted(_POLICY_FIELDS):
        digest = value[field_name]
        try:
            result[field_name] = validate_sha256(digest, f"operational_policy.{field_name}")
        except ValueError as exc:
            raise RemoteBindingError(str(exc)) from exc
    return result


def _evidence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RemoteProtocolError("remote lookup evidence must be an array")
    evidence = tuple(value)
    if not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise RemoteProtocolError("remote lookup evidence must contain non-empty strings")
    return evidence


class RemoteEnvironment(ExecutionEnvironment):
    """Provider-backed environment whose durable state belongs to a remote agent."""

    def __init__(
        self,
        transport: RemoteTransport,
        capabilities: EnvironmentCapabilities,
        operational_policy: Mapping[str, Any],
        *,
        provider_id: str,
        persistence_verified: bool = False,
    ) -> None:
        if not isinstance(transport, RemoteTransport):
            raise TypeError("transport must implement RemoteTransport")
        if not isinstance(capabilities, EnvironmentCapabilities):
            raise TypeError("capabilities must be EnvironmentCapabilities")
        if capabilities.capabilities_digest is None:
            raise RemoteBindingError("remote capabilities_digest is required")
        if type(persistence_verified) is not bool:
            raise TypeError("persistence_verified must be a boolean")
        if capabilities.persistent and not persistence_verified:
            raise RemoteBindingError("remote persistence must be verified by the host before it is advertised")
        unsupported = _UNSUPPORTED_REMOTE_CAPABILITIES.intersection(capabilities.environment_capabilities)
        if unsupported:
            raise RemoteUnsupportedOperation(
                f"remote adapter does not implement advertised capability {sorted(unsupported)[0]!r}"
            )
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        self.provider_id = provider_id
        self.capabilities = capabilities
        self.operational_policy = _policy(operational_policy)
        self._transport = transport
        self._uncertain_starts: set[tuple[str, str, str]] = set()
        self._uncertain_lock = threading.Lock()

    @staticmethod
    def _uncertain_key(binding: LookupBinding) -> tuple[str, str, str]:
        return (binding.tenant_id, binding.environment_id, binding.idempotency_key)

    def _set_start_state(self, binding: LookupBinding, status: LookupStatus) -> None:
        key = self._uncertain_key(binding)
        with self._uncertain_lock:
            if status in {LookupStatus.NOT_STARTED, LookupStatus.RUNNING, LookupStatus.EXITED}:
                self._uncertain_starts.discard(key)
            else:
                self._uncertain_starts.add(key)

    def _start_is_uncertain(self, binding: LookupBinding) -> bool:
        with self._uncertain_lock:
            return self._uncertain_key(binding) in self._uncertain_starts

    def validate_binding(
        self,
        binding: LookupBinding,
        *,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any],
        expected_operational_policy: Mapping[str, Any],
    ) -> tuple[EnvironmentCapabilities, dict[str, str]]:
        """Compare a durable frozen tuple with this attached remote incarnation."""
        if not isinstance(binding, LookupBinding):
            raise RemoteBindingError("binding must be a frozen LookupBinding")
        if isinstance(expected_capabilities, Mapping):
            try:
                expected_capabilities = EnvironmentCapabilities.from_dict(cast(Mapping[str, Any], expected_capabilities))
            except ValueError as exc:
                raise RemoteBindingError(str(exc)) from exc
        if not isinstance(expected_capabilities, EnvironmentCapabilities):
            raise RemoteBindingError("expected_capabilities must be EnvironmentCapabilities")
        policy = _policy(expected_operational_policy)
        if expected_capabilities != self.capabilities:
            raise RemoteBindingError("frozen capabilities do not match the attached remote environment")
        if policy != self.operational_policy:
            raise RemoteBindingError("frozen operational policy does not match the attached remote environment")
        if binding.provider_id != self.provider_id:
            raise RemoteBindingError("binding provider_id does not match RemoteEnvironment")
        if binding.environment_id != expected_capabilities.environment_id:
            raise RemoteBindingError("binding environment_id does not match frozen capabilities")
        if binding.generation != expected_capabilities.generation:
            raise RemoteBindingError("binding generation does not match frozen capabilities")
        return expected_capabilities, policy

    @staticmethod
    def _validate_handle(binding: LookupBinding, handle: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(handle, Mapping):
            raise RemoteBindingError("command handle must be an object")
        candidate = dict(handle)
        try:
            validate_lookup_provider_handle(binding, candidate, feature_level=3)
        except ProviderResultInvalid as exc:
            raise RemoteBindingError("command handle does not match the frozen execution binding") from exc
        return candidate

    def _request_identity(
        self,
        operation: str,
        binding: LookupBinding,
        capabilities: EnvironmentCapabilities,
        policy: dict[str, str],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "operation": operation,
            "binding": binding.to_dict(),
            "environment": capabilities.to_dict(),
            "operational_policy": policy,
            **extra,
        }

    def _unknown(
        self,
        binding: LookupBinding,
        *,
        reason: str,
        evidence: str,
    ) -> ProviderLookupResult:
        return ProviderLookupResult(
            binding,
            LookupStatus.UNKNOWN,
            Unknown(reason),
            (evidence,),
            feature_level=3,
        )

    def start(
        self,
        command: CommandSpec,
        binding: LookupBinding,
        *,
        expected_capabilities: EnvironmentCapabilities | Mapping[str, Any],
        expected_operational_policy: Mapping[str, Any],
    ) -> ProviderLookupResult:
        """Submit one command exactly once under its stable execution identity."""
        if not isinstance(command, CommandSpec):
            raise TypeError("command must be CommandSpec")
        capabilities, policy = self.validate_binding(
            binding,
            expected_capabilities=expected_capabilities,
            expected_operational_policy=expected_operational_policy,
        )
        if self._start_is_uncertain(binding):
            return self._unknown(
                binding,
                reason="a prior start has no proven execution state; query the original execution",
                evidence="remote:start-uncertain",
            )
        payload = self._request_identity(
            "start",
            binding,
            capabilities,
            policy,
            command=command.to_dict(),
        )
        try:
            response = self._transport.request("POST", "/start", payload)
        except (RemoteTransportUnavailable, TimeoutError, OSError):
            self._set_start_state(binding, LookupStatus.UNKNOWN)
            return self._unknown(
                binding,
                reason="remote start acknowledgment was unavailable; the process may have started",
                evidence="remote:start-transport-unavailable",
            )
        result = self._decode_result(
            response,
            binding,
            allowed_statuses={LookupStatus.RUNNING, LookupStatus.EXITED, LookupStatus.UNKNOWN},
        )
        self._set_start_state(binding, result.status)
        return result

    def lookup(self, execution_id: str, binding: LookupBinding) -> ProviderLookupResult:
        """Recover the original execution without issuing a replacement start."""
        if not isinstance(binding, LookupBinding):
            raise RemoteBindingError("binding must be a frozen LookupBinding")
        if execution_id != binding.execution_id:
            raise RemoteBindingError("lookup execution_id does not match the authenticated binding")
        if binding.provider_id != self.provider_id:
            raise RemoteBindingError("lookup provider_id does not match RemoteEnvironment")
        payload = self._request_identity("lookup", binding, self.capabilities, self.operational_policy)
        try:
            response = self._transport.request("POST", "/lookup", payload)
        except (RemoteTransportUnavailable, TimeoutError, OSError):
            self._set_start_state(binding, LookupStatus.UNKNOWN)
            return self._unknown(
                binding,
                reason="remote execution registry is unavailable",
                evidence="remote:lookup-transport-unavailable",
            )
        result = self._decode_result(response, binding, allowed_statuses=set(LookupStatus))
        self._set_start_state(binding, result.status)
        return result

    def poll(self, handle: Mapping[str, Any], binding: LookupBinding) -> ProviderLookupResult:
        """Poll an already accepted execution; this method never submits a start."""
        capabilities, policy = self.validate_binding(
            binding,
            expected_capabilities=self.capabilities,
            expected_operational_policy=self.operational_policy,
        )
        payload = self._request_identity(
            "poll",
            binding,
            capabilities,
            policy,
            handle=self._validate_handle(binding, handle),
        )
        try:
            response = self._transport.request("POST", "/poll", payload)
        except (RemoteTransportUnavailable, TimeoutError, OSError):
            return self._unknown(
                binding,
                reason="remote poll was unavailable; execution state remains unproved",
                evidence="remote:poll-transport-unavailable",
            )
        return self._decode_result(
            response,
            binding,
            allowed_statuses={LookupStatus.RUNNING, LookupStatus.EXITED, LookupStatus.UNKNOWN},
        )

    def write_stdin(
        self,
        handle: Mapping[str, Any],
        data: bytes,
        binding: LookupBinding,
    ) -> ProviderLookupResult:
        """Write stdin once; an unacknowledged write is never replayed."""
        if not isinstance(data, bytes):
            raise TypeError("stdin data must be bytes")
        capabilities, policy = self.validate_binding(
            binding,
            expected_capabilities=self.capabilities,
            expected_operational_policy=self.operational_policy,
        )
        payload = self._request_identity(
            "write_stdin",
            binding,
            capabilities,
            policy,
            handle=self._validate_handle(binding, handle),
            stdin_base64=base64.b64encode(data).decode("ascii"),
        )
        try:
            response = self._transport.request("POST", "/stdin", payload)
        except (RemoteTransportUnavailable, TimeoutError, OSError):
            return self._unknown(
                binding,
                reason="remote stdin acknowledgment was unavailable; the write was not replayed",
                evidence="remote:stdin-transport-unavailable",
            )
        return self._decode_result(
            response,
            binding,
            allowed_statuses={LookupStatus.RUNNING, LookupStatus.EXITED, LookupStatus.UNKNOWN},
        )

    def stop(self, handle: Mapping[str, Any], binding: LookupBinding) -> ProviderLookupResult:
        """Request stop for the original execution and return only proven state."""
        capabilities, policy = self.validate_binding(
            binding,
            expected_capabilities=self.capabilities,
            expected_operational_policy=self.operational_policy,
        )
        payload = self._request_identity(
            "stop",
            binding,
            capabilities,
            policy,
            handle=self._validate_handle(binding, handle),
        )
        try:
            response = self._transport.request("POST", "/stop", payload)
        except (RemoteTransportUnavailable, TimeoutError, OSError):
            return self._unknown(
                binding,
                reason="remote stop acknowledgment was unavailable; termination is unconfirmed",
                evidence="remote:stop-transport-unavailable",
            )
        return self._decode_result(
            response,
            binding,
            allowed_statuses={LookupStatus.RUNNING, LookupStatus.EXITED, LookupStatus.UNKNOWN},
        )

    def search(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise RemoteUnsupportedOperation("the minimal remote protocol does not define search")

    def _decode_result(
        self,
        payload: Mapping[str, Any],
        binding: LookupBinding,
        *,
        allowed_statuses: set[LookupStatus],
    ) -> ProviderLookupResult:
        if not isinstance(payload, Mapping) or set(payload) != _LOOKUP_FIELDS:
            raise RemoteProtocolError("remote lookup result must be a closed object")
        if payload["schema_version"] != _LOOKUP_SCHEMA_VERSION:
            raise RemoteProtocolError("remote lookup schema_version is not recognized")
        try:
            status = LookupStatus(payload["status"])
        except (TypeError, ValueError) as exc:
            raise RemoteProtocolError("remote lookup status is not recognized") from exc
        if status not in allowed_statuses:
            raise RemoteProtocolError(f"remote status {status.value!r} is invalid for this operation")
        try:
            response_binding = LookupBinding.from_dict(payload["binding"])
        except (TypeError, ValueError) as exc:
            raise RemoteProtocolError("remote lookup binding is malformed") from exc
        if response_binding != binding:
            raise RemoteProtocolError("remote lookup binding does not match the authenticated request")
        outcome = self._decode_outcome(payload["outcome"], status)
        try:
            result = ProviderLookupResult(
                binding,
                status,
                outcome,
                _evidence(payload["evidence"]),
                feature_level=3,
            )
        except (InvalidProviderLookupResult, ProviderResultInvalid) as exc:
            raise RemoteProtocolError("remote lookup result failed strict identity validation") from exc
        return result

    @staticmethod
    def _decode_outcome(value: Any, status: LookupStatus) -> Accepted | Definitive | Unknown | None:
        if status is LookupStatus.NOT_STARTED:
            if value is not None:
                raise RemoteProtocolError("not_started lookup must not carry an outcome")
            return None
        if not isinstance(value, Mapping):
            raise RemoteProtocolError("remote lookup outcome must be an object")
        if status is LookupStatus.RUNNING:
            if set(value) != _ACCEPTED_FIELDS or value["kind"] != "Accepted":
                raise RemoteProtocolError("running lookup requires a closed Accepted outcome")
            if type(value["yield_on_steer"]) is not bool or not isinstance(value["handle"], Mapping):
                raise RemoteProtocolError("Accepted outcome fields are invalid")
            return Accepted(dict(value["handle"]), value["yield_on_steer"])
        if status is LookupStatus.EXITED:
            if (
                set(value) != _DEFINITIVE_FIELDS
                or value["kind"] != "Definitive"
                or not isinstance(value["result"], Mapping)
                or not isinstance(value["usage"], Mapping)
                or (value["shared_state"] is not None and not isinstance(value["shared_state"], Mapping))
            ):
                raise RemoteProtocolError("exited lookup requires a closed Definitive outcome")
            return Definitive(
                dict(value["result"]),
                _evidence(value["evidence"]),
                dict(value["usage"]),
                None if value["shared_state"] is None else dict(value["shared_state"]),
            )
        if set(value) != _UNKNOWN_FIELDS or value["kind"] != "Unknown":
            raise RemoteProtocolError("unknown lookup requires a closed Unknown outcome")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise RemoteProtocolError("Unknown outcome reason is invalid")
        return Unknown(reason)


__all__ = [
    "RemoteBindingError",
    "RemoteEnvironment",
    "RemoteEnvironmentError",
    "RemoteProtocolError",
    "RemoteRequestRejected",
    "RemoteTransport",
    "RemoteTransportUnavailable",
    "RemoteUnsupportedOperation",
    "UrllibRemoteTransport",
]
