"""Task-neutral v8 controller admission and host-interaction wire types.

The controller protocol is deliberately independent of any application task
model.  Stores and distributed backends consume these immutable values at
their CAS boundary; this module only owns strict wire validation and digest
derivation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vv_agent.checkpoint import CheckpointError, canonical_json_bytes, canonical_json_sha256

if TYPE_CHECKING:
    from vv_agent.runtime.backends.distributed import DistributedRunHandle

CONTROLLER_COMMAND_SCHEMA = "vv-agent.controller-command.v1"
CONTROLLER_RECEIPT_SCHEMA = "vv-agent.controller-command-receipt.v1"
CONTROLLER_RESOLUTION_SCHEMA = "vv-agent.controller-command-resolution.v1"
HOST_REQUEST_SCHEMA = "vv-agent.host-interaction-request.v1"
HOST_OUTCOME_SCHEMA = "vv-agent.host-interaction-outcome.v1"
HOST_RESPONSE_SCHEMA = "vv-agent.host-interaction-response.v1"
HOST_RECOVERY_SCHEMA = "vv-agent.host-interaction-recovery.v1"
HOST_RECOVERY_RESULT_SCHEMA = "vv-agent.host-interaction-recovery-result.v1"
HOST_RECORD_SCHEMA = "vv-agent.host-interaction-record.v1"
HOST_NOTIFICATION_SCHEMA = "vv-agent.host-interaction-notification.v1"
CONTROLLER_COMMAND_ID_SCHEMA = "vv-agent.controller-command-id.v1"
CONTROLLER_COMMAND_ID_DOMAIN = "vv-agent.controller-command-id.v1"

_MAX_ID_BYTES = 512
_MAX_CONTENT_BYTES = 65536
_MAX_WIRE_INTEGER = (1 << 53) - 1
_SHA256_FIELDS = frozenset({"request_digest", "response_digest", "command_digest", "notification_payload_digest"})
_CREDENTIAL_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+"
    r"|\b(?:sk|pk)[-_][A-Za-z0-9._~-]+"
    r"|\b(?:token|secret|password|api[_-]?key|authorization)\s*(?:[:=]|\s+)\s*"
    r"(?:bearer\s+)?[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_LOCATOR_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def sanitize_host_prompt(prompt: str) -> str:
    """Create the public notification projection without credentials/locators."""
    # Remove complete locators first.  Query strings commonly contain
    # ``token=``/``secret=`` and replacing the credential fragment first would
    # leave part of the URL after the locator substitution.
    sanitized = _LOCATOR_PATTERN.sub("[external locator redacted]", _content(prompt, "prompt"))
    sanitized = _CREDENTIAL_PATTERN.sub("[credential redacted]", sanitized)
    return sanitized


def derive_controller_command_id(thread_id: str, turn_id: str, action_id: str) -> str:
    """Derive the App Server command identity without accepting a client id.

    The length prefix is part of the central contract so concatenation cannot
    make two different public scopes collide.  The payload uses snake_case
    keys because it is an internal identity envelope, not App Server wire.
    """
    thread = _text(thread_id, "thread_id")
    turn = _text(turn_id, "turn_id")
    action = _text(action_id, "action_id")
    payload = {
        "action_id": action,
        "schema_version": CONTROLLER_COMMAND_ID_SCHEMA,
        "thread_id": thread,
        "turn_id": turn,
    }
    canonical = canonical_json_bytes(payload, "controller_command_id")
    framed = CONTROLLER_COMMAND_ID_DOMAIN.encode("utf-8") + b"\x00" + len(canonical).to_bytes(8, "big") + canonical
    return hashlib.sha256(framed).hexdigest()


def derive_host_interaction_record_id(checkpoint_key: str, request: HostInteractionRequest | Mapping[str, Any]) -> str:
    """Derive the stable record identity from the canonical producer inputs."""
    request_value = request if isinstance(request, HostInteractionRequest) else HostInteractionRequest.from_dict(request)
    return canonical_json_sha256(
        {
            "checkpoint_key": _text(checkpoint_key, "checkpoint_key"),
            "interaction_id": request_value.interaction_id,
            "logical_cycle": request_value.logical_cycle,
            "request_digest": request_value.request_digest,
            "schema_version": HOST_RECORD_SCHEMA,
        },
        "host_interaction_record_id",
    )


def derive_host_interaction_notification_id(record_id: str) -> str:
    return canonical_json_sha256(
        {
            "record_id": _text(record_id, "record_id"),
            "schema_version": HOST_NOTIFICATION_SCHEMA,
            "transition": "host_interaction_requested",
        },
        "host_interaction_notification_id",
    )


def derive_controller_receipt_outbox_id(command_id: str, command_digest: str) -> str:
    """Derive the stable recovery-wake outbox identity from its command."""
    return canonical_json_sha256(
        {
            "command_digest": _digest(command_digest, "command_digest"),
            "command_id": _text(command_id, "command_id"),
            "schema_version": CONTROLLER_RECEIPT_SCHEMA,
        },
        "controller_receipt_outbox_id",
    )


def derive_host_response_digest(
    *,
    interaction_id: str,
    logical_cycle: int,
    operation_id: str,
    tool_call_id: str,
    request_digest: str,
    command_id: str,
    response: Mapping[str, Any],
) -> str:
    """Hash the complete resolved response without its derived digest."""
    response_value = _response(response)
    # The Rust constructor applies the public host-text policy before it
    # computes the response digest.  Keep this helper safe for callers that
    # construct a response from an untrusted/public value instead of relying
    # on the controller admission path to normalize it first.
    response_value["content"] = sanitize_host_prompt(response_value["content"])
    return canonical_json_sha256(
        {
            "command_id": _text(command_id, "command_id"),
            "interaction_id": _text(interaction_id, "interaction_id"),
            "logical_cycle": _integer(logical_cycle, "logical_cycle", minimum=1),
            "operation_id": _text(operation_id, "operation_id"),
            "request_digest": _digest(request_digest, "request_digest"),
            "response": response_value,
            "schema_version": HOST_RESPONSE_SCHEMA,
            "tool_call_id": _text(tool_call_id, "tool_call_id"),
        },
        "host_interaction_response",
    )


def _strict_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(f"{label} fields do not match v8 schema: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{label} field names must be strings")


def _closed_fields(payload: Mapping[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    actual = set(payload)
    missing = required - actual
    unknown = actual - allowed
    if missing or unknown:
        raise ValueError(f"{label} fields do not match v8 schema: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{label} field names must be strings")


def _text(value: Any, label: str, *, max_bytes: int = _MAX_ID_BYTES) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds the UTF-8 byte limit")
    return value


def _content(value: Any, label: str) -> str:
    return _text(value, label, max_bytes=_MAX_CONTENT_BYTES)


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int = _MAX_WIRE_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    return value


def _response(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("host interaction response must be an object")
    _strict_fields(value, {"role", "content"}, "host interaction response")
    if value.get("role") != "user":
        raise ValueError("host interaction response role must be user")
    return {"role": "user", "content": _content(value.get("content"), "host interaction response content")}


def _handle(value: Any) -> DistributedRunHandle:
    from vv_agent.runtime.backends.distributed import DistributedRunHandle

    if isinstance(value, DistributedRunHandle):
        return value
    return DistributedRunHandle.from_dict(value)


def _request_without_digest(request: HostInteractionRequest) -> dict[str, Any]:
    return {
        "interaction_id": request.interaction_id,
        "logical_cycle": request.logical_cycle,
        "operation_id": request.operation_id,
        "prompt": request.prompt,
        "schema_version": HOST_REQUEST_SCHEMA,
        "tool_call_id": request.tool_call_id,
    }


@dataclass(frozen=True, slots=True)
class HostInteractionAdmissionContext:
    """Framework-owned claim fence for a host-interaction producer call.

    The context is intentionally not part of the host request/outcome wire.
    A runner obtains it from its active checkpoint claim and binds it to the
    producer instance.  Stores must receive this complete context at the CAS
    boundary; discovering a claim by scanning the store is unsafe when more
    than one run is live.
    """

    checkpoint_key: str
    expected_revision: int
    claim_token: str
    claimed_cycle: int
    now_ms: int
    lease_expires_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_key", _text(self.checkpoint_key, "checkpoint_key"))
        object.__setattr__(self, "expected_revision", _integer(self.expected_revision, "expected_revision"))
        object.__setattr__(self, "claim_token", _text(self.claim_token, "claim_token"))
        object.__setattr__(self, "claimed_cycle", _integer(self.claimed_cycle, "claimed_cycle", minimum=1))
        object.__setattr__(self, "now_ms", _integer(self.now_ms, "now_ms"))
        object.__setattr__(
            self,
            "lease_expires_at_ms",
            _integer(self.lease_expires_at_ms, "lease_expires_at_ms"),
        )

    def validate(self) -> None:
        """Re-run strict validation immediately before a store call."""
        type(self)(
            checkpoint_key=self.checkpoint_key,
            expected_revision=self.expected_revision,
            claim_token=self.claim_token,
            claimed_cycle=self.claimed_cycle,
            now_ms=self.now_ms,
            lease_expires_at_ms=self.lease_expires_at_ms,
        )


@dataclass(frozen=True, slots=True)
class HostInteractionRequest:
    interaction_id: str
    logical_cycle: int
    operation_id: str
    tool_call_id: str
    prompt: str
    request_digest: str | None = None

    def __post_init__(self) -> None:
        interaction_id = _text(self.interaction_id, "interaction_id")
        operation_id = _text(self.operation_id, "operation_id")
        tool_call_id = _text(self.tool_call_id, "tool_call_id")
        logical_cycle = _integer(self.logical_cycle, "logical_cycle", minimum=1)
        prompt = _content(sanitize_host_prompt(self.prompt), "prompt")
        object.__setattr__(self, "interaction_id", interaction_id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "logical_cycle", logical_cycle)
        object.__setattr__(self, "prompt", prompt)
        expected = canonical_json_sha256(
            {
                "interaction_id": interaction_id,
                "logical_cycle": logical_cycle,
                "operation_id": operation_id,
                "prompt": prompt,
                "schema_version": HOST_REQUEST_SCHEMA,
                "tool_call_id": tool_call_id,
            },
            "host_interaction_request",
        )
        if self.request_digest is not None and _digest(self.request_digest, "request_digest") != expected:
            raise ValueError("request_digest does not match the canonical host interaction request")
        object.__setattr__(self, "request_digest", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_REQUEST_SCHEMA,
            "interaction_id": self.interaction_id,
            "logical_cycle": self.logical_cycle,
            "operation_id": self.operation_id,
            "tool_call_id": self.tool_call_id,
            "request_digest": self.request_digest,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> HostInteractionRequest:
        if not isinstance(payload, Mapping):
            raise ValueError("host interaction request must be an object")
        _strict_fields(
            payload,
            {"schema_version", "interaction_id", "logical_cycle", "operation_id", "tool_call_id", "request_digest", "prompt"},
            "host interaction request",
        )
        if payload["schema_version"] != HOST_REQUEST_SCHEMA:
            raise ValueError("unsupported host interaction request schema")
        request_digest = _digest(payload["request_digest"], "request_digest")
        return cls(
            interaction_id=payload["interaction_id"],
            logical_cycle=payload["logical_cycle"],
            operation_id=payload["operation_id"],
            tool_call_id=payload["tool_call_id"],
            request_digest=request_digest,
            prompt=payload["prompt"],
        )


@dataclass(frozen=True, slots=True)
class HostInteractionResponse:
    """The complete, closed resolved response persisted by controller CAS."""

    interaction_id: str
    logical_cycle: int
    operation_id: str
    tool_call_id: str
    request_digest: str
    command_id: str
    response: dict[str, str]
    response_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "interaction_id", _text(self.interaction_id, "interaction_id"))
        object.__setattr__(self, "logical_cycle", _integer(self.logical_cycle, "logical_cycle", minimum=1))
        object.__setattr__(self, "operation_id", _text(self.operation_id, "operation_id"))
        object.__setattr__(self, "tool_call_id", _text(self.tool_call_id, "tool_call_id"))
        object.__setattr__(self, "request_digest", _digest(self.request_digest, "request_digest"))
        object.__setattr__(self, "command_id", _text(self.command_id, "command_id"))
        response = _response(self.response)
        response["content"] = sanitize_host_prompt(response["content"])
        object.__setattr__(self, "response", response)
        expected = derive_host_response_digest(
            interaction_id=self.interaction_id,
            logical_cycle=self.logical_cycle,
            operation_id=self.operation_id,
            tool_call_id=self.tool_call_id,
            request_digest=self.request_digest,
            command_id=self.command_id,
            response=response,
        )
        if self.response_digest is not None and _digest(self.response_digest, "response_digest") != expected:
            raise ValueError("response_digest does not match the canonical resolved response")
        object.__setattr__(self, "response_digest", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_RESPONSE_SCHEMA,
            "interaction_id": self.interaction_id,
            "logical_cycle": self.logical_cycle,
            "operation_id": self.operation_id,
            "tool_call_id": self.tool_call_id,
            "request_digest": self.request_digest,
            "command_id": self.command_id,
            "response": dict(self.response),
            "response_digest": self.response_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> HostInteractionResponse:
        if not isinstance(payload, Mapping):
            raise ValueError("host interaction response record must be an object")
        _strict_fields(
            payload,
            {
                "schema_version",
                "interaction_id",
                "logical_cycle",
                "operation_id",
                "tool_call_id",
                "request_digest",
                "command_id",
                "response",
                "response_digest",
            },
            "host interaction response record",
        )
        if payload["schema_version"] != HOST_RESPONSE_SCHEMA:
            raise ValueError("unsupported host interaction response schema")
        raw_response = _response(payload["response"])
        if sanitize_host_prompt(raw_response["content"]) != raw_response["content"]:
            raise ValueError("resolved host interaction response is not sanitized")
        response_digest = _digest(payload["response_digest"], "response_digest")
        return cls(
            interaction_id=payload["interaction_id"],
            logical_cycle=payload["logical_cycle"],
            operation_id=payload["operation_id"],
            tool_call_id=payload["tool_call_id"],
            request_digest=payload["request_digest"],
            command_id=payload["command_id"],
            response=raw_response,
            response_digest=response_digest,
        )


@dataclass(frozen=True, slots=True)
class HostInteractionRecoveryEnvelope:
    """Strict v8 recovery envelope; no lease/default fields are accepted."""

    record_id: str
    checkpoint_key: str
    run_id: str
    trace_id: str
    claim_mode: str
    resume_attempt: int
    expected_revision: int
    logical_cycle: int
    interaction_id: str
    operation_id: str
    tool_call_id: str
    request_digest: str
    command_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "checkpoint_key",
            "run_id",
            "trace_id",
            "interaction_id",
            "operation_id",
            "tool_call_id",
            "command_id",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        if self.claim_mode != "recovery":
            raise ValueError("host interaction recovery claim_mode must be recovery")
        object.__setattr__(self, "resume_attempt", _integer(self.resume_attempt, "resume_attempt", minimum=1))
        object.__setattr__(self, "expected_revision", _integer(self.expected_revision, "expected_revision"))
        object.__setattr__(self, "logical_cycle", _integer(self.logical_cycle, "logical_cycle", minimum=1))
        object.__setattr__(self, "request_digest", _digest(self.request_digest, "request_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_RECOVERY_SCHEMA,
            "record_id": self.record_id,
            "checkpoint_key": self.checkpoint_key,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "claim_mode": self.claim_mode,
            "resume_attempt": self.resume_attempt,
            "expected_revision": self.expected_revision,
            "logical_cycle": self.logical_cycle,
            "interaction_id": self.interaction_id,
            "operation_id": self.operation_id,
            "tool_call_id": self.tool_call_id,
            "request_digest": self.request_digest,
            "command_id": self.command_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> HostInteractionRecoveryEnvelope:
        if not isinstance(payload, Mapping):
            raise ValueError("host interaction recovery envelope must be an object")
        fields = {
            "schema_version",
            "record_id",
            "checkpoint_key",
            "run_id",
            "trace_id",
            "claim_mode",
            "resume_attempt",
            "expected_revision",
            "logical_cycle",
            "interaction_id",
            "operation_id",
            "tool_call_id",
            "request_digest",
            "command_id",
        }
        _strict_fields(payload, fields, "host interaction recovery envelope")
        if payload["schema_version"] != HOST_RECOVERY_SCHEMA:
            raise ValueError("unsupported host interaction recovery schema")
        return cls(**{field: payload[field] for field in fields if field != "schema_version"})


def validate_host_interaction_record(
    payload: Mapping[str, Any],
    *,
    checkpoint_key: str | None = None,
) -> dict[str, Any]:
    """Strictly decode one durable interaction record before any CAS write."""

    if not isinstance(payload, Mapping):
        raise ValueError("host interaction record must be an object")
    allowed = {
        "schema_version",
        "record_id",
        "checkpoint_key",
        "interaction_id",
        "logical_cycle",
        "request",
        "request_digest",
        "state",
        "attempt",
        "claim_token",
        "lease_expires_at_ms",
        "response",
        "response_digest",
        "command_id",
        "resolved_revision",
        "consumed_revision",
        "last_error",
    }
    required = allowed - {"last_error"}
    _closed_fields(payload, allowed, required, "host interaction record")
    if payload["schema_version"] != HOST_RECORD_SCHEMA:
        raise ValueError("unsupported host interaction record schema")
    record_key = _text(payload["checkpoint_key"], "checkpoint_key")
    if checkpoint_key is not None and record_key != _text(checkpoint_key, "checkpoint_key"):
        raise ValueError("host interaction record checkpoint binding is stale")
    request = HostInteractionRequest.from_dict(payload["request"])
    if payload["request_digest"] != request.request_digest:
        raise ValueError("host interaction record request digest conflicts")
    if payload["record_id"] != derive_host_interaction_record_id(record_key, request):
        raise ValueError("host interaction record id does not match request identity")
    if payload["interaction_id"] != request.interaction_id or payload["logical_cycle"] != request.logical_cycle:
        raise ValueError("host interaction record identity conflicts with request")
    state = payload["state"]
    if state not in {"active", "resolved_pending", "resolved_claimed", "consumed"}:
        raise ValueError("host interaction record state is invalid")
    _integer(payload["attempt"], "host interaction record attempt")
    claim_token = payload["claim_token"]
    lease_expires_at_ms = payload["lease_expires_at_ms"]
    if claim_token is not None:
        _text(claim_token, "host interaction record claim_token")
    if lease_expires_at_ms is not None:
        _integer(lease_expires_at_ms, "host interaction record lease_expires_at_ms")
    if (claim_token is None) != (lease_expires_at_ms is None):
        raise ValueError("host interaction record claim and lease must be both present or null")
    command_id = payload["command_id"]
    if command_id is not None:
        _text(command_id, "host interaction record command_id")
    resolved_revision = payload["resolved_revision"]
    consumed_revision = payload["consumed_revision"]
    if resolved_revision is not None:
        _integer(resolved_revision, "host interaction record resolved_revision")
    if consumed_revision is not None:
        _integer(consumed_revision, "host interaction record consumed_revision")
    response = payload["response"]
    response_digest = payload["response_digest"]
    if state == "active":
        if any(value is not None for value in (response, response_digest, command_id, resolved_revision, consumed_revision)):
            raise ValueError("active host interaction record cannot contain a resolved response")
        if claim_token is not None:
            raise ValueError("active host interaction record cannot be claimed")
    else:
        if response is None or response_digest is None or command_id is None or resolved_revision is None:
            raise ValueError("resolved host interaction record is incomplete")
        parsed_response = HostInteractionResponse.from_dict(response)
        if sanitize_host_prompt(parsed_response.response["content"]) != parsed_response.response["content"]:
            raise ValueError("resolved host interaction response is not sanitized")
        if (
            parsed_response.interaction_id != request.interaction_id
            or parsed_response.logical_cycle != request.logical_cycle
            or parsed_response.operation_id != request.operation_id
            or parsed_response.tool_call_id != request.tool_call_id
            or parsed_response.request_digest != request.request_digest
            or parsed_response.command_id != command_id
            or parsed_response.response_digest != response_digest
        ):
            raise ValueError("resolved host interaction record identity or digest conflicts")
        if state == "resolved_pending" and claim_token is not None:
            raise ValueError("resolved_pending host interaction record cannot be claimed")
        if state == "resolved_claimed" and claim_token is None:
            raise ValueError("resolved_claimed host interaction record requires a claim")
        if state == "consumed" and claim_token is not None:
            raise ValueError("consumed host interaction record cannot be claimed")
        if state != "consumed" and consumed_revision is not None:
            raise ValueError("unconsumed host interaction record cannot contain consumed_revision")
        if state == "consumed" and consumed_revision is None:
            raise ValueError("consumed host interaction record requires consumed_revision")
    if "last_error" in payload and payload["last_error"] is not None:
        _content(payload["last_error"], "host interaction record last_error")
    return dict(payload)


def validate_host_interaction_notification(
    payload: Mapping[str, Any],
    *,
    notification_id: str | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Strictly decode the independent, sanitized UI notification payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("host interaction notification must be an object")
    fields = {
        "schema_version",
        "notification_id",
        "record_id",
        "interaction_id",
        "logical_cycle",
        "status",
        "wait_reason",
        "prompt",
    }
    _strict_fields(payload, fields, "host interaction notification")
    if payload["schema_version"] != HOST_NOTIFICATION_SCHEMA:
        raise ValueError("unsupported host interaction notification schema")
    parsed_notification_id = _text(payload["notification_id"], "notification_id")
    parsed_record_id = _text(payload["record_id"], "record_id")
    if notification_id is not None and parsed_notification_id != _text(notification_id, "notification_id"):
        raise ValueError("host interaction notification id conflicts")
    if record_id is not None and parsed_record_id != _text(record_id, "record_id"):
        raise ValueError("host interaction notification record binding conflicts")
    if parsed_notification_id != derive_host_interaction_notification_id(parsed_record_id):
        raise ValueError("host interaction notification id does not match record identity")
    _text(payload["interaction_id"], "interaction_id")
    _integer(payload["logical_cycle"], "logical_cycle", minimum=1)
    if payload["status"] != "host_interaction" or payload["wait_reason"] != "host_interaction":
        raise ValueError("host interaction notification status or wait_reason is invalid")
    prompt = _content(payload["prompt"], "prompt")
    if sanitize_host_prompt(prompt) != prompt:
        raise ValueError("host interaction notification prompt is not sanitized")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class HostInteractionOutcome:
    interaction_id: str
    logical_cycle: int
    checkpoint_revision: int
    status: str
    outbox_state: str
    record_id: str
    notification_id: str
    notification_payload_digest: str
    notification_outbox_action: str
    notification_outbox_destination: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "interaction_id", _text(self.interaction_id, "interaction_id"))
        object.__setattr__(self, "logical_cycle", _integer(self.logical_cycle, "logical_cycle", minimum=1))
        object.__setattr__(self, "checkpoint_revision", _integer(self.checkpoint_revision, "checkpoint_revision"))
        if self.status not in {"admitted", "replayed"}:
            raise ValueError("host interaction outcome status must be admitted or replayed")
        if self.outbox_state != "pending":
            raise ValueError("host interaction outcome outbox_state must be pending")
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(self, "notification_id", _text(self.notification_id, "notification_id"))
        object.__setattr__(
            self, "notification_payload_digest", _digest(self.notification_payload_digest, "notification_payload_digest")
        )
        if self.notification_outbox_action != "host_interaction_notification":
            raise ValueError("host interaction outcome notification_outbox_action is invalid")
        if self.notification_outbox_destination != "host_interaction_observer":
            raise ValueError("host interaction outcome notification_outbox_destination is invalid")
        object.__setattr__(
            self,
            "notification_outbox_destination",
            _text(self.notification_outbox_destination, "notification_outbox_destination"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_OUTCOME_SCHEMA,
            "interaction_id": self.interaction_id,
            "logical_cycle": self.logical_cycle,
            "checkpoint_revision": self.checkpoint_revision,
            "status": self.status,
            "outbox_state": self.outbox_state,
            "record_id": self.record_id,
            "notification_id": self.notification_id,
            "notification_payload_digest": self.notification_payload_digest,
            "notification_outbox_action": self.notification_outbox_action,
            "notification_outbox_destination": self.notification_outbox_destination,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> HostInteractionOutcome:
        if not isinstance(payload, Mapping):
            raise ValueError("host interaction outcome must be an object")
        fields = {
            "schema_version",
            "interaction_id",
            "logical_cycle",
            "checkpoint_revision",
            "status",
            "outbox_state",
            "record_id",
            "notification_id",
            "notification_payload_digest",
            "notification_outbox_action",
            "notification_outbox_destination",
        }
        _strict_fields(payload, fields, "host interaction outcome")
        if payload["schema_version"] != HOST_OUTCOME_SCHEMA:
            raise ValueError("unsupported host interaction outcome schema")
        return cls(**{field: payload[field] for field in fields if field != "schema_version"})


def _command_payload(command: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(command, Mapping):
        raise ValueError("controller command variant must be an object")
    kind = command.get("kind")
    if kind == "host_interaction_response":
        _strict_fields(
            command,
            {"kind", "interaction_id", "logical_cycle", "operation_id", "tool_call_id", "request_digest", "response"},
            "host interaction response command",
        )
        response = _response(command["response"])
        # Normalize before the command digest is derived so credentials and
        # transport locators cannot cross the durable controller CAS boundary.
        response["content"] = sanitize_host_prompt(response["content"])
        return {
            "kind": kind,
            "interaction_id": _text(command["interaction_id"], "interaction_id"),
            "logical_cycle": _integer(command["logical_cycle"], "logical_cycle", minimum=1),
            "operation_id": _text(command["operation_id"], "operation_id"),
            "tool_call_id": _text(command["tool_call_id"], "tool_call_id"),
            "request_digest": _digest(command["request_digest"], "request_digest"),
            "response": response,
        }
    if kind in {"suspend", "resume", "cancel", "abort"}:
        _strict_fields(command, {"kind"}, f"{kind} command")
        return {"kind": kind}
    raise ValueError("unsupported controller command variant")


@dataclass(frozen=True, slots=True)
class ControllerCommand:
    command_id: str
    handle: DistributedRunHandle
    resume_attempt: int
    expected_revision: int
    command: Mapping[str, Any]
    command_digest: str | None = None

    def __post_init__(self) -> None:
        command_id = _text(self.command_id, "command_id")
        handle = _handle(self.handle)
        resume_attempt = _integer(self.resume_attempt, "resume_attempt", minimum=1)
        expected_revision = _integer(self.expected_revision, "expected_revision")
        command = _command_payload(self.command)
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "handle", handle)
        object.__setattr__(self, "resume_attempt", resume_attempt)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "command", command)
        unsigned = {
            "schema_version": CONTROLLER_COMMAND_SCHEMA,
            "command_id": command_id,
            "handle": handle.to_dict(),
            "resume_attempt": resume_attempt,
            "expected_revision": expected_revision,
            "command": command,
        }
        expected = canonical_json_sha256(unsigned, "controller_command")
        if self.command_digest is not None and _digest(self.command_digest, "command_digest") != expected:
            raise ValueError("command_digest does not match the canonical controller command")
        object.__setattr__(self, "command_digest", expected)

    @property
    def kind(self) -> str:
        return str(self.command["kind"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROLLER_COMMAND_SCHEMA,
            "command_id": self.command_id,
            "command_digest": self.command_digest,
            "handle": self.handle.to_dict(),
            "resume_attempt": self.resume_attempt,
            "expected_revision": self.expected_revision,
            "command": dict(self.command),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControllerCommand:
        if not isinstance(payload, Mapping):
            raise ValueError("controller command must be an object")
        _strict_fields(
            payload,
            {"schema_version", "command_id", "command_digest", "handle", "resume_attempt", "expected_revision", "command"},
            "controller command",
        )
        if payload["schema_version"] != CONTROLLER_COMMAND_SCHEMA:
            raise ValueError("unsupported controller command schema")
        command_digest = _digest(payload["command_digest"], "command_digest")
        return cls(
            command_id=payload["command_id"],
            command_digest=command_digest,
            handle=_handle(payload["handle"]),
            resume_attempt=payload["resume_attempt"],
            expected_revision=payload["expected_revision"],
            command=payload["command"],
        )


@dataclass(frozen=True, slots=True)
class ControllerCommandReceipt:
    command_id: str
    command_digest: str
    handle: DistributedRunHandle
    resume_attempt: int
    expected_revision: int
    resulting_revision: int
    resulting_status: str
    outbox_state: str
    outbox_action: str
    outbox_destination: str | None
    outbox_attempt: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _text(self.command_id, "command_id"))
        object.__setattr__(self, "command_digest", _digest(self.command_digest, "command_digest"))
        object.__setattr__(self, "handle", _handle(self.handle))
        object.__setattr__(self, "resume_attempt", _integer(self.resume_attempt, "resume_attempt", minimum=1))
        object.__setattr__(self, "expected_revision", _integer(self.expected_revision, "expected_revision"))
        object.__setattr__(self, "resulting_revision", _integer(self.resulting_revision, "resulting_revision"))
        object.__setattr__(self, "resulting_status", _text(self.resulting_status, "resulting_status"))
        if self.outbox_state not in {"pending", "claimed", "delivered", "ambiguous"}:
            raise ValueError("controller receipt outbox_state is invalid")
        if self.outbox_action not in {"none", "recovery_dispatch"}:
            raise ValueError("controller receipt outbox_action is invalid")
        if self.outbox_action == "none" and self.outbox_destination is not None:
            raise ValueError("controller receipt none action cannot have a destination")
        if self.outbox_action == "recovery_dispatch" and self.outbox_destination != "distributed_advance":
            raise ValueError("controller receipt recovery_dispatch destination is invalid")
        object.__setattr__(self, "outbox_attempt", _integer(self.outbox_attempt, "outbox_attempt"))
        if self.outbox_action == "none" and (self.outbox_state != "delivered" or self.outbox_attempt != 0):
            raise ValueError("controller receipt none action must be durably delivered without an attempt")
        if self.outbox_action == "recovery_dispatch" and self.outbox_state != "pending" and self.outbox_attempt < 1:
            raise ValueError("controller receipt recovery action requires a claimed attempt")
        if self.outbox_destination is not None:
            object.__setattr__(self, "outbox_destination", _text(self.outbox_destination, "outbox_destination"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROLLER_RECEIPT_SCHEMA,
            "command_id": self.command_id,
            "command_digest": self.command_digest,
            "handle": self.handle.to_dict(),
            "resume_attempt": self.resume_attempt,
            "expected_revision": self.expected_revision,
            "resulting_revision": self.resulting_revision,
            "resulting_status": self.resulting_status,
            "outbox_state": self.outbox_state,
            "outbox_action": self.outbox_action,
            "outbox_destination": self.outbox_destination,
            "outbox_attempt": self.outbox_attempt,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControllerCommandReceipt:
        if not isinstance(payload, Mapping):
            raise ValueError("controller command receipt must be an object")
        fields = {
            "schema_version",
            "command_id",
            "command_digest",
            "handle",
            "resume_attempt",
            "expected_revision",
            "resulting_revision",
            "resulting_status",
            "outbox_state",
            "outbox_action",
            "outbox_destination",
            "outbox_attempt",
        }
        _strict_fields(payload, fields, "controller command receipt")
        if payload["schema_version"] != CONTROLLER_RECEIPT_SCHEMA:
            raise ValueError("unsupported controller command receipt schema")
        values = {field: payload[field] for field in fields if field != "schema_version"}
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ControllerWake:
    action: str
    destination: str | None
    logical_cycle: int
    claim_mode: str

    def __post_init__(self) -> None:
        if self.action not in {"recovery_dispatch", "none"}:
            raise ValueError("controller wake action is invalid")
        if self.action == "recovery_dispatch" and self.destination != "distributed_advance":
            raise ValueError("recovery_dispatch wake destination is invalid")
        if self.action == "none" and self.destination is not None:
            raise ValueError("none wake destination must be null")
        object.__setattr__(self, "logical_cycle", _integer(self.logical_cycle, "logical_cycle", minimum=1))
        if self.claim_mode not in {"recovery", "continue", "none"}:
            raise ValueError("controller wake claim_mode is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "destination": self.destination,
            "logical_cycle": self.logical_cycle,
            "claim_mode": self.claim_mode,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControllerWake:
        if not isinstance(payload, Mapping):
            raise ValueError("controller wake must be an object")
        _strict_fields(payload, {"action", "destination", "logical_cycle", "claim_mode"}, "controller wake")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ControllerCommandResolution:
    kind: str
    receipt: ControllerCommandReceipt | None = None
    wake: ControllerWake | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"applied", "replayed", "rejected"}:
            raise ValueError("controller command resolution kind is invalid")
        if self.kind == "rejected":
            if self.receipt is not None or self.wake is not None or not self.error:
                raise ValueError("rejected controller resolution requires only error")
        elif self.receipt is None or self.wake is None or self.error is not None:
            raise ValueError("applied/replayed controller resolution requires receipt and wake")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": CONTROLLER_RESOLUTION_SCHEMA, "kind": self.kind}
        if self.receipt is not None:
            payload["receipt"] = self.receipt.to_dict()
        if self.wake is not None:
            payload["wake"] = self.wake.to_dict()
        if self.error is not None:
            payload["error"] = _text(self.error, "controller resolution error")
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ControllerCommandResolution:
        if not isinstance(payload, Mapping):
            raise ValueError("controller command resolution must be an object")
        kind = payload.get("kind")
        if kind in {"applied", "replayed"}:
            _strict_fields(payload, {"schema_version", "kind", "receipt", "wake"}, "controller command resolution")
            if payload.get("schema_version") != CONTROLLER_RESOLUTION_SCHEMA:
                raise ValueError("unsupported controller resolution schema")
            return cls(
                kind=kind,
                receipt=ControllerCommandReceipt.from_dict(payload["receipt"]),
                wake=ControllerWake.from_dict(payload["wake"]),
            )
        if kind == "rejected":
            _strict_fields(payload, {"schema_version", "kind", "error"}, "controller command resolution")
            if payload.get("schema_version") != CONTROLLER_RESOLUTION_SCHEMA:
                raise ValueError("unsupported controller resolution schema")
            return cls(kind=kind, error=_text(payload["error"], "controller resolution error"))
        raise ValueError("unsupported controller command resolution kind")


@dataclass(frozen=True, slots=True)
class HostInteractionRecoveryResult:
    kind: str
    record_id: str
    checkpoint_revision: int | None
    consumed_revision: int | None
    claim_mode: str
    resume_attempt: int | None
    injection_count: int
    checkpoint_execution_claim_state: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"applied", "replayed", "rejected"}:
            raise ValueError("host interaction recovery result kind is invalid")
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        if self.checkpoint_revision is not None:
            object.__setattr__(self, "checkpoint_revision", _integer(self.checkpoint_revision, "checkpoint_revision"))
        if self.consumed_revision is not None:
            object.__setattr__(self, "consumed_revision", _integer(self.consumed_revision, "consumed_revision"))
        if self.claim_mode != "recovery":
            raise ValueError("host interaction recovery claim_mode must be recovery")
        if self.resume_attempt is not None:
            object.__setattr__(self, "resume_attempt", _integer(self.resume_attempt, "resume_attempt", minimum=1))
        object.__setattr__(self, "injection_count", _integer(self.injection_count, "injection_count"))
        if self.checkpoint_execution_claim_state not in {"retained", "released", "not_acquired"}:
            raise ValueError("host interaction recovery claim state is invalid")
        if self.kind == "rejected" and self.injection_count != 0:
            raise ValueError("rejected host interaction recovery cannot inject a response")
        if self.kind == "applied" and (
            self.consumed_revision is None
            or self.resume_attempt is None
            or self.injection_count != 1
            or self.checkpoint_execution_claim_state != "retained"
        ):
            raise ValueError("applied host interaction recovery result is incomplete")
        if self.kind == "replayed" and (
            self.consumed_revision is None
            or self.resume_attempt is None
            or self.injection_count != 1
            or self.checkpoint_execution_claim_state not in {"retained", "released"}
        ):
            raise ValueError("replayed host interaction recovery result is incomplete")
        if self.error is not None:
            object.__setattr__(self, "error", _text(self.error, "host interaction recovery error"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": HOST_RECOVERY_RESULT_SCHEMA,
            "kind": self.kind,
            "record_id": self.record_id,
            "checkpoint_revision": self.checkpoint_revision,
            "consumed_revision": self.consumed_revision,
            "claim_mode": self.claim_mode,
            "resume_attempt": self.resume_attempt,
            "injection_count": self.injection_count,
            "checkpoint_execution_claim_state": self.checkpoint_execution_claim_state,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> HostInteractionRecoveryResult:
        if not isinstance(payload, Mapping):
            raise ValueError("host interaction recovery result must be an object")
        _closed_fields(
            payload,
            {
                "schema_version",
                "kind",
                "record_id",
                "checkpoint_revision",
                "consumed_revision",
                "claim_mode",
                "resume_attempt",
                "injection_count",
                "checkpoint_execution_claim_state",
                "error",
            },
            {
                "schema_version",
                "kind",
                "record_id",
                "checkpoint_revision",
                "consumed_revision",
                "claim_mode",
                "resume_attempt",
                "injection_count",
                "checkpoint_execution_claim_state",
            },
            "host interaction recovery result",
        )
        if payload["schema_version"] != HOST_RECOVERY_RESULT_SCHEMA:
            raise ValueError("unsupported host interaction recovery result schema")
        return cls(
            kind=payload["kind"],
            record_id=payload["record_id"],
            checkpoint_revision=payload["checkpoint_revision"],
            consumed_revision=payload["consumed_revision"],
            claim_mode=payload["claim_mode"],
            resume_attempt=payload["resume_attempt"],
            injection_count=payload["injection_count"],
            checkpoint_execution_claim_state=payload["checkpoint_execution_claim_state"],
            error=payload.get("error"),
        )


class DistributedBackend:
    """Public task-neutral seam over a durable :class:`CheckpointStore`.

    Application code supplies the authoritative store.  A producer must also
    be bound to an explicit :class:`HostInteractionAdmissionContext` obtained
    from the runner's active checkpoint claim.  No backend or task-specific
    fields enter the wire values.
    """

    def __init__(
        self,
        store: Any,
        *,
        admission_context: HostInteractionAdmissionContext | None = None,
    ) -> None:
        if store is None:
            raise TypeError("DistributedBackend requires a checkpoint store")
        self.store = store
        self.admission_context = admission_context

    def produce_host_interaction(self, request: HostInteractionRequest) -> HostInteractionOutcome:
        context = self.admission_context
        if context is None:
            raise CheckpointError(
                "host interaction producer requires an explicit admission context",
                code="host_interaction_claim_required",
            )
        context.validate()
        return self.store.produce_host_interaction(request, admission_context=context)

    def resolve_controller_command(self, command: ControllerCommand) -> ControllerCommandResolution:
        return self.store.resolve_controller_command(command)

    def claim_and_consume_host_interaction_response(
        self,
        envelope: Mapping[str, Any],
    ) -> HostInteractionRecoveryResult:
        return self.store.claim_and_consume_host_interaction_response(envelope)
