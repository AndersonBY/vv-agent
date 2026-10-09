"""Host-interaction request values and their closed wire validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vv_agent.canonical_json import canonical_json_bytes, canonical_json_sha256

CONTROLLER_COMMAND_ID_SCHEMA = "vv-agent.controller-command-id.v1"
CONTROLLER_COMMAND_ID_DOMAIN = "vv-agent.controller-command-id.v1"
HOST_REQUEST_SCHEMA = "vv-agent.host-interaction-request.v1"
_MAX_ID_BYTES = 512
_MAX_CONTENT_BYTES = 65536
_MAX_WIRE_INTEGER = (1 << 53) - 1


def _strict_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(f"{label} fields do not match current schema: missing={sorted(missing)}, unknown={sorted(unknown)}")
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
        prompt = _content(self.prompt, "prompt")
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
        prompt = _content(payload["prompt"], "prompt")
        return cls(
            interaction_id=payload["interaction_id"],
            logical_cycle=payload["logical_cycle"],
            operation_id=payload["operation_id"],
            tool_call_id=payload["tool_call_id"],
            request_digest=request_digest,
            prompt=prompt,
        )


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


HOST_OUTCOME_SCHEMA = "vv-agent.host-interaction-outcome.v1"


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
            self,
            "notification_payload_digest",
            _digest(self.notification_payload_digest, "notification_payload_digest"),
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
