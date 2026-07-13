from __future__ import annotations

import json
from typing import Any

from vv_agent.types import AgentResult, AgentStatus, CycleRecord, Message

from .state import Checkpoint

_MAX_U32 = (1 << 32) - 1
_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def checkpoint_to_dict(checkpoint: Checkpoint) -> dict[str, Any]:
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    task_id = _task_id(checkpoint.task_id)
    cycle_index = _u32(checkpoint.cycle_index, "cycle_index")
    if not isinstance(checkpoint.status, AgentStatus):
        raise TypeError("checkpoint status must be an AgentStatus")
    if not isinstance(checkpoint.messages, list) or not all(isinstance(item, Message) for item in checkpoint.messages):
        raise TypeError("checkpoint messages must be a list of Message values")
    if not isinstance(checkpoint.cycles, list) or not all(isinstance(item, CycleRecord) for item in checkpoint.cycles):
        raise TypeError("checkpoint cycles must be a list of CycleRecord values")

    payload = {
        "task_id": task_id,
        "cycle_index": cycle_index,
        "status": checkpoint.status.value,
        "messages": [item.to_dict() for item in checkpoint.messages],
        "cycles": [item.to_dict() for item in checkpoint.cycles],
        "shared_state": checkpoint.shared_state,
    }
    if checkpoint.revision:
        payload["revision"] = _u64(checkpoint.revision, "revision")
    if checkpoint.claim_token is not None:
        if not isinstance(checkpoint.claim_token, str) or not checkpoint.claim_token:
            raise ValueError("checkpoint claim_token must be a non-empty string or null")
        payload["claim_token"] = checkpoint.claim_token
    if checkpoint.claimed_cycle is not None:
        payload["claimed_cycle"] = _u32(checkpoint.claimed_cycle, "claimed_cycle")
    if checkpoint.lease_expires_at_ms is not None:
        payload["lease_expires_at_ms"] = _u64(checkpoint.lease_expires_at_ms, "lease_expires_at_ms")
    if checkpoint.terminal_result is not None:
        if not isinstance(checkpoint.terminal_result, AgentResult):
            raise TypeError("checkpoint terminal_result must be an AgentResult or None")
        payload["terminal_result"] = checkpoint.terminal_result.to_dict()
    _validate_control_fields(payload)
    return _json_object(payload, "checkpoint")


def checkpoint_from_dict(payload: Any) -> Checkpoint:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be an object")

    task_id = _task_id(payload.get("task_id"))
    cycle_index = _u32(payload.get("cycle_index"), "cycle_index")
    status_value = payload.get("status")
    if not isinstance(status_value, str):
        raise ValueError("checkpoint status must be a string")
    try:
        status = AgentStatus(status_value)
    except ValueError as exc:
        raise ValueError(f"unknown checkpoint status: {status_value}") from exc

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("checkpoint messages must be an array")
    messages = [_message_from_dict(item, index) for index, item in enumerate(raw_messages)]

    raw_cycles = payload.get("cycles")
    if not isinstance(raw_cycles, list):
        raise ValueError("checkpoint cycles must be an array")
    cycles = [_cycle_from_dict(item, index) for index, item in enumerate(raw_cycles)]

    shared_state = payload.get("shared_state")
    if not isinstance(shared_state, dict):
        raise ValueError("checkpoint shared_state must be an object")

    control = _decode_control_fields(payload, cycle_index=cycle_index, status=status)
    return Checkpoint(
        task_id=task_id,
        cycle_index=cycle_index,
        status=status,
        messages=messages,
        cycles=cycles,
        shared_state=_json_object(shared_state, "checkpoint shared_state"),
        **control,
    )


def checkpoint_to_json(checkpoint: Checkpoint) -> str:
    return json.dumps(
        checkpoint_to_dict(checkpoint),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def checkpoint_from_json(payload: str | bytes) -> Checkpoint:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("checkpoint JSON payload must be a string or bytes")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint JSON: {exc.msg}") from exc
    return checkpoint_from_dict(value)


def clone_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return checkpoint_from_dict(checkpoint_to_dict(checkpoint))


def _task_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint task_id must be a non-empty string")
    return value


def _u32(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_U32:
        raise ValueError(f"checkpoint {field_name} must be an unsigned 32-bit integer")
    return value


def _u64(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
        raise ValueError(f"checkpoint {field_name} must be an unsigned 64-bit integer")
    return value


def _message_from_dict(value: Any, index: int) -> Message:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint messages[{index}] must be an object")
    role = value.get("role")
    if role not in _MESSAGE_ROLES:
        raise ValueError(f"checkpoint messages[{index}].role is invalid")
    if not isinstance(value.get("content"), str):
        raise ValueError(f"checkpoint messages[{index}].content must be a string")
    if "tool_calls" in value and not isinstance(value["tool_calls"], list):
        raise ValueError(f"checkpoint messages[{index}].tool_calls must be an array")
    if "metadata" in value and not isinstance(value["metadata"], dict):
        raise ValueError(f"checkpoint messages[{index}].metadata must be an object")
    for key in ("name", "tool_call_id", "reasoning_content", "image_url"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"checkpoint messages[{index}].{key} must be a string or null")
    try:
        return Message.from_dict(_json_object(value, f"checkpoint messages[{index}]"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint messages[{index}]: {exc}") from exc


def _cycle_from_dict(value: Any, index: int) -> CycleRecord:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint cycles[{index}] must be an object")
    _u32(value.get("index"), f"cycles[{index}].index")
    if not isinstance(value.get("assistant_message"), str):
        raise ValueError(f"checkpoint cycles[{index}].assistant_message must be a string")
    for key in ("tool_calls", "tool_results"):
        if key not in value or not isinstance(value[key], list):
            raise ValueError(f"checkpoint cycles[{index}].{key} must be an array")
    if "memory_compacted" not in value or not isinstance(value["memory_compacted"], bool):
        raise ValueError(f"checkpoint cycles[{index}].memory_compacted must be a boolean")
    if "token_usage" not in value or not isinstance(value["token_usage"], dict):
        raise ValueError(f"checkpoint cycles[{index}].token_usage must be an object")
    try:
        return CycleRecord.from_dict(_json_object(value, f"checkpoint cycles[{index}]"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint cycles[{index}]: {exc}") from exc


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only JSON-compatible values") from exc
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    return decoded


def _decode_control_fields(payload: dict[str, Any], *, cycle_index: int, status: AgentStatus) -> dict[str, Any]:
    revision = _u64(payload.get("revision", 0), "revision")
    claim_token = payload.get("claim_token")
    if claim_token is not None and (not isinstance(claim_token, str) or not claim_token):
        raise ValueError("checkpoint claim_token must be a non-empty string or null")
    raw_claimed_cycle = payload.get("claimed_cycle")
    claimed_cycle = None if raw_claimed_cycle is None else _u32(raw_claimed_cycle, "claimed_cycle")
    raw_lease = payload.get("lease_expires_at_ms")
    lease_expires_at_ms = None if raw_lease is None else _u64(raw_lease, "lease_expires_at_ms")
    raw_terminal = payload.get("terminal_result")
    if raw_terminal is not None and not isinstance(raw_terminal, dict):
        raise ValueError("checkpoint terminal_result must be an object or null")
    try:
        terminal_result = (
            AgentResult.from_dict(_json_object(raw_terminal, "checkpoint terminal_result")) if raw_terminal is not None else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint terminal_result: {exc}") from exc
    fields = {
        "revision": revision,
        "claim_token": claim_token,
        "claimed_cycle": claimed_cycle,
        "lease_expires_at_ms": lease_expires_at_ms,
        "terminal_result": terminal_result,
    }
    _validate_control_fields({**fields, "cycle_index": cycle_index, "status": status})
    return fields


def _validate_control_fields(payload: dict[str, Any]) -> None:
    claim_token = payload.get("claim_token")
    claimed_cycle = payload.get("claimed_cycle")
    lease = payload.get("lease_expires_at_ms")
    claim_values = (claim_token, claimed_cycle, lease)
    if any(value is not None for value in claim_values) and not all(value is not None for value in claim_values):
        raise ValueError("checkpoint claim_token, claimed_cycle, and lease_expires_at_ms must be set together")
    terminal = payload.get("terminal_result")
    if terminal is not None and claim_token is not None:
        raise ValueError("checkpoint terminal_result cannot have an active claim")
    if claimed_cycle is not None:
        cycle_index = payload.get("cycle_index")
        if cycle_index is not None and claimed_cycle != cycle_index + 1:
            raise ValueError("checkpoint claimed_cycle must be exactly cycle_index + 1")
        if claimed_cycle == 0:
            raise ValueError("checkpoint claimed_cycle must be greater than zero")
    terminal_status = (
        terminal.status if isinstance(terminal, AgentResult) else terminal.get("status") if isinstance(terminal, dict) else None
    )
    status = payload.get("status")
    status_value = status.value if isinstance(status, AgentStatus) else status
    terminal_status_value = terminal_status.value if isinstance(terminal_status, AgentStatus) else terminal_status
    if terminal is not None and terminal_status_value != status_value:
        raise ValueError("checkpoint terminal_result status must match checkpoint status")
