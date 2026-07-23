from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import (
    DEFAULT_MAX_EXTENSION_STATE_BYTES,
    MAX_EXTENSION_ENTRY_BYTES,
    RUN_DEFINITION_SCHEMA,
    CheckpointError,
    EventCursor,
    canonical_json_bytes,
    validate_checkpoint_extension,
)
from vv_agent.types import AgentResult, AgentStatus, CycleRecord, Message, ModelCallRecord

from .state import (
    CHECKPOINT_SCHEMA,
    Checkpoint,
    EventOutboxEntry,
    ExtensionStateEntry,
    OperationJournalEntry,
    validate_checkpoint,
)

_KNOWN_FIELDS = frozenset(
    {
        "schema_version",
        "run_definition_schema",
        "run_definition",
        "checkpoint_key",
        "task_id",
        "root_run_id",
        "trace_id",
        "run_definition_digest",
        "resume_attempt",
        "cycle_index",
        "status",
        "messages",
        "cycles",
        "model_calls",
        "shared_state",
        "budget_usage",
        "event_cursor",
        "event_outbox",
        "extension_state",
        "model_call_journal",
        "tool_journal",
        "revision",
        "claim_token",
        "claimed_cycle",
        "lease_expires_at_ms",
        "terminal_result",
        "terminal_acknowledged",
    }
)
_MIN_I64 = -(1 << 63)
_MAX_U64 = (1 << 64) - 1


def checkpoint_to_dict(
    checkpoint: Checkpoint,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
) -> dict[str, Any]:
    validate_checkpoint(checkpoint)
    extension_state = {namespace: entry.to_dict() for namespace, entry in sorted(checkpoint.extension_state.items())}
    validate_extension_state_size(
        extension_state,
        max_extension_state_bytes=max_extension_state_bytes,
    )
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "run_definition_schema": checkpoint.run_definition_schema,
        "run_definition": checkpoint.run_definition,
        "checkpoint_key": checkpoint.checkpoint_key,
        "task_id": checkpoint.task_id,
        "root_run_id": checkpoint.root_run_id,
        "trace_id": checkpoint.trace_id,
        "run_definition_digest": checkpoint.run_definition_digest,
        "resume_attempt": checkpoint.resume_attempt,
        "cycle_index": checkpoint.cycle_index,
        "status": checkpoint.status.value,
        "messages": [message.to_dict() for message in checkpoint.messages],
        "cycles": [_cycle_to_dict(cycle) for cycle in checkpoint.cycles],
        "model_calls": [record.to_dict() for record in checkpoint.model_calls],
        "shared_state": checkpoint.shared_state,
        "budget_usage": (checkpoint.budget_usage.to_dict() if checkpoint.budget_usage is not None else None),
        "event_cursor": (checkpoint.event_cursor.to_dict() if checkpoint.event_cursor is not None else None),
        "event_outbox": [entry.to_dict() for entry in checkpoint.event_outbox],
        "extension_state": extension_state,
        "model_call_journal": [entry.to_dict() for entry in checkpoint.model_call_journal],
        "tool_journal": [entry.to_dict() for entry in checkpoint.tool_journal],
        "revision": checkpoint.revision,
        "claim_token": checkpoint.claim_token,
        "claimed_cycle": checkpoint.claimed_cycle,
        "lease_expires_at_ms": checkpoint.lease_expires_at_ms,
        "terminal_result": (checkpoint.terminal_result.to_dict() if checkpoint.terminal_result is not None else None),
        "terminal_acknowledged": checkpoint.terminal_acknowledged,
    }
    return _json_object(payload, "checkpoint v3")


def checkpoint_from_dict(
    payload: Any,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
    registered_extensions: Iterable[Any] | None = None,
) -> Checkpoint:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint v3 payload must be an object")
    unknown_fields = set(payload) - _KNOWN_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise CheckpointError(
            f"checkpoint contains unknown field(s): {names}",
            code="checkpoint_unknown_field",
        )
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"unsupported checkpoint schema_version: {schema_version!r}",
            code="checkpoint_schema_unsupported",
        )
    missing_fields = _KNOWN_FIELDS - set(payload)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise CheckpointError(
            f"checkpoint is missing required field(s): {names}",
            code="checkpoint_missing_field",
        )
    run_definition_schema = payload.get("run_definition_schema")
    if run_definition_schema != RUN_DEFINITION_SCHEMA:
        raise CheckpointError(
            f"unsupported checkpoint run_definition_schema: {run_definition_schema!r}",
            code="checkpoint_definition_schema_unsupported",
        )
    run_definition = payload.get("run_definition")
    if not isinstance(run_definition, dict):
        raise CheckpointError(
            "checkpoint run_definition must be an object",
            code="checkpoint_definition_invalid",
        )
    status_raw = payload.get("status")
    if not isinstance(status_raw, str):
        raise ValueError("checkpoint status must be a string")
    try:
        status = AgentStatus(status_raw)
    except ValueError as exc:
        raise ValueError(f"unknown checkpoint status: {status_raw}") from exc
    messages_raw = _array(payload.get("messages"), "checkpoint messages")
    cycles_raw = _array(payload.get("cycles"), "checkpoint cycles")
    model_calls_raw = _array(payload.get("model_calls"), "checkpoint model_calls")
    shared_state = _object(payload.get("shared_state"), "checkpoint shared_state")
    budget_raw = payload.get("budget_usage")
    if budget_raw is not None and not isinstance(budget_raw, dict):
        raise ValueError("checkpoint budget_usage must be an object or null")
    cursor_raw = payload.get("event_cursor")
    if cursor_raw is not None and not isinstance(cursor_raw, dict):
        raise ValueError("checkpoint event_cursor must be an object or null")
    outbox_raw = _array(payload.get("event_outbox"), "checkpoint event_outbox")
    extensions_raw = _object(payload.get("extension_state"), "checkpoint extension_state")
    validate_extension_state_size(
        extensions_raw,
        max_extension_state_bytes=max_extension_state_bytes,
    )
    _validate_registered_extensions(extensions_raw, registered_extensions)
    model_journal_raw = _array(
        payload.get("model_call_journal"),
        "checkpoint model_call_journal",
    )
    tool_journal_raw = _array(payload.get("tool_journal"), "checkpoint tool_journal")
    terminal_raw = payload.get("terminal_result")
    if terminal_raw is not None and not isinstance(terminal_raw, dict):
        raise ValueError("checkpoint terminal_result must be an object or null")
    checkpoint = Checkpoint(
        schema_version=CHECKPOINT_SCHEMA,
        run_definition_schema=run_definition_schema,
        run_definition=run_definition,
        checkpoint_key=payload.get("checkpoint_key"),
        task_id=payload.get("task_id"),
        root_run_id=payload.get("root_run_id"),
        trace_id=payload.get("trace_id"),
        run_definition_digest=payload.get("run_definition_digest"),
        resume_attempt=payload.get("resume_attempt"),
        cycle_index=payload.get("cycle_index"),
        status=status,
        messages=[Message.from_dict(_object(item, "checkpoint message")) for item in messages_raw],
        cycles=[CycleRecord.from_dict(_object(item, "checkpoint cycle")) for item in cycles_raw],
        model_calls=[
            ModelCallRecord.from_dict(_object(item, "checkpoint model call"))
            for item in model_calls_raw
        ],
        shared_state=shared_state,
        budget_usage=BudgetUsageSnapshot.from_dict(budget_raw) if budget_raw is not None else None,
        event_cursor=EventCursor.from_dict(cursor_raw) if cursor_raw is not None else None,
        event_outbox=[EventOutboxEntry.from_dict(_object(item, "checkpoint outbox entry")) for item in outbox_raw],
        extension_state={
            namespace: ExtensionStateEntry.from_dict(_object(entry, f"checkpoint extension {namespace}"))
            for namespace, entry in extensions_raw.items()
        },
        model_call_journal=[
            OperationJournalEntry.from_dict(_object(item, "checkpoint model journal entry")) for item in model_journal_raw
        ],
        tool_journal=[
            OperationJournalEntry.from_dict(_object(item, "checkpoint tool journal entry")) for item in tool_journal_raw
        ],
        revision=payload.get("revision"),
        claim_token=payload.get("claim_token"),
        claimed_cycle=payload.get("claimed_cycle"),
        lease_expires_at_ms=payload.get("lease_expires_at_ms"),
        terminal_result=AgentResult.from_dict(terminal_raw) if terminal_raw is not None else None,
        terminal_acknowledged=payload.get("terminal_acknowledged"),
    )
    validate_checkpoint(checkpoint)
    return checkpoint


def checkpoint_to_json(
    checkpoint: Checkpoint,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
) -> str:
    return canonical_json_bytes(
        checkpoint_to_dict(
            checkpoint,
            max_extension_state_bytes=max_extension_state_bytes,
        ),
        "checkpoint v3",
    ).decode("utf-8")


def checkpoint_from_json(
    payload: str | bytes,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
    registered_extensions: Iterable[Any] | None = None,
) -> Checkpoint:
    if not isinstance(payload, str | bytes):
        raise TypeError("checkpoint v3 JSON must be str or bytes")
    try:
        decoded = _strict_json_loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("checkpoint v3 JSON is invalid") from exc
    return checkpoint_from_dict(
        decoded,
        max_extension_state_bytes=max_extension_state_bytes,
        registered_extensions=registered_extensions,
    )


def clone_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return checkpoint_from_json(checkpoint_to_json(checkpoint))


def validate_extension_state_size(
    extension_state: dict[str, Any],
    *,
    max_extension_state_bytes: int,
) -> None:
    if (
        isinstance(max_extension_state_bytes, bool)
        or not isinstance(max_extension_state_bytes, int)
        or max_extension_state_bytes < 0
    ):
        raise ValueError("max_extension_state_bytes must be a non-negative integer")
    total = 0
    for namespace, entry in extension_state.items():
        if not isinstance(namespace, str):
            raise ValueError("checkpoint extension namespace must be a string")
        entry_bytes = len(canonical_json_bytes(entry, f"checkpoint extension {namespace}"))
        if entry_bytes > MAX_EXTENSION_ENTRY_BYTES:
            raise CheckpointError(
                f"checkpoint extension {namespace} exceeds {MAX_EXTENSION_ENTRY_BYTES} UTF-8 bytes",
                code="checkpoint_extension_entry_too_large",
            )
        total += entry_bytes
    if total > max_extension_state_bytes:
        raise CheckpointError(
            f"checkpoint extension state exceeds {max_extension_state_bytes} UTF-8 bytes",
            code="checkpoint_extension_state_too_large",
        )


def _validate_registered_extensions(
    extension_state: dict[str, Any],
    registered_extensions: Iterable[Any] | None,
) -> None:
    if registered_extensions is None:
        return
    registered: dict[str, Any] = {}
    for extension in registered_extensions:
        validate_checkpoint_extension(extension)
        namespace = extension.namespace
        if namespace in registered:
            raise CheckpointError(
                f"duplicate checkpoint extension {namespace}",
                code="checkpoint_extension_namespace_duplicate",
            )
        registered[namespace] = extension
    for namespace, raw_entry in extension_state.items():
        entry = _object(raw_entry, f"checkpoint extension {namespace}")
        extension = registered.get(namespace)
        if extension is None:
            if entry.get("required") is True:
                raise CheckpointError(
                    f"required checkpoint extension {namespace} is unavailable",
                    code="checkpoint_required_extension_unavailable",
                )
            continue
        if entry.get("version") != extension.version:
            raise CheckpointError(
                f"checkpoint extension {namespace} version does not match",
                code="checkpoint_extension_version_mismatch",
            )


def _array(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    canonical = canonical_json_bytes(value, field_name)
    decoded = _strict_json_loads(canonical)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must be an object")
    return decoded


def _strict_json_loads(payload: str | bytes) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def parse_integer(value: str) -> int | float:
        parsed = int(value)
        if _MIN_I64 <= parsed <= _MAX_U64:
            return parsed
        # Match serde_json's portable number boundary. JCS can serialize a
        # finite float such as 1e20 without an exponent, so values outside the
        # integer wire range must be restored as IEEE-754 numbers.
        return float(value)

    return json.loads(
        payload,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
        parse_int=parse_integer,
    )


def _cycle_to_dict(cycle: CycleRecord) -> dict[str, Any]:
    return cycle.to_dict()
