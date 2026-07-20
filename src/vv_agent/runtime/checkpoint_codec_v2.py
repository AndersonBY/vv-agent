from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import (
    DEFAULT_MAX_EXTENSION_STATE_BYTES,
    MAX_EXTENSION_ENTRY_BYTES,
    RUN_DEFINITION_V1_SCHEMA,
    CheckpointError,
    EventCursor,
    canonical_json_bytes,
    compute_run_definition_digest,
    validate_checkpoint_extension,
)
from vv_agent.runtime.checkpoint_codec import (
    checkpoint_from_dict as checkpoint_v1_from_dict,
)
from vv_agent.runtime.state import Checkpoint
from vv_agent.types import AgentResult, AgentStatus, CycleRecord, Message

from .state_v2 import (
    CHECKPOINT_V2_SCHEMA,
    CheckpointV2,
    EventOutboxEntry,
    ExtensionStateEntry,
    OperationJournalEntry,
    validate_checkpoint_v2,
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


def checkpoint_v2_to_dict(
    checkpoint: CheckpointV2,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
) -> dict[str, Any]:
    validate_checkpoint_v2(checkpoint)
    overlapping_unknown = set(checkpoint.unknown_fields) & _KNOWN_FIELDS
    if overlapping_unknown:
        raise CheckpointError(
            f"checkpoint unknown_fields overlap known fields: {sorted(overlapping_unknown)!r}",
            code="checkpoint_unknown_field_invalid",
        )
    extension_state = {namespace: entry.to_dict() for namespace, entry in sorted(checkpoint.extension_state.items())}
    validate_extension_state_size(
        extension_state,
        max_extension_state_bytes=max_extension_state_bytes,
    )
    payload: dict[str, Any] = {
        **checkpoint.unknown_fields,
        "schema_version": CHECKPOINT_V2_SCHEMA,
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
    return _json_object(payload, "checkpoint v2")


def checkpoint_v2_from_dict(
    payload: Any,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
    registered_extensions: Iterable[Any] | None = None,
) -> CheckpointV2:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint v2 payload must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_V2_SCHEMA:
        raise CheckpointError(
            f"unsupported checkpoint schema_version: {schema_version!r}",
            code="checkpoint_schema_unsupported",
        )
    run_definition_schema = payload.get("run_definition_schema")
    if run_definition_schema != RUN_DEFINITION_V1_SCHEMA:
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
    checkpoint = CheckpointV2(
        schema_version=CHECKPOINT_V2_SCHEMA,
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
        unknown_fields={key: value for key, value in payload.items() if key not in _KNOWN_FIELDS},
    )
    validate_checkpoint_v2(checkpoint)
    return checkpoint


def checkpoint_v2_to_json(
    checkpoint: CheckpointV2,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
) -> str:
    return canonical_json_bytes(
        checkpoint_v2_to_dict(
            checkpoint,
            max_extension_state_bytes=max_extension_state_bytes,
        ),
        "checkpoint v2",
    ).decode("utf-8")


def checkpoint_v2_from_json(
    payload: str | bytes,
    *,
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES,
    registered_extensions: Iterable[Any] | None = None,
) -> CheckpointV2:
    if not isinstance(payload, str | bytes):
        raise TypeError("checkpoint v2 JSON must be str or bytes")
    try:
        decoded = _strict_json_loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("checkpoint v2 JSON is invalid") from exc
    return checkpoint_v2_from_dict(
        decoded,
        max_extension_state_bytes=max_extension_state_bytes,
        registered_extensions=registered_extensions,
    )


def clone_checkpoint_v2(checkpoint: CheckpointV2) -> CheckpointV2:
    return checkpoint_v2_from_json(checkpoint_v2_to_json(checkpoint))


def run_definition_comparison_copy(
    definition: dict[str, Any],
) -> dict[str, Any]:
    """Add only contract-0.8 nested defaults to an in-memory comparison copy."""

    comparison = deepcopy(definition)
    tools = comparison.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                tool.setdefault("tool_metadata", None)
    policy = comparison.get("tool_policy")
    if isinstance(policy, dict):
        policy.setdefault("denied_side_effects", [])
        policy.setdefault("denied_capability_tags", [])
        policy.setdefault("deny_terminal_tools", False)
        policy.setdefault("denied_cost_dimensions", [])
    return comparison


def decode_checkpoint_dict(payload: Any) -> Checkpoint | CheckpointV2:
    """Strictly select v1 only when the discriminator is absent."""

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be an object")
    if "schema_version" not in payload:
        return checkpoint_v1_from_dict(payload)
    if payload.get("schema_version") == CHECKPOINT_V2_SCHEMA:
        return checkpoint_v2_from_dict(payload)
    raise CheckpointError(
        f"unsupported checkpoint schema_version: {payload.get('schema_version')!r}",
        code="checkpoint_schema_unsupported",
    )


def decode_checkpoint_json(payload: str | bytes) -> Checkpoint | CheckpointV2:
    if not isinstance(payload, str | bytes):
        raise TypeError("checkpoint JSON must be str or bytes")
    try:
        decoded = _strict_json_loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("checkpoint JSON is invalid") from exc
    return decode_checkpoint_dict(decoded)


def migrate_terminal_checkpoint_v1(
    checkpoint: Checkpoint,
    *,
    checkpoint_key: str,
    root_run_id: str,
    trace_id: str,
    run_definition: dict[str, Any],
) -> CheckpointV2:
    """Build a v2 terminal receipt from an explicitly selected v1 terminal."""

    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a v1 Checkpoint")
    if any(
        value is not None
        for value in (
            checkpoint.claim_token,
            checkpoint.claimed_cycle,
            checkpoint.lease_expires_at_ms,
        )
    ):
        raise CheckpointError(
            "an actively claimed v1 checkpoint cannot be migrated",
            code="checkpoint_migration_active_claim",
        )
    if checkpoint.terminal_result is None:
        raise CheckpointError(
            "a non-terminal v1 checkpoint requires reconciliation",
            code="checkpoint_migration_requires_reconciliation",
        )
    definition_digest = compute_run_definition_digest(run_definition)
    terminal_result = AgentResult.from_dict(checkpoint.terminal_result.to_dict())
    terminal_result.checkpoint_key = checkpoint_key
    migrated = CheckpointV2(
        checkpoint_key=checkpoint_key,
        task_id=checkpoint.task_id,
        root_run_id=root_run_id,
        trace_id=trace_id,
        run_definition=run_definition,
        run_definition_digest=definition_digest,
        resume_attempt=1,
        cycle_index=checkpoint.cycle_index,
        status=checkpoint.status,
        messages=checkpoint.messages,
        cycles=checkpoint.cycles,
        shared_state=checkpoint.shared_state,
        budget_usage=checkpoint.budget_usage,
        terminal_result=terminal_result,
    )
    return clone_checkpoint_v2(migrated)


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
    payload = cycle.to_dict()
    token_usage = payload.get("token_usage")
    if isinstance(token_usage, dict) and token_usage.get("raw") == {}:
        token_usage.pop("raw")
    return payload
