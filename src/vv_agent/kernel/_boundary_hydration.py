"""Hydrate typed producer values from their durable record-owner links."""

from __future__ import annotations

import hmac
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ._budget_values import BudgetExhaustion, BudgetUsageSnapshot, RunBudgetLimits
from ._content import INPUT_ACTOR_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES
from ._content_publication import (
    _MISSING,
    _REFERENCE_DIGESTS,
    _content_slot_specs,
    _expand_nested_slots,
    _expand_slot,
    declared_content_ref,
    input_content_slots,
)
from ._driver import CoreDriver
from ._execution_values import AgentTask
from ._hydration import (
    DefinitionContent,
    ReferenceDigestMismatch,
    TypedSchemaFailure,
    _parse_budget,
    _parse_definition,
    _parse_principal_attributes,
    reference_digest,
)
from ._message_values import COMPACTION_METADATA_KEY, Message, validate_compaction_metadata
from ._records import InboxItem, Record, copy_json
from ._store import StoredRecord


def definition_payload(driver: CoreDriver, record: Record) -> dict[str, Any]:
    """Return the hydrated definition as its current canonical JSON container."""
    return _definition_value(driver.hydrated_definition(record))


def active_task(driver: CoreDriver) -> AgentTask:
    """Resolve the active frozen task through the driver's owner-bound definition."""
    tid = driver.state.active_turn_id
    if tid is None:
        raise TypedSchemaFailure("active task requires an active turn")
    return driver.hydrated_definition(driver.state.turns[tid].start).task


def hydrate_record(driver: CoreDriver, record: Record) -> Record:
    """Hydrate one retained record without changing its committed envelope."""
    stored = driver.stored_record(record)
    return hydrate_stored_record(driver, stored)


def hydrate_stored_record(driver: CoreDriver, stored: StoredRecord) -> Record:
    """Hydrate one stored record for a read-only producer decision."""
    record = stored.record
    retained_body = record.encode()
    payload = hydrate_stored_payload(driver, stored)
    hydrated = replace(record, payload=payload)
    # The committed skeleton remains the wire identity used by reducer dedupe.
    object.__setattr__(hydrated, "_body", retained_body)
    if hydrated.kind == "turn_started":
        task = AgentTask.from_dict(copy_json(hydrated.payload["definition"]["task"]))
        object.__setattr__(hydrated, "_task", lambda task=task: task)
    return hydrated


def hydrate_stored_records(driver: CoreDriver, records: Iterable[StoredRecord]) -> tuple[StoredRecord, ...]:
    """Hydrate a chronological record prefix while retaining commit metadata."""
    return tuple(replace(stored, record=hydrate_stored_record(driver, stored)) for stored in records)


def hydrate_stored_payload(
    driver: CoreDriver,
    stored: StoredRecord,
    *,
    include_input: bool = True,
) -> dict[str, Any]:
    """Replace declared C paths while preserving every other stored field."""
    record = stored.record
    payload = copy_json(record.payload)
    owner = f"record:{stored.seq}"

    # Publication and hydration share the same closed outer and nested path table.
    for content_slot in _content_slot_specs(record.kind, record.payload):
        for path, slot, value in _expand_slot(payload, content_slot.path, content_slot.slot):
            if not include_input and record.kind == "input_applied" and path[:1] == ("input",):
                continue
            if value is _MISSING or value is None:
                continue
            hydrated = _hydrate_declared(driver, owner, record, payload, slot, value, content_slot.nested)
            _path_set(payload, path, hydrated)
    return payload


def hydrate_boundary_data(driver: CoreDriver, record: Record) -> dict[str, Any]:
    """Return the hydrated data object for one boundary record."""
    if record.kind != "boundary_recorded":
        raise TypedSchemaFailure("boundary hydration requires a boundary_recorded record")
    hydrated = hydrate_record(driver, record)
    data = hydrated.payload.get("data")
    if type(data) is not dict:
        raise TypedSchemaFailure("boundary_recorded.data must be an object")
    return copy_json(data)


def hydrate_context_compacted(driver: CoreDriver, record: Record) -> Record:
    """Hydrate a compaction replacement and its evidence manifest in array order."""
    if record.kind != "context_compacted":
        raise TypedSchemaFailure("context compaction hydration requires context_compacted")
    return hydrate_record(driver, record)


def hydrate_turn_ended(driver: CoreDriver, record: Record) -> Record:
    """Hydrate terminal reason, result and budget values."""
    if record.kind != "turn_ended":
        raise TypedSchemaFailure("terminal hydration requires turn_ended")
    return hydrate_record(driver, record)


def _hydrate_declared(
    driver: CoreDriver,
    owner: str,
    record: Record,
    payload: dict[str, Any],
    slot: str,
    value: Any,
    nested: tuple[Any, ...] = (),
) -> Any:
    ref = declared_content_ref(value)
    if ref is None:
        raise TypedSchemaFailure(f"{record.kind}/{slot} must be a ContentRef, not plaintext")

    expected = _REFERENCE_DIGESTS.get((record.kind, slot))
    if expected is not None:
        expected_value = payload.get(expected)
        if not isinstance(expected_value, str) or not hmac.compare_digest(reference_digest(ref), expected_value):
            raise ReferenceDigestMismatch(f"{record.kind}/{slot} reference digest does not match {expected}")

    content = driver.read_content_json(owner, slot, ref)
    for _path, nested_slot, nested_value in _expand_nested_slots(content, nested, slot):
        if nested_value is _MISSING or nested_value is None:
            continue
        nested_ref = declared_content_ref(nested_value)
        if nested_ref is None:
            raise TypedSchemaFailure(f"{record.kind}/{nested_slot} must be a ContentRef, not plaintext")
        _path_set(content, _path, driver.read_content_json(owner, nested_slot, nested_ref))
    return _parse_slot(slot, content)


def _parse_slot(slot: str, value: Any) -> Any:
    base = slot.split("[", 1)[0]
    if base == SESSION_PRINCIPAL_ATTRIBUTES:
        try:
            return _parse_principal_attributes(value)
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("principal attributes do not match the current schema") from exc
    if base == "definition":
        if isinstance(value, DefinitionContent):
            return _definition_value(value)
        try:
            value = _parse_definition(value)
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("definition does not match the current schema") from exc
        return _definition_value(value)
    if base == "budget":
        if not isinstance(value, RunBudgetLimits):
            try:
                value = _parse_budget(value)
            except (TypeError, ValueError) as exc:
                raise TypedSchemaFailure("budget does not match the current schema") from exc
        return value.to_dict()
    if base in {"message", "replacement"}:
        try:
            return Message.from_dict(copy_json(value)).to_dict()
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure(f"{slot} must contain one current Message") from exc
    if base == "evidence_manifest":
        if type(value) is not dict:
            raise TypedSchemaFailure("evidence_manifest must be an object")
        try:
            validate_compaction_metadata({COMPACTION_METADATA_KEY: value})
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("evidence_manifest does not match the current schema") from exc
        return copy_json(value)
    if base == "budget_usage":
        try:
            return BudgetUsageSnapshot.from_dict(value).to_dict()
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("budget_usage does not match the current schema") from exc
    if base == "budget_exhaustion":
        try:
            return BudgetExhaustion.from_dict(value).to_dict()
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("budget_exhaustion does not match the current schema") from exc
    if base in {"memory_event", "shared_state"}:
        if type(value) is not dict:
            raise TypedSchemaFailure(f"{slot} must be an object")
        return copy_json(value)
    if base in {"boundary_error", "output_reason", "turn_ended_reason", "session_memory_content"}:
        if value is not None and type(value) is not str:
            raise TypedSchemaFailure(f"{slot} must be a string or null")
        return value
    return copy_json(value)


def hydrate_input(driver: CoreDriver, item: InboxItem) -> InboxItem:
    """Hydrate one inbox item through its exact input owner and slot paths."""

    payload = copy_json(item.payload)
    owner = f"input:{item.input_id}"
    for content_slot in input_content_slots(item.kind):
        for path, slot, value in _expand_slot(payload, content_slot.path, content_slot.slot):
            if value is _MISSING or value is None:
                continue
            ref = declared_content_ref(value)
            if ref is None:
                raise TypedSchemaFailure(f"input:{item.input_id}/{slot} must be a ContentRef, not plaintext")
            _path_set(payload, path, driver.read_content_json(owner, slot, ref))

    actor = copy_json(item.actor)
    if actor is not None:
        attributes = _path_get(actor, ("attributes",))
        ref = declared_content_ref(attributes)
        if ref is None:
            raise TypedSchemaFailure(f"input:{item.input_id}/{INPUT_ACTOR_ATTRIBUTES} must be a ContentRef")
        _path_set(actor, ("attributes",), driver.read_content_json(owner, INPUT_ACTOR_ATTRIBUTES, ref))
    return replace(item, payload=payload, actor=actor)


def _definition_value(definition: DefinitionContent) -> dict[str, Any]:
    return {
        "agent_name": definition.agent_name,
        "memory_settings": copy_json(definition.memory_settings),
        "child_tools": list(definition.child_tools),
        "model_binding": copy_json(definition.model_binding),
        "capabilities": copy_json(definition.capabilities),
        "task": definition.task.to_dict(),
        "tools": copy_json(list(definition.tools)),
    }


def _path_get(value: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        value = value[int(part)] if part.isdecimal() else value[part]
    return value


def _path_set(value: Any, path: tuple[str, ...], replacement: Any) -> None:
    if not path:
        raise ValueError("content path cannot be empty")
    parent = _path_get(value, path[:-1]) if len(path) > 1 else value
    key: str | int = int(path[-1]) if path[-1].isdecimal() else path[-1]
    parent[key] = replacement


__all__ = [
    "active_task",
    "definition_payload",
    "hydrate_boundary_data",
    "hydrate_context_compacted",
    "hydrate_input",
    "hydrate_record",
    "hydrate_stored_payload",
    "hydrate_stored_record",
    "hydrate_stored_records",
    "hydrate_turn_ended",
]
