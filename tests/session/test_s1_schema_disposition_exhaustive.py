"""Machine-enforced S1 schema dispositions and skeleton plaintext boundaries."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
from support.s1_fixture_values import content_ref, fixture_key_provider, wrap_fixture_store

from vv_agent.kernel._content import (
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    INPUT_SELECTION,
    ContentRef,
    ContentSessionWriter,
    Principal,
)
from vv_agent.kernel._content_publication import (
    _BOUNDARY_CONTENT_SLOTS,
    _CONTENT_SLOTS,
    _INPUT_SLOTS,
    APPROVAL_METADATA,
    APPROVAL_REASON,
    APPROVAL_SCOPE,
    CHILD_RESULT,
    PROVIDER_EVIDENCE,
    PROVIDER_HANDLE_CANCEL_REF,
    PROVIDER_HANDLE_EVIDENCE,
    PROVIDER_HANDLE_JOB_ID,
    PROVIDER_HANDLE_QUERY_REF,
    PROVIDER_RESULT,
    PROVIDER_USAGE,
    declare_input_content,
    link_declared_input_content,
    make_record_before_publication,
    publish_record_content,
)
from vv_agent.kernel._record_codec import RecordError, validate
from vv_agent.kernel._record_schema import (
    BOUNDARY_DATA,
    CONTENT_REF,
    HANDLE,
    INPUT_PAYLOADS,
    INPUT_SCHEMA,
    INPUT_SCHEMA_LEVEL_2,
    PAYLOADS,
    RECORD_SCHEMA,
    SESSION_CREATED_LEVEL_2,
)
from vv_agent.kernel._records import InboxItem, digest
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_store
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore

SESSION_ID = "s1-disposition-exhaustive"
TENANT_ID = "default"


class Bucket(StrEnum):
    SKELETON = "S"
    CONTENT = "C"
    DERIVED = "DERIVED"
    VALIDATED = "VALIDATED"
    RESERVED = "RESERVED"


@dataclass(frozen=True)
class Disposition:
    bucket: Bucket
    slot: str | None = None


def _put(
    target: dict[str, dict[str, Disposition]],
    kind: str,
    *,
    skeleton: tuple[str, ...] = (),
    content: dict[str, str] | None = None,
    derived: tuple[str, ...] = (),
    validated: tuple[str, ...] = (),
    reserved: tuple[str, ...] = (),
) -> None:
    values = target[kind]
    for bucket, paths in (
        (Bucket.SKELETON, skeleton),
        (Bucket.DERIVED, derived),
        (Bucket.VALIDATED, validated),
        (Bucket.RESERVED, reserved),
    ):
        for path in paths:
            assert path not in values, f"duplicate disposition for {kind}/{path}"
            values[path] = Disposition(bucket)
    for path, slot in (content or {}).items():
        assert path not in values, f"duplicate disposition for {kind}/{path}"
        values[path] = Disposition(Bucket.CONTENT, slot)


RECORD_DISPOSITIONS: dict[str, dict[str, Disposition]] = defaultdict(dict)
INPUT_DISPOSITIONS: dict[str, dict[str, Disposition]] = defaultdict(dict)

_put(
    RECORD_DISPOSITIONS,
    "session_created",
    skeleton=(
        "payload.principal.tenant_id",
        "payload.principal.subject_id",
        "payload.principal.subject_kind",
        "payload.parent_session_id",
        "payload.parent_operation_id",
        "payload.bootstrap.level",
        "payload.bootstrap.handler_version",
        "payload.bootstrap.capabilities",
    ),
    content={
        "payload.principal.attributes": "session.principal.attributes",
        "payload.workspace": "session.workspace",
        "payload.attributes": "session.attributes",
    },
)
_put(
    RECORD_DISPOSITIONS,
    "turn_started",
    skeleton=(
        "payload.input_ids",
        "payload.handler_version",
        "payload.binding",
        "payload.generation",
        "payload.requirements.level",
        "payload.requirements.handler_version",
        "payload.requirements.capabilities",
    ),
    content={
        "payload.definition": "definition",
        "payload.budget": "budget",
        "payload.environment.descriptor": "environment_descriptor",
        "payload.environment.operational_policy": "environment_operational_policy",
    },
    derived=(
        "payload.definition_digest",
        "payload.environment.capabilities_digest",
        "payload.environment.network_policy_ref_digest",
        "payload.environment.isolation_policy_ref_digest",
    ),
    reserved=(
        "payload.policy_digest",
        "payload.tokenizer_binding.model_identity",
        "payload.tokenizer_binding.counting_method",
        "payload.tokenizer_binding.implementation_id",
        "payload.tokenizer_binding.asset_identity",
        "payload.tokenizer_binding.normalization_id",
        "payload.environment.environment_capabilities",
    ),
)
_put(
    RECORD_DISPOSITIONS,
    "input_applied",
    skeleton=(
        "payload.input.schema_version",
        "payload.input.feature_level",
        "payload.input.actor.tenant_id",
        "payload.input.actor.subject_id",
        "payload.input.actor.subject_kind",
        "payload.input.input_id",
        "payload.input.kind",
        "payload.input.target_turn_id",
        "payload.input.generation",
        "payload.input.available_ms",
        "payload.input.requirements.level",
        "payload.input.requirements.handler_version",
        "payload.input.requirements.capabilities",
        "payload.disposition",
        "payload.target_operation_id",
        "payload.target_wait_id",
        "payload.position",
    ),
    content={
        "payload.input.actor.attributes": INPUT_ACTOR_ATTRIBUTES,
        "payload.reason": "input_reason",
        "payload.model_binding": "model_binding",
    },
    derived=("payload.input_digest",),
    validated=(
        "payload.input.admission_proof.admission_key_id",
        "payload.input.admission_proof.admission_signature",
        "payload.input.admission_proof.admission_signed_ms",
        "payload.input_proof.input_seq",
        "payload.input_proof.received_ms",
        "payload.input_proof.key_id",
        "payload.input_proof.signature",
        "payload.input_proof.signed_ms",
    ),
)
_put(
    RECORD_DISPOSITIONS,
    "op_planned",
    skeleton=(
        "payload.op_kind",
        "payload.purpose",
        "payload.context_version",
        "payload.dependencies",
        "payload.idempotency_key",
        "payload.not_before_ms",
        "payload.provider_binding",
        "payload.consumed_unknowns.operation_id",
        "payload.consumed_unknowns.attempt",
    ),
    content={
        "payload.request": "request",
        "payload.tool": "tool",
        "payload.budget_admission": "budget_admission",
    },
    derived=("payload.request_digest",),
    reserved=(
        "payload.policy_decision.policy_digest",
        "payload.policy_decision.rule_id",
        "payload.policy_decision.effect",
    ),
)
_put(
    RECORD_DISPOSITIONS,
    "op_prepared",
    skeleton=("payload.op_kind", "payload.provider_binding", "payload.idempotency_key"),
    content={
        "payload.request": "request",
        "payload.tool": "tool",
        "payload.hook_result": "hook_result",
        "payload.shared_state": "shared_state",
    },
    derived=("payload.request_digest",),
    reserved=(
        "payload.policy_decision.policy_digest",
        "payload.policy_decision.rule_id",
        "payload.policy_decision.effect",
    ),
)
_put(
    RECORD_DISPOSITIONS,
    "turn_parked",
    skeleton=("payload.interaction_id", "payload.source_operation_id", "payload.source_attempt"),
    content={"payload.question": "turn_question"},
)
_put(
    RECORD_DISPOSITIONS,
    "op_started",
    skeleton=("payload.dispatch_id", "payload.authorization_version", "payload.epoch", "payload.mode", "payload.endpoint_id"),
)
_put(
    RECORD_DISPOSITIONS,
    "op_parked",
    skeleton=(
        "payload.phase",
        "payload.poll_at_ms",
        "payload.deadline_ms",
        "payload.yield_on_steer",
        "payload.yieldable_children",
        "payload.handle.kind",
        "payload.handle.provider",
        "payload.handle.operation_id",
        "payload.handle.attempt",
        "payload.handle.request_digest",
        "payload.handle.request_id",
        "payload.handle.interaction_id",
        "payload.handle.session_id",
        "payload.handle.turn_id",
        "payload.handle.generation",
        "payload.handle.background",
        "payload.handle.delivery_target.session_id",
        "payload.handle.delivery_target.turn_id",
        "payload.handle.delivery_target.generation",
        "payload.handle.delivery_target.operation_id",
        "payload.handle.delivery_target.attempt",
        "payload.handle.siblings.session_id",
        "payload.handle.siblings.turn_id",
        "payload.handle.siblings.generation",
        "payload.delegation.mode",
        "payload.delegation.handoff_count",
        "payload.delegation.max_handoffs",
    ),
    content={
        "payload.handle.job_id": "provider_handle_job_id",
        "payload.handle.evidence": "provider_handle_evidence",
        "payload.handle.query_ref": "provider_handle_query_ref",
        "payload.handle.cancel_ref": "provider_handle_cancel_ref",
        "payload.handle.scope": "approval_scope",
        "payload.handle.question": "user_question",
        "payload.interaction_result": "interaction_result",
        "payload.delegation.agent_name": "delegation_agent_name",
        "payload.delegation.metadata": "delegation_metadata",
    },
)
_put(
    RECORD_DISPOSITIONS,
    "op_completed",
    skeleton=(
        "payload.execution_started",
        "payload.context",
        "payload.provider_binding",
        "payload.yielded_by",
    ),
    content={
        "payload.result": "result",
        "payload.usage": "usage",
        "payload.shared_state": "shared_state",
        "payload.evidence": "evidence",
        "payload.usage_counts.budget.host_cost.unit": "usage_counts.host_cost.unit",
        "payload.usage_counts.budget.host_cost.currency": "usage_counts.host_cost.currency",
    },
    derived=("payload.result_digest", "payload.request_digest"),
    validated=(
        "payload.usage_counts.tokens.input_tokens",
        "payload.usage_counts.tokens.output_tokens",
        "payload.usage_counts.tokens.total_tokens",
        "payload.usage_counts.tokens.reasoning_tokens",
        "payload.usage_counts.tokens.usage_source",
        "payload.usage_counts.tokens.cache_usage.status",
        "payload.usage_counts.tokens.cache_usage.read_input_tokens",
        "payload.usage_counts.tokens.cache_usage.write_input_tokens",
        "payload.usage_counts.tokens.cache_usage.uncached_input_tokens",
        "payload.usage_counts.budget.cycles",
        "payload.usage_counts.budget.tool_calls",
        "payload.usage_counts.budget.elapsed_ms",
        "payload.usage_counts.budget.total_tokens",
        "payload.usage_counts.budget.uncached_input_tokens",
        "payload.usage_counts.budget.tool_calls_by_name",
        "payload.usage_counts.budget.host_cost.amount_microunits",
        "payload.usage_counts.budget.unavailable_dimensions.dimension",
        "payload.usage_counts.budget.unavailable_dimensions.reason",
    ),
    reserved=(
        "payload.policy_decision.policy_digest",
        "payload.policy_decision.rule_id",
        "payload.policy_decision.effect",
    ),
)
_put(
    RECORD_DISPOSITIONS,
    "op_unknown",
    skeleton=("payload.retry", "payload.retry_at_ms", "payload.duplicate_cost_risk", "payload.measurement_missing"),
    content={
        "payload.reason": "op_unknown_reason",
        "payload.dispatch_evidence": "op_unknown_dispatch_evidence",
        "payload.observation": "op_unknown_observation",
    },
)
_put(
    RECORD_DISPOSITIONS,
    "context_compacted",
    skeleton=(
        "payload.prefix_ids",
        "payload.tail_ids",
        "payload.mode",
        "payload.summary_operation_id",
        "payload.micro_usage.archived_count",
        "payload.micro_usage.reclaimed_tokens",
        "payload.micro_usage.artifact_failure_count",
    ),
    content={"payload.replacement": "replacement[0]", "payload.evidence_manifest": "evidence_manifest"},
    derived=("payload.source_digest",),
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded",
    skeleton=("payload.boundary_id", "payload.stage", "payload.source_operation_id"),
    derived=("payload.source_digest",),
)
_put(
    RECORD_DISPOSITIONS,
    "usage_observed",
    skeleton=("payload.meter_id", "payload.observation", "payload.mode"),
    content={"payload.usage": "usage", "payload.source": "usage_source"},
)
_put(
    RECORD_DISPOSITIONS,
    "turn_ended",
    skeleton=(
        "payload.status",
        "payload.adopted_results.operation_id",
        "payload.adopted_results.attempt",
        "payload.unconfirmed_operations",
    ),
    content={
        "payload.reason": "turn_ended_reason",
        "payload.result": "turn_ended_result",
        "payload.budget": "turn_ended_budget",
    },
)

_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.before_memory",
    content={"payload.data.messages": "message[0]", "payload.data.shared_state": "shared_state"},
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.after_cycle",
    skeleton=("payload.data.action", "payload.data.disallow_tools", "payload.data.stop.code"),
    content={
        "payload.data.steering_messages": "message[0]",
        "payload.data.stop.message": "stop_message",
        "payload.data.shared_state": "shared_state",
        "payload.data.error": "boundary_error",
    },
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.memory_started",
    content={"payload.data.event": "memory_event"},
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.memory_completed",
    content={"payload.data.event": "memory_event"},
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.session_memory_saved",
    skeleton=(
        "payload.data.state.last_extracted_message_index",
        "payload.data.state.tokens_at_last_extraction",
        "payload.data.state.initialized",
        "payload.data.state.entries.category",
        "payload.data.state.entries.source_cycle",
        "payload.data.state.entries.importance",
    ),
    content={"payload.data.state.entries.content": "session_memory_content[0]"},
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.output_checked",
    skeleton=("payload.data.status",),
    content={
        "payload.data.reason": "output_reason",
        "payload.data.value": "output_value",
        "payload.data.partial_output": "partial_output",
    },
)
_put(
    RECORD_DISPOSITIONS,
    "boundary_recorded.budget",
    skeleton=("payload.data.tool_names",),
    content={"payload.data.usage": "budget_usage", "payload.data.exhaustion": "budget_exhaustion"},
)
for _boundary_stage in BOUNDARY_DATA:
    RECORD_DISPOSITIONS["boundary_recorded"].update(RECORD_DISPOSITIONS[f"boundary_recorded.{_boundary_stage}"])

_COMMON_INPUT_SKELETON = (
    "schema_version",
    "feature_level",
    "input_id",
    "kind",
    "target_turn_id",
    "generation",
    "available_ms",
    "actor.tenant_id",
    "actor.subject_id",
    "actor.subject_kind",
    "requirements.level",
    "requirements.handler_version",
    "requirements.capabilities",
)
_COMMON_INPUT_VALIDATED = (
    "admission_proof.admission_key_id",
    "admission_proof.admission_signature",
    "admission_proof.admission_signed_ms",
)
for _input_kind in INPUT_PAYLOADS:
    _put(
        INPUT_DISPOSITIONS,
        _input_kind,
        skeleton=_COMMON_INPUT_SKELETON,
        content={"actor.attributes": INPUT_ACTOR_ATTRIBUTES},
        validated=_COMMON_INPUT_VALIDATED,
    )

for _input_kind in ("user", "steer", "follow_up"):
    _put(INPUT_DISPOSITIONS, _input_kind, content={"payload.content": INPUT_CONTENT})
_put(INPUT_DISPOSITIONS, "model_selection", content={"payload.selection": INPUT_SELECTION})
_put(
    INPUT_DISPOSITIONS,
    "provider_result",
    skeleton=("payload.operation_id", "payload.attempt", "payload.provider_binding"),
    content={
        "payload.result": PROVIDER_RESULT,
        "payload.usage": PROVIDER_USAGE,
        "payload.evidence": PROVIDER_EVIDENCE,
    },
    derived=("payload.request_digest",),
)
_put(
    INPUT_DISPOSITIONS,
    "approval_answer",
    skeleton=("payload.operation_id", "payload.attempt", "payload.request_id", "payload.decision"),
    content={
        "payload.scope": APPROVAL_SCOPE,
        "payload.reason": APPROVAL_REASON,
        "payload.metadata": APPROVAL_METADATA,
    },
    derived=("payload.request_digest",),
    reserved=("payload.policy_digest", "payload.rule_id", "payload.expires_ms"),
)
_put(
    INPUT_DISPOSITIONS,
    "child_result",
    skeleton=(
        "payload.session_id",
        "payload.turn_id",
        "payload.operation_id",
        "payload.attempt",
        "payload.status",
        "payload.terminal_seq",
    ),
    content={"payload.result": CHILD_RESULT},
    derived=("payload.terminal_digest",),
)
_put(INPUT_DISPOSITIONS, "control", skeleton=("payload.action",))
_put(
    INPUT_DISPOSITIONS,
    "provider_evidence",
    skeleton=(
        "payload.operation_id",
        "payload.attempt",
        "payload.handle.kind",
        "payload.handle.provider",
        "payload.handle.operation_id",
        "payload.handle.attempt",
        "payload.yield_on_steer",
    ),
    content={
        "payload.handle.job_id": PROVIDER_HANDLE_JOB_ID,
        "payload.handle.evidence": PROVIDER_HANDLE_EVIDENCE,
        "payload.handle.query_ref": PROVIDER_HANDLE_QUERY_REF,
        "payload.handle.cancel_ref": PROVIDER_HANDLE_CANCEL_REF,
    },
    derived=("payload.request_digest", "payload.handle.request_digest"),
)

for _input_kind, _payload_disposition in INPUT_DISPOSITIONS.items():
    for _path, _disposition in _payload_disposition.items():
        if _path == "actor.attributes":
            continue
        if _disposition.bucket is not Bucket.CONTENT:
            continue
        _suffix = _path.removeprefix("payload.")
        _record_path = f"payload.input.payload.{_suffix}"
        _record_slot = f"input.payload.{_suffix}"
        existing = RECORD_DISPOSITIONS["input_applied"].get(_record_path)
        if existing is None:
            RECORD_DISPOSITIONS["input_applied"][_record_path] = Disposition(Bucket.CONTENT, _record_slot)
        else:
            assert existing == Disposition(Bucket.CONTENT, _record_slot)
        # Embed every non-content inbox field disposition with the same category.
    for _path, _disposition in _payload_disposition.items():
        if _path == "actor.attributes":
            continue
        _record_path = f"payload.input.{_path}"
        if _record_path in RECORD_DISPOSITIONS["input_applied"]:
            continue
        RECORD_DISPOSITIONS["input_applied"][_record_path] = _disposition

EXPECTED_RECORD_KINDS = frozenset(
    {
        "session_created",
        "turn_started",
        "input_applied",
        "op_planned",
        "op_prepared",
        "turn_parked",
        "op_started",
        "op_parked",
        "op_completed",
        "op_unknown",
        "context_compacted",
        "boundary_recorded",
        "usage_observed",
        "turn_ended",
    }
)
EXPECTED_INPUT_KINDS = frozenset(
    {
        "user",
        "steer",
        "follow_up",
        "model_selection",
        "provider_result",
        "approval_answer",
        "child_result",
        "control",
        "provider_evidence",
    }
)
EXPECTED_BOUNDARIES = frozenset(
    {
        "before_memory",
        "after_cycle",
        "memory_started",
        "memory_completed",
        "session_memory_saved",
        "output_checked",
        "budget",
    }
)
EXPECTED_HANDLE_KINDS = frozenset({"provider", "approval", "user", "child"})

_NESTED_LINK_SLOTS = {
    "session_created": frozenset(
        {
            "session.attributes.seed.messages[0]",
            "session.attributes.child_admission.definition.task.initial_messages[0]",
            "session.attributes.child_admission.definition.task.metadata.vv_session.input_messages[0]",
        }
    ),
    "turn_started": frozenset(
        {
            "definition.task.initial_messages[0]",
            "definition.task.metadata.vv_session.input_messages[0]",
        }
    ),
    "op_planned": frozenset({"request.messages[0]"}),
    "op_prepared": frozenset(
        {
            "request.messages[0]",
            "hook_result.metadata.children[0]",
            "hook_result.metadata.running_children[0]",
        }
    ),
}


def _marker(name: str) -> str:
    return f"PLAINTEXT::{name}::8f2b0d9c"


PLAINTEXT = {
    "prompt": _marker("prompt"),
    "reply": _marker("reply"),
    "tool_arguments": _marker("tool-arguments"),
    "tool_result": _marker("tool-result"),
    "error_text": _marker("error-text"),
    "one_byte_error": "\x1f",
    "attributes": _marker("attributes"),
    "seed": _marker("seed"),
    "route": _marker("model-route"),
    "approval_reason": _marker("approval-reason"),
    "memory": _marker("memory"),
    "archive": _marker("archive/path"),
    "evidence": _marker("evidence"),
    "usage": _marker("usage-source"),
}


def _walk_schema(schema: dict[str, Any], prefix: str, stops: set[str], paths: set[str]) -> None:
    if prefix in stops:
        paths.add(prefix)
        return
    for discriminator in ("oneOf", "anyOf"):
        branches = schema.get(discriminator)
        if branches is not None:
            for branch in branches:
                if branch.get("type") != "null":
                    _walk_schema(branch, prefix, stops, paths)
            return
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        if not properties:
            paths.add(prefix)
            return
        for name, child in properties.items():
            _walk_schema(child, f"{prefix}.{name}" if prefix else name, stops, paths)
        return
    if schema_type == "array":
        _walk_schema(schema["items"], prefix, stops, paths)
        return
    paths.add(prefix)


def _input_schema_for(kind: str) -> dict[str, Any]:
    level_one = dict(INPUT_SCHEMA)
    level_one["properties"] = {**INPUT_SCHEMA["properties"], "payload": INPUT_PAYLOADS[kind]}
    level_two = dict(INPUT_SCHEMA_LEVEL_2)
    level_two["properties"] = {**INPUT_SCHEMA_LEVEL_2["properties"], "payload": INPUT_PAYLOADS[kind]}
    return {"oneOf": [level_one, level_two]}


def _record_payload_schemas(kind: str) -> dict[str, dict[str, Any]]:
    if kind == "session_created":
        return {
            "level-1": PAYLOADS[kind],
            "level-2": SESSION_CREATED_LEVEL_2,
        }
    if kind == "boundary_recorded":
        variants: dict[str, dict[str, Any]] = {}
        for stage, data_schema in BOUNDARY_DATA.items():
            schema = dict(PAYLOADS[kind])
            schema["properties"] = {**PAYLOADS[kind]["properties"], "data": data_schema}
            variants[stage] = schema
        return variants
    if kind == "input_applied":
        variants = {}
        for input_kind in INPUT_PAYLOADS:
            schema = dict(PAYLOADS[kind])
            schema["properties"] = {
                **PAYLOADS[kind]["properties"],
                "input": _input_schema_for(input_kind),
            }
            variants[input_kind] = schema
        return variants
    return {"default": PAYLOADS[kind]}


def _actual_record_paths(kind: str) -> set[str]:
    paths: set[str] = set()
    stops = set(RECORD_DISPOSITIONS[kind])
    for schema in _record_payload_schemas(kind).values():
        _walk_schema(schema, "payload", stops, paths)
    return paths


def _actual_input_paths(kind: str) -> set[str]:
    paths: set[str] = set()
    _walk_schema(_input_schema_for(kind), "", set(INPUT_DISPOSITIONS[kind]), paths)
    return paths


def _schemas_at(schema: dict[str, Any], path: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path:
        return [schema]
    for discriminator in ("oneOf", "anyOf"):
        branches = schema.get(discriminator)
        if branches is not None:
            result: list[dict[str, Any]] = []
            for branch in branches:
                if branch.get("type") != "null":
                    result.extend(_schemas_at(branch, path))
            return result
    if schema.get("type") == "array":
        return _schemas_at(schema["items"], path)
    head, *tail = path
    properties = schema.get("properties", {})
    if head not in properties:
        return []
    return _schemas_at(properties[head], tuple(tail))


def _is_content_boundary(schema: dict[str, Any]) -> bool:
    for discriminator in ("oneOf", "anyOf"):
        branches = schema.get(discriminator)
        if branches is not None:
            non_null = [branch for branch in branches if branch.get("type") != "null"]
            return bool(non_null) and all(_is_content_boundary(branch) for branch in non_null)
    if schema == CONTENT_REF:
        return True
    if schema.get("type") == "array":
        return _is_content_boundary(schema["items"])
    return False


def _record_variant_at(kind: str, variant: str, path: str) -> list[dict[str, Any]]:
    return _schemas_at(
        _record_payload_schemas(kind)[variant],
        tuple(path.removeprefix("payload.").split(".")),
    )


def _input_schema_at(kind: str, path: str) -> list[dict[str, Any]]:
    return _schemas_at(_input_schema_for(kind), tuple(path.split(".")))


def test_s1_kind_inventory_and_producer_inventories_are_exhaustive() -> None:
    assert len(PAYLOADS) == 14
    assert set(RECORD_SCHEMA["properties"]["kind"]["enum"]) == EXPECTED_RECORD_KINDS
    assert set(INPUT_PAYLOADS) == EXPECTED_INPUT_KINDS
    assert len(INPUT_PAYLOADS) == 9
    assert set(INPUT_SCHEMA["properties"]["kind"]["enum"]) == EXPECTED_INPUT_KINDS
    assert set(INPUT_SCHEMA_LEVEL_2["properties"]["kind"]["enum"]) == EXPECTED_INPUT_KINDS
    assert set(_CONTENT_SLOTS) == EXPECTED_RECORD_KINDS
    assert set(_INPUT_SLOTS) == EXPECTED_INPUT_KINDS
    assert set(_BOUNDARY_CONTENT_SLOTS) == EXPECTED_BOUNDARIES
    assert len(EXPECTED_BOUNDARIES) == 7
    assert {variant["properties"]["kind"]["const"] for variant in HANDLE["oneOf"]} == EXPECTED_HANDLE_KINDS


@pytest.mark.parametrize("kind", sorted(EXPECTED_RECORD_KINDS))
def test_every_record_field_has_exactly_one_disposition(kind: str) -> None:
    expected = RECORD_DISPOSITIONS[kind]
    actual = _actual_record_paths(kind)
    assert actual == set(expected), f"unclassified={sorted(actual - set(expected))}; stale={sorted(set(expected) - actual)}"
    for path, disposition in expected.items():
        if disposition.bucket is not Bucket.CONTENT:
            continue
        schema_candidates = [
            candidate
            for variant_name in _record_payload_schemas(kind)
            for candidate in _record_variant_at(kind, variant_name, path)
        ]
        assert schema_candidates, f"no schema at {kind}/{path}"
        assert all(_is_content_boundary(candidate) for candidate in schema_candidates), (
            f"{kind}/{path} is not a typed ContentRef boundary"
        )


@pytest.mark.parametrize("kind", sorted(EXPECTED_INPUT_KINDS))
def test_every_inbox_field_has_exactly_one_disposition(kind: str) -> None:
    expected = INPUT_DISPOSITIONS[kind]
    actual = _actual_input_paths(kind)
    assert actual == set(expected), f"unclassified={sorted(actual - set(expected))}; stale={sorted(set(expected) - actual)}"
    for path, disposition in expected.items():
        if disposition.bucket is not Bucket.CONTENT:
            continue
        schemas = _input_schema_at(kind, path)
        assert schemas and all(_is_content_boundary(schema) for schema in schemas), (
            f"inbox {kind}/{path} is not a typed ContentRef boundary"
        )


def test_after_cycle_stop_code_disposition_rejects_free_text() -> None:
    value = {
        "action": "stop_non_success",
        "steering_messages": [],
        "disallow_tools": [],
        "stop": {"code": PLAINTEXT["prompt"], "message": content_ref("stop-message")},
        "shared_state": content_ref("shared-state"),
        "error": None,
    }
    with pytest.raises(RecordError, match="invalid session schema"):
        validate(value, BOUNDARY_DATA["after_cycle"])


@dataclass(frozen=True)
class BuiltStorage:
    raw: SQLiteStore
    wrapped: Any
    records: tuple[Any, ...]
    links: tuple[Any, ...]
    markers: tuple[str, ...]


def _content(prefix: str) -> dict[str, Any]:
    return {prefix: PLAINTEXT[prefix], "components": [PLAINTEXT["prompt"], PLAINTEXT["reply"]]}


def _input_payload(kind: str) -> dict[str, Any]:
    if kind in {"user", "steer", "follow_up"}:
        value = "prompt" if kind == "user" else "reply"
        return {"content": _content(value)}
    if kind == "model_selection":
        return {"selection": {"model": "model/opaque", "model_settings": {}, "route": PLAINTEXT["route"]}}
    if kind == "provider_result":
        return {
            "operation_id": "op/opaque",
            "attempt": 1,
            "request_digest": "1" * 64,
            "provider_binding": "provider/opaque",
            "result": _content("tool_result"),
            "usage": _content("usage"),
            "evidence": _content("evidence"),
        }
    if kind == "approval_answer":
        return {
            "operation_id": "op/opaque",
            "attempt": 1,
            "request_id": "approval/opaque",
            "request_digest": "2" * 64,
            "decision": "approve",
            "scope": {"tool": "opaque", "arguments": PLAINTEXT["tool_arguments"]},
            "reason": _content("approval_reason"),
            "metadata": _content("attributes"),
            "policy_digest": None,
            "rule_id": None,
            "expires_ms": None,
        }
    if kind == "child_result":
        return {
            "session_id": "child/fixture",
            "turn_id": "turn/opaque",
            "operation_id": "op/opaque",
            "attempt": 1,
            "status": "completed",
            "result": _content("reply"),
            "terminal_seq": 4,
            "terminal_digest": "3" * 64,
        }
    if kind == "control":
        return {"action": "cancel"}
    if kind == "provider_evidence":
        return {
            "operation_id": "op/opaque",
            "attempt": 1,
            "request_digest": "4" * 64,
            "handle": {
                "kind": "provider",
                "provider": "provider/opaque",
                "job_id": _content("tool_result"),
                "operation_id": "op/opaque",
                "attempt": 1,
                "request_digest": "4" * 64,
                "evidence": _content("evidence"),
                "query_ref": _content("route"),
                "cancel_ref": _content("archive"),
            },
            "yield_on_steer": True,
        }
    raise AssertionError(kind)


def _push_input(wrapped: Any, kind: str) -> None:
    input_id = f"input-{kind}"
    with wrapped.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT_ID, session_id=SESSION_ID)
        actor_attributes = writer.put_json({"attributes": PLAINTEXT["attributes"]})
        actor = Principal(
            tenant_id=TENANT_ID,
            subject_id=f"subject-{kind}",
            subject_kind="service" if kind in {"provider_result", "provider_evidence", "child_result"} else "user",
            attributes=actor_attributes,
        ).to_dict()
        item = InboxItem(
            input_id=input_id,
            kind=kind,
            payload=_input_payload(kind),
            target_turn_id=None if kind == "model_selection" else "turn/opaque",
            generation=None if kind == "model_selection" else 1,
            available_ms=0,
            actor=actor,
        )
        published = declare_input_content(writer, item)
        tx.push(SESSION_ID, published.item)
        link_declared_input_content(writer, published.item)


def _create_session(wrapped: Any) -> None:
    with wrapped.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT_ID, session_id=SESSION_ID)

        def spec(content: ContentSessionWriter) -> SessionSpec:
            principal_attributes = content.put_json({"attributes": PLAINTEXT["attributes"]})
            principal = Principal(
                tenant_id=TENANT_ID,
                subject_id="fixture/host",
                subject_kind="service",
                attributes=principal_attributes,
            ).to_dict()
            attributes = {
                "seed": {
                    "messages": [{"role": "user", "content": PLAINTEXT["seed"]}],
                    "shared_state": {"memory": PLAINTEXT["memory"]},
                },
                "app_server": {
                    "agent_key": PLAINTEXT["attributes"],
                    "cwd": PLAINTEXT["archive"],
                    "metadata": {"route": PLAINTEXT["route"]},
                },
                "child_admission": {
                    "definition": {
                        "task": {
                            "initial_messages": [{"role": "user", "content": PLAINTEXT["prompt"]}],
                            "metadata": {
                                "vv_session": {
                                    "input_messages": [{"role": "user", "content": PLAINTEXT["reply"]}],
                                }
                            },
                        }
                    }
                },
            }
            return SessionSpec(
                SESSION_ID,
                principal,
                content.put_json({"workspace": PLAINTEXT["archive"]}).to_dict(),
                attributes=attributes,
            )

        writer.create_session(spec, consumers=("events",))
        child_writer = SessionContentWriter.bind(tx, tenant_id=TENANT_ID, session_id="child/fixture")

        def child_spec(content: ContentSessionWriter) -> SessionSpec:
            actor, _ = content.service_principal("fixture/child")
            return SessionSpec(
                "child/fixture",
                actor,
                content.put_json("/tmp").to_dict(),
                parent_session_id=SESSION_ID,
                parent_operation_id="op/opaque",
                attributes={"seed": {"messages": [], "shared_state": {}}},
            )

        child_writer.create_session(child_spec, consumers=())


def _record(kind: str, payload: dict[str, Any], *, turn_id: str | None = "turn/opaque", operation_id: str | None = None):
    is_operation = kind.startswith("op_")
    return make_record_before_publication(
        kind,
        session_id=SESSION_ID,
        turn_id=turn_id,
        operation_id=operation_id if operation_id is not None else ("op/opaque" if is_operation else None),
        attempt=1 if is_operation else None,
        payload=payload,
    )


def _record_fixtures(wrapped: Any) -> tuple[Any, ...]:
    request = {
        "messages": [{"role": "user", "content": PLAINTEXT["prompt"]}],
        "metadata": {"vv_session": {"endpoint_order": [PLAINTEXT["route"]]}},
    }
    records = [
        _record(
            "turn_started",
            {
                "input_ids": ["input-user"],
                "definition": {
                    "task": {
                        "initial_messages": [{"role": "user", "content": PLAINTEXT["prompt"]}],
                        "metadata": {"vv_session": {"input_messages": [{"role": "user", "content": PLAINTEXT["seed"]}]}},
                    }
                },
                "definition_digest": digest(
                    {
                        "task": {
                            "initial_messages": [{"role": "user", "content": PLAINTEXT["prompt"]}],
                            "metadata": {"vv_session": {"input_messages": [{"role": "user", "content": PLAINTEXT["seed"]}]}},
                        }
                    }
                ),
                "handler_version": "1",
                "budget": {"max_cycles": 3, "note": PLAINTEXT["usage"]},
                "binding": "binding/opaque",
                "generation": 1,
                "policy_digest": None,
                "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                "tokenizer_binding": None,
                "environment": {
                    "descriptor": {"route": PLAINTEXT["route"]},
                    "environment_capabilities": ["env/opaque"],
                    "capabilities_digest": "5" * 64,
                    "network_policy_ref_digest": "6" * 64,
                    "isolation_policy_ref_digest": "7" * 64,
                    "operational_policy": {"network": PLAINTEXT["route"]},
                },
            },
        ),
        _record(
            "op_planned",
            {
                "op_kind": "model",
                "purpose": "primary",
                "request": request,
                "request_digest": digest(request),
                "context_version": "context/opaque",
                "dependencies": ["dependency/opaque"],
                "tool": {"arguments": PLAINTEXT["tool_arguments"]},
                "idempotency_key": "idempotency/opaque",
                "budget_admission": {"usage": PLAINTEXT["usage"]},
                "not_before_ms": 0,
                "provider_binding": "provider/opaque",
                "consumed_unknowns": [{"operation_id": "op/unknown", "attempt": 1}],
                "policy_decision": None,
            },
        ),
        _record(
            "op_prepared",
            {
                "request": request,
                "request_digest": digest(request),
                "op_kind": "tool",
                "tool": {"arguments": PLAINTEXT["tool_arguments"]},
                "provider_binding": "provider/opaque",
                "idempotency_key": "idempotency/opaque",
                "hook_result": {
                    "metadata": {
                        "children": [{"result": PLAINTEXT["reply"]}],
                        "running_children": [{"evidence": PLAINTEXT["evidence"]}],
                    }
                },
                "shared_state": {"memory": PLAINTEXT["memory"]},
                "policy_decision": None,
            },
        ),
        _record(
            "turn_parked",
            {
                "interaction_id": "interaction/opaque",
                "question": {"text": PLAINTEXT["prompt"]},
                "source_operation_id": "op/source",
                "source_attempt": 1,
            },
        ),
        _record(
            "op_started",
            {
                "dispatch_id": "dispatch/opaque",
                "authorization_version": "authorization/opaque",
                "epoch": 1,
                "mode": "sync",
                "endpoint_id": "endpoint/opaque",
            },
        ),
        _record(
            "op_parked",
            {
                "phase": "after_dispatch",
                "handle": {
                    "kind": "provider",
                    "provider": "provider/opaque",
                    "job_id": _content("tool_result"),
                    "operation_id": "op/provider",
                    "attempt": 1,
                    "request_digest": "8" * 64,
                    "evidence": _content("evidence"),
                    "query_ref": _content("route"),
                    "cancel_ref": _content("archive"),
                },
                "poll_at_ms": None,
                "deadline_ms": None,
                "interaction_result": _content("reply"),
                "delegation": {
                    "mode": "configured",
                    "agent_name": _content("attributes"),
                    "handoff_count": 0,
                    "max_handoffs": 1,
                    "metadata": _content("memory"),
                },
                "yield_on_steer": True,
                "yieldable_children": [],
            },
            operation_id="op/provider",
        ),
        _record(
            "op_parked",
            {
                "phase": "before_dispatch",
                "handle": {
                    "kind": "approval",
                    "request_id": "approval/opaque",
                    "request_digest": "9" * 64,
                    "scope": _content("tool_arguments"),
                },
                "poll_at_ms": None,
                "deadline_ms": None,
            },
            operation_id="op/approval",
        ),
        _record(
            "op_parked",
            {
                "phase": "after_dispatch",
                "handle": {
                    "kind": "user",
                    "interaction_id": "interaction/opaque",
                    "question": _content("prompt"),
                },
                "poll_at_ms": None,
                "deadline_ms": None,
            },
            operation_id="op/user",
        ),
        _record(
            "op_parked",
            {
                "phase": "after_dispatch",
                "handle": {
                    "kind": "child",
                    "session_id": "child/opaque",
                    "turn_id": "turn/child",
                    "generation": 1,
                    "background": False,
                    "delivery_target": {
                        "session_id": "parent/opaque",
                        "turn_id": "turn/parent",
                        "generation": 1,
                        "operation_id": "op/child",
                        "attempt": 1,
                    },
                },
                "poll_at_ms": None,
                "deadline_ms": None,
            },
            operation_id="op/child",
        ),
        _record(
            "op_completed",
            {
                "result": _content("reply"),
                "result_digest": digest(_content("reply")),
                "usage": _content("usage"),
                "usage_counts": {
                    "tokens": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "reasoning_tokens": None,
                        "usage_source": "provider_reported",
                        "cache_usage": {
                            "status": "provider_reported",
                            "read_input_tokens": 0,
                            "write_input_tokens": 0,
                            "uncached_input_tokens": 1,
                        },
                    },
                    "budget": {
                        "cycles": 1,
                        "tool_calls": 1,
                        "elapsed_ms": 1,
                        "total_tokens": 2,
                        "uncached_input_tokens": 1,
                        "tool_calls_by_name": {"opaque": 1},
                        "host_cost": {
                            "amount_microunits": 1,
                            "unit": _content("usage"),
                            "currency": _content("route"),
                        },
                        "unavailable_dimensions": [{"dimension": "wall_time", "reason": "accounting_missing"}],
                    },
                },
                "shared_state": _content("memory"),
                "evidence": _content("evidence"),
                "execution_started": True,
                "context": "normal",
                "request_digest": "a" * 64,
                "provider_binding": "provider/opaque",
                "policy_decision": None,
            },
        ),
        _record(
            "op_unknown",
            {
                "reason": _content("error_text"),
                "dispatch_evidence": _content("evidence"),
                "observation": _content("tool_result"),
                "retry": "manual",
                "retry_at_ms": None,
                "duplicate_cost_risk": True,
                "measurement_missing": True,
            },
        ),
        _record(
            "context_compacted",
            {
                "source_digest": "b" * 64,
                "prefix_ids": ["record/opaque"],
                "tail_ids": ["record/opaque-2"],
                "mode": "summary",
                "summary_operation_id": "op/opaque",
                "replacement": [_content("reply")],
                "evidence_manifest": {"archive": PLAINTEXT["archive"]},
                "micro_usage": {"archived_count": 1, "reclaimed_tokens": 1, "artifact_failure_count": 0},
            },
        ),
        _record(
            "usage_observed",
            {
                "meter_id": "meter/opaque",
                "observation": 1,
                "mode": "cumulative",
                "usage": _content("usage"),
                "source": _content("attributes"),
            },
            turn_id=None,
        ),
        _record(
            "turn_ended",
            {
                "status": "completed",
                "reason": _content("error_text"),
                "result": _content("reply"),
                "adopted_results": [{"operation_id": "op/opaque", "attempt": 1}],
                "budget": _content("usage"),
                "unconfirmed_operations": [],
            },
        ),
    ]

    for input_id in (
        stored.item.input_id for stored in (wrapped.read_input(SESSION_ID, f"input-{kind}") for kind in INPUT_PAYLOADS)
    ):
        stored = wrapped.read_input(SESSION_ID, input_id)
        assert stored is not None
        records.append(
            _record(
                "input_applied",
                {
                    "input": stored.item.to_dict(),
                    "input_digest": stored.item.digest,
                    "input_proof": None,
                    "disposition": "applied",
                    "reason": _content("approval_reason"),
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "before_dispatch",
                    "model_binding": _content("route"),
                },
                turn_id="turn/opaque",
            )
        )

    boundary_data = {
        "before_memory": {
            "messages": [{"role": "user", "content": PLAINTEXT["prompt"]}],
            "shared_state": {"memory": PLAINTEXT["memory"]},
        },
        "after_cycle": {
            "action": "stop_non_success",
            "steering_messages": [{"role": "user", "content": PLAINTEXT["reply"]}],
            "disallow_tools": ["tool/opaque"],
            "stop": {"code": "hook_stop", "message": {"text": PLAINTEXT["error_text"]}},
            "shared_state": {"memory": PLAINTEXT["memory"]},
            "error": PLAINTEXT["one_byte_error"],
        },
        "memory_started": {"event": {"memory": PLAINTEXT["memory"]}},
        "memory_completed": {"event": {"memory": PLAINTEXT["memory"]}},
        "session_memory_saved": {
            "state": {
                "last_extracted_message_index": 1,
                "tokens_at_last_extraction": 1,
                "initialized": True,
                "entries": [{"category": "fact", "source_cycle": 1, "importance": 1, "content": _content("memory")}],
            }
        },
        "output_checked": {
            "status": "completed",
            "reason": _content("error_text"),
            "value": _content("reply"),
            "partial_output": _content("prompt"),
        },
        "budget": {
            "tool_names": ["tool/opaque"],
            "usage": _content("usage"),
            "exhaustion": _content("error_text"),
        },
    }
    for stage, data in boundary_data.items():
        records.append(
            _record(
                "boundary_recorded",
                {
                    "boundary_id": f"boundary/{stage}",
                    "stage": stage,
                    "source_operation_id": "op/opaque",
                    "source_digest": "c" * 64,
                    "data": data,
                },
            )
        )
    return tuple(records)


def _build_storage(tmp_path: Path) -> Iterator[BuiltStorage]:
    with SQLiteStore.standalone(str(tmp_path / "s1-dispositions.sqlite")) as raw:
        raw.install_schema()
        raw.configure_input_admission(allow_unsigned=True)
        wrapped = wrap_fixture_store(raw, fixture_key_provider())
        _create_session(wrapped)
        for kind in INPUT_PAYLOADS:
            _push_input(wrapped, kind)
        records = _record_fixtures(wrapped)
        with wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT_ID, session_id=SESSION_ID)
            publication = publish_record_content(writer, records)
        yield BuiltStorage(
            raw,
            wrapped,
            publication.records,
            publication.links,
            tuple(PLAINTEXT.values()),
        )


@pytest.fixture
def s1_storage(tmp_path: Path) -> Iterator[BuiltStorage]:
    yield from _build_storage(tmp_path)


def _expected_record_slots(kind: str) -> set[str]:
    slots = {item.slot for item in RECORD_DISPOSITIONS[kind].values() if item.slot is not None}
    slots.update(_NESTED_LINK_SLOTS.get(kind, ()))
    return slots


def test_every_content_disposition_has_a_typed_link(s1_storage: BuiltStorage) -> None:
    raw = s1_storage.raw
    sql_links = raw._rows(
        "SELECT r.kind,r.seq,r.record_id,l.slot FROM sk_content_link l "
        "JOIN sk_record r ON r.tenant_id=l.tenant_id AND r.session_id=l.session_id AND r.seq=l.record_seq "
        "WHERE r.tenant_id=%s AND r.session_id=%s ORDER BY r.seq,l.slot",
        (TENANT_ID, SESSION_ID),
    )
    slots_by_kind: dict[str, set[str]] = defaultdict(set)
    for kind, _seq, _record_id, slot in sql_links:
        slots_by_kind[kind].add(slot)
    kinds_by_record_id = {record.record_id: record.kind for record in s1_storage.records}
    for link in s1_storage.links:
        slots_by_kind[kinds_by_record_id[link.record_id]].add(link.slot)

    for kind in EXPECTED_RECORD_KINDS:
        expected = _expected_record_slots(kind)
        actual = slots_by_kind[kind]
        assert actual == expected, (
            f"{kind}: missing links={sorted(expected - actual)}; undeclared links={sorted(actual - expected)}"
        )

    input_links = raw._rows(
        "SELECT i.kind,l.slot FROM sk_content_link l JOIN sk_inbox i "
        "ON i.tenant_id=l.tenant_id AND i.session_id=l.session_id AND i.input_id=l.input_id "
        "WHERE i.tenant_id=%s AND i.session_id=%s ORDER BY i.kind,l.slot",
        (TENANT_ID, SESSION_ID),
    )
    slots_by_input_kind: dict[str, set[str]] = defaultdict(set)
    for kind, slot in input_links:
        slots_by_input_kind[kind].add(slot)
    for kind in EXPECTED_INPUT_KINDS:
        expected = {item.slot for item in INPUT_DISPOSITIONS[kind].values() if item.slot is not None}
        assert slots_by_input_kind[kind] == expected, (
            f"inbox {kind}: missing links={sorted(expected - slots_by_input_kind[kind])}; "
            f"undeclared links={sorted(slots_by_input_kind[kind] - expected)}"
        )


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, bytes):
        yield from (part.decode("utf-8", "ignore") for part in value.split(b"\x00") if part)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _leaks(label: str, value: Any, markers: tuple[str, ...]) -> list[str]:
    strings = tuple(_strings(value))
    return [f"{label}: {marker!r}" for marker in markers if any(marker in item for item in strings)]


def test_s1_encoded_and_sql_skeleton_contain_no_content_plaintext(s1_storage: BuiltStorage) -> None:
    raw = s1_storage.raw

    sql_record_bodies = raw._rows(
        "SELECT seq,body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
        (TENANT_ID, SESSION_ID),
    )
    inbox_bodies = raw._rows(
        "SELECT input_id,body FROM sk_inbox WHERE tenant_id=%s AND session_id=%s ORDER BY input_seq",
        (TENANT_ID, SESSION_ID),
    )
    sql_record_kinds = {
        kind
        for kind, _ in raw._rows("SELECT kind,1 FROM sk_record WHERE tenant_id=%s AND session_id=%s", (TENANT_ID, SESSION_ID))
    }
    assert sql_record_kinds <= EXPECTED_RECORD_KINDS
    assert {
        kind for kind, _ in raw._rows("SELECT kind,1 FROM sk_inbox WHERE tenant_id=%s AND session_id=%s", (TENANT_ID, SESSION_ID))
    } == EXPECTED_INPUT_KINDS

    leaks: list[str] = []
    for label, body in (
        *((f"sql-record[{seq}].body", body) for seq, body in sql_record_bodies),
        *((f"produced-record[{record.kind}].body", record.encode()) for record in s1_storage.records),
        *((f"inbox[{input_id}].body", body) for input_id, body in inbox_bodies),
    ):
        parsed = json.loads(body)
        leaks.extend(_leaks(label + ".bytes", body, s1_storage.markers))
        leaks.extend(_leaks(label + ".json", parsed, s1_storage.markers))

    produced_kinds = {record.kind for record in s1_storage.records} | sql_record_kinds
    assert produced_kinds == EXPECTED_RECORD_KINDS

    encrypted_columns = {"admission_signature", "nonce", "ciphertext", "object_locator", "wrapped_key"}
    tables = raw._rows("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sk_%' ORDER BY name")
    for (table,) in tables:
        columns = [row[1] for row in raw._rows(f"PRAGMA table_info('{table}')") if row[1] not in encrypted_columns]
        if not columns:
            continue
        projection = ",".join(f'"{column}"' for column in columns)
        for row_number, row in enumerate(raw._rows(f'SELECT {projection} FROM "{table}"'), start=1):
            for column, value in zip(columns, row, strict=True):
                leaks.extend(_leaks(f"sql.{table}[{row_number}].{column}", value, s1_storage.markers))
                if isinstance(value, str) and value[:1] in {"{", "["}:
                    with suppress(json.JSONDecodeError):
                        leaks.extend(_leaks(f"sql.{table}[{row_number}].{column}.json", json.loads(value), s1_storage.markers))

    assert not leaks, "plaintext content reached S1 skeleton storage:\n" + "\n".join(leaks)

    content_rows = raw._rows(
        "SELECT content_id,key_id,mac,size,labels FROM sk_content WHERE tenant_id=%s AND session_id=%s",
        (TENANT_ID, SESSION_ID),
    )
    content_values = []
    durable = durable_content_store(s1_storage.wrapped)
    for content_id, key_id, mac, size, labels in content_rows:
        labels_value = json.loads(labels)
        ref = ContentRef.from_dict(
            {
                "content_id": content_id,
                "key_id": key_id,
                "mac": mac,
                "size": size,
                "labels": labels_value,
            }
        )
        content_values.append(durable.get(TENANT_ID, SESSION_ID, ref))
    decrypted_strings: list[str] = []
    for content_value in content_values:
        try:
            decrypted_strings.extend(_strings(json.loads(content_value)))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decrypted_strings.append(content_value.decode("utf-8", "ignore"))
    for marker in s1_storage.markers:
        assert any(marker in value for value in decrypted_strings), (
            f"fixture marker was not exercised in encrypted content: {marker!r}"
        )
