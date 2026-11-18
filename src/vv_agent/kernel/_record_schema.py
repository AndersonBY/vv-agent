"""Authoritative S1 logical session schemas and decoded-content validators.

The runtime remains stdlib-only. These are the current Python S1 candidate
logical shapes. The plaintext schemas below validate typed content after the
implemented S1 content-store hydration step and are never accepted in a logical
C slot.
"""

from __future__ import annotations

from typing import Any

from ._schema import _compile_check


def closed(**fields: Any) -> dict[str, Any]:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def array(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": schema}


TEXT = {"type": "string", "minLength": 1}
ANY_OBJECT = {"type": "object"}
NAT = {"type": "integer", "minimum": 0}
POS = {"type": "integer", "minimum": 1}
BOOL = {"type": "boolean"}
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$", "maxLength": 64}
STRING = {"type": "string"}
STRINGS = array(TEXT)
NULLABLE_TEXT = nullable(TEXT)
NULLABLE_HASH = nullable(HASH)
NULLABLE_NAT = nullable(NAT)
SIGNATURE = {
    "type": "string",
    "pattern": "^(?:[A-Za-z0-9+/]{4})+(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\\Z",
    "maxLength": 2_147_483_647,
}
CONTROL = enum("close", "archive", "suspend", "resume", "cancel", "abort")
# Only registered kernel codes remain in the clear S payload. Host-defined
# codes are represented by the generic code while their detail stays in C.
AFTER_CYCLE_KERNEL_STOP_CODES = (
    "host.policy_stop",
    "host_stop",
    "hook_stop",
    "stopped",
)
AFTER_CYCLE_GENERIC_STOP_CODE = "host_stop"


# Shared logical definitions keep one schema owner for every Principal,
# ContentRef, proof, usage and handle slot.
CONTENT_LABELS = closed(
    source=array(enum("system", "user", "internal", "untrusted")),
    classification=enum("public", "internal", "confidential", "restricted"),
)
CONTENT_REF = closed(content_id=TEXT, mac=HASH, size=NAT, key_id=TEXT, labels=CONTENT_LABELS)
PRINCIPAL = closed(
    tenant_id=TEXT,
    subject_id=TEXT,
    subject_kind=enum("user", "service", "agent"),
    attributes=CONTENT_REF,
)
ADMISSION_PROOF = nullable(closed(admission_key_id=TEXT, admission_signature=SIGNATURE, admission_signed_ms=NAT))
INPUT_PROOF = nullable(closed(input_seq=POS, received_ms=NAT, key_id=TEXT, signature=SIGNATURE, signed_ms=NAT))
POLICY_DECISION = nullable(closed(policy_digest=HASH, rule_id=TEXT, effect=enum("allow", "deny", "require_approval")))
TOKENIZER_BINDING = nullable(
    closed(
        model_identity=TEXT,
        counting_method=enum("provider_reported", "tokenizer"),
        implementation_id=TEXT,
        asset_identity=TEXT,
        normalization_id=TEXT,
    )
)
CACHE_USAGE = closed(
    status=enum("provider_reported", "accounting_missing", "unsupported"),
    read_input_tokens=NULLABLE_NAT,
    write_input_tokens=NULLABLE_NAT,
    uncached_input_tokens=NULLABLE_NAT,
)
TOKEN_COUNTS = closed(
    input_tokens=NULLABLE_NAT,
    output_tokens=NULLABLE_NAT,
    total_tokens=NULLABLE_NAT,
    reasoning_tokens=NULLABLE_NAT,
    usage_source=enum("provider_reported", "estimated", "accounting_missing"),
    cache_usage=CACHE_USAGE,
)
BUDGET_DIMENSION = enum(
    "total_tokens",
    "uncached_input_tokens",
    "tool_calls",
    "tool_calls_by_name",
    "wall_time",
    "host_cost",
)
BUDGET_REASON = enum(
    "accounting_missing",
    "usage_missing",
    "meter_missing",
    "meter_unavailable",
    "meter_error",
    "unit_mismatch",
    "currency_mismatch",
    "non_monotonic",
    "integer_overflow",
)
BUDGET_COUNTS = closed(
    cycles=NAT,
    tool_calls=NAT,
    elapsed_ms=NAT,
    total_tokens=NULLABLE_NAT,
    uncached_input_tokens=NULLABLE_NAT,
    tool_calls_by_name={"type": "object", "additionalProperties": NAT},
    host_cost=nullable(closed(amount_microunits=NAT, unit=CONTENT_REF, currency=nullable(CONTENT_REF))),
    unavailable_dimensions=array(closed(dimension=BUDGET_DIMENSION, reason=BUDGET_REASON)),
)
USAGE_COUNTS = closed(tokens=nullable(TOKEN_COUNTS), budget=nullable(BUDGET_COUNTS))
REQUIREMENTS = closed(level=POS, handler_version=NULLABLE_TEXT, capabilities=STRINGS)
ENVIRONMENT_BINDING = nullable(
    closed(
        descriptor=CONTENT_REF,
        environment_capabilities=STRINGS,
        capabilities_digest=NULLABLE_HASH,
        network_policy_ref_digest=NULLABLE_HASH,
        isolation_policy_ref_digest=NULLABLE_HASH,
        operational_policy=nullable(CONTENT_REF),
    )
)
EXECUTION_ENVIRONMENT_BINDING = closed(
    descriptor=CONTENT_REF,
    environment_id=TEXT,
    generation=NAT,
    isolation=enum("none", "container", "microvm"),
    network=enum("open", "allowlist", "none"),
    enforces_network_policy=BOOL,
    enforces_resource_limits=BOOL,
    persistent=BOOL,
    environment_capabilities=STRINGS,
    capabilities_digest=NULLABLE_HASH,
    network_policy_ref_digest=NULLABLE_HASH,
    isolation_policy_ref_digest=NULLABLE_HASH,
    operational_policy=CONTENT_REF,
)
EXECUTION_IDENTITY = closed(
    operation_key=TEXT,
    tenant_id=TEXT,
    environment_id=TEXT,
    session_id=TEXT,
    operation_id=TEXT,
    execution_id=HASH,
)
RESULT_ID = closed(operation_id=TEXT, attempt=POS)
DELIVERY_TARGET = closed(session_id=TEXT, turn_id=TEXT, generation=NAT, operation_id=TEXT, attempt=POS)
CHILD_IDENTITY = closed(session_id=TEXT, turn_id=TEXT, generation=NAT)

PROVIDER_HANDLE = closed(
    kind={"const": "provider"},
    provider=TEXT,
    job_id=CONTENT_REF,
    operation_id=TEXT,
    attempt=POS,
    request_digest=HASH,
    evidence=CONTENT_REF,
    query_ref=nullable(CONTENT_REF),
    cancel_ref=nullable(CONTENT_REF),
)
PROVIDER_LOOKUP_BINDING = closed(
    tenant_id=TEXT,
    session_id=TEXT,
    operation_id=TEXT,
    attempt=POS,
    request_digest=HASH,
    idempotency_key=TEXT,
    execution_id=HASH,
    command_digest=HASH,
    environment_id=TEXT,
    generation=NAT,
    provider_id=TEXT,
)
PROVIDER_LOOKUP_HANDLE = closed(
    kind={"const": "provider"},
    provider=TEXT,
    job_id=TEXT,
    operation_id=TEXT,
    attempt=POS,
    request_digest=HASH,
    execution_id=HASH,
    command_digest=HASH,
    environment_id=TEXT,
    generation=NAT,
    evidence=STRINGS,
    query_ref=NULLABLE_TEXT,
    cancel_ref=NULLABLE_TEXT,
)
PROVIDER_LOOKUP_OUTCOME = nullable(
    {
        "oneOf": [
            closed(kind={"const": "Accepted"}, handle=PROVIDER_LOOKUP_HANDLE, yield_on_steer=BOOL),
            closed(
                kind={"const": "Definitive"},
                result=ANY_OBJECT,
                usage=ANY_OBJECT,
                shared_state=nullable(ANY_OBJECT),
                evidence=STRINGS,
            ),
            closed(kind={"const": "Unknown"}, reason=TEXT),
        ]
    }
)
PROVIDER_LOOKUP_RESULT = closed(
    schema_version={"const": "vv-agent.provider-lookup.v1"},
    status=enum("not_started", "running", "exited", "unknown"),
    binding=PROVIDER_LOOKUP_BINDING,
    outcome=PROVIDER_LOOKUP_OUTCOME,
    evidence=STRINGS,
)
APPROVAL_HANDLE = closed(kind={"const": "approval"}, request_id=TEXT, request_digest=HASH, scope=CONTENT_REF)
USER_HANDLE = closed(kind={"const": "user"}, interaction_id=TEXT, question=CONTENT_REF)
CHILD_HANDLE = closed(
    kind={"const": "child"},
    session_id=TEXT,
    turn_id=TEXT,
    generation=NAT,
    background=BOOL,
    delivery_target=DELIVERY_TARGET,
    siblings=array(CHILD_IDENTITY),
)
CHILD_HANDLE["required"].remove("siblings")
HANDLE = {"oneOf": [PROVIDER_HANDLE, APPROVAL_HANDLE, USER_HANDLE, CHILD_HANDLE]}
DELEGATION = closed(
    mode=enum("configured", "agent_as_tool", "background_task", "handoff"),
    agent_name=CONTENT_REF,
    handoff_count=NAT,
    max_handoffs=NAT,
    metadata=CONTENT_REF,
)


# Decoded C-content validators. These are used by content hydration adapters and
# internal framework values, never as an alternate logical envelope.
CHILD_ADMISSION = closed(
    mode=enum("configured", "agent_as_tool", "background_task", "handoff"),
    selector=TEXT,
    definition=ANY_OBJECT,
    definition_digest=HASH,
    budget=ANY_OBJECT,
    handler_version=TEXT,
    sub_config=nullable(
        closed(
            model=TEXT,
            description=STRING,
            backend=nullable(STRING),
            system_prompt=nullable(STRING),
            max_cycles=NAT,
            session_memory_enabled=BOOL,
            exclude_tools=array(STRING),
            metadata=ANY_OBJECT,
            denied_side_effects=STRINGS,
            denied_capability_tags=STRINGS,
            deny_terminal_tools=BOOL,
            denied_cost_dimensions=STRINGS,
        )
    ),
    exclude_files_pattern=nullable(TEXT),
    handoff_count=NAT,
    max_handoffs=NAT,
    handoff_metadata=ANY_OBJECT,
)

SEED_POLICIES = ("unseen_input_protection", "response_rejection")
SEED = {
    "oneOf": [
        closed(**bootstrap, **{name: BOOL for index, name in enumerate(SEED_POLICIES) if mask & (1 << index)})
        for bootstrap in ({}, {"messages": array(ANY_OBJECT), "shared_state": ANY_OBJECT})
        for mask in range(1 << len(SEED_POLICIES))
        if bootstrap or mask
    ]
}

TASK_SESSION_METADATA = closed(
    host_binding_names=STRINGS,
    max_handoffs=NAT,
    handoff_targets=ANY_OBJECT,
    input_messages=array(ANY_OBJECT),
    memory_initial_state=ANY_OBJECT,
    input_blocked=TEXT,
)
TASK_SESSION_METADATA["required"] = []
REQUEST_SESSION_METADATA = closed(endpoint_order=STRINGS, endpoint_id=TEXT, shared_state=ANY_OBJECT, cycle_index=POS)
REQUEST_SESSION_METADATA["required"] = []

MODEL_CAPABILITY = closed(
    backend=TEXT,
    model=TEXT,
    context_length=nullable(POS),
    max_output_tokens=NULLABLE_NAT,
)
MODEL_BINDING = closed(
    backend=TEXT,
    model=TEXT,
    endpoints=STRINGS,
    context_length=nullable(POS),
    max_output_tokens=NULLABLE_NAT,
)
DEFINITION_MODEL_BINDING = closed(
    backend=TEXT,
    model=TEXT,
    context_length=nullable(POS),
    max_output_tokens=NULLABLE_NAT,
    internal={"type": "object", "additionalProperties": MODEL_CAPABILITY},
)
DEFINITION_MODEL_BINDING["required"].remove("internal")
MODEL_SETTINGS = closed(
    temperature=nullable({"type": "number", "minimum": 0}),
    top_p=nullable({"type": "number", "minimum": 0, "maximum": 1}),
    max_tokens=nullable(POS),
    tool_choice=nullable(
        {"oneOf": [enum("auto", "required", "none"), closed(type={"const": "function"}, function=closed(name=TEXT))]}
    ),
    parallel_tool_calls=nullable(BOOL),
    reasoning=nullable(ANY_OBJECT),
    response_format=nullable(
        {"oneOf": [closed(type=enum("text", "json_object")), closed(type={"const": "json_schema"}, json_schema=ANY_OBJECT)]}
    ),
    timeout_seconds=nullable({"type": "number", "exclusiveMinimum": 0}),
    retry=nullable(closed(max_attempts=POS, backoff_seconds={"type": "number", "minimum": 0})),
    extra_headers={"type": "object", "additionalProperties": STRING},
    extra_body=ANY_OBJECT,
    extra_args=ANY_OBJECT,
)
MODEL_SETTINGS["required"] = []
MODEL_SETTINGS["properties"]["retry"]["anyOf"][0]["required"] = []
MODEL_SELECTION = closed(model=TEXT, model_settings=MODEL_SETTINGS, route=ANY_OBJECT)
MODEL_SELECTION["required"] = ["model", "route"]
REQUEST_SESSION_METADATA["properties"]["model_binding"] = MODEL_BINDING
APP_SERVER_ATTRIBUTES = closed(agent_key=STRING, cwd=nullable(STRING), metadata=ANY_OBJECT)

BOUNDARY_DATA = {
    "before_memory": closed(messages=array(CONTENT_REF), shared_state=CONTENT_REF),
    "after_cycle": closed(
        action=enum("continue", "steer", "stop_non_success"),
        steering_messages=array(CONTENT_REF),
        disallow_tools=STRINGS,
        stop=nullable(closed(code=enum(*AFTER_CYCLE_KERNEL_STOP_CODES), message=CONTENT_REF)),
        shared_state=CONTENT_REF,
        error=nullable(CONTENT_REF),
    ),
    "memory_started": closed(event=CONTENT_REF),
    "memory_completed": closed(event=CONTENT_REF),
    "session_memory_saved": closed(
        state=closed(
            last_extracted_message_index=nullable(POS),
            tokens_at_last_extraction=NAT,
            initialized=BOOL,
            entries=array(closed(category=TEXT, source_cycle=NAT, importance=NAT, content=CONTENT_REF)),
        )
    ),
    "output_checked": closed(
        status=enum("completed", "repair", "failed"),
        reason=nullable(CONTENT_REF),
        value=CONTENT_REF,
        partial_output=nullable(CONTENT_REF),
    ),
    "budget": closed(tool_names=STRINGS, usage=CONTENT_REF, exhaustion=nullable(CONTENT_REF)),
}
BOUNDARY_DATA["output_checked"]["required"].remove("partial_output")

INPUT_PAYLOADS = {
    "user": closed(content=CONTENT_REF),
    "steer": closed(content=CONTENT_REF),
    "follow_up": closed(content=CONTENT_REF),
    "provider_result": closed(
        operation_id=TEXT,
        attempt=POS,
        request_digest=HASH,
        provider_binding=NULLABLE_TEXT,
        result=nullable(CONTENT_REF),
        usage=CONTENT_REF,
        evidence=CONTENT_REF,
    ),
    "approval_answer": closed(
        operation_id=TEXT,
        attempt=POS,
        request_id=TEXT,
        request_digest=HASH,
        decision=enum("allow_session", "approve", "deny", "timeout", "approve_once", "approve_for_session"),
        scope=CONTENT_REF,
        reason=nullable(CONTENT_REF),
        metadata=nullable(CONTENT_REF),
        policy_digest=NULLABLE_HASH,
        rule_id=NULLABLE_TEXT,
        expires_ms=NULLABLE_NAT,
    ),
    "child_result": closed(
        session_id=TEXT,
        turn_id=TEXT,
        operation_id=TEXT,
        attempt=POS,
        status=enum("running", "completed", "failed", "cancelled", "aborted"),
        result=nullable(CONTENT_REF),
        terminal_seq=nullable(POS),
        terminal_digest=NULLABLE_HASH,
    ),
    "control": closed(action=CONTROL),
    "provider_evidence": closed(
        operation_id=TEXT,
        attempt=POS,
        request_digest=HASH,
        handle=PROVIDER_HANDLE,
        yield_on_steer={"const": True, "type": "boolean"},
    ),
    "model_selection": closed(selection=nullable(CONTENT_REF)),
}
for name in ("reason", "metadata", "policy_digest", "rule_id", "expires_ms"):
    INPUT_PAYLOADS["approval_answer"]["required"].remove(name)
INPUT_PAYLOADS["provider_evidence"]["required"].remove("yield_on_steer")
PROVIDER_EVIDENCE_LEVEL_3 = closed(
    operation_id=TEXT,
    attempt=POS,
    request_digest=HASH,
    provider_lookup=CONTENT_REF,
)

INPUT_SCHEMA = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 1, "type": "integer"},
    actor=PRINCIPAL,
    admission_proof=ADMISSION_PROOF,
    input_id=TEXT,
    kind=enum(*INPUT_PAYLOADS),
    target_turn_id=NULLABLE_TEXT,
    generation=nullable(NAT),
    available_ms=NAT,
    payload=ANY_OBJECT,
)

INPUT_SCHEMA_LEVEL_2 = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 2, "type": "integer"},
    actor=PRINCIPAL,
    admission_proof=ADMISSION_PROOF,
    input_id=TEXT,
    kind=enum(*INPUT_PAYLOADS),
    target_turn_id=NULLABLE_TEXT,
    generation=nullable(NAT),
    available_ms=NAT,
    requirements=REQUIREMENTS,
    payload=ANY_OBJECT,
)
INPUT_SCHEMA_LEVEL_3 = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 3, "type": "integer"},
    actor=PRINCIPAL,
    admission_proof=ADMISSION_PROOF,
    input_id=TEXT,
    kind=enum("provider_evidence"),
    target_turn_id=NULLABLE_TEXT,
    generation=nullable(NAT),
    available_ms=NAT,
    payload=ANY_OBJECT,
)
INBOX_ITEM_SCHEMA = {"oneOf": [INPUT_SCHEMA, INPUT_SCHEMA_LEVEL_2, INPUT_SCHEMA_LEVEL_3]}

PAYLOADS = {
    "session_created": closed(
        principal=PRINCIPAL,
        workspace=CONTENT_REF,
        parent_session_id=NULLABLE_TEXT,
        parent_operation_id=NULLABLE_TEXT,
        attributes=CONTENT_REF,
    ),
    "turn_started": closed(
        input_ids=STRINGS,
        definition=CONTENT_REF,
        definition_digest=HASH,
        handler_version=TEXT,
        budget=CONTENT_REF,
        binding=NULLABLE_TEXT,
        generation=NAT,
        policy_digest=NULLABLE_HASH,
        requirements=REQUIREMENTS,
        tokenizer_binding=TOKENIZER_BINDING,
        environment=ENVIRONMENT_BINDING,
    ),
    "input_applied": closed(
        input=INBOX_ITEM_SCHEMA,
        input_digest=HASH,
        input_proof=INPUT_PROOF,
        disposition=enum("applied", "rejected", "noop", "queued"),
        reason=nullable(CONTENT_REF),
        target_operation_id=NULLABLE_TEXT,
        target_wait_id=NULLABLE_TEXT,
        position=TEXT,
        model_binding=CONTENT_REF,
    ),
    "op_planned": closed(
        op_kind=enum("model", "tool", "interaction"),
        purpose=nullable(enum("primary", "compaction", "session_memory", "output_repair")),
        request=CONTENT_REF,
        request_digest=HASH,
        context_version=TEXT,
        dependencies=STRINGS,
        tool=nullable(CONTENT_REF),
        idempotency_key=NULLABLE_TEXT,
        budget_admission=CONTENT_REF,
        not_before_ms=NULLABLE_NAT,
        provider_binding=NULLABLE_TEXT,
        consumed_unknowns=array(RESULT_ID),
        policy_decision=POLICY_DECISION,
    ),
    "op_prepared": closed(
        request=CONTENT_REF,
        request_digest=HASH,
        op_kind=enum("tool", "interaction"),
        tool=CONTENT_REF,
        provider_binding=NULLABLE_TEXT,
        idempotency_key=NULLABLE_TEXT,
        hook_result=nullable(CONTENT_REF),
        shared_state=CONTENT_REF,
        policy_decision=POLICY_DECISION,
    ),
    "turn_parked": closed(interaction_id=TEXT, question=CONTENT_REF, source_operation_id=TEXT, source_attempt=POS),
    "op_started": closed(
        dispatch_id=TEXT,
        authorization_version=TEXT,
        epoch=POS,
        mode=enum("sync", "provider", "managed"),
        endpoint_id=TEXT,
    ),
    "op_parked": closed(
        phase=enum("before_dispatch", "after_dispatch"),
        handle=HANDLE,
        poll_at_ms=NULLABLE_NAT,
        deadline_ms=NULLABLE_NAT,
        interaction_result=nullable(CONTENT_REF),
        delegation=DELEGATION,
        yield_on_steer={"const": True, "type": "boolean"},
        yieldable_children=STRINGS,
    ),
    "op_completed": closed(
        result=CONTENT_REF,
        result_digest=HASH,
        usage=CONTENT_REF,
        usage_counts=USAGE_COUNTS,
        shared_state=nullable(CONTENT_REF),
        evidence=CONTENT_REF,
        execution_started=BOOL,
        context=enum("normal", "correction", "audit"),
        request_digest=HASH,
        provider_binding=NULLABLE_TEXT,
        policy_decision=POLICY_DECISION,
        yielded_by=TEXT,
    ),
    "op_unknown": closed(
        reason=CONTENT_REF,
        dispatch_evidence=CONTENT_REF,
        observation=CONTENT_REF,
        retry=enum("retry", "stop", "manual"),
        retry_at_ms=NULLABLE_NAT,
        duplicate_cost_risk=BOOL,
        measurement_missing=BOOL,
    ),
    "context_compacted": closed(
        source_digest=HASH,
        prefix_ids=STRINGS,
        tail_ids=STRINGS,
        mode=enum("micro", "summary", "emergency"),
        summary_operation_id=NULLABLE_TEXT,
        replacement=array(CONTENT_REF),
        evidence_manifest=CONTENT_REF,
        micro_usage=closed(archived_count=NAT, reclaimed_tokens=NAT, artifact_failure_count=NAT),
    ),
    "boundary_recorded": closed(
        boundary_id=TEXT,
        stage=enum(*BOUNDARY_DATA),
        source_operation_id=NULLABLE_TEXT,
        source_digest=NULLABLE_HASH,
        data=ANY_OBJECT,
    ),
    "usage_observed": closed(
        meter_id=TEXT,
        observation=POS,
        mode=enum("cumulative", "correction"),
        usage=CONTENT_REF,
        source=CONTENT_REF,
    ),
    "turn_ended": closed(
        status=enum("completed", "failed", "cancelled", "aborted"),
        reason=nullable(CONTENT_REF),
        result=nullable(CONTENT_REF),
        adopted_results=array(RESULT_ID),
        budget=CONTENT_REF,
        unconfirmed_operations=STRINGS,
    ),
}

SESSION_CREATED_LEVEL_2 = closed(
    principal=PRINCIPAL,
    workspace=CONTENT_REF,
    parent_session_id=NULLABLE_TEXT,
    parent_operation_id=NULLABLE_TEXT,
    attributes=CONTENT_REF,
    bootstrap=REQUIREMENTS,
)

PAYLOADS["input_applied"]["required"].remove("model_binding")
PAYLOADS["op_started"]["required"].remove("endpoint_id")
for name in ("interaction_result", "delegation", "yield_on_steer", "yieldable_children"):
    PAYLOADS["op_parked"]["required"].remove(name)
for name in ("usage_counts", "yielded_by"):
    PAYLOADS["op_completed"]["required"].remove(name)
PAYLOADS["context_compacted"]["required"].remove("micro_usage")

TURN_STARTED_LEVEL_3 = closed(
    input_ids=STRINGS,
    definition=CONTENT_REF,
    definition_digest=HASH,
    handler_version=TEXT,
    budget=CONTENT_REF,
    binding=NULLABLE_TEXT,
    generation=NAT,
    policy_digest=NULLABLE_HASH,
    requirements=REQUIREMENTS,
    tokenizer_binding=TOKENIZER_BINDING,
    environment=EXECUTION_ENVIRONMENT_BINDING,
)
OP_PLANNED_LEVEL_3 = closed(
    op_kind=enum("tool"),
    purpose={"type": "null"},
    request=CONTENT_REF,
    request_digest=HASH,
    context_version=TEXT,
    dependencies=STRINGS,
    tool=nullable(CONTENT_REF),
    idempotency_key=NULLABLE_TEXT,
    budget_admission=CONTENT_REF,
    not_before_ms=NULLABLE_NAT,
    provider_binding=NULLABLE_TEXT,
    consumed_unknowns=array(RESULT_ID),
    policy_decision=POLICY_DECISION,
    execution_identity=EXECUTION_IDENTITY,
    command_digest=HASH,
)
OP_PREPARED_LEVEL_3 = closed(
    request=CONTENT_REF,
    request_digest=HASH,
    op_kind={"const": "tool"},
    tool=CONTENT_REF,
    provider_binding=TEXT,
    idempotency_key=TEXT,
    hook_result=nullable(CONTENT_REF),
    shared_state=CONTENT_REF,
    policy_decision=POLICY_DECISION,
    command_digest=HASH,
)

RECORD_SCHEMA = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 1, "type": "integer"},
    record_id=TEXT,
    kind=enum(*PAYLOADS),
    session_id=TEXT,
    turn_id=NULLABLE_TEXT,
    operation_id=NULLABLE_TEXT,
    attempt=nullable(POS),
    payload=ANY_OBJECT,
)

RECORD_SCHEMA_LEVEL_2 = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 2, "type": "integer"},
    record_id=TEXT,
    kind=enum("session_created"),
    session_id=TEXT,
    turn_id=NULLABLE_TEXT,
    operation_id=NULLABLE_TEXT,
    attempt=nullable(POS),
    payload=ANY_OBJECT,
)
RECORD_SCHEMA_LEVEL_3 = closed(
    schema_version={"const": "S1"},
    feature_level={"const": 3, "type": "integer"},
    record_id=TEXT,
    kind=enum("turn_started", "op_planned", "op_prepared"),
    session_id=TEXT,
    turn_id=NULLABLE_TEXT,
    operation_id=NULLABLE_TEXT,
    attempt=nullable(POS),
    payload=ANY_OBJECT,
)
RECORD_ITEM_SCHEMA = {"oneOf": [RECORD_SCHEMA, RECORD_SCHEMA_LEVEL_2, RECORD_SCHEMA_LEVEL_3]}

SCHEMAS = (
    RECORD_SCHEMA,
    RECORD_SCHEMA_LEVEL_2,
    RECORD_SCHEMA_LEVEL_3,
    RECORD_ITEM_SCHEMA,
    INPUT_SCHEMA,
    INPUT_SCHEMA_LEVEL_2,
    INPUT_SCHEMA_LEVEL_3,
    INBOX_ITEM_SCHEMA,
    CONTENT_LABELS,
    CONTENT_REF,
    PRINCIPAL,
    ADMISSION_PROOF,
    INPUT_PROOF,
    POLICY_DECISION,
    TOKENIZER_BINDING,
    CACHE_USAGE,
    TOKEN_COUNTS,
    BUDGET_COUNTS,
    USAGE_COUNTS,
    REQUIREMENTS,
    SESSION_CREATED_LEVEL_2,
    ENVIRONMENT_BINDING,
    EXECUTION_ENVIRONMENT_BINDING,
    EXECUTION_IDENTITY,
    PROVIDER_LOOKUP_BINDING,
    PROVIDER_LOOKUP_HANDLE,
    PROVIDER_LOOKUP_OUTCOME,
    PROVIDER_LOOKUP_RESULT,
    PROVIDER_EVIDENCE_LEVEL_3,
    TURN_STARTED_LEVEL_3,
    OP_PLANNED_LEVEL_3,
    OP_PREPARED_LEVEL_3,
    RESULT_ID,
    CHILD_ADMISSION,
    APP_SERVER_ATTRIBUTES,
    SEED,
    TASK_SESSION_METADATA,
    REQUEST_SESSION_METADATA,
    MODEL_CAPABILITY,
    MODEL_BINDING,
    DEFINITION_MODEL_BINDING,
    MODEL_SELECTION,
    HANDLE,
    *PAYLOADS.values(),
    *INPUT_PAYLOADS.values(),
    *BOUNDARY_DATA.values(),
)

CHECKS = {id(schema): _compile_check(schema) for schema in SCHEMAS}

# Private aliases preserve the existing import surface without defining another
# wire format.
_CHECKS = CHECKS
_SCHEMAS = SCHEMAS
