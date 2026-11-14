"""Authoritative closed record, inbox, handle and boundary schemas and checks."""

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
JSON_OBJECT = {"type": "object"}  # Explicit opaque host/provider data, never kernel control fields.
JSON_VALUE: dict[str, Any] = {}
NAT = {"type": "integer", "minimum": 0}
POS = {"type": "integer", "minimum": 1}
BOOL = {"type": "boolean"}
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$", "maxLength": 64}
STRINGS = array(TEXT)
CONTROL = enum("close", "archive", "suspend", "resume", "cancel", "abort")
RESULT_ID = closed(operation_id=TEXT, attempt=POS)
HANDLE = {
    "oneOf": [
        closed(
            kind={"const": "provider"},
            provider=TEXT,
            job_id=TEXT,
            operation_id=TEXT,
            attempt=POS,
            request_digest=HASH,
            evidence=TEXT,
            query_ref=nullable(TEXT),
            cancel_ref=nullable(TEXT),
        ),
        closed(kind={"const": "approval"}, request_id=TEXT, request_digest=HASH, scope=STRINGS),
        closed(kind={"const": "user"}, interaction_id=TEXT, question=TEXT),
        closed(
            kind={"const": "child"},
            session_id=TEXT,
            turn_id=TEXT,
            generation=NAT,
            background=BOOL,
            delivery_target=closed(session_id=TEXT, turn_id=TEXT, generation=NAT, operation_id=TEXT, attempt=POS),
        ),
    ]
}
HANDLE["oneOf"][-1]["properties"]["siblings"] = array(closed(session_id=TEXT, turn_id=TEXT, generation=NAT))

DELEGATION = closed(
    mode=enum("configured", "agent_as_tool", "background_task", "handoff"),
    agent_name=TEXT,
    handoff_count=NAT,
    max_handoffs=NAT,
    metadata=JSON_OBJECT,
)

CHILD_ADMISSION = closed(
    mode=enum("configured", "agent_as_tool", "background_task", "handoff"),
    selector=TEXT,
    definition=JSON_OBJECT,
    definition_digest=HASH,
    budget=JSON_OBJECT,
    handler_version=TEXT,
    sub_config=nullable(
        closed(
            model=TEXT,
            description={"type": "string"},
            backend=nullable({"type": "string"}),
            system_prompt=nullable({"type": "string"}),
            max_cycles=NAT,
            session_memory_enabled=BOOL,
            exclude_tools=array({"type": "string"}),
            metadata=JSON_OBJECT,
            denied_side_effects=STRINGS,
            denied_capability_tags=STRINGS,
            deny_terminal_tools=BOOL,
            denied_cost_dimensions=STRINGS,
        )
    ),
    exclude_files_pattern=nullable(TEXT),
    handoff_count=NAT,
    max_handoffs=NAT,
    handoff_metadata=JSON_OBJECT,
)

SEED_POLICIES = ("unseen_input_protection", "response_rejection")
SEED = {
    "oneOf": [
        closed(**bootstrap, **{name: BOOL for index, name in enumerate(SEED_POLICIES) if mask & (1 << index)})
        for bootstrap in ({}, {"messages": array(JSON_OBJECT), "shared_state": JSON_OBJECT})
        for mask in range(1 << len(SEED_POLICIES))
        if bootstrap or mask
    ]
}


TASK_SESSION_METADATA = closed(
    host_binding_names=STRINGS,
    max_handoffs=NAT,
    handoff_targets=JSON_OBJECT,
    input_messages=array(JSON_OBJECT),
    memory_initial_state=JSON_OBJECT,
    input_blocked=TEXT,
)
TASK_SESSION_METADATA["required"] = []
REQUEST_SESSION_METADATA = closed(endpoint_order=STRINGS, endpoint_id=TEXT, shared_state=JSON_OBJECT, cycle_index=POS)
REQUEST_SESSION_METADATA["required"] = []

MODEL_BINDING = closed(backend=TEXT, model=TEXT, endpoints=STRINGS, context_length=nullable(POS), max_output_tokens=nullable(NAT))
MODEL_SETTINGS = closed(
    temperature=nullable({"type": "number", "minimum": 0}),
    top_p=nullable({"type": "number", "minimum": 0, "maximum": 1}),
    max_tokens=nullable(POS),
    tool_choice=nullable(
        {"oneOf": [enum("auto", "required", "none"), closed(type={"const": "function"}, function=closed(name=TEXT))]}
    ),
    parallel_tool_calls=nullable(BOOL),
    reasoning=nullable(JSON_OBJECT),
    response_format=nullable(
        {"oneOf": [closed(type=enum("text", "json_object")), closed(type={"const": "json_schema"}, json_schema=JSON_OBJECT)]}
    ),
    timeout_seconds=nullable({"type": "number", "exclusiveMinimum": 0}),
    retry=nullable(closed(max_attempts=POS, backoff_seconds={"type": "number", "minimum": 0})),
    extra_headers={"type": "object", "additionalProperties": {"type": "string"}},
    extra_body=JSON_OBJECT,
    extra_args=JSON_OBJECT,
)
MODEL_SETTINGS["required"] = []
MODEL_SETTINGS["properties"]["retry"]["anyOf"][0]["required"] = []
MODEL_SELECTION = closed(model=TEXT, model_settings=MODEL_SETTINGS, route=JSON_OBJECT)
MODEL_SELECTION["required"] = ["model", "route"]
REQUEST_SESSION_METADATA["properties"]["model_binding"] = MODEL_BINDING

APP_SERVER_ATTRIBUTES = closed(agent_key={"type": "string"}, cwd=nullable({"type": "string"}), metadata=JSON_OBJECT)


BOUNDARY_DATA = {
    "before_memory": closed(messages=array(JSON_OBJECT), shared_state=JSON_OBJECT),
    "after_cycle": closed(
        action=enum("continue", "steer", "stop_non_success"),
        steering_messages=STRINGS,
        disallow_tools=STRINGS,
        stop=nullable(closed(code=TEXT, message=TEXT)),
        shared_state=JSON_OBJECT,
        error=nullable(TEXT),
    ),
    "memory_started": closed(event=JSON_OBJECT),
    "memory_completed": closed(event=JSON_OBJECT),
    "session_memory_saved": closed(
        state=closed(
            entries=array(
                closed(
                    category=enum("user_intent", "decision", "file_change", "error_fix", "key_fact"),
                    content=TEXT,
                    source_cycle=NAT,
                    importance={"type": "integer", "minimum": 1, "maximum": 10},
                )
            ),
            last_extracted_message_index={"type": "integer", "minimum": -1},
            tokens_at_last_extraction=NAT,
            initialized=BOOL,
        )
    ),
    "output_checked": closed(status=enum("completed", "failed", "repair"), reason=nullable(TEXT), value=JSON_VALUE),
    "budget": closed(usage=JSON_OBJECT, exhaustion=nullable(JSON_OBJECT), tool_names=STRINGS),
}
INPUT_PAYLOADS = {
    "model_selection": closed(selection=nullable(MODEL_SELECTION)),
    "user": closed(content=JSON_VALUE),
    "steer": closed(content=JSON_VALUE),
    "follow_up": closed(content=JSON_VALUE),
    "provider_result": closed(
        operation_id=TEXT,
        attempt=POS,
        request_digest=HASH,
        provider_binding=nullable(TEXT),
        result=JSON_VALUE,
        usage=JSON_OBJECT,
        evidence=STRINGS,
    ),
    "approval_answer": closed(
        operation_id=TEXT,
        attempt=POS,
        request_id=TEXT,
        request_digest=HASH,
        decision=enum("approve", "deny", "allow_session", "timeout"),
        scope=STRINGS,
    ),
    "child_result": closed(
        session_id=TEXT,
        turn_id=TEXT,
        operation_id=TEXT,
        attempt=POS,
        result=JSON_VALUE,
        status=enum("completed", "failed", "cancelled", "aborted"),
        terminal_seq=POS,
        terminal_digest=HASH,
    ),
    "control": closed(action=CONTROL),
    "provider_evidence": closed(operation_id=TEXT, attempt=POS, request_digest=HASH, handle=HANDLE),
}
# Optional typed decision details remain inside the closed approval answer.
INPUT_PAYLOADS["approval_answer"]["properties"].update(reason={"type": "string"}, metadata=JSON_OBJECT)
INPUT_PAYLOADS["provider_evidence"]["properties"]["yield_on_steer"] = {"const": True, "type": "boolean"}

INPUT_SCHEMA = closed(
    schema_version={"type": "integer", "const": 3},
    input_id=TEXT,
    kind=enum(*INPUT_PAYLOADS),
    target_turn_id=nullable(TEXT),
    generation=nullable(NAT),
    available_ms=NAT,
    payload=JSON_OBJECT,
)
PAYLOADS = {
    "session_created": closed(
        principal=TEXT,
        workspace=TEXT,
        parent_session_id=nullable(TEXT),
        parent_operation_id=nullable(TEXT),
        attributes=JSON_OBJECT,
    ),
    "turn_started": closed(
        input_ids=STRINGS,
        definition=JSON_OBJECT,
        definition_digest=HASH,
        handler_version=TEXT,
        budget=JSON_OBJECT,
        binding=nullable(TEXT),
        generation=NAT,
    ),
    "input_applied": closed(
        input=INPUT_SCHEMA,
        input_digest=HASH,
        disposition=enum("applied", "rejected", "noop", "queued"),
        reason=nullable(TEXT),
        target_operation_id=nullable(TEXT),
        target_wait_id=nullable(TEXT),
        position=TEXT,
    ),
    "op_planned": closed(
        op_kind=enum("model", "tool", "interaction"),
        purpose=nullable(enum("primary", "compaction", "session_memory", "output_repair")),
        request=JSON_OBJECT,
        request_digest=HASH,
        context_version=TEXT,
        dependencies=STRINGS,
        tool=nullable(JSON_OBJECT),
        idempotency_key=nullable(TEXT),
        budget_admission=JSON_OBJECT,
        not_before_ms=nullable(NAT),
        provider_binding=nullable(TEXT),
        consumed_unknowns=array(RESULT_ID),
    ),
    "op_prepared": closed(
        request=JSON_OBJECT,
        request_digest=HASH,
        op_kind=enum("tool", "interaction"),
        tool=JSON_OBJECT,
        provider_binding=nullable(TEXT),
        idempotency_key=nullable(TEXT),
        hook_result=JSON_VALUE,
        shared_state=JSON_OBJECT,
    ),
    "turn_parked": closed(interaction_id=TEXT, question={"type": "string"}, source_operation_id=TEXT, source_attempt=POS),
    "op_started": closed(dispatch_id=TEXT, authorization_version=TEXT, epoch=POS, mode=enum("sync", "provider", "managed")),
    "op_parked": closed(
        phase=enum("before_dispatch", "after_dispatch"), handle=HANDLE, poll_at_ms=nullable(NAT), deadline_ms=nullable(NAT)
    ),
    "op_completed": closed(
        result=JSON_VALUE,
        result_digest=HASH,
        usage=JSON_OBJECT,
        shared_state=nullable(JSON_OBJECT),
        evidence=STRINGS,
        execution_started=BOOL,
        context=enum("normal", "correction", "audit"),
        request_digest=HASH,
        provider_binding=nullable(TEXT),
    ),
    "op_unknown": closed(
        reason=TEXT,
        dispatch_evidence=STRINGS,
        observation=JSON_OBJECT,
        retry=enum("retry", "stop", "manual"),
        retry_at_ms=nullable(NAT),
        duplicate_cost_risk=BOOL,
        measurement_missing=BOOL,
    ),
    "context_compacted": closed(
        source_digest=HASH,
        prefix_ids=STRINGS,
        tail_ids=STRINGS,
        mode=enum("micro", "summary", "emergency"),
        summary_operation_id=nullable(TEXT),
        replacement=array(JSON_VALUE),
        evidence_manifest=JSON_OBJECT,
    ),
    "boundary_recorded": closed(
        boundary_id=TEXT,
        stage=enum(*BOUNDARY_DATA),
        source_operation_id=nullable(TEXT),
        source_digest=nullable(HASH),
        data=JSON_OBJECT,
    ),
    "usage_observed": closed(
        meter_id=TEXT, observation=POS, mode=enum("cumulative", "correction"), usage=JSON_OBJECT, source=TEXT
    ),
    "turn_ended": closed(
        status=enum("completed", "failed", "cancelled", "aborted"),
        reason=nullable(TEXT),
        result=JSON_VALUE,
        adopted_results=array(RESULT_ID),
        budget=JSON_OBJECT,
        unconfirmed_operations=STRINGS,
    ),
}
BOUNDARY_DATA["output_checked"]["properties"]["partial_output"] = JSON_VALUE
PAYLOADS["op_started"]["properties"]["endpoint_id"] = TEXT
PAYLOADS["op_parked"]["properties"]["interaction_result"] = JSON_OBJECT
PAYLOADS["op_parked"]["properties"]["yield_on_steer"] = {"const": True, "type": "boolean"}
PAYLOADS["op_parked"]["properties"]["yieldable_children"] = STRINGS
PAYLOADS["op_completed"]["properties"]["yielded_by"] = TEXT
PAYLOADS["context_compacted"]["properties"]["micro_usage"] = closed(
    archived_count=NAT,
    reclaimed_tokens=NAT,
    artifact_failure_count=NAT,
)
PAYLOADS["input_applied"]["properties"]["model_binding"] = MODEL_BINDING

RECORD_SCHEMA = closed(
    schema_version={"type": "integer", "const": 1},
    record_id=TEXT,
    kind=enum(*PAYLOADS),
    session_id=TEXT,
    turn_id=nullable(TEXT),
    operation_id=nullable(TEXT),
    attempt=nullable(POS),
    payload=JSON_OBJECT,
)
PAYLOADS["op_parked"]["properties"]["delegation"] = DELEGATION

_SCHEMAS = (
    RECORD_SCHEMA,
    INPUT_SCHEMA,
    CHILD_ADMISSION,
    APP_SERVER_ATTRIBUTES,
    SEED,
    TASK_SESSION_METADATA,
    REQUEST_SESSION_METADATA,
    MODEL_BINDING,
    HANDLE,
    *PAYLOADS.values(),
    *INPUT_PAYLOADS.values(),
    *BOUNDARY_DATA.values(),
)

_CHECKS = {id(schema): _compile_check(schema) for schema in _SCHEMAS}
