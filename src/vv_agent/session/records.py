"""Experimental v1 closed records. Store envelopes never participate in identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.validators import extend

from vv_agent.canonical_json import canonical_json_bytes


class RecordError(ValueError):
    pass


def digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


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
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
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
INPUT_PAYLOADS = {
    "user": closed(content=JSON_VALUE),
    "steer": closed(content=JSON_VALUE),
    "follow_up": closed(content=JSON_VALUE),
    "deferred_result": closed(
        operation_id=TEXT, attempt=POS, request_digest=HASH, provider_binding=nullable(TEXT), result=JSON_VALUE, evidence=STRINGS
    ),
    "approval_answer": closed(
        operation_id=TEXT, attempt=POS, request_id=TEXT, request_digest=HASH, decision=enum("approve", "deny"), scope=STRINGS
    ),
    "child_result": closed(
        session_id=TEXT,
        turn_id=TEXT,
        operation_id=TEXT,
        attempt=POS,
        result=JSON_VALUE,
        terminal_seq=POS,
        terminal_digest=HASH,
    ),
    "control": closed(action=CONTROL),
    "provider_evidence": closed(operation_id=TEXT, attempt=POS, request_digest=HASH, handle=HANDLE),
}
INPUT_SCHEMA = closed(
    schema_version={"type": "integer", "const": 1},
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
    "op_started": closed(dispatch_id=TEXT, authorization_version=TEXT, epoch=POS, mode=enum("sync", "provider", "managed")),
    "op_parked": closed(
        phase=enum("before_dispatch", "after_dispatch"), handle=HANDLE, poll_at_ms=nullable(NAT), deadline_ms=nullable(NAT)
    ),
    "op_completed": closed(
        result=JSON_VALUE,
        result_digest=HASH,
        usage=JSON_OBJECT,
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
_StrictValidator = extend(
    Draft202012Validator, type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda _, value: type(value) is int)
)
_VALIDATORS = {id(s): _StrictValidator(s) for s in [RECORD_SCHEMA, INPUT_SCHEMA, *PAYLOADS.values(), *INPUT_PAYLOADS.values()]}


def validate(value: Any, schema: dict[str, Any], *, canonical: bool = True) -> None:
    try:
        if canonical:
            canonical_json_bytes(value)
        _VALIDATORS[id(schema)].validate(value)
    except (ValueError, TypeError, ValidationError) as exc:
        # jsonschema.ValidationError and the existing canonical codec have different bases.
        raise RecordError(str(exc)) from exc


def _load(body: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecordError(f"duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(body, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise RecordError(str(exc)) from exc
    if not isinstance(value, dict):
        raise RecordError("expected object")
    return value


def _check_digest(payload: dict[str, Any], field: str) -> None:
    if payload[f"{field}_digest"] != digest(payload[field]):
        raise RecordError(f"{field} digest mismatch")


@dataclass(frozen=True)
class InboxItem:
    input_id: str
    kind: str
    payload: dict[str, Any]
    target_turn_id: str | None = None
    generation: int | None = None
    available_ms: int = 0
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _validate(self, value: dict[str, Any]) -> None:
        validate(value, INPUT_SCHEMA, canonical=False)
        validate(self.payload, INPUT_PAYLOADS[self.kind], canonical=False)

    def encode(self) -> bytes:
        value = {f.name: getattr(self, f.name) for f in fields(self)}
        self._validate(value)
        try:
            return canonical_json_bytes(value)
        except (ValueError, TypeError) as exc:
            raise RecordError(str(exc)) from exc

    @property
    def digest(self) -> str:
        return sha256(self.encode()).hexdigest()

    @classmethod
    def parse(cls, body: bytes) -> InboxItem:
        value = _load(body)
        validate(value, INPUT_SCHEMA, canonical=False)
        item = cls(**value)
        validate(item.payload, INPUT_PAYLOADS[item.kind], canonical=False)
        try:
            canonical_json_bytes(value)
        except (ValueError, TypeError) as exc:
            raise RecordError(str(exc)) from exc
        return item


def record_identity(
    kind: str, session_id: str, turn_id: str | None, operation_id: str | None, attempt: int | None, payload: dict[str, Any]
) -> str:
    if kind == "session_created":
        return f"session/{session_id}/created"
    if kind in {"turn_started", "turn_ended"}:
        return f"turn/{turn_id}/{kind.removeprefix('turn_')}"
    if kind == "input_applied":
        return f"input/{payload['input']['input_id']}/applied"
    if kind.startswith("op_"):
        suffix = "result" if kind == "op_completed" else kind.removeprefix("op_")
        if kind == "op_parked":
            suffix += f"/{payload['phase']}"
        return f"op/{operation_id}/{attempt}/{suffix}"
    if kind == "context_compacted":
        return f"compact/{payload['source_digest']}/{payload['mode']}/{payload['summary_operation_id']}"
    return f"usage/{payload['meter_id']}/{payload['observation']}"


@dataclass(frozen=True)
class Record:
    record_id: str
    kind: str
    session_id: str
    turn_id: str | None
    operation_id: str | None
    attempt: int | None
    payload: dict[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _validate(self, value: dict[str, Any]) -> None:
        validate(value, RECORD_SCHEMA, canonical=False)
        validate(self.payload, PAYLOADS[self.kind], canonical=False)
        op = self.kind.startswith("op_")
        if op != (self.operation_id is not None) or op != (self.attempt is not None):
            raise RecordError("operation and attempt required only for operation records")
        if self.kind == "session_created" and self.turn_id is not None:
            raise RecordError("session_created has no turn")
        if self.kind not in {"session_created", "input_applied", "usage_observed"} and self.turn_id is None:
            raise RecordError("turn_id required")
        if self.record_id != record_identity(
            self.kind, self.session_id, self.turn_id, self.operation_id, self.attempt, self.payload
        ):
            raise RecordError("record_id does not match semantic position")
        for kind, field in (
            ("turn_started", "definition"),
            ("op_planned", "request"),
            ("op_completed", "result"),
            ("input_applied", "input"),
        ):
            if self.kind == kind:
                _check_digest(self.payload, field)
        if self.kind == "input_applied":
            InboxItem.parse(canonical_json_bytes(self.payload["input"]))
        if self.kind == "op_planned" and (self.payload["op_kind"] == "model") != (self.payload["purpose"] is not None):
            raise RecordError("purpose required only for model operations")

    def encode(self) -> bytes:
        value = {f.name: getattr(self, f.name) for f in fields(self)}
        self._validate(value)
        try:
            return canonical_json_bytes(value)
        except (ValueError, TypeError) as exc:
            raise RecordError(str(exc)) from exc

    @property
    def digest(self) -> str:
        return sha256(self.encode()).hexdigest()

    @classmethod
    def parse(cls, body: bytes) -> Record:
        value = _load(body)
        try:
            record = cls(**value)
        except TypeError as exc:
            raise RecordError(str(exc)) from exc
        record._validate(value)
        try:
            canonical_json_bytes(value)
        except (ValueError, TypeError) as exc:
            raise RecordError(str(exc)) from exc
        return record


def make_record(
    kind: str,
    *,
    session_id: str,
    payload: dict[str, Any],
    turn_id: str | None = None,
    operation_id: str | None = None,
    attempt: int | None = None,
) -> Record:
    if kind not in PAYLOADS:
        raise RecordError("unknown record kind")
    validate(payload, PAYLOADS[kind], canonical=False)
    record = Record(
        record_identity(kind, session_id, turn_id, operation_id, attempt, payload),
        kind,
        session_id,
        turn_id,
        operation_id,
        attempt,
        payload,
    )
    record.encode()
    return record


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    principal: str
    workspace: str
    parent_session_id: str | None = None
    parent_operation_id: str | None = None
    attributes: dict[str, Any] | None = None

    def record(self) -> Record:
        return make_record(
            "session_created",
            session_id=self.session_id,
            payload={
                "principal": self.principal,
                "workspace": self.workspace,
                "parent_session_id": self.parent_session_id,
                "parent_operation_id": self.parent_operation_id,
                "attributes": self.attributes if self.attributes is not None else {},
            },
        )
