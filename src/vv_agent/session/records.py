"""Experimental v1 closed records. Store envelopes never participate in identity."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.validators import extend

from vv_agent.canonical_json import canonical_json_bytes

if TYPE_CHECKING:
    from vv_agent.types import AgentTask


class RecordError(ValueError):
    pass


def copy_json(value: Any) -> Any:
    """Detach validated JSON at mutable input/output boundaries, without reparsing."""
    if isinstance(value, dict):
        return {key: copy_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [copy_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: copy_json(item) for key, item in value.items()}
    return value


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
        terminal_seq=POS,
        terminal_digest=HASH,
    ),
    "control": closed(action=CONTROL),
    "provider_evidence": closed(operation_id=TEXT, attempt=POS, request_digest=HASH, handle=HANDLE),
}
# Optional typed decision details remain inside the closed approval answer.
INPUT_PAYLOADS["approval_answer"]["properties"].update(reason={"type": "string"}, metadata=JSON_OBJECT)

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


def _compile_check(schema: dict[str, Any]) -> Callable[[Any], bool]:
    """Compile only the closed schema vocabulary above; reject new keywords at import."""
    if not schema:
        return lambda _: True
    if set(schema) == {"anyOf"}:
        checks = tuple(_compile_check(s) for s in schema["anyOf"])
        return lambda v: any(check(v) for check in checks)
    if set(schema) == {"oneOf"}:
        checks = tuple(_compile_check(s) for s in schema["oneOf"])
        return lambda v: sum(check(v) for check in checks) == 1
    if set(schema) == {"const"} and isinstance(schema["const"], str):
        return lambda v: isinstance(v, str) and v == schema["const"]
    kind = schema.get("type")
    if kind == "object":
        if set(schema) == {"type"}:
            return lambda v: isinstance(v, dict)
        if set(schema) == {"type", "properties", "required", "additionalProperties"}:
            props = schema["properties"]
            assert set(schema["required"]) <= set(props) and schema["additionalProperties"] is False
            required = frozenset(schema["required"])
            fields = tuple((k, _compile_check(s)) for k, s in props.items())
            if required == props.keys():
                return lambda v: isinstance(v, dict) and v.keys() == props.keys() and all(check(v[k]) for k, check in fields)
            return lambda v: (
                isinstance(v, dict)
                and required <= v.keys() <= props.keys()
                and all(k not in v or check(v[k]) for k, check in fields)
            )
    if kind == "array" and set(schema) == {"type", "items"}:
        check = _compile_check(schema["items"])
        return lambda v: isinstance(v, list) and all(check(item) for item in v)
    if kind == "integer":
        if set(schema) == {"type", "minimum"}:
            return lambda v: type(v) is int and v >= schema["minimum"]
        if set(schema) == {"type", "const"}:
            return lambda v: type(v) is int and v == schema["const"]
    if kind == "string":
        if set(schema) == {"type"}:
            return lambda v: isinstance(v, str)
        if set(schema) == {"type", "minLength"}:
            return lambda v: isinstance(v, str) and len(v) >= schema["minLength"]
        if set(schema) == {"type", "enum"}:
            values = frozenset(schema["enum"])
            return lambda v: isinstance(v, str) and v in values
        if set(schema) == {"type", "pattern"}:
            pattern = re.compile(schema["pattern"])
            return lambda v: isinstance(v, str) and pattern.search(v) is not None
    if schema == {"type": "boolean"}:
        return lambda v: type(v) is bool
    if schema == {"type": "null"}:
        return lambda v: v is None
    raise ValueError(f"unsupported session schema: {schema}")


_CHECKS = {key: _compile_check(validator.schema) for key, validator in _VALIDATORS.items()}


def validate(value: Any, schema: dict[str, Any], *, canonical: bool = True) -> None:
    try:
        if canonical:
            canonical_json_bytes(value)
        if not _CHECKS[id(schema)](value):
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
    if kind == "turn_parked":
        return f"turn/{turn_id}/wait/{payload['interaction_id']}"
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
    """encode() freezes logical JSON; payload and to_dict() return detached values."""

    record_id: str
    kind: str
    session_id: str
    turn_id: str | None
    operation_id: str | None
    attempt: int | None
    payload: dict[str, Any]
    schema_version: int = 1

    _body: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _digest: str | None = field(default=None, init=False, repr=False, compare=False)
    _value: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _task_value: AgentTask | None = field(default=None, init=False, repr=False, compare=False)

    def _task(self) -> AgentTask:
        from vv_agent.types import AgentTask

        if self._task_value is None:
            object.__setattr__(self, "_task_value", AgentTask.from_dict(copy_json(self._payload["definition"]["task"])))
        assert self._task_value is not None
        return self._task_value

    def task(self) -> AgentTask:
        from vv_agent.model_settings import ResponseFormat

        task = self._task()
        settings = task.model_settings
        if settings is not None and (
            any(
                value is not None
                for value in (settings.reasoning, settings.extra_headers, settings.extra_body, settings.extra_args)
            )
            or (isinstance(settings.response_format, ResponseFormat) and settings.response_format.json_schema is not None)
        ):
            settings = deepcopy(settings)
        return replace(
            task,
            metadata=copy_json(task.metadata),
            initial_shared_state=copy_json(task.initial_shared_state),
            initial_messages=deepcopy(task.initial_messages),
            sub_agents=deepcopy(task.sub_agents),
            model_settings=settings,
            extra_tool_names=list(task.extra_tool_names),
            exclude_tools=list(task.exclude_tools),
        )

    def __getattribute__(self, name: str) -> Any:
        if name == "payload":
            value = object.__getattribute__(self, "_value")
            if value is not None:
                return copy_json(value["payload"])
        return object.__getattribute__(self, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> Record:
        if self._body is not None:
            return self
        return Record(**deepcopy(self.to_dict(), memo))

    def to_dict(self) -> dict[str, Any]:
        if self._body is not None:
            return copy_json(self._value)
        return {f.name: deepcopy(getattr(self, f.name)) for f in fields(self) if f.init}

    @property
    def _payload(self) -> dict[str, Any]:
        # Private read-only view; detach before passing nested data to host callbacks.
        return self._value["payload"] if self._value is not None else self.payload

    def _retain(self, body: bytes, value: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_value", value if value is not None else json.loads(body))
        object.__setattr__(self, "payload", {})
        object.__setattr__(self, "_digest", sha256(body).hexdigest())
        object.__setattr__(self, "_body", body)

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
        for kind, digest_field in (
            ("turn_started", "definition"),
            ("op_planned", "request"),
            ("op_prepared", "request"),
            ("op_completed", "result"),
            ("input_applied", "input"),
        ):
            if self.kind == kind:
                _check_digest(self.payload, digest_field)
        if self.kind == "input_applied":
            incoming = self.payload["input"]
            validate(incoming["payload"], INPUT_PAYLOADS[incoming["kind"]], canonical=False)
        if self.kind == "op_planned" and (self.payload["op_kind"] == "model") != (self.payload["purpose"] is not None):
            raise RecordError("purpose required only for model operations")

    def encode(self) -> bytes:
        if self._body is None:
            value = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
            self._validate(value)
            try:
                body = canonical_json_bytes(value)
            except (ValueError, TypeError) as exc:
                raise RecordError(str(exc)) from exc
            self._retain(body)
        assert self._body is not None
        return self._body

    @property
    def digest(self) -> str:
        self.encode()
        assert self._digest is not None
        return self._digest

    @classmethod
    def parse(cls, body: bytes) -> Record:
        value = _load(body)
        try:
            record = cls(**value)
        except TypeError as exc:
            raise RecordError(str(exc)) from exc
        record._validate(value)
        try:
            canonical = canonical_json_bytes(value)
            record._retain(canonical, value if canonical == body else None)
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
