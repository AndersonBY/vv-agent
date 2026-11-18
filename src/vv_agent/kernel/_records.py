"""Closed session records and retained JSON values, independent of execution or stores."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256
from typing import Any

from vv_agent.kernel._canonical_json import _stdlib_compatible, canonical_json_bytes, utf16_sort_key
from vv_agent.kernel._execution_values import AgentTask, execution_id_for_values
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._model_settings import ResponseFormat
from vv_agent.kernel._record_codec import RecordError, _load, validate
from vv_agent.kernel._record_schema import (
    BOUNDARY_DATA,
    CONTENT_REF,
    INBOX_ITEM_SCHEMA,
    INPUT_PAYLOADS,
    INPUT_SCHEMA,
    INPUT_SCHEMA_LEVEL_2,
    INPUT_SCHEMA_LEVEL_3,
    OP_PLANNED_LEVEL_3,
    PAYLOADS,
    PRINCIPAL,
    PROVIDER_EVIDENCE_LEVEL_3,
    RECORD_ITEM_SCHEMA,
    RECORD_SCHEMA,
    RECORD_SCHEMA_LEVEL_2,
    RECORD_SCHEMA_LEVEL_3,
    SEED_POLICIES,
    SESSION_CREATED_LEVEL_2,
    TURN_STARTED_LEVEL_3,
)
from vv_agent.kernel._record_schema import CHILD_ADMISSION as CHILD_ADMISSION
from vv_agent.kernel._requirements import (
    SessionRequirements,
    normalize_requirements,
)


def _walk(value: Any):
    yield value
    if type(value) is dict:
        for item in value.values():
            yield from _walk(item)
    elif type(value) is list:
        for item in value:
            yield from _walk(item)


def _validate_content_refs(value: Any) -> None:
    source_order = {"system": 0, "user": 1, "internal": 2, "untrusted": 3}
    for node in _walk(value):
        if type(node) is dict and set(node) == {"content_id", "mac", "size", "key_id", "labels"}:
            source = node["labels"]["source"]
            if len(source) != len(set(source)):
                raise RecordError("ContentRef labels.source contains duplicates")
            if [source_order[item] for item in source] != sorted(source_order[item] for item in source):
                raise RecordError("ContentRef labels.source is not in source-kind order")


def _validate_signature(value: str) -> None:
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise RecordError("proof signature is not canonical base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != value:
        raise RecordError("proof signature is not canonical base64")


def _validate_admission_proof(proof: dict[str, Any] | None) -> None:
    if proof is not None:
        _validate_signature(proof["admission_signature"])


def _validate_input_proof(proof: dict[str, Any] | None) -> None:
    if proof is not None:
        _validate_signature(proof["signature"])


def _validate_ref(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise RecordError(f"{label} requires a ContentRef produced by the content store")
    try:
        validate(value, CONTENT_REF, canonical=False)
    except RecordError as exc:
        raise RecordError(f"{label} requires a ContentRef produced by the content store") from exc


def _validate_identifier(value: str, label: str) -> None:
    if not value or value != value.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise RecordError(f"{label} must be a non-empty identifier without surrounding whitespace or controls")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RecordError(f"{label} must be valid UTF-8 text") from exc


def _validate_level_3_environment(environment: dict[str, Any]) -> None:
    _validate_identifier(environment["environment_id"], "environment_id")
    capabilities = environment["environment_capabilities"]
    if capabilities != sorted(capabilities) or len(capabilities) != len(set(capabilities)):
        raise RecordError("environment capabilities must be sorted and unique")
    for capability in capabilities:
        _validate_identifier(capability, "environment capability")
    if environment["network"] != "open" and not environment["enforces_network_policy"]:
        raise RecordError("level-3 network policy must be enforced")
    capabilities_digest = environment["capabilities_digest"]
    if capabilities_digest is None:
        return
    preimage = {
        key: environment[key]
        for key in (
            "environment_id",
            "generation",
            "isolation",
            "network",
            "enforces_network_policy",
            "enforces_resource_limits",
            "persistent",
            "environment_capabilities",
            "network_policy_ref_digest",
            "isolation_policy_ref_digest",
        )
    }
    expected = sha256(canonical_json_bytes(preimage, "environment capabilities")).hexdigest()
    if capabilities_digest != expected:
        raise RecordError("capabilities_digest does not match the environment binding")


def _validate_execution_identity(record: Record) -> None:
    identity = record.payload["execution_identity"]
    for field_name in ("operation_key", "tenant_id", "environment_id", "session_id", "operation_id"):
        _validate_identifier(identity[field_name], field_name)
    if record.operation_id is None or identity["session_id"] != record.session_id:
        raise RecordError("execution identity session does not match the record envelope")
    if identity["operation_id"] != record.operation_id:
        raise RecordError("execution identity operation does not match the record envelope")
    expected_key = f"{record.session_id}/{record.operation_id}"
    if identity["operation_key"] != expected_key:
        raise RecordError("execution identity operation_key does not match the stable framing")
    expected_execution_id = execution_id_for_values(identity["tenant_id"], identity["environment_id"], expected_key)
    if identity["execution_id"] != expected_execution_id:
        raise RecordError("execution identity execution_id does not match its stable inputs")


def _validate_provider_evidence(item: InboxItem) -> None:
    payload = item.payload
    lookup = payload["lookup"]
    binding = lookup["binding"]
    for field_name in ("tenant_id", "session_id", "operation_id", "idempotency_key", "environment_id", "provider_id"):
        _validate_identifier(binding[field_name], field_name)
    if item.actor is None or binding["tenant_id"] != item.actor["tenant_id"]:
        raise RecordError("provider lookup tenant does not match the inbox actor")
    if any(binding[key] != payload[key] for key in ("operation_id", "attempt", "request_digest")):
        raise RecordError("provider lookup binding does not match provider_evidence")
    expected_key = f"{binding['session_id']}/{binding['operation_id']}"
    if binding["idempotency_key"] != expected_key:
        raise RecordError("provider lookup idempotency_key does not match the stable operation key")
    expected_execution_id = execution_id_for_values(
        binding["tenant_id"],
        binding["environment_id"],
        expected_key,
    )
    if binding["execution_id"] != expected_execution_id:
        raise RecordError("provider lookup execution_id does not match its stable inputs")
    if not lookup["evidence"] or any(not value.strip() for value in lookup["evidence"]):
        raise RecordError("provider lookup evidence is incomplete")

    status = lookup["status"]
    outcome = lookup["outcome"]
    expected_kind = {
        "not_started": None,
        "running": "Accepted",
        "exited": "Definitive",
        "unknown": "Unknown",
    }[status]
    outcome_kind = None if outcome is None else outcome["kind"]
    if outcome_kind != expected_kind:
        raise RecordError(f"provider lookup {status} requires {expected_kind or 'no outcome'}")
    if outcome_kind == "Accepted":
        handle = outcome["handle"]
        expected_handle = binding | {"provider": binding["provider_id"]}
        if any(
            handle[key] != expected_handle[key]
            for key in (
                "provider",
                "operation_id",
                "attempt",
                "request_digest",
                "execution_id",
                "command_digest",
                "environment_id",
                "generation",
            )
        ):
            raise RecordError("provider lookup handle does not match the authenticated binding")
        if not set(handle["evidence"]).issubset(lookup["evidence"]):
            raise RecordError("provider lookup lost acceptance evidence")
    if outcome_kind == "Definitive" and not set(outcome["evidence"]).issubset(lookup["evidence"]):
        raise RecordError("provider lookup lost terminal evidence")


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
    return sha256(_json_bytes(value)).hexdigest()


def _internable(value: Any) -> bool:
    # Private keys must not coerce tuples, object keys or scalar subclasses.
    kind = type(value)
    if kind is dict:
        return all(type(key) is str and _internable(item) for key, item in value.items())
    if kind is list:
        return all(_internable(item) for item in value)
    return value is None or kind in (str, bool, int, float)


class _MessageCache:
    """Private immutable message values, validated forms and their canonical bytes."""

    def __init__(self) -> None:
        self.wires: dict[bytes, dict[str, Any]] = {}
        self._known: set[int] = set()
        self._decoded: dict[int, tuple[dict[str, Any], Message]] = {}
        self._bodies: dict[int, bytes | None] = {}
        self._wire_bodies: dict[int, bytes] = {}

    def copy(self) -> _MessageCache:
        result = _MessageCache()
        result.wires = self.wires.copy()
        result._known = self._known.copy()
        result._decoded = self._decoded.copy()
        result._bodies = self._bodies.copy()
        result._wire_bodies = self._wire_bodies.copy()
        return result

    def share(self, value: dict[str, Any], *, detach: bool = False) -> dict[str, Any]:
        if id(value) in self._known:
            return value
        if not _internable(value):
            return copy_json(value) if detach else value
        # An equality key, not canonical bytes: keep 1.0, 1, true and NaN distinct.
        # ASCII escapes preserve future message strings without validating them.
        key = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        retained = self.wires.get(key)
        if retained is None:
            retained = copy_json(value) if detach else value
            self.wires[key] = retained
        self._known.add(id(retained))
        return retained

    def retain_array(self, values: list[dict[str, Any]]) -> _RetainedMessages | list[dict[str, Any]]:
        # Only canonical-safe values may supply retained wire fragments.
        if not _stdlib_compatible(values):
            return values
        private = [self.share(value, detach=True) for value in values]
        return _RetainedMessages(private, self.wire_array(private))

    def wire_array(self, values: list[Any]) -> bytes:
        parts = [b"["]
        for index, value in enumerate(values):
            if index:
                parts.append(b",")
            if isinstance(value, dict):
                value = values[index] = self.share(value)
                body = self._wire_bodies.get(id(value))
                if body is None:
                    body = canonical_json_bytes(value)
                    self._wire_bodies[id(value)] = body
            else:
                body = canonical_json_bytes(value)
            parts.append(body)
        parts.append(b"]")
        return b"".join(parts)

    def parse(self, value: dict[str, Any]) -> Message:
        cached = self._decoded.get(id(value))
        if cached is None:
            message = Message.from_dict(copy_json(value))
            self._decoded[id(value)] = (value, message)
            self._bodies[id(message)] = None
            return message
        return cached[1]

    def source_digest(self, messages: list[Message]) -> str:
        checksum = sha256(b"[")
        for index, message in enumerate(messages):
            if index:
                checksum.update(b",")
            identity = id(message)
            body = self._bodies.get(identity)
            if body is None:
                body = canonical_json_bytes(message.to_dict())
                if identity in self._bodies:
                    self._bodies[identity] = body
            checksum.update(body)
        checksum.update(b"]")
        return checksum.hexdigest()


def inherited_seed_policies(attributes: dict[str, Any]) -> dict[str, bool]:
    """Derivation freezes each source choice, including an undeclared source's false."""
    if set(attributes) == set(CONTENT_REF["properties"]):
        raise RecordError("S1 content hydration is unavailable in the producer/codec phase")
    seed = attributes.get("seed", {})
    return {name: seed.get(name, False) for name in SEED_POLICIES}


def _check_digest(payload: dict[str, Any], field: str, encoded: bytes | None = None) -> bytes:
    encoded = _json_bytes(payload[field]) if encoded is None else encoded
    if payload[f"{field}_digest"] != sha256(encoded).hexdigest():
        raise RecordError(f"{field} digest mismatch")
    return encoded


@dataclass(frozen=True)
class _RetainedTools:
    """Producer-only reference to a validated, read-only schema subtree."""

    source: Record

    def encode(self) -> bytes:
        self.source.encode()
        if self.source._tools_body is None:
            object.__setattr__(self.source, "_tools_body", canonical_json_bytes(self.value))
        assert self.source._tools_body is not None
        return self.source._tools_body

    @property
    def value(self) -> list[dict[str, Any]]:
        return self.source._payload["definition"]["tools"]


@dataclass(frozen=True)
class _RetainedMessages:
    """Producer-only canonical fragments from detached, private message values."""

    value: list[dict[str, Any]]
    body: bytes

    def encode(self) -> bytes:
        return self.body


def _json_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        retained = {key: item.encode() for key, item in value.items() if isinstance(item, _RetainedTools | _RetainedMessages)}
        if retained:
            return _object_bytes(value, retained)
    return canonical_json_bytes(value)


def _object_bytes(value: dict[str, Any], encoded_fields: dict[str, bytes]) -> bytes:
    parts = [b"{"]
    pending = {}
    keys = sorted(value) if all(type(key) is str and key.isascii() for key in value) else sorted(value, key=utf16_sort_key)
    for key in keys:
        if key in encoded_fields:
            if pending:
                if len(parts) > 1:
                    parts.append(b",")
                parts.append(_json_bytes(pending)[1:-1])
                pending.clear()
            if len(parts) > 1:
                parts.append(b",")
            parts.extend((canonical_json_bytes(key), b":", encoded_fields[key]))
        else:
            pending[key] = value[key]
    if pending:
        if len(parts) > 1:
            parts.append(b",")
        parts.append(_json_bytes(pending)[1:-1])
    parts.append(b"}")
    return b"".join(parts)


def _record_bytes(value: dict[str, Any], encoded_fields: dict[str, bytes]) -> bytes:
    if not encoded_fields:
        return _json_bytes(value)
    payload = _object_bytes(value["payload"], encoded_fields)
    envelope = _json_bytes(value | {"payload": None})
    return envelope.replace(b'"payload":null', b'"payload":' + payload, 1)


@dataclass(frozen=True)
class InboxItem:
    input_id: str
    kind: str
    payload: dict[str, Any]
    target_turn_id: str | None = None
    generation: int | None = None
    available_ms: int = 0
    actor: dict[str, Any] | None = None
    admission_proof: dict[str, Any] | None = None
    schema_version: str = "S1"
    feature_level: int = 1
    requirements: SessionRequirements | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        if self.requirements is None:
            value.pop("requirements")
        else:
            value["requirements"] = self._requirements().to_dict()
        return value

    def _requirements(self) -> SessionRequirements:
        if self.requirements is None:
            raise RecordError("level-2 inbox requirements are required")
        try:
            return normalize_requirements(self.requirements)
        except (TypeError, ValueError) as exc:
            raise RecordError(str(exc)) from exc

    def _validate(self, value: dict[str, Any]) -> None:
        if self.actor is None:
            raise RecordError("S1 actor requires a closed Principal")
        schemas = {1: INPUT_SCHEMA, 2: INPUT_SCHEMA_LEVEL_2, 3: INPUT_SCHEMA_LEVEL_3}
        try:
            schema = schemas[self.feature_level]
        except KeyError as exc:
            raise RecordError("unsupported inbox feature level") from exc
        validate(value, schema, canonical=False)
        payload_schema = PROVIDER_EVIDENCE_LEVEL_3 if self.feature_level == 3 else INPUT_PAYLOADS[self.kind]
        validate(self.payload, payload_schema, canonical=False)
        validate(self.actor, PRINCIPAL, canonical=False)
        _validate_content_refs(value)
        _validate_admission_proof(self.admission_proof)
        if self.kind == "model_selection" and (self.target_turn_id is not None or self.generation is not None):
            raise RecordError("model_selection requires session scope")
        if self.feature_level == 1:
            if self.requirements is not None:
                raise RecordError("level-1 inbox items cannot contain requirements")
        elif self.feature_level == 2:
            if self.kind not in {"user", "follow_up", "model_selection"}:
                raise RecordError("level-2 inbox requirements are limited to user, follow_up and model_selection")
            requirements = self._requirements()
            if self.kind == "model_selection":
                if requirements.handler_version is None:
                    raise RecordError("model_selection requirements need the current handler")
                if any(not item.startswith("model:") for item in requirements.capabilities):
                    raise RecordError("model_selection requirements may contain model capabilities only")
            elif requirements.handler_version is None and not requirements.capabilities:
                raise RecordError("null-handler requirements need an explicit family capability")
        elif self.feature_level == 3:
            if self.requirements is not None:
                raise RecordError("level-3 inbox items cannot contain requirements")
            _validate_provider_evidence(self)
        else:
            raise RecordError("unsupported inbox feature level")

    def encode(self) -> bytes:
        value = self.to_dict()
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
        validate(value, INBOX_ITEM_SCHEMA, canonical=False)
        try:
            item = cls(**value)
        except TypeError as exc:
            raise RecordError(str(exc)) from exc
        item._validate(value)
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
    if kind == "boundary_recorded":
        return f"turn/{turn_id}/boundary/{payload['stage']}/{payload['boundary_id']}"
    if kind == "context_compacted":
        identity = f"compact/{payload['source_digest']}/{payload['mode']}"
        return identity if payload["summary_operation_id"] is None else f"{identity}/{payload['summary_operation_id']}"
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
    schema_version: str = "S1"
    feature_level: int = 1

    _body: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _digest: str | None = field(default=None, init=False, repr=False, compare=False)
    _value: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _task_value: AgentTask | None = field(default=None, init=False, repr=False, compare=False)
    _tools_body: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _message_cache_id: int | None = field(default=None, init=False, repr=False, compare=False)

    def _task(self) -> AgentTask:
        if self._task_value is not None:
            return self._task_value
        raise RecordError("S1 content hydration is unavailable in the producer/codec phase")

    def task(self) -> AgentTask:
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
        value = value if value is not None else json.loads(body)
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "payload", {})
        object.__setattr__(self, "_digest", sha256(body).hexdigest())
        object.__setattr__(self, "_body", body)

    def _share_messages(self, messages: _MessageCache) -> None:
        object.__setattr__(self, "_message_cache_id", id(messages))

    def _validate(self, value: dict[str, Any], messages: _MessageCache | None = None) -> dict[str, bytes]:
        level_schemas = {1: RECORD_SCHEMA, 2: RECORD_SCHEMA_LEVEL_2, 3: RECORD_SCHEMA_LEVEL_3}
        try:
            level_schema = level_schemas[self.feature_level]
        except KeyError as exc:
            raise RecordError("unsupported record feature level") from exc
        validate(value, level_schema, canonical=False)
        if self.feature_level == 2:
            payload_schema = SESSION_CREATED_LEVEL_2
        elif self.feature_level == 3:
            payload_schema = {
                "turn_started": TURN_STARTED_LEVEL_3,
                "op_planned": OP_PLANNED_LEVEL_3,
            }[self.kind]
        else:
            payload_schema = PAYLOADS[self.kind]
        validate(self.payload, payload_schema, canonical=False)
        _validate_content_refs(value)
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
            try:
                nested = InboxItem(**incoming)
            except TypeError as exc:
                raise RecordError(str(exc)) from exc
            nested._validate(incoming)
            proof = self.payload["input_proof"]
            admission = incoming["admission_proof"]
            if (proof is None) != (admission is None):
                raise RecordError("input proof admission mismatch")
            _validate_input_proof(proof)
            if (
                proof is not None
                and admission is not None
                and (
                    proof["key_id"] != admission["admission_key_id"]
                    or proof["signature"] != admission["admission_signature"]
                    or proof["signed_ms"] != admission["admission_signed_ms"]
                )
            ):
                raise RecordError("input proof does not copy admission proof")
        if self.feature_level == 2:
            try:
                bootstrap = normalize_requirements(self.payload["bootstrap"])
            except (TypeError, ValueError) as exc:
                raise RecordError(str(exc)) from exc
            if bootstrap.handler_version is None and not bootstrap.capabilities:
                raise RecordError("null-handler bootstrap needs an explicit family capability")
        if self.kind == "turn_started":
            requirements = self.payload["requirements"]
            if requirements["capabilities"] != sorted(requirements["capabilities"]):
                raise RecordError("requirements capabilities are not sorted")
            if len(requirements["capabilities"]) != len(set(requirements["capabilities"])):
                raise RecordError("requirements capabilities contain duplicates")
            environment = self.payload["environment"]
            if environment is not None:
                capabilities = environment["environment_capabilities"]
                if capabilities != sorted(capabilities):
                    raise RecordError("environment capabilities are not sorted")
                if len(capabilities) != len(set(capabilities)):
                    raise RecordError("environment capabilities contain duplicates")
                if self.feature_level == 3:
                    _validate_level_3_environment(environment)
        if self.kind == "boundary_recorded":
            validate(self.payload["data"], BOUNDARY_DATA[self.payload["stage"]], canonical=False)
        if self.kind == "op_planned" and (self.payload["op_kind"] == "model") != (self.payload["purpose"] is not None):
            raise RecordError("purpose required only for model operations")
        if self.feature_level == 3 and self.kind == "op_planned":
            _validate_execution_identity(self)
        return {}

    def encode(self) -> bytes:
        if self._body is None:
            value = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
            encoded_fields = self._validate(value)
            try:
                body = _record_bytes(value, encoded_fields)
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
        return cls._parse(body)

    @classmethod
    def _parse(cls, body: bytes, messages: _MessageCache | None = None) -> Record:
        value = _load(body)
        validate(value, RECORD_ITEM_SCHEMA, canonical=False)
        try:
            record = cls(**value)
        except TypeError as exc:
            raise RecordError(str(exc)) from exc
        encoded_fields = record._validate(value, messages)
        try:
            canonical = _record_bytes(value, encoded_fields)
            if canonical == body:
                record._retain(body if type(body) is bytes else canonical, value)
                if messages is not None:
                    object.__setattr__(record, "_message_cache_id", id(messages))
            else:
                record._retain(canonical)
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
    feature_level: int = 1,
    bootstrap: SessionRequirements | dict[str, Any] | None = None,
) -> Record:
    if kind not in PAYLOADS:
        raise RecordError("unknown record kind")
    if feature_level not in {1, 2, 3}:
        raise RecordError("unsupported record feature level")
    payload = copy_json(payload)
    if feature_level == 2 and kind != "session_created":
        raise RecordError("record feature level 2 is limited to session_created")
    if feature_level == 3 and kind not in {"turn_started", "op_planned"}:
        raise RecordError("record feature level 3 is limited to turn_started and op_planned")
    if bootstrap is not None:
        if feature_level != 2 or kind != "session_created":
            raise RecordError("bootstrap is limited to level-2 session_created records")
        try:
            payload["bootstrap"] = normalize_requirements(bootstrap).to_dict()
        except (TypeError, ValueError) as exc:
            raise RecordError(str(exc)) from exc
    if kind == "turn_started":
        payload.setdefault("policy_digest", None)
        payload.setdefault(
            "requirements",
            {"level": 1, "handler_version": payload.get("handler_version"), "capabilities": []},
        )
        payload.setdefault("tokenizer_binding", None)
        payload.setdefault("environment", None)
    elif kind in {"op_planned", "op_prepared", "op_completed"}:
        payload.setdefault("policy_decision", None)
    if feature_level == 2:
        payload_schema = SESSION_CREATED_LEVEL_2
    elif feature_level == 3:
        payload_schema = {
            "turn_started": TURN_STARTED_LEVEL_3,
            "op_planned": OP_PLANNED_LEVEL_3,
        }[kind]
    else:
        payload_schema = PAYLOADS[kind]
    validate(payload, payload_schema, canonical=False)
    record = Record(
        record_identity(kind, session_id, turn_id, operation_id, attempt, payload),
        kind,
        session_id,
        turn_id,
        operation_id,
        attempt,
        payload,
        feature_level=feature_level,
    )
    record.encode()
    return record


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    principal: dict[str, Any]
    workspace: dict[str, Any]
    parent_session_id: str | None = None
    parent_operation_id: str | None = None
    attributes: dict[str, Any] | None = None
    feature_level: int = 1
    bootstrap: SessionRequirements | dict[str, Any] | None = None

    def record(self, *, _existing: Record | None = None) -> Record:
        attributes = copy_json(self.attributes)
        if attributes is None and _existing is not None:
            attributes = copy_json(_existing._payload["attributes"])
        if attributes is None:
            raise RecordError("attributes requires a ContentRef produced by the content store")
        bootstrap = copy_json(self.bootstrap)
        if bootstrap is None and _existing is not None and _existing.feature_level == 2:
            bootstrap = copy_json(_existing._payload["bootstrap"])
        validate(self.principal, PRINCIPAL, canonical=False)
        _validate_ref(self.workspace, "workspace")
        _validate_ref(attributes, "attributes")
        return make_record(
            "session_created",
            session_id=self.session_id,
            payload={
                "principal": self.principal,
                "workspace": self.workspace,
                "parent_session_id": self.parent_session_id,
                "parent_operation_id": self.parent_operation_id,
                "attributes": attributes,
            },
            feature_level=self.feature_level,
            bootstrap=bootstrap,
        )
