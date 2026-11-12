"""Experimental v1 closed records. Store envelopes never participate in identity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import orjson

from vv_agent.kernel._canonical_json import _stdlib_compatible, canonical_json_bytes, utf16_sort_key
from vv_agent.kernel._model_settings import ModelSettings
from vv_agent.kernel._record_codec import RecordError as RecordError
from vv_agent.kernel._record_codec import _load as _load
from vv_agent.kernel._record_codec import validate as validate
from vv_agent.kernel._record_schema import _CHECKS as _CHECKS
from vv_agent.kernel._record_schema import _SCHEMAS as _SCHEMAS
from vv_agent.kernel._record_schema import APP_SERVER_ATTRIBUTES as APP_SERVER_ATTRIBUTES
from vv_agent.kernel._record_schema import BOUNDARY_DATA as BOUNDARY_DATA
from vv_agent.kernel._record_schema import CHILD_ADMISSION as CHILD_ADMISSION
from vv_agent.kernel._record_schema import HANDLE as HANDLE
from vv_agent.kernel._record_schema import INPUT_PAYLOADS as INPUT_PAYLOADS
from vv_agent.kernel._record_schema import INPUT_SCHEMA as INPUT_SCHEMA
from vv_agent.kernel._record_schema import MODEL_BINDING as MODEL_BINDING
from vv_agent.kernel._record_schema import MODEL_SELECTION as MODEL_SELECTION
from vv_agent.kernel._record_schema import PAYLOADS as PAYLOADS
from vv_agent.kernel._record_schema import RECORD_SCHEMA as RECORD_SCHEMA
from vv_agent.kernel._record_schema import REQUEST_SESSION_METADATA as REQUEST_SESSION_METADATA
from vv_agent.kernel._record_schema import SEED as SEED
from vv_agent.kernel._record_schema import SEED_POLICIES as SEED_POLICIES
from vv_agent.kernel._record_schema import TASK_SESSION_METADATA as TASK_SESSION_METADATA
from vv_agent.kernel._record_schema import nullable as nullable

if TYPE_CHECKING:
    from vv_agent.kernel._message_values import Message
    from vv_agent.types import AgentTask


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

    def share(self, value: dict[str, Any], *, strict: bool = False, detach: bool = False) -> dict[str, Any]:
        if id(value) in self._known:
            return value
        # This is an equality key, never persisted or used as a protocol digest.
        options = orjson.OPT_STRICT_INTEGER if strict else 0
        sorted_key = None
        try:
            key = orjson.dumps(value, option=options)
        except orjson.JSONEncodeError:
            if strict:
                canonical_json_bytes(value)
                raise
            key = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            sorted_key = key
        retained = self.wires.get(key)
        if retained is None:
            sorted_key = sorted_key if sorted_key is not None else orjson.dumps(value, option=options | orjson.OPT_SORT_KEYS)
            retained = self.wires.get(sorted_key)
            if retained is None:
                retained = copy_json(value) if detach else value
                self.wires[sorted_key] = retained
            self.wires[key] = retained
        self._known.add(id(retained))
        return retained

    def retain_array(self, values: list[dict[str, Any]]) -> _RetainedMessages | list[dict[str, Any]]:
        # orjson can map NaN to null and serialize non-JSON objects; do not intern those.
        if not _stdlib_compatible(values):
            return values
        private = [self.share(value, strict=True, detach=True) for value in values]
        return _RetainedMessages(private, self.wire_array(private))

    def wire_array(self, values: list[Any]) -> bytes:
        parts = [b"["]
        for index, value in enumerate(values):
            if index:
                parts.append(b",")
            if isinstance(value, dict):
                value = values[index] = self.share(value, strict=True)
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
        from vv_agent.kernel._message_values import Message

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
    # The checked subset has identical stdlib/JCS and orjson encodings. Emit
    # bytes directly instead of allocating a full Unicode encoding first.
    if _stdlib_compatible(value):
        try:
            return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
        except orjson.JSONEncodeError:
            # orjson has a lower nesting limit than the canonical encoder.
            pass
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
    schema_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _validate(self, value: dict[str, Any]) -> None:
        validate(value, INPUT_SCHEMA, canonical=False)
        validate(self.payload, INPUT_PAYLOADS[self.kind], canonical=False)
        if self.kind == "model_selection":
            if self.target_turn_id is not None or self.generation is not None:
                raise RecordError("model_selection requires session scope")
            selection = self.payload["selection"]
            if selection is not None:
                try:
                    ModelSettings.from_dict(selection.get("model_settings", {}))
                except (ValueError, TypeError) as exc:
                    raise RecordError(str(exc)) from exc

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
    schema_version: int = 1

    _body: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _digest: str | None = field(default=None, init=False, repr=False, compare=False)
    _value: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _task_value: AgentTask | None = field(default=None, init=False, repr=False, compare=False)
    _tools_body: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _message_cache_id: int | None = field(default=None, init=False, repr=False, compare=False)

    def _task(self) -> AgentTask:
        from vv_agent.types import AgentTask

        if self._task_value is None:
            object.__setattr__(self, "_task_value", AgentTask.from_dict(copy_json(self._payload["definition"]["task"])))
        assert self._task_value is not None
        return self._task_value

    def task(self) -> AgentTask:
        from vv_agent.kernel._model_settings import ResponseFormat

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
        for container_field in ("definition", "request"):
            container = self._payload.get(container_field)
            if isinstance(container, dict) and isinstance(container.get("tools"), _RetainedTools):
                retained = container["tools"]
                value["payload"][container_field]["tools"] = retained.value
                if container_field == "definition":
                    object.__setattr__(self, "_tools_body", retained.encode())
            if isinstance(container, dict) and isinstance(container.get("messages"), _RetainedMessages):
                value["payload"][container_field]["messages"] = container["messages"].value
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "payload", {})
        object.__setattr__(self, "_digest", sha256(body).hexdigest())
        object.__setattr__(self, "_body", body)

    def _share_messages(self, messages: _MessageCache) -> None:
        # Retained JSON is private and read-only; public payloads still detach it.
        # This marker only avoids repeated interning, never validation or digest checks.
        if self._message_cache_id == id(messages):
            return
        p = self._payload
        snapshots = []
        if self.kind == "op_planned" and p["op_kind"] == "model":
            snapshots = p["request"].get("messages", [])
        elif self.kind == "boundary_recorded" and p["stage"] == "before_memory":
            snapshots = p["data"]["messages"]
        if isinstance(snapshots, list):
            for index, message in enumerate(snapshots):
                if isinstance(message, dict):
                    snapshots[index] = messages.share(message)
        object.__setattr__(self, "_message_cache_id", id(messages))

    def _validate(self, value: dict[str, Any], messages: _MessageCache | None = None) -> dict[str, bytes]:
        validate(value, RECORD_SCHEMA, canonical=False)
        validate(self.payload, PAYLOADS[self.kind], canonical=False)
        op = self.kind.startswith("op_")
        if op != (self.operation_id is not None) or op != (self.attempt is not None):
            raise RecordError("operation and attempt required only for operation records")
        encoded_fields = {}
        if self.kind == "session_created" and "seed" in self.payload["attributes"]:
            validate(self.payload["attributes"]["seed"], SEED, canonical=False)
            from vv_agent.kernel._message_values import Message

            for message in self.payload["attributes"]["seed"].get("messages", []):
                try:
                    Message.from_dict(message)
                except (ValueError, TypeError, KeyError) as exc:
                    raise RecordError(str(exc)) from exc
        for field_name, schema in (("definition", TASK_SESSION_METADATA), ("request", REQUEST_SESSION_METADATA)):
            container = self.payload.get(field_name, {})
            if field_name == "definition" and self.kind == "session_created":
                container = self.payload["attributes"].get("child_admission", {}).get("definition", {})
            metadata = (
                container.get("task", {}).get("metadata", {}) if field_name == "definition" else container.get("metadata", {})
            )
            if "vv_session" in metadata:
                validate(metadata["vv_session"], schema, canonical=False)
        if self.kind == "session_created" and "app_server" in self.payload["attributes"]:
            validate(self.payload["attributes"]["app_server"], APP_SERVER_ATTRIBUTES, canonical=False)
        if self.kind == "session_created" and "child_admission" in self.payload["attributes"]:
            attributes = self.payload["attributes"]
            admission = attributes["child_admission"]
            validate(admission, CHILD_ADMISSION, canonical=False)
            definition = _check_digest(admission, "definition")
            encoded_fields["attributes"] = _object_bytes(
                attributes, {"child_admission": _object_bytes(admission, {"definition": definition})}
            )
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
                encoded = None
                if self.kind == "op_planned" and self.payload["op_kind"] == "model" and messages is not None:
                    request = self.payload["request"]
                    if isinstance(request.get("messages"), list) and _stdlib_compatible(request["messages"]):
                        encoded = _object_bytes(request, {"messages": messages.wire_array(request["messages"])})
                encoded_fields[digest_field] = _check_digest(self.payload, digest_field, encoded)
        if self.kind == "input_applied":
            incoming = self.payload["input"]
            InboxItem(**incoming)._validate(incoming)
        if self.kind == "boundary_recorded":
            validate(self.payload["data"], BOUNDARY_DATA[self.payload["stage"]], canonical=False)
            if self.payload["stage"] == "before_memory" and messages is not None:
                data = self.payload["data"]
                if _stdlib_compatible(data["messages"]):
                    try:
                        encoded_fields["data"] = _object_bytes(data, {"messages": messages.wire_array(data["messages"])})
                    except (ValueError, TypeError) as exc:
                        raise RecordError(str(exc)) from exc
        if self.kind == "op_planned" and (self.payload["op_kind"] == "model") != (self.payload["purpose"] is not None):
            raise RecordError("purpose required only for model operations")
        return encoded_fields

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

    def record(self, *, _existing: Record | None = None) -> Record:
        attributes = copy_json(self.attributes) if self.attributes is not None else {}
        if not isinstance(attributes, dict):
            raise RecordError("attributes must be an object")
        if "seed" in attributes and not isinstance(attributes["seed"], dict):
            raise RecordError("seed must be an object")
        if _existing is None:
            for name in SEED_POLICIES:
                attributes.setdefault("seed", {}).setdefault(name, True)
        else:
            original = _existing._payload["attributes"].get("seed", {})
            declarations = {name: original[name] for name in SEED_POLICIES if name in original}
            if "seed" in attributes:
                for name, value in declarations.items():
                    attributes["seed"].setdefault(name, value)
            elif original and set(original) <= set(SEED_POLICIES):
                attributes["seed"] = copy_json(original)
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
        )
