from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

MAX_WIRE_INTEGER = (1 << 53) - 1
MAX_CHECKPOINT_KEY_BYTES = 512
MAX_EXTENSION_NAMESPACE_BYTES = 128
MAX_EXTENSION_ENTRY_BYTES = 65_536
DEFAULT_MAX_EXTENSION_STATE_BYTES = 262_144
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXTENSION_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*\.[a-z0-9._-]+$")
_CAPABILITY_SLOT_RE = re.compile(r"^[a-z][a-z0-9_.:-]*$")
_JSON_POINTER_ESCAPE_RE = re.compile(r"~(?:0|1)")
RUN_DEFINITION_V1_SCHEMA = "vv-agent.run-definition.v1"
OPERATION_REQUEST_V1_SCHEMA = "vv-agent.operation-request.v1"
CREDENTIAL_REDACTION_VALUE = "<credential-redacted>"

_RUN_DEFINITION_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "agent",
        "root_input",
        "compiled_prompt",
        "initial_messages",
        "initial_shared_state",
        "run_metadata",
        "context_ref",
        "model",
        "credential_slots",
        "runtime_controls",
        "tools",
        "tool_policy",
        "checkpoint_policy",
        "budget_limits",
        "output_schema",
        "workspace_ref",
        "session_ref",
        "extensions",
        "capability_refs",
    }
)


class ResumePolicy(StrEnum):
    NEW = "new"
    RESUME_IF_PRESENT = "resume_if_present"
    REQUIRE_EXISTING = "require_existing"


class AmbiguousModelPolicy(StrEnum):
    REQUIRE_RECONCILIATION = "require_reconciliation"
    RETRY_WITH_DUPLICATE_RISK = "retry_with_duplicate_risk"


class AmbiguousToolPolicy(StrEnum):
    REQUIRE_RECONCILIATION = "require_reconciliation"
    RETRY_IDEMPOTENT_ONLY = "retry_idempotent_only"


class ToolIdempotency(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class OperationKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


class OperationState(StrEnum):
    PLANNED = "planned"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


class ReconciliationDecisionKind(StrEnum):
    DEFER = "defer"
    RETRY = "retry"
    REPLAY_SUCCESS = "replay_success"
    RECORD_FAILURE = "record_failure"
    ABORT = "abort"


class CheckpointError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class EventCursor:
    store_ref: dict[str, str]
    value: Any
    last_event_id: str | None = None
    schema_version: str = "vv-agent.event-cursor.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "vv-agent.event-cursor.v1":
            raise ValueError("unsupported event cursor schema_version")
        _validate_capability_ref(self.store_ref, "event cursor store_ref")
        _canonical_json(self.value, "event cursor value")
        if self.last_event_id is not None and (
            not isinstance(self.last_event_id, str) or not self.last_event_id.strip()
        ):
            raise ValueError("event cursor last_event_id must be a non-empty string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "store_ref": dict(self.store_ref),
            "value": self.value,
            "last_event_id": self.last_event_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventCursor:
        if not isinstance(payload, Mapping):
            raise ValueError("event cursor must be an object")
        if set(payload) != {"schema_version", "store_ref", "value", "last_event_id"}:
            raise ValueError("event cursor has missing or unknown fields")
        raw_ref = payload.get("store_ref")
        if not isinstance(raw_ref, Mapping):
            raise ValueError("event cursor store_ref must be an object")
        _validate_capability_ref(raw_ref, "event cursor store_ref")
        ref_id = raw_ref["id"]
        ref_version = raw_ref["version"]
        assert isinstance(ref_id, str)
        assert isinstance(ref_version, str)
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            store_ref={"id": ref_id, "version": ref_version},
            value=payload.get("value"),
            last_event_id=payload.get("last_event_id"),
        )


@dataclass(frozen=True, slots=True)
class ResumeObservation:
    operation_id: str
    operation_kind: OperationKind
    cycle_index: int
    state: OperationState = OperationState.AMBIGUOUS
    risk: str = ""
    idempotency_support: ToolIdempotency | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_kind, OperationKind):
            object.__setattr__(self, "operation_kind", OperationKind(self.operation_kind))
        if not isinstance(self.state, OperationState):
            object.__setattr__(self, "state", OperationState(self.state))
        if self.state is not OperationState.AMBIGUOUS:
            raise ValueError("resume observation state must be ambiguous")
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("resume observation operation_id must be non-empty")
        _positive_wire_integer(self.cycle_index, "resume observation cycle_index")
        if not isinstance(self.risk, str) or not self.risk:
            raise ValueError("resume observation risk must be non-empty")
        if self.idempotency_support is not None and not isinstance(
            self.idempotency_support, ToolIdempotency
        ):
            object.__setattr__(
                self,
                "idempotency_support",
                ToolIdempotency(self.idempotency_support),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_kind": self.operation_kind.value,
            "cycle_index": self.cycle_index,
            "state": self.state.value,
            "risk": self.risk,
            "idempotency_support": (
                self.idempotency_support.value if self.idempotency_support is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResumeObservation:
        if not isinstance(payload, Mapping):
            raise ValueError("resume observation must be an object")
        if set(payload) != {
            "operation_id",
            "operation_kind",
            "cycle_index",
            "state",
            "risk",
            "idempotency_support",
        }:
            raise ValueError("resume observation has missing or unknown fields")
        return cls(
            operation_id=_required_string(payload, "operation_id"),
            operation_kind=OperationKind(_required_string(payload, "operation_kind")),
            cycle_index=_required_integer(payload, "cycle_index"),
            state=OperationState(_required_string(payload, "state")),
            risk=_required_string(payload, "risk"),
            idempotency_support=(
                ToolIdempotency(payload["idempotency_support"])
                if payload.get("idempotency_support") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("reconciliation error code and message must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReconciliationError:
        if not isinstance(payload, Mapping):
            raise ValueError("reconciliation error must be an object")
        if set(payload) != {"code", "message", "retryable"}:
            raise ValueError("reconciliation error has missing or unknown fields")
        retryable = payload.get("retryable")
        if not isinstance(retryable, bool):
            raise ValueError("reconciliation error retryable must be a boolean")
        return cls(
            code=_required_string(payload, "code"),
            message=_required_string(payload, "message"),
            retryable=retryable,
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    kind: ReconciliationDecisionKind
    response: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: ReconciliationError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReconciliationDecisionKind):
            object.__setattr__(self, "kind", ReconciliationDecisionKind(self.kind))
        if self.kind is ReconciliationDecisionKind.REPLAY_SUCCESS:
            if (self.response is None) == (self.result is None):
                raise ValueError("replay_success requires exactly one response or result")
        elif self.response is not None or self.result is not None:
            raise ValueError("only replay_success accepts a response or result")
        if self.kind in {
            ReconciliationDecisionKind.RECORD_FAILURE,
            ReconciliationDecisionKind.ABORT,
        }:
            if self.error is None:
                raise ValueError(f"{self.kind.value} requires a typed error")
        elif self.error is not None:
            raise ValueError(f"{self.kind.value} does not accept an error")


@runtime_checkable
class ReconciliationProvider(Protocol):
    def reconcile(self, observation: ResumeObservation) -> ReconciliationDecision: ...


@runtime_checkable
class CheckpointExtension(Protocol):
    namespace: str
    version: str
    required: bool

    def snapshot(self) -> Any: ...

    def restore(self, state: Any) -> None: ...


@dataclass(slots=True)
class CheckpointConfig:
    store: Any | None = None
    store_ref: dict[str, str] | None = None
    key: str | None = None
    resume_policy: ResumePolicy = ResumePolicy.NEW
    ambiguous_model_policy: AmbiguousModelPolicy = AmbiguousModelPolicy.REQUIRE_RECONCILIATION
    ambiguous_tool_policy: AmbiguousToolPolicy = AmbiguousToolPolicy.REQUIRE_RECONCILIATION
    required_extension_namespaces: list[str] = field(default_factory=list)
    max_extension_state_bytes: int = DEFAULT_MAX_EXTENSION_STATE_BYTES
    credential_slots: list[str] = field(default_factory=list)
    capability_refs: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.resume_policy, ResumePolicy):
                self.resume_policy = ResumePolicy(self.resume_policy)
            if not isinstance(self.ambiguous_model_policy, AmbiguousModelPolicy):
                self.ambiguous_model_policy = AmbiguousModelPolicy(self.ambiguous_model_policy)
            if not isinstance(self.ambiguous_tool_policy, AmbiguousToolPolicy):
                self.ambiguous_tool_policy = AmbiguousToolPolicy(self.ambiguous_tool_policy)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint policy is invalid",
                code="checkpoint_policy_invalid",
            ) from exc
        if (self.store is None) == (self.store_ref is None):
            raise CheckpointError(
                "CheckpointConfig requires exactly one of store or store_ref",
                code="checkpoint_store_selection_invalid",
            )
        if self.store_ref is not None:
            try:
                _validate_capability_ref(self.store_ref, "CheckpointConfig.store_ref")
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    str(exc),
                    code="checkpoint_capability_ref_invalid",
                ) from exc
            self.store_ref = dict(self.store_ref)
        if self.key is not None:
            if not isinstance(self.key, str) or not self.key.strip():
                raise CheckpointError(
                    "CheckpointConfig.key must be a non-empty string or None",
                    code="checkpoint_key_invalid",
                )
            if len(self.key.encode("utf-8")) > MAX_CHECKPOINT_KEY_BYTES:
                raise CheckpointError(
                    f"CheckpointConfig.key must be at most {MAX_CHECKPOINT_KEY_BYTES} UTF-8 bytes",
                    code="checkpoint_key_invalid",
                )
        elif self.resume_policy is not ResumePolicy.NEW:
            raise CheckpointError(
                f"{self.resume_policy.value} requires an explicit checkpoint key",
                code="checkpoint_key_required",
            )
        if (
            isinstance(self.max_extension_state_bytes, bool)
            or not isinstance(self.max_extension_state_bytes, int)
            or not 0 <= self.max_extension_state_bytes <= MAX_WIRE_INTEGER
        ):
            raise CheckpointError(
                f"max_extension_state_bytes must be between 0 and {MAX_WIRE_INTEGER}",
                code="checkpoint_extension_limit_invalid",
            )
        normalized: list[str] = []
        for namespace in self.required_extension_namespaces:
            try:
                validate_extension_namespace(namespace)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    str(exc),
                    code="checkpoint_extension_namespace_invalid",
                ) from exc
            normalized.append(namespace)
        if len(normalized) != len(set(normalized)):
            raise CheckpointError(
                "required_extension_namespaces must be unique",
                code="checkpoint_extension_namespace_duplicate",
            )
        self.required_extension_namespaces = sorted(normalized, key=utf16_sort_key)
        if not isinstance(self.credential_slots, list) or not all(
            isinstance(pointer, str) for pointer in self.credential_slots
        ):
            raise CheckpointError(
                "CheckpointConfig.credential_slots must be an array of strings",
                code="checkpoint_credential_slots_invalid",
            )
        normalized_slots = sorted(set(self.credential_slots), key=utf16_sort_key)
        if self.credential_slots != normalized_slots:
            raise CheckpointError(
                "CheckpointConfig.credential_slots must be sorted and unique",
                code="checkpoint_credential_slots_invalid",
            )
        for pointer in normalized_slots:
            try:
                _validate_json_pointer_syntax(pointer)
            except ValueError as exc:
                raise CheckpointError(
                    str(exc),
                    code="checkpoint_credential_slots_invalid",
                ) from exc
        self.credential_slots = normalized_slots
        if not isinstance(self.capability_refs, Mapping):
            raise CheckpointError(
                "CheckpointConfig.capability_refs must be an object",
                code="checkpoint_capability_ref_invalid",
            )
        normalized_refs: dict[str, dict[str, str]] = {}
        for slot, reference in self.capability_refs.items():
            if not isinstance(slot, str) or _CAPABILITY_SLOT_RE.fullmatch(slot) is None:
                raise CheckpointError(
                    "CheckpointConfig capability slot is invalid",
                    code="checkpoint_capability_ref_invalid",
                )
            try:
                _validate_capability_ref(reference, f"CheckpointConfig.capability_refs.{slot}")
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    str(exc),
                    code="checkpoint_capability_ref_invalid",
                ) from exc
            normalized_refs[slot] = dict(reference)
        self.capability_refs = dict(
            sorted(normalized_refs.items(), key=lambda item: utf16_sort_key(item[0]))
        )
        if self.store is not None:
            required_methods = (
                "create_checkpoint_v2",
                "load_checkpoint_v2",
                "claim_checkpoint_v2",
                "progress_checkpoint_v2",
                "suspend_checkpoint_v2",
                "commit_checkpoint_v2",
                "finalize_claimed_checkpoint_v2",
                "finalize_checkpoint_v2",
                "record_event_delivery_v2",
                "renew_checkpoint_claim_v2",
                "acknowledge_terminal_v2",
                "delete_checkpoint_v2",
            )
            missing = [name for name in required_methods if not callable(getattr(self.store, name, None))]
            if missing:
                raise CheckpointError(
                    f"CheckpointConfig.store is missing methods: {', '.join(missing)}",
                    code="checkpoint_store_invalid",
                )


def validate_extension_namespace(namespace: str) -> None:
    if not isinstance(namespace, str):
        raise TypeError("checkpoint extension namespace must be a string")
    if len(namespace.encode("ascii", errors="ignore")) != len(namespace.encode("utf-8")):
        raise ValueError("checkpoint extension namespace must contain ASCII characters only")
    if len(namespace.encode("ascii")) > MAX_EXTENSION_NAMESPACE_BYTES:
        raise ValueError(
            f"checkpoint extension namespace must be at most {MAX_EXTENSION_NAMESPACE_BYTES} bytes"
        )
    if _EXTENSION_NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError("checkpoint extension namespace is invalid")


def validate_checkpoint_extension(extension: Any) -> None:
    """Validate an extension structurally without requiring inheritance."""

    try:
        namespace = extension.namespace
        version = extension.version
        required = extension.required
        snapshot = extension.snapshot
        restore = extension.restore
    except (AttributeError, TypeError) as exc:
        raise CheckpointError(
            "checkpoint extension is missing required attributes",
            code="checkpoint_extension_invalid",
        ) from exc
    try:
        validate_extension_namespace(namespace)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_extension_namespace_invalid",
        ) from exc
    if not isinstance(version, str) or not version:
        raise CheckpointError(
            "checkpoint extension version must be a non-empty string",
            code="checkpoint_extension_invalid",
        )
    if not isinstance(required, bool):
        raise CheckpointError(
            "checkpoint extension required must be a boolean",
            code="checkpoint_extension_invalid",
        )
    if not callable(snapshot) or not callable(restore):
        raise CheckpointError(
            "checkpoint extension snapshot and restore must be callable",
            code="checkpoint_extension_invalid",
        )


def canonical_json_bytes(value: Any, field_name: str = "value") -> bytes:
    return _canonical_json(value, field_name).encode("utf-8")


def canonical_json_sha256(value: Any, field_name: str = "value") -> str:
    return hashlib.sha256(canonical_json_bytes(value, field_name)).hexdigest()


def validate_run_definition(run_definition: Any) -> dict[str, Any]:
    if not isinstance(run_definition, Mapping):
        raise CheckpointError(
            "run_definition must be an object",
            code="checkpoint_definition_invalid",
        )
    definition = dict(run_definition)
    schema = definition.get("schema_version")
    if schema != RUN_DEFINITION_V1_SCHEMA:
        raise CheckpointError(
            f"unsupported run definition schema: {schema!r}",
            code="checkpoint_definition_schema_unsupported",
        )
    if set(definition) != _RUN_DEFINITION_REQUIRED_FIELDS:
        raise CheckpointError(
            "run_definition has missing or unknown top-level fields",
            code="checkpoint_definition_invalid",
        )
    credential_slots = definition.get("credential_slots")
    if not isinstance(credential_slots, list) or not all(
        isinstance(pointer, str) for pointer in credential_slots
    ):
        raise CheckpointError(
            "run_definition credential_slots must be an array of strings",
            code="checkpoint_credential_slots_invalid",
        )
    credential_slot_values: list[str] = credential_slots
    if credential_slot_values != sorted(
        set(credential_slot_values),
        key=utf16_sort_key,
    ):
        raise CheckpointError(
            "run_definition credential_slots must be sorted and unique",
            code="checkpoint_credential_slots_invalid",
        )
    for pointer in credential_slot_values:
        try:
            value = _resolve_json_pointer(definition, pointer)
        except ValueError as exc:
            code = (
                "checkpoint_credential_slots_invalid"
                if "escape" in str(exc) or "pointer" in str(exc)
                else "checkpoint_credential_slot_unresolved"
            )
            raise CheckpointError(str(exc), code=code) from exc
        if value != CREDENTIAL_REDACTION_VALUE:
            raise CheckpointError(
                f"credential slot {pointer!r} is not redacted",
                code="checkpoint_credential_value_not_redacted",
            )
    try:
        canonical_json_bytes(definition, "run_definition")
    except ValueError as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_definition_not_i_json",
        ) from exc
    return definition


def compute_run_definition_digest(run_definition: Any) -> str:
    definition = validate_run_definition(run_definition)
    return canonical_json_sha256(definition, "run_definition")


def compute_operation_request_digest(request: Any) -> str:
    if not isinstance(request, Mapping):
        raise CheckpointError(
            "operation request must be an object",
            code="operation_request_invalid",
        )
    projection = dict(request)
    if set(projection) != {"schema_version", "kind", "request"}:
        raise CheckpointError(
            "operation request has missing or unknown fields",
            code="operation_request_invalid",
        )
    if projection.get("schema_version") != OPERATION_REQUEST_V1_SCHEMA:
        raise CheckpointError(
            "operation request schema is unsupported",
            code="operation_request_schema_unsupported",
        )
    if projection.get("kind") not in {OperationKind.MODEL.value, OperationKind.TOOL.value}:
        raise CheckpointError(
            "operation request kind is invalid",
            code="operation_request_invalid",
        )
    if not isinstance(projection.get("request"), Mapping):
        raise CheckpointError(
            "operation request payload must be an object",
            code="operation_request_invalid",
        )
    request_payload = dict(projection["request"])
    if projection["kind"] == OperationKind.MODEL.value:
        required = {
            "model",
            "messages",
            "settings",
            "tools",
            "output_schema",
            "idempotency_key",
        }
    else:
        required = {
            "tool_call_id",
            "tool_name",
            "arguments",
            "idempotency_key",
        }
    if set(request_payload) != required:
        raise CheckpointError(
            "operation request payload has missing or unknown fields",
            code="operation_request_invalid",
        )
    try:
        return canonical_json_sha256(projection, "operation request")
    except ValueError as exc:
        raise CheckpointError(
            str(exc),
            code="operation_request_not_i_json",
        ) from exc


def compute_event_payload_digest(event: Any) -> str:
    if not isinstance(event, Mapping):
        raise CheckpointError(
            "checkpoint event must be an object",
            code="checkpoint_event_invalid",
        )
    try:
        return canonical_json_sha256(dict(event), "checkpoint event")
    except ValueError as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_event_not_i_json",
        ) from exc


def validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json(value: Any, field_name: str) -> str:
    try:
        return _jcs_encode(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{field_name} must be RFC 8785 I-JSON: {exc}") from exc


def _jcs_encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, int):
        if not -MAX_WIRE_INTEGER <= value <= MAX_WIRE_INTEGER:
            raise ValueError("integer is outside the I-JSON safe range")
        return str(value)
    if isinstance(value, float):
        return _jcs_float(value)
    if isinstance(value, Mapping):
        items: list[str] = []
        keys: list[str] = []
        for key in value:
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
            keys.append(key)
        for key in sorted(keys, key=utf16_sort_key):
            items.append(f"{_jcs_quote(key)}:{_jcs_encode(value[key])}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_jcs_encode(item) for item in value) + "]"
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def _jcs_quote(value: str) -> str:
    parts = ['"']
    escapes = {
        0x08: "\\b",
        0x09: "\\t",
        0x0A: "\\n",
        0x0C: "\\f",
        0x0D: "\\r",
    }
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise UnicodeError("unpaired UTF-16 surrogate")
        if character == '"':
            parts.append('\\"')
        elif character == "\\":
            parts.append("\\\\")
        elif codepoint in escapes:
            parts.append(escapes[codepoint])
        elif codepoint <= 0x1F:
            parts.append(f"\\u{codepoint:04x}")
        else:
            parts.append(character)
    parts.append('"')
    return "".join(parts)


def _jcs_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    if value == 0:
        return "0"
    negative = value < 0
    absolute = -value if negative else value
    source = repr(absolute).lower()
    if "e" in source:
        mantissa, exponent_text = source.split("e", 1)
        exponent = int(exponent_text)
        digits = mantissa.replace(".", "").rstrip("0")
        digits = digits or "0"
    else:
        exponent = 0
        digits = source

    if 1e-6 <= absolute < 1e21:
        if "e" in source:
            decimal_at = exponent + 1
            if decimal_at <= 0:
                rendered = "0." + ("0" * -decimal_at) + digits
            elif decimal_at >= len(digits):
                rendered = digits + ("0" * (decimal_at - len(digits)))
            else:
                rendered = digits[:decimal_at] + "." + digits[decimal_at:]
        else:
            rendered = source.removesuffix(".0")
    else:
        if "e" not in source:
            integer, _, fraction = source.partition(".")
            all_digits = (integer + fraction).lstrip("0")
            first_index = next(index for index, char in enumerate(source) if char not in "0.")
            dot_index = source.find(".")
            exponent = (dot_index if dot_index >= 0 else len(source)) - first_index - 1
            digits = all_digits.rstrip("0")
        mantissa = digits[0]
        if len(digits) > 1:
            mantissa += "." + digits[1:]
        sign = "+" if exponent >= 0 else ""
        rendered = f"{mantissa}e{sign}{exponent}"
    return "-" + rendered if negative else rendered


def utf16_sort_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise TypeError("object keys must be strings")
    _jcs_quote(value)
    return tuple(value.encode("utf-16-be"))


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    _validate_json_pointer_syntax(pointer)
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"credential slot {pointer!r} does not resolve")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError(f"credential slot {pointer!r} does not resolve")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"credential slot {pointer!r} does not resolve")
            current = current[index]
        else:
            raise ValueError(f"credential slot {pointer!r} does not resolve")
    return current


def _validate_json_pointer_syntax(pointer: str) -> None:
    if not pointer.startswith("/"):
        raise ValueError("credential slot must be a non-empty RFC 6901 JSON pointer")
    for raw_token in pointer[1:].split("/"):
        if "~" in _JSON_POINTER_ESCAPE_RE.sub("", raw_token):
            raise ValueError("credential slot contains an invalid RFC 6901 escape")


def _validate_capability_ref(value: Mapping[str, Any], field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if set(value) != {"id", "version"}:
        raise ValueError(f"{field_name} must contain exactly id and version")
    if not all(isinstance(value[key], str) and value[key].strip() for key in ("id", "version")):
        raise ValueError(f"{field_name} id and version must be non-empty strings")


def _positive_wire_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_WIRE_INTEGER:
        raise ValueError(f"{field_name} must be between 1 and {MAX_WIRE_INTEGER}")
    return value


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_integer(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value
