from __future__ import annotations

import hashlib
import json
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
RUN_DEFINITION_SCHEMA = "vv-agent.run-definition.v2"
OPERATION_REQUEST_SCHEMA = "vv-agent.operation-request.v1"
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
_RUN_DEFINITION_AGENT_FIELDS = frozenset({"name", "type"})
_RUN_DEFINITION_MODEL_FIELDS = frozenset({"backend", "model_id", "settings", "transport_timeout_seconds"})
_RUN_DEFINITION_RUNTIME_CONTROL_FIELDS = frozenset(
    {
        "max_cycles",
        "max_handoffs",
        "no_tool_policy",
        "session_memory_enabled",
        "memory_compact_threshold",
        "memory_threshold_percentage",
        "allow_interruption",
        "native_multimodal",
        "tool_use_behavior",
        "stop_at_tool_names",
    }
)
_RUN_DEFINITION_TOOL_FIELDS = frozenset({"schema", "tool_metadata", "timeout_seconds", "approval"})
_RUN_DEFINITION_TOOL_SCHEMA_FIELDS = frozenset({"type", "function"})
_RUN_DEFINITION_FUNCTION_SCHEMA_REQUIRED_FIELDS = frozenset({"name", "description", "parameters"})
_RUN_DEFINITION_FUNCTION_SCHEMA_FIELDS = frozenset({*_RUN_DEFINITION_FUNCTION_SCHEMA_REQUIRED_FIELDS, "strict"})
_RUN_DEFINITION_TOOL_METADATA_FIELDS = frozenset({"side_effect", "idempotency", "terminal", "capability_tags", "cost_dimensions"})
_RUN_DEFINITION_TOOL_POLICY_FIELDS = frozenset(
    {
        "allowed_tools",
        "disallowed_tools",
        "approval",
        "predicate_ref",
        "approval_timeout_seconds",
        "denied_side_effects",
        "denied_capability_tags",
        "deny_terminal_tools",
        "denied_cost_dimensions",
    }
)
_RUN_DEFINITION_CHECKPOINT_POLICY_FIELDS = frozenset(
    {"ambiguous_model_policy", "ambiguous_tool_policy", "max_extension_state_bytes"}
)
_RUN_DEFINITION_BUDGET_FIELDS = frozenset(
    {
        "max_total_tokens",
        "max_uncached_input_tokens",
        "max_tool_calls",
        "max_tool_calls_by_name",
        "max_wall_time_ms",
        "max_host_cost",
        "unavailable_metric_policy",
    }
)
_RUN_DEFINITION_EXTENSION_FIELDS = frozenset({"namespace", "version", "required"})
_RUN_DEFINITION_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
        "image_url",
        "metadata",
    }
)
_RUN_DEFINITION_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "extra_content"})
_RUN_DEFINITION_TOOL_CALL_REQUIRED_FIELDS = frozenset({"id", "type", "function"})
_RUN_DEFINITION_TOOL_CALL_FUNCTION_FIELDS = frozenset({"name", "arguments"})


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
        if self.last_event_id is not None and (not isinstance(self.last_event_id, str) or not self.last_event_id.strip()):
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
        if self.idempotency_support is not None and not isinstance(self.idempotency_support, ToolIdempotency):
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
            "idempotency_support": (self.idempotency_support.value if self.idempotency_support is not None else None),
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
                ToolIdempotency(payload["idempotency_support"]) if payload.get("idempotency_support") is not None else None
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
        if not isinstance(self.credential_slots, list) or not all(isinstance(pointer, str) for pointer in self.credential_slots):
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
        self.capability_refs = dict(sorted(normalized_refs.items(), key=lambda item: utf16_sort_key(item[0])))
        if self.store is not None:
            required_methods = (
                "create_checkpoint",
                "load_checkpoint",
                "claim_checkpoint",
                "progress_checkpoint",
                "suspend_checkpoint",
                "commit_checkpoint",
                "finalize_claimed_checkpoint",
                "finalize_checkpoint",
                "record_event_delivery",
                "renew_checkpoint_claim",
                "acknowledge_terminal",
                "delete_checkpoint",
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
        raise ValueError(f"checkpoint extension namespace must be at most {MAX_EXTENSION_NAMESPACE_BYTES} bytes")
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
    if schema != RUN_DEFINITION_SCHEMA:
        raise CheckpointError(
            f"unsupported run definition schema: {schema!r}",
            code="checkpoint_definition_schema_unsupported",
        )
    if set(definition) != _RUN_DEFINITION_REQUIRED_FIELDS:
        raise CheckpointError(
            "run_definition has missing or unknown top-level fields",
            code="checkpoint_definition_invalid",
        )
    try:
        _validate_run_definition_shape(definition)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_definition_invalid",
        ) from exc
    credential_slots = definition.get("credential_slots")
    if not isinstance(credential_slots, list) or not all(isinstance(pointer, str) for pointer in credential_slots):
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


def _validate_run_definition_shape(definition: dict[str, Any]) -> None:
    agent = _closed_definition_object(
        definition["agent"],
        _RUN_DEFINITION_AGENT_FIELDS,
        "run_definition.agent",
    )
    _non_empty_definition_string(agent["name"], "run_definition.agent.name")
    _optional_non_empty_definition_string(agent["type"], "run_definition.agent.type")
    _definition_string(definition["root_input"], "run_definition.root_input")
    _definition_string(definition["compiled_prompt"], "run_definition.compiled_prompt")

    messages = _definition_array(definition["initial_messages"], "run_definition.initial_messages")
    for index, message in enumerate(messages):
        _validate_run_definition_message(message, index=index)
    _open_definition_object(definition["initial_shared_state"], "run_definition.initial_shared_state")
    _open_definition_object(definition["run_metadata"], "run_definition.run_metadata")
    _optional_capability_ref(definition["context_ref"], "run_definition.context_ref")

    model = _closed_definition_object(
        definition["model"],
        _RUN_DEFINITION_MODEL_FIELDS,
        "run_definition.model",
    )
    _non_empty_definition_string(model["backend"], "run_definition.model.backend")
    _non_empty_definition_string(model["model_id"], "run_definition.model.model_id")
    settings = _open_definition_object(model["settings"], "run_definition.model.settings")
    if "timeout_seconds" in settings:
        raise ValueError("run_definition.model.settings must not contain transport timeout_seconds")
    from vv_agent.model_settings import ModelSettings

    parsed_settings = ModelSettings.from_dict(dict(settings))
    if parsed_settings.to_dict() != dict(settings):
        raise ValueError("run_definition.model.settings must use the complete current wire shape")
    _optional_positive_definition_number(
        model["transport_timeout_seconds"],
        "run_definition.model.transport_timeout_seconds",
    )

    controls = _closed_definition_object(
        definition["runtime_controls"],
        _RUN_DEFINITION_RUNTIME_CONTROL_FIELDS,
        "run_definition.runtime_controls",
    )
    _definition_integer(controls["max_cycles"], "run_definition.runtime_controls.max_cycles", minimum=1)
    _definition_integer(controls["max_handoffs"], "run_definition.runtime_controls.max_handoffs")
    if controls["no_tool_policy"] not in {"continue", "wait_user", "finish"}:
        raise ValueError("run_definition.runtime_controls.no_tool_policy is invalid")
    _definition_boolean(
        controls["session_memory_enabled"],
        "run_definition.runtime_controls.session_memory_enabled",
    )
    _definition_integer(
        controls["memory_compact_threshold"],
        "run_definition.runtime_controls.memory_compact_threshold",
    )
    _definition_integer(
        controls["memory_threshold_percentage"],
        "run_definition.runtime_controls.memory_threshold_percentage",
        maximum=255,
    )
    _definition_boolean(controls["allow_interruption"], "run_definition.runtime_controls.allow_interruption")
    _definition_boolean(controls["native_multimodal"], "run_definition.runtime_controls.native_multimodal")
    if controls["tool_use_behavior"] not in {
        "run_llm_again",
        "stop_on_first_tool",
        "stop_at_tool_names",
    }:
        raise ValueError("run_definition.runtime_controls.tool_use_behavior is invalid")
    _definition_string_array(
        controls["stop_at_tool_names"],
        "run_definition.runtime_controls.stop_at_tool_names",
        unique=True,
    )

    tools = _definition_array(definition["tools"], "run_definition.tools")
    for index, tool in enumerate(tools):
        _validate_run_definition_tool(tool, index=index)
    _validate_run_definition_tool_policy(definition["tool_policy"])

    checkpoint_policy = _closed_definition_object(
        definition["checkpoint_policy"],
        _RUN_DEFINITION_CHECKPOINT_POLICY_FIELDS,
        "run_definition.checkpoint_policy",
    )
    if checkpoint_policy["ambiguous_model_policy"] not in {
        "require_reconciliation",
        "retry_with_duplicate_risk",
    }:
        raise ValueError("run_definition.checkpoint_policy.ambiguous_model_policy is invalid")
    if checkpoint_policy["ambiguous_tool_policy"] not in {
        "require_reconciliation",
        "retry_idempotent_only",
    }:
        raise ValueError("run_definition.checkpoint_policy.ambiguous_tool_policy is invalid")
    _definition_integer(
        checkpoint_policy["max_extension_state_bytes"],
        "run_definition.checkpoint_policy.max_extension_state_bytes",
    )

    budget_limits = definition["budget_limits"]
    if budget_limits is not None:
        budget = _closed_definition_object(
            budget_limits,
            _RUN_DEFINITION_BUDGET_FIELDS,
            "run_definition.budget_limits",
        )
        from vv_agent.budget import RunBudgetLimits

        parsed_budget = RunBudgetLimits.from_dict(budget)
        if parsed_budget.to_dict() != dict(budget):
            raise ValueError("run_definition.budget_limits must use the complete current wire shape")

    output_schema = definition["output_schema"]
    if output_schema is not None:
        _open_definition_object(output_schema, "run_definition.output_schema")
    _optional_capability_ref(definition["workspace_ref"], "run_definition.workspace_ref")
    _optional_capability_ref(definition["session_ref"], "run_definition.session_ref")

    extensions = _definition_array(definition["extensions"], "run_definition.extensions")
    namespaces: list[str] = []
    for index, extension_value in enumerate(extensions):
        label = f"run_definition.extensions[{index}]"
        extension = _closed_definition_object(
            extension_value,
            _RUN_DEFINITION_EXTENSION_FIELDS,
            label,
        )
        namespace = _non_empty_definition_string(extension["namespace"], f"{label}.namespace")
        validate_extension_namespace(namespace)
        namespaces.append(namespace)
        _non_empty_definition_string(extension["version"], f"{label}.version")
        _definition_boolean(extension["required"], f"{label}.required")
    if namespaces != sorted(set(namespaces), key=utf16_sort_key):
        raise ValueError("run_definition.extensions must be sorted by unique namespace")

    refs = _open_definition_object(definition["capability_refs"], "run_definition.capability_refs")
    for slot, reference in refs.items():
        if _CAPABILITY_SLOT_RE.fullmatch(slot) is None:
            raise ValueError(f"run_definition.capability_refs contains invalid slot {slot!r}")
        _validate_capability_ref(reference, f"run_definition.capability_refs[{slot!r}]")


def _validate_run_definition_message(value: Any, *, index: int) -> None:
    label = f"run_definition.initial_messages[{index}]"
    message = _closed_definition_object(
        value,
        _RUN_DEFINITION_MESSAGE_FIELDS,
        label,
        required={"role", "content"},
    )
    if message["role"] not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"{label}.role is invalid")
    _definition_string(message["content"], f"{label}.content")
    for field_name in ("name", "tool_call_id", "reasoning_content", "image_url"):
        if field_name in message:
            _definition_string(message[field_name], f"{label}.{field_name}")
    if "metadata" in message:
        _open_definition_object(message["metadata"], f"{label}.metadata")
    if "tool_calls" not in message:
        return
    calls = _definition_array(message["tool_calls"], f"{label}.tool_calls")
    for call_index, value in enumerate(calls):
        call_label = f"{label}.tool_calls[{call_index}]"
        call = _closed_definition_object(
            value,
            _RUN_DEFINITION_TOOL_CALL_FIELDS,
            call_label,
            required=_RUN_DEFINITION_TOOL_CALL_REQUIRED_FIELDS,
        )
        _non_empty_definition_string(call["id"], f"{call_label}.id")
        if call["type"] != "function":
            raise ValueError(f"{call_label}.type must be function")
        function = _closed_definition_object(
            call["function"],
            _RUN_DEFINITION_TOOL_CALL_FUNCTION_FIELDS,
            f"{call_label}.function",
        )
        _non_empty_definition_string(function["name"], f"{call_label}.function.name")
        arguments = _definition_string(function["arguments"], f"{call_label}.function.arguments")
        try:
            decoded_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{call_label}.function.arguments must contain JSON") from exc
        if not isinstance(decoded_arguments, dict):
            raise ValueError(f"{call_label}.function.arguments must contain a JSON object")
        if "extra_content" in call:
            _open_definition_object(call["extra_content"], f"{call_label}.extra_content")


def _validate_run_definition_tool(value: Any, *, index: int) -> None:
    label = f"run_definition.tools[{index}]"
    tool = _closed_definition_object(value, _RUN_DEFINITION_TOOL_FIELDS, label)
    schema = _closed_definition_object(
        tool["schema"],
        _RUN_DEFINITION_TOOL_SCHEMA_FIELDS,
        f"{label}.schema",
    )
    if schema["type"] != "function":
        raise ValueError(f"{label}.schema.type must be function")
    function = _closed_definition_object(
        schema["function"],
        _RUN_DEFINITION_FUNCTION_SCHEMA_FIELDS,
        f"{label}.schema.function",
        required=_RUN_DEFINITION_FUNCTION_SCHEMA_REQUIRED_FIELDS,
    )
    _non_empty_definition_string(function["name"], f"{label}.schema.function.name")
    _definition_string(function["description"], f"{label}.schema.function.description")
    _open_definition_object(function["parameters"], f"{label}.schema.function.parameters")
    if "strict" in function:
        _definition_boolean(function["strict"], f"{label}.schema.function.strict")

    metadata_value = tool["tool_metadata"]
    if metadata_value is not None:
        metadata = _closed_definition_object(
            metadata_value,
            _RUN_DEFINITION_TOOL_METADATA_FIELDS,
            f"{label}.tool_metadata",
        )
        if metadata["side_effect"] not in {
            "unknown",
            "none",
            "read",
            "write",
            "execute",
            "network",
            "external",
        }:
            raise ValueError(f"{label}.tool_metadata.side_effect is invalid")
        if metadata["idempotency"] not in {"supported", "unsupported", "unknown"}:
            raise ValueError(f"{label}.tool_metadata.idempotency is invalid")
        _definition_boolean(metadata["terminal"], f"{label}.tool_metadata.terminal")
        _definition_string_array(
            metadata["capability_tags"],
            f"{label}.tool_metadata.capability_tags",
            sorted_unique=True,
        )
        _definition_string_array(
            metadata["cost_dimensions"],
            f"{label}.tool_metadata.cost_dimensions",
            sorted_unique=True,
        )
    _optional_positive_definition_number(tool["timeout_seconds"], f"{label}.timeout_seconds")
    approval = _open_definition_object(tool["approval"], f"{label}.approval")
    mode = approval.get("mode")
    if mode == "static":
        approval = _closed_definition_object(
            approval,
            {"mode", "required"},
            f"{label}.approval",
        )
        _definition_boolean(approval["required"], f"{label}.approval.required")
    elif mode == "referenced":
        approval = _closed_definition_object(
            approval,
            {"mode", "ref"},
            f"{label}.approval",
        )
        _validate_capability_ref(approval["ref"], f"{label}.approval.ref")
    else:
        raise ValueError(f"{label}.approval.mode is invalid")


def _validate_run_definition_tool_policy(value: Any) -> None:
    label = "run_definition.tool_policy"
    policy = _closed_definition_object(value, _RUN_DEFINITION_TOOL_POLICY_FIELDS, label)
    allowed = policy["allowed_tools"]
    if allowed is not None:
        _definition_string_array(allowed, f"{label}.allowed_tools", sorted_unique=True)
    _definition_string_array(policy["disallowed_tools"], f"{label}.disallowed_tools", sorted_unique=True)
    if policy["approval"] not in {"default", "always", "never", "on_request"}:
        raise ValueError(f"{label}.approval is invalid")
    _optional_capability_ref(policy["predicate_ref"], f"{label}.predicate_ref")
    _optional_positive_definition_number(
        policy["approval_timeout_seconds"],
        f"{label}.approval_timeout_seconds",
    )
    effects = _definition_string_array(
        policy["denied_side_effects"],
        f"{label}.denied_side_effects",
        sorted_unique=True,
    )
    if not set(effects).issubset({"unknown", "none", "read", "write", "execute", "network", "external"}):
        raise ValueError(f"{label}.denied_side_effects contains an invalid value")
    _definition_string_array(
        policy["denied_capability_tags"],
        f"{label}.denied_capability_tags",
        sorted_unique=True,
    )
    _definition_boolean(policy["deny_terminal_tools"], f"{label}.deny_terminal_tools")
    _definition_string_array(
        policy["denied_cost_dimensions"],
        f"{label}.denied_cost_dimensions",
        sorted_unique=True,
    )


def _closed_definition_object(
    value: Any,
    allowed: set[str] | frozenset[str],
    field_name: str,
    *,
    required: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    payload = _open_definition_object(value, field_name)
    required_fields = set(allowed if required is None else required)
    actual = set(payload)
    if not required_fields.issubset(actual) or not actual.issubset(allowed):
        missing = sorted(required_fields - actual)
        unknown = sorted(actual - set(allowed))
        raise ValueError(f"{field_name} has invalid fields: missing={missing}, unknown={unknown}")
    return payload


def _open_definition_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be an object with string keys")
    return value


def _definition_array(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _definition_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _non_empty_definition_string(value: Any, field_name: str) -> str:
    text = _definition_string(value, field_name)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _optional_non_empty_definition_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_definition_string(value, field_name)


def _definition_boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _definition_integer(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_WIRE_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be an integer between {minimum} and {maximum}")
    return value


def _optional_positive_definition_number(value: Any, field_name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number or null")
    return value


def _definition_string_array(
    value: Any,
    field_name: str,
    *,
    unique: bool = False,
    sorted_unique: bool = False,
) -> list[str]:
    values = _definition_array(value, field_name)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise TypeError(f"{field_name} must contain non-empty strings")
    typed_values = list(values)
    if unique and len(typed_values) != len(set(typed_values)):
        raise ValueError(f"{field_name} must contain unique values")
    if sorted_unique and typed_values != sorted(set(typed_values), key=utf16_sort_key):
        raise ValueError(f"{field_name} must be sorted and unique")
    return typed_values


def _optional_capability_ref(value: Any, field_name: str) -> None:
    if value is None:
        return
    _validate_capability_ref(value, field_name)


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
    if projection.get("schema_version") != OPERATION_REQUEST_SCHEMA:
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
