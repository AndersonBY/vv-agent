from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vv_agent.budget import RunBudgetLimits
from vv_agent.checkpoint import (
    MAX_CHECKPOINT_KEY_BYTES,
    MAX_WIRE_INTEGER,
    RUN_DEFINITION_SCHEMA,
    AmbiguousModelPolicy,
    AmbiguousToolPolicy,
    CheckpointConfig,
    ResumePolicy,
    canonical_json_bytes,
    utf16_sort_key,
    validate_extension_namespace,
)
from vv_agent.model_settings import ModelSettings
from vv_agent.run_config import ToolPolicy
from vv_agent.tools import ToolRegistry, build_default_registry
from vv_agent.tools.metadata import (
    ToolSideEffect,
    normalize_denied_side_effects,
    normalize_metadata_labels,
)
from vv_agent.types import AgentResult, AgentStatus, AgentTask, SubAgentConfig

DISTRIBUTED_RUN_SCHEMA_VERSION = "vv-agent.distributed-run.v2"
DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION = "vv-agent.distributed-worker-response.v1"
DEFAULT_TOOLSET_ID = "vv-agent.builtin-tools"
DEFAULT_TOOLSET_VERSION = "1"
DEFAULT_TOOLSET_SCHEMA_DIGEST = "24d8f7bde18b11374820f742cfa244c83666626a315e09d4b6e1b69e899a70aa"
DEFAULT_CYCLE_NAME = "vv_agent.distributed.run_single_cycle"
DEFAULT_LEASE_DURATION_MS = 5 * 60 * 1000
_MAX_U64 = (1 << 64) - 1

CapabilityKind = Literal[
    "llm_client",
    "workspace_backend",
    "approval_provider",
    "approval_broker",
    "cancellation",
    "event_sink",
    "host_cost_meter",
    "app_state",
    "memory_provider",
    "hook",
    "after_cycle_hook",
    "observer",
    "sub_task_manager",
    "tool_predicate",
    "checkpoint_store",
    "checkpoint_event_store",
    "checkpoint_extension",
    "reconciliation_provider",
]
ClaimMode = Literal["continue", "recovery"]
DistributedWorkerResponseType = Literal[
    "pending",
    "committed",
    "terminal_candidate",
    "terminal_replay",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JSON_POINTER_ESCAPE_RE = re.compile(r"~(?:0|1)")

_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "run_id",
        "task",
        "budget_limits",
        "recipe",
        "cycle_name",
        "cycle_index",
        "idempotency_key",
        "deadline_unix_ms",
        "lease_duration_ms",
        "root_run_id",
        "trace_id",
        "run_definition_schema",
        "run_definition_digest",
        "claim_mode",
        "resume_attempt",
        "checkpoint_config",
    }
)
_TASK_FIELDS = frozenset(
    {
        "task_id",
        "model",
        "system_prompt",
        "user_prompt",
        "max_cycles",
        "memory_compact_threshold",
        "memory_threshold_percentage",
        "no_tool_policy",
        "allow_interruption",
        "use_workspace",
        "agent_type",
        "native_multimodal",
        "sub_agents",
        "extra_tool_names",
        "exclude_tools",
        "model_settings",
        "initial_messages",
        "initial_shared_state",
        "metadata",
    }
)
_SUB_AGENT_FIELDS = frozenset(
    {
        "model",
        "description",
        "backend",
        "system_prompt",
        "max_cycles",
        "session_memory_enabled",
        "exclude_tools",
        "denied_side_effects",
        "denied_capability_tags",
        "deny_terminal_tools",
        "denied_cost_dimensions",
        "metadata",
    }
)
_MODEL_SETTINGS_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "response_format",
        "timeout_seconds",
        "retry",
        "extra_headers",
        "extra_body",
        "extra_args",
    }
)
_TOOL_CHOICE_FIELDS = frozenset({"type", "function"})
_TOOL_CHOICE_FUNCTION_FIELDS = frozenset({"name"})
_RESPONSE_FORMAT_JSON_SCHEMA_FIELDS = frozenset({"type", "json_schema"})
_MESSAGE_FIELDS = frozenset(
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
_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "extra_content"})
_TOOL_CALL_REQUIRED_FIELDS = frozenset({"id", "type", "function"})
_TOOL_FUNCTION_FIELDS = frozenset({"name", "arguments"})
_BUDGET_FIELDS = frozenset(
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
_HOST_COST_FIELDS = frozenset({"unit", "currency", "amount_microunits"})
_RECIPE_FIELDS = frozenset(
    {
        "settings_file",
        "backend",
        "model",
        "workspace",
        "timeout_seconds",
        "log_preview_chars",
        "capabilities",
    }
)
_CAPABILITY_FIELDS = frozenset(
    {
        "toolset_ref",
        "tool_policy",
        "llm_client_ref",
        "workspace_backend_ref",
        "approval_provider_ref",
        "approval_broker_ref",
        "approval_timeout_seconds",
        "cancellation_ref",
        "event_sink_ref",
        "host_cost_meter_ref",
        "app_state_ref",
        "sub_task_manager_ref",
        "memory_provider_refs",
        "hook_refs",
        "after_cycle_hook_refs",
        "observer_refs",
        "checkpoint_store_ref",
        "checkpoint_event_store_ref",
        "checkpoint_extension_refs",
        "reconciliation_provider_ref",
    }
)
_TOOLSET_REF_FIELDS = frozenset({"id", "version", "schema_digest"})
_TOOL_POLICY_FIELDS = frozenset(
    {
        "allowed_tools",
        "disallowed_tools",
        "approval",
        "predicate_ref",
        "denied_side_effects",
        "denied_capability_tags",
        "deny_terminal_tools",
        "denied_cost_dimensions",
    }
)
_CAPABILITY_REF_FIELDS = frozenset({"id", "version"})
_CHECKPOINT_EXTENSION_REF_FIELDS = frozenset({"namespace", "reference", "required"})
_CHECKPOINT_CONFIG_FIELDS = frozenset(
    {
        "key",
        "resume_policy",
        "ambiguous_model_policy",
        "ambiguous_tool_policy",
        "required_extension_namespaces",
        "max_extension_state_bytes",
        "credential_slots",
    }
)


class DistributedContractError(ValueError):
    """A distributed envelope or capability contract is invalid."""


class DistributedCapabilityError(RuntimeError):
    """A worker cannot resolve a declared distributed capability."""


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    if not all(isinstance(key, str) for key in payload):
        raise DistributedContractError(f"{label} must use string field names")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DistributedContractError(f"{label} fields do not match the current schema: missing={missing}, unknown={unknown}")


def _require_closed_fields(
    payload: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    label: str,
    *,
    required: set[str] | frozenset[str] = frozenset(),
) -> None:
    if not all(isinstance(key, str) for key in payload):
        raise DistributedContractError(f"{label} must use string field names")
    actual = set(payload)
    missing = set(required) - actual
    unknown = actual - set(allowed)
    if missing or unknown:
        raise DistributedContractError(
            f"{label} fields do not match the current schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _required_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DistributedContractError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise DistributedContractError(f"{label} must use string field names")
    return value


def _required_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DistributedContractError(f"{label} must be an array")
    return value


def _worker_response_integer(value: Any, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WIRE_INTEGER:
        raise DistributedContractError(f"{field_name} must be a JSON-safe unsigned integer")
    if positive and value == 0:
        raise DistributedContractError(f"{field_name} must be a positive JSON-safe integer")
    return value


@dataclass(frozen=True, slots=True)
class DistributedWorkerResponse:
    response_type: DistributedWorkerResponseType
    checkpoint_revision: int | None = None
    committed_cycle: int | None = None
    result: AgentResult | None = None

    def __post_init__(self) -> None:
        if self.response_type == "pending":
            if any(value is not None for value in (self.checkpoint_revision, self.committed_cycle, self.result)):
                raise DistributedContractError("distributed worker response fields do not match type pending")
            return
        if self.response_type == "committed":
            _worker_response_integer(self.checkpoint_revision, "checkpoint_revision")
            _worker_response_integer(self.committed_cycle, "committed_cycle", positive=True)
            if self.result is not None:
                raise DistributedContractError("distributed worker response fields do not match type committed")
            return
        if self.response_type not in {"terminal_candidate", "terminal_replay"}:
            raise DistributedContractError("unsupported distributed worker response type")
        _worker_response_integer(self.checkpoint_revision, "checkpoint_revision")
        if self.committed_cycle is not None or not isinstance(self.result, AgentResult):
            raise DistributedContractError(f"distributed worker response fields do not match type {self.response_type}")
        accepted_statuses = {
            "terminal_candidate": {
                AgentStatus.RECONCILIATION_REQUIRED,
                AgentStatus.WAIT_USER,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.MAX_CYCLES,
            },
            "terminal_replay": {
                AgentStatus.WAIT_USER,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.MAX_CYCLES,
            },
        }[self.response_type]
        if self.result.status not in accepted_statuses:
            raise DistributedContractError("distributed worker response result must be a complete current AgentResult")

    @property
    def is_terminal(self) -> bool:
        return self.response_type in {"terminal_candidate", "terminal_replay"}

    @classmethod
    def pending(cls) -> DistributedWorkerResponse:
        return cls(response_type="pending")

    @classmethod
    def committed(
        cls,
        *,
        checkpoint_revision: int,
        committed_cycle: int,
    ) -> DistributedWorkerResponse:
        return cls(
            response_type="committed",
            checkpoint_revision=checkpoint_revision,
            committed_cycle=committed_cycle,
        )

    @classmethod
    def terminal_candidate(
        cls,
        *,
        checkpoint_revision: int,
        result: AgentResult,
    ) -> DistributedWorkerResponse:
        return cls(
            response_type="terminal_candidate",
            checkpoint_revision=checkpoint_revision,
            result=result,
        )

    @classmethod
    def terminal_replay(
        cls,
        *,
        checkpoint_revision: int,
        result: AgentResult,
    ) -> DistributedWorkerResponse:
        return cls(
            response_type="terminal_replay",
            checkpoint_revision=checkpoint_revision,
            result=result,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION,
            "type": self.response_type,
        }
        if self.response_type == "committed":
            payload["checkpoint_revision"] = self.checkpoint_revision
            payload["committed_cycle"] = self.committed_cycle
        elif self.is_terminal:
            assert self.result is not None
            payload["checkpoint_revision"] = self.checkpoint_revision
            payload["result"] = self.result.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedWorkerResponse:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed worker response must be an object")
        if payload.get("schema_version") != DISTRIBUTED_WORKER_RESPONSE_SCHEMA_VERSION:
            raise DistributedContractError("unsupported distributed worker response schema_version")
        response_type = payload.get("type")
        if response_type not in {
            "pending",
            "committed",
            "terminal_candidate",
            "terminal_replay",
        }:
            raise DistributedContractError("unsupported distributed worker response type")
        expected_fields = {
            "pending": {"schema_version", "type"},
            "committed": {
                "schema_version",
                "type",
                "checkpoint_revision",
                "committed_cycle",
            },
            "terminal_candidate": {
                "schema_version",
                "type",
                "checkpoint_revision",
                "result",
            },
            "terminal_replay": {
                "schema_version",
                "type",
                "checkpoint_revision",
                "result",
            },
        }[response_type]
        if not all(isinstance(key, str) for key in payload) or set(payload) != expected_fields:
            raise DistributedContractError(f"distributed worker response fields do not match type {response_type}")
        if response_type == "pending":
            return cls.pending()
        checkpoint_revision = _worker_response_integer(
            payload["checkpoint_revision"],
            "checkpoint_revision",
        )
        if response_type == "committed":
            return cls.committed(
                checkpoint_revision=checkpoint_revision,
                committed_cycle=_worker_response_integer(
                    payload["committed_cycle"],
                    "committed_cycle",
                    positive=True,
                ),
            )
        raw_result = payload["result"]
        if not isinstance(raw_result, dict):
            raise DistributedContractError("distributed worker response result must be a complete current AgentResult")
        try:
            result = AgentResult.from_dict(raw_result)
        except (KeyError, TypeError, ValueError) as exc:
            raise DistributedContractError("distributed worker response result must be a complete current AgentResult") from exc
        if result.to_dict() != raw_result:
            raise DistributedContractError("distributed worker response result must be a complete current AgentResult")
        factory = cls.terminal_candidate if response_type == "terminal_candidate" else cls.terminal_replay
        return factory(
            checkpoint_revision=checkpoint_revision,
            result=result,
        )


def _validate_open_json(value: Any, label: str) -> None:
    def validate_wire_value(candidate: Any, path: str) -> None:
        if candidate is None or isinstance(candidate, (str, bool, int, float)):
            return
        if isinstance(candidate, list):
            for index, item in enumerate(candidate):
                validate_wire_value(item, f"{path}[{index}]")
            return
        if isinstance(candidate, Mapping):
            if not all(isinstance(key, str) for key in candidate):
                raise DistributedContractError(f"{path} must use string field names")
            for key, item in candidate.items():
                validate_wire_value(item, f"{path}[{key!r}]")
            return
        raise DistributedContractError(f"{path} contains non-JSON wire value {type(candidate).__name__}")

    validate_wire_value(value, label)
    try:
        canonical_json_bytes(value, label)
    except ValueError as exc:
        raise DistributedContractError(str(exc)) from exc


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"unsupported JSON constant {constant}")


def _validate_message_wire(value: Any, *, index: int) -> None:
    label = f"task.initial_messages[{index}]"
    message = _required_object(value, label)
    _require_closed_fields(
        message,
        _MESSAGE_FIELDS,
        label,
        required={"role", "content"},
    )
    role = message["role"]
    if role not in {"system", "user", "assistant", "tool"}:
        raise DistributedContractError(f"{label}.role is invalid")
    if not isinstance(message["content"], str):
        raise DistributedContractError(f"{label}.content must be a string")
    for field_name in ("name", "tool_call_id", "reasoning_content", "image_url"):
        if field_name in message and not isinstance(message[field_name], str):
            raise DistributedContractError(f"{label}.{field_name} must be a string")
    if "metadata" in message:
        metadata = _required_object(message["metadata"], f"{label}.metadata")
        _validate_open_json(metadata, f"{label}.metadata")
    if "tool_calls" not in message:
        return
    tool_calls = _required_array(message["tool_calls"], f"{label}.tool_calls")
    for call_index, value in enumerate(tool_calls):
        call_label = f"{label}.tool_calls[{call_index}]"
        call = _required_object(value, call_label)
        _require_closed_fields(
            call,
            _TOOL_CALL_FIELDS,
            call_label,
            required=_TOOL_CALL_REQUIRED_FIELDS,
        )
        call_id = call["id"]
        if not isinstance(call_id, str) or not call_id:
            raise DistributedContractError(f"{call_label}.id must be a non-empty string")
        if call["type"] != "function":
            raise DistributedContractError(f"{call_label}.type must be function")
        function = _required_object(call["function"], f"{call_label}.function")
        _require_exact_fields(function, _TOOL_FUNCTION_FIELDS, f"{call_label}.function")
        function_name = function["name"]
        if not isinstance(function_name, str) or not function_name:
            raise DistributedContractError(f"{call_label}.function.name must be a non-empty string")
        arguments = function["arguments"]
        if not isinstance(arguments, str):
            raise DistributedContractError(f"{call_label}.function.arguments must be a string")
        try:
            decoded_arguments = json.loads(
                arguments,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DistributedContractError(f"{call_label}.function.arguments must contain a JSON object") from exc
        if not isinstance(decoded_arguments, dict):
            raise DistributedContractError(f"{call_label}.function.arguments must contain a JSON object")
        _validate_open_json(decoded_arguments, f"{call_label}.function.arguments")
        if "extra_content" in call:
            extra_content = _required_object(call["extra_content"], f"{call_label}.extra_content")
            _validate_open_json(extra_content, f"{call_label}.extra_content")


def _validate_model_settings_wire(value: Any) -> None:
    if value is None:
        return
    settings = _required_object(value, "task.model_settings")
    _require_closed_fields(settings, _MODEL_SETTINGS_FIELDS, "task.model_settings")
    retry = settings.get("retry")
    if retry is not None:
        retry_payload = _required_object(retry, "task.model_settings.retry")
        _require_exact_fields(
            retry_payload,
            {"max_attempts", "backoff_seconds"},
            "task.model_settings.retry",
        )
    tool_choice = settings.get("tool_choice")
    if isinstance(tool_choice, Mapping):
        _require_exact_fields(
            tool_choice,
            _TOOL_CHOICE_FIELDS,
            "task.model_settings.tool_choice",
        )
        if tool_choice["type"] != "function":
            raise DistributedContractError("task.model_settings.tool_choice.type must be function")
        function = _required_object(
            tool_choice["function"],
            "task.model_settings.tool_choice.function",
        )
        _require_exact_fields(
            function,
            _TOOL_CHOICE_FUNCTION_FIELDS,
            "task.model_settings.tool_choice.function",
        )
    response_format = settings.get("response_format")
    if response_format is not None:
        response = _required_object(
            response_format,
            "task.model_settings.response_format",
        )
        response_type = response.get("type")
        if response_type in {"text", "json_object"}:
            _require_exact_fields(
                response,
                {"type"},
                "task.model_settings.response_format",
            )
        elif response_type == "json_schema":
            _require_exact_fields(
                response,
                _RESPONSE_FORMAT_JSON_SCHEMA_FIELDS,
                "task.model_settings.response_format",
            )
            schema = _required_object(
                response["json_schema"],
                "task.model_settings.response_format.json_schema",
            )
            _validate_open_json(
                schema,
                "task.model_settings.response_format.json_schema",
            )
        else:
            raise DistributedContractError("task.model_settings.response_format.type is invalid")
    for field_name in ("reasoning", "extra_headers", "extra_body", "extra_args"):
        if field_name in settings:
            extension_map = _required_object(
                settings[field_name],
                f"task.model_settings.{field_name}",
            )
            _validate_open_json(extension_map, f"task.model_settings.{field_name}")
    _validate_open_json(settings, "task.model_settings")
    try:
        decoded = ModelSettings.from_dict(dict(settings))
    except (TypeError, ValueError) as exc:
        raise DistributedContractError(f"task.model_settings is invalid: {exc}") from exc
    if decoded.to_dict() != dict(settings):
        raise DistributedContractError("task.model_settings must use the complete canonical current wire shape")


def _decode_agent_task(value: Any) -> AgentTask:
    task = _required_object(value, "task")
    _require_exact_fields(task, _TASK_FIELDS, "task")
    sub_agents = _required_object(task["sub_agents"], "task.sub_agents")
    for name, sub_agent_value in sub_agents.items():
        if not name.strip():
            raise DistributedContractError("task.sub_agents keys must be non-empty strings")
        label = f"task.sub_agents[{name!r}]"
        sub_agent = _required_object(sub_agent_value, label)
        _require_exact_fields(sub_agent, _SUB_AGENT_FIELDS, label)
        metadata = _required_object(sub_agent["metadata"], f"{label}.metadata")
        _validate_open_json(metadata, f"{label}.metadata")
        try:
            decoded_sub_agent = SubAgentConfig.from_dict(dict(sub_agent))
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(f"{label} is invalid: {exc}") from exc
        if decoded_sub_agent.to_dict() != dict(sub_agent):
            raise DistributedContractError(f"{label} must use the complete canonical current wire shape")
    _validate_model_settings_wire(task["model_settings"])
    messages = _required_array(task["initial_messages"], "task.initial_messages")
    for index, message in enumerate(messages):
        _validate_message_wire(message, index=index)
    for field_name in ("initial_shared_state", "metadata"):
        state = _required_object(task[field_name], f"task.{field_name}")
        _validate_open_json(state, f"task.{field_name}")
    try:
        decoded = AgentTask.from_dict(dict(task))
    except (TypeError, ValueError, KeyError) as exc:
        raise DistributedContractError(f"task is invalid: {exc}") from exc
    if decoded.to_dict() != dict(task):
        raise DistributedContractError("task must use the complete canonical current wire shape")
    return decoded


def _decode_budget_limits(value: Any) -> RunBudgetLimits | None:
    if value is None:
        return None
    budget = _required_object(value, "budget_limits")
    _require_exact_fields(budget, _BUDGET_FIELDS, "budget_limits")
    named_limits = _required_object(
        budget["max_tool_calls_by_name"],
        "budget_limits.max_tool_calls_by_name",
    )
    host_cost = budget["max_host_cost"]
    if host_cost is not None:
        host_cost_payload = _required_object(host_cost, "budget_limits.max_host_cost")
        _require_exact_fields(
            host_cost_payload,
            _HOST_COST_FIELDS,
            "budget_limits.max_host_cost",
        )
    try:
        decoded = RunBudgetLimits.from_dict(budget)
    except (TypeError, ValueError) as exc:
        raise DistributedContractError(f"budget_limits is invalid: {exc}") from exc
    if decoded.to_dict() != dict(budget):
        raise DistributedContractError("budget_limits must use the complete canonical current wire shape")
    _validate_open_json(named_limits, "budget_limits.max_tool_calls_by_name")
    return decoded


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DistributedContractError(f"{key} must be a non-empty string")
    return value


def _optional_ref(payload: Mapping[str, Any], key: str) -> CapabilityRef | None:
    value = payload.get(key)
    if value is None:
        return None
    return CapabilityRef.from_dict(value, field_name=key)


def _validate_json_pointer(pointer: str) -> None:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise DistributedContractError("credential slot must be a non-empty RFC 6901 JSON pointer")
    for token in pointer[1:].split("/"):
        if "~" in _JSON_POINTER_ESCAPE_RE.sub("", token):
            raise DistributedContractError("credential slot contains an invalid RFC 6901 escape")


def _validate_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DistributedContractError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    id: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise DistributedContractError("capability id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise DistributedContractError("capability version must be a non-empty string")

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version}

    @classmethod
    def from_dict(cls, payload: Any, *, field_name: str = "capability_ref") -> CapabilityRef:
        if not isinstance(payload, Mapping):
            raise DistributedContractError(f"{field_name} must be an object")
        _require_exact_fields(payload, _CAPABILITY_REF_FIELDS, field_name)
        identifier = payload.get("id")
        version = payload.get("version")
        if not isinstance(identifier, str) or not identifier.strip():
            raise DistributedContractError(f"{field_name}.id must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise DistributedContractError(f"{field_name}.version must be a non-empty string")
        return cls(
            id=identifier,
            version=version,
        )


@dataclass(frozen=True, slots=True)
class CheckpointExtensionRef:
    namespace: str
    reference: CapabilityRef
    required: bool

    def __post_init__(self) -> None:
        try:
            validate_extension_namespace(self.namespace)
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(str(exc)) from exc
        if not isinstance(self.required, bool):
            raise DistributedContractError("checkpoint extension required must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "reference": self.reference.to_dict(),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Any, *, index: int) -> CheckpointExtensionRef:
        field_name = f"checkpoint_extension_refs[{index}]"
        if not isinstance(payload, Mapping):
            raise DistributedContractError(f"{field_name} must be an object")
        _require_exact_fields(payload, _CHECKPOINT_EXTENSION_REF_FIELDS, field_name)
        namespace = payload.get("namespace")
        if not isinstance(namespace, str):
            raise DistributedContractError(f"{field_name}.namespace must be a string")
        required = payload.get("required")
        if not isinstance(required, bool):
            raise DistributedContractError(f"{field_name}.required must be a boolean")
        return cls(
            namespace=namespace,
            reference=CapabilityRef.from_dict(
                payload.get("reference"),
                field_name=f"{field_name}.reference",
            ),
            required=required,
        )


@dataclass(frozen=True, slots=True)
class DistributedCheckpointConfig:
    key: str
    resume_policy: ResumePolicy
    ambiguous_model_policy: AmbiguousModelPolicy
    ambiguous_tool_policy: AmbiguousToolPolicy
    required_extension_namespaces: tuple[str, ...] = ()
    max_extension_state_bytes: int = 262_144
    credential_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise DistributedContractError("checkpoint_config.key must be a non-empty string")
        if len(self.key.encode("utf-8")) > MAX_CHECKPOINT_KEY_BYTES:
            raise DistributedContractError(f"checkpoint_config.key must be at most {MAX_CHECKPOINT_KEY_BYTES} UTF-8 bytes")
        try:
            object.__setattr__(self, "resume_policy", ResumePolicy(self.resume_policy))
            object.__setattr__(
                self,
                "ambiguous_model_policy",
                AmbiguousModelPolicy(self.ambiguous_model_policy),
            )
            object.__setattr__(
                self,
                "ambiguous_tool_policy",
                AmbiguousToolPolicy(self.ambiguous_tool_policy),
            )
        except (TypeError, ValueError) as exc:
            raise DistributedContractError("checkpoint_config contains an unsupported policy") from exc
        if (
            isinstance(self.max_extension_state_bytes, bool)
            or not isinstance(self.max_extension_state_bytes, int)
            or not 0 <= self.max_extension_state_bytes <= MAX_WIRE_INTEGER
        ):
            raise DistributedContractError(f"max_extension_state_bytes must be between 0 and {MAX_WIRE_INTEGER}")
        namespaces = tuple(self.required_extension_namespaces)
        if namespaces != tuple(sorted(set(namespaces), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_config.required_extension_namespaces must be sorted and unique")
        for namespace in namespaces:
            try:
                validate_extension_namespace(namespace)
            except (TypeError, ValueError) as exc:
                raise DistributedContractError(str(exc)) from exc
        slots = tuple(self.credential_slots)
        if slots != tuple(sorted(set(slots), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_config.credential_slots must be sorted and unique")
        for pointer in slots:
            _validate_json_pointer(pointer)
        object.__setattr__(self, "required_extension_namespaces", namespaces)
        object.__setattr__(self, "credential_slots", slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "resume_policy": self.resume_policy.value,
            "ambiguous_model_policy": self.ambiguous_model_policy.value,
            "ambiguous_tool_policy": self.ambiguous_tool_policy.value,
            "required_extension_namespaces": list(self.required_extension_namespaces),
            "max_extension_state_bytes": self.max_extension_state_bytes,
            "credential_slots": list(self.credential_slots),
        }

    def to_runtime_config(
        self,
        *,
        store: Any,
        capability_refs: Mapping[str, Mapping[str, str]] | None = None,
    ) -> CheckpointConfig:
        return CheckpointConfig(
            store=store,
            key=self.key,
            resume_policy=self.resume_policy,
            ambiguous_model_policy=self.ambiguous_model_policy,
            ambiguous_tool_policy=self.ambiguous_tool_policy,
            required_extension_namespaces=list(self.required_extension_namespaces),
            max_extension_state_bytes=self.max_extension_state_bytes,
            credential_slots=list(self.credential_slots),
            capability_refs={key: dict(value) for key, value in (capability_refs or {}).items()},
        )

    @classmethod
    def from_checkpoint_config(
        cls,
        config: CheckpointConfig,
        *,
        require_existing: bool = True,
    ) -> DistributedCheckpointConfig:
        if config.key is None:
            raise DistributedContractError("distributed run requires an explicit checkpoint key")
        return cls(
            key=config.key,
            resume_policy=(ResumePolicy.REQUIRE_EXISTING if require_existing else config.resume_policy),
            ambiguous_model_policy=config.ambiguous_model_policy,
            ambiguous_tool_policy=config.ambiguous_tool_policy,
            required_extension_namespaces=tuple(config.required_extension_namespaces),
            max_extension_state_bytes=config.max_extension_state_bytes,
            credential_slots=tuple(config.credential_slots),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedCheckpointConfig:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("distributed run requires checkpoint_config")
        _require_exact_fields(payload, _CHECKPOINT_CONFIG_FIELDS, "checkpoint_config")

        namespaces = payload.get("required_extension_namespaces", [])
        credential_slots = payload.get("credential_slots", [])
        if not isinstance(namespaces, list):
            raise DistributedContractError("checkpoint_config.required_extension_namespaces must be an array")
        if not isinstance(credential_slots, list):
            raise DistributedContractError("checkpoint_config.credential_slots must be an array")
        decoded = cls(
            key=_required_string(payload, "key"),
            resume_policy=payload.get("resume_policy"),
            ambiguous_model_policy=payload.get("ambiguous_model_policy"),
            ambiguous_tool_policy=payload.get("ambiguous_tool_policy"),
            required_extension_namespaces=tuple(namespaces),
            max_extension_state_bytes=payload.get("max_extension_state_bytes", 262_144),
            credential_slots=tuple(credential_slots),
        )
        if decoded.to_dict() != dict(payload):
            raise DistributedContractError("checkpoint_config must use the complete canonical current wire shape")
        return decoded


@dataclass(frozen=True, slots=True)
class ToolsetRef:
    id: str = DEFAULT_TOOLSET_ID
    version: str = DEFAULT_TOOLSET_VERSION
    schema_digest: str = DEFAULT_TOOLSET_SCHEMA_DIGEST

    def __post_init__(self) -> None:
        CapabilityRef(self.id, self.version)
        if not isinstance(self.schema_digest, str) or len(self.schema_digest) != 64:
            raise DistributedContractError("toolset_ref.schema_digest must be a lowercase SHA-256 hex digest")
        if self.schema_digest != self.schema_digest.lower() or any(
            character not in "0123456789abcdef" for character in self.schema_digest
        ):
            raise DistributedContractError("toolset_ref.schema_digest must be a lowercase SHA-256 hex digest")

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version, "schema_digest": self.schema_digest}

    @classmethod
    def from_dict(cls, payload: Any) -> ToolsetRef:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("toolset_ref must be an object")
        _require_exact_fields(payload, _TOOLSET_REF_FIELDS, "toolset_ref")
        identifier = _required_string(payload, "id")
        version = _required_string(payload, "version")
        schema_digest = payload.get("schema_digest")
        if not isinstance(schema_digest, str) or not schema_digest.strip():
            raise DistributedContractError("toolset_ref.schema_digest must be a non-empty string")
        return cls(
            id=identifier,
            version=version,
            schema_digest=schema_digest,
        )


@dataclass(frozen=True, slots=True)
class DistributedToolPolicy:
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    approval: Literal["default", "always", "never", "on_request"] = "default"
    predicate_ref: CapabilityRef | None = None
    denied_side_effects: tuple[ToolSideEffect | str, ...] = ()
    denied_capability_tags: tuple[str, ...] = ()
    deny_terminal_tools: bool = False
    denied_cost_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.approval not in {"default", "always", "never", "on_request"}:
            raise DistributedContractError("tool_policy.approval is unsupported")
        for field_name, values in (
            ("tool_policy.allowed_tools", self.allowed_tools),
            ("tool_policy.disallowed_tools", self.disallowed_tools),
        ):
            if values is None:
                if field_name == "tool_policy.disallowed_tools":
                    raise DistributedContractError("tool_policy.disallowed_tools must be an array")
                continue
            if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
                raise DistributedContractError(f"{field_name} must be an array")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise DistributedContractError(f"{field_name} must contain non-empty strings")
        allowed_tools = None if self.allowed_tools is None else tuple(sorted(set(self.allowed_tools), key=utf16_sort_key))
        disallowed_tools = tuple(sorted(set(self.disallowed_tools), key=utf16_sort_key))
        try:
            denied_side_effects = tuple(value.value for value in normalize_denied_side_effects(self.denied_side_effects))
            denied_capability_tags = tuple(
                normalize_metadata_labels(
                    self.denied_capability_tags,
                    field_name="denied_capability_tags",
                )
            )
            denied_cost_dimensions = tuple(
                normalize_metadata_labels(
                    self.denied_cost_dimensions,
                    field_name="denied_cost_dimensions",
                )
            )
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(str(exc)) from exc
        if not isinstance(self.deny_terminal_tools, bool):
            raise DistributedContractError("tool_policy.deny_terminal_tools must be a boolean")
        object.__setattr__(self, "allowed_tools", allowed_tools)
        object.__setattr__(self, "disallowed_tools", disallowed_tools)
        object.__setattr__(self, "denied_side_effects", denied_side_effects)
        object.__setattr__(
            self,
            "denied_capability_tags",
            denied_capability_tags,
        )
        object.__setattr__(
            self,
            "denied_cost_dimensions",
            denied_cost_dimensions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "approval": self.approval,
            "predicate_ref": self.predicate_ref.to_dict() if self.predicate_ref is not None else None,
            "denied_side_effects": list(self.denied_side_effects),
            "denied_capability_tags": list(self.denied_capability_tags),
            "deny_terminal_tools": self.deny_terminal_tools,
            "denied_cost_dimensions": list(self.denied_cost_dimensions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedToolPolicy:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("tool_policy must be an object")
        _require_exact_fields(payload, _TOOL_POLICY_FIELDS, "tool_policy")
        allowed = payload.get("allowed_tools")
        disallowed = payload.get("disallowed_tools")
        if allowed is not None and not isinstance(allowed, list):
            raise DistributedContractError("tool_policy.allowed_tools must be an array or null")
        if not isinstance(disallowed, list):
            raise DistributedContractError("tool_policy.disallowed_tools must be an array")
        denied_side_effects = payload.get("denied_side_effects")
        denied_capability_tags = payload.get("denied_capability_tags")
        denied_cost_dimensions = payload.get("denied_cost_dimensions")
        for field_name, values in (
            ("denied_side_effects", denied_side_effects),
            ("denied_capability_tags", denied_capability_tags),
            ("denied_cost_dimensions", denied_cost_dimensions),
        ):
            if not isinstance(values, list):
                raise DistributedContractError(f"tool_policy.{field_name} must be an array")
        approval = payload.get("approval")
        if not isinstance(approval, str):
            raise DistributedContractError("tool_policy.approval must be a string")
        predicate = payload.get("predicate_ref")
        decoded = cls(
            allowed_tools=tuple(allowed) if allowed is not None else None,
            disallowed_tools=tuple(disallowed),
            approval=approval,  # type: ignore[arg-type]
            predicate_ref=(
                CapabilityRef.from_dict(predicate, field_name="tool_policy.predicate_ref") if predicate is not None else None
            ),
            denied_side_effects=tuple(denied_side_effects),
            denied_capability_tags=tuple(denied_capability_tags),
            deny_terminal_tools=payload.get("deny_terminal_tools"),
            denied_cost_dimensions=tuple(denied_cost_dimensions),
        )
        if decoded.to_dict() != dict(payload):
            raise DistributedContractError("tool_policy must use the complete canonical current wire shape")
        return decoded

    def resolve(self, registry: DistributedCapabilityRegistry) -> ToolPolicy:
        predicate = None
        if self.predicate_ref is not None:
            predicate = registry.resolve("tool_predicate", self.predicate_ref)
            if not callable(predicate):
                raise DistributedCapabilityError(
                    f"distributed capability tool_predicate {self.predicate_ref.id}@{self.predicate_ref.version} "
                    "did not resolve to a callable"
                )
        return ToolPolicy(
            allowed_tools=list(self.allowed_tools) if self.allowed_tools is not None else None,
            disallowed_tools=list(self.disallowed_tools),
            approval=self.approval,
            can_use_tool=predicate,
            denied_side_effects=list(self.denied_side_effects),
            denied_capability_tags=list(self.denied_capability_tags),
            deny_terminal_tools=self.deny_terminal_tools,
            denied_cost_dimensions=list(self.denied_cost_dimensions),
        )


def _policy_with_task_metadata_denials(
    policy: DistributedToolPolicy,
    task: AgentTask,
) -> DistributedToolPolicy:
    metadata = getattr(task, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise DistributedContractError("task.metadata must be an object")

    def values(key: str) -> tuple[Any, ...]:
        if key not in metadata:
            return ()
        value = metadata[key]
        if not isinstance(value, list):
            raise DistributedContractError(f"task.metadata.{key} must be an array")
        return tuple(value)

    deny_terminal_tools = policy.deny_terminal_tools
    terminal_key = "_vv_agent_deny_terminal_tools"
    if terminal_key in metadata:
        terminal_value = metadata[terminal_key]
        if not isinstance(terminal_value, bool):
            raise DistributedContractError(f"task.metadata.{terminal_key} must be a boolean")
        deny_terminal_tools = deny_terminal_tools or terminal_value

    return replace(
        policy,
        denied_side_effects=(*policy.denied_side_effects, *values("_vv_agent_denied_side_effects")),
        denied_capability_tags=(
            *policy.denied_capability_tags,
            *values("_vv_agent_denied_capability_tags"),
        ),
        deny_terminal_tools=deny_terminal_tools,
        denied_cost_dimensions=(
            *policy.denied_cost_dimensions,
            *values("_vv_agent_denied_cost_dimensions"),
        ),
    )


def _recipe_with_task_metadata_denials(
    recipe: RuntimeRecipe,
    task: AgentTask,
) -> RuntimeRecipe:
    capabilities = replace(
        recipe.capabilities,
        tool_policy=_policy_with_task_metadata_denials(recipe.capabilities.tool_policy, task),
    )
    return replace(recipe, capabilities=capabilities)


@dataclass(frozen=True, slots=True)
class DistributedCapabilities:
    toolset_ref: ToolsetRef = field(default_factory=ToolsetRef)
    tool_policy: DistributedToolPolicy = field(default_factory=DistributedToolPolicy)
    llm_client_ref: CapabilityRef | None = None
    workspace_backend_ref: CapabilityRef | None = None
    approval_provider_ref: CapabilityRef | None = None
    approval_broker_ref: CapabilityRef | None = None
    approval_timeout_seconds: float | None = None
    cancellation_ref: CapabilityRef | None = None
    event_sink_ref: CapabilityRef | None = None
    host_cost_meter_ref: CapabilityRef | None = None
    app_state_ref: CapabilityRef | None = None
    sub_task_manager_ref: CapabilityRef | None = None
    memory_provider_refs: tuple[CapabilityRef, ...] = ()
    hook_refs: tuple[CapabilityRef, ...] = ()
    after_cycle_hook_refs: tuple[CapabilityRef, ...] = ()
    observer_refs: tuple[CapabilityRef, ...] = ()
    checkpoint_store_ref: CapabilityRef | None = None
    checkpoint_event_store_ref: CapabilityRef | None = None
    checkpoint_extension_refs: tuple[CheckpointExtensionRef, ...] = ()
    reconciliation_provider_ref: CapabilityRef | None = None

    def __post_init__(self) -> None:
        if (self.approval_provider_ref is None) != (self.approval_broker_ref is None):
            raise DistributedContractError("approval_provider_ref and approval_broker_ref must be declared together")
        if self.approval_timeout_seconds is not None and (
            isinstance(self.approval_timeout_seconds, bool)
            or not isinstance(self.approval_timeout_seconds, (int, float))
            or not math.isfinite(float(self.approval_timeout_seconds))
            or self.approval_timeout_seconds <= 0
        ):
            raise DistributedContractError("approval_timeout_seconds must be a finite positive number or null")
        namespaces = tuple(reference.namespace for reference in self.checkpoint_extension_refs)
        if namespaces != tuple(sorted(set(namespaces), key=utf16_sort_key)):
            raise DistributedContractError("checkpoint_extension_refs must be sorted by unique namespace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolset_ref": self.toolset_ref.to_dict(),
            "tool_policy": self.tool_policy.to_dict(),
            "llm_client_ref": self.llm_client_ref.to_dict() if self.llm_client_ref else None,
            "workspace_backend_ref": self.workspace_backend_ref.to_dict() if self.workspace_backend_ref else None,
            "approval_provider_ref": self.approval_provider_ref.to_dict() if self.approval_provider_ref else None,
            "approval_broker_ref": self.approval_broker_ref.to_dict() if self.approval_broker_ref else None,
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "cancellation_ref": self.cancellation_ref.to_dict() if self.cancellation_ref else None,
            "event_sink_ref": self.event_sink_ref.to_dict() if self.event_sink_ref else None,
            "host_cost_meter_ref": self.host_cost_meter_ref.to_dict() if self.host_cost_meter_ref else None,
            "app_state_ref": self.app_state_ref.to_dict() if self.app_state_ref else None,
            "sub_task_manager_ref": self.sub_task_manager_ref.to_dict() if self.sub_task_manager_ref else None,
            "memory_provider_refs": [reference.to_dict() for reference in self.memory_provider_refs],
            "hook_refs": [reference.to_dict() for reference in self.hook_refs],
            "after_cycle_hook_refs": [reference.to_dict() for reference in self.after_cycle_hook_refs],
            "observer_refs": [reference.to_dict() for reference in self.observer_refs],
            "checkpoint_store_ref": (self.checkpoint_store_ref.to_dict() if self.checkpoint_store_ref is not None else None),
            "checkpoint_event_store_ref": (
                self.checkpoint_event_store_ref.to_dict() if self.checkpoint_event_store_ref is not None else None
            ),
            "checkpoint_extension_refs": [reference.to_dict() for reference in self.checkpoint_extension_refs],
            "reconciliation_provider_ref": (
                self.reconciliation_provider_ref.to_dict() if self.reconciliation_provider_ref is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedCapabilities:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("capabilities must be an object")
        _require_exact_fields(payload, _CAPABILITY_FIELDS, "capabilities")

        def refs(key: str) -> tuple[CapabilityRef, ...]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                raise DistributedContractError(f"capabilities.{key} must be an array")
            return tuple(
                CapabilityRef.from_dict(value, field_name=f"capabilities.{key}[{index}]") for index, value in enumerate(values)
            )

        checkpoint_extension_values = payload.get("checkpoint_extension_refs", [])
        if not isinstance(checkpoint_extension_values, list):
            raise DistributedContractError("capabilities.checkpoint_extension_refs must be an array")

        decoded = cls(
            toolset_ref=ToolsetRef.from_dict(payload.get("toolset_ref")),
            tool_policy=DistributedToolPolicy.from_dict(payload.get("tool_policy")),
            llm_client_ref=_optional_ref(payload, "llm_client_ref"),
            workspace_backend_ref=_optional_ref(payload, "workspace_backend_ref"),
            approval_provider_ref=_optional_ref(payload, "approval_provider_ref"),
            approval_broker_ref=_optional_ref(payload, "approval_broker_ref"),
            approval_timeout_seconds=payload.get("approval_timeout_seconds"),
            cancellation_ref=_optional_ref(payload, "cancellation_ref"),
            event_sink_ref=_optional_ref(payload, "event_sink_ref"),
            host_cost_meter_ref=_optional_ref(payload, "host_cost_meter_ref"),
            app_state_ref=_optional_ref(payload, "app_state_ref"),
            sub_task_manager_ref=_optional_ref(payload, "sub_task_manager_ref"),
            memory_provider_refs=refs("memory_provider_refs"),
            hook_refs=refs("hook_refs"),
            after_cycle_hook_refs=refs("after_cycle_hook_refs"),
            observer_refs=refs("observer_refs"),
            checkpoint_store_ref=_optional_ref(payload, "checkpoint_store_ref"),
            checkpoint_event_store_ref=_optional_ref(payload, "checkpoint_event_store_ref"),
            checkpoint_extension_refs=tuple(
                CheckpointExtensionRef.from_dict(value, index=index) for index, value in enumerate(checkpoint_extension_values)
            ),
            reconciliation_provider_ref=_optional_ref(payload, "reconciliation_provider_ref"),
        )
        if decoded.to_dict() != dict(payload):
            raise DistributedContractError("capabilities must use the complete canonical current wire shape")
        return decoded


@dataclass(slots=True)
class RuntimeRecipe:
    settings_file: str
    backend: str
    model: str
    workspace: str
    timeout_seconds: float = 90.0
    log_preview_chars: int | None = None
    capabilities: DistributedCapabilities = field(default_factory=DistributedCapabilities)

    def __post_init__(self) -> None:
        for field_name in ("settings_file", "backend", "model", "workspace"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"runtime_recipe.{field_name} must be a non-empty string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise DistributedContractError("runtime_recipe.timeout_seconds must be a finite positive number")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise DistributedContractError("runtime_recipe.timeout_seconds must be a finite positive number")
        if self.log_preview_chars is not None and (
            isinstance(self.log_preview_chars, bool) or not isinstance(self.log_preview_chars, int) or self.log_preview_chars < 0
        ):
            raise DistributedContractError("runtime_recipe.log_preview_chars must be a non-negative integer or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_file": self.settings_file,
            "backend": self.backend,
            "model": self.model,
            "workspace": self.workspace,
            "timeout_seconds": float(self.timeout_seconds),
            "log_preview_chars": self.log_preview_chars,
            "capabilities": self.capabilities.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RuntimeRecipe:
        if not isinstance(payload, Mapping):
            raise DistributedContractError("runtime_recipe must be an object")
        _require_exact_fields(payload, _RECIPE_FIELDS, "runtime_recipe")
        timeout = payload.get("timeout_seconds")
        log_preview_chars = payload.get("log_preview_chars")
        decoded = cls(
            settings_file=_required_string(payload, "settings_file"),
            backend=_required_string(payload, "backend"),
            model=_required_string(payload, "model"),
            workspace=_required_string(payload, "workspace"),
            timeout_seconds=timeout,
            log_preview_chars=log_preview_chars,
            capabilities=DistributedCapabilities.from_dict(payload.get("capabilities")),
        )
        if decoded.to_dict() != dict(payload):
            raise DistributedContractError("runtime_recipe must use the complete canonical current wire shape")
        return decoded


def _decode_current_envelope_components(
    payload: Any,
) -> tuple[AgentTask, RunBudgetLimits | None, RuntimeRecipe, DistributedCheckpointConfig]:
    if not isinstance(payload, Mapping):
        raise DistributedContractError("distributed envelope must be an object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        raise DistributedContractError("distributed envelope schema_version must be a string")
    if schema_version != DISTRIBUTED_RUN_SCHEMA_VERSION:
        raise DistributedContractError(f"unsupported distributed schema_version: {schema_version}")
    if payload.get("run_definition_schema") != RUN_DEFINITION_SCHEMA:
        raise DistributedContractError("checkpoint_definition_schema_unsupported")
    _require_exact_fields(payload, _ENVELOPE_FIELDS, "distributed envelope")
    return (
        _decode_agent_task(payload["task"]),
        _decode_budget_limits(payload["budget_limits"]),
        RuntimeRecipe.from_dict(payload["recipe"]),
        DistributedCheckpointConfig.from_dict(payload["checkpoint_config"]),
    )


@dataclass(frozen=True, slots=True)
class DistributedRunEnvelope:
    job_id: str
    run_id: str
    task: AgentTask
    recipe: RuntimeRecipe
    cycle_name: str
    cycle_index: int
    idempotency_key: str
    deadline_unix_ms: int | None
    root_run_id: str
    trace_id: str
    run_definition_digest: str
    claim_mode: ClaimMode
    resume_attempt: int
    checkpoint_config: DistributedCheckpointConfig
    budget_limits: RunBudgetLimits | None = None
    lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS
    schema_version: str = DISTRIBUTED_RUN_SCHEMA_VERSION
    run_definition_schema: str = RUN_DEFINITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DISTRIBUTED_RUN_SCHEMA_VERSION:
            raise DistributedContractError(f"unsupported distributed schema_version: {self.schema_version}")
        for field_name in ("job_id", "run_id", "cycle_name", "idempotency_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"distributed envelope {field_name} must be a non-empty string")
        if isinstance(self.cycle_index, bool) or not isinstance(self.cycle_index, int) or not 1 <= self.cycle_index < 1 << 32:
            raise DistributedContractError("distributed envelope cycle_index must be between 1 and 4294967295")
        if self.deadline_unix_ms is not None and (
            isinstance(self.deadline_unix_ms, bool)
            or not isinstance(self.deadline_unix_ms, int)
            or self.deadline_unix_ms < 0
            or self.deadline_unix_ms > _MAX_U64
        ):
            raise DistributedContractError("distributed envelope deadline_unix_ms must be a non-negative integer or null")
        if (
            isinstance(self.lease_duration_ms, bool)
            or not isinstance(self.lease_duration_ms, int)
            or self.lease_duration_ms <= 0
            or self.lease_duration_ms > _MAX_U64
        ):
            raise DistributedContractError("distributed envelope lease_duration_ms must be a positive integer")
        if self.budget_limits is not None and not isinstance(self.budget_limits, RunBudgetLimits):
            raise DistributedContractError("distributed envelope budget_limits must be an object or null")
        if not isinstance(self.task, AgentTask):
            raise DistributedContractError("distributed envelope task must be an AgentTask")
        if not isinstance(self.recipe, RuntimeRecipe):
            raise DistributedContractError("distributed envelope recipe must be a RuntimeRecipe")
        if self.run_definition_schema != RUN_DEFINITION_SCHEMA:
            raise DistributedContractError("checkpoint_definition_schema_unsupported")
        for field_name in ("root_run_id", "trace_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistributedContractError(f"distributed {field_name} must be a non-empty string")
        _validate_sha256(self.run_definition_digest, "run_definition_digest")
        if self.claim_mode not in {"continue", "recovery"}:
            raise DistributedContractError("checkpoint_claim_mode_invalid")
        if (
            isinstance(self.resume_attempt, bool)
            or not isinstance(self.resume_attempt, int)
            or not 1 <= self.resume_attempt <= MAX_WIRE_INTEGER
        ):
            raise DistributedContractError(f"resume_attempt must be between 1 and {MAX_WIRE_INTEGER}")
        if not isinstance(self.checkpoint_config, DistributedCheckpointConfig):
            raise DistributedContractError("distributed run requires checkpoint_config")
        capabilities = self.recipe.capabilities
        if capabilities.checkpoint_store_ref is None:
            raise DistributedContractError("distributed run requires checkpoint_store_ref")
        extensions_by_namespace = {reference.namespace: reference for reference in capabilities.checkpoint_extension_refs}
        for namespace in self.checkpoint_config.required_extension_namespaces:
            reference = extensions_by_namespace.get(namespace)
            if reference is None or not reference.required:
                raise DistributedContractError(f"required checkpoint extension {namespace} is unavailable")

    @classmethod
    def for_cycle(
        cls,
        *,
        task: AgentTask,
        recipe: RuntimeRecipe,
        cycle_index: int,
        root_run_id: str,
        trace_id: str,
        run_definition_digest: str,
        claim_mode: ClaimMode,
        resume_attempt: int,
        checkpoint_config: DistributedCheckpointConfig,
        cycle_name: str = DEFAULT_CYCLE_NAME,
        run_id: str | None = None,
        deadline_unix_ms: int | None = None,
        lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS,
        budget_limits: RunBudgetLimits | None = None,
    ) -> DistributedRunEnvelope:
        recipe = _recipe_with_task_metadata_denials(recipe, task)
        effective_run_id = run_id or str(task.metadata.get("_vv_agent_run_id") or root_run_id)
        idempotency_key = f"{effective_run_id}:cycle:{cycle_index}"
        return cls(
            job_id=idempotency_key,
            run_id=effective_run_id,
            task=task,
            budget_limits=budget_limits,
            recipe=recipe,
            cycle_name=cycle_name,
            cycle_index=cycle_index,
            idempotency_key=idempotency_key,
            deadline_unix_ms=deadline_unix_ms,
            lease_duration_ms=lease_duration_ms,
            root_run_id=root_run_id,
            trace_id=trace_id,
            run_definition_schema=RUN_DEFINITION_SCHEMA,
            run_definition_digest=run_definition_digest,
            claim_mode=claim_mode,
            resume_attempt=resume_attempt,
            checkpoint_config=checkpoint_config,
        )

    def remaining_seconds(self, *, now_ms: int | None = None) -> float | None:
        if self.deadline_unix_ms is None:
            return None
        current = time.time_ns() // 1_000_000 if now_ms is None else now_ms
        return max(0.0, (self.deadline_unix_ms - current) / 1000)

    def ensure_not_expired(self, *, now_ms: int | None = None) -> None:
        remaining = self.remaining_seconds(now_ms=now_ms)
        if remaining is not None and remaining <= 0:
            raise DistributedContractError(f"distributed job {self.job_id} deadline has expired")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "task": self.task.to_dict(),
            "budget_limits": self.budget_limits.to_dict() if self.budget_limits is not None else None,
            "recipe": self.recipe.to_dict(),
            "cycle_name": self.cycle_name,
            "cycle_index": self.cycle_index,
            "idempotency_key": self.idempotency_key,
            "deadline_unix_ms": self.deadline_unix_ms,
            "lease_duration_ms": self.lease_duration_ms,
            "root_run_id": self.root_run_id,
            "trace_id": self.trace_id,
            "run_definition_schema": self.run_definition_schema,
            "run_definition_digest": self.run_definition_digest,
            "claim_mode": self.claim_mode,
            "resume_attempt": self.resume_attempt,
            "checkpoint_config": self.checkpoint_config.to_dict(),
        }
        _decode_current_envelope_components(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> DistributedRunEnvelope:
        task, budget_limits, recipe, checkpoint_config = _decode_current_envelope_components(payload)
        assert isinstance(payload, Mapping)
        decoded = cls(
            schema_version=DISTRIBUTED_RUN_SCHEMA_VERSION,
            job_id=_required_string(payload, "job_id"),
            run_id=_required_string(payload, "run_id"),
            task=task,
            budget_limits=budget_limits,
            recipe=recipe,
            cycle_name=_required_string(payload, "cycle_name"),
            cycle_index=payload.get("cycle_index"),
            idempotency_key=_required_string(payload, "idempotency_key"),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            lease_duration_ms=payload.get("lease_duration_ms"),
            root_run_id=payload.get("root_run_id"),
            trace_id=payload.get("trace_id"),
            run_definition_schema=payload.get("run_definition_schema"),
            run_definition_digest=payload.get("run_definition_digest"),
            claim_mode=payload.get("claim_mode"),
            resume_attempt=payload.get("resume_attempt"),
            checkpoint_config=checkpoint_config,
        )
        if decoded.to_dict() != dict(payload):
            raise DistributedContractError("distributed envelope must use the complete canonical current wire shape")
        return decoded


def toolset_schema_digest(registry: ToolRegistry) -> str:
    canonical = json.dumps(
        registry.list_openai_schemas(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class DistributedCapabilityRegistry:
    """Worker-local registry for resolving language-neutral capability references."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._capabilities: dict[tuple[str, str, str], Any] = {}
        self._toolsets: dict[tuple[str, str], ToolRegistry] = {}
        if include_defaults:
            self.register_toolset(ToolsetRef(), build_default_registry())

    def register_toolset(self, reference: ToolsetRef, registry: ToolRegistry) -> None:
        actual = toolset_schema_digest(registry)
        if actual != reference.schema_digest:
            raise DistributedCapabilityError(
                f"toolset {reference.id}@{reference.version} schema digest mismatch: "
                f"expected {reference.schema_digest}, got {actual}"
            )
        self._toolsets[reference.key] = registry

    def register(self, kind: CapabilityKind, reference: CapabilityRef, capability: Any) -> None:
        self._capabilities[(kind, *reference.key)] = capability

    def resolve(self, kind: CapabilityKind, reference: CapabilityRef) -> Any:
        key = (kind, *reference.key)
        if key not in self._capabilities:
            raise DistributedCapabilityError(f"unknown distributed capability {kind} {reference.id}@{reference.version}")
        return self._capabilities[key]

    def resolve_toolset(self, reference: ToolsetRef) -> ToolRegistry:
        registry = self._toolsets.get(reference.key)
        if registry is None:
            raise DistributedCapabilityError(f"unknown distributed toolset {reference.id}@{reference.version}")
        actual = toolset_schema_digest(registry)
        if actual != reference.schema_digest:
            raise DistributedCapabilityError(
                f"toolset {reference.id}@{reference.version} schema digest mismatch: "
                f"expected {reference.schema_digest}, got {actual}"
            )
        return registry

    def validate(
        self,
        capabilities: DistributedCapabilities,
    ) -> None:
        self.resolve_toolset(capabilities.toolset_ref)
        capabilities.tool_policy.resolve(self)
        for kind, reference in (
            ("llm_client", capabilities.llm_client_ref),
            ("workspace_backend", capabilities.workspace_backend_ref),
            ("approval_provider", capabilities.approval_provider_ref),
            ("approval_broker", capabilities.approval_broker_ref),
            ("cancellation", capabilities.cancellation_ref),
            ("event_sink", capabilities.event_sink_ref),
            ("host_cost_meter", capabilities.host_cost_meter_ref),
            ("app_state", capabilities.app_state_ref),
            ("sub_task_manager", capabilities.sub_task_manager_ref),
            ("checkpoint_store", capabilities.checkpoint_store_ref),
            ("checkpoint_event_store", capabilities.checkpoint_event_store_ref),
            ("reconciliation_provider", capabilities.reconciliation_provider_ref),
        ):
            if reference is not None:
                self.resolve(kind, reference)  # type: ignore[arg-type]
        for kind, references in (
            ("memory_provider", capabilities.memory_provider_refs),
            ("hook", capabilities.hook_refs),
            ("after_cycle_hook", capabilities.after_cycle_hook_refs),
            ("observer", capabilities.observer_refs),
        ):
            for reference in references:
                self.resolve(kind, reference)  # type: ignore[arg-type]
        for extension in capabilities.checkpoint_extension_refs:
            self.resolve("checkpoint_extension", extension.reference)
        if capabilities.checkpoint_store_ref is None:
            raise DistributedCapabilityError("distributed run requires checkpoint_store_ref")
