"""Shared run-event wire and retained-event validation without event construction."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Any, Literal, cast

from vv_agent.kernel._budget_values import BudgetEnforcementBoundary, BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.kernel._execution_values import CompletionReason, ModelCallOperation, TokenUsage
from vv_agent.kernel._tool_metadata import ToolMetadata


class OperationKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


ApprovalAction = Literal["allow", "allow_session", "deny", "timeout"]


MemoryCompactTrigger = Literal["micro_threshold", "full_threshold", "prompt_too_long"]


MemoryCompactMode = Literal["none", "micro", "structural", "summary", "emergency"]


ReservedOutputSource = Literal[
    "model_settings",
    "task_metadata",
    "framework_fallback",
    "framework_fallback_capped_by_model_capability",
]


DiagnosticLevel = Literal["debug", "info", "warning", "error"]


_APPROVAL_ACTIONS = frozenset({"allow", "allow_session", "deny", "timeout"})


_MEMORY_COMPACT_TRIGGER_VALUES = frozenset({"micro_threshold", "full_threshold", "prompt_too_long"})


_MEMORY_COMPACT_MODE_VALUES = frozenset({"none", "micro", "structural", "summary", "emergency"})


_RESERVED_OUTPUT_SOURCE_VALUES = frozenset(
    {
        "model_settings",
        "task_metadata",
        "framework_fallback",
        "framework_fallback_capped_by_model_capability",
    }
)


_DIAGNOSTIC_LEVEL_VALUES = frozenset({"debug", "info", "warning", "error"})


_MODEL_CALL_OUTCOME_VALUES = frozenset({"definitive", "ambiguous"})


_MEMORY_COMPACT_STARTED_FIELDS = (
    "trigger",
    "configured_threshold",
    "effective_threshold",
    "microcompact_threshold",
    "microcompact_target",
    "candidate_count",
    "estimated_reclaimable_tokens",
    "model_context_window",
    "model_max_output_tokens",
    "reserved_output_tokens",
    "reserved_output_source",
    "autocompact_buffer_tokens",
)


_MEMORY_COMPACT_COMPLETED_FIELDS = (
    "mode",
    "changed",
    "archived_count",
    "reclaimed_tokens",
    "artifact_failure_count",
)


_JSON_SAFE_INTEGER_MAX = (1 << 53) - 1


_TOOL_STATUS_VALUES = frozenset({"success", "error", "wait_response", "running", "pending_compress"})


_TOOL_DIRECTIVE_VALUES = frozenset({"continue", "finish", "wait_user"})


_COMMON_EVENT_FIELDS = frozenset(
    {
        "version",
        "type",
        "event_id",
        "run_id",
        "trace_id",
        "created_at",
        "session_id",
        "parent_event_id",
        "parent_run_id",
        "cycle_index",
        "agent_name",
        "metadata",
    }
)


_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "run_started": frozenset({"input"}),
    "agent_started": frozenset(),
    "cycle_started": frozenset(),
    "cycle_aborted": frozenset({"logical_cycle", "reason"}),
    "model_call_started": frozenset({"call_id", "operation_id", "attempt", "operation", "backend", "model"}),
    "model_call_completed": frozenset({"call_id", "operation_id", "attempt", "operation", "backend", "model", "usage"}),
    "model_call_failed": frozenset(
        {
            "call_id",
            "operation_id",
            "attempt",
            "operation",
            "backend",
            "model",
            "outcome",
            "usage",
            "error_code",
        }
    ),
    "run_state_changed": frozenset({"state", "cancel_requested"}),
    "host_interaction_requested": frozenset(
        {
            "interaction_id",
            "logical_cycle",
            "operation_id",
            "tool_call_id",
            "request_digest",
            "prompt",
        }
    ),
    "host_interaction_response_consumed": frozenset(
        {
            "interaction_id",
            "logical_cycle",
            "operation_id",
            "tool_call_id",
            "request_digest",
            "command_id",
            "response_digest",
        }
    ),
    "diagnostic": frozenset({"level", "code", "details"}),
    "memory_compact_started": frozenset(
        {
            "message_count",
            "estimated_tokens",
            *_MEMORY_COMPACT_STARTED_FIELDS,
        }
    ),
    "memory_compact_completed": frozenset({"before_count", "after_count", "summary_tokens", *_MEMORY_COMPACT_COMPLETED_FIELDS}),
    "assistant_delta": frozenset({"delta", "content_chars", "estimated_tokens"}),
    "reasoning_delta": frozenset({"delta", "reasoning_chars", "estimated_tokens"}),
    "model_tool_call_started": frozenset({"tool_call_id", "tool_name", "tool_call_index", "arguments_chars", "estimated_tokens"}),
    "model_tool_call_progress": frozenset(
        {"tool_call_id", "tool_name", "tool_call_index", "arguments_chars", "estimated_tokens"}
    ),
    "tool_call_planned": frozenset({"tool_name", "tool_call_id", "arguments", "tool_metadata"}),
    "tool_call_started": frozenset({"tool_name", "tool_call_id", "arguments", "tool_metadata"}),
    "tool_call_completed": frozenset(
        {
            "tool_name",
            "tool_call_id",
            "status",
            "directive",
            "error_code",
            "execution_started",
            "duration_ms",
            "tool_metadata",
            "operation_id",
            "attempt",
        }
    ),
    "approval_requested": frozenset({"request_id", "tool_name", "tool_call_id", "message"}),
    "approval_resolved": frozenset({"request_id", "tool_name", "tool_call_id", "action"}),
    "sub_run_started": frozenset({"parent_tool_call_id", "status", "child_session_id", "task_id"}),
    "sub_run_completed": frozenset(
        {
            "parent_tool_call_id",
            "status",
            "child_session_id",
            "task_id",
            "final_output",
            "wait_reason",
            "error",
            "completion_reason",
            "completion_tool_name",
            "partial_output",
            "token_usage",
            "budget_usage",
            "budget_exhaustion",
        }
    ),
    "handoff_started": frozenset({"source_agent", "target_agent", "tool_call_id", "status", "child_session_id"}),
    "handoff_completed": frozenset(
        {"source_agent", "target_agent", "tool_call_id", "status", "child_session_id", "child_run_id"}
    ),
    "run_completed": frozenset(
        {
            "final_output",
            "status",
            "completion_reason",
            "completion_tool_name",
            "partial_output",
            "budget_usage",
            "budget_exhaustion",
        }
    ),
    "run_failed": frozenset(
        {
            "error",
            "status",
            "completion_reason",
            "completion_tool_name",
            "partial_output",
            "budget_usage",
            "budget_exhaustion",
        }
    ),
    "run_cancelled": frozenset({"reason", "completion_reason", "partial_output", "budget_usage", "budget_exhaustion"}),
    "budget_snapshot": frozenset({"enforcement_boundary", "budget_usage"}),
    "budget_exhausted": frozenset({"enforcement_boundary", "budget_usage", "budget_exhaustion"}),
    "operation_ambiguous": frozenset({"operation_id", "operation_kind", "risk", "idempotency_support"}),
    "model_retry_duplicate_risk": frozenset({"operation_id", "operation_kind", "risk"}),
}


_EVENT_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "run_started": frozenset({"input"}),
    "cycle_aborted": frozenset({"cycle_index", "logical_cycle", "reason"}),
    "model_call_started": frozenset({"call_id", "operation_id", "attempt", "operation", "cycle_index", "backend", "model"}),
    "model_call_completed": frozenset(
        {"call_id", "operation_id", "attempt", "operation", "cycle_index", "backend", "model", "usage"}
    ),
    "model_call_failed": frozenset(
        {
            "call_id",
            "operation_id",
            "attempt",
            "operation",
            "cycle_index",
            "backend",
            "model",
            "outcome",
            "usage",
            "error_code",
        }
    ),
    "assistant_delta": frozenset({"delta", "cycle_index"}),
    "reasoning_delta": frozenset({"delta", "cycle_index"}),
    "model_tool_call_started": frozenset({"tool_call_id", "tool_name", "cycle_index"}),
    "model_tool_call_progress": frozenset({"tool_call_id", "tool_name", "cycle_index"}),
    "tool_call_planned": frozenset({"tool_name", "tool_call_id", "arguments"}),
    "tool_call_started": frozenset({"tool_name", "tool_call_id", "arguments"}),
    "tool_call_completed": frozenset(
        {
            "tool_name",
            "tool_call_id",
            "status",
            "directive",
            "error_code",
            "execution_started",
            "duration_ms",
        }
    ),
    "approval_requested": frozenset({"request_id", "tool_name", "tool_call_id", "message"}),
    "approval_resolved": frozenset({"request_id", "tool_name", "tool_call_id", "action"}),
    "memory_compact_started": frozenset({"message_count", *_MEMORY_COMPACT_STARTED_FIELDS}),
    "memory_compact_completed": frozenset({"before_count", "after_count", *_MEMORY_COMPACT_COMPLETED_FIELDS}),
    "sub_run_started": frozenset({"parent_tool_call_id", "status"}),
    "sub_run_completed": frozenset({"parent_tool_call_id", "status"}),
    "handoff_started": frozenset({"source_agent", "target_agent", "tool_call_id", "status"}),
    "handoff_completed": frozenset({"source_agent", "target_agent", "tool_call_id", "status"}),
    "run_state_changed": frozenset({"state"}),
    "host_interaction_requested": frozenset(
        {
            "interaction_id",
            "logical_cycle",
            "operation_id",
            "tool_call_id",
            "request_digest",
            "prompt",
            "cycle_index",
        }
    ),
    "host_interaction_response_consumed": frozenset(
        {
            "interaction_id",
            "logical_cycle",
            "operation_id",
            "tool_call_id",
            "request_digest",
            "command_id",
            "response_digest",
            "cycle_index",
        }
    ),
    "diagnostic": frozenset({"level", "code", "details"}),
    "run_completed": frozenset({"status"}),
    "run_failed": frozenset({"error"}),
    "run_cancelled": frozenset({"reason"}),
    "budget_snapshot": frozenset({"enforcement_boundary", "budget_usage"}),
    "budget_exhausted": frozenset({"enforcement_boundary", "budget_usage", "budget_exhaustion"}),
    "operation_ambiguous": frozenset({"operation_id", "operation_kind", "risk", "idempotency_support", "cycle_index"}),
    "model_retry_duplicate_risk": frozenset({"operation_id", "operation_kind", "risk", "cycle_index"}),
}


def _stream_delta(value: Any, field_name: str = "delta") -> str:
    if not isinstance(value, str):
        raise ValueError(f"Run event {field_name} must be a string")
    return value


def _optional_stream_counter(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _JSON_SAFE_INTEGER_MAX:
        raise ValueError(f"Run event {field_name} must be a non-negative JSON-safe integer or null")
    return value


def _memory_compact_counter(value: Any, field_name: str) -> int:
    counter = _optional_stream_counter(value, field_name)
    if counter is None:
        raise ValueError(f"Run event {field_name} must be a non-negative JSON-safe integer")
    return counter


def _memory_compact_trigger(value: Any) -> MemoryCompactTrigger:
    if not isinstance(value, str) or value not in _MEMORY_COMPACT_TRIGGER_VALUES:
        raise ValueError(f"Unsupported memory compact trigger: {value!r}")
    return cast(MemoryCompactTrigger, value)


def _memory_compact_mode(value: Any) -> MemoryCompactMode:
    if not isinstance(value, str) or value not in _MEMORY_COMPACT_MODE_VALUES:
        raise ValueError(f"Unsupported memory compact mode: {value!r}")
    return cast(MemoryCompactMode, value)


def _reserved_output_source(value: Any) -> ReservedOutputSource:
    if not isinstance(value, str) or value not in _RESERVED_OUTPUT_SOURCE_VALUES:
        raise ValueError(f"Unsupported memory compact reserved_output_source: {value!r}")
    return cast(ReservedOutputSource, value)


def _memory_compact_changed(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Run event changed must be a boolean")
    return value


def _completion_reason(value: Any) -> CompletionReason | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Run event completion_reason must be a string or null")
    try:
        return CompletionReason(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported run event completion_reason: {value!r}") from exc


def _completion_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Run event {field_name} must be a string or null")
    return value


def _diagnostic_level(value: Any) -> DiagnosticLevel:
    if not isinstance(value, str) or value not in _DIAGNOSTIC_LEVEL_VALUES:
        raise ValueError(f"Unsupported diagnostic level: {value!r}")
    return cast(DiagnosticLevel, value)


def _diagnostic_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        if not -_JSON_SAFE_INTEGER_MAX <= value <= _JSON_SAFE_INTEGER_MAX:
            raise ValueError(f"Run event {path} integer exceeds the JSON-safe range")
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"Run event {path} number must be finite")
        return value
    if isinstance(value, list):
        return [_diagnostic_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Run event {path} object keys must be strings")
            normalized[key] = _diagnostic_json_value(item, f"{path}.{key}")
        return normalized
    raise ValueError(f"Run event {path} must contain JSON values")


def _diagnostic_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Run event diagnostic details must be an object")
    return cast(dict[str, Any], _diagnostic_json_value(value, "diagnostic.details"))


def _budget_usage(value: Any) -> BudgetUsageSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Run event budget_usage must be an object or null")
    return BudgetUsageSnapshot.from_dict(value)


def _budget_exhaustion(value: Any) -> BudgetExhaustion | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Run event budget_exhaustion must be an object or null")
    return BudgetExhaustion.from_dict(value)


def _tool_status(value: Any) -> str:
    if not isinstance(value, str) or value not in _TOOL_STATUS_VALUES:
        raise ValueError(f"Unsupported tool event status: {value!r}")
    return value


def _tool_directive(value: Any) -> str:
    if not isinstance(value, str) or value not in _TOOL_DIRECTIVE_VALUES:
        raise ValueError(f"Unsupported tool event directive: {value!r}")
    return value


def _tool_error_code(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("Run event error_code must be a string or null")
    return value


def _tool_execution_started(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Run event execution_started must be a boolean")
    return value


def _tool_duration_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _JSON_SAFE_INTEGER_MAX:
        raise ValueError("Run event duration_ms must be a non-negative JSON-safe integer or null")
    return value


def _event_tool_metadata(value: ToolMetadata | dict[str, Any] | None) -> ToolMetadata | None:
    if value is None:
        return None
    from vv_agent.kernel._tool_metadata import ToolMetadata

    if isinstance(value, ToolMetadata):
        source = value.to_dict()
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError("Run event tool_metadata must be an object")
    try:
        return ToolMetadata.from_dict(source)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid run event tool_metadata: {exc}") from exc


def _approval_action(action: Any) -> ApprovalAction:
    if not isinstance(action, str) or action not in _APPROVAL_ACTIONS:
        raise ValueError(f"Unsupported approval action: {action!r}")
    return cast(ApprovalAction, action)


def _host_interaction_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"Run event {field_name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"Run event {field_name} must be a lowercase SHA-256 digest") from exc
    return value


def _host_interaction_identity(value: Any, field_name: str) -> str:
    text = _required_event_text(value, field_name)
    if len(text.encode("utf-8")) > 512:
        raise ValueError(f"Run event {field_name} exceeds the UTF-8 identity limit")
    return text


def _host_interaction_cycle(value: Any, field_name: str) -> int:
    return _positive_event_integer(value, field_name)


def _host_interaction_prompt(value: Any) -> str:
    text = _required_event_text(value, "prompt")
    if len(text.encode("utf-8")) > 65536:
        raise ValueError("Run event prompt exceeds the UTF-8 content limit")
    return text


def _required_event_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Run event {field_name} must be a non-empty string")
    return value


def _cancel_requested_transition(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != {"from", "to"}:
        raise ValueError("Run event cancel_requested must contain exactly from and to")
    if value["from"] is not False or value["to"] is not True:
        raise ValueError("Run event cancel_requested must transition from false to true")
    return {"from": False, "to": True}


def _positive_event_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _JSON_SAFE_INTEGER_MAX:
        raise ValueError(f"Run event {field_name} must be a positive JSON-safe integer")
    return value


def _validate_event_wire(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Run event payload must be an object")
    if payload.get("version") != ("v6"):
        raise ValueError(f"Unsupported run event version: {payload.get('version')!r}")

    event_type = payload.get("type")
    if not isinstance(event_type, str) or event_type not in _EVENT_FIELDS:
        raise ValueError(f"Unsupported run event type: {event_type!r}")
    allowed_fields = _COMMON_EVENT_FIELDS | _EVENT_FIELDS[event_type]
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Run event {event_type} contains unknown fields: {names}")
    required_fields = {
        "version",
        "type",
        "event_id",
        "run_id",
        "trace_id",
        "created_at",
        *_EVENT_REQUIRED_FIELDS.get(event_type, ()),
    }
    required_fields.add("session_id")
    missing_fields = required_fields - set(payload)
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise ValueError(f"Run event {event_type} is missing required fields: {names}")

    for field_name in ("type", "event_id", "run_id", "trace_id"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run event {field_name} must be a non-empty string")

    for field_name in ("session_id", "parent_event_id", "parent_run_id", "agent_name"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Run event {field_name} must be a string or null")

    created_at = payload["created_at"]
    if isinstance(created_at, bool) or not isinstance(created_at, int | float) or not isfinite(created_at) or created_at < 0:
        raise ValueError("Run event created_at must be a finite non-negative number")

    cycle_index = payload.get("cycle_index")
    if cycle_index is not None and (isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0):
        raise ValueError("Run event cycle_index must be a non-negative integer or null")
    if payload["type"] == "cycle_aborted":
        if cycle_index is None or payload.get("logical_cycle") != cycle_index + 1:
            raise ValueError("Run event cycle_aborted logical_cycle must equal cycle_index + 1")
        if payload.get("reason") not in {"cancelled", "lease_lost", "operator_abort"}:
            raise ValueError("Run event cycle_aborted reason is invalid")

    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise ValueError("Run event metadata must be an object")
    if payload["type"] == "run_state_changed":
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and "cancel_requested" in metadata:
            raise ValueError("Run event cancel_requested must be a top-level field")
        if "cancel_requested" in payload:
            _cancel_requested_transition(payload["cancel_requested"])
    if payload["type"] == "diagnostic":
        _diagnostic_level(payload.get("level"))
        _required_event_text(payload.get("code"), "code")
        _diagnostic_details(payload.get("details"))
    if payload["type"] == "host_interaction_requested":
        _host_interaction_identity(payload.get("interaction_id"), "interaction_id")
        _host_interaction_cycle(payload.get("logical_cycle"), "logical_cycle")
        _host_interaction_identity(payload.get("operation_id"), "operation_id")
        _host_interaction_identity(payload.get("tool_call_id"), "tool_call_id")
        request_digest = _host_interaction_digest(payload.get("request_digest"), "request_digest")
        prompt = _host_interaction_prompt(payload.get("prompt"))
        expected_digest = canonical_json_sha256(
            {
                "interaction_id": payload["interaction_id"],
                "logical_cycle": payload["logical_cycle"],
                "operation_id": payload["operation_id"],
                "prompt": prompt,
                "schema_version": "vv-agent.host-interaction-request.v1",
                "tool_call_id": payload["tool_call_id"],
            },
            "host_interaction_request",
        )
        if request_digest != expected_digest:
            raise ValueError("Run event request_digest does not match host interaction request")
    if payload["type"] == "host_interaction_response_consumed":
        _host_interaction_identity(payload.get("interaction_id"), "interaction_id")
        _host_interaction_cycle(payload.get("logical_cycle"), "logical_cycle")
        _host_interaction_identity(payload.get("operation_id"), "operation_id")
        _host_interaction_identity(payload.get("tool_call_id"), "tool_call_id")
        _host_interaction_digest(payload.get("request_digest"), "request_digest")
        _host_interaction_identity(payload.get("command_id"), "command_id")
        _host_interaction_digest(payload.get("response_digest"), "response_digest")
    if payload["type"] in {"model_call_started", "model_call_completed", "model_call_failed"}:
        _required_event_text(payload.get("call_id"), "call_id")
        _required_event_text(payload.get("operation_id"), "operation_id")
        _positive_event_integer(payload.get("attempt"), "attempt")
        _positive_event_integer(payload.get("cycle_index"), "cycle_index")
        ModelCallOperation(payload.get("operation"))
        _required_event_text(payload.get("backend"), "backend")
        _required_event_text(payload.get("model"), "model")
        if payload["type"] != "model_call_started":
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                raise ValueError(f"Run event {payload['type']} requires usage")
            TokenUsage.from_dict(usage)
        if payload["type"] == "model_call_failed":
            if payload.get("outcome") not in _MODEL_CALL_OUTCOME_VALUES:
                raise ValueError(f"Unsupported model call outcome: {payload.get('outcome')!r}")
            _required_event_text(payload.get("error_code"), "error_code")
    tool_lifecycle_types = {"tool_call_planned", "tool_call_started", "tool_call_completed"}
    if payload["type"] in tool_lifecycle_types:
        _required_event_text(payload.get("tool_call_id"), "tool_call_id")
        _required_event_text(payload.get("tool_name"), "tool_name")
        if "tool_metadata" in payload:
            if not isinstance(payload["tool_metadata"], dict):
                raise ValueError("Run event tool_metadata must be an object")
            _event_tool_metadata(payload["tool_metadata"])
    if payload["type"] in {"tool_call_planned", "tool_call_started"} and not isinstance(payload.get("arguments"), dict):
        raise ValueError("Run event tool arguments must be an object")
    if payload["type"] == "tool_call_completed":
        _tool_status(payload.get("status"))
        _tool_directive(payload["directive"])
        _tool_error_code(payload["error_code"])
        _tool_execution_started(payload["execution_started"])
        _tool_duration_ms(payload["duration_ms"])
        if payload.get("execution_started") is False and payload.get("duration_ms") is not None:
            raise ValueError("Run event duration_ms must be null when execution_started is false")
        if "operation_id" in payload:
            _required_event_text(payload.get("operation_id"), "operation_id")
        if "attempt" in payload:
            _positive_event_integer(payload.get("attempt"), "attempt")
    if (
        payload["type"] == "reconciliation_resolved"
        and "claim_mode" in payload
        and payload["claim_mode"] not in {"continue", "recovery"}
    ):
        raise ValueError("Run event claim_mode must be 'continue' or 'recovery'")
    if payload["type"] in {"approval_requested", "approval_resolved"}:
        _required_event_text(payload.get("request_id"), "request_id")
        _required_event_text(payload.get("tool_call_id"), "tool_call_id")
        _required_event_text(payload.get("tool_name"), "tool_name")
    if payload["type"] == "approval_resolved":
        _approval_action(payload["action"])
    if payload["type"] == "memory_compact_started":
        _memory_compact_counter(payload["message_count"], "message_count")
        if "estimated_tokens" in payload:
            _optional_stream_counter(payload["estimated_tokens"], "estimated_tokens")
        _memory_compact_trigger(payload["trigger"])
        for field_name in (
            "configured_threshold",
            "effective_threshold",
            "microcompact_threshold",
            "microcompact_target",
            "candidate_count",
            "estimated_reclaimable_tokens",
            "model_context_window",
            "model_max_output_tokens",
            "reserved_output_tokens",
            "autocompact_buffer_tokens",
        ):
            if field_name == "model_max_output_tokens":
                _optional_stream_counter(payload[field_name], field_name)
            else:
                _memory_compact_counter(payload[field_name], field_name)
        _reserved_output_source(payload["reserved_output_source"])
    if payload["type"] == "memory_compact_completed":
        _memory_compact_counter(payload["before_count"], "before_count")
        _memory_compact_counter(payload["after_count"], "after_count")
        if "summary_tokens" in payload:
            _optional_stream_counter(payload["summary_tokens"], "summary_tokens")
        _memory_compact_mode(payload["mode"])
        _memory_compact_changed(payload["changed"])
        for field_name in ("archived_count", "reclaimed_tokens", "artifact_failure_count"):
            _memory_compact_counter(payload[field_name], field_name)
    if payload["type"] in {"assistant_delta", "reasoning_delta"}:
        _stream_delta(payload.get("delta"))
    typed_stream_events = {
        "assistant_delta",
        "reasoning_delta",
        "model_tool_call_started",
        "model_tool_call_progress",
    }
    if payload["type"] in typed_stream_events and (
        isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 1
    ):
        raise ValueError("Run event stream variants require a positive cycle_index")
    stream_counter_fields = {
        "assistant_delta": ("content_chars", "estimated_tokens"),
        "reasoning_delta": ("reasoning_chars", "estimated_tokens"),
        "model_tool_call_started": ("tool_call_index", "arguments_chars", "estimated_tokens"),
        "model_tool_call_progress": ("tool_call_index", "arguments_chars", "estimated_tokens"),
    }
    for field_name in stream_counter_fields.get(payload["type"], ()):
        _optional_stream_counter(payload.get(field_name), field_name)
    if payload["type"] in {"model_tool_call_started", "model_tool_call_progress"}:
        _required_event_text(payload.get("tool_call_id"), "tool_call_id")
        _required_event_text(payload.get("tool_name"), "tool_name")

    _completion_reason(payload.get("completion_reason"))
    _completion_text(payload.get("completion_tool_name"), "completion_tool_name")
    _completion_text(payload.get("partial_output"), "partial_output")
    budget_usage = _budget_usage(payload.get("budget_usage"))
    budget_exhaustion = _budget_exhaustion(payload.get("budget_exhaustion"))
    if payload["type"] in {"budget_snapshot", "budget_exhausted"}:
        boundary_raw = payload.get("enforcement_boundary")
        try:
            boundary = BudgetEnforcementBoundary(boundary_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported budget enforcement boundary: {boundary_raw!r}") from exc
        if budget_usage is None:
            raise ValueError(f"Run event {payload['type']} requires budget_usage")
        if payload["type"] == "budget_exhausted":
            if budget_exhaustion is None:
                raise ValueError("Run event budget_exhausted requires budget_exhaustion")
            if budget_exhaustion.enforcement_boundary is not boundary:
                raise ValueError("Run event budget exhaustion boundaries must match")

    if payload["type"] in {"operation_ambiguous", "model_retry_duplicate_risk"}:
        if not isinstance(payload.get("cycle_index"), int):
            raise ValueError("Run event cycle_index is required for operation lifecycle events")
        _required_event_text(payload.get("operation_id"), "operation_id")
        OperationKind(payload.get("operation_kind"))
    if payload["type"] in {"operation_ambiguous", "model_retry_duplicate_risk"}:
        _required_event_text(payload.get("risk"), "risk")
    if payload["type"] == "operation_ambiguous":
        operation_kind = OperationKind(payload.get("operation_kind"))
        support = payload.get("idempotency_support")
        if operation_kind is OperationKind.TOOL:
            if support is None:
                raise ValueError("ambiguous tool event requires idempotency_support")
            from vv_agent.kernel._tool_metadata import ToolIdempotency

            ToolIdempotency(support)
        elif support is not None:
            raise ValueError("ambiguous model event idempotency_support must be null")
    if payload["type"] == "model_retry_duplicate_risk" and payload.get("operation_kind") != "model":
        raise ValueError("model retry duplicate risk event requires model operation_kind")


def _validate_cycle_aborted_fields(cycle_index: int, logical_cycle: int, reason: str) -> None:
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
        raise ValueError("cycle_aborted cycle_index must be a non-negative integer")
    if isinstance(logical_cycle, bool) or not isinstance(logical_cycle, int) or logical_cycle != cycle_index + 1:
        raise ValueError("cycle_aborted logical_cycle must equal cycle_index + 1")
    if reason not in {"cancelled", "lease_lost", "operator_abort"}:
        raise ValueError("cycle_aborted reason is invalid")


def validate_retained_event(payload: dict[str, Any]) -> None:
    _validate_event_wire(payload)
    if payload["type"] == "cycle_aborted":
        _validate_cycle_aborted_fields(payload["cycle_index"], payload["logical_cycle"], payload["reason"])
