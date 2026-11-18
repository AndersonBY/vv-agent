"""One pre-dispatch admission algorithm over explicit tool and call facts."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ._call_context import matches_preapproved_call
from ._records import Record
from ._tool_metadata import metadata_policy_denial_source
from ._tool_policy import _TASK_TOOL_POLICY_METADATA_KEYS
from ._tool_schema import SchemaCheck, invalid_tool_arguments_result, validate_tool_arguments
from ._tool_values import ToolCall, ToolExecutionResult, ToolResultStatus


class Read(Enum):
    SCHEMA = auto()
    HANDLER_MANAGED = auto()
    POLICY_METADATA = auto()
    APPROVAL_MODE = auto()
    TOOL_METADATA = auto()


@dataclass(frozen=True, slots=True)
class ApprovedCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyPredicate:
    predicate: Callable[[str, dict[str, Any]], Any]
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RequiresApproval:
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RequestApproval:
    """Run the existing host approval transport using its retained snapshot."""


@dataclass(frozen=True, slots=True)
class Dispatch:
    mark_started: bool


class _AdmissionStopIteration(Exception):
    def __init__(self, original: StopIteration):
        self.original = original


POLICY_KEYS = (*_TASK_TOOL_POLICY_METADATA_KEYS, "_vv_agent_tool_policy_can_use_tool")

AdmissionRequest = Read | ApprovedCall | PolicyPredicate | RequiresApproval | RequestApproval | SchemaCheck


def _error_result(tool_call_id: str, message: str, *, error_code: str | None = None) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=json.dumps({"ok": False, "error": message, "error_code": error_code}, ensure_ascii=False),
    )


def _parse_arguments(
    tool_call_id: str,
    raw_arguments: Any,
) -> tuple[dict[str, Any], ToolExecutionResult | None]:
    if raw_arguments is None:
        return {}, None

    if isinstance(raw_arguments, dict):
        return raw_arguments, None

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}, None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return {}, _error_result(tool_call_id, f"Invalid tool arguments JSON: {exc}", error_code="invalid_arguments_json")

        if isinstance(parsed, dict):
            return parsed, None
        return {}, _error_result(
            tool_call_id,
            "Tool arguments must decode to an object",
            error_code="invalid_arguments_payload",
        )

    return {}, _error_result(
        tool_call_id,
        f"Unsupported tool argument type: {type(raw_arguments).__name__}",
        error_code="invalid_arguments_type",
    )


def _policy_steps(
    call: ToolCall, *, executor_present: bool, allowed_tool_names: Collection[str] | None
) -> Generator[AdmissionRequest, Any, ToolExecutionResult | None]:
    try:
        if executor_present and (yield Read.HANDLER_MANAGED):
            return None
        metadata = yield Read.POLICY_METADATA
        allowed_tools = metadata.get("_vv_agent_allowed_tools")
        disallowed_tools = metadata.get("_vv_agent_disallowed_tools")
        can_use_tool = metadata.get("_vv_agent_tool_policy_can_use_tool")

        policy_source: str | None = None
        if isinstance(allowed_tools, list) and call.name not in allowed_tools:
            policy_source = "allowed_tools"
        elif isinstance(disallowed_tools, list) and call.name in disallowed_tools:
            policy_source = "disallowed_tools"
        elif callable(can_use_tool) and not (yield PolicyPredicate(can_use_tool, call.name, dict(call.arguments))):
            policy_source = "can_use_tool"
        elif executor_present:
            policy_source = metadata_policy_denial_source(
                (yield Read.TOOL_METADATA),
                denied_side_effects=metadata.get("_vv_agent_denied_side_effects", []),
                denied_capability_tags=metadata.get(
                    "_vv_agent_denied_capability_tags",
                    [],
                ),
                deny_terminal_tools=metadata.get("_vv_agent_deny_terminal_tools", False),
                denied_cost_dimensions=metadata.get(
                    "_vv_agent_denied_cost_dimensions",
                    [],
                ),
            )
        elif allowed_tool_names is not None and call.name not in allowed_tool_names:
            policy_source = "planned_name"

        if policy_source is None and allowed_tool_names is not None and call.name not in allowed_tool_names:
            policy_source = "planned_name"

        if policy_source is None:
            return None
        message = f"Tool {call.name} is not allowed for these arguments."
        return ToolExecutionResult(
            tool_call_id=call.id,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": "tool_not_allowed",
                    "tool_name": call.name,
                },
                ensure_ascii=False,
            ),
            status_code=ToolResultStatus.ERROR,
            error_code="tool_not_allowed",
            metadata={
                "mode": "permission_denied",
                "policy_source": policy_source,
                "tool_name": call.name,
                "arguments": dict(call.arguments),
                "message": message,
            },
        )

    except StopIteration as error:
        raise _AdmissionStopIteration(error) from error


def admit_steps(
    call: ToolCall, *, original_call: ToolCall, executor_present: bool, allowed_tool_names: Collection[str] | None
) -> Generator[AdmissionRequest, Any, ToolExecutionResult | Dispatch]:
    try:
        if not executor_present:
            result = yield from _policy_steps(call, executor_present=False, allowed_tool_names=allowed_tool_names)
            if result is not None:
                return result
            return _error_result(original_call.id, f"Unknown tool: {original_call.name}", error_code="tool_not_found")
        tool_call_id = call.id
        schema = yield Read.SCHEMA
        issues = yield from validate_tool_arguments(schema, call.arguments)
        result = invalid_tool_arguments_result(tool_call_id=tool_call_id, issues=issues)
        if result is not None:
            return result
        result = yield from _policy_steps(call, executor_present=True, allowed_tool_names=allowed_tool_names)
        if result is not None:
            return result
        if not (yield Read.HANDLER_MANAGED):
            query = ApprovedCall(call.id, call.name, dict(call.arguments))
            approved = yield query
            if not matches_preapproved_call(
                approved, tool_call_id=query.tool_call_id, tool_name=query.tool_name, arguments=query.arguments
            ):
                mode = str((yield Read.APPROVAL_MODE) or "default")
                if mode != "never":
                    needs_approval = mode == "always"
                    if not needs_approval:
                        needs_approval = yield RequiresApproval(call.arguments)
                    if needs_approval:
                        result = yield RequestApproval()
                        if result is not None:
                            return result
        return Dispatch(mark_started=not (yield Read.HANDLER_MANAGED))
    except StopIteration as error:
        raise _AdmissionStopIteration(error) from error


def session_preflight_result(plan: Record, *, handler_managed: Any) -> ToolExecutionResult | None:
    if handler_managed:
        return ToolExecutionResult(
            tool_call_id=plan.payload["request"]["id"],
            status_code=ToolResultStatus.ERROR,
            content="Handler-managed dispatch requires an explicit session provider adapter.",
            error_code="session_dispatch_boundary_required",
        )
    return None
