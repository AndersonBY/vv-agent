from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from vv_agent.app_server.processor import CLIENT_METHODS, SERVER_NOTIFICATION_METHODS, SERVER_REQUEST_METHODS
from vv_agent.app_server.protocol.approval import ApprovalDecision
from vv_agent.app_server.protocol.identity import CLIENT_ID_PATTERN
from vv_agent.session.records import MODEL_SELECTION, nullable

SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
REQUEST_ID_SCHEMA = {
    "oneOf": [
        {"type": "string"},
        {"type": "integer", "minimum": -(2**63), "maximum": 2**63 - 1},
    ]
}
NULLABLE_STRING = {"type": ["string", "null"]}
STRING_MAP = {"type": "object", "additionalProperties": {"type": "string"}}
JSON_OBJECT = {"type": "object", "additionalProperties": True}
JSON_SCHEMA_NAMES = (
    "AppItem",
    "AppThread",
    "AppTurn",
    "ApprovalDecision",
    "ApprovalRequestParams",
    "ApprovalResolveParams",
    "ClientRequest",
    "InitializeParams",
    "InitializeResponse",
    "JsonRpcMessage",
    "SchemaExportResponse",
    "ServerNotification",
    "ServerRequest",
    "ThreadCompactStartResponse",
    "ThreadContextReadResponse",
    "ThreadReadResponse",
    "ThreadResumeResponse",
    "ThreadStartResponse",
    "TurnResumeParams",
    "TurnResumeResponse",
    "TurnStartResponse",
)
TYPESCRIPT_SCHEMA_NAMES = tuple(name for name in JSON_SCHEMA_NAMES if name != "JsonRpcMessage")


def _object(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    return schema


def _array(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item_schema}


def _definitions() -> dict[str, dict[str, Any]]:
    approval_decisions = {"type": "string", "enum": [decision.value for decision in ApprovalDecision]}
    input_item = _object({}, additional_properties=True)
    thread_item = _object(
        {
            "itemId": {"type": "string"},
            "threadId": {"type": "string"},
            "turnId": {"type": "string"},
            "type": {"type": "string"},
            "status": {"type": "string"},
            "payload": JSON_OBJECT,
            "createdAt": {"type": "number"},
            "updatedAt": {"type": "number"},
        },
        required=["itemId", "threadId", "turnId", "type", "status", "payload", "createdAt", "updatedAt"],
    )
    thread_record = _object(
        {
            "threadId": {"type": "string"},
            "agentKey": {"type": "string"},
            "cwd": NULLABLE_STRING,
            "createdAt": {"type": "number"},
            "updatedAt": {"type": "number"},
            "archivedAt": {"type": ["number", "null"]},
            "status": {"type": "string"},
            "metadata": JSON_OBJECT,
        },
        required=["threadId", "agentKey", "cwd", "createdAt", "updatedAt", "archivedAt", "status", "metadata"],
    )
    turn_record = _object(
        {
            "turnId": {"type": "string"},
            "threadId": {"type": "string"},
            "runId": NULLABLE_STRING,
            "status": {"type": "string"},
            "startedAt": {"type": "number"},
            "completedAt": {"type": ["number", "null"]},
            "input": _array({"$ref": "#/$defs/InputItem"}),
            "result": JSON_OBJECT,
        },
        required=["turnId", "threadId", "runId", "status", "startedAt", "completedAt", "input", "result"],
    )
    approval_request = _object(
        {
            "requestId": {"type": "string"},
            "threadId": {"type": "string"},
            "turnId": {"type": "string"},
            "toolCallId": {"type": "string"},
            "toolName": {"type": "string"},
            "preview": {"type": "string"},
            "arguments": JSON_OBJECT,
        },
        required=["requestId", "threadId", "turnId", "toolCallId", "toolName", "preview", "arguments"],
    )
    approval_resolve = _object(
        {
            "requestId": {"type": "string"},
            "threadId": {"type": "string"},
            "turnId": {"type": "string"},
            "decision": approval_decisions,
            "reason": {"type": "string"},
            "metadata": JSON_OBJECT,
        },
        required=["requestId", "threadId", "turnId", "decision"],
    )
    definitions: dict[str, dict[str, Any]] = {
        "ApprovalDecision": approval_decisions,
        "EmptyParams": _object({}),
        "ClientInfo": _object(
            {"name": {"type": "string"}, "title": {"type": "string"}, "version": {"type": "string"}},
            required=["name"],
        ),
        "ClientCapabilities": _object(
            {
                "experimentalApi": {"type": "boolean"},
                "optOutNotificationMethods": _array({"type": "string"}),
            }
        ),
        "ServerCapabilities": _object(
            {
                "modelList": {"type": "boolean"},
                "threadLifecycle": {"type": "boolean"},
                "notificationOptOut": {"type": "boolean"},
                "schemaExport": {"type": "boolean"},
                "approvalResolve": {"type": "boolean"},
                "controllerAdmission": {"type": "boolean"},
            },
            required=["modelList", "threadLifecycle", "notificationOptOut", "schemaExport", "approvalResolve"],
        ),
        "InitializeParams": _object(
            {
                "clientInfo": {"$ref": "#/$defs/ClientInfo"},
                "capabilities": {"$ref": "#/$defs/ClientCapabilities"},
            },
            required=["clientInfo"],
        ),
        "ModelListParams": _object(
            {
                "agentKey": {"type": "string"},
                "provider": {"type": "string"},
            }
        ),
        "ThreadStartParams": _object(
            {
                "inputId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                "agentKey": {"type": "string"},
                "cwd": {"type": "string"},
                "metadata": JSON_OBJECT,
            },
            required=["inputId"],
        ),
        "ThreadIdParams": _object({"threadId": {"type": "string"}}, required=["threadId"]),
        "ThreadResumeParams": _object(
            {"threadId": {"type": "string"}, "subscribe": {"type": "boolean"}},
            required=["threadId"],
        ),
        "ThreadModelSelectResponse": _object(
            {"threadId": {"type": "string"}, "inputId": {"type": "string"}, "accepted": {"const": True}},
            required=["threadId", "inputId", "accepted"],
        ),
        "ThreadModelSelectParams": _object(
            {
                "threadId": {"type": "string", "minLength": 1},
                "inputId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                "selection": nullable(MODEL_SELECTION),
            },
            required=["threadId", "inputId", "selection"],
        ),
        "ThreadReadParams": _object(
            {"threadId": {"type": "string"}},
            required=["threadId"],
        ),
        "ThreadStatusParams": _object({"threadId": {"type": "string"}}, required=["threadId"]),
        "ThreadListParams": _object(
            {
                "includeArchived": {"type": "boolean"},
                "archived": {"type": "boolean"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 0},
            }
        ),
        "TurnStartParams": _object(
            {
                "threadId": {"type": "string"},
                "inputId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                "expectedTurnId": NULLABLE_STRING,
                "input": _array({"$ref": "#/$defs/InputItem"}),
                "metadata": JSON_OBJECT,
            },
            required=["threadId", "inputId", "expectedTurnId", "input"],
        ),
        "TurnResumeParams": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
            },
            required=["threadId", "turnId"],
        ),
        **{
            name: _object(
                {
                    "threadId": {"type": "string"},
                    "inputId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                    "expectedTurnId": {"type": "string"},
                    "input": _array({"$ref": "#/$defs/InputItem"}),
                },
                required=["threadId", "inputId", "expectedTurnId", "input"],
            )
            for name in ("TurnSteerParams", "TurnFollowUpParams")
        },
        "TurnInterruptParams": _object(
            {
                "threadId": {"type": "string"},
                "expectedTurnId": {"type": "string"},
                "actionId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
            },
            required=["threadId", "expectedTurnId", "actionId"],
        ),
        "EventCursor": _object(
            {
                "threadId": {"type": "string"},
                "seq": {"type": "integer", "minimum": 1},
                "offset": {"type": "integer", "minimum": 0},
                "recordDigest": {"type": "string"},
            },
            required=["threadId", "seq", "offset", "recordDigest"],
        ),
        "ThreadEventsListParams": _object(
            {
                "threadId": {"type": "string"},
                "afterCursor": {"$ref": "#/$defs/EventCursor"},
                "throughCursor": {"$ref": "#/$defs/EventCursor"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 256},
            },
            required=["threadId", "limit"],
        ),
        "ThreadEventsListResponse": _object(
            {
                "events": _array(
                    _object({"cursor": {"$ref": "#/$defs/EventCursor"}, "event": JSON_OBJECT}, required=["cursor", "event"])
                ),
                "nextCursor": {"$ref": "#/$defs/EventCursor"},
                "throughCursor": {"$ref": "#/$defs/EventCursor"},
                "more": {"type": "boolean"},
            },
            required=["events", "nextCursor", "throughCursor", "more"],
        ),
        "TurnActionParams": _object(
            {
                "threadId": {"type": "string", "minLength": 1, "maxLength": 512},
                "turnId": {"type": "string", "minLength": 1, "maxLength": 512},
                "actionId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                "action": {
                    "oneOf": [
                        _object(
                            {
                                "kind": {"const": "respond"},
                                "message": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"const": "user"},
                                        "content": {"type": "string", "minLength": 1, "maxLength": 65536},
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False,
                                },
                            },
                            required=["kind", "message"],
                        ),
                        *[
                            _object({"kind": {"const": kind}}, required=["kind"])
                            for kind in ("suspend", "resume", "cancel", "abort")
                        ],
                    ]
                },
            },
            required=["threadId", "turnId", "actionId", "action"],
        ),
        "ApprovalRequestParams": approval_request,
        "ApprovalResolveParams": approval_resolve,
        "InputItem": input_item,
        "AppItem": thread_item,
        "AppThread": thread_record,
        "AppTurn": turn_record,
        "ThreadStartedParams": _object(
            {
                "threadId": {"type": "string"},
                "agentKey": {"type": "string"},
                "cwd": NULLABLE_STRING,
                "status": {"type": "string"},
            },
            required=["threadId", "agentKey", "cwd", "status"],
        ),
        "ThreadArchivedParams": _object(
            {"threadId": {"type": "string"}, "archived": {"type": "boolean"}},
            required=["threadId", "archived"],
        ),
        "TurnStartedParams": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "runId": {"type": "string"},
                "status": {"type": "string"},
            },
            required=["threadId", "turnId"],
        ),
        "AgentMessageDeltaParams": {
            **deepcopy(thread_item),
            "properties": {**thread_item["properties"], "delta": {"type": "string"}},
            "required": [*thread_item["required"], "delta"],
        },
        "ToolCallDeltaParams": {
            **deepcopy(thread_item),
            "properties": {**thread_item["properties"], "delta": {}},
            "required": [*thread_item["required"], "delta"],
        },
        "WarningParams": _object(
            {
                "message": {"type": "string"},
                "code": {"type": "string"},
            },
            required=["message"],
        ),
        "TurnCompletedParams": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "runId": {"type": "string"},
                "status": {"type": "string"},
                "finalOutput": {},
                "completionReason": {"type": "string"},
                "completionToolName": {"type": "string"},
                "partialOutput": {"type": "string"},
                "waitReason": {"type": "string"},
                "tokenUsage": JSON_OBJECT,
                "budgetUsage": JSON_OBJECT,
                "budgetExhaustion": JSON_OBJECT,
                "error": {"type": "string"},
            },
            required=["threadId", "turnId", "status"],
        ),
    }
    definitions.update(_result_definitions())
    definitions.update(_context_definitions())
    from vv_agent.session.app_server import ThreadStatus

    status = {"type": "string", "enum": [value.value for value in ThreadStatus]}
    for name in (
        "AppThread",
        "ThreadStatusChangedParams",
        "ThreadStatusResponse",
        "ThreadStartResponse",
        "ThreadStartedParams",
    ):
        definitions[name]["properties"]["status"] = status
    definitions["ThreadStartResponse"]["properties"].update(accepted={"const": True}, replayed={"type": "boolean"})
    definitions["ThreadStartResponse"]["required"].extend(["accepted", "replayed"])
    definitions["TurnInterruptResponse"] = deepcopy(definitions["TurnActionResponse"])
    definitions["Interaction"] = _object(
        {
            "sessionId": {"type": "string"},
            "turnId": {"type": "string"},
            "interactionId": {"type": "string"},
            "requestId": {"type": "string"},
            "prompt": {"type": "string"},
        },
        required=["sessionId", "turnId"],
    )
    definitions["CacheUsage"] = _object(
        {
            "status": {"type": "string", "enum": ["provider_reported", "accounting_missing", "unsupported"]},
            **{
                key: {"type": ["integer", "null"], "minimum": 0}
                for key in ("readInputTokens", "writeInputTokens", "uncachedInputTokens")
            },
            "source": NULLABLE_STRING,
        },
        required=["status", "readInputTokens", "writeInputTokens", "uncachedInputTokens", "source"],
    )
    counts = {
        key: {"type": ["integer", "null"], "minimum": 0}
        for key in ("inputTokens", "outputTokens", "totalTokens", "reasoningTokens")
    }
    definitions["TokenUsage"] = _object(
        {
            "schemaVersion": {"const": "vv-agent.token-usage.v1"},
            **counts,
            "usageSource": {"type": "string", "enum": ["provider_reported", "estimated", "accounting_missing"]},
            "cacheUsage": {"$ref": "#/$defs/CacheUsage"},
            "providerUsage": JSON_OBJECT,
        },
        required=["schemaVersion", *counts, "usageSource", "cacheUsage", "providerUsage"],
    )
    call_fields = {
        "schemaVersion": {"const": "vv-agent.model-call.v2"},
        **{key: {"type": "string", "minLength": 1} for key in ("callId", "operationId", "backend", "model")},
        **{key: {"type": "integer", "minimum": 1} for key in ("attempt", "cycleIndex")},
        "operation": {"type": "string", "enum": ["agent_cycle", "memory_compaction", "session_memory", "output_repair"]},
        "status": {"type": "string", "enum": ["completed", "failed", "ambiguous"]},
        "usage": {"$ref": "#/$defs/TokenUsage"},
        "errorCode": NULLABLE_STRING,
    }
    definitions["ModelCallRecord"] = _object(call_fields, required=list(call_fields))
    definitions["TaskTokenUsage"] = _object(
        {
            "schemaVersion": {"const": "vv-agent.task-token-usage.v3"},
            **counts,
            "cacheUsage": {"$ref": "#/$defs/CacheUsage"},
            "modelCalls": _array({"$ref": "#/$defs/ModelCallRecord"}),
        },
        required=["schemaVersion", *counts, "cacheUsage", "modelCalls"],
    )
    result = {
        "finalOutput": {},
        "waitReason": {"type": "string"},
        "completionReason": {"type": "string"},
        "completionToolName": {"type": "string"},
        "partialOutput": {},
        "error": {"type": "string"},
        "tokenUsage": {"$ref": "#/$defs/TaskTokenUsage"},
        "budgetUsage": JSON_OBJECT,
        "budgetExhaustion": JSON_OBJECT,
    }
    definitions["AppTurn"]["properties"]["result"] = _object(result)
    for name in ("TurnResumeResponse", "TurnCompletedParams"):
        definitions[name]["properties"].update(result)

    return definitions


def _result_definitions() -> dict[str, dict[str, Any]]:
    capabilities = {
        "contextLength": nullable({"type": "integer", "minimum": 1}),
        "maxOutputTokens": nullable({"type": "integer", "minimum": 0}),
        **{
            name: {"type": ["boolean", "null"]}
            for name in ("supportsTools", "supportsVision", "supportsReasoning", "supportsPromptCache", "supportsContinuation")
        },
        "inputModalities": nullable(_array({"type": "string"})),
        "supportedReasoningEfforts": nullable(_array({"type": "string"})),
        "defaultReasoningEffort": NULLABLE_STRING,
    }
    model_summary = _object(
        {
            "id": {"type": "string"},
            "provider": {"type": "string"},
            "displayName": {"type": "string"},
            "metadata": JSON_OBJECT,
            **capabilities,
        },
        required=["id", *capabilities],
    )
    return {
        "InitializeResponse": _object(
            {
                "userAgent": {"type": "string"},
                "protocolVersion": {"const": "v4"},
                "capabilities": {"$ref": "#/$defs/ServerCapabilities"},
            },
            required=["userAgent", "protocolVersion", "capabilities"],
        ),
        "ModelSummary": model_summary,
        "ModelListResponse": _object(
            {"models": _array({"$ref": "#/$defs/ModelSummary"})},
            required=["models"],
        ),
        "ThreadStartResponse": _object(
            {
                "threadId": {"type": "string"},
                "agentKey": {"type": "string"},
                "cwd": NULLABLE_STRING,
                "status": {"type": "string"},
            },
            required=["threadId", "agentKey", "cwd", "status"],
        ),
        "ThreadSnapshotResponse": _object(
            {
                "thread": {"$ref": "#/$defs/AppThread"},
                "turns": _array({"$ref": "#/$defs/AppTurn"}),
                "items": _array({"$ref": "#/$defs/AppItem"}),
            },
            required=["thread", "turns", "items"],
        ),
        "ThreadListResponse": _object({"threads": _array({"$ref": "#/$defs/AppThread"})}, required=["threads"]),
        "ThreadArchiveResponse": _object(
            {"threadId": {"type": "string"}, "archived": {"type": "boolean"}},
            required=["threadId", "archived"],
        ),
        "ThreadUnsubscribeResponse": _object(
            {
                "threadId": {"type": "string"},
                "subscribed": {"type": "boolean"},
            },
            required=["threadId", "subscribed"],
        ),
        **{
            name: _object(
                {
                    "threadId": {"type": "string"},
                    "status": {"type": "string"},
                    "phase": {"type": "string", "enum": ["idle", "active", "parked", "suspended", "closed"]},
                    "waitKind": {"enum": [None, "provider", "child", "user", "approval", "retry"]},
                    "nextDriveAt": {"type": ["integer", "null"], "minimum": 0},
                    "pendingInteraction": nullable({"$ref": "#/$defs/Interaction"}),
                    "unconfirmedCount": {"type": "integer", "minimum": 0},
                    "projectionWatermark": {"type": "integer", "minimum": 1},
                },
                required=[
                    "threadId",
                    "status",
                    "phase",
                    "waitKind",
                    "nextDriveAt",
                    "pendingInteraction",
                    "unconfirmedCount",
                    "projectionWatermark",
                ],
            )
            for name in ("ThreadStatusResponse", "ThreadStatusChangedParams")
        },
        **{
            name: _object(
                {
                    "threadId": {"type": "string"},
                    "turnId": {"type": "string"},
                    "inputId": {"type": "string"},
                    "inputSeq": {"type": "integer", "minimum": 1},
                    "accepted": {"const": True},
                    "replayed": {"type": "boolean"},
                },
                required=["threadId", "turnId", "inputId", "inputSeq", "accepted", "replayed"],
            )
            for name in ("TurnStartResponse", "TurnQueueResponse")
        },
        "TurnResumeResponse": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "runId": {"type": "string"},
                "status": {"type": "string"},
                "finalOutput": {},
                "completionReason": {"type": "string"},
                "completionToolName": {"type": "string"},
                "partialOutput": {"type": "string"},
                "waitReason": {"type": "string"},
                "error": {"type": "string"},
            },
            required=["threadId", "turnId", "runId", "status"],
        ),
        "TurnActionResponse": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "actionId": {"type": "string"},
                "accepted": {"type": "boolean"},
                "status": {"type": "string"},
            },
            required=["threadId", "turnId", "actionId", "accepted", "status"],
        ),
        "ApprovalResolveResponse": _object({}),
        "SchemaExportResponse": _object(
            {"jsonSchema": STRING_MAP, "typescript": STRING_MAP}, required=["jsonSchema", "typescript"]
        ),
    }


def _context_definitions() -> dict[str, dict[str, Any]]:
    integer = {"type": "integer", "minimum": 0}
    count = nullable(integer)
    text = {"type": "string"}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$", "minLength": 64, "maxLength": 64}

    def closed(fields):
        return _object(fields, required=list(fields))

    notice = closed(
        {
            "threadId": text,
            "turnId": text,
            "requestId": text,
            "eventId": text,
            "changed": {"type": "boolean"},
            "error": NULLABLE_STRING,
        }
    )
    return {
        "ThreadContextReadParams": closed({"threadId": text}),
        "ThreadCompactStartParams": _object(
            {
                "threadId": text,
                "actionId": {"type": "string", "pattern": CLIENT_ID_PATTERN, "minLength": 1, "maxLength": 128},
                "expectedContextVersion": digest,
            },
            required=["threadId", "actionId"],
        ),
        "ThreadCompactStartResponse": closed(
            {
                "threadId": text,
                "actionId": text,
                "requestId": text,
                "inputSeq": {"type": "integer", "minimum": 1},
                "accepted": {"const": True},
                "replayed": {"type": "boolean"},
            }
        ),
        "ContextCompactionParams": notice,
        "ThreadContextReadResponse": closed(
            {
                "threadId": text,
                "contextVersion": digest,
                "projectionStage": {"const": "durable_pre_hooks"},
                "estimates": closed(
                    {
                        "estimated": {"const": True},
                        "estimator": {"const": "local_cjk_character_estimate_v1"},
                        "parts": closed(
                            {name: count for name in ("instructions", "tools", "history", "summary", "pendingInput")}
                        ),
                        "total": count,
                    }
                ),
                "limits": closed(
                    {
                        "contextWindow": count,
                        "maxOutputTokens": count,
                        "runtimeContextWindow": count,
                        "outputReserve": count,
                        "outputReserveSource": NULLABLE_STRING,
                        "compactionThreshold": count,
                    }
                ),
                "lastCompaction": nullable(
                    closed(
                        {"seq": integer, "turnId": text, "createdAt": integer, "replacedMessageCount": integer, "summary": text}
                    )
                ),
                "counts": closed(
                    {
                        name: integer
                        for name in (
                            "rawRecordCount",
                            "rawHistoryMessages",
                            "modelVisibleMessages",
                            "modelVisibleHistoryMessages",
                            "pendingInputs",
                        )
                    }
                ),
            }
        ),
    }


CLIENT_METHOD_SPECS: dict[str, tuple[str | None, bool, bool]] = {
    "initialize": ("InitializeParams", True, True),
    "initialized": (None, False, False),
    "model/list": ("ModelListParams", True, False),
    "thread/start": ("ThreadStartParams", True, True),
    "thread/resume": ("ThreadResumeParams", True, True),
    "thread/read": ("ThreadReadParams", True, True),
    "thread/context/read": ("ThreadContextReadParams", True, True),
    "thread/compact/start": ("ThreadCompactStartParams", True, True),
    "thread/events/list": ("ThreadEventsListParams", True, True),
    "thread/model/select": ("ThreadModelSelectParams", True, True),
    "thread/status": ("ThreadStatusParams", True, True),
    "thread/list": ("ThreadListParams", True, False),
    "thread/archive": ("ThreadIdParams", True, True),
    "thread/unsubscribe": ("ThreadIdParams", True, True),
    "turn/start": ("TurnStartParams", True, True),
    "turn/resume": ("TurnResumeParams", True, True),
    "turn/steer": ("TurnSteerParams", True, True),
    "turn/followUp": ("TurnFollowUpParams", True, True),
    "turn/interrupt": ("TurnInterruptParams", True, True),
    "turn/action": ("TurnActionParams", True, True),
    "approval/resolve": ("ApprovalResolveParams", True, True),
    "schema/export": ("EmptyParams", True, False),
}

SERVER_NOTIFICATION_SPECS: dict[str, str] = {
    "thread/started": "ThreadStartedParams",
    "thread/status/changed": "ThreadStatusChangedParams",
    "thread/archived": "ThreadArchivedParams",
    "turn/started": "TurnStartedParams",
    "item/started": "AppItem",
    "item/agentMessage/delta": "AgentMessageDeltaParams",
    "item/toolCall/delta": "ToolCallDeltaParams",
    "item/completed": "AppItem",
    "approval/requested": "ApprovalRequestParams",
    "approval/resolved": "ApprovalResolveParams",
    "error/warning": "WarningParams",
    "turn/completed": "TurnCompletedParams",
    **{f"contextCompaction/{stage}": "ContextCompactionParams" for stage in ("started", "completed", "failed")},
}

SERVER_REQUEST_SPECS: dict[str, str] = {"approval/request": "ApprovalRequestParams"}


def _method_variant(
    method: str,
    params_name: str | None,
    *,
    request: bool,
    params_required: bool = True,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "jsonrpc": {"type": "string", "const": "2.0"},
        "method": {"const": method},
    }
    required = ["jsonrpc", "method"]
    if request:
        properties["id"] = REQUEST_ID_SCHEMA
        required.insert(1, "id")
    if params_name is not None:
        properties["params"] = {"$ref": f"#/$defs/{params_name}"}
        if params_required:
            required.append("params")
    return _object(properties, required=required)


def _envelope_schema(title: str, variants: list[dict[str, Any]], definitions: dict[str, Any]) -> dict[str, Any]:
    return {"$schema": SCHEMA_URI, "title": title, "oneOf": variants, "$defs": deepcopy(definitions)}


def _schema_bundle() -> dict[str, Any]:
    definitions = _definitions()
    if set(CLIENT_METHOD_SPECS) != set(CLIENT_METHODS):
        raise RuntimeError("Client method schema registry does not match processor methods")
    if set(SERVER_NOTIFICATION_SPECS) != set(SERVER_NOTIFICATION_METHODS):
        raise RuntimeError("Server notification schema registry does not match processor methods")
    if set(SERVER_REQUEST_SPECS) != set(SERVER_REQUEST_METHODS):
        raise RuntimeError("Server request schema registry does not match processor methods")

    client_request = _envelope_schema(
        "ClientRequest",
        [
            _method_variant(method, params_name, request=request, params_required=params_required)
            for method, (params_name, request, params_required) in CLIENT_METHOD_SPECS.items()
        ],
        definitions,
    )
    server_notification = _envelope_schema(
        "ServerNotification",
        [_method_variant(method, params_name, request=False) for method, params_name in SERVER_NOTIFICATION_SPECS.items()],
        definitions,
    )
    server_request = _envelope_schema(
        "ServerRequest",
        [_method_variant(method, params_name, request=True) for method, params_name in SERVER_REQUEST_SPECS.items()],
        definitions,
    )
    success_response = _object(
        {
            "jsonrpc": {"type": "string", "const": "2.0"},
            "id": REQUEST_ID_SCHEMA,
            "result": {},
        },
        required=["jsonrpc", "id", "result"],
    )
    error_response = _object(
        {
            "jsonrpc": {"type": "string", "const": "2.0"},
            "id": {"oneOf": [*deepcopy(REQUEST_ID_SCHEMA["oneOf"]), {"type": "null"}]},
            "error": _object(
                {
                    "code": {"type": "integer"},
                    "message": {"type": "string"},
                    "data": {},
                },
                required=["code", "message"],
            ),
        },
        required=["jsonrpc", "id", "error"],
    )
    json_rpc_message = {
        "$schema": SCHEMA_URI,
        "title": "JsonRpcMessage",
        "oneOf": [
            *deepcopy(client_request["oneOf"]),
            *deepcopy(server_notification["oneOf"]),
            *deepcopy(server_request["oneOf"]),
            success_response,
            error_response,
        ],
        "$defs": deepcopy(definitions),
    }
    standalone_sources = {
        "AppItem": definitions["AppItem"],
        "AppThread": definitions["AppThread"],
        "AppTurn": definitions["AppTurn"],
        "ApprovalDecision": definitions["ApprovalDecision"],
        "ApprovalRequestParams": definitions["ApprovalRequestParams"],
        "ApprovalResolveParams": definitions["ApprovalResolveParams"],
        "InitializeParams": definitions["InitializeParams"],
        "InitializeResponse": definitions["InitializeResponse"],
        "SchemaExportResponse": definitions["SchemaExportResponse"],
        "ThreadReadResponse": definitions["ThreadSnapshotResponse"],
        "ThreadContextReadResponse": definitions["ThreadContextReadResponse"],
        "ThreadCompactStartResponse": definitions["ThreadCompactStartResponse"],
        "ThreadResumeResponse": definitions["ThreadSnapshotResponse"],
        "ThreadStartResponse": definitions["ThreadStartResponse"],
        "TurnResumeParams": definitions["TurnResumeParams"],
        "TurnResumeResponse": definitions["TurnResumeResponse"],
        "TurnStartResponse": definitions["TurnStartResponse"],
    }
    bundle: dict[str, Any] = {
        "ClientRequest": client_request,
        "JsonRpcMessage": json_rpc_message,
        "ServerNotification": server_notification,
        "ServerRequest": server_request,
    }
    for name, schema in standalone_sources.items():
        standalone = {"$schema": SCHEMA_URI, "title": name, **deepcopy(schema)}
        if _contains_ref(standalone):
            standalone["$defs"] = _referenced_definitions(standalone, definitions)
        bundle[name] = standalone
    if set(bundle) != set(JSON_SCHEMA_NAMES):
        raise RuntimeError("App Server JSON schema bundle does not match the parity contract")
    return {name: bundle[name] for name in JSON_SCHEMA_NAMES}


def _referenced_definitions(schema: Any, definitions: dict[str, Any]) -> dict[str, Any]:
    needed: dict[str, Any] = {}

    def visit(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in needed:
                    needed[name] = deepcopy(definitions[name])
                    visit(needed[name])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)
    return needed


def _contains_ref(value: Any) -> bool:
    if isinstance(value, dict):
        return "$ref" in value or any(_contains_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False


def generate_json_schema(out_dir: str | Path) -> None:
    root = Path(out_dir)
    json_dir = root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    schemas = _schema_bundle()
    for name, schema in schemas.items():
        (json_dir / f"{name}.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (json_dir / "vv_agent_app_server.schemas.json").write_text(
        json.dumps(schemas, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_typescript(out_dir: str | Path) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, source in typescript_schema_bundle().items():
        (root / name).write_text(source, encoding="utf-8")


def json_schema_bundle() -> dict[str, str]:
    return {
        name: json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n" for name, schema in _schema_bundle().items()
    }


def typescript_schema_bundle() -> dict[str, str]:
    source = _typescript_protocol_source()
    return {f"{name}.ts": source for name in TYPESCRIPT_SCHEMA_NAMES}


def export_schema_bundles() -> dict[str, dict[str, str]]:
    return {"jsonSchema": json_schema_bundle(), "typescript": typescript_schema_bundle()}


def _typescript_protocol_source() -> str:
    return (
        "// Generated by vv-agent. This file is self-contained.\n"
        "export type RequestId = string | number;\n"
        "export type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject;\n"
        "export type JsonObject = { [key: string]: JsonValue };\n"
        'export type ApprovalDecision = "allow" | "allow_session" | "deny" | "timeout";\n'
        'export type ThreadStatus = "idle" | "running" | "provider_running" | "child_running" | '
        '"user_wait" | "approval_wait" | "retry_backoff" | "suspended" | "terminal" | "archived" | '
        '"closed";\n'
        'export type TurnStatus = "queued" | "running" | "completed" | "failed" | "interrupted";\n'
        'export type AppItemStatus = "started" | "inProgress" | "completed" | "failed";\n'
        "\n"
        "export interface ClientInfo { name: string; title?: string; version?: string; }\n"
        "export interface ClientCapabilities { experimentalApi?: boolean; optOutNotificationMethods?: string[]; }\n"
        "export interface ServerCapabilities {\n"
        "  modelList: boolean; threadLifecycle: boolean; notificationOptOut: boolean;\n"
        "  schemaExport: boolean; approvalResolve: boolean;\n"
        "}\n"
        "export interface InitializeParams { clientInfo: ClientInfo; capabilities?: ClientCapabilities; }\n"
        "export interface ModelListParams { agentKey?: string; provider?: string; }\n"
        "export interface ThreadStartParams { inputId: string; agentKey?: string; cwd?: string; metadata?: JsonObject; }\n"
        "export interface ThreadIdParams { threadId: string; }\n"
        "export interface ThreadResumeParams { threadId: string; subscribe?: boolean; }\n"
        "export interface ThreadModelSelectResponse { threadId: string; inputId: string; accepted: true; }\n"
        "export interface ThreadModelSelectParams { threadId: string; inputId: string; "
        "selection: { model: string; model_settings?: Record<string, unknown>; route: Record<string, unknown> } | null; }\n"
        "export interface EventCursor { threadId: string; seq: number; offset: number; recordDigest: string; }\n"
        "export interface ThreadEventsListParams { threadId: string; afterCursor?: EventCursor; "
        "throughCursor?: EventCursor; limit: number; }\n"
        "export interface ThreadEventsListResponse { events: { cursor: EventCursor; event: JsonObject "
        "}[]; nextCursor: EventCursor; throughCursor: EventCursor; more: boolean; }\n"
        "export interface ThreadReadParams { threadId: string; }\n"
        "export interface ThreadContextReadParams { threadId: string; }\n"
        "export interface ThreadCompactStartParams { threadId: string; actionId: string; expectedContextVersion?: string; }\n"
        "export interface ThreadCompactStartResponse { threadId: string; actionId: string; requestId: "
        "string; inputSeq: number; accepted: true; replayed: boolean; }\n"
        "export interface ContextCompactionParams { threadId: string; turnId: string; requestId: string; "
        "eventId: string; changed: boolean; error: string | null; }\n"
        "export interface ThreadContextReadResponse { threadId: string; contextVersion: string; "
        "projectionStage: 'durable_pre_hooks';\n"
        "  estimates: { estimated: true; estimator: 'local_cjk_character_estimate_v1'; parts: { "
        "instructions: number | null; tools: number | null; history: number | null; summary: number | null; "
        "pendingInput: number | null; }; total: number | null; };\n"
        "  limits: { contextWindow: number | null; maxOutputTokens: number | null; runtimeContextWindow: "
        "number | null; outputReserve: number | null; outputReserveSource: string | null; "
        "compactionThreshold: number | null; };\n"
        "  lastCompaction: { seq: number; turnId: string; createdAt: number; replacedMessageCount: number; "
        "summary: string; } | null;\n"
        "  counts: { rawRecordCount: number; rawHistoryMessages: number; modelVisibleMessages: number; "
        "modelVisibleHistoryMessages: number; pendingInputs: number; }; }\n"
        "export interface ThreadListParams {\n"
        "  includeArchived?: boolean; archived?: boolean; offset?: number; limit?: number;\n"
        "}\n"
        "export type InputItem = JsonObject;\n"
        "export interface TurnStartParams { threadId: string; inputId: string; expectedTurnId: string | "
        "null; input: InputItem[]; metadata?: JsonObject; }\n"
        "export interface TurnResumeParams { threadId: string; turnId: string; }\n"
        "export interface TurnSteerParams { threadId: string; inputId: string; expectedTurnId: string; input: InputItem[]; }\n"
        "export interface TurnFollowUpParams { threadId: string; inputId: string; expectedTurnId: string; input: InputItem[]; }\n"
        "export interface TurnInterruptParams { threadId: string; actionId: string; expectedTurnId: string; }\n"
        "export interface ApprovalRequestParams {\n"
        "  requestId: string; threadId: string; turnId: string; toolCallId: string;\n"
        "  toolName: string; preview: string; arguments: JsonObject;\n"
        "}\n"
        "export interface ApprovalResolveParams {\n"
        "  requestId: string; threadId: string; turnId: string; decision: ApprovalDecision;\n"
        "  reason?: string; metadata?: JsonObject;\n"
        "}\n"
        "\n"
        "export interface AppItem {\n"
        "  itemId: string; threadId: string; turnId: string; type: string; status: AppItemStatus;\n"
        "  payload: JsonObject; createdAt: number; updatedAt: number;\n"
        "}\n"
        "export interface AppThread {\n"
        "  threadId: string; agentKey: string; cwd: string | null; createdAt: number; updatedAt: number;\n"
        "  archivedAt: number | null; status: ThreadStatus; metadata: JsonObject;\n"
        "}\n"
        "export interface AppTurn {\n"
        "  turnId: string; threadId: string; runId: string | null; status: TurnStatus;\n"
        "  startedAt: number; completedAt: number | null; input: InputItem[]; result: JsonObject;\n"
        "}\n"
        "export interface ToolCallDeltaParams extends AppItem { delta: JsonValue; }\n"
        "export interface WarningParams { message: string; code?: string; }\n"
        'export interface InitializeResponse { userAgent: string; protocolVersion: "v4"; '
        "capabilities: ServerCapabilities; }\n"
        "export interface ModelSummary {\n"
        "  id: string; provider?: string; displayName?: string; contextLength: number | null;\n"
        "  supportsTools: boolean | null; maxOutputTokens: number | null; supportsVision: boolean | null;\n"
        "  inputModalities: string[] | null; supportsReasoning: boolean | null; supportedReasoningEfforts: string[] | null;\n"
        "  defaultReasoningEffort: string | null; supportsPromptCache: boolean | null; supportsContinuation: "
        "boolean | null; metadata?: JsonObject;\n"
        "}\n"
        "export interface ModelListResponse { models: ModelSummary[]; }\n"
        "export interface ThreadStartResponse { threadId: string; agentKey: string; cwd: string | "
        "null; status: ThreadStatus; accepted: true; replayed: boolean; }\n"
        "export interface ThreadReadResponse { thread: AppThread; turns: AppTurn[]; items: AppItem[]; }\n"
        "export interface ThreadResumeResponse { thread: AppThread; turns: AppTurn[]; items: AppItem[]; }\n"
        "export interface ThreadListResponse { threads: AppThread[]; }\n"
        "export interface ThreadArchiveResponse { threadId: string; archived: boolean; }\n"
        "export interface ThreadUnsubscribeResponse { threadId: string; subscribed: boolean; }\n"
        "export interface ThreadStatusResponse { threadId: string; status: ThreadStatus; "
        'phase: string; waitKind: "provider" | "child" | "user" | "approval" | "retry" | null; '
        "nextDriveAt: number | null; pendingInteraction: Interaction | null; unconfirmedCount: number; "
        "projectionWatermark: number; }\n"
        "export interface TurnStartResponse { threadId: string; turnId: string; inputId: string; "
        "inputSeq: number; accepted: true; replayed: boolean; }\n"
        "export interface TurnResumeResponse {\n"
        "  threadId: string; turnId: string; runId: string; status: TurnStatus; finalOutput?: JsonValue;\n"
        "  completionReason?: string; completionToolName?: string; partialOutput?: string; waitReason?: string;\n"
        "  error?: string; tokenUsage?: TaskTokenUsage; budgetUsage?: JsonObject; budgetExhaustion?: JsonObject;\n"
        "}\n"
        "export interface TurnQueueResponse extends TurnStartResponse {}\n"
        "export interface TurnInterruptResponse extends TurnActionResponse {}\n"
        "export type TurnAction =\n"
        '  | { kind: "respond"; message: { role: "user"; content: string } }\n'
        '  | { kind: "suspend" } | { kind: "resume" } | { kind: "cancel" } | { kind: "abort" };\n'
        "export interface TurnActionParams { threadId: string; turnId: string; actionId: string; action: TurnAction; }\n"
        "export interface TurnActionResponse {\n"
        "  threadId: string; turnId: string; actionId: string; accepted: boolean;\n"
        "  status: TurnStatus;\n"
        "}\n"
        "export interface ThreadStatusChangedParams { threadId: string; status: ThreadStatus; "
        'phase: string; waitKind: "provider" | "child" | "user" | "approval" | "retry" | null; '
        "nextDriveAt: number | null; pendingInteraction: Interaction | null; unconfirmedCount: number; "
        "projectionWatermark: number; }\n"
        "export interface TurnStartedParams { threadId: string; turnId: string; runId?: string; status?: TurnStatus; }\n"
        "export interface TurnCompletedParams {\n"
        "  threadId: string; turnId: string; runId?: string; status: TurnStatus; finalOutput?: JsonValue;\n"
        "  completionReason?: string; completionToolName?: string; partialOutput?: string; waitReason?: string;\n"
        "  tokenUsage?: TaskTokenUsage; budgetUsage?: JsonObject; budgetExhaustion?: JsonObject;\n"
        "  error?: string;\n"
        "}\n"
        "export type ApprovalResolveResponse = Record<string, never>;\n"
        "export interface SchemaExportResponse { jsonSchema: Record<string, string>; typescript: "
        "Record<string, string>; }\n"
        "\n"
        "export type ClientRequest =\n"
        '  | { jsonrpc: "2.0"; id: RequestId; method: "initialize"; params: InitializeParams }\n'
        '  | { jsonrpc: "2.0"; method: "initialized" }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "model/list"; params?: ModelListParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/start"; params: ThreadStartParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/resume"; params: ThreadResumeParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/model/select"; params: ThreadModelSelectParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/read"; params: ThreadReadParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/context/read"; params: ThreadContextReadParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/compact/start"; params: ThreadCompactStartParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/events/list"; params: ThreadEventsListParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/status"; params: ThreadIdParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/archive" | "thread/unsubscribe"; '
        "params: ThreadIdParams }\n"
        '  | { jsonrpc: "2.0"; id: RequestId; method: "thread/list"; params?: ThreadListParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/start"; params: TurnStartParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/resume"; params: TurnResumeParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/steer"; params: TurnSteerParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/followUp"; params: TurnFollowUpParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/interrupt"; params: TurnInterruptParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "turn/action"; params: TurnActionParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "approval/resolve"; params: ApprovalResolveParams }\n'
        '  | { jsonrpc: "2.0"; id: RequestId; method: "schema/export"; params?: Record<string, never> };\n'
        "\n"
        "export type ServerNotification =\n"
        '  | { jsonrpc: "2.0"; method: "thread/started"; params: ThreadStartResponse }\n'
        '  | { jsonrpc: "2.0"; method: "thread/status/changed"; params: ThreadStatusChangedParams }\n'
        '  | { jsonrpc: "2.0"; method: "thread/archived"; params: ThreadArchiveResponse }\n'
        '  | { jsonrpc: "2.0"; method: "turn/started"; params: TurnStartedParams }\n'
        '  | { jsonrpc: "2.0"; method: "turn/completed"; params: TurnCompletedParams }\n'
        '  | { jsonrpc: "2.0"; method: "item/started" | "item/completed"; params: AppItem }\n'
        '  | { jsonrpc: "2.0"; method: "item/agentMessage/delta"; params: AppItem & { delta: string } }\n'
        '  | { jsonrpc: "2.0"; method: "item/toolCall/delta"; params: ToolCallDeltaParams }\n'
        '  | { jsonrpc: "2.0"; method: "approval/requested"; params: ApprovalRequestParams }\n'
        '  | { jsonrpc: "2.0"; method: "approval/resolved"; params: ApprovalResolveParams }\n'
        '  | { jsonrpc: "2.0"; method: "contextCompaction/started" | "contextCompaction/completed" | '
        '"contextCompaction/failed"; params: ContextCompactionParams }\n'
        '  | { jsonrpc: "2.0"; method: "error/warning"; params: WarningParams };\n'
        "\n"
        "export type ServerRequest = {\n"
        '  jsonrpc: "2.0"; id: RequestId; method: "approval/request"; params: ApprovalRequestParams;\n'
        "};\n"
        "export type ClientResult =\n"
        "  | ThreadContextReadResponse | ThreadCompactStartResponse | InitializeResponse | ModelListResponse "
        "| ThreadStartResponse | ThreadReadResponse\n"
        "  | ThreadModelSelectResponse | ThreadResumeResponse | ThreadListResponse | ThreadArchiveResponse | "
        "ThreadUnsubscribeResponse | ThreadStatusResponse\n"
        "  | TurnStartResponse | TurnResumeResponse | TurnQueueResponse | TurnInterruptResponse | "
        "ApprovalResolveResponse\n"
        "  | SchemaExportResponse;\n"
        'export type JsonRpcSuccess = { jsonrpc: "2.0"; id: RequestId; result: ClientResult | JsonValue };\n'
        "export type JsonRpcError = {\n"
        '  jsonrpc: "2.0"; id: RequestId | null;\n'
        "  error: { code: number; message: string; data?: JsonValue };\n"
        "};\n"
        "export type JsonRpcMessage = ClientRequest | ServerNotification | ServerRequest | "
        "JsonRpcSuccess | JsonRpcError;\n"
        "\n"
        "export interface Interaction { sessionId: string; turnId: string; interactionId?: string; "
        "requestId?: string; prompt?: string; }\n"
        "\n"
        "export interface CacheUsage {\n"
        '  status: "provider_reported" | "accounting_missing" | "unsupported";\n'
        "  readInputTokens: number | null; writeInputTokens: number | null;\n"
        "  uncachedInputTokens: number | null; source: string | null;\n"
        "}\n"
        "export interface TokenUsage {\n"
        '  schemaVersion: "vv-agent.token-usage.v1"; inputTokens: number | null;\n'
        "  outputTokens: number | null; totalTokens: number | null; reasoningTokens: number | null;\n"
        '  usageSource: "provider_reported" | "estimated" | "accounting_missing";\n'
        "  cacheUsage: CacheUsage; providerUsage: JsonObject;\n"
        "}\n"
        "export interface ModelCallRecord {\n"
        '  schemaVersion: "vv-agent.model-call.v2"; callId: string; operationId: string;\n'
        "  attempt: number; cycleIndex: number; backend: string; model: string;\n"
        '  operation: "agent_cycle" | "memory_compaction" | "session_memory" | "output_repair";\n'
        '  status: "completed" | "failed" | "ambiguous"; usage: TokenUsage; errorCode: string | null;\n'
        "}\n"
        "export interface TaskTokenUsage {\n"
        '  schemaVersion: "vv-agent.task-token-usage.v3"; inputTokens: number | null;\n'
        "  outputTokens: number | null; totalTokens: number | null; reasoningTokens: number | null;\n"
        "  cacheUsage: CacheUsage; modelCalls: ModelCallRecord[];\n"
        "}\n"
        "export interface TurnResult {\n"
        "  finalOutput?: JsonValue; waitReason?: string; completionReason?: string;\n"
        "  completionToolName?: string; partialOutput?: JsonValue; error?: string;\n"
        "  tokenUsage?: TaskTokenUsage; budgetUsage?: JsonObject; budgetExhaustion?: JsonObject;\n"
        "}\n"
    )
