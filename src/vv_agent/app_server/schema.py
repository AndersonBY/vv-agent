from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from vv_agent.app_server.processor import CLIENT_METHODS, SERVER_NOTIFICATION_METHODS, SERVER_REQUEST_METHODS
from vv_agent.app_server.protocol.approval import ApprovalDecision

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
    "ThreadReadResponse",
    "ThreadResumeResponse",
    "ThreadStartResponse",
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
    definitions = {
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
        "ThreadStartParams": _object({"agentKey": {"type": "string"}, "cwd": {"type": "string"}, "metadata": JSON_OBJECT}),
        "ThreadIdParams": _object({"threadId": {"type": "string"}}, required=["threadId"]),
        "ThreadResumeParams": _object(
            {"threadId": {"type": "string"}, "subscribe": {"type": "boolean"}},
            required=["threadId"],
        ),
        "ThreadReadParams": _object(
            {"threadId": {"type": "string"}, "afterItemId": {"type": "string"}},
            required=["threadId"],
        ),
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
                "input": _array({"$ref": "#/$defs/InputItem"}),
                "metadata": JSON_OBJECT,
            },
            required=["threadId"],
        ),
        "TurnSteerParams": _object(
            {
                "threadId": {"type": "string"},
                "expectedTurnId": {"type": "string"},
                "input": _array({"$ref": "#/$defs/InputItem"}),
            },
            required=["threadId"],
        ),
        "TurnFollowUpParams": _object(
            {
                "threadId": {"type": "string"},
                "expectedTurnId": {"type": "string"},
                "input": _array({"$ref": "#/$defs/InputItem"}),
            },
            required=["threadId"],
        ),
        "TurnInterruptParams": _object(
            {
                "threadId": {"type": "string"},
                "expectedTurnId": {"type": "string"},
                "reason": {"type": "string"},
            },
            required=["threadId"],
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
        "ThreadStatusChangedParams": _object(
            {"threadId": {"type": "string"}, "status": {"type": "string"}},
            required=["threadId", "status"],
        ),
        "ThreadArchivedParams": _object(
            {"threadId": {"type": "string"}, "archived": {"type": "boolean"}},
            required=["threadId", "archived"],
        ),
        "ThreadClosedParams": _object({"threadId": {"type": "string"}}, required=["threadId"]),
        "TurnStartedParams": _object(
            {"threadId": {"type": "string"}, "turnId": {"type": "string"}},
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
                "tokenUsage": JSON_OBJECT,
                "error": {"type": "string"},
            },
            required=["threadId", "turnId", "status"],
        ),
    }
    definitions.update(_result_definitions())
    return definitions


def _result_definitions() -> dict[str, dict[str, Any]]:
    model_summary = _object(
        {
            "id": {"type": "string"},
            "provider": {"type": "string"},
            "displayName": {"type": "string"},
            "contextLength": {"type": "integer", "minimum": 0},
            "supportsTools": {"type": "boolean"},
            "metadata": JSON_OBJECT,
        },
        required=["id", "supportsTools"],
    )
    return {
        "InitializeResponse": _object(
            {
                "userAgent": {"type": "string"},
                "protocolVersion": {"const": "v1"},
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
                "closed": {"type": "boolean"},
            },
            required=["threadId", "subscribed", "closed"],
        ),
        "TurnStartResponse": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "status": {"type": "string"},
            },
            required=["threadId", "turnId", "status"],
        ),
        "TurnQueueResponse": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "queued": {"type": "boolean"},
            },
            required=["threadId", "turnId", "queued"],
        ),
        "TurnInterruptResponse": _object(
            {
                "threadId": {"type": "string"},
                "turnId": {"type": "string"},
                "cancelled": {"type": "boolean"},
            },
            required=["threadId", "turnId", "cancelled"],
        ),
        "ApprovalResolveResponse": _object({}),
        "SchemaExportResponse": _object(
            {"jsonSchema": STRING_MAP, "typescript": STRING_MAP}, required=["jsonSchema", "typescript"]
        ),
    }


CLIENT_METHOD_SPECS: dict[str, tuple[str | None, bool, bool]] = {
    "initialize": ("InitializeParams", True, True),
    "initialized": (None, False, False),
    "model/list": ("ModelListParams", True, False),
    "thread/start": ("ThreadStartParams", True, False),
    "thread/resume": ("ThreadResumeParams", True, True),
    "thread/read": ("ThreadReadParams", True, True),
    "thread/list": ("ThreadListParams", True, False),
    "thread/archive": ("ThreadIdParams", True, True),
    "thread/unsubscribe": ("ThreadIdParams", True, True),
    "turn/start": ("TurnStartParams", True, True),
    "turn/steer": ("TurnSteerParams", True, True),
    "turn/followUp": ("TurnFollowUpParams", True, True),
    "turn/interrupt": ("TurnInterruptParams", True, True),
    "approval/resolve": ("ApprovalResolveParams", True, True),
    "schema/export": ("EmptyParams", True, False),
}

SERVER_NOTIFICATION_SPECS: dict[str, str] = {
    "thread/started": "ThreadStartedParams",
    "thread/status/changed": "ThreadStatusChangedParams",
    "thread/archived": "ThreadArchivedParams",
    "thread/closed": "ThreadClosedParams",
    "turn/started": "TurnStartedParams",
    "item/started": "AppItem",
    "item/agentMessage/delta": "AgentMessageDeltaParams",
    "item/toolCall/delta": "ToolCallDeltaParams",
    "item/completed": "AppItem",
    "approval/requested": "ApprovalRequestParams",
    "approval/resolved": "ApprovalResolveParams",
    "error/warning": "WarningParams",
    "turn/completed": "TurnCompletedParams",
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
        "ThreadResumeResponse": definitions["ThreadSnapshotResponse"],
        "ThreadStartResponse": definitions["ThreadStartResponse"],
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
            standalone["$defs"] = deepcopy(definitions)
        bundle[name] = standalone
    if set(bundle) != set(JSON_SCHEMA_NAMES):
        raise RuntimeError("App Server JSON schema bundle does not match the parity contract")
    return {name: bundle[name] for name in JSON_SCHEMA_NAMES}


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
    return """// Generated by vv-agent. This file is self-contained.
export type RequestId = string | number;
export type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };
export type ApprovalDecision = "allow" | "allow_session" | "deny" | "timeout";
export type ThreadStatus = "idle" | "running" | "archived" | "closed";
export type TurnStatus = "queued" | "running" | "completed" | "failed" | "interrupted";
export type AppItemStatus = "started" | "inProgress" | "completed" | "failed";

export interface ClientInfo { name: string; title?: string; version?: string; }
export interface ClientCapabilities { experimentalApi?: boolean; optOutNotificationMethods?: string[]; }
export interface ServerCapabilities {
  modelList: boolean; threadLifecycle: boolean; notificationOptOut: boolean;
  schemaExport: boolean; approvalResolve: boolean;
}
export interface InitializeParams { clientInfo: ClientInfo; capabilities?: ClientCapabilities; }
export interface ModelListParams { agentKey?: string; provider?: string; }
export interface ThreadStartParams { agentKey?: string; cwd?: string; metadata?: JsonObject; }
export interface ThreadIdParams { threadId: string; }
export interface ThreadResumeParams { threadId: string; subscribe?: boolean; }
export interface ThreadReadParams { threadId: string; afterItemId?: string; }
export interface ThreadListParams {
  includeArchived?: boolean; archived?: boolean; offset?: number; limit?: number;
}
export type InputItem = JsonObject;
export interface TurnStartParams { threadId: string; input?: InputItem[]; metadata?: JsonObject; }
export interface TurnSteerParams { threadId: string; expectedTurnId?: string; input?: InputItem[]; }
export interface TurnFollowUpParams { threadId: string; expectedTurnId?: string; input?: InputItem[]; }
export interface TurnInterruptParams { threadId: string; expectedTurnId?: string; reason?: string; }
export interface ApprovalRequestParams {
  requestId: string; threadId: string; turnId: string; toolCallId: string;
  toolName: string; preview: string; arguments: JsonObject;
}
export interface ApprovalResolveParams {
  requestId: string; threadId: string; turnId: string; decision: ApprovalDecision;
  reason?: string; metadata?: JsonObject;
}

export interface AppItem {
  itemId: string; threadId: string; turnId: string; type: string; status: AppItemStatus;
  payload: JsonObject; createdAt: number; updatedAt: number;
}
export interface AppThread {
  threadId: string; agentKey: string; cwd: string | null; createdAt: number; updatedAt: number;
  archivedAt: number | null; status: ThreadStatus; metadata: JsonObject;
}
export interface AppTurn {
  turnId: string; threadId: string; runId: string | null; status: TurnStatus;
  startedAt: number; completedAt: number | null; input: InputItem[]; result: JsonObject;
}
export interface ToolCallDeltaParams extends AppItem { delta: JsonValue; }
export interface WarningParams { message: string; code?: string; }
export interface InitializeResponse { userAgent: string; protocolVersion: "v1"; capabilities: ServerCapabilities; }
export interface ModelSummary {
  id: string; provider?: string; displayName?: string; contextLength?: number;
  supportsTools: boolean; metadata?: JsonObject;
}
export interface ModelListResponse { models: ModelSummary[]; }
export interface ThreadStartResponse { threadId: string; agentKey: string; cwd: string | null; status: ThreadStatus; }
export interface ThreadReadResponse { thread: AppThread; turns: AppTurn[]; items: AppItem[]; }
export interface ThreadResumeResponse { thread: AppThread; turns: AppTurn[]; items: AppItem[]; }
export interface ThreadListResponse { threads: AppThread[]; }
export interface ThreadArchiveResponse { threadId: string; archived: boolean; }
export interface ThreadUnsubscribeResponse { threadId: string; subscribed: boolean; closed: boolean; }
export interface TurnStartResponse { threadId: string; turnId: string; status: TurnStatus; }
export interface TurnQueueResponse { threadId: string; turnId: string; queued: boolean; }
export interface TurnInterruptResponse { threadId: string; turnId: string; cancelled: boolean; }
export interface ThreadStatusChangedParams { threadId: string; status: ThreadStatus; }
export interface ThreadClosedParams { threadId: string; }
export interface TurnStartedParams { threadId: string; turnId: string; }
export interface TurnCompletedParams {
  threadId: string; turnId: string; runId?: string; status: TurnStatus; finalOutput?: JsonValue;
  completionReason?: string; completionToolName?: string; partialOutput?: string;
  tokenUsage?: JsonObject; error?: string;
}
export type ApprovalResolveResponse = Record<string, never>;
export interface SchemaExportResponse { jsonSchema: Record<string, string>; typescript: Record<string, string>; }

export type ClientRequest =
  | { jsonrpc: "2.0"; id: RequestId; method: "initialize"; params: InitializeParams }
  | { jsonrpc: "2.0"; method: "initialized" }
  | { jsonrpc: "2.0"; id: RequestId; method: "model/list"; params?: ModelListParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "thread/start"; params?: ThreadStartParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "thread/resume"; params: ThreadResumeParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "thread/read"; params: ThreadReadParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "thread/archive" | "thread/unsubscribe"; params: ThreadIdParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "thread/list"; params?: ThreadListParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "turn/start"; params: TurnStartParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "turn/steer"; params: TurnSteerParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "turn/followUp"; params: TurnFollowUpParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "turn/interrupt"; params: TurnInterruptParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "approval/resolve"; params: ApprovalResolveParams }
  | { jsonrpc: "2.0"; id: RequestId; method: "schema/export"; params?: Record<string, never> };

export type ServerNotification =
  | { jsonrpc: "2.0"; method: "thread/started"; params: ThreadStartResponse }
  | { jsonrpc: "2.0"; method: "thread/status/changed"; params: ThreadStatusChangedParams }
  | { jsonrpc: "2.0"; method: "thread/archived"; params: ThreadArchiveResponse }
  | { jsonrpc: "2.0"; method: "thread/closed"; params: ThreadClosedParams }
  | { jsonrpc: "2.0"; method: "turn/started"; params: TurnStartedParams }
  | { jsonrpc: "2.0"; method: "turn/completed"; params: TurnCompletedParams }
  | { jsonrpc: "2.0"; method: "item/started" | "item/completed"; params: AppItem }
  | { jsonrpc: "2.0"; method: "item/agentMessage/delta"; params: AppItem & { delta: string } }
  | { jsonrpc: "2.0"; method: "item/toolCall/delta"; params: ToolCallDeltaParams }
  | { jsonrpc: "2.0"; method: "approval/requested"; params: ApprovalRequestParams }
  | { jsonrpc: "2.0"; method: "approval/resolved"; params: ApprovalResolveParams }
  | { jsonrpc: "2.0"; method: "error/warning"; params: WarningParams };

export type ServerRequest = {
  jsonrpc: "2.0"; id: RequestId; method: "approval/request"; params: ApprovalRequestParams;
};
export type ClientResult =
  | InitializeResponse | ModelListResponse | ThreadStartResponse | ThreadReadResponse
  | ThreadResumeResponse | ThreadListResponse | ThreadArchiveResponse | ThreadUnsubscribeResponse
  | TurnStartResponse | TurnQueueResponse | TurnInterruptResponse | ApprovalResolveResponse
  | SchemaExportResponse;
export type JsonRpcSuccess = { jsonrpc: "2.0"; id: RequestId; result: ClientResult | JsonValue };
export type JsonRpcError = {
  jsonrpc: "2.0"; id: RequestId | null;
  error: { code: number; message: string; data?: JsonValue };
};
export type JsonRpcMessage = ClientRequest | ServerNotification | ServerRequest | JsonRpcSuccess | JsonRpcError;
"""
