from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vv_agent.app_server.client import AppServerClient as AppServerClient
    from vv_agent.app_server.client import AppServerClientError as AppServerClientError
    from vv_agent.app_server.host import AgentResolutionRequest as AgentResolutionRequest
    from vv_agent.app_server.host import AppServerApprovalProvider as AppServerApprovalProvider
    from vv_agent.app_server.host import AppServerHost as AppServerHost
    from vv_agent.app_server.host import DefaultAppServerHost as DefaultAppServerHost
    from vv_agent.app_server.host import RunConfigResolutionRequest as RunConfigResolutionRequest
    from vv_agent.app_server.host import ToolResultResourceAuthorizationRequest as ToolResultResourceAuthorizationRequest
    from vv_agent.app_server.outgoing import OutgoingRouter as OutgoingRouter
    from vv_agent.app_server.outgoing import PendingServerRequest as PendingServerRequest
    from vv_agent.app_server.processor import MessageProcessor as MessageProcessor
    from vv_agent.app_server.protocol import ApprovalDecision as ApprovalDecision
    from vv_agent.app_server.protocol import ApprovalRequestParams as ApprovalRequestParams
    from vv_agent.app_server.protocol import ApprovalResolveParams as ApprovalResolveParams
    from vv_agent.app_server.protocol import AppServerError as AppServerError
    from vv_agent.app_server.protocol import AppServerErrorCode as AppServerErrorCode
    from vv_agent.app_server.protocol import ClientInfo as ClientInfo
    from vv_agent.app_server.protocol import InitializeParams as InitializeParams
    from vv_agent.app_server.protocol import InitializeResponse as InitializeResponse
    from vv_agent.app_server.protocol import JsonRpcError as JsonRpcError
    from vv_agent.app_server.protocol import JsonRpcMessage as JsonRpcMessage
    from vv_agent.app_server.protocol import JsonRpcNotification as JsonRpcNotification
    from vv_agent.app_server.protocol import JsonRpcRequest as JsonRpcRequest
    from vv_agent.app_server.protocol import JsonRpcResponse as JsonRpcResponse
    from vv_agent.app_server.protocol import ModelListRequest as ModelListRequest
    from vv_agent.app_server.protocol import ModelListResponse as ModelListResponse
    from vv_agent.app_server.protocol import ModelSummary as ModelSummary
    from vv_agent.app_server.protocol import RequestId as RequestId
    from vv_agent.app_server.protocol import ThreadArchiveParams as ThreadArchiveParams
    from vv_agent.app_server.protocol import ThreadCompactStartParams as ThreadCompactStartParams
    from vv_agent.app_server.protocol import ThreadContextReadParams as ThreadContextReadParams
    from vv_agent.app_server.protocol import ThreadEventsListParams as ThreadEventsListParams
    from vv_agent.app_server.protocol import ThreadForkParams as ThreadForkParams
    from vv_agent.app_server.protocol import ThreadItem as ThreadItem
    from vv_agent.app_server.protocol import ThreadListParams as ThreadListParams
    from vv_agent.app_server.protocol import ThreadModelSelectParams as ThreadModelSelectParams
    from vv_agent.app_server.protocol import ThreadReadParams as ThreadReadParams
    from vv_agent.app_server.protocol import ThreadResumeParams as ThreadResumeParams
    from vv_agent.app_server.protocol import ThreadStartParams as ThreadStartParams
    from vv_agent.app_server.protocol import ThreadStatusParams as ThreadStatusParams
    from vv_agent.app_server.protocol import ThreadUnsubscribeParams as ThreadUnsubscribeParams
    from vv_agent.app_server.protocol import TurnActionParams as TurnActionParams
    from vv_agent.app_server.protocol import TurnActionResponse as TurnActionResponse
    from vv_agent.app_server.protocol import TurnFollowUpParams as TurnFollowUpParams
    from vv_agent.app_server.protocol import TurnInterruptParams as TurnInterruptParams
    from vv_agent.app_server.protocol import TurnResumeParams as TurnResumeParams
    from vv_agent.app_server.protocol import TurnResumeResponse as TurnResumeResponse
    from vv_agent.app_server.protocol import TurnStartParams as TurnStartParams
    from vv_agent.app_server.protocol import TurnSteerParams as TurnSteerParams
    from vv_agent.app_server.protocol import WarningParams as WarningParams
    from vv_agent.app_server.protocol.tool_result import ToolResultReadParams as ToolResultReadParams
    from vv_agent.app_server.protocol.tool_result import ToolResultReadResponse as ToolResultReadResponse
    from vv_agent.app_server.protocol.tool_result import ToolResultRef as ToolResultRef
    from vv_agent.app_server.protocol.tool_result import ToolResultSummary as ToolResultSummary
    from vv_agent.app_server.server import AppServer as AppServer
    from vv_agent.app_server.transport import ChannelTransport as ChannelTransport
    from vv_agent.app_server.transport import StdioJsonlTransport as StdioJsonlTransport

_EXPORT_MODULES = {
    "AgentResolutionRequest": "vv_agent.app_server.host",
    "AppServer": "vv_agent.app_server.server",
    "AppServerApprovalProvider": "vv_agent.app_server.host",
    "AppServerClient": "vv_agent.app_server.client",
    "AppServerClientError": "vv_agent.app_server.client",
    "AppServerError": "vv_agent.app_server.protocol",
    "AppServerErrorCode": "vv_agent.app_server.protocol",
    "AppServerHost": "vv_agent.app_server.host",
    "ApprovalDecision": "vv_agent.app_server.protocol",
    "ApprovalRequestParams": "vv_agent.app_server.protocol",
    "ApprovalResolveParams": "vv_agent.app_server.protocol",
    "ChannelTransport": "vv_agent.app_server.transport",
    "ClientInfo": "vv_agent.app_server.protocol",
    "DefaultAppServerHost": "vv_agent.app_server.host",
    "InitializeParams": "vv_agent.app_server.protocol",
    "InitializeResponse": "vv_agent.app_server.protocol",
    "JsonRpcError": "vv_agent.app_server.protocol",
    "JsonRpcMessage": "vv_agent.app_server.protocol",
    "JsonRpcNotification": "vv_agent.app_server.protocol",
    "JsonRpcRequest": "vv_agent.app_server.protocol",
    "JsonRpcResponse": "vv_agent.app_server.protocol",
    "MessageProcessor": "vv_agent.app_server.processor",
    "ModelListRequest": "vv_agent.app_server.protocol",
    "ModelListResponse": "vv_agent.app_server.protocol",
    "ModelSummary": "vv_agent.app_server.protocol",
    "OutgoingRouter": "vv_agent.app_server.outgoing",
    "PendingServerRequest": "vv_agent.app_server.outgoing",
    "RequestId": "vv_agent.app_server.protocol",
    "RunConfigResolutionRequest": "vv_agent.app_server.host",
    "StdioJsonlTransport": "vv_agent.app_server.transport",
    "ThreadArchiveParams": "vv_agent.app_server.protocol",
    "ThreadCompactStartParams": "vv_agent.app_server.protocol",
    "ThreadContextReadParams": "vv_agent.app_server.protocol",
    "ThreadEventsListParams": "vv_agent.app_server.protocol",
    "ThreadForkParams": "vv_agent.app_server.protocol",
    "ThreadItem": "vv_agent.app_server.protocol",
    "ThreadListParams": "vv_agent.app_server.protocol",
    "ThreadModelSelectParams": "vv_agent.app_server.protocol",
    "ThreadReadParams": "vv_agent.app_server.protocol",
    "ThreadResumeParams": "vv_agent.app_server.protocol",
    "ThreadStartParams": "vv_agent.app_server.protocol",
    "ThreadStatusParams": "vv_agent.app_server.protocol",
    "ThreadUnsubscribeParams": "vv_agent.app_server.protocol",
    "ToolResultReadParams": "vv_agent.app_server.protocol.tool_result",
    "ToolResultReadResponse": "vv_agent.app_server.protocol.tool_result",
    "ToolResultRef": "vv_agent.app_server.protocol.tool_result",
    "ToolResultResourceAuthorizationRequest": "vv_agent.app_server.host",
    "ToolResultSummary": "vv_agent.app_server.protocol.tool_result",
    "TurnActionParams": "vv_agent.app_server.protocol",
    "TurnActionResponse": "vv_agent.app_server.protocol",
    "TurnFollowUpParams": "vv_agent.app_server.protocol",
    "TurnInterruptParams": "vv_agent.app_server.protocol",
    "TurnResumeParams": "vv_agent.app_server.protocol",
    "TurnResumeResponse": "vv_agent.app_server.protocol",
    "TurnStartParams": "vv_agent.app_server.protocol",
    "TurnSteerParams": "vv_agent.app_server.protocol",
    "WarningParams": "vv_agent.app_server.protocol",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
