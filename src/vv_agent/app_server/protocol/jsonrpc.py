from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vv_agent.app_server.protocol.errors import AppServerError

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JSON_RPC_VERSION = "2.0"
_MIN_REQUEST_ID = -(2**63)
_MAX_REQUEST_ID = 2**63 - 1


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str]) -> None:
    if set(payload) - allowed:
        raise ValueError("Invalid JSON-RPC message")


def _request_id(value: Any, *, allow_null: bool = False) -> RequestId:
    if value is None and allow_null:
        return RequestId(None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int))
        or (isinstance(value, int) and not _MIN_REQUEST_ID <= value <= _MAX_REQUEST_ID)
    ):
        raise ValueError("Invalid JSON-RPC message")
    return RequestId(value)


@dataclass(frozen=True, slots=True)
class RequestId:
    value: str | int | None

    def to_wire(self) -> str | int | None:
        return self.value

    def require_wire(self) -> str | int:
        if self.value is None:
            raise ValueError("JSON-RPC request id cannot be null")
        return self.value


@dataclass(frozen=True, slots=True)
class JsonRpcRequest:
    id: RequestId
    method: str
    params: JsonValue = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSON_RPC_VERSION, "id": self.id.require_wire(), "method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True, slots=True)
class JsonRpcNotification:
    method: str
    params: JsonValue = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSON_RPC_VERSION, "method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True, slots=True)
class JsonRpcResponse:
    id: RequestId
    result: JsonValue

    def to_dict(self) -> dict[str, Any]:
        return {"jsonrpc": JSON_RPC_VERSION, "id": self.id.require_wire(), "result": self.result}


@dataclass(frozen=True, slots=True)
class JsonRpcError:
    id: RequestId
    error: AppServerError

    def to_dict(self) -> dict[str, Any]:
        return {"jsonrpc": JSON_RPC_VERSION, "id": self.id.to_wire(), "error": self.error.to_dict()}


@dataclass(frozen=True, slots=True)
class JsonRpcMessage:
    message: JsonRpcRequest | JsonRpcNotification | JsonRpcResponse | JsonRpcError

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JsonRpcMessage:
        if not isinstance(payload, dict) or payload.get("jsonrpc") != JSON_RPC_VERSION:
            raise ValueError("Invalid JSON-RPC message")
        if "method" in payload and "id" in payload:
            if "result" in payload or "error" in payload:
                raise ValueError("Invalid JSON-RPC message")
            method = payload["method"]
            if not isinstance(method, str) or not method:
                raise ValueError("Invalid JSON-RPC message")
            _reject_unknown_fields(payload, {"jsonrpc", "id", "method", "params"})
            return cls(JsonRpcRequest(id=_request_id(payload["id"]), method=method, params=payload.get("params")))
        if "method" in payload:
            if "result" in payload or "error" in payload:
                raise ValueError("Invalid JSON-RPC message")
            method = payload["method"]
            if not isinstance(method, str) or not method:
                raise ValueError("Invalid JSON-RPC message")
            _reject_unknown_fields(payload, {"jsonrpc", "method", "params"})
            return cls(JsonRpcNotification(method=method, params=payload.get("params")))
        if "result" in payload and "error" not in payload and "id" in payload:
            _reject_unknown_fields(payload, {"jsonrpc", "id", "result"})
            return cls(JsonRpcResponse(id=_request_id(payload["id"]), result=payload["result"]))
        if "error" in payload and "result" not in payload and "id" in payload:
            _reject_unknown_fields(payload, {"jsonrpc", "id", "error"})
            raw_error = payload["error"]
            if not isinstance(raw_error, dict):
                raise ValueError("Invalid JSON-RPC message")
            code = raw_error.get("code")
            message = raw_error.get("message")
            if isinstance(code, bool) or not isinstance(code, int) or not isinstance(message, str):
                raise ValueError("Invalid JSON-RPC message")
            return cls(
                JsonRpcError(
                    id=_request_id(payload["id"], allow_null=True),
                    error=AppServerError(
                        code=code,
                        message=message,
                        data=raw_error.get("data"),
                    ),
                )
            )
        raise ValueError("Invalid JSON-RPC message")
