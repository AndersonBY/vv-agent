from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vv_agent.app_server.protocol.errors import AppServerError

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RequestId:
    value: str | int

    def to_wire(self) -> str | int:
        return self.value


@dataclass(frozen=True, slots=True)
class JsonRpcRequest:
    id: RequestId
    method: str
    params: JsonValue = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id.to_wire(), "method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True, slots=True)
class JsonRpcNotification:
    method: str
    params: JsonValue = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True, slots=True)
class JsonRpcResponse:
    id: RequestId
    result: JsonValue

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id.to_wire(), "result": self.result}


@dataclass(frozen=True, slots=True)
class JsonRpcError:
    id: RequestId
    error: AppServerError

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id.to_wire(), "error": self.error.to_dict()}


@dataclass(frozen=True, slots=True)
class JsonRpcMessage:
    message: JsonRpcRequest | JsonRpcNotification | JsonRpcResponse | JsonRpcError

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JsonRpcMessage:
        if "method" in payload and "id" in payload:
            return cls(JsonRpcRequest(id=RequestId(payload["id"]), method=str(payload["method"]), params=payload.get("params")))
        if "method" in payload:
            return cls(JsonRpcNotification(method=str(payload["method"]), params=payload.get("params")))
        if "result" in payload and "id" in payload:
            return cls(JsonRpcResponse(id=RequestId(payload["id"]), result=payload["result"]))
        if "error" in payload and "id" in payload:
            raw_error = payload["error"]
            if not isinstance(raw_error, dict):
                raise ValueError("Invalid JSON-RPC message")
            return cls(
                JsonRpcError(
                    id=RequestId(payload["id"]),
                    error=AppServerError(
                        code=int(raw_error["code"]),
                        message=str(raw_error["message"]),
                        data=raw_error.get("data"),
                    ),
                )
            )
        raise ValueError("Invalid JSON-RPC message")
