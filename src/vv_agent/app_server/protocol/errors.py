from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


class AppServerErrorCode:
    PARSE_ERROR: ClassVar[int] = -32700
    INVALID_REQUEST: ClassVar[int] = -32600
    METHOD_NOT_FOUND: ClassVar[int] = -32601
    INVALID_PARAMS: ClassVar[int] = -32602
    INTERNAL_ERROR: ClassVar[int] = -32603
    SERVER_OVERLOADED: ClassVar[int] = -32001
    NOT_INITIALIZED: ClassVar[int] = -32010
    ALREADY_INITIALIZED: ClassVar[int] = -32011
    THREAD_NOT_FOUND: ClassVar[int] = -32020
    ACTIVE_TURN_NOT_FOUND: ClassVar[int] = -32030
    TURN_ID_MISMATCH: ClassVar[int] = -32031


@dataclass(frozen=True, slots=True)
class AppServerError:
    code: int
    message: str
    data: dict[str, Any] | None = field(default=None)

    @classmethod
    def not_initialized(cls) -> AppServerError:
        return cls(code=AppServerErrorCode.NOT_INITIALIZED, message="Not initialized")

    @classmethod
    def already_initialized(cls) -> AppServerError:
        return cls(code=AppServerErrorCode.ALREADY_INITIALIZED, message="Already initialized")

    @classmethod
    def method_not_found(cls, method: str) -> AppServerError:
        return cls(code=AppServerErrorCode.METHOD_NOT_FOUND, message=f"Method not found: {method}")

    @classmethod
    def invalid_params(cls, message: str, data: dict[str, Any] | None = None) -> AppServerError:
        return cls(code=AppServerErrorCode.INVALID_PARAMS, message=message, data=data)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = dict(self.data)
        return payload
