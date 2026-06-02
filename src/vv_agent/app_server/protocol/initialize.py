from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ClientInfo:
    name: str
    title: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"name": self.name}
        if self.title is not None:
            payload["title"] = self.title
        if self.version is not None:
            payload["version"] = self.version
        return payload


@dataclass(frozen=True, slots=True)
class ClientCapabilities:
    experimental_api: bool = False
    opt_out_notification_methods: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, bool | list[str]]:
        return {
            "experimentalApi": self.experimental_api,
            "optOutNotificationMethods": list(self.opt_out_notification_methods),
        }


@dataclass(frozen=True, slots=True)
class InitializeParams:
    client_info: ClientInfo
    capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)

    def to_dict(self) -> dict[str, object]:
        return {"clientInfo": self.client_info.to_dict(), "capabilities": self.capabilities.to_dict()}


@dataclass(frozen=True, slots=True)
class InitializeResponse:
    user_agent: str
    protocol_version: str
    capabilities: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "userAgent": self.user_agent,
            "protocolVersion": self.protocol_version,
            "capabilities": dict(self.capabilities),
        }
