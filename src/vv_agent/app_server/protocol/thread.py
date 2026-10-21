from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ThreadStartParams:
    agent_key: str
    input_id: str
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"agentKey": self.agent_key, "inputId": self.input_id}
        if self.cwd is not None:
            payload["cwd"] = self.cwd
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ThreadResumeParams:
    thread_id: str
    subscribe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"threadId": self.thread_id, "subscribe": self.subscribe}


@dataclass(frozen=True, slots=True)
class ThreadReadParams:
    thread_id: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"threadId": self.thread_id}
        return payload


@dataclass(frozen=True, slots=True)
class ThreadStatusParams:
    thread_id: str

    def to_dict(self) -> dict[str, str]:
        return {"threadId": self.thread_id}


@dataclass(frozen=True, slots=True)
class ThreadListParams:
    include_archived: bool = False
    archived: bool | None = None
    offset: int | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.include_archived:
            payload["includeArchived"] = True
        if self.archived is not None:
            payload["archived"] = self.archived
        if self.offset is not None:
            payload["offset"] = self.offset
        if self.limit is not None:
            payload["limit"] = self.limit
        return payload


@dataclass(frozen=True, slots=True)
class ThreadArchiveParams:
    thread_id: str

    def to_dict(self) -> dict[str, str]:
        return {"threadId": self.thread_id}


@dataclass(frozen=True, slots=True)
class ThreadUnsubscribeParams:
    thread_id: str

    def to_dict(self) -> dict[str, str]:
        return {"threadId": self.thread_id}


@dataclass(frozen=True, slots=True)
class ThreadModelSelectParams:
    thread_id: str
    input_id: str
    selection: dict[str, Any] | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ThreadModelSelectParams:
        from vv_agent.session.records import InboxItem

        if set(value) != {"threadId", "inputId", "selection"}:
            raise ValueError("thread/model/select requires exactly threadId, inputId and selection")
        for key in ("threadId", "inputId"):
            if not isinstance(value[key], str) or not value[key] or len(value[key].encode("utf-8")) > 512:
                raise ValueError(f"{key} must be a nonempty identity of at most 512 bytes")
        from .identity import client_id

        client_id(value["inputId"], "inputId")
        InboxItem(value["inputId"], "model_selection", {"selection": value["selection"]}).encode()
        return cls(value["threadId"], value["inputId"], value["selection"])

    def to_dict(self) -> dict[str, Any]:
        return {"threadId": self.thread_id, "inputId": self.input_id, "selection": self.selection}


@dataclass(frozen=True, slots=True)
class ThreadEventsListParams:
    thread_id: str
    limit: int
    after_cursor: dict[str, Any] | None = None
    through_cursor: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"threadId": self.thread_id, "limit": self.limit}
        if self.after_cursor is not None:
            result["afterCursor"] = self.after_cursor
        if self.through_cursor is not None:
            result["throughCursor"] = self.through_cursor
        return result
