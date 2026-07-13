from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ThreadStartParams:
    agent_key: str
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"agentKey": self.agent_key}
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
    after_item_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"threadId": self.thread_id}
        if self.after_item_id is not None:
            payload["afterItemId"] = self.after_item_id
        return payload


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
