from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock

from vv_agent.sessions.base import SessionCommitError, _normalize_message, validate_session_commit
from vv_agent.types import Message


@dataclass(slots=True)
class MemorySession:
    session_id: str
    _items: list[Message] = field(default_factory=list)
    _commits: dict[str, str] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get_items(self, limit: int | None = None) -> list[Message]:
        with self._lock:
            items = deepcopy(self._items)
        if limit is not None:
            normalized_limit = max(int(limit), 0)
            return [] if normalized_limit == 0 else items[-normalized_limit:]
        return items

    def add_items(self, items: list[Message]) -> None:
        normalized = [_normalize_message(item) for item in items]
        with self._lock:
            self._items.extend(normalized)

    def add_items_once(
        self,
        commit_id: str,
        payload_digest: str,
        items: list[Message],
    ) -> str:
        normalized = validate_session_commit(commit_id, payload_digest, items)
        with self._lock:
            existing = self._commits.get(commit_id)
            if existing is not None:
                if existing != payload_digest:
                    raise SessionCommitError(
                        "session commit id already has a different payload",
                        code="session_commit_identity_conflict",
                    )
                return "replayed"
            self._items.extend(normalized)
            self._commits[commit_id] = payload_digest
            return "committed"

    def pop_item(self) -> Message | None:
        with self._lock:
            if not self._items:
                return None
            return deepcopy(self._items.pop())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._commits.clear()

    def clear_session(self) -> None:
        self.clear()


@dataclass(slots=True)
class MemorySessionStore:
    _sessions: dict[str, MemorySession] = field(default_factory=dict)

    def session(self, session_id: str) -> MemorySession:
        if session_id not in self._sessions:
            self._sessions[session_id] = MemorySession(session_id)
        return self._sessions[session_id]
