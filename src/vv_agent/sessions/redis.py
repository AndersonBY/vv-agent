from __future__ import annotations

from typing import Any

from vv_agent.sessions.base import (
    SessionCommitError,
    _deserialize_message,
    _serialize_message,
    validate_session_commit,
)
from vv_agent.types import Message


class RedisSession:
    _ADD_ITEMS_ONCE_SCRIPT = """
local existing = redis.call('HGET', KEYS[2], ARGV[1])
if existing then
    if existing == ARGV[2] then
        return 0
    end
    return -1
end
for index = 3, #ARGV do
    redis.call('RPUSH', KEYS[1], ARGV[index])
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
return 1
"""

    def __init__(
        self,
        session_id: str,
        *,
        redis_client: Any | None = None,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "vv-agent-session",
    ) -> None:
        self.session_id = session_id
        self.key_prefix = key_prefix
        self._client = redis_client if redis_client is not None else self._build_client(redis_url)

    @property
    def key(self) -> str:
        return f"{self.key_prefix}:{self.session_id}"

    @property
    def commit_key(self) -> str:
        return f"{self.key}:commits"

    def get_items(self, limit: int | None = None) -> list[Message]:
        if limit is None:
            raw_items = self._client.lrange(self.key, 0, -1)
        else:
            parsed_limit = max(int(limit), 0)
            if parsed_limit == 0:
                return []
            raw_items = self._client.lrange(self.key, -parsed_limit, -1)
        return [_deserialize_message(item) for item in raw_items]

    def add_items(self, items: list[Message]) -> None:
        if not items:
            return
        payloads = [_serialize_message(item) for item in items]
        self._client.rpush(self.key, *payloads)

    def add_items_once(
        self,
        commit_id: str,
        payload_digest: str,
        items: list[Message],
    ) -> str:
        normalized = validate_session_commit(commit_id, payload_digest, items)
        payloads = [_serialize_message(item) for item in normalized]
        evaluate = getattr(self._client, "eval", None)
        if not callable(evaluate):
            raise SessionCommitError(
                "Redis session client does not support atomic Lua evaluation",
                code="checkpoint_session_idempotency_unsupported",
            )
        outcome = int(
            evaluate(
                self._ADD_ITEMS_ONCE_SCRIPT,
                2,
                self.key,
                self.commit_key,
                commit_id,
                payload_digest,
                *payloads,
            )
        )
        if outcome == -1:
            raise SessionCommitError(
                "session commit id already has a different payload",
                code="session_commit_identity_conflict",
            )
        return "committed" if outcome == 1 else "replayed"

    def pop_item(self) -> Message | None:
        raw = self._client.rpop(self.key)
        if raw is None:
            return None
        return _deserialize_message(raw)

    def clear(self) -> None:
        self._client.delete(self.key, self.commit_key)

    def clear_session(self) -> None:
        self.clear()

    @staticmethod
    def _build_client(redis_url: str) -> Any:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("RedisSession requires the optional redis package or an injected redis_client.") from exc
        return redis.Redis.from_url(redis_url)


class RedisSessionStore:
    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "vv-agent-session",
    ) -> None:
        self.key_prefix = key_prefix
        self._client = redis_client if redis_client is not None else RedisSession._build_client(redis_url)

    def session(self, session_id: str) -> RedisSession:
        return RedisSession(
            session_id,
            redis_client=self._client,
            key_prefix=self.key_prefix,
        )
