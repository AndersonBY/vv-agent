from __future__ import annotations

from typing import Any

from vv_agent.sessions.base import _deserialize_message, _serialize_message
from vv_agent.types import Message


class RedisSession:
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

    def pop_item(self) -> Message | None:
        raw = self._client.rpop(self.key)
        if raw is None:
            return None
        return _deserialize_message(raw)

    def clear(self) -> None:
        self._client.delete(self.key)

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
