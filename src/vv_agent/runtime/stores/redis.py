"""RedisStateStore — checkpoint persistence backed by Redis.

Reuses the same Redis instance that Celery already depends on.
Data is stored as JSON under ``vv_agent:checkpoint:{task_id}`` keys.
"""
from __future__ import annotations

import json
from typing import Any

from vv_agent.runtime.state import Checkpoint
from vv_agent.types import AgentStatus, CycleRecord, Message

_KEY_PREFIX = "vv_agent:checkpoint:"


class RedisStateStore:
    """StateStore implementation backed by Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis as _redis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisStateStore. "
                "Install with: pip install redis"
            ) from exc
        self._client: Any = _redis.Redis.from_url(
            redis_url, decode_responses=True,
        )

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        payload = json.dumps(
            {
                "task_id": checkpoint.task_id,
                "cycle_index": checkpoint.cycle_index,
                "status": checkpoint.status.value,
                "messages": [m.to_dict() for m in checkpoint.messages],
                "cycles": [c.to_dict() for c in checkpoint.cycles],
                "shared_state": checkpoint.shared_state,
            },
            ensure_ascii=False,
            default=str,
        )
        self._client.set(f"{_KEY_PREFIX}{checkpoint.task_id}", payload)

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        raw = self._client.get(f"{_KEY_PREFIX}{task_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        return Checkpoint(
            task_id=data["task_id"],
            cycle_index=data["cycle_index"],
            status=AgentStatus(data["status"]),
            messages=[Message.from_dict(m) for m in data["messages"]],
            cycles=[
                CycleRecord.from_dict(c) for c in data["cycles"]
            ],
            shared_state=data.get("shared_state", {}),
        )

    def delete_checkpoint(self, task_id: str) -> None:
        self._client.delete(f"{_KEY_PREFIX}{task_id}")

    def list_checkpoints(self) -> list[str]:
        keys: list[str] = []
        for key in self._client.scan_iter(f"{_KEY_PREFIX}*"):
            keys.append(str(key).removeprefix(_KEY_PREFIX))
        return sorted(keys)
