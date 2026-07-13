"""RedisStateStore — checkpoint persistence backed by Redis.

Reuses the same Redis instance that Celery already depends on.
Data is stored as JSON under ``vv_agent:checkpoint:{task_id}`` keys.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vv_agent.runtime.checkpoint_codec import checkpoint_from_json, checkpoint_to_json
from vv_agent.runtime.state import (
    Checkpoint,
    StateStoreSpec,
    _check_claim,
    _claim_matches,
    _validate_claim,
    _validate_renew,
)

_KEY_PREFIX = "vv_agent:checkpoint:"


class RedisStateStore:
    """StateStore implementation backed by Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis as _redis
        except ImportError as exc:
            raise ImportError("redis is required for RedisStateStore. Install with: pip install redis") from exc
        self._redis_url = redis_url
        self._watch_error = _redis.WatchError
        self._client: Any = _redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        payload = checkpoint_to_json(checkpoint)
        self._client.set(f"{_KEY_PREFIX}{checkpoint.task_id}", payload)

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        payload = checkpoint_to_json(checkpoint)
        return bool(self._client.set(f"{_KEY_PREFIX}{checkpoint.task_id}", payload, nx=True))

    def commit_checkpoint(self, checkpoint: Checkpoint, *, claim_token: str, expected_revision: int) -> bool:
        key = f"{_KEY_PREFIX}{checkpoint.task_id}"
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = checkpoint_from_json(raw) if raw is not None else None
                    if not _claim_matches(current, checkpoint, claim_token, expected_revision):
                        pipe.unwatch()
                        return False
                    checkpoint = replace(
                        checkpoint,
                        revision=expected_revision + 1,
                        claim_token=None,
                        claimed_cycle=None,
                        lease_expires_at_ms=None,
                    )
                    pipe.multi()
                    pipe.set(key, checkpoint_to_json(checkpoint))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        if checkpoint.terminal_result is None:
            raise ValueError("finalized checkpoint must include terminal_result")
        key = f"{_KEY_PREFIX}{checkpoint.task_id}"
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = checkpoint_from_json(raw) if raw is not None else None
                    if (
                        current is None
                        or current.revision != expected_revision
                        or current.claim_token is not None
                        or current.terminal_result is not None
                    ):
                        pipe.unwatch()
                        return False
                    terminal = replace(
                        checkpoint,
                        revision=expected_revision + 1,
                        claim_token=None,
                        claimed_cycle=None,
                        lease_expires_at_ms=None,
                    )
                    pipe.multi()
                    pipe.set(key, checkpoint_to_json(terminal))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue

    def renew_checkpoint_claim(
        self,
        task_id: str,
        *,
        claim_token: str,
        expected_revision: int,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, expected_revision, lease_expires_at_ms, now_ms)
        key = f"{_KEY_PREFIX}{task_id}"
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    checkpoint = checkpoint_from_json(raw) if raw is not None else None
                    if (
                        checkpoint is None
                        or checkpoint.revision != expected_revision
                        or checkpoint.claim_token != claim_token
                        or (checkpoint.lease_expires_at_ms or 0) <= now_ms
                    ):
                        pipe.unwatch()
                        return False
                    checkpoint.lease_expires_at_ms = lease_expires_at_ms
                    pipe.multi()
                    pipe.set(key, checkpoint_to_json(checkpoint))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        raw = self._client.get(f"{_KEY_PREFIX}{task_id}")
        if raw is None:
            return None
        return checkpoint_from_json(raw)

    def claim_checkpoint(
        self,
        task_id: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Checkpoint | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        key = f"{_KEY_PREFIX}{task_id}"
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    checkpoint = checkpoint_from_json(raw)
                    _check_claim(checkpoint, cycle_index, now_ms)
                    checkpoint.revision += 1
                    checkpoint.claim_token = claim_token
                    checkpoint.claimed_cycle = cycle_index
                    checkpoint.lease_expires_at_ms = lease_expires_at_ms
                    pipe.multi()
                    pipe.set(key, checkpoint_to_json(checkpoint))
                    pipe.execute()
                    return checkpoint
                except self._watch_error:
                    continue

    def delete_checkpoint(self, task_id: str) -> None:
        self._client.delete(f"{_KEY_PREFIX}{task_id}")

    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool:
        key = f"{_KEY_PREFIX}{task_id}"
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    checkpoint = checkpoint_from_json(raw)
                    if checkpoint.revision != expected_revision or checkpoint.terminal_result is None:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.delete(key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue

    def list_checkpoints(self) -> list[str]:
        keys: list[str] = []
        for key in self._client.scan_iter(f"{_KEY_PREFIX}*"):
            keys.append(str(key).removeprefix(_KEY_PREFIX))
        return sorted(keys)

    def state_store_spec(self) -> StateStoreSpec:
        return StateStoreSpec(kind="redis", location=self._redis_url)

    @staticmethod
    def checkpoint_to_json(checkpoint: Checkpoint) -> str:
        return checkpoint_to_json(checkpoint)

    @staticmethod
    def checkpoint_from_json(payload: str | bytes) -> Checkpoint:
        return checkpoint_from_json(payload)
