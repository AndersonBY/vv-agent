"""RedisStateStore — checkpoint persistence backed by Redis.

Reuses the same Redis instance that Celery already depends on.
Data is stored as JSON under ``vv_agent:checkpoint:{task_id}`` keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from vv_agent.checkpoint import EventCursor, canonical_json_bytes
from vv_agent.runtime.checkpoint_codec import checkpoint_from_json, checkpoint_to_json
from vv_agent.runtime.checkpoint_codec_v2 import (
    _strict_json_loads,
    checkpoint_v2_from_dict,
    checkpoint_v2_from_json,
    checkpoint_v2_to_dict,
    checkpoint_v2_to_json,
)
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    StateStoreSpec,
    _check_claim,
    _claim_matches,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
)
from vv_agent.runtime.state_v2 import (
    CheckpointV2,
    ClaimMode,
    check_claim_v2,
    checkpoint_definition_matches_v2,
    prepare_claimed_terminal_v2,
    prepare_event_delivery_v2,
)
from vv_agent.types import AgentStatus

_KEY_PREFIX = "vv_agent:checkpoint:"
_KEY_PREFIX_V2 = "vv-agent:checkpoint:v2:"
_IO_TIMEOUT_SECONDS = 1.0
_TRANSACTION_MAX_ATTEMPTS = 8
_RENEW_CLAIM_SCRIPT = """
local redis_time = redis.call("TIME")
local server_now_ms = tonumber(redis_time[1]) * 1000 + math.floor(tonumber(redis_time[2]) / 1000)
local client_now_ms = tonumber(ARGV[5])
local current_now_ms = math.max(server_now_ms, client_now_ms)
local previous_expiry_ms = tonumber(ARGV[3])
local requested_expiry_ms = tonumber(ARGV[4])
if previous_expiry_ms <= current_now_ms or requested_expiry_ms <= current_now_ms then
  return 2
end
local current = redis.call("GET", KEYS[1])
if current ~= ARGV[1] then
  return 0
end
redis.call("SET", KEYS[1], ARGV[2])
return 1
"""


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
            socket_connect_timeout=_IO_TIMEOUT_SECONDS,
            socket_timeout=_IO_TIMEOUT_SECONDS,
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
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
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
        raise RuntimeError("redis checkpoint commit exceeded transaction retry limit")

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        if checkpoint.terminal_result is None:
            raise ValueError("finalized checkpoint must include terminal_result")
        key = f"{_KEY_PREFIX}{checkpoint.task_id}"
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
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
        raise RuntimeError("redis checkpoint finalization exceeded transaction retry limit")

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
        clock = _LeaseOperationClock(now_ms)
        key = f"{_KEY_PREFIX}{task_id}"
        raw = self._client.get(key)
        checkpoint = checkpoint_from_json(raw) if raw is not None else None
        current_now_ms = clock.now_ms()
        if (
            checkpoint is None
            or checkpoint.revision != expected_revision
            or checkpoint.claim_token != claim_token
            or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
            or lease_expires_at_ms <= current_now_ms
        ):
            return False
        previous_lease_expires_at_ms = checkpoint.lease_expires_at_ms
        assert previous_lease_expires_at_ms is not None
        checkpoint.lease_expires_at_ms = lease_expires_at_ms
        result = self._client.eval(
            _RENEW_CLAIM_SCRIPT,
            1,
            key,
            raw,
            checkpoint_to_json(checkpoint),
            str(previous_lease_expires_at_ms),
            str(lease_expires_at_ms),
            str(clock.now_ms()),
        )
        if result == 2:
            raise CheckpointConflictError("claim lease expired")
        if result not in {0, 1}:
            raise RuntimeError(f"redis checkpoint renewal returned unexpected result: {result!r}")
        return bool(result)

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
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
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
        raise RuntimeError("redis checkpoint claim exceeded transaction retry limit")

    def delete_checkpoint(self, task_id: str) -> None:
        self._client.delete(f"{_KEY_PREFIX}{task_id}")

    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool:
        key = f"{_KEY_PREFIX}{task_id}"
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
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
        raise RuntimeError("redis checkpoint acknowledgement exceeded transaction retry limit")

    def list_checkpoints(self) -> list[str]:
        keys: list[str] = []
        for key in self._client.scan_iter(f"{_KEY_PREFIX}*"):
            keys.append(str(key).removeprefix(_KEY_PREFIX))
        return sorted(keys)

    def save_checkpoint_v2(self, checkpoint: CheckpointV2) -> None:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        payload, lease = _checkpoint_v2_to_storage(checkpoint)
        with self._client.pipeline() as pipe:
            pipe.multi()
            pipe.set(data_key, payload)
            if lease is None:
                pipe.delete(lease_key)
            else:
                pipe.set(lease_key, str(lease))
            pipe.execute()

    def create_checkpoint_v2(self, checkpoint: CheckpointV2) -> bool:
        if (
            checkpoint.revision != 0
            or checkpoint.resume_attempt != 1
            or checkpoint.claim_token is not None
        ):
            raise ValueError("new checkpoint v2 records must be unclaimed at revision zero")
        data_key, _lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        payload, _lease = _checkpoint_v2_to_storage(checkpoint)
        return bool(self._client.set(data_key, payload, nx=True))

    def load_checkpoint_v2(self, checkpoint_key: str) -> CheckpointV2 | None:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint_key)
        for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
            raw = self._client.get(data_key)
            if raw is None:
                return None
            lease = self._client.get(lease_key)
            if self._client.get(data_key) == raw and self._client.get(lease_key) == lease:
                return _checkpoint_v2_from_storage(raw, lease)
        raise RuntimeError("redis checkpoint v2 load could not obtain a stable snapshot")

    def claim_checkpoint_v2(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> CheckpointV2 | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    checkpoint = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    try:
                        check_claim_v2(checkpoint, cycle_index, now_ms, claim_mode)
                    except ValueError as exc:
                        raise CheckpointConflictError(str(exc)) from exc
                    if checkpoint.claim_token is not None and claim_mode != "recovery":
                        raise CheckpointConflictError("expired checkpoint claims require recovery mode")
                    if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED and claim_mode != "recovery":
                        raise CheckpointConflictError("reconciliation checkpoints require recovery mode")
                    checkpoint.revision += 1
                    if claim_mode == "recovery":
                        checkpoint.resume_attempt += 1
                    checkpoint.status = AgentStatus.RUNNING
                    checkpoint.claim_token = claim_token
                    checkpoint.claimed_cycle = cycle_index
                    checkpoint.lease_expires_at_ms = lease_expires_at_ms
                    payload, _lease = _checkpoint_v2_to_storage(checkpoint)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.set(lease_key, str(lease_expires_at_ms))
                    pipe.execute()
                    return checkpoint
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 claim exceeded transaction retry limit")

    def progress_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, self._client.get(lease_key))
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or current.status is not AgentStatus.RUNNING
                        or checkpoint.status is not AgentStatus.RUNNING
                        or not checkpoint_definition_matches_v2(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    snapshot = checkpoint_v2_from_json(checkpoint_v2_to_json(checkpoint))
                    snapshot.revision = expected_revision + 1
                    snapshot.claim_token = current.claim_token
                    snapshot.claimed_cycle = current.claimed_cycle
                    snapshot.lease_expires_at_ms = current.lease_expires_at_ms
                    payload, _lease = _checkpoint_v2_to_storage(snapshot)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 progress exceeded transaction retry limit")

    def suspend_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or checkpoint.cycle_index != current.cycle_index
                        or checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED
                        or not checkpoint_definition_matches_v2(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    snapshot = replace(
                        checkpoint,
                        revision=expected_revision + 1,
                        claim_token=None,
                        claimed_cycle=None,
                        lease_expires_at_ms=None,
                    )
                    payload, _lease = _checkpoint_v2_to_storage(snapshot)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 suspend exceeded transaction retry limit")

    def commit_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or checkpoint.terminal_result is not None
                        or checkpoint.status is not AgentStatus.RUNNING
                        or not checkpoint_definition_matches_v2(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    claimed_cycle = current.claimed_cycle
                    assert claimed_cycle is not None
                    if checkpoint.cycle_index != claimed_cycle:
                        pipe.unwatch()
                        return False
                    committed = replace(
                        checkpoint,
                        revision=expected_revision + 1,
                        claim_token=None,
                        claimed_cycle=None,
                        lease_expires_at_ms=None,
                        model_call_journal=[],
                        tool_journal=[],
                    )
                    payload, _lease = _checkpoint_v2_to_storage(committed)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 commit exceeded transaction retry limit")

    def finalize_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        expected_revision: int,
    ) -> bool:
        if checkpoint.terminal_result is None or checkpoint.claim_token is not None:
            raise ValueError("finalized checkpoint v2 must be terminal and unclaimed")
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token is not None
                        or current.terminal_result is not None
                        or not checkpoint_definition_matches_v2(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    terminal = replace(checkpoint, revision=expected_revision + 1)
                    payload, _lease = _checkpoint_v2_to_storage(terminal)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 finalization exceeded transaction retry limit")

    def finalize_claimed_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    terminal = prepare_claimed_terminal_v2(
                        current,
                        checkpoint,
                        claim_token=claim_token,
                        expected_revision=expected_revision,
                    )
                    if terminal is None:
                        pipe.unwatch()
                        return False
                    payload, _lease = _checkpoint_v2_to_storage(terminal)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis claimed checkpoint v2 finalization exceeded transaction retry limit")

    def record_event_delivery_v2(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    delivered = prepare_event_delivery_v2(
                        current,
                        event_id=event_id,
                        payload_digest=payload_digest,
                        cursor=cursor,
                        expected_revision=expected_revision,
                        claim_token=claim_token,
                    )
                    if delivered is None:
                        pipe.unwatch()
                        return False
                    payload, _lease = _checkpoint_v2_to_storage(delivered)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if delivered.claim_token is None:
                        pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 event delivery exceeded transaction retry limit")

    def renew_checkpoint_claim_v2(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, 0, lease_expires_at_ms, now_ms)
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    raw_lease = pipe.get(lease_key)
                    if raw is None or raw_lease is None:
                        pipe.unwatch()
                        return False
                    checkpoint = _checkpoint_v2_from_storage(raw, raw_lease)
                    current_now_ms = max(now_ms, _redis_server_now_ms(self._client))
                    if (
                        checkpoint.claim_token != claim_token
                        or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                        or lease_expires_at_ms <= current_now_ms
                    ):
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.set(lease_key, str(lease_expires_at_ms))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 renewal exceeded transaction retry limit")

    def acknowledge_terminal_v2(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        data_key, lease_key = self._checkpoint_v2_keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw = pipe.get(data_key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    checkpoint = _checkpoint_v2_from_storage(raw, pipe.get(lease_key))
                    if (
                        checkpoint.revision != expected_revision
                        or checkpoint.terminal_result is None
                        or checkpoint.claim_token is not None
                        or checkpoint.terminal_acknowledged
                    ):
                        pipe.unwatch()
                        return False
                    checkpoint.revision += 1
                    checkpoint.terminal_acknowledged = True
                    payload, _lease = _checkpoint_v2_to_storage(checkpoint)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v2 acknowledgement exceeded transaction retry limit")

    def delete_checkpoint_v2(self, checkpoint_key: str) -> None:
        self._client.delete(*self._checkpoint_v2_keys(checkpoint_key))

    def list_checkpoints_v2(self) -> list[str]:
        checkpoint_keys: set[str] = set()
        for key in self._client.scan_iter(f"{_KEY_PREFIX_V2}*"):
            normalized = str(key)
            if normalized.endswith(":lease"):
                continue
            raw = self._client.get(normalized)
            if raw is not None:
                checkpoint_keys.add(_checkpoint_key_from_storage(raw))
        return sorted(checkpoint_keys)

    def state_store_spec(self) -> StateStoreSpec:
        return StateStoreSpec(kind="redis", location=self._redis_url)

    @staticmethod
    def checkpoint_to_json(checkpoint: Checkpoint) -> str:
        return checkpoint_to_json(checkpoint)

    @staticmethod
    def checkpoint_from_json(payload: str | bytes) -> Checkpoint:
        return checkpoint_from_json(payload)

    @staticmethod
    def checkpoint_v2_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX_V2}{digest}"

    @classmethod
    def _checkpoint_v2_keys(cls, checkpoint_key: str) -> tuple[str, str]:
        data_key = cls.checkpoint_v2_key(checkpoint_key)
        return data_key, f"{data_key}:lease"

    @staticmethod
    def checkpoint_v2_to_json(checkpoint: CheckpointV2) -> str:
        return checkpoint_v2_to_json(checkpoint)

    @staticmethod
    def checkpoint_v2_from_json(payload: str | bytes) -> CheckpointV2:
        return checkpoint_v2_from_json(payload)


def _checkpoint_v2_to_storage(checkpoint: CheckpointV2) -> tuple[str, int | None]:
    payload = checkpoint_v2_to_dict(checkpoint)
    lease = payload.pop("lease_expires_at_ms")
    return canonical_json_bytes(payload, "redis checkpoint v2").decode("utf-8"), lease


def _checkpoint_v2_from_storage(raw: str | bytes, raw_lease: object | None) -> CheckpointV2:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis checkpoint v2 payload is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis checkpoint v2 payload must be an object")
    payload["lease_expires_at_ms"] = _lease_from_storage(raw_lease)
    return checkpoint_v2_from_dict(payload)


def _checkpoint_key_from_storage(raw: str | bytes) -> str:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis checkpoint v2 payload is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis checkpoint v2 payload must be an object")
    checkpoint_key = payload.get("checkpoint_key")
    if not isinstance(checkpoint_key, str) or not checkpoint_key:
        raise ValueError("redis checkpoint v2 checkpoint_key must be a non-empty string")
    return checkpoint_key


def _lease_from_storage(raw_lease: object | None) -> int | None:
    if raw_lease is None:
        return None
    if isinstance(raw_lease, bool) or not isinstance(raw_lease, str | bytes | int):
        raise ValueError("redis checkpoint v2 lease must be an integer")
    try:
        return int(raw_lease)
    except ValueError as exc:
        raise ValueError("redis checkpoint v2 lease must be an integer") from exc


def _redis_server_now_ms(client: Any) -> int:
    time_method = getattr(client, "time", None)
    if not callable(time_method):
        return 0
    seconds, microseconds = time_method()
    return int(seconds) * 1000 + int(microseconds) // 1000
