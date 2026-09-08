"""RedisCheckpointStore — checkpoint persistence backed by Redis.

Reuses the same Redis instance that Celery already depends on. Data is stored
under SHA-256-addressed ``vv-agent:checkpoint:`` keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from threading import RLock
from typing import Any, Literal, cast

from vv_agent.checkpoint import CheckpointError, EventCursor, canonical_json_bytes, canonical_json_sha256
from vv_agent.deferred import DeferredResolutionReceipt, DeferredResolveDecision, DeferredToolHandle
from vv_agent.runtime.checkpoint_codec import (
    _strict_json_loads,
    checkpoint_from_dict,
    checkpoint_from_json,
    checkpoint_to_dict,
    checkpoint_to_json,
)
from vv_agent.runtime.controller import (
    ControllerCommand,
    ControllerCommandReceipt,
    ControllerCommandResolution,
    ControllerWake,
    HostInteractionAdmissionContext,
    HostInteractionOutcome,
    HostInteractionRecoveryEnvelope,
    HostInteractionRecoveryResult,
    HostInteractionRequest,
    derive_controller_receipt_outbox_id,
    derive_host_interaction_notification_id,
    derive_host_interaction_record_id,
    validate_host_interaction_notification,
    validate_host_interaction_record,
)
from vv_agent.runtime.dispatch_outbox import (
    DispatchOutboxClaim,
    DispatchOutboxRecord,
    claim_dispatch,
    complete_dispatch,
    reap_dispatch,
    reconcile_dispatch,
)
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    CheckpointRenewal,
    ClaimMode,
    OperationState,
    RenewOutcome,
    _validate_claim,
    _validate_renew,
    check_claim,
    checkpoint_definition_matches,
    merge_event_outbox,
    prepare_claimed_terminal,
    prepare_event_delivery,
    prepare_unclaimed_terminal,
    validate_checkpoint_creation,
    validate_model_journal_accounting,
)
from vv_agent.types import AgentStatus

_KEY_PREFIX = "vv-agent:checkpoint:"
_DEFERRED_RECEIPT_PREFIX = "vv-agent:deferred-receipt:"
_DEFERRED_RECEIPT_SET_PREFIX = "vv-agent:deferred-receipts-by-checkpoint:"
_CONTROLLER_RECEIPT_PREFIX = "vv-agent:controller-command:"
_CONTROLLER_RECEIPT_SET_PREFIX = "vv-agent:controller-commands-by-checkpoint:"
_CONTROLLER_COMMAND_PAYLOAD_SUFFIX = ":command"
_CONTROLLER_OUTBOX_SUFFIX = ":outbox"
_HOST_RECORD_PREFIX = "vv-agent:host-interaction:"
_HOST_RECORD_SET_PREFIX = "vv-agent:host-interactions-by-checkpoint:"
_HOST_NOTIFICATION_PREFIX = "vv-agent:host-interaction-notification:"
_HOST_NOTIFICATION_SET_PREFIX = "vv-agent:host-interaction-notifications-by-checkpoint:"
_DISPATCH_OUTBOX_PREFIX = "vv-agent:distributed-dispatch:"
_DISPATCH_OUTBOX_SET_PREFIX = "vv-agent:distributed-dispatches-by-checkpoint:"
_IO_TIMEOUT_SECONDS = 1.0
_TRANSACTION_MAX_ATTEMPTS = 8


class RedisCheckpointStore:
    """Current checkpoint store backed by Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis as _redis
        except ImportError as exc:
            raise ImportError("redis is required for RedisCheckpointStore. Install with: pip install redis") from exc
        self._watch_error = _redis.WatchError
        self._client: Any = _redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=_IO_TIMEOUT_SECONDS,
            socket_timeout=_IO_TIMEOUT_SECONDS,
        )

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        validate_checkpoint_creation(checkpoint)
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        payload, _lease = _checkpoint_to_storage(checkpoint)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    if pipe.get(data_key) is not None:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.set(data_key, payload, nx=True)
                    pipe.delete(lease_key)
                    results = pipe.execute()
                    return bool(results[0])
                except self._watch_error:
                    pipe.unwatch()
                    continue
                except Exception:
                    pipe.unwatch()
                    raise
        raise RuntimeError("redis checkpoint v10 creation exceeded transaction retry limit")

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None:
        data_key, lease_key = self._keys(checkpoint_key)
        raw, lease = self._client.mget([data_key, lease_key])
        return None if raw is None else _checkpoint_from_storage(raw, lease, checkpoint_key=checkpoint_key)

    def claim_checkpoint(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> Checkpoint | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        data_key, lease_key = self._keys(checkpoint_key)
        record_set_key = self._host_record_set_key(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key, record_set_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return None
                    checkpoint = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    try:
                        check_claim(checkpoint, cycle_index, now_ms, claim_mode)
                    except ValueError as exc:
                        raise CheckpointConflictError(str(exc)) from exc
                    if checkpoint.claim_token is not None and claim_mode != "recovery":
                        raise CheckpointConflictError("expired checkpoint claims require recovery mode")
                    if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED and claim_mode != "recovery":
                        raise CheckpointConflictError("reconciliation checkpoints require recovery mode")
                    for raw_record_key in tuple(pipe.smembers(record_set_key)):
                        record_key = raw_record_key.decode("utf-8") if isinstance(raw_record_key, bytes) else str(raw_record_key)
                        raw_record = pipe.get(record_key)
                        if raw_record is None:
                            continue
                        record = _host_record_from_storage(raw_record, expected_key=record_key)
                        if record["checkpoint_key"] != checkpoint_key:
                            continue
                        if record["state"] in {"resolved_pending", "resolved_claimed"}:
                            raise CheckpointError(
                                "host interaction response requires dedicated recovery",
                                code="host_interaction_recovery_required",
                            )
                    checkpoint.revision += 1
                    if claim_mode == "recovery":
                        checkpoint.resume_attempt += 1
                    checkpoint.status = AgentStatus.RUNNING
                    checkpoint.claim_token = claim_token
                    checkpoint.claimed_cycle = cycle_index
                    checkpoint.lease_expires_at_ms = lease_expires_at_ms
                    payload, _lease = _checkpoint_to_storage(checkpoint)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.set(lease_key, str(lease_expires_at_ms))
                    pipe.execute()
                    return checkpoint
                except CheckpointError:
                    pipe.unwatch()
                    raise
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 claim exceeded transaction retry limit")

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(
                        raw,
                        raw_lease,
                        checkpoint_key=checkpoint.checkpoint_key,
                    )
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or current.status is not AgentStatus.RUNNING
                        or checkpoint.status is not AgentStatus.RUNNING
                        or not checkpoint_definition_matches(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
                    snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
                    snapshot.event_cursor = deepcopy(current.event_cursor)
                    snapshot.cancel_requested = snapshot.cancel_requested or current.cancel_requested
                    snapshot.revision = expected_revision + 1
                    snapshot.claim_token = current.claim_token
                    snapshot.claimed_cycle = current.claimed_cycle
                    snapshot.lease_expires_at_ms = current.lease_expires_at_ms
                    payload, _lease = _checkpoint_to_storage(snapshot)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 progress exceeded transaction retry limit")

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or checkpoint.cycle_index != current.cycle_index
                        or checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED
                        or not checkpoint_definition_matches(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    snapshot = checkpoint_from_json(
                        checkpoint_to_json(
                            replace(
                                checkpoint,
                                revision=expected_revision + 1,
                                claim_token=None,
                                claimed_cycle=None,
                                lease_expires_at_ms=None,
                            )
                        )
                    )
                    snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
                    snapshot.event_cursor = deepcopy(current.event_cursor)
                    snapshot.cancel_requested = current.cancel_requested
                    payload, _lease = _checkpoint_to_storage(snapshot)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 suspend exceeded transaction retry limit")

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != checkpoint.claimed_cycle
                        or current.terminal_result is not None
                        or checkpoint.terminal_result is not None
                        or checkpoint.status is not AgentStatus.RUNNING
                        or current.cancel_requested
                        or checkpoint.cancel_requested
                        or any(
                            entry.state
                            in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
                            for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
                        )
                        or not checkpoint_definition_matches(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    claimed_cycle = current.claimed_cycle
                    assert claimed_cycle is not None
                    if checkpoint.cycle_index != claimed_cycle:
                        pipe.unwatch()
                        return False
                    validate_model_journal_accounting(checkpoint)
                    checkpoint.event_outbox = merge_event_outbox(current.event_outbox, checkpoint.event_outbox)
                    checkpoint.event_cursor = deepcopy(current.event_cursor)
                    committed = replace(
                        checkpoint,
                        revision=expected_revision + 1,
                        claim_token=None,
                        claimed_cycle=None,
                        lease_expires_at_ms=None,
                        event_outbox=[entry for entry in checkpoint.event_outbox if entry.state == "pending"],
                        model_call_journal=[],
                        tool_journal=[],
                    )
                    payload, _lease = _checkpoint_to_storage(committed)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 commit exceeded transaction retry limit")

    def finalize_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int,
    ) -> bool:
        if checkpoint.terminal_result is None or checkpoint.claim_token is not None:
            raise ValueError("finalized checkpoint v10 must be terminal and unclaimed")
        checkpoint = prepare_unclaimed_terminal(checkpoint)
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token is not None
                        or current.terminal_result is not None
                        or not checkpoint_definition_matches(current, checkpoint)
                    ):
                        pipe.unwatch()
                        return False
                    checkpoint.cancel_requested = current.cancel_requested
                    checkpoint.event_outbox = merge_event_outbox(current.event_outbox, checkpoint.event_outbox)
                    checkpoint.event_cursor = deepcopy(current.event_cursor)
                    terminal = replace(checkpoint, revision=expected_revision + 1)
                    payload, _lease = _checkpoint_to_storage(terminal)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 finalization exceeded transaction retry limit")

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    candidate = deepcopy(checkpoint)
                    terminal = prepare_claimed_terminal(
                        current,
                        candidate,
                        claim_token=claim_token,
                        expected_revision=expected_revision,
                    )
                    if terminal is None:
                        pipe.unwatch()
                        return False
                    payload, _lease = _checkpoint_to_storage(terminal)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis claimed checkpoint v10 finalization exceeded transaction retry limit")

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool:
        data_key, lease_key = self._keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    delivered = prepare_event_delivery(
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
                    payload, _lease = _checkpoint_to_storage(delivered)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if delivered.claim_token is None:
                        pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 event delivery exceeded transaction retry limit")

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal:
        _validate_renew(claim_token, lease_expires_at_ms, now_ms)
        data_key, lease_key = self._keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=0)
                    checkpoint = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    current_now_ms = max(now_ms, _redis_server_now_ms(self._client))
                    if (
                        checkpoint.claim_token != claim_token
                        or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                        or lease_expires_at_ms <= current_now_ms
                    ):
                        pipe.unwatch()
                        return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=checkpoint.revision)
                    pipe.multi()
                    pipe.set(lease_key, str(lease_expires_at_ms))
                    pipe.execute()
                    return CheckpointRenewal(
                        outcome=(RenewOutcome.CANCEL_REQUESTED if checkpoint.cancel_requested else RenewOutcome.RENEWED),
                        lease_expires_at_ms=lease_expires_at_ms,
                    )
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 renewal exceeded transaction retry limit")

    def record_tool_receipt(
        self,
        checkpoint: Checkpoint,
        *,
        operation_id: str,
        attempt: int,
        tool_call_id: str,
        request_digest: str,
        result: Any,
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    helper = InMemoryCheckpointStore()
                    helper._store[current.checkpoint_key] = current  # type: ignore[attr-defined]
                    updated = helper.record_tool_receipt(
                        checkpoint,
                        operation_id=operation_id,
                        attempt=attempt,
                        tool_call_id=tool_call_id,
                        request_digest=request_digest,
                        result=result,
                        claim_token=claim_token,
                        expected_revision=expected_revision,
                        claimed_cycle=claimed_cycle,
                    )
                    authoritative = helper._store[current.checkpoint_key]  # type: ignore[attr-defined]
                    if not updated or authoritative.revision == current.revision:
                        pipe.unwatch()
                        return updated
                    payload, _lease = _checkpoint_to_storage(authoritative)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if authoritative.lease_expires_at_ms is not None:
                        pipe.set(lease_key, str(authoritative.lease_expires_at_ms))
                    pipe.execute()
                    return True
                except CheckpointError:
                    pipe.unwatch()
                    raise
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 tool receipt exceeded transaction retry limit")

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        data_key, lease_key = self._keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    checkpoint = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
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
                    payload, _lease = _checkpoint_to_storage(checkpoint)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 acknowledgement exceeded transaction retry limit")

    def delete_checkpoint(self, checkpoint_key: str) -> None:
        def cleanup_member_key(member: object, label: str) -> str:
            if isinstance(member, bytes):
                try:
                    member = member.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise CheckpointError(
                        f"redis {label} reverse index member is invalid",
                        code="checkpoint_store_conflict",
                    ) from exc
            if not isinstance(member, str) or not member:
                raise CheckpointError(
                    f"redis {label} reverse index member is invalid",
                    code="checkpoint_store_conflict",
                )
            return member

        data_key, lease_key = self._keys(checkpoint_key)
        receipt_set_key = self._receipt_set_key(checkpoint_key)
        controller_set_key = self._controller_receipt_set_key(checkpoint_key)
        record_set_key = self._host_record_set_key(checkpoint_key)
        notification_set_key = self._host_notification_set_key(checkpoint_key)
        dispatch_set_key = self._dispatch_outbox_set_key(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    # The reverse index is watched together with the
                    # checkpoint. A resolver that inserts a new receipt (and
                    # sadds its index) therefore retries instead of creating
                    # an orphan tombstone after this SMEMBERS snapshot.
                    pipe.watch(
                        data_key,
                        lease_key,
                        receipt_set_key,
                        controller_set_key,
                        record_set_key,
                        notification_set_key,
                        dispatch_set_key,
                    )
                    smembers = getattr(pipe, "smembers", None)
                    if callable(smembers):
                        receipt_keys = tuple(smembers(receipt_set_key))
                        controller_keys = tuple(smembers(controller_set_key))
                        record_keys = tuple(smembers(record_set_key))
                        notification_keys = tuple(smembers(notification_set_key))
                        dispatch_keys = tuple(smembers(dispatch_set_key))
                    else:
                        client_smembers = getattr(self._client, "smembers", None)
                        receipt_keys = tuple(client_smembers(receipt_set_key)) if callable(client_smembers) else ()
                        controller_keys = tuple(client_smembers(controller_set_key)) if callable(client_smembers) else ()
                        record_keys = tuple(client_smembers(record_set_key)) if callable(client_smembers) else ()
                        notification_keys = tuple(client_smembers(notification_set_key)) if callable(client_smembers) else ()
                        dispatch_keys = tuple(client_smembers(dispatch_set_key)) if callable(client_smembers) else ()
                    receipt_keys = tuple(cleanup_member_key(key, "deferred receipt") for key in receipt_keys)
                    controller_keys = tuple(cleanup_member_key(key, "controller receipt") for key in controller_keys)
                    record_keys = tuple(cleanup_member_key(key, "host record") for key in record_keys)
                    notification_keys = tuple(cleanup_member_key(key, "host notification") for key in notification_keys)
                    dispatch_keys = tuple(cleanup_member_key(key, "dispatch outbox") for key in dispatch_keys)
                    raw_checkpoint, raw_lease = pipe.mget([data_key, lease_key])
                    if raw_checkpoint is not None:
                        _checkpoint_from_storage(
                            raw_checkpoint,
                            raw_lease,
                            checkpoint_key=checkpoint_key,
                        )
                    if not receipt_keys:
                        # Legacy/test doubles may not expose the reverse set.
                        # Keep the scan under WATCH; a concurrent modern
                        # resolver mutating the set invalidates this attempt.
                        scan_iter = getattr(self._client, "scan_iter", None)
                        if callable(scan_iter):
                            for candidate in scan_iter(f"{_DEFERRED_RECEIPT_PREFIX}*"):
                                candidate = cleanup_member_key(candidate, "deferred receipt")
                                raw = self._client.get(candidate)
                                if raw is None:
                                    continue
                                try:
                                    receipt = _receipt_from_storage(raw)
                                    if receipt.handle.checkpoint_key == checkpoint_key and candidate == self._receipt_key(
                                        receipt.handle.key
                                    ):
                                        receipt_keys = (*receipt_keys, candidate)
                                except (TypeError, ValueError):
                                    continue
                    scan_iter = getattr(self._client, "scan_iter", None)
                    if callable(scan_iter):
                        if not controller_keys:
                            for candidate in scan_iter(f"{_CONTROLLER_RECEIPT_PREFIX}*"):
                                candidate = cleanup_member_key(candidate, "controller receipt")
                                raw = self._client.get(candidate)
                                if raw is None:
                                    continue
                                try:
                                    receipt = _controller_receipt_from_storage(raw)
                                except (TypeError, ValueError):
                                    continue
                                if receipt.handle.checkpoint_key == checkpoint_key and candidate == self._controller_receipt_key(
                                    receipt.command_id
                                ):
                                    controller_keys = (*controller_keys, candidate)
                        if not record_keys:
                            for candidate in scan_iter(f"{_HOST_RECORD_PREFIX}*"):
                                candidate = cleanup_member_key(candidate, "host record")
                                raw = self._client.get(candidate)
                                if raw is None:
                                    continue
                                try:
                                    record = _host_record_from_storage(raw, expected_key=candidate)
                                except (TypeError, ValueError):
                                    continue
                                if record.get("checkpoint_key") == checkpoint_key:
                                    record_keys = (*record_keys, candidate)
                        if not notification_keys:
                            for candidate in scan_iter(f"{_HOST_NOTIFICATION_PREFIX}*"):
                                candidate = cleanup_member_key(candidate, "host notification")
                                raw = self._client.get(candidate)
                                if raw is None:
                                    continue
                                try:
                                    notification = _notification_from_storage(raw, expected_key=candidate)
                                except (TypeError, ValueError):
                                    continue
                                if notification.get("checkpoint_key") == checkpoint_key:
                                    notification_keys = (*notification_keys, candidate)
                        if not dispatch_keys:
                            for candidate in scan_iter(f"{_DISPATCH_OUTBOX_PREFIX}*"):
                                candidate = cleanup_member_key(candidate, "dispatch outbox")
                                raw = self._client.get(candidate)
                                if raw is None:
                                    continue
                                try:
                                    dispatch = _dispatch_outbox_from_storage(raw, expected_key=candidate)
                                except (TypeError, ValueError):
                                    continue
                                if dispatch.checkpoint_key == checkpoint_key:
                                    dispatch_keys = (*dispatch_keys, candidate)
                    for receipt_key in receipt_keys:
                        raw = pipe.get(receipt_key)
                        if raw is None:
                            raise CheckpointError(
                                "redis deferred receipt reverse index member is missing",
                                code="checkpoint_store_conflict",
                            )
                        try:
                            receipt = _receipt_from_storage(raw)
                        except (TypeError, ValueError) as exc:
                            raise CheckpointError(
                                "redis deferred receipt reverse index member is invalid",
                                code="checkpoint_store_conflict",
                            ) from exc
                        if receipt.handle.checkpoint_key != checkpoint_key or receipt_key != self._receipt_key(
                            receipt.handle.key
                        ):
                            raise CheckpointError(
                                "redis deferred receipt reverse index member is foreign",
                                code="checkpoint_store_conflict",
                            )
                    for controller_key in controller_keys:
                        raw = pipe.get(controller_key)
                        if raw is None:
                            raise CheckpointError(
                                "redis controller receipt reverse index member is missing",
                                code="checkpoint_store_conflict",
                            )
                        try:
                            receipt = _controller_receipt_from_storage(raw)
                        except (TypeError, ValueError) as exc:
                            raise CheckpointError(
                                "redis controller receipt reverse index member is invalid",
                                code="checkpoint_store_conflict",
                            ) from exc
                        if receipt.handle.checkpoint_key != checkpoint_key or controller_key != self._controller_receipt_key(
                            receipt.command_id
                        ):
                            raise CheckpointError(
                                "redis controller receipt reverse index member is foreign",
                                code="checkpoint_store_conflict",
                            )
                    for record_key in record_keys:
                        raw = pipe.get(record_key)
                        if raw is None:
                            raise CheckpointError(
                                "redis host record reverse index member is missing",
                                code="checkpoint_store_conflict",
                            )
                        try:
                            record = _host_record_from_storage(
                                raw,
                                expected_checkpoint_key=checkpoint_key,
                                expected_key=record_key,
                            )
                        except (TypeError, ValueError) as exc:
                            raise CheckpointError(
                                "redis host record reverse index member is invalid",
                                code="checkpoint_store_conflict",
                            ) from exc
                    for notification_key in notification_keys:
                        raw = pipe.get(notification_key)
                        if raw is None:
                            raise CheckpointError(
                                "redis host notification reverse index member is missing",
                                code="checkpoint_store_conflict",
                            )
                        try:
                            notification = _notification_from_storage(
                                raw,
                                expected_checkpoint_key=checkpoint_key,
                                expected_key=notification_key,
                            )
                        except (TypeError, ValueError) as exc:
                            raise CheckpointError(
                                "redis host notification reverse index member is invalid",
                                code="checkpoint_store_conflict",
                            ) from exc
                    for dispatch_key in dispatch_keys:
                        raw = pipe.get(dispatch_key)
                        if raw is None:
                            raise CheckpointError(
                                "redis dispatch outbox reverse index member is missing",
                                code="checkpoint_store_conflict",
                            )
                        try:
                            dispatch = _dispatch_outbox_from_storage(
                                raw,
                                expected_checkpoint_key=checkpoint_key,
                                expected_key=dispatch_key,
                            )
                        except (TypeError, ValueError) as exc:
                            raise CheckpointError(
                                "redis dispatch outbox reverse index member is invalid",
                                code="checkpoint_store_conflict",
                            ) from exc
                    pipe.multi()
                    for key in (
                        data_key,
                        lease_key,
                        receipt_set_key,
                        controller_set_key,
                        record_set_key,
                        notification_set_key,
                        dispatch_set_key,
                        *map(str, receipt_keys),
                        *map(str, controller_keys),
                        *(f"{key}{_CONTROLLER_COMMAND_PAYLOAD_SUFFIX}" for key in map(str, controller_keys)),
                        *(f"{key}{_CONTROLLER_OUTBOX_SUFFIX}" for key in map(str, controller_keys)),
                        *map(str, record_keys),
                        *map(str, notification_keys),
                        *map(str, dispatch_keys),
                    ):
                        pipe.delete(key)
                    pipe.execute()
                    return
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 cleanup exceeded transaction retry limit")

    def claim_distributed_dispatch(
        self,
        envelope: Mapping[str, Any],
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> DispatchOutboxClaim:
        candidate = DispatchOutboxRecord.pending(envelope)
        data_key, lease_key = self._keys(candidate.checkpoint_key)
        outbox_key = self._dispatch_outbox_key(candidate.dispatch_id)
        index_key = self._dispatch_outbox_set_key(candidate.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key, outbox_key, index_key)
                    raw_checkpoint, raw_lease = pipe.mget([data_key, lease_key])
                    if raw_checkpoint is None:
                        pipe.unwatch()
                        raise CheckpointError("dispatch checkpoint was not found", code="checkpoint_not_found")
                    _checkpoint_from_storage(
                        raw_checkpoint,
                        raw_lease,
                        checkpoint_key=candidate.checkpoint_key,
                    )
                    raw = pipe.get(outbox_key)
                    current = (
                        candidate
                        if raw is None
                        else _dispatch_outbox_from_storage(
                            raw,
                            expected_checkpoint_key=candidate.checkpoint_key,
                            expected_key=outbox_key,
                        )
                    )
                    if current.envelope_digest != candidate.envelope_digest:
                        pipe.unwatch()
                        raise CheckpointError(
                            "dispatch id was reused with a different immutable envelope",
                            code="dispatch_outbox_conflict",
                        )
                    claim = claim_dispatch(
                        current,
                        claim_token=claim_token,
                        lease_expires_at_ms=lease_expires_at_ms,
                        now_ms=now_ms,
                    )
                    if claim.record == current:
                        pipe.unwatch()
                        return claim
                    pipe.multi()
                    pipe.set(outbox_key, _dispatch_outbox_to_storage(claim.record))
                    pipe.sadd(index_key, outbox_key)
                    pipe.execute()
                    return claim
                except self._watch_error:
                    continue
        raise RuntimeError("redis distributed dispatch claim exceeded transaction retry limit")

    def complete_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        outbox_key = self._dispatch_outbox_key(dispatch_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(outbox_key)
                    raw = pipe.get(outbox_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    current = _dispatch_outbox_from_storage(raw, expected_key=outbox_key)
                    if current.envelope_digest != envelope_digest:
                        pipe.unwatch()
                        raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
                    if outcome not in {"delivered", "ambiguous"}:
                        pipe.unwatch()
                        raise ValueError("dispatch completion outcome must be delivered or ambiguous")
                    typed_outcome = cast(Literal["delivered", "ambiguous"], outcome)
                    if current.state == typed_outcome:
                        pipe.unwatch()
                        return current
                    updated = complete_dispatch(
                        current,
                        claim_token=claim_token,
                        attempt=attempt,
                        outcome=typed_outcome,
                        now_ms=now_ms,
                        error=error,
                    )
                    pipe.multi()
                    pipe.set(outbox_key, _dispatch_outbox_to_storage(updated))
                    pipe.execute()
                    return updated
                except self._watch_error:
                    continue
        raise RuntimeError("redis distributed dispatch completion exceeded transaction retry limit")

    def reconcile_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        outbox_key = self._dispatch_outbox_key(dispatch_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(outbox_key)
                    raw = pipe.get(outbox_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    current = _dispatch_outbox_from_storage(raw, expected_key=outbox_key)
                    if current.envelope_digest != envelope_digest:
                        pipe.unwatch()
                        raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
                    if outcome not in {"retry", "delivered"}:
                        pipe.unwatch()
                        raise ValueError("dispatch reconciliation outcome must be retry or delivered")
                    typed_outcome = cast(Literal["retry", "delivered"], outcome)
                    if (typed_outcome == "retry" and current.state == "pending") or (
                        typed_outcome == "delivered" and current.state == "delivered"
                    ):
                        pipe.unwatch()
                        return current
                    updated = reconcile_dispatch(
                        current,
                        outcome=typed_outcome,
                        now_ms=now_ms,
                        error=error,
                    )
                    pipe.multi()
                    pipe.set(outbox_key, _dispatch_outbox_to_storage(updated))
                    pipe.execute()
                    return updated
                except self._watch_error:
                    continue
        raise RuntimeError("redis distributed dispatch reconciliation exceeded transaction retry limit")

    def get_distributed_dispatch(self, dispatch_id: str) -> DispatchOutboxRecord | None:
        raw = self._client.get(self._dispatch_outbox_key(dispatch_id))
        return (
            _dispatch_outbox_from_storage(raw, expected_key=self._dispatch_outbox_key(dispatch_id)) if raw is not None else None
        )

    def reap_distributed_dispatches(
        self,
        *,
        checkpoint_key: str | None = None,
        now_ms: int,
    ) -> list[DispatchOutboxRecord]:
        if checkpoint_key is None:
            scan_iter = getattr(self._client, "scan_iter", None)
            keys = tuple(scan_iter(f"{_DISPATCH_OUTBOX_PREFIX}*")) if callable(scan_iter) else ()
        else:
            keys = tuple(self._client.smembers(self._dispatch_outbox_set_key(checkpoint_key)))
        rows: list[DispatchOutboxRecord] = []
        for raw_key in sorted(keys, key=str):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            with self._client.pipeline() as pipe:
                for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if raw is None:
                            pipe.unwatch()
                            break
                        current = _dispatch_outbox_from_storage(
                            raw,
                            expected_checkpoint_key=checkpoint_key,
                            expected_key=key,
                        )
                        updated = reap_dispatch(current, now_ms=now_ms)
                        if updated is None:
                            pipe.unwatch()
                            break
                        pipe.multi()
                        pipe.set(key, _dispatch_outbox_to_storage(updated))
                        pipe.execute()
                        rows.append(updated)
                        break
                    except self._watch_error:
                        continue
        return rows

    def preflight_tool_batch(
        self,
        checkpoint: Checkpoint,
        *,
        tool_call_count: int,
        expected_revision: int,
        claim_token: str,
        claimed_cycle: int,
    ) -> bool:
        if isinstance(tool_call_count, bool) or not isinstance(tool_call_count, int) or tool_call_count <= 0:
            return False
        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    if (
                        current.revision != expected_revision
                        or checkpoint.revision != expected_revision
                        or current.claim_token != claim_token
                        or current.claimed_cycle != claimed_cycle
                        or current.status is not AgentStatus.RUNNING
                        or current.terminal_result is not None
                    ):
                        pipe.unwatch()
                        return False
                    # Same-value transaction is the Redis writeability proof;
                    # the lifecycle outbox has no fixed cardinality/byte cap.
                    payload, _lease = _checkpoint_to_storage(current)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if current.lease_expires_at_ms is not None:
                        pipe.set(lease_key, str(current.lease_expires_at_ms))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 outbox preflight exceeded transaction retry limit")

    def admit_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        outcomes: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        """Atomically persist one mixed model-tool batch and its barrier.

        The Python helper is only used to prepare a fully validated snapshot;
        Redis WATCH/MULTI performs the authoritative compare-and-swap.  There
        is deliberately no bounded outbox or receipt cardinality check here.
        """
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    helper = InMemoryCheckpointStore()
                    helper._store[current.checkpoint_key] = current  # type: ignore[attr-defined]
                    if not helper.admit_deferred_batch(
                        current,
                        outcomes=outcomes,
                        claim_token=claim_token,
                        expected_revision=expected_revision,
                        claimed_cycle=claimed_cycle,
                    ):
                        pipe.unwatch()
                        return False
                    updated = helper._store[current.checkpoint_key]  # type: ignore[attr-defined]
                    payload, _lease = _checkpoint_to_storage(updated)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if updated.claim_token is None:
                        pipe.delete(lease_key)
                    else:
                        pipe.set(lease_key, str(updated.lease_expires_at_ms))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 deferred admission exceeded transaction retry limit")

    def resolve_deferred(self, handle: DeferredToolHandle, result: Any) -> DeferredResolveDecision:
        """Resolve one handle with a receipt-first Redis WATCH/MULTI CAS."""
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        data_key, lease_key = self._keys(handle.checkpoint_key)
        receipt_key = self._receipt_key(handle.key)
        receipt_set_key = self._receipt_set_key(handle.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key, receipt_key, receipt_set_key)
                    raw_receipt = pipe.get(receipt_key)
                    helper = InMemoryCheckpointStore()
                    if raw_receipt is not None:
                        receipt = _receipt_from_storage(raw_receipt)
                        if receipt.handle.key != handle.key or receipt.handle_key != handle.key:
                            raise ValueError("deferred_receipt_identity_invalid")
                        helper._deferred_receipts[handle.key] = receipt  # type: ignore[attr-defined]
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is not None:
                        current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=handle.checkpoint_key)
                        helper._store[current.checkpoint_key] = current  # type: ignore[attr-defined]
                    decision = helper.resolve_deferred(handle, result)
                    if decision.kind in {"replayed", "not_admitted", "reconciliation_required"}:
                        pipe.unwatch()
                        return decision
                    updated = helper._store[handle.checkpoint_key]  # type: ignore[attr-defined]
                    receipt = decision.receipt
                    assert receipt is not None
                    payload, _lease = _checkpoint_to_storage(updated)
                    receipt_payload = _receipt_to_storage(receipt)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if updated.claim_token is None:
                        pipe.delete(lease_key)
                    else:
                        pipe.set(lease_key, str(updated.lease_expires_at_ms))
                    pipe.set(receipt_key, receipt_payload)
                    sadd = getattr(pipe, "sadd", None)
                    if callable(sadd):
                        sadd(receipt_set_key, receipt_key)
                    pipe.execute()
                    return decision
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 deferred resolution exceeded transaction retry limit")

    def accept_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        decisions: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        data_key, lease_key = self._keys(checkpoint.checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        return False
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint.checkpoint_key)
                    helper = InMemoryCheckpointStore()
                    helper._store[current.checkpoint_key] = current  # type: ignore[attr-defined]
                    if not helper.accept_deferred_batch(
                        current,
                        decisions=decisions,
                        claim_token=claim_token,
                        expected_revision=expected_revision,
                        claimed_cycle=claimed_cycle,
                    ):
                        pipe.unwatch()
                        return False
                    updated = helper._store[current.checkpoint_key]  # type: ignore[attr-defined]
                    # Exact repeat acceptance is a no-write replay and does
                    # not require another active recovery claim.
                    if updated.revision == current.revision:
                        pipe.unwatch()
                        return True
                    payload, _lease = _checkpoint_to_storage(updated)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    pipe.delete(lease_key)
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis checkpoint v10 deferred reconciliation exceeded transaction retry limit")

    def produce_host_interaction(
        self,
        request: HostInteractionRequest | Mapping[str, Any],
        *,
        admission_context: HostInteractionAdmissionContext,
    ) -> HostInteractionOutcome:
        """Admit a host request and its UI notification in one Redis CAS."""
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        request_value = request if isinstance(request, HostInteractionRequest) else HostInteractionRequest.from_dict(request)
        admission_context.validate()
        if request_value.logical_cycle != admission_context.claimed_cycle:
            raise CheckpointError("host interaction logical cycle does not match its claim", code="host_interaction_stale")
        checkpoint_key = admission_context.checkpoint_key
        record_id = derive_host_interaction_record_id(checkpoint_key, request_value)
        record_key = self._host_record_key(checkpoint_key, request_value.interaction_id)
        notification_id = derive_host_interaction_notification_id(record_id)
        notification_key = self._host_notification_key(notification_id)
        data_key, lease_key = self._keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key, record_key, notification_key)
                    raw, raw_lease, raw_record, notification_raw = pipe.mget([data_key, lease_key, record_key, notification_key])
                    if raw is None:
                        pipe.unwatch()
                        raise CheckpointError("host interaction checkpoint was not found", code="host_interaction_claim_required")
                    if raw_record is not None:
                        record = _host_record_from_storage(
                            raw_record,
                            expected_checkpoint_key=checkpoint_key,
                            expected_key=record_key,
                        )
                        if (
                            record["request_digest"] != request_value.request_digest
                            or record["request"] != request_value.to_dict()
                        ):
                            pipe.unwatch()
                            raise CheckpointError(
                                "host interaction identity or digest conflicts", code="host_interaction_conflict"
                            )
                        if notification_raw is None:
                            pipe.unwatch()
                            raise CheckpointError(
                                "host interaction notification row is missing", code="host_interaction_conflict"
                            )
                        checkpoint = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                        pipe.unwatch()
                        if checkpoint.revision < admission_context.expected_revision + 1:
                            raise CheckpointError("host interaction replay revision is stale", code="host_interaction_stale")
                        return _host_interaction_outcome(
                            record,
                            _notification_from_storage(
                                notification_raw,
                                expected_checkpoint_key=checkpoint_key,
                                expected_key=notification_key,
                            ),
                            status="replayed",
                            checkpoint_revision=checkpoint.revision,
                        )
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    helper = InMemoryCheckpointStore()
                    helper._store[checkpoint_key] = current  # type: ignore[attr-defined]
                    outcome = helper.produce_host_interaction(
                        request_value,
                        admission_context=admission_context,
                    )
                    updated = helper._store[checkpoint_key]  # type: ignore[attr-defined]
                    record = helper._host_interaction_records[(checkpoint_key, request_value.interaction_id)]  # type: ignore[attr-defined]
                    notification = helper._host_interaction_notifications[notification_id]  # type: ignore[attr-defined]
                    payload, lease = _checkpoint_to_storage(updated)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if lease is None:
                        pipe.delete(lease_key)
                    else:
                        pipe.set(lease_key, str(lease))
                    pipe.set(record_key, _host_record_to_storage(record))
                    pipe.set(notification_key, _notification_to_storage(notification))
                    pipe.sadd(self._host_record_set_key(checkpoint_key), record_key)
                    pipe.sadd(self._host_notification_set_key(checkpoint_key), notification_key)
                    pipe.execute()
                    return outcome
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction admission exceeded transaction retry limit")

    def admit_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandReceipt:
        """Admit one closed controller variant under checkpoint CAS."""
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        checkpoint_key = command_value.handle.checkpoint_key
        data_key, lease_key = self._keys(checkpoint_key)
        receipt_key = self._controller_receipt_key(command_value.command_id)
        command_key = f"{receipt_key}{_CONTROLLER_COMMAND_PAYLOAD_SUFFIX}"
        outbox_key = self._controller_outbox_key(command_value.command_id)
        record_key: str | None = None
        record_id: str | None = None
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    record_id = None
                    watch_keys = [data_key, lease_key, receipt_key, command_key, outbox_key]
                    pipe.watch(*watch_keys)
                    raw_receipt = pipe.get(receipt_key)
                    if raw_receipt is not None:
                        receipt = _controller_receipt_from_storage(raw_receipt)
                        if receipt.command_id != command_value.command_id:
                            pipe.unwatch()
                            raise CheckpointError(
                                "redis controller receipt identity conflicts", code="controller_command_conflict"
                            )
                        raw_outbox = pipe.get(outbox_key)
                        if raw_outbox is None:
                            pipe.unwatch()
                            raise CheckpointError("controller wake outbox is missing", code="controller_command_conflict")
                        wake = _controller_wake_from_storage(raw_outbox)
                        if (
                            wake["command_id"] != receipt.command_id
                            or wake["command_digest"] != receipt.command_digest
                            or wake["outbox_state"] != receipt.outbox_state
                            or wake["attempt"] != receipt.outbox_attempt
                        ):
                            pipe.unwatch()
                            raise CheckpointError(
                                "controller wake outbox conflicts with receipt", code="controller_command_conflict"
                            )
                        if receipt.command_digest != command_value.command_digest:
                            pipe.unwatch()
                            raise CheckpointError(
                                "controller command id was reused with a different digest", code="controller_command_conflict"
                            )
                        pipe.unwatch()
                        return receipt
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    if raw is None:
                        pipe.unwatch()
                        raise CheckpointError("controller command checkpoint was not found", code="controller_command_stale")
                    try:
                        current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    except CheckpointError:
                        pipe.unwatch()
                        raise
                    except ValueError:
                        pipe.unwatch()
                        continue
                    active = current.active_host_interaction
                    if command_value.kind in {"host_interaction_response", "resume"} and isinstance(
                        current.suspended_origin, dict
                    ):
                        origin_active = current.suspended_origin.get("active_host_interaction")
                        if command_value.kind == "host_interaction_response" and isinstance(origin_active, dict):
                            active = origin_active
                    if command_value.kind == "host_interaction_response" and isinstance(active, dict):
                        record_id = derive_host_interaction_record_id(checkpoint_key, active)
                    elif command_value.kind == "resume" and isinstance(current.suspended_origin, dict):
                        origin_active = current.suspended_origin.get("active_host_interaction")
                        if isinstance(origin_active, dict):
                            record_id = derive_host_interaction_record_id(checkpoint_key, origin_active)
                            active = origin_active
                    record_key = (
                        self._host_record_key(checkpoint_key, active["interaction_id"])
                        if record_id is not None and isinstance(active, dict)
                        else None
                    )
                    if record_key is not None:
                        pipe.watch(record_key)
                    lease_now_ms = _redis_server_now_ms(pipe)
                    helper = InMemoryCheckpointStore()
                    helper._store[checkpoint_key] = current  # type: ignore[attr-defined]
                    if record_key is not None:
                        raw_record = pipe.get(record_key)
                        if raw_record is not None:
                            record = _host_record_from_storage(
                                raw_record,
                                expected_checkpoint_key=checkpoint_key,
                                expected_key=record_key,
                            )
                            helper._host_interaction_records[(checkpoint_key, record["interaction_id"])] = record  # type: ignore[attr-defined]
                    receipt = helper._admit_controller_command(command_value, lease_now_ms=lease_now_ms)  # type: ignore[attr-defined]
                    updated = helper._store[checkpoint_key]  # type: ignore[attr-defined]
                    staged = None
                    if record_key is not None:
                        records = helper._host_interaction_records  # type: ignore[attr-defined]
                        staged = next((row for row in records.values() if row["record_id"] == record_id), None)
                    payload, lease = _checkpoint_to_storage(updated)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if lease is None:
                        pipe.delete(lease_key)
                    else:
                        pipe.set(lease_key, str(lease))
                    pipe.set(receipt_key, _controller_receipt_to_storage(receipt))
                    pipe.set(
                        outbox_key,
                        _controller_wake_to_storage(
                            {
                                "command_id": receipt.command_id,
                                "command_digest": receipt.command_digest,
                                "outbox_id": derive_controller_receipt_outbox_id(receipt.command_id, receipt.command_digest),
                                "outbox_action": receipt.outbox_action,
                                "outbox_destination": receipt.outbox_destination,
                                "outbox_state": receipt.outbox_state,
                                "attempt": receipt.outbox_attempt,
                                "claim_token": None,
                                "lease_expires_at_ms": None,
                                "delivered_at_ms": None,
                                "last_error": None,
                            }
                        ),
                    )
                    pipe.set(command_key, _controller_command_to_storage(command_value))
                    pipe.sadd(self._controller_receipt_set_key(checkpoint_key), receipt_key)
                    if staged is not None:
                        pipe.set(record_key, _host_record_to_storage(staged))
                        pipe.sadd(self._host_record_set_key(checkpoint_key), record_key)
                    pipe.execute()
                    return receipt
                except self._watch_error:
                    continue
        raise RuntimeError("redis controller admission exceeded transaction retry limit")

    def resolve_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandResolution:
        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        resolution_lock = getattr(self, "_controller_resolution_lock", None)
        if resolution_lock is None:
            resolution_lock = RLock()
            self._controller_resolution_lock = resolution_lock
        with resolution_lock:
            existed = self._client.get(self._controller_receipt_key(command_value.command_id)) is not None
            try:
                receipt = self.admit_controller_command(command_value)
            except CheckpointError as exc:
                return ControllerCommandResolution(kind="rejected", error=getattr(exc, "code", None) or str(exc))
        checkpoint = self.load_checkpoint(command_value.handle.checkpoint_key)
        if checkpoint is None:
            return ControllerCommandResolution(kind="rejected", error="controller_command_stale")
        return ControllerCommandResolution(
            kind="replayed" if existed else "applied",
            receipt=receipt,
            wake=ControllerWake(
                action=receipt.outbox_action,
                destination=receipt.outbox_destination,
                logical_cycle=(
                    int(command_value.command["logical_cycle"])
                    if command_value.kind == "host_interaction_response"
                    else checkpoint.cycle_index + 1
                ),
                claim_mode="recovery" if receipt.outbox_action == "recovery_dispatch" else "none",
            ),
        )

    def get_controller_command_receipt(self, command_id: str) -> ControllerCommandReceipt | None:
        raw = self._client.get(self._controller_receipt_key(command_id))
        if raw is None:
            return None
        receipt = _controller_receipt_from_storage(raw)
        if receipt.command_id != command_id:
            raise ValueError("redis controller receipt identity conflicts")
        outbox_raw = self._client.get(self._controller_outbox_key(command_id))
        if outbox_raw is None:
            raise ValueError("redis controller wake outbox is missing")
        wake = _controller_wake_from_storage(outbox_raw)
        if (
            wake["command_id"] != receipt.command_id
            or wake["command_digest"] != receipt.command_digest
            or wake["outbox_state"] != receipt.outbox_state
            or wake["attempt"] != receipt.outbox_attempt
        ):
            raise ValueError("redis controller receipt and wake outbox conflict")
        return receipt

    def get_host_interaction_notification(self, notification_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._host_notification_key(notification_id))
        return (
            _notification_from_storage(raw, expected_key=self._host_notification_key(notification_id))
            if raw is not None
            else None
        )

    def _find_resolved_pending_host_interaction(self, *, checkpoint_key: str) -> dict[str, Any] | None:
        found: dict[str, Any] | None = None
        for candidate in self._client.smembers(self._host_record_set_key(checkpoint_key)):
            record_key = candidate.decode("utf-8") if isinstance(candidate, bytes) else str(candidate)
            raw = self._client.get(record_key)
            if raw is None:
                continue
            record = _host_record_from_storage(raw, expected_key=record_key)
            if record["checkpoint_key"] != checkpoint_key or record["state"] != "resolved_pending":
                continue
            if found is not None:
                raise CheckpointError(
                    "checkpoint has multiple pending host interaction responses",
                    code="host_interaction_conflict",
                )
            found = record
        return found

    @staticmethod
    def _controller_wake_matches_receipt(receipt: ControllerCommandReceipt, wake: Mapping[str, Any]) -> None:
        if (
            wake["command_id"] != receipt.command_id
            or wake["command_digest"] != receipt.command_digest
            or wake["outbox_state"] != receipt.outbox_state
            or wake["attempt"] != receipt.outbox_attempt
        ):
            raise CheckpointError("controller receipt and wake outbox conflict", code="controller_command_conflict")

    def _redis_controller_wake_snapshot(
        self,
        pipe: Any,
        command_id: str,
    ) -> tuple[ControllerCommandReceipt, dict[str, Any]] | None:
        raw_receipt = pipe.get(self._controller_receipt_key(command_id))
        if raw_receipt is None:
            return None
        raw_wake = pipe.get(self._controller_outbox_key(command_id))
        if raw_wake is None:
            raise CheckpointError("controller wake outbox is missing", code="controller_command_conflict")
        receipt = _controller_receipt_from_storage(raw_receipt)
        if receipt.command_id != command_id:
            raise CheckpointError("redis controller receipt identity conflicts", code="controller_command_conflict")
        wake = _controller_wake_from_storage(raw_wake)
        self._controller_wake_matches_receipt(receipt, wake)
        return receipt, wake

    def _redis_controller_wake_write(
        self,
        pipe: Any,
        receipt: ControllerCommandReceipt,
        wake: Mapping[str, Any],
    ) -> None:
        pipe.set(self._controller_receipt_key(receipt.command_id), _controller_receipt_to_storage(receipt))
        pipe.set(self._controller_outbox_key(receipt.command_id), _controller_wake_to_storage(wake))

    def claim_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(claim_token, str) or not claim_token.strip():
            raise ValueError("controller wake claim_token must be non-empty")
        if isinstance(lease_expires_at_ms, bool) or not isinstance(lease_expires_at_ms, int) or lease_expires_at_ms <= now_ms:
            raise ValueError("controller wake lease must be greater than now_ms")
        receipt_key = self._controller_receipt_key(command_id)
        outbox_key = self._controller_outbox_key(command_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(receipt_key, outbox_key)
                    snapshot = self._redis_controller_wake_snapshot(pipe, command_id)
                    if snapshot is None:
                        pipe.unwatch()
                        return None
                    receipt, current = snapshot
                    if receipt.command_digest != command_digest:
                        pipe.unwatch()
                        raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                    if current["outbox_action"] == "none" or current["outbox_state"] == "delivered":
                        pipe.unwatch()
                        return current
                    if current["outbox_state"] == "ambiguous":
                        pipe.unwatch()
                        raise CheckpointError("controller wake requires reconciliation", code="controller_command_stale")
                    if current["outbox_state"] == "claimed":
                        if current["claim_token"] == claim_token:
                            pipe.unwatch()
                            return current
                        if int(current["lease_expires_at_ms"] or 0) > now_ms:
                            pipe.unwatch()
                            raise CheckpointError("controller wake is claimed by another owner", code="controller_command_stale")
                    staged = dict(current)
                    staged.update(
                        outbox_state="claimed",
                        claim_token=claim_token,
                        lease_expires_at_ms=lease_expires_at_ms,
                        attempt=int(current["attempt"]) + 1,
                    )
                    updated = replace(
                        receipt,
                        outbox_state="claimed",
                        outbox_attempt=int(staged["attempt"]),
                    )
                    pipe.multi()
                    self._redis_controller_wake_write(pipe, updated, staged)
                    pipe.execute()
                    return staged
                except self._watch_error:
                    continue
        raise RuntimeError("redis controller wake claim exceeded transaction retry limit")

    def complete_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if outcome not in {"delivered", "ambiguous"}:
            raise ValueError("controller wake completion outcome must be delivered or ambiguous")
        receipt_key = self._controller_receipt_key(command_id)
        outbox_key = self._controller_outbox_key(command_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(receipt_key, outbox_key)
                    snapshot = self._redis_controller_wake_snapshot(pipe, command_id)
                    if snapshot is None:
                        pipe.unwatch()
                        return None
                    receipt, current = snapshot
                    if receipt.command_digest != command_digest:
                        pipe.unwatch()
                        raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                    if current["outbox_state"] in {"delivered", "ambiguous"}:
                        if current["outbox_state"] == outcome:
                            pipe.unwatch()
                            return current
                        pipe.unwatch()
                        raise CheckpointError("controller wake has already completed", code="controller_command_stale")
                    if (
                        current["outbox_state"] != "claimed"
                        or current["claim_token"] != claim_token
                        or current["attempt"] != attempt
                    ):
                        pipe.unwatch()
                        raise CheckpointError("controller wake owner or attempt is stale", code="controller_command_stale")
                    staged = dict(current)
                    staged.update(
                        outbox_state=outcome,
                        claim_token=None,
                        lease_expires_at_ms=None,
                        delivered_at_ms=now_ms if outcome == "delivered" else None,
                        last_error=error,
                    )
                    updated = replace(receipt, outbox_state=outcome)
                    pipe.multi()
                    self._redis_controller_wake_write(pipe, updated, staged)
                    pipe.execute()
                    return staged
                except self._watch_error:
                    continue
        raise RuntimeError("redis controller wake completion exceeded transaction retry limit")

    def reconcile_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        outcome: str,
        now_ms: int,
    ) -> dict[str, Any] | None:
        if outcome not in {"delivered", "retry"}:
            raise ValueError("controller wake reconciliation outcome must be delivered or retry")
        receipt_key = self._controller_receipt_key(command_id)
        outbox_key = self._controller_outbox_key(command_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(receipt_key, outbox_key)
                    snapshot = self._redis_controller_wake_snapshot(pipe, command_id)
                    if snapshot is None:
                        pipe.unwatch()
                        return None
                    receipt, current = snapshot
                    if receipt.command_digest != command_digest:
                        pipe.unwatch()
                        raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                    target = "delivered" if outcome == "delivered" else "pending"
                    if current["outbox_state"] == target:
                        pipe.unwatch()
                        return current
                    if current["outbox_state"] != "ambiguous":
                        pipe.unwatch()
                        raise CheckpointError("controller wake is not ambiguous", code="controller_command_stale")
                    staged = dict(current)
                    staged.update(
                        outbox_state=target,
                        delivered_at_ms=now_ms if target == "delivered" else None,
                        last_error=None,
                    )
                    updated = replace(receipt, outbox_state=target)
                    pipe.multi()
                    self._redis_controller_wake_write(pipe, updated, staged)
                    pipe.execute()
                    return staged
                except self._watch_error:
                    continue
        raise RuntimeError("redis controller wake reconciliation exceeded transaction retry limit")

    def _reap_controller_command_wake(self, *, command_id: str, now_ms: int) -> dict[str, Any] | None:
        receipt_key = self._controller_receipt_key(command_id)
        outbox_key = self._controller_outbox_key(command_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(receipt_key, outbox_key)
                    snapshot = self._redis_controller_wake_snapshot(pipe, command_id)
                    if snapshot is None:
                        pipe.unwatch()
                        return None
                    receipt, current = snapshot
                    if current["outbox_state"] != "claimed" or int(current["lease_expires_at_ms"] or 0) > now_ms:
                        pipe.unwatch()
                        return current
                    staged = dict(current)
                    staged.update(outbox_state="pending", claim_token=None, lease_expires_at_ms=None)
                    updated = replace(receipt, outbox_state="pending")
                    pipe.multi()
                    self._redis_controller_wake_write(pipe, updated, staged)
                    pipe.execute()
                    return staged
                except self._watch_error:
                    continue
        raise RuntimeError("redis controller wake reaper exceeded transaction retry limit")

    def reap_controller_command_wakes(self, checkpoint_key: str, now_ms: int) -> list[dict[str, Any]]:
        candidates: list[tuple[int, str]] = []
        for member in self._client.smembers(self._controller_receipt_set_key(checkpoint_key)):
            receipt_key = member.decode("utf-8") if isinstance(member, bytes) else str(member)
            raw_receipt = self._client.get(receipt_key)
            if raw_receipt is None:
                continue
            receipt = _controller_receipt_from_storage(raw_receipt)
            if receipt.handle.checkpoint_key != checkpoint_key:
                continue
            raw_wake = self._client.get(f"{receipt_key}{_CONTROLLER_OUTBOX_SUFFIX}")
            if raw_wake is None:
                raise CheckpointError("controller wake outbox is missing", code="controller_command_conflict")
            wake = _controller_wake_from_storage(raw_wake)
            if wake["outbox_action"] != "recovery_dispatch" or not (
                wake["outbox_state"] == "pending"
                or (wake["outbox_state"] == "claimed" and int(wake["lease_expires_at_ms"] or 0) <= now_ms)
            ):
                continue
            candidates.append((receipt.expected_revision, receipt.command_id))
        candidates.sort()
        rows: list[dict[str, Any]] = []
        for _expected_revision, command_id in candidates:
            row = self._reap_controller_command_wake(command_id=command_id, now_ms=now_ms)
            if row is not None and row["outbox_action"] == "recovery_dispatch" and row["outbox_state"] == "pending":
                rows.append(row)
        return rows

    def get_controller_command(self, command_id: str) -> ControllerCommand | None:
        receipt = self.get_controller_command_receipt(command_id)
        if receipt is None:
            return None
        raw = self._client.get(f"{self._controller_receipt_key(command_id)}{_CONTROLLER_COMMAND_PAYLOAD_SUFFIX}")
        if raw is None:
            raise ValueError("redis controller command payload is missing")
        command = _controller_command_from_storage(raw)
        if command.command_id != receipt.command_id or command.command_digest != receipt.command_digest:
            raise ValueError("redis controller command and receipt conflict")
        return command

    def claim_and_consume_host_interaction_response(self, envelope: Mapping[str, Any]) -> HostInteractionRecoveryResult:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        try:
            envelope_value = (
                envelope
                if isinstance(envelope, HostInteractionRecoveryEnvelope)
                else HostInteractionRecoveryEnvelope.from_dict(envelope)
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(str(exc), code="host_interaction_recovery_stale") from exc
        checkpoint_key = envelope_value.checkpoint_key
        data_key, lease_key = self._keys(checkpoint_key)
        record_key = self._host_record_key(checkpoint_key, envelope_value.interaction_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(data_key, lease_key, record_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    raw_record = pipe.get(record_key)
                    if raw is None or raw_record is None:
                        pipe.unwatch()
                        raise CheckpointError(
                            "host interaction recovery record was not found", code="host_interaction_recovery_stale"
                        )
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    record = _host_record_from_storage(
                        raw_record,
                        expected_checkpoint_key=checkpoint_key,
                        expected_key=record_key,
                    )
                    lease_now_ms = _redis_server_now_ms(pipe) if record["state"] == "resolved_pending" else None
                    helper = InMemoryCheckpointStore()
                    helper._store[checkpoint_key] = current  # type: ignore[attr-defined]
                    helper._host_interaction_records[(checkpoint_key, record["interaction_id"])] = record  # type: ignore[attr-defined]
                    result = helper._claim_and_consume_host_interaction_response(  # type: ignore[attr-defined]
                        envelope_value,
                        lease_now_ms=lease_now_ms,
                    )
                    updated = helper._store[checkpoint_key]  # type: ignore[attr-defined]
                    if result.kind != "applied":
                        pipe.unwatch()
                        return result
                    updated_record = helper._host_interaction_records[(checkpoint_key, record["interaction_id"])]  # type: ignore[attr-defined]
                    payload, lease = _checkpoint_to_storage(updated)
                    pipe.multi()
                    pipe.set(data_key, payload)
                    if lease is None:
                        pipe.delete(lease_key)
                    else:
                        pipe.set(lease_key, str(lease))
                    pipe.set(record_key, _host_record_to_storage(updated_record))
                    pipe.execute()
                    return result
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction recovery exceeded transaction retry limit")

    def reap_host_interaction_record(self, *, record_id: str, checkpoint_key: str, now_ms: int) -> bool:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise CheckpointError("now_ms is invalid", code="host_interaction_claim_required")
        data_key, lease_key = self._keys(checkpoint_key)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    index_key = self._host_record_set_key(checkpoint_key)
                    pipe.watch(data_key, lease_key, index_key)
                    raw, raw_lease = pipe.mget([data_key, lease_key])
                    candidate_keys = tuple(pipe.smembers(index_key))
                    record_key: str | None = None
                    raw_record: str | bytes | None = None
                    for candidate in candidate_keys:
                        candidate_value = candidate.decode("utf-8") if isinstance(candidate, bytes) else candidate
                        candidate_raw = pipe.get(candidate_value)
                        if candidate_raw is None:
                            continue
                        candidate_record = _host_record_from_storage(candidate_raw, expected_key=candidate_value)
                        if candidate_record["record_id"] == record_id and candidate_record["checkpoint_key"] == checkpoint_key:
                            record_key = candidate_value
                            raw_record = candidate_raw
                            break
                    if raw is None or raw_record is None or record_key is None:
                        pipe.unwatch()
                        return False
                    pipe.watch(record_key)
                    current = _checkpoint_from_storage(raw, raw_lease, checkpoint_key=checkpoint_key)
                    record = _host_record_from_storage(
                        raw_record,
                        expected_checkpoint_key=checkpoint_key,
                        expected_key=record_key,
                    )
                    helper = InMemoryCheckpointStore()
                    helper._store[checkpoint_key] = current  # type: ignore[attr-defined]
                    helper._host_interaction_records[(checkpoint_key, record["interaction_id"])] = record  # type: ignore[attr-defined]
                    if not helper.reap_host_interaction_record(record_id=record_id, checkpoint_key=checkpoint_key, now_ms=now_ms):
                        pipe.unwatch()
                        return False
                    updated_record = helper._host_interaction_records[(checkpoint_key, record["interaction_id"])]  # type: ignore[attr-defined]
                    pipe.multi()
                    pipe.set(record_key, _host_record_to_storage(updated_record))
                    pipe.execute()
                    return True
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction recovery reap exceeded transaction retry limit")

    def claim_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> dict[str, Any] | None:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        notification_key = self._host_notification_key(notification_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(notification_key)
                    raw = pipe.get(notification_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    row = _notification_from_storage(raw, expected_key=notification_key)
                    before = deepcopy(row)
                    helper = InMemoryCheckpointStore()
                    helper._host_interaction_notifications[notification_id] = row  # type: ignore[attr-defined]
                    result = helper.claim_host_interaction_notification(
                        notification_id=notification_id,
                        payload_digest=payload_digest,
                        claim_token=claim_token,
                        lease_expires_at_ms=lease_expires_at_ms,
                        now_ms=now_ms,
                    )
                    if result is None:
                        pipe.unwatch()
                        return None
                    if result == before:
                        pipe.unwatch()
                        return result
                    pipe.multi()
                    pipe.set(notification_key, _notification_to_storage(result))
                    pipe.execute()
                    return result
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction notification claim exceeded transaction retry limit")

    def complete_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        notification_key = self._host_notification_key(notification_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(notification_key)
                    raw = pipe.get(notification_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    row = _notification_from_storage(raw, expected_key=notification_key)
                    before = deepcopy(row)
                    helper = InMemoryCheckpointStore()
                    helper._host_interaction_notifications[notification_id] = row  # type: ignore[attr-defined]
                    result = helper.complete_host_interaction_notification(
                        notification_id=notification_id,
                        payload_digest=payload_digest,
                        claim_token=claim_token,
                        attempt=attempt,
                        outcome=outcome,
                        now_ms=now_ms,
                        error=error,
                    )
                    if result == before:
                        pipe.unwatch()
                        return result
                    if result is None:
                        pipe.unwatch()
                        return None
                    pipe.multi()
                    pipe.set(notification_key, _notification_to_storage(result))
                    pipe.execute()
                    return result
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction notification completion exceeded transaction retry limit")

    def reconcile_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        outcome: str,
        now_ms: int,
        abort_reason: str | None = None,
    ) -> dict[str, Any] | None:
        from vv_agent.runtime.stores.memory import InMemoryCheckpointStore

        notification_key = self._host_notification_key(notification_id)
        with self._client.pipeline() as pipe:
            for _attempt in range(_TRANSACTION_MAX_ATTEMPTS):
                try:
                    pipe.watch(notification_key)
                    raw = pipe.get(notification_key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    row = _notification_from_storage(raw, expected_key=notification_key)
                    before = deepcopy(row)
                    helper = InMemoryCheckpointStore()
                    helper._host_interaction_notifications[notification_id] = row  # type: ignore[attr-defined]
                    result = helper.reconcile_host_interaction_notification(
                        notification_id=notification_id,
                        payload_digest=payload_digest,
                        outcome=outcome,
                        now_ms=now_ms,
                        abort_reason=abort_reason,
                    )
                    if result == before:
                        pipe.unwatch()
                        return result
                    if result is None:
                        pipe.unwatch()
                        return None
                    pipe.multi()
                    pipe.set(notification_key, _notification_to_storage(result))
                    pipe.execute()
                    return result
                except self._watch_error:
                    continue
        raise RuntimeError("redis host interaction notification reconciliation exceeded transaction retry limit")

    @staticmethod
    def data_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    @classmethod
    def _keys(cls, checkpoint_key: str) -> tuple[str, str]:
        data_key = cls.data_key(checkpoint_key)
        return data_key, f"{data_key}:lease"

    @staticmethod
    def _receipt_key(handle_key: str) -> str:
        return f"{_DEFERRED_RECEIPT_PREFIX}{handle_key}"

    @staticmethod
    def _receipt_set_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_DEFERRED_RECEIPT_SET_PREFIX}{digest}"

    @staticmethod
    def _controller_receipt_key(command_id: str) -> str:
        return f"{_CONTROLLER_RECEIPT_PREFIX}{hashlib.sha256(command_id.encode('utf-8')).hexdigest()}"

    @classmethod
    def _controller_outbox_key(cls, command_id: str) -> str:
        return f"{cls._controller_receipt_key(command_id)}{_CONTROLLER_OUTBOX_SUFFIX}"

    @staticmethod
    def _controller_receipt_set_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_CONTROLLER_RECEIPT_SET_PREFIX}{digest}"

    @staticmethod
    def _host_record_key(checkpoint_key: str, interaction_id: str) -> str:
        # Rust and the central store contract address the durable record by
        # its composite owner identity, not by the derived record_id.  The NUL
        # frame prevents concatenation collisions while remaining portable
        # across Redis clients and languages.
        identity = f"{checkpoint_key}\x00{interaction_id}".encode()
        return f"{_HOST_RECORD_PREFIX}{hashlib.sha256(identity).hexdigest()}"

    @staticmethod
    def _host_record_set_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_HOST_RECORD_SET_PREFIX}{digest}"

    @staticmethod
    def _host_notification_key(notification_id: str) -> str:
        return f"{_HOST_NOTIFICATION_PREFIX}{hashlib.sha256(notification_id.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _host_notification_set_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_HOST_NOTIFICATION_SET_PREFIX}{digest}"

    @staticmethod
    def _dispatch_outbox_key(dispatch_id: str) -> str:
        return f"{_DISPATCH_OUTBOX_PREFIX}{hashlib.sha256(dispatch_id.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _dispatch_outbox_set_key(checkpoint_key: str) -> str:
        digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
        return f"{_DISPATCH_OUTBOX_SET_PREFIX}{digest}"


def _host_record_to_storage(record: Mapping[str, Any]) -> str:
    validate_host_interaction_record(record, checkpoint_key=str(record.get("checkpoint_key")))
    return canonical_json_bytes(dict(record), "redis host interaction record").decode("utf-8")


def _dispatch_outbox_to_storage(record: DispatchOutboxRecord) -> str:
    return canonical_json_bytes(record.to_dict(), "redis distributed dispatch outbox").decode("utf-8")


def _dispatch_outbox_from_storage(
    raw: str | bytes,
    *,
    expected_checkpoint_key: str | None = None,
    expected_key: str | None = None,
) -> DispatchOutboxRecord:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis distributed dispatch outbox is invalid") from exc
    record = DispatchOutboxRecord.from_dict(payload)
    if expected_checkpoint_key is not None and record.checkpoint_key != expected_checkpoint_key:
        raise CheckpointError("redis distributed dispatch checkpoint identity conflicts", code="dispatch_outbox_conflict")
    if expected_key is not None and expected_key != RedisCheckpointStore._dispatch_outbox_key(record.dispatch_id):
        raise CheckpointError("redis distributed dispatch outbox identity conflicts", code="dispatch_outbox_conflict")
    return record


def _host_record_from_storage(
    raw: str | bytes,
    *,
    expected_checkpoint_key: str | None = None,
    expected_key: str | None = None,
) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis host interaction record is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis host interaction record must be an object")
    if expected_checkpoint_key is not None and payload.get("checkpoint_key") != expected_checkpoint_key:
        raise CheckpointError("redis host interaction record checkpoint identity conflicts", code="host_interaction_conflict")
    try:
        record = validate_host_interaction_record(payload, checkpoint_key=str(payload.get("checkpoint_key")))
    except (TypeError, ValueError) as exc:
        raise ValueError("redis host interaction record is invalid") from exc
    if expected_key is not None and expected_key != RedisCheckpointStore._host_record_key(
        record["checkpoint_key"], record["interaction_id"]
    ):
        raise CheckpointError("redis host interaction record identity conflicts", code="host_interaction_conflict")
    return record


def _notification_to_storage(row: Mapping[str, Any]) -> str:
    checked = _validate_redis_notification_row(row)
    return canonical_json_bytes(checked, "redis host interaction notification").decode("utf-8")


def _notification_from_storage(
    raw: str | bytes,
    *,
    expected_checkpoint_key: str | None = None,
    expected_key: str | None = None,
) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis host interaction notification is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis host interaction notification must be an object")
    try:
        payload = _validate_redis_notification_row(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("redis host interaction notification is invalid") from exc
    if expected_checkpoint_key is not None and payload["checkpoint_key"] != expected_checkpoint_key:
        raise CheckpointError(
            "redis host interaction notification checkpoint identity conflicts", code="host_interaction_conflict"
        )
    if expected_key is not None and expected_key != RedisCheckpointStore._host_notification_key(str(payload["notification_id"])):
        raise CheckpointError("redis host interaction notification identity conflicts", code="host_interaction_conflict")
    return payload


def _validate_redis_notification_row(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "notification_id",
        "checkpoint_key",
        "record_id",
        "payload",
        "payload_digest",
        "outbox_state",
        "claim_token",
        "lease_expires_at_ms",
        "attempt",
        "delivered_at_ms",
        "aborted_at_ms",
        "abort_reason",
        "last_error",
    }
    if set(row) != expected:
        raise ValueError("notification row fields are invalid")
    checked = dict(row)
    notification_id = _redis_wire_text(checked["notification_id"], "notification_id", 512)
    record_id = _redis_wire_text(checked["record_id"], "record_id", 512)
    checked["notification_id"] = notification_id
    checked["checkpoint_key"] = _redis_wire_text(checked["checkpoint_key"], "checkpoint_key", 512)
    checked["record_id"] = record_id
    validate_host_interaction_notification(checked["payload"], notification_id=notification_id, record_id=record_id)
    checked["payload_digest"] = _redis_wire_digest(checked["payload_digest"], "payload_digest")
    if checked["payload_digest"] != canonical_json_sha256(checked["payload"], "notification_payload"):
        raise ValueError("notification payload digest conflicts")
    state = checked["outbox_state"]
    if state not in {"pending", "claimed", "delivered", "ambiguous", "aborted"}:
        raise ValueError("notification state is invalid")
    checked["attempt"] = _redis_wire_int(checked["attempt"], "notification attempt", minimum=0)
    claim_token = checked["claim_token"]
    lease = checked["lease_expires_at_ms"]
    if (claim_token is None) != (lease is None):
        raise ValueError("notification claim and lease are inconsistent")
    if claim_token is not None:
        checked["claim_token"] = _redis_wire_text(claim_token, "notification claim_token", 512)
    if lease is not None:
        checked["lease_expires_at_ms"] = _redis_wire_int(lease, "notification lease_expires_at_ms", minimum=0)
    if state == "claimed" and claim_token is None:
        raise ValueError("claimed notification has no owner")
    if state != "claimed" and claim_token is not None:
        raise ValueError("unclaimed notification has an owner")
    for field_name in ("delivered_at_ms", "aborted_at_ms"):
        value = checked[field_name]
        if value is not None:
            checked[field_name] = _redis_wire_int(value, f"notification {field_name}", minimum=0)
    abort_reason = checked["abort_reason"]
    if abort_reason is not None:
        checked["abort_reason"] = _redis_wire_text(abort_reason, "notification abort_reason", 65536)
    last_error = checked["last_error"]
    if last_error is not None:
        checked["last_error"] = _redis_wire_text(last_error, "notification last_error", 65536)
    if state == "delivered" and (
        checked["delivered_at_ms"] is None or checked["aborted_at_ms"] is not None or abort_reason is not None
    ):
        raise ValueError("delivered notification fields are invalid")
    if state == "aborted" and (
        checked["aborted_at_ms"] is None or checked["delivered_at_ms"] is not None or abort_reason is None
    ):
        raise ValueError("aborted notification fields are invalid")
    if state not in {"delivered", "aborted"} and (
        checked["delivered_at_ms"] is not None or checked["aborted_at_ms"] is not None or abort_reason is not None
    ):
        raise ValueError("pending notification fields are invalid")
    return checked


def _controller_receipt_to_storage(receipt: ControllerCommandReceipt) -> str:
    return canonical_json_bytes(receipt.to_dict(), "redis controller command receipt").decode("utf-8")


def _controller_receipt_from_storage(raw: str | bytes) -> ControllerCommandReceipt:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis controller command receipt is invalid") from exc
    return ControllerCommandReceipt.from_dict(payload)


def _controller_wake_to_storage(row: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": "vv-agent.controller-command-wake.v1",
        "command_id": row["command_id"],
        "command_digest": row["command_digest"],
        "outbox_id": row["outbox_id"],
        "outbox_action": row["outbox_action"],
        "outbox_destination": row["outbox_destination"],
        "outbox_state": row["outbox_state"],
        "attempt": row["attempt"],
        "claim_token": row["claim_token"],
        "lease_expires_at_ms": row["lease_expires_at_ms"],
        "delivered_at_ms": row["delivered_at_ms"],
        "last_error": row.get("last_error"),
    }
    return canonical_json_bytes(payload, "redis controller wake outbox").decode("utf-8")


def _controller_wake_from_storage(raw: str | bytes) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("controller wake outbox must be an object")
        expected = {
            "schema_version",
            "command_id",
            "command_digest",
            "outbox_id",
            "outbox_action",
            "outbox_destination",
            "outbox_state",
            "attempt",
            "claim_token",
            "lease_expires_at_ms",
            "delivered_at_ms",
            "last_error",
        }
        if set(payload) != expected or payload["schema_version"] != "vv-agent.controller-command-wake.v1":
            raise ValueError("controller wake outbox fields are invalid")
        row = dict(payload)
        command_id = _redis_wire_text(row["command_id"], "controller wake command_id", 512)
        command_digest = _redis_wire_digest(row["command_digest"], "controller wake command_digest")
        row["command_id"] = command_id
        row["command_digest"] = command_digest
        if row["outbox_id"] != derive_controller_receipt_outbox_id(command_id, command_digest):
            raise ValueError("controller wake outbox identity conflicts")
        if row["outbox_action"] not in {"none", "recovery_dispatch"}:
            raise ValueError("controller wake action is invalid")
        if row["outbox_action"] == "none" and (
            row["outbox_destination"] is not None or row["outbox_state"] != "delivered" or row["attempt"] != 0
        ):
            raise ValueError("controller wake none action is invalid")
        if row["outbox_action"] == "recovery_dispatch" and row["outbox_destination"] != "distributed_advance":
            raise ValueError("controller wake destination is invalid")
        if row["outbox_state"] not in {"pending", "claimed", "delivered", "ambiguous"}:
            raise ValueError("controller wake state is invalid")
        row["attempt"] = _redis_wire_int(row["attempt"], "controller wake attempt", minimum=0)
        if (row["claim_token"] is None) != (row["lease_expires_at_ms"] is None):
            raise ValueError("controller wake claim and lease are inconsistent")
        if row["outbox_state"] == "claimed" and row["claim_token"] is None:
            raise ValueError("claimed controller wake has no owner")
        if row["outbox_state"] != "claimed" and row["claim_token"] is not None:
            raise ValueError("unclaimed controller wake has an owner")
        if row["outbox_action"] == "recovery_dispatch" and row["outbox_state"] != "pending" and row["attempt"] < 1:
            raise ValueError("controller wake recovery action requires a claimed attempt")
        if row["claim_token"] is not None:
            row["claim_token"] = _redis_wire_text(row["claim_token"], "controller wake claim_token", 512)
        if row["lease_expires_at_ms"] is not None:
            row["lease_expires_at_ms"] = _redis_wire_int(
                row["lease_expires_at_ms"], "controller wake lease_expires_at_ms", minimum=0
            )
        if row["delivered_at_ms"] is not None:
            row["delivered_at_ms"] = _redis_wire_int(row["delivered_at_ms"], "controller wake delivered_at_ms", minimum=0)
        if row["last_error"] is not None:
            row["last_error"] = _redis_wire_text(row["last_error"], "controller wake last_error", 65536)
        return row
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis controller wake outbox is invalid") from exc


def _redis_wire_text(value: object, field_name: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} is invalid")
    return value


def _redis_wire_digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} is invalid")
    return value


def _redis_wire_int(value: object, field_name: str, *, minimum: int) -> int:
    maximum = (1 << 53) - 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{field_name} is invalid")
    return value


def _controller_command_to_storage(command: ControllerCommand) -> str:
    return canonical_json_bytes(command.to_dict(), "redis controller command").decode("utf-8")


def _controller_command_from_storage(raw: str | bytes) -> ControllerCommand:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis controller command is invalid") from exc
    return ControllerCommand.from_dict(payload)


def _host_interaction_outcome(
    record: Mapping[str, Any],
    notification: Mapping[str, Any],
    *,
    status: str,
    checkpoint_revision: int,
) -> HostInteractionOutcome:
    return HostInteractionOutcome(
        interaction_id=str(record["interaction_id"]),
        logical_cycle=int(record["logical_cycle"]),
        checkpoint_revision=checkpoint_revision,
        status=status,
        # Delivery is an independent at-least-once lifecycle.  A producer
        # outcome only reports that the notification is durably pending.
        outbox_state="pending",
        record_id=str(record["record_id"]),
        notification_id=str(notification["notification_id"]),
        notification_payload_digest=str(notification["payload_digest"]),
        notification_outbox_action="host_interaction_notification",
        notification_outbox_destination="host_interaction_observer",
    )


def _checkpoint_to_storage(checkpoint: Checkpoint) -> tuple[str, int | None]:
    payload = checkpoint_to_dict(checkpoint)
    lease = payload.pop("lease_expires_at_ms")
    return canonical_json_bytes(payload, "redis checkpoint v10").decode("utf-8"), lease


def _receipt_to_storage(receipt: DeferredResolutionReceipt) -> str:
    return canonical_json_bytes(receipt.to_dict(), "redis deferred resolution receipt").decode("utf-8")


def _receipt_from_storage(raw: str | bytes) -> DeferredResolutionReceipt:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis deferred resolution receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis deferred resolution receipt must be an object")
    return DeferredResolutionReceipt.from_dict(payload)


def _checkpoint_from_storage(
    raw: str | bytes,
    raw_lease: object | None,
    *,
    checkpoint_key: str,
) -> Checkpoint:
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("redis checkpoint v10 payload is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("redis checkpoint v10 payload must be an object")
    if payload.get("checkpoint_key") != checkpoint_key:
        raise CheckpointError(
            "redis checkpoint payload key does not match the requested key",
            code="checkpoint_store_conflict",
        )
    payload["lease_expires_at_ms"] = _lease_from_storage(raw_lease)
    return checkpoint_from_dict(payload)


def _lease_from_storage(raw_lease: object | None) -> int | None:
    if raw_lease is None:
        return None
    if isinstance(raw_lease, bool) or not isinstance(raw_lease, str | bytes | int):
        raise ValueError("redis checkpoint v10 lease must be an integer")
    try:
        return int(raw_lease)
    except ValueError as exc:
        raise ValueError("redis checkpoint v10 lease must be an integer") from exc


def _redis_server_now_ms(client: Any) -> int:
    time_method = getattr(client, "time", None)
    if not callable(time_method):
        return 0
    seconds, microseconds = time_method()
    return int(seconds) * 1000 + int(microseconds) // 1000
