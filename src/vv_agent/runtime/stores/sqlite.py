from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

import vv_agent.events as run_events
from vv_agent.checkpoint import CheckpointError, EventCursor, canonical_json_sha256
from vv_agent.deferred import DeferredResolutionReceipt, DeferredResolveDecision, DeferredToolHandle
from vv_agent.runtime.checkpoint_codec import (
    _strict_json_loads,
    checkpoint_from_dict,
    checkpoint_from_json,
    checkpoint_to_dict,
    checkpoint_to_json,
)
from vv_agent.runtime.controller import (
    HOST_RECORD_SCHEMA,
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
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
    check_claim,
    merge_event_outbox,
    prepare_claimed_terminal,
    prepare_deferred_acceptance,
    prepare_deferred_admission,
    prepare_deferred_resolution,
    prepare_event_delivery,
    prepare_tool_receipt,
    prepare_unclaimed_terminal,
    validate_checkpoint_creation,
    validate_model_journal_accounting,
)
from vv_agent.runtime.stores.controller_store import (
    prepare_controller_command,
    prepare_host_interaction,
    prepare_host_response_consumption,
    validate_host_tool_receipt_replay,
)
from vv_agent.types import AgentStatus


class SqliteCheckpointStore:
    """Current checkpoint store backed by SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        raw_path = str(db_path)
        self._db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            self._lock = RLock()
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_table()
            self._conn.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            with suppress(BaseException):
                self._conn.close()
            raise

    def _create_table(self) -> None:
        definitions = (
            ("table", "checkpoints", _CREATE_TABLE_SQL),
            ("index", "checkpoints_status_idx", _CREATE_INDEX_SQL),
            ("table", "host_interaction_records", _CREATE_HOST_RECORDS_TABLE_SQL),
            ("index", "host_interaction_records_checkpoint_idx", _CREATE_HOST_RECORDS_INDEX_SQL),
            ("index", "host_interaction_records_recovery_idx", _CREATE_HOST_RECORDS_RECOVERY_INDEX_SQL),
            ("table", "host_interaction_notification_outbox", _CREATE_NOTIFICATION_TABLE_SQL),
            ("index", "host_interaction_notification_outbox_checkpoint_idx", _CREATE_NOTIFICATION_CHECKPOINT_INDEX_SQL),
            ("index", "host_interaction_notification_outbox_lease_idx", _CREATE_NOTIFICATION_LEASE_INDEX_SQL),
            ("table", "deferred_resolution_receipts", _CREATE_RECEIPTS_TABLE_SQL),
            ("index", "deferred_receipts_checkpoint_idx", _CREATE_RECEIPTS_INDEX_SQL),
            ("table", "controller_command_receipts", _CREATE_CONTROLLER_RECEIPTS_TABLE_SQL),
            ("index", "controller_command_receipts_checkpoint_idx", _CREATE_CONTROLLER_CHECKPOINT_INDEX_SQL),
            ("index", "controller_command_receipts_outbox_idx", _CREATE_CONTROLLER_OUTBOX_INDEX_SQL),
        )
        existing = {name: self._schema_object(name) for _object_type, name, _sql in definitions}
        if not any(schema is not None for schema in existing.values()):
            for _object_type, _name, sql in definitions:
                self._conn.execute(sql)
            self._conn.commit()
            return

        for expected_type, name, expected_sql in definitions:
            schema = existing[name]
            if schema is None:
                raise RuntimeError(
                    f"checkpoint_store_schema_mismatch: existing {name} schema is incomplete; create a new database"
                )
            actual_type, actual_sql = schema
            if (
                actual_type != expected_type
                or actual_sql is None
                or _normalize_schema_sql(actual_sql) != _normalize_schema_sql(expected_sql)
            ):
                raise RuntimeError(
                    f"checkpoint_store_schema_mismatch: existing {name} does not match the current schema; create a new database"
                )

    def _schema_object(self, name: str) -> tuple[str, str | None] | None:
        rows = self._conn.execute(
            "SELECT type, sql FROM sqlite_master WHERE lower(name) = lower(?)",
            (name,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError(f"checkpoint_store_schema_mismatch: multiple schema objects match {name}; create a new database")
        row = rows[0]
        return str(row[0]), str(row[1]) if row[1] is not None else None

    def _schema_sql(self, object_type: str, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
        validate_checkpoint_creation(snapshot)
        row = _checkpoint_row(snapshot)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO checkpoints ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})",
                row,
            )
            return cursor.rowcount == 1

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint_key,),
            ).fetchone()
        return _checkpoint_from_row(row) if row is not None else None

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                checkpoint = _checkpoint_from_row(row)
                try:
                    check_claim(checkpoint, cycle_index, now_ms, claim_mode)
                except ValueError as exc:
                    raise CheckpointConflictError(str(exc)) from exc
                if checkpoint.claim_token is not None and claim_mode != "recovery":
                    raise CheckpointConflictError("expired checkpoint claims require recovery mode")
                if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED and claim_mode != "recovery":
                    raise CheckpointConflictError("reconciliation checkpoints require recovery mode")
                if (
                    self._conn.execute(
                        "SELECT 1 FROM host_interaction_records "
                        "WHERE checkpoint_key = ? AND state IN ('resolved_pending', 'resolved_claimed') LIMIT 1",
                        (checkpoint_key,),
                    ).fetchone()
                    is not None
                ):
                    raise CheckpointError(
                        "host interaction response requires dedicated recovery",
                        code="host_interaction_recovery_required",
                    )
                cursor = self._conn.execute(
                    """
                    UPDATE checkpoints
                    SET revision = revision + 1,
                        resume_attempt = resume_attempt + ?, status = ?,
                        claim_token = ?, claimed_cycle = ?, lease_expires_at_ms = ?
                    WHERE checkpoint_key = ? AND revision = ?
                      AND (claim_token IS NULL OR lease_expires_at_ms <= ?)
                      AND terminal_result IS NULL
                    """,
                    (
                        int(claim_mode == "recovery"),
                        AgentStatus.RUNNING.value,
                        claim_token,
                        cycle_index,
                        lease_expires_at_ms,
                        checkpoint_key,
                        checkpoint.revision,
                        now_ms,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CheckpointConflictError(f"checkpoint cycle {cycle_index} for key {checkpoint_key} is already claimed")
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        checkpoint.revision += 1
        if claim_mode == "recovery":
            checkpoint.resume_attempt += 1
        checkpoint.status = AgentStatus.RUNNING
        checkpoint.claim_token = claim_token
        checkpoint.claimed_cycle = cycle_index
        checkpoint.lease_expires_at_ms = lease_expires_at_ms
        return checkpoint

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
        if snapshot.status is not AgentStatus.RUNNING or snapshot.terminal_result is not None:
            return False
        snapshot.revision = expected_revision + 1
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?",
                (snapshot.checkpoint_key, expected_revision, claim_token),
            ).fetchone()
            if current_row is None:
                return False
            current = _checkpoint_from_row(current_row)
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.event_cursor = deepcopy(current.event_cursor)
            snapshot.cancel_requested = snapshot.cancel_requested or current.cancel_requested
            row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    snapshot.claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        claimed_cycle = checkpoint.claimed_cycle
        if claimed_cycle is None:
            raise ValueError("checkpoint v11 suspend requires an active claim")
        snapshot = replace(
            checkpoint,
            revision=expected_revision + 1,
            claim_token=None,
            claimed_cycle=None,
            lease_expires_at_ms=None,
        )
        snapshot = checkpoint_from_json(checkpoint_to_json(snapshot))
        if snapshot.status is not AgentStatus.RECONCILIATION_REQUIRED or snapshot.cycle_index != claimed_cycle - 1:
            return False
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?",
                (snapshot.checkpoint_key, expected_revision, claim_token),
            ).fetchone()
            if current_row is None:
                return False
            current = _checkpoint_from_row(current_row)
            snapshot.cancel_requested = current.cancel_requested
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.event_cursor = deepcopy(current.event_cursor)
            row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        claimed_cycle = checkpoint.claimed_cycle
        if claimed_cycle is None:
            raise ValueError("checkpoint v11 commit requires an active claim")
        if (
            checkpoint.cycle_index != claimed_cycle
            or checkpoint.status is not AgentStatus.RUNNING
            or checkpoint.terminal_result is not None
            or checkpoint.cancel_requested
            or any(
                entry.state in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
                for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
            )
        ):
            return False
        validate_model_journal_accounting(checkpoint)
        snapshot = replace(
            checkpoint,
            revision=expected_revision + 1,
            claim_token=None,
            claimed_cycle=None,
            lease_expires_at_ms=None,
            event_outbox=checkpoint.event_outbox,
            model_call_journal=[],
            tool_journal=[],
        )
        snapshot = checkpoint_from_json(checkpoint_to_json(snapshot))
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?",
                (snapshot.checkpoint_key, expected_revision, claim_token),
            ).fetchone()
            if current_row is None:
                return False
            current = _checkpoint_from_row(current_row)
            if current.cancel_requested:
                return False
            snapshot.event_outbox = [
                entry for entry in merge_event_outbox(current.event_outbox, snapshot.event_outbox) if entry.state == "pending"
            ]
            snapshot.event_cursor = deepcopy(current.event_cursor)
            snapshot.cancel_requested = current.cancel_requested
            row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def finalize_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        snapshot = prepare_unclaimed_terminal(checkpoint)
        snapshot = checkpoint_from_json(checkpoint_to_json(snapshot))
        if snapshot.terminal_result is None or snapshot.claim_token is not None:
            raise ValueError("finalized checkpoint v11 must be terminal and unclaimed")
        snapshot.revision = expected_revision + 1
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _FINALIZE_COLUMNS
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ? AND revision = ? AND claim_token IS NULL",
                (snapshot.checkpoint_key, expected_revision),
            ).fetchone()
            if current_row is None:
                return False
            current = _checkpoint_from_row(current_row)
            snapshot.cancel_requested = current.cancel_requested
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.event_cursor = deepcopy(current.event_cursor)
            row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + " WHERE checkpoint_key = ? AND revision = ?"
                + " AND claim_token IS NULL AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint.checkpoint_key,),
            ).fetchone()
            if current_row is None:
                return False
            terminal = prepare_claimed_terminal(
                _checkpoint_from_row(current_row),
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            )
            if terminal is None:
                return False
            row = dict(zip(_COLUMNS, _checkpoint_row(terminal), strict=True))
            columns = _FINALIZE_COLUMNS
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    terminal.checkpoint_key,
                    expected_revision,
                    claim_token,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

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
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint_key,),
            ).fetchone()
            if current_row is None:
                return False
            delivered = prepare_event_delivery(
                _checkpoint_from_row(current_row),
                event_id=event_id,
                payload_digest=payload_digest,
                cursor=cursor,
                expected_revision=expected_revision,
                claim_token=claim_token,
            )
            if delivered is None:
                return False
            row = dict(zip(_COLUMNS, _checkpoint_row(delivered), strict=True))
            claim_clause = "claim_token IS NULL" if claim_token is None else "claim_token = ?"
            parameters: tuple[object, ...] = (
                row["revision"],
                row["event_cursor"],
                row["event_outbox"],
                checkpoint_key,
                expected_revision,
            )
            if claim_token is not None:
                parameters = (*parameters, claim_token)
            result = self._conn.execute(
                "UPDATE checkpoints SET revision = ?, event_cursor = ?, event_outbox = ?"
                + f" WHERE checkpoint_key = ? AND revision = ? AND {claim_clause}",
                parameters,
            )
            return result.rowcount == 1

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal:
        _validate_renew(claim_token, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current_now_ms = clock.now_ms()
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                ).fetchone()
                current = _checkpoint_from_row(row) if row is not None else None
                if current is None:
                    self._conn.rollback()
                    return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=0)
                if (
                    current.claim_token != claim_token
                    or (current.lease_expires_at_ms or 0) <= current_now_ms
                    or lease_expires_at_ms <= current_now_ms
                ):
                    self._conn.rollback()
                    return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=current.revision)
                cursor = self._conn.execute(
                    """
                    UPDATE checkpoints
                    SET lease_expires_at_ms = ?
                    WHERE checkpoint_key = ? AND claim_token = ?
                      AND lease_expires_at_ms > ?
                    """,
                    (
                        lease_expires_at_ms,
                        checkpoint_key,
                        claim_token,
                        current_now_ms,
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=current.revision)
                self._conn.commit()
                return CheckpointRenewal(
                    outcome=(RenewOutcome.CANCEL_REQUESTED if current.cancel_requested else RenewOutcome.RENEWED),
                    lease_expires_at_ms=lease_expires_at_ms,
                )
            except BaseException:
                self._conn.rollback()
                raise

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint.checkpoint_key,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                current = _checkpoint_from_row(row)
                authoritative = prepare_tool_receipt(
                    current,
                    checkpoint,
                    operation_id=operation_id,
                    attempt=attempt,
                    tool_call_id=tool_call_id,
                    request_digest=request_digest,
                    result=result,
                    claim_token=claim_token,
                    expected_revision=expected_revision,
                    claimed_cycle=claimed_cycle,
                    created_at=run_events.event_created_at(),
                )
                if authoritative is None or authoritative is current:
                    self._conn.rollback()
                    return authoritative is not None
                self._write_checkpoint_tx(authoritative, expected_revision=expected_revision, claim_token=claim_token)
                self._conn.commit()
                return True
            except BaseException:
                self._conn.rollback()
                raise

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE checkpoints
                SET revision = revision + 1, terminal_acknowledged = 1
                WHERE checkpoint_key = ? AND revision = ?
                  AND terminal_result IS NOT NULL AND claim_token IS NULL
                  AND terminal_acknowledged = 0
                """,
                (checkpoint_key, expected_revision),
            )
            return cursor.rowcount == 1

    def delete_checkpoint(self, checkpoint_key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_key = ?",
                (checkpoint_key,),
            )
            self._conn.execute(
                "DELETE FROM deferred_resolution_receipts WHERE checkpoint_key = ?",
                (checkpoint_key,),
            )
            if self._schema_object("vv_agent_distributed_dispatch_outbox") is not None:
                self._conn.execute(
                    "DELETE FROM vv_agent_distributed_dispatch_outbox WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                )

    def _ensure_dispatch_outbox_schema(self) -> None:
        self._conn.execute(_CREATE_DISPATCH_OUTBOX_TABLE_SQL)
        self._conn.execute(_CREATE_DISPATCH_OUTBOX_INDEX_SQL)

    def claim_distributed_dispatch(
        self,
        envelope: Mapping[str, Any],
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> DispatchOutboxClaim:
        candidate = DispatchOutboxRecord.pending(envelope)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_dispatch_outbox_schema()
                if (
                    self._conn.execute(
                        "SELECT 1 FROM checkpoints WHERE checkpoint_key = ?",
                        (candidate.checkpoint_key,),
                    ).fetchone()
                    is None
                ):
                    raise CheckpointError("dispatch checkpoint was not found", code="checkpoint_not_found")
                row = self._conn.execute(
                    _SELECT_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                    (candidate.dispatch_id,),
                ).fetchone()
                current = candidate if row is None else _dispatch_from_row(row)
                if current.envelope_digest != candidate.envelope_digest:
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
                if row is None:
                    self._conn.execute(
                        _INSERT_DISPATCH_OUTBOX,
                        _dispatch_values(claim.record),
                    )
                elif claim.record != current:
                    self._conn.execute(
                        _UPDATE_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                        (*_dispatch_values(claim.record), candidate.dispatch_id),
                    )
                self._conn.commit()
                return claim
            except BaseException:
                self._conn.rollback()
                raise

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_dispatch_outbox_schema()
                row = self._conn.execute(
                    _SELECT_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                    (dispatch_id,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                current = _dispatch_from_row(row)
                if current.envelope_digest != envelope_digest:
                    raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
                if outcome not in {"delivered", "ambiguous"}:
                    raise ValueError("dispatch completion outcome must be delivered or ambiguous")
                typed_outcome = cast(Literal["delivered", "ambiguous"], outcome)
                if current.state == typed_outcome:
                    self._conn.commit()
                    return current
                updated = complete_dispatch(
                    current,
                    claim_token=claim_token,
                    attempt=attempt,
                    outcome=typed_outcome,
                    now_ms=now_ms,
                    error=error,
                )
                self._conn.execute(
                    _UPDATE_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                    (*_dispatch_values(updated), dispatch_id),
                )
                self._conn.commit()
                return updated
            except BaseException:
                self._conn.rollback()
                raise

    def reconcile_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_dispatch_outbox_schema()
                row = self._conn.execute(
                    _SELECT_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                    (dispatch_id,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                current = _dispatch_from_row(row)
                if current.envelope_digest != envelope_digest:
                    raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
                if outcome not in {"retry", "delivered"}:
                    raise ValueError("dispatch reconciliation outcome must be retry or delivered")
                typed_outcome = cast(Literal["retry", "delivered"], outcome)
                if (typed_outcome == "retry" and current.state == "pending") or (
                    typed_outcome == "delivered" and current.state == "delivered"
                ):
                    self._conn.commit()
                    return current
                updated = reconcile_dispatch(
                    current,
                    outcome=typed_outcome,
                    now_ms=now_ms,
                    error=error,
                )
                self._conn.execute(
                    _UPDATE_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                    (*_dispatch_values(updated), dispatch_id),
                )
                self._conn.commit()
                return updated
            except BaseException:
                self._conn.rollback()
                raise

    def get_distributed_dispatch(self, dispatch_id: str) -> DispatchOutboxRecord | None:
        with self._lock:
            if self._schema_object("vv_agent_distributed_dispatch_outbox") is None:
                return None
            row = self._conn.execute(
                _SELECT_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return _dispatch_from_row(row) if row is not None else None

    def reap_distributed_dispatches(
        self,
        *,
        checkpoint_key: str | None = None,
        now_ms: int,
    ) -> list[DispatchOutboxRecord]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_dispatch_outbox_schema()
                if checkpoint_key is None:
                    rows = self._conn.execute(
                        _SELECT_DISPATCH_OUTBOX
                        + " WHERE state = 'claimed' AND lease_expires_at_ms <= ? ORDER BY dispatch_id ASC",
                        (now_ms,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        _SELECT_DISPATCH_OUTBOX + " WHERE checkpoint_key = ? AND state = 'claimed' AND lease_expires_at_ms <= ? "
                        "ORDER BY dispatch_id ASC",
                        (checkpoint_key, now_ms),
                    ).fetchall()
                updated_rows: list[DispatchOutboxRecord] = []
                for row in rows:
                    current = _dispatch_from_row(row)
                    updated = reap_dispatch(current, now_ms=now_ms)
                    if updated is None:
                        continue
                    self._conn.execute(
                        _UPDATE_DISPATCH_OUTBOX + " WHERE dispatch_id = ?",
                        (*_dispatch_values(updated), current.dispatch_id),
                    )
                    updated_rows.append(updated)
                self._conn.commit()
                return updated_rows
            except BaseException:
                self._conn.rollback()
                raise

    def _update_checkpoint_row(self, snapshot: Checkpoint, *, expected_revision: int) -> bool:
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        assignments = ", ".join(f"{column} = ?" for column in _COLUMNS if column != "checkpoint_key")
        values = tuple(row[column] for column in _COLUMNS if column != "checkpoint_key")
        result = self._conn.execute(
            f"UPDATE checkpoints SET {assignments} WHERE checkpoint_key = ? AND revision = ?",
            (*values, snapshot.checkpoint_key, expected_revision),
        )
        return result.rowcount == 1

    @staticmethod
    def _host_record_values(record: dict[str, Any]) -> tuple[object, ...]:
        validate_host_interaction_record(
            {**record, "schema_version": HOST_RECORD_SCHEMA}, checkpoint_key=str(record["checkpoint_key"])
        )
        return (
            record["record_id"],
            record["checkpoint_key"],
            record["interaction_id"],
            record["logical_cycle"],
            _json_dump(record["request"]),
            record["request_digest"],
            record["state"],
            record["attempt"],
            record["claim_token"],
            record["lease_expires_at_ms"],
            _json_dump(record["response"]) if record["response"] is not None else None,
            record["response_digest"],
            record["command_id"],
            record["resolved_revision"],
            record["consumed_revision"],
            record.get("last_error"),
        )

    @staticmethod
    def _host_record_from_row(row: tuple[object, ...]) -> dict[str, Any]:
        names = (
            "record_id",
            "checkpoint_key",
            "interaction_id",
            "logical_cycle",
            "request",
            "request_digest",
            "state",
            "attempt",
            "claim_token",
            "lease_expires_at_ms",
            "response",
            "response_digest",
            "command_id",
            "resolved_revision",
            "consumed_revision",
            "last_error",
        )
        values = dict(zip(names, row, strict=True))
        values["request"] = _json_load(values["request"], "host interaction request")
        values["response"] = (
            _json_load(values["response"], "host interaction response") if values["response"] is not None else None
        )
        values["schema_version"] = HOST_RECORD_SCHEMA
        try:
            return validate_host_interaction_record(values, checkpoint_key=str(values["checkpoint_key"]))
        except (TypeError, ValueError) as exc:
            raise CheckpointError("host interaction record is invalid", code="host_interaction_conflict") from exc

    def _notification_from_row(self, row: tuple[object, ...]) -> dict[str, Any]:
        names = (
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
        )
        values = dict(zip(names, row, strict=True))
        values["payload"] = _json_load(values["payload"], "host interaction notification")
        try:
            validate_host_interaction_notification(
                values["payload"],
                notification_id=str(values["notification_id"]),
                record_id=str(values["record_id"]),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError("host interaction notification is invalid", code="notification_conflict") from exc
        if values["payload_digest"] != canonical_json_sha256(values["payload"], "notification_payload"):
            raise CheckpointError("notification payload digest conflicts", code="notification_conflict")
        if values["outbox_state"] not in {"pending", "claimed", "delivered", "ambiguous", "aborted"}:
            raise CheckpointError("notification state is invalid", code="notification_conflict")
        if (values["claim_token"] is None) != (values["lease_expires_at_ms"] is None):
            raise CheckpointError("notification claim and lease are inconsistent", code="notification_conflict")
        if values["outbox_state"] == "claimed" and values["claim_token"] is None:
            raise CheckpointError("claimed notification has no owner", code="notification_conflict")
        if values["outbox_state"] != "claimed" and values["claim_token"] is not None:
            raise CheckpointError("unclaimed notification has an owner", code="notification_conflict")
        attempt = values["attempt"]
        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 0 <= attempt <= (1 << 53) - 1:
            raise CheckpointError("notification attempt is invalid", code="notification_conflict")
        for field_name in ("lease_expires_at_ms", "delivered_at_ms", "aborted_at_ms"):
            value = values[field_name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (1 << 53) - 1
            ):
                raise CheckpointError(f"notification {field_name} is invalid", code="notification_conflict")
        claim_token = values["claim_token"]
        if claim_token is not None and (
            not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token.encode("utf-8")) > 512
        ):
            raise CheckpointError("notification claim token is invalid", code="notification_conflict")
        abort_reason = values["abort_reason"]
        if abort_reason is not None and (
            not isinstance(abort_reason, str) or not abort_reason.strip() or len(abort_reason.encode("utf-8")) > 65536
        ):
            raise CheckpointError("notification abort reason is invalid", code="notification_conflict")
        last_error = values["last_error"]
        if last_error is not None and (not isinstance(last_error, str) or len(last_error.encode("utf-8")) > 65536):
            raise CheckpointError("notification last error is invalid", code="notification_conflict")
        if values["outbox_state"] == "delivered" and (
            values["delivered_at_ms"] is None or values["aborted_at_ms"] is not None or abort_reason is not None
        ):
            raise CheckpointError("delivered notification fields are invalid", code="notification_conflict")
        if values["outbox_state"] == "aborted" and (
            values["aborted_at_ms"] is None or values["delivered_at_ms"] is not None or abort_reason is None
        ):
            raise CheckpointError("aborted notification fields are invalid", code="notification_conflict")
        if values["outbox_state"] not in {"delivered", "aborted"} and (
            values["delivered_at_ms"] is not None or values["aborted_at_ms"] is not None or abort_reason is not None
        ):
            raise CheckpointError("pending notification fields are invalid", code="notification_conflict")
        return values

    def _notification_row(self, notification_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT notification_id, checkpoint_key, record_id, payload, payload_digest, outbox_state, "
            "claim_token, lease_expires_at_ms, attempt, delivered_at_ms, aborted_at_ms, abort_reason, last_error "
            "FROM host_interaction_notification_outbox WHERE notification_id = ?",
            (notification_id,),
        ).fetchone()
        return self._notification_from_row(row) if row is not None else None

    def get_host_interaction_notification(self, notification_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._notification_row(notification_id)

    def _find_resolved_pending_host_interaction(self, *, checkpoint_key: str) -> dict[str, Any] | None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, checkpoint_key, interaction_id, logical_cycle, request, request_digest, state, "
                "attempt, claim_token, lease_expires_at_ms, response, response_digest, command_id, "
                "resolved_revision, consumed_revision, last_error "
                "FROM host_interaction_records WHERE checkpoint_key = ? AND state = 'resolved_pending' LIMIT 2",
                (checkpoint_key,),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise CheckpointError(
                    "checkpoint has multiple pending host interaction responses",
                    code="host_interaction_conflict",
                )
            return self._host_record_from_row(rows[0])

    @staticmethod
    def _notification_values(row: dict[str, Any]) -> tuple[object, ...]:
        validate_host_interaction_notification(
            row["payload"],
            notification_id=str(row["notification_id"]),
            record_id=str(row["record_id"]),
        )
        if row["payload_digest"] != canonical_json_sha256(row["payload"], "notification_payload"):
            raise ValueError("notification payload digest conflicts")
        return (
            row["notification_id"],
            row["checkpoint_key"],
            row["record_id"],
            _json_dump(row["payload"]),
            row["payload_digest"],
            row["outbox_state"],
            row["claim_token"],
            row["lease_expires_at_ms"],
            row["attempt"],
            row.get("delivered_at_ms"),
            row.get("aborted_at_ms"),
            row.get("abort_reason"),
            row.get("last_error"),
        )

    def _host_outcome(self, record: dict[str, Any], *, status: str, checkpoint_revision: int) -> HostInteractionOutcome:
        notification_id = derive_host_interaction_notification_id(record["record_id"])
        notification = self._notification_row(notification_id)
        if notification is None:
            raise CheckpointError("host interaction notification row is missing", code="host_interaction_conflict")
        return HostInteractionOutcome(
            interaction_id=record["interaction_id"],
            logical_cycle=int(record["logical_cycle"]),
            checkpoint_revision=checkpoint_revision,
            status=status,
            outbox_state="pending",
            record_id=str(record["record_id"]),
            notification_id=notification_id,
            notification_payload_digest=str(notification["payload_digest"]),
            notification_outbox_action="host_interaction_notification",
            notification_outbox_destination="host_interaction_observer",
        )

    def produce_host_interaction(
        self,
        request: HostInteractionRequest | Mapping[str, Any],
        *,
        admission_context: HostInteractionAdmissionContext,
    ) -> HostInteractionOutcome:
        request_value = request if isinstance(request, HostInteractionRequest) else HostInteractionRequest.from_dict(request)
        admission_context.validate()
        if request_value.logical_cycle != admission_context.claimed_cycle:
            raise CheckpointError("host interaction logical cycle does not match its claim", code="host_interaction_stale")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                checkpoint_key = admission_context.checkpoint_key
                if checkpoint_key is not None:
                    existing_row = self._conn.execute(
                        "SELECT record_id, checkpoint_key, interaction_id, logical_cycle, request, request_digest, state, "
                        "attempt, claim_token, lease_expires_at_ms, response, response_digest, command_id, "
                        "resolved_revision, consumed_revision, last_error FROM host_interaction_records "
                        "WHERE checkpoint_key = ? AND interaction_id = ?",
                        (checkpoint_key, request_value.interaction_id),
                    ).fetchone()
                    if existing_row is not None:
                        existing = self._host_record_from_row(existing_row)
                        if (
                            existing["request_digest"] != request_value.request_digest
                            or existing["request"] != request_value.to_dict()
                        ):
                            raise CheckpointError(
                                "host interaction identity or digest conflicts", code="host_interaction_conflict"
                            )
                        checkpoint = self.load_checkpoint(checkpoint_key)
                        if checkpoint is None:
                            raise CheckpointError("host interaction checkpoint was deleted", code="host_interaction_conflict")
                        if checkpoint.revision < admission_context.expected_revision + 1:
                            raise CheckpointError("host interaction replay revision is stale", code="host_interaction_stale")
                        validate_host_tool_receipt_replay(checkpoint, request_value, admission_context)
                        self._conn.commit()
                        return self._host_outcome(existing, status="replayed", checkpoint_revision=checkpoint.revision)
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                ).fetchone()
                current = _checkpoint_from_row(row) if row is not None else None
                snapshot, record, notification, outcome = prepare_host_interaction(
                    current,
                    request_value,
                    admission_context=admission_context,
                    created_at=time.time(),
                )
                if not self._update_checkpoint_row(snapshot, expected_revision=admission_context.expected_revision):
                    raise CheckpointError("host interaction producer CAS lost", code="host_interaction_claim_required")
                self._conn.execute(
                    "INSERT INTO host_interaction_records (record_id, checkpoint_key, interaction_id, logical_cycle, request, "
                    "request_digest, state, attempt, claim_token, lease_expires_at_ms, response, response_digest, command_id, "
                    "resolved_revision, consumed_revision, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._host_record_values(record),
                )
                self._conn.execute(
                    "INSERT INTO host_interaction_notification_outbox (notification_id, checkpoint_key, record_id, "
                    "payload, payload_digest, "
                    "outbox_state, claim_token, lease_expires_at_ms, attempt, delivered_at_ms, aborted_at_ms, abort_reason, "
                    "last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._notification_values(notification),
                )
                self._conn.commit()
                return outcome
            except BaseException:
                self._conn.rollback()
                raise

    @staticmethod
    def _controller_receipt_values(receipt: ControllerCommandReceipt, command: ControllerCommand) -> tuple[object, ...]:
        wire = receipt.to_dict()
        return (
            receipt.command_id,
            command.handle.checkpoint_key,
            _json_dump(command.handle.to_dict()),
            receipt.command_digest,
            _json_dump(command.command),
            receipt.resume_attempt,
            receipt.expected_revision,
            _json_dump(wire),
            receipt.resulting_status,
            receipt.resulting_revision,
            receipt.outbox_state,
            derive_controller_receipt_outbox_id(receipt.command_id, receipt.command_digest),
            receipt.outbox_action,
            receipt.outbox_destination,
            receipt.outbox_attempt,
            None,
            None,
            None,
            None,
        )

    def _controller_receipt_from_row(self, row: tuple[object, ...]) -> ControllerCommandReceipt:
        # receipt is the strict public source; the other columns are indexed
        # facts and are checked by SQLite constraints before this decode.
        raw = _json_load(row[7], "controller command receipt")
        receipt = ControllerCommandReceipt.from_dict(raw)
        if (
            str(row[0]) != receipt.command_id
            or str(row[3]) != receipt.command_digest
            or _sqlite_int(row[5], "resume_attempt") != receipt.resume_attempt
            or _sqlite_int(row[6], "expected_revision") != receipt.expected_revision
            or str(row[8]) != receipt.resulting_status
            or _sqlite_int(row[9], "resulting_revision") != receipt.resulting_revision
            or str(row[10]) != receipt.outbox_state
            or str(row[11]) != derive_controller_receipt_outbox_id(receipt.command_id, receipt.command_digest)
            or str(row[12]) != receipt.outbox_action
            or row[13] != receipt.outbox_destination
            or _sqlite_int(row[14], "outbox_attempt") != receipt.outbox_attempt
        ):
            raise CheckpointError("controller command receipt scalar fields conflict", code="controller_command_conflict")
        return receipt

    def admit_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandReceipt:
        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing_row = self._conn.execute(
                    "SELECT command_id, checkpoint_key, handle, command_digest, command, resume_attempt, expected_revision, "
                    "receipt, resulting_status, resulting_revision, outbox_state, outbox_id, outbox_action, "
                    "outbox_destination, attempt, claim_token, lease_expires_at_ms, delivered_at_ms, last_error "
                    "FROM controller_command_receipts WHERE command_id = ?",
                    (command_value.command_id,),
                ).fetchone()
                if existing_row is not None:
                    if existing_row[3] != command_value.command_digest:
                        raise CheckpointError(
                            "controller command id was reused with a different digest", code="controller_command_conflict"
                        )
                    receipt = self._controller_receipt_from_row(existing_row)
                    self._controller_wake_from_row(existing_row, receipt)
                    self._conn.commit()
                    return receipt
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (command_value.handle.checkpoint_key,),
                ).fetchone()
                if row is None:
                    raise CheckpointError("controller command checkpoint was not found", code="controller_command_stale")
                current = _checkpoint_from_row(row)
                active = current.active_host_interaction
                if command_value.kind in {"resume", "host_interaction_response"} and isinstance(current.suspended_origin, dict):
                    active = current.suspended_origin.get("active_host_interaction")
                record: dict[str, Any] | None = None
                if isinstance(active, dict):
                    record_row = self._conn.execute(
                        "SELECT record_id, checkpoint_key, interaction_id, logical_cycle, request, request_digest, state, "
                        "attempt, claim_token, lease_expires_at_ms, response, response_digest, command_id, "
                        "resolved_revision, consumed_revision, last_error FROM host_interaction_records "
                        "WHERE checkpoint_key = ? AND interaction_id = ?",
                        (current.checkpoint_key, active.get("interaction_id")),
                    ).fetchone()
                    if record_row is not None:
                        record = self._host_record_from_row(record_row)
                snapshot, staged, receipt = prepare_controller_command(
                    current,
                    record,
                    command_value,
                    now_ms=time.time_ns() // 1_000_000,
                    event_id=run_events.new_event_id(),
                    created_at=run_events.event_created_at(),
                )
                if not self._update_checkpoint_row(snapshot, expected_revision=current.revision):
                    raise CheckpointError("controller command checkpoint CAS lost", code="controller_command_stale")
                if staged is not None:
                    self._conn.execute(
                        "UPDATE host_interaction_records SET state = ?, attempt = ?, claim_token = ?, "
                        "lease_expires_at_ms = ?, response = ?, response_digest = ?, command_id = ?, "
                        "resolved_revision = ?, consumed_revision = ?, last_error = ? "
                        "WHERE record_id = ? AND checkpoint_key = ? AND interaction_id = ?",
                        (
                            staged["state"],
                            staged["attempt"],
                            staged["claim_token"],
                            staged["lease_expires_at_ms"],
                            _json_dump(staged["response"]) if staged["response"] is not None else None,
                            staged["response_digest"],
                            staged["command_id"],
                            staged["resolved_revision"],
                            staged["consumed_revision"],
                            staged.get("last_error"),
                            staged["record_id"],
                            staged["checkpoint_key"],
                            staged["interaction_id"],
                        ),
                    )
                    if self._conn.execute("SELECT changes()").fetchone()[0] != 1:
                        raise CheckpointError("controller host interaction record CAS lost", code="controller_command_stale")
                self._conn.execute(
                    "INSERT INTO controller_command_receipts (command_id, checkpoint_key, handle, command_digest, command, "
                    "resume_attempt, expected_revision, receipt, resulting_status, resulting_revision, outbox_state, outbox_id, "
                    "outbox_action, outbox_destination, attempt, claim_token, lease_expires_at_ms, delivered_at_ms, last_error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._controller_receipt_values(receipt, command_value),
                )
                self._conn.commit()
                return receipt
            except BaseException:
                self._conn.rollback()
                raise

    def resolve_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandResolution:
        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        with self._lock:
            was_present = (
                self._conn.execute(
                    "SELECT 1 FROM controller_command_receipts WHERE command_id = ?",
                    (command_value.command_id,),
                ).fetchone()
                is not None
            )
            try:
                receipt = self.admit_controller_command(command_value)
            except CheckpointError as exc:
                return ControllerCommandResolution(kind="rejected", error=getattr(exc, "code", None) or str(exc))
        checkpoint = self.load_checkpoint(command_value.handle.checkpoint_key)
        if checkpoint is None:
            return ControllerCommandResolution(kind="rejected", error="controller_command_stale")
        return ControllerCommandResolution(
            kind="replayed" if was_present else "applied",
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
        with self._lock:
            row = self._conn.execute(
                "SELECT command_id, checkpoint_key, handle, command_digest, command, resume_attempt, expected_revision, "
                "receipt, resulting_status, resulting_revision, outbox_state, outbox_id, outbox_action, "
                "outbox_destination, attempt, claim_token, lease_expires_at_ms, delivered_at_ms, last_error "
                "FROM controller_command_receipts WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                return None
            receipt = self._controller_receipt_from_row(row)
            self._controller_wake_from_row(row, receipt)
            return receipt

    @staticmethod
    def _controller_wake_from_row(row: tuple[object, ...], receipt: ControllerCommandReceipt) -> dict[str, Any]:
        wake = {
            "command_id": receipt.command_id,
            "command_digest": receipt.command_digest,
            "outbox_id": str(row[11]),
            "outbox_action": receipt.outbox_action,
            "outbox_destination": receipt.outbox_destination,
            "outbox_state": receipt.outbox_state,
            "attempt": _sqlite_int(row[14], "outbox_attempt"),
            "claim_token": row[15],
            "lease_expires_at_ms": row[16],
            "delivered_at_ms": row[17],
            "last_error": row[18],
        }
        if wake["outbox_id"] != derive_controller_receipt_outbox_id(wake["command_id"], wake["command_digest"]):
            raise CheckpointError("controller wake outbox identity conflicts", code="controller_command_conflict")
        if wake["outbox_action"] == "none" and (
            wake["outbox_destination"] is not None or wake["outbox_state"] != "delivered" or wake["attempt"] != 0
        ):
            raise CheckpointError("controller wake none action is invalid", code="controller_command_conflict")
        claim_token = wake["claim_token"]
        lease = wake["lease_expires_at_ms"]
        if (claim_token is None) != (lease is None):
            raise CheckpointError("controller wake claim and lease are inconsistent", code="controller_command_conflict")
        if wake["outbox_state"] == "claimed" and claim_token is None:
            raise CheckpointError("claimed controller wake has no owner", code="controller_command_conflict")
        if wake["outbox_state"] != "claimed" and claim_token is not None:
            raise CheckpointError("unclaimed controller wake has an owner", code="controller_command_conflict")
        if claim_token is not None and (
            not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token.encode("utf-8")) > 512
        ):
            raise CheckpointError("controller wake claim token is invalid", code="controller_command_conflict")
        for field_name in ("lease_expires_at_ms", "delivered_at_ms"):
            value = wake[field_name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (1 << 53) - 1
            ):
                raise CheckpointError(f"controller wake {field_name} is invalid", code="controller_command_conflict")
        if wake["last_error"] is not None and (
            not isinstance(wake["last_error"], str) or len(wake["last_error"].encode("utf-8")) > 65536
        ):
            raise CheckpointError("controller wake last_error is invalid", code="controller_command_conflict")
        return wake

    def _controller_receipt_row(self, command_id: str) -> tuple[object, ...] | None:
        return self._conn.execute(
            "SELECT command_id, checkpoint_key, handle, command_digest, command, resume_attempt, expected_revision, "
            "receipt, resulting_status, resulting_revision, outbox_state, outbox_id, outbox_action, "
            "outbox_destination, attempt, claim_token, lease_expires_at_ms, delivered_at_ms, last_error "
            "FROM controller_command_receipts WHERE command_id = ?",
            (command_id,),
        ).fetchone()

    def _update_controller_wake_row(
        self,
        row: tuple[object, ...],
        receipt: ControllerCommandReceipt,
        staged: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = replace(
            receipt,
            outbox_state=str(staged["outbox_state"]),
            outbox_attempt=int(staged["attempt"]),
        )
        result = self._conn.execute(
            "UPDATE controller_command_receipts SET receipt = ?, outbox_state = ?, attempt = ?, claim_token = ?, "
            "lease_expires_at_ms = ?, delivered_at_ms = ?, last_error = ? WHERE command_id = ? AND outbox_state = ? "
            "AND attempt = ?",
            (
                _json_dump(updated.to_dict()),
                updated.outbox_state,
                updated.outbox_attempt,
                staged["claim_token"],
                staged["lease_expires_at_ms"],
                staged["delivered_at_ms"],
                staged.get("last_error"),
                updated.command_id,
                row[10],
                row[14],
            ),
        )
        if result.rowcount != 1:
            raise CheckpointError("controller wake owner or attempt is stale", code="controller_command_stale")
        return {
            "command_id": updated.command_id,
            "command_digest": updated.command_digest,
            "outbox_id": str(row[11]),
            "outbox_action": updated.outbox_action,
            "outbox_destination": updated.outbox_destination,
            "outbox_state": updated.outbox_state,
            "attempt": updated.outbox_attempt,
            "claim_token": staged["claim_token"],
            "lease_expires_at_ms": staged["lease_expires_at_ms"],
            "delivered_at_ms": staged["delivered_at_ms"],
            "last_error": staged.get("last_error"),
        }

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._controller_receipt_row(command_id)
                if row is None:
                    self._conn.commit()
                    return None
                receipt = self._controller_receipt_from_row(row)
                if receipt.command_digest != command_digest:
                    raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                current = self._controller_wake_from_row(row, receipt)
                if current["outbox_action"] == "none" or current["outbox_state"] == "delivered":
                    self._conn.commit()
                    return current
                if current["outbox_state"] == "ambiguous":
                    raise CheckpointError("controller wake requires reconciliation", code="controller_command_stale")
                if current["outbox_state"] == "claimed":
                    if current["claim_token"] == claim_token:
                        self._conn.commit()
                        return current
                    if int(current["lease_expires_at_ms"] or 0) > now_ms:
                        raise CheckpointError("controller wake is claimed by another owner", code="controller_command_stale")
                staged = dict(current)
                staged["outbox_state"] = "claimed"
                staged["claim_token"] = claim_token
                staged["lease_expires_at_ms"] = lease_expires_at_ms
                staged["attempt"] = int(current["attempt"]) + 1
                result = self._update_controller_wake_row(row, receipt, staged)
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._controller_receipt_row(command_id)
                if row is None:
                    self._conn.commit()
                    return None
                receipt = self._controller_receipt_from_row(row)
                if receipt.command_digest != command_digest:
                    raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                current = self._controller_wake_from_row(row, receipt)
                if current["outbox_state"] in {"delivered", "ambiguous"}:
                    if current["outbox_state"] == outcome:
                        self._conn.commit()
                        return current
                    raise CheckpointError("controller wake has already completed", code="controller_command_stale")
                if current["outbox_state"] != "claimed" or current["claim_token"] != claim_token or current["attempt"] != attempt:
                    raise CheckpointError("controller wake owner or attempt is stale", code="controller_command_stale")
                staged = dict(current)
                staged.update(
                    outbox_state=outcome,
                    claim_token=None,
                    lease_expires_at_ms=None,
                    delivered_at_ms=now_ms if outcome == "delivered" else None,
                    last_error=error,
                )
                result = self._update_controller_wake_row(row, receipt, staged)
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

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
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._controller_receipt_row(command_id)
                if row is None:
                    self._conn.commit()
                    return None
                receipt = self._controller_receipt_from_row(row)
                if receipt.command_digest != command_digest:
                    raise CheckpointError("controller command digest conflicts", code="controller_command_conflict")
                current = self._controller_wake_from_row(row, receipt)
                target = "delivered" if outcome == "delivered" else "pending"
                if current["outbox_state"] == target:
                    self._conn.commit()
                    return current
                if current["outbox_state"] != "ambiguous":
                    raise CheckpointError("controller wake is not ambiguous", code="controller_command_stale")
                staged = dict(current)
                staged["outbox_state"] = target
                staged["delivered_at_ms"] = now_ms if target == "delivered" else None
                staged["last_error"] = None
                result = self._update_controller_wake_row(row, receipt, staged)
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

    def _reap_controller_command_wake(self, *, command_id: str, now_ms: int) -> dict[str, Any] | None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._controller_receipt_row(command_id)
                if row is None:
                    self._conn.commit()
                    return None
                receipt = self._controller_receipt_from_row(row)
                current = self._controller_wake_from_row(row, receipt)
                if current["outbox_state"] != "claimed" or int(current["lease_expires_at_ms"] or 0) > now_ms:
                    self._conn.commit()
                    return current
                staged = dict(current)
                staged.update(outbox_state="pending", claim_token=None, lease_expires_at_ms=None)
                result = self._update_controller_wake_row(row, receipt, staged)
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

    def reap_controller_command_wakes(self, checkpoint_key: str, now_ms: int) -> list[dict[str, Any]]:
        with self._lock:
            command_ids = tuple(
                str(item[0])
                for item in self._conn.execute(
                    "SELECT command_id FROM controller_command_receipts "
                    "WHERE checkpoint_key = ? AND outbox_action = 'recovery_dispatch' "
                    "AND (outbox_state = 'pending' OR "
                    "(outbox_state = 'claimed' AND lease_expires_at_ms <= ?)) "
                    "ORDER BY expected_revision ASC, command_id ASC",
                    (checkpoint_key, now_ms),
                ).fetchall()
            )
        rows: list[dict[str, Any]] = []
        for command_id in command_ids:
            row = self._reap_controller_command_wake(command_id=command_id, now_ms=now_ms)
            if row is not None and row["outbox_action"] == "recovery_dispatch" and row["outbox_state"] == "pending":
                rows.append(row)
        return rows

    def get_controller_command(self, command_id: str) -> ControllerCommand | None:
        with self._lock:
            row = self._controller_receipt_row(command_id)
            if row is None:
                return None
            receipt = self._controller_receipt_from_row(row)
            self._controller_wake_from_row(row, receipt)
            return ControllerCommand(
                command_id=str(row[0]),
                handle=_json_load(row[2], "controller command handle"),
                command_digest=str(row[3]),
                command=_json_load(row[4], "controller command"),
                resume_attempt=_sqlite_int(row[5], "resume_attempt"),
                expected_revision=_sqlite_int(row[6], "expected_revision"),
            )

    def claim_and_consume_host_interaction_response(self, envelope: Mapping[str, Any]) -> HostInteractionRecoveryResult:
        try:
            envelope_value = (
                envelope
                if isinstance(envelope, HostInteractionRecoveryEnvelope)
                else HostInteractionRecoveryEnvelope.from_dict(envelope)
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(str(exc), code="host_interaction_recovery_stale") from exc
        record_id = envelope_value.record_id
        checkpoint_key = envelope_value.checkpoint_key
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                checkpoint_row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                ).fetchone()
                record_row = self._conn.execute(
                    "SELECT record_id, checkpoint_key, interaction_id, logical_cycle, request, request_digest, state, attempt, "
                    "claim_token, lease_expires_at_ms, response, response_digest, command_id, resolved_revision, "
                    "consumed_revision, last_error FROM host_interaction_records "
                    "WHERE record_id = ? AND checkpoint_key = ? AND interaction_id = ?",
                    (record_id, checkpoint_key, envelope_value.interaction_id),
                ).fetchone()
                if checkpoint_row is None or record_row is None:
                    raise CheckpointError(
                        "host interaction recovery record was not found", code="host_interaction_recovery_stale"
                    )
                checkpoint = _checkpoint_from_row(checkpoint_row)
                record = self._host_record_from_row(record_row)
                if record["checkpoint_key"] != checkpoint_key:
                    raise CheckpointError("host interaction recovery binding is stale", code="host_interaction_recovery_stale")
                snapshot, staged_record, result = prepare_host_response_consumption(
                    checkpoint,
                    record,
                    envelope_value,
                    now_ms=time.time_ns() // 1_000_000,
                    created_at=run_events.event_created_at(),
                )
                if result.kind != "applied":
                    self._conn.commit()
                    return result
                if not self._update_checkpoint_row(snapshot, expected_revision=checkpoint.revision):
                    raise CheckpointError("host interaction recovery CAS lost", code="host_interaction_recovery_stale")
                self._conn.execute(
                    "UPDATE host_interaction_records SET state = ?, attempt = ?, claim_token = ?, "
                    "lease_expires_at_ms = ?, response = ?, response_digest = ?, command_id = ?, "
                    "resolved_revision = ?, consumed_revision = ?, last_error = ? "
                    "WHERE record_id = ? AND checkpoint_key = ? AND interaction_id = ?",
                    (
                        staged_record["state"],
                        staged_record["attempt"],
                        staged_record["claim_token"],
                        staged_record["lease_expires_at_ms"],
                        _json_dump(staged_record["response"]),
                        staged_record["response_digest"],
                        staged_record["command_id"],
                        staged_record["resolved_revision"],
                        staged_record["consumed_revision"],
                        staged_record.get("last_error"),
                        record_id,
                        checkpoint_key,
                        record["interaction_id"],
                    ),
                )
                if self._conn.execute("SELECT changes()").fetchone()[0] != 1:
                    raise CheckpointError("host interaction recovery record CAS lost", code="host_interaction_recovery_stale")
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

    def reap_host_interaction_record(self, *, record_id: str, checkpoint_key: str, now_ms: int) -> bool:
        """Return an expired response claim to the durable pending state."""
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise CheckpointError("now_ms is invalid", code="host_interaction_claim_required")
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT claim_token, lease_expires_at_ms, state, checkpoint_key, interaction_id "
                "FROM host_interaction_records WHERE record_id = ? AND checkpoint_key = ?",
                (record_id, checkpoint_key),
            ).fetchone()
            if row is None or row[3] != checkpoint_key:
                return False
            if row[2] != "resolved_claimed" or row[0] is None:
                return False
            checkpoint = self.load_checkpoint(checkpoint_key)
            if checkpoint is None or checkpoint.claim_token is None:
                return False
            if checkpoint.status is not AgentStatus.RUNNING or checkpoint.claim_token != row[0]:
                return False
            if checkpoint.lease_expires_at_ms is None or checkpoint.lease_expires_at_ms > now_ms:
                return False
            # Keep the checkpoint execution fence in the same SQLite CAS as
            # the record repair.  The assignment is deliberately a no-op:
            # reaping repairs an in-transaction record phase and must not
            # manufacture a new checkpoint revision, but a concurrent claim
            # replacement must invalidate the whole transaction.
            checkpoint_cas = self._conn.execute(
                "UPDATE checkpoints SET lease_expires_at_ms = lease_expires_at_ms "
                "WHERE checkpoint_key = ? AND claim_token = ? AND lease_expires_at_ms <= ?",
                (checkpoint_key, row[0], now_ms),
            )
            if checkpoint_cas.rowcount != 1:
                return False
            updated = self._conn.execute(
                "UPDATE host_interaction_records SET state = 'resolved_pending', claim_token = NULL, "
                "lease_expires_at_ms = NULL, last_error = ? WHERE record_id = ? AND checkpoint_key = ? AND interaction_id = ? "
                "AND state = 'resolved_claimed' AND claim_token = ?",
                ("host_interaction_response_claim_expired", record_id, checkpoint_key, row[4], row[0]),
            )
            return updated.rowcount == 1

    def claim_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token.encode("utf-8")) > 512:
            raise ValueError("notification claim_token must be non-empty and bounded")
        if isinstance(lease_expires_at_ms, bool) or not isinstance(lease_expires_at_ms, int) or lease_expires_at_ms <= now_ms:
            raise ValueError("notification lease must be greater than now_ms")
        with self._lock, self._conn:
            row = self._notification_row(notification_id)
            if row is None:
                return None
            if row["payload_digest"] != payload_digest:
                raise CheckpointError("notification payload digest conflicts", code="notification_conflict")
            if row["outbox_state"] in {"delivered", "aborted"}:
                return row
            if (
                row["outbox_state"] == "claimed"
                and row["claim_token"] != claim_token
                and (row["lease_expires_at_ms"] or 0) > now_ms
            ):
                raise CheckpointError("notification is claimed by another owner", code="notification_stale")
            result = self._conn.execute(
                "UPDATE host_interaction_notification_outbox SET outbox_state = 'claimed', claim_token = ?, "
                "lease_expires_at_ms = ?, attempt = attempt + 1 WHERE notification_id = ? AND payload_digest = ? "
                "AND (outbox_state = 'pending' OR (outbox_state = 'claimed' AND lease_expires_at_ms <= ?))",
                (claim_token, lease_expires_at_ms, notification_id, payload_digest, now_ms),
            )
            if result.rowcount != 1:
                raise CheckpointError("notification owner or attempt is stale", code="notification_stale")
            return self._notification_row(notification_id)

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
        if outcome not in {"delivered", "ambiguous"}:
            raise ValueError("notification completion outcome must be delivered or ambiguous")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 0 <= attempt <= (1 << 53) - 1:
            raise ValueError("notification attempt is invalid")
        with self._lock, self._conn:
            row = self._notification_row(notification_id)
            if row is None:
                return None
            if row["payload_digest"] != payload_digest:
                raise CheckpointError("notification payload digest conflicts", code="notification_conflict")
            updated = self._conn.execute(
                "UPDATE host_interaction_notification_outbox SET outbox_state = ?, claim_token = NULL, "
                "lease_expires_at_ms = NULL, delivered_at_ms = ?, last_error = ? WHERE notification_id = ? "
                "AND payload_digest = ? AND outbox_state = 'claimed' AND claim_token = ? AND attempt = ?",
                (
                    outcome,
                    now_ms if outcome == "delivered" else None,
                    error,
                    notification_id,
                    payload_digest,
                    claim_token,
                    attempt,
                ),
            )
            if updated.rowcount != 1:
                raise CheckpointError("notification owner or attempt is stale", code="notification_stale")
            return self._notification_row(notification_id)

    def reconcile_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        outcome: str,
        now_ms: int,
        abort_reason: str | None = None,
    ) -> dict[str, Any] | None:
        if outcome not in {"delivered", "retry", "abort"}:
            raise ValueError("notification reconciliation outcome must be delivered, retry, or abort")
        if outcome == "abort" and (
            not isinstance(abort_reason, str) or not abort_reason.strip() or len(abort_reason.encode("utf-8")) > 65536
        ):
            raise ValueError("notification abort_reason is required")
        with self._lock, self._conn:
            row = self._notification_row(notification_id)
            if row is None:
                return None
            if row["payload_digest"] != payload_digest:
                raise CheckpointError("notification payload digest conflicts", code="notification_conflict")
            target = {"delivered": "delivered", "retry": "pending", "abort": "aborted"}[outcome]
            if row["outbox_state"] == target:
                return row
            if row["outbox_state"] != "ambiguous":
                raise CheckpointError("notification is not ambiguous", code="notification_stale")
            self._conn.execute(
                "UPDATE host_interaction_notification_outbox SET outbox_state = ?, delivered_at_ms = ?, "
                "aborted_at_ms = ?, abort_reason = ?, claim_token = NULL, lease_expires_at_ms = NULL "
                "WHERE notification_id = ? AND payload_digest = ? AND outbox_state = 'ambiguous'",
                (
                    target,
                    now_ms if target == "delivered" else None,
                    now_ms if target == "aborted" else None,
                    abort_reason if target == "aborted" else None,
                    notification_id,
                    payload_digest,
                ),
            )
            return self._notification_row(notification_id)

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
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?", (checkpoint.checkpoint_key,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                current = _checkpoint_from_row(row)
                if (
                    current.revision != expected_revision
                    or checkpoint.revision != expected_revision
                    or current.claim_token != claim_token
                    or current.claimed_cycle != claimed_cycle
                    or current.status is not AgentStatus.RUNNING
                    or current.terminal_result is not None
                ):
                    self._conn.rollback()
                    return False
                # A same-value write inside the immediate transaction proves
                # that the checkpoint/outbox row is writable before a tool
                # provider can perform an external effect. It is rolled back,
                # so no revision or event is added by the preflight.
                self._conn.execute(
                    "UPDATE checkpoints SET event_outbox = event_outbox WHERE checkpoint_key = ?",
                    (checkpoint.checkpoint_key,),
                )
                self._conn.rollback()
                return True
            except sqlite3.DatabaseError:
                self._conn.rollback()
                return False

    def admit_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        outcomes: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?", (checkpoint.checkpoint_key,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                current = _checkpoint_from_row(row)
                updated = prepare_deferred_admission(
                    current,
                    checkpoint,
                    outcomes=outcomes,
                    claim_token=claim_token,
                    expected_revision=expected_revision,
                    claimed_cycle=claimed_cycle,
                    created_at=run_events.event_created_at(),
                )
                if updated is None:
                    self._conn.rollback()
                    return False
                self._write_checkpoint_tx(updated, expected_revision=expected_revision, claim_token=claim_token)
                self._conn.commit()
                return True
            except BaseException:
                self._conn.rollback()
                raise

    def resolve_deferred(self, handle: DeferredToolHandle, result: Any) -> DeferredResolveDecision:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                receipt_row = self._conn.execute(
                    "SELECT handle_key, checkpoint_key, handle, result, result_digest, "
                    "event_id, event_payload_digest, receipt_status "
                    "FROM deferred_resolution_receipts WHERE handle_key = ?",
                    (handle.key,),
                ).fetchone()
                receipt = _receipt_from_row(receipt_row) if receipt_row is not None else None
                row = self._conn.execute(_SELECT_CHECKPOINT + " WHERE checkpoint_key = ?", (handle.checkpoint_key,)).fetchone()
                current = _checkpoint_from_row(row) if row is not None else None
                updated, decision = prepare_deferred_resolution(
                    current, receipt, handle, result, created_at=run_events.event_created_at()
                )
                if updated is None:
                    self._conn.rollback()
                    return decision
                assert current is not None
                current_revision = current.revision
                self._write_checkpoint_tx(updated, expected_revision=current_revision, claim_token=None)
                receipt = decision.receipt
                assert receipt is not None
                self._conn.execute(
                    "INSERT OR REPLACE INTO deferred_resolution_receipts ("
                    "handle_key, checkpoint_key, handle, result, result_digest, "
                    "event_id, event_payload_digest, receipt_status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        receipt.handle.key,
                        receipt.handle.checkpoint_key,
                        _json_dump(receipt.handle.to_dict()),
                        _json_dump(receipt.result.to_dict()),
                        receipt.result_digest,
                        receipt.event_id,
                        receipt.event_payload_digest,
                        receipt.receipt_status,
                    ),
                )
                self._conn.commit()
                return decision
            except BaseException:
                self._conn.rollback()
                raise

    def accept_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        decisions: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?", (checkpoint.checkpoint_key,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                current = _checkpoint_from_row(row)
                updated = prepare_deferred_acceptance(
                    current,
                    checkpoint,
                    decisions=decisions,
                    claim_token=claim_token,
                    expected_revision=expected_revision,
                    claimed_cycle=claimed_cycle,
                    created_at=run_events.event_created_at(),
                )
                if updated is None:
                    self._conn.rollback()
                    return False
                if updated is current:
                    # Exact repeated accept_deferred decisions are already
                    # durable replays; no checkpoint write or revision bump.
                    self._conn.rollback()
                    return True
                self._write_checkpoint_tx(updated, expected_revision=expected_revision, claim_token=claim_token)
                self._conn.commit()
                return True
            except BaseException:
                self._conn.rollback()
                raise

    def _write_checkpoint_tx(self, checkpoint: Checkpoint, *, expected_revision: int, claim_token: str | None) -> None:
        row = dict(zip(_COLUMNS, _checkpoint_row(checkpoint), strict=True))
        where = "checkpoint_key = ? AND revision = ?"
        params: tuple[Any, ...] = (checkpoint.checkpoint_key, expected_revision)
        if claim_token is not None:
            where += " AND claim_token = ?"
            params += (claim_token,)
        else:
            where += " AND claim_token IS NULL"
        cursor = self._conn.execute(
            "UPDATE checkpoints SET "
            + ", ".join(f"{column} = ?" for column in _COLUMNS if column != "checkpoint_key")
            + " WHERE "
            + where,
            tuple(row[column] for column in _COLUMNS if column != "checkpoint_key") + params,
        )
        if cursor.rowcount != 1:
            raise CheckpointConflictError("checkpoint revision conflict")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_key TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL CHECK (schema_version = 'vv-agent.checkpoint.v11'),
    run_definition_schema TEXT NOT NULL CHECK (run_definition_schema = 'vv-agent.run-definition.v5'),
    run_definition TEXT NOT NULL,
    task_id TEXT NOT NULL,
    root_run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    run_definition_digest TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    cycle_index INTEGER NOT NULL CHECK (cycle_index >= 0),
    status TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL CHECK (cancel_requested IN (0, 1)),
    active_host_interaction TEXT,
    suspended_origin TEXT,
    messages TEXT NOT NULL,
    cycles TEXT NOT NULL,
    model_calls TEXT NOT NULL,
    shared_state TEXT NOT NULL,
    budget_usage TEXT,
    event_cursor TEXT,
    event_outbox TEXT NOT NULL,
    extension_state TEXT NOT NULL,
    model_call_journal TEXT NOT NULL,
    tool_journal TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    claim_token TEXT,
    claimed_cycle INTEGER,
    lease_expires_at_ms INTEGER,
    terminal_result TEXT,
    terminal_acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (terminal_acknowledged IN (0, 1)),
    CHECK (status <> 'deferred' OR (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)),
    CHECK (status <> 'deferred' OR tool_journal <> '[]'),
    CHECK (
        (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
        OR
        (claim_token IS NOT NULL AND claimed_cycle IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (claim_token IS NULL OR claimed_cycle = cycle_index + 1),
    CHECK (terminal_result IS NULL OR claim_token IS NULL),
    CHECK (
        (status = 'host_interaction' AND active_host_interaction IS NOT NULL AND suspended_origin IS NULL)
        OR
        (status = 'suspended' AND active_host_interaction IS NULL AND suspended_origin IS NOT NULL)
        OR
        (status NOT IN ('host_interaction', 'suspended') AND active_host_interaction IS NULL AND suspended_origin IS NULL)
    ),
    CHECK (
        status NOT IN ('host_interaction', 'suspended')
        OR (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
    )
)
"""
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS checkpoints_status_idx ON checkpoints(status)
"""
_CREATE_HOST_RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS host_interaction_records (
    record_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    interaction_id TEXT NOT NULL,
    logical_cycle INTEGER NOT NULL CHECK (logical_cycle >= 1),
    request TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'resolved_pending', 'resolved_claimed', 'consumed')),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    response TEXT,
    response_digest TEXT,
    command_id TEXT,
    resolved_revision INTEGER,
    consumed_revision INTEGER,
    last_error TEXT,
    UNIQUE (checkpoint_key, interaction_id),
    CHECK (
        (claim_token IS NULL AND lease_expires_at_ms IS NULL)
        OR (claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (
        (state = 'active' AND response IS NULL AND response_digest IS NULL AND command_id IS NULL)
        OR
        (state IN ('resolved_pending', 'resolved_claimed', 'consumed')
         AND response IS NOT NULL AND response_digest IS NOT NULL AND command_id IS NOT NULL)
    ),
    CHECK (state <> 'resolved_claimed' OR (claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)),
    CHECK (state <> 'resolved_pending' OR claim_token IS NULL),
    CHECK (state <> 'consumed' OR consumed_revision IS NOT NULL),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
)
"""
_CREATE_HOST_RECORDS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS host_interaction_records_checkpoint_idx
    ON host_interaction_records(checkpoint_key, state)
"""
_CREATE_HOST_RECORDS_RECOVERY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS host_interaction_records_recovery_idx
    ON host_interaction_records(state, lease_expires_at_ms)
"""
_CREATE_NOTIFICATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS host_interaction_notification_outbox (
    notification_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    outbox_state TEXT NOT NULL CHECK (outbox_state IN ('pending', 'claimed', 'delivered', 'ambiguous', 'aborted')),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    delivered_at_ms INTEGER,
    aborted_at_ms INTEGER,
    abort_reason TEXT,
    last_error TEXT,
    UNIQUE (checkpoint_key, record_id),
    CHECK (
        (outbox_state = 'claimed' AND claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
        OR (outbox_state <> 'claimed' AND claim_token IS NULL AND lease_expires_at_ms IS NULL)
    ),
    CHECK (
        (outbox_state = 'delivered' AND delivered_at_ms IS NOT NULL AND aborted_at_ms IS NULL AND abort_reason IS NULL)
        OR
        (outbox_state = 'aborted' AND aborted_at_ms IS NOT NULL AND delivered_at_ms IS NULL AND abort_reason IS NOT NULL)
        OR
        (outbox_state NOT IN ('delivered', 'aborted') AND delivered_at_ms IS NULL AND aborted_at_ms IS NULL
         AND abort_reason IS NULL)
    ),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE,
    FOREIGN KEY (record_id) REFERENCES host_interaction_records(record_id) ON DELETE CASCADE
)
"""
_CREATE_NOTIFICATION_CHECKPOINT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS host_interaction_notification_outbox_checkpoint_idx
    ON host_interaction_notification_outbox(checkpoint_key, outbox_state)
"""
_CREATE_NOTIFICATION_LEASE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS host_interaction_notification_outbox_lease_idx
    ON host_interaction_notification_outbox(outbox_state, lease_expires_at_ms)
"""
_CREATE_RECEIPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS deferred_resolution_receipts (
    handle_key TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    handle TEXT NOT NULL,
    result TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_payload_digest TEXT NOT NULL,
    receipt_status TEXT NOT NULL CHECK (receipt_status IN ('succeeded', 'failed')),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
)
"""
_CREATE_RECEIPTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS deferred_receipts_checkpoint_idx ON deferred_resolution_receipts(checkpoint_key)
"""
_CREATE_CONTROLLER_RECEIPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS controller_command_receipts (
    command_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    handle TEXT NOT NULL,
    command_digest TEXT NOT NULL,
    command TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    expected_revision INTEGER NOT NULL CHECK (expected_revision >= 0),
    receipt TEXT NOT NULL,
    resulting_status TEXT NOT NULL,
    resulting_revision INTEGER NOT NULL CHECK (resulting_revision >= 0),
    outbox_state TEXT NOT NULL CHECK (outbox_state IN ('pending', 'claimed', 'delivered', 'ambiguous')),
    outbox_id TEXT NOT NULL,
    outbox_action TEXT NOT NULL CHECK (outbox_action IN ('none', 'recovery_dispatch')),
    outbox_destination TEXT,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    delivered_at_ms INTEGER,
    last_error TEXT,
    CHECK (
        (outbox_action = 'none' AND outbox_destination IS NULL)
        OR
        (outbox_action = 'recovery_dispatch' AND outbox_destination = 'distributed_advance')
    ),
    CHECK (outbox_state = 'delivered' OR outbox_action = 'recovery_dispatch'),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
)
"""
_CREATE_CONTROLLER_CHECKPOINT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS controller_command_receipts_checkpoint_idx
    ON controller_command_receipts(checkpoint_key)
"""
_CREATE_CONTROLLER_OUTBOX_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS controller_command_receipts_outbox_idx
    ON controller_command_receipts(outbox_state, lease_expires_at_ms)
"""
_CREATE_DISPATCH_OUTBOX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vv_agent_distributed_dispatch_outbox (
    schema_version TEXT NOT NULL CHECK (schema_version = 'vv-agent.distributed-dispatch.v1'),
    dispatch_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    cycle_index INTEGER NOT NULL CHECK (cycle_index >= 1),
    envelope TEXT NOT NULL,
    envelope_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'claimed', 'delivered', 'ambiguous')),
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    delivered_at_ms INTEGER,
    last_error TEXT,
    CHECK (
        (state = 'claimed' AND claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
        OR (state <> 'claimed' AND claim_token IS NULL AND lease_expires_at_ms IS NULL)
    ),
    CHECK (
        (state = 'delivered' AND delivered_at_ms IS NOT NULL)
        OR (state <> 'delivered' AND delivered_at_ms IS NULL)
    ),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
)
"""
_CREATE_DISPATCH_OUTBOX_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS vv_agent_distributed_dispatch_checkpoint_idx
    ON vv_agent_distributed_dispatch_outbox(checkpoint_key, state)
"""
_DISPATCH_OUTBOX_COLUMNS = (
    "schema_version",
    "dispatch_id",
    "checkpoint_key",
    "cycle_index",
    "envelope",
    "envelope_digest",
    "state",
    "attempt",
    "claim_token",
    "lease_expires_at_ms",
    "delivered_at_ms",
    "last_error",
)
_SELECT_DISPATCH_OUTBOX = "SELECT " + ", ".join(_DISPATCH_OUTBOX_COLUMNS) + " FROM vv_agent_distributed_dispatch_outbox"
_INSERT_DISPATCH_OUTBOX = (
    "INSERT INTO vv_agent_distributed_dispatch_outbox ("
    + ", ".join(_DISPATCH_OUTBOX_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _DISPATCH_OUTBOX_COLUMNS)
    + ")"
)
_UPDATE_DISPATCH_OUTBOX = "UPDATE vv_agent_distributed_dispatch_outbox SET " + ", ".join(
    f"{column} = ?" for column in _DISPATCH_OUTBOX_COLUMNS
)


def _normalize_schema_sql(sql: str) -> str:
    return " ".join(sql.replace("IF NOT EXISTS", "").split())


def _dispatch_values(record: DispatchOutboxRecord) -> tuple[object, ...]:
    payload = record.to_dict()
    return (
        payload["schema_version"],
        payload["dispatch_id"],
        payload["checkpoint_key"],
        payload["cycle_index"],
        _json_dump(payload["envelope"]),
        payload["envelope_digest"],
        payload["state"],
        payload["attempt"],
        payload["claim_token"],
        payload["lease_expires_at_ms"],
        payload["delivered_at_ms"],
        payload["last_error"],
    )


def _dispatch_from_row(row: tuple[object, ...]) -> DispatchOutboxRecord:
    values = dict(zip(_DISPATCH_OUTBOX_COLUMNS, row, strict=True))
    values["envelope"] = _json_load(values["envelope"], "distributed dispatch envelope")
    return DispatchOutboxRecord.from_dict(values)


_COLUMNS = (
    "checkpoint_key",
    "schema_version",
    "run_definition_schema",
    "run_definition",
    "task_id",
    "root_run_id",
    "trace_id",
    "run_definition_digest",
    "resume_attempt",
    "cycle_index",
    "status",
    "cancel_requested",
    "active_host_interaction",
    "suspended_origin",
    "messages",
    "cycles",
    "model_calls",
    "shared_state",
    "budget_usage",
    "event_cursor",
    "event_outbox",
    "extension_state",
    "model_call_journal",
    "tool_journal",
    "revision",
    "claim_token",
    "claimed_cycle",
    "lease_expires_at_ms",
    "terminal_result",
    "terminal_acknowledged",
)
_PROGRESS_COLUMNS = tuple(
    column
    for column in _COLUMNS
    if column
    not in {
        "checkpoint_key",
        "schema_version",
        "run_definition_schema",
        "run_definition",
        "task_id",
        "root_run_id",
        "trace_id",
        "run_definition_digest",
        "resume_attempt",
        "claim_token",
        "claimed_cycle",
        "lease_expires_at_ms",
        "terminal_result",
        "terminal_acknowledged",
    }
)
_FINALIZE_COLUMNS = (*_PROGRESS_COLUMNS, "terminal_result")
_IDENTITY_WHERE = (
    " AND schema_version = ? AND run_definition_schema = ? AND run_definition = ?"
    " AND task_id = ? AND root_run_id = ? AND trace_id = ?"
    " AND run_definition_digest = ? AND resume_attempt = ?"
    " AND cancel_requested = ? AND terminal_acknowledged = ?"
)
_SELECT_CHECKPOINT = f"SELECT {', '.join(_COLUMNS)} FROM checkpoints"


def _checkpoint_row(checkpoint: Checkpoint) -> tuple[object, ...]:
    full = checkpoint_to_dict(checkpoint)
    return (
        full["checkpoint_key"],
        full["schema_version"],
        full["run_definition_schema"],
        _json_dump(full["run_definition"]),
        full["task_id"],
        full["root_run_id"],
        full["trace_id"],
        full["run_definition_digest"],
        full["resume_attempt"],
        full["cycle_index"],
        full["status"],
        int(full["cancel_requested"]),
        _json_dump(full["active_host_interaction"]) if full["active_host_interaction"] is not None else None,
        _json_dump(full["suspended_origin"]) if full["suspended_origin"] is not None else None,
        _json_dump(full["messages"]),
        _json_dump(full["cycles"]),
        _json_dump(full["model_calls"]),
        _json_dump(full["shared_state"]),
        _json_dump(full["budget_usage"]) if full["budget_usage"] is not None else None,
        _json_dump(full["event_cursor"]) if full["event_cursor"] is not None else None,
        _json_dump(full["event_outbox"]),
        _json_dump(full["extension_state"]),
        _json_dump(full["model_call_journal"]),
        _json_dump(full["tool_journal"]),
        full["revision"],
        full["claim_token"],
        full["claimed_cycle"],
        full["lease_expires_at_ms"],
        _json_dump(full["terminal_result"]) if full["terminal_result"] is not None else None,
        int(full["terminal_acknowledged"]),
    )


def _checkpoint_from_row(row: tuple[object, ...]) -> Checkpoint:
    values = dict(zip(_COLUMNS, row, strict=True))
    payload = {
        "checkpoint_key": values["checkpoint_key"],
        "schema_version": values["schema_version"],
        "run_definition_schema": values["run_definition_schema"],
        "run_definition": (
            _json_load(values["run_definition"], "run_definition") if values["run_definition"] is not None else None
        ),
        "task_id": values["task_id"],
        "root_run_id": values["root_run_id"],
        "trace_id": values["trace_id"],
        "run_definition_digest": values["run_definition_digest"],
        "resume_attempt": values["resume_attempt"],
        "cycle_index": values["cycle_index"],
        "status": values["status"],
        "cancel_requested": bool(values["cancel_requested"]),
        "active_host_interaction": (
            _json_load(values["active_host_interaction"], "active_host_interaction")
            if values["active_host_interaction"] is not None
            else None
        ),
        "suspended_origin": (
            _json_load(values["suspended_origin"], "suspended_origin") if values["suspended_origin"] is not None else None
        ),
        "messages": _json_load(values["messages"], "messages"),
        "cycles": _json_load(values["cycles"], "cycles"),
        "model_calls": _json_load(values["model_calls"], "model_calls"),
        "shared_state": _json_load(values["shared_state"], "shared_state"),
        "budget_usage": (_json_load(values["budget_usage"], "budget_usage") if values["budget_usage"] is not None else None),
        "event_cursor": (_json_load(values["event_cursor"], "event_cursor") if values["event_cursor"] is not None else None),
        "event_outbox": _json_load(values["event_outbox"], "event_outbox"),
        "extension_state": _json_load(values["extension_state"], "extension_state"),
        "model_call_journal": _json_load(
            values["model_call_journal"],
            "model_call_journal",
        ),
        "tool_journal": _json_load(values["tool_journal"], "tool_journal"),
        "revision": values["revision"],
        "claim_token": values["claim_token"],
        "claimed_cycle": values["claimed_cycle"],
        "lease_expires_at_ms": values["lease_expires_at_ms"],
        "terminal_result": (
            _json_load(values["terminal_result"], "terminal_result") if values["terminal_result"] is not None else None
        ),
        "terminal_acknowledged": bool(values["terminal_acknowledged"]),
    }
    return checkpoint_from_dict(payload)


def _identity_values(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["schema_version"],
        row["run_definition_schema"],
        row["run_definition"],
        row["task_id"],
        row["root_run_id"],
        row["trace_id"],
        row["run_definition_digest"],
        row["resume_attempt"],
        row["cancel_requested"],
        row["terminal_acknowledged"],
    )


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_load(value: object, field_name: str) -> Any:
    try:
        return _strict_json_loads(str(value))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint v11 {field_name} JSON") from exc


def _sqlite_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointError(f"invalid SQLite {field_name}", code="controller_command_conflict")
    return value


def _receipt_from_row(row: tuple[object, ...]) -> DeferredResolutionReceipt:
    values = dict(
        zip(
            (
                "handle_key",
                "checkpoint_key",
                "handle",
                "result",
                "result_digest",
                "event_id",
                "event_payload_digest",
                "receipt_status",
            ),
            row,
            strict=True,
        )
    )
    handle = DeferredToolHandle.from_dict(_json_load(values["handle"], "deferred receipt handle"))
    if values["handle_key"] != handle.key or values["checkpoint_key"] != handle.checkpoint_key:
        raise ValueError("deferred_receipt_identity_invalid")
    return DeferredResolutionReceipt.from_dict(
        {
            "handle_key": values["handle_key"],
            "handle": handle.to_dict(),
            "result": _json_load(values["result"], "deferred receipt result"),
            "result_digest": values["result_digest"],
            "event_id": values["event_id"],
            "event_payload_digest": values["event_payload_digest"],
            "receipt_status": values["receipt_status"],
        }
    )
