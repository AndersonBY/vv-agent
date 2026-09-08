"""Shared v9 controller/host-interaction CAS logic for checkpoint stores.

The mixin is intentionally storage-neutral: a concrete store supplies its
authoritative checkpoint dictionary and lock.  SQLite/Redis adapters may use
the same transition rules while wrapping the mutation in their native
transaction primitive.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from vv_agent.checkpoint import CheckpointError, OperationState, ResumeObservation, canonical_json_sha256
from vv_agent.events import (
    CheckpointResumedEvent,
    HostInteractionRequestedEvent,
    HostInteractionResponseConsumedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunStateChangedEvent,
)
from vv_agent.runtime.checkpoint_codec import clone_checkpoint
from vv_agent.runtime.controller import (
    HOST_NOTIFICATION_SCHEMA,
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
    HostInteractionResponse,
    derive_controller_receipt_outbox_id,
    derive_host_interaction_notification_id,
    derive_host_interaction_record_id,
    derive_host_response_digest,
    validate_host_interaction_notification,
    validate_host_interaction_record,
)
from vv_agent.runtime.state import (
    EventOutboxEntry,
    prepare_claimed_terminal,
    prepare_unclaimed_terminal,
    validate_checkpoint,
)
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import AgentResult, AgentStatus, CompletionReason, Message


def _controller_error(message: str, code: str) -> CheckpointError:
    return CheckpointError(message, code=code)


_RECOVERY_LEASE_DURATION_MS = 60_000


class ControllerStoreMixin:
    """Implement v9 controller transitions over ``_store`` and ``_lock``.

    A concrete store should initialize the three index dictionaries with
    ``_init_controller_indexes``.  All public transitions are performed while
    the concrete store lock is held, so retries and duplicate deliveries do
    not create a second receipt, event, or notification.
    """

    _store: MutableMapping[str, Any]
    _lock: Any

    def _init_controller_indexes(self) -> None:
        self._host_interaction_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._controller_command_receipts: dict[str, ControllerCommandReceipt] = {}
        self._controller_commands: dict[str, ControllerCommand] = {}
        self._host_interaction_notifications: dict[str, dict[str, Any]] = {}
        self._controller_command_outboxes: dict[str, dict[str, Any]] = {}

    def _load_controller_checkpoint(self, checkpoint_key: str) -> Any:
        current = self._store.get(checkpoint_key)
        return current

    @staticmethod
    def _lease_now_ms(lease_now_ms: int | None) -> int:
        now_ms = time.time_ns() // 1_000_000 if lease_now_ms is None else lease_now_ms
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise _controller_error("lease_now_ms is invalid", "host_interaction_claim_required")
        return now_ms

    @staticmethod
    def _require_execution_claim(
        checkpoint: Any,
        *,
        claim_token: str | None,
        claimed_cycle: int | None,
        now_ms: int,
    ) -> None:
        """Reject stale execution owners before any controller side effect."""
        if (
            checkpoint.status is not AgentStatus.RUNNING
            or checkpoint.claim_token is None
            or checkpoint.claimed_cycle is None
            or checkpoint.lease_expires_at_ms is None
            or checkpoint.lease_expires_at_ms <= now_ms
            or claim_token is None
            or checkpoint.claim_token != claim_token
            or claimed_cycle is None
            or checkpoint.claimed_cycle != claimed_cycle
        ):
            raise _controller_error(
                "host interaction producer requires an active claim; checkpoint execution claim is stale or expired",
                "host_interaction_claim_required",
            )

    def _find_producer_checkpoint(
        self,
        request: HostInteractionRequest,
        *,
        admission_context: HostInteractionAdmissionContext,
    ) -> tuple[str, Any, str, int, int]:
        admission_context.validate()
        if request.logical_cycle != admission_context.claimed_cycle:
            raise _controller_error("host interaction logical cycle does not match its claim", "host_interaction_stale")
        key = admission_context.checkpoint_key
        current = self._load_controller_checkpoint(key)
        if current is None:
            raise _controller_error("host interaction checkpoint was not found", "host_interaction_claim_required")
        if current.revision != admission_context.expected_revision:
            raise _controller_error("host interaction producer revision is stale", "host_interaction_stale")
        if current.lease_expires_at_ms != admission_context.lease_expires_at_ms:
            raise _controller_error("host interaction producer lease is stale", "host_interaction_claim_required")
        if current.claimed_cycle != current.cycle_index + 1:
            raise _controller_error("host interaction producer cycle fence is invalid", "host_interaction_stale")
        self._require_execution_claim(
            current,
            claim_token=admission_context.claim_token,
            claimed_cycle=admission_context.claimed_cycle,
            now_ms=admission_context.now_ms,
        )
        assert current.claim_token is not None and current.claimed_cycle is not None
        return key, current, admission_context.claim_token, current.revision, current.claimed_cycle

    @staticmethod
    def _notification_payload(record_id: str, request: HostInteractionRequest, notification_id: str) -> dict[str, Any]:
        return {
            "schema_version": HOST_NOTIFICATION_SCHEMA,
            "notification_id": notification_id,
            "record_id": record_id,
            "interaction_id": request.interaction_id,
            "logical_cycle": request.logical_cycle,
            "status": "host_interaction",
            "wait_reason": "host_interaction",
            "prompt": request.prompt,
        }

    @staticmethod
    def _record(
        *,
        checkpoint_key: str,
        request: HostInteractionRequest,
        record_id: str,
        state: str = "active",
        response: dict[str, Any] | None = None,
        response_digest: str | None = None,
        command_id: str | None = None,
        resolved_revision: int | None = None,
        consumed_revision: int | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": HOST_RECORD_SCHEMA,
            "record_id": record_id,
            "checkpoint_key": checkpoint_key,
            "interaction_id": request.interaction_id,
            "logical_cycle": request.logical_cycle,
            "request": request.to_dict(),
            "request_digest": request.request_digest,
            "state": state,
            "attempt": 0,
            "claim_token": None,
            "lease_expires_at_ms": None,
            "response": response,
            "response_digest": response_digest,
            "command_id": command_id,
            "resolved_revision": resolved_revision,
            "consumed_revision": consumed_revision,
            "last_error": None,
        }

    @staticmethod
    def _checked_host_record(record: Mapping[str, Any], *, checkpoint_key: str | None = None) -> dict[str, Any]:
        try:
            return validate_host_interaction_record(record, checkpoint_key=checkpoint_key)
        except (TypeError, ValueError) as exc:
            raise _controller_error("host interaction record is invalid", "host_interaction_conflict") from exc

    @staticmethod
    def _checked_notification(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            notification_id = str(row["notification_id"])
            record_id = str(row["record_id"])
            payload = validate_host_interaction_notification(row["payload"], notification_id=notification_id, record_id=record_id)
            if row["payload_digest"] != canonical_json_sha256(payload, "notification_payload"):
                raise ValueError("notification payload digest conflicts")
            state = row["outbox_state"]
            if state not in {"pending", "claimed", "delivered", "ambiguous", "aborted"}:
                raise ValueError("notification state is invalid")
            claim_token = row.get("claim_token")
            lease = row.get("lease_expires_at_ms")
            if (claim_token is None) != (lease is None):
                raise ValueError("notification claim and lease are inconsistent")
            if claim_token is not None:
                if not isinstance(claim_token, str) or not claim_token.strip():
                    raise ValueError("notification claim token is invalid")
                if not isinstance(lease, int) or isinstance(lease, bool) or lease < 0:
                    raise ValueError("notification lease is invalid")
            if state == "claimed" and claim_token is None:
                raise ValueError("claimed notification has no owner")
            if state != "claimed" and claim_token is not None:
                raise ValueError("unclaimed notification has an owner")
            if not isinstance(row.get("attempt"), int) or isinstance(row.get("attempt"), bool) or row["attempt"] < 0:
                raise ValueError("notification attempt is invalid")
            if row["attempt"] > (1 << 53) - 1:
                raise ValueError("notification attempt is invalid")
            delivered_at_ms = row.get("delivered_at_ms")
            aborted_at_ms = row.get("aborted_at_ms")
            for field_name, value in (("delivered_at_ms", delivered_at_ms), ("aborted_at_ms", aborted_at_ms)):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (1 << 53) - 1
                ):
                    raise ValueError(f"{field_name} is invalid")
            abort_reason = row.get("abort_reason")
            last_error = row.get("last_error")
            if abort_reason is not None and (
                not isinstance(abort_reason, str) or not abort_reason.strip() or len(abort_reason.encode("utf-8")) > 65536
            ):
                raise ValueError("notification abort_reason is invalid")
            if last_error is not None and (not isinstance(last_error, str) or len(last_error.encode("utf-8")) > 65536):
                raise ValueError("notification last_error is invalid")
            if state == "delivered" and (delivered_at_ms is None or aborted_at_ms is not None or abort_reason is not None):
                raise ValueError("delivered notification timestamps are invalid")
            if state == "aborted" and (aborted_at_ms is None or delivered_at_ms is not None or abort_reason is None):
                raise ValueError("aborted notification fields are invalid")
            if state not in {"delivered", "aborted"} and (
                delivered_at_ms is not None or aborted_at_ms is not None or abort_reason is not None
            ):
                raise ValueError("pending notification timestamps are invalid")
            return dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise _controller_error("host interaction notification is invalid", "notification_conflict") from exc

    def get_host_interaction_notification(self, notification_id: str) -> dict[str, Any] | None:
        """Read the sanitized UI projection without exposing controller internals."""
        with self._lock:
            row = self._host_interaction_notifications.get(notification_id)
            if row is None:
                return None
            return deepcopy(self._checked_notification(row))

    def _find_resolved_pending_host_interaction(self, *, checkpoint_key: str) -> dict[str, Any] | None:
        with self._lock:
            for (key, _interaction_id), record in self._host_interaction_records.items():
                if key == checkpoint_key and record["state"] == "resolved_pending":
                    return deepcopy(self._checked_host_record(record, checkpoint_key=checkpoint_key))
        return None

    @staticmethod
    def _controller_wake_from_receipt(receipt: ControllerCommandReceipt) -> dict[str, Any]:
        return {
            "command_id": receipt.command_id,
            "command_digest": receipt.command_digest,
            "outbox_id": derive_controller_receipt_outbox_id(receipt.command_id, receipt.command_digest),
            "outbox_action": receipt.outbox_action,
            "outbox_destination": receipt.outbox_destination,
            "outbox_state": receipt.outbox_state,
            "attempt": receipt.outbox_attempt,
            "claim_token": None,
            "lease_expires_at_ms": None,
            "delivered_at_ms": (None if receipt.outbox_action == "none" or receipt.outbox_state != "delivered" else 0),
            "last_error": None,
        }

    @staticmethod
    def _checked_controller_wake(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            command_id = row["command_id"]
            command_digest = row["command_digest"]
            if not isinstance(command_id, str) or not command_id.strip():
                raise ValueError("controller wake command_id is invalid")
            if (
                not isinstance(command_digest, str)
                or len(command_digest) != 64
                or command_digest != command_digest.lower()
                or any(character not in "0123456789abcdef" for character in command_digest)
            ):
                raise ValueError("controller wake command_digest is invalid")
            if row["outbox_id"] != derive_controller_receipt_outbox_id(command_id, command_digest):
                raise ValueError("controller wake outbox_id is invalid")
            action = row["outbox_action"]
            destination = row["outbox_destination"]
            state = row["outbox_state"]
            if action not in {"none", "recovery_dispatch"}:
                raise ValueError("controller wake action is invalid")
            if action == "none" and (destination is not None or state != "delivered" or row.get("attempt") != 0):
                raise ValueError("controller wake none action is not delivered")
            if action == "recovery_dispatch" and destination != "distributed_advance":
                raise ValueError("controller wake destination is invalid")
            if state not in {"pending", "claimed", "delivered", "ambiguous"}:
                raise ValueError("controller wake state is invalid")
            attempt = row["attempt"]
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0 or attempt > (1 << 53) - 1:
                raise ValueError("controller wake attempt is invalid")
            if action == "recovery_dispatch" and state != "pending" and attempt < 1:
                raise ValueError("controller wake recovery action requires a claimed attempt")
            claim_token = row["claim_token"]
            lease = row["lease_expires_at_ms"]
            if (claim_token is None) != (lease is None):
                raise ValueError("controller wake claim and lease are inconsistent")
            if state == "claimed" and claim_token is None:
                raise ValueError("claimed controller wake has no owner")
            if state != "claimed" and claim_token is not None:
                raise ValueError("unclaimed controller wake has an owner")
            if claim_token is not None and (
                not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token.encode("utf-8")) > 512
            ):
                raise ValueError("controller wake claim token is invalid")
            if lease is not None and (
                isinstance(lease, bool) or not isinstance(lease, int) or lease < 0 or lease > (1 << 53) - 1
            ):
                raise ValueError("controller wake lease is invalid")
            delivered_at_ms = row.get("delivered_at_ms")
            if delivered_at_ms is not None and (
                isinstance(delivered_at_ms, bool)
                or not isinstance(delivered_at_ms, int)
                or delivered_at_ms < 0
                or delivered_at_ms > (1 << 53) - 1
            ):
                raise ValueError("controller wake delivered_at_ms is invalid")
            last_error = row.get("last_error")
            if last_error is not None and (not isinstance(last_error, str) or len(last_error.encode("utf-8")) > 65536):
                raise ValueError("controller wake last_error is invalid")
            return dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise _controller_error("controller recovery wake outbox is invalid", "controller_command_conflict") from exc

    def _ensure_controller_wake(self, receipt: ControllerCommandReceipt) -> dict[str, Any]:
        row = self._controller_command_outboxes.get(receipt.command_id)
        if row is None:
            raise _controller_error("controller command wake outbox is missing", "controller_command_conflict")
        checked = self._checked_controller_wake(row)
        if checked["command_id"] != receipt.command_id or checked["command_digest"] != receipt.command_digest:
            raise _controller_error("controller receipt and wake identity conflict", "controller_command_conflict")
        return checked

    def _set_controller_wake(self, row: Mapping[str, Any]) -> ControllerCommandReceipt:
        checked = self._checked_controller_wake(row)
        command_id = str(checked["command_id"])
        receipt = self._controller_command_receipts.get(command_id)
        if receipt is None:
            raise _controller_error("controller command receipt is missing", "controller_command_stale")
        updated = replace(
            receipt,
            outbox_state=str(checked["outbox_state"]),
            outbox_attempt=int(checked["attempt"]),
        )
        self._controller_command_outboxes[command_id] = checked
        self._controller_command_receipts[command_id] = updated
        return updated

    def _outcome_for_record(self, record: dict[str, Any], *, status: str, checkpoint_revision: int) -> HostInteractionOutcome:
        self._checked_host_record(record, checkpoint_key=str(record["checkpoint_key"]))
        notification_id = derive_host_interaction_notification_id(record["record_id"])
        notification = self._host_interaction_notifications.get(notification_id)
        if notification is None:
            raise _controller_error("host interaction notification row is missing", "host_interaction_conflict")
        self._checked_notification(notification)
        return HostInteractionOutcome(
            interaction_id=record["interaction_id"],
            logical_cycle=record["logical_cycle"],
            checkpoint_revision=checkpoint_revision,
            status=status,
            # Notification delivery is an independent at-least-once
            # lifecycle.  The producer outcome only acknowledges that the
            # pending notification was durably admitted.
            outbox_state="pending",
            record_id=record["record_id"],
            notification_id=notification_id,
            notification_payload_digest=notification["payload_digest"],
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
        with self._lock:
            # Replay lookup happens before active-claim lookup.  A crash after
            # the CAS but before the producer returned must be a zero-write
            # replay even though the claim has already been released.
            key = admission_context.checkpoint_key
            existing = self._host_interaction_records.get((key, request_value.interaction_id))
            if existing is not None:
                self._checked_host_record(existing, checkpoint_key=key)
                if existing["request_digest"] != request_value.request_digest or existing["request"] != request_value.to_dict():
                    raise _controller_error("host interaction identity or digest conflicts", "host_interaction_conflict")
                current = self._load_controller_checkpoint(key)
                if current is None:
                    raise _controller_error("host interaction checkpoint was deleted", "host_interaction_conflict")
                if current.revision < admission_context.expected_revision + 1:
                    raise _controller_error("host interaction replay revision is stale", "host_interaction_stale")
                return self._outcome_for_record(existing, status="replayed", checkpoint_revision=current.revision)
            key, current, _owner, revision, _cycle = self._find_producer_checkpoint(
                request_value,
                admission_context=admission_context,
            )
            record_id = derive_host_interaction_record_id(key, request_value)
            existing = self._host_interaction_records.get((key, request_value.interaction_id))
            if existing is not None:
                self._checked_host_record(existing, checkpoint_key=key)
                if existing["request_digest"] != request_value.request_digest or existing["record_id"] != record_id:
                    raise _controller_error("host interaction identity or digest conflicts", "host_interaction_conflict")
                return self._outcome_for_record(existing, status="replayed", checkpoint_revision=current.revision)
            notification_id = derive_host_interaction_notification_id(record_id)
            notification_payload = self._notification_payload(record_id, request_value, notification_id)
            event = HostInteractionRequestedEvent(
                run_id=current.root_run_id,
                trace_id=current.trace_id,
                checkpoint_key=key,
                resume_attempt=current.resume_attempt,
                interaction_id=request_value.interaction_id,
                logical_cycle=request_value.logical_cycle,
                operation_id=request_value.operation_id,
                tool_call_id=request_value.tool_call_id,
                request_digest=request_value.request_digest or "",
                prompt=request_value.prompt,
                # cycle_index is the last committed cycle.  The in-flight
                # logical cycle is carried separately as ``logical_cycle``.
                cycle_index=current.cycle_index,
                event_id=f"evt_host_interaction_requested_{record_id[:16]}",
            ).to_dict()
            record = self._record(checkpoint_key=key, request=request_value, record_id=record_id)
            notification = {
                "notification_id": notification_id,
                "checkpoint_key": key,
                "record_id": record_id,
                "payload": notification_payload,
                "payload_digest": canonical_json_sha256(notification_payload, "notification_payload"),
                "outbox_state": "pending",
                "claim_token": None,
                "lease_expires_at_ms": None,
                "attempt": 0,
                "delivered_at_ms": None,
                "aborted_at_ms": None,
                "abort_reason": None,
                "last_error": None,
            }
            validate_host_interaction_record(record)
            validate_host_interaction_notification(
                notification_payload,
                notification_id=notification_id,
                record_id=record_id,
            )
            snapshot = clone_checkpoint(current)
            snapshot.active_host_interaction = request_value.to_dict()
            snapshot.status = AgentStatus.HOST_INTERACTION
            snapshot.claim_token = None
            snapshot.claimed_cycle = None
            snapshot.lease_expires_at_ms = None
            snapshot.revision = revision + 1
            snapshot.event_outbox.append(EventOutboxEntry.pending(event["event_id"], event))
            validate_checkpoint(snapshot)
            record_key = (key, request_value.interaction_id)
            previous_checkpoint = self._store.get(key)
            previous_record = self._host_interaction_records.get(record_key)
            previous_notification = self._host_interaction_notifications.get(notification_id)
            try:
                self._store[key] = snapshot
                self._host_interaction_records[record_key] = record
                self._host_interaction_notifications[notification_id] = notification
            except BaseException:
                if previous_checkpoint is None:
                    self._store.pop(key, None)
                else:
                    self._store[key] = previous_checkpoint
                if previous_record is None:
                    self._host_interaction_records.pop(record_key, None)
                else:
                    self._host_interaction_records[record_key] = previous_record
                if previous_notification is None:
                    self._host_interaction_notifications.pop(notification_id, None)
                else:
                    self._host_interaction_notifications[notification_id] = previous_notification
                raise
            return self._outcome_for_record(record, status="admitted", checkpoint_revision=snapshot.revision)

    def _validate_command_binding(self, command: ControllerCommand, *, lease_now_ms: int) -> Any:
        current = self._load_controller_checkpoint(command.handle.checkpoint_key)
        if current is None:
            raise _controller_error("controller command checkpoint was not found", "controller_command_stale")
        if (current.root_run_id, current.trace_id) != (command.handle.run_id, command.handle.trace_id):
            raise _controller_error("controller command handle does not match checkpoint", "controller_command_stale")
        if current.terminal_result is not None:
            raise _controller_error("controller command cannot rewrite a committed terminal", "controller_command_terminal")
        if current.claim_token is not None and command.kind != "cancel":
            claim_expired = (current.lease_expires_at_ms or 0) <= lease_now_ms
            if command.kind != "suspend" or not claim_expired:
                raise _controller_error(
                    "controller command cannot clear a live execution claim", "controller_command_claim_active"
                )
        journals = [*current.model_call_journal, *current.tool_journal]
        if command.kind != "abort" and (
            current.status is AgentStatus.RECONCILIATION_REQUIRED
            or (
                any(entry.state is OperationState.AMBIGUOUS for entry in journals)
                and not (command.kind == "cancel" and current.claim_token is not None)
            )
        ):
            raise _controller_error(
                "controller command is blocked by an unresolved external effect",
                "controller_command_ambiguity_requires_reconciliation",
            )
        if (command.kind != "cancel" or current.claim_token is None) and (
            current.status is AgentStatus.DEFERRED or any(entry.state is OperationState.DEFERRED for entry in journals)
        ):
            raise _controller_error(
                "controller command is blocked by a deferred effect",
                "controller_command_deferred_pending",
            )
        if current.resume_attempt != command.resume_attempt or current.revision != command.expected_revision:
            raise _controller_error("controller command fences are stale", "controller_command_stale")
        return current

    def _append_control_events(
        self,
        checkpoint: Any,
        *,
        state: str,
        cancelled: bool = False,
        error: str | None = None,
        error_code: str | None = None,
        cancel_transition: bool = False,
    ) -> None:
        metadata: dict[str, Any] = {}
        if error_code is not None:
            metadata["error_code"] = error_code
        common: dict[str, Any] = {
            "run_id": checkpoint.root_run_id,
            "trace_id": checkpoint.trace_id,
            "agent_name": None,
            "session_id": None,
            "cycle_index": checkpoint.cycle_index,
            "event_id": None,
            "created_at": None,
            "metadata": metadata or None,
        }
        events: list[dict[str, Any]] = [
            RunStateChangedEvent(
                state=state,
                cancel_requested={"from": False, "to": True} if cancel_transition else None,
                **common,
            ).to_dict()
        ]
        if cancelled:
            events.append(RunCancelledEvent(reason="cancelled", completion_reason=CompletionReason.CANCELLED, **common).to_dict())
        elif error is not None:
            events.append(RunFailedEvent(error=error, completion_reason=CompletionReason.FAILED, **common).to_dict())
        for event in events:
            checkpoint.event_outbox.append(EventOutboxEntry.pending(event["event_id"], event))

    @staticmethod
    def _append_resume_event(checkpoint: Any) -> None:
        event = CheckpointResumedEvent(
            run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
            checkpoint_key=checkpoint.checkpoint_key,
            resume_attempt=checkpoint.resume_attempt,
            cycle_index=checkpoint.cycle_index,
        ).to_dict()
        checkpoint.event_outbox.append(EventOutboxEntry.pending(event["event_id"], event))

    def _terminal_result(
        self,
        checkpoint: Any,
        *,
        reason: CompletionReason,
        error: dict[str, Any] | None,
        resume_observations: list[ResumeObservation] | None = None,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            messages=deepcopy(checkpoint.messages),
            cycles=deepcopy(checkpoint.cycles),
            final_answer=None,
            error=deepcopy(error),
            shared_state=deepcopy(checkpoint.shared_state),
            token_usage=summarize_task_token_usage(checkpoint.model_calls),
            completion_reason=reason,
            checkpoint_key=checkpoint.checkpoint_key,
            resume_observations=resume_observations or [],
        )

    def admit_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandReceipt:
        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        return self._admit_controller_command(command_value, lease_now_ms=None)

    def _admit_controller_command(
        self,
        command_value: ControllerCommand,
        *,
        lease_now_ms: int | None,
    ) -> ControllerCommandReceipt:
        with self._lock:
            existing = self._controller_command_receipts.get(command_value.command_id)
            if existing is not None:
                if existing.command_digest != command_value.command_digest:
                    raise _controller_error(
                        "controller command id was reused with a different digest", "controller_command_conflict"
                    )
                self._ensure_controller_wake(existing)
                return deepcopy(self._controller_command_receipts[command_value.command_id])
            now_ms = self._lease_now_ms(lease_now_ms)
            checkpoint = clone_checkpoint(self._validate_command_binding(command_value, lease_now_ms=now_ms))
            kind = command_value.kind
            resulting_status = checkpoint.status.value
            wake_action = "none"
            wake_destination: str | None = None
            staged_record: dict[str, Any] | None = None
            if kind == "host_interaction_response":
                active = checkpoint.active_host_interaction
                suspended_host_origin = (
                    checkpoint.status is AgentStatus.SUSPENDED
                    and isinstance(checkpoint.suspended_origin, dict)
                    and checkpoint.suspended_origin.get("status") == AgentStatus.HOST_INTERACTION.value
                )
                if suspended_host_origin:
                    active = checkpoint.suspended_origin.get("active_host_interaction")
                if not isinstance(active, dict):
                    raise _controller_error(
                        "host interaction response has no pending interaction", "host_interaction_recovery_required"
                    )
                command_payload = command_value.command
                identity_fields = ("interaction_id", "logical_cycle", "operation_id", "tool_call_id", "request_digest")
                if any(command_payload[field] != active[field] for field in identity_fields):
                    raise _controller_error("host interaction response identity conflicts", "host_interaction_conflict")
                record = self._host_interaction_records.get((checkpoint.checkpoint_key, active["interaction_id"]))
                if record is None:
                    raise _controller_error("host interaction record is missing", "host_interaction_recovery_required")
                try:
                    validate_host_interaction_record(record, checkpoint_key=checkpoint.checkpoint_key)
                except (TypeError, ValueError) as exc:
                    raise _controller_error("host interaction record is invalid", "host_interaction_conflict") from exc
                if record["request_digest"] != active["request_digest"] or record["state"] != "active":
                    raise _controller_error("host interaction request digest conflicts", "host_interaction_conflict")
                record = deepcopy(record)
                response_value = dict(command_payload["response"])
                response_digest = derive_host_response_digest(
                    interaction_id=active["interaction_id"],
                    logical_cycle=active["logical_cycle"],
                    operation_id=active["operation_id"],
                    tool_call_id=active["tool_call_id"],
                    request_digest=active["request_digest"],
                    command_id=command_value.command_id,
                    response=response_value,
                )
                resolved_response = HostInteractionResponse(
                    interaction_id=active["interaction_id"],
                    logical_cycle=active["logical_cycle"],
                    operation_id=active["operation_id"],
                    tool_call_id=active["tool_call_id"],
                    request_digest=active["request_digest"],
                    command_id=command_value.command_id,
                    response=response_value,
                    response_digest=response_digest,
                )
                record["response"] = resolved_response.to_dict()
                record["response_digest"] = resolved_response.response_digest
                record["command_id"] = command_value.command_id
                record["state"] = "resolved_pending"
                record["resolved_revision"] = checkpoint.revision + 1
                validate_host_interaction_record(record, checkpoint_key=checkpoint.checkpoint_key)
                staged_record = record
                if suspended_host_origin:
                    # A response submitted while suspended is durable but does
                    # not wake a worker.  Resume is the sole wake transition.
                    checkpoint.status = AgentStatus.SUSPENDED
                    checkpoint.active_host_interaction = None
                    resulting_status = AgentStatus.SUSPENDED.value
                else:
                    checkpoint.active_host_interaction = None
                    checkpoint.status = AgentStatus.RUNNING
                    resulting_status = AgentStatus.RUNNING.value
                    wake_action = "recovery_dispatch"
                    wake_destination = "distributed_advance"
                checkpoint.revision += 1
            elif kind == "suspend":
                if checkpoint.status not in {AgentStatus.RUNNING, AgentStatus.HOST_INTERACTION}:
                    raise _controller_error("suspend command is not admissible for this state", "controller_command_stale")
                checkpoint.suspended_origin = {
                    "status": checkpoint.status.value,
                    "active_host_interaction": deepcopy(checkpoint.active_host_interaction),
                }
                checkpoint.active_host_interaction = None
                checkpoint.status = AgentStatus.SUSPENDED
                checkpoint.claim_token = None
                checkpoint.claimed_cycle = None
                checkpoint.lease_expires_at_ms = None
                checkpoint.revision += 1
                self._append_control_events(checkpoint, state=AgentStatus.SUSPENDED.value)
                resulting_status = AgentStatus.SUSPENDED.value
            elif kind == "resume":
                if checkpoint.status is not AgentStatus.SUSPENDED or checkpoint.suspended_origin is None:
                    raise _controller_error("resume command requires a suspended checkpoint", "controller_command_stale")
                origin = checkpoint.suspended_origin
                active = origin.get("active_host_interaction")
                pending = False
                if isinstance(active, dict):
                    record = self._host_interaction_records.get((checkpoint.checkpoint_key, active.get("interaction_id")))
                    if record is None:
                        raise _controller_error(
                            "suspended host interaction record is missing", "host_interaction_recovery_required"
                        )
                    self._checked_host_record(record, checkpoint_key=checkpoint.checkpoint_key)
                    pending = record["state"] in {"resolved_pending", "resolved_claimed"}
                if origin["status"] == AgentStatus.HOST_INTERACTION.value and not pending:
                    checkpoint.status = AgentStatus.HOST_INTERACTION
                    checkpoint.active_host_interaction = deepcopy(active)
                    checkpoint.suspended_origin = None
                else:
                    checkpoint.status = AgentStatus.RUNNING
                    checkpoint.active_host_interaction = None
                    checkpoint.suspended_origin = None
                    wake_action = "recovery_dispatch"
                    wake_destination = "distributed_advance"
                checkpoint.revision += 1
                self._append_resume_event(checkpoint)
                resulting_status = checkpoint.status.value
            elif kind == "cancel":
                if checkpoint.claim_token is not None:
                    claim_expired = (checkpoint.lease_expires_at_ms or 0) <= now_ms
                    if not claim_expired:
                        cancel_transition = not checkpoint.cancel_requested
                        checkpoint.cancel_requested = True
                        if cancel_transition:
                            self._append_control_events(
                                checkpoint,
                                state=checkpoint.status.value,
                                cancelled=False,
                                cancel_transition=True,
                            )
                        resulting_status = checkpoint.status.value
                    else:
                        recovery_claim = f"controller-recovery-{command_value.command_id}"
                        authority = clone_checkpoint(checkpoint)
                        authority.resume_attempt += 1
                        authority.claim_token = recovery_claim
                        authority.claimed_cycle = checkpoint.claimed_cycle
                        authority.lease_expires_at_ms = now_ms + _RECOVERY_LEASE_DURATION_MS
                        authority.cancel_requested = True
                        terminal_candidate = clone_checkpoint(authority)
                        terminal_candidate.terminal_result = self._terminal_result(
                            authority,
                            reason=CompletionReason.CANCELLED,
                            error={
                                "code": "cancelled_with_unknown_outcome",
                                "message": "Cancellation was accepted while the external outcome remained unknown.",
                                "retryable": False,
                            },
                        )
                        terminal_candidate.status = AgentStatus.FAILED
                        terminal_candidate.active_host_interaction = None
                        terminal_candidate.suspended_origin = None
                        checkpoint = prepare_claimed_terminal(
                            authority,
                            terminal_candidate,
                            claim_token=recovery_claim,
                            expected_revision=authority.revision,
                        )
                        if checkpoint is None:
                            raise _controller_error(
                                "expired claim cancellation lost its recovery fence",
                                "controller_command_stale",
                            )
                        self._append_control_events(checkpoint, state=AgentStatus.FAILED.value, cancelled=True)
                        resulting_status = AgentStatus.FAILED.value
                else:
                    checkpoint.terminal_result = self._terminal_result(
                        checkpoint,
                        reason=CompletionReason.CANCELLED,
                        error={
                            "code": "cancelled_with_unknown_outcome",
                            "message": "Cancellation was accepted while the external outcome remained unknown.",
                            "retryable": False,
                        },
                    )
                    checkpoint.status = AgentStatus.FAILED
                    checkpoint.active_host_interaction = None
                    checkpoint.suspended_origin = None
                    checkpoint.claim_token = None
                    checkpoint.claimed_cycle = None
                    checkpoint.lease_expires_at_ms = None
                    checkpoint = prepare_unclaimed_terminal(checkpoint)
                    checkpoint.revision += 1
                    self._append_control_events(checkpoint, state=AgentStatus.FAILED.value, cancelled=True)
                    resulting_status = AgentStatus.FAILED.value
            elif kind == "abort":
                if checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED:
                    raise _controller_error("abort requires reconciliation_required", "controller_command_stale")
                ambiguous = next(
                    (
                        entry
                        for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
                        if entry.state is OperationState.AMBIGUOUS
                    ),
                    None,
                )
                if ambiguous is None:
                    raise _controller_error("abort requires an unresolved external effect", "controller_command_stale")
                observation = ResumeObservation(
                    operation_id=ambiguous.operation_id,
                    operation_kind=ambiguous.kind,
                    cycle_index=ambiguous.cycle_index,
                    risk="operator abort with unknown outcome",
                    idempotency_support=ambiguous.idempotency_support,
                )
                checkpoint.terminal_result = self._terminal_result(
                    checkpoint,
                    reason=CompletionReason.FAILED,
                    error={
                        "code": "operator_abort_with_unknown_outcome",
                        "message": "Operator accepted that the external outcome is unknown.",
                        "retryable": False,
                    },
                    resume_observations=[observation],
                )
                checkpoint.status = AgentStatus.FAILED
                checkpoint.claim_token = None
                checkpoint.claimed_cycle = None
                checkpoint.lease_expires_at_ms = None
                checkpoint = prepare_unclaimed_terminal(checkpoint)
                checkpoint.revision += 1
                self._append_control_events(
                    checkpoint,
                    state=AgentStatus.FAILED.value,
                    error="failed",
                    error_code="operator_abort_with_unknown_outcome",
                )
                resulting_status = AgentStatus.FAILED.value
            else:  # pragma: no cover - ControllerCommand closes the variants.
                raise _controller_error("unsupported controller command", "controller_command_invalid")
            validate_checkpoint(checkpoint)
            receipt = ControllerCommandReceipt(
                command_id=command_value.command_id,
                command_digest=command_value.command_digest or "",
                handle=command_value.handle,
                resume_attempt=command_value.resume_attempt,
                expected_revision=command_value.expected_revision,
                resulting_revision=checkpoint.revision,
                resulting_status=resulting_status,
                outbox_state="pending" if wake_action == "recovery_dispatch" else "delivered",
                outbox_action=wake_action,
                outbox_destination=wake_destination,
                outbox_attempt=0,
            )
            checkpoint_key = checkpoint.checkpoint_key
            record_key = (checkpoint_key, staged_record["interaction_id"]) if staged_record is not None else None
            previous_checkpoint = self._store.get(checkpoint_key)
            previous_record = self._host_interaction_records.get(record_key) if record_key is not None else None
            previous_receipt = self._controller_command_receipts.get(command_value.command_id)
            previous_command = self._controller_commands.get(command_value.command_id)
            previous_outbox = self._controller_command_outboxes.get(command_value.command_id)
            try:
                self._store[checkpoint_key] = checkpoint
                if staged_record is not None and record_key is not None:
                    self._host_interaction_records[record_key] = staged_record
                self._controller_command_receipts[command_value.command_id] = receipt
                self._controller_commands[command_value.command_id] = deepcopy(command_value)
                self._controller_command_outboxes[command_value.command_id] = self._controller_wake_from_receipt(receipt)
            except BaseException:
                if previous_checkpoint is None:
                    self._store.pop(checkpoint_key, None)
                else:
                    self._store[checkpoint_key] = previous_checkpoint
                if record_key is not None:
                    if previous_record is None:
                        self._host_interaction_records.pop(record_key, None)
                    else:
                        self._host_interaction_records[record_key] = previous_record
                if previous_receipt is None:
                    self._controller_command_receipts.pop(command_value.command_id, None)
                else:
                    self._controller_command_receipts[command_value.command_id] = previous_receipt
                if previous_command is None:
                    self._controller_commands.pop(command_value.command_id, None)
                else:
                    self._controller_commands[command_value.command_id] = previous_command
                if previous_outbox is None:
                    self._controller_command_outboxes.pop(command_value.command_id, None)
                else:
                    self._controller_command_outboxes[command_value.command_id] = previous_outbox
                raise
            return deepcopy(receipt)

    def get_controller_command_receipt(self, command_id: str) -> ControllerCommandReceipt | None:
        with self._lock:
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is None:
                return None
            self._ensure_controller_wake(receipt)
            return deepcopy(self._controller_command_receipts[command_id])

    def claim_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> dict[str, Any] | None:
        """Claim a pending recovery wake without changing checkpoint state."""
        if not isinstance(claim_token, str) or not claim_token.strip():
            raise ValueError("controller wake claim_token must be non-empty")
        if isinstance(lease_expires_at_ms, bool) or not isinstance(lease_expires_at_ms, int) or lease_expires_at_ms <= now_ms:
            raise ValueError("controller wake lease must be greater than now_ms")
        with self._lock:
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is None:
                return None
            if receipt.command_digest != command_digest:
                raise _controller_error("controller command digest conflicts", "controller_command_conflict")
            row = self._ensure_controller_wake(receipt)
            if row["outbox_action"] == "none" or row["outbox_state"] == "delivered":
                return deepcopy(row)
            if row["outbox_state"] == "ambiguous":
                raise _controller_error("controller wake requires reconciliation", "controller_command_stale")
            if row["outbox_state"] == "claimed":
                if row["claim_token"] == claim_token:
                    return deepcopy(row)
                if int(row["lease_expires_at_ms"] or 0) > now_ms:
                    raise _controller_error("controller wake is claimed by another owner", "controller_command_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = "claimed"
            staged["claim_token"] = claim_token
            staged["lease_expires_at_ms"] = lease_expires_at_ms
            staged["attempt"] = int(row["attempt"]) + 1
            self._set_controller_wake(staged)
            return deepcopy(staged)

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
        """Owner-attempt CAS for a recovery wake delivery."""
        if outcome not in {"delivered", "ambiguous"}:
            raise ValueError("controller wake completion outcome must be delivered or ambiguous")
        with self._lock:
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is None:
                return None
            if receipt.command_digest != command_digest:
                raise _controller_error("controller command digest conflicts", "controller_command_conflict")
            row = self._ensure_controller_wake(receipt)
            if row["outbox_state"] in {"delivered", "ambiguous"}:
                if row["outbox_state"] == outcome:
                    return deepcopy(row)
                raise _controller_error("controller wake has already completed", "controller_command_stale")
            if row["outbox_state"] != "claimed" or row["claim_token"] != claim_token or row["attempt"] != attempt:
                raise _controller_error("controller wake owner or attempt is stale", "controller_command_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = outcome
            staged["claim_token"] = None
            staged["lease_expires_at_ms"] = None
            staged["delivered_at_ms"] = now_ms if outcome == "delivered" else None
            staged["last_error"] = error
            self._set_controller_wake(staged)
            return deepcopy(staged)

    def reconcile_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        outcome: str,
        now_ms: int,
    ) -> dict[str, Any] | None:
        """Resolve an ambiguous wake without synthesizing a second command."""
        if outcome not in {"delivered", "retry"}:
            raise ValueError("controller wake reconciliation outcome must be delivered or retry")
        with self._lock:
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is None:
                return None
            if receipt.command_digest != command_digest:
                raise _controller_error("controller command digest conflicts", "controller_command_conflict")
            row = self._ensure_controller_wake(receipt)
            target = "delivered" if outcome == "delivered" else "pending"
            if row["outbox_state"] == target:
                return deepcopy(row)
            if row["outbox_state"] != "ambiguous":
                raise _controller_error("controller wake is not ambiguous", "controller_command_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = target
            staged["delivered_at_ms"] = now_ms if target == "delivered" else None
            staged["last_error"] = None
            self._set_controller_wake(staged)
            return deepcopy(staged)

    def _reap_controller_command_wake(self, *, command_id: str, now_ms: int) -> dict[str, Any] | None:
        """Return one wake to pending after an expired claim; never retry ambiguous."""
        with self._lock:
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is None:
                return None
            row = self._ensure_controller_wake(receipt)
            if row["outbox_state"] != "claimed" or int(row["lease_expires_at_ms"] or 0) > now_ms:
                return deepcopy(row)
            staged = deepcopy(row)
            staged["outbox_state"] = "pending"
            staged["claim_token"] = None
            staged["lease_expires_at_ms"] = None
            self._set_controller_wake(staged)
            return deepcopy(staged)

    def reap_controller_command_wakes(self, checkpoint_key: str, now_ms: int) -> list[dict[str, Any]]:
        with self._lock:
            candidates = sorted(
                (
                    receipt.expected_revision,
                    command_id,
                )
                for command_id, receipt in self._controller_command_receipts.items()
                if receipt.handle.checkpoint_key == checkpoint_key
                and (row := self._ensure_controller_wake(receipt))["outbox_action"] == "recovery_dispatch"
                and (
                    row["outbox_state"] == "pending"
                    or (row["outbox_state"] == "claimed" and int(row["lease_expires_at_ms"] or 0) <= now_ms)
                )
            )
        rows: list[dict[str, Any]] = []
        for _expected_revision, command_id in candidates:
            row = self._reap_controller_command_wake(command_id=command_id, now_ms=now_ms)
            if row is not None and row["outbox_action"] == "recovery_dispatch" and row["outbox_state"] == "pending":
                rows.append(row)
        return rows

    def get_controller_command(self, command_id: str) -> ControllerCommand | None:
        """Return the canonical command for App Server idempotency replay.

        The receipt intentionally exposes only framework fields.  The App
        Server projection still needs the original closed command to compare
        a retried public action with the durable command digest; keeping this
        lookup beside the receipt avoids treating a reused action id as an
        unconditional success.
        """
        with self._lock:
            command = self._controller_commands.get(command_id)
            receipt = self._controller_command_receipts.get(command_id)
            if receipt is not None:
                self._checked_controller_wake(self._ensure_controller_wake(receipt))
            return deepcopy(command) if command is not None else None

    def resolve_controller_command(self, command: ControllerCommand | Mapping[str, Any]) -> ControllerCommandResolution:
        command_value = command if isinstance(command, ControllerCommand) else ControllerCommand.from_dict(command)
        with self._lock:
            had_receipt = command_value.command_id in self._controller_command_receipts
            try:
                receipt = self.admit_controller_command(command_value)
            except CheckpointError as exc:
                return ControllerCommandResolution(kind="rejected", error=getattr(exc, "code", None) or str(exc))
        wake = ControllerWake(
            action=receipt.outbox_action,
            destination=receipt.outbox_destination,
            logical_cycle=(
                int(command_value.command["logical_cycle"])
                if command_value.kind == "host_interaction_response"
                else self._load_controller_checkpoint(command_value.handle.checkpoint_key).cycle_index + 1
            ),
            claim_mode="recovery" if receipt.outbox_action == "recovery_dispatch" else "none",
        )
        kind = "replayed" if had_receipt else "applied"
        return ControllerCommandResolution(kind=kind, receipt=receipt, wake=wake)

    def claim_and_consume_host_interaction_response(self, envelope: Mapping[str, Any]) -> HostInteractionRecoveryResult:
        try:
            envelope_value = (
                envelope
                if isinstance(envelope, HostInteractionRecoveryEnvelope)
                else HostInteractionRecoveryEnvelope.from_dict(envelope)
            )
        except (TypeError, ValueError) as exc:
            raise _controller_error(str(exc), "host_interaction_recovery_stale") from exc
        return self._claim_and_consume_host_interaction_response(envelope_value, lease_now_ms=None)

    def _claim_and_consume_host_interaction_response(
        self,
        envelope_value: HostInteractionRecoveryEnvelope,
        *,
        lease_now_ms: int | None,
    ) -> HostInteractionRecoveryResult:
        record_id = envelope_value.record_id
        checkpoint_key = envelope_value.checkpoint_key
        with self._lock:
            checkpoint = self._load_controller_checkpoint(checkpoint_key)
            record = self._host_interaction_records.get((checkpoint_key, envelope_value.interaction_id))
            if checkpoint is None or record is None or record["checkpoint_key"] != checkpoint_key:
                raise _controller_error("host interaction recovery record was not found", "host_interaction_recovery_stale")
            try:
                validate_host_interaction_record(record, checkpoint_key=checkpoint_key)
            except (TypeError, ValueError) as exc:
                raise _controller_error("host interaction recovery record is invalid", "host_interaction_recovery_stale") from exc
            if record["record_id"] != record_id:
                raise _controller_error("host interaction recovery record identity conflicts", "host_interaction_recovery_stale")
            if (checkpoint.root_run_id, checkpoint.trace_id) != (envelope_value.run_id, envelope_value.trace_id):
                raise _controller_error("host interaction recovery handle is stale", "host_interaction_recovery_stale")
            request = HostInteractionRequest.from_dict(record["request"])
            if (
                request.interaction_id != envelope_value.interaction_id
                or request.logical_cycle != envelope_value.logical_cycle
                or request.operation_id != envelope_value.operation_id
                or request.tool_call_id != envelope_value.tool_call_id
                or request.request_digest != envelope_value.request_digest
                or record.get("command_id") != envelope_value.command_id
            ):
                raise _controller_error("host interaction recovery identity conflicts", "host_interaction_recovery_stale")
            if record["state"] == "consumed":
                return HostInteractionRecoveryResult(
                    kind="replayed",
                    record_id=record_id,
                    checkpoint_revision=checkpoint.revision,
                    consumed_revision=record["consumed_revision"],
                    claim_mode="recovery",
                    resume_attempt=checkpoint.resume_attempt,
                    injection_count=1,
                    checkpoint_execution_claim_state="retained" if checkpoint.claim_token else "released",
                )
            # ``resolved_claimed`` is an in-transaction phase only.  If it is
            # ever observed durably, refuse a record-only continuation rather
            # than allowing a second worker to consume it.
            if record["state"] != "resolved_pending":
                raise _controller_error(
                    "host interaction response is not ready for recovery", "host_interaction_recovery_required"
                )
            if (
                checkpoint.revision != envelope_value.expected_revision
                or checkpoint.resume_attempt != envelope_value.resume_attempt
            ):
                raise _controller_error("host interaction recovery fences are stale", "host_interaction_recovery_stale")
            resolved_revision = record.get("resolved_revision")
            if resolved_revision != checkpoint.revision:
                resumed_after_admission = (
                    isinstance(resolved_revision, int)
                    and checkpoint.revision == resolved_revision + 1
                    and checkpoint.status is AgentStatus.RUNNING
                    and any(entry.event.get("type") == "checkpoint_resumed" for entry in checkpoint.event_outbox)
                )
                if not resumed_after_admission:
                    raise _controller_error("host interaction recovery revision is stale", "host_interaction_recovery_stale")
            if (
                checkpoint.status in {AgentStatus.SUSPENDED, AgentStatus.DEFERRED, AgentStatus.RECONCILIATION_REQUIRED}
                or checkpoint.terminal_result is not None
            ):
                raise _controller_error(
                    "host interaction recovery checkpoint is not executable", "host_interaction_recovery_stale"
                )
            if (
                checkpoint.claim_token is not None
                or checkpoint.claimed_cycle is not None
                or checkpoint.lease_expires_at_ms is not None
            ):
                raise _controller_error(
                    "host interaction recovery cannot replace an existing execution claim",
                    "host_interaction_recovery_stale",
                )
            response = record["response"]
            try:
                resolved_response = HostInteractionResponse.from_dict(response)
            except (TypeError, ValueError):
                raise _controller_error("resolved host response is incomplete", "host_interaction_recovery_stale") from None
            if resolved_response.command_id != envelope_value.command_id:
                raise _controller_error("resolved host response command binding is stale", "host_interaction_recovery_stale")
            # The combined operation is one lock/CAS: claim the next cycle,
            # inject exactly one user message, append the consumed event, and
            # retain the checkpoint execution claim for the worker.
            snapshot = clone_checkpoint(checkpoint)
            staged_record = deepcopy(record)
            snapshot.resume_attempt += 1
            snapshot.revision += 1
            snapshot.claim_token = f"host-response:{record_id}:{record['attempt'] + 1}"
            snapshot.claimed_cycle = snapshot.cycle_index + 1
            now_ms = self._lease_now_ms(lease_now_ms)
            snapshot.lease_expires_at_ms = now_ms + _RECOVERY_LEASE_DURATION_MS
            snapshot.status = AgentStatus.RUNNING
            snapshot.messages.append(Message(role="user", content=resolved_response.response["content"]))
            staged_record["state"] = "resolved_claimed"
            staged_record["claim_token"] = snapshot.claim_token
            staged_record["lease_expires_at_ms"] = snapshot.lease_expires_at_ms
            staged_record["attempt"] += 1
            validate_host_interaction_record(staged_record, checkpoint_key=checkpoint_key)
            consumed_event = HostInteractionResponseConsumedEvent(
                run_id=snapshot.root_run_id,
                trace_id=snapshot.trace_id,
                checkpoint_key=checkpoint_key,
                resume_attempt=snapshot.resume_attempt,
                interaction_id=record["interaction_id"],
                logical_cycle=record["logical_cycle"],
                operation_id=record["request"]["operation_id"],
                tool_call_id=record["request"]["tool_call_id"],
                request_digest=record["request_digest"],
                command_id=record["command_id"],
                response_digest=record["response_digest"],
                consumed_revision=snapshot.revision,
                cycle_index=snapshot.cycle_index,
                event_id=f"evt_host_interaction_consumed_{record_id[:16]}",
            ).to_dict()
            snapshot.event_outbox.append(EventOutboxEntry.pending(consumed_event["event_id"], consumed_event))
            staged_record["state"] = "consumed"
            staged_record["consumed_revision"] = snapshot.revision
            staged_record["claim_token"] = None
            staged_record["lease_expires_at_ms"] = None
            validate_host_interaction_record(staged_record, checkpoint_key=checkpoint_key)
            validate_checkpoint(snapshot)
            return_value = HostInteractionRecoveryResult(
                kind="applied",
                record_id=record_id,
                checkpoint_revision=snapshot.revision,
                consumed_revision=staged_record["consumed_revision"],
                claim_mode="recovery",
                resume_attempt=snapshot.resume_attempt,
                injection_count=1,
                checkpoint_execution_claim_state="retained",
            )
            record_key = (checkpoint_key, request.interaction_id)
            previous_checkpoint = self._store.get(checkpoint_key)
            previous_record = self._host_interaction_records.get(record_key)
            try:
                self._store[checkpoint_key] = snapshot
                self._host_interaction_records[record_key] = staged_record
            except BaseException:
                if previous_checkpoint is None:
                    self._store.pop(checkpoint_key, None)
                else:
                    self._store[checkpoint_key] = previous_checkpoint
                if previous_record is None:
                    self._host_interaction_records.pop(record_key, None)
                else:
                    self._host_interaction_records[record_key] = previous_record
                raise
            return return_value

    def reap_host_interaction_record(self, *, record_id: str, checkpoint_key: str, now_ms: int) -> bool:
        now_ms = self._lease_now_ms(now_ms)
        with self._lock:
            record = next(
                (
                    item
                    for (key, _interaction), item in self._host_interaction_records.items()
                    if key == checkpoint_key and item["record_id"] == record_id
                ),
                None,
            )
            checkpoint = self._load_controller_checkpoint(checkpoint_key)
            if record is None or checkpoint is None or record["checkpoint_key"] != checkpoint_key:
                return False
            self._checked_host_record(record, checkpoint_key=checkpoint_key)
            if record["state"] != "resolved_claimed" or record["claim_token"] is None:
                return False
            if (
                checkpoint.claim_token is None
                or checkpoint.status is not AgentStatus.RUNNING
                or record["claim_token"] != checkpoint.claim_token
                or checkpoint.lease_expires_at_ms is None
                or checkpoint.lease_expires_at_ms > now_ms
            ):
                return False
            staged = deepcopy(record)
            staged["state"] = "resolved_pending"
            staged["claim_token"] = None
            staged["lease_expires_at_ms"] = None
            staged["last_error"] = "host_interaction_response_claim_expired"
            self._checked_host_record(staged, checkpoint_key=checkpoint_key)
            self._host_interaction_records[(checkpoint_key, record["interaction_id"])] = staged
            return True

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
        with self._lock:
            row = self._host_interaction_notifications.get(notification_id)
            if row is None:
                return None
            self._checked_notification(row)
            if row["payload_digest"] != payload_digest:
                raise _controller_error("notification payload digest conflicts", "notification_conflict")
            if row["outbox_state"] in {"delivered", "aborted"}:
                return deepcopy(row)
            if row["outbox_state"] == "ambiguous":
                raise _controller_error("ambiguous notification requires reconciliation", "notification_stale")
            if (
                row["outbox_state"] == "claimed"
                and row["claim_token"] != claim_token
                and (row["lease_expires_at_ms"] or 0) > now_ms
            ):
                raise _controller_error("notification is claimed by another owner", "notification_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = "claimed"
            staged["claim_token"] = claim_token
            staged["lease_expires_at_ms"] = lease_expires_at_ms
            staged["attempt"] += 1
            self._checked_notification(staged)
            self._host_interaction_notifications[notification_id] = staged
            return deepcopy(staged)

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
        with self._lock:
            row = self._host_interaction_notifications.get(notification_id)
            if row is None:
                return None
            self._checked_notification(row)
            if row["payload_digest"] != payload_digest:
                raise _controller_error("notification payload digest conflicts", "notification_conflict")
            if row["claim_token"] != claim_token or row["attempt"] != attempt or row["outbox_state"] != "claimed":
                raise _controller_error("notification owner or attempt is stale", "notification_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = outcome
            staged["claim_token"] = None
            staged["lease_expires_at_ms"] = None
            staged["last_error"] = error
            staged["delivered_at_ms"] = now_ms if outcome == "delivered" else None
            self._checked_notification(staged)
            self._host_interaction_notifications[notification_id] = staged
            return deepcopy(staged)

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
        with self._lock:
            row = self._host_interaction_notifications.get(notification_id)
            if row is None:
                return None
            self._checked_notification(row)
            if row["payload_digest"] != payload_digest:
                raise _controller_error("notification payload digest conflicts", "notification_conflict")
            if row["outbox_state"] not in {"ambiguous", "delivered", "pending", "aborted"}:
                raise _controller_error("notification state is not reconcilable", "notification_stale")
            target = {"delivered": "delivered", "retry": "pending", "abort": "aborted"}[outcome]
            if row["outbox_state"] == target:
                return deepcopy(row)
            if row["outbox_state"] != "ambiguous":
                raise _controller_error("notification is not ambiguous", "notification_stale")
            staged = deepcopy(row)
            staged["outbox_state"] = target
            staged["delivered_at_ms"] = now_ms if target == "delivered" else None
            staged["aborted_at_ms"] = now_ms if target == "aborted" else None
            staged["abort_reason"] = abort_reason if target == "aborted" else None
            self._checked_notification(staged)
            self._host_interaction_notifications[notification_id] = staged
            return deepcopy(staged)
