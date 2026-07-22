from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import (
    RUN_DEFINITION_SCHEMA,
    CheckpointError,
    EventCursor,
    OperationKind,
    OperationState,
    ToolIdempotency,
    canonical_json_bytes,
    compute_event_payload_digest,
    compute_operation_request_digest,
    compute_run_definition_digest,
    validate_extension_namespace,
    validate_sha256,
)
from vv_agent.types import AgentResult, AgentStatus, CycleRecord, Message

CHECKPOINT_SCHEMA = "vv-agent.checkpoint.v2"
MAX_WIRE_INTEGER = (1 << 53) - 1
ClaimMode = Literal["continue", "recovery"]


class CheckpointConflictError(RuntimeError):
    """The requested checkpoint transition lost its compare-and-swap race."""


class _LeaseOperationClock:
    def __init__(self, now_ms: int) -> None:
        self._now_ms = now_ms
        self._started_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        elapsed_ms = max(0, time.monotonic_ns() - self._started_ns) // 1_000_000
        return min((1 << 64) - 1, self._now_ms + elapsed_ms)


def _validate_claim(cycle_index: int, claim_token: str, lease_expires_at_ms: int, now_ms: int) -> None:
    _positive_wire_integer(cycle_index, "claimed cycle_index")
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


def _validate_renew(claim_token: str, lease_expires_at_ms: int, now_ms: int) -> None:
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


@dataclass(frozen=True, slots=True)
class OperationError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("operation error code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("operation error message must be non-empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("operation error retryable must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperationError:
        if not isinstance(payload, dict):
            raise ValueError("operation error must be an object")
        if set(payload) != {"code", "message", "retryable"}:
            raise ValueError("operation error has missing or unknown fields")
        return cls(
            code=_required_string(payload, "code"),
            message=_required_string(payload, "message"),
            retryable=_required_boolean(payload, "retryable", default=False),
        )


@dataclass(slots=True)
class OperationJournalEntry:
    kind: OperationKind
    operation_id: str
    cycle_index: int
    attempt: int
    state: OperationState
    request_digest: str
    idempotency_key: str | None = None
    response: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: OperationError | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    idempotency_support: ToolIdempotency | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperationKind):
            try:
                self.kind = OperationKind(self.kind)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "operation kind is invalid",
                    code="operation_kind_fields_invalid",
                ) from exc
        if not isinstance(self.state, OperationState):
            try:
                self.state = OperationState(self.state)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "operation state is invalid",
                    code="operation_state_invalid",
                ) from exc
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise CheckpointError(
                "operation_id must be non-empty",
                code="operation_id_invalid",
            )
        try:
            _positive_wire_integer(self.cycle_index, "operation cycle_index")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="operation_cycle_invalid") from exc
        try:
            _positive_wire_integer(self.attempt, "operation attempt")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="operation_attempt_invalid") from exc
        try:
            validate_sha256(self.request_digest, "operation request_digest")
        except ValueError as exc:
            raise CheckpointError(
                str(exc),
                code="operation_request_digest_invalid",
            ) from exc
        if self.idempotency_key is not None and (not isinstance(self.idempotency_key, str) or not self.idempotency_key):
            raise CheckpointError(
                "operation idempotency_key must be a non-empty string or null",
                code="operation_idempotency_key_invalid",
            )
        if self.kind is OperationKind.MODEL:
            if any(
                value is not None
                for value in (
                    self.tool_call_id,
                    self.tool_name,
                    self.arguments,
                    self.idempotency_support,
                    self.result,
                )
            ):
                raise CheckpointError(
                    "model journal entry cannot contain tool fields",
                    code="operation_kind_fields_invalid",
                )
        else:
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
                raise CheckpointError(
                    "tool journal entry requires tool_call_id",
                    code="operation_kind_fields_invalid",
                )
            if not isinstance(self.tool_name, str) or not self.tool_name:
                raise CheckpointError(
                    "tool journal entry requires tool_name",
                    code="operation_kind_fields_invalid",
                )
            if not isinstance(self.arguments, dict):
                raise CheckpointError(
                    "tool journal entry requires object arguments",
                    code="operation_kind_fields_invalid",
                )
            canonical_json_bytes(self.arguments, "tool journal arguments")
            if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
                raise CheckpointError(
                    "tool journal entry requires idempotency_key",
                    code="tool_idempotency_key_required",
                )
            if not isinstance(self.idempotency_support, ToolIdempotency):
                try:
                    self.idempotency_support = ToolIdempotency(self.idempotency_support)
                except (TypeError, ValueError) as exc:
                    raise CheckpointError(
                        "tool idempotency support is invalid",
                        code="operation_kind_fields_invalid",
                    ) from exc
            if self.response is not None:
                raise CheckpointError(
                    "tool journal entry cannot contain a model response",
                    code="operation_kind_fields_invalid",
                )
        if self.state is OperationState.SUCCEEDED:
            receipt = self.response if self.kind is OperationKind.MODEL else self.result
            if receipt is None or self.error is not None:
                raise CheckpointError(
                    "succeeded operation requires exactly one success receipt",
                    code="operation_receipt_required",
                )
            canonical_json_bytes(receipt, "operation success receipt")
        elif self.state is OperationState.FAILED:
            if self.error is None or self.response is not None or self.result is not None:
                raise CheckpointError(
                    "failed operation requires exactly one typed error",
                    code="operation_error_required",
                )
        elif self.response is not None or self.result is not None or self.error is not None:
            raise CheckpointError(
                f"{self.state.value} operation cannot contain a receipt",
                code="operation_receipt_forbidden",
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "operation_id": self.operation_id,
            "cycle_index": self.cycle_index,
            "attempt": self.attempt,
            "state": self.state.value,
            "request_digest": self.request_digest,
            "idempotency_key": self.idempotency_key,
        }
        if self.kind is OperationKind.MODEL:
            payload["response"] = self.response
        else:
            idempotency_support = self.idempotency_support
            assert idempotency_support is not None
            payload.update(
                {
                    "tool_call_id": self.tool_call_id,
                    "tool_name": self.tool_name,
                    "arguments": self.arguments,
                    "idempotency_support": idempotency_support.value,
                    "result": self.result,
                }
            )
        payload["error"] = self.error.to_dict() if self.error is not None else None
        return payload

    def verify_request(self, request: dict[str, Any]) -> None:
        if compute_operation_request_digest(request) != self.request_digest:
            raise CheckpointError(
                "operation request digest does not match the durable journal",
                code="checkpoint_journal_integrity_mismatch",
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperationJournalEntry:
        if not isinstance(payload, dict):
            raise CheckpointError(
                "operation journal entry must be an object",
                code="operation_entry_invalid",
            )
        try:
            kind = OperationKind(payload.get("kind"))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "operation kind is invalid",
                code="operation_kind_fields_invalid",
            ) from exc
        try:
            state = OperationState(_required_string(payload, "state"))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "operation state is invalid",
                code="operation_state_invalid",
            ) from exc
        model_fields = {
            "kind",
            "operation_id",
            "cycle_index",
            "attempt",
            "request_digest",
            "idempotency_key",
            "state",
            "response",
            "error",
        }
        tool_fields = {
            "kind",
            "operation_id",
            "cycle_index",
            "attempt",
            "request_digest",
            "tool_call_id",
            "tool_name",
            "arguments",
            "idempotency_key",
            "idempotency_support",
            "state",
            "result",
            "error",
        }
        if set(payload) != (model_fields if kind is OperationKind.MODEL else tool_fields):
            raise CheckpointError(
                "operation journal fields do not match operation kind",
                code="operation_kind_fields_invalid",
            )
        error_raw = payload.get("error")
        return cls(
            kind=kind,
            operation_id=_required_string(payload, "operation_id"),
            cycle_index=_required_integer(payload, "cycle_index"),
            attempt=_required_integer(payload, "attempt"),
            state=state,
            request_digest=_required_string(payload, "request_digest"),
            idempotency_key=payload.get("idempotency_key"),
            response=payload.get("response"),
            result=payload.get("result"),
            error=OperationError.from_dict(error_raw) if error_raw is not None else None,
            tool_call_id=payload.get("tool_call_id"),
            tool_name=payload.get("tool_name"),
            arguments=payload.get("arguments"),
            idempotency_support=(payload.get("idempotency_support") if kind is OperationKind.TOOL else None),
        )


@dataclass(frozen=True, slots=True)
class ExtensionStateEntry:
    version: str
    required: bool
    state: Any

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("checkpoint extension version must be non-empty")
        if not isinstance(self.required, bool):
            raise TypeError("checkpoint extension required must be a boolean")
        canonical_json_bytes(self.state, "checkpoint extension state")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "required": self.required, "state": self.state}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtensionStateEntry:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint extension state entry must be an object")
        if set(payload) != {"version", "required", "state"}:
            raise ValueError("checkpoint extension state entry has missing or unknown fields")
        return cls(
            version=_required_string(payload, "version"),
            required=_required_boolean(payload, "required"),
            state=payload.get("state"),
        )


@dataclass(slots=True)
class EventOutboxEntry:
    event_id: str
    payload_digest: str
    state: str
    event: dict[str, Any]
    cursor: EventCursor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("outbox event_id must be non-empty")
        validate_sha256(self.payload_digest, "outbox payload_digest")
        if self.state not in {"pending", "delivered"}:
            raise ValueError("outbox state must be pending or delivered")
        if not isinstance(self.event, dict):
            raise ValueError("outbox event must be an object")
        from vv_agent.events import event_from_dict

        try:
            event = event_from_dict(self.event)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "outbox event must match the current RunEvent wire contract",
                code="checkpoint_event_invalid",
            ) from exc
        if event.event_id != self.event_id:
            raise CheckpointError(
                "outbox event_id must match the embedded RunEvent event_id",
                code="event_identity_conflict",
            )
        if event.to_dict() != self.event:
            raise CheckpointError(
                "outbox event must use the canonical current RunEvent shape",
                code="checkpoint_event_invalid",
            )
        canonical_json_bytes(self.event, "outbox event")
        if self.state == "pending" and self.cursor is not None:
            raise ValueError("pending outbox event cannot have a cursor")
        if self.state == "delivered" and self.cursor is None:
            raise ValueError("delivered outbox event requires a cursor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "payload_digest": self.payload_digest,
            "state": self.state,
            "event": dict(self.event),
            "cursor": self.cursor.to_dict() if self.cursor is not None else None,
        }

    def verify_payload(self) -> None:
        if compute_event_payload_digest(self.event) != self.payload_digest:
            raise CheckpointError(
                "event payload digest does not match the durable outbox entry",
                code="event_identity_conflict",
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventOutboxEntry:
        if not isinstance(payload, dict):
            raise ValueError("event outbox entry must be an object")
        if set(payload) != {"event_id", "payload_digest", "state", "event", "cursor"}:
            raise ValueError("event outbox entry has missing or unknown fields")
        cursor_raw = payload.get("cursor")
        return cls(
            event_id=_required_string(payload, "event_id"),
            payload_digest=_required_string(payload, "payload_digest"),
            state=_required_string(payload, "state"),
            event=_required_object(payload, "event"),
            cursor=EventCursor.from_dict(cursor_raw) if cursor_raw is not None else None,
        )

    @classmethod
    def pending(cls, event_id: str, event: dict[str, Any]) -> EventOutboxEntry:
        return cls(
            event_id=event_id,
            payload_digest=compute_event_payload_digest(event),
            state="pending",
            event=event,
        )


@dataclass(slots=True)
class Checkpoint:
    checkpoint_key: str
    task_id: str
    root_run_id: str
    trace_id: str
    run_definition: dict[str, Any]
    run_definition_digest: str
    resume_attempt: int
    cycle_index: int
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    shared_state: dict[str, Any] = field(default_factory=dict)
    budget_usage: BudgetUsageSnapshot | None = None
    event_cursor: EventCursor | None = None
    event_outbox: list[EventOutboxEntry] = field(default_factory=list)
    extension_state: dict[str, ExtensionStateEntry] = field(default_factory=dict)
    model_call_journal: list[OperationJournalEntry] = field(default_factory=list)
    tool_journal: list[OperationJournalEntry] = field(default_factory=list)
    revision: int = 0
    claim_token: str | None = None
    claimed_cycle: int | None = None
    lease_expires_at_ms: int | None = None
    terminal_result: AgentResult | None = None
    terminal_acknowledged: bool = False
    schema_version: str = CHECKPOINT_SCHEMA
    run_definition_schema: str = RUN_DEFINITION_SCHEMA


@runtime_checkable
class CheckpointStore(Protocol):
    def create_checkpoint(self, checkpoint: Checkpoint) -> bool: ...

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None: ...

    def claim_checkpoint(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> Checkpoint | None: ...

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def finalize_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int,
    ) -> bool: ...

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool: ...

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool: ...

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool: ...

    def delete_checkpoint(self, checkpoint_key: str) -> None: ...


def validate_checkpoint(checkpoint: Checkpoint) -> None:
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            "unsupported checkpoint v2 schema_version",
            code="checkpoint_schema_unsupported",
        )
    if checkpoint.run_definition_schema != RUN_DEFINITION_SCHEMA:
        raise CheckpointError(
            "unsupported checkpoint run_definition_schema",
            code="checkpoint_definition_schema_unsupported",
        )
    for value, name in (
        (checkpoint.checkpoint_key, "checkpoint_key"),
        (checkpoint.task_id, "task_id"),
        (checkpoint.root_run_id, "root_run_id"),
        (checkpoint.trace_id, "trace_id"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CheckpointError(
                f"checkpoint {name} must be non-empty",
                code=("checkpoint_key_invalid" if name == "checkpoint_key" else "checkpoint_identity_invalid"),
            )
    if len(checkpoint.checkpoint_key.encode("utf-8")) > 512:
        raise CheckpointError(
            "checkpoint_key must be at most 512 UTF-8 bytes",
            code="checkpoint_key_invalid",
        )
    expected_definition_digest = compute_run_definition_digest(checkpoint.run_definition)
    try:
        validate_sha256(checkpoint.run_definition_digest, "run_definition_digest")
    except ValueError as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_definition_digest_invalid",
        ) from exc
    if checkpoint.run_definition_digest != expected_definition_digest:
        raise CheckpointError(
            "run_definition_digest does not match the embedded run_definition",
            code="checkpoint_definition_mismatch",
        )
    try:
        _positive_wire_integer(checkpoint.resume_attempt, "resume_attempt")
    except ValueError as exc:
        raise CheckpointError(str(exc), code="checkpoint_resume_attempt_invalid") from exc
    try:
        _wire_integer(checkpoint.cycle_index, "cycle_index")
        _wire_integer(checkpoint.revision, "revision")
    except ValueError as exc:
        raise CheckpointError(str(exc), code="checkpoint_revision_invalid") from exc
    if not isinstance(checkpoint.status, AgentStatus):
        raise TypeError("checkpoint status must be an AgentStatus")
    if not isinstance(checkpoint.messages, list) or not all(isinstance(item, Message) for item in checkpoint.messages):
        raise TypeError("checkpoint messages must contain Message values")
    if not isinstance(checkpoint.cycles, list) or not all(isinstance(item, CycleRecord) for item in checkpoint.cycles):
        raise TypeError("checkpoint cycles must contain CycleRecord values")
    canonical_json_bytes(checkpoint.shared_state, "checkpoint shared_state")
    if checkpoint.budget_usage is not None and not isinstance(
        checkpoint.budget_usage,
        BudgetUsageSnapshot,
    ):
        raise TypeError("checkpoint budget_usage must be a BudgetUsageSnapshot or None")
    if checkpoint.event_cursor is not None and not isinstance(checkpoint.event_cursor, EventCursor):
        raise TypeError("checkpoint event_cursor must be an EventCursor or None")
    if not isinstance(checkpoint.event_outbox, list) or not all(
        isinstance(item, EventOutboxEntry) for item in checkpoint.event_outbox
    ):
        raise TypeError("checkpoint event_outbox must contain EventOutboxEntry values")
    event_ids: set[str] = set()
    for entry in checkpoint.event_outbox:
        entry.verify_payload()
        if entry.event_id in event_ids:
            raise CheckpointError(
                f"checkpoint event_outbox contains duplicate event id {entry.event_id!r}",
                code="event_identity_conflict",
            )
        event_ids.add(entry.event_id)
    if not isinstance(checkpoint.extension_state, dict) or not all(
        isinstance(entry, ExtensionStateEntry) for entry in checkpoint.extension_state.values()
    ):
        raise TypeError("checkpoint extension_state must contain ExtensionStateEntry values")
    if not isinstance(checkpoint.model_call_journal, list) or not isinstance(
        checkpoint.tool_journal,
        list,
    ):
        raise TypeError("checkpoint journals must be arrays")
    claim_values = (
        checkpoint.claim_token,
        checkpoint.claimed_cycle,
        checkpoint.lease_expires_at_ms,
    )
    if any(value is None for value in claim_values) != all(value is None for value in claim_values):
        raise CheckpointError(
            "checkpoint claim fields must be all present or all null",
            code="checkpoint_claim_invalid",
        )
    if checkpoint.claim_token is not None:
        if not checkpoint.claim_token:
            raise CheckpointError(
                "checkpoint claim_token must be non-empty",
                code="checkpoint_claim_invalid",
            )
        if checkpoint.claimed_cycle != checkpoint.cycle_index + 1:
            raise CheckpointError(
                "checkpoint claimed_cycle must equal cycle_index + 1",
                code="checkpoint_claim_invalid",
            )
        try:
            _positive_wire_integer(checkpoint.lease_expires_at_ms, "lease_expires_at_ms")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="checkpoint_claim_invalid") from exc
    if checkpoint.terminal_result is not None and checkpoint.claim_token is not None:
        raise CheckpointError(
            "terminal checkpoint cannot have an active claim",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_acknowledged and checkpoint.terminal_result is None:
        raise CheckpointError(
            "terminal acknowledgement requires a terminal result",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_result is None and checkpoint.status not in {
        AgentStatus.RUNNING,
        AgentStatus.RECONCILIATION_REQUIRED,
    }:
        raise CheckpointError(
            "non-terminal checkpoint status must be running or reconciliation_required",
            code="checkpoint_status_invalid",
        )
    active_cycle = checkpoint.claimed_cycle or (checkpoint.cycle_index + 1)
    journals = [*checkpoint.model_call_journal, *checkpoint.tool_journal]
    if not all(isinstance(entry, OperationJournalEntry) for entry in journals):
        raise TypeError("checkpoint journals must contain OperationJournalEntry values")
    for entry in journals:
        if entry.cycle_index != active_cycle:
            raise CheckpointError(
                "journal cycle_index must equal active cycle",
                code="checkpoint_journal_cycle_invalid",
            )
    for entry in checkpoint.model_call_journal:
        if entry.kind is not OperationKind.MODEL:
            raise CheckpointError(
                "model_call_journal contains a non-model entry",
                code="operation_kind_fields_invalid",
            )
    for entry in checkpoint.tool_journal:
        if entry.kind is not OperationKind.TOOL:
            raise CheckpointError(
                "tool_journal contains a non-tool entry",
                code="operation_kind_fields_invalid",
            )
    ambiguous = [entry for entry in journals if entry.state is OperationState.AMBIGUOUS]
    if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED:
        if checkpoint.claim_token is not None or not ambiguous or checkpoint.terminal_result is not None:
            raise CheckpointError(
                "reconciliation_required requires ambiguity and no claim or terminal result",
                code="checkpoint_status_invalid",
            )
    elif checkpoint.status is AgentStatus.RUNNING and ambiguous and checkpoint.claim_token is None:
        raise CheckpointError(
            "running checkpoint ambiguity requires an active recovery claim",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_result is not None:
        if checkpoint.terminal_result.status is not checkpoint.status:
            raise CheckpointError(
                "checkpoint terminal status must match terminal_result status",
                code="checkpoint_status_invalid",
            )
        if checkpoint.terminal_result.checkpoint_key not in {None, checkpoint.checkpoint_key}:
            raise CheckpointError(
                "terminal_result checkpoint_key does not match checkpoint",
                code="checkpoint_status_invalid",
            )
        if journals and not _is_operator_abort_terminal(checkpoint, ambiguous):
            raise CheckpointError(
                "terminal checkpoint cannot retain active journals",
                code="checkpoint_status_invalid",
            )
    for namespace in checkpoint.extension_state:
        try:
            validate_extension_namespace(namespace)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                str(exc),
                code="checkpoint_extension_namespace_invalid",
            ) from exc


def check_claim(
    checkpoint: Checkpoint,
    cycle_index: int,
    now_ms: int,
    claim_mode: ClaimMode,
) -> None:
    validate_claim_mode(claim_mode)
    _positive_wire_integer(cycle_index, "claimed cycle_index")
    if checkpoint.terminal_result is not None:
        raise CheckpointError(
            "checkpoint is terminal",
            code="checkpoint_terminal_immutable",
        )
    if checkpoint.status not in {
        AgentStatus.RUNNING,
        AgentStatus.RECONCILIATION_REQUIRED,
    }:
        raise CheckpointError(
            "checkpoint is not resumable",
            code="checkpoint_status_invalid",
        )
    if checkpoint.cycle_index != cycle_index - 1:
        raise CheckpointError(
            "checkpoint cycle conflict",
            code="checkpoint_cycle_conflict",
        )
    if checkpoint.claim_token is not None and (checkpoint.lease_expires_at_ms or 0) > now_ms:
        raise CheckpointError(
            "checkpoint claim is active",
            code="checkpoint_claim_active",
        )


def validate_claim_mode(claim_mode: str) -> ClaimMode:
    if claim_mode not in {"continue", "recovery"}:
        raise CheckpointError(
            "checkpoint claim_mode must be continue or recovery",
            code="checkpoint_claim_mode_invalid",
        )
    return cast(ClaimMode, claim_mode)


def claim_matches(
    current: Checkpoint | None,
    snapshot: Checkpoint,
    claim_token: str,
    expected_revision: int,
) -> bool:
    return bool(
        current is not None
        and current.revision == expected_revision
        and snapshot.revision == expected_revision
        and current.claim_token == claim_token
        and current.claimed_cycle == snapshot.claimed_cycle
        and snapshot.checkpoint_key == current.checkpoint_key
        and checkpoint_definition_matches(current, snapshot)
    )


def checkpoint_definition_matches(current: Checkpoint, snapshot: Checkpoint) -> bool:
    return bool(
        current.schema_version == snapshot.schema_version
        and current.checkpoint_key == snapshot.checkpoint_key
        and current.task_id == snapshot.task_id
        and current.root_run_id == snapshot.root_run_id
        and current.trace_id == snapshot.trace_id
        and current.run_definition_schema == snapshot.run_definition_schema
        and current.run_definition_digest == snapshot.run_definition_digest
        and canonical_json_bytes(current.run_definition) == canonical_json_bytes(snapshot.run_definition)
        and current.resume_attempt == snapshot.resume_attempt
        and current.terminal_acknowledged is snapshot.terminal_acknowledged
    )


def prepare_claimed_terminal(
    current: Checkpoint,
    checkpoint: Checkpoint,
    *,
    claim_token: str,
    expected_revision: int,
) -> Checkpoint | None:
    if (
        current.revision != expected_revision
        or checkpoint.revision != expected_revision
        or current.claim_token != claim_token
        or checkpoint.claim_token != claim_token
        or current.claimed_cycle != checkpoint.claimed_cycle
        or current.terminal_result is not None
        or checkpoint.terminal_result is None
        or not checkpoint_definition_matches(current, checkpoint)
    ):
        return None
    journals = [*checkpoint.model_call_journal, *checkpoint.tool_journal]
    ambiguous = [entry for entry in journals if entry.state is OperationState.AMBIGUOUS]
    preserve_ambiguity = _is_operator_abort_terminal(checkpoint, ambiguous)
    terminal = replace(
        deepcopy(checkpoint),
        revision=expected_revision + 1,
        claim_token=None,
        claimed_cycle=None,
        lease_expires_at_ms=None,
        model_call_journal=(deepcopy(checkpoint.model_call_journal) if preserve_ambiguity else []),
        tool_journal=(deepcopy(checkpoint.tool_journal) if preserve_ambiguity else []),
    )
    validate_checkpoint(terminal)
    return terminal


def prepare_event_delivery(
    current: Checkpoint,
    *,
    event_id: str,
    payload_digest: str,
    cursor: EventCursor,
    expected_revision: int,
    claim_token: str | None,
) -> Checkpoint | None:
    if current.revision != expected_revision or current.claim_token != claim_token:
        return None
    if cursor.last_event_id != event_id:
        raise CheckpointError(
            "event cursor last_event_id does not match the delivered event",
            code="event_cursor_invalid",
        )
    matches = [entry for entry in current.event_outbox if entry.event_id == event_id]
    if len(matches) != 1:
        raise CheckpointError(
            "checkpoint outbox does not contain exactly one matching event",
            code="event_identity_conflict",
        )
    entry = matches[0]
    entry.verify_payload()
    if entry.payload_digest != payload_digest:
        raise CheckpointError(
            "event payload digest conflicts with the durable outbox entry",
            code="event_identity_conflict",
        )
    if entry.state != "pending":
        return None
    delivered = deepcopy(current)
    delivered_entry = next(item for item in delivered.event_outbox if item.event_id == event_id)
    delivered_entry.state = "delivered"
    delivered_entry.cursor = deepcopy(cursor)
    delivered.event_cursor = deepcopy(cursor)
    delivered.revision = expected_revision + 1
    validate_checkpoint(delivered)
    return delivered


def _is_operator_abort_terminal(
    checkpoint: Checkpoint,
    ambiguous: list[OperationJournalEntry],
) -> bool:
    terminal = checkpoint.terminal_result
    journals = [*checkpoint.model_call_journal, *checkpoint.tool_journal]
    observation = terminal.resume_observation if terminal is not None else None
    return bool(
        terminal is not None
        and checkpoint.status is AgentStatus.FAILED
        and terminal.error == "operator_abort_with_unknown_outcome"
        and observation is not None
        and ambiguous
        and len(ambiguous) == len(journals)
        and any(
            entry.operation_id == observation.operation_id
            and entry.kind is observation.operation_kind
            and entry.cycle_index == observation.cycle_index
            for entry in ambiguous
        )
    )


def _wire_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WIRE_INTEGER:
        raise ValueError(f"{field_name} must be between 0 and {MAX_WIRE_INTEGER}")
    return value


def _positive_wire_integer(value: Any, field_name: str) -> int:
    result = _wire_integer(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_integer(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _required_boolean(
    payload: dict[str, Any],
    field_name: str,
    *,
    default: bool | None = None,
) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _required_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value
