"""Synchronous session transactions with caller-owned connections."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from .reducer import ExecutionState

from .records import InboxItem, Record, SessionSpec


class Conflict(ValueError):
    pass


class SequenceConflict(Conflict):
    pass


class LeaseLost(Conflict):
    pass


@dataclass(frozen=True)
class Lease:
    session_id: str
    owner: str
    epoch: int
    expires_at_ms: int


@dataclass(frozen=True)
class StoredRecord:
    record: Record
    seq: int
    commit_id: str
    writer_epoch: int
    created_ms: int


@dataclass(frozen=True)
class StoredInput:
    item: InboxItem
    input_seq: int
    received_ms: int
    consumed_seq: int | None


@dataclass(frozen=True)
class ReadPage:
    head_seq: int
    inbox_seq: int
    records: tuple[StoredRecord, ...]


@dataclass(frozen=True)
class CommitReceipt:
    commit_id: str
    record_sequences: tuple[tuple[str, int], ...]
    head_seq: int
    replayed: bool = False
    records: tuple[StoredRecord, ...] = field(default=(), compare=False, repr=False)
    inbox_seq: int = field(default=0, compare=False)


@dataclass(frozen=True)
class CreateReceipt:
    session_id: str
    record_seq: int
    received_ms: int
    replayed: bool = False


@dataclass(frozen=True)
class InputReceipt:
    input_id: str
    input_seq: int
    received_ms: int
    replayed: bool = False


@dataclass(frozen=True)
class ConsumerBatch:
    session_id: str
    consumer: str
    from_seq: int
    through_seq: int
    records: tuple[StoredRecord, ...]


@dataclass(frozen=True, order=True)
class WorkCursor:
    session_id: str
    kind: Literal["drive", "project"]
    consumer: str = ""


@dataclass(frozen=True)
class WorkItem:
    session_id: str
    kind: Literal["drive", "project"]
    consumer: str = ""

    @property
    def cursor(self) -> WorkCursor:
        return WorkCursor(self.session_id, self.kind, self.consumer)


@dataclass(frozen=True)
class LeasePoll:
    lease: Lease
    db_now_ms: int
    inbox_seq: int
    controls: tuple[InboxItem, ...]


class SessionTx(Protocol):
    def create(self, spec: SessionSpec, *, consumers: tuple[str, ...]) -> CreateReceipt: ...
    def push(self, session_id: str, item: InboxItem) -> InputReceipt: ...
    def append(
        self,
        session_id: str,
        *,
        lease: Lease,
        expected_seq: int,
        commit_id: str,
        records: tuple[Record, ...],
        consume_input_ids: tuple[str, ...] = (),
        expected_inbox_seq: int | None = None,
    ) -> CommitReceipt: ...
    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None: ...
    def ack(self, batch: ConsumerBatch) -> None: ...


class SessionStore(Protocol):
    def read_state(self, session_id: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]: ...
    def list_sessions(self, *, limit: int = 100, after: str | None = None) -> tuple[str, ...]: ...
    def atomic(self) -> AbstractContextManager[SessionTx]: ...
    def read(self, session_id: str, *, after_seq: int = 0, through_seq: int | None = None, limit: int = 1024) -> ReadPage: ...
    def peek_inbox(
        self, session_id: str, *, through_input_seq: int | None = None, limit: int = 256
    ) -> tuple[StoredInput, ...]: ...
    def acquire(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None: ...
    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll: ...
    def release(self, lease: Lease) -> bool: ...
    def next_drive_delay_ms(self, session_id: str, *, lease: Lease | None = None) -> int | None: ...
    def defer_idle_drive(self, lease: Lease, *, poll_ms: int) -> bool: ...
    def defer_drive(self, lease: Lease, *, retry_after_ms: int) -> None: ...
    def defer_projection(self, session_id: str, consumer: str, *, retry_after_ms: int) -> None: ...
    def is_runnable(self, session_id: str) -> bool: ...
    def list_runnable(self, *, limit: int = 100, after: WorkCursor | None = None) -> tuple[WorkItem, ...]: ...


class LeaseRetryExhausted(LeaseLost):
    """The host could not restore its execution lease within the retry limit."""

    code = "lease_retry_exhausted"

    def __init__(self, session_id: str, turn_id: str, attempts: int) -> None:
        self.session_id, self.turn_id, self.attempts = session_id, turn_id, attempts
        super().__init__(f"Execution lease recovery exhausted after {attempts} attempts for {session_id}/{turn_id}")
