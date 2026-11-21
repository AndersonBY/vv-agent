"""Synchronous session transactions with caller-owned connections."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vv_agent.kernel._state import ExecutionState

from vv_agent.kernel._content import ContentSessionWriter, SessionHead
from vv_agent.kernel._records import InboxItem, Record, SessionSpec

DEFAULT_TENANT_ID = "default"
SESSION_DELETION_INVENTORY = (
    "sk_session",
    "sk_record",
    "sk_inbox",
    "sk_consumer",
    "sk_commit",
    "sk_checkpoint",
    "sk_session_key",
    "sk_content",
    "sk_content_link",
    "sk_tombstone",
)
SESSION_PURGE_DELETE_ORDER = (
    "sk_content_link",
    "sk_content",
    "sk_inbox",
    "sk_checkpoint",
    "sk_consumer",
    "sk_commit",
    "sk_record",
    "sk_session_key",
    "sk_session",
)


class Conflict(ValueError):
    pass


class SequenceConflict(Conflict):
    pass


class LeaseLost(Conflict):
    pass


@dataclass(frozen=True)
class WorkerSupports:
    """The immutable, deterministically serializable worker support declaration."""

    storage_major: Literal["S1"] = "S1"
    max_level: int = 1
    handler_versions: tuple[str, ...] = ("1",)
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.storage_major != "S1":
            raise ValueError("worker storage major must be S1")
        if type(self.max_level) is not int or self.max_level <= 0:
            raise ValueError("worker max_level must be a positive integer")
        object.__setattr__(
            self,
            "handler_versions",
            _support_ids(self.handler_versions, "worker handler_version"),
        )
        object.__setattr__(self, "capabilities", _support_ids(self.capabilities, "worker capability"))

    def accepts(
        self,
        *,
        level: int,
        handler_version: str | None,
        capabilities: Iterable[str],
    ) -> bool:
        required_capabilities = set(capabilities)
        return (
            level <= self.max_level
            and (handler_version is None or handler_version in self.handler_versions)
            and required_capabilities.issubset(self.capabilities)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation with stable sequence ordering."""

        return {
            "storage_major": self.storage_major,
            "max_level": self.max_level,
            "handler_versions": list(self.handler_versions),
            "capabilities": list(self.capabilities),
        }


def _support_ids(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} values must be an iterable of identifiers")
    normalized = set(values)
    if any(not isinstance(value, str) or not value or "\x00" in value for value in normalized):
        raise ValueError(f"{name} values must be nonempty text without NUL")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class Lease:
    session_id: str
    owner: str
    epoch: int
    expires_at_ms: int
    tenant_id: str = DEFAULT_TENANT_ID


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
    tenant_id: str = DEFAULT_TENANT_ID


@dataclass(frozen=True)
class WorkItem:
    session_id: str
    kind: Literal["drive", "project"]
    consumer: str = ""
    tenant_id: str = DEFAULT_TENANT_ID

    @property
    def cursor(self) -> WorkCursor:
        return WorkCursor(self.session_id, self.kind, self.consumer, self.tenant_id)


@dataclass(frozen=True)
class ClaimedWork:
    """A runnable work item and the drive lease atomically claimed with it."""

    work: WorkItem
    lease: Lease | None = None


ProjectionConsumers = tuple[str, ...]
WorkKind = Literal["drive", "project"]
WorkKinds = tuple[WorkKind, ...]


@dataclass(frozen=True)
class LeasePoll:
    lease: Lease
    db_now_ms: int
    inbox_seq: int
    controls: tuple[InboxItem, ...]


class SessionTx(Protocol):
    def lock_session(self, session_id: str) -> None: ...

    def bind_lease_guard(self, guard: Callable[[], Lease]) -> None: ...

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
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> CommitReceipt: ...
    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None: ...
    def ack(self, batch: ConsumerBatch) -> None: ...


@runtime_checkable
class ContentSessionTx(SessionTx, Protocol):
    """A session transaction with an explicitly bound trusted-content capability."""

    def content_writer(self, *, tenant_id: str, session_id: str) -> ContentSessionWriter: ...


class SessionStore(Protocol):
    @property
    def max_feature_level(self) -> int:
        """Return the highest session feature level this store can durably write."""
        ...

    def for_tenant(self, tenant_id: str) -> SessionStore: ...

    def heartbeat_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        """Return a repeatable factory that opens a new renewal connection per call."""
        ...

    def heartbeat_store(self) -> AbstractContextManager[SessionStore]:
        """Open a runtime-independent renewal connection; memory stores may share their lock."""
        ...

    def read_input(self, session_id: str, input_id: str) -> StoredInput | None: ...

    def projection_context(self, session_id: str, records: tuple[StoredRecord, ...]) -> tuple[StoredRecord, ...]: ...

    def control_records(self, session_id: str, *, through_seq: int) -> tuple[StoredRecord, ...]: ...

    def read_state(
        self,
        session_id: str,
        *,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]: ...
    def list_sessions(self, *, limit: int = 100, after: str | None = None) -> tuple[str, ...]: ...
    def atomic(self) -> AbstractContextManager[SessionTx]: ...
    def read(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        through_seq: int | None = None,
        limit: int = 1024,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> ReadPage: ...
    def verify_ledger(
        self,
        session_id: str,
        *,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> tuple[int, str]: ...
    def peek_inbox(
        self, session_id: str, *, through_input_seq: int | None = None, limit: int = 256, ready_only: bool = True
    ) -> tuple[StoredInput, ...]: ...
    def acquire(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None: ...
    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll: ...
    def release(self, lease: Lease) -> bool: ...
    def lease_status(self, session_id: str) -> tuple[int, int, int]:
        """Return head sequence, lease epoch and remaining lease milliseconds using database time."""
        ...

    def next_drive_delay_ms(self, session_id: str, *, lease: Lease | None = None) -> int | None: ...
    def defer_idle_drive(self, lease: Lease, *, poll_ms: int) -> bool: ...
    def defer_drive(self, lease: Lease, *, retry_after_ms: int) -> None: ...
    def defer_projection(self, session_id: str, consumer: str, *, retry_after_ms: int) -> None: ...
    def is_runnable(self, session_id: str) -> bool: ...
    def list_runnable(
        self,
        *,
        limit: int = 100,
        after: WorkCursor | None = None,
        work_kinds: WorkKinds | None = None,
        projection_consumers: ProjectionConsumers | None = None,
    ) -> tuple[WorkItem, ...]: ...
    def claim_runnable(
        self,
        *,
        owner: str,
        ttl_ms: int,
        limit: int = 100,
        after: WorkCursor | None = None,
        work_kinds: WorkKinds | None = None,
        projection_consumers: ProjectionConsumers | None = None,
    ) -> tuple[ClaimedWork, ...]: ...


class LeaseRetryExhausted(LeaseLost):
    """The host could not restore its execution lease within the retry limit."""

    code = "lease_retry_exhausted"

    def __init__(self, session_id: str, turn_id: str, attempts: int) -> None:
        self.session_id, self.turn_id, self.attempts = session_id, turn_id, attempts
        super().__init__(f"Execution lease recovery exhausted after {attempts} attempts for {session_id}/{turn_id}")
