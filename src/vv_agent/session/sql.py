"""Shared SQL transaction rules for the PostgreSQL reference and single-host SQLite store."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from vv_agent.canonical_json import canonical_json_bytes

from .records import InboxItem, Record, SessionSpec
from .reducer import ExecutionState, Fold, fold
from .store import (
    CommitReceipt,
    Conflict,
    ConsumerBatch,
    CreateReceipt,
    InputReceipt,
    Lease,
    LeaseLost,
    LeasePoll,
    ReadPage,
    SequenceConflict,
    StoredInput,
    StoredRecord,
    WorkCursor,
    WorkItem,
)


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _natural(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _identity(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("identity must be nonempty text without NUL")


@dataclass
class _Prefix:
    session_id: str
    epoch: int
    head_digest: str
    reducer: Fold
    records: tuple[StoredRecord, ...]


class SQLStore:
    """One connection per thread. atomic() nests as a savepoint in host transactions."""

    lock_clause = ""
    read_lock_clause = ""
    clock_sql = ""
    _fold_cache: _Prefix | None = None

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        raise NotImplementedError

    def _transaction(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def _transaction_id(self) -> object:
        raise NotImplementedError

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...]:
        rows = self._rows(query, params)
        if not rows:
            raise Conflict("session or cursor not found")
        return rows[0]

    def _now(self) -> int:
        return self._one(f"SELECT {self.clock_sql}")[0]

    def list_sessions(self, *, limit: int = 100, after: str | None = None) -> tuple[str, ...]:
        """Read-only catalog, including idle sessions absent from list_runnable."""
        _positive(limit, "limit")
        if after is not None:
            _identity(after)
        return tuple(
            row[0]
            for row in self._rows(
                "SELECT session_id FROM sk_session WHERE (%s OR session_id > %s) ORDER BY session_id LIMIT %s",
                (after is None, after or "", limit),
            )
        )

    @contextmanager
    def atomic(self) -> Iterator[SQLSessionTx]:
        with self._transaction():
            yield SQLSessionTx(self)

    def _lock(self, session_id: str) -> tuple[Any, ...]:
        return self._one(
            "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms "
            f"FROM sk_session WHERE session_id=%s{self.lock_clause}",
            (session_id,),
        )

    def _check_lease(self, session_id: str, lease: Lease, row: tuple[Any, ...], now: int) -> None:
        if (
            lease.session_id != session_id
            or row[2] != lease.epoch
            or row[3] != lease.owner
            or row[4] is None
            or row[4] <= now
            or lease.expires_at_ms <= now
        ):
            raise LeaseLost("lease expired or fenced")

    @staticmethod
    def _stored(row: tuple[Any, ...]) -> StoredRecord:
        body, seq, commit, epoch, created, checksum = row
        if sha256(body).hexdigest() != checksum:
            raise Conflict("stored record digest mismatch")
        return StoredRecord(Record.parse(body), seq, commit, epoch, created)

    def _prefix(self, session_id: str, row: tuple[Any, ...]) -> _Prefix:
        head, epoch = row[0], row[2]
        cached = self._fold_cache
        if cached is not None:
            valid = cached.session_id == session_id and cached.epoch == epoch and len(cached.records) <= head
            if valid:
                body, checksum = self._one(
                    "SELECT body,digest FROM sk_record WHERE session_id=%s AND seq=%s",
                    (session_id, len(cached.records)),
                )
                valid = checksum == cached.head_digest and sha256(body).hexdigest() == checksum
            if not valid:
                self._fold_cache = cached = None
        through = len(cached.records) if cached else 0
        if cached is not None and through == head:
            return cached
        rows = self._rows(
            "SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record "
            "WHERE session_id=%s AND seq>%s AND seq<=%s ORDER BY seq",
            (session_id, through, head),
        )
        records = tuple(self._stored(r) for r in rows)
        if tuple(r.seq for r in records) != tuple(range(through + 1, head + 1)):
            raise Conflict("log sequence gap")
        consumed = []
        for body, checksum in self._rows(
            "SELECT body,digest FROM sk_inbox WHERE session_id=%s AND consumed_seq>%s AND consumed_seq<=%s ORDER BY input_seq",
            (session_id, through, head),
        ):
            if sha256(body).hexdigest() != checksum:
                raise Conflict("stored input digest mismatch")
            consumed.append(InboxItem.parse(body))
        reducer = cached.reducer.fork() if cached else Fold()
        reducer.extend((r.record for r in records), consumed_inputs=consumed, bodies=(r[0] for r in rows))
        prefix = _Prefix(session_id, epoch, rows[-1][5], reducer, (cached.records if cached else ()) + records)
        self._fold_cache = prefix
        return prefix

    def read_state(self, session_id: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
        # Preserve the public read boundary used by host snapshot/reconnect hooks.
        self.read(session_id, limit=1)
        with self._transaction():
            row = self._one(
                "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms "
                f"FROM sk_session WHERE session_id=%s{self.read_lock_clause}",
                (session_id,),
            )
            prefix = self._prefix(session_id, row)
            # Validated log payloads are JSON trees. Clone them in C while preserving the
            # shared Record references between the detached state and detached history.
            memo = {id(r.record.payload): json.loads(json.dumps(r.record.payload)) for r in prefix.records}
            state, records = deepcopy((prefix.reducer.state, prefix.records), memo)
            return state, records, row[1]

    def _schedule(self, session_id: str, state: ExecutionState) -> None:
        self._rows(
            "UPDATE sk_session SET phase=%s,next_drive_ms=%s,active_turn_id=%s,terminal_seq=%s WHERE session_id=%s",
            (state.phase, state.next_drive_ms, state.active_turn_id, state.terminal_seq, session_id),
        )

    def rebuild_schedule(self, session_id: str) -> ExecutionState:
        with self._transaction():
            row = self._lock(session_id)
            self._fold_cache = None
            state = self._prefix(session_id, row).reducer.state
            self._schedule(session_id, state)
            return deepcopy(state)

    def read(self, session_id: str, *, after_seq: int = 0, through_seq: int | None = None, limit: int = 1024) -> ReadPage:
        _positive(limit, "limit")
        _natural(after_seq, "after_seq")
        if through_seq is not None:
            _natural(through_seq, "through_seq")
        with self._transaction():
            head, inbox = self._one(
                f"SELECT head_seq,inbox_seq FROM sk_session WHERE session_id=%s{self.read_lock_clause}", (session_id,)
            )
            rows = self._rows(
                "SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record "
                "WHERE session_id=%s AND seq>%s AND seq<=%s ORDER BY seq LIMIT %s",
                (session_id, after_seq, min(head, through_seq) if through_seq is not None else head, limit),
            )
            return ReadPage(head, inbox, tuple(self._stored(row) for row in rows))

    def peek_inbox(self, session_id: str, *, through_input_seq: int | None = None, limit: int = 256) -> tuple[StoredInput, ...]:
        _positive(limit, "limit")
        if through_input_seq is not None:
            _natural(through_input_seq, "through_input_seq")
        rows = self._rows(
            f"SELECT body,input_seq,received_ms,consumed_seq,digest FROM sk_inbox WHERE session_id=%s "
            f"AND consumed_seq IS NULL AND available_ms<={self.clock_sql} "
            "AND (%s OR input_seq<=%s) ORDER BY input_seq LIMIT %s",
            (session_id, through_input_seq is None, through_input_seq or 0, limit),
        )
        result = []
        for body, seq, received, consumed, checksum in rows:
            if sha256(body).hexdigest() != checksum:
                raise Conflict("stored input digest mismatch")
            result.append(StoredInput(InboxItem.parse(body), seq, received, consumed))
        return tuple(result)

    def acquire(self, session_id: str, *, owner: str, ttl_ms: int) -> Lease | None:
        _identity(owner)
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(session_id)
            now = self._now()  # Must be read after the row lock, not at transaction start.
            if row[4] is not None and row[4] > now:
                return None
            lease = Lease(session_id, owner, row[2] + 1, now + ttl_ms)
            self._rows(
                "UPDATE sk_session SET lease_epoch=%s,lease_owner=%s,lease_until_ms=%s WHERE session_id=%s",
                (lease.epoch, owner, lease.expires_at_ms, session_id),
            )
            return lease

    def renew(self, lease: Lease, *, ttl_ms: int) -> LeasePoll:
        _positive(ttl_ms, "ttl_ms")
        with self._transaction():
            row = self._lock(lease.session_id)
            now = self._now()
            self._check_lease(lease.session_id, lease, row, now)
            renewed = Lease(lease.session_id, lease.owner, lease.epoch, now + ttl_ms)
            self._rows("UPDATE sk_session SET lease_until_ms=%s WHERE session_id=%s", (renewed.expires_at_ms, lease.session_id))
            controls = tuple(
                InboxItem.parse(r[0])
                for r in self._rows(
                    "SELECT body FROM sk_inbox WHERE session_id=%s AND consumed_seq IS NULL AND available_ms<=%s "
                    "AND kind='control' ORDER BY input_seq",
                    (lease.session_id, now),
                )
            )
            return LeasePoll(renewed, now, row[1], controls)

    def release(self, lease: Lease) -> bool:
        with self._transaction():
            row = self._lock(lease.session_id)
            try:
                self._check_lease(lease.session_id, lease, row, self._now())
            except LeaseLost:
                return False
            self._rows("UPDATE sk_session SET lease_owner=NULL,lease_until_ms=NULL WHERE session_id=%s", (lease.session_id,))
            return True

    def list_runnable(self, *, limit: int = 100, after: WorkCursor | None = None) -> tuple[WorkItem, ...]:
        _positive(limit, "limit")
        cursor = (after.session_id, after.kind, after.consumer) if after else ("", "", "")
        rows = self._rows(
            f"""
            WITH clock AS MATERIALIZED (SELECT {self.clock_sql} AS ms), driving AS (
                SELECT s.session_id,'drive' AS work_kind,'' AS consumer
                FROM sk_session s CROSS JOIN clock t
                WHERE s.session_id >= %s AND (s.session_id,'drive','')>(%s,%s,%s)
                  AND (s.lease_until_ms IS NULL OR s.lease_until_ms<=t.ms)
                  AND (s.next_drive_ms<=t.ms OR EXISTS (
                      SELECT 1 FROM sk_inbox i WHERE i.session_id=s.session_id
                      AND i.consumed_seq IS NULL AND i.available_ms<=t.ms))
                ORDER BY s.session_id LIMIT %s
            ), projecting AS (
                SELECT c.session_id,'project' AS work_kind,c.consumer FROM sk_consumer c
                JOIN sk_session s USING(session_id)
                WHERE c.session_id >= %s AND (c.session_id,'project',c.consumer)>(%s,%s,%s)
                  AND c.last_seq<s.head_seq
                ORDER BY c.session_id,c.consumer LIMIT %s
            ), work AS (
                SELECT * FROM driving UNION ALL SELECT * FROM projecting
            ) SELECT session_id,work_kind,consumer FROM work
            ORDER BY session_id,work_kind,consumer LIMIT %s
            """,
            (cursor[0], *cursor, limit, cursor[0], *cursor, limit, limit),
        )
        return tuple(WorkItem(*row) for row in rows)


class SQLSessionTx:
    def __init__(self, store: SQLStore):
        self.store = store
        self.transaction_id = store._transaction_id()
        self.batches: dict[tuple[str, str], ConsumerBatch] = {}

    def _active(self) -> None:
        if self.store._transaction_id() != self.transaction_id:
            raise RuntimeError("SessionTx must be used within its original caller transaction")

    def _commit(
        self,
        session_id: str,
        commit_id: str,
        body: bytes,
        manifest: bytes,
        sequences: tuple[tuple[str, int], ...],
        head: int,
        now: int,
    ) -> None:
        self.store._rows(
            "INSERT INTO sk_commit VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (session_id, commit_id, manifest, sha256(body).hexdigest(), canonical_json_bytes(sequences), head, now),
        )

    def _write(
        self,
        record: Record,
        *,
        seq: int,
        commit_id: str,
        commit_digest: str,
        position: int,
        epoch: int,
        now: int,
        body: bytes | None = None,
    ) -> None:
        body = record.encode() if body is None else body
        self.store._rows(
            "INSERT INTO sk_record VALUES (%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                record.session_id,
                seq,
                record.record_id,
                record.kind,
                record.turn_id,
                record.operation_id,
                record.attempt,
                body,
                sha256(body).hexdigest(),
                commit_id,
                commit_digest,
                position,
                epoch,
                now,
            ),
        )

    def create(self, spec: SessionSpec, *, consumers: tuple[str, ...]) -> CreateReceipt:
        self._active()
        record = spec.record()
        for consumer in consumers:
            _identity(consumer)
        if len(consumers) != len(set(consumers)):
            raise Conflict("duplicate consumer")
        body = canonical_json_bytes({"create": record.to_dict(), "consumers": sorted(consumers)})
        manifest = canonical_json_bytes({"create": record.record_id, "consumers": sorted(consumers)})
        commit_id = "session/create"
        s = self.store
        with s._transaction():
            inserted = s._rows(
                "INSERT INTO sk_session(session_id,schema_version,parent_session_id,parent_operation_id,phase) "
                "VALUES (%s,1,%s,%s,'idle') ON CONFLICT(session_id) DO NOTHING RETURNING session_id",
                (spec.session_id, spec.parent_session_id, spec.parent_operation_id),
            )
            s._lock(spec.session_id)
            if not inserted:
                old = s._one(
                    "SELECT body,created_ms FROM sk_commit WHERE session_id=%s AND commit_id=%s", (spec.session_id, commit_id)
                )
                stored = s._one("SELECT body FROM sk_record WHERE session_id=%s AND seq=1", (spec.session_id,))[0]
                if old[0] != manifest or stored != record.encode():
                    raise Conflict("create identity has different bytes")
                return CreateReceipt(spec.session_id, 1, old[1], True)
            now = s._now()
            state = fold((record,))
            self._write(record, seq=1, commit_id=commit_id, commit_digest=sha256(body).hexdigest(), position=0, epoch=0, now=now)
            self._commit(spec.session_id, commit_id, body, manifest, ((record.record_id, 1),), 1, now)
            for consumer in consumers:
                s._rows("INSERT INTO sk_consumer(session_id,consumer) VALUES (%s,%s)", (spec.session_id, consumer))
            s._rows("UPDATE sk_session SET head_seq=1 WHERE session_id=%s", (spec.session_id,))
            s._schedule(spec.session_id, state)
            return CreateReceipt(spec.session_id, 1, now)

    def push(self, session_id: str, item: InboxItem) -> InputReceipt:
        self._active()
        body = item.encode()
        s = self.store
        with s._transaction():
            row = s._lock(session_id)
            old = s._rows(
                "SELECT body,input_seq,received_ms FROM sk_inbox WHERE session_id=%s AND input_id=%s", (session_id, item.input_id)
            )
            if old:
                if old[0][0] != body:
                    raise Conflict("input identity has different bytes")
                return InputReceipt(item.input_id, old[0][1], old[0][2], True)
            now, seq = s._now(), row[1] + 1
            s._rows(
                "INSERT INTO sk_inbox VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL)",
                (session_id, item.input_id, seq, item.kind, body, sha256(body).hexdigest(), item.available_ms, now),
            )
            s._rows("UPDATE sk_session SET inbox_seq=%s WHERE session_id=%s", (seq, session_id))
            return InputReceipt(item.input_id, seq, now)

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
    ) -> CommitReceipt:
        self._active()
        _identity(commit_id)
        _natural(expected_seq, "expected_seq")
        if expected_inbox_seq is not None:
            _natural(expected_inbox_seq, "expected_inbox_seq")
        if len(set(consume_input_ids)) != len(consume_input_ids):
            raise Conflict("duplicate consume_input_ids")
        encoded = []
        for record in records:
            encoded.append(record.encode())
            if record.session_id != session_id:
                raise Conflict("record belongs to another session")
        if len({r.record_id for r in records}) != len(records):
            raise Conflict("duplicate record_id in batch")
        # Cache the same JSON values that a cold reader sees, detached from caller-owned payloads.
        records = tuple(Record(**json.loads(body)) for body in encoded)
        # CAS/lease are admission facts, not logical commit bytes: replay cannot renew authority.
        body = b'{"consume_input_ids":' + canonical_json_bytes(consume_input_ids) + b',"records":[' + b",".join(encoded) + b"]}"
        manifest = canonical_json_bytes(
            {"record_ids": [r.record_id for r in records], "consume_input_ids": list(consume_input_ids)}
        )
        s = self.store
        with s._transaction():
            row = s._lock(session_id)
            old = s._rows(
                "SELECT body,record_sequences,head_seq FROM sk_commit WHERE session_id=%s AND commit_id=%s",
                (session_id, commit_id),
            )
            if old:
                stored = {
                    r.record_id: s._one(
                        "SELECT body FROM sk_record WHERE session_id=%s AND record_id=%s", (session_id, r.record_id)
                    )[0]
                    for r in records
                }
                if old[0][0] != manifest or any(stored.get(r.record_id) != r.encode() for r in records):
                    raise Conflict("commit identity has different bytes")
                return CommitReceipt(commit_id, tuple((rid, seq) for rid, seq in json.loads(old[0][1])), old[0][2], True)
            s._check_lease(session_id, lease, row, s._now())
            if expected_seq != row[0] or (expected_inbox_seq is not None and expected_inbox_seq != row[1]):
                raise SequenceConflict("head or inbox watermark changed")
            prefix = s._prefix(session_id, row)
            sequences: list[tuple[str, int]] = []
            new: list[tuple[int, Record, int]] = []
            head = row[0]
            for position, record in enumerate(records):
                prior = s._rows(
                    "SELECT body,seq FROM sk_record WHERE session_id=%s AND record_id=%s", (session_id, record.record_id)
                )
                if prior:
                    if prior[0][0] != record.encode():
                        raise Conflict("record identity has different bytes")
                    sequences.append((record.record_id, prior[0][1]))
                else:
                    head += 1
                    sequences.append((record.record_id, head))
                    new.append((position, record, head))
            applications = {r.payload["input"]["input_id"]: (r, seq) for _, r, seq in new if r.kind == "input_applied"}
            if set(applications) != set(consume_input_ids):
                raise Conflict("new input_applied and consume_input_ids must match exactly")
            consumed = []
            now = s._now()
            for input_id in consume_input_ids:
                incoming = s._one(
                    "SELECT body,consumed_seq,available_ms FROM sk_inbox WHERE session_id=%s AND input_id=%s",
                    (session_id, input_id),
                )
                applied = applications[input_id][0]
                if incoming[1] is not None or incoming[2] > now or incoming[0] != canonical_json_bytes(applied.payload["input"]):
                    raise Conflict("input already consumed, unavailable, or different bytes")
                consumed.append(InboxItem.parse(incoming[0]))
            reducer = prefix.reducer.fork()
            state = reducer.extend((r for _, r, _ in new), consumed_inputs=consumed, bodies=(encoded[p] for p, _, _ in new))
            now = s._now()
            s._check_lease(session_id, lease, row, now)  # Validation may outlive the lease.
            commit_digest = sha256(body).hexdigest()
            for position, record, seq in new:
                if record.kind == "op_started":
                    if record.payload["epoch"] != lease.epoch:
                        raise Conflict("dispatch epoch does not match writer")
                    assert record.operation_id is not None and record.attempt is not None
                    op = state.operations[record.operation_id]
                    planned = op.attempts[record.attempt].plan
                    if (planned.payload["not_before_ms"] or 0) > now:
                        raise Conflict("dispatch before not-before")
                self._write(
                    record,
                    seq=seq,
                    commit_id=commit_id,
                    commit_digest=commit_digest,
                    position=position,
                    epoch=lease.epoch,
                    now=now,
                    body=encoded[position],
                )
            for input_id, (_, seq) in applications.items():
                s._rows("UPDATE sk_inbox SET consumed_seq=%s WHERE session_id=%s AND input_id=%s", (seq, session_id, input_id))
            s._rows("UPDATE sk_session SET head_seq=%s WHERE session_id=%s", (head, session_id))
            s._schedule(session_id, state)
            receipt = CommitReceipt(commit_id, tuple(sequences), head)
            self._commit(session_id, commit_id, body, manifest, receipt.record_sequences, head, now)
            added = tuple(StoredRecord(r, seq, commit_id, lease.epoch, now) for _, r, seq in new)
            checksum = sha256(encoded[new[-1][0]]).hexdigest() if new else prefix.head_digest
            s._fold_cache = _Prefix(session_id, lease.epoch, checksum, reducer, prefix.records + added)
            return receipt

    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None:
        self._active()
        _positive(limit, "limit")
        s = self.store
        last = s._one(
            f"SELECT last_seq FROM sk_consumer WHERE session_id=%s AND consumer=%s{s.lock_clause}", (session_id, consumer)
        )[0]
        # One statement snapshot. Include all remaining rows of the commit touching the soft limit.
        rows = s._rows(
            """
            WITH page AS MATERIALIZED (
                SELECT seq,commit_id FROM sk_record WHERE session_id=%s AND seq>%s ORDER BY seq LIMIT %s
            ), boundary AS (
                SELECT max(r.seq) AS seq FROM sk_record r WHERE r.session_id=%s
                AND r.commit_id=(SELECT commit_id FROM page ORDER BY seq DESC LIMIT 1)
            )
            SELECT body,seq,commit_id,writer_epoch,created_ms,digest FROM sk_record
            WHERE session_id=%s AND seq>%s AND seq<=(SELECT seq FROM boundary) ORDER BY seq
            """,
            (session_id, last, limit, session_id, session_id, last),
        )
        key = (session_id, consumer)
        if not rows:
            self.batches.pop(key, None)
            return None
        records = tuple(s._stored(row) for row in rows)
        batch = ConsumerBatch(session_id, consumer, last + 1, records[-1].seq, records)
        self.batches[key] = batch
        return batch

    def ack(self, batch: ConsumerBatch) -> None:
        self._active()
        key = (batch.session_id, batch.consumer)
        if self.batches.get(key) is not batch:
            raise Conflict("ack requires the batch issued by this transaction")
        updated = self.store._rows(
            "UPDATE sk_consumer SET last_seq=%s WHERE session_id=%s AND consumer=%s AND last_seq=%s RETURNING last_seq",
            (batch.through_seq, *key, batch.from_seq - 1),
        )
        if not updated:
            raise Conflict("consumer cursor changed")
        del self.batches[key]
