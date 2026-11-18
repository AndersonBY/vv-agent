"""Single-thread, seeded scheduling around the production driver and SQL transactions.

SQLite uses participant-owned connections to a real file and an injected clock.
PostgreSQL keeps clock_timestamp(); expiry is an explicit SQL fault. No state
machine or storage rules are reimplemented here.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.s1_fixture_values import (
    content_ref,
    create_fixture_session,
    push_fixture_input,
    wrap_fixture_store,
)
from vv_agent.kernel._boundary_hydration import SessionHydration
from vv_agent.kernel._content_runtime import durable_content_reader
from vv_agent.kernel._execution_values import LLMResponse
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.kernel._records import InboxItem, SessionSpec, make_record
from vv_agent.kernel._sql import LedgerIntegrityError, _initial_chain_digest
from vv_agent.kernel._sqlite import DEFAULT_BUSY_TIMEOUT_MS, DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS, SQLiteStore
from vv_agent.kernel._store import LeaseLost
from vv_agent.kernel._tool_values import ToolCall

from .assembly import (
    SIM_HANDLER_VERSION,
    CoreRuntimeAssembly,
    SimContentReader,
    SimRuntimeNotReady,
    extend_hydrated_fold,
    fold_factory,
)
from .content import SimAESGCMCipher, SimContentKeyProvider

_DEFAULT_FOLD_FACTORY = fold_factory


def sqlite_root(store):
    current, seen = store, set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SQLiteStore):
            return current
        for name in ("_root", "_root_store", "_store", "store"):
            parent = getattr(current, name, None)
            if parent is not None and parent is not current:
                current = parent
                break
        else:
            return None
    return None


def sqlite_state_version(store, sid):
    root = sqlite_root(store)
    if root is None:
        return None
    head = root._one(
        "SELECT head_seq,inbox_seq FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, sid),
    )
    data_version = root.connection.execute("PRAGMA data_version").fetchone()[0]
    return data_version, root.connection.total_changes, *head


class SimSQLiteStore(SQLiteStore):
    def __init__(
        self,
        path,
        *,
        fold_factory=None,
        clock=None,
        busy_timeout_ms=DEFAULT_BUSY_TIMEOUT_MS,
        heartbeat_busy_timeout_ms=DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS,
    ):
        selected_fold_factory = fold_factory or _DEFAULT_FOLD_FACTORY
        selected_clock = clock or SimClock()
        super().__init__(
            path,
            fold_factory=selected_fold_factory,
            clock=selected_clock,
            busy_timeout_ms=busy_timeout_ms,
            heartbeat_busy_timeout_ms=heartbeat_busy_timeout_ms,
        )
        self._sim_verified_ledger_cache = None

    def _verified_records(self, session_id, through, **kwargs):
        if kwargs:
            return super()._verified_records(session_id, through, **kwargs)
        params = (self.tenant_id, session_id, through)
        record_rows = tuple(
            self._rows(
                "SELECT body,seq,commit_id,writer_epoch,created_ms,digest,chain_digest,record_id,"
                "schema_version,kind,turn_id,operation_id,attempt,commit_position FROM sk_record "
                "WHERE tenant_id=%s AND session_id=%s AND seq<=%s ORDER BY seq",
                params,
            )
        )
        checkpoint_rows = tuple(
            self._rows(
                "SELECT seq,chain_digest,key_id,signature,signed_ms FROM sk_checkpoint "
                "WHERE tenant_id=%s AND session_id=%s AND seq<=%s ORDER BY seq",
                params,
            )
        )
        commit_rows = tuple(
            self._rows(
                "SELECT commit_id,body,digest,record_sequences,head_seq,created_ms FROM sk_commit "
                "WHERE tenant_id=%s AND session_id=%s ORDER BY created_ms,commit_id",
                (self.tenant_id, session_id),
            )
        )
        signature = (through, record_rows, checkpoint_rows, commit_rows)
        cached = self._sim_verified_ledger_cache
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        stored_rows, chains = super()._verified_records(session_id, through)
        self._sim_verified_ledger_cache = (signature, stored_rows, chains)
        return stored_rows, chains


class SimPostgresStore(PostgresStore):
    def __init__(self, connection, *, fold_factory=None, runtime_binding=None):
        selected_fold_factory = fold_factory or _DEFAULT_FOLD_FACTORY
        super().__init__(connection, fold_factory=selected_fold_factory, runtime_binding=runtime_binding)


PENDING = {}


class SimLedgerSigner:
    """Deterministic Ed25519 signer using the production proof contracts."""

    def __init__(self, seed: int) -> None:
        self.key_id = f"sim-ledger-{seed}"
        self.key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"sim-ledger:{seed}".encode()).digest())

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


class InvariantViolation(AssertionError):
    def __init__(self, invariant, detail):
        self.invariant = invariant
        super().__init__(f"{invariant}: {detail}")


def require(condition, invariant, detail):
    if not condition:
        raise InvariantViolation(invariant, detail)


@dataclass(frozen=True)
class CachedHydratedState:
    signature: tuple[Any, ...]
    fold: Any
    records: tuple[Any, ...]
    inbox_seq: int
    version: tuple[Any, ...] | None


class WorkerDeath(BaseException):
    """Unwind a synchronous worker without a graceful lease release."""


@dataclass
class SimClock:
    milliseconds: int = 1_000_000

    def now_ms(self):
        return self.milliseconds

    def monotonic(self):
        return self.milliseconds / 1000

    def monotonic_ns(self):
        return self.milliseconds * 1_000_000

    def sleep(self, seconds):
        self.milliseconds += max(1, int(seconds * 1000))


class SystemClock:
    def now_ms(self):
        return time.time_ns() // 1_000_000

    def monotonic(self):
        return time.monotonic()

    def monotonic_ns(self):
        return time.monotonic_ns()

    def sleep(self, seconds):
        time.sleep(seconds)


class Choices:
    """Named decisions survive deletion during bounded delta debugging."""

    def __init__(self, seed, replay=None):
        self.random = random.Random(seed)
        self.replay = replay
        self.overrides = {d["at"]: d["value"] for d in replay or []}
        self.occurrences = Counter()
        self.used = set()
        self.trace = []

    def choose(self, label, options):
        at = f"{label}#{self.occurrences[label]}"
        self.occurrences[label] += 1
        if at in self.overrides:
            value = self.overrides[at]
            if value not in options:
                raise ValueError(f"replay decision {at} is unavailable")
            self.used.add(at)
        else:
            value = options[0] if self.replay is not None else self.random.choice(options)
        self.trace.append({"at": at, "value": value})
        return value


class FaultyStore:
    """Instrument an actual store in place, preserving SQLStore/SQLSessionTx paths.

    All transaction work is delegated. A precommit fault is raised INSIDE the
    real outer atomic block; a lost reply/death is raised AFTER its commit.
    The driver receives ``store`` itself so its SQL latch ordering is unchanged.
    """

    def __init__(self, store, world, participant):
        self.store, self.world, self.participant = store, world, participant
        self.fault = None
        self.dead = False
        self._atomic, self._release = store.atomic, store.release
        store.atomic = self.atomic
        store.release = self.release

    def __getattr__(self, name):
        return getattr(self.store, name)

    def release(self, lease):
        return False if self.dead else self._release(lease)

    def die(self, reason):
        self.dead = True
        raise WorkerDeath(reason)

    @contextmanager
    def atomic(self):
        if self.dead:
            raise WorkerDeath("dead participant used its connection")
        matched = False
        with self._atomic() as tx:
            append = tx.append

            def observe_append(*args, **kwargs):
                nonlocal matched
                if self.fault:
                    _, kind, op_kind = self.fault
                    matched = any(
                        r.kind == kind and (op_kind is None or ("/tool/" in (r.operation_id or "")) == (op_kind == "tool"))
                        for r in kwargs["records"]
                    )
                return append(*args, **kwargs)

            tx.append = observe_append
            yield tx
            fault = self.fault if matched else None
            if fault:
                self.fault = None
            if fault and fault[0] == "precommit":
                self.world.events.append({"fault": "precommit", "participant": self.participant})
                self.die("before SQL commit")
        if fault:
            self.world.events.append({"fault": fault[0], "participant": self.participant})
            if fault[0] == "reply_loss":
                raise TimeoutError("committed SQL reply lost")
            self.die("after SQL commit")


class SimWorld:
    def __init__(self, database, *, seed=0, replay=None, signed=True):
        self.database, self.seed = database, seed
        self.clock = SimClock() if isinstance(database, Path) else SystemClock()
        self.choices = Choices(seed, replay)
        self.signed = signed
        self.ledger_signer = SimLedgerSigner(seed) if signed else None
        self.stack = ExitStack()
        self.connections, self.workers, self.keepers = {}, {}, {}
        self.events, self.calls, self.admissions = [], [], []
        self.worker_claims = []
        self.model_receipts = {}
        self.tasks, self.running = {}, set()
        self.folds, self.prefixes, self.runtimes, self.snapshots = {}, {}, {}, {}
        self.verified_folds, self.verified_prefixes = {}, {}
        self._signature_cache = {}
        self._ledger_checks = {}
        self._content_signatures, self._content_bodies = {}, {}
        self.queue = []
        self.after_external = None
        self.scenario = None
        self.covered = set()
        self.memory_settings = {}
        self.postgres_runtime_ready = False
        self.content_provider = SimContentKeyProvider(seed)
        self.content_cipher = SimAESGCMCipher(seed)

    def __enter__(self):
        host = self.connection("host")
        try:
            host.install_schema()
        except BaseException:
            self.stack.close()
            raise
        if not isinstance(self.database, Path):
            self.postgres_runtime_ready = True
            self._set_postgres_runtime_timeouts(host)
        return self

    def __exit__(self, *args):
        if isinstance(args[1], Exception):
            args[1].add_note(json.dumps(self.summary()))
        return self.stack.__exit__(*args)

    def connection(self, participant):
        if participant not in self.connections:
            if isinstance(self.database, Path):
                context = SimSQLiteStore.standalone(
                    str(self.database),
                    fold_factory=fold_factory,
                    clock=self.clock,
                )
            else:
                context = SimPostgresStore.standalone(self.database, fold_factory=fold_factory)
            raw_store = self.stack.enter_context(context)
            raw_store.configure_worker(handler_versions=(SIM_HANDLER_VERSION,))
            raw_store.configure_reader(level=2)
            raw_store.configure_input_admission(
                self.ledger_signer,
                allow_unsigned=self.ledger_signer is None,
            )
            if not isinstance(raw_store, SQLiteStore) and self.postgres_runtime_ready:
                self._set_postgres_runtime_timeouts(raw_store)
            elif not isinstance(raw_store, SQLiteStore):
                raw_store._rows("SELECT set_config('lock_timeout','0',false),set_config('statement_timeout','0',false)")
            store = wrap_fixture_store(raw_store, self.content_provider, cipher=self.content_cipher)
            self.connections[participant] = store
        return self.connections[participant]

    def hydration(self, sid):
        store = self.connection("observer")
        return SessionHydration(
            SimContentReader(
                durable_content_reader(store, tenant_id=store.tenant_id, session_id=sid),
                self,
            ),
            tenant_id=store.tenant_id,
            session_id=sid,
        )

    def read_state(self, sid):
        store = self.connection("observer")
        cached = self.snapshots.get(sid)
        version = sqlite_state_version(store, sid)
        if cached is not None and version is not None and cached.version == version:
            return cached.fold.state, cached.records, cached.inbox_seq
        head = store._head(sid)
        verified, _chains = store._verified_records(sid, head[0])
        records = tuple(store._stored(sid, row) for row in verified)
        inbox_seq = head[1]
        signature = self.cached_state_signature(store, sid)
        self._content_signatures[sid] = signature
        hydration = self.hydration(sid)
        prefix = cached.records if cached is not None else ()
        if cached is not None and cached.signature[0] == signature[0][: len(prefix)]:
            fold, _hydrated = extend_hydrated_fold(cached.fold, records[len(prefix) :], hydration, raw_applied_inputs=False)
        else:
            fold, _hydrated = extend_hydrated_fold(None, records, hydration, raw_applied_inputs=False)
        version = sqlite_state_version(store, sid)
        self.snapshots[sid] = CachedHydratedState(signature, fold, records, inbox_seq, version)
        return fold.state, records, inbox_seq

    def check_ledger(self, sid):
        """Run the production ledger verifier and compare its complete SQL prefix."""

        store = self.connection("observer")
        version = sqlite_state_version(store, sid)
        cached = self._ledger_checks.get(sid)
        if version is not None and cached is not None and cached[0] == version:
            return cached[1]
        head, inbox_seq = store._head(sid)[:2]
        try:
            verified, chains = store._verified_records(sid, head)
            require(
                len(verified) == len(chains) == head,
                "I9",
                f"seed={self.seed} session={sid} verifier returned an incomplete prefix",
            )
            if self.signed:
                checkpoint_count = store._one(
                    "SELECT count(*) FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s",
                    (store.tenant_id, sid),
                )[0]
                require(
                    checkpoint_count == head,
                    "I9",
                    f"seed={self.seed} session={sid} checkpoint count {checkpoint_count}, head={head}",
                )
            else:
                checkpoint_count = store._one(
                    "SELECT count(*) FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s",
                    (store.tenant_id, sid),
                )[0]
                require(
                    checkpoint_count == 0,
                    "I9",
                    f"seed={self.seed} session={sid} hash-only ledger contains checkpoint evidence",
                )
            store._stored_inputs(sid, inbox_seq)
        except LedgerIntegrityError as exc:
            raise InvariantViolation("I9", f"seed={self.seed} session={sid}: {exc}") from exc
        previous = chains[-1] if chains else _initial_chain_digest(store.tenant_id, sid)
        if head:
            self.covered.add("I9")
        if version is not None:
            self._ledger_checks[sid] = (version, previous)
        return previous

    def cached_state_signature(self, store, sid, records=None):
        version = sqlite_state_version(store, sid)
        if version is None:
            return self.state_signature(store, sid, records)
        root = sqlite_root(store)
        key = (id(root), store.tenant_id, sid, records is not None)
        cached = self._signature_cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
        signature = self.state_signature(store, sid, records)
        self._signature_cache[key] = (version, signature)
        return signature

    @staticmethod
    def state_signature(store, sid, records=None):
        params = (store.tenant_id, sid)
        return (
            tuple((row.seq, row.record.encode()) for row in records)
            if records is not None
            else tuple(
                store._rows(
                    "SELECT seq,body,commit_id,writer_epoch,created_ms,digest,chain_digest,record_id,schema_version,"
                    "kind,turn_id,operation_id,attempt,commit_position FROM sk_record "
                    "WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                    params,
                )
            ),
            tuple(
                store._rows(
                    "SELECT input_id,input_seq,kind,body,digest,admission_key_id,admission_signature,"
                    "admission_signed_ms,available_ms,received_ms,consumed_seq FROM sk_inbox "
                    "WHERE tenant_id=%s AND session_id=%s ORDER BY input_seq",
                    params,
                )
            ),
            tuple(
                store._rows(
                    "SELECT key_id,provider_id,wrapped_key,state,created_ms FROM sk_session_key "
                    "WHERE tenant_id=%s AND session_id=%s",
                    params,
                )
            ),
            tuple(
                store._rows(
                    "SELECT content_id,key_id,mac,size,labels,nonce,created_ms,state,pin_until_ms,ciphertext,"
                    "object_locator FROM sk_content WHERE tenant_id=%s AND session_id=%s ORDER BY content_id",
                    params,
                )
            ),
            tuple(
                store._rows(
                    "SELECT record_seq,input_id,content_id,slot FROM sk_content_link "
                    "WHERE tenant_id=%s AND session_id=%s ORDER BY record_seq,input_id,content_id,slot",
                    params,
                )
            ),
        )

    @staticmethod
    def _set_postgres_runtime_timeouts(store):
        store._rows("SELECT set_config('lock_timeout','10ms',false),set_config('statement_timeout','100ms',false)")

    def install_write_observer(self):
        """Per-world patch restored by ExitStack; no production files are changed."""
        from unittest.mock import patch

        from vv_agent.kernel._sql import SQLSessionTx

        actual = SQLSessionTx._write

        def write(tx, record, **kwargs):
            row = tx.store._lock(record.session_id)
            now = tx.store._now()
            self.admissions.append((record.session_id, record.record_id, kwargs["epoch"], row[2], row[4], now))
            return actual(tx, record, **kwargs)

        self.stack.enter_context(patch.object(SQLSessionTx, "_write", write))

    def create(self, sid="s", *, feature_level=1, bootstrap=None):
        create_fixture_session(
            self.connection("host"),
            sid,
            consumers=(),
            feature_level=feature_level,
            bootstrap=bootstrap,
        )
        self.push(sid, InboxItem("initial", "user", {"content": "go"}))

    def child_spec(self, parent_session_id, session_id):
        created = self.connection("observer").read(session_id=parent_session_id, limit=1).records[0].record._payload
        return SessionSpec(
            session_id,
            created["principal"],
            created["workspace"],
            attributes={},
        )

    def push(self, sid, item):
        store = self.connection("host")
        before = store.read_input(sid, item.input_id)
        try:
            if before is not None and item == before.item:
                with store.atomic() as tx:
                    receipt = tx.push(sid, item)
            else:
                receipt = push_fixture_input(store, sid, item)
        except Exception as exc:
            if before is not None:
                raise InvariantViolation("I3", "duplicate input rejected instead of replayed") from exc
            raise
        if before is not None:
            require(receipt.replayed and receipt.input_seq == before.input_seq, "I3", "duplicate input got a new receipt")
        self.queue.append(sid)
        return receipt

    def deliver(self, **kwargs):
        if not self.queue:
            return False
        index = self.choices.choose("wake-delivery", list(range(len(self.queue))))
        sid = self.queue.pop(index)
        self.events.append({"wake_delivery": sid, "index": index})
        return self.drive(sid, **kwargs)

    def enqueue(self, label, callback):
        self.tasks[label] = callback

    def step(self):
        if not self.tasks:
            return False
        label = self.choices.choose("participant", sorted(self.tasks))
        callback = self.tasks.pop(label)
        self.events.append({"participant": label})
        try:
            callback()
        except sqlite3.OperationalError as exc:
            if not any(text in str(exc).lower() for text in ("busy", "locked")):
                raise
            self.events.append({"blocked": label})
            self.tasks[label] = callback
        except Exception as exc:
            if getattr(exc, "sqlstate", None) not in {"55P03", "57014"}:
                raise
            self.events.append({"blocked": label})
            self.tasks[label] = callback
        self.check()
        return True

    def checkpoint(self, point):
        # Only run nested participants at hooks outside an SQL atomic block.
        if any(getattr(c.connection, "in_transaction", False) for c in self.connections.values()):
            return
        if self.tasks and self.choices.choose(f"yield/{point}", [False, True]):
            self.step()

    def expire(self, sid="s"):
        with self.connection("fault").atomic():
            self.connection("fault")._rows(
                "UPDATE sk_session SET lease_until_ms=0 WHERE session_id=%s AND lease_owner IS NOT NULL", (sid,)
            )
        self.events.append({"expire": sid})

    def advance(self, ms=1000):
        if isinstance(self.clock, SimClock):
            self.clock.sleep(ms / 1000)
        else:
            # PostgreSQL clocks stay real; move persisted schedule deadlines only.
            with self.connection("fault").atomic():
                self.connection("fault")._rows(
                    "UPDATE sk_session SET next_drive_ms=0,drive_retry_at_ms=0 WHERE next_drive_ms IS NOT NULL"
                )
        self.events.append({"advance_ms": ms, "clock": type(self.clock).__name__})

    def scripted_model_callback(self, sid):
        """Model replies owned by the scenario, independent of SDK or core assembly."""

        def model(request):
            op = request.metadata["operation_id"]
            attempt = request.metadata["attempt"]
            self.calls.append((sid, op, attempt, "model", None))
            if self.scenario == "model_unknown":
                raise TimeoutError("lost model response")
            if request.metadata.get("purpose") == "compaction":
                self.model_receipts[(sid, op, attempt)] = (5, 1)
                return LLMResponse(
                    json.dumps(
                        {
                            "summary_version": "2.0",
                            "original_user_messages": ["go"],
                            "user_constraints": [],
                            "decisions": [],
                            "files_examined_or_modified": [],
                            "errors_and_fixes": [],
                            "progress": ["scripted effect completed"],
                            "key_facts": [],
                            "open_issues": [],
                            "current_work_state": "tool result summarized",
                            "next_steps": [],
                        }
                    ),
                    raw={"usage": {"prompt_tokens": 5, "completion_tokens": 1}},
                )
            issues_tool = (
                (self.scenario == "context_compacted" and request.metadata["vv_session"]["cycle_index"] in {1, 2})
                or request.metadata["vv_session"]["cycle_index"] == 1
            ) and (sid == "s" or (self.scenario == "cancel_descendants" and sid in {"child", "grandchild"}))
            self.model_receipts[(sid, op, attempt)] = (11, 3) if issues_tool else (7, 1)
            if issues_tool:
                return LLMResponse(
                    "", [ToolCall("job", "effect", {})], raw={"usage": {"prompt_tokens": 11, "completion_tokens": 3}}
                )
            return LLMResponse("done", raw={"usage": {"prompt_tokens": 7, "completion_tokens": 1}})

        return model

    def scripted_hook(self, participant):
        """Durable observation hook shared by test-owned SDK and core runtimes."""

        def hook(point, record):
            if record is not None and point in {"after_commit", "before_external_call", "after_external_call"}:
                self.events.append(
                    {"hook": point, "kind": record.kind, "operation": record.operation_id, "attempt": record.attempt}
                )
                if point == "after_external_call" and self.after_external and "/tool/" in (record.operation_id or ""):
                    callback, self.after_external = self.after_external, None
                    callback(self.workers[participant])
                self.checkpoint(point)

        return hook

    def runtime(self, sid, participant, *, provider=None, children=None):
        model = self.scripted_model_callback(sid)
        hook = self.scripted_hook(participant)

        assembly = CoreRuntimeAssembly(
            self,
            sid,
            participant,
            provider=provider,
            children=children,
            model_callback=model,
            hook=hook,
            scenario=self.scenario,
            memory_settings=self.memory_settings,
        )
        self.keepers[participant] = assembly.lease_keeper
        return assembly

    @contextmanager
    def heartbeat_connection(self, participant):
        yield self.connection(f"{participant}/heartbeat")

    def _drive_runtime(self, store, sid, *, runtime, factory):
        """Call the explicitly selected runtime; SDK tests override only this seam."""
        if factory is not None:
            return runtime.drive_factory(store, sid, factory)
        return runtime.drive(store, sid)

    def drive(self, sid="s", *, participant="worker-0", provider=None, children=None, fault=None, factory=None):
        require(participant not in self.running, "scheduler", "recursive use of one worker connection")
        proxy = self.workers.get(participant)
        if proxy is None or proxy.dead:
            self.runtimes.pop((sid, participant), None)
            if proxy is not None:
                self.connections.pop(participant).connection.close()
            proxy = self.workers[participant] = FaultyStore(self.connection(participant), self, participant)
        proxy.fault = fault
        key = (sid, participant)
        if key not in self.runtimes:
            self.runtimes[key] = self.runtime(sid, participant, provider=provider, children=children)
        rt = self.runtimes[key]
        self.running.add(participant)
        try:
            return self._drive_runtime(proxy.store, sid, runtime=rt, factory=factory)
        except (WorkerDeath, LeaseLost, TimeoutError):
            self.events.append({"worker_stopped": participant, "dead": proxy.dead})
        finally:
            self.running.remove(participant)

    def scan(self, *, provider=None, children=None, participant="scanner-worker", rounds=4):
        """Recover SQL-ready work after losing hints; zero candidates prove nothing."""
        require(not self.tasks, "scheduler", "scanner recovery requires drained participant tasks")
        self.queue.clear()
        scanner = self.connection("scanner")
        progressed = False
        for _ in range(rounds):
            work = [item for item in scanner.list_runnable() if item.kind == "drive"]
            self.events.append({"scan_candidates": [item.session_id for item in work]})
            if not work:
                break
            for item in work:
                before = scanner._head(item.session_id)[0]
                self.drive(item.session_id, provider=provider, children=children, participant=participant)
                after = scanner._head(item.session_id)[0]
                progressed |= after > before
                self.events.append({"scan_drive": item.session_id, "before": before, "after": after})
                self.check()
        # Use another connection and predicate: a broken scanner returning no
        # candidates must not turn its own empty result into liveness evidence.
        observer = self.connection("observer")
        ready = [sid for sid in observer.list_sessions() if observer.is_runnable(sid)]
        require(not ready, "I4", f"fair SQL scan left ready work stranded: {ready}")
        if progressed:
            self.covered.add("I4")
        return progressed

    def check(self, *, quiescent=False, every_prefix=False):
        store = self.connection("observer")
        for sid in store.list_sessions():
            self.check_ledger(sid)
            state, rows, _ = self.read_state(sid)
            checked_inputs = self._check_inputs(sid, rows)
            _inputs, applications = checked_inputs
            records = [r.record for r in rows]
            identities = {(sid, r.record.record_id) for r in rows}
            for session, rid, epoch, actual, until, now in self.admissions:
                if (session, rid) in identities and epoch:
                    require(epoch == actual and until is not None and until > now, "I2", f"unfenced write {rid}")
            current = self.folds.setdefault(sid, fold_factory())
            old = self.prefixes.get(sid, 0)
            for end in range(old + 1, len(records) + 1):
                r = records[end - 1]
                consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                current.extend([r], consumed_inputs=consumed)
            self.prefixes[sid] = len(records)
            # Two independent append-only branches catch reducer drift without
            # replaying the full history for every head. The raw signature
            # forces a rebuild if same-head SQL record bytes change.
            if len(records) != old:
                raw_prefix = self.snapshots[sid].signature[0]
                verified = self.verified_folds.setdefault(sid, fold_factory())
                verified_old = len(self.verified_prefixes.get(sid, ()))
                if raw_prefix[:verified_old] == self.verified_prefixes.get(sid, ()):
                    for end in range(verified_old + 1, len(records) + 1):
                        r = records[end - 1]
                        consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                        verified.extend([r], consumed_inputs=consumed)
                else:
                    verified = self.verified_folds[sid] = fold_factory()
                    for r in records:
                        consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                        verified.extend([r], consumed_inputs=consumed)
                self.verified_prefixes[sid] = raw_prefix
                require(current.state == verified.state, "I8", "append-only fold branches diverged")
            if every_prefix:
                replay = fold_factory()
                for end, r in enumerate(records, 1):
                    consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                    replay.extend([r], consumed_inputs=consumed)
                    self.compare_prefix(records[:end], replay.state)
            for oid, op in state.operations.items():
                first_key = op.attempts[1].plan._payload["idempotency_key"]
                for number, attempt in op.attempts.items():
                    key = (sid, oid, number)
                    external = [c for c in self.calls if c[:3] == key]
                    require(len(external) <= 1, "I1", f"duplicate dispatch {key}")
                    require(attempt.plan._payload["idempotency_key"] == first_key, "I1", f"retry changed key {key}")
                    if (
                        attempt.unknown
                        and op.kind != "model"
                        and (attempt.plan._payload["tool"] or {}).get("idempotency") != "supported"
                    ):
                        require(attempt.unknown._payload["retry"] != "retry", "I10", f"unsafe unknown authorized resend {key}")
                    if number > 1:
                        first = op.attempts[1].plan._payload
                        bound = max(2, len(first["request"].get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])))
                        if op.kind != "model":
                            bound = 2
                        allowed = op.kind == "model" or (attempt.plan._payload["tool"] or {}).get("idempotency") == "supported"
                        require(allowed and number <= bound, "I1", f"retry not permitted {key}")
                    if attempt.unknown and attempt.result:
                        evidence = attempt.result._payload["evidence"]
                        if attempt.child_handle:
                            child = attempt.child_handle["session_id"]
                            terminal_turn = attempt.child_handle["turn_id"]
                            declared_child = attempt.result._payload["result"].get("metadata", {}).get("child_session_id")
                            trusted = declared_child == child and any(
                                r.record.kind == "turn_ended"
                                and r.record.turn_id == terminal_turn
                                and f"child/{r.seq}/{r.record.digest}" in evidence
                                for r in store.read(child).records
                            )
                        else:
                            receipts = self.connection("provider")._rows("SELECT handle FROM sim_effect")
                            trusted = any(
                                json.loads(h)["operation_id"] == oid and json.loads(h)["evidence"] in evidence
                                for (h,) in receipts
                            )
                        require(trusted, "I10", f"unknown settled without durable external or child evidence {key}")
                    outcomes = [r for r in records if r.kind == "op_completed" and r.operation_id == oid and r.attempt == number]
                    require(len(outcomes) <= 1, "I5", f"duplicate attempt usage {key}")
                    if attempt.state == "unknown":
                        require(attempt.unknown._payload["measurement_missing"], "I5", f"unknown lost missing-meter marker {key}")
            if quiescent and state.active_turn_id and store.peek_inbox(sid):
                require(bool(state.waits) or store.is_runnable(sid), "I4", f"stranded active input {sid}")
            if state.operations:
                self.covered.update({"I1", "I8"})
            if any(a.unknown for op in state.operations.values() for a in op.attempts.values()):
                self.covered.add("I10")
            if applications:
                self.covered.add("I3")

    def _check_inputs(self, sid, rows):
        signature = self.snapshots[sid].signature
        inputs = tuple((row[0], row[10]) for row in signature[1])
        applications = Counter(r.record._payload["input"]["input_id"] for r in rows if r.record.kind == "input_applied")
        for row in rows:
            if row.record.kind == "input_applied":
                require(
                    row.created_ms >= row.record._payload["input"]["available_ms"],
                    "I3",
                    "input consumed before its durable availability time",
                )
        for input_id, consumed in inputs:
            require(applications[input_id] == int(consumed is not None), "I3", f"input {sid}/{input_id} consumption differs")
        require(set(applications) <= {key for key, _ in inputs}, "I3", "application has no inbox input")
        return inputs, applications

    @staticmethod
    def compare_prefix(records, state):
        consumed = [InboxItem(**r.payload["input"]) for r in records if r.kind == "input_applied"]
        replay = fold_factory().extend(records, consumed_inputs=consumed)
        require(replay == state, "I8", "cold prefix replay differs from incremental fold")

    def stale_writer_probe(self):
        store = self.connection("stale-writer")
        old = store.acquire("s", owner="stale", ttl_ms=15000)
        require(old is not None, "probe", "stale probe could not acquire")
        self.expire()
        fresh = self.connection("new-writer").acquire("s", owner="fresh", ttl_ms=15000)
        require(fresh is not None, "probe", "takeover did not acquire")
        rec = make_record(
            "usage_observed",
            session_id="s",
            payload={
                "meter_id": "stale-probe",
                "observation": 1,
                "mode": "cumulative",
                "usage": content_ref("stale-probe/usage"),
                "source": content_ref("stale-probe/source"),
            },
        )
        try:
            with store.atomic() as tx:
                tx.append("s", lease=old, expected_seq=store.read("s").head_seq, commit_id="stale", records=(rec,))
        except LeaseLost:
            pass
        else:
            raise InvariantViolation("I2", "actual SQL append accepted a superseded lease")
        finally:
            self.connection("new-writer").release(fresh)
        self.covered.add("I2")
        self.check()

    def check_accounting(self):
        from vv_agent.kernel._usage import project_model_usage

        store = self.connection("observer")
        exercised = False
        for sid in store.list_sessions():
            snapshot = self.read_state(sid)
            state, rows, _ = snapshot
            meters, observations = {}, {}
            for stored in rows:
                r = stored.record
                if r.kind == "usage_observed":
                    payload = r._payload
                    meter = payload["meter_id"]
                    require(
                        payload["observation"] == observations.get(meter, 0) + 1,
                        "I5",
                        f"missing or repeated accounting observation {meter}",
                    )
                    observations[meter] = payload["observation"]
                    meters[meter] = (r.record_id, r.digest)
            retained = {meter: (record.record_id, record.digest) for meter, record in state.usage_values.items()}
            require(retained == meters, "I5", "cumulative accounting did not retain exactly the latest value")
            exercised |= bool(meters)
            for tid in state.turns:
                usage, _ = project_model_usage(state, tid)
                calls = usage.model_calls
                exercised |= bool(calls)
                expected = {
                    (oid, number)
                    for oid, op in state.operations.items()
                    if op.kind == "model" and op.turn_id == tid
                    for number, attempt in op.attempts.items()
                    if attempt.started
                }
                require(
                    Counter((c.operation_id, c.attempt) for c in calls) == Counter({key: 1 for key in expected}),
                    "I5",
                    "model attempt accounting was lost or counted twice",
                )
                for call in calls:
                    attempt = state.operations[call.operation_id].attempts[call.attempt]
                    # External observations cannot supply a measurement whose
                    # result transaction rolled back. Only a durable receipt can.
                    committed = attempt.result is not None
                    receipt = self.model_receipts.get((sid, call.operation_id, call.attempt)) if committed else None
                    if receipt is None:
                        require(
                            call.usage.usage_source.value == "accounting_missing"
                            and call.usage.input_tokens is None
                            and call.usage.output_tokens is None,
                            "I5",
                            "missing measurement invented usage",
                        )
                    else:
                        require(
                            (call.usage.input_tokens, call.usage.output_tokens) == receipt,
                            "I5",
                            "provider usage differs from durable billing projection",
                        )
                for field in ("input_tokens", "output_tokens"):
                    values = [getattr(call.usage, field) for call in calls]
                    total = None if None in values else sum(values)
                    require(getattr(usage, field) == total, "I5", f"{field} total differs from conserved attempt ledger")
        if exercised:
            self.covered.add("I5")

    def not_ready(self, sid):
        raise SimRuntimeNotReady(1)

    def summary(self):
        return {
            "seed": self.seed,
            "scenario": self.scenario,
            "decisions": self.choices.trace,
            "events": self.events,
            "calls": self.calls,
            "worker_claims": self.worker_claims,
            "covered": sorted(self.covered),
            "pending": PENDING,
            "backend": "sqlite" if isinstance(self.database, Path) else "postgres",
            "ledger_proof": "ed25519" if self.signed else "hash-only",
        }
