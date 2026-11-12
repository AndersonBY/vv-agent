"""Single-thread, seeded scheduling around the production driver and SQL transactions.

SQLite uses participant-owned connections to a real file and an injected clock.
PostgreSQL keeps clock_timestamp(); expiry is an explicit SQL fault. No state
machine or storage rules are reimplemented here.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session import kernel
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.providers import Unknown
from vv_agent.session.records import InboxItem, SessionSpec, make_record
from vv_agent.session.reducer import Fold, fold
from vv_agent.session.runtime import Runtime, RuntimeNotReady
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost
from vv_agent.session.timing import SystemClock
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, ToolCall

PENDING = {
    "I9": "W2c: chain_digest and chain verifier do not exist at the P1 baseline",
    "I11": "W4: worker support sets and claim compatibility filtering do not exist at the P1 baseline",
}


class InvariantViolation(AssertionError):
    def __init__(self, invariant, detail):
        self.invariant = invariant
        super().__init__(f"{invariant}: {detail}")


def require(condition, invariant, detail):
    if not condition:
        raise InvariantViolation(invariant, detail)


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


class ScheduledLeaseKeeper:
    """Schedule the real heartbeat.poll; never create a heartbeat thread."""

    def __init__(self, world, participant):
        self.world, self.participant = world, participant

    def start(self, heartbeat):
        self.poll(heartbeat)
        self.world.enqueue(f"renew/{self.participant}", lambda: self.poll(heartbeat))

    def poll(self, heartbeat):
        try:
            heartbeat.poll(self.world.connection(f"{self.participant}/heartbeat"))
        except LeaseLost:
            heartbeat.lost = True
            if hasattr(heartbeat, "token"):
                heartbeat.token.cancel("lease_lost")
            raise
        finally:
            heartbeat.ready.set()

    def close(self, heartbeat):
        self.world.tasks.pop(f"renew/{self.participant}", None)

    def is_alive(self, heartbeat):
        return False


class SynchronousInvoker:
    def __init__(self, world):
        self.world = world

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        self.world.checkpoint("invoke")
        try:
            return callback()
        except InvariantViolation:
            raise
        except Exception as exc:
            return Unknown(f"{type(exc).__name__}: scripted outcome unavailable")


class SimWorld:
    def __init__(self, database, *, seed=0, replay=None):
        self.database, self.seed = database, seed
        self.clock = SimClock() if isinstance(database, Path) else SystemClock()
        self.choices = Choices(seed, replay)
        self.stack = ExitStack()
        self.connections, self.workers, self.keepers = {}, {}, {}
        self.events, self.calls, self.admissions = [], [], []
        self.model_receipts = {}
        self.tasks, self.running = {}, set()
        self.folds, self.prefixes, self.runtimes, self.snapshots = {}, {}, {}, {}
        self.queue = []
        self.after_external = None
        self.scenario = None
        self.covered = set()

    def __enter__(self):
        self.connection("host").install_schema()
        return self

    def __exit__(self, *args):
        if isinstance(args[1], Exception):
            args[1].add_note(json.dumps(self.summary()))
        return self.stack.__exit__(*args)

    def connection(self, participant):
        if participant not in self.connections:
            if isinstance(self.database, Path):
                context = SQLiteStore.standalone(str(self.database), clock=self.clock)
            else:
                context = PostgresStore.standalone(self.database)
            store = self.stack.enter_context(context)
            if isinstance(store, SQLiteStore):
                store._rows("PRAGMA busy_timeout=10")
            else:
                store._rows("SELECT set_config('lock_timeout','10ms',false),set_config('statement_timeout','100ms',false)")
            self.connections[participant] = store
        return self.connections[participant]

    def install_write_observer(self):
        """Per-world patch restored by ExitStack; no production files are changed."""
        from unittest.mock import patch

        from vv_agent.session.sql import SQLSessionTx

        actual = SQLSessionTx._write

        def write(tx, record, **kwargs):
            row = tx.store._lock(record.session_id)
            now = tx.store._now()
            self.admissions.append((record.session_id, record.record_id, kwargs["epoch"], row[2], row[4], now))
            return actual(tx, record, **kwargs)

        self.stack.enter_context(patch.object(SQLSessionTx, "_write", write))

    def create(self, sid="s"):
        with self.connection("host").atomic() as tx:
            tx.create(SessionSpec(sid, "sim", "/tmp"), consumers=())
        self.push(sid, InboxItem("initial", "user", {"content": "go"}))

    def push(self, sid, item):
        store = self.connection("host")
        before = store.read_input(sid, item.input_id)
        try:
            with store.atomic() as tx:
                receipt = tx.push(sid, item)
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

    def runtime(self, sid, participant, *, provider=None, children=None):
        world = self

        def model(request):
            meta = request.metadata["vv_session"]
            op = request.metadata["operation_id"]
            attempt = request.metadata["attempt"]
            world.calls.append((sid, op, attempt, "model", None))
            if world.scenario == "model_unknown":
                raise TimeoutError("lost model response")
            world.model_receipts[(sid, op, attempt)] = (11, 3) if sid == "s" and meta["cycle_index"] == 1 else (7, 1)
            if sid == "s" and meta["cycle_index"] == 1:
                return LLMResponse(
                    "", [ToolCall("job", "effect", {})], raw={"usage": {"prompt_tokens": 11, "completion_tokens": 3}}
                )
            return LLMResponse("done", raw={"usage": {"prompt_tokens": 7, "completion_tokens": 1}})

        class Model(ScriptedLLM):
            def complete(self, request):
                return model(request)

        def forbidden(_ctx, _args):
            raise AssertionError("external effects must go through the scripted provider")

        idempotent = self.scenario == "unknown_idempotent"
        tool = FunctionTool(
            name="effect",
            description="scripted effect",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke=forbidden,
            tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED if idempotent else ToolIdempotency.UNSUPPORTED),
        )
        keeper = self.keepers.setdefault(participant, ScheduledLeaseKeeper(self, participant))

        def hook(point, record):
            if record is not None and point in {"after_commit", "before_external_call", "after_external_call"}:
                self.events.append(
                    {"hook": point, "kind": record.kind, "operation": record.operation_id, "attempt": record.attempt}
                )
                if point == "after_external_call" and self.after_external and "/tool/" in (record.operation_id or ""):
                    callback, self.after_external = self.after_external, None
                    callback(self.workers[participant])
                self.checkpoint(point)

        return Runtime(
            agent=Agent(name="sim", instructions="Be precise.", tools=[tool]),
            config=RunConfig(workspace=Path("/tmp"), session_memory_enabled=False, tool_registry_factory=ToolRegistry),
            resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            llm=Model(),
            heartbeat_store=lambda: self.heartbeat_connection(participant),
            providers={"effect": provider} if provider is not None and not children else {},
            children=children or {},
            clock=self.clock,
            lease_keeper=keeper,
            invoker=SynchronousInvoker(self),
            hook=hook,
            lease_retry_sleep=self.clock.sleep,
            lease_retry_jitter=lambda a, b: a,
            wake=self.queue.append,
        )

    @contextmanager
    def heartbeat_connection(self, participant):
        yield self.connection(f"{participant}/heartbeat")

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
            if factory is not None:
                return kernel._drive(proxy.store, sid, runtime=factory, clock=self.clock, lease_keeper=rt.lease_keeper)
            return kernel.drive(proxy.store, sid, runtime=rt)
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
            version = store._head(sid)[:2]
            cached = self.snapshots.get(sid)
            if cached is None or cached[0] != version or every_prefix:
                cached = self.snapshots[sid] = (version, store.read_state(sid))
            state, rows, _ = cached[1]
            records = [r.record for r in rows]
            identities = {(sid, r.record.record_id) for r in rows}
            for session, rid, epoch, actual, until, now in self.admissions:
                if (session, rid) in identities and epoch:
                    require(epoch == actual and until is not None and until > now, "I2", f"unfenced write {rid}")
            inputs = store._rows("SELECT input_id,consumed_seq FROM sk_inbox WHERE session_id=%s", (sid,))
            applications = Counter(r._payload["input"]["input_id"] for r in records if r.kind == "input_applied")
            for input_id, consumed in inputs:
                require(applications[input_id] == int(consumed is not None), "I3", f"input {sid}/{input_id} consumption differs")
            require(set(applications) <= {key for key, _ in inputs}, "I3", "application has no inbox input")
            current = self.folds.setdefault(sid, Fold())
            old = self.prefixes.get(sid, 0)
            for end in range(old + 1, len(records) + 1):
                r = records[end - 1]
                consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                current.extend([r], consumed_inputs=consumed)
            self.prefixes[sid] = len(records)
            # An unchanged prefix has already passed cold replay. Still compare
            # the actual store cache and external counters on every step below.
            if len(records) != old:
                self.compare_prefix(records, current.state)
            if every_prefix:
                replay = Fold()
                for end, r in enumerate(records, 1):
                    consumed = [InboxItem(**r._payload["input"])] if r.kind == "input_applied" else []
                    replay.extend([r], consumed_inputs=consumed)
                    self.compare_prefix(records[:end], replay.state)
            require(state == current.state, "I8", "store cache differs from incremental replay")
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
                            child = attempt.result._payload["result"].get("metadata", {}).get("child_session_id")
                            terminals = store.read(child).records if child else ()
                            trusted = any(
                                r.record.kind == "turn_ended" and f"{child}/{r.seq}/{r.record.digest}" in evidence
                                for r in terminals
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

    @staticmethod
    def compare_prefix(records, state):
        consumed = [InboxItem(**r.payload["input"]) for r in records if r.kind == "input_applied"]
        require(fold(records, consumed_inputs=consumed) == state, "I8", "cold prefix replay differs from incremental fold")

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
                "usage": {},
                "source": "host",
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
        from vv_agent.session.result import project_result

        store = self.connection("observer")
        exercised = False
        for sid in store.list_sessions():
            snapshot = store.read_state(sid)
            state, rows, _ = snapshot
            meters = {}
            for stored in rows:
                r = stored.record
                if r.kind == "usage_observed":
                    payload = r._payload
                    meter = payload["meter_id"]
                    previous = meters.get(meter)
                    require(
                        payload["observation"] == (previous._payload["observation"] + 1 if previous else 1),
                        "I5",
                        f"missing or repeated accounting observation {meter}",
                    )
                    meters[meter] = r
            require(state.usage_values == meters, "I5", "cumulative accounting did not retain exactly the latest value")
            exercised |= bool(meters)
            for tid in state.turns:
                usage = project_result(store, sid, tid, snapshot=snapshot).token_usage
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
        raise RuntimeNotReady(1)

    def summary(self):
        return {
            "seed": self.seed,
            "scenario": self.scenario,
            "decisions": self.choices.trace,
            "events": self.events,
            "calls": self.calls,
            "covered": sorted(self.covered),
            "pending": PENDING,
            "backend": "sqlite" if isinstance(self.database, Path) else "postgres",
        }
