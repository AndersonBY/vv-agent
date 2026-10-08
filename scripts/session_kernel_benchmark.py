"""Measure the real PostgreSQL kernel, using disposable databases only."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.session.helpers import record
from tests.session.test_capacity import seed_history
from tests.session.test_recovery_matrix import runtime
from vv_agent.session.kernel import _Scope, drive, read_state
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.reducer import fold
from vv_agent.session.store import LeaseLost, WorkCursor


def measure(callback, samples):
    values = []
    for _ in range(samples):
        begin = time.perf_counter()
        callback()
        values.append((time.perf_counter() - begin) * 1000)
    return {"samples_ms": values, "median_ms": statistics.median(values), "max_ms": max(values)}


def benchmark_history(store, database, size, samples):
    tid, logical = seed_history(store, database, size)
    consumed = tuple(InboxItem(**r.payload["input"]) for r in logical if r.kind == "input_applied")
    lease = store.acquire("s", owner="append-benchmark", ttl_ms=600000)
    assert lease is not None
    addition = record("usage_observed", tid=tid, meter_id="measurement", observation=1)
    appended, restore = [], []
    for index in range(samples):
        before = time.perf_counter()
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=size, commit_id=f"measurement/{index}", records=(addition,))
        appended.append((time.perf_counter() - before) * 1000)
        # Fixture reset, outside the measured durable transaction. Only our n+1 row/receipt.
        before = time.perf_counter()
        with store.connection.transaction():
            assert store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq=%s", (size + 1,)).rowcount == 1
            assert (
                store.connection.execute(
                    "DELETE FROM sk_commit WHERE session_id='s' AND commit_id=%s", (f"measurement/{index}",)
                ).rowcount
                == 1
            )
            store.connection.execute("UPDATE sk_session SET head_seq=%s WHERE session_id='s'", (size,))
        restore.append((time.perf_counter() - before) * 1000)
    assert store.read("s").head_seq == size
    append = {"samples_ms": appended, "median_ms": statistics.median(appended), "max_ms": max(appended)}
    store.release(lease)
    rt = runtime(database, [])
    lease = store.acquire("s", owner="steady-benchmark", ttl_ms=rt.ttl_ms)
    assert lease is not None and rt.ttl_ms == 15000 and rt.heartbeat_seconds == 0.25
    scope = _Scope(lease, rt)
    scope.thread.start()
    steady_values = []
    try:
        read_state(store, "s")  # Warm this exact head/epoch before consecutive durable appends.
        for index in range(samples):
            addition = record("usage_observed", tid=tid, meter_id="steady", observation=index + 1)
            before = time.perf_counter()
            with scope.lock:
                assert not scope.lost
                with store.atomic() as tx:
                    tx.append("s", lease=scope.lease, expected_seq=size + index, commit_id=f"steady/{index}", records=(addition,))
            steady_values.append((time.perf_counter() - before) * 1000)
        assert not scope.lost
    finally:
        scope.stop.set()
        scope.thread.join(timeout=2)
        store.release(scope.lease)
    steady = {"samples_ms": steady_values, "median_ms": statistics.median(steady_values), "max_ms": max(steady_values)}
    with store.connection.transaction():
        store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq>%s", (size,))
        store.connection.execute("DELETE FROM sk_commit WHERE session_id='s' AND head_seq>%s", (size,))
        store.connection.execute("UPDATE sk_session SET head_seq=%s WHERE session_id='s'", (size,))
    full_fold = measure(lambda: fold(logical, consumed_inputs=consumed), samples)

    def read_cold():
        with PostgresStore.standalone(database) as fresh:
            read_state(fresh, "s")

    cold_read = measure(read_cold, samples)
    cold_drive = []
    cold_failures: list[dict[str, Any]] = []
    for index in range(min(samples, 3)):
        # New connection and Runtime per invocation, with no retained reducer/cache.
        begin = time.perf_counter()
        with PostgresStore.standalone(database) as fresh:
            rt = runtime(database, [])
            with patch.object(rt.llm, "complete", wraps=rt.llm.complete) as completion:
                try:
                    drive(fresh, "s", runtime=rt)
                except LeaseLost as exc:
                    cold_failures.append({"sample": index, "error": type(exc).__name__, "reason": str(exc)})
                assert completion.call_count == 0
        cold_drive.append((time.perf_counter() - begin) * 1000)
        state, recovered, _ = read_state(store, "s")
        ends = [r.record for r in recovered if r.record.kind == "turn_ended"]
        if ends:
            assert state.active_turn_id is None and len(ends) == 1
            assert ends[0].payload["status"] == "completed" and ends[0].payload["result"] == "retained answer"
        else:
            assert any(f["sample"] == index for f in cold_failures), "drive returned without terminal or LeaseLost"
            cold_failures[-1]["elapsed_ms"] = cold_drive[-1]
        with store.connection.transaction():
            store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq>%s", (size,))
            store.connection.execute("DELETE FROM sk_commit WHERE session_id='s' AND head_seq>%s", (size,))
            store.connection.execute("UPDATE sk_session SET head_seq=%s,terminal_seq=0 WHERE session_id='s'", (size,))
        store.rebuild_schedule("s")
    state, rows, _ = read_state(store, "s")
    assert state.phase == "active" and len(rows) == size
    return {
        "records": size,
        "logical_bytes": sum(len(r.encode()) for r in logical),
        "append": append,
        "steady_append": steady | {"lease_ttl_ms": rt.ttl_ms, "heartbeat_seconds": rt.heartbeat_seconds},
        "full_fold": full_fold,
        "read_and_fold": cold_read,
        "cold_drive": {
            "samples_ms": cold_drive,
            "median_ms": statistics.median(cold_drive),
            "max_ms": max(cold_drive),
            "failures": cold_failures,
        },
        "reset_samples_ms": restore,
        "provider_call_increment": 0,
    }


def benchmark_runnable(store, size, samples):
    with store.atomic() as tx:
        for index in range(size):
            sid = f"catalog/{index:05d}"
            tx.create(SessionSpec(sid, "benchmark", "/tmp"), consumers=("events",))
            if index % 2 == 0:
                tx.push(sid, InboxItem("input", "user", {"content": "go"}))
    store.connection.execute("ANALYZE sk_session")
    store.connection.execute("ANALYZE sk_inbox")
    store.connection.execute("ANALYZE sk_consumer")
    first_page = measure(lambda: store.list_runnable(limit=100), samples)
    late_page = measure(
        lambda: store.list_runnable(limit=100, after=WorkCursor(f"catalog/{size - 100:05d}", "drive", "")), samples
    )

    def scan():
        count, after = 0, None
        while page := store.list_runnable(limit=100, after=after):
            count += len(page)
            after = page[-1].cursor
        assert count == size + (size + 1) // 2

    return {
        "sessions": size,
        "runnable_items": size + (size + 1) // 2,
        "page_size": 100,
        "first_page": first_page,
        "late_page": late_page,
        "full_scan": measure(scan, samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/kernel-m5/performance.json"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 5000, 20000])
    parser.add_argument("--sessions", nargs="*", type=int, default=[1000, 10000])
    parser.add_argument("--assert-capacity", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_unix_s": time.time(),
        "samples": args.samples,
        "cold_samples": min(args.samples, 3),
        "method": (
            "Single active turn with retained final model receipt before terminal commit, "
            "completed tool triples with bounded 1KiB receipts; append is committed; steady append uses consecutive heads; "
            "fresh connection/runtime for cold drive through terminal commit; OS/PG buffers remain warm."
        ),
        "history": [],
        "runnable": [],
    }
    with psycopg.connect("dbname=postgres", autocommit=True) as admin:
        version = admin.execute("SELECT version()").fetchone()
        assert version is not None
        payload["postgres"] = version[0]
        for category, sizes in (("history", args.sizes), ("runnable", args.sessions)):
            for size in sizes:
                name = f"vvsk_test_m5_bench_{uuid4().hex[:16]}"
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                try:
                    with PostgresStore.standalone(f"dbname={name}") as store:
                        store.install_schema()
                        value = (
                            benchmark_history(store, f"dbname={name}", size, args.samples)
                            if category == "history"
                            else benchmark_runnable(store, size, args.samples)
                        )
                        payload[category].append(value)
                        print(category, size, json.dumps(value), flush=True)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(payload, indent=2) + "\n")
                finally:
                    admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    if args.assert_capacity:
        failures = []
        for case in payload["history"]:
            size = case["records"]
            if case["cold_drive"]["failures"] or (size >= 2000 and case["cold_drive"]["max_ms"] > size):
                failures.append(f"{size}: cold drive exceeds {size}ms or loses lease")
            if case["steady_append"]["median_ms"] > 50:
                failures.append(f"{size}: steady append median exceeds 50ms")
        for case in payload["runnable"]:
            if case["full_scan"]["max_ms"] > 1000:
                failures.append(f"{case['sessions']}: full pagination exceeds 1000ms")
        assert not failures, "\n".join(failures)
    return int(any(case["cold_drive"]["failures"] for case in payload["history"]))


if __name__ == "__main__":
    raise SystemExit(main())
