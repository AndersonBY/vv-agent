"""Benchmark the real PostgreSQL session kernel with disposable databases.

The default command preserves the small synthetic capacity suite. Use
``--turns 100 --history-mb-per-turn 10`` for the S1-A12 signed, encrypted content-history
workload. New connections alone are not described as cold: cold measurements
require ``--cold-mode restart`` and explicit PostgreSQL cluster paths.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext, suppress
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import psycopg
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.s1_fixture_values import (  # noqa: E402
    content_ref,
    fixture_content_reader,
    fixture_key_provider_for_store,
    principal,
    wrap_fixture_store,
)
from tests.session import test_capacity  # noqa: E402
from tests.session.helpers import record  # noqa: E402
from tests.session.test_capacity import DEFAULT_BUDGET, turn_record  # noqa: E402
from tests.session.test_recovery_matrix import runtime  # noqa: E402
from tests.support.s1_fixture_values import (  # noqa: E402
    append_fixture_records,
    clear_fixture_state_cache,
    create_fixture_session,
    push_fixture_input,
    read_fixture_state,
)
from vv_agent.kernel._canonical_json import canonical_json_bytes  # noqa: E402
from vv_agent.kernel._content_publication import make_record_before_publication  # noqa: E402
from vv_agent.session.kernel import _Scope, drive, read_state  # noqa: E402
from vv_agent.session.postgres import PostgresStore  # noqa: E402
from vv_agent.session.records import InboxItem, SessionSpec  # noqa: E402
from vv_agent.session.reducer import fold  # noqa: E402
from vv_agent.session.store import LeaseLost, WorkCursor  # noqa: E402
from vv_agent.types import LLMResponse  # noqa: E402

DEFAULT_SIZES = [100, 1000, 5000, 20000]
DEFAULT_SESSIONS = [1000, 10000]
MIB = 1024 * 1024
_BENCHMARK_PROTOCOL = "S1"
_A12_WARM_TIMING_LIMIT_MS = 60_000.0
_A12_COLD_TIMING_LIMIT_MS = 120_000.0
_A12_STANDARD_TIMINGS = frozenset(
    {
        "warm_cached_read",
        "warm_uncached_read",
        "warm_fold",
        "warm_append",
        "warm_drive",
        "cold_read",
        "cold_drive",
    }
)
_FIXTURE_KEY_ID = "vv-agent-benchmark-ed25519-v1"
_FIXTURE_SIGNING_SEED = sha256(b"vv-agent-benchmark-fixture-signer-v1").digest()


class ReceiptCut(BaseException):
    """Stop immediately after the durable final model receipt."""


class _BoundFixtureSigner:
    key_id = _FIXTURE_KEY_ID

    def __init__(self, key: Ed25519PrivateKey) -> None:
        self._key = key

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)


class _FixtureSigner:
    """Deterministic benchmark-only Ed25519 fixture key."""

    def __init__(self) -> None:
        self._key = Ed25519PrivateKey.from_private_bytes(_FIXTURE_SIGNING_SEED)

    def bind_key(self) -> _BoundFixtureSigner:
        return _BoundFixtureSigner(self._key)

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != _FIXTURE_KEY_ID:
            return False
        try:
            self._key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


@contextmanager
def _open_store(database: str, *, install_schema: bool = False) -> Iterator[Any]:
    with PostgresStore.standalone(database) as raw:
        if install_schema:
            raw.install_schema()
        raw.configure_input_admission(_FixtureSigner())
        yield wrap_fixture_store(raw, fixture_key_provider_for_store(raw))


def _benchmark_runtime(database: str, steps: Sequence[Any], **kwargs: Any):
    value = runtime(database, steps, **kwargs)
    value.heartbeat_store = lambda: _open_store(database)
    return value


def _seed_signed_history(store, database: str, size: int):
    original_runtime = test_capacity.runtime

    def signed_runtime(runtime_store, runtime_database, steps, *args, **kwargs):
        value = original_runtime(runtime_store, runtime_database, steps, *args, **kwargs)
        value.heartbeat_store = lambda: _open_store(runtime_database)
        return value

    with patch.object(test_capacity, "runtime", signed_runtime):
        return test_capacity.seed_history(store, database, size)


def timing(values: Sequence[float]) -> dict[str, Any]:
    assert values, "timing requires at least one sample"
    return {
        "n": len(values),
        "samples_ms": list(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure(callback: Callable[[], Any], samples: int) -> dict[str, Any]:
    values = []
    for _ in range(samples):
        begin = time.perf_counter()
        callback()
        values.append((time.perf_counter() - begin) * 1000)
    return timing(values)


def elapsed(callback: Callable[[], Any]) -> tuple[float, Any]:
    begin = time.perf_counter()
    value = callback()
    return (time.perf_counter() - begin) * 1000, value


def table_bytes(connection, name: str) -> dict[str, int]:
    row = connection.execute(
        "SELECT pg_relation_size(%s::regclass), pg_table_size(%s::regclass), "
        "pg_indexes_size(%s::regclass), pg_total_relation_size(%s::regclass)",
        (name, name, name, name),
    ).fetchone()
    assert row is not None
    return {
        "relation_bytes": int(row[0]),
        "table_bytes_including_toast": int(row[1]),
        "index_bytes": int(row[2]),
        "total_bytes": int(row[3]),
    }


def fixture_security(connection) -> dict[str, Any]:
    inputs = connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE admission_key_id IS NOT NULL "
        "AND admission_signature IS NOT NULL AND admission_signed_ms IS NOT NULL) "
        "FROM sk_inbox WHERE session_id='s'"
    ).fetchone()
    content = connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE ciphertext IS NOT NULL AND octet_length(ciphertext)>0) "
        "FROM sk_content WHERE session_id='s'"
    ).fetchone()
    checkpoints = connection.execute("SELECT count(*) FROM sk_checkpoint WHERE session_id='s'").fetchone()
    assert inputs is not None and content is not None and checkpoints is not None
    return {
        "admission": "ed25519:vv-agent-benchmark-ed25519-v1",
        "input_rows": int(inputs[0]),
        "signed_input_rows": int(inputs[1]),
        "checkpoint_rows": int(checkpoints[0]),
        "content_cipher": "AES-GCM",
        "content_rows": int(content[0]),
        "encrypted_content_rows": int(content[1]),
        "all_inputs_signed": int(inputs[0]) > 0 and inputs[0] == inputs[1],
        "all_content_encrypted": int(content[0]) > 0 and content[0] == content[1],
    }


def message_reuse_metrics(linked: Sequence[Any], physical: Sequence[Any]) -> dict[str, Any]:
    """Build reuse metrics from linked rows and distinct physical content rows."""

    assert len(linked) == 4 and len(physical) == 2
    message_links = int(linked[0])
    linked_plaintext = int(linked[1])
    linked_ciphertext = int(linked[2])
    unique_messages = int(linked[3])
    physical_plaintext = int(physical[0])
    physical_ciphertext = int(physical[1])
    return {
        "link_rows": message_links,
        "unique_content_rows": unique_messages,
        "physical_plaintext_size_sum": physical_plaintext,
        "linked_logical_plaintext_bytes": linked_plaintext,
        "ciphertext_octet_bytes": linked_ciphertext,
        "physical_ciphertext_octet_bytes": physical_ciphertext,
        "reuse_ratio": message_links / unique_messages if unique_messages else None,
        "physical_to_linked_ratio": physical_plaintext / linked_plaintext if linked_plaintext else None,
    }


def collect_database_metrics(connection) -> dict[str, Any]:
    records = connection.execute(
        "SELECT count(*),coalesce(sum(octet_length(body)),0),coalesce(sum(pg_column_size(body)),0) "
        "FROM sk_record WHERE session_id='s'"
    ).fetchone()
    content = connection.execute(
        "SELECT count(*),coalesce(sum(size),0),coalesce(sum(octet_length(ciphertext)),0) FROM sk_content WHERE session_id='s'"
    ).fetchone()
    links = connection.execute("SELECT count(*) FROM sk_content_link WHERE session_id='s'").fetchone()
    slots = connection.execute(
        "SELECT slot,count(*) FROM sk_content_link WHERE session_id='s' GROUP BY slot ORDER BY slot"
    ).fetchall()
    messages = connection.execute(
        "SELECT count(l.slot),coalesce(sum(c.size),0),coalesce(sum(octet_length(c.ciphertext)),0),count(DISTINCT c.content_id) "
        "FROM sk_content_link l JOIN sk_content c ON c.tenant_id=l.tenant_id "
        "AND c.session_id=l.session_id AND c.content_id=l.content_id "
        "WHERE l.session_id=%s AND l.slot LIKE %s",
        ("s", "%initial_messages[%"),
    ).fetchone()
    physical_messages = connection.execute(
        "SELECT coalesce(sum(c.size),0),coalesce(sum(octet_length(c.ciphertext)),0) "
        "FROM sk_content c WHERE c.session_id='s' AND EXISTS ("
        "SELECT 1 FROM sk_content_link l WHERE l.tenant_id=c.tenant_id "
        "AND l.session_id=c.session_id AND l.content_id=c.content_id "
        "AND l.slot LIKE %s)",
        ("%initial_messages[%",),
    ).fetchone()
    database_size = connection.execute("SELECT pg_database_size(current_database())").fetchone()
    assert records and content and links and messages and physical_messages and database_size

    relations = {
        name: table_bytes(connection, name)
        for name in (
            "sk_record",
            "sk_commit",
            "sk_checkpoint",
            "sk_content",
            "sk_content_link",
            "sk_inbox",
            "sk_session",
        )
    }
    return {
        "database_size_bytes": int(database_size[0]),
        "relations": relations,
        "sql_records": {
            "table": relations["sk_record"],
            "rows": int(records[0]),
            "body_octet_bytes": int(records[1]),
            "body_pg_column_bytes": int(records[2]),
        },
        "content_store": {
            "table": relations["sk_content"],
            "relations": {
                "sk_content_link": relations["sk_content_link"],
                "sk_content_link_total_bytes": relations["sk_content_link"]["total_bytes"],
            },
            "rows": int(content[0]),
            "stored_plaintext_size_sum": int(content[1]),
            "ciphertext_octet_bytes": int(content[2]),
            "link_rows": int(links[0]),
            "slot_counts": {str(slot): int(count) for slot, count in slots},
        },
        "message_reuse": message_reuse_metrics(messages, physical_messages),
    }


def a12_message(target_bytes_per_turn: int) -> tuple[dict[str, str], int, int]:
    """Return the fixed message used as the complete history of every A12 turn."""

    if target_bytes_per_turn <= 0:
        raise ValueError("target_bytes_per_turn must be positive")
    empty_message = {"role": "assistant", "content": ""}
    content_chars = target_bytes_per_turn - len(canonical_json_bytes(empty_message))
    if content_chars < 0:
        raise ValueError("target_bytes_per_turn is smaller than the message JSON overhead")
    message = {"role": "assistant", "content": "x" * content_chars}
    logical_bytes = len(canonical_json_bytes(message))
    if logical_bytes < target_bytes_per_turn:
        raise AssertionError("message construction did not reach the requested logical size")
    return message, logical_bytes, content_chars


def seed_a12(store, database: str, turns: int, target_bytes_per_turn: int) -> dict[str, Any]:
    started_at = time.perf_counter()
    create_fixture_session(store, "s")
    message, logical_message_bytes, content_chars = a12_message(target_bytes_per_turn)
    rows = []
    turn_ids = []
    for index in range(turns):
        tid = f"history-{index:03d}"
        turn_ids.append(tid)
        started = turn_record(store, database, tid, f"history turn {index}")
        payload = deepcopy(started.payload)
        definition = payload["definition"]
        definition["task"]["initial_messages"].append(deepcopy(message))
        payload["definition"] = definition
        rows.append(
            make_record_before_publication(
                "turn_started",
                session_id="s",
                turn_id=tid,
                payload=payload,
            )
        )
        rows.append(
            make_record_before_publication(
                "turn_ended",
                session_id="s",
                turn_id=tid,
                payload=record("turn_ended", tid=tid).payload | {"budget": DEFAULT_BUDGET},
            )
        )
    append_fixture_records(store, "s", rows, commit_id="a12-history")
    push_fixture_input(store, "s", InboxItem("input", "user", {"content": "go"}))

    def hook(point, value):
        if point == "after_commit" and value.kind == "op_completed":
            raise ReceiptCut

    with suppress(ReceiptCut):
        drive(
            store,
            "s",
            runtime=_benchmark_runtime(
                database,
                [LLMResponse("retained answer")],
                content_reader=fixture_content_reader(store, "s"),
                hook=hook,
            ),
        )
    clear_fixture_state_cache(store)
    state, hydrated, _ = read_fixture_state(store, "s")
    assert state.active_turn_id is not None and state.phase == "active"
    assert sum(row.record.kind == "turn_ended" for row in hydrated) == turns
    _, raw_rows, _ = store.read_state("s")
    logical = tuple(row.record for row in raw_rows)
    base_head = store.read("s").head_seq
    reset_row = store.connection.execute(
        "SELECT head_seq,terminal_seq,level,handler_version,required_capabilities,active_turn_id,"
        "next_drive_ms,drive_retry_at_ms,phase FROM sk_session WHERE session_id='s'"
    ).fetchone()
    assert reset_row is not None
    consumed = tuple(InboxItem(**row.payload["input"]) for row in logical if row.kind == "input_applied")
    return {
        "setup_ms": (time.perf_counter() - started_at) * 1000,
        "active_turn_id": state.active_turn_id,
        "base_head": base_head,
        "reset_state": reset_row,
        "logical": logical,
        "consumed": consumed,
        "logical_records": len(logical),
        "logical_history_bytes_per_turn": logical_message_bytes,
        "logical_history_bytes_total": logical_message_bytes * turns,
        "message_plaintext_bytes": len(canonical_json_bytes(message)),
        "message_string_bytes": len(message["content"].encode()),
        "message_content_chars": content_chars,
        "turn_ids": turn_ids,
    }


def reset_measured_records(store, base_head: int, reset_state) -> dict[str, int]:
    with store.connection.transaction():
        links = store.connection.execute(
            "DELETE FROM sk_content_link WHERE session_id='s' AND record_seq>%s",
            (base_head,),
        ).rowcount
        checkpoints = store.connection.execute(
            "DELETE FROM sk_checkpoint WHERE session_id='s' AND seq>%s",
            (base_head,),
        ).rowcount
        records = store.connection.execute(
            "DELETE FROM sk_record WHERE session_id='s' AND seq>%s",
            (base_head,),
        ).rowcount
        commits = store.connection.execute(
            "DELETE FROM sk_commit WHERE session_id='s' AND head_seq>%s",
            (base_head,),
        ).rowcount
        store.connection.execute(
            "UPDATE sk_session SET head_seq=%s,terminal_seq=%s,level=%s,handler_version=%s,"
            "required_capabilities=%s,active_turn_id=%s,next_drive_ms=%s,drive_retry_at_ms=%s,phase=%s "
            "WHERE session_id='s'",
            reset_state,
        )
    clear_fixture_state_cache(store)
    return {"content_links": links, "checkpoints": checkpoints, "records": records, "commits": commits}


def _append_once(store, lease, base_head: int, index: int, addition) -> None:
    with store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=base_head,
            commit_id=f"a12-append/{index}",
            records=(addition,),
        )


def _drive_only(store, database: str) -> None:
    rt = _benchmark_runtime(database, [], content_reader=fixture_content_reader(store, "s"))
    with patch.object(rt.llm, "complete", wraps=rt.llm.complete) as completion:
        drive(store, "s", runtime=rt)
    assert completion.call_count == 0


def _drive_once(store, database: str) -> None:
    _drive_only(store, database)
    state, _, _ = store.read_state("s")
    assert state.active_turn_id is None


def restart_postgres(pg_ctl: Path, pgdata: Path, pglog: Path) -> float:
    begin = time.perf_counter()
    subprocess.run(
        [str(pg_ctl), "-D", str(pgdata), "-l", str(pglog), "-m", "fast", "restart"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    return (time.perf_counter() - begin) * 1000


def benchmark_a12(
    store,
    database: str,
    *,
    turns: int,
    history_mb_per_turn: int,
    samples: int,
    cold_mode: str,
    cold_samples: int,
    pg_ctl: Path | None,
    pgdata: Path | None,
    pglog: Path | None,
) -> dict[str, Any]:
    target_bytes_per_turn = history_mb_per_turn * MIB
    seeded = seed_a12(store, database, turns, target_bytes_per_turn)
    base_head = seeded["base_head"]
    reset_state = seeded["reset_state"]
    logical = seeded["logical"]
    consumed = seeded["consumed"]
    storage = collect_database_metrics(store.connection)
    security = fixture_security(store.connection)

    clear_fixture_state_cache(store)
    read_state(store, "s")
    warm_cached_read = measure(lambda: read_state(store, "s"), samples)
    warm_uncached_values = []
    for _ in range(samples):
        clear_fixture_state_cache(store)
        duration, _ = elapsed(lambda: read_state(store, "s"))
        warm_uncached_values.append(duration)
    warm_uncached_read = timing(warm_uncached_values)
    warm_fold = measure(lambda: fold(logical, consumed_inputs=consumed), samples)

    lease = store.acquire("s", owner="a12-append", ttl_ms=600000)
    assert lease is not None
    warm_append_values = []
    append_reset = {"content_links": 0, "checkpoints": 0, "records": 0, "commits": 0}
    for index in range(samples):
        addition = record(
            "usage_observed",
            tid=seeded["active_turn_id"],
            meter_id=f"append-{index}",
            observation=index + 1,
        )
        duration, _ = elapsed(lambda index=index, addition=addition: _append_once(store, lease, base_head, index, addition))
        warm_append_values.append(duration)
        counts = reset_measured_records(store, base_head, reset_state)
        for key, value in counts.items():
            append_reset[key] += value
    store.release(lease)

    warm_drive_values = []
    drive_reset = {"content_links": 0, "checkpoints": 0, "records": 0, "commits": 0}
    for _ in range(samples):
        reset_measured_records(store, base_head, reset_state)
        duration, _ = elapsed(lambda: _drive_only(store, database))
        warm_drive_values.append(duration)
        state, _, _ = store.read_state("s")
        assert state.active_turn_id is None
        counts = reset_measured_records(store, base_head, reset_state)
        for key, value in counts.items():
            drive_reset[key] += value

    timings: dict[str, Any] = {
        "warm_cached_read": warm_cached_read,
        "warm_uncached_read": warm_uncached_read,
        "warm_fold": warm_fold,
        "warm_append": timing(warm_append_values),
        "warm_drive": timing(warm_drive_values),
    }
    result: dict[str, Any] = {
        "semantic": "S1-A12 complete per-turn history with one deduplicated message",
        "cold_mode": cold_mode,
        "cold_samples": cold_samples if cold_mode == "restart" else 0,
        "workload": {
            "protocol": _BENCHMARK_PROTOCOL,
            "semantic": "one fixed historical message referenced in full by every turn",
            "completed_turns": turns,
            "active_turn_after_history": 1,
            "target_history_bytes_per_turn": target_bytes_per_turn,
            "message": "one identical assistant JSON message nested in every historical turn definition",
            "content_dedup_key": "tenant/session/key_id/HMAC/labels",
            "logical_records": seeded["logical_records"],
            "logical_history_bytes_per_turn": seeded["logical_history_bytes_per_turn"],
            "logical_history_bytes_total": seeded["logical_history_bytes_total"],
            "message_plaintext_bytes_each": seeded["message_plaintext_bytes"],
            "message_string_bytes_each": seeded["message_string_bytes"],
            "message_content_chars_each": seeded["message_content_chars"],
            "message_links_expected": turns,
            "unique_messages_expected": 1,
            "setup_ms": seeded["setup_ms"],
            "base_head": base_head,
        },
        "timings": timings,
        "bytes_and_reuse": storage,
        "fixture_security": security,
        "reset_counts": {"append": append_reset, "warm_drive": drive_reset},
    }

    if cold_mode == "restart":
        assert pg_ctl is not None and pgdata is not None and pglog is not None
        cold_restarts = []
        cold_reads = []
        cold_drives = []
        for _ in range(cold_samples):
            cold_restarts.append(restart_postgres(pg_ctl, pgdata, pglog))
            with _open_store(database) as cold_store:
                clear_fixture_state_cache(cold_store)
                duration, _ = elapsed(lambda: read_state(cold_store, "s"))
                cold_reads.append(duration)
            cold_restarts.append(restart_postgres(pg_ctl, pgdata, pglog))
            with _open_store(database) as cold_store:
                duration, _ = elapsed(lambda: _drive_only(cold_store, database))
                cold_drives.append(duration)
                state, _, _ = cold_store.read_state("s")
                assert state.active_turn_id is None
            with _open_store(database) as repair_store:
                reset_measured_records(repair_store, base_head, reset_state)
        timings["cold_read"] = timing(cold_reads)
        timings["cold_drive"] = timing(cold_drives)
        timings["postgres_restart_ms"] = cold_restarts
        result["cold_method"] = (
            "PostgreSQL cluster restarted immediately before each cold read/drive. "
            "PostgreSQL shared buffers are cold on the first access; OS page cache and CPU caches are not flushed."
        )
    else:
        result["cold_method"] = (
            "Cold mode not run. New connections and cleared Python state do not denote cold OS or PostgreSQL caches."
        )
    return result


def benchmark_history(store, database: str, size: int, samples: int) -> dict[str, Any]:
    tid, logical = _seed_signed_history(store, database, size)
    consumed = tuple(InboxItem(**r.payload["input"]) for r in logical if r.kind == "input_applied")
    storage = collect_database_metrics(store.connection)
    security = fixture_security(store.connection)
    lease = store.acquire("s", owner="append-benchmark", ttl_ms=600000)
    assert lease is not None
    addition = record("usage_observed", tid=tid, meter_id="measurement", observation=1)
    appended, restore = [], []
    for index in range(samples):
        begin = time.perf_counter()
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=size, commit_id=f"measurement/{index}", records=(addition,))
        appended.append((time.perf_counter() - begin) * 1000)
        begin = time.perf_counter()
        with store.connection.transaction():
            assert (
                store.connection.execute("DELETE FROM sk_checkpoint WHERE session_id='s' AND seq=%s", (size + 1,)).rowcount == 1
            )
            assert store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq=%s", (size + 1,)).rowcount == 1
            assert (
                store.connection.execute(
                    "DELETE FROM sk_commit WHERE session_id='s' AND commit_id=%s", (f"measurement/{index}",)
                ).rowcount
                == 1
            )
            store.connection.execute("UPDATE sk_session SET head_seq=%s WHERE session_id='s'", (size,))
        restore.append((time.perf_counter() - begin) * 1000)
    assert store.read("s").head_seq == size
    append = timing(appended)
    store.release(lease)
    rt = _benchmark_runtime(database, [])
    lease = store.acquire("s", owner="steady-benchmark", ttl_ms=rt.ttl_ms)
    assert lease is not None and rt.ttl_ms == 15000 and rt.heartbeat_seconds == 0.25
    scope = _Scope(lease, rt)
    scope.start()
    steady_values = []
    try:
        assert scope.ready.wait(5), "lease heartbeat did not become ready"
        read_state(store, "s")
        for index in range(samples):
            addition = record("usage_observed", tid=tid, meter_id="steady", observation=index + 1)
            before = time.perf_counter()
            with scope.committing() as current_lease, store.atomic() as tx:
                tx.append(
                    "s",
                    lease=current_lease,
                    expected_seq=size + index,
                    commit_id=f"steady/{index}",
                    records=(addition,),
                )
            steady_values.append((time.perf_counter() - before) * 1000)
        assert not scope.lost
    finally:
        cleanup = suppress(Exception) if sys.exc_info()[0] is not None else nullcontext()
        with cleanup:
            try:
                scope.close()
            finally:
                store.release(scope.lease)
    steady = timing(steady_values) | {"lease_ttl_ms": rt.ttl_ms, "heartbeat_seconds": rt.heartbeat_seconds}
    with store.connection.transaction():
        store.connection.execute("DELETE FROM sk_checkpoint WHERE session_id='s' AND seq>%s", (size,))
        store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq>%s", (size,))
        store.connection.execute("DELETE FROM sk_commit WHERE session_id='s' AND head_seq>%s", (size,))
        store.connection.execute("UPDATE sk_session SET head_seq=%s WHERE session_id='s'", (size,))
    full_fold = measure(lambda: fold(logical, consumed_inputs=consumed), samples)

    runtime_reload_read = measure(lambda: _runtime_reload_read(database), samples)
    reload_drive = []
    reload_failures: list[dict[str, Any]] = []
    for index in range(min(samples, 3)):
        begin = time.perf_counter()
        try:
            _runtime_reload_drive(database)
        except LeaseLost as exc:
            reload_failures.append({"sample": index, "error": type(exc).__name__, "reason": str(exc)})
        reload_drive.append((time.perf_counter() - begin) * 1000)
        state, recovered, _ = read_fixture_state(store, "s")
        ends = [r.record for r in recovered if r.record.kind == "turn_ended"]
        if ends:
            assert state.active_turn_id is None and len(ends) == 1
            assert ends[0].payload["status"] == "completed" and ends[0].payload["result"] == "retained answer"
        else:
            assert any(f["sample"] == index for f in reload_failures), "drive returned without terminal or LeaseLost"
            reload_failures[-1]["elapsed_ms"] = reload_drive[-1]
        with store.connection.transaction():
            store.connection.execute("DELETE FROM sk_content_link WHERE session_id='s' AND record_seq>%s", (size,))
            store.connection.execute("DELETE FROM sk_checkpoint WHERE session_id='s' AND seq>%s", (size,))
            store.connection.execute("DELETE FROM sk_record WHERE session_id='s' AND seq>%s", (size,))
            store.connection.execute("DELETE FROM sk_commit WHERE session_id='s' AND head_seq>%s", (size,))
            store.connection.execute("UPDATE sk_session SET head_seq=%s,terminal_seq=0 WHERE session_id='s'", (size,))
        store.rebuild_schedule("s")
    state, rows, _ = read_state(store, "s")
    assert state.phase == "active" and len(rows) == size
    return {
        "semantic": "synthetic capacity prefix; this is not an A12 content-history workload",
        "records": size,
        "logical_bytes": sum(len(r.encode()) for r in logical),
        "target_history_bytes_per_turn": None,
        "append": append,
        "steady_append": steady,
        "full_fold": full_fold,
        "runtime_reload_read": runtime_reload_read
        | {"buffers": "PostgreSQL and OS buffers warm; Python runtime and connection are new per sample"},
        "runtime_reload_drive": timing(reload_drive)
        | {
            "buffers": "PostgreSQL and OS buffers warm; Python runtime and connection are new per sample",
            "failures": reload_failures,
        },
        "reset_samples_ms": restore,
        "provider_call_increment": 0,
        "bytes_and_reuse": storage,
        "fixture_security": security,
    }


def _runtime_reload_read(database: str) -> None:
    with _open_store(database) as store:
        read_state(store, "s")


def _runtime_reload_drive(database: str) -> None:
    with _open_store(database) as store:
        _drive_once(store, database)


def benchmark_runnable(store, size: int, samples: int) -> dict[str, Any]:
    with store.atomic() as tx:
        for index in range(size):
            sid = f"catalog/{index:05d}"
            tx.create(
                SessionSpec(
                    sid,
                    principal("catalog/benchmark", "service"),
                    content_ref(f"{sid}/workspace"),
                    attributes=content_ref(f"{sid}/attributes"),
                ),
                consumers=("events",),
            )
    for index in range(0, size, 2):
        push_fixture_input(store, f"catalog/{index:05d}", InboxItem("input", "user", {"content": "go"}))
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


def create_database(admin_dsn: str, name: str) -> None:
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def drop_database(admin_dsn: str, name: str) -> None:
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        try:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        except psycopg.errors.SyntaxError:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))


@contextmanager
def disposable_database(admin_dsn: str, cleanup: dict[str, int]) -> Iterator[str]:
    name = f"vvsk_bench_{uuid4().hex[:20]}"
    create_database(admin_dsn, name)
    cleanup["database_created"] += 1
    try:
        yield make_conninfo(admin_dsn, dbname=name)
    finally:
        drop_database(admin_dsn, name)
        cleanup["database_dropped"] += 1


def server_metadata(admin_dsn: str) -> dict[str, Any]:
    with psycopg.connect(admin_dsn) as admin:
        version = admin.execute("SELECT version()").fetchone()
        setting_names = (
            "server_version",
            "server_version_num",
            "server_encoding",
            "shared_buffers",
            "work_mem",
            "maintenance_work_mem",
            "effective_cache_size",
            "fsync",
            "synchronous_commit",
            "full_page_writes",
            "jit",
            "max_connections",
            "max_wal_size",
            "checkpoint_timeout",
        )
        settings = dict(
            admin.execute(
                "SELECT name,setting FROM pg_settings WHERE name=ANY(%s)",
                (list(setting_names),),
            ).fetchall()
        )
        database = admin.execute("SELECT datcollate,datctype FROM pg_database WHERE datname=current_database()").fetchone()
    assert version is not None and database is not None and set(settings) == set(setting_names)
    settings["datcollate"], settings["datctype"] = database
    return {
        "version": version[0],
        "config": settings,
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_bytes(*args: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def code_provenance() -> dict[str, Any]:
    """Bind benchmark evidence to the commit and exact dirty source inputs."""

    revision = _git("rev-parse", "HEAD")
    source_status = (_git("status", "--porcelain=v1") or "").splitlines()
    fingerprint = sha256()
    fingerprint.update((revision or "unknown").encode())
    fingerprint.update(b"\0tracked-diff\0")
    fingerprint.update(_git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--") or b"")
    untracked = _git_bytes("ls-files", "--others", "--exclude-standard", "-z", "--", "src", "scripts") or b""
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        path = ROOT / relative
        fingerprint.update(relative.encode("utf-8", "surrogateescape"))
        fingerprint.update(b"\0")
        if path.is_file():
            fingerprint.update(path.read_bytes())
        fingerprint.update(b"\0")
    dirty = bool(source_status)
    digest = fingerprint.hexdigest()
    return {
        "revision": revision,
        "dirty": dirty,
        "dirty_fingerprint": digest if dirty else None,
        "fingerprint": digest,
        "source_status": source_status,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def workload_signature(workload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol",
        "semantic",
        "completed_turns",
        "active_turn_after_history",
        "target_history_bytes_per_turn",
        "logical_records",
        "logical_history_bytes_per_turn",
        "logical_history_bytes_total",
        "message_plaintext_bytes_each",
        "message_string_bytes_each",
        "message_content_chars_each",
        "message_links_expected",
        "unique_messages_expected",
        "content_dedup_key",
        "message",
    )
    return {key: workload.get(key) for key in keys}


def _timing_value(timing_value: Any, field: str) -> float | None:
    if isinstance(timing_value, dict):
        value = timing_value.get(field)
    elif field == "median_ms" and isinstance(timing_value, (int, float)):
        value = timing_value
    else:
        value = None
    return float(value) if isinstance(value, (int, float)) else None


def _postgres_signature(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {"version": value.get("version"), "config": value.get("config")}


def _code_evidence_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    revision = value.get("revision")
    fingerprint = value.get("fingerprint")
    dirty = value.get("dirty")
    dirty_fingerprint = value.get("dirty_fingerprint")
    return (
        isinstance(revision, str)
        and bool(revision)
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and isinstance(dirty, bool)
        and (not dirty or (isinstance(dirty_fingerprint, str) and len(dirty_fingerprint) == 64))
    )


def _comparison_identity(payload: dict[str, Any]) -> dict[str, Any]:
    root = payload.get("a12", payload)
    workload = workload_signature(root.get("workload", {}))
    cold_mode = payload.get("cold_mode", root.get("cold_mode"))
    cold_samples = payload.get("cold_samples", root.get("cold_samples", 0))
    return {
        "protocol": workload.get("protocol"),
        "workload": workload,
        "warm_samples": payload.get("samples", root.get("samples")),
        "cold_mode": cold_mode,
        "cold_samples": cold_samples if cold_mode == "restart" else 0,
        "postgres": _postgres_signature(payload.get("postgres", root.get("postgres"))),
        "code": payload.get("code", root.get("code")),
    }


def _timing_diagnostics(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name in sorted(set(current) | set(baseline)):
        if name in _A12_STANDARD_TIMINGS:
            continue
        diagnostics[name] = {"baseline": baseline.get(name), "current": current.get(name)}
    return diagnostics


def compare_a12_baseline(current: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    baseline_root = baseline.get("a12", baseline)
    current_identity = _comparison_identity(current)
    baseline_identity = _comparison_identity(baseline)
    checks["comparable_protocol"] = (
        current_identity["protocol"] is not None and current_identity["protocol"] == baseline_identity["protocol"]
    )
    checks["comparable_workload"] = current_identity["workload"] == baseline_identity["workload"]
    checks["comparable_warm_samples"] = current_identity["warm_samples"] == baseline_identity["warm_samples"]
    checks["comparable_cold_mode"] = (
        current_identity["cold_mode"] is not None and current_identity["cold_mode"] == baseline_identity["cold_mode"]
    )
    checks["comparable_cold_samples"] = current_identity["cold_samples"] == baseline_identity["cold_samples"]
    checks["comparable_postgres"] = (
        current_identity["postgres"] is not None and current_identity["postgres"] == baseline_identity["postgres"]
    )
    checks["code_provenance_complete"] = _code_evidence_complete(current_identity["code"]) and _code_evidence_complete(
        baseline_identity["code"]
    )
    mismatch_labels = (
        ("comparable_protocol", "protocol"),
        ("comparable_workload", "workload"),
        ("comparable_warm_samples", "warm sample count"),
        ("comparable_cold_mode", "cold mode"),
        ("comparable_cold_samples", "cold sample count"),
        ("comparable_postgres", "PostgreSQL server/config"),
        ("code_provenance_complete", "code revision/dirty fingerprint"),
    )
    for check, label in mismatch_labels:
        if not checks[check]:
            failures.append(f"incomparable {label}")
    diagnostics = _timing_diagnostics(current.get("timings", {}), baseline_root.get("timings", {}))
    comparison_summary = {
        "baseline_revision": baseline.get("revision", baseline_root.get("revision")),
        "current_revision": current.get("revision"),
        "baseline_code": baseline_identity["code"],
        "current_code": current_identity["code"],
        "baseline_signature": baseline_identity,
        "current_signature": current_identity,
        "tolerance": tolerance,
        "checks": checks,
        "diagnostics": diagnostics,
        "time_checks": {},
        "storage_checks": {},
        "message_reuse": {},
        "passed": False,
        "failures": failures,
    }
    if failures:
        return comparison_summary

    timing_checks: dict[str, Any] = {}
    for name, baseline_timing in baseline_root.get("timings", {}).items():
        if name not in _A12_STANDARD_TIMINGS:
            continue
        current_timing = current.get("timings", {}).get(name)
        baseline_median = _timing_value(baseline_timing, "median_ms")
        current_median = _timing_value(current_timing, "median_ms")
        baseline_max = _timing_value(baseline_timing, "max_ms")
        current_max = _timing_value(current_timing, "max_ms")
        median_ok = baseline_median is not None and current_median is not None and current_median <= baseline_median * tolerance
        max_ok = baseline_max is None or (current_max is not None and current_max <= baseline_max * tolerance)
        timing_checks[name] = {
            "baseline_median_ms": baseline_median,
            "current_median_ms": current_median,
            "median_ratio": (
                current_median / baseline_median if current_median is not None and baseline_median not in (None, 0) else None
            ),
            "baseline_observed_max_ms": baseline_max,
            "current_observed_max_ms": current_max,
            "max_ratio": (current_max / baseline_max if current_max is not None and baseline_max not in (None, 0) else None),
            "median_not_greater": median_ok,
            "observed_max_not_greater": max_ok,
        }
        checks[f"timing:{name}"] = median_ok and max_ok
        if not checks[f"timing:{name}"]:
            failures.append(f"timing regression: {name}")

    storage_checks: dict[str, Any] = {}
    paths = {
        "record_table_bytes_including_toast": ("sql_records", "table", "table_bytes_including_toast"),
        "record_total_bytes": ("sql_records", "table", "total_bytes"),
        "content_table_bytes_including_toast": ("content_store", "table", "table_bytes_including_toast"),
        "content_total_bytes": ("content_store", "table", "total_bytes"),
        "content_link_total_bytes": ("content_store", "relations", "sk_content_link_total_bytes"),
    }

    def dig(value: dict[str, Any], path: tuple[str, ...]) -> Any:
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    baseline_storage = baseline_root.get("bytes_and_reuse", {})
    current_storage = current.get("bytes_and_reuse", {})
    for label, path in paths.items():
        baseline_value = dig(baseline_storage, path)
        if not isinstance(baseline_value, int):
            continue
        current_value = dig(current_storage, path)
        accepted = isinstance(current_value, int) and current_value <= baseline_value * tolerance
        storage_checks[label] = {
            "baseline": baseline_value,
            "current": current_value,
            "current_not_greater": accepted,
        }
        checks[f"storage:{label}"] = accepted
        if not accepted:
            failures.append(f"storage regression: {label}")

    baseline_reuse = baseline_storage.get("message_reuse", {}).get("reuse_ratio")
    current_reuse = current_storage.get("message_reuse", {}).get("reuse_ratio")
    reuse_ok = (
        isinstance(baseline_reuse, (int, float))
        and isinstance(current_reuse, (int, float))
        and current_reuse >= baseline_reuse / tolerance
    )
    checks["message_reuse_not_lower"] = reuse_ok
    if not reuse_ok:
        failures.append(f"message reuse regression: current={current_reuse} baseline={baseline_reuse}")

    comparison_summary.update(
        {
            "time_checks": timing_checks,
            "storage_checks": storage_checks,
            "message_reuse": {"baseline": baseline_reuse, "current": current_reuse, "current_not_lower": reuse_ok},
            "passed": not failures,
            "failures": failures,
        }
    )
    return comparison_summary


def _timing_within_limit(value: Any, limit_ms: float) -> bool:
    if not isinstance(value, dict):
        return False
    samples = value.get("samples_ms")
    n = value.get("n")
    minimum = value.get("min_ms")
    median = value.get("median_ms")
    maximum = value.get("max_ms")
    return (
        isinstance(samples, list)
        and isinstance(n, int)
        and n > 0
        and len(samples) == n
        and all(isinstance(sample, (int, float)) and sample >= 0 for sample in samples)
        and isinstance(minimum, (int, float))
        and isinstance(median, (int, float))
        and isinstance(maximum, (int, float))
        and 0 <= minimum <= median <= maximum <= limit_ms
    )


def capacity_failures(payload: dict[str, Any], *, a12_requested: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for case in payload["history"]:
        size = case["records"]
        checks[f"history:{size}:reload_no_lease_failures"] = not case["runtime_reload_drive"]["failures"]
        if size >= 2000:
            checks[f"history:{size}:reload_bound"] = case["runtime_reload_drive"]["max_ms"] <= size
        checks[f"history:{size}:steady_append"] = case["steady_append"]["median_ms"] <= 50
        checks[f"history:{size}:signed_fixture"] = case["fixture_security"]["all_inputs_signed"]
        checks[f"history:{size}:encrypted_fixture"] = case["fixture_security"]["all_content_encrypted"]
    for case in payload["runnable"]:
        checks[f"runnable:{case['sessions']}:full_scan"] = case["full_scan"]["max_ms"] <= 1000
    if a12_requested:
        case = payload["a12"]
        workload = case["workload"]
        reuse = case["bytes_and_reuse"]["message_reuse"]
        logical_total = workload["logical_history_bytes_total"]
        logical_per_turn = workload["logical_history_bytes_per_turn"]
        target_per_turn = workload["target_history_bytes_per_turn"]
        completed_turns = workload["completed_turns"]
        checks["a12:protocol"] = workload["protocol"] == _BENCHMARK_PROTOCOL
        checks["a12:logical_per_turn_target"] = logical_per_turn >= target_per_turn
        checks["a12:logical_history_total"] = logical_total == logical_per_turn * completed_turns
        checks["a12:one_physical_message"] = reuse["unique_content_rows"] == 1
        checks["a12:all_turn_links"] = reuse["link_rows"] == completed_turns
        checks["a12:linked_logical_bytes"] = reuse["linked_logical_plaintext_bytes"] == logical_total
        checks["a12:physical_message_bytes"] = reuse["physical_plaintext_size_sum"] == workload["message_plaintext_bytes_each"]
        checks["a12:reuse_ratio"] = reuse["reuse_ratio"] == completed_turns
        checks["a12:signed_fixture"] = case["fixture_security"]["all_inputs_signed"]
        checks["a12:encrypted_fixture"] = case["fixture_security"]["all_content_encrypted"]
        relations = case["bytes_and_reuse"]["relations"]
        checks["a12:storage_relations"] = all(
            isinstance(relations.get(name, {}).get("total_bytes"), int) and relations[name]["total_bytes"] > 0
            for name in ("sk_record", "sk_content", "sk_content_link")
        )
        for name in ("warm_cached_read", "warm_uncached_read", "warm_fold", "warm_append", "warm_drive"):
            checks[f"a12:timing:{name}"] = _timing_within_limit(case["timings"].get(name), _A12_WARM_TIMING_LIMIT_MS)
        if workload.get("cold_mode") == "restart" or payload.get("cold_mode") == "restart":
            for name in ("cold_read", "cold_drive"):
                checks[f"a12:timing:{name}"] = _timing_within_limit(case["timings"].get(name), _A12_COLD_TIMING_LIMIT_MS)
            restarts = case["timings"].get("postgres_restart_ms", [])
            checks["a12:restart_diagnostics"] = (
                isinstance(restarts, list)
                and len(restarts) == 2 * case["cold_samples"]
                and all(isinstance(value, (int, float)) and value >= 0 for value in restarts)
            )
    cleanup = payload.get("cleanup", {})
    checks["cleanup:databases"] = cleanup.get("database_created") == cleanup.get("database_dropped")
    return checks


def _safe_exception(exc: BaseException, admin_dsn: str) -> str:
    message = str(exc)
    password = conninfo_to_dict(admin_dsn).get("password")
    if password:
        message = message.replace(str(password), "<redacted>")
    return message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "With no workload selector, the command runs the legacy synthetic capacity suite. "
            "Examples:\n"
            "  session_kernel_benchmark.py --sizes 100 --samples 1\n"
            "  session_kernel_benchmark.py --turns 100 --history-mb-per-turn 10 --samples 7\n\n"
            "The admin DSN defaults to VV_AGENT_TEST_POSTGRES_DSN or dbname=postgres."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/kernel-m5/performance.json"))
    parser.add_argument("--sizes", nargs="+", type=int, help="legacy synthetic record-history sizes")
    parser.add_argument("--sessions", nargs="*", type=int, help="legacy runnable-catalog sizes")
    parser.add_argument("--turns", type=int, help="completed historical turns for the A12 workload")
    parser.add_argument(
        "--history-mb-per-turn",
        type=int,
        help="A12 logical history MiB referenced in full by every turn (integer)",
    )
    parser.add_argument("--samples", type=int, default=7, help="warm timing samples (default: 7)")
    parser.add_argument(
        "--cold-mode",
        choices=("none", "restart"),
        default="none",
        help="'restart' measures first access after a real PostgreSQL cluster restart",
    )
    parser.add_argument("--cold-samples", type=int, default=3, help="restart-mode cold samples (default: 3)")
    parser.add_argument("--pg-ctl", type=Path, help="pg_ctl path required by --cold-mode restart")
    parser.add_argument("--pgdata", type=Path, help="PostgreSQL data directory required by --cold-mode restart")
    parser.add_argument("--pg-log", type=Path, help="PostgreSQL restart log required by --cold-mode restart")
    parser.add_argument("--dsn", help="admin PostgreSQL DSN; credentials are never written to the output")
    parser.add_argument("--assert-capacity", action="store_true", help="fail on structural/capacity violations")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="explicit A12 baseline JSON; comparison failure returns nonzero",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=1.0,
        help="maximum timing/storage ratio against the baseline (default: 1.0)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for flag in ("sizes", "sessions"):
        values = getattr(args, flag)
        if values is not None and any(value <= 0 for value in values):
            parser.error(f"--{flag} values must be positive")
    for flag in ("turns", "history_mb_per_turn", "samples", "cold_samples"):
        value = getattr(args, flag)
        if value is not None and value <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if (args.turns is None) != (args.history_mb_per_turn is None):
        parser.error("--turns and --history-mb-per-turn must be supplied together")
    args.a12_requested = args.turns is not None and args.history_mb_per_turn is not None
    if args.cold_mode == "restart" and not args.a12_requested:
        parser.error("--cold-mode restart is available only with --turns/--history-mb-per-turn")
    if args.cold_mode == "restart" and not all((args.pg_ctl, args.pgdata, args.pg_log)):
        parser.error("--cold-mode restart requires --pg-ctl, --pgdata and --pg-log")
    if args.regression_tolerance < 1.0:
        parser.error("--regression-tolerance must be at least 1.0")
    if args.sizes is None and args.sessions is None and not args.a12_requested:
        args.sizes = list(DEFAULT_SIZES)
        args.sessions = list(DEFAULT_SESSIONS)
    else:
        args.sizes = [] if args.sizes is None else args.sizes
        args.sessions = [] if args.sessions is None else args.sessions
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    admin_dsn = args.dsn or os.environ.get("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres")
    cleanup = {"database_created": 0, "database_dropped": 0}
    code = code_provenance()
    payload: dict[str, Any] = {
        "schema_version": 3,
        "mode": "a12" if args.a12_requested else "capacity",
        "protocol": _BENCHMARK_PROTOCOL,
        "revision": code["revision"],
        "source_status": code["source_status"],
        "script_sha256": code["script_sha256"],
        "code": code,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_unix_s": time.time(),
        "samples": args.samples,
        "cold_mode": args.cold_mode,
        "cold_samples": args.cold_samples if args.cold_mode == "restart" else 0,
        "history": [],
        "runnable": [],
        "failures": [],
        "cleanup": cleanup,
    }
    try:
        payload["postgres"] = server_metadata(admin_dsn)
        for size in args.sizes:
            with disposable_database(admin_dsn, cleanup) as database:
                with _open_store(database, install_schema=True) as store:
                    value = benchmark_history(store, database, size, args.samples)
                payload["history"].append(value)
        for size in args.sessions:
            with disposable_database(admin_dsn, cleanup) as database:
                with _open_store(database, install_schema=True) as store:
                    value = benchmark_runnable(store, size, args.samples)
                payload["runnable"].append(value)
        if args.a12_requested:
            with disposable_database(admin_dsn, cleanup) as database:
                with _open_store(database, install_schema=True) as store:
                    value = benchmark_a12(
                        store,
                        database,
                        turns=args.turns,
                        history_mb_per_turn=args.history_mb_per_turn,
                        samples=args.samples,
                        cold_mode=args.cold_mode,
                        cold_samples=args.cold_samples,
                        pg_ctl=args.pg_ctl,
                        pgdata=args.pgdata,
                        pglog=args.pg_log,
                    )
                payload["a12"] = value
                payload["workload"] = value["workload"]
                payload["timings"] = value["timings"]
                payload["bytes_and_reuse"] = value["bytes_and_reuse"]
                payload["fixture_security"] = value["fixture_security"]
                payload["cold_method"] = value["cold_method"]
        checks = capacity_failures(payload, a12_requested=args.a12_requested)
        if args.baseline_json is not None:
            baseline = json.loads(args.baseline_json.read_text())
            if not args.a12_requested:
                raise ValueError("--baseline-json currently requires the A12 --turns/--history-mb-per-turn workload")
            comparison = compare_a12_baseline(payload, baseline, args.regression_tolerance)
            payload["comparison"] = comparison
            checks["baseline:passed"] = comparison["passed"]
        failed = [name for name, passed in checks.items() if not passed]
        payload["hard_gate"] = {"passed": not failed, "checks": checks, "failures": failed}
    except BaseException as exc:
        payload["failures"].append({"type": type(exc).__name__, "message": _safe_exception(exc, admin_dsn)})
        raise
    finally:
        payload["finished_unix_s"] = time.time()
        payload["duration_s"] = payload["finished_unix_s"] - payload["started_unix_s"]
        payload["cleanup"] = cleanup
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if "comparison" in payload:
        print(json.dumps(payload["comparison"], indent=2, sort_keys=True), flush=True)
    assert_required = args.assert_capacity or args.baseline_json is not None
    if assert_required and not payload["hard_gate"]["passed"]:
        print(json.dumps(payload["hard_gate"], indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
