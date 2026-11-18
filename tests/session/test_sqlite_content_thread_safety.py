"""Thread-safety regressions for durable content on a shared SQLite connection."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from vv_agent.kernel import _content as content
from vv_agent.kernel._records import SessionSpec
from vv_agent.session import content_store
from vv_agent.session.content_reader import DurableContentReader, record_owner
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

MASTER_KEY = b"k" * 32
TENANT = "tenant-a"
SESSION = "session-a"


def _content_ref(label: str) -> dict[str, Any]:
    payload = label.encode()
    return {
        "content_id": f"content/contract/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/contract",
        "labels": {"source": [], "classification": "internal"},
    }


def _principal(subject_id: str) -> dict[str, Any]:
    return {
        "tenant_id": TENANT,
        "subject_id": subject_id,
        "subject_kind": "user",
        "attributes": _content_ref(f"principal/{subject_id}"),
    }


@contextmanager
def _open_sqlite(path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        yield store


def _create_session(store: SQLiteStore) -> None:
    with store.for_tenant(TENANT).atomic() as tx:
        tx.create(
            SessionSpec(
                SESSION,
                _principal(f"user/{SESSION}"),
                _content_ref(f"workspace/{SESSION}"),
                attributes=_content_ref(f"attributes/{SESSION}"),
            ),
            consumers=(),
        )


def test_content_reads_and_transactional_writes_share_store_lock(tmp_path: Path) -> None:
    path = tmp_path / "content-isolation.sqlite"
    plaintext = b"published only after the writer commits"
    with _open_sqlite(path) as store:
        _create_session(store)
        provider = content_store.LocalKeyProvider(MASTER_KEY)
        published: list[content.ContentRef] = []
        write_finished = threading.Event()
        write_ready = threading.Event()
        read_attempted = threading.Event()
        read_finished = threading.Event()
        release_writer = threading.Event()
        errors: list[BaseException] = []
        result: list[bytes] = []

        def write() -> None:
            try:
                with store.atomic() as tx:
                    adapter = SQLiteContentStore(tx, provider)
                    ref = adapter.put(TENANT, SESSION, plaintext, content.ContentLabels())
                    adapter.link_record(TENANT, SESSION, 1, ref, "body")
                    published.append(ref)
                    write_ready.set()
                    if not release_writer.wait(timeout=5):
                        raise TimeoutError("test did not release the SQLite content writer")
            except BaseException as exc:
                errors.append(exc)
            finally:
                write_finished.set()

        writer = threading.Thread(target=write, name="content-writer")
        writer.start()
        assert write_ready.wait(timeout=5), "writer did not reach its paused transaction"

        adapter = SQLiteContentStore(store, provider)
        reader = DurableContentReader(adapter, tenant_id=TENANT, session_id=SESSION)

        def read() -> None:
            read_attempted.set()
            try:
                result.append(reader.read(TENANT, SESSION, record_owner(1), "body", published[0]))
            except BaseException as exc:
                errors.append(exc)
            finally:
                read_finished.set()

        reader_thread = threading.Thread(target=read, name="content-reader")
        reader_thread.start()
        assert read_attempted.wait(timeout=5), "reader did not attempt to read"
        read_during_transaction = read_finished.wait(timeout=0.2)
        release_writer.set()
        writer.join(timeout=5)
        reader_thread.join(timeout=5)

        assert not writer.is_alive()
        assert not reader_thread.is_alive()
        assert not errors
        assert not read_during_transaction, "reader observed an uncommitted content transaction"
        assert result == [plaintext]

        assert store._one("SELECT 1") == (1,)
        assert reader.read(TENANT, SESSION, record_owner(1), "body", published[0]) == plaintext


def test_content_writer_and_reader_threads_complete_repeated_rounds(tmp_path: Path) -> None:
    path = tmp_path / "content-concurrency.sqlite"
    anchor = b"stable owner content"
    rounds = 24
    with _open_sqlite(path) as store:
        _create_session(store)
        provider = content_store.LocalKeyProvider(MASTER_KEY)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            anchor_ref = adapter.put(TENANT, SESSION, anchor, content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, anchor_ref, "anchor")

        reader = DurableContentReader(
            SQLiteContentStore(store, provider),
            tenant_id=TENANT,
            session_id=SESSION,
        )
        done = threading.Event()
        started = threading.Event()
        errors: list[BaseException] = []
        read_count = 0
        counter_lock = threading.Lock()

        def write() -> None:
            try:
                for round_number in range(rounds):
                    with store.atomic() as tx:
                        adapter = SQLiteContentStore(tx, provider)
                        ref = adapter.put(
                            TENANT,
                            SESSION,
                            f"round-{round_number}".encode(),
                            content.ContentLabels(),
                        )
                        adapter.link_record(TENANT, SESSION, 1, ref, f"round-{round_number}")
                        started.set()
                        if round_number == 0:
                            time.sleep(0.02)
                    deadline = time.monotonic() + 5
                    while read_count <= round_number and time.monotonic() < deadline:
                        time.sleep(0.001)
                    if read_count <= round_number:
                        raise TimeoutError("reader did not run alongside a content write")
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        def read() -> None:
            nonlocal read_count
            if not started.wait(timeout=5):
                errors.append(TimeoutError("content writer did not start"))
                return
            try:
                while not done.is_set():
                    assert reader.read(TENANT, SESSION, record_owner(1), "anchor", anchor_ref) == anchor
                    with counter_lock:
                        read_count += 1
            except BaseException as exc:
                errors.append(exc)

        writer = threading.Thread(target=write, name="round-writer")
        read_thread = threading.Thread(target=read, name="round-reader")
        writer.start()
        read_thread.start()
        writer.join(timeout=10)
        done.set()
        read_thread.join(timeout=5)

        assert not writer.is_alive()
        assert not read_thread.is_alive()
        assert not errors
        assert read_count >= rounds
        assert (
            store._one("SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s", (TENANT, SESSION))[0]
            == rounds + 1
        )
        assert reader.read(TENANT, SESSION, record_owner(1), "anchor", anchor_ref) == anchor


def test_raw_connection_source_keeps_caller_transaction_ownership(tmp_path: Path) -> None:
    path = tmp_path / "content-raw-connection.sqlite"
    with _open_sqlite(path) as store:
        _create_session(store)

    connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.create_function("session_now_ms", 0, lambda: 1_000)
        connection.execute("BEGIN IMMEDIATE")
        adapter = SQLiteContentStore(connection, content_store.LocalKeyProvider(MASTER_KEY))
        adapter.put(TENANT, SESSION, b"caller owns rollback", content.ContentLabels())
        assert connection.in_transaction
        connection.execute("ROLLBACK")

        assert connection.execute(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=? AND session_id=?",
            (TENANT, SESSION),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=? AND session_id=?",
            (TENANT, SESSION),
        ).fetchone() == (0,)
    finally:
        connection.close()
