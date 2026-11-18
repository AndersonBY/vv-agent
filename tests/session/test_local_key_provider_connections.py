"""Connection lifecycle regressions for the development local key provider."""

from __future__ import annotations

import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vv_agent.kernel._content import KeyDestroyStatus, KeyUnavailable
from vv_agent.session import content_store
from vv_agent.session.content_store import LocalKeyProvider

MASTER_KEY = b"k" * 32
TENANT = "tenant-a"
SESSION = "session-a"


def test_first_initialization_creates_nested_state_directory(tmp_path: Path) -> None:
    state_path = tmp_path / "new" / "nested" / "state.sqlite3"
    provider = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    try:
        wrapped = provider.generate_data_key(TENANT, SESSION)
        assert len(provider.unwrap(TENANT, SESSION, wrapped)) == 32
        assert (tmp_path / "new").is_dir()
        assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    finally:
        provider.close()

    state_path.unlink()


def test_unwrap_uses_one_provider_owned_state_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "key-state.sqlite3"
    real_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def counting_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(content_store.sqlite3, "connect", counting_connect)
    provider = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    try:
        assert len(connections) == 1
        wrapped = provider.generate_data_key(TENANT, SESSION)
        expected = provider.unwrap(TENANT, SESSION, wrapped)

        for _ in range(50):
            assert provider.unwrap(TENANT, SESSION, wrapped) == expected
            provider.restore_data_key(wrapped)

        assert len(connections) == 1
        outcome = provider.destroy(TENANT, wrapped.key_id)
        assert outcome.status is KeyDestroyStatus.DESTROYED
        assert len(connections) == 1
        with pytest.raises(KeyUnavailable):
            provider.unwrap(TENANT, SESSION, wrapped)
        assert len(connections) == 1
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    finally:
        provider.close()

    provider.close()
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")
    state_path.unlink()
    assert not state_path.exists()


def test_destroy_is_visible_across_instances_and_remains_idempotent(tmp_path: Path) -> None:
    state_path = tmp_path / "shared-key-state.sqlite3"
    owner = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    observer = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    try:
        wrapped = owner.generate_data_key(TENANT, SESSION)
        assert observer.unwrap(TENANT, SESSION, wrapped)
        observer.restore_data_key(wrapped)

        mismatch = observer.destroy("tenant-b", wrapped.key_id)
        assert mismatch.status is KeyDestroyStatus.BINDING_MISMATCH
        assert observer.unwrap(TENANT, SESSION, wrapped)

        destroyed = owner.destroy(TENANT, wrapped.key_id)
        assert destroyed.status is KeyDestroyStatus.DESTROYED
        with pytest.raises(KeyUnavailable):
            observer.unwrap(TENANT, SESSION, wrapped)
        with pytest.raises(KeyUnavailable):
            observer.restore_data_key(wrapped)

        repeated = owner.destroy(TENANT, wrapped.key_id)
        assert repeated.status is KeyDestroyStatus.ALREADY_DESTROYED
        assert repeated.evidence_id == destroyed.evidence_id
    finally:
        owner.close()
        observer.close()

    restarted = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    try:
        with pytest.raises(KeyUnavailable):
            restarted.unwrap(TENANT, SESSION, wrapped)
        with pytest.raises(KeyUnavailable):
            restarted.restore_data_key(wrapped)
    finally:
        restarted.close()


def test_wrong_master_key_rejects_state_and_closes_failed_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "fingerprinted-key-state.sqlite3"
    LocalKeyProvider(MASTER_KEY, state_path=state_path).close()
    real_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def recording_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(content_store.sqlite3, "connect", recording_connect)
    with pytest.raises(ValueError, match="different master key"):
        LocalKeyProvider(b"w" * 32, state_path=state_path)

    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_shared_connection_is_lock_serialized_across_threads(tmp_path: Path) -> None:
    state_path = tmp_path / "threaded-key-state.sqlite3"
    provider = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    barrier = threading.Barrier(3)

    def exercise(worker_index: int) -> None:
        barrier.wait(timeout=5)
        for round_index in range(8):
            tenant_id = f"tenant-{worker_index}"
            session_id = f"session-{round_index}"
            wrapped = provider.generate_data_key(tenant_id, session_id)
            assert len(provider.unwrap(tenant_id, session_id, wrapped)) == 32
            outcome = provider.destroy(tenant_id, wrapped.key_id)
            assert outcome.status is KeyDestroyStatus.DESTROYED
            with pytest.raises(KeyUnavailable):
                provider.unwrap(tenant_id, session_id, wrapped)

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(exercise, index) for index in range(3)]
            for future in futures:
                future.result(timeout=10)

        with provider._lock:
            connection = provider._state_connection()
            assert not connection.in_transaction
            assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
    finally:
        provider.close()

    state_path.unlink()
