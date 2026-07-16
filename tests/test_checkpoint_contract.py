from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from threading import Thread

import pytest

from vv_agent.runtime.backends.celery import RuntimeRecipe
from vv_agent.runtime.backends.celery_tasks import run_single_cycle
from vv_agent.runtime.checkpoint_codec import checkpoint_from_dict, checkpoint_to_dict, checkpoint_to_json
from vv_agent.runtime.state import Checkpoint, CheckpointConflictError, InMemoryStateStore, StateStore
from vv_agent.runtime.stores.redis import RedisStateStore
from vv_agent.runtime.stores.sqlite import SqliteStateStore
from vv_agent.types import AgentResult, AgentStatus, AgentTask, Message

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "checkpoint_codec_v1.json"
RUST_REPO = Path(os.environ.get("VV_AGENT_RS_REPO", Path(__file__).parents[2] / "vv-agent-rs"))
RUST_FIXTURE_PATH = RUST_REPO / "crates" / "vv-agent" / "tests" / "fixtures" / "parity" / FIXTURE_PATH.name
FIXTURE_SHA256 = "e7be2cfafca7f741d32b4537cb003f0179f69162171432c17cd746a0ff2119cf"


class _FakeWatchError(Exception):
    pass


class _FakeRedisPipeline:
    def __init__(self, client: _FakeRedisClient) -> None:
        self._client = client
        self._commands: list[tuple[str, str, str | None]] = []
        self._transaction = False

    def __enter__(self) -> _FakeRedisPipeline:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def watch(self, _key: str) -> None:
        return None

    def unwatch(self) -> None:
        self._transaction = False
        self._commands.clear()

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def multi(self) -> None:
        self._transaction = True

    def set(self, key: str, value: str) -> None:
        if self._transaction:
            self._commands.append(("set", key, value))
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        if self._transaction:
            self._commands.append(("delete", key, None))
        else:
            self._client.delete(key)

    def execute(self) -> list[object]:
        results: list[object] = []
        for command, key, value in self._commands:
            if command == "set":
                assert value is not None
                results.append(self._client.set(key, value))
            else:
                results.append(self._client.delete(key))
        self._transaction = False
        self._commands.clear()
        return results


class _FakeRedisClient:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self.eval_server_now_ms: int | None = None
        self.eval_replacement: tuple[str, str] | None = None

    def set(self, key: str, value: str, *, nx: bool = False) -> bool:
        if nx and key in self._values:
            return False
        self._values[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def delete(self, key: str) -> int:
        return int(self._values.pop(key, None) is not None)

    def pipeline(self) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    def eval(self, _script: str, numkeys: int, *args: str) -> int:
        assert numkeys == 1
        key, expected, updated, previous_expiry, requested_expiry, client_now = args
        if self.eval_replacement is not None:
            replacement_key, replacement_value = self.eval_replacement
            self._values[replacement_key] = replacement_value
        server_now = self.eval_server_now_ms if self.eval_server_now_ms is not None else int(client_now)
        current_now = max(server_now, int(client_now))
        if int(previous_expiry) <= current_now or int(requested_expiry) <= current_now:
            return 2
        if self._values.get(key) != expected:
            return 0
        self._values[key] = updated
        return 1

    def scan_iter(self, pattern: str) -> list[str]:
        prefix = pattern.removesuffix("*")
        return [key for key in self._values if key.startswith(prefix)]


def _redis_store() -> RedisStateStore:
    store = RedisStateStore.__new__(RedisStateStore)
    store._redis_url = "redis://contract.test/0"
    store._watch_error = _FakeWatchError
    store._client = _FakeRedisClient()
    return store


def _checkpoint(task_id: str = "claim") -> Checkpoint:
    return Checkpoint(
        task_id=task_id,
        cycle_index=0,
        status=AgentStatus.RUNNING,
        messages=[Message(role="user", content="hello")],
        cycles=[],
        shared_state={"nested": {"value": 1}},
    )


def _stores(tmp_path: Path) -> list[StateStore]:
    return [InMemoryStateStore(), SqliteStateStore(tmp_path / "checkpoints.sqlite3"), _redis_store()]


def test_checkpoint_fixture_hash_and_rust_copy_match() -> None:
    payload = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_SHA256
    assert RUST_FIXTURE_PATH.read_bytes() == payload


def test_checkpoint_codec_matches_shared_valid_and_invalid_corpus() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in [fixture["canonical"], *(item["payload"] for item in fixture["valid_cases"])]:
        decoded = checkpoint_from_dict(case)
        assert checkpoint_to_dict(decoded) == case
        assert json.loads(checkpoint_to_json(decoded)) == case

    for case in fixture["invalid_cases"]:
        with pytest.raises((TypeError, ValueError), match="checkpoint"):
            checkpoint_from_dict(case["payload"])


def test_redis_renewal_atomic_compare_and_set_does_not_overwrite_a_new_owner() -> None:
    store = _redis_store()
    checkpoint = _checkpoint("renewal-cas-mismatch")
    checkpoint.revision = 1
    checkpoint.claim_token = "owner"
    checkpoint.claimed_cycle = 1
    checkpoint.lease_expires_at_ms = 200
    store.save_checkpoint(checkpoint)
    replacement = _checkpoint(checkpoint.task_id)
    replacement.revision = 2
    replacement.claim_token = "contender"
    replacement.claimed_cycle = 1
    replacement.lease_expires_at_ms = 500
    store._client.eval_replacement = (
        f"vv_agent:checkpoint:{checkpoint.task_id}",
        checkpoint_to_json(replacement),
    )

    assert not store.renew_checkpoint_claim(
        checkpoint.task_id,
        claim_token="owner",
        expected_revision=1,
        lease_expires_at_ms=300,
        now_ms=150,
    )
    persisted = store.load_checkpoint(checkpoint.task_id)
    assert persisted is not None
    assert persisted.claim_token == "contender"
    assert persisted.lease_expires_at_ms == 500


def test_redis_renewal_checks_server_time_atomically_before_writing() -> None:
    store = _redis_store()
    checkpoint = _checkpoint("renewal-expired-at-write")
    checkpoint.revision = 1
    checkpoint.claim_token = "owner"
    checkpoint.claimed_cycle = 1
    checkpoint.lease_expires_at_ms = 110
    store.save_checkpoint(checkpoint)
    store._client.eval_server_now_ms = 110

    with pytest.raises(CheckpointConflictError, match="claim lease expired"):
        store.renew_checkpoint_claim(
            checkpoint.task_id,
            claim_token="owner",
            expected_revision=1,
            lease_expires_at_ms=1_000,
            now_ms=100,
        )
    persisted = store.load_checkpoint(checkpoint.task_id)
    assert persisted is not None
    assert persisted.lease_expires_at_ms == 110

    contender = store.claim_checkpoint(
        checkpoint.task_id,
        1,
        claim_token="contender",
        lease_expires_at_ms=1_000,
        now_ms=110,
    )
    assert contender is not None
    assert contender.claim_token == "contender"


def test_redis_renewal_expiry_precedes_compare_and_set_mismatch() -> None:
    store = _redis_store()
    checkpoint = _checkpoint("renewal-expiry-before-cas")
    checkpoint.revision = 1
    checkpoint.claim_token = "owner"
    checkpoint.claimed_cycle = 1
    checkpoint.lease_expires_at_ms = 200
    store.save_checkpoint(checkpoint)
    replacement = _checkpoint(checkpoint.task_id)
    replacement.revision = 2
    replacement.claim_token = None
    replacement.claimed_cycle = None
    replacement.lease_expires_at_ms = None
    replacement_raw = checkpoint_to_json(replacement)
    store._client.eval_replacement = (
        f"vv_agent:checkpoint:{checkpoint.task_id}",
        replacement_raw,
    )
    store._client.eval_server_now_ms = 200

    with pytest.raises(CheckpointConflictError, match="claim lease expired"):
        store.renew_checkpoint_claim(
            checkpoint.task_id,
            claim_token="owner",
            expected_revision=1,
            lease_expires_at_ms=300,
            now_ms=150,
        )
    assert store._client.get(f"vv_agent:checkpoint:{checkpoint.task_id}") == replacement_raw


def test_state_stores_snapshot_values_and_reject_duplicate_or_skipped_claims(tmp_path: Path) -> None:
    for store in _stores(tmp_path):
        original = _checkpoint(type(store).__name__)
        assert store.create_checkpoint(original) is True
        assert store.create_checkpoint(original) is False

        original.messages[0].content = "mutated after save"
        original.shared_state["nested"]["value"] = 2
        loaded = store.load_checkpoint(original.task_id)
        assert loaded is not None
        assert loaded.messages[0].content == "hello"
        assert loaded.shared_state == {"nested": {"value": 1}}

        claimed = store.claim_checkpoint(
            original.task_id,
            1,
            claim_token="first",
            lease_expires_at_ms=200,
            now_ms=100,
        )
        assert claimed is not None
        claimed.messages[0].content = "mutated after load"
        reloaded = store.load_checkpoint(original.task_id)
        assert reloaded is not None
        assert reloaded.messages[0].content == "hello"
        assert store.renew_checkpoint_claim(
            original.task_id,
            claim_token="first",
            expected_revision=claimed.revision,
            lease_expires_at_ms=300,
            now_ms=150,
        )
        assert not store.renew_checkpoint_claim(
            original.task_id,
            claim_token="wrong",
            expected_revision=claimed.revision,
            lease_expires_at_ms=320,
            now_ms=160,
        )
        renewed = store.load_checkpoint(original.task_id)
        assert renewed is not None and renewed.lease_expires_at_ms == 300

        with pytest.raises(CheckpointConflictError):
            store.claim_checkpoint(
                original.task_id,
                1,
                claim_token="duplicate",
                lease_expires_at_ms=250,
                now_ms=150,
            )
        with pytest.raises(CheckpointConflictError):
            store.claim_checkpoint(
                original.task_id,
                3,
                claim_token="skipped",
                lease_expires_at_ms=250,
                now_ms=150,
            )

        claimed.messages[0].content = "cycle complete"
        claimed.cycle_index = 1
        expired = store.claim_checkpoint(
            original.task_id,
            1,
            claim_token="retry",
            lease_expires_at_ms=400,
            now_ms=300,
        )
        assert expired is not None
        assert store.commit_checkpoint(claimed, claim_token="stale", expected_revision=claimed.revision) is False
        expired.messages[0].content = "cycle complete"
        expired.cycle_index = 1
        assert store.commit_checkpoint(expired, claim_token="retry", expected_revision=expired.revision) is True
        committed = store.load_checkpoint(original.task_id)
        assert committed is not None
        assert committed.cycle_index == 1
        assert committed.revision == expired.revision + 1

        terminal_result = AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=committed.messages,
            cycles=committed.cycles,
            final_answer="done",
            shared_state=committed.shared_state,
        )
        committed.status = terminal_result.status
        committed.terminal_result = terminal_result
        assert store.finalize_checkpoint(committed, expected_revision=committed.revision) is True
        terminal = store.load_checkpoint(original.task_id)
        assert terminal is not None and terminal.terminal_result is not None

        replacement_result = AgentResult(
            status=AgentStatus.FAILED,
            messages=terminal.messages,
            cycles=terminal.cycles,
            error="must not replace the durable terminal",
            shared_state=terminal.shared_state,
        )
        replacement = Checkpoint(
            task_id=terminal.task_id,
            cycle_index=terminal.cycle_index,
            status=replacement_result.status,
            messages=replacement_result.messages,
            cycles=replacement_result.cycles,
            shared_state=replacement_result.shared_state,
            revision=terminal.revision,
            terminal_result=replacement_result,
        )
        assert store.finalize_checkpoint(replacement, expected_revision=terminal.revision) is False
        retained = store.load_checkpoint(original.task_id)
        assert retained is not None and retained.terminal_result is not None
        assert retained.terminal_result.final_answer == "done"
        assert store.acknowledge_terminal(original.task_id, expected_revision=terminal.revision - 1) is False
        assert store.acknowledge_terminal(original.task_id, expected_revision=terminal.revision) is True


def test_sqlite_store_spec_is_reconstructable_and_memory_store_is_not(tmp_path: Path) -> None:
    memory = InMemoryStateStore()
    sqlite = SqliteStateStore(tmp_path / "state.sqlite3")

    assert memory.state_store_spec() is None
    spec = sqlite.state_store_spec()
    assert spec is not None
    assert spec.to_dict() == {"kind": "sqlite", "location": str((tmp_path / "state.sqlite3").resolve())}


def test_sqlite_migrates_legacy_checkpoint_table_in_place(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE checkpoints (
            task_id TEXT PRIMARY KEY,
            cycle_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            messages TEXT NOT NULL,
            cycles TEXT NOT NULL,
            shared_state TEXT NOT NULL
        );
        INSERT INTO checkpoints VALUES ('legacy-task', 0, 'running', '[]', '[]', '{}');
        """
    )
    legacy.close()

    store = SqliteStateStore(db_path)
    checkpoint = store.load_checkpoint("legacy-task")

    assert checkpoint is not None
    assert checkpoint.revision == 0
    assert checkpoint.claim_token is None
    assert checkpoint.terminal_result is None
    claimed = store.claim_checkpoint(
        "legacy-task",
        1,
        claim_token="migrated-worker",
        lease_expires_at_ms=200,
        now_ms=100,
    )
    assert claimed is not None and claimed.revision == 1


def test_sqlite_second_connection_waits_for_short_write_contention(tmp_path: Path) -> None:
    db_path = tmp_path / "contention.sqlite3"
    store = SqliteStateStore(db_path)
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("PRAGMA busy_timeout=5000")
    locker.execute("BEGIN IMMEDIATE")
    outcome: dict[str, object] = {}

    def create_checkpoint() -> None:
        try:
            outcome["created"] = store.create_checkpoint(_checkpoint("contended-task"))
        except BaseException as exc:
            outcome["error"] = exc

    worker = Thread(target=create_checkpoint)
    worker.start()
    time.sleep(0.1)
    assert worker.is_alive(), "the second connection should wait instead of failing immediately"
    locker.commit()
    worker.join(2)
    locker.close()

    assert not worker.is_alive()
    assert "error" not in outcome
    assert outcome["created"] is True
    assert store.load_checkpoint("contended-task") is not None


def test_sqlite_renewal_refreshes_time_after_write_lock_wait(tmp_path: Path) -> None:
    db_path = tmp_path / "renewal-contention.sqlite3"
    store = SqliteStateStore(db_path)
    checkpoint = _checkpoint("contended-renewal")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.task_id,
        1,
        claim_token="owner",
        lease_expires_at_ms=150,
        now_ms=100,
    )
    assert claimed is not None
    locker = sqlite3.connect(db_path, check_same_thread=False)
    locker.execute("PRAGMA busy_timeout=5000")
    locker.execute("BEGIN IMMEDIATE")
    outcome: dict[str, object] = {}

    def renew_checkpoint() -> None:
        try:
            outcome["renewed"] = store.renew_checkpoint_claim(
                checkpoint.task_id,
                claim_token="owner",
                expected_revision=claimed.revision,
                lease_expires_at_ms=300,
                now_ms=100,
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = Thread(target=renew_checkpoint)
    worker.start()
    time.sleep(0.08)
    assert worker.is_alive(), "renewal should wait for the SQLite writer lock"
    locker.commit()
    worker.join(2)
    locker.close()

    assert not worker.is_alive()
    assert "error" not in outcome
    assert outcome["renewed"] is False
    persisted = store.load_checkpoint(checkpoint.task_id)
    assert persisted is not None
    assert persisted.lease_expires_at_ms == 150


def test_sqlite_renewal_rolls_back_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SqliteStateStore(tmp_path / "renewal-commit-failure.sqlite3")
    checkpoint = _checkpoint("renewal-commit-failure")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.task_id,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
    )
    assert claimed is not None
    connection = store._conn

    class CommitFailingConnection:
        def __init__(self) -> None:
            self.rollback_calls = 0

        def __getattr__(self, name: str):
            return getattr(connection, name)

        def commit(self) -> None:
            raise sqlite3.OperationalError("commit failed")

        def rollback(self) -> None:
            self.rollback_calls += 1
            connection.rollback()

    failing_connection = CommitFailingConnection()
    monkeypatch.setattr(store, "_conn", failing_connection)

    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        store.renew_checkpoint_claim(
            checkpoint.task_id,
            claim_token="owner",
            expected_revision=claimed.revision,
            lease_expires_at_ms=300,
            now_ms=150,
        )

    assert failing_connection.rollback_calls == 1
    persisted = store.load_checkpoint(checkpoint.task_id)
    assert persisted is not None
    assert persisted.lease_expires_at_ms == 200


def test_claimed_terminal_result_commits_before_scheduler_acknowledgement(tmp_path: Path) -> None:
    for store in _stores(tmp_path):
        task_id = f"terminal-{type(store).__name__}"
        assert store.create_checkpoint(_checkpoint(task_id))
        claimed = store.claim_checkpoint(
            task_id,
            1,
            claim_token="terminal",
            lease_expires_at_ms=200,
            now_ms=100,
        )
        assert claimed is not None
        claimed.cycle_index = 1
        claimed.status = AgentStatus.COMPLETED
        claimed.terminal_result = AgentResult(
            status=AgentStatus.COMPLETED,
            messages=claimed.messages,
            cycles=claimed.cycles,
            final_answer="done",
            shared_state=claimed.shared_state,
        )
        revision = claimed.revision

        assert store.commit_checkpoint(claimed, claim_token="terminal", expected_revision=revision)
        persisted = store.load_checkpoint(task_id)
        assert persisted is not None
        assert persisted.claim_token is None
        assert persisted.terminal_result is not None
        assert persisted.terminal_result.final_answer == "done"
        assert store.acknowledge_terminal(task_id, expected_revision=revision + 1)


def test_worker_redelivery_replays_persisted_terminal_before_runtime_rebuild(tmp_path: Path) -> None:
    db_path = tmp_path / "redelivery.sqlite3"
    store = SqliteStateStore(db_path)
    task = AgentTask(task_id="redelivery", model="model", system_prompt="system", user_prompt="prompt")
    terminal = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[Message(role="assistant", content="done")],
        cycles=[],
        final_answer="persisted done",
        shared_state={"calls": 1},
    )
    assert store.create_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=1,
            status=terminal.status,
            messages=terminal.messages,
            cycles=terminal.cycles,
            shared_state=terminal.shared_state,
            revision=2,
            terminal_result=terminal,
        )
    )
    recipe = RuntimeRecipe(
        settings_file=str(tmp_path / "missing-settings.py"),
        backend="missing",
        model="missing",
        workspace=str(tmp_path),
        state_store=store.state_store_spec(),
    )

    payload = run_single_cycle(task_dict=task.to_dict(), recipe_dict=recipe.to_dict(), cycle_index=1)

    assert payload["finished"] is True
    assert payload["checkpoint_revision"] == 2
    assert payload["result"]["final_answer"] == "persisted done"
