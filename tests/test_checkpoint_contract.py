from __future__ import annotations

import hashlib
import json
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
RUST_FIXTURE_PATH = (
    Path(__file__).parents[2] / "vv-agent-rs" / "crates" / "vv-agent" / "tests" / "fixtures" / "parity" / FIXTURE_PATH.name
)
FIXTURE_SHA256 = "375baeb13c961a3a50ae23501e000839ac6baf5d2e2878d7858d79d3bab91cb8"


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
