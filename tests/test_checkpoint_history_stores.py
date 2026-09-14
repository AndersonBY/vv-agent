from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from test_checkpoint import _minimal_checkpoint, _redis_store

from vv_agent.checkpoint import CheckpointError
from vv_agent.runtime.checkpoint_codec import checkpoint_to_json
from vv_agent.runtime.checkpoint_history import (
    compact_checkpoint,
    cumulative_checkpoint_usage,
    decode_history_batches,
    hydrate_checkpoint_result,
)
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.types import AgentResult, AgentStatus, CycleRecord, ModelCallOperation, ModelCallRecord, ModelCallStatus, TokenUsage


@pytest.fixture(params=["memory", "sqlite", "redis", "real_redis"])
def store(request: Any, tmp_path: Path) -> Any:
    if request.param == "memory":
        value: Any = InMemoryCheckpointStore()
    elif request.param == "sqlite":
        value = SqliteCheckpointStore(tmp_path / "history.sqlite3")
    elif request.param == "real_redis":
        redis_url = os.getenv("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for real Redis history transactions")
        value = RedisCheckpointStore(redis_url)
    else:
        value = _redis_store()
    checkpoint = _minimal_checkpoint(key=f"history-{uuid4().hex}")
    assert value.create_checkpoint(checkpoint)
    yield value, checkpoint.checkpoint_key
    value.delete_checkpoint(checkpoint.checkpoint_key)
    if isinstance(value, SqliteCheckpointStore):
        value.close()


def _candidate(store: Any, key: str, index: int) -> Any:
    checkpoint = store.claim_checkpoint(
        key, index, claim_token=f"owner-{index}", lease_expires_at_ms=1000, now_ms=0, claim_mode="continue"
    )
    assert checkpoint is not None
    checkpoint.cycles.append(CycleRecord(index=index, assistant_message="x" * 8192))
    checkpoint.model_calls.append(
        ModelCallRecord(
            call_id=f"call-{index}",
            operation_id=f"operation-{index}",
            attempt=1,
            operation=ModelCallOperation.AGENT_CYCLE,
            cycle_index=index,
            backend="test",
            model="test",
            status=ModelCallStatus.COMPLETED,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=0),
        )
    )
    checkpoint.cycle_index = index
    return checkpoint


def _commit(store: Any, checkpoint: Any) -> bool:
    return store.commit_checkpoint(checkpoint, claim_token=checkpoint.claim_token, expected_revision=checkpoint.revision)


def test_history_is_bounded_complete_and_deleted_with_checkpoint(store: Any) -> None:
    value, key = store
    sizes = []
    for index in range(1, 41):
        candidate = _candidate(value, key, index)
        assert _commit(value, candidate)
        current = value.load_checkpoint(key)
        assert [cycle.index for cycle in current.cycles] == [index]
        assert [call.call_id for call in current.model_calls] == [f"call-{index}"]
        assert current.history["cycle_count"] == index - 1
        assert current.history["model_call_count"] == index - 1
        assert cumulative_checkpoint_usage(current).total_tokens == index * 15
        sizes.append(len(checkpoint_to_json(current)))
    assert max(sizes[1:]) < min(sizes[1:]) * 1.02
    # Retained snapshot encoding stays bounded as history grows.
    assert sum(sizes[:40]) < sum(sizes[:20]) * 2.1
    history = value.load_checkpoint_history(key)
    assert [cycle.index for cycle in history.cycles] == list(range(1, 40))
    assert [record.call_id for record in history.model_calls] == [f"call-{i}" for i in range(1, 40)]
    value.delete_checkpoint(key)
    assert value.load_checkpoint_history(key).cycles == []
    if isinstance(value, SqliteCheckpointStore):
        assert value._conn.execute("SELECT count(*) FROM checkpoint_history").fetchone()[0] == 0
        assert value._conn.execute("SELECT count(*) FROM checkpoint_history_call_ids").fetchone()[0] == 0
    elif isinstance(value, RedisCheckpointStore):
        assert value._client.hgetall(f"{value.data_key(key)}:history") == {}
        assert value._client.smembers(f"{value.data_key(key)}:history:call_ids") == set()
    else:
        assert key not in value._history_call_ids


def test_failed_cas_does_not_append_and_frontier_cannot_be_replaced(store: Any) -> None:
    value, key = store
    assert _commit(value, _candidate(value, key, 1))
    second = _candidate(value, key, 2)
    stale = deepcopy(second)
    stale.revision -= 1
    assert not _commit(value, stale)
    assert value.load_checkpoint_history(key).cycles == []
    assert _commit(value, second)
    third = _candidate(value, key, 3)
    forged = deepcopy(third)
    forged.history["head_digest"] = "f" * 64
    assert not _commit(value, forged)
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1]
    assert _commit(value, third)
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1, 2]


def test_history_corruption_is_detected_only_on_explicit_history_read(store: Any) -> None:
    value, key = store
    for index in (1, 2):
        assert _commit(value, _candidate(value, key, index))
    if isinstance(value, SqliteCheckpointStore):
        payload = value._conn.execute("SELECT payload FROM checkpoint_history WHERE checkpoint_key = ?", (key,)).fetchone()[0]
    elif isinstance(value, RedisCheckpointStore):
        payload = value._client.hget(f"{value.data_key(key)}:history", "1")
    else:
        payload = json.dumps(value._history[key][0])
    batch = json.loads(payload)
    batch["cycles"][0]["assistant_message"] = "corrupted"
    if isinstance(value, SqliteCheckpointStore):
        with value._conn:
            value._conn.execute("UPDATE checkpoint_history SET payload = ? WHERE checkpoint_key = ?", (json.dumps(batch), key))
    elif isinstance(value, RedisCheckpointStore):
        value._client.hset(f"{value.data_key(key)}:history", "1", json.dumps(batch))
    else:
        value._history[key][0] = batch
    assert value.load_checkpoint(key).cycle_index == 2
    with pytest.raises(CheckpointError, match="integrity"):
        value.load_checkpoint_history(key)


@pytest.mark.parametrize("table", ["checkpoint_history", "checkpoint_history_call_ids"])
def test_sqlite_history_append_failure_rolls_back_checkpoint(tmp_path: Path, table: str) -> None:
    value = SqliteCheckpointStore(tmp_path / "rollback.sqlite3")
    key = "history-rollback"
    assert value.create_checkpoint(_minimal_checkpoint(key=key))
    assert _commit(value, _candidate(value, key, 1))
    second = _candidate(value, key, 2)
    initial = value.load_checkpoint(key)
    assert initial is not None
    before = checkpoint_to_json(initial)
    value._conn.execute(
        f"CREATE TRIGGER fail_history BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'history write failure'); END"
    )
    with pytest.raises(Exception, match="history write failure"):
        _commit(value, second)
    current = value.load_checkpoint(key)
    assert current is not None
    assert checkpoint_to_json(current) == before
    assert value.load_checkpoint_history(key).cycles == []
    assert value._conn.execute("SELECT count(*) FROM checkpoint_history_call_ids").fetchone()[0] == 0
    value._conn.execute("DROP TRIGGER fail_history")
    assert _commit(value, second)
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1]
    value.close()


def test_real_redis_watch_retry_appends_exactly_once(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    value, key = store
    if not isinstance(value, RedisCheckpointStore) or not hasattr(value._client, "connection_pool"):
        pytest.skip("requires real Redis transaction conflict")
    assert _commit(value, _candidate(value, key, 1))
    second = _candidate(value, key, 2)
    original_pipeline = value._client.pipeline
    conflicted = False

    def pipeline(*args: Any, **kwargs: Any) -> Any:
        pipe = original_pipeline(*args, **kwargs)
        execute = pipe.execute

        def execute_with_conflict(*args: Any, **kwargs: Any) -> Any:
            nonlocal conflicted
            if not conflicted:
                conflicted = True
                data_key = value.data_key(key)
                # Even an identical write from another connection invalidates
                # WATCH, so neither queued checkpoint nor archive may commit.
                value._client.set(data_key, value._client.get(data_key))
            return execute(*args, **kwargs)

        pipe.execute = execute_with_conflict
        return pipe

    monkeypatch.setattr(value._client, "pipeline", pipeline)
    assert _commit(value, second)
    assert conflicted
    assert list(value._client.hgetall(f"{value.data_key(key)}:history")) == ["1"]
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1]
    assert value.load_checkpoint(key).history["sequence"] == 1


@pytest.mark.parametrize("suffix", [":history", ":history:call_ids"])
def test_real_redis_wrong_history_type_cannot_commit_partial_checkpoint(store: Any, suffix: str) -> None:
    value, key = store
    if not isinstance(value, RedisCheckpointStore) or not hasattr(value._client, "connection_pool"):
        pytest.skip("requires real Redis command type enforcement")
    assert _commit(value, _candidate(value, key, 1))
    second = _candidate(value, key, 2)
    before = checkpoint_to_json(value.load_checkpoint(key))
    history_key = f"{value.data_key(key)}{suffix}"
    value._client.set(history_key, "wrong type")
    with pytest.raises(Exception, match="WRONGTYPE"):
        _commit(value, second)
    assert checkpoint_to_json(value.load_checkpoint(key)) == before
    value._client.delete(history_key)
    assert _commit(value, second)
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1]


def _cross_history_cycle(value: Any, index: int) -> None:
    checkpoint = value.claim_checkpoint(
        "cross-history", index, claim_token=f"owner-{index}", lease_expires_at_ms=1000, now_ms=0, claim_mode="continue"
    )
    assert checkpoint is not None
    checkpoint.cycles.append(CycleRecord(index=index, assistant_message="x" * 2048))
    checkpoint.model_calls.append(
        ModelCallRecord(
            call_id=f"call-{index}",
            operation_id=f"operation-{index}",
            attempt=1,
            operation=ModelCallOperation.AGENT_CYCLE,
            cycle_index=index,
            backend="test",
            model="model",
            status=ModelCallStatus.COMPLETED,
            usage=TokenUsage(input_tokens=index),
        )
    )
    checkpoint.cycle_index = index
    if index == 4:
        forged = deepcopy(checkpoint)
        forged.model_calls[-1].call_id = "call-1"
        with pytest.raises(CheckpointError) as error:
            _commit(value, forged)
        assert error.value.code == "checkpoint_history_invalid"
    assert _commit(value, checkpoint)


def test_cross_runtime_history_store() -> None:
    """Run write/read in opposite runtimes against the same disposable store."""
    kind = os.getenv("VV_AGENT_CROSS_HISTORY_STORE")
    if kind is None:
        pytest.skip("requires a paired cross-runtime history store")
    location = os.environ["VV_AGENT_CROSS_HISTORY_LOCATION"]
    mode = os.environ["VV_AGENT_CROSS_HISTORY_MODE"]
    assert kind in {"sqlite", "redis"} and mode in {"write", "read"}
    value: Any = SqliteCheckpointStore(location) if kind == "sqlite" else RedisCheckpointStore(location)
    try:
        if mode == "write":
            value.delete_checkpoint("cross-history")
            assert value.create_checkpoint(_minimal_checkpoint(key="cross-history"))
            for index in range(1, 4):
                _cross_history_cycle(value, index)
        checkpoint = value.load_checkpoint("cross-history")
        assert checkpoint.history["cycle_count"] == 2
        assert checkpoint.history["model_call_count"] == 2
        history = value.load_checkpoint_history("cross-history")
        assert [cycle.index for cycle in [*history.cycles, *checkpoint.cycles]] == [1, 2, 3]
        assert [record.call_id for record in [*history.model_calls, *checkpoint.model_calls]] == ["call-1", "call-2", "call-3"]
        assert cumulative_checkpoint_usage(checkpoint).input_tokens == 6
        if mode == "read":
            _cross_history_cycle(value, 4)
            assert [cycle.index for cycle in value.load_checkpoint_history("cross-history").cycles] == [1, 2, 3]
            assert cumulative_checkpoint_usage(value.load_checkpoint("cross-history")).input_tokens == 10
    finally:
        if isinstance(value, SqliteCheckpointStore):
            value.close()


@pytest.mark.parametrize("changed", ["cycle", "model_call"])
def test_committed_history_prefix_cannot_be_changed(store: Any, changed: str) -> None:
    value, key = store
    assert _commit(value, _candidate(value, key, 1))
    second = _candidate(value, key, 2)
    before = checkpoint_to_json(value.load_checkpoint(key))
    if changed == "cycle":
        second.cycles[0].assistant_message = "rewritten committed evidence"
    else:
        second.model_calls[0].usage.input_tokens = 99
    assert not _commit(value, second)
    assert checkpoint_to_json(value.load_checkpoint(key)) == before
    assert value.load_checkpoint_history(key).cycles == []


@pytest.mark.parametrize("changed", ["duplicate_cycle", "retroactive_call", "archived_call_id"])
def test_history_writes_reject_reintroduced_evidence(store: Any, changed: str) -> None:
    value, key = store
    for index in (1, 2):
        assert _commit(value, _candidate(value, key, index))
    third = _candidate(value, key, 3)
    valid = deepcopy(third)
    before = checkpoint_to_json(value.load_checkpoint(key))
    if changed == "duplicate_cycle":
        third.cycles.insert(1, deepcopy(third.cycles[0]))
    elif changed == "retroactive_call":
        third.model_calls[-1].cycle_index = 1
    else:
        # A forged current-cycle record must not reuse an archived identity.
        third.model_calls[-1].call_id = "call-1"
    try:
        assert not _commit(value, third)
    except CheckpointError as error:
        assert error.code == "checkpoint_history_invalid"
    assert checkpoint_to_json(value.load_checkpoint(key)) == before
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1]
    assert _commit(value, valid)
    assert [cycle.index for cycle in value.load_checkpoint_history(key).cycles] == [1, 2]


def test_history_archive_matches_canonical_golden() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/parity/checkpoint_codec.json").read_text())
    golden = fixture["history_archive_cases"]["one_committed_cycle"]
    checkpoint = _minimal_checkpoint()
    checkpoint.cycle_index = 2
    checkpoint.cycles = [CycleRecord.from_dict(golden["batch"]["cycles"][0]), CycleRecord(index=2, assistant_message="second")]
    checkpoint.model_calls = [ModelCallRecord.from_dict(golden["batch"]["model_calls"][0])]
    batch = compact_checkpoint(checkpoint)
    assert batch is not None
    assert batch == golden["batch"]
    assert checkpoint.history == golden["frontier"]
    assert [cycle.index for cycle in checkpoint.cycles] == golden["retained_cycle_indices"]
    assert [record.call_id for record in checkpoint.model_calls] == golden["retained_model_call_ids"]
    history = decode_history_batches(checkpoint, [batch])
    assert history.frontier == golden["frontier"]
    assert [cycle.to_dict() for cycle in history.cycles] == golden["batch"]["cycles"]
    assert [record.to_dict() for record in history.model_calls] == golden["batch"]["model_calls"]


@pytest.mark.parametrize("kind", ["memory", "sqlite", "redis"])
def test_hydration_rejects_changed_archive_frontier(kind: str, tmp_path: Path) -> None:
    if kind == "sqlite":
        value: Any = SqliteCheckpointStore(tmp_path / "history-hydration.sqlite3")
    elif kind == "redis":
        redis_url = os.getenv("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for real Redis hydration race")
        value = RedisCheckpointStore(redis_url)
    else:
        value = InMemoryCheckpointStore()
    key = f"history-hydration-{uuid4().hex}"
    try:
        assert value.create_checkpoint(_minimal_checkpoint(key=key))
        for index in range(1, 4):
            assert _commit(value, _candidate(value, key, index))
        old_checkpoint = value.load_checkpoint(key)
        assert old_checkpoint is not None
        assert old_checkpoint.history["cycle_count"] == 2
        tail_result = AgentResult(status=AgentStatus.COMPLETED, messages=old_checkpoint.messages, cycles=old_checkpoint.cycles)
        original_result = deepcopy(tail_result)
        assert value.load_checkpoint_history(key).frontier == old_checkpoint.history
        assert [cycle.index for cycle in hydrate_checkpoint_result(value, old_checkpoint, tail_result).cycles] == [1, 2, 3]
        assert _commit(value, _candidate(value, key, 4))
        with pytest.raises(CheckpointError) as error:
            hydrate_checkpoint_result(value, old_checkpoint, tail_result)
        assert error.value.code == "checkpoint_history_changed"
        assert tail_result == original_result
    finally:
        value.delete_checkpoint(key)
        if isinstance(value, SqliteCheckpointStore):
            value.close()
