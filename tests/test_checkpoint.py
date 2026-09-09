from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Set
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock, Thread
from typing import Any
from uuid import uuid4

import pytest

from vv_agent.checkpoint import (
    AmbiguousModelPolicy,
    AmbiguousToolPolicy,
    CheckpointConfig,
    CheckpointError,
    CheckpointExtension,
    EventCursor,
    OperationKind,
    OperationState,
    ReconciliationDecisionKind,
    ResumeObservation,
    ResumePolicy,
    ToolIdempotency,
    canonical_json_bytes,
    canonical_json_sha256,
    compute_event_payload_digest,
    compute_operation_request_digest,
    compute_run_definition_digest,
    validate_checkpoint_extension,
)
from vv_agent.events import (
    CheckpointCreatedEvent,
    CheckpointResumedEvent,
    CycleAbortedEvent,
    ModelCallCompletedEvent,
    ModelCallFailedEvent,
    ModelCallStartedEvent,
    OperationAmbiguousEvent,
    OperationReplayedEvent,
    ReconciliationRequiredEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunStateChangedEvent,
)
from vv_agent.runtime.checkpoint_codec import (
    _strict_json_loads,
    checkpoint_from_dict,
    checkpoint_from_json,
    checkpoint_to_dict,
    checkpoint_to_json,
    validate_extension_state_size,
)
from vv_agent.runtime.checkpoint_resume import CheckpointReconciliationRequired, CheckpointResumeController
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    CheckpointStore,
    EventOutboxEntry,
    OperationError,
    OperationJournalEntry,
    compute_tool_identity_key,
)
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    CompletionReason,
    CycleRecord,
    Message,
    ModelCallRecord,
    ModelCallStatus,
    TokenUsage,
    ToolArtifactRef,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
CHECKPOINT_SQL_FIXTURE = FIXTURE_DIR / "checkpoint_sqlite_canonical.sql"


def _seed_checkpoint_sqlite_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(CHECKPOINT_SQL_FIXTURE.read_text(encoding="utf-8"))


def _sqlite_master_state(path: Path) -> tuple[tuple[str, str, str, str | None], ...]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    return tuple(
        (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            None if row[3] is None else str(row[3]),
        )
        for row in rows
    )


def _sqlite_pragma_state(path: Path) -> tuple[int, int, str]:
    with sqlite3.connect(path) as connection:
        schema_version = int(connection.execute("PRAGMA schema_version").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    return schema_version, user_version, journal_mode


def _seed_business_probe(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE business_probe (business_key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO business_probe (business_key, value) VALUES ('probe', 'unchanged')")


def _sqlite_business_state(path: Path) -> tuple[tuple[str, str], ...]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT business_key, value FROM business_probe ORDER BY business_key").fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.replace("IF NOT EXISTS", "").split())


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

    def watch(self, *_keys: str) -> None:
        return None

    def unwatch(self) -> None:
        self._transaction = False
        self._commands.clear()

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def mget(self, keys: list[str]) -> list[str | None]:
        return self._client.mget(keys)

    def smembers(self, key: str) -> Set[str]:
        return self._client.smembers(key)

    def multi(self) -> None:
        self._transaction = True

    def set(self, key: str, value: str, *, nx: bool = False) -> None:
        if self._transaction:
            self._commands.append(("set_nx" if nx else "set", key, value))
        else:
            self._client.set(key, value, nx=nx)

    def sadd(self, key: str, value: str) -> None:
        if self._transaction:
            self._commands.append(("sadd", key, value))
        else:
            self._client.sadd(key, value)

    def srem(self, key: str, value: str) -> None:
        if self._transaction:
            self._commands.append(("srem", key, value))
        else:
            self._client.srem(key, value)

    def delete(self, key: str) -> None:
        if self._transaction:
            self._commands.append(("delete", key, None))
        else:
            self._client.delete(key)

    def execute(self) -> list[object]:
        results: list[object] = []
        for command, key, value in self._commands:
            if command in {"set", "set_nx"}:
                assert value is not None
                results.append(self._client.set(key, value, nx=command == "set_nx"))
            elif command == "sadd":
                assert value is not None
                results.append(self._client.sadd(key, value))
            elif command == "srem":
                assert value is not None
                results.append(self._client.srem(key, value))
            else:
                results.append(self._client.delete(key))
        self._transaction = False
        self._commands.clear()
        return results


class _FakeRedisClient:
    fail_once: bool
    injected: bool
    second_receipt_key: str
    second_receipt_payload: str
    receipt_set_key: str
    winner_applied: bool

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}
        self.server_now_ms = 0

    def set(self, key: str, value: str, *, nx: bool = False) -> bool:
        if nx and key in self._values:
            return False
        self._values[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self._values.get(key) for key in keys]

    def sadd(self, key: str, value: str) -> int:
        members = self._sets.setdefault(key, set())
        before = len(members)
        members.add(value)
        return int(len(members) != before)

    def srem(self, key: str, value: str) -> int:
        members = self._sets.get(key)
        if members is None or value not in members:
            return 0
        members.remove(value)
        if not members:
            self._sets.pop(key, None)
        return 1

    def smembers(self, key: str) -> Set[str]:
        return set(self._sets.get(key, set()))

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            deleted += int(self._values.pop(key, None) is not None)
            deleted += int(self._sets.pop(key, None) is not None)
        return deleted

    def pipeline(self) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    def scan_iter(self, pattern: str) -> list[str]:
        prefix = pattern.removesuffix("*")
        return [key for key in self._values if key.startswith(prefix)]

    def time(self) -> tuple[int, int]:
        return self.server_now_ms // 1000, (self.server_now_ms % 1000) * 1000


def _redis_store() -> RedisCheckpointStore:
    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = _FakeRedisClient()
    return store


def test_redis_claim_uses_one_atomic_checkpoint_and_lease_snapshot() -> None:
    from unittest.mock import patch

    store = _redis_store()
    checkpoint = _minimal_checkpoint(key="atomic-claim")
    assert store.create_checkpoint(checkpoint)
    with patch.object(store._client, "mget", wraps=store._client.mget) as snapshot:
        with patch.object(_FakeRedisPipeline, "get", side_effect=AssertionError("non-atomic snapshot")):
            claimed = store.claim_checkpoint(
                checkpoint.checkpoint_key,
                1,
                claim_token="owner",
                lease_expires_at_ms=200,
                now_ms=100,
                claim_mode="continue",
            )
        snapshot.assert_called_once_with(list(store._keys(checkpoint.checkpoint_key)))
    assert claimed is not None
    assert claimed.claim_token == "owner"
    assert claimed.lease_expires_at_ms == 200


def test_redis_load_uses_one_atomic_checkpoint_and_lease_snapshot() -> None:
    store = _redis_store()
    checkpoint = _minimal_checkpoint(key="atomic-load")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    from unittest.mock import patch

    with patch.object(store._client, "mget", wraps=store._client.mget) as snapshot:
        with patch.object(store._client, "get", side_effect=AssertionError("non-atomic snapshot")):
            loaded = store.load_checkpoint(checkpoint.checkpoint_key)
        snapshot.assert_called_once_with(list(store._keys(checkpoint.checkpoint_key)))
    assert loaded is not None
    assert loaded.claim_token == "owner"
    assert loaded.lease_expires_at_ms == 200
    assert loaded.revision == claimed.revision
    assert store.load_checkpoint("missing-atomic-load") is None


def test_redis_create_transaction_failure_leaves_no_half_state() -> None:
    class FailingPipeline(_FakeRedisPipeline):
        def execute(self) -> list[object]:
            raise RuntimeError("injected EXEC failure")

    class FailingClient(_FakeRedisClient):
        def pipeline(self) -> FailingPipeline:
            return FailingPipeline(self)

    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = FailingClient()
    checkpoint = _minimal_checkpoint(key="redis-create-exec-failure")
    data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
    store._client.set(lease_key, "orphan-lease")  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="injected EXEC failure"):
        store.create_checkpoint(checkpoint)

    assert store._client.get(data_key) is None  # type: ignore[attr-defined]
    assert store._client.get(lease_key) == "orphan-lease"  # type: ignore[attr-defined]


def _fixture(name: str) -> dict[str, Any]:
    text = (FIXTURE_DIR / name).read_text(encoding="utf-8")
    return json.loads(text) if name == "checkpoint_store.json" else _strict_json_loads(text)


def _codec_case(name: str) -> dict[str, Any]:
    fixture = _fixture("checkpoint_codec.json")
    return deepcopy(next(case["payload"] for case in fixture["valid_cases"] if case["name"] == name))


def _minimal_checkpoint(*, key: str = "fresh") -> Checkpoint:
    payload = _codec_case("minimal_running")
    payload["checkpoint_key"] = key
    return checkpoint_from_dict(payload)


@pytest.mark.parametrize("field_name", ["active_host_interaction", "suspended_origin"])
def test_checkpoint_preserves_host_prompt(field_name: str) -> None:
    from vv_agent.runtime.controller import HostInteractionRequest

    request = HostInteractionRequest(
        interaction_id="unsanitized-interaction",
        logical_cycle=1,
        operation_id="unsanitized-operation",
        tool_call_id="unsanitized-tool",
        prompt="Approve sk-state-123 at https://example.invalid/state",
    )
    request_payload = request.to_dict()
    request_payload["prompt"] = "Approve sk-state-123 at https://example.invalid/state"
    payload = checkpoint_to_dict(_minimal_checkpoint(key=f"unsanitized-{field_name}"))
    if field_name == "active_host_interaction":
        payload["status"] = AgentStatus.HOST_INTERACTION.value
        payload["active_host_interaction"] = request_payload
        payload["suspended_origin"] = None
    else:
        payload["status"] = AgentStatus.SUSPENDED.value
        payload["active_host_interaction"] = None
        payload["suspended_origin"] = {
            "status": AgentStatus.HOST_INTERACTION.value,
            "active_host_interaction": request_payload,
        }

    assert checkpoint_to_dict(checkpoint_from_dict(payload)) == payload


def _checkpoint_created_event(*, event_id: str, checkpoint: Checkpoint) -> dict[str, Any]:
    return CheckpointCreatedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        checkpoint_key=checkpoint.checkpoint_key,
        resume_attempt=checkpoint.resume_attempt,
        cycle_index=checkpoint.cycle_index,
        event_id=event_id,
        created_at=123.0,
    ).to_dict()


def _store(store_kind: str, tmp_path: Path, name: str) -> Any:
    if store_kind == "memory":
        return InMemoryCheckpointStore()
    if store_kind == "sqlite":
        return SqliteCheckpointStore(tmp_path / f"{name}.sqlite3")
    if store_kind == "real_redis":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        return RedisCheckpointStore(redis_url)
    return _redis_store()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_create_rejects_cancel_requested_before_any_store_write(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"cancelled-create-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"cancelled-create-{store_kind}")
    checkpoint.cancel_requested = True

    with pytest.raises(CheckpointError) as error:
        store.create_checkpoint(checkpoint)

    assert error.value.code == "checkpoint_initial_invalid"
    assert store.load_checkpoint(checkpoint.checkpoint_key) is None
    if store_kind == "redis":
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        assert store._client.get(data_key) is None  # type: ignore[attr-defined]
        assert store._client.get(lease_key) is None  # type: ignore[attr-defined]


@pytest.mark.skipif(not os.getenv("VV_AGENT_TEST_REDIS_URL"), reason="set VV_AGENT_TEST_REDIS_URL for live Redis")
def test_real_redis_create_rejects_cancelled_random_key_before_write() -> None:
    store = RedisCheckpointStore(os.environ["VV_AGENT_TEST_REDIS_URL"])
    checkpoint = _minimal_checkpoint(key=f"real-redis-cancelled-create-{uuid4().hex}")
    checkpoint.cancel_requested = True
    data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]

    store._client.delete(data_key, lease_key)  # type: ignore[attr-defined]
    store._client.set(lease_key, "orphan-lease")  # type: ignore[attr-defined]
    try:
        with pytest.raises(CheckpointError) as error:
            store.create_checkpoint(checkpoint)
        assert error.value.code == "checkpoint_initial_invalid"
        assert store._client.get(data_key) is None  # type: ignore[attr-defined]
        assert store._client.get(lease_key) == "orphan-lease"  # type: ignore[attr-defined]
    finally:
        store.delete_checkpoint(checkpoint.checkpoint_key)


@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_checkpoint_payload_key_binding_fails_closed(store_kind: str) -> None:
    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()
    first_key = f"redis-payload-first-{uuid4().hex}"
    second_key = f"redis-payload-second-{uuid4().hex}"
    first = _minimal_checkpoint(key=first_key)
    second = _minimal_checkpoint(key=second_key)
    assert store.create_checkpoint(first)
    assert store.create_checkpoint(second)
    first_data_key, first_lease_key = store._keys(first_key)  # type: ignore[attr-defined]
    second_data_key, second_lease_key = store._keys(second_key)  # type: ignore[attr-defined]
    first_payload = store._client.get(first_data_key)  # type: ignore[attr-defined]
    second_payload = store._client.get(second_data_key)  # type: ignore[attr-defined]
    assert first_payload is not None and second_payload is not None

    try:
        store._client.set(first_data_key, second_payload)  # type: ignore[attr-defined]
        store._client.set(second_data_key, first_payload)  # type: ignore[attr-defined]
        before = (
            store._client.get(first_data_key),  # type: ignore[attr-defined]
            store._client.get(second_data_key),  # type: ignore[attr-defined]
            store._client.get(first_lease_key),  # type: ignore[attr-defined]
            store._client.get(second_lease_key),  # type: ignore[attr-defined]
        )
        with pytest.raises(CheckpointError) as load_error:
            store.load_checkpoint(first_key)
        assert load_error.value.code == "checkpoint_store_conflict"
        with pytest.raises(CheckpointError) as claim_error:
            store.claim_checkpoint(
                first_key,
                1,
                claim_token="payload-swap-owner",
                lease_expires_at_ms=10_000,
                now_ms=1,
                claim_mode="continue",
            )
        assert claim_error.value.code == "checkpoint_store_conflict"
        after = (
            store._client.get(first_data_key),  # type: ignore[attr-defined]
            store._client.get(second_data_key),  # type: ignore[attr-defined]
            store._client.get(first_lease_key),  # type: ignore[attr-defined]
            store._client.get(second_lease_key),  # type: ignore[attr-defined]
        )
        assert after == before
    finally:
        store._client.set(first_data_key, first_payload)  # type: ignore[attr-defined]
        store._client.set(second_data_key, second_payload)  # type: ignore[attr-defined]
        store.delete_checkpoint(first_key)
        store.delete_checkpoint(second_key)


@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_delete_rejects_foreign_deferred_index_member(store_kind: str) -> None:
    from vv_agent.deferred import DeferredResolutionReceipt, DeferredToolHandle
    from vv_agent.runtime.stores.redis import _receipt_to_storage

    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()
    first_key = f"redis-delete-first-{uuid4().hex}"
    second_key = f"redis-delete-second-{uuid4().hex}"
    first = _minimal_checkpoint(key=first_key)
    second = _minimal_checkpoint(key=second_key)
    assert store.create_checkpoint(first)
    assert store.create_checkpoint(second)
    handle = DeferredToolHandle(
        checkpoint_key=second_key,
        operation_id="foreign-delete-operation",
        attempt=1,
        request_digest="c" * 64,
    )
    result = ToolExecutionResult(
        tool_call_id="foreign-delete-call",
        content="accepted",
        status_code=ToolResultStatus.SUCCESS,
    )
    receipt = DeferredResolutionReceipt(
        handle=handle,
        result=result,
        result_digest=canonical_json_sha256(result.to_dict(), "deferred result"),
        event_id=(
            "evt_receipt_"
            + compute_tool_identity_key(
                handle.checkpoint_key,
                handle.operation_id,
                handle.attempt,
                result.tool_call_id,
                handle.request_digest,
            )
        ),
        event_payload_digest="d" * 64,
        receipt_status="succeeded",
    )
    receipt_key = store._receipt_key(handle.key)  # type: ignore[attr-defined]
    first_set_key = store._receipt_set_key(first_key)  # type: ignore[attr-defined]
    first_data_key, first_lease_key = store._keys(first_key)  # type: ignore[attr-defined]
    try:
        store._client.set(receipt_key, _receipt_to_storage(receipt))  # type: ignore[attr-defined]
        store._client.sadd(first_set_key, receipt_key)  # type: ignore[attr-defined]
        before = (
            store._client.get(first_data_key),  # type: ignore[attr-defined]
            store._client.get(first_lease_key),  # type: ignore[attr-defined]
            set(store._client.smembers(first_set_key)),  # type: ignore[attr-defined]
            store._client.get(receipt_key),  # type: ignore[attr-defined]
        )
        with pytest.raises(CheckpointError) as error:
            store.delete_checkpoint(first_key)
        assert error.value.code == "checkpoint_store_conflict"
        after = (
            store._client.get(first_data_key),  # type: ignore[attr-defined]
            store._client.get(first_lease_key),  # type: ignore[attr-defined]
            set(store._client.smembers(first_set_key)),  # type: ignore[attr-defined]
            store._client.get(receipt_key),  # type: ignore[attr-defined]
        )
        assert after == before
    finally:
        store._client.srem(first_set_key, receipt_key)  # type: ignore[attr-defined]
        store._client.delete(receipt_key)  # type: ignore[attr-defined]
        store.delete_checkpoint(first_key)
        store.delete_checkpoint(second_key)


def _invalid_initial_checkpoint(field_name: str) -> Checkpoint:
    checkpoint = _minimal_checkpoint(key=f"invalid-initial-{field_name}")
    if field_name == "revision":
        checkpoint.revision = 1
    elif field_name == "resume_attempt":
        checkpoint.resume_attempt = 2
    elif field_name == "claim":
        checkpoint.claim_token = "owner"
        checkpoint.claimed_cycle = 1
        checkpoint.lease_expires_at_ms = 100
    elif field_name == "terminal":
        checkpoint.status = AgentStatus.COMPLETED
        checkpoint.terminal_result = AgentResult(
            status=AgentStatus.COMPLETED,
            messages=[],
            cycles=[],
            completion_reason=CompletionReason.NO_TOOL_FINISH,
            final_answer="done",
            checkpoint_key=checkpoint.checkpoint_key,
        )
    elif field_name == "host":
        request = {
            "schema_version": "vv-agent.host-interaction-request.v1",
            "interaction_id": "interaction-create",
            "logical_cycle": 1,
            "operation_id": "op-host-create",
            "tool_call_id": "call-host-create",
            "prompt": "Choose an option.",
        }
        request["request_digest"] = canonical_json_sha256(request, "host interaction request")
        checkpoint.status = AgentStatus.HOST_INTERACTION
        checkpoint.active_host_interaction = request
    elif field_name == "suspended":
        checkpoint.status = AgentStatus.SUSPENDED
        checkpoint.suspended_origin = {"status": "running", "active_host_interaction": None}
    elif field_name == "outbox":
        event = RunStateChangedEvent(
            run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
            state="running",
            cycle_index=1,
            event_id="evt-invalid-create",
        ).to_dict()
        checkpoint.event_outbox = [EventOutboxEntry.pending("evt-invalid-create", event)]
    elif field_name == "cursor":
        checkpoint.event_cursor = EventCursor(
            store_ref={"id": "events.test", "version": "1"},
            value={"sequence": 1},
            last_event_id="evt-cursor",
        )
    elif field_name == "journal":
        entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
        entry.cycle_index = 1
        checkpoint.tool_journal = [entry]
    else:
        raise AssertionError(field_name)
    return checkpoint


@pytest.mark.parametrize(
    "field_name",
    ["revision", "resume_attempt", "claim", "terminal", "host", "suspended", "outbox", "cursor", "journal"],
)
@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_create_rejects_every_non_initial_state_before_store_write(
    field_name: str,
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"initial-invalid-{field_name}-{store_kind}")
    checkpoint = _invalid_initial_checkpoint(field_name)

    with pytest.raises(CheckpointError) as error:
        store.create_checkpoint(checkpoint)

    assert error.value.code == "checkpoint_initial_invalid"
    assert store.load_checkpoint(checkpoint.checkpoint_key) is None
    if store_kind == "redis":
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        assert store._client.get(data_key) is None  # type: ignore[attr-defined]
        assert store._client.get(lease_key) is None  # type: ignore[attr-defined]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_tool_batch_outbox_preflight_proves_writable_before_provider_effect(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"outbox-preflight-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"outbox-preflight-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="claim-preflight",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None

    assert store.preflight_tool_batch(
        claimed,
        tool_call_count=2,
        expected_revision=claimed.revision,
        claim_token="claim-preflight",
        claimed_cycle=1,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_tool_batch_outbox_preflight_rejects_without_active_claim(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"outbox-preflight-negative-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"outbox-preflight-negative-{store_kind}")
    assert store.create_checkpoint(checkpoint)

    assert not store.preflight_tool_batch(
        checkpoint,
        tool_call_count=1,
        expected_revision=checkpoint.revision,
        claim_token="missing-claim",
        claimed_cycle=1,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_progress_merges_authoritative_and_caller_event_outbox(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"progress-outbox-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"progress-outbox-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    authoritative_event = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="cancel_requested",
        cycle_index=1,
        event_id="evt-controller-cancel",
    ).to_dict()
    claimed.event_outbox = [EventOutboxEntry.pending("evt-controller-cancel", authoritative_event)]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed = store.load_checkpoint(checkpoint.checkpoint_key)
    assert claimed is not None
    caller_event = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="running",
        cycle_index=1,
        event_id="evt-caller-progress",
    ).to_dict()
    claimed.event_outbox = [EventOutboxEntry.pending("evt-caller-progress", caller_event)]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert [entry.event_id for entry in persisted.event_outbox] == ["evt-controller-cancel", "evt-caller-progress"]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_terminal_cas_preserves_authoritative_cancel_cursor_and_events(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"terminal-authority-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"terminal-authority-{store_kind}")
    first = EventOutboxEntry.pending(
        "evt-authoritative",
        _checkpoint_created_event(event_id="evt-authoritative", checkpoint=checkpoint),
    )
    checkpoint.event_outbox = [first]
    assert store.create_checkpoint(checkpoint)
    cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id=first.event_id,
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=first.event_id,
        payload_digest=first.payload_digest,
        cursor=cursor,
        expected_revision=0,
        claim_token=None,
    )
    current = store.load_checkpoint(checkpoint.checkpoint_key)
    assert current is not None
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-authority",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None

    cancelled = store.load_checkpoint(checkpoint.checkpoint_key)
    assert cancelled is not None
    cancelled.cancel_requested = True
    if store_kind == "memory":
        store._store[checkpoint.checkpoint_key] = cancelled  # type: ignore[attr-defined]
    elif store_kind == "sqlite":
        with store._lock, store._conn:  # type: ignore[attr-defined]
            store._conn.execute(  # type: ignore[attr-defined]
                "UPDATE checkpoints SET cancel_requested = 1 WHERE checkpoint_key = ?",
                (checkpoint.checkpoint_key,),
            )
    else:
        from vv_agent.runtime.stores.redis import _checkpoint_to_storage

        payload, lease = _checkpoint_to_storage(cancelled)
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        store._client.set(data_key, payload)  # type: ignore[attr-defined]
        if lease is not None:
            store._client.set(lease_key, str(lease))  # type: ignore[attr-defined]

    candidate = store.load_checkpoint(checkpoint.checkpoint_key)
    assert candidate is not None
    candidate.status = AgentStatus.FAILED
    candidate.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=candidate.messages,
        cycles=candidate.cycles,
        error={"code": "agent_failed", "message": "terminal failure", "retryable": False},
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=candidate.checkpoint_key,
    )
    second_event = RunFailedEvent(
        run_id=candidate.root_run_id,
        trace_id=candidate.trace_id,
        error="terminal failure",
        cycle_index=candidate.cycle_index,
        event_id="evt-candidate-terminal",
    ).to_dict()
    candidate.event_outbox.append(EventOutboxEntry.pending(second_event["event_id"], second_event))
    assert store.finalize_claimed_checkpoint(
        candidate,
        claim_token="owner-authority",
        expected_revision=candidate.revision,
    )

    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.cancel_requested
    assert terminal.event_cursor == cursor
    assert [entry.event_id for entry in terminal.event_outbox] == [first.event_id, "evt-candidate-terminal"]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_unclaimed_terminal_cas_preserves_authoritative_cancel_cursor_and_events(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"unclaimed-terminal-authority-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"unclaimed-terminal-authority-{store_kind}")
    first = EventOutboxEntry.pending(
        "evt-unclaimed-authoritative",
        _checkpoint_created_event(event_id="evt-unclaimed-authoritative", checkpoint=checkpoint),
    )
    checkpoint.event_outbox = [first]
    assert store.create_checkpoint(checkpoint)
    cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id=first.event_id,
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=first.event_id,
        payload_digest=first.payload_digest,
        cursor=cursor,
        expected_revision=0,
        claim_token=None,
    )
    current = store.load_checkpoint(checkpoint.checkpoint_key)
    assert current is not None
    current.cancel_requested = True
    if store_kind == "memory":
        store._store[checkpoint.checkpoint_key] = current  # type: ignore[attr-defined]
    elif store_kind == "sqlite":
        with store._lock, store._conn:  # type: ignore[attr-defined]
            store._conn.execute(  # type: ignore[attr-defined]
                "UPDATE checkpoints SET cancel_requested = 1 WHERE checkpoint_key = ?",
                (checkpoint.checkpoint_key,),
            )
    else:
        from vv_agent.runtime.stores.redis import _checkpoint_to_storage

        payload, lease = _checkpoint_to_storage(current)
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        store._client.set(data_key, payload)  # type: ignore[attr-defined]
        if lease is not None:
            store._client.set(lease_key, str(lease))  # type: ignore[attr-defined]

    candidate = store.load_checkpoint(checkpoint.checkpoint_key)
    assert candidate is not None
    candidate.status = AgentStatus.FAILED
    candidate.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=candidate.messages,
        cycles=candidate.cycles,
        error={"code": "agent_failed", "message": "terminal failure", "retryable": False},
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=candidate.checkpoint_key,
    )
    second_event = RunFailedEvent(
        run_id=candidate.root_run_id,
        trace_id=candidate.trace_id,
        error="terminal failure",
        cycle_index=candidate.cycle_index,
        event_id="evt-unclaimed-candidate-terminal",
    ).to_dict()
    candidate.event_outbox.append(EventOutboxEntry.pending(second_event["event_id"], second_event))
    assert store.finalize_checkpoint(candidate, expected_revision=candidate.revision)

    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.cancel_requested
    assert terminal.event_cursor == cursor
    assert [entry.event_id for entry in terminal.event_outbox] == [
        first.event_id,
        "evt-unclaimed-candidate-terminal",
    ]


def _journal_case(name: str) -> dict[str, Any]:
    fixture = _fixture("operation_journal.json")
    return deepcopy(next(case["entry"] for case in fixture["valid_entries"] if case["name"] == name))


@pytest.mark.parametrize(
    "vector_name",
    ["error_custom_metadata_non_default_directive", "error_truncated_artifact", "error_truncated_cursor"],
)
def test_failed_tool_receipt_keeps_complete_result_and_projection(
    vector_name: str,
) -> None:
    fixture = _fixture("operation_journal.json")
    vector = next(item for item in fixture["receipt_identity"]["result_digest_vectors"] if item["name"] == vector_name)
    result = ToolExecutionResult.from_dict(vector["wire"])
    payload = _journal_case("tool_failed")
    payload.update(
        {
            "tool_call_id": result.tool_call_id,
            "result": result.to_dict(),
            "result_digest": vector["rfc8785_sha256"],
            "error": {
                "code": result.error_code or "tool_operation_failed",
                "message": result.content or "tool operation failed",
                "retryable": result.metadata.get("retryable", False),
            },
        }
    )

    entry = OperationJournalEntry.from_dict(payload)

    assert entry.result == result.to_dict()
    assert entry.result_digest == vector["rfc8785_sha256"]
    assert entry.error == OperationError(
        code=result.error_code or "tool_operation_failed",
        message=result.content or "tool operation failed",
        retryable=result.metadata.get("retryable", False),
    )


def test_failed_tool_receipt_uses_projection_fallback_for_empty_error_code() -> None:
    result = ToolExecutionResult(
        tool_call_id="call-projection-fallback",
        content="provider rejected without a typed code",
        status_code=ToolResultStatus.ERROR,
        error_code="",
    )
    payload = _journal_case("tool_failed")
    payload.update(
        {
            "tool_call_id": result.tool_call_id,
            "result": result.to_dict(),
            "result_digest": canonical_json_sha256(result.to_dict(), "tool result"),
            "error": {
                "code": "tool_operation_failed",
                "message": result.content,
                "retryable": False,
            },
        }
    )

    entry = OperationJournalEntry.from_dict(payload)

    assert result.to_dict()["error_code"] == ""
    assert entry.error == OperationError(
        code="tool_operation_failed",
        message=result.content,
        retryable=False,
    )


@pytest.mark.parametrize(
    "vector_name",
    ["error_custom_metadata_non_default_directive", "error_truncated_artifact", "error_truncated_cursor"],
)
def test_failed_tool_receipt_recovery_replays_exact_result_without_dispatch(
    vector_name: str,
) -> None:
    fixture = _fixture("operation_journal.json")
    vector = next(item for item in fixture["receipt_identity"]["result_digest_vectors"] if item["name"] == vector_name)
    result = ToolExecutionResult.from_dict(vector["wire"])
    key = f"failed-replay-{vector_name}"
    store = InMemoryCheckpointStore()
    seed = _minimal_checkpoint(key=key)
    first = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
        task_id=seed.task_id,
        run_id=seed.root_run_id,
        trace_id=seed.trace_id,
        run_definition=deepcopy(seed.run_definition),
        run_definition_digest=seed.run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )
    call = ToolCall(id=result.tool_call_id, name="write_record", arguments={})
    try:
        assert first.admit() is None
        first.plan_tool(cycle_index=1, call=call, idempotency_support=ToolIdempotency.UNKNOWN)
        first.tool_started(cycle_index=1, call=call)
        first.finish_tool(cycle_index=1, call=call, result=result)
        retained = store.load_checkpoint(key)
        assert retained is not None
        assert retained.tool_journal[0].result == result.to_dict()
        assert retained.tool_journal[0].result_digest == vector["rfc8785_sha256"]
    finally:
        first.close()

    with store._lock:
        store._store[key].lease_expires_at_ms = 1

    second = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
        task_id=seed.task_id,
        run_id=seed.root_run_id,
        trace_id=seed.trace_id,
        run_definition=deepcopy(seed.run_definition),
        run_definition_digest=seed.run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )
    try:
        assert second.admit() is None
        plan = second.plan_tool(cycle_index=1, call=call, idempotency_support=ToolIdempotency.UNKNOWN)
        assert plan.replay_result is not None
        assert plan.replay_result.to_dict() == result.to_dict()
        assert plan.replay_result.directive is result.directive
    finally:
        second.close()
        store.delete_checkpoint(key)


def _model_event_kwargs(checkpoint: Checkpoint, journal: OperationJournalEntry) -> dict[str, Any]:
    assert journal.call_id is not None
    assert journal.model_operation is not None
    assert journal.backend is not None
    assert journal.model is not None
    return {
        "run_id": checkpoint.root_run_id,
        "trace_id": checkpoint.trace_id,
        "call_id": journal.call_id,
        "operation_id": journal.operation_id,
        "attempt": journal.attempt,
        "operation": journal.model_operation,
        "cycle_index": journal.cycle_index,
        "backend": journal.backend,
        "model": journal.model,
    }


def _append_outbox_event(checkpoint: Checkpoint, event: Any) -> None:
    payload = event.to_dict()
    checkpoint.event_outbox.append(EventOutboxEntry.pending(payload["event_id"], payload))


def _attach_model_accounting(
    checkpoint: Checkpoint,
    journal: OperationJournalEntry,
    *,
    status: ModelCallStatus | None = None,
    usage: TokenUsage | None = None,
    error_code: str = "provider_rejected",
) -> None:
    identity = _model_event_kwargs(checkpoint, journal)
    _append_outbox_event(
        checkpoint,
        ModelCallStartedEvent(
            **identity,
            event_id=f"evt-{journal.call_id}-started",
            created_at=100.0,
        ),
    )
    if status is None:
        return

    effective_usage = usage or TokenUsage()
    checkpoint.model_calls.append(
        ModelCallRecord(
            call_id=identity["call_id"],
            operation_id=identity["operation_id"],
            attempt=identity["attempt"],
            operation=identity["operation"],
            cycle_index=identity["cycle_index"],
            backend=identity["backend"],
            model=identity["model"],
            status=status,
            usage=deepcopy(effective_usage),
            error_code=None if status is ModelCallStatus.COMPLETED else error_code,
        )
    )
    if status is ModelCallStatus.COMPLETED:
        terminal_event = ModelCallCompletedEvent(
            **identity,
            usage=deepcopy(effective_usage),
            event_id=f"evt-{journal.call_id}-completed",
            created_at=101.0,
        )
    else:
        terminal_event = ModelCallFailedEvent(
            **identity,
            outcome="ambiguous" if status is ModelCallStatus.AMBIGUOUS else "definitive",
            usage=deepcopy(effective_usage),
            error_code=error_code,
            event_id=f"evt-{journal.call_id}-failed",
            created_at=101.0,
        )
    _append_outbox_event(checkpoint, terminal_event)


def _checkpoint_with_model_journal(
    journal_case: str,
    *,
    status: ModelCallStatus | None = None,
    error_code: str = "provider_rejected",
) -> Checkpoint:
    checkpoint = _minimal_checkpoint(key=f"accounting-{journal_case}")
    checkpoint.claim_token = "owner"
    checkpoint.claimed_cycle = 1
    checkpoint.lease_expires_at_ms = 200
    journal = OperationJournalEntry.from_dict(_journal_case(journal_case))
    checkpoint.model_call_journal = [journal]
    if journal.state is OperationState.AMBIGUOUS:
        checkpoint.status = AgentStatus.RECONCILIATION_REQUIRED
        checkpoint.claim_token = None
        checkpoint.claimed_cycle = None
        checkpoint.lease_expires_at_ms = None
    if journal.state is not OperationState.PLANNED and not (journal.state is OperationState.FAILED and status is None):
        _attach_model_accounting(
            checkpoint,
            journal,
            status=status,
            error_code=error_code,
        )
    return checkpoint


def test_rfc8785_vectors_match_canonical_bytes_and_digests() -> None:
    vectors = (
        ("run_definition.json", ("golden_cases",), "definition"),
        ("operation_journal.json", ("request_digest", "golden_cases"), "request"),
        ("checkpoint_codec.json", ("extension_limits", "canonicalization_vectors"), "entry"),
        ("checkpoint_store.json", ("event_payload_digest", "golden_cases"), "event"),
    )
    for fixture_name, path, value_field in vectors:
        value: Any = _fixture(fixture_name)
        for part in path:
            value = value[part]
        for vector in value:
            canonical = canonical_json_bytes(vector[value_field])
            assert base64.b64encode(canonical).decode("ascii") == vector["canonical_json_base64"]
            assert len(canonical) == vector["canonical_json_utf8_bytes"]
            assert hashlib.sha256(canonical).hexdigest() == vector["sha256"]


def test_run_definition_digest_matches_both_golden_vectors() -> None:
    fixture = _fixture("run_definition.json")
    for case in fixture["golden_cases"]:
        assert compute_run_definition_digest(case["definition"]) == case["sha256"]


def test_checkpoint_round_trip_uses_jcs() -> None:
    fixture = _fixture("checkpoint_codec.json")
    checkpoint = checkpoint_from_dict(fixture["canonical_checkpoint"])
    encoded = checkpoint_to_dict(checkpoint)

    assert encoded == fixture["canonical_checkpoint"]
    wire = checkpoint_to_json(checkpoint)
    assert wire.encode("utf-8") == canonical_json_bytes(encoded)
    assert checkpoint_from_json(wire) == checkpoint


def test_checkpoint_round_trip_preserves_message_artifact_ref() -> None:
    checkpoint = _minimal_checkpoint(key="message-artifact-ref")
    artifact_ref = ToolArtifactRef(
        path=".vv-agent/artifacts/run-7/call-search.txt",
        media_type="text/plain",
        encoding="utf-8",
        size_bytes=42,
        sha256="0" * 64,
    )
    checkpoint.messages.append(
        Message(
            role="tool",
            content="<Tool Result Compact>\nartifact_path: .vv-agent/artifacts/run-7/call-search.txt",
            tool_call_id="call-search",
            artifact_ref=artifact_ref,
        )
    )

    restored = checkpoint_from_json(checkpoint_to_json(checkpoint))

    assert restored.messages[-1].artifact_ref == artifact_ref


def test_checkpoint_and_operation_receipt_preserve_bounded_tool_result_fields() -> None:
    checkpoint_payload = _fixture("checkpoint_codec.json")["canonical_checkpoint"]
    expected_cycle_result = checkpoint_payload["cycles"][0]["tool_results"][0]
    checkpoint = checkpoint_from_dict(checkpoint_payload)

    assert checkpoint.cycles[0].tool_results[0].to_dict() == expected_cycle_result
    assert checkpoint_to_dict(checkpoint)["cycles"][0]["tool_results"][0] == expected_cycle_result

    expected_receipt = _journal_case("tool_succeeded_truncated_bash")
    entry = OperationJournalEntry.from_dict(expected_receipt)
    assert entry.result == expected_receipt["result"]
    assert entry.to_dict() == expected_receipt


def test_cycle_aborted_follows_live_cancel_signal_before_terminal_event() -> None:
    from vv_agent.runtime.state import _append_cycle_aborted_event

    checkpoint = _minimal_checkpoint(key="cycle-aborted-live-cancel-order")
    live_cancel = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="running",
        cancel_requested={"from": False, "to": True},
        event_id="evt-live-cancel",
    ).to_dict()
    terminal = RunCancelledEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        reason="cancelled",
        event_id="evt-run-cancelled",
    ).to_dict()
    checkpoint.event_outbox = [
        EventOutboxEntry.pending("evt-live-cancel", live_cancel),
        EventOutboxEntry.pending("evt-run-cancelled", terminal),
    ]

    _append_cycle_aborted_event(checkpoint, logical_cycle=1, reason="cancelled")

    assert [entry.event["type"] for entry in checkpoint.event_outbox] == [
        "run_state_changed",
        "cycle_aborted",
        "run_cancelled",
    ]


def test_checkpoint_round_trip_restores_jcs_large_float_through_codec_and_sqlite(
    tmp_path: Path,
) -> None:
    definition_case = next(
        case for case in _fixture("run_definition.json")["golden_cases"] if case["name"] == "full_unicode_float_and_capabilities"
    )
    checkpoint = _minimal_checkpoint(key="jcs-large-float")
    checkpoint.run_definition = deepcopy(definition_case["definition"])
    checkpoint.run_definition_digest = definition_case["sha256"]

    wire = checkpoint_to_json(checkpoint)
    assert '"large_number":100000000000000000000' in wire
    decoded = checkpoint_from_json(wire)
    large_number = decoded.run_definition["model"]["settings"]["extra_body"]["large_number"]
    assert isinstance(large_number, float)
    assert large_number == 1e20

    store = SqliteCheckpointStore(tmp_path / "jcs-large-float.sqlite3")
    assert store.create_checkpoint(checkpoint)
    restored = store.load_checkpoint(checkpoint.checkpoint_key)
    assert restored is not None
    restored_large_number = restored.run_definition["model"]["settings"]["extra_body"]["large_number"]
    assert isinstance(restored_large_number, float)
    assert restored_large_number == 1e20
    assert restored.run_definition_digest == definition_case["sha256"]


def test_run_definition_rejects_host_integer_above_i_json_safe_range() -> None:
    definition = deepcopy(_fixture("run_definition.json")["golden_cases"][0]["definition"])
    definition["model"]["settings"] = {
        "extra_body": {"count": 9_007_199_254_740_992},
    }

    with pytest.raises(CheckpointError) as error:
        compute_run_definition_digest(definition)

    assert error.value.code == "checkpoint_definition_not_i_json"


def test_checkpoint_validates_embedded_definition_schema_and_digest() -> None:
    payload = _codec_case("minimal_running")
    assert checkpoint_from_dict(payload).run_definition_digest == compute_run_definition_digest(payload["run_definition"])

    missing_definition = deepcopy(payload)
    missing_definition.pop("run_definition")
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(missing_definition)
    assert error.value.code == "checkpoint_missing_field"

    missing_definition_schema = deepcopy(payload)
    missing_definition_schema.pop("run_definition_schema")
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(missing_definition_schema)
    assert error.value.code == "checkpoint_missing_field"

    mismatch = deepcopy(payload)
    mismatch["run_definition"]["root_input"] = "different"
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(mismatch)
    assert error.value.code == "checkpoint_definition_mismatch"

    unknown = deepcopy(payload)
    unknown["schema_version"] = "vv-agent.checkpoint.v3"
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(unknown)
    assert error.value.code == "checkpoint_schema_unsupported"


def test_checkpoint_resume_rejects_definition_mismatch_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(case for case in _fixture("run_definition.json")["golden_cases"] if case["name"] == "minimal")
    stored_definition = deepcopy(case["definition"])
    stored_digest = case["sha256"]
    current_definition = deepcopy(stored_definition)
    current_definition["root_input"] = "Different input"
    checkpoint_key = "definition-mismatch"
    store = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        checkpoint_key=checkpoint_key,
        task_id="task",
        root_run_id="run",
        trace_id="trace",
        run_definition=deepcopy(stored_definition),
        run_definition_digest=stored_digest,
        resume_attempt=1,
        cycle_index=0,
        status=AgentStatus.RUNNING,
        messages=[],
        cycles=[],
    )
    assert store.create_checkpoint(checkpoint)
    claims = 0
    original_claim = store.claim_checkpoint

    def count_claim(*args: Any, **kwargs: Any) -> Any:
        nonlocal claims
        claims += 1
        return original_claim(*args, **kwargs)

    monkeypatch.setattr(store, "claim_checkpoint", count_claim)
    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=checkpoint_key,
            resume_policy=ResumePolicy.REQUIRE_EXISTING,
        ),
        task_id="task",
        run_id="run",
        trace_id="trace",
        run_definition=current_definition,
        run_definition_digest=compute_run_definition_digest(current_definition),
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )

    with pytest.raises(CheckpointError) as error:
        controller.admit()

    assert error.value.code == "checkpoint_definition_mismatch"
    assert claims == 0
    retained = store.load_checkpoint(checkpoint_key)
    assert retained is not None
    assert retained.run_definition == stored_definition
    assert retained.run_definition_digest == stored_digest


def test_checkpoint_invalid_fixture_cases_have_stable_codes() -> None:
    fixture = _fixture("checkpoint_codec.json")
    expected_codes = {
        "old_v9_schema_is_rejected_forward_only": "checkpoint_schema_unsupported",
        "unknown_schema": "checkpoint_schema_unsupported",
        "blank_checkpoint_key": "checkpoint_key_invalid",
        "bad_definition_digest": "checkpoint_definition_digest_invalid",
        "zero_resume_attempt": "checkpoint_resume_attempt_invalid",
        "partial_claim": "checkpoint_claim_invalid",
        "claimed_cycle_not_next": "checkpoint_claim_invalid",
        "terminal_with_claim": "checkpoint_status_invalid",
        "journal_cycle_not_active": "checkpoint_journal_cycle_invalid",
        "unknown_required_extension": "checkpoint_required_extension_unavailable",
        "invalid_extension_namespace": "checkpoint_extension_namespace_invalid",
        "unknown_top_level_is_rejected": "checkpoint_unknown_field",
        "cancel_requested_not_boolean": "checkpoint_status_invalid",
        "terminal_result_with_active_tool_journal_is_invalid": "checkpoint_status_invalid",
    }
    for case in fixture["invalid_cases"]:
        registered = [] if case["name"] == "unknown_required_extension" else None
        payload = case.get("payload")
        if payload is None:
            base_name = case.get("base_valid_case")
            payload = deepcopy(next(item["payload"] for item in fixture["valid_cases"] if item["name"] == base_name))
            for field_name, replacement in case.get("mutation", {}).get("replace", {}).items():
                payload[field_name] = replacement
        with pytest.raises(CheckpointError) as error:
            checkpoint_from_dict(
                payload,
                registered_extensions=registered,
            )
        assert error.value.code == expected_codes[case["name"]]

    for case in fixture["status_cases"]:
        if "base_valid_case" not in case:
            continue
        payload = _codec_case(case["base_valid_case"])
        payload.update(case["mutation"]["replace"])
        with pytest.raises(CheckpointError) as error:
            checkpoint_from_dict(payload)
        assert error.value.code == case["error_code"]


def test_extension_size_counts_complete_entry_jcs_bytes() -> None:
    exact = {"version": "1", "required": False, "state": "x" * 65_493}
    over = {"version": "1", "required": False, "state": "x" * 65_494}
    assert len(canonical_json_bytes(exact)) == 65_536
    validate_extension_state_size(
        {"com.example.exact": exact},
        max_extension_state_bytes=262_144,
    )
    with pytest.raises(CheckpointError) as error:
        validate_extension_state_size(
            {"com.example.over": over},
            max_extension_state_bytes=262_144,
        )
    assert error.value.code == "checkpoint_extension_entry_too_large"

    entries: dict[str, dict[str, Any]] = {
        f"com.example.e{index}": {
            "version": "1",
            "required": False,
            "state": "x" * repetitions,
        }
        for index, repetitions in enumerate((65_493, 65_493, 65_493, 65_450, 0))
    }
    validate_extension_state_size(entries, max_extension_state_bytes=262_144)
    entries["com.example.e3"]["state"] += "x"
    with pytest.raises(CheckpointError) as error:
        validate_extension_state_size(entries, max_extension_state_bytes=262_144)
    assert error.value.code == "checkpoint_extension_state_too_large"


def test_extensions_are_validated_by_duck_typing() -> None:
    class DuckExtension:
        namespace = "org.example.future"
        version = "9"
        required = True

        def snapshot(self) -> Any:
            return {"opaque": True}

        def restore(self, state: Any) -> None:
            self.state = state

    extension = DuckExtension()
    assert isinstance(extension, CheckpointExtension)
    validate_checkpoint_extension(extension)
    payload = _codec_case("minimal_running")
    payload["extension_state"] = {
        extension.namespace: {
            "version": extension.version,
            "required": True,
            "state": {"opaque": True},
        }
    }
    assert checkpoint_from_dict(payload, registered_extensions=[extension])
    with pytest.raises(CheckpointError) as error:
        checkpoint_from_dict(payload, registered_extensions=[])
    assert error.value.code == "checkpoint_required_extension_unavailable"


def test_operation_and_event_digest_golden_vectors() -> None:
    journal_fixture = _fixture("operation_journal.json")
    for case in journal_fixture["request_digest"]["golden_cases"]:
        assert compute_operation_request_digest(case["request"]) == case["sha256"]
    model_request = journal_fixture["request_digest"]["golden_cases"][0]["request"]
    model_entry = OperationJournalEntry.from_dict(_journal_case("model_planned"))
    model_entry.verify_request(model_request)
    changed_request = deepcopy(model_request)
    changed_request["request"]["messages"][0]["content"] = "different"
    with pytest.raises(CheckpointError) as error:
        model_entry.verify_request(changed_request)
    assert error.value.code == "checkpoint_journal_integrity_mismatch"

    store_fixture = _fixture("checkpoint_store.json")
    event_case = store_fixture["event_payload_digest"]["golden_cases"][0]
    assert compute_event_payload_digest(event_case["event"]) == event_case["sha256"]
    pending = EventOutboxEntry.pending("evt-1", event_case["event"])
    assert pending.payload_digest == event_case["sha256"]
    pending.verify_payload()


@pytest.mark.parametrize(
    ("support", "key", "error_code"),
    [
        ("unsupported", "idem_unexpected", "tool_idempotency_key_invalid"),
        ("supported", None, "tool_idempotency_key_required"),
        ("unknown", None, "tool_idempotency_key_required"),
    ],
)
def test_tool_journal_idempotency_key_matches_support(support: str, key: str | None, error_code: str) -> None:
    entry = _journal_case("tool_started")
    entry["idempotency_support"] = support
    entry["idempotency_key"] = key
    with pytest.raises(CheckpointError) as error:
        OperationJournalEntry.from_dict(entry)
    assert error.value.code == error_code


def test_operation_journal_invalid_cases_have_stable_codes() -> None:
    fixture = _fixture("operation_journal.json")
    for case in fixture["valid_entries"]:
        OperationJournalEntry.from_dict(case["entry"])
    for case in fixture["invalid_entries"]:
        if "base_valid_entry" in case:
            entry = _journal_case(case["base_valid_entry"])
            mutation = case["mutation"]
            if "remove" in mutation:
                entry.pop(mutation["remove"])
            if "replace" in mutation:
                for field_name, replacement in mutation["replace"].items():
                    target = entry
                    parts = field_name.split(".")
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = replacement
            if "add" in mutation:
                entry.update(mutation["add"])
        else:
            entry = case["entry"]
        with pytest.raises(CheckpointError) as error:
            OperationJournalEntry.from_dict(entry)
        assert error.value.code == case["error_code"], case["name"]


@pytest.mark.parametrize(
    ("journal_case", "mutation", "error_code"),
    [
        ("tool_started", ("remove", "result"), "operation_kind_fields_invalid"),
        ("tool_started", ("remove", "idempotency_key"), "operation_kind_fields_invalid"),
        ("tool_failed", ("null", "result"), "operation_result_required"),
        ("tool_failed_tool_cancelled_closure", ("remove", "result"), "operation_result_required"),
        (
            "tool_failed_tool_cancelled_closure",
            ("null", "result_digest"),
            "operation_closure_receipt_forbidden",
        ),
        ("tool_succeeded", ("null", "result_digest"), "operation_result_digest_required"),
        ("model_planned", ("remove", "response"), "operation_kind_fields_invalid"),
        ("model_planned", ("null", "call_id"), "model_identity_invalid"),
        ("tool_started", ("add", "future_field"), "operation_entry_unknown_field"),
        ("model_planned", ("add", "future_field"), "operation_entry_unknown_field"),
        ("tool_failed_tool_cancelled_closure", ("add", "future_field"), "operation_entry_unknown_field"),
    ],
)
def test_operation_journal_reader_rejects_missing_null_and_unknown_fields(
    journal_case: str,
    mutation: tuple[str, str],
    error_code: str,
) -> None:
    entry = _journal_case(journal_case)
    action, field_name = mutation
    if action == "remove":
        entry.pop(field_name)
    elif action == "null":
        entry[field_name] = None
    else:
        entry[field_name] = True

    with pytest.raises(CheckpointError) as error:
        OperationJournalEntry.from_dict(entry)
    assert error.value.code == error_code


@pytest.mark.parametrize(
    ("journal_case", "state", "error_code"),
    [
        ("tool_planned", "planned", "operation_receipt_unexpected"),
        ("tool_started", "started", "operation_receipt_unexpected"),
        ("tool_started", "ambiguous", "operation_receipt_unexpected"),
        ("tool_deferred", "deferred", "operation_deferred_fields_invalid"),
    ],
)
def test_active_tool_journal_reader_rejects_resume_observation(
    journal_case: str,
    state: str,
    error_code: str,
) -> None:
    entry = _journal_case(journal_case)
    entry["state"] = state
    entry["resume_observation"] = _journal_case("tool_failed_tool_cancelled_closure")["resume_observation"]

    with pytest.raises(CheckpointError) as error:
        OperationJournalEntry.from_dict(entry)

    assert error.value.code == error_code


def test_closed_tool_cancelled_journal_reader_accepts_resume_observation() -> None:
    entry = _journal_case("tool_failed_tool_cancelled_closure")

    decoded = OperationJournalEntry.from_dict(entry)

    assert decoded.error is not None
    assert decoded.error.code == "tool_cancelled"
    assert decoded.resume_observation is not None
    assert decoded.to_dict() == entry


@pytest.mark.parametrize(
    ("journal_case", "status"),
    [
        ("model_planned", None),
        ("model_started", None),
        ("model_succeeded", ModelCallStatus.COMPLETED),
        ("model_failed", None),
        ("model_failed", ModelCallStatus.FAILED),
        ("model_ambiguous", ModelCallStatus.AMBIGUOUS),
    ],
)
def test_model_journal_accepts_complete_atomic_accounting_states(
    journal_case: str,
    status: ModelCallStatus | None,
) -> None:
    checkpoint_to_dict(_checkpoint_with_model_journal(journal_case, status=status))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("call_id", "different-call"),
        ("operation_id", "different-operation"),
        ("attempt", 2),
        ("operation", "session_memory"),
        ("cycle_index", 2),
        ("backend", "different-backend"),
        ("model", "different-model"),
    ],
)
def test_model_started_event_identity_must_match_journal(
    field: str,
    replacement: Any,
) -> None:
    checkpoint = _checkpoint_with_model_journal("model_started")
    event = deepcopy(checkpoint.event_outbox[0].event)
    event[field] = replacement
    checkpoint.event_outbox[0] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_model_started_journal_requires_atomic_started_event() -> None:
    checkpoint = _checkpoint_with_model_journal("model_started")
    checkpoint.event_outbox.clear()

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("missing", ["started_event", "terminal_event", "ledger_record"])
def test_terminal_model_journal_requires_complete_atomic_evidence(missing: str) -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    if missing == "started_event":
        checkpoint.event_outbox.pop(0)
    elif missing == "terminal_event":
        checkpoint.event_outbox.pop()
    else:
        checkpoint.model_calls.clear()

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_terminal_model_event_usage_must_match_ledger() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    event = deepcopy(checkpoint.event_outbox[-1].event)
    event["usage"] = TokenUsage(input_tokens=1, total_tokens=1).to_dict()
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_failed_model_event_error_must_match_ledger() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_failed",
        status=ModelCallStatus.FAILED,
    )
    event = deepcopy(checkpoint.event_outbox[-1].event)
    event["error_code"] = "different_error"
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(event["event_id"], event)

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_terminal_model_event_type_must_match_ledger_status() -> None:
    checkpoint = _checkpoint_with_model_journal(
        "model_succeeded",
        status=ModelCallStatus.COMPLETED,
    )
    journal = checkpoint.model_call_journal[0]
    failed_event = ModelCallFailedEvent(
        **_model_event_kwargs(checkpoint, journal),
        outcome="definitive",
        usage=deepcopy(checkpoint.model_calls[0].usage),
        error_code="provider_rejected",
        event_id=f"evt-{journal.call_id}-failed",
        created_at=101.0,
    )
    checkpoint.event_outbox[-1] = EventOutboxEntry.pending(
        failed_event.event_id,
        failed_event.to_dict(),
    )

    with pytest.raises(CheckpointError) as error:
        checkpoint_to_dict(checkpoint)

    assert error.value.code == "checkpoint_status_invalid"


def test_checkpoint_config_validates_store_key_capabilities_and_stable_codes() -> None:
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(
        store=store,
        key=None,
        capability_refs={"runtime_hook:0": {"id": "hook.audit", "version": "1"}},
    )
    assert config.resume_policy is ResumePolicy.NEW
    assert isinstance(store, CheckpointStore)

    invalid_cases: tuple[tuple[Callable[[], CheckpointConfig], str], ...] = (
        (lambda: CheckpointConfig(store=store, key="x" * 513), "checkpoint_key_invalid"),
        (
            lambda: CheckpointConfig(
                store=store,
                key=None,
                resume_policy=ResumePolicy.REQUIRE_EXISTING,
            ),
            "checkpoint_key_required",
        ),
        (
            lambda: CheckpointConfig(
                store=store,
                store_ref={"id": "x", "version": "1"},
            ),
            "checkpoint_store_selection_invalid",
        ),
        (
            lambda: CheckpointConfig(
                store=store,
                capability_refs={"bad slot": {"id": "x", "version": "1"}},
            ),
            "checkpoint_capability_ref_invalid",
        ),
    )
    for factory, code in invalid_cases:
        with pytest.raises(CheckpointError) as error:
            factory()
        assert error.value.code == code


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claim_modes_update_resume_attempt_atomically(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claim-mode")
    checkpoint = _minimal_checkpoint(key=f"claim-{store_kind}")
    assert store.create_checkpoint(checkpoint)

    continued = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-a",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert continued is not None
    assert continued.resume_attempt == 1
    assert continued.revision == 1

    with pytest.raises(CheckpointConflictError):
        store.claim_checkpoint(
            checkpoint.checkpoint_key,
            1,
            claim_token="owner-b",
            lease_expires_at_ms=300,
            now_ms=199,
            claim_mode="recovery",
        )
    rejected = store.load_checkpoint(checkpoint.checkpoint_key)
    assert rejected is not None
    assert rejected.resume_attempt == 1

    recovered = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-b",
        lease_expires_at_ms=300,
        now_ms=200,
        claim_mode="recovery",
    )
    assert recovered is not None
    assert recovered.resume_attempt == 2
    assert recovered.revision == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_concurrent_recovery_claims_increment_once(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "concurrent-recovery")
    checkpoint = _minimal_checkpoint(key=f"concurrent-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    assert store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="expired-owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    barrier = Barrier(3)
    result_lock = Lock()
    claims: list[Checkpoint] = []
    conflicts: list[CheckpointConflictError] = []

    def recover(owner: str) -> None:
        barrier.wait()
        try:
            claimed = store.claim_checkpoint(
                checkpoint.checkpoint_key,
                1,
                claim_token=owner,
                lease_expires_at_ms=300,
                now_ms=200,
                claim_mode="recovery",
            )
        except CheckpointConflictError as exc:
            with result_lock:
                conflicts.append(exc)
        else:
            assert claimed is not None
            with result_lock:
                claims.append(claimed)

    workers = [Thread(target=recover, args=(owner,)) for owner in ("owner-a", "owner-b")]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()

    assert len(claims) == 1
    assert len(conflicts) == 1
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.revision == 2
    assert persisted.resume_attempt == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_progress_and_heartbeat_preserve_claim_and_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "progress")
    checkpoint = _minimal_checkpoint(key=f"progress-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    started = OperationJournalEntry.from_dict(_journal_case("model_started"))
    claimed.model_call_journal = [started]
    _attach_model_accounting(claimed, started)
    claimed.shared_state["progress"] = "started"
    assert store.progress_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    assert store.renew_checkpoint_claim(
        checkpoint.checkpoint_key,
        claim_token="owner",
        lease_expires_at_ms=300,
        now_ms=150,
    )

    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.revision == 2
    assert persisted.claim_token == "owner"
    assert persisted.lease_expires_at_ms == 300
    assert persisted.model_call_journal[0].state is OperationState.STARTED
    assert persisted.shared_state["progress"] == "started"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_live_cancel_preserves_owner_progress_and_one_tool_receipt(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"live-cancel-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"live-cancel-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    entry.cycle_index = 1
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1

    current = store.load_checkpoint(checkpoint.checkpoint_key)
    assert current is not None
    current.cancel_requested = True
    if store_kind == "memory":
        store._store[checkpoint.checkpoint_key] = current  # type: ignore[attr-defined]
    elif store_kind == "sqlite":
        with store._lock, store._conn:  # type: ignore[attr-defined]
            store._conn.execute(  # type: ignore[attr-defined]
                "UPDATE checkpoints SET cancel_requested = 1 WHERE checkpoint_key = ?",
                (checkpoint.checkpoint_key,),
            )
    else:
        from vv_agent.runtime.stores.redis import _checkpoint_to_storage

        payload, lease = _checkpoint_to_storage(current)
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        store._client.set(data_key, payload)  # type: ignore[attr-defined]
        if lease is not None:
            store._client.set(lease_key, str(lease))  # type: ignore[attr-defined]

    claimed.shared_state["after_cancel"] = True
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    result = ToolExecutionResult(
        tool_call_id=entry.tool_call_id or "call-tool",
        content="done",
        status_code=ToolResultStatus.SUCCESS,
    )
    assert store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-tool",
        request_digest=entry.request_digest,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.cancel_requested
    assert persisted.tool_journal[0].state is OperationState.SUCCEEDED
    revision = persisted.revision
    assert store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-tool",
        request_digest=entry.request_digest,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    replayed = store.load_checkpoint(checkpoint.checkpoint_key)
    assert replayed is not None and replayed.revision == revision


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_record_tool_receipt_rejects_planned_operation(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"planned-receipt-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"planned-receipt-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_planned"))
    entry.cycle_index = 1
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    result = ToolExecutionResult(
        tool_call_id=entry.tool_call_id or "call-1",
        content="Tool is not allowed for this run.",
        status_code=ToolResultStatus.ERROR,
        error_code="tool_not_allowed",
    )
    assert not store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-1",
        request_digest=entry.request_digest,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    planned = persisted.tool_journal[0]
    assert planned.state is OperationState.PLANNED
    assert planned.identity_key is None
    assert planned.result_digest is None
    assert planned.error is None
    completed = [event.event for event in persisted.event_outbox if event.event.get("type") == "tool_call_completed"]
    assert completed == []


@pytest.mark.parametrize("journal_state", [OperationState.PLANNED, OperationState.STARTED, OperationState.AMBIGUOUS])
def test_finish_tool_rejects_mismatched_result_before_any_write(journal_state: OperationState) -> None:
    key = f"finish-tool-identity-{journal_state.value}"
    store = InMemoryCheckpointStore()
    checkpoint = _minimal_checkpoint(key=key)
    assert store.create_checkpoint(checkpoint)
    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=key,
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
        ),
        task_id=checkpoint.task_id,
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        run_definition=deepcopy(checkpoint.run_definition),
        run_definition_digest=checkpoint.run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )
    call = ToolCall(id="call-finish-tool", name="write_record", arguments={})
    try:
        assert controller.admit() is None
        controller.plan_tool(
            cycle_index=1,
            call=call,
            idempotency_support=ToolIdempotency.UNKNOWN,
        )
        entry = controller._find_tool_call(cycle_index=1, tool_call_id=call.id)
        assert entry is not None
        if journal_state is not OperationState.PLANNED:
            controller.tool_started(cycle_index=1, call=call)
        if journal_state is OperationState.AMBIGUOUS:
            entry = controller._find_tool_call(cycle_index=1, tool_call_id=call.id)
            assert entry is not None
            entry.state = OperationState.AMBIGUOUS
            controller._progress()
        before = store.load_checkpoint(key)
        assert before is not None
        before_wire = checkpoint_to_dict(before)

        with pytest.raises(CheckpointError) as error:
            controller.finish_tool(
                cycle_index=1,
                call=call,
                result=ToolExecutionResult(
                    tool_call_id="call-wrong-finish-tool",
                    content="rejected",
                    status_code=ToolResultStatus.ERROR,
                    error_code="provider_rejected",
                ),
            )

        assert error.value.code == "tool_receipt_identity_invalid"
        after = store.load_checkpoint(key)
        assert after is not None
        assert checkpoint_to_dict(after) == before_wire
    finally:
        controller.close()
        store.delete_checkpoint(key)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_record_tool_receipt_rejects_result_identity_before_planned_short_circuit(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"mismatched-receipt-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"mismatched-receipt-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_planned"))
    entry.cycle_index = 1
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    before = store.load_checkpoint(checkpoint.checkpoint_key)
    assert before is not None
    before_payload = checkpoint_to_dict(before)

    with pytest.raises(CheckpointError) as error:
        store.record_tool_receipt(
            claimed,
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            tool_call_id=entry.tool_call_id or "call-planned",
            request_digest=entry.request_digest,
            result=ToolExecutionResult(
                tool_call_id="call-does-not-match",
                content="wrong identity",
                status_code=ToolResultStatus.ERROR,
                error_code="tool_not_allowed",
            ),
            claim_token="owner",
            expected_revision=claimed.revision,
            claimed_cycle=1,
        )

    assert error.value.code == "tool_receipt_identity_invalid"
    after = store.load_checkpoint(checkpoint.checkpoint_key)
    assert after is not None
    assert checkpoint_to_dict(after) == before_payload


@pytest.mark.parametrize(
    ("claim_token", "expected_revision", "expected_code"),
    [
        ("", 2, "checkpoint_claim_required"),
        ("stale-owner", 2, "checkpoint_claim_conflict"),
        ("owner", 1, "checkpoint_revision_conflict"),
    ],
)
@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_record_tool_receipt_identity_miss_rejects_typed_without_write(
    store_kind: str,
    claim_token: str,
    expected_revision: int,
    expected_code: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"receipt-identity-miss-{store_kind}-{expected_code}")
    checkpoint = _minimal_checkpoint(key=f"receipt-identity-miss-{store_kind}-{expected_code}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    entry.cycle_index = 1
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    result = ToolExecutionResult(
        tool_call_id=entry.tool_call_id or "call-identity-miss",
        content="written",
        status_code=ToolResultStatus.SUCCESS,
    )
    before = store.load_checkpoint(checkpoint.checkpoint_key)
    assert before is not None
    before_payload = checkpoint_to_dict(before)

    with pytest.raises(CheckpointError) as error:
        store.record_tool_receipt(
            claimed,
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            tool_call_id=entry.tool_call_id or "call-identity-miss",
            request_digest=entry.request_digest,
            result=result,
            claim_token=claim_token,
            expected_revision=expected_revision,
            claimed_cycle=1,
        )

    assert error.value.code == expected_code
    after = store.load_checkpoint(checkpoint.checkpoint_key)
    assert after is not None
    assert checkpoint_to_dict(after) == before_payload


def test_canonical_unknown_journal_recovers_once_through_controller() -> None:
    import time

    fixture = _fixture("operation_journal.json")
    case = next(
        item for item in fixture["recovery_cases"] if item["name"] == "started_tool_surfaces_unknown_outcome_to_model_by_default"
    )
    expected = case["expected"]
    receipt = _journal_case(case["receipt_entry"])
    seed = _minimal_checkpoint(key=fixture["receipt_identity"]["golden_identity"]["checkpoint_key"])
    store = InMemoryCheckpointStore()
    assert store.create_checkpoint(seed)
    claimed = store.claim_checkpoint(
        seed.checkpoint_key, 1, claim_token="expired-owner", lease_expires_at_ms=200, now_ms=100, claim_mode="continue"
    )
    assert claimed is not None
    claimed.tool_journal = [OperationJournalEntry.from_dict(_journal_case(case["entry"]))]
    assert store.progress_checkpoint(claimed, claim_token="expired-owner", expected_revision=claimed.revision)

    original_completion = None
    observed: list[Any] = []
    for recovery in range(2):
        before = store.load_checkpoint(seed.checkpoint_key)
        assert before is not None
        time.sleep(max(0, (before.lease_expires_at_ms or 0) / 1000 - time.time()))
        controller = CheckpointResumeController(
            config=CheckpointConfig(
                store=store,
                key=seed.checkpoint_key,
                resume_policy=ResumePolicy.REQUIRE_EXISTING,
                ambiguous_tool_policy=AmbiguousToolPolicy(case["policy"]),
            ),
            task_id=seed.task_id,
            run_id=seed.root_run_id,
            trace_id=seed.trace_id,
            run_definition=seed.run_definition,
            run_definition_digest=seed.run_definition_digest,
            initial_messages=[],
            initial_shared_state={},
            initial_budget_usage=None,
            extensions=[],
            reconciliation_provider=None,
            event_sink=observed.append,
            lease_duration_ms=1000,
        )
        try:
            assert controller.admit() is None
            controller._ensure_claim(1)
            retained = store.load_checkpoint(seed.checkpoint_key)
            assert retained is not None
            assert retained.resume_attempt == before.resume_attempt + 1
            assert len(retained.tool_journal) == 1
            entry = retained.tool_journal[0]
            assert entry.state.value == expected["persisted_state"]
            assert entry.result == receipt["result"]
            assert entry.result_digest == expected["result_digest"]
            assert entry.resume_observation is not None
            assert entry.resume_observation.to_dict() == expected["resume_observation"]
            completion = [item for item in retained.event_outbox if item.event["type"] == expected["event"]]
            assert len(completion) == 1
            assert completion[0].event_id == expected["event_id"]
            if recovery == 0:
                original_completion = deepcopy(completion[0].event)
            else:
                assert completion[0].event == original_completion
        finally:
            controller.close()
    assert sum(event.type == expected["event"] for event in observed) == 1


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis", "real_redis"])
def test_tool_outcome_unknown_receipt_replay_preserves_digest_and_zero_writes(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"unknown-receipt-replay-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"unknown-receipt-replay-{store_kind}-{uuid4().hex}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_unknown_idempotency"))
    entry.cycle_index = 1
    entry.state = OperationState.AMBIGUOUS
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    observation = ResumeObservation(
        operation_id=entry.operation_id,
        operation_kind=OperationKind.TOOL,
        cycle_index=entry.cycle_index,
        risk="unknown_tool_side_effect",
        idempotency_support=entry.idempotency_support,
    )
    claimed.tool_journal[0].resume_observation = observation
    result = ToolExecutionResult(
        tool_call_id=entry.tool_call_id or "call-2",
        content="The tool outcome is unknown.",
        status_code=ToolResultStatus.ERROR,
        error_code="tool_outcome_unknown",
    )
    digest = canonical_json_sha256(result.to_dict(), "tool result")
    assert store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-2",
        request_digest=entry.request_digest,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert persisted.tool_journal[0].state is OperationState.FAILED
    assert persisted.tool_journal[0].result == result.to_dict()
    assert persisted.tool_journal[0].result_digest == digest
    assert persisted.tool_journal[0].resume_observation == observation
    revision = persisted.revision
    events = tuple(entry.event_id for entry in persisted.event_outbox)

    assert store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-2",
        request_digest=entry.request_digest,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    replayed = store.load_checkpoint(checkpoint.checkpoint_key)
    assert replayed is not None
    assert replayed.revision == revision
    assert tuple(entry.event_id for entry in replayed.event_outbox) == events

    conflict = replace(result, content="A different unknown outcome.")
    with pytest.raises(CheckpointError) as error:
        store.record_tool_receipt(
            claimed,
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            tool_call_id=entry.tool_call_id or "call-2",
            request_digest=entry.request_digest,
            result=conflict,
            claim_token="owner",
            expected_revision=claimed.revision,
            claimed_cycle=1,
        )
    assert error.value.code == "tool_receipt_conflict"
    unchanged = store.load_checkpoint(checkpoint.checkpoint_key)
    assert unchanged is not None
    assert unchanged.revision == revision
    assert tuple(entry.event_id for entry in unchanged.event_outbox) == events
    store.delete_checkpoint(checkpoint.checkpoint_key)


@pytest.mark.parametrize("source_case", ["started", "missing_observation", "wrong_cycle", "wrong_observation"])
@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis", "real_redis"])
def test_tool_outcome_unknown_receipt_rejects_incomplete_observation_without_write(
    source_case: str,
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"unknown-receipt-observation-{source_case}-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"unknown-receipt-observation-{source_case}-{store_kind}-{uuid4().hex}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_unknown_idempotency"))
    entry.cycle_index = 1
    entry.state = OperationState.STARTED if source_case == "started" else OperationState.AMBIGUOUS
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    claimed.tool_journal[0].resume_observation = ResumeObservation(
        operation_id=entry.operation_id,
        operation_kind=OperationKind.TOOL,
        cycle_index=entry.cycle_index,
        risk="unknown_tool_side_effect",
        idempotency_support=entry.idempotency_support,
    )
    if source_case in {"started", "missing_observation"}:
        claimed.tool_journal[0].resume_observation = None
    elif source_case == "wrong_cycle":
        claimed.tool_journal[0].cycle_index += 1
    else:
        assert claimed.tool_journal[0].resume_observation is not None
        claimed.tool_journal[0].resume_observation = replace(
            claimed.tool_journal[0].resume_observation,
            operation_id="op-wrong-observation",
        )

    result = ToolExecutionResult(
        tool_call_id=entry.tool_call_id or "call-2",
        content="The tool outcome is unknown.",
        status_code=ToolResultStatus.ERROR,
        error_code="tool_outcome_unknown",
    )
    before = store.load_checkpoint(checkpoint.checkpoint_key)
    assert before is not None
    before_payload = checkpoint_to_dict(before)

    with pytest.raises(CheckpointError) as error:
        store.record_tool_receipt(
            claimed,
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            tool_call_id=entry.tool_call_id or "call-2",
            request_digest=entry.request_digest,
            result=result,
            claim_token="owner",
            expected_revision=claimed.revision,
            claimed_cycle=1,
        )

    assert error.value.code == "checkpoint_journal_integrity_mismatch"
    after = store.load_checkpoint(checkpoint.checkpoint_key)
    assert after is not None
    assert checkpoint_to_dict(after) == before_payload
    store.delete_checkpoint(checkpoint.checkpoint_key)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_tool_receipt_mutates_the_full_journal_identity(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"receipt-identity-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"receipt-identity-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    first = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    second = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    for candidate, call_id, digest in ((first, "call-first", "a" * 64), (second, "call-second", "b" * 64)):
        candidate.cycle_index = 1
        candidate.operation_id = "same-operation"
        candidate.tool_call_id = call_id
        candidate.request_digest = digest
    claimed.tool_journal = [first, second]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    result = ToolExecutionResult(
        tool_call_id="call-second",
        content="done",
        status_code=ToolResultStatus.SUCCESS,
    )
    assert store.record_tool_receipt(
        claimed,
        operation_id="same-operation",
        attempt=1,
        tool_call_id="call-second",
        request_digest="b" * 64,
        result=result,
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    assert [entry.state for entry in persisted.tool_journal] == [OperationState.STARTED, OperationState.SUCCEEDED]


def test_checkpoint_codec_rejects_tampered_terminal_tool_identity() -> None:
    store = InMemoryCheckpointStore()
    checkpoint = _minimal_checkpoint(key="codec-terminal-tool-identity")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    entry.cycle_index = 1
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    assert store.record_tool_receipt(
        claimed,
        operation_id=entry.operation_id,
        attempt=entry.attempt,
        tool_call_id=entry.tool_call_id or "call-codec",
        request_digest=entry.request_digest,
        result=ToolExecutionResult(
            tool_call_id=entry.tool_call_id or "call-codec",
            content="done",
            status_code=ToolResultStatus.SUCCESS,
        ),
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    persisted = store.load_checkpoint(checkpoint.checkpoint_key)
    assert persisted is not None
    payload = checkpoint_to_dict(persisted)

    identity_tampered = deepcopy(payload)
    identity_tampered["tool_journal"][0]["identity_key"] = "0" * 64
    with pytest.raises(CheckpointError) as identity_error:
        checkpoint_from_dict(identity_tampered)
    assert identity_error.value.code == "tool_receipt_identity_invalid"

    result_tampered = deepcopy(payload)
    result_tampered["tool_journal"][0]["result"]["tool_call_id"] = "call-tampered"
    result_tampered["tool_journal"][0]["result_digest"] = canonical_json_sha256(
        result_tampered["tool_journal"][0]["result"],
        "tool result",
    )
    with pytest.raises(CheckpointError) as result_error:
        checkpoint_from_dict(result_tampered)
    assert result_error.value.code == "tool_receipt_identity_invalid"

    event_tampered = deepcopy(payload)
    receipt_event = next(item for item in event_tampered["event_outbox"] if item["event"]["type"] == "tool_call_completed")
    receipt_event_id = "evt_receipt_" + ("f" * 64)
    receipt_event["event_id"] = receipt_event_id
    receipt_event["event"]["event_id"] = receipt_event_id
    receipt_event["payload_digest"] = compute_event_payload_digest(receipt_event["event"])
    with pytest.raises(CheckpointError) as event_error:
        checkpoint_from_dict(event_tampered)
    assert event_error.value.code == "tool_receipt_identity_invalid"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cross_cycle_tool_call_id_uses_distinct_receipt_event_identity(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"cross-cycle-tool-call-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"cross-cycle-tool-call-{store_kind}")
    assert store.create_checkpoint(checkpoint)

    first_claim = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert first_claim is not None
    first_entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    first_entry.cycle_index = 1
    first_entry.operation_id = "operation-cycle-one"
    first_entry.tool_call_id = "reused-tool-call"
    first_entry.request_digest = "a" * 64
    first_claim.tool_journal = [first_entry]
    assert store.progress_checkpoint(first_claim, claim_token="owner", expected_revision=first_claim.revision)
    first_claim.revision += 1
    assert store.record_tool_receipt(
        first_claim,
        operation_id=first_entry.operation_id,
        attempt=first_entry.attempt,
        tool_call_id=first_entry.tool_call_id or "reused-tool-call",
        request_digest=first_entry.request_digest,
        result=ToolExecutionResult(
            tool_call_id="reused-tool-call",
            content="cycle one",
            status_code=ToolResultStatus.SUCCESS,
        ),
        claim_token="owner",
        expected_revision=first_claim.revision,
        claimed_cycle=1,
    )
    first_done = store.load_checkpoint(checkpoint.checkpoint_key)
    assert first_done is not None
    first_done.cycle_index = 1
    assert store.commit_checkpoint(
        first_done,
        claim_token="owner",
        expected_revision=first_done.revision,
    )

    second_claim = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        2,
        claim_token="owner-two",
        lease_expires_at_ms=400,
        now_ms=300,
        claim_mode="continue",
    )
    assert second_claim is not None
    second_entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    second_entry.cycle_index = 2
    second_entry.operation_id = "operation-cycle-two"
    second_entry.tool_call_id = "reused-tool-call"
    second_entry.request_digest = "b" * 64
    second_claim.tool_journal = [second_entry]
    assert store.progress_checkpoint(second_claim, claim_token="owner-two", expected_revision=second_claim.revision)
    second_claim.revision += 1
    assert store.record_tool_receipt(
        second_claim,
        operation_id=second_entry.operation_id,
        attempt=second_entry.attempt,
        tool_call_id=second_entry.tool_call_id or "reused-tool-call",
        request_digest=second_entry.request_digest,
        result=ToolExecutionResult(
            tool_call_id="reused-tool-call",
            content="cycle two",
            status_code=ToolResultStatus.SUCCESS,
        ),
        claim_token="owner-two",
        expected_revision=second_claim.revision,
        claimed_cycle=2,
    )
    after = store.load_checkpoint(checkpoint.checkpoint_key)
    assert after is not None
    completed = [entry for entry in after.tool_journal if entry.state is OperationState.SUCCEEDED]
    assert len(completed) == 1
    identity = compute_tool_identity_key(
        checkpoint.checkpoint_key,
        second_entry.operation_id,
        second_entry.attempt,
        second_entry.tool_call_id or "reused-tool-call",
        second_entry.request_digest,
    )
    receipt_events = [entry.event_id for entry in after.event_outbox if entry.event.get("type") == "tool_call_completed"]
    assert receipt_events[-1] == f"evt_receipt_{identity}"


def test_tool_call_id_reuse_within_cycle_is_a_typed_journal_conflict() -> None:
    first = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    second = deepcopy(first)
    first.cycle_index = second.cycle_index = 1
    first.operation_id = "operation-first"
    second.operation_id = "operation-second"
    second.request_digest = "b" * 64
    checkpoint = _minimal_checkpoint(key="tool-call-id-conflict")
    checkpoint.tool_journal = [first, second]
    controller = object.__new__(CheckpointResumeController)
    controller.checkpoint = checkpoint

    with pytest.raises(CheckpointError) as error:
        controller._find_tool_call(cycle_index=1, tool_call_id=first.tool_call_id or "call-1")
    assert error.value.code == "checkpoint_journal_integrity_mismatch"


def test_default_model_retry_stops_after_the_first_attempt() -> None:
    entry = OperationJournalEntry.from_dict(_journal_case("model_ambiguous"))
    entry.attempt = 2
    controller = object.__new__(CheckpointResumeController)
    controller.config = CheckpointConfig(
        store=InMemoryCheckpointStore(),
        key="model-retry-cap",
        ambiguous_model_policy=AmbiguousModelPolicy.RETRY_WITH_DUPLICATE_RISK,
    )
    controller.reconciliation_provider = None
    observation = ResumeObservation(
        operation_id=entry.operation_id,
        operation_kind=OperationKind.MODEL,
        cycle_index=entry.cycle_index,
        risk="duplicate_model_request_and_cost",
        idempotency_support=None,
    )

    decision = controller._reconciliation_decision(entry, observation)

    assert decision.kind is ReconciliationDecisionKind.DEFER


def test_memory_terminal_acknowledgement_rejects_active_claim() -> None:
    store = InMemoryCheckpointStore()
    checkpoint = _minimal_checkpoint(key="terminal-active-claim")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    with store._lock:
        current = store._store[checkpoint.checkpoint_key]
        current.status = AgentStatus.COMPLETED
        current.terminal_result = AgentResult(
            status=AgentStatus.COMPLETED,
            messages=[],
            cycles=[],
            final_answer="done",
            completion_reason=CompletionReason.NO_TOOL_FINISH,
            token_usage=summarize_task_token_usage([]),
            checkpoint_key=checkpoint.checkpoint_key,
        )
        revision = current.revision

    assert not store.acknowledge_terminal(checkpoint.checkpoint_key, expected_revision=revision)
    with store._lock:
        retained = store._store[checkpoint.checkpoint_key]
        assert retained.revision == revision
        assert retained.claim_token == "owner"
        assert not retained.terminal_acknowledged


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_suspend_preserves_ambiguity_and_recovery_claims_it(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "suspend")
    checkpoint = _minimal_checkpoint(key=f"suspend-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    ambiguous = OperationJournalEntry.from_dict(_journal_case("model_ambiguous"))
    claimed.model_call_journal = [ambiguous]
    _attach_model_accounting(
        claimed,
        ambiguous,
        status=ModelCallStatus.AMBIGUOUS,
        error_code="model_outcome_unknown",
    )
    claimed.status = AgentStatus.RECONCILIATION_REQUIRED
    assert store.suspend_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    suspended = store.load_checkpoint(checkpoint.checkpoint_key)
    assert suspended is not None
    assert suspended.status is AgentStatus.RECONCILIATION_REQUIRED
    assert suspended.revision == 2
    assert suspended.resume_attempt == 1
    assert suspended.claim_token is None
    assert suspended.model_call_journal[0].state is OperationState.AMBIGUOUS

    recovery = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="reconciler",
        lease_expires_at_ms=400,
        now_ms=300,
        claim_mode="recovery",
    )
    assert recovery is not None
    assert recovery.status is AgentStatus.RUNNING
    assert recovery.resume_attempt == 2
    assert recovery.model_call_journal[0].state is OperationState.AMBIGUOUS


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cycle_commit_finalize_and_acknowledgement_are_separate(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "terminal")
    checkpoint = _minimal_checkpoint(key=f"terminal-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    succeeded = OperationJournalEntry.from_dict(_journal_case("model_succeeded"))
    claimed.model_call_journal = [succeeded]
    _attach_model_accounting(
        claimed,
        succeeded,
        status=ModelCallStatus.COMPLETED,
    )
    claimed.cycle_index = 1
    assert store.commit_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )

    committed = store.load_checkpoint(checkpoint.checkpoint_key)
    assert committed is not None
    assert committed.claim_token is None
    assert committed.model_call_journal == []
    assert committed.revision == 2
    committed.status = AgentStatus.COMPLETED
    committed.terminal_result = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=committed.messages,
        cycles=committed.cycles,
        final_answer="done",
        completion_reason=CompletionReason.NO_TOOL_FINISH,
        token_usage=summarize_task_token_usage(committed.model_calls),
        checkpoint_key=committed.checkpoint_key,
    )
    assert store.finalize_checkpoint(committed, expected_revision=committed.revision)

    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.revision == 3
    assert terminal.terminal_result is not None
    assert not store.finalize_checkpoint(committed, expected_revision=terminal.revision)
    assert store.acknowledge_terminal(
        checkpoint.checkpoint_key,
        expected_revision=terminal.revision,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.revision == 4
    assert retained.terminal_acknowledged
    assert retained.terminal_result is not None
    assert not store.acknowledge_terminal(
        checkpoint.checkpoint_key,
        expected_revision=retained.revision,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cycle_commit_retains_only_pending_outbox_entries(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"commit-pending-outbox-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"commit-pending-outbox-{store_kind}")
    created = EventOutboxEntry.pending(
        "evt-commit-created",
        _checkpoint_created_event(event_id="evt-commit-created", checkpoint=checkpoint),
    )
    checkpoint.event_outbox = [created]
    assert store.create_checkpoint(checkpoint)
    cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id=created.event_id,
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=created.event_id,
        payload_digest=created.payload_digest,
        cursor=cursor,
        expected_revision=0,
        claim_token=None,
    )
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    pending_event = RunStateChangedEvent(
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        state="running",
        cycle_index=1,
        event_id="evt-commit-pending",
    ).to_dict()
    claimed.event_outbox.append(EventOutboxEntry.pending("evt-commit-pending", pending_event))
    claimed.cycle_index = 1
    claimed.status = AgentStatus.RUNNING
    assert store.commit_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    committed = store.load_checkpoint(checkpoint.checkpoint_key)
    assert committed is not None
    assert [entry.event_id for entry in committed.event_outbox] == ["evt-commit-pending"]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_controller_cycle_commit_delivers_outbox_before_releasing_claim(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"controller-commit-delivery-{store_kind}")
    seed = _minimal_checkpoint(key=f"controller-commit-delivery-{store_kind}")
    attempts: list[str] = []
    fail_once = True

    def event_sink(event: Any) -> None:
        nonlocal fail_once
        if event.event_id != "evt-controller-commit-delivery":
            return
        attempts.append(event.event_id)
        if fail_once:
            fail_once = False
            raise RuntimeError("trusted sink unavailable")

    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=seed.checkpoint_key,
            resume_policy=ResumePolicy.NEW,
        ),
        task_id=seed.task_id,
        run_id=seed.root_run_id,
        trace_id=seed.trace_id,
        run_definition=seed.run_definition,
        run_definition_digest=seed.run_definition_digest,
        initial_messages=seed.messages,
        initial_shared_state=seed.shared_state,
        initial_budget_usage=seed.budget_usage,
        extensions=[],
        reconciliation_provider=None,
        event_sink=event_sink,
    )
    messages: list[Message] = []
    cycles: list[CycleRecord] = []
    shared_state: dict[str, Any] = {}
    try:
        assert controller.admit() is None
        controller.bind_runtime_state(messages=messages, cycles=cycles, shared_state=shared_state)
        controller._ensure_claim(1)
        checkpoint = controller._require_checkpoint()
        controller._queue_outbox_event(
            checkpoint,
            RunStateChangedEvent(
                run_id=seed.root_run_id,
                trace_id=seed.trace_id,
                state="running",
                cycle_index=1,
                event_id="evt-controller-commit-delivery",
            ),
        )
        cycles.append(CycleRecord(index=1, assistant_message="finished"))

        with pytest.raises(RuntimeError, match="trusted sink unavailable"):
            controller.commit_cycle(
                cycle_index=1,
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
            )

        retained = store.load_checkpoint(seed.checkpoint_key)
        assert retained is not None
        assert retained.claim_token is not None
        assert retained.claimed_cycle == 1
        assert retained.status is AgentStatus.RUNNING
        assert any(
            entry.event_id == "evt-controller-commit-delivery" and entry.state == "pending" for entry in retained.event_outbox
        )

        controller.commit_cycle(
            cycle_index=1,
            messages=messages,
            cycles=cycles,
            shared_state=shared_state,
        )
        committed = store.load_checkpoint(seed.checkpoint_key)
        assert committed is not None
        assert committed.claim_token is None
        assert committed.claimed_cycle is None
        assert committed.cycle_index == 1
        assert committed.event_outbox == []
        assert attempts == ["evt-controller-commit-delivery", "evt-controller-commit-delivery"]
    finally:
        controller.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cross_cycle_tool_call_id_reuse_succeeds_after_delivery(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, f"cross-cycle-tool-call-delivered-{store_kind}")
    checkpoint = _minimal_checkpoint(key=f"cross-cycle-tool-call-delivered-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    first_claim = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert first_claim is not None
    first_entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    first_entry.cycle_index = 1
    first_entry.operation_id = "operation-cycle-one-delivered"
    first_entry.tool_call_id = "reused-tool-call-delivered"
    first_entry.request_digest = "a" * 64
    first_claim.tool_journal = [first_entry]
    assert store.progress_checkpoint(first_claim, claim_token="owner", expected_revision=first_claim.revision)
    first_claim.revision += 1
    assert store.record_tool_receipt(
        first_claim,
        operation_id=first_entry.operation_id,
        attempt=first_entry.attempt,
        tool_call_id=first_entry.tool_call_id or "reused-tool-call-delivered",
        request_digest=first_entry.request_digest,
        result=ToolExecutionResult(
            tool_call_id="reused-tool-call-delivered",
            content="cycle one",
            status_code=ToolResultStatus.SUCCESS,
        ),
        claim_token="owner",
        expected_revision=first_claim.revision,
        claimed_cycle=1,
    )
    pending = store.load_checkpoint(checkpoint.checkpoint_key)
    assert pending is not None
    completed_event = next(entry for entry in pending.event_outbox if entry.event.get("type") == "tool_call_completed")
    cursor = EventCursor(
        store_ref={"id": "events.cross-cycle", "version": "1"},
        value={"sequence": 1},
        last_event_id=completed_event.event_id,
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=completed_event.event_id,
        payload_digest=completed_event.payload_digest,
        cursor=cursor,
        expected_revision=pending.revision,
        claim_token="owner",
    )
    delivered = store.load_checkpoint(checkpoint.checkpoint_key)
    assert delivered is not None
    delivered.cycle_index = 1
    assert store.commit_checkpoint(
        delivered,
        claim_token="owner",
        expected_revision=delivered.revision,
    )

    second_claim = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        2,
        claim_token="owner-two",
        lease_expires_at_ms=400,
        now_ms=300,
        claim_mode="continue",
    )
    assert second_claim is not None
    second_entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    second_entry.cycle_index = 2
    second_entry.operation_id = "operation-cycle-two-delivered"
    second_entry.tool_call_id = "reused-tool-call-delivered"
    second_entry.request_digest = "b" * 64
    second_claim.tool_journal = [second_entry]
    assert store.progress_checkpoint(second_claim, claim_token="owner-two", expected_revision=second_claim.revision)
    second_claim.revision += 1
    assert store.record_tool_receipt(
        second_claim,
        operation_id=second_entry.operation_id,
        attempt=second_entry.attempt,
        tool_call_id=second_entry.tool_call_id or "reused-tool-call-delivered",
        request_digest=second_entry.request_digest,
        result=ToolExecutionResult(
            tool_call_id="reused-tool-call-delivered",
            content="cycle two",
            status_code=ToolResultStatus.SUCCESS,
        ),
        claim_token="owner-two",
        expected_revision=second_claim.revision,
        claimed_cycle=2,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_cycle_commit_rejects_model_journal_without_atomic_accounting(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "invalid-commit-accounting")
    checkpoint = _minimal_checkpoint(key=f"invalid-commit-accounting-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.model_call_journal = [OperationJournalEntry.from_dict(_journal_case("model_succeeded"))]
    claimed.cycle_index = 1

    with pytest.raises(CheckpointError) as error:
        store.commit_checkpoint(
            claimed,
            claim_token="owner",
            expected_revision=claimed.revision,
        )

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_terminal_rejects_model_journal_without_atomic_accounting(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "invalid-terminal-accounting")
    checkpoint = _minimal_checkpoint(key=f"invalid-terminal-accounting-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.model_call_journal = [OperationJournalEntry.from_dict(_journal_case("model_succeeded"))]
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error={"code": "agent_failed", "message": "failed after model dispatch", "retryable": False},
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
    )

    with pytest.raises(CheckpointError) as error:
        store.finalize_claimed_checkpoint(
            claimed,
            claim_token="owner",
            expected_revision=claimed.revision,
        )

    assert error.value.code == "checkpoint_status_invalid"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_operator_abort_finalize_preserves_ambiguous_evidence(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "operator-abort")
    checkpoint = _minimal_checkpoint(key=f"abort-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    ambiguous = OperationJournalEntry.from_dict(_journal_case("tool_unknown_idempotency"))
    ambiguous.cycle_index = 1
    ambiguous.state = OperationState.AMBIGUOUS
    claimed.tool_journal = [ambiguous]
    claimed.status = AgentStatus.RECONCILIATION_REQUIRED
    assert store.suspend_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    checkpoint = store.load_checkpoint(checkpoint.checkpoint_key)
    assert checkpoint is not None

    checkpoint.status = AgentStatus.FAILED
    checkpoint.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=checkpoint.messages,
        cycles=checkpoint.cycles,
        error={
            "code": "operator_abort_with_unknown_outcome",
            "message": "Operator accepted that the external outcome is unknown.",
            "retryable": False,
        },
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=checkpoint.checkpoint_key,
        resume_observations=[
            ResumeObservation(
                operation_id=checkpoint.tool_journal[0].operation_id,
                operation_kind=OperationKind.TOOL,
                cycle_index=1,
                risk="unknown external tool outcome",
                idempotency_support=ToolIdempotency.UNKNOWN,
            )
        ],
    )
    assert store.finalize_checkpoint(checkpoint, expected_revision=checkpoint.revision)
    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    assert terminal.status is AgentStatus.FAILED
    assert terminal.tool_journal[0].state is OperationState.FAILED
    assert terminal.terminal_result is not None
    assert terminal.terminal_result.resume_observations


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_cycle_aborted_replay_reuses_durable_event_payload(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "cycle-aborted-replay")
    checkpoint = _minimal_checkpoint(key=f"cycle-aborted-replay-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_unknown_idempotency"))
    entry.cycle_index = 1
    entry.state = OperationState.STARTED
    claimed.tool_journal = [entry]
    cycle_event = CycleAbortedEvent(
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        cycle_index=0,
        logical_cycle=1,
        reason="cancelled",
        event_id="evt_cycle_aborted_cancelled",
        created_at=111.0,
    ).to_dict()
    claimed.event_outbox = [EventOutboxEntry.pending(cycle_event["event_id"], cycle_event)]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)

    candidate = store.load_checkpoint(checkpoint.checkpoint_key)
    assert candidate is not None
    candidate.status = AgentStatus.FAILED
    candidate.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=candidate.messages,
        cycles=candidate.cycles,
        error={
            "code": "cancelled_with_unknown_outcome",
            "message": "Cancellation was accepted while the external outcome remained unknown.",
            "retryable": False,
        },
        completion_reason=CompletionReason.CANCELLED,
        checkpoint_key=candidate.checkpoint_key,
    )
    assert store.finalize_claimed_checkpoint(
        candidate,
        claim_token="owner",
        expected_revision=candidate.revision,
    )

    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    retained = next(entry.event for entry in terminal.event_outbox if entry.event_id == cycle_event["event_id"])
    assert retained == cycle_event


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_terminal_finalize_clears_claim_and_ordinary_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claimed-terminal")
    checkpoint = _minimal_checkpoint(key=f"claimed-terminal-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-failure",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    failed_entry = _journal_case("model_failed")
    failed_entry["cycle_index"] = 1
    claimed.model_call_journal = [OperationJournalEntry.from_dict(failed_entry)]
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error={"code": "agent_failed", "message": "definitive model rejection", "retryable": False},
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
    )

    assert store.finalize_claimed_checkpoint(
        claimed,
        claim_token="owner-failure",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint(claimed.checkpoint_key)
    assert terminal is not None
    assert terminal.claim_token is None
    assert terminal.claimed_cycle is None
    assert terminal.lease_expires_at_ms is None
    assert terminal.model_call_journal == []
    assert terminal.terminal_result is not None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_claimed_operator_abort_preserves_ambiguous_journal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "claimed-abort")
    checkpoint = _minimal_checkpoint(key=f"claimed-abort-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-reconcile",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="recovery",
    )
    assert claimed is not None
    ambiguous_entry = _journal_case("tool_unknown_idempotency")
    ambiguous_entry["cycle_index"] = 1
    ambiguous_entry["state"] = "ambiguous"
    claimed.tool_journal = [OperationJournalEntry.from_dict(ambiguous_entry)]
    observation = ResumeObservation(
        operation_id=claimed.tool_journal[0].operation_id,
        operation_kind=OperationKind.TOOL,
        cycle_index=1,
        risk="unknown_tool_side_effect",
        idempotency_support=ToolIdempotency.UNKNOWN,
    )
    claimed.status = AgentStatus.FAILED
    claimed.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=claimed.messages,
        cycles=claimed.cycles,
        error={
            "code": "operator_abort_with_unknown_outcome",
            "message": "Operator accepted that the external outcome is unknown.",
            "retryable": False,
        },
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=claimed.checkpoint_key,
        resume_observations=[observation],
    )

    assert store.finalize_claimed_checkpoint(
        claimed,
        claim_token="owner-reconcile",
        expected_revision=claimed.revision,
    )
    terminal = store.load_checkpoint(claimed.checkpoint_key)
    assert terminal is not None
    assert terminal.claim_token is None
    assert len(terminal.tool_journal) == 1
    assert terminal.tool_journal[0].state is OperationState.FAILED


def test_redis_claimed_finalize_retries_from_an_unmodified_candidate() -> None:
    class RetryPipeline(_FakeRedisPipeline):
        def execute(self) -> list[object]:
            if not self._client.fail_once:
                self._client.fail_once = True
                self._commands.clear()
                self._transaction = False
                raise _FakeWatchError()
            return super().execute()

    class RetryClient(_FakeRedisClient):
        fail_once: bool

        def __init__(self) -> None:
            super().__init__()
            self.fail_once = False

        def pipeline(self) -> RetryPipeline:
            return RetryPipeline(self)

    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = RetryClient()
    checkpoint = _minimal_checkpoint(key="redis-finalize-retry")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_unknown_idempotency"))
    entry.cycle_index = 1
    entry.state = OperationState.AMBIGUOUS
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1

    candidate = store.load_checkpoint(checkpoint.checkpoint_key)
    assert candidate is not None
    candidate.status = AgentStatus.FAILED
    candidate.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=candidate.messages,
        cycles=candidate.cycles,
        error={
            "code": "operator_abort_with_unknown_outcome",
            "message": "Operator accepted that the external outcome is unknown.",
            "retryable": False,
        },
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=candidate.checkpoint_key,
        resume_observations=[
            ResumeObservation(
                operation_id=entry.operation_id,
                operation_kind=OperationKind.TOOL,
                cycle_index=entry.cycle_index,
                risk="operator abort with unknown outcome",
                idempotency_support=entry.idempotency_support,
            )
        ],
    )
    assert candidate.tool_journal[0].state is OperationState.AMBIGUOUS

    assert store.finalize_claimed_checkpoint(
        candidate,
        claim_token="owner",
        expected_revision=candidate.revision,
    )
    assert candidate.tool_journal[0].state is OperationState.AMBIGUOUS
    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None and terminal.terminal_result is not None
    assert terminal.tool_journal[0].state is OperationState.FAILED
    assert terminal.tool_journal[0].error is not None
    assert terminal.tool_journal[0].error.code == "tool_cancelled"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_event_delivery_cas_preserves_claim_and_terminal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "event-delivery")
    checkpoint = _minimal_checkpoint(key=f"event-delivery-{store_kind}")
    first = EventOutboxEntry.pending(
        "evt-created",
        _checkpoint_created_event(event_id="evt-created", checkpoint=checkpoint),
    )
    checkpoint.event_outbox = [first]
    assert store.create_checkpoint(checkpoint)
    first_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 1},
        last_event_id="evt-created",
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=first.event_id,
        payload_digest=first.payload_digest,
        cursor=first_cursor,
        expected_revision=0,
        claim_token=None,
    )

    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner-events",
        lease_expires_at_ms=300,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    second = EventOutboxEntry.pending(
        "evt-resumed",
        CheckpointResumedEvent(
            run_id=claimed.root_run_id,
            trace_id=claimed.trace_id,
            checkpoint_key=claimed.checkpoint_key,
            resume_attempt=claimed.resume_attempt,
            cycle_index=claimed.cycle_index,
            event_id="evt-resumed",
            created_at=124.0,
        ).to_dict(),
    )
    claimed.event_outbox.append(second)
    assert store.progress_checkpoint(
        claimed,
        claim_token="owner-events",
        expected_revision=claimed.revision,
    )
    claimed.revision += 1
    second_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 2},
        last_event_id="evt-resumed",
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=second.event_id,
        payload_digest=second.payload_digest,
        cursor=second_cursor,
        expected_revision=claimed.revision,
        claim_token="owner-events",
    )
    delivered = store.load_checkpoint(checkpoint.checkpoint_key)
    assert delivered is not None
    assert delivered.claim_token == "owner-events"
    assert delivered.event_cursor == second_cursor
    assert delivered.event_outbox[-1].state == "delivered"

    delivered.status = AgentStatus.FAILED
    delivered.terminal_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=delivered.messages,
        cycles=delivered.cycles,
        error={"code": "agent_failed", "message": "terminal", "retryable": False},
        completion_reason=CompletionReason.FAILED,
        checkpoint_key=delivered.checkpoint_key,
    )
    terminal_event = EventOutboxEntry.pending(
        "evt-terminal",
        RunFailedEvent(
            run_id=delivered.root_run_id,
            trace_id=delivered.trace_id,
            error="terminal",
            cycle_index=delivered.cycle_index,
            event_id="evt-terminal",
            created_at=125.0,
        ).to_dict(),
    )
    delivered.event_outbox.append(terminal_event)
    assert store.finalize_claimed_checkpoint(
        delivered,
        claim_token="owner-events",
        expected_revision=delivered.revision,
    )
    terminal = store.load_checkpoint(checkpoint.checkpoint_key)
    assert terminal is not None
    terminal_cursor = EventCursor(
        store_ref={"id": "events.test", "version": "1"},
        value={"sequence": 3},
        last_event_id="evt-terminal",
    )
    assert store.record_event_delivery(
        checkpoint.checkpoint_key,
        event_id=terminal_event.event_id,
        payload_digest=terminal_event.payload_digest,
        cursor=terminal_cursor,
        expected_revision=terminal.revision,
        claim_token=None,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.terminal_result is not None
    assert retained.event_outbox[-1].state == "delivered"


def test_event_outbox_rejects_partial_unknown_and_mismatched_current_events() -> None:
    checkpoint = _minimal_checkpoint(key="strict-outbox-event")
    current = _checkpoint_created_event(event_id="evt-current", checkpoint=checkpoint)

    partial = deepcopy(current)
    partial.pop("run_id")
    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-current", partial)
    assert error.value.code == "checkpoint_event_invalid"

    unknown = deepcopy(current)
    unknown["unknown_field"] = True
    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-current", unknown)
    assert error.value.code == "checkpoint_event_invalid"

    with pytest.raises(CheckpointError) as error:
        EventOutboxEntry.pending("evt-other", current)
    assert error.value.code == "event_identity_conflict"


@pytest.mark.parametrize("event_type", ["operation_ambiguous", "operation_replayed", "reconciliation_required"])
def test_recovery_event_replay_reuses_existing_payload_timestamp(event_type: str) -> None:
    checkpoint = _minimal_checkpoint(key=f"recovery-event-{event_type}")
    event_id = f"evt-{event_type}"
    common: dict[str, Any] = {
        "run_id": checkpoint.root_run_id,
        "trace_id": checkpoint.trace_id,
        "checkpoint_key": checkpoint.checkpoint_key,
        "operation_id": "op-recovery",
        "operation_kind": OperationKind.TOOL,
        "cycle_index": 1,
        "event_id": event_id,
    }
    conflicting_common: dict[str, Any] = {**common, "operation_id": "op-other"}
    observation = ResumeObservation(
        operation_id="op-recovery",
        operation_kind=OperationKind.TOOL,
        cycle_index=1,
        risk="unknown_tool_side_effect",
        idempotency_support=ToolIdempotency.UNKNOWN,
    )
    conflicting_observation = replace(observation, operation_id="op-other")
    if event_type == "operation_ambiguous":
        first_event = OperationAmbiguousEvent(
            **common,
            risk="unknown_tool_side_effect",
            idempotency_support=ToolIdempotency.UNKNOWN,
            created_at=100.0,
        )
        replay_event = OperationAmbiguousEvent(
            **common,
            risk="unknown_tool_side_effect",
            idempotency_support=ToolIdempotency.UNKNOWN,
            created_at=200.0,
        )
        conflicting_event = OperationAmbiguousEvent(
            **conflicting_common,
            risk="unknown_tool_side_effect",
            idempotency_support=ToolIdempotency.UNKNOWN,
            created_at=200.0,
        )
    elif event_type == "operation_replayed":
        first_event = OperationReplayedEvent(
            **common,
            receipt_state=OperationState.FAILED,
            created_at=100.0,
        )
        replay_event = OperationReplayedEvent(
            **common,
            receipt_state=OperationState.FAILED,
            created_at=200.0,
        )
        conflicting_event = OperationReplayedEvent(
            **conflicting_common,
            receipt_state=OperationState.FAILED,
            created_at=200.0,
        )
    else:
        first_event = ReconciliationRequiredEvent(
            **common,
            interruption_reason="resume_requires_reconciliation",
            resume_observation=observation,
            created_at=100.0,
        )
        replay_event = ReconciliationRequiredEvent(
            **common,
            interruption_reason="resume_requires_reconciliation",
            resume_observation=observation,
            created_at=200.0,
        )
        conflicting_event = ReconciliationRequiredEvent(
            **conflicting_common,
            interruption_reason="resume_requires_reconciliation",
            resume_observation=conflicting_observation,
            created_at=200.0,
        )
    checkpoint.event_outbox = [EventOutboxEntry.pending(event_id, first_event.to_dict())]

    CheckpointResumeController._queue_outbox_event(checkpoint, replay_event)

    assert len(checkpoint.event_outbox) == 1
    assert checkpoint.event_outbox[0].event == first_event.to_dict()
    with pytest.raises(CheckpointError) as error:
        CheckpointResumeController._queue_outbox_event(checkpoint, conflicting_event)
    assert error.value.code == "event_identity_conflict"


def test_stable_event_replay_reuses_authoritative_timestamp_for_any_event_type() -> None:
    checkpoint = _minimal_checkpoint(key="stable-event-generic")
    first = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="running",
        cycle_index=1,
        event_id="evt-stable-generic",
        created_at=100.0,
    )
    replay = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="running",
        cycle_index=1,
        event_id="evt-stable-generic",
        created_at=200.0,
    )
    conflict = RunStateChangedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        state="cancel_requested",
        cycle_index=1,
        event_id="evt-stable-generic",
        created_at=200.0,
    )
    checkpoint.event_outbox = [EventOutboxEntry.pending(first.event_id, first.to_dict())]

    CheckpointResumeController._queue_outbox_event(checkpoint, replay)

    assert len(checkpoint.event_outbox) == 1
    assert checkpoint.event_outbox[0].event == first.to_dict()
    with pytest.raises(CheckpointError) as error:
        CheckpointResumeController._queue_outbox_event(checkpoint, conflict)
    assert error.value.code == "event_identity_conflict"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_store_rejects_run_definition_replacement_during_progress(
    store_kind: str,
    tmp_path: Path,
) -> None:
    store = _store(store_kind, tmp_path, "definition-immutable")
    checkpoint = _minimal_checkpoint(key=f"definition-{store_kind}")
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        checkpoint.checkpoint_key,
        1,
        claim_token="owner",
        lease_expires_at_ms=200,
        now_ms=100,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.run_definition["root_input"] = "replacement"
    claimed.run_definition_digest = compute_run_definition_digest(claimed.run_definition)
    assert not store.progress_checkpoint(
        claimed,
        claim_token="owner",
        expected_revision=claimed.revision,
    )
    retained = store.load_checkpoint(checkpoint.checkpoint_key)
    assert retained is not None
    assert retained.run_definition["root_input"] == "Summarize the status."


def test_sqlite_uses_only_the_current_checkpoint_schema(tmp_path: Path) -> None:
    store = SqliteCheckpointStore(tmp_path / "schema.sqlite3")
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(checkpoints)").fetchall()}
    assert {"run_definition_schema", "run_definition"} <= columns
    checkpoint = _minimal_checkpoint()
    assert store.create_checkpoint(checkpoint)
    assert store.load_checkpoint(checkpoint.checkpoint_key) is not None


def test_sqlite_rejects_non_current_checkpoint_table_schema(tmp_path: Path) -> None:
    database = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE checkpoints (task_id TEXT PRIMARY KEY)")

    with pytest.raises(RuntimeError, match="does not match"):
        SqliteCheckpointStore(database)


def test_sqlite_schema_mismatch_closes_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "mismatch-close.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE checkpoints (task_id TEXT PRIMARY KEY)")

    class _TrackingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection
            self.closed = False

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return self._connection.execute(*args, **kwargs)

        def close(self) -> None:
            self.closed = True
            self._connection.close()

    real_connect = sqlite3.connect
    connections: list[_TrackingConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> _TrackingConnection:
        connection = _TrackingConnection(real_connect(*args, **kwargs))
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)

    assert len(connections) == 1
    assert connections[0].closed
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_sqlite_ignores_unrelated_checkpoint_prefixed_tables(tmp_path: Path) -> None:
    database = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE checkpoint_archive (task_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TRIGGER checkpoint_archive_trigger AFTER INSERT ON checkpoint_archive BEGIN SELECT 1; END")

    store = SqliteCheckpointStore(database)
    try:
        assert store._schema_sql("table", "checkpoint_archive") is not None
        assert store._schema_sql("table", "checkpoints") is not None
    finally:
        store.close()


def test_sqlite_opens_vendored_canonical_checkpoint_schema(tmp_path: Path) -> None:
    database = tmp_path / "canonical.sqlite3"
    _seed_checkpoint_sqlite_database(database)

    store = SqliteCheckpointStore(database)
    try:
        assert store._schema_sql("table", "checkpoints") is not None
        assert store._schema_sql("index", "checkpoints_status_idx") is not None
        assert store._schema_sql("table", "deferred_resolution_receipts") is not None
        assert store._schema_sql("index", "deferred_receipts_checkpoint_idx") is not None
    finally:
        store.close()


def test_sqlite_schema_sql_matches_vendored_canonical_fixture(tmp_path: Path) -> None:
    canonical_database = tmp_path / "canonical.sqlite3"
    generated_database = tmp_path / "generated.sqlite3"
    _seed_checkpoint_sqlite_database(canonical_database)
    generated_store = SqliteCheckpointStore(generated_database)
    generated_store.close()

    canonical_objects = {
        name: sql
        for _object_type, name, _table_name, sql in _sqlite_master_state(canonical_database)
        if name
        in {
            "checkpoints",
            "checkpoints_status_idx",
            "deferred_resolution_receipts",
            "deferred_receipts_checkpoint_idx",
        }
    }
    generated_objects = {
        name: sql
        for _object_type, name, _table_name, sql in _sqlite_master_state(generated_database)
        if name in canonical_objects
    }
    assert set(generated_objects) == set(canonical_objects)
    for name, canonical_sql in canonical_objects.items():
        assert canonical_sql is not None
        generated_sql = generated_objects[name]
        assert generated_sql is not None
        assert _normalized_sql(generated_sql) == _normalized_sql(canonical_sql)


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("missing-receipt", "DROP TABLE deferred_resolution_receipts"),
        ("missing-receipt-index", "DROP INDEX deferred_receipts_checkpoint_idx"),
    ],
)
def test_sqlite_rejects_incomplete_related_schema_without_ddl(
    tmp_path: Path,
    name: str,
    mutation: str,
) -> None:
    database = tmp_path / f"{name}.sqlite3"
    _seed_checkpoint_sqlite_database(database)
    _seed_business_probe(database)
    with sqlite3.connect(database) as connection:
        connection.execute(mutation)

    before_master = _sqlite_master_state(database)
    before_pragmas = _sqlite_pragma_state(database)
    before_business = _sqlite_business_state(database)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)
    assert _sqlite_master_state(database) == before_master
    assert _sqlite_pragma_state(database) == before_pragmas
    assert _sqlite_business_state(database) == before_business


def test_sqlite_rejects_malformed_auxiliary_schema_without_ddl(tmp_path: Path) -> None:
    database = tmp_path / "malformed-receipt.sqlite3"
    _seed_checkpoint_sqlite_database(database)
    _seed_business_probe(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE deferred_resolution_receipts")
        connection.execute(
            """
            CREATE TABLE deferred_resolution_receipts (
                handle_key TEXT PRIMARY KEY,
                checkpoint_key TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX deferred_receipts_checkpoint_idx ON deferred_resolution_receipts(checkpoint_key)")

    before_master = _sqlite_master_state(database)
    before_pragmas = _sqlite_pragma_state(database)
    before_business = _sqlite_business_state(database)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)
    assert _sqlite_master_state(database) == before_master
    assert _sqlite_pragma_state(database) == before_pragmas
    assert _sqlite_business_state(database) == before_business


@pytest.mark.parametrize(
    ("name", "seed_canonical", "mutation"),
    [
        (
            "case-variant-table",
            False,
            "CREATE TABLE CheckPoints (status TEXT)",
        ),
        (
            "case-variant-index",
            True,
            "DROP INDEX checkpoints_status_idx; CREATE INDEX CheckPoints_Status_Idx ON checkpoints(status)",
        ),
        (
            "case-variant-receipt-table",
            True,
            "DROP TABLE deferred_resolution_receipts; CREATE VIEW Deferred_Resolution_Receipts AS SELECT 1",
        ),
        (
            "case-variant-receipt-index",
            True,
            "DROP INDEX deferred_receipts_checkpoint_idx; CREATE INDEX "
            "Deferred_Receipts_Checkpoint_Idx ON deferred_resolution_receipts(checkpoint_key)",
        ),
    ],
)
def test_sqlite_rejects_case_variant_or_conflicting_related_objects_without_changes(
    tmp_path: Path,
    name: str,
    seed_canonical: bool,
    mutation: str,
) -> None:
    database = tmp_path / f"{name}.sqlite3"
    if seed_canonical:
        _seed_checkpoint_sqlite_database(database)
    _seed_business_probe(database)
    with sqlite3.connect(database) as connection:
        connection.executescript(mutation)

    before_master = _sqlite_master_state(database)
    before_pragmas = _sqlite_pragma_state(database)
    before_business = _sqlite_business_state(database)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)
    assert _sqlite_master_state(database) == before_master
    assert _sqlite_pragma_state(database) == before_pragmas
    assert _sqlite_business_state(database) == before_business


@pytest.mark.parametrize(
    ("name", "trigger_name", "trigger_table"),
    [
        ("canonical-table-trigger", "CHECKPOINTS", "checkpoints"),
        ("canonical-index-trigger", "CHECKPOINTS_STATUS_IDX", "checkpoints"),
        ("canonical-receipt-table-trigger", "DEFERRED_RESOLUTION_RECEIPTS", "deferred_resolution_receipts"),
        ("canonical-receipt-index-trigger", "DEFERRED_RECEIPTS_CHECKPOINT_IDX", "deferred_resolution_receipts"),
    ],
)
def test_sqlite_rejects_canonical_object_and_same_name_trigger_without_changes(
    tmp_path: Path,
    name: str,
    trigger_name: str,
    trigger_table: str,
) -> None:
    database = tmp_path / f"{name}.sqlite3"
    _seed_checkpoint_sqlite_database(database)
    _seed_business_probe(database)
    with sqlite3.connect(database) as connection:
        connection.execute(f'CREATE TRIGGER "{trigger_name}" AFTER INSERT ON "{trigger_table}" BEGIN SELECT 1; END')

    before_master = _sqlite_master_state(database)
    before_pragmas = _sqlite_pragma_state(database)
    before_business = _sqlite_business_state(database)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)
    assert _sqlite_master_state(database) == before_master
    assert _sqlite_pragma_state(database) == before_pragmas
    assert _sqlite_business_state(database) == before_business


def test_sqlite_schema_collision_closes_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "collision-close.sqlite3"
    _seed_checkpoint_sqlite_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TRIGGER "CHECKPOINTS" AFTER INSERT ON checkpoints BEGIN SELECT 1; END')

    class _TrackingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection
            self.closed = False

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return self._connection.execute(*args, **kwargs)

        def close(self) -> None:
            self.closed = True
            self._connection.close()

    real_connect = sqlite3.connect
    connections: list[_TrackingConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> _TrackingConnection:
        connection = _TrackingConnection(real_connect(*args, **kwargs))
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)

    assert len(connections) == 1
    assert connections[0].closed
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_sqlite_rejects_auxiliary_only_schema_without_ddl(tmp_path: Path) -> None:
    database = tmp_path / "auxiliary-only.sqlite3"
    _seed_checkpoint_sqlite_database(database)
    _seed_business_probe(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP INDEX checkpoints_status_idx")
        connection.execute("DROP TABLE checkpoints")

    before_master = _sqlite_master_state(database)
    before_pragmas = _sqlite_pragma_state(database)
    before_business = _sqlite_business_state(database)
    with pytest.raises(RuntimeError, match="checkpoint_store_schema_mismatch"):
        SqliteCheckpointStore(database)
    assert _sqlite_master_state(database) == before_master
    assert _sqlite_pragma_state(database) == before_pragmas
    assert _sqlite_business_state(database) == before_business


def test_redis_key_vectors_match_contract() -> None:
    fixture = _fixture("checkpoint_store.json")
    for vector in fixture["redis_key_vectors"]:
        data_key = RedisCheckpointStore.data_key(vector["checkpoint_key"])
        assert data_key == vector["data_key"]
        assert RedisCheckpointStore._keys(vector["checkpoint_key"]) == (
            vector["data_key"],
            vector["lease_key"],
        )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_accept_deferred_batch_replays_identity_before_revision_and_claim_fences(
    store_kind: str,
    tmp_path: Path,
) -> None:
    from vv_agent import AcceptDeferredDecision, DeferredToolHandle, ToolCallOutcome
    from vv_agent.types import ToolCall

    key = f"deferred-replay-fences-{store_kind}"
    store = _store(store_kind, tmp_path, key)
    checkpoint = _minimal_checkpoint(key=key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="owner",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    handle = DeferredToolHandle(
        checkpoint_key=key,
        operation_id="op_tool_cycle_1_call_1",
        attempt=1,
        request_digest="a" * 64,
    )
    claimed.tool_journal.append(
        OperationJournalEntry(
            kind=OperationKind.TOOL,
            operation_id=handle.operation_id,
            cycle_index=1,
            attempt=handle.attempt,
            state=OperationState.STARTED,
            request_digest=handle.request_digest,
            tool_call_id="call-defer",
            tool_name="defer",
            arguments={},
            idempotency_key="idem-defer",
            idempotency_support=ToolIdempotency.SUPPORTED,
        )
    )
    assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
    claimed.revision += 1
    assert store.admit_deferred_batch(
        claimed,
        outcomes=[
            (
                ToolCall(id="call-defer", name="defer", arguments={}),
                ToolCallOutcome.Deferred(handle),
            )
        ],
        claim_token="owner",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    admitted = store.load_checkpoint(key)
    assert admitted is not None
    before = checkpoint_to_dict(admitted)

    stale = deepcopy(admitted)
    stale.revision -= 1
    stale.run_definition_digest = "f" * 64
    assert store.accept_deferred_batch(
        stale,
        decisions=[AcceptDeferredDecision(handle=handle)],
        claim_token="expired-owner",
        expected_revision=stale.revision,
        claimed_cycle=1,
    )
    retained = store.load_checkpoint(key)
    assert retained is not None
    assert checkpoint_to_dict(retained) == before

    wrong_handle = DeferredToolHandle(
        checkpoint_key=key,
        operation_id=handle.operation_id,
        attempt=handle.attempt,
        request_digest="b" * 64,
    )
    assert not store.accept_deferred_batch(
        stale,
        decisions=[AcceptDeferredDecision(handle=wrong_handle)],
        claim_token="expired-owner",
        expected_revision=stale.revision,
        claimed_cycle=1,
    )
    retained_after_rejection = store.load_checkpoint(key)
    assert retained_after_rejection is not None
    assert checkpoint_to_dict(retained_after_rejection) == before


def test_redis_cleanup_retries_when_resolution_adds_receipt_after_smembers() -> None:
    from vv_agent.checkpoint import canonical_json_sha256
    from vv_agent.deferred import DeferredResolutionReceipt, DeferredToolHandle
    from vv_agent.runtime.state import compute_tool_identity_key
    from vv_agent.runtime.stores.redis import _receipt_to_storage
    from vv_agent.types import ToolExecutionResult, ToolResultStatus

    class RacingPipeline(_FakeRedisPipeline):
        def execute(self) -> list[object]:
            if not self._client.injected:
                self._client.injected = True
                self._client._values[self._client.second_receipt_key] = self._client.second_receipt_payload
                self._client.sadd(self._client.receipt_set_key, self._client.second_receipt_key)
                raise _FakeWatchError()
            return super().execute()

    class RacingClient(_FakeRedisClient):
        def __init__(self) -> None:
            super().__init__()
            self.injected = False
            self.second_receipt_key = ""
            self.second_receipt_payload = ""
            self.receipt_set_key = ""

        def pipeline(self) -> RacingPipeline:
            return RacingPipeline(self)

    def receipt(operation_id: str) -> tuple[str, str, DeferredToolHandle]:
        handle = DeferredToolHandle(
            checkpoint_key="redis-cleanup-race",
            operation_id=operation_id,
            attempt=1,
            request_digest=(operation_id.encode("ascii").hex() + "0" * 64)[:64],
        )
        result = ToolExecutionResult(
            tool_call_id=operation_id,
            content="accepted",
            status_code=ToolResultStatus.SUCCESS,
        )
        value = DeferredResolutionReceipt(
            handle=handle,
            result=result,
            result_digest=canonical_json_sha256(result.to_dict(), "deferred result"),
            event_id=(
                "evt_receipt_"
                + compute_tool_identity_key(
                    handle.checkpoint_key,
                    handle.operation_id,
                    handle.attempt,
                    result.tool_call_id,
                    handle.request_digest,
                )
            ),
            event_payload_digest="a" * 64,
            receipt_status="succeeded",
        )
        key = RedisCheckpointStore._receipt_key(handle.key)
        return key, _receipt_to_storage(value), handle

    client = RacingClient()
    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = client
    first_key, first_payload, first_handle = receipt("op_first")
    second_key, second_payload, _second_handle = receipt("op_second")
    receipt_set_key = RedisCheckpointStore._receipt_set_key("redis-cleanup-race")
    client._values[first_key] = first_payload
    client.sadd(receipt_set_key, first_key)
    client.second_receipt_key = second_key
    client.second_receipt_payload = second_payload
    client.receipt_set_key = receipt_set_key

    store.delete_checkpoint("redis-cleanup-race")

    assert first_key not in client._values
    assert second_key not in client._values
    assert receipt_set_key not in client._sets
    assert first_handle.checkpoint_key == "redis-cleanup-race"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_deferred_failed_tombstone_replays_complete_result_without_writes(
    store_kind: str,
    tmp_path: Path,
) -> None:
    from vv_agent import DeferredToolHandle, ToolCallOutcome

    key = f"deferred-failed-tombstone-{store_kind}"
    handle = DeferredToolHandle(
        checkpoint_key=key,
        operation_id="op_tool_cycle_1_call_failed_deferred",
        attempt=1,
        request_digest="a" * 64,
    )
    result = ToolExecutionResult(
        tool_call_id="call-failed-deferred",
        content="provider rejected",
        status_code=ToolResultStatus.ERROR,
        directive=ToolDirective.WAIT_USER,
        error_code="provider_rejected",
        metadata={"provider": "gateway", "retryable": False},
    )
    store = _store(store_kind, tmp_path, key)
    try:
        checkpoint = _minimal_checkpoint(key=key)
        assert store.create_checkpoint(checkpoint)
        claimed = store.claim_checkpoint(
            key,
            1,
            claim_token="owner",
            lease_expires_at_ms=200,
            now_ms=100,
            claim_mode="continue",
        )
        assert claimed is not None
        entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
        entry.operation_id = handle.operation_id
        entry.request_digest = handle.request_digest
        entry.tool_call_id = result.tool_call_id
        entry.tool_name = "remote_write"
        claimed.tool_journal = [entry]
        assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
        claimed.revision += 1
        assert store.admit_deferred_batch(
            claimed,
            outcomes=[
                (
                    ToolCall(id=result.tool_call_id, name="remote_write", arguments={}),
                    ToolCallOutcome.Deferred(handle),
                )
            ],
            claim_token="owner",
            expected_revision=claimed.revision,
            claimed_cycle=1,
        )

        decision = store.resolve_deferred(handle, result)

        assert decision.kind == "applied_ready"
        assert decision.receipt is not None
        assert decision.receipt.result.to_dict() == result.to_dict()
        assert decision.receipt.result_digest == canonical_json_sha256(result.to_dict(), "deferred result")
        persisted = store.load_checkpoint(key)
        assert persisted is not None
        failed = persisted.tool_journal[0]
        assert failed.state is OperationState.FAILED
        assert failed.result == result.to_dict()
        assert failed.result_digest == decision.receipt.result_digest
        assert failed.error == OperationError(
            code="provider_rejected",
            message="provider rejected",
            retryable=False,
        )
        revision = persisted.revision
        event_ids = tuple(item.event_id for item in persisted.event_outbox)

        replay = store.resolve_deferred(handle, result)

        assert replay.kind == "replayed"
        assert replay.receipt is not None
        assert replay.receipt.to_dict() == decision.receipt.to_dict()
        replayed = store.load_checkpoint(key)
        assert replayed is not None
        assert replayed.revision == revision
        assert tuple(item.event_id for item in replayed.event_outbox) == event_ids
    finally:
        store.delete_checkpoint(key)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_deferred_receipt_index_tamper_is_rejected_without_checkpoint_write(
    store_kind: str,
    tmp_path: Path,
) -> None:
    from vv_agent.deferred import DeferredResolutionReceipt, DeferredToolHandle, ToolCallOutcome
    from vv_agent.runtime.stores.redis import _strict_json_loads
    from vv_agent.types import ToolCall

    key = f"deferred-receipt-tamper-{store_kind}"
    store = _store(store_kind, tmp_path, key)
    handle = DeferredToolHandle(
        checkpoint_key=key,
        operation_id="op_tool_cycle_1_call_defer",
        attempt=1,
        request_digest="a" * 64,
    )
    result = ToolExecutionResult(
        tool_call_id="call-defer",
        content="accepted",
        status_code=ToolResultStatus.SUCCESS,
    )
    tampered_receipt_payload: str | None = None
    wrong_checkpoint_key: str | None = None
    try:
        checkpoint = _minimal_checkpoint(key=key)
        assert store.create_checkpoint(checkpoint)
        claimed = store.claim_checkpoint(
            key,
            1,
            claim_token="owner",
            lease_expires_at_ms=200,
            now_ms=100,
            claim_mode="continue",
        )
        assert claimed is not None
        entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
        entry.cycle_index = 1
        entry.operation_id = handle.operation_id
        entry.request_digest = handle.request_digest
        entry.tool_call_id = result.tool_call_id
        claimed.tool_journal = [entry]
        assert store.progress_checkpoint(claimed, claim_token="owner", expected_revision=claimed.revision)
        claimed.revision += 1
        assert store.admit_deferred_batch(
            claimed,
            outcomes=[
                (
                    ToolCall(id=result.tool_call_id, name=entry.tool_name or "defer", arguments={}),
                    ToolCallOutcome.Deferred(handle),
                )
            ],
            claim_token="owner",
            expected_revision=claimed.revision,
            claimed_cycle=1,
        )
        decision = store.resolve_deferred(handle, result)
        assert decision.kind == "applied_ready"
        before = store.load_checkpoint(key)
        assert before is not None
        before_wire = checkpoint_to_dict(before)

        if store_kind == "memory":
            with store._lock:  # type: ignore[attr-defined]
                receipt = store._deferred_receipts[handle.key]  # type: ignore[attr-defined]
                wrong_handle = DeferredToolHandle(
                    checkpoint_key=key,
                    operation_id=handle.operation_id,
                    attempt=handle.attempt,
                    request_digest="b" * 64,
                )
                wrong_identity = compute_tool_identity_key(
                    wrong_handle.checkpoint_key,
                    wrong_handle.operation_id,
                    wrong_handle.attempt,
                    result.tool_call_id,
                    wrong_handle.request_digest,
                )
                store._deferred_receipts[handle.key] = DeferredResolutionReceipt(  # type: ignore[attr-defined]
                    handle=wrong_handle,
                    result=result,
                    result_digest=receipt.result_digest,
                    event_id=f"evt_receipt_{wrong_identity}",
                    event_payload_digest=receipt.event_payload_digest,
                    receipt_status="succeeded",
                )
        elif store_kind == "sqlite":
            wrong_checkpoint_key = f"{key}-wrong"
            assert store.create_checkpoint(_minimal_checkpoint(key=wrong_checkpoint_key))
            with store._lock, store._conn:  # type: ignore[attr-defined]
                store._conn.execute(  # type: ignore[attr-defined]
                    "UPDATE deferred_resolution_receipts SET checkpoint_key = ? WHERE handle_key = ?",
                    (wrong_checkpoint_key, handle.key),
                )
        else:
            receipt_key = store._receipt_key(handle.key)  # type: ignore[attr-defined]
            tampered_receipt_payload = store._client.get(receipt_key)  # type: ignore[attr-defined]
            assert tampered_receipt_payload is not None
            payload = _strict_json_loads(tampered_receipt_payload)
            payload["handle"]["checkpoint_key"] = "wrong-checkpoint"
            store._client.set(  # type: ignore[attr-defined]
                receipt_key,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            )

        with pytest.raises(ValueError, match="deferred_receipt_identity_invalid"):
            store.resolve_deferred(handle, result)
        after = store.load_checkpoint(key)
        assert after is not None
        assert checkpoint_to_dict(after) == before_wire
    finally:
        if store_kind == "memory" and handle.key in store._deferred_receipts:  # type: ignore[attr-defined]
            store._deferred_receipts.pop(handle.key, None)  # type: ignore[attr-defined]
        elif store_kind == "sqlite":
            with store._lock, store._conn:  # type: ignore[attr-defined]
                store._conn.execute(
                    "DELETE FROM deferred_resolution_receipts WHERE handle_key = ?",
                    (handle.key,),
                )
        elif tampered_receipt_payload is not None:
            store._client.set(store._receipt_key(handle.key), tampered_receipt_payload)  # type: ignore[attr-defined]
        if wrong_checkpoint_key is not None:
            store.delete_checkpoint(wrong_checkpoint_key)
        store.delete_checkpoint(key)


def test_redis_resolution_retries_to_receipt_replay_after_concurrent_winner() -> None:
    from vv_agent.deferred import DeferredToolHandle, ToolCallOutcome
    from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
    from vv_agent.types import ToolCall, ToolExecutionResult, ToolResultStatus

    handle = DeferredToolHandle(
        checkpoint_key="redis-resolve-race",
        operation_id="op_tool_cycle_1_call_resolve",
        attempt=1,
        request_digest="c" * 64,
    )
    memory = InMemoryCheckpointStore()
    checkpoint = _minimal_checkpoint(key=handle.checkpoint_key)
    assert memory.create_checkpoint(checkpoint)
    claimed = memory.claim_checkpoint(
        handle.checkpoint_key,
        1,
        claim_token="claim-resolve",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.tool_journal.append(
        OperationJournalEntry(
            kind=OperationKind.TOOL,
            operation_id=handle.operation_id,
            cycle_index=1,
            attempt=1,
            state=OperationState.STARTED,
            request_digest=handle.request_digest,
            tool_call_id="call-resolve",
            tool_name="remote_write",
            arguments={},
            idempotency_key="idem-resolve",
            idempotency_support=ToolIdempotency.SUPPORTED,
        )
    )
    assert memory.progress_checkpoint(claimed, claim_token="claim-resolve", expected_revision=claimed.revision)
    claimed.revision += 1
    assert memory.admit_deferred_batch(
        claimed,
        outcomes=[
            (
                ToolCall(id="call-resolve", name="remote_write", arguments={}),
                ToolCallOutcome.Deferred(handle),
            )
        ],
        claim_token="claim-resolve",
        expected_revision=claimed.revision,
        claimed_cycle=1,
    )
    admitted = memory.load_checkpoint(handle.checkpoint_key)
    assert admitted is not None

    class RacingPipeline(_FakeRedisPipeline):
        def execute(self) -> list[object]:
            result = super().execute()
            if not self._client.winner_applied:
                self._client.winner_applied = True
                raise _FakeWatchError()
            return result

    class RacingClient(_FakeRedisClient):
        def __init__(self) -> None:
            super().__init__()
            self.winner_applied = False

        def pipeline(self) -> RacingPipeline:
            return RacingPipeline(self)

    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = _FakeWatchError
    store._client = RacingClient()
    from vv_agent.runtime.stores.redis import _checkpoint_to_storage

    raw, lease = _checkpoint_to_storage(admitted)
    data_key, lease_key = store._keys(handle.checkpoint_key)
    store._client.set(data_key, raw)
    if lease is not None:
        store._client.set(lease_key, str(lease))
    result = ToolExecutionResult(
        tool_call_id="call-resolve",
        content="accepted",
        status_code=ToolResultStatus.SUCCESS,
    )

    decision = store.resolve_deferred(handle, result)

    assert decision.kind == "replayed"
    assert decision.receipt is not None
    assert store._client.winner_applied is True
    assert store._client.smembers(store._receipt_set_key(handle.checkpoint_key)) == {store._receipt_key(handle.key)}


def test_cross_runtime_sqlite_probe_from_environment() -> None:
    import time

    database = os.environ.get("VV_AGENT_CROSS_RUNTIME_DB")
    if database is None:
        pytest.skip("requires a cross-runtime SQLite database")
    mode = os.environ.get("VV_AGENT_CROSS_RUNTIME_MODE", "read_rust")
    store = SqliteCheckpointStore(database)
    if mode == "write_python":
        checkpoint = _minimal_checkpoint(key="python-wrote")
        checkpoint.messages = [Message(role="user", content="from Python")]
        checkpoint.shared_state = {"writer": "python", "format": "checkpoint"}
    elif mode == "read_rust":
        checkpoint = store.load_checkpoint("rust-wrote")
        assert checkpoint is not None
        assert checkpoint.messages == [Message(role="user", content="from Rust")]
        assert checkpoint.shared_state == {"format": "checkpoint", "writer": "rust"}
        assert checkpoint.run_definition_digest == compute_run_definition_digest(checkpoint.run_definition)
        entry = checkpoint.tool_journal[0]
        assert entry.idempotency_support is ToolIdempotency.UNSUPPORTED
        assert entry.idempotency_key is None
        entry.verify_request(
            {
                "schema_version": "vv-agent.operation-request.v1",
                "kind": "tool",
                "request": {"tool_call_id": "cross-tool", "tool_name": "unsafe_write", "arguments": {}, "idempotency_key": None},
            }
        )
        expiry = checkpoint.lease_expires_at_ms or 0
        time.sleep(max(0, expiry / 1000 - time.time()))
    else:
        raise AssertionError(f"unknown cross-runtime mode: {mode}")

    controller = CheckpointResumeController(
        config=CheckpointConfig(
            store=store,
            key=checkpoint.checkpoint_key,
            resume_policy=ResumePolicy.NEW if mode == "write_python" else ResumePolicy.REQUIRE_EXISTING,
        ),
        task_id=checkpoint.task_id,
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        run_definition=checkpoint.run_definition,
        run_definition_digest=checkpoint.run_definition_digest,
        initial_messages=checkpoint.messages,
        initial_shared_state=checkpoint.shared_state,
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
        lease_duration_ms=1000,
    )
    try:
        assert controller.admit() is None
        plan = controller.plan_tool(
            cycle_index=1,
            call=ToolCall(id="cross-tool", name="unsafe_write", arguments={}),
            idempotency_support=ToolIdempotency.UNSUPPORTED,
        )
        assert plan.idempotency_key is None
        assert plan.replay_result is None
        retained = store.load_checkpoint(checkpoint.checkpoint_key)
        assert retained is not None
        assert len(retained.tool_journal) == 1
        assert retained.tool_journal[0].idempotency_key is None
        if mode == "read_rust":
            assert plan.operation_id == entry.operation_id
            assert plan.attempt == entry.attempt
            assert plan.request_digest == entry.request_digest
            assert retained.resume_attempt == checkpoint.resume_attempt + 1
    finally:
        controller.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "redis"])
def test_incomplete_deferred_batch_reconciles_unclassified_wait_user_across_stores(
    store_kind: str,
    tmp_path: Path,
) -> None:
    from vv_agent.deferred import DeferredToolHandle, ToolCallOutcome

    key = f"incomplete-deferred-wait-user-{store_kind}"
    store = _store(store_kind, tmp_path, key)
    seed = _minimal_checkpoint(key=key)
    controller = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.NEW),
        task_id=seed.task_id,
        run_id=seed.root_run_id,
        trace_id=seed.trace_id,
        run_definition=deepcopy(seed.run_definition),
        run_definition_digest=seed.run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
    )
    defer_call = ToolCall(id="call-defer", name="defer", arguments={})
    wait_call = ToolCall(id="call-wait", name="wait_user", arguments={})
    try:
        assert controller.admit() is None
        controller._ensure_claim(1)
        deferred_plan = controller.plan_tool(
            cycle_index=1,
            call=defer_call,
            idempotency_support=ToolIdempotency.SUPPORTED,
        )
        controller.tool_started(cycle_index=1, call=defer_call)
        controller.plan_tool(
            cycle_index=1,
            call=wait_call,
            idempotency_support=ToolIdempotency.UNKNOWN,
        )
        controller.tool_started(cycle_index=1, call=wait_call)
        checkpoint = controller._require_checkpoint()
        handle = DeferredToolHandle(
            checkpoint_key=key,
            operation_id=deferred_plan.operation_id,
            attempt=deferred_plan.attempt,
            request_digest=deferred_plan.request_digest,
        )
        outcomes = [(defer_call, ToolCallOutcome.Deferred(handle))]

        with pytest.raises(CheckpointError) as admission_error:
            store.admit_deferred_batch(
                checkpoint,
                outcomes=outcomes,
                claim_token=checkpoint.claim_token or "",
                expected_revision=checkpoint.revision,
                claimed_cycle=checkpoint.claimed_cycle or 1,
            )
        assert admission_error.value.code == "deferred_batch_incomplete"

        with pytest.raises(CheckpointReconciliationRequired):
            controller._suspend_incomplete_deferred_batch(outcomes)

        retained = store.load_checkpoint(key)
        assert retained is not None
        assert retained.status is AgentStatus.RECONCILIATION_REQUIRED
        assert retained.claim_token is None
        assert [entry.state for entry in retained.tool_journal] == [OperationState.STARTED, OperationState.AMBIGUOUS]
        assert retained.tool_journal[0].deferred_handle is None
        decision = store.resolve_deferred(
            handle,
            ToolExecutionResult(
                tool_call_id="call-defer",
                content="accepted",
                status_code=ToolResultStatus.SUCCESS,
            ),
        )
        assert decision.kind == "not_admitted"

        recovered = store.claim_checkpoint(
            key,
            1,
            claim_token="recovery-owner",
            lease_expires_at_ms=10_000,
            now_ms=1,
            claim_mode="recovery",
        )
        assert recovered is not None
        assert recovered.claim_token == "recovery-owner"
    finally:
        controller.close()
        store.delete_checkpoint(key)
