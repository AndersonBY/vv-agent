from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from test_checkpoint import _minimal_checkpoint, _redis_store
from test_distributed_checkpoint import _strict_envelope

import vv_agent.runtime as runtime_api
from vv_agent.checkpoint import CheckpointError
from vv_agent.runtime.backends.celery import CeleryBackend
from vv_agent.runtime.backends.distributed import DistributedRunEnvelope, DistributedRunHandle
from vv_agent.runtime.dispatch_outbox import DispatchOutboxRecord, DispatchOutboxStore, _validate_envelope, claim_dispatch
from vv_agent.runtime.state import CheckpointStore
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore


def test_dispatch_receipts_are_not_part_of_the_checkpoint_store_protocol() -> None:
    dispatch_methods = {
        "claim_distributed_dispatch",
        "complete_distributed_dispatch",
        "reconcile_distributed_dispatch",
        "get_distributed_dispatch",
        "reap_distributed_dispatches",
    }
    assert dispatch_methods.isdisjoint(vars(CheckpointStore))
    assert dispatch_methods.issubset(vars(DispatchOutboxStore))
    assert isinstance(InMemoryCheckpointStore(), DispatchOutboxStore)
    assert "DispatchOutboxStore" not in runtime_api.__all__


def _stores(tmp_path: Path) -> list[Any]:
    return [
        InMemoryCheckpointStore(),
        SqliteCheckpointStore(tmp_path / "dispatch-outbox.sqlite3"),
        _redis_store(),
    ]


def test_dispatch_record_decodes_each_input_once_and_rejects_digest_drift() -> None:
    payload = _strict_envelope().to_dict()
    with patch("vv_agent.runtime.dispatch_outbox._validate_envelope", wraps=_validate_envelope) as decode:
        pending = DispatchOutboxRecord.pending(payload)
        assert decode.call_count == 1
        claimed = claim_dispatch(pending, claim_token="owner", lease_expires_at_ms=200, now_ms=100)
        assert decode.call_count == 2
    assert claimed.record.envelope_digest == pending.envelope_digest
    wire = claimed.record.to_dict()
    wire["envelope"]["task"]["user_prompt"] = "changed request"
    with pytest.raises(CheckpointError) as error:
        DispatchOutboxRecord.from_dict(wire)
    assert error.value.code == "dispatch_outbox_conflict"
    payload["unknown"] = True
    with pytest.raises(CheckpointError) as error:
        DispatchOutboxRecord.pending(payload)
    assert error.value.code == "dispatch_outbox_conflict"


@pytest.mark.parametrize("store_index", [0, 1, 2])
def test_dispatch_receipt_replay_has_one_claim_and_rejects_identity_drift(
    tmp_path: Path,
    store_index: int,
) -> None:
    store = _stores(tmp_path)[store_index]
    envelope = _strict_envelope()
    checkpoint = _minimal_checkpoint(key=envelope.checkpoint_config.key)
    assert store.create_checkpoint(checkpoint)
    payload = envelope.to_dict()

    first = store.claim_distributed_dispatch(
        payload,
        claim_token="dispatch-owner-a",
        lease_expires_at_ms=10_000,
        now_ms=1_000,
    )
    assert first.should_enqueue
    assert first.record.attempt == 1

    replay = store.claim_distributed_dispatch(
        payload,
        claim_token="dispatch-owner-b",
        lease_expires_at_ms=10_000,
        now_ms=1_001,
    )
    assert not replay.should_enqueue
    assert replay.record.attempt == 1
    assert replay.record.claim_token == "dispatch-owner-a"

    delivered = store.complete_distributed_dispatch(
        dispatch_id=payload["job_id"],
        envelope_digest=first.record.envelope_digest,
        claim_token="dispatch-owner-a",
        attempt=1,
        outcome="delivered",
        now_ms=1_002,
    )
    assert isinstance(delivered, DispatchOutboxRecord)
    assert delivered.state == "delivered"
    assert not store.claim_distributed_dispatch(
        payload,
        claim_token="dispatch-owner-c",
        lease_expires_at_ms=10_000,
        now_ms=1_003,
    ).should_enqueue

    volatile_replay = copy.deepcopy(payload)
    volatile_replay["deadline_unix_ms"] += 1_000
    assert not store.claim_distributed_dispatch(
        volatile_replay,
        claim_token="dispatch-owner-d",
        lease_expires_at_ms=10_000,
        now_ms=1_004,
    ).should_enqueue

    immutable_drift = copy.deepcopy(payload)
    immutable_drift["task"]["user_prompt"] = "different immutable task input"
    with pytest.raises(CheckpointError) as exc_info:
        store.claim_distributed_dispatch(
            immutable_drift,
            claim_token="dispatch-owner-e",
            lease_expires_at_ms=10_000,
            now_ms=1_005,
        )
    assert exc_info.value.code == "dispatch_outbox_conflict"
    assert store.get_distributed_dispatch(payload["job_id"]).state == "delivered"


@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_dispatch_payload_key_binding_fails_closed(store_kind: str) -> None:
    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()

    first_key = f"redis-dispatch-first-{uuid4().hex}"
    second_key = f"redis-dispatch-second-{uuid4().hex}"
    first_payload = _strict_envelope().to_dict()
    second_payload = _strict_envelope().to_dict()
    first_payload["checkpoint_config"]["key"] = first_key
    first_payload["job_id"] = f"{first_payload['run_id']}:cycle:{first_payload['cycle_index']}:first"
    second_payload["checkpoint_config"]["key"] = second_key
    second_payload["job_id"] = f"{second_payload['run_id']}:cycle:{second_payload['cycle_index']}:second"
    first_checkpoint = _minimal_checkpoint(key=first_key)
    second_checkpoint = _minimal_checkpoint(key=second_key)
    first_key_store = store._dispatch_outbox_key(first_payload["job_id"])  # type: ignore[attr-defined]
    second_key_store = store._dispatch_outbox_key(second_payload["job_id"])  # type: ignore[attr-defined]
    first_raw: str | None = None
    second_raw: str | None = None
    try:
        assert store.create_checkpoint(first_checkpoint)
        assert store.create_checkpoint(second_checkpoint)
        first_claim = store.claim_distributed_dispatch(
            first_payload,
            claim_token="dispatch-binding-first",
            lease_expires_at_ms=10_000,
            now_ms=1,
        )
        store.claim_distributed_dispatch(
            second_payload,
            claim_token="dispatch-binding-second",
            lease_expires_at_ms=10_000,
            now_ms=1,
        )
        first_raw = store._client.get(first_key_store)  # type: ignore[attr-defined]
        second_raw = store._client.get(second_key_store)  # type: ignore[attr-defined]
        assert first_raw is not None and second_raw is not None
        store._client.set(first_key_store, second_raw)  # type: ignore[attr-defined]
        store._client.set(second_key_store, first_raw)  # type: ignore[attr-defined]
        before = (store._client.get(first_key_store), store._client.get(second_key_store))  # type: ignore[attr-defined]
        with pytest.raises(CheckpointError) as get_error:
            store.get_distributed_dispatch(first_payload["job_id"])
        assert get_error.value.code == "dispatch_outbox_conflict"
        with pytest.raises(CheckpointError) as complete_error:
            store.complete_distributed_dispatch(
                dispatch_id=first_payload["job_id"],
                envelope_digest=first_claim.record.envelope_digest,
                claim_token="dispatch-binding-first",
                attempt=1,
                outcome="delivered",
                now_ms=2,
            )
        assert complete_error.value.code == "dispatch_outbox_conflict"
        assert (store._client.get(first_key_store), store._client.get(second_key_store)) == before  # type: ignore[attr-defined]
    finally:
        if first_raw is not None and second_raw is not None:
            store._client.set(first_key_store, first_raw)  # type: ignore[attr-defined]
            store._client.set(second_key_store, second_raw)  # type: ignore[attr-defined]
        store.delete_checkpoint(first_key)
        store.delete_checkpoint(second_key)


@pytest.mark.parametrize("store_index", [0, 1, 2])
def test_dispatch_reaper_and_reconcile_retry_keep_stable_task_id(tmp_path: Path, store_index: int) -> None:
    store = _stores(tmp_path)[store_index]
    envelope = _strict_envelope()
    payload = envelope.to_dict()
    assert store.create_checkpoint(_minimal_checkpoint(key=envelope.checkpoint_config.key))
    first = store.claim_distributed_dispatch(
        payload,
        claim_token="crashed-owner",
        lease_expires_at_ms=100,
        now_ms=1,
    )
    expired = store.reap_distributed_dispatches(checkpoint_key=envelope.checkpoint_config.key, now_ms=100)
    assert [row.dispatch_id for row in expired] == [payload["job_id"]]
    assert expired[0].state == "ambiguous"

    pending = store.reconcile_distributed_dispatch(
        dispatch_id=payload["job_id"],
        envelope_digest=first.record.envelope_digest,
        outcome="retry",
        now_ms=101,
        error="broker outcome unknown",
    )
    assert pending is not None
    assert pending.state == "pending"
    retry = store.claim_distributed_dispatch(
        payload,
        claim_token="recovery-owner",
        lease_expires_at_ms=200,
        now_ms=102,
    )
    assert retry.should_enqueue
    assert retry.record.dispatch_id == payload["job_id"]
    assert retry.record.attempt == 2


def test_memory_dispatch_claim_race_has_one_enqueue() -> None:
    store = InMemoryCheckpointStore()
    envelope = _strict_envelope()
    payload = envelope.to_dict()
    assert store.create_checkpoint(_minimal_checkpoint(key=envelope.checkpoint_config.key))
    results: list[bool] = []

    def claim(owner: str) -> None:
        results.append(
            store.claim_distributed_dispatch(
                payload,
                claim_token=owner,
                lease_expires_at_ms=10_000,
                now_ms=1,
            ).should_enqueue
        )

    threads = [threading.Thread(target=claim, args=(owner,)) for owner in ("race-a", "race-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False, True]
    raced = store.get_distributed_dispatch(payload["job_id"])
    assert raced is not None
    assert raced.attempt == 1


class _RecordingBroker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_on_send = False

    def send_task(self, _name: str, *, kwargs: dict[str, Any], **options: Any) -> object:
        self.calls.append({"kwargs": copy.deepcopy(kwargs), "options": copy.deepcopy(options)})
        if self.raise_on_send:
            raise RuntimeError("broker send failed")
        return object()


def _backend_for_outbox(
    store: InMemoryCheckpointStore,
    broker: _RecordingBroker,
    *,
    with_dispatch_adapter: bool = True,
) -> tuple[CeleryBackend, DistributedRunEnvelope]:
    envelope = _strict_envelope()
    ref = envelope.recipe.capabilities.checkpoint_store_ref
    assert ref is not None
    from vv_agent.runtime.backends.distributed import DistributedCapabilityRegistry

    registry = DistributedCapabilityRegistry()
    registry.register("checkpoint_store", ref, store)
    backend = CeleryBackend(
        broker,
        runtime_recipe=envelope.recipe,
        capability_registry=registry,
        dispatch_outbox_store=store if with_dispatch_adapter else None,
    )
    return backend, envelope


def test_celery_enqueue_failure_reconciles_and_retries_same_task_id() -> None:
    store = InMemoryCheckpointStore()
    broker = _RecordingBroker()
    backend, envelope = _backend_for_outbox(store, broker)
    assert store.create_checkpoint(_minimal_checkpoint(key=envelope.checkpoint_config.key))
    handle = DistributedRunHandle(
        envelope.checkpoint_config.key,
        envelope.root_run_id,
        envelope.trace_id,
    )
    broker.raise_on_send = True
    with pytest.raises(RuntimeError, match="broker send failed"):
        backend._enqueue_envelope(envelope, handle=handle, continuation=None)
    ambiguous = store.get_distributed_dispatch(envelope.job_id)
    assert ambiguous is not None and ambiguous.state == "ambiguous"
    store.reconcile_distributed_dispatch(
        dispatch_id=envelope.job_id,
        envelope_digest=ambiguous.envelope_digest,
        outcome="retry",
        now_ms=2,
        error="explicit broker retry",
    )
    broker.raise_on_send = False
    backend._enqueue_envelope(envelope, handle=handle, continuation=None)
    assert [call["options"]["task_id"] for call in broker.calls] == [envelope.job_id, envelope.job_id]
    delivered = store.get_distributed_dispatch(envelope.job_id)
    assert delivered is not None and delivered.state == "delivered" and delivered.attempt == 2


def test_celery_crash_after_broker_return_is_reaped_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryCheckpointStore()
    broker = _RecordingBroker()
    backend, envelope = _backend_for_outbox(store, broker)
    assert store.create_checkpoint(_minimal_checkpoint(key=envelope.checkpoint_config.key))
    handle = DistributedRunHandle(envelope.checkpoint_config.key, envelope.root_run_id, envelope.trace_id)
    original_complete = store.complete_distributed_dispatch

    def crash_after_send(**_kwargs: Any) -> Any:
        raise KeyboardInterrupt("crash after broker accepted task")

    monkeypatch.setattr(store, "complete_distributed_dispatch", crash_after_send)
    with pytest.raises(KeyboardInterrupt, match="crash after broker accepted task"):
        backend._enqueue_envelope(envelope, handle=handle, continuation=None)
    assert len(broker.calls) == 1
    claimed = store.get_distributed_dispatch(envelope.job_id)
    assert claimed is not None and claimed.state == "claimed"
    assert claimed.lease_expires_at_ms is not None
    monkeypatch.setattr(store, "complete_distributed_dispatch", original_complete)
    reaped = store.reap_distributed_dispatches(
        checkpoint_key=envelope.checkpoint_config.key,
        now_ms=claimed.lease_expires_at_ms,
    )
    assert reaped and reaped[0].state == "ambiguous"
    store.reconcile_distributed_dispatch(
        dispatch_id=envelope.job_id,
        envelope_digest=claimed.envelope_digest,
        outcome="retry",
        now_ms=claimed.lease_expires_at_ms + 1,
        error="operator reconciled broker ambiguity",
    )
    backend._enqueue_envelope(envelope, handle=handle, continuation=None)
    assert len(broker.calls) == 2
    assert broker.calls[0]["options"]["task_id"] == broker.calls[1]["options"]["task_id"] == envelope.job_id


def test_celery_without_dispatch_adapter_uses_stable_task_id_and_no_checkpoint_receipt() -> None:
    store = InMemoryCheckpointStore()
    broker = _RecordingBroker()
    backend, envelope = _backend_for_outbox(store, broker, with_dispatch_adapter=False)
    assert store.create_checkpoint(_minimal_checkpoint(key=envelope.checkpoint_config.key))
    handle = DistributedRunHandle(envelope.checkpoint_config.key, envelope.root_run_id, envelope.trace_id)

    backend._enqueue_envelope(envelope, handle=handle, continuation=None)
    backend._enqueue_envelope(envelope, handle=handle, continuation=None)

    assert [call["options"]["task_id"] for call in broker.calls] == [envelope.job_id, envelope.job_id]
    assert store.get_distributed_dispatch(envelope.job_id) is None


@pytest.mark.skipif(not os.getenv("VV_AGENT_TEST_REDIS_URL"), reason="set VV_AGENT_TEST_REDIS_URL for live Redis")
def test_real_redis_dispatch_receipt_replay_and_reconcile() -> None:
    store = RedisCheckpointStore(os.environ["VV_AGENT_TEST_REDIS_URL"])
    envelope = _strict_envelope()
    payload = envelope.to_dict()
    key = f"{envelope.checkpoint_config.key}-live-dispatch"
    payload["checkpoint_config"]["key"] = key
    payload["job_id"] = f"{payload['run_id']}:cycle:{payload['cycle_index']}"
    checkpoint = _minimal_checkpoint(key=key)
    try:
        assert store.create_checkpoint(checkpoint)
        first = store.claim_distributed_dispatch(
            payload,
            claim_token="live-owner",
            lease_expires_at_ms=10_000,
            now_ms=1,
        )
        replay = store.claim_distributed_dispatch(
            payload,
            claim_token="live-replay",
            lease_expires_at_ms=10_000,
            now_ms=2,
        )
        assert first.should_enqueue and not replay.should_enqueue
        store.complete_distributed_dispatch(
            dispatch_id=payload["job_id"],
            envelope_digest=first.record.envelope_digest,
            claim_token="live-owner",
            attempt=1,
            outcome="delivered",
            now_ms=3,
        )
        delivered = store.get_distributed_dispatch(payload["job_id"])
        assert delivered is not None
        assert delivered.state == "delivered"
    finally:
        store.delete_checkpoint(key)
