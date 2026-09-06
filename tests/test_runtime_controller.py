from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from vv_agent.checkpoint import CheckpointConfig, CheckpointError, OperationState, ResumePolicy
from vv_agent.runtime.backends.distributed import DistributedRunHandle
from vv_agent.runtime.checkpoint_codec import checkpoint_from_dict, checkpoint_to_dict
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.controller import (
    ControllerCommand,
    DistributedBackend,
    HostInteractionAdmissionContext,
    HostInteractionRecoveryEnvelope,
    HostInteractionRequest,
    HostInteractionResponse,
    derive_controller_command_id,
    derive_host_response_digest,
    sanitize_host_prompt,
)
from vv_agent.runtime.engine import AgentRuntime
from vv_agent.runtime.state import OperationJournalEntry
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.runtime.stores.redis import RedisCheckpointStore
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore
from vv_agent.types import AgentStatus

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
SetOfStrings = set[str]


def _checkpoint(key: str) -> Any:
    fixture = json.loads((FIXTURE_DIR / "checkpoint_codec.json").read_text(encoding="utf-8"))
    payload = next(case["payload"] for case in fixture["valid_cases"] if case["name"] == "minimal_running")
    payload = dict(payload)
    payload["checkpoint_key"] = key
    return checkpoint_from_dict(payload)


def _redis_store() -> RedisCheckpointStore:
    class WatchError(Exception):
        pass

    class Pipeline:
        def __init__(self, client: Client) -> None:
            self.client = client
            self.commands: list[tuple[str, str, str | None]] = []
            self.transaction = False

        def __enter__(self) -> Pipeline:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def watch(self, *_keys: str) -> None:
            return None

        def unwatch(self) -> None:
            self.transaction = False
            self.commands.clear()

        def get(self, key: str) -> str | None:
            return self.client.get(key)

        def time(self) -> tuple[int, int]:
            return self.client.time()

        def smembers(self, key: str) -> SetOfStrings:
            return self.client.smembers(key)

        def multi(self) -> None:
            self.transaction = True

        def set(self, key: str, value: str, *, nx: bool = False) -> None:
            if self.transaction:
                self.commands.append(("set_nx" if nx else "set", key, value))
            else:
                self.client.set(key, value, nx=nx)

        def sadd(self, key: str, value: str) -> None:
            if self.transaction:
                self.commands.append(("sadd", key, value))
            else:
                self.client.sadd(key, value)

        def srem(self, key: str, value: str) -> None:
            if self.transaction:
                self.commands.append(("srem", key, value))
            else:
                self.client.srem(key, value)

        def delete(self, key: str) -> None:
            if self.transaction:
                self.commands.append(("delete", key, None))
            else:
                self.client.delete(key)

        def execute(self) -> list[object]:
            result: list[object] = []
            for kind, key, value in self.commands:
                if kind in {"set", "set_nx"}:
                    assert value is not None
                    result.append(self.client.set(key, value, nx=kind == "set_nx"))
                elif kind == "sadd":
                    assert value is not None
                    result.append(self.client.sadd(key, value))
                elif kind == "srem":
                    assert value is not None
                    result.append(self.client.srem(key, value))
                else:
                    result.append(self.client.delete(key))
            self.commands.clear()
            self.transaction = False
            return result

    class Client:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.sets: dict[str, SetOfStrings] = {}
            self.server_now_ms = 0

        def set(self, key: str, value: str, *, nx: bool = False) -> bool:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

        def get(self, key: str) -> str | None:
            return self.values.get(key)

        def sadd(self, key: str, value: str) -> int:
            members = self.sets.setdefault(key, set())
            before = len(members)
            members.add(value)
            return int(len(members) != before)

        def srem(self, key: str, value: str) -> int:
            members = self.sets.get(key)
            if members is None or value not in members:
                return 0
            members.remove(value)
            if not members:
                self.sets.pop(key, None)
            return 1

        def smembers(self, key: str) -> SetOfStrings:
            return set(self.sets.get(key, set()))

        def delete(self, *keys: str) -> int:
            return sum(int(self.values.pop(key, None) is not None) for key in keys)

        def pipeline(self) -> Pipeline:
            return Pipeline(self)

        def scan_iter(self, pattern: str) -> list[str]:
            return [key for key in self.values if key.startswith(pattern.removesuffix("*"))]

        def time(self) -> tuple[int, int]:
            return self.server_now_ms // 1000, (self.server_now_ms % 1000) * 1000

    store = RedisCheckpointStore.__new__(RedisCheckpointStore)
    store._watch_error = WatchError
    store._client = Client()
    return store


@pytest.fixture(params=["memory", "sqlite", "redis"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        return InMemoryCheckpointStore()
    if request.param == "sqlite":
        return SqliteCheckpointStore(tmp_path / "controller.sqlite3")
    return _redis_store()


def _admit_host(store: Any, key: str = "controller-run") -> tuple[Any, HostInteractionRequest, Any]:
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    request = HostInteractionRequest(
        interaction_id="interaction-1",
        logical_cycle=1,
        operation_id="operation-1",
        tool_call_id="tool-1",
        prompt="Approve sk-test-123 at https://example.invalid/run?secret=abc",
    )
    outcome = store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=key,
            claim_token="worker-claim",
            expected_revision=claimed.revision,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    return checkpoint, request, outcome


@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_claim_ignores_foreign_resolved_host_record(store_kind: str) -> None:
    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()
    foreign_key = f"redis-foreign-record-{uuid4().hex}"
    target_key = f"redis-foreign-target-{uuid4().hex}"
    _foreign_checkpoint, request, _outcome = _admit_host(store, foreign_key)
    target = _checkpoint(target_key)
    assert store.create_checkpoint(target)
    command = _host_response_command(store, request, key=foreign_key, command_id=f"foreign-command-{uuid4().hex}")
    assert store.resolve_controller_command(command).kind == "applied"
    record_key = store._host_record_key(foreign_key, request.interaction_id)  # type: ignore[attr-defined]
    target_record_set_key = store._host_record_set_key(target_key)  # type: ignore[attr-defined]
    store._client.sadd(target_record_set_key, record_key)  # type: ignore[attr-defined]
    try:
        claimed = store.claim_checkpoint(
            target_key,
            1,
            claim_token="target-owner",
            lease_expires_at_ms=10_000,
            now_ms=1,
            claim_mode="continue",
        )
        assert claimed is not None
    finally:
        store._client.srem(target_record_set_key, record_key)  # type: ignore[attr-defined]
        store.delete_checkpoint(target_key)
        store.delete_checkpoint(foreign_key)


def _host_response_command(store: Any, request: HostInteractionRequest, *, key: str, command_id: str) -> ControllerCommand:
    current = store.load_checkpoint(key)
    assert current is not None
    return ControllerCommand(
        command_id=command_id,
        handle=DistributedRunHandle(key, current.root_run_id, current.trace_id),
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={
            "kind": "host_interaction_response",
            "interaction_id": request.interaction_id,
            "logical_cycle": request.logical_cycle,
            "operation_id": request.operation_id,
            "tool_call_id": request.tool_call_id,
            "request_digest": request.request_digest,
            "response": {"role": "user", "content": "Approved sk-response-456 at https://example.invalid/next"},
        },
    )


def _recovery_envelope(store: Any, request: HostInteractionRequest, outcome: Any, *, key: str, command_id: str) -> dict[str, Any]:
    current = store.load_checkpoint(key)
    assert current is not None
    return HostInteractionRecoveryEnvelope(
        record_id=outcome.record_id,
        checkpoint_key=key,
        run_id=current.root_run_id,
        trace_id=current.trace_id,
        claim_mode="recovery",
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        logical_cycle=request.logical_cycle,
        interaction_id=request.interaction_id,
        operation_id=request.operation_id,
        tool_call_id=request.tool_call_id,
        request_digest=request.request_digest or "",
        command_id=command_id,
    ).to_dict()


def test_command_id_matches_contract_vectors() -> None:
    assert derive_controller_command_id("thread-1", "turn-1", "same-action") == (
        "48d6ee2d2a12b910a61370db73c06835bfe3946258bff4eff1cfd6739bd5be9a"
    )
    assert derive_controller_command_id("thread/中文", "turn-😀", "same-action") == (
        "a8e2cd09beba02260172b94cfd8b2023b5e9f3182fa7bd25377d5c5e957411f5"
    )


def test_host_producer_requires_explicit_admission_context() -> None:
    store = InMemoryCheckpointStore()
    request = HostInteractionRequest(
        interaction_id="unbound-interaction",
        logical_cycle=1,
        operation_id="unbound-operation",
        tool_call_id="unbound-tool",
        prompt="Choose.",
    )
    with pytest.raises(CheckpointError, match="explicit admission context"):
        DistributedBackend(store).produce_host_interaction(request)


def test_engine_binds_typed_host_producer_to_the_active_checkpoint_claim() -> None:
    store = InMemoryCheckpointStore()
    key = "engine-host-producer"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="engine-claim",
        lease_expires_at_ms=9_999_999_999_999,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    controller = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
        task_id=claimed.task_id,
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        run_definition=claimed.run_definition,
        run_definition_digest=claimed.run_definition_digest,
        initial_messages=claimed.messages,
        initial_shared_state=claimed.shared_state,
        initial_budget_usage=claimed.budget_usage,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
        preloaded_checkpoint=claimed,
    )
    controller.checkpoint = claimed
    context = ExecutionContext(metadata={"_vv_agent_checkpoint_controller": controller})
    AgentRuntime._bind_host_interaction_producer(context)
    outcome = context.produce_host_interaction(
        HostInteractionRequest(
            interaction_id="engine-interaction",
            logical_cycle=1,
            operation_id="engine-operation",
            tool_call_id="engine-tool",
            prompt="Choose.",
        )
    )
    assert outcome.status == "admitted"
    assert outcome.outbox_state == "pending"


def test_checkpoint_heartbeat_refreshes_host_producer_lease_fence() -> None:
    store = InMemoryCheckpointStore()
    key = "heartbeat-host-producer"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="heartbeat-claim",
        lease_expires_at_ms=9_999_999_999_999,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    controller = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
        task_id=claimed.task_id,
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        run_definition=claimed.run_definition,
        run_definition_digest=claimed.run_definition_digest,
        initial_messages=claimed.messages,
        initial_shared_state=claimed.shared_state,
        initial_budget_usage=claimed.budget_usage,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
        preloaded_checkpoint=claimed,
        lease_duration_ms=60,
    )
    controller.checkpoint = claimed
    before = claimed.lease_expires_at_ms
    controller._start_heartbeat()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and claimed.lease_expires_at_ms == before:
            time.sleep(0.01)
        assert claimed.lease_expires_at_ms is not None and claimed.lease_expires_at_ms != before
        context = controller.host_interaction_admission_context()
        assert context.lease_expires_at_ms == claimed.lease_expires_at_ms
        authoritative = store.load_checkpoint(key)
        assert authoritative is not None
        assert authoritative.lease_expires_at_ms == context.lease_expires_at_ms
    finally:
        controller.close()


def test_host_producer_reads_authoritative_renewed_lease() -> None:
    store = InMemoryCheckpointStore()
    key = "authoritative-host-producer"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    now_ms = int(time.time() * 1000)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="authoritative-claim",
        lease_expires_at_ms=now_ms + 100_000,
        now_ms=now_ms,
        claim_mode="continue",
    )
    assert claimed is not None
    controller = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=key, resume_policy=ResumePolicy.RESUME_IF_PRESENT),
        task_id=claimed.task_id,
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        run_definition=claimed.run_definition,
        run_definition_digest=claimed.run_definition_digest,
        initial_messages=claimed.messages,
        initial_shared_state=claimed.shared_state,
        initial_budget_usage=claimed.budget_usage,
        extensions=[],
        reconciliation_provider=None,
        event_sink=lambda _event: None,
        preloaded_checkpoint=claimed,
    )
    controller.checkpoint = claimed
    renewed_lease = now_ms + 200_000
    assert store.renew_checkpoint_claim(
        key,
        claim_token="authoritative-claim",
        lease_expires_at_ms=renewed_lease,
        now_ms=now_ms + 1,
    )
    assert claimed.lease_expires_at_ms != renewed_lease
    context = controller.host_interaction_admission_context()
    assert context.lease_expires_at_ms == renewed_lease
    assert controller.checkpoint is not None
    assert controller.checkpoint.lease_expires_at_ms == renewed_lease


def test_host_response_constructor_sanitizes_and_from_dict_rejects_unsanitized() -> None:
    request_digest = "a" * 64
    content = "Approve sk-response-456 at https://example.invalid/next"
    response = HostInteractionResponse(
        interaction_id="response-interaction",
        logical_cycle=1,
        operation_id="response-operation",
        tool_call_id="response-tool",
        request_digest=request_digest,
        command_id="response-command",
        response={"role": "user", "content": content},
    )
    expected_content = sanitize_host_prompt(content)
    assert response.response["content"] == expected_content
    assert response.response_digest == derive_host_response_digest(
        interaction_id=response.interaction_id,
        logical_cycle=response.logical_cycle,
        operation_id=response.operation_id,
        tool_call_id=response.tool_call_id,
        request_digest=response.request_digest,
        command_id=response.command_id,
        response=response.response,
    )
    persisted = response.to_dict()
    persisted["response"] = {"role": "user", "content": content}
    with pytest.raises(ValueError, match="not sanitized"):
        HostInteractionResponse.from_dict(persisted)


def test_host_request_constructor_sanitizes_and_from_dict_rejects_unsanitized() -> None:
    prompt = "Approve sk-request-123 at https://example.invalid/request"
    request = HostInteractionRequest(
        interaction_id="request-interaction",
        logical_cycle=1,
        operation_id="request-operation",
        tool_call_id="request-tool",
        prompt=prompt,
    )
    assert request.prompt == sanitize_host_prompt(prompt)
    persisted = request.to_dict()
    persisted["prompt"] = prompt
    with pytest.raises(ValueError, match="not sanitized"):
        HostInteractionRequest.from_dict(persisted)


def test_controller_wire_readers_reject_null_and_unknown_digest_fields() -> None:
    request = HostInteractionRequest(
        interaction_id="strict-request",
        logical_cycle=1,
        operation_id="strict-operation",
        tool_call_id="strict-tool",
        prompt="Choose.",
    )
    response = HostInteractionResponse(
        interaction_id=request.interaction_id,
        logical_cycle=request.logical_cycle,
        operation_id=request.operation_id,
        tool_call_id=request.tool_call_id,
        request_digest=request.request_digest or "",
        command_id="strict-command",
        response={"role": "user", "content": "Approved."},
    )
    command = ControllerCommand(
        command_id="strict-command",
        handle=DistributedRunHandle("strict-checkpoint", "strict-run", "strict-trace"),
        resume_attempt=1,
        expected_revision=0,
        command={"kind": "suspend"},
    )
    readers = (
        ("request_digest", HostInteractionRequest.from_dict, request.to_dict()),
        ("response_digest", HostInteractionResponse.from_dict, response.to_dict()),
        ("command_digest", ControllerCommand.from_dict, command.to_dict()),
    )
    for digest_field, reader, payload in readers:
        null_digest = dict(payload)
        null_digest[digest_field] = None
        with pytest.raises(ValueError, match=f"{digest_field} must be a lowercase SHA-256 digest"):
            reader(null_digest)

        unknown_field = dict(payload)
        unknown_field["unknown_digest_alias"] = "ignored"
        with pytest.raises(ValueError, match="unknown="):
            reader(unknown_field)


def test_notification_reconcile_replay_conflict_and_closed_state(store: Any) -> None:
    _checkpoint_value, _request, outcome = _admit_host(store, "notification-reconcile-contract")
    claimed = store.claim_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="notification-reconcile-owner",
        lease_expires_at_ms=10_000,
        now_ms=1,
    )
    assert claimed is not None
    ambiguous = store.complete_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="notification-reconcile-owner",
        attempt=int(claimed["attempt"]),
        outcome="ambiguous",
        now_ms=2,
    )
    assert ambiguous is not None
    retried = store.reconcile_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        outcome="retry",
        now_ms=3,
    )
    assert retried is not None and retried["outbox_state"] == "pending"
    assert (
        store.reconcile_host_interaction_notification(
            notification_id=outcome.notification_id,
            payload_digest=outcome.notification_payload_digest,
            outcome="retry",
            now_ms=4,
        )
        == retried
    )
    with pytest.raises(CheckpointError, match="digest"):
        store.reconcile_host_interaction_notification(
            notification_id=outcome.notification_id,
            payload_digest="b" * 64,
            outcome="retry",
            now_ms=5,
        )
    delivered_claim = store.claim_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="notification-reconcile-owner-2",
        lease_expires_at_ms=20_000,
        now_ms=5,
    )
    assert delivered_claim is not None
    delivered = store.complete_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="notification-reconcile-owner-2",
        attempt=int(delivered_claim["attempt"]),
        outcome="delivered",
        now_ms=6,
    )
    assert delivered is not None and delivered["outbox_state"] == "delivered"
    assert (
        store.reconcile_host_interaction_notification(
            notification_id=outcome.notification_id,
            payload_digest=outcome.notification_payload_digest,
            outcome="delivered",
            now_ms=7,
        )
        == delivered
    )
    with pytest.raises(CheckpointError, match="ambiguous"):
        store.reconcile_host_interaction_notification(
            notification_id=outcome.notification_id,
            payload_digest=outcome.notification_payload_digest,
            outcome="retry",
            now_ms=8,
        )


@pytest.mark.parametrize("operation", ["get", "claim", "complete", "reconcile"])
@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_notification_payload_key_binding_fails_closed(operation: str, store_kind: str) -> None:
    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()
    first_key = f"redis-notification-first-{uuid4().hex}"
    second_key = f"redis-notification-second-{uuid4().hex}"
    _first_checkpoint, _first_request, first_outcome = _admit_host(store, first_key)
    _second_checkpoint, _second_request, second_outcome = _admit_host(store, second_key)
    first_notification_key = store._host_notification_key(first_outcome.notification_id)  # type: ignore[attr-defined]
    second_notification_key = store._host_notification_key(second_outcome.notification_id)  # type: ignore[attr-defined]
    first_payload = store._client.get(first_notification_key)  # type: ignore[attr-defined]
    second_payload = store._client.get(second_notification_key)  # type: ignore[attr-defined]
    assert first_payload is not None and second_payload is not None
    claim_attempt = 0
    try:
        if operation in {"complete", "reconcile"}:
            claimed = store.claim_host_interaction_notification(
                notification_id=first_outcome.notification_id,
                payload_digest=first_outcome.notification_payload_digest,
                claim_token="notification-binding-owner",
                lease_expires_at_ms=10_000,
                now_ms=1,
            )
            assert claimed is not None
            claim_attempt = int(claimed["attempt"])
        if operation == "reconcile":
            ambiguous = store.complete_host_interaction_notification(
                notification_id=first_outcome.notification_id,
                payload_digest=first_outcome.notification_payload_digest,
                claim_token="notification-binding-owner",
                attempt=claim_attempt,
                outcome="ambiguous",
                now_ms=2,
            )
            assert ambiguous is not None
        store._client.set(first_notification_key, second_payload)  # type: ignore[attr-defined]
        store._client.set(second_notification_key, first_payload)  # type: ignore[attr-defined]
        swapped = (store._client.get(first_notification_key), store._client.get(second_notification_key))  # type: ignore[attr-defined]
        with pytest.raises(CheckpointError) as error:
            if operation == "get":
                store.get_host_interaction_notification(first_outcome.notification_id)
            elif operation == "claim":
                store.claim_host_interaction_notification(
                    notification_id=first_outcome.notification_id,
                    payload_digest=first_outcome.notification_payload_digest,
                    claim_token="notification-binding-owner",
                    lease_expires_at_ms=10_000,
                    now_ms=1,
                )
            elif operation == "complete":
                store.complete_host_interaction_notification(
                    notification_id=first_outcome.notification_id,
                    payload_digest=first_outcome.notification_payload_digest,
                    claim_token="notification-binding-owner",
                    attempt=claim_attempt,
                    outcome="delivered",
                    now_ms=3,
                )
            else:
                store.reconcile_host_interaction_notification(
                    notification_id=first_outcome.notification_id,
                    payload_digest=first_outcome.notification_payload_digest,
                    outcome="retry",
                    now_ms=3,
                )
        assert error.value.code == "host_interaction_conflict"
        assert (store._client.get(first_notification_key), store._client.get(second_notification_key)) == swapped  # type: ignore[attr-defined]
    finally:
        store._client.set(first_notification_key, first_payload)  # type: ignore[attr-defined]
        store._client.set(second_notification_key, second_payload)  # type: ignore[attr-defined]
        store.delete_checkpoint(first_key)
        store.delete_checkpoint(second_key)


@pytest.mark.parametrize("store_kind", ["fake", "real"])
def test_redis_host_replay_payload_key_binding_fails_closed(store_kind: str) -> None:
    if store_kind == "real":
        redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
        if not redis_url:
            pytest.skip("set VV_AGENT_TEST_REDIS_URL for live Redis")
        store: Any = RedisCheckpointStore(redis_url)
    else:
        store = _redis_store()
    first_key = f"redis-host-replay-first-{uuid4().hex}"
    second_key = f"redis-host-replay-second-{uuid4().hex}"
    _first_checkpoint, first_request, first_outcome = _admit_host(store, first_key)
    _second_checkpoint, _second_request, second_outcome = _admit_host(store, second_key)
    first_record_key = store._host_record_key(first_key, first_request.interaction_id)  # type: ignore[attr-defined]
    second_record_key = store._host_record_key(second_key, _second_request.interaction_id)  # type: ignore[attr-defined]
    first_notification_key = store._host_notification_key(first_outcome.notification_id)  # type: ignore[attr-defined]
    second_notification_key = store._host_notification_key(second_outcome.notification_id)  # type: ignore[attr-defined]
    first_record = store._client.get(first_record_key)  # type: ignore[attr-defined]
    second_record = store._client.get(second_record_key)  # type: ignore[attr-defined]
    first_notification = store._client.get(first_notification_key)  # type: ignore[attr-defined]
    second_notification = store._client.get(second_notification_key)  # type: ignore[attr-defined]
    assert all(value is not None for value in (first_record, second_record, first_notification, second_notification))
    context = HostInteractionAdmissionContext(
        checkpoint_key=first_key,
        claim_token="worker-claim",
        expected_revision=first_outcome.checkpoint_revision - 1,
        claimed_cycle=1,
        now_ms=1,
        lease_expires_at_ms=10_000,
    )
    try:
        for swap_records, swap_notifications in ((True, False), (False, True)):
            store._client.set(first_record_key, second_record if swap_records else first_record)  # type: ignore[attr-defined]
            store._client.set(second_record_key, first_record if swap_records else second_record)  # type: ignore[attr-defined]
            store._client.set(  # type: ignore[attr-defined]
                first_notification_key,
                second_notification if swap_notifications else first_notification,
            )
            store._client.set(  # type: ignore[attr-defined]
                second_notification_key,
                first_notification if swap_notifications else second_notification,
            )
            before = (
                store._client.get(first_record_key),  # type: ignore[attr-defined]
                store._client.get(second_record_key),  # type: ignore[attr-defined]
                store._client.get(first_notification_key),  # type: ignore[attr-defined]
                store._client.get(second_notification_key),  # type: ignore[attr-defined]
            )
            with pytest.raises(CheckpointError) as error:
                store.produce_host_interaction(first_request, admission_context=context)
            assert error.value.code == "host_interaction_conflict"
            after = (
                store._client.get(first_record_key),  # type: ignore[attr-defined]
                store._client.get(second_record_key),  # type: ignore[attr-defined]
                store._client.get(first_notification_key),  # type: ignore[attr-defined]
                store._client.get(second_notification_key),  # type: ignore[attr-defined]
            )
            assert after == before
    finally:
        store._client.set(first_record_key, first_record)  # type: ignore[attr-defined]
        store._client.set(second_record_key, second_record)  # type: ignore[attr-defined]
        store._client.set(first_notification_key, first_notification)  # type: ignore[attr-defined]
        store._client.set(second_notification_key, second_notification)  # type: ignore[attr-defined]
        store.delete_checkpoint(first_key)
        store.delete_checkpoint(second_key)


def test_abort_admits_from_reconciliation_and_replays_terminal_result(store: Any) -> None:
    key = "controller-abort"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="abort-owner",
        lease_expires_at_ms=10_000,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    claimed.tool_journal = [
        OperationJournalEntry.from_dict(
            {
                "kind": "tool",
                "operation_id": "op_abort_tool",
                "cycle_index": 1,
                "attempt": 1,
                "state": OperationState.AMBIGUOUS.value,
                "request_digest": "a" * 64,
                "idempotency_key": "idem_abort_tool",
                "tool_call_id": "call_abort_tool",
                "tool_name": "write_record",
                "arguments": {"record_id": "42", "value": "approved"},
                "idempotency_support": "unknown",
                "result": None,
                "error": None,
            }
        )
    ]
    claimed.status = AgentStatus.RECONCILIATION_REQUIRED
    assert store.suspend_checkpoint(
        claimed,
        claim_token="abort-owner",
        expected_revision=claimed.revision,
    )
    current = store.load_checkpoint(key)
    assert current is not None
    command = ControllerCommand(
        command_id="abort-command",
        handle=DistributedRunHandle(key, current.root_run_id, current.trace_id),
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "abort"},
    )
    first = store.resolve_controller_command(command)
    assert first.kind == "applied"
    assert first.receipt is not None and first.receipt.resulting_status == "failed"
    terminal = store.load_checkpoint(key)
    assert terminal is not None and terminal.terminal_result is not None
    assert terminal.terminal_result.error == {
        "code": "operator_abort_with_unknown_outcome",
        "message": "Operator accepted that the external outcome is unknown.",
        "retryable": False,
    }
    assert terminal.terminal_result.error_code is None
    failed_events = [entry.event for entry in terminal.event_outbox if entry.event.get("type") == "run_failed"]
    assert failed_events and failed_events[-1]["error"] == "failed"
    assert failed_events[-1]["metadata"]["error_code"] == "operator_abort_with_unknown_outcome"
    lifecycle_types = [
        entry.event["type"]
        for entry in terminal.event_outbox
        if entry.event["type"] in {"cycle_aborted", "run_state_changed", "run_failed", "run_cancelled"}
    ]
    assert lifecycle_types[-3:] == ["cycle_aborted", "run_state_changed", "run_failed"]
    replay = store.resolve_controller_command(command)
    assert replay.kind == "replayed"
    assert replay.receipt == first.receipt


def test_distinct_live_cancel_after_signal_is_applied_noop(store: Any) -> None:
    key = "controller-live-cancel-noop"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    now_ms = int(time.time() * 1000)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=now_ms + 10_000,
        now_ms=now_ms,
        claim_mode="continue",
    )
    assert claimed is not None
    first_command = ControllerCommand(
        command_id="live-cancel-first",
        handle=DistributedRunHandle(key, claimed.root_run_id, claimed.trace_id),
        resume_attempt=claimed.resume_attempt,
        expected_revision=claimed.revision,
        command={"kind": "cancel"},
    )
    first = store.resolve_controller_command(first_command)
    assert first.kind == "applied"
    after_first = store.load_checkpoint(key)
    assert after_first is not None and after_first.cancel_requested is True
    assert after_first.claim_token == "worker-claim"
    assert len(after_first.event_outbox) == len(claimed.event_outbox) + 1
    cancel_event = after_first.event_outbox[-1].event
    assert cancel_event["type"] == "run_state_changed"
    assert cancel_event["cancel_requested"] == {"from": False, "to": True}
    assert "cancel_requested" not in cancel_event.get("metadata", {})

    second_command = ControllerCommand(
        command_id="live-cancel-second",
        handle=DistributedRunHandle(key, claimed.root_run_id, claimed.trace_id),
        resume_attempt=after_first.resume_attempt,
        expected_revision=after_first.revision,
        command={"kind": "cancel"},
    )
    second = store.resolve_controller_command(second_command)
    assert second.kind == "applied"
    assert second.receipt is not None
    assert second.receipt.outbox_action == "none"
    assert second.receipt.resulting_revision == after_first.revision
    after_second = store.load_checkpoint(key)
    assert after_second is not None
    assert checkpoint_to_dict(after_second) == checkpoint_to_dict(after_first)
    assert len(after_second.event_outbox) == len(after_first.event_outbox)
    if isinstance(store, InMemoryCheckpointStore):
        assert store._controller_command_outboxes["live-cancel-second"]["delivered_at_ms"] is None  # type: ignore[attr-defined]

    replay = store.resolve_controller_command(second_command)
    assert replay.kind == "replayed"
    assert replay.receipt == second.receipt
    after_replay = store.load_checkpoint(key)
    assert after_replay is not None
    assert checkpoint_to_dict(after_replay) == checkpoint_to_dict(after_second)


def test_host_request_notification_is_sanitized_and_replay_is_zero_write(store: Any) -> None:
    checkpoint, request, outcome = _admit_host(store)
    stored = store.load_checkpoint(checkpoint.checkpoint_key)
    assert stored is not None and stored.active_host_interaction is not None
    assert stored.active_host_interaction["prompt"] == "Approve [credential redacted] at [external locator redacted]"
    replay = store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=checkpoint.checkpoint_key,
            claim_token="worker-claim",
            expected_revision=outcome.checkpoint_revision - 1,
            claimed_cycle=1,
            now_ms=1,
            lease_expires_at_ms=10_000,
        ),
    )
    assert replay.status == "replayed"
    assert replay.checkpoint_revision == outcome.checkpoint_revision
    notification = store.claim_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="observer-1",
        lease_expires_at_ms=10_000,
        now_ms=1,
    )
    assert notification is not None
    prompt = notification["payload"]["prompt"]
    assert "sk-test" not in prompt
    assert "example.invalid" not in prompt


def test_host_producer_outcome_stays_pending_after_notification_delivery(store: Any) -> None:
    checkpoint, request, outcome = _admit_host(store, "producer-pending-outcome")
    claimed = store.claim_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="observer-delivery",
        lease_expires_at_ms=10_000,
        now_ms=1,
    )
    assert claimed is not None
    delivered = store.complete_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="observer-delivery",
        attempt=int(claimed["attempt"]),
        outcome="delivered",
        now_ms=2,
    )
    assert delivered is not None and delivered["outbox_state"] == "delivered"
    replay = store.produce_host_interaction(
        request,
        admission_context=HostInteractionAdmissionContext(
            checkpoint_key=checkpoint.checkpoint_key,
            claim_token="worker-claim",
            expected_revision=outcome.checkpoint_revision - 1,
            claimed_cycle=1,
            now_ms=2,
            lease_expires_at_ms=10_000,
        ),
    )
    assert replay.status == "replayed"
    assert replay.outbox_state == "pending"


def test_host_notification_sanitizer_is_exact_and_idempotent() -> None:
    from vv_agent.runtime.controller import sanitize_host_prompt

    prompt = "Approve token sk-test-123 at https://example.invalid/run?secret=abc"
    expected = "Approve [credential redacted] at [external locator redacted]"
    assert sanitize_host_prompt(prompt) == expected
    assert sanitize_host_prompt(expected) == expected


def test_host_producer_rejects_expired_or_wrong_execution_claim_without_writes(store: Any) -> None:
    key = "expired-producer"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="worker-claim",
        lease_expires_at_ms=10,
        now_ms=1,
        claim_mode="continue",
    )
    assert claimed is not None
    request = HostInteractionRequest(
        interaction_id="expired-interaction",
        logical_cycle=1,
        operation_id="expired-operation",
        tool_call_id="expired-tool",
        prompt="Choose.",
    )
    before = store.load_checkpoint(key)
    assert before is not None
    with pytest.raises(CheckpointError, match="active claim"):
        store.produce_host_interaction(
            request,
            admission_context=HostInteractionAdmissionContext(
                checkpoint_key=key,
                claim_token="worker-claim",
                expected_revision=claimed.revision,
                claimed_cycle=1,
                now_ms=10,
                lease_expires_at_ms=10,
            ),
        )
    after = store.load_checkpoint(key)
    assert after is not None and after.revision == before.revision and after.status is before.status
    with pytest.raises(CheckpointError, match="active claim"):
        store.produce_host_interaction(
            request,
            admission_context=HostInteractionAdmissionContext(
                checkpoint_key=key,
                claim_token="old-owner",
                expected_revision=claimed.revision,
                claimed_cycle=1,
                now_ms=1,
                lease_expires_at_ms=10,
            ),
        )
    unchanged = store.load_checkpoint(key)
    assert unchanged is not None and unchanged.revision == before.revision


def test_host_response_recovery_is_strict_sanitized_and_replayable(store: Any) -> None:
    _checkpoint_value, request, outcome = _admit_host(store, "response-recovery")
    command = _host_response_command(store, request, key="response-recovery", command_id="response-command")
    assert command.command["response"]["content"] == "Approved [credential redacted] at [external locator redacted]"
    resolution = store.resolve_controller_command(command)
    assert resolution.kind == "applied"
    assert resolution.receipt is not None and resolution.receipt.outbox_action == "recovery_dispatch"
    current = store.load_checkpoint("response-recovery")
    assert current is not None and current.status.value == "running"
    envelope = _recovery_envelope(
        store,
        request,
        outcome,
        key="response-recovery",
        command_id=command.command_id,
    )
    consumed = store.claim_and_consume_host_interaction_response(envelope)
    assert consumed.kind == "applied"
    after = store.load_checkpoint("response-recovery")
    assert after is not None
    assert after.claimed_cycle == after.cycle_index + 1
    assert after.messages[-1].content == "Approved [credential redacted] at [external locator redacted]"
    revision = after.revision
    replay = store.claim_and_consume_host_interaction_response(envelope)
    assert replay.kind == "replayed"
    assert store.load_checkpoint("response-recovery").revision == revision


def test_host_record_reaper_requires_the_checkpoint_execution_owner(store: Any) -> None:
    checkpoint, request, outcome = _admit_host(store, "record-reaper-fence")
    command = _host_response_command(store, request, key=checkpoint.checkpoint_key, command_id="record-reaper-command")
    assert store.resolve_controller_command(command).kind == "applied"
    current = store.load_checkpoint(checkpoint.checkpoint_key)
    assert current is not None and current.claim_token is None
    with pytest.raises(CheckpointError) as barrier:
        store.claim_checkpoint(
            checkpoint.checkpoint_key,
            current.cycle_index + 1,
            claim_token="checkpoint-owner",
            lease_expires_at_ms=10,
            now_ms=1,
            claim_mode="recovery",
        )
    assert barrier.value.code == "host_interaction_recovery_required"
    claim = deepcopy(current)
    claim.claim_token = "checkpoint-owner"
    claim.claimed_cycle = current.cycle_index + 1
    claim.lease_expires_at_ms = 10
    claim.status = AgentStatus.RUNNING
    if isinstance(store, InMemoryCheckpointStore):
        store._store[checkpoint.checkpoint_key] = deepcopy(claim)  # type: ignore[attr-defined]
    elif isinstance(store, SqliteCheckpointStore):
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE checkpoints SET status = ?, claim_token = ?, claimed_cycle = ?, lease_expires_at_ms = ? "
            "WHERE checkpoint_key = ?",
            (
                AgentStatus.RUNNING.value,
                claim.claim_token,
                claim.claimed_cycle,
                claim.lease_expires_at_ms,
                checkpoint.checkpoint_key,
            ),
        )
        store._conn.commit()  # type: ignore[attr-defined]
    else:
        from vv_agent.runtime.stores.redis import _checkpoint_to_storage

        payload, lease = _checkpoint_to_storage(claim)
        data_key, lease_key = store._keys(checkpoint.checkpoint_key)  # type: ignore[attr-defined]
        store._client.set(data_key, payload)  # type: ignore[attr-defined]
        assert lease is not None
        store._client.set(lease_key, str(lease))  # type: ignore[attr-defined]
    if isinstance(store, InMemoryCheckpointStore):
        stored_checkpoint = store._store[checkpoint.checkpoint_key]  # type: ignore[attr-defined]
        stored_record = store._host_interaction_records[(checkpoint.checkpoint_key, request.interaction_id)]  # type: ignore[attr-defined]
        stored_record["state"] = "resolved_claimed"
        stored_record["claim_token"] = "different-owner"
        stored_record["lease_expires_at_ms"] = 10
        assert not store.reap_host_interaction_record(
            record_id=outcome.record_id, checkpoint_key=checkpoint.checkpoint_key, now_ms=11
        )
        assert stored_record["state"] == "resolved_claimed"
        stored_record["claim_token"] = stored_checkpoint.claim_token
    elif isinstance(store, SqliteCheckpointStore):
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE host_interaction_records SET state = 'resolved_claimed', claim_token = ?, lease_expires_at_ms = ? "
            "WHERE record_id = ? AND checkpoint_key = ?",
            ("different-owner", 10, outcome.record_id, checkpoint.checkpoint_key),
        )
        store._conn.commit()  # type: ignore[attr-defined]
        assert not store.reap_host_interaction_record(
            record_id=outcome.record_id, checkpoint_key=checkpoint.checkpoint_key, now_ms=11
        )
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE host_interaction_records SET claim_token = ? WHERE record_id = ? AND checkpoint_key = ?",
            (claim.claim_token, outcome.record_id, checkpoint.checkpoint_key),
        )
        store._conn.commit()  # type: ignore[attr-defined]
    else:
        from vv_agent.runtime.stores.redis import _host_record_from_storage, _host_record_to_storage

        record_key = store._host_record_key(checkpoint.checkpoint_key, request.interaction_id)  # type: ignore[attr-defined]
        raw = store._client.get(record_key)  # type: ignore[attr-defined]
        assert raw is not None
        stored_record = _host_record_from_storage(raw)
        stored_record.update(state="resolved_claimed", claim_token="different-owner", lease_expires_at_ms=10)
        store._client.set(record_key, _host_record_to_storage(stored_record))  # type: ignore[attr-defined]
        assert not store.reap_host_interaction_record(
            record_id=outcome.record_id, checkpoint_key=checkpoint.checkpoint_key, now_ms=11
        )
        stored_record["claim_token"] = claim.claim_token
        store._client.set(record_key, _host_record_to_storage(stored_record))  # type: ignore[attr-defined]
    assert store.reap_host_interaction_record(record_id=outcome.record_id, checkpoint_key=checkpoint.checkpoint_key, now_ms=11)


def test_host_response_envelope_is_closed_and_no_lease_default(store: Any) -> None:
    _checkpoint_value, request, outcome = _admit_host(store, "strict-envelope")
    command = _host_response_command(store, request, key="strict-envelope", command_id="strict-command")
    assert store.resolve_controller_command(command).kind == "applied"
    envelope = _recovery_envelope(
        store,
        request,
        outcome,
        key="strict-envelope",
        command_id=command.command_id,
    )
    with pytest.raises(CheckpointError):
        store.claim_and_consume_host_interaction_response({**envelope, "lease_expires_at_ms": 1})
    current = store.load_checkpoint("strict-envelope")
    assert current is not None and current.revision == envelope["expected_revision"]


def test_suspended_host_response_waits_until_resume_then_wakes(store: Any) -> None:
    _checkpoint_value, request, outcome = _admit_host(store, "suspended-host")
    current = store.load_checkpoint("suspended-host")
    assert current is not None
    handle = DistributedRunHandle("suspended-host", current.root_run_id, current.trace_id)
    suspend = ControllerCommand(
        command_id="suspend-host",
        handle=handle,
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "suspend"},
    )
    suspended = store.resolve_controller_command(suspend)
    assert suspended.kind == "applied" and suspended.receipt is not None
    assert suspended.receipt.outbox_action == "none"
    current = store.load_checkpoint("suspended-host")
    assert current is not None and current.status.value == "suspended"
    response = _host_response_command(store, request, key="suspended-host", command_id="suspended-response")
    admitted = store.resolve_controller_command(response)
    assert admitted.kind == "applied" and admitted.receipt is not None
    assert admitted.receipt.outbox_action == "none"
    current = store.load_checkpoint("suspended-host")
    assert current is not None and current.status.value == "suspended"
    resume = ControllerCommand(
        command_id="resume-host",
        handle=handle,
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "resume"},
    )
    resumed = store.resolve_controller_command(resume)
    assert resumed.kind == "applied" and resumed.receipt is not None
    assert resumed.receipt.outbox_action == "recovery_dispatch"
    current = store.load_checkpoint("suspended-host")
    assert current is not None and current.status.value == "running"
    consumed = store.claim_and_consume_host_interaction_response(
        _recovery_envelope(
            store,
            request,
            outcome,
            key="suspended-host",
            command_id="suspended-response",
        )
    )
    assert consumed.kind == "applied"


def test_controller_recovery_wake_has_owner_attempt_and_ambiguity_lifecycle(store: Any) -> None:
    _checkpoint_value, request, _outcome = _admit_host(store, "wake-lifecycle")
    command = _host_response_command(store, request, key="wake-lifecycle", command_id="wake-command")
    assert store.resolve_controller_command(command).kind == "applied"
    receipt = store.get_controller_command_receipt(command.command_id)
    assert receipt is not None and receipt.outbox_state == "pending"
    claimed = store.claim_controller_command_wake(
        command_id=command.command_id,
        command_digest=command.command_digest or "",
        claim_token="wake-owner-a",
        lease_expires_at_ms=10_000,
        now_ms=1,
    )
    assert claimed is not None and claimed["outbox_state"] == "claimed" and claimed["attempt"] == 1
    with pytest.raises(CheckpointError):
        store.complete_controller_command_wake(
            command_id=command.command_id,
            command_digest=command.command_digest or "",
            claim_token="stale-owner",
            attempt=1,
            outcome="delivered",
            now_ms=2,
        )
    ambiguous = store.complete_controller_command_wake(
        command_id=command.command_id,
        command_digest=command.command_digest or "",
        claim_token="wake-owner-a",
        attempt=1,
        outcome="ambiguous",
        now_ms=2,
    )
    assert ambiguous is not None and ambiguous["outbox_state"] == "ambiguous"
    assert store.reap_controller_command_wakes("wake-lifecycle", 20_000) == []
    retried = store.reconcile_controller_command_wake(
        command_id=command.command_id,
        command_digest=command.command_digest or "",
        outcome="retry",
        now_ms=3,
    )
    assert retried is not None and retried["outbox_state"] == "pending"
    claimed_again = store.claim_controller_command_wake(
        command_id=command.command_id,
        command_digest=command.command_digest or "",
        claim_token="wake-owner-b",
        lease_expires_at_ms=20_000,
        now_ms=4,
    )
    assert claimed_again is not None and claimed_again["attempt"] == 2


def test_controller_recovery_wake_reaper_finds_expired_outbox(store: Any) -> None:
    checkpoint, request, _outcome = _admit_host(store, "wake-reaper")
    command = _host_response_command(store, request, key=checkpoint.checkpoint_key, command_id="reaper-command")
    assert store.resolve_controller_command(command).kind == "applied"
    claimed = store.claim_controller_command_wake(
        command_id=command.command_id,
        command_digest=command.command_digest or "",
        claim_token="reaper-owner",
        lease_expires_at_ms=10,
        now_ms=1,
    )
    assert claimed is not None and claimed["outbox_state"] == "claimed"
    assert store.reap_controller_command_wakes("other-checkpoint", 11) == []
    rows = store.reap_controller_command_wakes(checkpoint.checkpoint_key, 11)
    assert len(rows) == 1
    assert rows[0]["command_id"] == command.command_id
    assert rows[0]["outbox_state"] == "pending"
    receipt = store.get_controller_command_receipt(command.command_id)
    assert receipt is not None and receipt.outbox_state == "pending"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_host_response_recovery_dual_worker_one_revision(tmp_path: Path, store_kind: str) -> None:
    store = InMemoryCheckpointStore() if store_kind == "memory" else SqliteCheckpointStore(tmp_path / "recovery-race.sqlite3")
    _checkpoint_value, request, outcome = _admit_host(store, f"recovery-race-{store_kind}")
    command = _host_response_command(store, request, key=f"recovery-race-{store_kind}", command_id="race-command")
    assert store.resolve_controller_command(command).kind == "applied"
    envelope = _recovery_envelope(
        store,
        request,
        outcome,
        key=f"recovery-race-{store_kind}",
        command_id=command.command_id,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(store.claim_and_consume_host_interaction_response, [envelope, envelope]))
    assert {result.kind for result in results} == {"applied", "replayed"}
    current = store.load_checkpoint(f"recovery-race-{store_kind}")
    assert current is not None and current.revision == envelope["expected_revision"] + 1


def test_notification_owner_attempt_cas_rejects_stale_completion(store: Any) -> None:
    _checkpoint_value, _request, outcome = _admit_host(store, "notification-race")
    claimed = store.claim_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="observer-1",
        lease_expires_at_ms=10_000,
        now_ms=1,
    )
    assert claimed is not None
    with pytest.raises(CheckpointError):
        store.complete_host_interaction_notification(
            notification_id=outcome.notification_id,
            payload_digest=outcome.notification_payload_digest,
            claim_token="stale-owner",
            attempt=int(claimed["attempt"]),
            outcome="delivered",
            now_ms=2,
        )
    completed = store.complete_host_interaction_notification(
        notification_id=outcome.notification_id,
        payload_digest=outcome.notification_payload_digest,
        claim_token="observer-1",
        attempt=int(claimed["attempt"]),
        outcome="delivered",
        now_ms=2,
    )
    assert completed is not None and completed["outbox_state"] == "delivered"


@pytest.mark.parametrize("kind", ["suspend", "cancel"])
def test_control_variants_are_admitted_by_all_stores(store: Any, kind: str) -> None:
    checkpoint, _request, outcome = _admit_host(store, f"control-{kind}")
    handle = DistributedRunHandle(checkpoint.checkpoint_key, checkpoint.root_run_id, checkpoint.trace_id)
    command = ControllerCommand(
        command_id=f"command-{kind}",
        handle=handle,
        resume_attempt=1,
        expected_revision=outcome.checkpoint_revision,
        command={"kind": kind},
    )
    resolution = store.resolve_controller_command(command)
    assert resolution.kind == "applied"
    assert resolution.receipt is not None
    assert resolution.receipt.resulting_status in {"suspended", "failed"}
    stored = store.get_controller_command(command.command_id)
    assert stored is not None
    assert stored.command_digest == command.command_digest


def test_redis_controller_admission_keeps_live_claim_with_fast_process_clock() -> None:
    store = _redis_store()
    key = f"redis-live-claim-{uuid4().hex}"
    process_now_ms = time.time_ns() // 1_000_000
    redis_now_ms = process_now_ms - 300_000
    store._client.server_now_ms = redis_now_ms  # type: ignore[attr-defined]
    assert store.create_checkpoint(_checkpoint(key))
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="live-claim",
        lease_expires_at_ms=redis_now_ms + 60_000,
        now_ms=redis_now_ms,
        claim_mode="continue",
    )
    assert claimed is not None
    current = store.load_checkpoint(key)
    assert current is not None
    command = ControllerCommand(
        command_id="live-claim-command",
        handle=DistributedRunHandle(key, current.root_run_id, current.trace_id),
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "suspend"},
    )
    with pytest.raises(CheckpointError) as error:
        store.admit_controller_command(command)
    assert error.value.code == "controller_command_claim_active"
    current = store.load_checkpoint(key)
    assert current is not None and current.claim_token == "live-claim"
    store.delete_checkpoint(key)


def test_redis_controller_admission_recovers_expired_claim_from_redis_time() -> None:
    store = _redis_store()
    key = f"redis-expired-claim-{uuid4().hex}"
    process_now_ms = time.time_ns() // 1_000_000
    redis_now_ms = process_now_ms + 300_000
    store._client.server_now_ms = redis_now_ms  # type: ignore[attr-defined]
    assert store.create_checkpoint(_checkpoint(key))
    claimed = store.claim_checkpoint(
        key,
        1,
        claim_token="expired-claim",
        lease_expires_at_ms=redis_now_ms - 1,
        now_ms=redis_now_ms - 60_000,
        claim_mode="continue",
    )
    assert claimed is not None
    current = store.load_checkpoint(key)
    assert current is not None
    command = ControllerCommand(
        command_id="expired-claim-command",
        handle=DistributedRunHandle(key, current.root_run_id, current.trace_id),
        resume_attempt=current.resume_attempt,
        expected_revision=current.revision,
        command={"kind": "cancel"},
    )
    receipt = store.admit_controller_command(command)
    assert receipt.resulting_status == "failed"
    current = store.load_checkpoint(key)
    assert current is not None and current.claim_token is None and current.terminal_result is not None
    store.delete_checkpoint(key)


def test_real_redis_controller_cas_replay_and_notification_ambiguity() -> None:
    redis_url = os.environ.get("VV_AGENT_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("VV_AGENT_TEST_REDIS_URL is required for the live Redis integration test")
    store = RedisCheckpointStore(redis_url)
    key = f"real-redis-controller-{uuid4().hex}"
    checkpoint = _checkpoint(key)
    assert store.create_checkpoint(checkpoint)
    try:
        claimed = store.claim_checkpoint(
            key,
            1,
            claim_token="real-worker-claim",
            lease_expires_at_ms=10_000,
            now_ms=1,
            claim_mode="continue",
        )
        assert claimed is not None
        request = HostInteractionRequest(
            interaction_id="real-interaction",
            logical_cycle=1,
            operation_id="real-operation",
            tool_call_id="real-tool",
            prompt="Approve the live Redis operation",
        )

        def produce() -> Any:
            return store.produce_host_interaction(
                request,
                admission_context=HostInteractionAdmissionContext(
                    checkpoint_key=key,
                    claim_token="real-worker-claim",
                    expected_revision=claimed.revision,
                    claimed_cycle=1,
                    now_ms=1,
                    lease_expires_at_ms=10_000,
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _index: produce(), range(2)))
        assert {outcome.status for outcome in outcomes} == {"admitted", "replayed"}
        notification = outcomes[0]
        expected_record_key = (
            "vv-agent:host-interaction:" + hashlib.sha256(f"{key}\x00{request.interaction_id}".encode()).hexdigest()
        )
        assert store._host_record_key(key, request.interaction_id) == expected_record_key  # type: ignore[attr-defined]
        assert store._client.get(expected_record_key) is not None  # type: ignore[attr-defined]
        expected_notification_key = (
            "vv-agent:host-interaction-notification:" + hashlib.sha256(notification.notification_id.encode("utf-8")).hexdigest()
        )
        assert store._host_notification_key(notification.notification_id) == expected_notification_key  # type: ignore[attr-defined]
        assert store._client.get(expected_notification_key) is not None  # type: ignore[attr-defined]
        # A second Python process/client can decode the same network record;
        # the key assertion above is the cross-language canonical vector.
        peer = RedisCheckpointStore(redis_url)
        peer_checkpoint = peer.load_checkpoint(key)
        assert peer_checkpoint is not None and peer_checkpoint.revision == outcomes[0].checkpoint_revision
        current = store.load_checkpoint(key)
        assert current is not None
        assert current.revision == outcomes[0].checkpoint_revision

        command = ControllerCommand(
            command_id="real-controller-command",
            handle=DistributedRunHandle(key, checkpoint.root_run_id, checkpoint.trace_id),
            resume_attempt=1,
            expected_revision=outcomes[0].checkpoint_revision,
            command={
                "kind": "host_interaction_response",
                "interaction_id": request.interaction_id,
                "logical_cycle": request.logical_cycle,
                "operation_id": request.operation_id,
                "tool_call_id": request.tool_call_id,
                "request_digest": request.request_digest,
                "response": {"role": "user", "content": "approved"},
            },
        )

        def resolve() -> Any:
            return store.resolve_controller_command(command)

        with ThreadPoolExecutor(max_workers=2) as executor:
            resolutions = list(executor.map(lambda _index: resolve(), range(2)))
        assert {resolution.kind for resolution in resolutions} == {"applied", "replayed"}
        receipt = store.get_controller_command_receipt(command.command_id)
        assert receipt is not None
        assert store.get_controller_command(command.command_id) is not None
        expected_controller_set = "vv-agent:controller-commands-by-checkpoint:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert store._controller_receipt_set_key(key) == expected_controller_set  # type: ignore[attr-defined]
        assert store._client.smembers(expected_controller_set)  # type: ignore[attr-defined]
        current = store.load_checkpoint(key)
        assert current is not None
        recovery = HostInteractionRecoveryEnvelope(
            record_id=notification.record_id,
            checkpoint_key=key,
            run_id=current.root_run_id,
            trace_id=current.trace_id,
            claim_mode="recovery",
            resume_attempt=current.resume_attempt,
            expected_revision=current.revision,
            logical_cycle=request.logical_cycle,
            interaction_id=request.interaction_id,
            operation_id=request.operation_id,
            tool_call_id=request.tool_call_id,
            request_digest=request.request_digest or "",
            command_id=command.command_id,
        )
        consumed = store.claim_and_consume_host_interaction_response(recovery.to_dict())
        assert consumed.kind == "applied"
        replayed_consumed = store.claim_and_consume_host_interaction_response(recovery.to_dict())
        assert replayed_consumed.kind == "replayed"
        claimed_wake = store.claim_controller_command_wake(
            command_id=command.command_id,
            command_digest=command.command_digest or "",
            claim_token="real-wake-owner",
            lease_expires_at_ms=10_000,
            now_ms=4,
        )
        assert claimed_wake is not None and claimed_wake["attempt"] == 1
        completed_wake = store.complete_controller_command_wake(
            command_id=command.command_id,
            command_digest=command.command_digest or "",
            claim_token="real-wake-owner",
            attempt=1,
            outcome="delivered",
            now_ms=5,
        )
        assert completed_wake is not None and completed_wake["outbox_state"] == "delivered"

        claimed_notification = store.claim_host_interaction_notification(
            notification_id=notification.notification_id,
            payload_digest=notification.notification_payload_digest,
            claim_token="real-observer",
            lease_expires_at_ms=10_000,
            now_ms=2,
        )
        assert claimed_notification is not None
        ambiguous = store.complete_host_interaction_notification(
            notification_id=notification.notification_id,
            payload_digest=notification.notification_payload_digest,
            claim_token="real-observer",
            attempt=int(claimed_notification["attempt"]),
            outcome="ambiguous",
            now_ms=3,
            error="connection lost after delivery",
        )
        assert ambiguous is not None and ambiguous["outbox_state"] == "ambiguous"
        with pytest.raises(CheckpointError):
            store.claim_host_interaction_notification(
                notification_id=notification.notification_id,
                payload_digest=notification.notification_payload_digest,
                claim_token="blind-retry",
                lease_expires_at_ms=20_000,
                now_ms=4,
            )
        retried = store.reconcile_host_interaction_notification(
            notification_id=notification.notification_id,
            payload_digest=notification.notification_payload_digest,
            outcome="retry",
            now_ms=4,
        )
        assert retried is not None and retried["outbox_state"] == "pending"
    finally:
        store.delete_checkpoint(key)
