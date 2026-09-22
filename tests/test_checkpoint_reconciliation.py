"""Real recovery producers and receipt CAS, using the locked canonical seeds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from test_checkpoint import _journal_case, _minimal_checkpoint, _store

from vv_agent.checkpoint import (
    CheckpointConfig,
    CheckpointError,
    EventCursor,
    OperationState,
    ReconciliationDecision,
    ReconciliationDecisionKind,
    ReconciliationError,
    ResumePolicy,
)
from vv_agent.events import ReconciliationResolvedEvent, RunEvent
from vv_agent.runtime.checkpoint_codec import checkpoint_to_dict
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
from vv_agent.runtime.state import Checkpoint, EventOutboxEntry, OperationJournalEntry
from vv_agent.types import ToolExecutionResult, ToolResultStatus


class _Crash(BaseException):
    pass


class _ProbeStore:
    """Inject failure at the real store call, never in the recovery algorithm."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.fault: str | None = None
        self.commits: list[tuple[Checkpoint, Checkpoint]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _commit(self, method: str, checkpoint: Checkpoint, **kwargs: Any) -> bool:
        resolving = any(
            row.state == "pending" and row.event["type"] == "reconciliation_resolved" for row in checkpoint.event_outbox
        )
        fault = self.fault if resolving else None
        if fault is not None:
            self.fault = None
        before = self.inner.load_checkpoint(checkpoint.checkpoint_key)
        assert before is not None
        if fault == "before":
            raise _Crash("before resolution CAS")
        if fault == "claim_lost":
            now = (before.lease_expires_at_ms or 0) + 1
            self.inner.claim_checkpoint(
                before.checkpoint_key,
                1,
                claim_token="replacement-owner",
                now_ms=now,
                lease_expires_at_ms=now + 10_000,
                claim_mode="recovery",
            )
        written = getattr(self.inner, method)(checkpoint, **kwargs)
        if written and resolving:
            after = self.inner.load_checkpoint(checkpoint.checkpoint_key)
            assert after is not None
            self.commits.append((before, after))
            if fault == "after":
                raise _Crash("after resolution CAS")
        return written

    def progress_checkpoint(self, checkpoint: Checkpoint, **kwargs: Any) -> bool:
        return self._commit("progress_checkpoint", checkpoint, **kwargs)

    def record_tool_receipt(self, checkpoint: Checkpoint, **kwargs: Any) -> bool:
        return self._commit("record_tool_receipt", checkpoint, **kwargs)


class _Provider:
    def __init__(self, decision: ReconciliationDecision) -> None:
        self.decision = decision
        self.calls = 0

    def reconcile(self, observation: Any) -> ReconciliationDecision:
        self.calls += 1
        return self.decision


class _Events:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any], EventCursor]] = {}
        self.fail_after_append = False

    def append(self, event: RunEvent) -> None:
        raise AssertionError("checkpoint delivery must use append_once")

    def replay(self, *args: Any, **kwargs: Any) -> Iterator[RunEvent]:
        return iter(())

    def append_once(self, event_id: str, payload_digest: str, event: RunEvent) -> EventCursor:
        existing = self.rows.get(event_id)
        if existing is not None:
            assert existing[:2] == (payload_digest, event.to_dict())
            return existing[2]
        cursor = EventCursor(
            store_ref={"id": "test.reconciliation-events", "version": "1"},
            value={"sequence": len(self.rows) + 1},
            last_event_id=event_id,
        )
        self.rows[event_id] = (payload_digest, event.to_dict(), cursor)
        if self.fail_after_append and event.type == "reconciliation_resolved":
            self.fail_after_append = False
            raise _Crash("after append before delivery acknowledgement")
        return cursor


def _seed(store: _ProbeStore) -> Checkpoint:
    seed = _minimal_checkpoint(key="reconciliation-atomic")
    assert store.create_checkpoint(seed)
    claimed = store.claim_checkpoint(
        seed.checkpoint_key,
        1,
        claim_token="seed-owner",
        now_ms=100,
        lease_expires_at_ms=200,
        claim_mode="continue",
    )
    assert claimed is not None
    entry = OperationJournalEntry.from_dict(_journal_case("tool_started"))
    entry.cycle_index = 1
    entry.state = OperationState.AMBIGUOUS
    claimed.tool_journal = [entry]
    assert store.progress_checkpoint(claimed, claim_token="seed-owner", expected_revision=claimed.revision)
    return seed


@contextmanager
def _controller(
    store: _ProbeStore,
    provider: _Provider,
    events: _Events | None = None,
) -> Iterator[CheckpointResumeController]:
    before = store.load_checkpoint("reconciliation-atomic")
    assert before is not None
    now = (before.lease_expires_at_ms or 0) + 1
    token = f"recovery-{before.resume_attempt}"
    claimed = store.claim_checkpoint(
        before.checkpoint_key,
        1,
        claim_token=token,
        now_ms=now,
        lease_expires_at_ms=now + 10_000,
        claim_mode="recovery",
    )
    assert claimed is not None
    controller = CheckpointResumeController(
        config=CheckpointConfig(store=store, key=claimed.checkpoint_key, resume_policy=ResumePolicy.REQUIRE_EXISTING),
        task_id=claimed.task_id,
        run_id=claimed.root_run_id,
        trace_id=claimed.trace_id,
        run_definition=claimed.run_definition,
        run_definition_digest=claimed.run_definition_digest,
        initial_messages=[],
        initial_shared_state={},
        initial_budget_usage=None,
        extensions=[],
        reconciliation_provider=provider,
        event_sink=lambda _: None,
        event_store=events,
    )
    # An already-claimed worker isolates recovery from the wall-clock heartbeat.
    # The real store owns every revision, claim, receipt, and delivery transition.
    controller.checkpoint = claimed
    controller._owned_claim_token = token
    controller._active_claim_mode = "recovery"
    with patch.object(controller, "_now_ms", return_value=now):
        try:
            controller._deliver_pending_outbox()
            yield controller
        finally:
            controller.close()


def _decision(kind: str, entry: OperationJournalEntry) -> ReconciliationDecision:
    if kind == "retry":
        return ReconciliationDecision(ReconciliationDecisionKind.RETRY)
    if kind == "replay_success":
        return ReconciliationDecision(
            ReconciliationDecisionKind.REPLAY_SUCCESS,
            result=ToolExecutionResult(tool_call_id=entry.tool_call_id or "", content="retained result").to_dict(),
        )
    return ReconciliationDecision(
        ReconciliationDecisionKind.RECORD_FAILURE,
        error=ReconciliationError(code="tool_unavailable", message="still unavailable", retryable=True),
    )


def _audits(checkpoint: Checkpoint) -> list[EventOutboxEntry]:
    return [row for row in checkpoint.event_outbox if row.event["type"] == "reconciliation_resolved"]


@pytest.fixture(params=["memory", "sqlite", "redis"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> _ProbeStore:
    # The existing "redis" fixture is an in-process fake; no Redis server is used.
    result = _ProbeStore(_store(request.param, tmp_path, "reconciliation"))
    _seed(result)
    return result


@pytest.mark.parametrize("resolution", ["record_failure", "replay_success"])
def test_retry_then_ambiguous_resolution_is_atomic(store: _ProbeStore, resolution: str) -> None:
    provider = _Provider(ReconciliationDecision(ReconciliationDecisionKind.RETRY))
    with _controller(store, provider) as controller:
        controller._recover_ambiguous_operations()
        checkpoint = controller._require_checkpoint()
        entry = checkpoint.tool_journal[0]
        assert entry.attempt == 2 and entry.state is OperationState.PLANNED
        assert _audits(checkpoint)[0].event_id == controller._stable_event_id(
            "reconciliation_resolved",
            entry.operation_id,
            "1",
            "retry",
        )
        entry.state = OperationState.AMBIGUOUS
        controller._progress()
        decision = _decision(resolution, entry)

    with _controller(store, _Provider(decision)) as controller:
        controller._recover_ambiguous_operations()
        checkpoint = controller._require_checkpoint()
        entry = checkpoint.tool_journal[0]
        audits = _audits(checkpoint)
        assert [row.event["decision"] for row in audits] == ["retry", resolution]
        assert len({row.event_id for row in audits}) == 2
        assert audits[-1].event_id == controller._stable_event_id(
            "reconciliation_resolved",
            entry.operation_id,
            "2",
            resolution,
        )
        assert entry.attempt == 2
        assert entry.state is (OperationState.FAILED if resolution == "record_failure" else OperationState.SUCCEEDED)
        assert entry.result is not None and entry.result_digest is not None
        if resolution == "record_failure":
            assert entry.result["metadata"] == {"retryable": True}
            assert entry.error is not None and entry.error.retryable

    assert len(store.commits) == 2
    for before, after in store.commits:
        assert after.revision == before.revision + 1
        assert after.claim_token == before.claim_token
        assert len(_audits(after)) == len(_audits(before)) + 1
        assert _audits(after)[-1].state == "pending"


@pytest.mark.parametrize("resolution", ["record_failure", "replay_success"])
@pytest.mark.parametrize("delivered", [False, True])
def test_retained_destination_attempt_retry_keeps_original_bytes(
    store: _ProbeStore,
    resolution: str,
    delivered: bool,
) -> None:
    with _controller(store, _Provider(ReconciliationDecision(ReconciliationDecisionKind.RETRY))) as controller:
        checkpoint = controller._require_checkpoint()
        entry = checkpoint.tool_journal[0]
        entry.attempt = 2
        old = ReconciliationResolvedEvent(
            run_id=checkpoint.root_run_id,
            trace_id=checkpoint.trace_id,
            cycle_index=1,
            checkpoint_key=checkpoint.checkpoint_key,
            operation_id=entry.operation_id,
            operation_kind=entry.kind,
            decision="retry",
            created_at=123.0,
            event_id=controller._stable_event_id("reconciliation_resolved", entry.operation_id, "2"),
        )
        controller._queue_outbox_event(checkpoint, old)
        controller._progress()
        if delivered:
            controller._deliver_pending_outbox()
        retained = deepcopy(_audits(checkpoint)[0])
        decision = _decision(resolution, entry)

    with _controller(store, _Provider(decision)) as controller:
        controller._recover_ambiguous_operations()
        old_row, new_row = _audits(controller._require_checkpoint())
        assert (old_row.event_id, old_row.event, old_row.payload_digest) == (
            retained.event_id,
            retained.event,
            retained.payload_digest,
        )
        assert old_row.event_id != new_row.event_id
        assert new_row.event["decision"] == resolution


@pytest.mark.parametrize("resolution", ["retry", "record_failure", "replay_success"])
@pytest.mark.parametrize("fault", ["before", "after", "append"])
def test_reconciliation_crash_recovery_retains_one_decision(
    store: _ProbeStore,
    resolution: str,
    fault: str,
) -> None:
    checkpoint = store.load_checkpoint("reconciliation-atomic")
    provider = _Provider(_decision(resolution, checkpoint.tool_journal[0]))
    events = _Events()
    store.fault = fault if fault != "append" else None
    events.fail_after_append = fault == "append"
    with pytest.raises(_Crash), _controller(store, provider, events) as controller:
        controller._recover_ambiguous_operations()
    crashed = store.load_checkpoint("reconciliation-atomic")
    audits = deepcopy(_audits(crashed))
    if fault == "before":
        assert audits == []
        assert crashed.tool_journal[0].state is OperationState.AMBIGUOUS
        assert crashed.tool_journal[0].attempt == 1
        assert crashed.tool_journal[0].result is None
    else:
        assert len(audits) == 1 and audits[0].state == "pending"
        assert crashed.tool_journal[0].state is not OperationState.AMBIGUOUS
    with _controller(store, provider, events) as controller:
        controller._recover_ambiguous_operations()
        resumed = controller._require_checkpoint()
        assert len(_audits(resumed)) == 1
        assert _audits(resumed)[0].state == "delivered"
        if audits:
            assert _audits(resumed)[0].event == audits[0].event
            assert _audits(resumed)[0].payload_digest == audits[0].payload_digest
    assert provider.calls == (2 if fault == "before" else 1)
    assert sum(row[1]["type"] == "reconciliation_resolved" for row in events.rows.values()) == 1
    assert len(store.commits) == 1


@pytest.mark.parametrize("resolution", ["retry", "record_failure", "replay_success"])
def test_lost_claim_cannot_commit_resolution_or_audit(store: _ProbeStore, resolution: str) -> None:
    checkpoint = store.load_checkpoint("reconciliation-atomic")
    provider = _Provider(_decision(resolution, checkpoint.tool_journal[0]))
    store.fault = "claim_lost"
    with pytest.raises(CheckpointError) as error, _controller(store, provider) as controller:
        controller._recover_ambiguous_operations()
    assert error.value.code in {"checkpoint_store_conflict", "checkpoint_claim_conflict"}
    retained = store.load_checkpoint("reconciliation-atomic")
    assert retained.claim_token == "replacement-owner"
    assert retained.tool_journal[0].state is OperationState.AMBIGUOUS
    assert retained.tool_journal[0].attempt == 1
    assert retained.tool_journal[0].result is None
    assert _audits(retained) == []
    assert store.commits == []


def test_reconciled_receipt_still_rejects_different_result(store: _ProbeStore) -> None:
    before = store.load_checkpoint("reconciliation-atomic")
    provider = _Provider(_decision("replay_success", before.tool_journal[0]))
    with _controller(store, provider) as controller:
        stale = deepcopy(controller._require_checkpoint())
        controller._recover_ambiguous_operations()
    retained = store.load_checkpoint("reconciliation-atomic")
    expected = checkpoint_to_dict(retained)
    entry = retained.tool_journal[0]
    result = ToolExecutionResult.from_dict(entry.result)

    def write(value: ToolExecutionResult) -> bool:
        return store.record_tool_receipt(
            stale,
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            tool_call_id=entry.tool_call_id,
            request_digest=entry.request_digest,
            result=value,
            claim_token=stale.claim_token,
            expected_revision=stale.revision,
            claimed_cycle=1,
        )

    assert write(result)
    assert checkpoint_to_dict(store.load_checkpoint(retained.checkpoint_key)) == expected
    with pytest.raises(CheckpointError) as error:
        write(replace(result, content="different result"))
    assert error.value.code == "tool_receipt_conflict"
    assert checkpoint_to_dict(store.load_checkpoint(retained.checkpoint_key)) == expected


def test_resolved_event_timestamp_replay_and_real_payload_conflict(store: _ProbeStore) -> None:
    with _controller(store, _Provider(ReconciliationDecision(ReconciliationDecisionKind.RETRY))) as controller:
        controller._recover_ambiguous_operations()
        checkpoint = controller._require_checkpoint()
        original = deepcopy(_audits(checkpoint)[0])
        from vv_agent.events import event_from_dict

        replay = event_from_dict({**original.event, "created_at": original.event["created_at"] + 100})
        controller._queue_outbox_event(checkpoint, replay)
        assert _audits(checkpoint)[0] == original
        changed = event_from_dict({**original.event, "metadata": {"changed": True}})
        with pytest.raises(CheckpointError) as error:
            controller._queue_outbox_event(checkpoint, changed)
        assert error.value.code == "event_identity_conflict"
        assert _audits(checkpoint)[0] == original


def test_nondefinitive_reconciliation_cannot_commit_a_resolved_audit(store: _ProbeStore) -> None:
    checkpoint = store.load_checkpoint("reconciliation-atomic")
    provider = _Provider(
        ReconciliationDecision(
            ReconciliationDecisionKind.REPLAY_SUCCESS,
            result=ToolExecutionResult(
                tool_call_id=checkpoint.tool_journal[0].tool_call_id,
                content="unknown",
                status_code=ToolResultStatus.ERROR,
                error_code="tool_execution_failed",
            ).to_dict(),
        )
    )
    with pytest.raises(ValueError), _controller(store, provider) as controller:
        controller._recover_ambiguous_operations()
    retained = store.load_checkpoint("reconciliation-atomic")
    assert retained.tool_journal[0].state is OperationState.AMBIGUOUS
    assert _audits(retained) == []
