from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _boundary_hydration as boundary_hydration
from vv_agent.kernel import _compaction_phase, _driver_recovery, _summary_requests
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._boundary_hydration import (
    SessionHydration,
    fold_hydrated_records,
    hydrate_session_record_payload,
    hydrate_session_records,
    hydrate_session_stored_record,
)
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import KeyUnavailable, MissingContentLink
from vv_agent.kernel._reducer import Fold, TransitionError
from vv_agent.kernel._retained_budget import RetainedBudgetHydrationRequired, retained_budget_inputs
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import Lease, StoredRecord
from vv_agent.llm.errors import ModelRequestRejected
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.session import _recovery_phase
from vv_agent.session import kernel as kernel_module
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_reader import record_owner
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.records import InboxItem, Record, digest, record_identity
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import record as fixture_record

_CONTENT_BODIES = {
    "input.payload.content": b'{"text":"logical"}',
    "session.attributes": b'{"seed":{"messages":[],"shared_state":{}}}',
    "session.principal.attributes": b"{}",
    "session.workspace": b'"/workspace"',
}


@function_tool
def retry_probe_tool(text: str) -> str:
    return text


def _bind_fixture_content(reference: dict, body: bytes) -> None:
    reference["mac"] = sha256(body).hexdigest()
    reference["size"] = len(body)


def _logical_input_fixture():
    input_payload = fixture_record("input_applied")._payload["input"]
    _bind_fixture_content(input_payload["payload"]["content"], _CONTENT_BODIES["input.payload.content"])
    return fixture_record("input_applied", input=input_payload, input_digest=digest(input_payload))


def _created_fixture():
    created = fixture_record("session_created")
    payload = created._payload
    _bind_fixture_content(payload["principal"]["attributes"], _CONTENT_BODIES["session.principal.attributes"])
    _bind_fixture_content(payload["workspace"], _CONTENT_BODIES["session.workspace"])
    _bind_fixture_content(payload["attributes"], _CONTENT_BODIES["session.attributes"])
    return fixture_record(
        "session_created",
        principal=payload["principal"],
        workspace=payload["workspace"],
        parent_session_id=payload["parent_session_id"],
        parent_operation_id=payload["parent_operation_id"],
        attributes=payload["attributes"],
    )


def test_cold_driver_hydrates_seed_definition_and_budget_without_rewriting_references(tmp_path: Path) -> None:
    path = tmp_path / "cold-s1.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    limits = RunBudgetLimits(max_total_tokens=100, max_tool_calls=2)
    with open_store(path, allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create(
            "s",
            "/workspace",
            {
                "seed": {
                    "unseen_input_protection": True,
                    "messages": [{"role": "system", "content": "Seed context."}],
                    "shared_state": {},
                }
            },
        )
        sessions.push("s", InboxItem("input", "user", {"content": "Use the tool."}))
        agent = Agent("s1", "Work carefully.", model="scripted")
        runtime = sessions.runtime(
            agent,
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
                budget_limits=limits,
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="cold-s1", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        logical_state = driver._logical_state()
        assert logical_state._protection is not None
        assert logical_state.turns[turn_id].start._payload["budget"] == limits.to_dict()
        assert logical_state.turns[turn_id].start._payload["definition"]["task"]["initial_messages"] == [
            {"role": "system", "content": "Seed context."}
        ]
        active_budget = retained_budget_inputs(logical_state, turn_id)
        assert active_budget is not None and active_budget.limits == limits
        evaluator = driver.live_budget()
        assert evaluator is not None and evaluator.limits == limits
        persisted_bodies = [_row_body(store, row.seq) for row in driver.records]
        assert sessions.store.release(lease)

    with open_store(path, allow_unsigned=True) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.active_turn_id == turn_id
        with pytest.raises(RetainedBudgetHydrationRequired, match="requires hydrated logical state"):
            retained_budget_inputs(raw_state, turn_id)
        assert [_row_body(reopened, row.seq) for row in raw_records] == persisted_bodies

        sessions = SessionDriver(store=reopened, content_key_provider=provider)
        runtime = sessions.runtime(
            Agent("s1", "Work carefully.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
                budget_limits=limits,
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="cold-s1-reopen", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))

        logical_state = driver._logical_state()
        start = logical_state.turns[turn_id].start
        assert start._payload["budget"] == limits.to_dict()
        assert start._payload["definition"]["task"]["user_prompt"] == "Use the tool."
        retained = retained_budget_inputs(logical_state, turn_id)
        assert retained is not None
        assert retained.limits == limits
        assert logical_state._protection is not None
        evaluator = driver.live_budget()
        assert evaluator is not None and evaluator.limits == limits

        turn_started = next(row.record for row in driver.records if row.record.kind == "turn_started")
        ContentRef.from_dict(turn_started._payload["budget"])
        ContentRef.from_dict(turn_started._payload["definition"])
        created = next(row.record for row in driver.records if row.record.kind == "session_created")
        ContentRef.from_dict(created._payload["attributes"])
        assert all(b"Seed context." not in body for body in persisted_bodies)
        assert sessions.store.release(lease)


def test_logical_state_rehydrates_after_content_key_destroy(tmp_path: Path) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "destroyed.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)

        logical_state = driver._logical_state()
        turn_id = logical_state.active_turn_id
        assert turn_id is not None
        assert logical_state.turns[turn_id].start._payload["definition"]["task"]["user_prompt"] == "go"
        definition = ContentRef.from_dict(driver.state.turns[turn_id].start._payload["definition"])

        assert provider.destroy(store.tenant_id, definition.key_id).accepted
        with pytest.raises(KeyUnavailable):
            driver._logical_state()
        assert driver.state is driver.fold.state
        assert sessions.store.release(lease)


def test_step_scopes_logical_state_and_closes_with_retained_budget(tmp_path: Path) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "step-scope.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, _lease = _started_driver(sessions, store)
        scope = driver.scope
        scope.start()
        try:
            assert scope.ready.wait(5)
            assert scope.confirmed_poll().lease is scope.lease
            turn_id = driver.state.active_turn_id
            assert turn_id is not None
            assert driver.task().user_prompt == "go"
            driver.live_budget()
            assert driver.state is driver.fold.state

            refreshed_after_completion = False
            for _ in range(8):
                if driver.state.active_turn_id is None:
                    break
                assert driver.step()
                assert driver.state is driver.fold.state
                started = next(row.record for row in driver.records if row.record.kind == "turn_started")
                ContentRef.from_dict(started._payload["definition"])
                ContentRef.from_dict(started._payload["budget"])
                if not refreshed_after_completion and any(row.record.kind == "op_completed" for row in driver.records):
                    driver.refresh()
                    assert driver.state is driver.fold.state
                    refreshed_after_completion = True

            assert driver.state.active_turn_id is None
            assert any(row.record.kind == "turn_ended" for row in driver.records)
            assert refreshed_after_completion
        finally:
            scope.close()
            assert sessions.store.release(scope.lease)


def test_logical_scope_exit_does_not_rehydrate_discarded_commit_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    calls = 0
    hydrate_and_fold_records = kernel_module.hydrate_and_fold_records

    def counted_hydration(*args, **kwargs):
        nonlocal calls
        calls += 1
        return hydrate_and_fold_records(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "hydrate_and_fold_records", counted_hydration)
    with open_store(tmp_path / "scope-commit.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        boundary = make_record_before_publication(
            "boundary_recorded",
            session_id="s",
            turn_id=driver.state.active_turn_id,
            payload={
                "stage": "before_memory",
                "boundary_id": "scope-commit",
                "source_operation_id": None,
                "source_digest": digest([message.to_dict() for message in driver.transcript()]),
                "data": {"messages": [], "shared_state": {}},
            },
        )

        with driver._logical_state_scope():
            assert calls == 1
            driver.commit([boundary])
            assert calls == 2

        assert calls == 2
        assert driver.state is driver.fold.state
        assert sessions.store.release(lease)


def test_nested_logical_scope_reuses_the_outer_validated_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    calls = 0
    hydrate_and_fold_records = kernel_module.hydrate_and_fold_records

    def counted_hydration(*args, **kwargs):
        nonlocal calls
        calls += 1
        return hydrate_and_fold_records(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "hydrate_and_fold_records", counted_hydration)
    with open_store(tmp_path / "nested-scope.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)

        with driver._logical_state_scope():
            assert calls == 1
            with driver._logical_state_scope():
                assert calls == 1
            assert calls == 1

        assert calls == 1
        assert driver.state is driver.fold.state
        assert sessions.store.release(lease)


def test_logical_scope_refresh_invalidates_stale_fold_and_recovers_after_hydration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "scope-refresh.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        other = _Driver(sessions.store, "s", driver.runtime, driver.scope)
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        boundary = make_record_before_publication(
            "boundary_recorded",
            session_id="s",
            turn_id=turn_id,
            payload={
                "stage": "before_memory",
                "boundary_id": "scope-refresh",
                "source_operation_id": None,
                "source_digest": digest([message.to_dict() for message in driver.transcript()]),
                "data": {"messages": [], "shared_state": {}},
            },
        )
        stale_records = tuple(driver.records)
        other.commit([boundary])
        assert tuple(driver.records) == stale_records
        added = other.records[len(stale_records) :]
        assert [row.record.kind for row in added] == ["usage_observed", "boundary_recorded"]
        active_usage, committed_boundary = added
        assert active_usage.commit_id == committed_boundary.commit_id
        assert active_usage.record.turn_id == committed_boundary.record.turn_id == turn_id
        assert committed_boundary.record.record_id == boundary.record_id
        assert active_usage.writer_epoch == lease.epoch
        ContentRef.from_dict(active_usage.record._payload["usage"])
        ContentRef.from_dict(active_usage.record._payload["source"])
        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        usage_payload = hydrate_session_record_payload(active_usage, hydration)
        meter_id = f"active/{turn_id}/{lease.epoch}"
        assert set(usage_payload) == {"meter_id", "observation", "mode", "source", "usage"}
        assert usage_payload["meter_id"] == meter_id
        assert usage_payload["observation"] == 1
        assert usage_payload["mode"] == "cumulative"
        assert usage_payload["source"] == "kernel-active-scope"
        assert set(usage_payload["usage"]) == {"elapsed_ms", "host_cost"}
        assert usage_payload["usage"]["host_cost"] is None
        assert type(usage_payload["usage"]["elapsed_ms"]) is int
        assert usage_payload["usage"]["elapsed_ms"] >= 0

        original_hydration = kernel_module.hydrate_and_fold_records
        fail_hydration = False

        def controlled_hydration(*args, **kwargs):
            if fail_hydration:
                raise MissingContentLink("injected refresh hydration failure")
            return original_hydration(*args, **kwargs)

        monkeypatch.setattr(kernel_module, "hydrate_and_fold_records", controlled_hydration)
        with driver._logical_state_scope():
            initial_fold = driver._active_logical_fold
            assert initial_fold is not None
            fail_hydration = True
            driver.refresh()
            assert driver._logical_dirty
            assert driver._active_logical_fold is None
            with pytest.raises(MissingContentLink, match="injected"):
                _ = driver.state
            assert driver._logical_dirty
            assert driver._active_logical_fold is None

            fail_hydration = False
            refreshed = driver.state
            assert (turn_id, "before_memory", "scope-refresh") in refreshed.boundaries
            assert refreshed.usage_observations == {meter_id: 1}
            assert driver._active_logical_fold is not None
            assert driver._active_logical_fold is not initial_fold

        assert driver.state is driver.fold.state
        assert sessions.store.release(lease)


def test_logical_retry_payloads_skip_unrelated_attempt_one_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "retry-lazy.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        driver.records = tuple(
            StoredRecord(
                record=fixture_record("op_planned", oid=f"op-{index}"),
                seq=index,
                commit_id="retry-lazy",
                writer_epoch=1,
                created_ms=index,
            )
            for index in range(1, 21)
        )

        def unexpected_hydration(*_args, **_kwargs):
            raise AssertionError("unique first attempts do not need retry hydration")

        monkeypatch.setattr(kernel_module, "hydrate_stored_record", unexpected_hydration)
        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        assert driver._logical_retry_plan_payloads(hydration) == {}
        assert sessions.store.release(lease)


def test_retry_plan_hydrates_its_own_request_before_rejecting_changed_identity(tmp_path: Path) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "retry-tamper.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        request = {"id": "call", "name": "retry_probe_tool", "arguments": {"text": "one"}}
        plan = _tool_plan_record(driver, request, attempt=1, tool=None, budget_admission={})
        driver.commit([plan])
        started = driver.record(
            "op_started",
            {
                "dispatch_id": "call/1",
                "authorization_version": "1",
                "epoch": lease.epoch,
                "mode": "sync",
            },
            plan,
        )
        driver.commit([started])
        unknown = driver.record(
            "op_unknown",
            {
                "reason": {"kind": "worker_lost"},
                "dispatch_evidence": {"kind": "tool"},
                "observation": {"attempt": 1},
                "retry": "retry",
                "retry_at_ms": None,
                "duplicate_cost_risk": True,
                "measurement_missing": True,
            },
            plan,
        )
        driver.commit([unknown])
        before = store.read_state("s")[1]
        tampered = _tool_plan_record(
            driver,
            request | {"arguments": {"text": "tampered"}},
            attempt=2,
            tool=None,
            budget_admission={},
        )

        with driver._logical_state_scope(), pytest.raises(TransitionError, match="retry changed request"):
            driver.commit([tampered])

        assert store.read_state("s")[1] == before
        assert not any(row.record.record_id == tampered.record_id for row in store.read_state("s")[1])
        assert sessions.store.release(lease)


def test_retry_plan_preserves_current_prepared_hook_parameters(tmp_path: Path) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "retry-prepared.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_tool_driver(sessions, store)
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        definition = driver.hydrated_definition(driver.state.turns[turn_id].start)
        tool_name = "retry_probe_tool"
        capability = definition.capabilities.get(tool_name, {})
        request = {"id": "call", "name": tool_name, "arguments": {"text": "one"}}
        plan = _tool_plan_record(driver, request, attempt=1, tool=None, budget_admission={})
        driver.commit([plan])

        hook_result = {"patched": True}
        shared_state = {"prepared": True}
        prepared = driver.record(
            "op_prepared",
            {
                "request": request,
                "request_digest": digest(request),
                "op_kind": "tool",
                "tool": capability,
                "provider_binding": None,
                "idempotency_key": "call",
                "hook_result": hook_result,
                "shared_state": shared_state,
                "policy_decision": None,
            },
            plan,
        )
        driver.commit([prepared])
        started = driver.record(
            "op_started",
            {
                "dispatch_id": "call/1",
                "authorization_version": "1",
                "epoch": lease.epoch,
                "mode": "sync",
            },
            plan,
        )
        driver.commit([started])
        unknown = driver.record(
            "op_unknown",
            {
                "reason": {"kind": "worker_lost"},
                "dispatch_evidence": {"kind": "tool"},
                "observation": {"attempt": 1},
                "retry": "retry",
                "retry_at_ms": None,
                "duplicate_cost_risk": True,
                "measurement_missing": True,
            },
            plan,
        )
        driver.commit([unknown])

        retry = _tool_plan_record(
            driver,
            request,
            attempt=2,
            tool=capability,
            budget_admission={"hook_result": hook_result, "shared_state": shared_state},
        )
        driver.commit([retry])
        logical = driver._logical_state()
        retry_plan = logical.operations["tool"].attempts[2].plan
        assert retry_plan._payload["request"] == request
        assert retry_plan._payload["tool"] == capability
        assert retry_plan._payload["budget_admission"] == {
            "hook_result": hook_result,
            "shared_state": shared_state,
        }
        assert sessions.store.release(lease)


def test_logical_preflight_base_folds_only_pending_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "preflight-delta.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        boundary = make_record_before_publication(
            "boundary_recorded",
            session_id="s",
            turn_id=driver.state.active_turn_id,
            payload={
                "stage": "before_memory",
                "boundary_id": "preflight-delta",
                "source_operation_id": None,
                "source_digest": digest([message.to_dict() for message in driver.transcript()]),
                "data": {"messages": [], "shared_state": {}},
            },
        )

        def unexpected_history_hydration(*_args, **_kwargs):
            raise AssertionError("preflight reused its validated logical prefix")

        monkeypatch.setattr(boundary_hydration, "hydrate_session_records", unexpected_history_hydration)
        with driver._logical_state_scope():
            driver.commit([boundary])
        assert sessions.store.release(lease)


def test_input_applied_payload_hydrates_once_from_its_record_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "input-owner.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions, store)
        applied = next(row for row in driver.records if row.record.kind == "input_applied")
        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        calls: list[tuple[str, str]] = []
        read = hydration.reader.read

        def counted_read(tenant_id, session_id, owner, slot, ref):
            calls.append((owner, slot))
            return read(tenant_id, session_id, owner, slot, ref)

        monkeypatch.setattr(hydration.reader, "read", counted_read)
        payload = hydrate_session_record_payload(applied, hydration)

        owner = record_owner(applied.seq)
        assert payload["input"]["payload"]["content"] == "go"
        assert calls.count((owner, "input.payload.content")) == 1
        assert all(call_owner == owner for call_owner, _slot in calls)
        assert sessions.store.release(lease)


def test_fold_reuses_partial_input_hydration_once_and_keeps_raw_identity() -> None:
    raw = _logical_input_fixture()
    created = StoredRecord(_created_fixture(), 1, "commit", 1, 1)
    source = StoredRecord(raw, 2, "commit", 1, 2)
    hydration = _CountingHydration()
    hydrated = hydrate_session_records((source,), hydration)
    captured = {}

    class CapturingFold(Fold):
        def __init__(self) -> None:
            super().__init__(
                normalize_memory_settings=lambda settings, _model: settings,
                count_messages=lambda messages, *, model: 0,
            )

        def extend(self, records, *, consumed_inputs=(), bodies=None) -> ExecutionState:
            captured["records"] = list(records)
            record = captured["records"][1]
            captured["record"] = record
            captured["consumed"] = next(iter(consumed_inputs))
            assert bodies is not None
            captured["bodies"] = list(bodies)
            return ExecutionState(applied_inputs={"input": record})

    state = fold_hydrated_records(hydrated, (created, source), CapturingFold, hydration)

    owner = record_owner(source.seq)
    assert hydration.reader.calls.count((owner, "input.payload.content")) == 1
    assert captured["record"]._payload["input"]["payload"]["content"] == raw._payload["input"]["payload"]["content"]
    assert captured["record"]._business_payload["input"]["payload"]["content"] == {"text": "logical"}
    assert captured["consumed"].payload["content"] == {"text": "logical"}
    assert captured["bodies"] == [created.record.encode(), raw.encode()]
    assert state.applied_inputs["input"]._payload["input"]["payload"]["content"] == {"text": "logical"}


def test_fold_rejects_hydrated_records_with_mismatched_identity_or_body() -> None:
    source = StoredRecord(_logical_input_fixture(), 1, "commit", 1, 1)
    other = StoredRecord(fixture_record("turn_started"), 2, "commit", 1, 2)
    hydration = _CountingHydration()
    hydrated = hydrate_session_records((source,), hydration)[0]

    def unused_fold_factory():
        raise AssertionError("fold must not start for mismatched hydration")

    with pytest.raises(MissingContentLink, match="hydrated record identity"):
        fold_hydrated_records((replace(hydrated, seq=source.seq + 1),), (source,), unused_fold_factory, hydration)

    tampered_record = replace(hydrated.record)
    object.__setattr__(tampered_record, "_body", other.record.encode())
    with pytest.raises(MissingContentLink, match="hydrated record body"):
        fold_hydrated_records((replace(hydrated, record=tampered_record),), (source,), unused_fold_factory, hydration)


def test_finish_summary_reuses_hydrated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    source_digest = digest([])
    plan = SimpleNamespace(
        _payload={
            "purpose": "compaction",
            "request": {"metadata": {"mode": "summary", "source_digest": source_digest, "keep_recent": 0}},
        }
    )
    attempt = SimpleNamespace(plan=plan, result=object(), context="normal")
    operation = SimpleNamespace(turn_id="turn", kind="model", selected_attempt=1, attempts={1: attempt})
    driver = SimpleNamespace(
        state=SimpleNamespace(
            active_turn_id="turn",
            turns={"turn": SimpleNamespace(start=SimpleNamespace(_payload={"generation": 1}))},
            operations={"summary": operation},
            compactions=[],
        )
    )
    manager = SimpleNamespace(
        protected_start=0,
        recovery_tool_available=True,
        effective_context_window=10_000,
        settings={"language": "en", "summary_event_limit": 100, "model": "scripted"},
    )
    hydration_calls = 0
    accepted_text = []

    def hydrated_result(_driver, _record):
        nonlocal hydration_calls
        hydration_calls += 1
        return {"content": "summary body", "error_code": None}

    def accept_summary(_plan, text, **_kwargs):
        accepted_text.append(text)
        if False:
            yield None
        return SimpleNamespace(accepted=False, messages=[])

    monkeypatch.setattr(_compaction_phase, "_hydrated_completed_result", hydrated_result)
    monkeypatch.setattr(_compaction_phase._summary_compaction, "plan_summary", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(_compaction_phase._summary_compaction, "accept_summary", accept_summary)

    steps = _compaction_phase.finish_summary(driver)
    assert next(steps) is _compaction_phase.CompactionRequest.INBOX
    assert steps.send(()) is _compaction_phase.CompactionRequest.TRANSCRIPT
    assert steps.send(()) is _compaction_phase.CompactionRequest.MEMORY
    with pytest.raises(StopIteration) as stopped:
        steps.send(manager)

    assert stopped.value.value is False
    assert hydration_calls == 1
    assert accepted_text == ["summary body"]


class _CountingReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        del tenant_id, session_id, ref
        self.calls.append((owner, slot))
        body = _CONTENT_BODIES.get(slot)
        if body is None:
            raise AssertionError(f"unexpected hydration slot: {slot}")
        return body


class _CountingHydration(SessionHydration):
    reader: _CountingReader

    def __init__(self) -> None:
        super().__init__(_CountingReader(), tenant_id="default", session_id="s")


def _started_driver(sessions: SessionDriver, store: SQLiteStore) -> tuple[_Driver, Lease]:
    sessions.create(
        "s",
        "/workspace",
        {"seed": {"messages": [{"role": "system", "content": "Seed context."}], "shared_state": {}}},
    )
    sessions.push("s", InboxItem("input", "user", {"content": "go"}))
    runtime = sessions.runtime(
        Agent("s1", "Work carefully.", model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("done")]),
            workspace="/workspace",
        ),
        session_id="s",
    )
    lease = sessions.store.acquire("s", owner="s1-lifecycle", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))
    return driver, lease


def _started_tool_driver(sessions: SessionDriver, store: SQLiteStore) -> tuple[_Driver, Lease]:
    sessions.create(
        "s",
        "/workspace",
        {"seed": {"messages": [{"role": "system", "content": "Seed context."}], "shared_state": {}}},
    )
    sessions.push("s", InboxItem("input", "user", {"content": "go"}))
    runtime = sessions.runtime(
        Agent("s1", "Work carefully.", model="scripted", tools=[retry_probe_tool]),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("done")]),
            workspace="/workspace",
        ),
        session_id="s",
    )
    lease = sessions.store.acquire("s", owner="s1-tool-retry", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))
    return driver, lease


def _tool_plan_record(
    driver: _Driver,
    request: dict[str, Any],
    *,
    attempt: int,
    tool: dict[str, Any] | None,
    budget_admission: dict[str, Any],
) -> Record:
    turn_id = driver.state.active_turn_id
    assert turn_id is not None
    return make_record_before_publication(
        "op_planned",
        session_id="s",
        turn_id=turn_id,
        operation_id="tool",
        attempt=attempt,
        payload={
            "op_kind": "tool",
            "purpose": None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "v1",
            "dependencies": [],
            "tool": tool,
            "idempotency_key": "call",
            "budget_admission": budget_admission,
            "not_before_ms": None,
            "provider_binding": None,
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )


def _completed_tool_result(driver: _Driver, lease: Lease) -> tuple[Record, dict[str, Any]]:
    request = {"id": "call", "name": "retry_probe_tool", "arguments": {"text": "one"}}
    plan = _tool_plan_record(driver, request, attempt=1, tool=None, budget_admission={})
    driver.commit([plan])
    started = driver.record(
        "op_started",
        {
            "dispatch_id": "call/1",
            "authorization_version": "1",
            "epoch": lease.epoch,
            "mode": "sync",
        },
        plan,
    )
    driver.commit([started])
    attempt = driver.state.operations["tool"].attempts[1]
    admitted = attempt.execution_plan._payload
    result = {"content": "done", "items": [1]}
    completed = driver.record(
        "op_completed",
        {
            "result": result,
            "result_digest": digest(result),
            "usage": {},
            "shared_state": None,
            "evidence": ["receipt"],
            "execution_started": True,
            "context": "normal",
            "request_digest": admitted["request_digest"],
            "provider_binding": admitted["provider_binding"],
            "policy_decision": None,
        },
        plan,
    )
    driver.commit([completed])
    stored = next(row for row in driver.records if row.record.kind == "op_completed" and row.record.operation_id == "tool")
    return stored.record, result


def test_active_logical_scope_reuses_verified_completed_result_without_rereading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / "completed-result-scope.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_tool_driver(sessions, store)
        raw_completed, expected = _completed_tool_result(driver, lease)

        with driver._logical_state_scope():
            logical_completed = driver.state.operations["tool"].attempts[1].result
            assert logical_completed is not None
            read_calls: list[tuple[str, str]] = []
            unwrap_calls = 0
            original_read = driver.read_content_json
            original_unwrap = provider.unwrap

            def counted_read(owner: str, slot: str, ref: Any) -> Any:
                read_calls.append((owner, slot))
                return original_read(owner, slot, ref)

            def counted_unwrap(tenant_id: str, session_id: str, wrapped_key):
                nonlocal unwrap_calls
                unwrap_calls += 1
                return original_unwrap(tenant_id, session_id, wrapped_key)

            monkeypatch.setattr(driver, "read_content_json", counted_read)
            monkeypatch.setattr(provider, "unwrap", counted_unwrap)
            first = _driver_recovery._result(driver, logical_completed)
            second = _summary_requests._completed_result(driver, logical_completed)

            assert first == second == expected
            assert read_calls == []
            assert unwrap_calls == 0
            first["items"].append(2)
            assert second == expected
            assert logical_completed._payload["result"] == expected

            forged = replace(
                logical_completed,
                payload=logical_completed._payload | {"result": {"content": "forged"}},
            )
            assert _driver_recovery._result(driver, forged) == expected
            assert read_calls == [(record_owner(driver.stored_record(raw_completed).seq), "result")]

        assert driver._active_logical_fold is None
        assert sessions.store.release(lease)


@pytest.mark.parametrize("damage", ["key", "link"])
def test_completed_result_reuse_does_not_cross_independent_logical_scopes(tmp_path: Path, damage: str) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(tmp_path / f"completed-result-{damage}.sqlite", allow_unsigned=True) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_tool_driver(sessions, store)
        raw_completed, expected = _completed_tool_result(driver, lease)

        with driver._logical_state_scope():
            logical_completed = driver.state.operations["tool"].attempts[1].result
            assert logical_completed is not None
            assert _driver_recovery._result(driver, logical_completed) == expected

        if damage == "key":
            ref = ContentRef.from_dict(raw_completed._payload["result"])
            assert provider.destroy(store.tenant_id, ref.key_id).accepted
            expected_error = KeyUnavailable
        else:
            seq = driver.stored_record(raw_completed).seq
            with store.atomic():
                store._rows(
                    "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=%s AND slot='result'",
                    (store.tenant_id, "s", seq),
                )
            expected_error = MissingContentLink

        with pytest.raises(expected_error), driver._logical_state_scope():
            pass
        assert sessions.store.release(lease)


def _row_body(store: SQLiteStore, seq: int) -> bytes:
    row = store._one(
        "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        ("default", "s", seq),
    )
    assert row is not None
    body = row[0]
    assert isinstance(body, bytes)
    return body


def test_hydrated_turn_record_retains_bytes_and_caches_one_logical_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with open_store(tmp_path / "record-cache.sqlite", allow_unsigned=True) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
        driver, lease = _started_driver(sessions, store)
        stored = next(row for row in driver.records if row.record.kind == "turn_started")
        retained_body = stored.record.encode()
        hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id="s")
        decode_calls = 0
        original_decode = AgentTask.from_dict

        def counted_decode(cls, value):
            nonlocal decode_calls
            decode_calls += 1
            return original_decode(value)

        monkeypatch.setattr(AgentTask, "from_dict", classmethod(counted_decode))
        hydrated = hydrate_session_stored_record(stored, hydration).record
        first = hydrated.task()
        second = hydrated.task()

        assert decode_calls == 1
        assert hydrated.encode() == retained_body
        assert hydrated.to_dict()["payload"]["definition"]["task"]["user_prompt"] == "go"
        assert first.user_prompt == second.user_prompt == "go"
        assert first.initial_shared_state == second.initial_shared_state == {}
        first.initial_shared_state["isolated"] = True
        first.metadata["isolated"] = True
        assert second.initial_shared_state == {}
        assert "isolated" not in second.metadata
        store.release(lease)


def test_semantic_preflight_rejects_retained_endpoint_tamper_before_dispatch(tmp_path: Path) -> None:
    with open_store(tmp_path / "endpoint-preflight.sqlite", allow_unsigned=True) as store:
        store.install_schema()
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
        driver, lease = _started_driver(sessions, store)
        request = {"metadata": {"vv_session": {"endpoint_order": ["a", "b"]}}}
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        payload = {
            "op_kind": "model",
            "purpose": "primary",
            "request": request,
            "request_digest": digest(request),
            "context_version": str(driver.records[-1].seq),
            "dependencies": [],
            "tool": None,
            "idempotency_key": "s/o",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "model",
            "consumed_unknowns": [],
            "policy_decision": None,
        }
        plan = Record(
            record_identity("op_planned", "s", turn_id, "o", 1, payload),
            "op_planned",
            "s",
            turn_id,
            "o",
            1,
            payload,
        )
        driver.commit([plan])
        before = store.read_state("s")[1]
        started = driver.record(
            "op_started",
            {
                "dispatch_id": "tamper",
                "authorization_version": "1",
                "epoch": lease.epoch,
                "mode": "sync",
                "endpoint_id": "b",
            },
            plan,
        )

        with driver._logical_state_scope(), pytest.raises(TransitionError, match="endpoint"):
            driver.commit([started])

        assert store.read_state("s")[1] == before
        assert not [row for row in before if row.record.kind == "op_started"]
        store.release(lease)


def test_rejected_http_retry_reuses_hydrated_request_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def create(*, endpoint_id, **_kwargs):
        def complete(**_payload):
            calls.append(endpoint_id)
            if len(calls) == 1:
                raise ModelRequestRejected(403, "Forbidden")
            return [
                SimpleNamespace(
                    content="done",
                    tool_calls=[],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )
            ]

        return SimpleNamespace(create_completion=complete)

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", create)
    monkeypatch.setattr(VvLlmClient, "_sleep_backoff", lambda *_args, **_kwargs: None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with (
        open_store(tmp_path / "endpoint-retry.sqlite", allow_unsigned=True) as store,
        closing(SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))) as sessions,
    ):
        sessions.create("s", str(workspace), {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "go"}))
        client = VvLlmClient(
            [EndpointTarget(name, "unused", "https://example.invalid") for name in ("a", "b")],
            backend="openai",
            randomize_endpoints=False,
        )
        provider = ScriptedModelProvider(
            "openai",
            "scripted",
            client,
            context_length=8192,
            max_output_tokens=1024,
        )
        runtime = sessions.runtime(
            Agent("retry", "Retry the rejected request.", model="scripted"),
            RunConfig(model_provider=provider, workspace=workspace),
            session_id="s",
        )

        drive(sessions.store, "s", runtime=runtime)

        rows = sessions.store.read("s").records
        plans = [row.record for row in rows if row.record.kind == "op_planned"]
        starts = [row.record for row in rows if row.record.kind == "op_started"]
        terminal = next(row.record for row in rows if row.record.kind == "turn_ended")
        assert calls == ["a", "b"]
        assert [plan.attempt for plan in plans] == [1, 2]
        assert len({plan.payload["request_digest"] for plan in plans}) == 1
        assert [record.payload["endpoint_id"] for record in starts] == ["a", "b"]
        assert terminal.payload["reason"] is None
