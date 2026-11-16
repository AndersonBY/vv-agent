from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _compaction_phase
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._hydration import KeyUnavailable, MissingContentLink
from vv_agent.kernel._retained_budget import RetainedBudgetHydrationRequired, retained_budget_inputs
from vv_agent.kernel._store import StoredRecord
from vv_agent.session import _recovery_phase
from vv_agent.session import kernel as kernel_module
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_reader import record_owner
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.projection import (
    fold_hydrated_records,
    hydrate_record_payload,
    hydrate_records,
    hydration_for_session_store,
)
from vv_agent.session.records import InboxItem, digest
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .helpers import record as fixture_record


def test_cold_driver_hydrates_seed_definition_and_budget_without_rewriting_references(tmp_path: Path) -> None:
    path = tmp_path / "cold-s1.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    limits = RunBudgetLimits(max_total_tokens=100, max_tool_calls=2)
    with SQLiteStore.standalone(str(path)) as store:
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

    with SQLiteStore.standalone(str(path)) as reopened:
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
    with SQLiteStore.standalone(str(tmp_path / "destroyed.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions)

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
    with SQLiteStore.standalone(str(tmp_path / "step-scope.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions)
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
        assert sessions.store.release(lease)


def test_live_budget_reuses_one_logical_hydration_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    calls = 0
    fold_hydrated_records = kernel_module.fold_hydrated_records

    def counted_hydration(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fold_hydrated_records(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "fold_hydrated_records", counted_hydration)
    with SQLiteStore.standalone(str(tmp_path / "local-hydration.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions)

        driver.live_budget()
        assert calls == 1
        assert driver.state is driver.fold.state
        driver.refresh()
        driver.live_budget()
        assert calls == 2
        assert driver.state is driver.fold.state
        assert sessions.store.release(lease)


def test_input_applied_payload_hydrates_once_from_its_record_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LocalKeyProvider(b"k" * 32)
    with SQLiteStore.standalone(str(tmp_path / "input-owner.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        driver, lease = _started_driver(sessions)
        applied = next(row for row in driver.records if row.record.kind == "input_applied")
        hydration = hydration_for_session_store(store, "s")
        calls: list[tuple[str, str]] = []
        read = hydration.reader.read

        def counted_read(tenant_id, session_id, owner, slot, ref):
            calls.append((owner, slot))
            return read(tenant_id, session_id, owner, slot, ref)

        monkeypatch.setattr(hydration.reader, "read", counted_read)
        payload = hydrate_record_payload(applied, hydration)

        owner = record_owner(applied.seq)
        assert payload["input"]["payload"]["content"] == "go"
        assert calls.count((owner, "input.payload.content")) == 1
        assert all(call_owner == owner for call_owner, _slot in calls)
        assert sessions.store.release(lease)


def test_fold_reuses_partial_input_hydration_once_and_keeps_raw_identity() -> None:
    raw = fixture_record("input_applied")
    created = StoredRecord(fixture_record("session_created"), 1, "commit", 1, 1)
    source = StoredRecord(raw, 2, "commit", 1, 2)
    hydration = _CountingHydration()
    hydrated = hydrate_records((source,), hydration)
    captured = {}

    class Fold:
        def extend(self, records, *, consumed_inputs, bodies):
            captured["records"] = list(records)
            record = captured["records"][1]
            captured["record"] = record
            captured["consumed"] = next(iter(consumed_inputs))
            captured["bodies"] = list(bodies)
            return SimpleNamespace(applied_inputs={"input": record}, operations={})

    state = fold_hydrated_records(hydrated, (created, source), Fold, hydration)

    owner = record_owner(source.seq)
    assert hydration.calls.count((owner, "input.payload.content")) == 1
    assert captured["record"]._payload["input"]["payload"]["content"] == raw._payload["input"]["payload"]["content"]
    assert captured["record"]._business_payload["input"]["payload"]["content"] == {"text": "logical"}
    assert captured["consumed"].payload["content"] == {"text": "logical"}
    assert captured["bodies"] == [created.record.encode(), raw.encode()]
    assert state.applied_inputs["input"]._payload["input"]["payload"]["content"] == {"text": "logical"}


def test_fold_rejects_hydrated_records_with_mismatched_identity_or_body() -> None:
    source = StoredRecord(fixture_record("input_applied"), 1, "commit", 1, 1)
    other = StoredRecord(fixture_record("turn_started"), 2, "commit", 1, 2)
    hydration = _CountingHydration()
    hydrated = hydrate_records((source,), hydration)[0]

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


class _CountingHydration:
    session_id = "s"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def read(self, owner, slot, ref, **_kwargs):
        self.calls.append((owner, slot))
        if slot == "input.payload.content":
            return {"text": "logical"}
        return ref


def _started_driver(sessions: SessionDriver) -> tuple[_Driver, object]:
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
    driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=sessions.store._now()))
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))
    return driver, lease


def _row_body(store: SQLiteStore, seq: int) -> bytes:
    row = store._one(
        "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        ("default", "s", seq),
    )
    assert row is not None
    body = row[0]
    assert isinstance(body, bytes)
    return body
