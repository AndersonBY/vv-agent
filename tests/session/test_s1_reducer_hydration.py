from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._retained_budget import RetainedBudgetHydrationRequired, retained_budget_inputs
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver


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


def _row_body(store: SQLiteStore, seq: int) -> bytes:
    row = store._one(
        "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        ("default", "s", seq),
    )
    assert row is not None
    body = row[0]
    assert isinstance(body, bytes)
    return body
