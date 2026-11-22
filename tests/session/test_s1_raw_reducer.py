from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._budget_values import BudgetUsageSnapshot
from vv_agent.kernel._content_publication import declared_content_ref, make_record_before_publication
from vv_agent.kernel._driver_planning import effective_model
from vv_agent.kernel._hydration import DefinitionContent, KeyUnavailable
from vv_agent.kernel._record_codec import RecordError
from vv_agent.kernel._records import Record
from vv_agent.kernel._reducer import Fold, TransitionError, _operation
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn
from vv_agent.kernel._store import StoredRecord
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.records import InboxItem, digest
from vv_agent.session.reducer import fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import content_ref, record


class _ModelSelectionView:
    def __init__(self, state: ExecutionState) -> None:
        self._state = state

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def records(self) -> tuple[StoredRecord, ...]:
        return ()

    @property
    def fold(self) -> Fold:
        raise AssertionError("model selection must not inspect the fold")

    @property
    def sid(self) -> str:
        return "s"

    @property
    def tenant_id(self) -> str | None:
        raise AssertionError("model selection must not inspect the tenant")

    def hydrated_definition(self, record: Record) -> DefinitionContent:
        raise AssertionError(f"model selection must not hydrate {record.kind}")

    def hydrate_record_field(self, record: Record, slot: str, ref: object) -> None:
        raise AssertionError(f"model selection must not hydrate {record.kind}/{slot}")

    def hydrate_plan(self, record: Record) -> Record:
        raise AssertionError(f"model selection must not hydrate {record.kind}")

    def stored_record(self, record: Record) -> StoredRecord:
        raise AssertionError(f"model selection must not load {record.kind}")

    def session_attributes(self) -> dict[str, object]:
        raise AssertionError("model selection must not inspect session attributes")


def _prepared_record(payload: dict) -> Record:
    return Record(
        "op/o/prepared",
        "op_prepared",
        "s",
        "t",
        "o",
        1,
        payload,
    )


def _prepared_state(plan: Record) -> ExecutionState:
    state = ExecutionState(session_id="s", active_turn_id="t")
    state.turns["t"] = Turn(record("turn_started"))
    state.operations["o"] = Operation("t", "tool", attempts={1: Attempt(plan)})
    return state


def test_raw_prepared_transition_allows_null_tool_and_rejects_mixed_modes() -> None:
    plan = record("op_planned")
    payload = {
        "request": content_ref("prepared-request"),
        "request_digest": plan._payload["request_digest"],
        "tool": None,
        "op_kind": "tool",
        "provider_binding": "function",
        "idempotency_key": None,
        "hook_result": None,
        "shared_state": content_ref("shared-state"),
        "policy_decision": None,
    }
    prepared = _prepared_record(payload)
    state = _prepared_state(plan)

    _operation(state, prepared)

    assert state.operations["o"].attempts[1].prepared is prepared

    mixed = _prepared_record(payload | {"tool": {"idempotency": "supported"}})
    with pytest.raises(TransitionError, match="mixes retained and hydrated content"):
        _operation(_prepared_state(plan), mixed)


def test_logical_model_selection_objects_are_not_treated_as_raw_refs() -> None:
    selection = {"model": "B", "model_settings": {"max_tokens": 128}, "route": {"backend_type": "anthropic"}}
    binding = {"backend": "anthropic", "model": "B"}
    state = ExecutionState(
        session_id="s",
        model_selection=Record(
            "selection",
            "input_applied",
            "s",
            None,
            None,
            None,
            {
                "input": {"input_id": "select", "payload": {"selection": selection}},
                "model_binding": binding,
            },
        ),
    )

    steps = effective_model(_ModelSelectionView(state))
    with pytest.raises(StopIteration) as stopped:
        next(steps)

    assert stopped.value.value == selection | {"binding": binding}


def test_raw_model_selection_requires_logical_hydration() -> None:
    state = ExecutionState(
        session_id="s",
        model_selection=Record(
            "selection",
            "input_applied",
            "s",
            None,
            None,
            None,
            {
                "input": {"input_id": "select", "payload": {"selection": content_ref("selection")}},
                "model_binding": content_ref("binding"),
            },
        ),
    )

    with pytest.raises(KeyUnavailable, match="requires hydrated logical state"):
        next(effective_model(_ModelSelectionView(state)))


def test_malformed_content_and_replayed_operation_identity_fail_closed() -> None:
    created = record("session_created")
    started_turn = record("turn_started")
    plan = record("op_planned")
    first_dispatch = record("op_started")
    conflicting_replay = record("op_started", dispatch_id="other")
    with pytest.raises(TransitionError, match="record identity conflict"):
        fold([created, started_turn, plan, first_dispatch, conflicting_replay])

    malformed_request = content_ref("prepared-request") | {"mac": "not-a-hash"}
    malformed_prepared = Record(
        "op/o/prepared",
        "op_prepared",
        "s",
        "t",
        "o",
        1,
        {
            "request": malformed_request,
            "request_digest": plan._payload["request_digest"],
            "tool": content_ref("prepared-tool"),
            "op_kind": "tool",
            "provider_binding": "function",
            "idempotency_key": None,
            "hook_result": None,
            "shared_state": content_ref("shared-state"),
            "policy_decision": None,
        },
    )
    with pytest.raises(RecordError):
        fold([created, started_turn, plan, malformed_prepared])

    compacted = record("context_compacted")
    malformed_compaction = replace(
        compacted,
        payload=compacted._payload | {"replacement": [content_ref("compacted") | {"mac": "not-a-hash"}]},
    )
    with pytest.raises((RecordError, TransitionError)):
        fold([created, started_turn, malformed_compaction])


def test_raw_compacted_cold_read_and_logical_reopen_preserve_transition(tmp_path: Path) -> None:
    import sys

    tests_root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, tests_root)
    try:
        from sim.scenarios import run_scenario
        from sim.world import SimWorld
    finally:
        sys.path.remove(tests_root)

    path = tmp_path / "raw-compacted.sqlite"
    with SimWorld(path, seed=17) as world:
        result = run_scenario(world, "context_compacted")
        assert result["scenario"] == "context_compacted"
        signer = world.ledger_signer

    assert signer is not None
    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(signer)
        raw_state, raw_records, _ = reopened.read_state("s")
        compacted = [row.record for row in raw_records if row.record.kind == "context_compacted"]
        assert len(compacted) == 1
        assert raw_state.compactions == compacted
        assert declared_content_ref(compacted[0]._payload["evidence_manifest"]) is not None
        assert all(declared_content_ref(value) is not None for value in compacted[0]._payload["replacement"])

    with SimWorld(path, seed=17) as reopened_world:
        logical_state, logical_records, _ = reopened_world.read_state("s")
        logical_compacted = [row.record for row in logical_records if row.record.kind == "context_compacted"]
        assert [record.record_id for record in logical_state.compactions] == [record.record_id for record in logical_compacted]
        assert logical_state.compactions[0]._payload["mode"] == "summary"
        assert "role" in logical_state.compactions[0]._payload["replacement"][0]


def test_raw_wait_user_cold_read_defers_retained_definition_to_logical_fold(tmp_path: Path) -> None:
    path = tmp_path / "raw-wait-user.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    question = "Which deployment should run?"
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("initial", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", [LLMResponse(question)]),
                workspace="/workspace",
                no_tool_policy="wait_user",
            ),
            session_id="s",
        )
        assert drive(sessions.store, "s", runtime=runtime)
        parked = next(row.record for row in store.read("s").records if row.record.kind == "turn_parked")
        assert declared_content_ref(parked._payload["question"]) is not None
        assert all(question.encode() not in row.record.encode() for row in store.read("s").records)
        turn_id = parked.turn_id
        assert turn_id is not None

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        retained_turn = next(row.record for row in raw_records if row.record.kind == "turn_started")
        assert declared_content_ref(retained_turn._payload["definition"]) is not None
        assert raw_state.turns[turn_id].wait is not None

        sessions = SessionDriver(store=reopened, content_key_provider=provider)
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
                no_tool_policy="wait_user",
            ),
            session_id="s",
        )
        lease = reopened.acquire("s", owner="raw-wait-reopen", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(reopened, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))
        logical = driver._logical_state()
        assert logical.turns[turn_id].start._payload["definition"]["task"]["no_tool_policy"] == "wait_user"
        assert logical.turns[turn_id].wait is not None
        assert reopened.release(lease)


def test_raw_operations_endpoint_and_budget_boundary_survive_sqlite_reopen(tmp_path: Path) -> None:
    path = tmp_path / "raw-reducer.sqlite"
    provider = LocalKeyProvider(b"k" * 32)

    @function_tool
    def tool() -> str:
        return "ok"

    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create(
            "s",
            "/workspace",
            {"seed": {"messages": [{"role": "system", "content": "seed"}], "shared_state": {}}},
        )
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted", tools=[tool]),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-reducer", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None

        model_request = {
            "model": "scripted",
            "route": {},
            "messages": [{"role": "user", "content": "go"}],
            "tools": [],
            "metadata": {"vv_session": {"endpoint_order": ["a", "b"], "model_binding": "scripted"}},
            "prompt_bundle": {},
            "model_settings": {},
        }
        model_plan = _plan_record(driver, "model", "t/model/primary/1", model_request, provider_binding="model")
        driver.commit([model_plan])
        model_started = driver.record(
            "op_started",
            {
                "dispatch_id": "model/1",
                "authorization_version": "1",
                "epoch": driver.scope.lease.epoch,
                "mode": "sync",
                "endpoint_id": "a",
            },
            model_plan,
        )
        driver.commit([model_started])

        tool_request = {"id": "call-1", "name": "tool", "arguments": {}}
        tool_plan = _plan_record(driver, "tool", "tool", tool_request, provider_binding="function")
        driver.commit([tool_plan])
        logical = driver._logical_state()
        definition = logical.turns[turn_id].start._payload["definition"]
        assert any(schema["function"]["name"] == "tool" for schema in definition["tools"])
        tool_capability = definition["capabilities"].get("tool", {})
        prepared = driver.record(
            "op_prepared",
            {
                "request": tool_request,
                "request_digest": digest(tool_request),
                "tool": tool_capability,
                "op_kind": "tool",
                "provider_binding": "function",
                "idempotency_key": None,
                "hook_result": None,
                "shared_state": {},
            },
            tool_plan,
        )
        driver.commit([prepared])
        tool_started = driver.record(
            "op_started",
            {
                "dispatch_id": "tool/1",
                "authorization_version": "1",
                "epoch": driver.scope.lease.epoch,
                "mode": "provider",
            },
            tool_plan,
        )
        driver.commit([tool_started])
        boundary = driver.boundary_record(
            "budget",
            "raw-usage",
            {"tool_names": ["tool"], "usage": BudgetUsageSnapshot(tool_calls=1).to_dict(), "exhaustion": None},
        )
        driver.commit([boundary])

        hot = driver.state
        hot_model_attempt = hot.operations["t/model/primary/1"].attempts[1]
        assert hot_model_attempt.started
        assert hot_model_attempt.dispatch is not None
        assert hot_model_attempt.dispatch._payload["endpoint_id"] == "a"
        assert hot.operations["tool"].attempts[1].prepared is not None
        assert hot.operations["tool"].attempts[1].started
        assert (turn_id, "budget", "raw-usage") in hot.boundaries
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        raw_model_attempt = raw_state.operations["t/model/primary/1"].attempts[1]
        assert raw_model_attempt.started
        assert raw_model_attempt.dispatch is not None
        assert raw_model_attempt.dispatch._payload["endpoint_id"] == "a"
        assert raw_state.operations["tool"].attempts[1].prepared is not None
        assert raw_state.operations["tool"].attempts[1].started
        assert (turn_id, "budget", "raw-usage") in raw_state.boundaries
        assert any(row.record.kind == "op_prepared" for row in raw_records)


def test_raw_and_cold_memory_boundary_defers_event_wire_validation_to_hydration(tmp_path: Path) -> None:
    path = tmp_path / "raw-memory-boundary.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create(
            "s",
            "/workspace",
            {"seed": {"messages": [{"role": "system", "content": "seed"}], "shared_state": {}}},
        )
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-memory", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        event = {
            "version": "v6",
            "type": "memory_compact_started",
            "event_id": "memory/started",
            "run_id": turn_id,
            "trace_id": "trace",
            "created_at": 1,
            "session_id": "s",
            "message_count": 1,
            "configured_threshold": 1_024,
            "effective_threshold": 512,
            "microcompact_threshold": 486,
            "microcompact_target": 460,
            "candidate_count": 0,
            "estimated_reclaimable_tokens": 0,
            "model_context_window": 1_024,
            "model_max_output_tokens": 100,
            "reserved_output_tokens": 10,
            "reserved_output_source": "model_settings",
            "autocompact_buffer_tokens": 0,
            "trigger": "full_threshold",
        }
        boundary = driver.boundary_record("memory_started", "raw-memory", {"event": event})
        driver.commit([boundary])

        retained = driver.state.boundaries[(turn_id, "memory_started", "raw-memory")]
        assert declared_content_ref(retained._payload["data"]["event"]) is not None
        with driver._logical_state_scope():
            hydrated = driver.state.boundaries[(turn_id, "memory_started", "raw-memory")]
            assert hydrated._payload["data"]["event"] == event
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, _records, _ = reopened.read_state("s")
        retained = raw_state.boundaries[(turn_id, "memory_started", "raw-memory")]
        assert declared_content_ref(retained._payload["data"]["event"]) is not None

        sessions = SessionDriver(store=reopened, content_key_provider=provider)
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-memory-reopen", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))
        with driver._logical_state_scope():
            hydrated = driver.state.boundaries[(turn_id, "memory_started", "raw-memory")]
            assert hydrated._payload["data"]["event"] == event
        assert sessions.store.release(lease)


def test_model_selection_raw_pointer_hydrates_for_driver_planning(tmp_path: Path) -> None:
    path = tmp_path / "selection.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    selection = {"model": "scripted", "model_settings": {"max_tokens": 64}, "route": {}}
    requirements = SessionRequirements(level=2, handler_version="1", capabilities=())
    with open_store(path) as store:
        store.install_schema()
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        sessions = SessionDriver(store=store, content_key_provider=provider, worker_level=2)
        sessions._configure_static_worker(level=2, capabilities=(), requirements=requirements)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push(
            "s", InboxItem("select", "model_selection", {"selection": selection}, feature_level=2, requirements=requirements)
        )
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-selection", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        retained_item = sessions.store.peek_inbox("s")[0].item
        item = retained_item
        binding = {
            "backend": "scripted",
            "model": "scripted",
            "endpoints": [],
            "context_length": runtime.resolved.context_length,
            "max_output_tokens": runtime.resolved.max_output_tokens,
        }
        applied = driver.record(
            "input_applied",
            {
                "input": item.to_dict(),
                "input_digest": retained_item.digest,
                "input_proof": None,
                "disposition": "applied",
                "reason": None,
                "target_operation_id": None,
                "target_wait_id": None,
                "position": "between_actions",
                "model_binding": binding,
            },
            tid=None,
        )
        driver.commit([applied], (retained_item.input_id,), guarded=True)
        assert driver.state.model_selection is not None
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, _, _ = reopened.read_state("s")
        assert raw_state.model_selection is not None

        sessions = SessionDriver(store=reopened, content_key_provider=provider, worker_level=2)
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-selection-reopen", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))
        with driver._logical_state_scope():
            selected_record = driver.state.model_selection
            assert selected_record is not None
            assert selected_record._payload["input"]["payload"]["selection"] == selection
            assert selected_record._payload["model_binding"] == binding
            selected = driver.state.applied_inputs[selected_record._payload["input"]["input_id"]]._payload
            assert selected["input"]["payload"]["selection"] == selection
            steps = effective_model(driver)
            with pytest.raises(StopIteration) as stopped:
                next(steps)
            effective = stopped.value.value
        assert effective["model"] == "scripted"
        assert effective["model_settings"] == selection["model_settings"]
        assert effective["binding"]["model"] == "scripted"
        assert sessions.store.release(lease)


def test_commit_rejects_nonretryable_retry_before_append(tmp_path: Path) -> None:
    path = tmp_path / "retry.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    request = {
        "model": "scripted",
        "route": {},
        "messages": [{"role": "user", "content": "go"}],
        "tools": [],
        "metadata": {"vv_session": {"endpoint_order": ["a", "b"], "model_binding": "scripted"}},
        "prompt_bundle": {},
        "model_settings": {},
    }
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create(
            "s",
            "/workspace",
            {"seed": {"messages": [{"role": "system", "content": "seed"}], "shared_state": {}}},
        )
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-retry", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))

        oid = "model"
        plan = _plan_record(driver, "model", oid, request, provider_binding="model")
        driver.commit([plan])
        started = driver.record(
            "op_started",
            {
                "dispatch_id": "model/1",
                "authorization_version": "1",
                "epoch": driver.scope.lease.epoch,
                "mode": "sync",
                "endpoint_id": "a",
            },
            plan,
        )
        driver.commit([started])
        result = {"content": "invalid", "error_code": "invalid_output", "rejection": {"retryable": False}}
        stored_plan = driver.state.operations[oid].attempts[1].execution_plan
        completed = driver.record(
            "op_completed",
            {
                "result": result,
                "result_digest": digest(result),
                "usage": {},
                "shared_state": None,
                "evidence": {"kind": "model"},
                "execution_started": True,
                "context": "normal",
                "request_digest": stored_plan._payload["request_digest"],
                "provider_binding": stored_plan._payload["provider_binding"],
            },
            stored_plan,
        )
        driver.commit([completed])
        retry = _plan_record(
            driver,
            "model",
            oid,
            request,
            provider_binding="model",
            attempt=2,
            context_version=plan._payload["context_version"],
        )
        with pytest.raises(TransitionError, match="retry not authorized"):
            driver.commit([retry])
        assert 2 not in driver.state.operations[oid].attempts
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.operations[oid].attempts[1].state == "completed"
        assert 2 not in raw_state.operations[oid].attempts
        assert retry.record_id not in {row.record.record_id for row in raw_records}

        sessions = SessionDriver(store=reopened, content_key_provider=provider)
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-retry-reopen", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=reopened._now()))
        logical = driver._logical_state()
        assert 2 not in logical.operations[oid].attempts
        assert sessions.store.release(lease)


def test_commit_rejects_invalid_retained_wait_before_append(tmp_path: Path) -> None:
    path = tmp_path / "retained-wait.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    request = _model_request()
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-wait-preflight", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None

        oid = "model"
        plan = _start_model_attempt(driver, oid, request)
        completed = _complete_model(
            driver,
            oid,
            plan,
            {"content": "actual response", "tool_calls": []},
            {"kind": "model"},
        )
        driver.commit([completed])

        parked = driver.record(
            "turn_parked",
            {
                "interaction_id": "wait-invalid",
                "question": "unrelated question",
                "source_operation_id": oid,
                "source_attempt": 1,
            },
            tid=turn_id,
        )
        with pytest.raises(TransitionError, match="invalid user wait source"):
            driver.commit([parked])
        assert driver.state.turns[turn_id].wait is None
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.turns[turn_id].wait is None
        assert parked.record_id not in {row.record.record_id for row in raw_records}


def test_commit_rejects_unbound_retained_provider_evidence_before_append(tmp_path: Path) -> None:
    path = tmp_path / "retained-evidence.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    request = _model_request()
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-evidence-preflight", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))

        oid = "model"
        plan = _start_model_attempt(driver, oid, request)
        stored_plan = driver.state.operations[oid].attempts[1].execution_plan
        parked = driver.record(
            "op_parked",
            {
                "phase": "after_dispatch",
                "handle": {
                    "kind": "provider",
                    "provider": "model",
                    "job_id": "job-1",
                    "operation_id": oid,
                    "attempt": 1,
                    "request_digest": stored_plan._payload["request_digest"],
                    "evidence": ["provider/one"],
                    "query_ref": None,
                    "cancel_ref": None,
                },
                "poll_at_ms": None,
                "deadline_ms": None,
            },
            plan,
        )
        driver.commit([parked])

        completed = _complete_model(
            driver,
            oid,
            plan,
            {"content": "done", "tool_calls": []},
            ["provider/two"],
        )
        with pytest.raises(TransitionError, match="result does not bind parked evidence"):
            driver.commit([completed])
        assert driver.state.operations[oid].attempts[1].state == "parked"
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.operations[oid].attempts[1].state == "parked"
        assert completed.record_id not in {row.record.record_id for row in raw_records}


def test_commit_rejects_invalid_retained_boundary_before_append(tmp_path: Path) -> None:
    path = tmp_path / "retained-boundary.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-boundary-preflight", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))

        boundary = driver.boundary_record(
            "budget",
            "invalid-usage",
            {
                "tool_names": [],
                "usage": BudgetUsageSnapshot().to_dict() | {"unexpected": 1},
                "exhaustion": None,
            },
        )
        with pytest.raises(TransitionError, match="budget snapshot contains unknown or incomplete fields"):
            driver.commit([boundary])
        assert driver.state.boundaries == {}
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.boundaries == {}
        assert boundary.record_id not in {row.record.record_id for row in raw_records}


def test_commit_rejects_unbound_producer_ref_before_append(tmp_path: Path) -> None:
    path = tmp_path / "unbound-preflight.sqlite"
    provider = LocalKeyProvider(b"k" * 32)
    request = _model_request()
    with open_store(path) as store:
        sessions = SessionDriver(store=store, content_key_provider=provider)
        sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
        sessions.push("s", InboxItem("input", "user", {"content": "go"}))
        runtime = sessions.runtime(
            Agent("raw", "Work.", model="scripted"),
            RunConfig(
                model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
                workspace="/workspace",
            ),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="raw-unbound-preflight", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))

        oid = "model"
        plan = _start_model_attempt(driver, oid, request)
        stored_plan = driver.state.operations[oid].attempts[1].execution_plan
        retained_request = content_ref("request-owner")
        completed = driver.record(
            "op_completed",
            {
                "result": retained_request,
                "result_digest": digest(retained_request),
                "usage": {},
                "shared_state": None,
                "evidence": {"kind": "model"},
                "execution_started": True,
                "context": "normal",
                "request_digest": stored_plan._payload["request_digest"],
                "provider_binding": stored_plan._payload["provider_binding"],
            },
            plan,
        )
        with pytest.raises(RecordError, match="outside the retained ledger"):
            driver.commit([completed])
        assert driver.state.operations[oid].attempts[1].state == "started"
        assert sessions.store.release(lease)

    with open_store(path) as reopened:
        raw_state, raw_records, _ = reopened.read_state("s")
        assert raw_state.operations[oid].attempts[1].state == "started"
        assert completed.record_id not in {row.record.record_id for row in raw_records}


def _model_request() -> dict:
    return {
        "model": "scripted",
        "route": {},
        "messages": [{"role": "user", "content": "go"}],
        "tools": [],
        "metadata": {"vv_session": {"endpoint_order": ["a", "b"], "model_binding": "scripted"}},
        "prompt_bundle": {},
        "model_settings": {},
    }


def _start_model_attempt(driver: _Driver, oid: str, request: dict) -> Record:
    plan = _plan_record(driver, "model", oid, request, provider_binding="model")
    driver.commit([plan])
    started = driver.record(
        "op_started",
        {
            "dispatch_id": f"{oid}/1",
            "authorization_version": "1",
            "epoch": driver.scope.lease.epoch,
            "mode": "sync",
            "endpoint_id": "a",
        },
        plan,
    )
    driver.commit([started])
    return plan


def _complete_model(driver: _Driver, oid: str, plan: Record, result: dict, evidence: object) -> Record:
    stored_plan = driver.state.operations[oid].attempts[1].execution_plan
    return driver.record(
        "op_completed",
        {
            "result": result,
            "result_digest": digest(result),
            "usage": {},
            "shared_state": None,
            "evidence": evidence,
            "execution_started": True,
            "context": "normal",
            "request_digest": stored_plan._payload["request_digest"],
            "provider_binding": stored_plan._payload["provider_binding"],
        },
        plan,
    )


def _plan_record(
    driver: _Driver,
    kind: str,
    oid: str,
    request: dict,
    *,
    provider_binding: str,
    attempt: int = 1,
    context_version: str | None = None,
) -> Record:
    return make_record_before_publication(
        "op_planned",
        session_id="s",
        turn_id=driver.state.active_turn_id,
        operation_id=oid,
        attempt=attempt,
        payload={
            "op_kind": kind,
            "purpose": "primary" if kind == "model" else None,
            "request": request,
            "request_digest": digest(request),
            "context_version": context_version or str(driver.records[-1].seq),
            "dependencies": [],
            "tool": {"idempotency": "supported"} if kind == "tool" else None,
            "idempotency_key": f"s/{oid}",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": provider_binding,
            "consumed_unknowns": [],
        },
    )
