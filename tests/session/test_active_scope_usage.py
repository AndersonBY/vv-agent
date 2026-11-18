"""CoreDriver publishes usage for real active execution intervals."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.agent import ToolUseBehavior
from vv_agent.budget import HostCost, RunBudgetLimits
from vv_agent.kernel import _driver as driver_module
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._hydration import MissingContentLink, TypedSchemaFailure
from vv_agent.kernel._records import Record
from vv_agent.kernel._retained_budget import RetainedBudgetHydrationRequired, retained_budget_inputs
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver, drive, read_state
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.providers import Accepted, Definitive
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, NoToolPolicy, ToolCall, ToolExecutionResult

MASTER_KEY = b"k" * 32
ANSWER = "active-scope-finished-result"


class ManualClock:
    def __init__(self, ns: int = 1_000_000_000_000) -> None:
        self.ns = ns

    def now_ms(self) -> int:
        return self.ns // 1_000_000

    def monotonic(self) -> float:
        return self.ns / 1_000_000_000

    def monotonic_ns(self) -> int:
        return self.ns

    def sleep(self, seconds: float) -> None:
        raise AssertionError(f"active usage test must not sleep: {seconds}")

    def advance_ms(self, milliseconds: int) -> None:
        self.ns += milliseconds * 1_000_000


class Meter:
    def __init__(self) -> None:
        self.reads = 0

    def read(self) -> HostCost:
        self.reads += 1
        return HostCost("credits", self.reads, "USD")


class FixedMeter:
    def __init__(self, amount: int) -> None:
        self.amount = amount
        self.reads = 0

    def read(self) -> HostCost:
        self.reads += 1
        return HostCost("credits", self.amount, "USD")


class Cut(Exception):
    pass


@function_tool
def remote_work() -> str:
    raise AssertionError("provider operations must not execute the function body")


class PollingProvider:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.handle: dict = {}
        self.submits = 0
        self.queries = 0

    def submit(self, plan, *, context):
        self.submits += 1
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "polling-job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "provider-accepted",
            "query_ref": None,
            "cancel_ref": None,
        }
        return Accepted(self.handle)

    def query(self, handle):
        self.queries += 1
        if self.queries == 1:
            self.clock.advance_ms(5)
            return Accepted(handle)
        self.clock.advance_ms(7)
        return Definitive(
            ToolExecutionResult("remote-call", "provider query finished").to_dict(),
            (handle["evidence"],),
        )

    def cancel(self, handle):
        raise AssertionError("the provider handle must not be cancelled")

    def authenticate(self, item, plan):
        return False


def runtime(
    sessions: SessionDriver,
    provider: ScriptedModelProvider,
    *,
    no_tool_policy: NoToolPolicy = "finish",
    clock: ManualClock | None = None,
    meter: Meter | FixedMeter | None = None,
    budget_limits: RunBudgetLimits | None = None,
    tools=(),
    tool_use_behavior: ToolUseBehavior = "run_llm_again",
):
    selected = sessions.runtime(
        Agent(
            "active-scope",
            "Return a concise answer.",
            model="scripted",
            tools=list(tools),
            tool_use_behavior=tool_use_behavior,
        ),
        RunConfig(
            model_provider=provider,
            workspace=Path("/workspace"),
            no_tool_policy=no_tool_policy,
            host_cost_meter=meter,
            budget_limits=budget_limits,
        ),
        session_id="s",
    )
    if clock is not None:
        selected.clock = clock
    return selected


def start(sessions: SessionDriver, input_id: str = "initial") -> None:
    sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    sessions.push("s", InboxItem(input_id, "user", {"content": "start"}))


def rows(sessions: SessionDriver):
    return list(sessions.store.read("s").records)


def hydrated(sessions: SessionDriver, row) -> dict:
    return hydrate_record_payload(row, hydration_for_session_store(sessions.store, "s"))


def usage_rows(sessions: SessionDriver):
    return [row for row in rows(sessions) if row.record.kind == "usage_observed"]


def interrupt_before_model(sessions: SessionDriver) -> None:
    start(sessions)
    selected = runtime(
        sessions,
        ScriptedModelProvider.new("scripted", "scripted", []),
        budget_limits=RunBudgetLimits(max_wall_time_ms=10_000),
    )

    def cut_after_plan(point, record):
        if point == "after_commit" and record is not None and record.kind == "op_planned":
            raise Cut()

    selected.hook = cut_after_plan
    with pytest.raises(Cut):
        drive(sessions.store, "s", runtime=selected)


def test_real_s1_no_tool_finish_publishes_and_hydrates_four_active_observations(tmp_path: Path) -> None:
    calls = []
    path = tmp_path / "finish.sqlite"
    key_provider = LocalKeyProvider(MASTER_KEY)

    def complete(_request):
        calls.append(ANSWER)
        return LLMResponse(ANSWER)

    with SQLiteStore.standalone(str(path)) as store:
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        start(sessions)
        rt = runtime(sessions, ScriptedModelProvider.from_callback("scripted", "scripted", complete))

        assert drive(sessions.store, "s", runtime=rt)

        raw = rows(sessions)
        assert [row.record.kind for row in raw] == [
            "session_created",
            "input_applied",
            "turn_started",
            "usage_observed",
            "op_planned",
            "usage_observed",
            "op_started",
            "usage_observed",
            "op_completed",
            "usage_observed",
            "turn_ended",
        ]
        assert calls == [ANSWER]
        assert ANSWER.encode() not in b"".join(row.record.encode() for row in raw)
        assert b"kernel-active-scope" not in b"".join(row.record.encode() for row in raw)

    with SQLiteStore.standalone(str(path)) as reopened:
        sessions = SessionDriver(store=reopened, content_key_provider=key_provider)
        raw = rows(sessions)
        usages = usage_rows(sessions)
        assert len(usages) == 4
        turn_id = next(row.record.turn_id for row in raw if row.record.kind == "turn_started")
        assert turn_id is not None
        for row in usages:
            assert row.record.operation_id is None and row.record.attempt is None
            assert row.record.turn_id == turn_id
            assert row.record.payload["meter_id"] == f"active/{turn_id}/{row.writer_epoch}"
            assert row.record.payload["source"] != "kernel-active-scope"
            assert declared_content_ref(row.record.payload["usage"]) is not None
            assert declared_content_ref(row.record.payload["source"]) is not None
            payload = hydrated(sessions, row)
            assert payload["source"] == "kernel-active-scope"
            assert payload["usage"]["host_cost"] is None
        assert [hydrated(sessions, row)["observation"] for row in usages] == [1, 2, 3, 4]

        started = next(row for row in raw if row.record.kind == "op_started")
        completed = next(row for row in raw if row.record.kind == "op_completed")
        planned = next(row for row in raw if row.record.kind == "op_planned")
        assert started.record.operation_id == completed.record.operation_id == planned.record.operation_id
        assert started.record.attempt == completed.record.attempt == planned.record.attempt == 1
        assert hydrated(sessions, completed)["result"]["content"] == ANSWER


def test_wait_user_excludes_parked_downtime_and_starts_a_new_epoch_interval(tmp_path: Path) -> None:
    clock = ManualClock()
    meter = Meter()
    with SQLiteStore.standalone(str(tmp_path / "wait.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        start(sessions)
        question = "Resume with the approved answer?"
        rt = runtime(
            sessions,
            ScriptedModelProvider.new("scripted", "scripted", [LLMResponse(question)]),
            no_tool_policy="wait_user",
            clock=clock,
            meter=meter,
        )

        assert drive(sessions.store, "s", runtime=rt)
        parked = next(row.record for row in rows(sessions) if row.record.kind == "turn_parked")
        initial_usages = usage_rows(sessions)
        assert len(initial_usages) == 4
        assert [row.record.kind for row in rows(sessions) if row.record.turn_id == parked.turn_id].count("turn_ended") == 0
        assert meter.reads == 4

        clock.advance_ms(60_000)
        sessions.push(
            "s",
            InboxItem(
                "reply",
                "user",
                {"content": {"interaction_id": parked.payload["interaction_id"], "text": "continue"}},
                target_turn_id=parked.turn_id,
                generation=0,
            ),
        )

        def complete(_request):
            clock.advance_ms(7)
            return LLMResponse(ANSWER)

        resumed = runtime(
            sessions,
            ScriptedModelProvider.from_callback("scripted", "scripted", complete),
            clock=clock,
            meter=meter,
        )
        assert drive(sessions.store, "s", runtime=resumed)

        all_usages = usage_rows(sessions)
        resumed_usages = all_usages[len(initial_usages) :]
        assert len(resumed_usages) == 5
        initial_meter = initial_usages[0].record.payload["meter_id"]
        resumed_meter = resumed_usages[0].record.payload["meter_id"]
        assert initial_meter != resumed_meter
        assert resumed_meter.startswith(f"active/{parked.turn_id}/")
        for offset, row in enumerate(resumed_usages, start=len(initial_usages) + 1):
            payload = hydrated(sessions, row)
            assert payload["usage"]["elapsed_ms"] < 1_000
            assert payload["usage"]["host_cost"] == HostCost("credits", offset, "USD").to_dict()
        assert [hydrated(sessions, row)["observation"] for row in resumed_usages] == [1, 2, 3, 4, 5]


def test_provider_query_completion_from_parked_state_records_the_claimed_worker_interval(tmp_path: Path) -> None:
    clock = ManualClock()
    meter = FixedMeter(4)
    provider = PollingProvider(clock)
    model_calls = []
    completed_phases = []

    def complete(_request):
        model_calls.append("model")
        if len(model_calls) == 1:
            return LLMResponse("", [ToolCall("remote-call", "remote_work", {})])
        return LLMResponse("provider scenario complete")

    with SQLiteStore.standalone(str(tmp_path / "provider-query.sqlite"), clock=clock) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        start(sessions)
        rt = runtime(
            sessions,
            ScriptedModelProvider.from_callback("scripted", "scripted", complete),
            clock=clock,
            meter=meter,
            tools=(remote_work,),
            tool_use_behavior="stop_on_first_tool",
        )
        rt.providers["remote_work"] = provider

        def capture_completion_phase(point, record):
            if (
                point == "before_commit"
                and record is not None
                and record.kind == "op_completed"
                and record.operation_id == provider.handle.get("operation_id")
            ):
                completed_phases.append(read_state(sessions.store, "s")[0].phase)

        rt.hook = capture_completion_phase

        assert drive(sessions.store, "s", runtime=rt)
        parked = next(row.record for row in rows(sessions) if row.record.kind == "op_parked")
        assert read_state(sessions.store, "s")[0].phase == "active"
        initial_usage_count = len(usage_rows(sessions))
        initial_reads = meter.reads
        assert provider.submits == 1 and provider.queries == 0
        assert initial_usage_count >= 3

        clock.advance_ms(parked.payload["poll_at_ms"] - clock.now_ms())
        assert drive(sessions.store, "s", runtime=rt)
        assert provider.queries == 1
        assert len(usage_rows(sessions)) == initial_usage_count
        assert meter.reads == initial_reads

        clock.advance_ms(rt.poll_ms)
        assert drive(sessions.store, "s", runtime=rt)
        assert provider.queries == 2
        completed = next(
            row
            for row in rows(sessions)
            if row.record.kind == "op_completed" and row.record.operation_id == provider.handle["operation_id"]
        )
        preceding = next(row for row in rows(sessions) if row.seq == completed.seq - 1)
        assert preceding.record.kind == "usage_observed"
        payload = hydrated(sessions, preceding)
        assert payload["usage"]["elapsed_ms"] == 7
        assert payload["usage"]["host_cost"] == HostCost("credits", 4, "USD").to_dict()
        assert completed_phases == ["active"]
        assert model_calls == ["model", "model"]


@pytest.mark.parametrize("dimension", ["wall_time", "host_cost"])
def test_cold_reopen_budget_reconstruction_rejects_new_model_without_repeating_completed_call(
    tmp_path: Path,
    dimension: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock()
    meter = FixedMeter(5) if dimension == "host_cost" else None
    key_provider = LocalKeyProvider(MASTER_KEY)
    path = tmp_path / f"budget-{dimension}.sqlite"
    model_calls = []
    limits = (
        RunBudgetLimits(max_wall_time_ms=10)
        if dimension == "wall_time"
        else RunBudgetLimits(max_host_cost=HostCost("credits", 10, "USD"))
    )

    def first_model(_request):
        model_calls.append("first")
        clock.advance_ms(7)
        return LLMResponse("Budgeted question")

    with SQLiteStore.standalone(str(path), clock=clock) as store:
        sessions = SessionDriver(store=store, content_key_provider=key_provider)
        start(sessions)
        rt = runtime(
            sessions,
            ScriptedModelProvider.from_callback("scripted", "scripted", first_model),
            no_tool_policy="wait_user",
            clock=clock,
            meter=meter,
            budget_limits=limits,
        )
        assert drive(sessions.store, "s", runtime=rt)
        parked = next(row.record for row in rows(sessions) if row.record.kind == "turn_parked")
        assert parked.turn_id is not None
        assert len([row for row in rows(sessions) if row.record.kind == "op_started"]) == 1

    clock.advance_ms(60_000)
    if meter is not None:
        meter.amount = 11

    with SQLiteStore.standalone(str(path), clock=clock) as reopened:
        sessions = SessionDriver(store=reopened, content_key_provider=key_provider)
        sessions.push(
            "s",
            InboxItem(
                "reply",
                "user",
                {"content": {"interaction_id": parked.payload["interaction_id"], "text": "continue"}},
                target_turn_id=parked.turn_id,
                generation=0,
            ),
        )

        def second_model(_request):
            model_calls.append("second")
            return LLMResponse("must not dispatch")

        resumed = runtime(
            sessions,
            ScriptedModelProvider.from_callback("scripted", "scripted", second_model),
            clock=clock,
            meter=meter,
            budget_limits=limits,
        )

        logical_state = _Driver._logical_state

        def reject_budget_logical_state(driver: _Driver):
            if sys._getframe(1).f_code.co_name in {"live_budget_inputs_steps", "_retained_budget"}:
                raise AssertionError("retained budget must use the raw fold state")
            return logical_state(driver)

        monkeypatch.setattr(_Driver, "_logical_state", reject_budget_logical_state)
        if dimension == "wall_time":

            def advance_active_worker(point, record):
                if point == "before_commit" and record is not None and record.kind == "op_planned":
                    clock.advance_ms(4)

            resumed.hook = advance_active_worker

        assert drive(sessions.store, "s", runtime=resumed)
        assert model_calls == ["first"]
        assert len([row for row in rows(sessions) if row.record.kind == "op_started"]) == 1
        completed = [row for row in rows(sessions) if row.record.kind == "op_completed"]
        assert sum(hydrated(sessions, row)["execution_started"] is True for row in completed) == 1
        terminal = next(row for row in rows(sessions) if row.record.kind == "turn_ended")
        terminal_payload = hydrated(sessions, terminal)
        assert terminal_payload["status"] == "failed"
        assert terminal_payload["reason"] == "budget_exhausted"

        exhausted = next(
            hydrated(sessions, row)["data"]["exhaustion"]
            for row in reversed(rows(sessions))
            if row.record.kind == "boundary_recorded"
            and row.record.payload["stage"] == "budget"
            and hydrated(sessions, row)["data"]["exhaustion"] is not None
        )
        assert exhausted["dimension"] == dimension
        assert exhausted["observed"] == 11


def test_finalize_failure_rolls_back_content_records_and_usage_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    failed = False

    def complete(_request):
        calls.append(ANSWER)
        return LLMResponse(ANSWER)

    link_content = driver_module.link_declared_record_content

    def fail_finalize(writer, links):
        nonlocal failed
        if any(link.record_id.startswith("usage/") for link in links) and not failed:
            failed = True
            raise Cut()
        link_content(writer, links)

    monkeypatch.setattr(driver_module, "link_declared_record_content", fail_finalize)

    with SQLiteStore.standalone(str(tmp_path / "rollback.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        start(sessions)
        rt = runtime(sessions, ScriptedModelProvider.from_callback("scripted", "scripted", complete))

        with pytest.raises(Cut):
            drive(sessions.store, "s", runtime=rt)
        assert failed and not calls
        assert not usage_rows(sessions)
        assert not any(row.record.kind == "op_planned" for row in rows(sessions))

        retry = runtime(sessions, ScriptedModelProvider.from_callback("scripted", "scripted", complete))
        assert drive(sessions.store, "s", runtime=retry)
        assert calls == [ANSWER]
        assert len(usage_rows(sessions)) == 4
        assert sum(row.record.kind == "op_started" for row in rows(sessions)) == 1


def test_committed_plan_replay_adds_no_usage_and_resume_binds_the_same_operation(tmp_path: Path) -> None:
    calls = []
    captured: list[Record] = []

    def complete(_request):
        calls.append(ANSWER)
        return LLMResponse(ANSWER)

    def cut_after_plan(point, record):
        if point == "after_commit" and record is not None and record.kind == "op_planned" and not captured:
            captured.append(record)
            raise Cut()

    with SQLiteStore.standalone(str(tmp_path / "replay.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        start(sessions)
        rt = runtime(sessions, ScriptedModelProvider.from_callback("scripted", "scripted", complete))
        rt.hook = cut_after_plan

        with pytest.raises(Cut):
            drive(sessions.store, "s", runtime=rt)
        assert not calls
        assert len(usage_rows(sessions)) == 1
        plan = captured[0]
        plan_body = plan.encode()
        before_rows = [(row.seq, row.record.record_id, row.record.encode()) for row in rows(sessions)]

        lease = sessions.store.acquire("s", owner="replay", ttl_ms=15_000)
        assert lease is not None
        scope = _Scope(lease, rt, now_ms=rt.clock.now_ms())
        replay_driver = _Driver(sessions.store, "s", rt, scope)
        replay_driver.commit([plan])
        assert plan.encode() == plan_body
        assert [(row.seq, row.record.record_id, row.record.encode()) for row in rows(sessions)] == before_rows
        sessions.store.release(lease)

        resumed = runtime(sessions, ScriptedModelProvider.from_callback("scripted", "scripted", complete))
        assert drive(sessions.store, "s", runtime=resumed)
        assert calls == [ANSWER]
        assert len(usage_rows(sessions)) == 4
        started = [row.record for row in rows(sessions) if row.record.kind == "op_started"]
        completed = [row.record for row in rows(sessions) if row.record.kind == "op_completed"]
        assert len(started) == len(completed) == 1
        assert started[0].operation_id == completed[0].operation_id == plan.operation_id


def test_retained_budget_requires_reader_and_rejects_malformed_ref(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "retained-reader.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        interrupt_before_model(sessions)
        state, _, _ = read_state(store, "s")
        tid = state.active_turn_id
        assert tid is not None

        with pytest.raises(RetainedBudgetHydrationRequired):
            retained_budget_inputs(state, tid)

        turn = state.turns[tid]
        malformed = dict(turn.start.payload["budget"])
        malformed["mac"] = "not-a-sha256"
        state.turns[tid] = replace(turn, start=replace(turn.start, payload=turn.start.payload | {"budget": malformed}))
        with pytest.raises(TypedSchemaFailure, match=r"turn_started\.budget contains a malformed ContentRef"):
            retained_budget_inputs(state, tid)


@pytest.mark.parametrize("corruption", ["missing", "wrong-slot", "cross-owner"])
def test_retained_budget_rejects_broken_owner_links(tmp_path: Path, corruption: str) -> None:
    with SQLiteStore.standalone(str(tmp_path / f"retained-{corruption}.sqlite")) as store:
        sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        interrupt_before_model(sessions)
        turn = next(row for row in rows(sessions) if row.record.kind == "turn_started")
        record_seq = turn.seq

        if corruption == "missing":
            store.connection.execute(
                "DELETE FROM sk_content_link WHERE tenant_id=? AND session_id='s' AND record_seq=? AND slot='budget'",
                (store.tenant_id, record_seq),
            )
        elif corruption == "wrong-slot":
            store.connection.execute(
                "UPDATE sk_content_link SET slot='wrong_budget_slot' "
                "WHERE tenant_id=? AND session_id='s' AND record_seq=? AND slot='budget'",
                (store.tenant_id, record_seq),
            )
        else:
            foreign = store.connection.execute(
                "SELECT content_id FROM sk_content_link "
                "WHERE tenant_id=? AND session_id='s' AND record_seq<>? AND slot<>'budget' LIMIT 1",
                (store.tenant_id, record_seq),
            ).fetchone()
            assert foreign is not None
            store.connection.execute(
                "UPDATE sk_content_link SET content_id=? WHERE tenant_id=? AND session_id='s' AND record_seq=? AND slot='budget'",
                (foreign[0], store.tenant_id, record_seq),
            )

        cold = runtime(sessions, ScriptedModelProvider.new("scripted", "scripted", []))
        lease = store.acquire("s", owner=f"retained-{corruption}", ttl_ms=15_000)
        assert lease is not None
        driver = _Driver(store, "s", cold, _Scope(lease, cold, now_ms=cold.clock.now_ms()))
        try:
            state = driver.fold.state
            tid = state.active_turn_id
            assert tid is not None
            with pytest.raises(MissingContentLink):
                retained_budget_inputs(state, tid, hydrator=driver)
        finally:
            store.release(lease)
