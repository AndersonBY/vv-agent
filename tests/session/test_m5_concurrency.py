"""Real-store simultaneous same-value and different-value provider callbacks."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from vv_agent.budget import RunBudgetLimits
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import budget
from vv_agent.session.store import Conflict
from vv_agent.session.supervisor import tick
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .controlled import ControlledProvider
from .test_recovery_matrix import effect_tool, receive, runtime, spawn_worker, start


def test_concurrent_same_and_different_callback_bytes(store, database):
    provider = ControlledProvider(database, "accepted")
    provider.install()
    start(store)
    drive(
        store,
        "s",
        runtime=runtime(
            database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], providers={"effect": provider}
        ),
    )
    original = provider.callback()
    other = replace(original, payload=original.payload | {"result": {"different": True}})
    barrier = Barrier(20)

    def deliver(index):
        with open_store(database) as connection:
            barrier.wait(timeout=10)
            try:
                with connection.atomic() as tx:
                    receipt = tx.push("s", original if index % 2 else other)
                return index % 2, receipt.replayed
            except Conflict:
                return index % 2, "conflict"

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(deliver, range(20)))
    winning = [kind for kind, replay in results if replay is False]
    assert len(winning) == 1
    assert sum(replay == "conflict" for _, replay in results) == 10
    assert sum(replay is True for _, replay in results) == 9
    assert all(kind != winning[0] for kind, replay in results if replay == "conflict")
    assert len(store.peek_inbox("s")) == 1
    rt = runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider})
    drive(store, "s", runtime=rt)
    if winning == [0]:
        assert read_state(store, "s")[0].applied_inputs[original.input_id].payload["disposition"] == "rejected"
        with store.atomic() as tx:
            tx.push("s", replace(original, input_id="authenticated-replacement"))
        drive(store, "s", runtime=rt)
    state, rows, _ = read_state(store, "s")
    assert state.active_turn_id is None
    assert provider.counts() == (1, 1)
    assert sum(r.record.kind == "turn_ended" for r in rows) == 1


def test_model_attempt_cost_identity_survives_restart(store, database):
    requests = []

    def lost(request):
        requests.append(dict(request.metadata))
        raise TimeoutError("accepted model response lost")

    start(store)
    drive(store, "s", runtime=runtime(database, [lost, lost]))
    assert len(requests) == 2
    assert len({r["operation_id"] for r in requests}) == 1
    assert [r["attempt"] for r in requests] == [1, 2]
    assert all(r["call_id"] == f"{r['operation_id']}/{r['attempt']}" for r in requests)
    assert len({r["call_id"] for r in requests}) == 2
    with open_store(database) as fresh:
        state, rows, _ = read_state(fresh, "s")
        unknowns = [r.record for r in rows if r.record.kind == "op_unknown"]
        assert len(unknowns) == 2 and all(r.payload["duplicate_cost_risk"] for r in unknowns)
        projected = [e.to_dict() for e in project_records(rows)]
        failures = [e for e in projected if e["type"] == "model_call_failed"]
        assert [e["call_id"] for e in failures] == [r["call_id"] for r in requests]
        assert all(e["outcome"] == "ambiguous" for e in failures)
        assert len([e for e in projected if e["type"] == "model_retry_duplicate_risk"]) == 2
        assert state.active_turn_id is None
        drive(fresh, "s", runtime=runtime(database, []))
        assert [e.to_dict() for e in project_records(read_state(fresh, "s")[1])] == projected
    assert len(requests) == 2


@pytest.mark.parametrize("mode", ["normal", "unknown", "cancel"])
def test_budget_accounting_survives_result_unknown_and_cancel_restart(store, database, mode):
    provider = ControlledProvider(database, {"normal": "definitive", "unknown": "unknown", "cancel": "accepted"}[mode])
    provider.install()
    start(store)

    class Restart(BaseException):
        pass

    def cut(point, record):
        if point != "after_commit":
            return
        tool = "/tool/" in (record.operation_id or "")
        if (
            (mode == "normal" and tool and record.kind == "op_completed")
            or (mode == "unknown" and tool and record.kind == "op_unknown")
            or (mode == "cancel" and record.kind == "input_applied" and record.payload["input"]["kind"] == "control")
        ):
            raise Restart

    rt = runtime(
        database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], providers={"effect": provider}, hook=cut
    )
    rt.config.budget_limits = RunBudgetLimits(max_tool_calls=10)
    if mode == "cancel":
        drive(store, "s", runtime=rt)
        with store.atomic() as tx:
            tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, "s/turn/initial"))
    with pytest.raises(Restart):
        drive(store, "s", runtime=rt)
    state, _, _ = read_state(store, "s")
    assert state.active_turn_id is not None
    baseline = budget(state, state.active_turn_id)
    assert baseline is not None
    before = baseline.snapshot().to_dict()
    assert before["tool_calls"] == 1 and before["elapsed_ms"] > 0
    with open_store(database) as fresh:
        drive(
            fresh,
            "s",
            runtime=runtime(
                database, [] if mode == "cancel" else [LLMResponse("done")], [effect_tool()], providers={"effect": provider}
            ),
        )
        _, rows, _ = read_state(fresh, "s")
    end = next(r.record for r in rows if r.record.kind == "turn_ended")
    assert end.payload["status"] == ("cancelled" if mode == "cancel" else "completed")
    assert end.payload["budget"]["tool_calls"] == before["tool_calls"]
    assert end.payload["budget"]["tool_calls_by_name"] == before["tool_calls_by_name"]
    assert end.payload["budget"]["elapsed_ms"] >= before["elapsed_ms"]
    assert provider.counts() == (1, 1)


@pytest.mark.persistent_store
def test_default_lease_expiry_empty_inbox_recovers_without_wake(store, database):
    start(store)
    process, pipe = spawn_worker(database, stop_point="before_external_call", stop_tool=False)
    try:
        receive(pipe, "barrier")
        begin = time.monotonic()
        process.kill()
        process.join(5)
        assert not process.is_alive()
        pipe.close()
        assert not store.peek_inbox("s")
        recovered = runtime(database, [LLMResponse("done")], [effect_tool()])
        assert recovered.ttl_ms == 15000 and recovered.heartbeat_seconds == 0.25

        def project(sid, consumer):
            with store.atomic() as tx:
                batch = tx.consumer_batch(sid, consumer)
                if batch is not None:
                    tx.ack(batch)

        while time.monotonic() - begin < 20:
            tick(store, runtime=lambda _sid: recovered, project=project, page_size=1)
            if not store.list_runnable() and read_state(store, "s")[0].active_turn_id is None:
                break
            # Poll the natural database expiry; no fault cut is chosen by a sleep.
            time.sleep(1)
        state, rows, _ = read_state(store, "s")
        assert time.monotonic() - begin < 20
        assert state.active_turn_id is None and not recovered.llm.steps
        assert [r.record.attempt for r in rows if r.record.kind == "op_started"] == [1, 2]
        assert next(r.record for r in rows if r.record.kind == "turn_ended").payload["status"] == "completed"
        assert not store.list_runnable()
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        pipe.close()
