"""Seeded fault repetition on disposable SQL stores; no remote models or broker."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.session.conftest import open_store
from tests.session.test_recovery_matrix import runtime
from tests.session.transport import PollingProvider, Transport, WorkerKilled
from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.reducer import fold
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

SCENARIOS = ("tools", "approval", "child", "cancel", "delayed", "accepted", "kill_before", "kill_after")


@contextmanager
def disposable_database(backend):
    if backend == "sqlite":
        with TemporaryDirectory(prefix="vvsk-soak-") as directory:
            yield Path(directory) / "session.sqlite"
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres")
    name = f"vvsk_test_soak_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(dsn, dbname=name)
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def run_soak(database, *, seed=42, sessions=8, assert_invariants=True):
    if sessions < 1:
        raise ValueError("sessions must be positive")
    rng, counts = random.Random(seed), Counter()
    started = time.perf_counter()
    now = 1_000_000
    effects, models = Counter(), Counter()
    with open_store(database) as store, patch.object(type(store), "clock_sql", str(now)):
        store.install_schema()
        PollingProvider(database).install()
        transport = Transport(store, database)

        def advance(ms):
            nonlocal now
            now += ms
            type(store).clock_sql = str(now)

        for offset in range(0, sessions, len(SCENARIOS)):
            cases = list(SCENARIOS[: min(len(SCENARIOS), sessions - offset)])
            rng.shuffle(cases)
            expected, providers, scenarios = {}, {}, {}
            recoveries = {}
            for index, scenario in enumerate(cases, offset):
                sid = f"session/{index:06d}"
                counts[scenario] += 1
                scenarios[sid] = scenario

                def effect(sid=sid) -> str:
                    effects[sid] += 1
                    return "ok"

                tool = function_tool(effect, needs_approval=scenario == "approval")

                def model(_request, sid=sid, scenario=scenario):
                    models[sid] += 1
                    if scenario == "cancel":
                        transport.push(sid, InboxItem("cancel", "control", {"action": "cancel"}, f"{sid}/turn/initial"))
                        return LLMResponse("discarded")
                    return LLMResponse("", [ToolCall("job", "effect", {})])

                def final(_request, sid=sid):
                    models[sid] += 1
                    return LLMResponse("done")

                steps = [model, final] if scenario != "cancel" else [model]
                transport.create(sid, steps, [tool])
                rt = transport.runtimes[sid]
                expected[sid] = "cancelled" if scenario == "cancel" else "completed"
                if scenario == "child":
                    child_sid = f"{sid}/child"
                    rt.children["effect"] = lambda plan, child_sid=child_sid: ChildSession(
                        SessionSpec(child_sid, "test", "/tmp"), "child input"
                    )

                    def child_final(_request, sid=child_sid):
                        models[sid] += 1
                        return LLMResponse("child done")

                    transport.runtimes[child_sid] = runtime(database, [child_final], wake=transport.wake)
                    expected[child_sid] = "completed"
                if scenario == "accepted":
                    provider = providers[sid] = PollingProvider(database)
                    provider.deadline = now + rng.randint(2, 5) * rt.poll_ms
                    rt.providers["effect"] = provider
                if scenario.startswith("kill_"):
                    committed = scenario == "kill_after"

                    def kill(point, record, committed=committed):
                        if (
                            point == ("after_commit" if committed else "before_commit")
                            and record
                            and record.kind == "op_planned"
                            and record.payload["op_kind"] == "tool"
                        ):
                            raise WorkerKilled

                    rt.hook = kill
                    recoveries[sid] = (committed, model)
                available = now + rng.randint(1, 999) if scenario == "delayed" else 0
                item = InboxItem("initial", "user", {"content": "go"}, available_ms=available)
                transport.push(sid, item)
                transport.push(sid, item)  # Identical input admission replay as well as wake replay.
                transport.queue.extend([sid] * rng.randint(1, 3))

            # A retained queue is randomly reordered/duplicated/dropped; tick repairs loss.
            deliveries = 0
            while transport.queue:
                deliveries += 1
                if deliveries > 100:
                    raise AssertionError("idle self-wake in initial dispatch")
                index = rng.randrange(len(transport.queue))
                sid = transport.queue[index]
                if rng.random() < 0.25:
                    transport.queue.pop(index)
                    counts["dropped_wakes"] += 1
                    continue
                try:
                    with (
                        patch.object(store, "release", lambda lease: False)
                        if sid in recoveries
                        else patch.object(store, "release", store.release)
                    ):
                        transport.deliver(index)
                except WorkerKilled:
                    counts["abandoned_workers"] += 1
                    committed, callback = recoveries.pop(sid)
                    rt = transport.runtimes[sid]
                    rt.hook = lambda *_: None
                    if not committed:
                        rt.llm.steps.insert(0, callback)
            # Any killed session whose initial wakes all dropped still hits the same cut via tick.
            for sid in list(recoveries):
                transport.wake(sid)
                with patch.object(store, "release", lambda lease: False):
                    try:
                        transport.deliver()
                    except WorkerKilled:
                        counts["abandoned_workers"] += 1
                    else:
                        raise AssertionError("worker fault cut was not reached")
                committed, callback = recoveries.pop(sid)
                transport.runtimes[sid].hook = lambda *_: None
                if not committed:
                    transport.runtimes[sid].llm.steps.insert(0, callback)

            # Tick recovers drops, advances Accepted polls and finally expires abandoned leases.
            for _ in range(20):
                transport.tick(page_size=3)
                transport.drain()
                for sid in expected:
                    state, rows, _ = read_state(store, sid)
                    for row in rows:
                        park = row.record
                        if park.kind == "op_parked" and park.payload["handle"]["kind"] == "approval":
                            handle = park.payload["handle"]
                            item = InboxItem(
                                "approval",
                                "approval_answer",
                                {
                                    "operation_id": park.operation_id,
                                    "attempt": park.attempt,
                                    "request_id": handle["request_id"],
                                    "request_digest": handle["request_digest"],
                                    "scope": handle["scope"],
                                    "decision": "approve",
                                },
                                park.turn_id,
                            )
                            if state.active_turn_id:
                                transport.push(sid, item)
                transport.drain()
                if all(read_state(store, sid)[0].terminal_seq for sid in expected):
                    break
                advance(1000)

            for sid, status in expected.items():
                state, rows, _ = read_state(store, sid)
                logical = [r.record for r in rows]
                inputs = [
                    InboxItem.parse(body)
                    for (body,) in store._rows(
                        "SELECT body FROM sk_inbox WHERE session_id=%s AND consumed_seq IS NOT NULL", (sid,)
                    )
                ]
                counts["records"] += len(rows)
                counts["terminal_sessions"] += bool(state.terminal_seq)
                if assert_invariants:
                    assert [r.seq for r in rows] == list(range(1, len(rows) + 1)), sid
                    assert len({r.record_id for r in logical}) == len(rows), sid
                    assert fold(logical, consumed_inputs=inputs) == state, sid
                    ends = [r for r in logical if r.kind == "turn_ended"]
                    assert len(ends) == 1 and ends[0].payload["status"] == status, sid
                    assert state.active_turn_id is None and not store.peek_inbox(sid), sid
                    sql_state = store._one(
                        "SELECT phase,next_drive_ms,active_turn_id,terminal_seq FROM sk_session WHERE session_id=%s", (sid,)
                    )
                    assert sql_state == (state.phase, state.next_drive_ms, state.active_turn_id, state.terminal_seq), sid
                    starts = [r for r in logical if r.kind == "op_started"]
                    assert len({(r.operation_id, r.attempt) for r in starts}) == len(starts), sid
                    unknowns = [r for r in logical if r.kind == "op_unknown"]
                    if unknowns:
                        assert len(unknowns) == 1 and unknowns[0].payload["observation"]["code"] == (
                            "duplicate_model_request_and_cost"
                        ), sid
                    if sid in models:
                        assert models[sid] == sum(state.operations[r.operation_id].kind == "model" for r in starts), sid
                        assert models[sid] == (1 if scenarios.get(sid) in (None, "cancel") else 2) + bool(unknowns), sid
                        tool_plans = [
                            state.operations[r.operation_id].attempts[r.attempt].execution_plan
                            for r in starts
                            if state.operations[r.operation_id].kind == "tool"
                        ]
                        assert effects[sid] == sum(
                            r.payload["provider_binding"] == "function" and scenarios.get(sid) != "child" for r in tool_plans
                        ), sid
                if sid in providers:
                    queries = providers[sid].queries
                    period = transport.runtimes[sid].poll_ms
                    counts["provider_polls"] += len(queries)
                    if assert_invariants:
                        assert len(queries) <= 1 + (queries[-1] - queries[0]) // period, (sid, queries, period)
                        assert all(b - a >= period for a, b in pairwise(queries)), sid
            counts["model_submits"] = sum(models.values())
            counts["tool_effects"] = sum(effects.values())
            for _ in range(30):
                if not transport.tick():
                    break
                transport.drain()
            if assert_invariants:
                assert not transport.queue and not store.list_runnable(), "transport did not drain"
            transport.runtimes.clear()
        calls, provider_effects = PollingProvider(database).counts()
        if assert_invariants:
            assert calls == provider_effects == counts["accepted"], "duplicate provider submits"
        counts["provider_submits"] = calls
    return {
        "seed": seed,
        "sessions": sessions,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "asserted": assert_invariants,
        "queue_remaining": len(transport.queue),
        "counts": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=1024)
    parser.add_argument("--store", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--assert", dest="assert_invariants", action="store_true")
    args = parser.parse_args()
    with disposable_database(args.store) as database:
        summary = run_soak(database, seed=args.seed, sessions=args.sessions, assert_invariants=args.assert_invariants)
    print(json.dumps({"store": args.store, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
