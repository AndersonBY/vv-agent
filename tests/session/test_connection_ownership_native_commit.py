"""Native commit companion and file-H rollback use independent execution connections."""

import json
from threading import Event, current_thread

import pytest

from vv_agent.session.kernel import _Driver
from vv_agent.session.postgres import join_transaction
from vv_agent.session.records import InboxItem

from .test_connection_ownership_native_lifecycle import NativeCohort
from .test_yield_completion_errors import parked_server
from .test_yield_review6_edges import payload


@pytest.mark.persistent_store
@pytest.mark.parametrize("committed", [False, True])
def test_native_companion_accepted_work(store, tmp_path, monkeypatch, committed):
    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    cohort = NativeCohort(server.run_adapter, store, monkeypatch)
    server.run_adapter._native_test_cohort = cohort
    tid = store.read_state("completion")[0].active_turn_id
    entered, read, observed, finish = Event(), Event(), Event(), Event()
    cuts = []
    entries = []
    returns = []
    refresh = _Driver.refresh
    client = server.run_adapter._store.runtimes["completion"].llm
    callback = client.callback

    def model(request):
        entries.append(request)
        response = callback(request)
        returns.append(response)
        return response

    monkeypatch.setattr(client, "callback", model)

    def gate(driver):
        cut = driver.sid == "completion" and current_thread().name == "session-surface-driver" and not cuts
        if cut:
            cuts.append(driver)
            entered.set()
            assert read.wait(5)
        result = refresh(driver)
        if cut:
            observed.set()
            assert finish.wait(5)
        return result

    monkeypatch.setattr(_Driver, "refresh", gate)

    class Rollback(Exception):
        pass

    try:
        server.wake("completion")
        assert entered.wait(5)
        driver = cuts[0]
        assert driver.scope.ready.wait(5) and driver.scope.successful_ready and not driver.scope.lost
        assert driver.store.connection is not store.connection
        assert not driver.store._connection_lock._is_owned()
        try:
            if hasattr(store, "path"):
                with store.atomic() as tx:
                    tx.push("completion", InboxItem("native-companion", "provider_result", payload(provider), tid, 0))
                    server.wake("completion")
                    if not committed:
                        read.set()
                        assert observed.wait(5)
                        assert entries == returns == []
                        raise Rollback()
            else:
                with store.connection.transaction():
                    assert not store._connection_lock._is_owned()
                    join_transaction(store.connection).push(
                        "completion", InboxItem("native-companion", "provider_result", payload(provider), tid, 0)
                    )
                    server.wake("completion")
                    if not committed:
                        read.set()
                        assert observed.wait(5)
                        assert entries == returns == []
                        raise Rollback()
        except Rollback:
            assert not committed
        read.set()
        assert observed.wait(5)
        finish.set()
        server.run_adapter.join()
        cohort.wait()
        assert len(entries) == len(returns) == int(committed)
        assert len(calls) == 1 + int(committed) and provider.submits == 1
        item = store.read_input("completion", "native-companion")
        assert (item is not None) == committed
        if committed:
            assert item.consumed_seq is not None
        assert all(h.done() and h._error is None and not h._thread.is_alive() for h in server.kernel.handles)
        state, records, _ = store.read_state("completion")
        assert sum(r.record.kind == "turn_started" for r in records) == 1
        assert sum(r.record.kind == "turn_ended" for r in records) == int(committed)
        assert sum(r.record.kind == "op_completed" for r in records) == 1 + 2 * int(committed)
        assert not any(r.record.kind == "op_unknown" for r in records)
        assert (state.active_turn_id is None) == committed
        assert all(h.run_id == tid for h in server.kernel.handles)
        assert not provider.cancels and not store.peek_inbox("completion")
        assert not driver.scope.thread.is_alive()
        with (tmp_path / "business-ledger.json").open("x") as out:
            json.dump(
                {
                    "committed": committed,
                    "entries": len(entries),
                    "returns": len(returns),
                    "calls": len(calls),
                    "provider_submits": provider.submits,
                    "input_consumed_seq": item.consumed_seq if item is not None else None,
                    "H": id(store.connection),
                    "E": id(driver.store.connection),
                    "HBready": driver.scope.successful_ready,
                    "HBthread_exited": not driver.scope.thread.is_alive(),
                    "records": [
                        {"seq": r.seq, "kind": r.record.kind, "tid": r.record.turn_id, "payload": r.record.payload}
                        for r in records
                    ],
                },
                out,
                indent=2,
            )
        assert not any(method == "error/warning" for _, method, _ in notices)
        print(
            "CO2-N-COMPANION",
            type(store).__name__,
            committed,
            "entries",
            len(entries),
            "returns",
            len(returns),
            "HBready",
            driver.scope.successful_ready,
            flush=True,
        )
    finally:
        with (tmp_path / "native-cohort-before-cleanup.json").open("x") as out:
            json.dump(cohort.snapshot(), out, indent=2)
        read.set()
        finish.set()
        server.run_adapter.close()
        server.run_adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
def test_active_execution_progresses_with_unrelated_native_host_transaction(store, database):
    from concurrent.futures import ThreadPoolExecutor

    from vv_agent.session.kernel import drive
    from vv_agent.types import LLMResponse

    from .test_recovery_matrix import runtime, start

    start(store)
    ready, proceed, returned = Event(), Event(), Event()
    entries = []
    responses = []

    def actual(request):
        return LLMResponse("committed work")

    def model(request):
        entries.append(1)
        response = actual(request)
        responses.append(response)
        returned.set()
        return response

    def hook(point, record, *, tx):
        if point == "before_external_call":
            ready.set()
            assert proceed.wait(5)

    rt = runtime(database, [model], hook=hook)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(drive, store, "s", runtime=rt)
        try:
            assert ready.wait(5)
            if hasattr(store, "path"):
                store.connection.execute("BEGIN")
                try:
                    store.connection.execute("SELECT count(*) FROM sk_session").fetchone()
                    proceed.set()
                    assert returned.wait(5)
                    assert store.connection.in_transaction
                finally:
                    store.connection.execute("ROLLBACK")
            else:
                with store.connection.transaction():
                    store.connection.execute("SELECT count(*) FROM sk_session").fetchone()
                    assert not store._connection_lock._is_owned()
                    proceed.set()
                    assert returned.wait(5)
        finally:
            proceed.set()
        assert future.result(5)
    assert len(entries) == len(responses) == 1
