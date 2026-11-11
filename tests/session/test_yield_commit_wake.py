"""Admission under a real outer transaction must not wait for the turn owner."""

import subprocess
import sys
from pathlib import Path

import pytest


def lock_probe(database, backend, workspace):
    import faulthandler
    from threading import Event
    from time import monotonic

    from vv_agent.session.postgres import PostgresStore
    from vv_agent.session.sqlite import SQLiteStore

    from .test_yield_completion_errors import parked_server

    faulthandler.dump_traceback_later(12, exit=True)
    scope = PostgresStore.standalone(database) if backend == "postgres" else SQLiteStore.standalone(database)
    with scope as store, pytest.MonkeyPatch.context() as patch:
        store.install_schema()
        server, provider, calls, _ = parked_server(store, Path(workspace), patch)
        adapter = server.run_adapter
        held = store.acquire("completion", owner="manual", ttl_ms=500)
        assert held is not None
        pump_owns_turn, main_owns_store = Event(), Event()
        schedule = adapter._schedule_lease_retry
        hits = []

        def pause(sid, started, connection):
            if not hits:
                hits.append(1)
                print("PUMP_OWNS_TURN", flush=True)
                pump_owns_turn.set()
                assert main_owns_store.wait(5)
                print("PUMP_REQUESTS_STORE", flush=True)
            return schedule(sid, started, connection)

        patch.setattr(adapter, "_schedule_lease_retry", pause)
        try:
            adapter.recover(None, "completion")
            assert pump_owns_turn.wait(5)
            with store.atomic():
                print("MAIN_OWNS_STORE", flush=True)
                main_owns_store.set()
                tid = store.read_state("completion")[0].active_turn_id
                assert tid is not None
                receipt = adapter.controller_action(
                    connection_id=None, thread_id="completion", turn_id=tid, action_id="cancel-outer", action={"kind": "cancel"}
                )
                assert receipt["accepted"]
                print("CONTROLLER_RETURNED_BEFORE_COMMIT", flush=True)
                assert not store.read_state("completion")[0].turns[tid].ended
            store.release(held)
            deadline = monotonic() + 5
            while monotonic() < deadline and not store.read_state("completion")[0].turns[tid].ended:
                Event().wait(0.02)
            state, rows, _ = store.read_state("completion")
            assert state.turns[tid].ended and not store.peek_inbox("completion")
            assert [r.record.payload["status"] for r in rows if r.record.kind == "turn_ended"] == ["cancelled"]
            assert len(calls) == provider.submits == 1
            adapter.join()
            assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
            assert all(not p.is_alive() for p in adapter._pumps)
            print("COMMITTED_CONTROL_CONSUMED_ONCE", flush=True)
        finally:
            main_owns_store.set()
            adapter.close()
            adapter.join()
            server.kernel.close()
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.persistent_store
def test_outer_transaction_controller_and_lease_completion_succeed(database, store_backend, tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-c",
        "from session.test_yield_commit_wake import lock_probe; import sys; lock_probe(*sys.argv[1:])",
        str(database),
        store_backend,
        str(tmp_path / "workspace"),
    ]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "PYTHONPATH": f"{root}/src:{root}/tests",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=20)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONTROLLER_RETURNED_BEFORE_COMMIT" in result.stdout
    assert "COMMITTED_CONTROL_CONSUMED_ONCE" in result.stdout
    assert not result.stderr.strip(), result.stderr


@pytest.mark.persistent_store
@pytest.mark.parametrize("outcome", ["commit", "rollback", "savepoint_rollback", "savepoint_then_commit", "commit_failure"])
@pytest.mark.parametrize("private_scope", [False, True], ids=["host", "callback"])
def test_nested_admission_waits_for_outer_transaction(store, tmp_path, monkeypatch, outcome, private_scope):
    if private_scope:
        with store._connection_scope():
            _nested_admission(store, tmp_path, monkeypatch, outcome)
    else:
        _nested_admission(store, tmp_path, monkeypatch, outcome)


def _nested_admission(store, tmp_path, monkeypatch, outcome):
    from threading import Event, current_thread
    from time import monotonic

    from vv_agent.session.records import InboxItem

    from .test_yield_completion_errors import parked_server
    from .test_yield_review6_edges import payload

    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    before = len(server.kernel.handles)
    original_head = store.read_state("completion")[1][-1].seq
    tid = store.read_state("completion")[0].active_turn_id
    item = InboxItem("nested-reply", "provider_result", payload(provider), tid, 0)
    if outcome == "commit_failure":
        store._rows("CREATE TABLE gate_parent (id integer PRIMARY KEY)")
        store._rows("CREATE TABLE gate_child (id integer REFERENCES gate_parent(id) DEFERRABLE INITIALLY DEFERRED)")
    outer_open = Event()
    reads = []
    rows = store._rows
    transaction_store = store._scoped_store() or store

    def source_transaction_open():
        import sqlite3

        from vv_agent.session.sqlite import SQLiteStore

        if isinstance(transaction_store, SQLiteStore):
            try:
                return transaction_store.connection.in_transaction
            except sqlite3.ProgrammingError:
                return False
        from psycopg.pq import TransactionStatus

        return (
            not transaction_store.connection.closed
            and transaction_store.connection.info.transaction_status != TransactionStatus.IDLE
        )

    def observed_read(*a, **kw):
        if outer_open.is_set() and current_thread().name != "MainThread" and source_transaction_open():
            reads.append(current_thread().name)
        return rows(*a, **kw)

    monkeypatch.setattr(store, "_rows", observed_read)

    def admit():
        server.kernel.push("completion", item)
        for _ in range(100):
            server.wake("completion")
        assert len(adapter._pending_recoveries) == 1
        assert len(server.kernel.handles) == before
        Event().wait(0.12)
        assert len(server.kernel.handles) == before and not notices and not reads

    failed = None
    try:
        try:
            with store.atomic():
                outer_open.set()
                try:
                    with store.atomic(), store.atomic():
                        admit()
                        if outcome in {"savepoint_rollback", "savepoint_then_commit"}:
                            raise ValueError("rollback savepoint")
                except ValueError as exc:
                    assert outcome in {"savepoint_rollback", "savepoint_then_commit"}
                    assert str(exc) == "rollback savepoint"
                # Inner atomic exits are not outer commit notifications.
                Event().wait(0.12)
                assert len(server.kernel.handles) == before and not notices and not reads
                if outcome == "savepoint_then_commit":
                    admit()
                if outcome == "rollback":
                    raise ValueError("rollback outer")
                if outcome == "commit_failure":
                    store._rows("INSERT INTO gate_child(id) VALUES (999)")
        except Exception as exc:
            failed = exc
        finally:
            outer_open.clear()
        assert bool(failed) == (outcome in {"rollback", "commit_failure"})
        if outcome == "commit_failure":
            if hasattr(failed, "sqlstate"):
                assert failed.sqlstate == "23503"
            else:
                import sqlite3

                assert isinstance(failed, sqlite3.IntegrityError) and "FOREIGN KEY" in str(failed)
        elif outcome == "rollback":
            assert isinstance(failed, ValueError) and str(failed) == "rollback outer"
        committed = outcome in {"commit", "savepoint_then_commit"}
        deadline = monotonic() + 5
        while monotonic() < deadline:
            state = store.read_state("completion")[0]
            if (committed and state.active_turn_id is None) or (
                not committed
                and not adapter._pending_recoveries
                and not adapter._recovering
                and all(h.done() for h in server.kernel.handles)
            ):
                break
            Event().wait(0.02)
        adapter.join()
        state, records, _ = store.read_state("completion")
        if committed:
            assert state.active_turn_id is None and len(calls) == 2
            assert (
                sum(r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == item.input_id for r in records)
                == 1
            )
        else:
            assert state.active_turn_id == tid and len(calls) == 1
            assert records[-1].seq == original_head and store.read_input("completion", item.input_id) is None
            server.kernel.push("completion", item)
            server.wake("completion")
            adapter.join()
            assert store.read_state("completion")[0].active_turn_id is None and len(calls) == 2
        assert provider.submits == 1 and not provider.cancels and not store.peek_inbox("completion")
        assert not reads and all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        assert not adapter._pending_recoveries
    finally:
        outer_open.clear()
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.postgres
@pytest.mark.parametrize("rollback", [False, True])
def test_postgres_external_transaction_and_join_transaction(store, tmp_path, monkeypatch, rollback):
    from threading import Event
    from time import monotonic

    from vv_agent.session.postgres import join_transaction
    from vv_agent.session.records import InboxItem

    from .test_yield_completion_errors import parked_server
    from .test_yield_review6_edges import payload

    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    before = len(server.kernel.handles)
    tid = store.read_state("completion")[0].active_turn_id
    item = InboxItem("external-transaction-reply", "provider_result", payload(provider), tid, 0)
    try:
        try:
            # This outer transaction intentionally does not own store._connection_lock.
            with store.connection.transaction():
                with store.connection.transaction():
                    join_transaction(store.connection).push("completion", item)
                    server.wake("completion")
                Event().wait(0.15)
                assert len(server.kernel.handles) == before and not notices and len(calls) == 1
                assert len(adapter._pending_recoveries) == 1
                if rollback:
                    raise ValueError("external rollback")
        except ValueError as exc:
            assert rollback and str(exc) == "external rollback"
        deadline = monotonic() + 5
        while monotonic() < deadline:
            if rollback:
                if not adapter._pending_recoveries and not adapter._recovering and all(h.done() for h in server.kernel.handles):
                    break
            elif store.read_state("completion")[0].active_turn_id is None:
                break
            Event().wait(0.02)
        adapter.join()
        assert len(calls) == (1 if rollback else 2)
        if rollback:
            assert store.read_input("completion", item.input_id) is None
            server.kernel.push("completion", item)
            server.wake("completion")
            adapter.join()
        assert store.read_state("completion")[0].active_turn_id is None
        assert not store.peek_inbox("completion") and provider.submits == 1
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
    finally:
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
def test_transaction_started_after_idle_probe_cannot_dispatch_uncommitted_reply(store, tmp_path, monkeypatch):
    from threading import Event, Thread, current_thread
    from time import monotonic

    from vv_agent.session.records import InboxItem

    from .test_yield_completion_errors import parked_server
    from .test_yield_review6_edges import payload

    server, provider, calls, notices = parked_server(store, tmp_path, monkeypatch)
    adapter = server.run_adapter
    tid = store.read_state("completion")[0].active_turn_id
    ready, release = Event(), Event()
    checked = adapter._store_transaction_open
    cuts, errors = [], []

    def probe():
        value = checked()
        if current_thread().name == "commit-visibility-cut" and not cuts:
            assert not value
            cuts.append(1)
            ready.set()
            assert release.wait(5)
        return value

    monkeypatch.setattr(adapter, "_store_transaction_open", probe)

    def recover():
        try:
            adapter.recover(None, "completion")
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=recover, name="commit-visibility-cut")
    try:
        thread.start()
        assert ready.wait(5)
        with store.atomic():
            server.kernel.push("completion", InboxItem("probe-race-reply", "provider_result", payload(provider), tid, 0))
            server.wake("completion")
            release.set()
            Event().wait(0.2)
            assert len(calls) == provider.submits == 1 and not errors
            assert not any(method == "turn/completed" and params.get("status") == "completed" for _, method, params in notices)
        deadline = monotonic() + 5
        while monotonic() < deadline and store.read_state("completion")[0].active_turn_id is not None:
            Event().wait(0.02)
        thread.join(5)
        adapter.join()
        state, records, _ = store.read_state("completion")
        assert not thread.is_alive() and not errors and cuts == [1]
        assert state.active_turn_id is None and len(calls) == 2 and provider.submits == 1
        assert (
            sum(r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == "probe-race-reply" for r in records)
            == 1
        )
        assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
        assert all(not p.is_alive() for p in adapter._pumps)
    finally:
        release.set()
        thread.join(5)
        adapter.close()
        adapter.join()
        server.kernel.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("close_first", [False, True])
def test_deferred_wake_without_subscribers_with_two_nodes_and_close(store, tmp_path, monkeypatch, close_first):
    from threading import Event
    from time import monotonic

    from vv_agent.app_server import AppServer
    from vv_agent.session.records import InboxItem

    from .test_yield_completion_errors import parked_server
    from .test_yield_review6_edges import payload

    first, provider, calls, _ = parked_server(store, tmp_path, monkeypatch)
    second = AppServer(store=store, host=first.host)
    tid = store.read_state("completion")[0].active_turn_id
    try:
        with store.atomic():
            first.kernel.push("completion", InboxItem("two-node-reply", "provider_result", payload(provider), tid, 0))
            for _ in range(50):
                first.wake("completion")
                second.wake("completion")
            assert len(first.run_adapter._pending_recoveries) == len(second.run_adapter._pending_recoveries) == 1
            if close_first:
                first.run_adapter.close()
                assert not first.run_adapter._pending_recoveries
                assert not first.run_adapter._notification_thread.is_alive()
            assert len(calls) == 1
        deadline = monotonic() + 5
        while monotonic() < deadline and store.read_state("completion")[0].active_turn_id is not None:
            Event().wait(0.02)
        assert store.read_state("completion")[0].active_turn_id is None and len(calls) == 2
        assert not store.peek_inbox("completion") and provider.submits == 1
    finally:
        for server in (first, second):
            server.run_adapter.close()
            server.run_adapter.join()
            server.kernel.close()
            assert all(h.done() and not h._thread.is_alive() for h in server.kernel.handles)
            assert all(not p.is_alive() for p in server.run_adapter._pumps)


@pytest.mark.persistent_store
def test_scoped_connections_cannot_reuse_a_prior_transaction_token(store):
    from vv_agent.session.records import InboxItem, SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("scope-token", "test", "."), consumers=("events",))
    with store._connection_scope(), store.atomic() as retained:
        retained.push("scope-token", InboxItem("first", "user", {"content": "first"}))
    with store._connection_scope(), store.atomic(), pytest.raises(RuntimeError, match="original caller transaction"):
        retained.push("scope-token", InboxItem("stale", "user", {"content": "stale"}))
    assert [x.item.input_id for x in store.peek_inbox("scope-token")] == ["first"]
