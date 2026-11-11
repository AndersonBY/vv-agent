"""The necessary RunHandle seam reuses owned E for foreground children only."""

import json
from dataclasses import replace
from threading import Event, get_ident

import pytest

from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_handle import RunHandle
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.parametrize("background", [False])
def test_child_execution_identity_and_hook_scope(store, database, monkeypatch, background):
    start(store)
    kernel = SessionDriver(store=store)
    loads = []
    root_hooks = []
    opened = []
    projection_opens = []
    import vv_agent.run_handle as handle_module

    actual_checkout = handle_module.checkout

    def observed_checkout(source, **kwargs):
        if kwargs.get("role") == "projection":
            projection_opens.append(get_ident())
        return actual_checkout(source, **kwargs)

    monkeypatch.setattr(handle_module, "checkout", observed_checkout)
    model_entries = []
    model_returns = []
    initialize = _Driver.__init__
    factory = store.execution_store

    def loaded(driver, *a, **kw):
        initialize(driver, *a, **kw)
        assert driver.scope.successful_ready and not driver.scope.lost
        loads.append((driver.sid, driver.store.connection, get_ident()))

    def execution():
        opened.append(get_ident())
        return factory()

    monkeypatch.setattr(_Driver, "__init__", loaded)
    monkeypatch.setattr(store, "execution_store", execution)

    def hook(point, record, *, tx):
        if record is not None:
            root_hooks.append(record.session_id)

    rt = parent_runtime(database, background=background, hook=hook)

    def child_runtime(current, sid):
        assert current.connection is not store.connection if getattr(store, "path", None) != ":memory:" else current is store

        def model(request):
            model_entries.append(request)
            response = LLMResponse("child done")
            model_returns.append(response)
            return response

        child = rt.for_agent(rt.agent, replace(rt.config, stream=None))
        child.llm = ScriptedLLM([model])
        assert child.hook is not rt.hook
        return child

    rt.child_runtime = child_runtime
    handle = RunHandle(kernel, "s", "s/turn/initial", rt, "events")
    kernel.handles.append(handle)
    try:
        handle.start()
        result = handle.result(timeout=10)
        for owned in kernel.handles:
            owned.join(10)
            assert owned.done() and owned._error is None
        assert result.final_output == "parent done"
        parent = [entry for entry in loads if entry[0] == "s"]
        child = [entry for entry in loads if entry[0] == "child"]
        assert parent and child
        assert len(model_entries) == len(model_returns) == 1
        assert all(entry[1] is child[0][1] and entry[2] == child[0][2] for entry in child)
        child_records = store.read_state("child")[1]
        assert (
            sum(
                r.record.kind == "op_completed" and r.record.payload.get("result", {}).get("content") == "child done"
                for r in child_records
            )
            == 1
        )
        assert root_hooks and set(root_hooks) == {"s"}
        if background:
            assert child[0][2] != parent[0][2]
            if getattr(store, "path", None) != ":memory:":
                assert child[0][1] is not parent[0][1]
            assert len(opened) >= 2
        else:
            assert child[0][1] is parent[0][1] and child[0][2] == parent[0][2]
            assert len(opened) - len(projection_opens) == 1
        print(
            "CO2-F04-CHILD",
            type(store).__name__,
            background,
            [(sid, id(raw), thread) for sid, raw, thread in loads],
            "checkouts",
            len(opened),
            flush=True,
        )
    finally:
        kernel.close()


@pytest.mark.parametrize("timing", ["before_second_request", "frozen_before_terminal", "after_terminal"])
def test_background_child_ownership_and_notification_order(store, database, monkeypatch, timing):
    start(store)
    kernel = SessionDriver(store=store)
    loads, opened, root_hooks, calls, returns, deliveries = [], [], [], [], [], []
    admission, second_entered, terminal, delivered = Event(), Event(), Event(), Event()
    clients = {}
    initialize, factory, complete = _Driver.__init__, store.execution_store, ScriptedLLM.complete
    schedule, delivery = RunHandle._schedule_background, RunHandle._deliver_child

    def wait(event):
        assert not kernel._background_lock._is_owned()
        if getattr(store, "path", None) == ":memory:":
            assert not store._connection_lock._is_owned()
        assert event.wait(10), f"missing natural boundary: {timing}"

    def loaded(driver, *args, **kwargs):
        initialize(driver, *args, **kwargs)
        assert driver.scope.successful_ready and not driver.scope.lost
        loads.append((driver.sid, driver.store.connection, get_ident()))

    def execution():
        opened.append(get_ident())
        return factory()

    def model(client, request):
        sid = clients[id(client)]
        calls.append((sid, request, get_ident()))
        response = complete(client, request)
        returns.append((sid, response, get_ident()))
        return response

    def scheduled(handle):
        value = schedule(handle)
        # Scheduling has returned outside B/S; the child now owns its thread.
        if handle.session_id == "s" and timing == "before_second_request" and admission.is_set():
            wait(delivered)
        return value

    def observed_delivery(handle, sid, runtime):
        value = delivery(handle, sid, runtime)
        if sid == "child" and not delivered.is_set():
            items = [i.item for i in handle._execution_store.peek_inbox("s") if i.item.kind == "child_result"]
            if items:
                assert len(items) == 1
                deliveries.append(items[0])
                delivered.set()
        return value

    monkeypatch.setattr(_Driver, "__init__", loaded)
    monkeypatch.setattr(store, "execution_store", execution)
    monkeypatch.setattr(ScriptedLLM, "complete", model)
    monkeypatch.setattr(RunHandle, "_schedule_background", scheduled)
    monkeypatch.setattr(RunHandle, "_deliver_child", observed_delivery)

    def hook(point, record, *, tx):
        if record is None:
            return
        root_hooks.append(record.session_id)
        if point == "after_commit" and record.kind == "op_completed" and "/tool/" in (record.operation_id or ""):
            assert tx is None
            admission.set()
        if point == "after_commit" and record.kind == "turn_ended":
            assert tx is None
            terminal.set()
            if timing == "after_terminal":
                wait(delivered)

    rt = parent_runtime(database, background=True, hook=hook)
    assert rt.ttl_ms == 15000
    original_second = rt.llm.steps[1]

    def parent_second(request):
        second_entered.set()
        if timing == "frozen_before_terminal":
            wait(delivered)
        return original_second

    rt.llm.steps[1] = parent_second
    if timing == "frozen_before_terminal":
        rt.llm.steps.append(LLMResponse("parent done"))
    clients[id(rt.llm)] = "s"

    def child_runtime(current, sid):
        assert sid == "child"
        assert current is store if getattr(store, "path", None) == ":memory:" else current.connection is not store.connection

        def child_model(request):
            if timing == "frozen_before_terminal":
                wait(second_entered)
            elif timing == "after_terminal":
                wait(terminal)
            return LLMResponse("child done")

        child = rt.for_agent(rt.agent, replace(rt.config, stream=None))
        assert child.hook is not rt.hook and child.hook is not hook
        child.llm = ScriptedLLM([child_model])
        clients[id(child.llm)] = sid
        return child

    rt.child_runtime = child_runtime
    handle = RunHandle(kernel, "s", "s/turn/initial", rt, "events")
    kernel.handles.append(handle)
    try:
        handle.start()
        result = handle.result(timeout=10)
        assert result.final_output == "parent done"
        # Await existing owners' normal completion; no recovery or additional wake.
        for owned in kernel.handles:
            assert owned.result(timeout=10).final_output == ("parent done" if owned is handle else "child done")
            assert owned.done() and owned._error is None
        assert rt.hook is hook
        assert len(kernel.handles) == 2
        assert delivered.is_set() and terminal.is_set()
        parent = [entry for entry in loads if entry[0] == "s"]
        child = [entry for entry in loads if entry[0] == "child"]
        assert parent and child and child[0][2] != parent[0][2]
        assert all(raw is child[0][1] and thread == child[0][2] for _, raw, thread in child)
        assert all(raw is parent[0][1] and thread == parent[0][2] for _, raw, thread in parent)
        if getattr(store, "path", None) == ":memory:":
            assert parent[0][1] is child[0][1] is store.connection
        else:
            assert parent[0][1] is not child[0][1]
        assert len(opened) >= 2
        assert root_hooks and set(root_hooks) == {"s"}

        parent_calls = [request for sid, request, _ in calls if sid == "s"]
        parent_returns = [response for sid, response, _ in returns if sid == "s"]
        expected = 3 if timing == "frozen_before_terminal" else 2
        assert len(parent_calls) == len(parent_returns) == expected
        assert sum(sid == "child" for sid, _, _ in calls) == sum(sid == "child" for sid, _, _ in returns) == 1
        assert [thread for sid, _, thread in calls if sid == "child"] == [thread for sid, _, thread in returns if sid == "child"]
        notifications = [sum(m.role == "user" and "child done" in m.content for m in r.messages) for r in parent_calls]
        assert (
            notifications
            == ({"before_second_request": [0, 1], "frozen_before_terminal": [0, 0, 1], "after_terminal": [0, 0]}[timing])
        )
        assert not rt.llm.steps

        state, records, _ = store.read_state("s")
        _, child_records, _ = store.read_state("child")
        ended = [r for r in records if r.record.kind == "turn_ended"]
        child_completed = [
            r
            for r in child_records
            if r.record.kind == "op_completed" and r.record.payload.get("result", {}).get("content") == "child done"
        ]
        assert len(child_completed) == 1
        assert len([r for r in child_records if r.record.kind == "turn_ended"]) == 1
        assert len(ended) == len(state.turns) == 1 and state.active_turn_id is None
        assert ended[0].record.turn_id == handle.run_id == "s/turn/initial"
        assert ended[0].record.payload["status"] == "completed"
        assert ended[0].record.payload["result"] == "parent done"
        generation = state.turns[handle.run_id].start.payload["generation"]
        plans = [
            r
            for r in records
            if r.record.kind == "op_planned"
            and r.record.payload["op_kind"] == "model"
            and r.record.payload["purpose"] == "primary"
        ]
        assert len(plans) == expected
        completed = [
            r
            for r in records
            if r.record.kind == "op_completed" and r.record.operation_id in {p.record.operation_id for p in plans}
        ]
        assert len(completed) == expected
        assert [r.record.payload["result"]["content"] for r in completed] == [r.content for r in parent_returns]
        applied = [r for r in records if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "child_result"]
        assert len(applied) == len(deliveries) == 1
        assert deliveries[0].target_turn_id == handle.run_id and deliveries[0].generation == generation
        if timing == "before_second_request":
            assert applied[0].seq < plans[1].seq < ended[0].seq
        elif timing == "frozen_before_terminal":
            assert plans[1].seq < completed[1].seq < applied[0].seq < plans[2].seq < ended[0].seq
        else:
            assert ended[0].seq < applied[0].seq
            assert applied[0].record.payload["reason"] == "late background completion: audit only"
        assert not any(r.record.kind == "op_unknown" for r in records + child_records)
        print(
            "CO2-FIX2A-B2",
            json.dumps(
                {
                    "timing": timing,
                    "backend": type(store).__name__,
                    "notifications": notifications,
                    "model_entries": [sid for sid, _, _ in calls],
                    "model_returns": [sid for sid, _, _ in returns],
                    "parent_contents": [r.content for r in parent_returns],
                    "parent_requests": [[m.to_dict() for m in request.messages] for request in parent_calls],
                    "model_entry_threads": [(sid, thread) for sid, _, thread in calls],
                    "model_return_threads": [(sid, thread) for sid, _, thread in returns],
                    "parent_plans": [p.seq for p in plans],
                    "parent_completions": [r.seq for r in completed],
                    "notification_applied": applied[0].seq,
                    "notification_reason": applied[0].record.payload["reason"],
                    "terminal": ended[0].seq,
                    "child_completion": child_completed[0].seq,
                    "loads": [(sid, id(raw), thread) for sid, raw, thread in loads],
                    "checkouts": len(opened),
                    "handles": len(kernel.handles),
                    "turn_id": handle.run_id,
                    "generation": generation,
                }
            ),
            flush=True,
        )
        # Restored original hook must not access the worker's closed E.
        drive(store, "s", runtime=rt)
        assert store.read_state("s")[1] == records
        assert len(calls) == len(returns) == expected + 1
    finally:
        kernel.close()
        if getattr(store, "path", None) == ":memory:":
            assert store.connection.execute("SELECT 1").fetchone()[0] == 1
