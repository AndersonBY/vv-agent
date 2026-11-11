"""Local drive ownership, provider deadlines and durable child delivery recovery."""

from threading import Event

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .test_batch_children import complete, rows
from .test_children import child_task
from .test_recovery_matrix import runtime, start
from .test_yieldable_waits import YieldProvider, waiting_runtime


def parked_tree(store, database, monkeypatch):
    start(store)
    rt, observed = waiting_runtime(database)
    provider = YieldProvider(declared=False, callback=None)
    child_rt = runtime(
        database,
        [LLMResponse("", [ToolCall("work", "child_task", {})]), LLMResponse("child done")],
        [child_task],
        providers={"child_task": provider},
    )
    drive(store, "s", runtime=rt)
    drive(store, "yield-child-0", runtime=child_rt)
    monkeypatch.setattr(
        rt,
        "child_runtime",
        lambda *_: runtime(database, [LLMResponse("child done")], [child_task], providers={"child_task": provider}),
    )
    driver = SessionDriver(store=store)
    parent = RunHandle(driver, "s", "s/turn/initial", rt, None)
    driver.handles.append(parent)
    with store.atomic() as tx:
        tx.push("yield-child-0", InboxItem("initial-stale", "steer", {"content": "stale"}, "yield-child-0/turn/start", 99))
    parent._schedule_background()
    driver._background["yield-child-0"].result(timeout=5)
    return rt, observed, provider, driver, parent


@pytest.mark.sqlite_file
def test_steer_after_last_progress_snapshot_is_not_stranded(store, database, monkeypatch):
    _rt, observed, _provider, driver, parent = parked_tree(store, database, monkeypatch)
    reads = []
    injected = Event()
    original = parent._position

    def position(sid):
        snapshot = original(sid)
        if sid == "s":
            reads.append(snapshot)
            if len(reads) == 2:
                # Real input after the DB snapshot, before handle completion.
                parent.steer("real input after no-progress snapshot")
                injected.set()
        return snapshot

    monkeypatch.setattr(parent, "_position", position)
    try:
        parent.start()
        parent.result(timeout=5)
        assert injected.is_set()
        inbox = [i.item.input_id for i in store.peek_inbox("s")]
        print(
            "PARENT_EXIT_RACE",
            {
                "snapshot_reads": len(reads),
                "parent_done": parent.done(),
                "observed": len(observed),
                "pending_inputs": len(inbox),
                "active_handles": sum(not h.done() for h in driver.handles),
            },
        )
        assert len(observed) == 1, "real steer arrived before parent done but no driver remains to consume it"
    finally:
        driver.close()


@pytest.mark.sqlite_file
def test_child_exit_rechecks_new_provider_input(store, database, monkeypatch):
    rt, _observed, provider, driver, parent = parked_tree(store, database, monkeypatch)
    # A new child driver exits at the same no-progress snapshot, with a real receipt arriving.
    with store.atomic() as tx:
        tx.push("yield-child-0", InboxItem("wake-stale", "steer", {"content": "stale"}, "yield-child-0/turn/start", 99))
    injected = Event()
    h = provider.handle

    def child_hook(point, record):
        if point == "after_drive" and not injected.is_set():
            injected.set()
            with store.atomic() as tx:
                tx.push(
                    "yield-child-0",
                    InboxItem(
                        "real-provider-done",
                        "provider_result",
                        {
                            "operation_id": h["operation_id"],
                            "attempt": h["attempt"],
                            "request_digest": h["request_digest"],
                            "provider_binding": h["provider"],
                            "result": ToolExecutionResult("work", "done").to_dict(),
                            "usage": {},
                            "evidence": [h["evidence"]],
                        },
                        "yield-child-0/turn/start",
                        0,
                    ),
                )

    monkeypatch.setattr(
        rt,
        "child_runtime",
        lambda *_: runtime(
            database, [LLMResponse("child done")], [child_task], providers={"child_task": provider}, hook=child_hook
        ),
    )
    try:
        parent._schedule_background()
        for _ in range(5):
            driver._background["yield-child-0"].join(5)
            if rows(store, "yield-child-0", "turn_ended"):
                break
        assert injected.is_set()
        print(
            "CHILD_EXIT_RACE", {"child_ended": bool(rows(store, "yield-child-0", "turn_ended")), "handles": len(driver.handles)}
        )
        assert rows(store, "yield-child-0", "turn_ended")
        assert any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
    finally:
        driver.close()


@pytest.mark.sqlite_file
def test_child_terminal_without_delivery_is_recovered(store, database, monkeypatch):
    start(store)
    rt, observed = waiting_runtime(database)
    monkeypatch.setattr(rt, "child_runtime", lambda *_: runtime(database, []))
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    assert not any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
    driver = SessionDriver(store=store)
    parent = RunHandle(driver, "s", "s/turn/initial", rt, None)
    driver.handles.append(parent)
    try:
        parent.start()
        parent.result(timeout=5)
        print(
            "TERMINAL_WITHOUT_DELIVERY",
            {
                "child_runnable": store.is_runnable("yield-child-0"),
                "parent_done": parent.done(),
                "observed": len(observed),
                "handles": len(driver.handles),
            },
        )
        assert len(observed) == 1, "terminal child never delivered to retained foreground parent"
    finally:
        driver.close()


@pytest.mark.sqlite_file
def test_due_provider_poll_reschedules_without_log_or_inbox_change(store, database, monkeypatch):
    from vv_agent.session.providers import Definitive

    # Database time controls both provider poll admission and is_runnable.
    monkeypatch.setattr(type(store), "clock_sql", "1000000")
    _rt, _observed, provider, driver, parent = parked_tree(store, database, monkeypatch)
    first = driver._background["yield-child-0"]

    def position():
        page = store.read("yield-child-0", limit=1)
        return page.head_seq, page.inbox_seq

    original_position = position()
    monkeypatch.setattr(
        provider,
        "query",
        lambda _h: Definitive(ToolExecutionResult("work", "finished at provider").to_dict(), ("provider-accepted",)),
    )
    monkeypatch.setattr(type(store), "clock_sql", "1060000")
    assert store.is_runnable("yield-child-0")
    assert first._drive_positions["yield-child-0"] == original_position
    try:
        for _ in range(3):
            parent._schedule_background()
        active = driver._background["yield-child-0"]
        active.join(5)
        print(
            "DUE_CHILD_POLL",
            {
                "runnable": store.is_runnable("yield-child-0"),
                "position_unchanged": position() == original_position,
                "same_handle": active is first,
                "child_ended": bool(rows(store, "yield-child-0", "turn_ended")),
            },
        )
        assert rows(store, "yield-child-0", "turn_ended"), "due provider poll suppressed forever by same log/inbox position"
    finally:
        driver.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("independent", [False, True])
def test_due_poll_still_accepted_parks_until_next_store_deadline(store, database, monkeypatch, independent):
    monkeypatch.setattr(type(store), "clock_sql", "1000000")
    _rt, _observed, provider, driver, parent = parked_tree(store, database, monkeypatch)
    try:
        first = driver._background["yield-child-0"]
        position = first._position("yield-child-0")
        before = provider.queries
        monkeypatch.setattr(type(store), "clock_sql", "1060000")
        # Exercise the shared scheduling path for both ordinary background and yieldable children.
        child = first._background_child[1] | {"background": not independent}
        parent._schedule_child(child, independent=independent)
        second = driver._background["yield-child-0"]
        second.result(timeout=5)
        assert second is not first
        assert provider.queries == before + 1
        assert second._position("yield-child-0") == position
        delay = store.next_drive_delay_ms("yield-child-0")
        assert delay is not None and delay > 0
        count = len(driver.handles)
        for _ in range(3):
            parent._schedule_child(child, independent=independent)
        driver.close()
        assert count <= len(driver.handles) <= count + 3
        assert provider.queries == before + 1
        monkeypatch.setattr(type(store), "clock_sql", str(1060000 + delay))
        parent._schedule_child(child, independent=independent)
        third = driver._background["yield-child-0"]
        third.result(timeout=5)
        assert third is not second
        assert provider.queries == before + 2
        assert store.next_drive_delay_ms("yield-child-0") > 0
        print(
            "DUE_ACCEPTED_PARK",
            {"independent": independent, "queries": provider.queries - before, "delay_ms": delay, "handles": len(driver.handles)},
        )
    finally:
        driver.close()


@pytest.mark.sqlite_file
@pytest.mark.parametrize("arrival", ["after_final_check", "after_done"])
def test_root_exit_input_wake_stays_with_execution_owner(store, database, monkeypatch, arrival):
    from threading import Thread

    _rt, observed, _provider, driver, parent = parked_tree(store, database, monkeypatch)
    committed = Event()
    producer = None
    if arrival == "after_final_check":
        original_check = parent._drive_pass

        def check():
            nonlocal producer
            snapshot = original_check()
            if not committed.is_set():

                def submit():
                    parent.steer("input after final exit check")
                    committed.set()

                producer = Thread(target=submit)
                producer.start()
                assert committed.wait(5)
            return snapshot

        monkeypatch.setattr(parent, "_drive_pass", check)
    try:
        parent.start()
        parent.result(timeout=5)
        if producer is not None:
            producer.join(5)
            assert not producer.is_alive()
            assert len(observed) == 1
            commits = {r.commit_id for r in store.read_state("s")[1]}
            assert 2 <= len(driver.handles) <= len(commits) + 2
            assert sum(h.session_id == "s" for h in driver.handles) == 1
        else:
            before = len(driver.handles)
            parent.steer("input after root parked")
            assert len(driver.handles) == before  # push never creates a root run
            assert not driver._wake("s")
            resumed, observed = waiting_runtime(database)
            resumed.llm.steps.pop(0)
            monkeypatch.setattr(resumed, "child_runtime", _rt.child_runtime)
            successor = RunHandle(driver, "s", parent.run_id, resumed, None)
            driver.handles.append(successor)
            successor.start()
            successor.result(timeout=5)
        assert len(observed) == 1
        assert any("input after" in str(m.content) for m in observed[0].messages)
        assert not store.peek_inbox("s")
        assert sum(r.record.payload["input"]["kind"] == "steer" for r in rows(store, kind="input_applied")) == 1
        print(
            "ROOT_HANDOFF",
            {"arrival": arrival, "observed": len(observed), "active_handles": sum(not h.done() for h in driver.handles)},
        )
    finally:
        if producer is not None:
            producer.join(5)
        driver.close()


def child_completed_before_delivery_worker(database, pipe):
    from .conftest import open_store

    with open_store(database) as store:
        driver = SessionDriver(store=store)
        rt = runtime(database, [LLMResponse("child completed before exit")])

        class CutHandle(RunHandle):
            def _deliver_child(self, sid, runtime) -> int:
                assert rows(store, sid, "turn_ended")
                pipe.send("terminal-committed-before-delivery")
                pipe.recv()
                raise AssertionError("worker must be killed before delivery")

        handle = CutHandle(driver, "yield-child-0", "yield-child-0/turn/start", rt, None)
        handle._deliver_to_parent = True
        handle.start()
        handle.result(timeout=20)


@pytest.mark.persistent_store
def test_process_death_after_child_terminal_before_delivery_recovers_parent(store, database, monkeypatch):
    import multiprocessing

    from .test_recovery_matrix import force_expire

    start(store)
    rt, _ = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    process = ctx.Process(target=child_completed_before_delivery_worker, args=(database, child))
    process.start()
    child.close()
    driver = SessionDriver(store=store)
    try:
        assert pipe.poll(20)
        assert pipe.recv() == "terminal-committed-before-delivery"
        process.kill()
        process.join(10)
        assert process.exitcode is not None and process.exitcode != 0
        terminal_bytes = rows(store, "yield-child-0", "turn_ended")[0].record.encode()
        assert not any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
        force_expire(database)
        resumed, observed = waiting_runtime(database)
        resumed.llm.steps.pop(0)

        def unexpected_worker(*args):
            raise AssertionError("terminal child does not need a new runtime")

        monkeypatch.setattr(resumed, "child_runtime", unexpected_worker)
        parent = RunHandle(driver, "s", "s/turn/initial", resumed, None)
        driver.handles.append(parent)
        parent.start()
        parent.result(timeout=5)
        assert len(observed) == 1
        assert len(driver.handles) == 1
        assert rows(store, "yield-child-0", "turn_ended")[0].record.encode() == terminal_bytes
        # Recovery repeats delivery scans, but push + ACK has only one business input.
        parent._schedule_background()
        with store.atomic() as tx:
            assert tx.consumer_batch("yield-child-0", "child_delivery") is None
        notices = [r for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
        assert len(notices) == 1
        assert not any(i.item.kind == "child_result" for i in store.peek_inbox("s"))
        print("CHILD_CRASH_RECOVERED", {"exitcode": process.exitcode, "notices": len(notices), "new_child_workers": 0})
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        pipe.close()
        driver.close()
