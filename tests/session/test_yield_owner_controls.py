"""A3's same-input concurrency under the new explicit-owner protocol."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread, current_thread

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session.records import InboxItem

from .test_yield_local_recovery import parked_tree
from .test_yieldable_waits import waiting_runtime


@pytest.mark.sqlite_file
def test_eight_same_input_pushes_then_one_owner_run(store, database, monkeypatch):
    rt, _, _, driver, parent = parked_tree(store, database, monkeypatch)
    try:
        parent.start()
        parent.result(timeout=5)
        barrier = Barrier(8)
        item = InboxItem("concurrent-same", "steer", {"content": "once"}, parent.run_id, 0)

        def submit():
            barrier.wait(5)
            return driver.push("s", item)

        before = len(driver.handles)
        with ThreadPoolExecutor(8) as pool:
            receipts = [f.result(5) for f in [pool.submit(submit) for _ in range(8)]]
        assert len(driver.handles) == before and driver._wake("s") is False
        resumed, observed = waiting_runtime(database)
        resumed.llm.steps.pop(0)
        monkeypatch.setattr(resumed, "child_runtime", rt.child_runtime)
        successor = RunHandle(driver, "s", parent.run_id, resumed, None)
        driver.handles.append(successor)
        successor.start()
        successor.result(timeout=5)
        applied = [
            r
            for r in store.read_state("s")[1]
            if r.record.kind == "input_applied" and r.record.payload["input"]["input_id"] == item.input_id
        ]
        fresh = sum(not x.replayed for x in receipts)
        print(
            "EXPLICIT_OWNER_SAME_INPUT",
            dict(producers=8, new_receipts=fresh, applied=len(applied), observed=len(observed), handles=len(driver.handles)),
        )
        assert fresh == len(applied) == len(observed) == 1
        assert before < len(driver.handles) <= before + 5 and not store.peek_inbox("s")
    finally:
        driver.close()


@pytest.mark.sqlite_file
def test_child_previous_swapped_to_still_active_worker(store, database, monkeypatch):
    rt, _, _, driver, parent = parked_tree(store, database, monkeypatch)
    sid = "yield-child-0"
    old = driver._background[sid]
    child = old._background_child[1]
    driver.push(sid, InboxItem("identity-active", "steer", {"content": "stale"}, child["turn_id"], 99))
    a_ready, a_release, b_ready, b_release = Event(), Event(), Event(), Event()
    original_binding = rt.child_runtime
    original_check = RunHandle._drive_pass

    def binding(*a, **kw):
        result = original_binding(*a, **kw)
        if current_thread().name == "scheduler-A":
            a_ready.set()
            assert a_release.wait(8)
        return result

    def check(self):
        result = original_check(self)
        if self.session_id == sid and self is not old and not b_ready.is_set():
            b_ready.set()
            assert b_release.wait(8)
        return result

    monkeypatch.setattr(rt, "child_runtime", binding)
    monkeypatch.setattr(RunHandle, "_drive_pass", check)
    a = Thread(target=parent._schedule_child, args=(child,), kwargs={"independent": True}, name="scheduler-A")
    try:
        a.start()
        assert a_ready.wait(5)
        parent._schedule_child(child, independent=True)
        assert b_ready.wait(5)
        replacement = driver._background[sid]
        assert not replacement.done()
        before = len(driver.handles)
        a_release.set()
        a.join(5)
        assert not a.is_alive()
        assert len(driver.handles) == before and driver._background[sid] is replacement
        print("CHILD_SWAP_ACTIVE", dict(handles=before, unchanged=True, replacement_active=not replacement.done()))
        b_release.set()
        replacement.result(timeout=5)
    finally:
        a_release.set()
        b_release.set()
        a.join(5)
        driver.close()
