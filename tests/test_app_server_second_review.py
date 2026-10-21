"""Cross-node recovery and notification isolation through real producers."""

import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event

import pytest
from test_app_server_distributed import node, request, submission
from test_app_server_review_regressions import record_output

from vv_agent import Agent, ScriptedModelProvider
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import Runtime
from vv_agent.types import LLMResponse, ToolCall


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def lifecycle(messages):
    return [(m["method"], m["params"]["turnId"]) for m in messages if m.get("method") in {"turn/started", "turn/completed"}]


def provider():
    return ScriptedModelProvider.from_callback("test", "m", lambda r: LLMResponse("done"))


def test_abandoned_lease_runs_accepted_turn_after_expiry_without_external_wake(tmp_path, monkeypatch):
    calls = []
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            ScriptedModelProvider.from_callback("test", "m", lambda r: calls.append(r.model) or LLMResponse("done")),
        )
        messages = record_output(server, monkeypatch)
        request(server, 1, "thread/start", {"inputId": "thread"})
        assert server.kernel.store.acquire("thread", owner="dead-process", ttl_ms=2000)
        receipt = request(server, 2, "turn/start", submission())["result"]
        assert receipt["accepted"]
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns), timeout=4)
        server.run_adapter.join()
        state = server.kernel.store.read_state("thread")[0]
        assert state.turns[receipt["turnId"]].ended and calls == ["m"]
        assert not server.kernel.store.peek_inbox("thread")
        assert lifecycle(messages) == [("turn/started", receipt["turnId"]), ("turn/completed", receipt["turnId"])]


def test_resumed_turn_after_restart_retries_dead_lease_without_external_wake(tmp_path):
    models = ScriptedModelProvider.new(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose?"})]),
            LLMResponse("done"),
        ],
    )
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", models)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        a.run_adapter.join()
        a.processor.disconnect_connection("A")
        b = node(stack, tmp_path / "store", "B", models)
        _, agent, config = b.run_adapter._binding("thread")
        b.kernel.answer("thread", b.kernel.runtime(agent, config), "answer", "reply")
        b.kernel.store.acquire("thread", owner="dead-before-restart", ttl_ms=200)
        assert request(b, 3, "turn/resume", {"threadId": "thread", "turnId": tid})["result"]["status"] == "running"
        eventually(lambda: bool(b.kernel.store.read_state("thread")[0].turns[tid].ended))
        b.run_adapter.join()
        assert b.store.read_thread("thread").turns[0].result["finalOutput"] == "done"


@pytest.mark.parametrize("disconnect", [False, True])
def test_resubscribe_starts_at_head_and_releases_positions(tmp_path, monkeypatch, disconnect):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = node(stack, tmp_path / "store", "B", provider())
        messages = record_output(b, monkeypatch)
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        if disconnect:
            b.processor.disconnect_connection("B")
        else:
            request(b, 3, "thread/unsubscribe", {"threadId": "thread"})
        assert len(b.run_adapter._notification_positions) == 0
        previous = None
        for i in range(3):
            previous = request(a, f"t{i}", "turn/start", submission(identity=f"p{i}", expected=previous))["result"]["turnId"]
            a.run_adapter.join()
        if disconnect:
            b.router.register_transport(b.transport)
            request(b, "init-again", "initialize", {"clientInfo": {"name": "B"}})
        messages.clear()
        request(b, 9, "thread/resume", {"threadId": "thread"})
        b.run_adapter.join()
        assert lifecycle(messages) == []
        assert len(b.run_adapter._notification_positions) == 1
        for i in range(10):
            sid = f"other{i}"
            request(b, f"create{i}", "thread/start", {"inputId": sid})
            request(b, f"unsub{i}", "thread/unsubscribe", {"threadId": sid})
        assert len(b.run_adapter._notification_positions) == 1


def test_observer_recovers_from_transient_read_error_with_backoff(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = node(stack, tmp_path / "store", "B", provider())
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        messages = record_output(b, monkeypatch)
        read = b.kernel.store.read
        failures = []

        def transient(sid, **kwargs):
            if not failures:
                failures.append(time.monotonic())
                raise RuntimeError("transient reset")
            return read(sid, **kwargs)

        monkeypatch.setattr(b.kernel.store, "read", transient)
        eventually(lambda: bool(failures))
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        a.run_adapter.join()
        eventually(lambda: ("turn/completed", tid) in lifecycle(messages), timeout=3)
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        assert sum(m.get("method") == "error/warning" for m in messages) == 1
        assert b.run_adapter._notification_thread.is_alive()


@pytest.mark.parametrize("writer", ["local", "process"])
@pytest.mark.parametrize("publisher", ["observer", "local_pump"])
def test_blocked_notification_does_not_hold_store_or_admission_lock(tmp_path, monkeypatch, writer, publisher):
    blocked, release = Event(), Event()
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = node(stack, tmp_path / "store", "B", provider()) if publisher == "observer" else a
        stack.callback(release.set)
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        write = b.transport.write_outbound

        def slow_write(message):
            if message.get("method") == ("turn/completed" if publisher == "observer" else "item/completed"):
                blocked.set()
                assert release.wait(10)
            write(message)

        monkeypatch.setattr(b.transport, "write_outbound", slow_write)
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        if publisher == "observer":
            a.run_adapter.join()
        assert blocked.wait(5)
        if writer == "process":
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        """
import sys
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.records import InboxItem
with SQLiteStore.standalone(sys.argv[1]) as store:
    store.connection.execute('PRAGMA busy_timeout=100')
    with store.atomic() as tx:
        tx.push('thread', InboxItem('selection', 'model_selection', {'selection': None}))
""",
                        str(tmp_path / "store"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                assert result.returncode == 0, result.stderr
            finally:
                release.set()
            return
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                b.run_adapter.submit_input,
                connection_id="B",
                thread_id="thread",
                input_id="next",
                expected_turn_id=tid,
                kind="follow_up",
                input=[],
            )
            try:
                assert future.result(timeout=0.5)[0]["accepted"]
            finally:
                release.set()


def test_idle_read_backoff_is_per_session_while_one_session_is_busy(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = node(stack, tmp_path / "store", "B", provider())
        for sid in ["busy", *(f"idle{i}" for i in range(8))]:
            request(a, f"new{sid}", "thread/start", {"inputId": sid})
            request(b, f"sub{sid}", "thread/resume", {"threadId": sid})
        reads = Counter()
        read = b.kernel.store.read

        def counted(sid, **kwargs):
            reads[sid] += 1
            return read(sid, **kwargs)

        monkeypatch.setattr(b.kernel.store, "read", counted)
        time.sleep(2.1)
        reads.clear()
        end = time.monotonic() + 2.2
        i = 0
        while time.monotonic() < end:
            # Committed inbox records provide continuing activity without provider I/O.
            a.kernel.push("busy", InboxItem(f"select{i}", "model_selection", {"selection": None}))
            a.kernel.control("busy", "suspend", f"suspend/{i}")
            i += 1
            time.sleep(0.06)
        assert max(reads[f"idle{i}"] for i in range(8)) <= 3, reads
        assert reads["busy"] > 4, reads


def test_error_after_terminal_record_emits_completed_only_once(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)
        complete = server.run_adapter._complete_turn

        def failed_stream(connection, started, **kwargs):
            complete(connection, started, result=None, error=RuntimeError("late stream failure"))

        monkeypatch.setattr(server.run_adapter, "_complete_turn", failed_stream)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        assert next(m for m in messages if m.get("method") == "turn/completed")["params"]["status"] == "completed"


def test_persistent_observer_failure_is_rate_limited_and_does_not_stop_other_sessions(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = node(stack, tmp_path / "store", "B", provider())
        for sid in ("broken", "healthy"):
            request(a, sid, "thread/start", {"inputId": sid})
            request(b, sid, "thread/resume", {"threadId": sid})
        messages = record_output(b, monkeypatch)
        calls = []
        read = b.kernel.store.read

        def fail_one(sid, **kw):
            if sid == "broken":
                calls.append(time.monotonic())
                raise RuntimeError("database unavailable")
            return read(sid, **kw)

        monkeypatch.setattr(b.kernel.store, "read", fail_one)
        request(a, 3, "turn/start", submission(sid="healthy"))
        a.run_adapter.join()
        eventually(lambda: ("turn/completed", "healthy/turn/prompt") in lifecycle(messages))
        time.sleep(1.2)
        assert len(calls) <= 6
        assert sum(m.get("method") == "error/warning" for m in messages) <= 2
        assert b.run_adapter._notification_thread.is_alive()
        monkeypatch.undo()


def test_expiry_retry_failure_warns_once_with_two_acquires(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)
        server.kernel.store.acquire("thread", owner="dead", ttl_ms=150)
        acquire = server.kernel.store.acquire
        attempts = []

        def cannot_acquire(*args, **kw):
            attempts.append(time.monotonic())
            # Fail the second acquire even though the lease really has expired.
            return acquire(*args, **kw) if len(attempts) == 1 else None

        monkeypatch.setattr(server.kernel.store, "acquire", cannot_acquire)
        request(server, 2, "turn/start", submission())
        eventually(lambda: any(m.get("params", {}).get("code") == "lease_retry_exhausted" for m in messages))
        server.run_adapter.join()
        assert len(attempts) == 2 and attempts[1] - attempts[0] >= 0.1
        assert lifecycle(messages) == []
        assert not server.run_adapter._lease_retries


def test_renewed_holder_defers_retry_until_new_expiry_without_another_acquire(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        holder = server.kernel.store.acquire("thread", owner="live", ttl_ms=200)
        attempts = []
        acquire = server.kernel.store.acquire
        monkeypatch.setattr(server.kernel.store, "acquire", lambda *a, **kw: attempts.append(kw) or acquire(*a, **kw))
        request(server, 2, "turn/start", submission())
        server.run_adapter.join()
        server.kernel.store.renew(holder, ttl_ms=1000)
        time.sleep(0.35)
        assert server.run_adapter._lease_retries
        assert len(attempts) == 1
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns))
        server.run_adapter.join()
        assert len(attempts) == 2
        state = server.kernel.store.read_state("thread")[0]
        assert state.turns["thread/turn/prompt"].ended


def test_first_subscriber_during_active_turn_receives_started(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def model(r):
        entered.set()
        assert release.wait(5)
        return LLMResponse("done")

    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", model))
        b = node(stack, tmp_path / "store", "B", provider())
        stack.callback(release.set)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        messages = record_output(b, monkeypatch)
        request(b, 3, "thread/resume", {"threadId": "thread"})
        b.run_adapter.join()
        assert lifecycle(messages) == [("turn/started", tid)]
        release.set()
        a.run_adapter.join()
        eventually(lambda: ("turn/completed", tid) in lifecycle(messages))
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]


@pytest.mark.parametrize(
    "holder_kind,hold_seconds",
    [("abandoned", 0), ("live_handle", 0.05), ("live_handle", 1), ("live_handle", 3), ("wake_only", 0.3)],
)
def test_foreground_child_lease_unavailable_propagates_and_recovers(tmp_path, monkeypatch, holder_kind, hold_seconds):
    from vv_agent.run_handle import RunHandle

    entered, release = Event(), Event()
    child = Agent("child", "Answer.")
    parent = Agent("parent", "Delegate.", tools=[child.as_tool(name="child")])
    calls = []

    def model(_request):
        calls.append(True)
        if len(calls) == 1:
            return LLMResponse("", [ToolCall("child-call", "child", {"task_description": "go"})])
        if len(calls) == 2:
            if holder_kind != "abandoned":
                entered.set()
                assert release.wait(5)
            return LLMResponse("child done")
        return LLMResponse("parent done")

    models = ScriptedModelProvider.from_callback("test", "m", model)
    held = []
    child_runtime = Runtime.child_runtime

    def abandoned_child(self, store, sid):
        rt = child_runtime(self, store, sid)
        if not held:
            if holder_kind == "abandoned":
                held.append(store.acquire(sid, owner="abandoned-child-driver", ttl_ms=200))
            else:
                rt.ttl_ms = 5000 if holder_kind == "wake_only" else 200
                rt.heartbeat_seconds = 0.05
                pending = store.peek_inbox(sid)[0].item
                handle = RunHandle(server.kernel, sid, f"{sid}/turn/{pending.input_id}", rt, None)
                handle._deliver_to_parent = True
                held.append(handle)
                server.kernel.handles.append(handle)
                handle.start()
                assert entered.wait(5)
        return rt

    monkeypatch.setattr(Runtime, "child_runtime", abandoned_child)
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", models, agent=parent)
        stack.callback(release.set)
        messages = record_output(server, monkeypatch)
        acquisitions = []
        acquire = server.kernel.store.acquire
        monkeypatch.setattr(server.kernel.store, "acquire", lambda *a, **kw: acquisitions.append(a[0]) or acquire(*a, **kw))
        request(server, 1, "thread/start", {"inputId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        first = server.kernel.handles[0]
        assert first._lease_unavailable
        assert first._lease_unavailable_session == held[0].session_id != "thread"
        assert lifecycle(messages) == [("turn/started", tid)]
        initial_acquires = len(acquisitions)
        time.sleep(hold_seconds)
        assert len(acquisitions) == initial_acquires
        if holder_kind == "wake_only":
            # Delivery must drive the parent even when its expiry retry is lost.
            with server.run_adapter._notification_lock:
                server.run_adapter._lease_retries.clear()
        release.set()
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns[tid].ended))
        server.run_adapter.join()
        assert server.store.read_thread("thread").turns[0].result["finalOutput"] == "parent done"
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        assert len(calls) == 3
        assert not any(m.get("params", {}).get("code") == "lease_retry_exhausted" for m in messages)


def test_thread_list_authorizes_before_loading_and_never_projects_full_snapshots(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        for i in range(10):
            request(server, i, "thread/start", {"inputId": f"thread{i}"})
        read_state = server.kernel.store.read_state
        reads = []

        def authorized_read(sid):
            assert sid != "thread0"
            reads.append(sid)
            return read_state(sid)

        monkeypatch.setattr(server.kernel.store, "read_state", authorized_read)
        monkeypatch.setattr(server.host, "authorize", lambda conn, method, params: params.get("threadId") != "thread0")
        monkeypatch.setattr(server.store, "read_thread", lambda *a: pytest.fail("list projected full history"))
        response = request(server, "list", "thread/list", {"offset": 1, "limit": 2})["result"]
        assert [t["threadId"] for t in response["threads"]] == ["thread2", "thread3"]
        assert reads == ["thread2", "thread3"]
