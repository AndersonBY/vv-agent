"""Cross-node recovery and notification isolation through real producers."""

import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from itertools import pairwise
from threading import Event

import pytest
from support.interactions import target_of
from test_app_server_distributed import node, request, submission
from test_app_server_review_regressions import record_output

from vv_agent import InteractionAnswer, ScriptedModelProvider
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
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


def observer(stack, path, connection):
    server = node(stack, path, connection, provider())
    support = replace(
        server.kernel._worker.support,
        drive=None,
        project=None,
        projection_consumers=(),
    )
    server.kernel._worker.support = support
    return server


def test_abandoned_lease_runs_accepted_turn_after_expiry_without_external_wake(tmp_path, monkeypatch):
    calls = []
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            ScriptedModelProvider.from_callback("test", "m", lambda r: calls.append(r.model) or LLMResponse("done")),
            managed_worker=False,
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
        expected = [("turn/started", receipt["turnId"]), ("turn/completed", receipt["turnId"])]
        eventually(lambda: lifecycle(messages) == expected)


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
        eventually(lambda: bool(list(b.kernel.waits("thread"))))
        interaction = next(wait["interaction"] for _, _, wait in b.kernel.waits("thread") if "interaction" in wait)
        target = target_of(interaction)
        _, agent, config = b.run_adapter._binding("thread")
        b.kernel.answer(
            "thread",
            b.kernel.runtime(agent, config),
            target,
            InteractionAnswer("text", text="answer"),
            "reply",
        )
        b.kernel.store.acquire("thread", owner="dead-before-restart", ttl_ms=200)
        assert request(b, 3, "turn/resume", {"threadId": "thread", "turnId": tid})["result"]["status"] == "running"
        eventually(lambda: bool(b.kernel.store.read_state("thread")[0].turns[tid].ended))
        b.run_adapter.join()
        assert b.store.read_thread("thread").turns[0].result["finalOutput"] == "done"


@pytest.mark.parametrize("disconnect", [False, True])
def test_resubscribe_starts_at_head_and_releases_positions(tmp_path, monkeypatch, disconnect):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = observer(stack, tmp_path / "store", "B")
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
            response = request(a, f"t{i}", "turn/start", submission(identity=f"p{i}", expected=previous))
            assert "result" in response, response
            previous = response["result"]["turnId"]
            a.run_adapter.join()
            a.run_adapter.join()
            eventually(
                lambda turn_id=previous: (
                    (turn := a.kernel.store.read_state("thread")[0].turns.get(turn_id)) is not None and turn.ended
                ),
                timeout=10,
            )
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
        b = observer(stack, tmp_path / "store", "B")
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        messages = record_output(b, monkeypatch)
        failures = []
        publish = b.run_adapter._publish_turns

        def transient(*args, **kwargs):
            if not failures:
                failures.append(time.monotonic())
                raise RuntimeError("transient reset")
            return publish(*args, **kwargs)

        monkeypatch.setattr(b.run_adapter, "_publish_turns", transient)
        eventually(lambda: bool(failures))
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        a.run_adapter.join()
        eventually(lambda: ("turn/completed", tid) in lifecycle(messages), timeout=3)
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        assert sum(m.get("method") == "error/warning" for m in messages) == 1
        assert b.kernel._worker.running


@pytest.mark.parametrize("writer", ["local", "process"])
def test_blocked_observer_does_not_hold_store_or_admission_lock(tmp_path, monkeypatch, writer):
    blocked, release = Event(), Event()
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = observer(stack, tmp_path / "store", "B")
        stack.callback(release.set)
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        write = b.transport.write_outbound

        def slow_write(message):
            if message.get("method") == "turn/completed":
                blocked.set()
                assert release.wait(10)
            write(message)

        monkeypatch.setattr(b.transport, "write_outbound", slow_write)
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        a.run_adapter.join()
        assert blocked.wait(10)
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
    _, records, _ = store.read_state('thread')
    actor = records[0].record.payload['principal']
    with store.atomic() as tx:
        tx.push('thread', InboxItem('selection', 'model_selection', {'selection': None}, actor=actor))
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
        b = observer(stack, tmp_path / "store", "B")
        for sid in ["busy", *(f"idle{i}" for i in range(8))]:
            request(a, f"new{sid}", "thread/start", {"inputId": sid})
            request(b, f"sub{sid}", "thread/resume", {"threadId": sid})
        reads = Counter()
        publish = b.run_adapter._publish_turns

        def counted(sid, **kwargs):
            reads[sid] += 1
            return publish(sid, **kwargs)

        monkeypatch.setattr(b.run_adapter, "_publish_turns", counted)
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
        assert reads["busy"] > max(reads[f"idle{i}"] for i in range(8)), reads


def test_error_after_terminal_notification_emits_completed_only_once(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = []
        write = server.transport.write_outbound
        failed = False

        def late_failure(message):
            nonlocal failed
            messages.append(message)
            write(message)
            if message.get("method") == "turn/completed" and not failed:
                failed = True
                raise RuntimeError("late stream failure")

        monkeypatch.setattr(server.transport, "write_outbound", late_failure)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        eventually(lambda: failed)
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        assert next(m for m in messages if m.get("method") == "turn/completed")["params"]["status"] == "completed"


def test_persistent_observer_failure_is_rate_limited_and_does_not_stop_other_sessions(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider())
        b = observer(stack, tmp_path / "store", "B")
        for sid in ("broken", "healthy"):
            request(a, sid, "thread/start", {"inputId": sid})
            request(b, sid, "thread/resume", {"threadId": sid})
        messages = record_output(b, monkeypatch)
        calls = []
        warnings = []
        publish = b.run_adapter._publish_turns
        notify = b.run_adapter._notify_subscribers

        def fail_one(sid, **kw):
            if sid == "broken":
                calls.append(time.monotonic())
                raise RuntimeError("database unavailable")
            return publish(sid, **kw)

        def timed_warning(thread_id, method, params, *, exclude=None):
            if thread_id == "broken" and method == "error/warning":
                warnings.append(time.monotonic())
            return notify(thread_id, method, params, exclude=exclude)

        monkeypatch.setattr(b.run_adapter, "_publish_turns", fail_one)
        monkeypatch.setattr(b.run_adapter, "_notify_subscribers", timed_warning)
        request(a, 3, "turn/start", submission(sid="healthy"))
        a.run_adapter.join()
        eventually(lambda: ("turn/completed", "healthy/turn/prompt") in lifecycle(messages))
        time.sleep(1.2)
        assert len(calls) <= 6
        assert len(warnings) <= 3
        assert all(later - earlier >= 0.9 for earlier, later in pairwise(warnings))
        assert sum(m.get("method") == "error/warning" for m in messages) == len(warnings)
        assert b.kernel._worker.running
        monkeypatch.undo()


def test_live_lease_prevents_fake_started_until_release(tmp_path, monkeypatch):
    def unexpected_claim(*_args, **_kwargs):
        pytest.fail("App Server must acquire known sessions directly")

    monkeypatch.setattr(SQLiteStore, "claim_runnable", unexpected_claim)
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider(), managed_worker=False)
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)
        holder = server.kernel.store.acquire("thread", owner="live-process", ttl_ms=60_000)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        assert lifecycle(messages) == []
        state = server.kernel.store.read_state("thread")[0]
        assert state.active_turn_id is None
        assert not state.turns
        assert server.kernel.store.lease_status("thread")[2] > 0

        server.kernel.store.release(holder)
        server.run_adapter.wake("thread")
        server.run_adapter.join()
        expected = [("turn/started", tid), ("turn/completed", tid)]
        eventually(lambda: lifecycle(messages) == expected)
        assert server.kernel.store.read_state("thread")[0].turns[tid].ended


def test_renewed_live_lease_still_owns_the_turn(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)
        holder = server.kernel.store.acquire("thread", owner="live", ttl_ms=500)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        time.sleep(0.25)
        server.kernel.store.renew(holder, ttl_ms=500)
        server.run_adapter.join()
        time.sleep(0.3)
        assert lifecycle(messages) == []
        assert not server.kernel.store.read_state("thread")[0].turns

        server.kernel.store.release(holder)
        server.run_adapter.wake("thread")
        server.run_adapter.join()
        expected = [("turn/started", tid), ("turn/completed", tid)]
        eventually(lambda: lifecycle(messages) == expected)
        assert server.kernel.store.read_state("thread")[0].turns[tid].ended


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


def test_foreground_turn_does_not_complete_while_model_is_blocked(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def model(_request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("parent done")

    models = ScriptedModelProvider.from_callback("test", "m", model)
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", models)
        stack.callback(release.set)
        messages = record_output(server, monkeypatch)
        request(server, 1, "thread/start", {"inputId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        eventually(lambda: ("turn/started", tid) in lifecycle(messages))
        assert lifecycle(messages) == [("turn/started", tid)]
        release.set()
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns[tid].ended))
        server.run_adapter.join()
        assert server.store.read_thread("thread").turns[0].result["finalOutput"] == "parent done"
        assert lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]


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

        with monkeypatch.context() as guards:
            guards.setattr(server.kernel.store, "read_state", authorized_read)
            guards.setattr(server.host, "authorize", lambda conn, method, params: params.get("threadId") != "thread0")
            guards.setattr(server.store, "read_thread", lambda *a: pytest.fail("list projected full history"))
            response = request(server, "list", "thread/list", {"offset": 1, "limit": 2})["result"]
            assert [t["threadId"] for t in response["threads"]] == ["thread2", "thread3"]
            assert reads == ["thread2", "thread3"]
