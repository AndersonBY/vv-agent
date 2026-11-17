"""A1 real App Server producers sharing one durable store across API nodes."""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Event

import pytest
from support.s1_fixture_values import wrap_fixture_store

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.projection import hydration_for_session_store, project_records
from vv_agent.session.replay import InvalidCursor, list_events
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolCall


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def request(server, identity, method, params):
    transport = server.transport
    server.processor.process_message(
        transport.connection_id, {"jsonrpc": "2.0", "id": identity, "method": method, "params": params}
    )
    while True:
        message = transport.receive_outbound(timeout=5)
        if message.get("id") == identity:
            return message


def node(stack, path, connection, provider, *, agent=None, content_key_provider=None):
    store = stack.enter_context(SQLiteStore.standalone(str(path)))
    store.install_schema()
    if content_key_provider is None:
        resolved_path = Path(path).resolve()
        content_key_provider = LocalKeyProvider(
            sha256(b"vv-agent/app-server-distributed-test\0" + str(resolved_path).encode()).digest(),
            state_path=resolved_path.with_name(f"{resolved_path.name}.content-key-state.sqlite3"),
        )
        stack.callback(content_key_provider.close)
    # Bind through the shared S1 helper; AppServer constructs its canonical adapter from this raw root.
    wrap_fixture_store(store, content_key_provider)
    server = AppServer(
        transport=ChannelTransport(connection_id=connection, outbound_capacity=10000),
        store=store,
        content_key_provider=content_key_provider,
        host=DefaultAppServerHost(
            agent=agent or Agent("test", "Answer."), run_config=RunConfig(model_provider=provider, approval_timeout_seconds=5)
        ),
    )
    stack.callback(server.kernel.close)
    stack.callback(server.run_adapter.join)
    stack.callback(lambda: [h.cancel("teardown") for h in server.kernel.handles])
    assert "result" in request(server, "init", "initialize", {"clientInfo": {"name": connection}})
    return server


def submission(sid="thread", identity="prompt", expected=None, text="hello"):
    return {"threadId": sid, "inputId": identity, "expectedTurnId": expected, "input": [{"type": "text", "text": text}]}


def test_node_does_not_close_borrowed_content_key_provider(tmp_path):
    state_path = tmp_path / "borrowed-content-key-state.sqlite3"
    content_key_provider = LocalKeyProvider(b"b" * 32, state_path=state_path)
    try:
        with ExitStack() as stack:
            node(
                stack,
                tmp_path / "store",
                "A",
                ScriptedModelProvider.new("test", "m", []),
                content_key_provider=content_key_provider,
            )
        assert content_key_provider.generate_data_key("default", "after-node").wrapped_key
    finally:
        content_key_provider.close()


def test_two_processors_concurrent_create_submit_lost_ack_and_reconnect(tmp_path, monkeypatch):
    calls = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: calls.append(r.model) or LLMResponse("done"))
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        for server in (a, b):
            monkeypatch.setattr(
                server.kernel.store, "list_sessions", lambda **kw: pytest.fail("thread creation scanned sessions")
            )
        barrier = Barrier(2)

        def concurrent(server, method, params):
            barrier.wait(timeout=5)
            return request(server, "wire", method, params)

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(concurrent, s, "thread/start", {"inputId": "thread"}) for s in (a, b)]
            creates = [f.result(timeout=10)["result"] for f in futures]
            assert sorted(r["replayed"] for r in creates) == [False, True]
            futures = [pool.submit(concurrent, s, "turn/start", submission()) for s in (a, b)]
            receipts = [f.result(timeout=10)["result"] for f in futures]
        assert sorted(r["replayed"] for r in receipts) == [False, True]
        assert receipts[0] | {"replayed": True} == receipts[1] | {"replayed": True}
        for s in (a, b):
            s.run_adapter.join()
            eventually(lambda: bool(b.kernel.store.read_state("thread")[0].turns) and calls == ["m"])
        # The ACK has been lost to the client; reconnect uses another node and RPC ID.
        a.processor.disconnect_connection("A")
        replay = request(b, "new-wire-id", "turn/start", submission())["result"]
        assert replay == receipts[0] | {"replayed": True}
        assert request(b, "conflict", "turn/start", submission(text="changed"))["error"]["code"] == -32024
        assert (
            request(b, "create-conflict", "thread/start", {"inputId": "thread", "cwd": "/different"})["error"]["code"] == -32024
        )
        state, records, _ = b.kernel.store.read_state("thread")
        assert len(state.turns) == 1 and calls == ["m"]
        assert len([r for r in records if r.record.kind == "turn_started"]) == 1
        assert b.kernel.store.read_input("thread", "prompt").item.input_id == "prompt"


def test_steer_and_followup_receipts_survive_turn_completion(tmp_path):
    entered, release = Event(), Event()
    calls = []

    def model(r):
        calls.append(r.model)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return LLMResponse("done")

    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", model))
        stack.callback(release.set)
        b = node(stack, tmp_path / "store", "B", a.host._run_config.model_provider)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        assert request(b, 3, "thread/status", {"threadId": "thread"})["result"]["status"] == "provider_running"
        receipts = {}
        for method, identity in (("turn/steer", "context"), ("turn/followUp", "next")):
            params = submission(identity=identity, expected=tid)
            receipts[method] = request(b, method, method, params)["result"]
            assert request(a, method, method, params)["result"] == receipts[method] | {"replayed": True}
        release.set()
        a.run_adapter.join()
        b.run_adapter.join()
        for method, identity in (("turn/steer", "context"), ("turn/followUp", "next")):
            params = submission(identity=identity, expected=tid)
            assert request(b, "replay-" + method, method, params)["result"] == receipts[method] | {"replayed": True}
        assert len(b.kernel.store.read_state("thread")[0].turns) == 2


def test_control_resume_and_approval_from_second_processor_after_disconnect(tmp_path):
    effects = []

    @function_tool(needs_approval=True)
    def protected() -> str:
        effects.append("executed")
        return "ok"

    provider = ScriptedModelProvider.new(
        "test", "m", [LLMResponse("", [ToolCall("protected", "protected", {})]), LLMResponse("done")]
    )
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider, agent=Agent("test", "Use protected.", tools=[protected]))
        b = node(stack, tmp_path / "store", "B", provider, agent=a.host._agent)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        while (notice := a.transport.receive_outbound(timeout=5)).get("method") != "approval/request":
            pass
        approval = notice["params"]
        a.processor.disconnect_connection("A")
        a.run_adapter.join()
        status = request(b, 3, "thread/status", {"threadId": "thread"})["result"]
        assert status["status"] == "approval_wait" and status["waitKind"] == "approval"
        assert status["pendingInteraction"]["requestId"] == approval["requestId"]
        assert "result" in request(b, "attach", "thread/resume", {"threadId": "thread"})
        for kind in ("suspend", "resume"):
            assert request(
                b, kind, "turn/action", {"threadId": "thread", "turnId": tid, "actionId": kind, "action": {"kind": kind}}
            )["result"]["accepted"]
            # The reconnect already replays the durable approval once. A
            # suspend/resume cycle must not mint a second server request.
            if kind == "resume":
                outbound = []
                while True:
                    try:
                        outbound.append(b.transport.receive_outbound(timeout=0.05))
                    except Exception:
                        break
                assert not any(message.get("method") == "approval/request" for message in outbound)
            b.router.cancel_matching_server_requests()
            b.run_adapter.join()
        assert "result" in request(b, 4, "turn/resume", {"threadId": "thread", "turnId": tid})
        answer = request(
            b,
            5,
            "approval/resolve",
            {"threadId": "thread", "turnId": tid, "requestId": approval["requestId"], "decision": "allow"},
        )
        assert "result" in answer
        b.run_adapter.join()
        assert effects == ["executed"], [
            (r.record.kind, r.record.payload)
            for r in b.kernel.store.read_state("thread")[1]
            if r.record.kind in {"turn_ended", "input_applied", "op_completed"}
        ]
        assert request(b, 6, "thread/status", {"threadId": "thread"})["result"]["status"] == "terminal"
        answers = [
            r.record.payload["input"]["payload"]
            for r in b.kernel.store.read_state("thread")[1]
            if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "approval_answer"
        ]
        assert answers[0]["request_id"] == approval["requestId"]
        assert answers[0]["operation_id"] and len(answers[0]["request_digest"]) == 64


def test_interrupt_without_local_handle_is_durable(tmp_path):
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose?"})])])
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        a.run_adapter.join()
        eventually(lambda: "thread/turn/prompt" in a.kernel.store.read_state("thread")[0].turns)
        a.processor.disconnect_connection("A")
        assert b.state_manager.active_turn("thread") is None
        result = request(b, 3, "turn/interrupt", {"threadId": "thread", "expectedTurnId": tid, "actionId": "cancel"})["result"]
        assert result["accepted"]
        b.run_adapter.join()
        state, records, _ = b.kernel.store.read_state("thread")
        assert state.turns[tid].ended
        assert any(
            r.record.kind == "input_applied"
            and r.record.payload["input"]["kind"] == "control"
            and r.record.payload["input"]["payload"]["action"] == "cancel"
            for r in records
        )


def test_paged_replay_matches_full_producer_and_bounds_reads(tmp_path, monkeypatch):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", lambda r: LLMResponse("done")))
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = None
        for index in range(20):
            response = request(a, 2, "turn/start", submission(identity=f"prompt-{index}", expected=tid))
            assert "result" in response, response
            tid = response["result"]["turnId"]
            a.run_adapter.join()
            eventually(lambda turn_id=tid: a.kernel.store.read_state("thread")[0].turns[turn_id].ended)
        store = a.kernel.store
        hydration = hydration_for_session_store(store, "thread")
        expected = [e.to_dict() for e in project_records(store.read_state("thread")[1], hydration=hydration)]
        rows_read = []
        original = store._rows

        def counted(sql, params=()):
            rows = original(sql, params)
            if "FROM sk_record" in sql:
                rows_read.extend(rows)
            return rows

        monkeypatch.setattr(store, "_rows", counted)
        monkeypatch.setattr(store, "read_state", lambda *a, **k: pytest.fail("page folded full history"))
        after = through = None
        actual = []
        middle = None
        for _ in range(2000):
            rows_read.clear()
            page = list_events(store, "thread", limit=3, after=after, through=through)
            assert len(page["events"]) <= 3 and len(rows_read) <= 70
            actual.extend(x["event"] for x in page["events"])
            for x in page["events"]:
                if x["event"]["type"] == "run_started":
                    middle = x["cursor"]
            after, through = page["nextCursor"], page["throughCursor"]
            if not page["more"]:
                break
        else:
            pytest.fail("paging did not finish")
        monkeypatch.undo()
        assert actual == expected
        assert middle is not None
        page = list_events(store, "thread", limit=1, after=middle, through=through)
        assert page["events"][0]["event"]["type"] == "agent_started"
        for bad in [
            "old",
            {},
            after | {"seq": True},
            after | {"offset": 999},
            after | {"recordDigest": "unknown"},
            after | {"threadId": "other"},
        ]:
            with pytest.raises(InvalidCursor):
                list_events(store, "thread", limit=3, after=bad)
        assert (
            request(a, 9, "thread/events/list", {"threadId": "thread", "limit": 3, "afterCursor": {}})["error"]["code"] == -32026
        )


def test_submission_conflict_compares_canonical_bytes_including_boolean_types(tmp_path):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", [LLMResponse("done")]))
        request(server, 1, "thread/start", {"inputId": "thread"})
        original = submission() | {
            "metadata": {"number": 1, "other": "value"},
            "input": [{"type": "image", "url": "https://example.invalid/image.png"}],
        }
        receipt = request(server, 2, "turn/start", original)["result"]
        server.run_adapter.join()
        eventually(lambda: "thread/turn/prompt" in server.kernel.store.read_state("thread")[0].turns)
        reordered = original | {
            "metadata": {"other": "value", "number": 1.0},
            "input": [{"url": "https://example.invalid/image.png", "type": "image"}],
        }
        assert request(server, 3, "turn/start", reordered)["result"] == receipt | {"replayed": True}
        different_type = original | {"metadata": {"number": True, "other": "value"}}
        assert request(server, 4, "turn/start", different_type)["error"]["code"] == -32024
        assert len(server.kernel.store.read_state("thread")[0].turns) == 1


def test_lost_ack_before_drive_reuses_original_admitted_content_after_reconnect(tmp_path, monkeypatch):
    seen = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: seen.append(r) or LLMResponse("done"))
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        request(a, 1, "thread/start", {"inputId": "thread"})
        entered, release = Event(), Event()
        original_scan = a.kernel._worker._periodic_scan

        def gated_scan(store):
            entered.set()
            release.wait()
            return original_scan(store)

        monkeypatch.setattr(a.kernel._worker, "_periodic_scan", gated_scan)
        stack.callback(release.set)
        original = [{"type": "image", "url": "https://example.invalid/image.png"}]
        receipt, _ = a.run_adapter.submit_input(
            connection_id="A", thread_id="thread", input_id="prompt", expected_turn_id=None, kind="start", input=original
        )
        assert entered.wait(1), "node A reached drive before the admission barrier"
        # The API node disappears after the admission commit and before driving.
        a.processor.disconnect_connection("A")
        reordered = [{"url": "https://example.invalid/image.png", "type": "image"}]
        replay = request(b, 2, "turn/start", submission() | {"input": reordered})["result"]
        assert replay == receipt | {"replayed": True}
        b.run_adapter.join()
        eventually(
            lambda: (
                len(seen) == 1
                and (state := b.kernel.store.read_state("thread")[0]).active_turn_id is None
                and state.turns[replay["turnId"]].ended
            )
        )
        assert len(seen) == 1
        state = b.kernel.store.read_state("thread")[0]
        assert len(state.turns) == 1 and state.active_turn_id is None
        release.set()
