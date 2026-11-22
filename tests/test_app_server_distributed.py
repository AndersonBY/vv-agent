"""A1 real App Server producers sharing one durable store across API nodes."""

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest
from support.s1_fixture_values import wrap_fixture_store

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.kernel._content import KeyUnavailable
from vv_agent.kernel._hydration import HydrationError
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.kernel._sql import LedgerIntegrityError, UnsupportedFeatureLevel
from vv_agent.model import ModelError, ModelRef
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.projection import hydration_for_session_store, project_records
from vv_agent.session.replay import InvalidCursor, list_events
from vv_agent.session.runtime import static_model_selection_requirements
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Conflict
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


def request_with_outbound(server, identity, method, params):
    transport = server.transport
    server.processor.process_message(
        transport.connection_id, {"jsonrpc": "2.0", "id": identity, "method": method, "params": params}
    )
    outbound = []
    while True:
        message = transport.receive_outbound(timeout=5)
        outbound.append(message)
        if message.get("id") == identity:
            return message, outbound


def until_method(server, method, *, timeout=10, predicate=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = server.transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        if message.get("method") == method and (predicate is None or predicate(message)):
            return message
    raise AssertionError(f"Timed out waiting for {method}")


class _ProtocolOnlyStore(SQLiteStore):
    """Reject SQL-only query shortcuts while retaining the store contract."""

    @property
    def max_feature_level(self):
        return super().max_feature_level

    def _one(self, *_args, **_kwargs):
        frame = inspect.currentframe()
        assert frame is not None
        caller = frame.f_back
        assert caller is not None
        frames = []
        current = caller
        while current is not None:
            frames.append(current.f_code.co_name)
            current = current.f_back
        if "_restore_worker_support" in frames and "_verified_routing_cache" in frames:
            raise AssertionError("App Server worker support must use the public SessionStore API")
        processor_caller = (
            caller.f_code.co_name == "_handle_thread_start"
            and caller.f_globals.get("__name__") == "vv_agent.app_server.processor"
        )
        if processor_caller:
            raise AssertionError("app-server processors must not issue SQL from a custom store")
        return super()._one(*_args, **_kwargs)


def node(
    stack,
    path,
    connection,
    provider,
    *,
    agent=None,
    content_key_provider=None,
    write_level=2,
    host=None,
    store_class=SQLiteStore,
):
    store = stack.enter_context(store_class.standalone(str(path)))
    store.configure_input_admission(allow_unsigned=True)
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
        host=host
        or DefaultAppServerHost(
            agent=agent or Agent("test", "Answer."),
            run_config=RunConfig(model_provider=provider, approval_timeout_seconds=5),
        ),
    )
    root_store = server.kernel._root_store
    if not isinstance(root_store, SQLiteStore):
        raise TypeError("distributed test nodes require a SQLite session store")
    root_store._rows("UPDATE sk_settings SET write_level=? WHERE singleton=1", (write_level,))
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


def test_external_level_one_store_keeps_thread_start_at_level_one(tmp_path):
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            ScriptedModelProvider.new("test", "m", []),
            write_level=1,
        )

        created = request(server, 1, "thread/start", {"inputId": "thread"})
        assert created["result"]["replayed"] is False
        record = server.kernel.store.read("thread", limit=1).records[0].record
        assert record.feature_level == 1

        rejected = request(
            server,
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": {"model": "m", "route": {}}},
        )
        assert rejected["error"]["code"] == -32602
        assert rejected["error"]["data"]["reason"] == "model_selection_unavailable"


@pytest.mark.parametrize("write_level", [1, 2])
def test_thread_start_uses_custom_store_feature_level_without_sql_private_access(tmp_path, write_level):
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            ScriptedModelProvider.new("test", "m", []),
            write_level=write_level,
            store_class=_ProtocolOnlyStore,
        )

        created = request(server, 1, "thread/start", {"inputId": "thread"})
        assert "result" in created, created
        record = server.kernel.store.read("thread", limit=1).records[0].record
        assert record.feature_level == write_level
        assert ("bootstrap" in record._payload) is (write_level == 2)

        selected = request(
            server,
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": {"model": "m", "route": {}}},
        )
        if write_level == 1:
            assert selected["error"]["data"]["reason"] == "model_selection_unavailable"
        else:
            assert "result" in selected, selected


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


def test_control_resume_and_approval_from_second_processor_after_disconnect(tmp_path, monkeypatch):
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
        a.kernel._worker.stop()
        assert a.kernel._worker.join(5), "disconnected processor worker did not stop"
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
        original_scan = b.kernel._worker._periodic_scan
        handoff_armed, first_scan, handoff_claimed, handoff_release = Event(), Event(), Event(), Event()
        scan_state = {"count": 0}

        class ClaimHandoffGate:
            def __init__(self, store):
                self._store = store

            def __getattr__(self, name):
                return getattr(self._store, name)

            def claim_runnable(self, **kwargs):
                claims = self._store.claim_runnable(**kwargs)
                if handoff_armed.is_set() and any(
                    claim.work.session_id == "thread" and claim.lease is not None for claim in claims
                ):
                    handoff_armed.clear()
                    handoff_claimed.set()
                    assert handoff_release.wait(5), "claim handoff was not released"
                return claims

        def gated_scan(store):
            if handoff_armed.is_set():
                scan_state["count"] += 1
                if scan_state["count"] == 1:
                    first_scan.set()
                    return None
            return original_scan(ClaimHandoffGate(store))

        monkeypatch.setattr(b.kernel._worker, "_periodic_scan", gated_scan)
        original_post_observation = b.kernel._worker.post_observation

        def suppress_handoff_observations(callback):
            if not handoff_claimed.is_set():
                original_post_observation(callback)

        monkeypatch.setattr(b.kernel._worker, "post_observation", suppress_handoff_observations)
        handoff_armed.set()
        answer = request(
            b,
            5,
            "approval/resolve",
            {"threadId": "thread", "turnId": tid, "requestId": approval["requestId"], "decision": "allow"},
        )
        assert "result" in answer
        assert first_scan.wait(5), "worker did not reach the pre-claim scan"
        joined, join_errors = Event(), []

        def join_handoff():
            try:
                b.run_adapter.join()
            except BaseException as exc:
                join_errors.append(exc)
            finally:
                joined.set()

        join_thread = Thread(target=join_handoff, name="approval-handoff-join")
        join_thread.start()
        try:
            assert handoff_claimed.wait(5), "approval answer was not claimed"
            assert not joined.wait(0.2), "join returned while the local claim was still handing off"
        finally:
            handoff_release.set()
            join_thread.join(5)
        assert not join_thread.is_alive(), "join did not finish after claim handoff"
        assert not join_errors
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


def test_committed_turn_start_replays_before_host_runtime_resolution(tmp_path, monkeypatch):
    calls = []
    provider = ScriptedModelProvider.from_callback(
        "test", "m", lambda request: calls.append(request.model) or LLMResponse("done")
    )
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider)
        request(server, 1, "thread/start", {"inputId": "thread"})
        first = request(server, 2, "turn/start", submission())["result"]
        server.run_adapter.join()
        eventually(lambda: server.kernel.store.read_state("thread")[0].active_turn_id is None)
        head = server.kernel.store.read("thread", limit=1).head_seq

        def unavailable(_request):
            raise ValueError("model temporarily unavailable")

        monkeypatch.setattr(
            server.run_adapter._host,
            "build_run_config",
            unavailable,
        )
        replay = request(server, 3, "turn/start", submission())["result"]

        assert replay == first | {"replayed": True}
        assert server.run_adapter._turn_connections[("thread", first["turnId"])] == "A"
        assert server.kernel.store.read("thread", limit=1).head_seq == head
        assert calls == ["m"]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ModelError("unsupported model route"), "model_resolution_failed"),
        (UnsupportedFeatureLevel("worker cannot restore level 2"), "unsupported_feature_level"),
        (Conflict("runtime resume conflict"), "worker_support_conflict"),
    ],
)
def test_committed_turn_start_replay_reports_typed_restore_warning(tmp_path, monkeypatch, error, reason):
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider)
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(server, 2, "turn/start", submission())
        server.run_adapter.join()
        eventually(lambda: server.kernel.store.read_state("thread")[0].active_turn_id is None)
        monkeypatch.setattr(server.run_adapter._host, "build_run_config", lambda _request: (_ for _ in ()).throw(error))

        response, outbound = request_with_outbound(server, 3, "turn/start", submission())

        assert response["result"]["replayed"] is True
        warning = next(message for message in outbound if message.get("method") == "error/warning")
        assert warning["params"] == {
            "message": str(error),
            "code": "worker_support_restore",
            "reason": reason,
        }


def test_level_one_turn_start_replay_skips_level_two_runtime_restore(tmp_path, monkeypatch):
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider, write_level=1)
        request(server, 1, "thread/start", {"inputId": "thread"})
        receipt = request(server, 2, "turn/start", submission())["result"]
        server.run_adapter.join()
        eventually(lambda: server.kernel.store.read_state("thread")[0].active_turn_id is None)
        monkeypatch.setattr(
            server.run_adapter._host,
            "build_run_config",
            lambda _request: (_ for _ in ()).throw(ModelError("level 2 runtime must not be restored")),
        )

        response, outbound = request_with_outbound(server, 3, "turn/start", submission())

        assert response["result"] == receipt | {"replayed": True}
        assert not any(message.get("method") == "error/warning" for message in outbound)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ModelError("unsupported model route"), "model_resolution_failed"),
        (UnsupportedFeatureLevel("worker cannot restore level 2"), "unsupported_feature_level"),
    ],
)
def test_fresh_reconnect_turn_start_replays_before_runtime_support_failure(tmp_path, error, reason):
    calls = []
    arguments = {"path": "notes/private.txt", "force": True}

    @function_tool(needs_approval=True)
    def protected(path: str, force: bool) -> str:
        calls.append({"path": path, "force": force})
        return "allowed"

    agent = Agent("test", "Use protected.", tools=[protected])
    first_provider = ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("protected", "protected", arguments)])])
    with ExitStack() as stack:
        first = node(
            stack,
            tmp_path / "store",
            "A",
            first_provider,
            agent=agent,
            store_class=_ProtocolOnlyStore,
        )
        request(first, 1, "thread/start", {"inputId": "thread"})
        turn_id = request(first, 2, "turn/start", submission())["result"]["turnId"]
        assert until_method(first, "approval/request")["params"]["turnId"] == turn_id
        first.run_adapter.close()
        first.kernel.close()

        class FailingRestoreHost(DefaultAppServerHost):
            def build_run_config(self, request):
                del request
                raise error

        resumed = node(
            stack,
            tmp_path / "store",
            "B",
            ScriptedModelProvider.new("test", "m", []),
            agent=agent,
            host=FailingRestoreHost(agent=agent, run_config=RunConfig()),
            store_class=_ProtocolOnlyStore,
        )
        response, outbound = request_with_outbound(resumed, 3, "turn/start", submission())
        while not any(message.get("method") == "approval/request" for message in outbound):
            outbound.append(resumed.transport.receive_outbound(timeout=5))

        assert response["result"]["replayed"] is True
        warning = next(message for message in outbound if message.get("method") == "error/warning")
        assert warning["params"]["reason"] == reason
        methods = [message.get("method") for message in outbound]
        causal = ["turn/started", "item/started", "approval/request"]
        assert [methods.index(method) for method in causal] == sorted(methods.index(method) for method in causal)
        approval = next(message for message in outbound if message.get("method") == "approval/request")
        assert approval["params"]["arguments"] == arguments
        assert calls == []


@pytest.mark.parametrize(
    "error",
    [
        LedgerIntegrityError("damaged ledger"),
        HydrationError("damaged content"),
        KeyUnavailable("missing key"),
    ],
)
def test_committed_turn_start_replay_does_not_swallow_integrity_or_key_errors(tmp_path, monkeypatch, error):
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider)
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(server, 2, "turn/start", submission())
        server.run_adapter.join()
        eventually(lambda: server.kernel.store.read_state("thread")[0].active_turn_id is None)

        def fail_subscribe(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(server.run_adapter, "subscribe", fail_subscribe)

        with pytest.raises(type(error), match=str(error)):
            server.run_adapter.start_turn(
                connection_id="A",
                thread_id="thread",
                input=submission()["input"],
                input_id="prompt",
                expected_turn_id=None,
            )

        assert server.run_adapter._turn_connections[("thread", "thread/turn/prompt")] == "A"


def test_create_conflict_without_session_preserves_original_typed_error(tmp_path):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        server.kernel._root_store._rows("UPDATE sk_settings SET write_level=? WHERE singleton=1", (1,))
        with pytest.raises(UnsupportedFeatureLevel, match="write level 1"):
            server.store.create_thread_receipt(
                input_id="rejected",
                agent_key="default",
                bootstrap=SessionRequirements(2, "1", ()),
            )


def test_level_two_approval_turn_start_replays_after_process_restart(tmp_path):
    calls = []
    arguments = {"path": "notes/private.txt", "force": True}

    @function_tool(needs_approval=True)
    def protected(path: str, force: bool) -> str:
        calls.append({"path": path, "force": force})
        return "allowed"

    agent = Agent("test", "Use protected.", tools=[protected])
    first_provider = ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("protected", "protected", arguments)])])
    resumed_provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    with ExitStack() as stack:
        first = node(stack, tmp_path / "store", "A", first_provider, agent=agent)
        request(first, 1, "thread/start", {"inputId": "thread"})
        turn_id = request(first, 2, "turn/start", submission())["result"]["turnId"]
        approval = until_method(first, "approval/request")
        assert approval["params"]["arguments"] == arguments
        first.run_adapter.close()
        first.kernel.close()

        resumed = node(stack, tmp_path / "store", "B", resumed_provider, agent=agent)
        assert resumed.kernel._worker.support.level == 1
        response, outbound = request_with_outbound(resumed, 3, "turn/start", submission())
        assert response["result"] == {
            "threadId": "thread",
            "turnId": turn_id,
            "inputId": "prompt",
            "inputSeq": response["result"]["inputSeq"],
            "accepted": True,
            "replayed": True,
        }
        replayed = next((message for message in outbound if message.get("method") == "approval/request"), None)
        if replayed is None:
            replayed = until_method(resumed, "approval/request")
        assert replayed["id"] == approval["id"]
        assert replayed["params"]["arguments"] == arguments
        assert resumed.run_adapter._turn_connections[("thread", turn_id)] == "B"
        assert resumed.kernel._worker.support.level == 2

        resumed.processor.process_message(
            resumed.transport.connection_id,
            {"jsonrpc": "2.0", "id": replayed["id"], "result": {"decision": "allow"}},
        )
        completed = until_method(
            resumed,
            "turn/completed",
            predicate=lambda message: message["params"]["turnId"] == turn_id and message["params"]["status"] != "interrupted",
        )
        assert completed["params"]["status"] == "completed"
        assert calls == [arguments]


def test_level_two_resume_does_not_elevate_an_incompatible_host(tmp_path):
    @function_tool
    def original_only() -> str:
        return "original"

    provider = ScriptedModelProvider.new("test", "m", [])
    with ExitStack() as stack:
        first = node(stack, tmp_path / "store", "A", provider, agent=Agent("test", "Use the tool.", tools=[original_only]))
        request(first, 1, "thread/start", {"inputId": "thread"})
        first.run_adapter.close()
        first.kernel.close()

        resumed = node(stack, tmp_path / "store", "B", provider, agent=Agent("test", "No matching tool."))
        response = request(resumed, 2, "thread/resume", {"threadId": "thread"})

        assert response["error"]["code"] == -32602
        assert resumed.kernel._worker.support.level == 1


def test_restart_restores_persisted_custom_route_selection_for_claim(tmp_path):
    agent = Agent("test", "Answer.", model=ModelRef.named("a"))
    selection = {"model": "b", "route": {"pool": "b"}}

    class RoutedHost(DefaultAppServerHost):
        def __init__(self, default_provider, selected_provider):
            self._default_provider = default_provider
            self._selected_provider = selected_provider
            self.model_list_calls = 0
            super().__init__(agent=agent, run_config=RunConfig(model_provider=default_provider))

        def resolve_model(self, request):
            provider = None
            if request.get("model") == "a" and request.get("route") == {}:
                provider = self._default_provider
            elif request.get("model") == "b" and request.get("route") == {"pool": "b"}:
                provider = self._selected_provider
            if provider is None:
                raise ModelError("unsupported model route")
            resolved = provider.resolve(ModelRef.named(request["model"]))
            return provider.client(resolved), resolved

        def list_models(self, request):
            self.model_list_calls += 1
            return super().list_models(request)

    with ExitStack() as stack:
        before_a = ScriptedModelProvider.new("a-backend", "a", [])
        before_b = ScriptedModelProvider.new("b-backend", "b", [LLMResponse("before restart")])
        first_host = RoutedHost(before_a, before_b)
        first = node(
            stack,
            tmp_path / "store",
            "A",
            before_a,
            host=first_host,
        )
        request(first, 1, "thread/start", {"inputId": "thread"})
        selection_response = request(
            first,
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": selection},
        )
        assert "result" in selection_response, selection_response
        first_turn_id = request(first, 3, "turn/start", submission())["result"]["turnId"]
        first.wake("thread")
        eventually(
            lambda: (turns := first.store.read_thread("thread").turns) and turns[-1].result.get("finalOutput") == "before restart"
        )
        first.run_adapter.join()
        assert first_host.model_list_calls == 0
        first.kernel.close()

        after_a = ScriptedModelProvider.new("a-backend", "a", [])
        after_b = ScriptedModelProvider.new("b-backend", "b", [LLMResponse("after restart")])
        resumed_host = RoutedHost(after_a, after_b)
        resumed = node(
            stack,
            tmp_path / "store",
            "B",
            after_a,
            host=resumed_host,
        )
        assert resumed.kernel._worker.support.level == 1
        assert "result" in request(resumed, 4, "thread/resume", {"threadId": "thread"})

        selected = static_model_selection_requirements(
            selection,
            resolve_model=resumed_host.resolve_model,
            handler_version=resumed.kernel.default_handler_version,
        )
        assert set(selected.capabilities) <= set(resumed.kernel._worker.support.capabilities)
        assert resumed_host.model_list_calls == 0

        request(
            resumed,
            5,
            "turn/start",
            submission(identity="after", expected=first_turn_id),
        )
        resumed.wake("thread")
        eventually(
            lambda: (
                (turns := resumed.store.read_thread("thread").turns) and turns[-1].result.get("finalOutput") == "after restart"
            )
        )
        resumed.run_adapter.join()
        assert resumed.store.read_thread("thread").turns[-1].result["finalOutput"] == "after restart"


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
