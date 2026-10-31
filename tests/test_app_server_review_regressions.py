"""VA1 review regressions exercise real App Server, Runtime and durable stores."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event

import pytest
from support.interactions import displayed_target
from test_app_server_distributed import node, request, submission

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.app_server import AppServer, AppServerHost, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.processor import CLIENT_METHODS
from vv_agent.app_server.protocol import ModelListResponse
from vv_agent.types import LLMResponse, ToolCall


def record_output(server, monkeypatch):
    messages = []
    write = server.transport.write_outbound

    def recorded(message):
        messages.append(message)
        write(message)

    monkeypatch.setattr(server.transport, "write_outbound", recorded)
    return messages


def test_default_runtime_uses_custom_host_resolver(tmp_path, monkeypatch):
    import json

    from vv_agent.model import ModelRef, VvLlmModelProvider

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "VERSION": "2",
                "backends": {"test": {"models": {"a": {"id": "a-2024", "endpoints": ["local"]}}}},
                "endpoints": [{"id": "local", "api_key": "placeholder", "api_base": "https://example.invalid/v1"}],
            }
        )
    )
    provider = VvLlmModelProvider.from_settings_file(catalog)
    monkeypatch.setattr(
        VvLlmModelProvider, "client", lambda *args: ScriptedModelProvider.new("test", "a-2024", [LLMResponse("done")]).llm
    )

    resolved_requests = []

    class Host(AppServerHost):
        def authorize_tool_result_resource(self, request):
            return False

        def resolve_model(self, request):
            resolved_requests.append(request)
            assert request["model"] == "a-2024" and request["route"] == {}
            resolved = provider.resolve(ModelRef.backend("test", "a"))
            return provider.client(resolved), resolved

        def authorize(self, connection_id, method, params):
            return True

        def resolve_agent(self, request):
            return Agent("test", "Answer.", model=ModelRef.backend("test", "a"))

        def build_run_config(self, request):
            return RunConfig(model_provider=provider)

        def list_models(self, request):
            return ModelListResponse([])

    server = AppServer(transport=ChannelTransport(connection_id="A"), host=Host())
    try:
        request(server, 0, "initialize", {"clientInfo": {"name": "test"}})
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(server, 2, "turn/start", submission())
        server.run_adapter.join()
        assert server.store.read_thread("thread").turns[0].result["finalOutput"] == "done"
        assert len(resolved_requests) == 2
    finally:
        server.kernel.close()


def test_nonholder_acquires_once_and_both_nodes_observe_each_durable_turn(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def model(req):
        calls.append(req.model)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
        return LLMResponse("done")

    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", model))
        b = node(stack, tmp_path / "store", "B", a.host._run_config.model_provider)
        stack.callback(release.set)
        outputs = [record_output(s, monkeypatch) for s in (a, b)]
        acquisitions = []
        acquire = b.kernel.store.acquire
        monkeypatch.setattr(b.kernel.store, "acquire", lambda *args, **kw: acquisitions.append(kw) or acquire(*args, **kw))
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        request(b, 3, "thread/resume", {"threadId": "thread"})
        b.run_adapter.join()
        assert len(acquisitions) == 1
        request(b, 4, "turn/followUp", submission(identity="next", expected=tid))
        b.run_adapter.join()
        assert len(acquisitions) == 2  # One attempt for each explicit wake, no lease polling.
        release.set()
        a.run_adapter.join()
        b.run_adapter.join()
        expected = [(method, turn) for turn in (tid, "thread/turn/next") for method in ("turn/started", "turn/completed")]
        for messages in outputs:
            actual = [
                (m["method"], m["params"]["turnId"]) for m in messages if m.get("method") in {"turn/started", "turn/completed"}
            ]
            assert actual == expected
        assert calls == ["m", "m"]
        assert request(b, 5, "thread/start", {"inputId": "thread"})["result"]["status"] == "terminal"


def test_authorization_precedes_every_thread_dispatch(tmp_path, monkeypatch):
    class Host(DefaultAppServerHost):
        def authorize(self, connection_id, method, params):
            target = params.get("inputId") if method == "thread/start" else params.get("threadId")
            return not (connection_id == "B" and target == "T")

    server = AppServer(
        transport=ChannelTransport(connection_id="B"),
        host=Host(run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))),
    )
    try:
        request(server, 0, "initialize", {"clientInfo": {"name": "B"}})
        server.kernel.create("T", ".", {"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}})
        monkeypatch.setattr(server.store, "read_thread", lambda *a, **kw: pytest.fail("Unauthorized read"))
        monkeypatch.setattr(server.store, "list_threads", lambda **kw: pytest.fail("Unauthorized list"))
        assert request(server, "create", "thread/start", {"inputId": "T"})["error"]["code"] == -32027
        for identity, method in enumerate(CLIENT_METHODS, 1):
            if method.startswith(("thread/", "turn/")) or method == "approval/resolve":
                response = request(server, identity, method, {"threadId": "T", "inputId": "T"})
                expected_error: dict[str, object] = {"code": -32027, "message": "permission_denied"}
                if method == "thread/toolResult/read":
                    expected_error["data"] = {"reason": "permission_denied"}
                assert response["error"] == expected_error, method
        monkeypatch.undo()
        server.kernel.create("U", ".", {"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}})
        listed = request(server, "list", "thread/list", {})["result"]["threads"]
        assert [t["threadId"] for t in listed] == ["U"]
        assert not server.state_manager.subscribers("T")
        assert server.kernel.store.peek_inbox("T") == ()
    finally:
        server.kernel.close()


@pytest.mark.parametrize("method", CLIENT_METHODS)
def test_exactly_one_response_per_request_including_closed_params(tmp_path, monkeypatch, method):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)
        params = submission(expected="thread/turn/missing") | {"unknown": True}
        request(server, "review", method, params)
        responses = [m for m in messages if m.get("id") == "review" and ("result" in m or "error" in m)]
        assert len(responses) == 1


def test_post_receipt_start_failure_is_a_notification(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        request(server, 1, "thread/start", {"inputId": "thread"})
        messages = record_output(server, monkeypatch)

        def fail(*args, **kw):
            raise RuntimeError("start failed after durable admission")

        monkeypatch.setattr(server.run_adapter, "_start", fail)
        assert request(server, 2, "turn/start", submission())["result"]["accepted"]
        assert Counter(m["id"] for m in messages if "id" in m) == {2: 1}
        assert any(m.get("method") == "error/warning" and m["params"]["code"] == "drive_start" for m in messages)


@pytest.mark.parametrize(
    "identity",
    [
        "parent/child/digest",
        "approval/rid/answer",
        "app_server/archive",
        "controller/hash",
        " leading",
        "trailing ",
        "newline\n",
        "x" * 129,
        "",
        "模型",
    ],
)
def test_client_ids_cannot_squat_internal_sessions_or_inputs(tmp_path, identity):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        assert request(server, 1, "thread/start", {"inputId": identity})["error"]["code"] == -32602
        request(server, 2, "thread/start", {"inputId": "thread"})
        for method in ("turn/start", "turn/steer", "turn/followUp"):
            assert (
                request(server, method, method, submission(identity=identity, expected="thread/turn/initial"))["error"]["code"]
                == -32602
            )
        for method, params in (
            ("thread/model/select", {"threadId": "thread", "inputId": identity, "selection": None}),
            (
                "turn/action",
                {"threadId": "thread", "turnId": "thread/turn/initial", "actionId": identity, "action": {"kind": "cancel"}},
            ),
            ("turn/interrupt", {"threadId": "thread", "expectedTurnId": "thread/turn/initial", "actionId": identity}),
        ):
            assert request(server, method, method, params)["error"]["code"] == -32602
        assert server.kernel.store.peek_inbox("thread") == ()


def test_child_approval_answer_from_b_wakes_a_while_a_stays_connected(tmp_path):
    effects = []

    @function_tool(needs_approval=True)
    def protected() -> str:
        effects.append("once")
        return "ok"

    child = Agent("child", "Call protected.", tools=[protected])
    parent = Agent("parent", "Delegate.", tools=[child.as_tool(name="child")])
    provider = ScriptedModelProvider.new(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("child-call", "child", {"task_description": "go"})]),
            LLMResponse("", [ToolCall("protected-call", "protected", {})]),
            LLMResponse("child done"),
            LLMResponse("parent done"),
        ],
    )
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider, agent=parent)
        b = node(stack, tmp_path / "store", "B", provider, agent=parent)
        request(a, 1, "thread/start", {"inputId": "thread"})
        tid = request(a, 2, "turn/start", submission())["result"]["turnId"]
        while (notice := a.transport.receive_outbound(timeout=5)).get("method") != "approval/request":
            pass
        rid = notice["params"]["requestId"]
        assert a.router.is_registered("A")
        params = {"threadId": "thread", "turnId": tid, "requestId": rid, "decision": "allow"}
        assert request(b, 3, "approval/resolve", params)["result"] == {}
        a.run_adapter.join()
        b.run_adapter.join()
        assert effects == ["once"], {
            sid: [
                (r.record.kind, r.record.payload)
                for r in a.kernel.store.read_state(sid)[1]
                if r.record.kind in {"input_applied", "turn_ended", "op_parked"}
            ]
            for sid in a.kernel.store.list_sessions(limit=100)
        }
        assert a.store.read_thread("thread").turns[0].result["finalOutput"] == "parent done"
        assert a.router.pending_server_request_count() == 0
        assert request(b, 4, "approval/resolve", params)["result"] == {}
        assert request(b, 5, "approval/resolve", params | {"decision": "deny"})["error"]["code"] == -32024


def test_start_or_attach_and_compare_clear_with_barriers(tmp_path, monkeypatch):
    cleared, allow_completion, pushed, resumed, release_model = (Event() for _ in range(5))
    calls = []

    def model(req):
        calls.append(req)
        if len(calls) == 1:
            return LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose?"})])
        resumed.set()
        assert release_model.wait(10)
        return LLMResponse("done")

    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", model))
        stack.callback(release_model.set)
        stack.callback(allow_completion.set)
        request(server, 1, "thread/start", {"inputId": "thread"})
        clear = server.state_manager.clear_active_turn
        starts = []
        start = server.kernel.start

        def counted(*args, **kw):
            handle = start(*args, **kw)
            starts.append(handle)
            return handle

        def barrier_clear(*args, **kw):
            result = clear(*args, **kw)
            if len(starts) == 1:
                cleared.set()
                assert allow_completion.wait(10)
            return result

        push = server.kernel.push

        def barrier_push(sid, item):
            receipt = push(sid, item)
            if item.input_id.startswith("controller/"):
                pushed.set()
            return receipt

        monkeypatch.setattr(server.kernel, "start", counted)
        monkeypatch.setattr(server.kernel, "push", barrier_push)
        monkeypatch.setattr(server.state_manager, "clear_active_turn", barrier_clear)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        assert cleared.wait(5)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                server.run_adapter.controller_action,
                thread_id="thread",
                turn_id=tid,
                action_id="reply",
                action={
                    "kind": "respond",
                    "target": displayed_target(server.kernel, "thread").to_dict(),
                    "answer": {"kind": "text", "text": "go"},
                },
                connection_id="A",
            )
            assert pushed.wait(5)
            allow_completion.set()
            assert future.result(timeout=5)["accepted"]
        assert resumed.wait(5)
        assert len(starts) == 2
        active = server.state_manager.active_turn("thread")
        assert active.handle is starts[1]
        assert not clear("thread", tid, handle=starts[0])
        assert server.state_manager.active_turn("thread") is active
        release_model.set()
        server.run_adapter.join()


def test_approval_inbox_polling_uses_capped_backoff(monkeypatch):
    from types import SimpleNamespace

    from vv_agent.app_server.host import AppServerApprovalProvider
    from vv_agent.app_server.outgoing import OutgoingRouter
    from vv_agent.app_server.protocol import RequestId
    from vv_agent.approval import ApprovalRequest

    router = OutgoingRouter()
    router.register_transport(ChannelTransport(connection_id="A"))
    timeouts, reads = [], []

    def pending_result(timeout):
        timeouts.append(timeout)
        raise TimeoutError

    pending = SimpleNamespace(request_id=RequestId("rid"), result=pending_result)
    monkeypatch.setattr(router, "send_server_request", lambda *args, **kw: pending)

    def answered(request):
        reads.append(request.request_id)
        return len(reads) == 9

    provider = AppServerApprovalProvider(
        connection_id="A",
        thread_id="thread",
        turn_id="turn",
        router=router,
        timeout_seconds=None,
        has_durable_answer=answered,
    )
    assert provider.decide(ApprovalRequest("rid", "protected", "call")) is None
    assert timeouts == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0]


def test_controller_digest_cannot_be_squatted_in_client_input_namespace(tmp_path):
    from vv_agent.interaction import derive_controller_command_id

    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose?"})])])
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider)
        request(server, 1, "thread/start", {"inputId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        digest = derive_controller_command_id("thread", tid, "cancel")
        assert request(
            server,
            3,
            "thread/model/select",
            {
                "threadId": "thread",
                "inputId": digest,
                "selection": None,
            },
        )["result"]["accepted"]
        assert request(
            server,
            4,
            "turn/interrupt",
            {
                "threadId": "thread",
                "expectedTurnId": tid,
                "actionId": "cancel",
            },
        )["result"]["accepted"]
        server.run_adapter.join()
        assert server.kernel.store.read_input("thread", digest).item.kind == "model_selection"
        assert server.kernel.store.read_input("thread", "controller/" + digest).item.kind == "control"


def test_action_conflict_and_replay_precede_a_later_active_turn(tmp_path):
    entered, release = Event(), Event()
    calls = []

    def model(req):
        calls.append(req)
        if len(calls) == 1:
            return LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose?"})])
        entered.set()
        assert release.wait(10)
        return LLMResponse("done")

    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", model))
        stack.callback(release.set)
        request(server, 1, "thread/start", {"inputId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        action = {"threadId": "thread", "turnId": tid, "actionId": "pause", "action": {"kind": "suspend"}}
        receipt = request(server, 3, "turn/action", action)["result"]
        server.run_adapter.join()
        request(server, 4, "turn/interrupt", {"threadId": "thread", "expectedTurnId": tid, "actionId": "end"})
        server.run_adapter.join()
        request(server, 5, "turn/start", submission(identity="second", expected=tid))
        assert entered.wait(5)
        assert request(server, 6, "turn/action", action)["result"] == receipt
        assert request(server, 7, "turn/action", action | {"action": {"kind": "cancel"}})["error"]["code"] == -32024
        release.set()
        server.run_adapter.join()


def test_observer_cannot_overtake_local_completion_between_clear_and_notify(tmp_path, monkeypatch):
    from threading import current_thread

    cleared, observer_attempted, release = Event(), Event(), Event()
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", [LLMResponse("done")]))
        stack.callback(release.set)
        messages = record_output(server, monkeypatch)
        clear = server.state_manager.clear_active_turn
        turn_lock = server.state_manager.turn_lock

        def barrier_clear(*args, **kwargs):
            result = clear(*args, **kwargs)
            cleared.set()
            assert release.wait(10)
            return result

        class ObservedLock:
            def __init__(self, lock):
                self.lock = lock

            def __enter__(self):
                if current_thread().name == "app-server-notifications" and cleared.is_set():
                    observer_attempted.set()
                self.lock.acquire()

            def __exit__(self, *args):
                self.lock.release()

        monkeypatch.setattr(server.state_manager, "clear_active_turn", barrier_clear)
        monkeypatch.setattr(server.state_manager, "turn_lock", lambda sid: ObservedLock(turn_lock(sid)))
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(server, 2, "turn/start", submission())
        assert cleared.wait(5)
        assert observer_attempted.wait(5)
        assert not any(m.get("method") == "turn/completed" for m in messages)
        release.set()
        server.run_adapter.join()
        lifecycle = [m for m in messages if m.get("method") in {"turn/started", "turn/completed"}]
        assert [m["method"] for m in lifecycle] == ["turn/started", "turn/completed"]
        terminal = next(
            i
            for i, m in enumerate(messages)
            if m.get("method") == "thread/status/changed" and m["params"]["status"] == "terminal"
        )
        completed = next(i for i, m in enumerate(messages) if m.get("method") == "turn/completed")
        assert terminal < completed
