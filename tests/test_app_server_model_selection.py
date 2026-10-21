"""App Server admits strict selection commands into the canonical session inbox."""

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost, ThreadModelSelectParams
from vv_agent.app_server.schema import json_schema_bundle
from vv_agent.model import ScriptedModelProvider


def test_model_selection_method_admission_replay_conflict_and_schema():
    transport = ChannelTransport(connection_id="test")
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(
            agent=Agent("test", "Answer."), run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []))
        ),
    )

    def request(identity, method, params):
        server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (response := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return response

    try:
        request(0, "initialize", {"clientInfo": {"name": "test"}})
        thread = request(1, "thread/start", {"inputId": "thread_1"})["result"]["threadId"]
        params = ThreadModelSelectParams(thread, "command", {"model": "m", "route": {}}).to_dict()
        expected = {"threadId": thread, "inputId": "command", "accepted": True}
        assert request(2, "thread/model/select", params)["result"] == expected
        assert request(3, "thread/model/select", params)["result"] == expected
        assert request(4, "thread/model/select", params | {"selection": None})["error"]["code"] == -32024
        for identity, bad in enumerate(
            [
                params | {"extra": 1},
                {"threadId": thread},
                params | {"inputId": ""},
                params | {"selection": {"model": "m", "route": []}},
                params | {"selection": {"model": "m", "route": {}, "model_settings": {"max_tokens": False}}},
            ],
            5,
        ):
            assert request(identity, "thread/model/select", bad)["error"]["code"] == -32602
        stored = server.kernel.store.peek_inbox(thread)
        assert len(stored) == 1 and stored[0].item.kind == "model_selection"
        assert stored[0].item.target_turn_id is None and stored[0].item.generation is None
        schema = json.loads(json_schema_bundle()["ClientRequest"])
        validator = Draft202012Validator(schema)
        wire = {"jsonrpc": "2.0", "id": 1, "method": "thread/model/select", "params": params}
        validator.validate(wire)
        with pytest.raises(ValidationError):
            validator.validate(wire | {"params": params | {"extra": True}})
    finally:
        server.kernel.close()


@pytest.mark.parametrize("surface", ["app", "cli", "interactive"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("catalog_kind", ["base", "cross_backend", "endpoint_id"])
def test_default_surfaces_resolve_exact_model_id_and_bill_actual_model(
    tmp_path, monkeypatch, capsys, surface, selected, qualified, catalog_kind
):
    from vv_agent import AgentSessionOptions, create_agent_session
    from vv_agent.cli import _build_app_server_host, _run_task_cli
    from vv_agent.model import VvLlmModelProvider
    from vv_agent.session.projection import project_records
    from vv_agent.session.records import InboxItem
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse

    models: dict[str, Any] = {name: {"id": name + "-2024", "endpoints": ["local"]} for name in ("a", "b")}
    if catalog_kind == "endpoint_id":
        for name, model in models.items():
            model["endpoints"] = [{"endpoint_id": "local", "model_id": name + "-runtime"}]
    backends = {"test": {"models": models}}
    if catalog_kind == "cross_backend":
        backends["mirror"] = {"models": models}
    suffix = "-runtime" if catalog_kind == "endpoint_id" else "-2024"
    settings = tmp_path / "catalog.json"
    settings.write_text(
        json.dumps(
            {
                "VERSION": "2",
                "backends": backends,
                "endpoints": [{"id": "local", "api_key": "placeholder", "api_base": "https://example.invalid/v1"}],
            }
        )
    )
    seen = []

    def client(self, resolved):
        def call(request):
            seen.append(request.model)
            return LLMResponse("done", raw={"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})

        from vv_agent.model import ScriptedModelProvider

        return ScriptedModelProvider.from_callback("test", resolved.model_id, call).llm

    monkeypatch.setattr(VvLlmModelProvider, "client", client)
    driver = SessionDriver()
    original_create = SessionDriver.create
    original_runtime = SessionDriver.runtime

    def runtime(self, agent, config, task=None):
        from dataclasses import replace

        from vv_agent.model import ModelRef

        if qualified:
            agent = replace(agent, model=ModelRef.backend("test", "a"))
            config = replace(config, model=None)
        return original_runtime(self, agent, config, task)

    monkeypatch.setattr(SessionDriver, "runtime", runtime)

    def create(self, *args, **kwargs):
        original_create(self, *args, **kwargs)
        sid = args[0]
        if selected:
            self.push(sid, InboxItem("selection", "model_selection", {"selection": {"model": "b" + suffix, "route": {}}}))

    try:
        if surface == "app":
            host = _build_app_server_host(settings_file=settings, backend="test", model="a", workspace=tmp_path, max_cycles=3)
            transport = ChannelTransport(connection_id="test")
            server = AppServer(transport=transport, store=driver.store, host=host)

            def send(identity, method, params):
                server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
                while (wire := transport.receive_outbound(timeout=5)).get("id") != identity:
                    pass
                return wire

            send(0, "initialize", {"clientInfo": {"name": "test"}})
            sid = send(1, "thread/start", {"inputId": "thread"})["result"]["threadId"]
            models = send("models", "model/list", {})["result"]["models"]
            assert [model["id"] for model in models] == ["a" + suffix]
            for model in models:
                send(
                    "listed",
                    "thread/model/select",
                    {"threadId": sid, "inputId": "listed", "selection": {"model": model["id"], "route": {}}},
                )
            if selected:
                send(
                    2,
                    "thread/model/select",
                    {"threadId": sid, "inputId": "selection", "selection": {"model": "b" + suffix, "route": {}}},
                )
            send(
                3, "thread/model/select", {"threadId": sid, "inputId": "unknown", "selection": {"model": "unknown", "route": {}}}
            )
            send(
                4,
                "turn/start",
                {"threadId": sid, "inputId": "prompt", "expectedTurnId": None, "input": [{"type": "text", "text": "go"}]},
            )
            server.run_adapter.join()
        elif surface == "cli":
            monkeypatch.setattr(SessionDriver, "create", create)
            assert (
                _run_task_cli(
                    [
                        "--prompt",
                        "go",
                        "--settings-file",
                        str(settings),
                        "--backend",
                        "test",
                        "--model",
                        "a",
                        "--workspace",
                        str(tmp_path),
                    ],
                    store=driver.store,
                )
                == 0
            )
            sid = "cli"
        else:
            monkeypatch.setattr(SessionDriver, "create", create)
            session = create_agent_session(
                agent=Agent("test", "Answer.", model="a"),
                session_id="interactive",
                options=AgentSessionOptions(
                    model_provider=VvLlmModelProvider.from_settings_file(settings).with_default_backend("test"),
                    workspace=tmp_path,
                    session_store=driver.store,
                ),
            )
            session.prompt("go")
            sid = "interactive"
        state, records, _ = driver.store.read_state(sid)
        expected_model = ("b" if selected else "a") + suffix
        assert seen == [expected_model]
        plans = [r.record for r in records if r.record.kind == "op_planned" and r.record.payload["op_kind"] == "model"]
        assert plans[0].payload["request"]["model"] == expected_model
        events = [e.to_dict() for e in project_records(records) if e.type == "model_call_completed"]
        assert events[0]["model"] == expected_model and events[0]["backend"] == "test"
        from vv_agent.session.result import project_result

        usage = project_result(driver.store, sid, next(reversed(state.turns))).token_usage.model_calls
        assert len(usage) == 1 and usage[0].model == expected_model and usage[0].backend == "test"
        if surface == "app":
            assert state.applied_inputs["unknown"].payload["reason"] == "model_selection_unresolvable"
    finally:
        driver.close()


def test_custom_app_server_host_resolver_controls_real_selected_call_and_usage():
    from vv_agent.model import ModelRef
    from vv_agent.types import LLMResponse

    selected = ScriptedModelProvider.new("selected-backend", "b", [LLMResponse("selected client")])
    resolved_requests = []

    class RoutedHost(DefaultAppServerHost):
        def resolve_model(self, request):
            resolved_requests.append(request)
            assert request["model"] == "b" and request["route"] == {"pool": "authorized"}
            resolved = selected.resolve(ModelRef.named("b"))
            return selected.client(resolved), resolved

    transport = ChannelTransport(connection_id="host-hook")
    server = AppServer(
        transport=transport,
        host=RoutedHost(
            agent=Agent("test", "Answer.", model="a"),
            run_config=RunConfig(model_provider=ScriptedModelProvider.new("default-backend", "a", [LLMResponse("wrong client")])),
        ),
    )

    def send(identity, method, params):
        server.processor.process_message("host-hook", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (wire := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return wire

    try:
        send(0, "initialize", {"clientInfo": {"name": "host-hook"}})
        send(1, "thread/start", {"inputId": "thread"})
        send(
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": {"model": "b", "route": {"pool": "authorized"}}},
        )
        send(3, "turn/start", {"threadId": "thread", "inputId": "prompt", "expectedTurnId": None, "input": []})
        server.run_adapter.join()
        result = server.store.read_thread("thread").turns[0].result
        assert result["finalOutput"] == "selected client"
        calls = result["tokenUsage"]["modelCalls"]
        assert len(calls) == 1 and (calls[0]["backend"], calls[0]["model"]) == ("selected-backend", "b")
        assert len(resolved_requests) >= 2
    finally:
        server.run_adapter.join()
        server.kernel.close()


@pytest.mark.parametrize("endpoint_ids", [False, True])
def test_ambiguous_model_id_rejects_selection_in_real_runtime(tmp_path, monkeypatch, endpoint_ids):
    from vv_agent.cli import _build_app_server_host
    from vv_agent.model import VvLlmModelProvider
    from vv_agent.types import LLMResponse

    duplicate = {"id": "b-2024", "endpoints": ["local"]}
    if endpoint_ids:
        duplicate = {"id": "base-b", "endpoints": [{"endpoint_id": "local", "model_id": "b-2024"}]}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "VERSION": "2",
                "backends": {
                    "test": {
                        "models": {
                            "a": {"id": "a-2024", "endpoints": ["local"]},
                            "b": duplicate,
                            "b-other": duplicate,
                        }
                    }
                },
                "endpoints": [{"id": "local", "api_key": "placeholder", "api_base": "https://example.invalid/v1"}],
            }
        )
    )
    seen = []

    def client(self, resolved):
        return ScriptedModelProvider.from_callback(
            "test", resolved.model_id, lambda req: seen.append(req.model) or LLMResponse("done")
        ).llm

    monkeypatch.setattr(VvLlmModelProvider, "client", client)
    server = AppServer(
        transport=ChannelTransport(connection_id="A"),
        host=_build_app_server_host(settings_file=catalog, backend="test", model="a", workspace=tmp_path, max_cycles=3),
    )
    from test_app_server_distributed import request, submission

    try:
        request(server, 0, "initialize", {"clientInfo": {"name": "test"}})
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(
            server,
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "ambiguous", "selection": {"model": "b-2024", "route": {}}},
        )
        request(server, 3, "turn/start", submission())
        server.run_adapter.join()
        state = server.kernel.store.read_state("thread")[0]
        assert state.applied_inputs["ambiguous"].payload["reason"] == "model_selection_unresolvable"
        assert seen == ["a-2024"]
    finally:
        server.kernel.close()
