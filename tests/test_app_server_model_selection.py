"""App Server admits strict selection commands into the canonical session inbox."""

import json
from contextlib import suppress
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, ValidationError

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost, ThreadModelSelectParams
from vv_agent.app_server.host import model_summary
from vv_agent.app_server.schema import json_schema_bundle
from vv_agent.kernel._requirements import SessionRequirements, normalize_requirements
from vv_agent.model import ModelError, ModelRef, ScriptedModelProvider, resolve_model_selection
from vv_agent.session.runtime import static_model_selection_requirements, static_session_requirements


def _selection_bootstrap(selections, resolve_model):
    capabilities = set()
    for selection in selections:
        with suppress(ModelError, ValueError):
            capabilities.update(
                static_model_selection_requirements(
                    selection,
                    resolve_model=resolve_model,
                    handler_version="1",
                ).capabilities
            )
    return SessionRequirements(2, "1", tuple(sorted(capabilities)))


@pytest.mark.capability("app-server-model-selection")
def test_model_selection_method_admission_replay_conflict_and_schema():
    transport = ChannelTransport(connection_id="test")
    provider = ScriptedModelProvider.new("test", "m", [])
    agent = Agent("test", "Answer.", model=ModelRef.named("m"))
    config = RunConfig(model_provider=provider)
    host = DefaultAppServerHost(
        agent=agent,
        run_config=config,
        models=[model_summary(provider.resolve(ModelRef.named("m")))],
    )
    server = AppServer(
        transport=transport,
        host=host,
    )

    def request(identity, method, params):
        server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (response := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return response

    try:
        request(0, "initialize", {"clientInfo": {"name": "test"}})
        thread = request(1, "thread/start", {"inputId": "thread_1"})["result"]["threadId"]
        creation = server.kernel.store.read(thread, limit=1).records[0].record
        assert creation.feature_level == 2
        bootstrap = normalize_requirements(creation._payload["bootstrap"])
        session_requirements = static_session_requirements(
            agent,
            config,
            provider.resolve(ModelRef.named("m")),
            level=2,
            handler_version="1",
        )
        selection_requirements = static_model_selection_requirements(
            {"model": "m", "route": {}},
            resolve_model=host.resolve_model,
            handler_version="1",
        )
        assert bootstrap.handler_version == server.kernel.default_handler_version
        assert set(session_requirements.capabilities) | set(selection_requirements.capabilities) <= set(bootstrap.capabilities)
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
        assert stored[0].item.requirements == selection_requirements.to_dict()
        assert cast(Any, server.kernel.store).worker_supports.accepts(
            level=2,
            handler_version=server.kernel.default_handler_version,
            capabilities=selection_requirements.capabilities,
        )
        schema = json.loads(json_schema_bundle()["ClientRequest"])
        validator = Draft202012Validator(schema)
        wire = {"jsonrpc": "2.0", "id": 1, "method": "thread/model/select", "params": params}
        validator.validate(wire)
        with pytest.raises(ValidationError):
            validator.validate(wire | {"params": params | {"extra": True}})
    finally:
        server.run_adapter.join()
        server.kernel.close()


def test_model_selection_without_trusted_model_config_is_rejected_before_admission():
    transport = ChannelTransport(connection_id="test")
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(agent=Agent("test", "Answer."), run_config=RunConfig()),
    )

    def request(identity, method, params):
        server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (response := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return response

    try:
        request(0, "initialize", {"clientInfo": {"name": "test"}})
        thread = request(1, "thread/start", {"inputId": "thread_1"})["result"]["threadId"]
        creation = server.kernel.store.read(thread, limit=1).records[0].record
        assert creation.feature_level == 1 and "bootstrap" not in creation._payload
        response = request(
            2,
            "thread/model/select",
            {"threadId": thread, "inputId": "command", "selection": {"model": "m", "route": {}}},
        )
        assert response["error"] == {
            "code": -32602,
            "message": "model selection requires a level-2 session route with a bound handler",
            "data": {"reason": "model_selection_unavailable"},
        }
        assert server.kernel.store.peek_inbox(thread) == ()
    finally:
        server.run_adapter.join()
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
    session_provider = VvLlmModelProvider.from_settings_file(settings).with_default_backend("test")
    level_two = _selection_bootstrap(
        [{"model": name + suffix, "route": {}} for name in ("a", "b")],
        lambda request: resolve_model_selection(session_provider, request),
    )
    driver = None
    server = None
    original_create = SessionDriver.create
    original_runtime = SessionDriver.runtime

    def runtime(self, agent, config, task=None, **kwargs):
        from dataclasses import replace

        from vv_agent.model import ModelRef

        if qualified:
            agent = replace(agent, model=ModelRef.backend("test", "a"))
            config = replace(config, model=None)
        return original_runtime(self, agent, config, task, **kwargs)

    monkeypatch.setattr(SessionDriver, "runtime", runtime)

    def create(self, *args, **kwargs):
        self.capabilities = tuple(sorted(set(self.capabilities) | set(level_two.capabilities)))
        self.model_resolver = lambda request: resolve_model_selection(session_provider, request)
        self._configure_static_worker(
            level=2,
            capabilities=level_two.capabilities,
            requirements=level_two,
        )
        original_create(self, *args, **kwargs)
        sid = args[0]
        if selected:
            self.push(sid, InboxItem("selection", "model_selection", {"selection": {"model": "b" + suffix, "route": {}}}))

    try:
        if surface == "app":
            host = _build_app_server_host(settings_file=settings, backend="test", model="a", workspace=tmp_path, max_cycles=3)
            transport = ChannelTransport(connection_id="test")
            server = AppServer(transport=transport, host=host)
            driver = server.kernel

            def send(identity, method, params):
                server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
                while (wire := transport.receive_outbound(timeout=5)).get("id") != identity:
                    pass
                return wire

            send(0, "initialize", {"clientInfo": {"name": "test"}})
            sid = send(1, "thread/start", {"inputId": "thread"})["result"]["threadId"]
            creation = driver.store.read(sid, limit=1).records[0].record
            assert creation.feature_level == 2
            assert normalize_requirements(creation._payload["bootstrap"]).handler_version == "1"
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
            server.wake(sid)
            from test_app_server_distributed import eventually

            eventually(
                lambda: any(
                    row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "unknown"
                    for row in driver.store.read_state(sid)[1]
                )
            )
            server.run_adapter.join()
        elif surface == "cli":
            driver = SessionDriver()
            driver._ensure_static_worker_support(level=2, capabilities=level_two.capabilities)
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
            driver = SessionDriver()
            driver._ensure_static_worker_support(level=2, capabilities=level_two.capabilities)
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
        from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store

        hydration = hydration_for_session_store(driver.store, sid)
        plans = [r for r in records if r.record.kind == "op_planned" and r.record.payload["op_kind"] == "model"]
        plan_request = hydrate_record_payload(plans[0], hydration)["request"]
        assert plan_request["model"] == expected_model
        events = [e.to_dict() for e in project_records(records, hydration=hydration) if e.type == "model_call_completed"]
        assert events[0]["model"] == expected_model and events[0]["backend"] == "test"
        from vv_agent.session.result import project_result

        usage = project_result(driver.store, sid, next(reversed(state.turns))).token_usage.model_calls
        assert len(usage) == 1 and usage[0].model == expected_model and usage[0].backend == "test"
        if surface == "app":
            rejected = next(
                row
                for row in records
                if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "unknown"
            )
            payload = hydrate_record_payload(rejected, hydration_for_session_store(driver.store, sid))
            assert payload["reason"] == "model_selection_unresolvable"
    finally:
        if server is not None:
            server.run_adapter.join()
            server.kernel.close()
        elif driver is not None:
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

    host = RoutedHost(
        agent=Agent("test", "Answer.", model="a"),
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("default-backend", "a", [LLMResponse("wrong client")])),
    )
    transport = ChannelTransport(connection_id="host-hook")
    server = AppServer(
        transport=transport,
        host=host,
    )

    def send(identity, method, params):
        server.processor.process_message("host-hook", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
        while (wire := transport.receive_outbound(timeout=5)).get("id") != identity:
            pass
        return wire

    try:
        from test_app_server_distributed import eventually

        send(0, "initialize", {"clientInfo": {"name": "host-hook"}})
        send(1, "thread/start", {"inputId": "thread"})
        send(
            2,
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": {"model": "b", "route": {"pool": "authorized"}}},
        )
        send(3, "turn/start", {"threadId": "thread", "inputId": "prompt", "expectedTurnId": None, "input": []})
        server.wake("thread")
        try:
            eventually(
                lambda: bool(
                    (turns := server.store.read_thread("thread").turns)
                    and turns[0].result.get("finalOutput") == "selected client"
                )
            )
        except AssertionError:
            expected = static_model_selection_requirements(
                {"model": "b", "route": {"pool": "authorized"}},
                resolve_model=host.resolve_model,
                handler_version=server.kernel.default_handler_version,
            )
            state, records, _ = server.kernel.store.read_state("thread")
            support = server.kernel._worker.support
            snapshot = server.store.read_thread("thread")
            diagnostic = {
                "errors": {
                    sid: {"turn_id": failure.turn_id, "attempts": failure.attempts, "error": repr(failure.error)}
                    for sid, failure in server.kernel._errors.items()
                },
                "lease": server.kernel.store.lease_status("thread"),
                "worker": {
                    "started": server.kernel._worker.started,
                    "running": server.kernel._worker.running,
                    "last_error": repr(server.kernel._worker.last_error),
                    "active_wake_sessions": sorted(server.kernel._worker.active_wake_sessions),
                    "inflight_lease": repr(server.kernel._worker._inflight_lease),
                    "support": {
                        "level": support.level,
                        "handler_versions": support.handler_versions,
                        "capabilities": len(support.capabilities),
                        "missing_selection": sorted(set(expected.capabilities) - set(support.capabilities)),
                    },
                },
                "records": [
                    {
                        "seq": row.seq,
                        "kind": row.record.kind,
                        "turn_id": row.record.turn_id,
                        "operation_id": row.record.operation_id,
                        "attempt": row.record.attempt,
                        "input_id": row.record.payload.get("input", {}).get("input_id"),
                        "input_kind": row.record.payload.get("input", {}).get("kind"),
                        "disposition": row.record.payload.get("disposition"),
                        "reason": row.record.payload.get("reason"),
                        "error": row.record.payload.get("error"),
                        "status": row.record.payload.get("status"),
                    }
                    for row in records
                ],
                "state": {
                    "active_turn_id": state.active_turn_id,
                    "turns": list(state.turns),
                    "model_selection": state.model_selection.encode() if state.model_selection is not None else None,
                },
                "turn": snapshot.turns[0].result if snapshot.turns else None,
                "resolved_requests": [
                    {"model": request.get("model"), "route": request.get("route"), "keys": sorted(request)}
                    for request in resolved_requests
                ],
            }
            pytest.fail(repr(diagnostic))
        server.run_adapter.join()
        result = server.store.read_thread("thread").turns[0].result
        assert result.get("finalOutput") == "selected client", result
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
        server.wake("thread")
        server.run_adapter.join()
        from test_app_server_distributed import eventually

        from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store

        eventually(
            lambda: any(
                row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "ambiguous"
                for row in server.kernel.store.read_state("thread")[1]
            )
        )

        _, records, _ = server.kernel.store.read_state("thread")
        rejected = next(
            row
            for row in records
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "ambiguous"
        )
        payload = hydrate_record_payload(rejected, hydration_for_session_store(server.kernel.store, "thread"))
        assert payload["reason"] == "model_selection_unresolvable"
        eventually(
            lambda: (
                bool((turns := server.store.read_thread("thread").turns) and turns[0].result.get("finalOutput") == "done")
                and seen == ["a-2024"]
            )
        )
        assert seen == ["a-2024"]
    finally:
        server.run_adapter.join()
        server.kernel.close()
