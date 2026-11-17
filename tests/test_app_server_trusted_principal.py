from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from session.app_server_s1_helpers import bind_development_content

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.host import (
    AgentResolutionRequest,
    AppServerHost,
    ModelListRequest,
    ModelListResponse,
    RunConfigResolutionRequest,
    ToolResultResourceAuthorizationRequest,
    TrustedPrincipal,
)
from vv_agent.interaction import derive_controller_command_id
from vv_agent.kernel._content import Principal
from vv_agent.llm.base import LlmRequest
from vv_agent.session.projection import hydrate_record_payload
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

PRINCIPAL: dict[str, Any] = {
    "tenant_id": "tenant-a",
    "subject_id": "trusted-ingress-user",
    "subject_kind": "user",
    "attributes": {"role": "trusted-principal-marker"},
}


def _user_principal(subject_id: str, *, tenant_id: str = "tenant-a") -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "subject_kind": "user",
        "attributes": {"role": subject_id},
    }


class _TrustedHost(DefaultAppServerHost):
    def __init__(self, principals: dict[str, TrustedPrincipal | None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._principals = principals
        self.resolved: list[tuple[str, str, dict[str, Any]]] = []
        self.authorized: list[tuple[str, str, dict[str, Any]]] = []

    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        self.authorized.append((connection_id, method, dict(params)))
        return True

    def resolve_principal(
        self,
        connection_id: str,
        method: str,
        params: dict[str, Any],
    ) -> TrustedPrincipal | None:
        self.resolved.append((connection_id, method, dict(params)))
        principal = self._principals.get(connection_id)
        return dict(principal) if isinstance(principal, dict) else principal


class _LegacyHost(AppServerHost):
    def __init__(self, *, agent: Agent, run_config: RunConfig) -> None:
        self._agent = agent
        self._run_config = run_config
        self._default = DefaultAppServerHost(agent=agent, run_config=run_config)

    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        del connection_id, method, params
        return True

    def authorize_tool_result_resource(self, request: ToolResultResourceAuthorizationRequest) -> bool:
        return self._default.authorize_tool_result_resource(request)

    def resolve_model(self, request: dict[str, Any]):
        return self._default.resolve_model(request)

    def resolve_agent(self, request: AgentResolutionRequest) -> Agent:
        del request
        return self._agent

    def build_run_config(self, request: RunConfigResolutionRequest) -> RunConfig:
        del request
        return self._run_config

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        del request
        return ModelListResponse(models=[])


def _app_server(
    *,
    host: Any,
    database: Path,
    principal: dict[str, Any] | None = None,
    connection_id: str = "tenant-a",
) -> tuple[AppServer, ChannelTransport, SQLiteStore]:
    root_store = SQLiteStore(str(database))
    root_store.install_schema()
    bind_development_content(root_store)
    store = root_store.for_tenant(principal["tenant_id"] if principal is not None else "tenant-a")
    transport = ChannelTransport(connection_id=connection_id)
    server = AppServer(host=host, store=store, transport=transport)
    return server, transport, root_store


def _request(
    server: AppServer,
    transport: ChannelTransport,
    request_id: int,
    method: str,
    params: dict[str, Any] | None,
    *,
    observed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    transport.send_inbound(payload)
    server.processor.process_next(transport)
    while True:
        message = transport.receive_outbound(timeout=10)
        if observed is not None:
            observed.append(message)
        if message.get("id") == request_id:
            return message


def _initialize(server: AppServer, transport: ChannelTransport) -> None:
    response = _request(
        server,
        transport,
        0,
        "initialize",
        {"clientInfo": {"name": transport.connection_id}},
    )
    assert "result" in response
    transport.send_inbound({"jsonrpc": "2.0", "method": "initialized"})
    server.processor.process_next(transport)


def _actor(server: AppServer, thread_id: str, input_id: str) -> dict[str, Any]:
    stored = server.kernel.store.read_input(thread_id, input_id)
    assert stored is not None
    item = stored.item
    assert item.actor is not None
    return item.actor


def _assert_principal(actor: dict[str, Any], expected: dict[str, Any] = PRINCIPAL) -> None:
    assert {key: actor[key] for key in ("tenant_id", "subject_id", "subject_kind")} == {
        key: expected[key] for key in ("tenant_id", "subject_id", "subject_kind")
    }


def _creation_principal(server: AppServer, thread_id: str) -> dict[str, Any]:
    stored = server.kernel.store.read(thread_id, limit=1).records[0]
    return hydrate_record_payload(stored, server.store._hydration(thread_id))["principal"]


def _close(server: AppServer, root_store: SQLiteStore) -> None:
    server.run_adapter.join()
    server.kernel.close()
    root_store.connection.close()


def test_trusted_principal_reaches_creation_and_all_durable_write_paths(tmp_path) -> None:
    model_requests: list[LlmRequest] = []

    def complete(request: LlmRequest) -> LLMResponse:
        model_requests.append(request)
        return LLMResponse("done")

    provider = ScriptedModelProvider.from_callback("test", "m", complete)
    agent = Agent("assistant", "Answer.", model="m")
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL},
        agent=agent,
        run_config=RunConfig(model_provider=provider, workspace=tmp_path),
    )
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    observed: list[dict[str, Any]] = []
    try:
        _initialize(server, transport)
        started = _request(
            server,
            transport,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
            observed=observed,
        )
        thread_id = started["result"]["threadId"]
        creation = server.kernel.store.read(thread_id, limit=1).records[0].record.payload
        _assert_principal(creation["principal"])

        selected = _request(
            server,
            transport,
            2,
            "thread/model/select",
            {
                "threadId": thread_id,
                "inputId": "model-select",
                "selection": {"model": "m", "route": {}},
            },
            observed=observed,
        )
        assert "result" in selected, selected
        assert selected["result"]["accepted"] is True
        _assert_principal(_actor(server, thread_id, "model-select"))

        turn = _request(
            server,
            transport,
            3,
            "turn/start",
            {
                "threadId": thread_id,
                "inputId": "prompt",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "hello"}],
            },
            observed=observed,
        )
        turn_id = turn["result"]["turnId"]
        _assert_principal(_actor(server, thread_id, "prompt"))
        server.run_adapter.join()

        compacted = _request(
            server,
            transport,
            4,
            "thread/compact/start",
            {"threadId": thread_id, "actionId": "compact"},
            observed=observed,
        )
        compact_input = compacted["result"]["requestId"]
        _assert_principal(_actor(server, thread_id, compact_input))
        server.run_adapter.join()

        archived = _request(
            server,
            transport,
            5,
            "thread/archive",
            {"threadId": thread_id},
            observed=observed,
        )
        assert archived["result"] == {"threadId": thread_id, "archived": True}
        _assert_principal(_actor(server, thread_id, "app_server/archive"))

        status_resolved = len(host.resolved)
        assert "result" in _request(server, transport, 6, "thread/status", {"threadId": thread_id})
        assert len(host.resolved) == status_resolved + 1
        assert [method for _, method, _ in host.resolved] == [
            "thread/start",
            "thread/model/select",
            "turn/start",
            "thread/compact/start",
            "thread/archive",
            "thread/status",
        ]

        serialized_wire = json.dumps(observed, ensure_ascii=False)
        serialized_prompt = "\n".join(str(message.content) for request in model_requests for message in request.messages)
        assert PRINCIPAL["subject_id"] not in serialized_wire
        assert PRINCIPAL["subject_id"] not in serialized_prompt
        assert next(iter(PRINCIPAL["attributes"].values())) not in serialized_prompt
        assert turn_id == f"{thread_id}/turn/prompt"
    finally:
        _close(server, root_store)


def test_persisted_principal_object_is_owner_hydrated_without_reencryption(tmp_path) -> None:
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(model_provider=provider, workspace=tmp_path),
    )
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    try:
        _initialize(server, transport)
        thread_id = _request(
            server,
            transport,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]
        creation = server.kernel.store.read(thread_id, limit=1).records[0].record.payload
        host._principals["tenant-a"] = Principal.from_dict(creation["principal"])

        replay = _request(
            server,
            transport,
            2,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )
        assert replay["result"]["replayed"] is True

        selected = _request(
            server,
            transport,
            3,
            "thread/model/select",
            {
                "threadId": thread_id,
                "inputId": "persisted-principal",
                "selection": {"model": "m", "route": {}},
            },
        )
        assert "result" in selected, selected
        assert selected["result"]["accepted"] is True
        actor = _actor(server, thread_id, "persisted-principal")
        assert (actor["tenant_id"], actor["subject_id"], actor["subject_kind"]) == (
            "tenant-a",
            PRINCIPAL["subject_id"],
            PRINCIPAL["subject_kind"],
        )
        assert (
            server.kernel.read_input_content(
                thread_id,
                "persisted-principal",
                actor["attributes"],
                slot="input.actor.attributes",
            )
            == PRINCIPAL["attributes"]
        )
    finally:
        _close(server, root_store)


def test_processor_passes_trusted_principal_to_control_and_approval_hooks(monkeypatch, tmp_path) -> None:
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(model_provider=provider, workspace=tmp_path),
    )
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    submissions: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    try:
        _initialize(server, transport)
        thread_id = _request(
            server,
            transport,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]
        turn_id = _request(
            server,
            transport,
            2,
            "turn/start",
            {
                "threadId": thread_id,
                "inputId": "prompt",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "hello"}],
            },
        )["result"]["turnId"]
        server.run_adapter.join()

        def submit_input(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            submissions.append(kwargs)
            return ({"threadId": thread_id, "turnId": turn_id, "inputId": kwargs["input_id"], "accepted": True}, {})

        monkeypatch.setattr(server.run_adapter, "submit_input", submit_input)
        for request_id, method, input_id in ((3, "turn/steer", "steer"), (4, "turn/followUp", "follow-up")):
            assert "result" in _request(
                server,
                transport,
                request_id,
                method,
                {
                    "threadId": thread_id,
                    "inputId": input_id,
                    "expectedTurnId": turn_id,
                    "input": [{"type": "text", "text": method}],
                },
            )

        def controller_action(**kwargs: Any) -> dict[str, Any]:
            controls.append(kwargs)
            return {"threadId": thread_id, "turnId": turn_id, "actionId": kwargs["action_id"], "accepted": True}

        monkeypatch.setattr(server.run_adapter, "controller_action", controller_action)
        assert "result" in _request(
            server,
            transport,
            5,
            "turn/interrupt",
            {"threadId": thread_id, "expectedTurnId": turn_id, "actionId": "interrupt"},
        )
        assert "result" in _request(
            server,
            transport,
            6,
            "turn/action",
            {"threadId": thread_id, "turnId": turn_id, "actionId": "cancel", "action": {"kind": "cancel"}},
        )

        def resolve_approval(**kwargs: Any) -> None:
            approvals.append(kwargs)

        monkeypatch.setattr(server.run_adapter, "resolve_approval", resolve_approval)
        assert (
            _request(
                server,
                transport,
                7,
                "approval/resolve",
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "requestId": "approval",
                    "decision": "allow",
                    "reason": "approved",
                },
            )["result"]
            == {}
        )

        assert [call["actor"] for call in submissions] == [PRINCIPAL, PRINCIPAL]
        assert [call["actor"] for call in controls] == [PRINCIPAL, PRINCIPAL]
        assert approvals[0]["actor"] == PRINCIPAL
        assert [method for _, method, _ in host.resolved].count("turn/steer") == 1
        assert [method for _, method, _ in host.resolved].count("turn/followUp") == 1
        assert [method for _, method, _ in host.resolved].count("turn/interrupt") == 1
        assert [method for _, method, _ in host.resolved].count("turn/action") == 1
        assert [method for _, method, _ in host.resolved].count("approval/resolve") == 1
    finally:
        _close(server, root_store)


def test_cross_tenant_ingress_cannot_write_to_session(tmp_path) -> None:
    tenant_b_principal = {
        "tenant_id": "tenant-b",
        "subject_id": "foreign-user",
        "subject_kind": "user",
        "attributes": {"role": "foreign"},
    }
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL, "tenant-b": tenant_b_principal},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport_a, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    transport_b = ChannelTransport(connection_id="tenant-b")
    server.router.register_transport(transport_b)
    try:
        _initialize(server, transport_a)
        _initialize(server, transport_b)
        thread_id = _request(
            server,
            transport_a,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]

        rejected = _request(
            server,
            transport_b,
            2,
            "thread/model/select",
            {
                "threadId": thread_id,
                "inputId": "foreign-select",
                "selection": {"model": "m", "route": {}},
            },
        )
        assert rejected["error"]["code"] == -32027
        assert server.kernel.store.read_input(thread_id, "foreign-select") is None
    finally:
        _close(server, root_store)


def test_thread_start_replay_requires_exact_trusted_binding_and_no_subscription(tmp_path) -> None:
    other_user = _user_principal("same-tenant-user-b")
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL, "user-b": other_user},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport_a, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    transport_b = ChannelTransport(connection_id="user-b")
    server.router.register_transport(transport_b)
    try:
        _initialize(server, transport_a)
        _initialize(server, transport_b)
        params = {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path), "metadata": {"k": "v"}}
        thread_id = _request(server, transport_a, 1, "thread/start", params)["result"]["threadId"]

        replay = _request(server, transport_a, 2, "thread/start", params)
        assert replay["result"]["replayed"] is True
        assert set(server.state_manager.subscribers(thread_id)) == {"tenant-a"}

        for request_id, connection, changed in (
            (3, transport_b, params),
            (4, transport_a, params | {"agentKey": "other"}),
            (5, transport_a, params | {"cwd": str(tmp_path / "other")}),
            (6, transport_a, params | {"metadata": {"k": "different"}}),
        ):
            rejected = _request(server, connection, request_id, "thread/start", changed)
            assert rejected["error"]["code"] == -32024
            assert set(server.state_manager.subscribers(thread_id)) == {"tenant-a"}

        assert ("user-b", "thread/start", params) in host.authorized
    finally:
        _close(server, root_store)


def test_cross_tenant_read_list_resume_and_subscribe_are_denied(tmp_path) -> None:
    tenant_b = _user_principal("foreign-user", tenant_id="tenant-b")
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL, "tenant-b": tenant_b},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport_a, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    transport_b = ChannelTransport(connection_id="tenant-b")
    server.router.register_transport(transport_b)
    try:
        _initialize(server, transport_a)
        _initialize(server, transport_b)
        thread_id = _request(
            server,
            transport_a,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]

        requests = (
            ("thread/read", {"threadId": thread_id}),
            ("thread/context/read", {"threadId": thread_id}),
            ("thread/events/list", {"threadId": thread_id}),
            ("thread/toolResult/read", {"threadId": thread_id}),
            ("thread/status", {"threadId": thread_id}),
            ("thread/resume", {"threadId": thread_id}),
            ("turn/resume", {"threadId": thread_id, "turnId": "turn"}),
            ("thread/unsubscribe", {"threadId": thread_id}),
            ("thread/list", {}),
            (
                "turn/start",
                {
                    "threadId": thread_id,
                    "inputId": "foreign-prompt",
                    "expectedTurnId": None,
                    "input": [{"type": "text", "text": "foreign"}],
                },
            ),
        )
        for request_id, (method, params) in enumerate(requests, 2):
            rejected = _request(server, transport_b, request_id, method, params)
            assert rejected["error"]["code"] == -32027, method

        assert "tenant-b" not in server.state_manager.subscribers(thread_id)
        assert server.kernel.store.read_input(thread_id, "foreign-prompt") is None
        assert {method for connection, method, _ in host.authorized if connection == "tenant-b"} == {
            method for method, _ in requests
        }
    finally:
        _close(server, root_store)


def test_fork_uses_current_principal_and_rejects_foreign_tenant_and_replay(tmp_path) -> None:
    user_b = _user_principal("fork-user-b")
    user_c = _user_principal("fork-user-c")
    tenant_b = _user_principal("foreign-fork", tenant_id="tenant-b")
    host = _TrustedHost(
        {
            "tenant-a": PRINCIPAL,
            "user-b": user_b,
            "user-c": user_c,
            "tenant-b": tenant_b,
        },
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport_a, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    transport_b = ChannelTransport(connection_id="user-b")
    transport_c = ChannelTransport(connection_id="user-c")
    foreign = ChannelTransport(connection_id="tenant-b")
    for transport in (transport_b, transport_c, foreign):
        server.router.register_transport(transport)
    try:
        for transport in (transport_a, transport_b, transport_c, foreign):
            _initialize(server, transport)
        source = _request(
            server,
            transport_a,
            1,
            "thread/start",
            {"inputId": "source", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]
        source_principal = server.kernel.store.read(source, limit=1).records[0].record.payload["principal"]
        host._principals["user-c"] = Principal.from_dict(source_principal | {"subject_id": user_c["subject_id"]})
        user_b["attributes"] = {"role": "fork-user-b", "labels": "business", "size": 17}
        host._principals["user-b"] = user_b
        params = {"threadId": source, "forkId": "copy"}

        denied = _request(server, foreign, 2, "thread/fork", params)
        assert denied["error"]["code"] == -32027

        forked = _request(server, transport_b, 3, "thread/fork", params)
        destination = forked["result"]["threadId"]
        assert user_b["subject_id"] not in json.dumps(forked, ensure_ascii=False)
        _assert_principal(_creation_principal(server, destination), user_b)
        assert _creation_principal(server, destination)["attributes"] == user_b["attributes"]
        destination_encoded = server.kernel.store.read(destination, limit=1).records[0].record.payload["principal"]
        assert destination_encoded["attributes"] != source_principal["attributes"]
        _assert_principal(_creation_principal(server, source), PRINCIPAL)

        assert _request(server, transport_b, 4, "thread/fork", params)["result"]["replayed"] is True
        replay = _request(server, transport_c, 5, "thread/fork", params)
        assert replay["error"]["code"] == -32024

        typed = _request(
            server,
            transport_c,
            6,
            "thread/fork",
            {"threadId": source, "forkId": "typed-copy"},
        )
        typed_destination = typed["result"]["threadId"]
        _assert_principal(_creation_principal(server, typed_destination), user_c)
        typed_encoded = server.kernel.store.read(typed_destination, limit=1).records[0].record.payload["principal"]
        assert typed_encoded["attributes"] != source_principal["attributes"]
    finally:
        _close(server, root_store)


def test_controller_replay_requires_original_actor(tmp_path) -> None:
    same_tenant = _user_principal("controller-user-b")
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL, "user-b": same_tenant},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport_a, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    transport_b = ChannelTransport(connection_id="user-b")
    server.router.register_transport(transport_b)
    try:
        _initialize(server, transport_a)
        _initialize(server, transport_b)
        thread_id = _request(
            server,
            transport_a,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]
        turn_id = _request(
            server,
            transport_a,
            2,
            "turn/start",
            {
                "threadId": thread_id,
                "inputId": "prompt",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "go"}],
            },
        )["result"]["turnId"]
        server.run_adapter.join()
        action = {"threadId": thread_id, "turnId": turn_id, "actionId": "cancel", "action": {"kind": "cancel"}}

        assert "result" in _request(server, transport_a, 3, "turn/action", action)
        rejected = _request(server, transport_b, 4, "turn/action", action)
        assert rejected["error"]["code"] == -32024
        action_id = derive_controller_command_id(thread_id, turn_id, "cancel")
        retained = server.kernel._retained_reply(
            thread_id,
            "controller/" + action_id,
        )
        assert retained is not None
        _assert_principal(retained[1].actor, PRINCIPAL)
    finally:
        _close(server, root_store)


def test_malformed_principal_maps_to_invalid_params(tmp_path) -> None:
    malformed = {"tenant_id": "tenant-a", "subject_id": "", "subject_kind": "user", "attributes": {}}
    host = _TrustedHost(
        {"tenant-a": malformed},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    try:
        _initialize(server, transport)
        rejected = _request(server, transport, 1, "thread/list", {})
        assert rejected["error"]["code"] == -32602
        assert rejected["error"]["message"] == "subject_id must be a non-empty string"
        rejected = _request(
            server,
            transport,
            2,
            "thread/start",
            {"inputId": "malformed-principal", "agentKey": "default"},
        )
        assert rejected["error"]["code"] == -32602
        assert "malformed-principal" not in server.kernel.store.list_sessions()
    finally:
        _close(server, root_store)


def test_legacy_host_without_resolver_keeps_local_service_principal(tmp_path) -> None:
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    agent = Agent("assistant", "Answer.", model="m")
    host = _LegacyHost(agent=agent, run_config=RunConfig(model_provider=provider, workspace=tmp_path))
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite")
    try:
        _initialize(server, transport)
        thread_id = _request(
            server,
            transport,
            1,
            "thread/start",
            {"inputId": "thread", "agentKey": "default", "cwd": str(tmp_path)},
        )["result"]["threadId"]
        principal = server.kernel.store.read(thread_id, limit=1).records[0].record.payload["principal"]
        assert (principal["tenant_id"], principal["subject_id"], principal["subject_kind"]) == (
            "tenant-a",
            "app-server",
            "service",
        )

        assert "result" in _request(
            server,
            transport,
            2,
            "thread/model/select",
            {
                "threadId": thread_id,
                "inputId": "model-select",
                "selection": {"model": "m", "route": {}},
            },
        )
        actor = _actor(server, thread_id, "model-select")
        assert (actor["tenant_id"], actor["subject_id"], actor["subject_kind"]) == (
            "tenant-a",
            "app-server",
            "service",
        )
    finally:
        _close(server, root_store)


@pytest.mark.parametrize("forged_field", ["actor", "principal"])
def test_client_cannot_inject_identity_fields(forged_field: str, tmp_path) -> None:
    forged = {
        "tenant_id": "tenant-evil",
        "subject_id": "untrusted-client",
        "subject_kind": "user",
        "attributes": {"role": "admin"},
    }
    host = _TrustedHost(
        {"tenant-a": PRINCIPAL},
        agent=Agent("assistant", "Answer.", model="m"),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("done")]),
            workspace=tmp_path,
        ),
    )
    server, transport, root_store = _app_server(host=host, database=tmp_path / "trusted.sqlite", principal=PRINCIPAL)
    try:
        _initialize(server, transport)
        rejected = _request(
            server,
            transport,
            1,
            "thread/start",
            {"inputId": "forged-thread", "agentKey": "default", forged_field: forged},
        )
        assert rejected["error"]["code"] == -32602
        assert "forged-thread" not in server.kernel.store.list_sessions()
        assert host.resolved[-1][2][forged_field] == forged
    finally:
        _close(server, root_store)
