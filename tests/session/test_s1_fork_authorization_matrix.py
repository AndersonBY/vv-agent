"""A8 fork isolation and the current P2 A14 authorization boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.host import ToolResultResourceAuthorizationRequest, TrustedPrincipal
from vv_agent.app_server.protocol.tool_result import ToolResultRef
from vv_agent.kernel._content import ContentLabels, ContentRef, ContentUnavailable, Principal
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_store
from vv_agent.session.content_store import InMemoryEncryptedContentStore, LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.types import LLMResponse
from vv_agent.workspace.resources import ToolResultResourceRef


def _principal(subject_id: str, tenant_id: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "subject_kind": "user",
        "attributes": {"role": subject_id},
    }


class _Client:
    def __init__(
        self,
        server: AppServer,
        connection_id: str,
        *,
        transport: ChannelTransport | None = None,
    ) -> None:
        self.server = server
        self.transport = transport or ChannelTransport(connection_id=connection_id, outbound_capacity=10_000)
        if transport is None:
            server.router.register_transport(self.transport)
        self.next_id = 0

    def rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        identity = self.next_id
        self.next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": identity, "method": method}
        if params is not None:
            payload["params"] = params
        self.transport.send_inbound(payload)
        self.server.processor.process_next(self.transport)
        while True:
            response = self.transport.receive_outbound(timeout=10)
            if response.get("id") == identity:
                return response

    def initialize(self) -> None:
        assert "result" in self.rpc("initialize", {"clientInfo": {"name": self.transport.connection_id}})
        self.transport.send_inbound({"jsonrpc": "2.0", "method": "initialized"})
        self.server.processor.process_next(self.transport)


def _server(
    store: Any,
    host: DefaultAppServerHost,
    state_path: Path,
) -> tuple[AppServer, LocalKeyProvider, _Client]:
    provider = LocalKeyProvider(b"f" * 32, state_path=state_path)
    transport = ChannelTransport(connection_id="user-a", outbound_capacity=10_000)
    server = AppServer(store=store, content_key_provider=provider, host=host, transport=transport)
    client = _Client(server, "user-a", transport=transport)
    client.initialize()
    return server, provider, client


def _creation(server: AppServer, session_id: str) -> dict[str, Any]:
    return server.kernel.store.read(session_id, limit=1).records[0].record.payload


def _hydrated_session_attributes(server: AppServer, session_id: str) -> dict[str, Any]:
    stored = server.kernel.store.read(session_id, limit=1).records[0]
    hydrated = hydrate_record_payload(
        stored,
        hydration_for_session_store(server.kernel.store, session_id),
    )
    return hydrated["attributes"]


def _read_json(store: Any, session_id: str, ref: ContentRef) -> dict[str, Any]:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
        return writer.get_json(ref)


def _link_key_ids(store: Any, session_id: str) -> set[str]:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
        links = writer.durable.read_links(store.tenant_id, session_id, record_seq=1)
    return {link.ref.key_id for link in links}


def _link_count(store: Any, session_id: str) -> int:
    return store._one(
        "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )[0]


def _session_ids(store: Any) -> set[str]:
    return {row[0] for row in store._rows("SELECT session_id FROM sk_session WHERE tenant_id=%s", (store.tenant_id,))}


@pytest.mark.parametrize("operation", ["forget", "purge"])
@pytest.mark.parametrize("forgotten", ["source", "destination"])
def test_fork_rebinds_content_and_each_side_has_an_independent_forget(store, tmp_path, operation, forgotten):
    host = DefaultAppServerHost(
        agent=Agent("test", "Be precise."),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("answer")]),
            workspace=tmp_path,
        ),
    )
    server, provider, client = _server(store, host, tmp_path / "fork-keys.sqlite3")
    try:
        assert client.rpc("thread/start", {"inputId": "source"})["result"]["threadId"] == "source"
        turn = client.rpc(
            "turn/start",
            {
                "threadId": "source",
                "inputId": "message",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "same private message"}],
            },
        )["result"]["turnId"]
        server.run_adapter.join()
        forked = client.rpc(
            "thread/fork",
            {"threadId": "source", "forkId": "copy", "throughTurnId": turn},
        )["result"]
        destination = forked["threadId"]

        source_creation = _creation(server, "source")
        destination_creation = _creation(server, destination)
        source_principal = Principal.from_dict(source_creation["principal"])
        destination_principal = Principal.from_dict(destination_creation["principal"])
        source_workspace = ContentRef.from_dict(source_creation["workspace"])
        destination_workspace = ContentRef.from_dict(destination_creation["workspace"])
        source_attributes_ref = ContentRef.from_dict(source_creation["attributes"])
        destination_attributes_ref = ContentRef.from_dict(destination_creation["attributes"])

        assert destination_principal.attributes.key_id != source_principal.attributes.key_id
        assert destination_workspace.key_id != source_workspace.key_id
        assert destination_principal.attributes.content_id != source_principal.attributes.content_id
        assert destination_principal.attributes.mac != source_principal.attributes.mac
        assert destination_workspace.content_id != source_workspace.content_id
        assert destination_workspace.mac != source_workspace.mac
        assert destination_attributes_ref.key_id != source_attributes_ref.key_id
        assert destination_attributes_ref.content_id != source_attributes_ref.content_id
        assert destination_attributes_ref.mac != source_attributes_ref.mac

        destination_attributes = _hydrated_session_attributes(server, destination)
        copied_messages = destination_attributes["seed"]["messages"]
        assert any(message["role"] == "user" and message["content"] == "same private message" for message in copied_messages)
        assert _read_json(store, "source", source_workspace) == _read_json(store, destination, destination_workspace)
        assert _link_key_ids(store, "source") == {source_principal.attributes.key_id}
        assert _link_key_ids(store, destination) == {destination_principal.attributes.key_id}

        deletion = SessionDeletionStore(store, durable_content_store(store), allow_unsigned=True)
        forgotten_session = "source" if forgotten == "source" else destination
        survivor_session = destination if forgotten == "source" else "source"
        receipt = getattr(deletion, f"{operation}_session")(
            store.tenant_id,
            forgotten_session,
            actor=source_principal,
        )
        survivor = "destination" if forgotten == "source" else "source"
        forgotten_ref = source_workspace if forgotten == "source" else destination_workspace
        survivor_ref = destination_workspace if survivor == "destination" else source_workspace

        assert receipt.session_ids == (forgotten_session,)
        assert _read_json(store, survivor_session, survivor_ref)
        with pytest.raises(ContentUnavailable):
            _read_json(store, forgotten_session, forgotten_ref)
        # Destroying the other side does not make the already-forgotten side readable again.
        assert survivor_session in _session_ids(store)
        assert _link_count(store, forgotten_session) == 0
    finally:
        server.kernel.close()
        provider.close()


def test_same_plaintext_has_independent_tenant_ids_macs_and_storage(tmp_path):
    provider = LocalKeyProvider(b"t" * 32, state_path=tmp_path / "tenant-keys.sqlite3")
    try:
        store = InMemoryEncryptedContentStore(provider)
        labels = ContentLabels()
        tenant_a = store.put("tenant-a", "same-session", b"same private plaintext", labels)
        repeated = store.put("tenant-a", "same-session", b"same private plaintext", labels)
        tenant_b = store.put("tenant-b", "same-session", b"same private plaintext", labels)

        assert repeated == tenant_a
        assert tenant_b.content_id != tenant_a.content_id
        assert tenant_b.key_id != tenant_a.key_id
        assert tenant_b.mac != tenant_a.mac
        assert store.get("tenant-a", "same-session", tenant_a) == b"same private plaintext"
        with pytest.raises(ContentUnavailable):
            store.get("tenant-b", "same-session", tenant_a)
    finally:
        provider.close()


def test_duplicate_create_message_and_fork_replay_without_copying_and_conflicts(store, tmp_path):
    host = DefaultAppServerHost(
        agent=Agent("test", "Be precise."),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("answer")]),
            workspace=tmp_path,
        ),
    )
    server, provider, client = _server(store, host, tmp_path / "idempotency-keys.sqlite3")
    try:
        create = {"inputId": "source", "metadata": {"host": "stable"}}
        assert client.rpc("thread/start", create)["result"]["threadId"] == "source"
        assert client.rpc("thread/start", create)["result"]["replayed"] is True
        assert client.rpc("thread/start", create | {"metadata": {"host": "changed"}})["error"]["code"] == -32024
        assert _session_ids(store) == {"source"}

        message = {
            "threadId": "source",
            "inputId": "message",
            "expectedTurnId": None,
            "input": [{"type": "text", "text": "same message"}],
        }
        first_turn = client.rpc("turn/start", message)["result"]
        server.run_adapter.join()
        stored_input = store.read_input("source", "message")
        replay = client.rpc("turn/start", message)
        assert replay["result"]["turnId"] == first_turn["turnId"]
        assert replay["result"]["replayed"] is True
        assert store.read_input("source", "message") == stored_input
        assert store._one(
            "SELECT COUNT(*) FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            (store.tenant_id, "source", "message"),
        ) == (1,)
        changed = message | {"input": [{"type": "text", "text": "different message"}]}
        assert client.rpc("turn/start", changed)["error"]["code"] == -32024

        fork = {"threadId": "source", "forkId": "copy", "workspace": {"root": "prepared"}}
        created_fork = client.rpc("thread/fork", fork)["result"]
        destination = created_fork["threadId"]
        replay_fork = client.rpc("thread/fork", fork)["result"]
        assert replay_fork == {
            "threadId": destination,
            "accepted": True,
            "replayed": True,
            "lineage": created_fork["lineage"],
        }
        conflict = fork | {"workspace": {"root": "different"}}
        assert client.rpc("thread/fork", conflict)["error"]["code"] == -32024
        assert _session_ids(store) == {"source", destination}
        assert len(store.read_state(destination)[1]) == 1
        assert not store.peek_inbox(destination)
    finally:
        server.run_adapter.join()
        server.kernel.close()
        provider.close()


class _GrantHost(DefaultAppServerHost):
    """Test host implementing the documented account/resource authorization hook."""

    _ACTIONS: ClassVar[dict[str, str]] = {
        "thread/start": "create",
        "thread/fork": "control",
        "thread/model/select": "control",
        "thread/compact/start": "control",
        "turn/start": "control",
        "turn/steer": "control",
        "turn/followUp": "control",
        "turn/action": "control",
        "turn/interrupt": "control",
        "thread/read": "read",
        "thread/context/read": "read",
        "thread/events/list": "read",
        "thread/status": "read",
        "thread/resume": "reconnect",
        "thread/toolResult/read": "artifact",
    }

    def __init__(
        self,
        *,
        principals: dict[str, TrustedPrincipal],
        grants: set[tuple[str, str, str]],
        resource_grants: set[tuple[str, str]],
        agent: Agent,
        run_config: RunConfig,
    ) -> None:
        super().__init__(agent=agent, run_config=run_config)
        self.principals = principals
        self.grants = grants
        self.resource_grants = resource_grants
        self.authorizations: list[tuple[str, str, str | None]] = []
        self.resource_authorizations: list[ToolResultResourceAuthorizationRequest] = []

    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        target = params.get("threadId") or params.get("inputId")
        action = self._ACTIONS.get(method, method)
        self.authorizations.append((connection_id, method, target))
        return (connection_id, action, target) in self.grants

    def resolve_principal(self, connection_id: str, method: str, params: dict[str, Any]) -> TrustedPrincipal | None:
        del method, params
        return self.principals.get(connection_id)

    def authorize_tool_result_resource(self, request: ToolResultResourceAuthorizationRequest) -> bool:
        self.resource_authorizations.append(request)
        return (request.connection_id, request.thread_id) in self.resource_grants


def test_same_tenant_users_reuse_only_explicit_shared_resource_grants(store, tmp_path, monkeypatch):
    tenant = store.tenant_id
    user_a = _principal("user-a", tenant)
    user_b = _principal("user-b", tenant)
    user_c = _principal("user-c", tenant)
    foreign = _principal("foreign-user", f"{tenant}-other")
    host = _GrantHost(
        principals={"user-a": user_a, "user-b": user_b, "user-c": user_c, "foreign": foreign},
        grants={("user-a", "create", "source")},
        resource_grants=set(),
        agent=Agent("test", "Be precise."),
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", []), workspace=tmp_path),
    )
    server, provider, user_a_client = _server(store, host, tmp_path / "authorization-keys.sqlite3")
    try:
        user_b_client = _Client(server, "user-b")
        user_b_client.initialize()
        user_c_client = _Client(server, "user-c")
        user_c_client.initialize()
        foreign_client = _Client(server, "foreign")
        foreign_client.initialize()

        assert user_a_client.rpc("thread/start", {"inputId": "source"})["result"]["threadId"] == "source"
        storage_reads: list[str] = []
        read_thread = server.store.read_thread

        def tracked_read_thread(thread_id: str, *args: Any, **kwargs: Any):
            storage_reads.append(thread_id)
            return read_thread(thread_id, *args, **kwargs)

        monkeypatch.setattr(server.store, "read_thread", tracked_read_thread)
        assert user_b_client.rpc("thread/read", {"threadId": "source"})["error"]["code"] == -32027
        assert storage_reads == []

        host.grants.update(
            {
                ("user-b", "read", "source"),
                ("user-b", "reconnect", "source"),
                ("user-b", "artifact", "source"),
            }
        )
        read = user_b_client.rpc("thread/read", {"threadId": "source"})
        resumed = user_b_client.rpc("thread/resume", {"threadId": "source", "subscribe": False})
        artifact_route = user_b_client.rpc("thread/toolResult/read", {"threadId": "source"})
        assert read["result"]["thread"]["threadId"] == "source"
        assert resumed["result"]["thread"]["threadId"] == "source"
        assert artifact_route["error"]["code"] == -32602
        assert storage_reads

        control = user_b_client.rpc(
            "turn/start",
            {
                "threadId": "source",
                "inputId": "forbidden-control",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "change it"}],
            },
        )
        fork = user_b_client.rpc("thread/fork", {"threadId": "source", "forkId": "forbidden-copy"})
        assert control["error"]["code"] == -32027
        assert fork["error"]["code"] == -32027
        assert store.read_input("source", "forbidden-control") is None
        assert _session_ids(store) == {"source"}

        assert user_c_client.rpc("thread/read", {"threadId": "source"})["error"]["code"] == -32027
        host.grants.add(("foreign", "read", "source"))
        reads_before_foreign = len(storage_reads)
        assert foreign_client.rpc("thread/read", {"threadId": "source"})["error"]["code"] == -32027
        assert len(storage_reads) == reads_before_foreign

        resource = ToolResultResourceRef(
            "a" * 64,
            "b" * 64,
            ".vv-agent/resources/" + "b" * 64 + "/object",
            1,
            "c" * 64,
            "text/plain",
            "result.txt",
        )
        request = ToolResultResourceAuthorizationRequest(
            connection_id="user-b",
            thread_id="source",
            ref=ToolResultRef("source", "turn", "operation", 1, "call", 2, "d" * 64),
            workspace_backend=object(),
            resource=resource,
        )
        assert host.authorize_tool_result_resource(request) is False
        host.resource_grants.add(("user-b", "source"))
        assert host.authorize_tool_result_resource(request) is True
        assert ("user-b", "thread/toolResult/read", "source") in host.authorizations
        assert ("user-b", "turn/start", "source") in host.authorizations
        assert ("user-b", "thread/fork", "source") in host.authorizations
    finally:
        server.run_adapter.join()
        server.kernel.close()
        provider.close()


@pytest.mark.skip(
    reason=(
        "P3 boundary: P2 App Server exposes no thread export/delete surface, and the store export/delete APIs "
        "do not receive an authenticated user Principal for resource authorization."
    )
)
def test_a14_export_and_delete_user_authorization_matrix() -> None:
    raise AssertionError("P3 must add an authenticated App Server export/delete contract before A14 can pass")
