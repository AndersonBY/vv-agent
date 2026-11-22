"""Signed App Server root publication against disposable PostgreSQL and local RustFS."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from session.test_s1_checkpoint_signer_acceptance import Signer
from test_app_server_distributed import request
from test_postgres_object_publication_fence import object_publication as object_publication
from test_s1_postgres_rls import TENANT_A, TENANT_B
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.app_server.host import TrustedPrincipal
from vv_agent.kernel._content import Classification, ContentLabels, ContentRef, SourceKind
from vv_agent.kernel._content_runtime import SessionContentWriter, _provision_postgres_object_content, bind_content_runtime
from vv_agent.kernel._external_content import _PostgresObjectContent
from vv_agent.kernel._postgres_allocation_journal import PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.session import fork, s3_content
from vv_agent.session._s3_transport import _S3ProcessTransport, _S3Request
from vv_agent.session.content_runtime import SESSION_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES, SESSION_WORKSPACE
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import SessionSpec, digest
from vv_agent.session.s3_content import S3CiphertextStore
from vv_agent.types import LLMResponse, Message

_RUSTFS_ENDPOINT = os.environ.get("VV_P2_RUSTFS_ENDPOINT")
pytestmark = pytest.mark.skipif(not _RUSTFS_ENDPOINT, reason="requires disposable loopback RustFS")
_PRINCIPAL = {"tenant_id": TENANT_A, "subject_kind": "user", "subject_id": "fork-user", "attributes": {"team": "fixture"}}
_SOURCE = "source"
_PROMPT = "private source context"
_LABELS = ContentLabels((SourceKind.USER,), Classification.RESTRICTED)
_TABLES = (
    "sk_session",
    "sk_record",
    "sk_commit",
    "sk_checkpoint",
    "sk_session_key",
    "sk_content",
    "sk_content_link",
    "sk_consumer",
)


class _Host(DefaultAppServerHost):
    def resolve_principal(self, connection_id: str, method: str, params: dict[str, Any]) -> TrustedPrincipal:
        return _PRINCIPAL


def _client(endpoint: str, *, stall_after_ciphertext: bool = False) -> Any:
    boto3 = pytest.importorskip("boto3")
    config = pytest.importorskip("botocore.config")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUSTFS_SECRET_KEY"],
        region_name="us-east-1",
        config=config.Config(
            retries={"total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=2,
            proxies={},
            s3={"addressing_style": "path"},
        ),
    )
    if stall_after_ciphertext:
        puts = 0

        def stall(http_response, **_):
            nonlocal puts
            puts += 1
            if puts == 2 and http_response.status_code == 200:
                time.sleep(4)

        client.meta.events.register("after-call.s3.PutObject", stall)
    return client


@pytest.fixture
def rustfs():
    endpoint = _RUSTFS_ENDPOINT
    assert endpoint is not None and urlsplit(endpoint).hostname in {"127.0.0.1", "::1"}
    bucket = f"p2-fork-{uuid4().hex}"
    with closing(_client(endpoint)) as inspector:
        inspector.create_bucket(Bucket=bucket)
        try:
            yield endpoint, bucket, inspector
        finally:
            for entry in inspector.list_objects_v2(Bucket=bucket).get("Contents", ()):
                inspector.delete_object(Bucket=bucket, Key=entry["Key"])
            inspector.delete_bucket(Bucket=bucket)


def _keys(inspector: Any, bucket: str) -> set[str]:
    return {entry["Key"] for entry in inspector.list_objects_v2(Bucket=bucket).get("Contents", ())}


def _read_object(inspector: Any, bucket: str, key: str) -> bytes:
    with closing(inspector.get_object(Bucket=bucket, Key=key)["Body"]) as stream:
        return stream.read()


def _reopen(journal: PostgresObjectJournal) -> PostgresObjectJournal:
    def open_role(role: PostgresAllocationJournal) -> PostgresAllocationJournal:
        return PostgresAllocationJournal(role._conninfo, journal.identity, role._role)

    return PostgresObjectJournal(
        anonymous=open_role(journal._anonymous),
        binder=open_role(journal._binder),
        scope=open_role(journal._scope),
        keys=open_role(journal._keys),
    )


@contextmanager
def _server(
    dsn: str,
    endpoint: str,
    bucket: str,
    journal: PostgresObjectJournal,
    signer: Signer,
    key_path: Path,
    master_key: bytes,
    *,
    owner: PostgresStore | None = None,
    stall_after_ciphertext: bool = False,
) -> Iterator[tuple[AppServer, PostgresStore, S3CiphertextStore]]:
    with (
        closing(LocalKeyProvider(master_key, state_path=key_path)) as provider,
        _S3ProcessTransport(
            partial(_client, endpoint, stall_after_ciphertext=stall_after_ciphertext), endpoint_url=endpoint
        ) as io,
        PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store,
    ):
        instance = provider._provision_exact_key_lifecycle() if owner is not None else provider._exact_key_provider_instance_id()
        adapter = S3CiphertextStore(io, bucket=bucket, journal=journal)
        objects = _PostgresObjectContent(journal, adapter, instance)
        if owner is not None:
            _provision_postgres_object_content(owner, key_provider=provider, objects=objects)
        store.configure_input_admission(signer)
        bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        host = _Host(
            agent=Agent("test", "Answer."),
            run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("answer")])),
        )
        server = AppServer(store=store, host=host, transport=ChannelTransport(connection_id=uuid4().hex, outbound_capacity=100))
        with closing(server.kernel):
            assert "result" in request(server, "init", "initialize", {"clientInfo": {"name": "object-fork"}})
            yield server, store, adapter


def _seed_source(server: AppServer, workspace: str) -> None:
    host = server.host
    assert isinstance(host, _Host) and host._agent is not None and host._run_config is not None
    runtime = server.kernel.runtime(host._agent, host._run_config)
    context = runtime._definition(runtime.compile("", _SOURCE + "/context", control=True))
    with server.kernel._root_publication_atomic() as tx:
        destination = SessionContentWriter.bind(tx, tenant_id=TENANT_A, session_id=_SOURCE)

        def spec(writer: SessionContentWriter) -> SessionSpec:
            message = writer.put_json(Message("user", _PROMPT).to_dict(), labels=_LABELS)
            principal = _PRINCIPAL | {"attributes": writer.put_json(_PRINCIPAL["attributes"]).to_dict()}
            workspace_ref = writer.put_json(workspace, labels=_LABELS)
            return SessionSpec(
                _SOURCE,
                principal,
                workspace_ref.to_dict(),
                attributes={
                    "app_server": {"agent_key": "default", "cwd": workspace, "metadata": {}},
                    "app_server_context": context,
                    "seed": {"messages": [message.to_dict()], "shared_state": {}},
                },
            )

        destination.create_session(spec, consumers=("events", "app_server", "traces"))


def _workspace(server: AppServer, sid: str) -> ContentRef:
    created = server.kernel.store.read(sid, limit=1).records[0]
    return ContentRef.from_dict(created.record.payload["workspace"])


def _locator(store: PostgresStore, sid: str, ref: ContentRef) -> bytes:
    ciphertext, locator = store._one(
        "SELECT ciphertext,object_locator FROM sk_content WHERE session_id=%s AND content_id=%s", (sid, ref.content_id)
    )
    assert ciphertext is None and isinstance(locator, bytes) and len(locator) == 33 and locator[0] == 1
    return locator


def test_signed_postgres_object_root_fork_and_timeout(object_publication, rustfs, tmp_path, monkeypatch):
    owner, primary, journals, _ = object_publication
    endpoint, bucket, inspector = rustfs
    journal = journals[TENANT_A]
    signer = Signer()
    key_path, master_key = tmp_path / "keys.sqlite", os.urandom(32)
    open_server = partial(_server, primary[TENANT_A], endpoint, bucket, signer=signer, key_path=key_path, master_key=master_key)
    workspace = str(tmp_path)
    params = {"threadId": _SOURCE, "forkId": "stable"}
    destination = "fork/" + digest([_SOURCE, "stable"])
    timed_out = "fork/" + digest([_SOURCE, "timeout"])

    with open_server(journal, owner=owner) as (server, store, adapter), open_server(_reopen(journal)) as (peer, _, _):
        _seed_source(server, workspace)
        source = store.read(_SOURCE).records[0].record
        params = params | {"workspace": {"mode": "host-prepared"}}
        execute = adapter._transport.execute
        reads, fenced_writes = [], []

        def observe(request: _S3Request, **kwargs):
            if request.operation == "get_object":
                assert not store._transaction_in_progress(), "fork attempted S3 GET inside the publication fence"
                reads.append(request.key)
            elif request.operation == "put_object":
                assert store._transaction_in_progress(), "fork published an object outside the primary fence"
                fenced_writes.append(request.key)
            return execute(request, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(adapter._transport, "execute", observe)
            with peer.kernel._root_publication_atomic():
                busy = request(server, "busy", "thread/fork", params)
            assert busy["error"]["code"] == -32001, busy
            assert fenced_writes == []
            assert journal.key_intents_for_admission(TENANT_A, destination, destination, "session/create") == ()
            response = request(server, "fork", "thread/fork", params)
        assert response.get("result", {}).get("threadId") == destination, response
        assert reads and fenced_writes
        result = response["result"]
        assert result["accepted"] is True and result["replayed"] is False
        assert result["lineage"]["sourceRecordDigest"] == source.digest
        assert store.verify_ledger(_SOURCE)[0] == store.verify_ledger(destination)[0] == 1
        assert any(message["content"] == _PROMPT for message in server.kernel.session_attributes(destination)["seed"]["messages"])
        source_workspace, destination_workspace = (_workspace(server, sid) for sid in (_SOURCE, destination))
        assert source_workspace.key_id != destination_workspace.key_id
        assert source_workspace.content_id != destination_workspace.content_id
        assert source_workspace.labels == destination_workspace.labels == _LABELS
        created = store.read(destination, limit=1).records[0]
        ref = ContentRef.from_dict(created.record._payload["attributes"])
        attributes = server.kernel.read_record_content(destination, 1, ref, slot=SESSION_ATTRIBUTES)
        messages = [ContentRef.from_dict(value) for value in attributes["seed"]["messages"]]
        assert messages and all(SourceKind.USER in value.labels.source for value in messages)
        assert all(value.labels.classification is Classification.RESTRICTED for value in messages)
        source_locator = _locator(store, _SOURCE, source_workspace)
        destination_locator = _locator(store, destination, destination_workspace)
        assert source_locator != destination_locator
        assert _read_object(inspector, bucket, adapter._key(source_locator[1:])) != _read_object(
            inspector, bucket, adapter._key(destination_locator[1:])
        )
        published = _keys(inspector, bucket)
        with peer.kernel._root_publication_atomic():
            assert request(server, "replay", "thread/fork", params)["result"] == result | {"replayed": True}
        conflict = request(server, "conflict", "thread/fork", params | {"workspace": {"mode": "other"}})
        assert conflict["error"]["code"] == -32024
        assert _keys(inspector, bucket) == published

        verify_links = fork._ForkContentReader.verify_links

        def stale_link(reader, writer):
            owner_slot = ("record:1", SESSION_WORKSPACE)
            captured = reader.links[owner_slot]
            reader.links[owner_slot] = replace(captured, mac="0" * 64)
            return verify_links(reader, writer)

        mismatch = "fork/" + digest([_SOURCE, "changed-link"])
        with monkeypatch.context() as patch:
            patch.setattr(fork._ForkContentReader, "verify_links", stale_link)
            rejected = request(server, "changed-link", "thread/fork", params | {"forkId": "changed-link"})
        assert "content metadata does not match its reference" in rejected["error"]["message"]
        assert store._one("SELECT count(*) FROM sk_session WHERE session_id=%s", (mismatch,)) == (0,)
        assert journal.key_intents_for_admission(TENANT_A, mismatch, mismatch, "session/create") == ()
        assert _keys(inspector, bucket) == published

    with open_server(journal, stall_after_ciphertext=True) as (server, store, _), monkeypatch.context() as patch:
        patch.setattr(s3_content, "_OPERATION_NS", 1_200_000_000)
        started = time.monotonic()
        error = request(server, "timeout", "thread/fork", params | {"forkId": "timeout"})["error"]
        assert "deadline" in error["message"] and time.monotonic() - started < 3.5, error
        for table in _TABLES:
            assert store._one(f"SELECT count(*) FROM {table} WHERE session_id=%s", (timed_out,)) == (0,)
        [orphan_key] = _keys(inspector, bucket) - published
        orphan_id = bytes.fromhex(orphan_key.rsplit("/", 1)[1])
        orphan = journal._scope.get(orphan_id)
        assert orphan is not None and orphan.state == "bound" and orphan.owner is not None
        assert orphan.owner.session_id == timed_out and orphan.owner.admission_session_id == timed_out
        assert inspector.head_object(Bucket=bucket, Key=orphan_key)["ContentLength"] > 8

    reopened = _reopen(journal)
    assert reopened.verify_identity() == journal.identity and reopened._scope.get(orphan_id) == orphan
    with open_server(reopened) as (server, store, _):
        assert request(server, "cold-replay", "thread/fork", params)["result"] == result | {"replayed": True}
        for sid, ref, locator in (
            (_SOURCE, source_workspace, source_locator),
            (destination, destination_workspace, destination_locator),
        ):
            assert server.kernel.read_record_content(sid, 1, ref, slot=SESSION_WORKSPACE) == workspace
            allocation = reopened._scope.get(locator[1:])
            assert allocation is not None and allocation.state == "bound" and allocation.owner is not None
            assert (allocation.owner.session_id, allocation.owner.content_id, allocation.owner.key_id) == (
                sid,
                ref.content_id,
                ref.key_id,
            )
        assert store.verify_ledger(_SOURCE)[0] == store.verify_ledger(destination)[0] == 1
        assert _keys(inspector, bucket) == published | {orphan_key}
    with PostgresStore.tenant_standalone(primary[TENANT_B], tenant_id=TENANT_B) as foreign:
        assert foreign._rows("SELECT session_id FROM sk_session WHERE session_id IN (%s,%s)", (_SOURCE, destination)) == []


def test_signed_postgres_object_thread_start_replay_and_timeout(object_publication, rustfs, tmp_path, monkeypatch):
    owner, primary, journals, _ = object_publication
    endpoint, bucket, inspector = rustfs
    journal = journals[TENANT_A]
    signer = Signer()
    key_path, master_key = tmp_path / "keys.sqlite", os.urandom(32)
    open_server = partial(_server, primary[TENANT_A], endpoint, bucket, signer=signer, key_path=key_path, master_key=master_key)
    params = {"inputId": "app-root", "cwd": str(tmp_path), "metadata": {"project": "private"}}
    sid = params["inputId"]
    timed_out = "app-timeout"

    owner._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    with open_server(journal, owner=owner) as (server, store, adapter), open_server(_reopen(journal)) as (peer, _, _):
        execute = adapter._transport.execute
        writes = []

        def observe(operation: _S3Request, **kwargs):
            if operation.operation == "get_object":
                assert not store._transaction_in_progress(), "thread/start issued S3 GET inside the publication fence"
            elif operation.operation == "put_object":
                assert store._transaction_in_progress(), "thread/start issued S3 PUT outside the publication fence"
                writes.append(operation.key)
            return execute(operation, **kwargs)

        monkeypatch.setattr(adapter._transport, "execute", observe)
        with peer.kernel._root_publication_atomic():
            busy = request(server, "busy", "thread/start", params)
        assert busy["error"]["code"] == -32001
        assert writes == []
        assert journal.key_intents_for_admission(TENANT_A, sid, sid, "session/create") == ()

        first = request(server, "create", "thread/start", params)
        assert first.get("result") == {
            "threadId": sid,
            "agentKey": "default",
            "cwd": params["cwd"],
            "status": "idle",
            "accepted": True,
            "replayed": False,
        }, first
        assert writes
        created = store.read(sid, limit=1).records[0].record
        assert created.feature_level == 2
        assert created._payload["bootstrap"]["level"] == 2
        assert store.verify_ledger(sid)[0] == 1
        ref = ContentRef.from_dict(created.payload["principal"]["attributes"])
        assert server.kernel.read_record_content(sid, 1, ref, slot=SESSION_PRINCIPAL_ATTRIBUTES) == _PRINCIPAL["attributes"]
        assert _locator(store, sid, ref)
        attributes = server.kernel.session_attributes(sid)
        assert attributes["app_server"]["metadata"] == params["metadata"]
        assert attributes["app_server_context"]["task"]
        published = _keys(inspector, bucket)
        key_intents = journal.key_intents_for_admission(TENANT_A, sid, sid, "session/create")
        assert len(key_intents) == 1

        assert request(peer, "replay", "thread/start", params)["result"] == first["result"] | {"replayed": True}
        assert request(peer, "conflict", "thread/start", params | {"metadata": {}})["error"]["code"] == -32024
        with monkeypatch.context() as patch:
            patch.setattr(peer.host, "resolve_principal", lambda *_: _PRINCIPAL | {"subject_id": "other-user"})
            assert request(peer, "principal", "thread/start", params)["error"]["code"] == -32024
        assert _keys(inspector, bucket) == published
        assert journal.key_intents_for_admission(TENANT_A, sid, sid, "session/create") == key_intents

        raced_sid = "raced"
        raced_params = params | {"inputId": raced_sid}
        publication = server.kernel._root_publication_atomic
        writes.clear()

        def publish_after_peer():
            competing = request(peer, "wins-race", "thread/start", raced_params)
            assert competing.get("result", {}).get("replayed") is False, competing
            return publication()

        with monkeypatch.context() as patch:
            patch.setattr(server.kernel, "_root_publication_atomic", publish_after_peer)
            raced = request(server, "raced", "thread/start", raced_params)
        assert raced.get("result", {}).get("replayed") is True, raced
        assert writes == []
        assert store.verify_ledger(raced_sid)[0] == 1
        assert len(journal.key_intents_for_admission(TENANT_A, raced_sid, raced_sid, "session/create")) == 1
        published = _keys(inspector, bucket)

    with open_server(_reopen(journal), stall_after_ciphertext=True) as (server, store, _), monkeypatch.context() as patch:
        patch.setattr(s3_content, "_OPERATION_NS", 1_200_000_000)
        started = time.monotonic()
        failed = request(server, "timeout", "thread/start", params | {"inputId": timed_out})
        assert "deadline" in failed["error"]["message"] and time.monotonic() - started < 3.5, failed
        for table in _TABLES:
            assert store._one(f"SELECT count(*) FROM {table} WHERE session_id=%s", (timed_out,)) == (0,)
        [orphan_key] = _keys(inspector, bucket) - published
        orphan = journal._scope.get(bytes.fromhex(orphan_key.rsplit("/", 1)[1]))
        assert orphan is not None and orphan.state == "bound" and orphan.owner is not None
        assert (orphan.owner.session_id, orphan.owner.admission_session_id) == (timed_out, timed_out)
