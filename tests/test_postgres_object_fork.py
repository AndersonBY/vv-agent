"""Signed App Server root publication against disposable PostgreSQL and local RustFS."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, current_thread, local
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from session.test_s1_checkpoint_signer_acceptance import Signer
from test_app_server_distributed import eventually, request, submission
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
from vv_agent.kernel._store import Lease, SessionStore
from vv_agent.llm.base import LlmRequest
from vv_agent.memory.token_utils import count_tokens
from vv_agent.session import fork, s3_content
from vv_agent.session._s3_transport import _S3ProcessTransport, _S3Request, _S3TransportError
from vv_agent.session.content_runtime import SESSION_ATTRIBUTES, SESSION_PRINCIPAL_ATTRIBUTES, SESSION_WORKSPACE
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import SessionSpec, digest
from vv_agent.session.result import project_result
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
    open_claim_store: Callable[[], AbstractContextManager[SessionStore]] | None = None,
    host: DefaultAppServerHost | None = None,
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
        server = AppServer(
            store=store,
            host=host
            or _Host(
                agent=Agent("test", "Answer."),
                run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("answer")])),
            ),
            open_claim_store=open_claim_store,
            transport=ChannelTransport(connection_id=uuid4().hex, outbound_capacity=100),
        )
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


def test_signed_postgres_object_turn_start_and_stale_owner(object_publication, rustfs, tmp_path, monkeypatch):
    owner, primary, journals, worker_dsn = object_publication
    endpoint, bucket, inspector = rustfs
    journal = journals[TENANT_A]
    signer = Signer()
    sid = "turn-root"
    key_path, master_key = tmp_path / "keys.sqlite", os.urandom(32)
    open_server = partial(_server, primary[TENANT_A], endpoint, bucket, signer=signer, key_path=key_path, master_key=master_key)
    open_claim_store = partial(PostgresStore.worker_standalone, worker_dsn)
    owner._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")

    prompts = ("signed first prompt", "signed second prompt")
    count_tokens(prompts[0], "m")
    calls: list[tuple[str, LlmRequest]] = []

    def host(node: str) -> _Host:
        def reply(model_request: LlmRequest) -> LLMResponse:
            calls.append((node, model_request))
            return LLMResponse(f"answer from {node}")

        return _Host(
            agent=Agent("test", "Answer."),
            run_config=RunConfig(
                model_provider=ScriptedModelProvider.from_callback("test", "m", reply), metadata={"resolved_by": node}
            ),
        )

    enabled: set[str] = set()
    one_shot: set[str] = set()
    claims: list[tuple[int, Lease]] = []
    renewals: list[tuple[int, Lease]] = []
    drives: set[int] = set()
    deferred: list[Lease] = []
    deferred_event = Event()
    claim_runnable, renew, defer_drive = PostgresStore.claim_runnable, PostgresStore.renew, PostgresStore.defer_drive
    execute = _S3ProcessTransport.execute
    transport_errors: list[tuple[str, str, str]] = []
    fence = local()
    fenced_writes: list[str] = []

    @contextmanager
    def track_publication(publication):
        with publication() as tx:
            fence.active = True
            try:
                yield tx
            finally:
                fence.active = False

    def execute_with_diagnostics(transport, request: _S3Request, **kwargs):
        if getattr(fence, "active", False):
            assert request.operation != "get_object", "App Server input publication issued an S3 GET under the primary fence"
            if request.operation == "put_object":
                fenced_writes.append(request.operation)
        try:
            return execute(transport, request, **kwargs)
        except _S3TransportError as exc:
            transport_errors.append((request.operation, current_thread().name, str(exc)))
            raise

    def claim(store, *, owner, **kwargs):
        if owner not in enabled:
            return ()
        claimed = claim_runnable(store, owner=owner, **kwargs)
        for item in claimed:
            if item.lease is not None:
                assert store.runtime_mode == "worker" and not store.claims_all_tenants
                assert store._input_admission_policy() == (True, False, signer)
                claims.append((store.connection.info.backend_pid, item.lease))
                if owner in one_shot:
                    enabled.discard(owner)
        return claimed

    def heartbeat(store, lease, **kwargs):
        result = renew(store, lease, **kwargs)
        assert store.credential_tenant_id == TENANT_A
        assert store._input_admission_policy() == (True, False, signer)
        renewals.append((store.connection.info.backend_pid, result.lease))
        return result

    def drive(build_runtime, store, session_id):
        assert store.credential_tenant_id == TENANT_A
        assert store._input_admission_policy() == (True, False, signer)
        drives.add(store.connection.info.backend_pid)
        return build_runtime(store, session_id)

    def defer(store, lease, **kwargs):
        defer_drive(store, lease, **kwargs)
        deferred.append(lease)
        deferred_event.set()

    monkeypatch.setattr(PostgresStore, "claim_runnable", claim)
    monkeypatch.setattr(PostgresStore, "renew", heartbeat)
    monkeypatch.setattr(PostgresStore, "defer_drive", defer)
    monkeypatch.setattr(_S3ProcessTransport, "execute", execute_with_diagnostics)

    with (
        open_server(journal, owner=owner, open_claim_store=open_claim_store, host=host("A")) as (a, store, adapter),
        open_server(_reopen(journal), open_claim_store=open_claim_store, host=host("B")) as (b, peer, _),
    ):
        for server in (a, b):
            build_runtime = server.kernel._build_runtime_for_claim
            monkeypatch.setattr(server.kernel, "_build_runtime_for_claim", partial(drive, build_runtime))
            publication = server.kernel._root_publication_atomic
            monkeypatch.setattr(server.kernel, "_root_publication_atomic", partial(track_publication, publication))
        worker_a, worker_b = a.kernel._worker, b.kernel._worker
        created = request(a, "create", "thread/start", {"inputId": sid, "cwd": str(tmp_path)})
        assert created.get("result", {}).get("threadId") == sid, created
        assert store.read(sid, limit=1).records[0].record.feature_level == 2
        first = submission(sid, "first", text=prompts[0])
        enabled.add(worker_a._owner)

        def submit(server, params):
            for attempt in range(5):
                started = request(server, f"{params['inputId']}-{attempt}", "turn/start", params)
                if started.get("error", {}).get("code") != -32001:
                    break
                time.sleep(0.05)
            assert started.get("result", {}).get("replayed") is False, (started, transport_errors[-8:])
            return started["result"]

        fenced_writes.clear()
        first_receipt = submit(a, first)
        first_turn = first_receipt["turnId"]

        def ended(turn_id):
            return any(row.record.turn_id == turn_id and row.record.kind == "turn_ended" for row in store.read(sid).records)

        def wait_for_turn(turn_id, worker):
            try:
                eventually(lambda: ended(turn_id) and store.lease_status(sid)[2] == 0, timeout=30)
            except AssertionError as exc:
                records = [(row.seq, row.record.kind, row.record.turn_id) for row in store.read(sid).records][-12:]
                lease_and_route = store._one(
                    "SELECT lease_epoch,lease_owner,drive_retry_at_ms,level,handler_version FROM sk_session WHERE session_id=%s",
                    (sid,),
                )
                claim_ids = [(lease.owner, lease.epoch) for _, lease in claims][-8:]
                defer_ids = [(lease.owner, lease.epoch) for lease in deferred][-8:]
                raise AssertionError(
                    f"Turn {turn_id} did not settle. Worker error={worker.last_error!r}, claims={claim_ids}, "
                    f"deferrals={defer_ids}, lease and route={lease_and_route}, records={records}, "
                    f"transport errors={transport_errors[-8:]}"
                ) from exc

        wait_for_turn(first_turn, worker_a)
        enabled.discard(worker_a._owner)
        assert a.kernel._runtimes[sid].turn_id == first_turn
        assert [node for node, _ in calls] == ["A"]
        first_head = store.verify_ledger(sid)

        second = submission(sid, "second", expected=first_turn, text=prompts[1])
        second_receipt = submit(b, second)
        second_turn = second_receipt["turnId"]
        assert fenced_writes
        assert b.kernel._runtimes[sid].turn_id == second_turn
        assert a.kernel._runtimes[sid].turn_id == first_turn
        published = _keys(inspector, bucket)

        steer = submission(sid, "steer", expected=first_turn, text="unexpected steer")
        assert "error" in request(b, "steer", "turn/steer", steer)
        cancel = {"threadId": sid, "expectedTurnId": first_turn, "actionId": "cancel"}
        assert "error" in request(b, "cancel", "turn/interrupt", cancel)
        assert [row.item.input_id for row in store.peek_inbox(sid)] == ["second"]
        assert _keys(inspector, bucket) == published

        stale_claim_offset, stale_defer_offset = len(claims), len(deferred)
        deferred_event.clear()
        one_shot.add(worker_a._owner)
        enabled.add(worker_a._owner)
        a.wake(sid)
        assert deferred_event.wait(15), worker_a.last_error
        eventually(lambda: store.lease_status(sid)[2] == 0)
        [stale_lease] = deferred[stale_defer_offset:]
        [(_, stale_claim)] = claims[stale_claim_offset:]
        assert (stale_lease.tenant_id, stale_lease.session_id, stale_lease.owner, stale_lease.epoch) == (
            TENANT_A,
            sid,
            worker_a._owner,
            stale_claim.epoch,
        )
        assert store._one("SELECT drive_retry_at_ms FROM sk_session WHERE session_id=%s", (sid,))[0] > 0
        assert [node for node, _ in calls] == ["A"]
        assert store.verify_ledger(sid) == first_head
        assert [row.item.input_id for row in store.peek_inbox(sid)] == ["second"]
        assert _keys(inspector, bucket) == published

        enabled.add(worker_b._owner)
        b.wake(sid)
        wait_for_turn(second_turn, worker_b)
        enabled.discard(worker_b._owner)
        assert {lease.owner for _, lease in claims[:stale_claim_offset]} == {worker_a._owner}
        assert {lease.owner for _, lease in claims[stale_claim_offset + 1 :]} == {worker_b._owner}
        assert all((lease.tenant_id, lease.session_id) == (TENANT_A, sid) for _, lease in claims)
        assert renewals and {lease.owner for _, lease in renewals} >= {worker_a._owner, worker_b._owner}
        roots = {store.connection.info.backend_pid, peer.connection.info.backend_pid}
        claim_pids, heartbeat_pids = ({pid for pid, _ in seen} for seen in (claims, renewals))
        assert drives.isdisjoint(roots | claim_pids | heartbeat_pids)
        assert roots.isdisjoint(claim_pids | heartbeat_pids) and claim_pids.isdisjoint(heartbeat_pids)
        for (node, model_request), prompt in zip(calls, prompts, strict=True):
            assert model_request.metadata["resolved_by"] == node
            assert any(message.role == "user" and message.content == prompt for message in model_request.messages)
        assert [node for node, _ in calls] == ["A", "B"]
        for turn_id in (first_turn, second_turn):
            [start] = [
                row.record
                for row in store.read(sid).records
                if row.record.turn_id == turn_id and row.record.kind == "turn_started"
            ]
            assert start._payload["requirements"]["level"] == 2
        assert store.peek_inbox(sid) == ()
        for input_id in ("first", "second"):
            retained = store.read_input(sid, input_id)
            assert retained is not None and retained.consumed_seq is not None and retained.item.admission_proof is not None
            assert retained.item.feature_level == 2
            assert retained.item.actor is not None and retained.item.actor["subject_id"] == _PRINCIPAL["subject_id"]
        objects = store._rows("SELECT ciphertext,object_locator FROM sk_content WHERE session_id=%s", (sid,))
        assert objects and all(ciphertext is None and locator is not None for ciphertext, locator in objects)
        physical = _keys(inspector, bucket)
        assert {adapter._key(locator[1:]) for _, locator in objects} == physical
        head = store.verify_ledger(sid)

    with open_server(_reopen(journal)) as (cold, store, _):
        for params, receipt in ((first, first_receipt), (second, second_receipt)):
            replay = request(cold, "replay-" + params["inputId"], "turn/start", params)
            assert replay.get("result") == receipt | {"replayed": True}, replay
        conflicting = second | {"input": [{"type": "text", "text": "different prompt"}]}
        assert request(cold, "conflict", "turn/start", conflicting)["error"]["code"] == -32024
        assert project_result(cold.kernel.store, sid, first_turn).final_output == "answer from A"
        assert project_result(cold.kernel.store, sid, second_turn).final_output == "answer from B"
        assert store.verify_ledger(sid) == head
        assert _keys(inspector, bucket) == physical
