from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel._allocation_journal import SQLiteAllocationJournal
from vv_agent.kernel._content import (
    ContentCleanupUnavailable,
    ContentIntegrityError,
    ContentRef,
    ContentUnavailable,
    KeyUnavailable,
    Principal,
    WrappedDataKey,
)
from vv_agent.kernel._external_content import _SQLiteObjectContent
from vv_agent.kernel._records import InboxItem
from vv_agent.session.content_runtime import (
    SESSION_WORKSPACE,
    bind_content_runtime,
    durable_content_store,
)
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.s3_content import S3CiphertextStore
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .test_s1_deletion_recovery_acceptance import Signer
from .test_s3_reservation_adapter import _Client


class S3Fixture(_Client):
    def __init__(self, journal: SQLiteAllocationJournal) -> None:
        super().__init__(journal.get)
        self.journal = journal

    def put_object(self, **request: Any) -> dict[str, Any]:
        if "IfMatch" not in request:
            return super().put_object(**request)
        row = self.journal.get(bytes.fromhex(request["Key"].rsplit("/", maxsplit=1)[1]))
        assert row is not None and row.owner is not None
        assert row.state == "bound" and row.reservation_etag == request["IfMatch"]
        assert self.journal.get_key_intent(row.owner.provider_instance_id, row.owner.key_id) is not None
        self.calls.append(request)
        self.objects[request["Key"]] = request["Body"]
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


def test_sqlite_session_driver_object_round_trip(tmp_path: Path) -> None:
    primary = tmp_path / "sessions.sqlite"
    sidecar = tmp_path / "allocation.sqlite"
    key_path = tmp_path / "keys.sqlite"
    identity = SQLiteAllocationJournal.provision(sidecar, database_id="test-primary", namespace="fixture")
    journal = SQLiteAllocationJournal(sidecar, identity)
    client = S3Fixture(journal)
    objects = S3CiphertextStore(client, bucket="fixture", journal=journal)
    with (
        SQLiteStore.standalone(str(primary)) as raw,
        closing(LocalKeyProvider(b"x" * 32, state_path=key_path)) as provider,
    ):
        raw.configure_input_admission(allow_unsigned=True)
        with closing(SessionDriver(store=raw, content_key_provider=provider)) as inline:
            inline.create("inline", "/inline")
            inline_record = inline.store.read("inline", limit=1).records[0].record
            inline_ref = ContentRef.from_dict(inline_record.payload["workspace"])
        provider_id = provider._provision_exact_key_lifecycle()
        external = _SQLiteObjectContent(journal, objects, provider_id)
        bind_content_runtime(raw, key_provider=provider, cipher=AESGCMCipher(), _object_content=external)
        with closing(SessionDriver(store=raw)) as driver:
            raw.connection.execute("BEGIN DEFERRED")
            try:
                with pytest.raises(RuntimeError, match="store-owned SQLite write transaction"):
                    driver.create("host-transaction", "/rejected")
            finally:
                raw.connection.rollback()
            assert not client.calls
            assert journal.scoped_key_intents("default", ("host-transaction",)) == ()
            driver.create("objects", "/external", {"scope": "test"})
            created = driver.store.read("objects", limit=1).records[0].record
            workspace = ContentRef.from_dict(created.payload["workspace"])
            driver.push("objects", InboxItem("input", "user", {"content": "object payload"}))
            inbox_body = raw._rows("SELECT body FROM sk_inbox WHERE session_id=%s", ("objects",))
            item = InboxItem.parse(inbox_body[0][0])
            input_ref = ContentRef.from_dict(item.payload["content"])
            assert driver.read_input_content("objects", "input", input_ref) == "object payload"
            assert driver.read_record_content("inline", 1, inline_ref, slot=SESSION_WORKSPACE) == "/inline"
            assert driver.read_record_content("objects", 1, workspace, slot=SESSION_WORKSPACE) == "/external"
            assert durable_content_store(driver.store).get("default", "objects", workspace) == b'"/external"'
            assert not raw._rows(
                "SELECT 1 FROM sk_content WHERE session_id=%s AND (ciphertext IS NOT NULL OR object_locator IS NULL)",
                ("objects",),
            )
            assert raw._rows("SELECT 1 FROM sk_content WHERE session_id=%s AND ciphertext IS NOT NULL", ("inline",))
            assert len(journal.scoped_key_intents("default", ("objects",))) == 1
            deletion = SessionDeletionStore(raw, SQLiteContentStore(raw, provider))
            actor = Principal.from_dict(created.payload["principal"])
            with pytest.raises(ContentCleanupUnavailable):
                deletion.purge_session("default", "objects", actor=actor)
            assert raw._rows("SELECT deletion_state FROM sk_session WHERE session_id=%s", ("objects",)) == [("live",)]

    reopened_journal = SQLiteAllocationJournal(sidecar, identity)
    reopened_objects = S3CiphertextStore(client, bucket="fixture", journal=reopened_journal)
    with (
        SQLiteStore.standalone(str(primary)) as reopened,
        closing(LocalKeyProvider(b"x" * 32, state_path=key_path)) as provider,
    ):
        reopened.configure_input_admission(allow_unsigned=True)
        unbound_deletion = SessionDeletionStore(reopened, SQLiteContentStore(reopened, provider))
        for erase in (unbound_deletion.forget_session, unbound_deletion.purge_session):
            with pytest.raises(ContentCleanupUnavailable):
                erase("default", "objects", actor=actor)
        assert reopened._rows("SELECT deletion_state FROM sk_session WHERE session_id=%s", ("objects",)) == [("live",)]
        bind_content_runtime(
            reopened,
            key_provider=provider,
            cipher=AESGCMCipher(),
            _object_content=_SQLiteObjectContent(reopened_journal, reopened_objects, provider_id),
        )
        with closing(SessionDriver(store=reopened)) as driver:
            assert driver.read_record_content("objects", 1, workspace, slot=SESSION_WORKSPACE) == "/external"
            (locator,) = reopened._rows(
                "SELECT object_locator FROM sk_content WHERE session_id=%s AND content_id=%s",
                ("objects", workspace.content_id),
            )[0]
            key = objects._key(locator[1:])
            client.objects[key] = bytes(len(client.objects[key]))
            with pytest.raises(ContentIntegrityError):
                durable_content_store(driver.store).get("default", "objects", workspace)


@pytest.mark.parametrize("operation", ["forget", "purge"])
@pytest.mark.parametrize("cut", ["provider", "delete", "object_ack", "sql_commit", "journal"])
def test_external_root_erasure_resumes_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, cut: str
) -> None:
    primary, key_path, sidecar = (tmp_path / name for name in ("primary.sqlite", "keys.sqlite", "allocation.sqlite"))
    identity = SQLiteAllocationJournal.provision(sidecar, database_id="primary", namespace="fixture")
    journal = SQLiteAllocationJournal(sidecar, identity)
    client = S3Fixture(journal)
    signer = Signer()
    delete_keys: list[str] = []
    original_delete = client.delete_object

    def observe_delete(*, Bucket: str, Key: str) -> dict[str, Any]:
        delete_keys.append(Key)
        return original_delete(Bucket=Bucket, Key=Key)

    monkeypatch.setattr(client, "delete_object", observe_delete)

    class Interrupted(RuntimeError):
        pass

    with SQLiteStore.standalone(str(primary)) as raw, closing(LocalKeyProvider(b"x" * 32, state_path=key_path)) as provider:
        raw.configure_input_admission(signer=signer)
        provider_id = provider._provision_exact_key_lifecycle()
        binding = _SQLiteObjectContent(
            journal, S3CiphertextStore(client, bucket="fixture", journal=journal), provider_id, terminal_bound_delete=True
        )
        bind_content_runtime(raw, key_provider=provider, cipher=AESGCMCipher(), _object_content=binding)
        with closing(SessionDriver(store=raw)) as driver:
            driver.create("objects", "/external")
            actor = Principal.from_dict(driver.store.read("objects", limit=1).records[0].record.payload["principal"])
        ((key_id, wrapped_key),) = raw._rows("SELECT key_id,wrapped_key FROM sk_session_key")
        wrapped = WrappedDataKey("default", "objects", key_id, wrapped_key)
        service = SessionDeletionStore(raw, SQLiteContentStore(raw, provider))
        erase = service.forget_session if operation == "forget" else service.purge_session
        with monkeypatch.context() as patch:
            if cut == "provider":

                def after_revocation(*_args: Any) -> None:
                    with SQLiteStore.standalone(str(primary)) as observer:
                        tombstone = observer._tombstone("objects")
                        assert tombstone is not None and tombstone.state.value == "pending"
                    with pytest.raises(KeyUnavailable):
                        provider.unwrap("default", "objects", wrapped)
                    raise Interrupted("lost key retirement receipt")

                patch.setattr(journal, "record_key_retirement", after_revocation)
            elif cut == "delete":
                client.fail = "delete"
            elif cut == "object_ack":
                original_confirm = journal.confirm_bound_delete

                def after_confirm(*args: Any) -> None:
                    assert original_confirm(*args).delete_confirmed
                    raise Interrupted("lost bound object confirmation acknowledgement")

                patch.setattr(journal, "confirm_bound_delete", after_confirm)
            elif cut == "sql_commit":

                def after_content_removal(*_args: Any, **_kwargs: Any) -> None:
                    assert raw._rows("SELECT object_locator FROM sk_content") == []
                    raise Interrupted("lost cleanup acknowledgement")

                patch.setattr(journal, "finish_reap", after_content_removal)
            else:
                original_finish = journal.finish_key_intent

                def after_key_intent_finish(*args: Any) -> None:
                    assert original_finish(*args)
                    raise Interrupted("lost last journal acknowledgement")

                patch.setattr(journal, "finish_key_intent", after_key_intent_finish)

            with pytest.raises(ContentCleanupUnavailable if cut == "delete" else Interrupted):
                erase("default", "objects", actor=actor)
        client.fail = ""
        pending = raw._tombstone("objects")
        assert pending is not None and pending.state.value == "pending"
        complete = raw.complete_forget if operation == "forget" else raw.complete_purge
        with pytest.raises(ContentCleanupUnavailable):
            complete("objects")

    with SQLiteStore.standalone(str(primary)) as raw, closing(LocalKeyProvider(b"x" * 32, state_path=key_path)) as provider:
        raw.configure_input_admission(signer=signer)
        complete = raw.complete_forget if operation == "forget" else raw.complete_purge
        with pytest.raises(ContentCleanupUnavailable):
            complete("objects")
        with pytest.raises(KeyUnavailable):
            provider.unwrap("default", "objects", wrapped)

        reopened = SQLiteAllocationJournal(sidecar, identity)
        for candidate in (
            SQLiteAllocationJournal(
                tmp_path / "wrong.sqlite",
                SQLiteAllocationJournal.provision(tmp_path / "wrong.sqlite", database_id="primary", namespace="fixture"),
            ),
            reopened,
        ):
            wrong_client = S3Fixture(candidate)
            wrong_client.endpoint_url = "http://another-fixture.invalid" if candidate is reopened else client.endpoint_url
            wrong = _SQLiteObjectContent(
                candidate,
                S3CiphertextStore(wrong_client, bucket="fixture", journal=candidate),
                provider_id,
                terminal_bound_delete=True,
            )
            bind_content_runtime(raw, key_provider=provider, cipher=AESGCMCipher(), _object_content=wrong)
            service = SessionDeletionStore(raw, SQLiteContentStore(raw, provider))
            erase = service.forget_session if operation == "forget" else service.purge_session
            with pytest.raises(ContentCleanupUnavailable, match="matching journal"):
                erase("default", "objects", actor=actor)
        binding = _SQLiteObjectContent(
            reopened, S3CiphertextStore(client, bucket="fixture", journal=reopened), provider_id, terminal_bound_delete=True
        )
        bind_content_runtime(raw, key_provider=provider, cipher=AESGCMCipher(), _object_content=binding)
        service = SessionDeletionStore(raw, SQLiteContentStore(raw, provider))
        erase = service.forget_session if operation == "forget" else service.purge_session
        missing = sidecar.with_suffix(".missing")
        sidecar.rename(missing)
        sent = len(delete_keys)
        try:
            with pytest.raises(ContentUnavailable):
                erase("default", "objects", actor=actor)
        finally:
            missing.rename(sidecar)
        assert len(delete_keys) == sent
        if cut == "journal":
            with monkeypatch.context() as patch:
                patch.setattr(provider, "_verify_exact_retirement", lambda _intent: None)
                with pytest.raises(ContentCleanupUnavailable, match="retirement evidence"):
                    erase("default", "objects", actor=actor)
        receipt = erase("default", "objects", actor=actor)
        assert receipt.authenticity.value == "signed"
        assert erase("default", "objects", actor=actor) == receipt
        complete_tombstone = raw._tombstone("objects")
        assert complete_tombstone is not None and complete_tombstone.state.value == "complete"
        assert raw._rows("SELECT key_id FROM sk_session_key") == []
        assert raw._rows("SELECT content_id FROM sk_content") == []
        assert reopened.scoped_key_intents("default", ("objects",)) == ()
        assert reopened.scoped_allocations("default", ("objects",)) == ()
        assert client.objects == {}
        if cut in {"sql_commit", "journal"}:
            assert len(delete_keys) == sent
        if cut == "object_ack":
            assert len(delete_keys) == len(set(delete_keys))
