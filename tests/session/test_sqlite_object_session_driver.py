from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel._allocation_journal import SQLiteAllocationJournal
from vv_agent.kernel._content import ContentCleanupUnavailable, ContentIntegrityError, ContentRef, Principal
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

from .test_s3_reservation_adapter import _Client


class S3Fixture(_Client):
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
