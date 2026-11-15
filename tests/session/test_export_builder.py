"""Public export_session builder over a real S1 SQLite session."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentLabels,
    ContentRef,
    ExportContentMode,
    ExportLineKind,
    ExportModeMismatch,
    InvalidExport,
    Principal,
    SessionHead,
)
from vv_agent.kernel._sql import LedgerIntegrityError, SessionNotLive, stored_input_proof
from vv_agent.kernel._store import Conflict
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.export import export_session
from vv_agent.session.records import SessionSpec, digest
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

from .helpers import content_ref, item, principal, record

TENANT = "default"
SESSION = "export-builder-session"
MASTER_KEY = b"e" * 32


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "export-builder-ledger-key"

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _create(store: SQLiteStore) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(SESSION, principal(), content_ref("workspace"), attributes=content_ref("attributes")),
            consumers=(),
        )


def _append_turn(store: SQLiteStore) -> None:
    lease = store.acquire(SESSION, owner="export-builder-test", ttl_ms=10_000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append(
            SESSION,
            lease=lease,
            expected_seq=1,
            commit_id="turn",
            records=(record("turn_started", sid=SESSION),),
        )
    assert store.release(lease)


def _consume_input(store: SQLiteStore, input_id: str) -> None:
    stored = store.read_input(SESSION, input_id)
    assert stored is not None
    lease = store.acquire(SESSION, owner="export-builder-test", ttl_ms=10_000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append(
            SESSION,
            lease=lease,
            expected_seq=2,
            commit_id=f"consume-{input_id}",
            records=(
                record(
                    "input_applied",
                    sid=SESSION,
                    input=stored.item.to_dict(),
                    input_digest=digest(stored.item.to_dict()),
                    input_proof=stored_input_proof(stored),
                    disposition="applied",
                ),
            ),
            consume_input_ids=(input_id,),
        )


def _deletion_principal(key_id: str) -> Principal:
    return Principal(
        TENANT,
        "export-deleter",
        "service",
        ContentRef("content/export-deleter", "0" * 64, 0, key_id, ContentLabels()),
    )


def _head(document, key_id: str) -> SessionHead:
    return SessionHead(
        tenant_id=document.header.tenant_id,
        session_id=document.header.session_id,
        final_seq=document.header.head_seq,
        chain_digest=document.header.chain_digest,
        key_id=key_id,
    )


def test_export_session_builds_complete_verified_skeleton(tmp_path) -> None:
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "signed.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("input-a"))
            tx.push(SESSION, item("input-b"))
        _append_turn(store)
        _consume_input(store, "input-a")

        document = export_session(store.for_tenant(TENANT), SESSION)
        head = _head(document, signer.key_id)
        document.verify(head, signer, expected_head=(head.final_seq, head.chain_digest))

        assert [line.kind for line in document.lines] == [
            ExportLineKind.RECORD,
            ExportLineKind.RECORD,
            ExportLineKind.RECORD,
            ExportLineKind.INBOX,
            ExportLineKind.INBOX,
            ExportLineKind.CHECKPOINT,
            ExportLineKind.CHECKPOINT,
            ExportLineKind.CHECKPOINT,
            ExportLineKind.MANIFEST,
        ]
        manifest = document.lines[-1].value
        assert (manifest["record_lines"], manifest["inbox_lines"], manifest["checkpoint_lines"]) == (3, 2, 3)
        assert document.header.content_mode is ExportContentMode.SKELETON

        stored_records = store._rows(
            "SELECT seq,digest,chain_digest,writer_epoch,created_ms FROM sk_record "
            "WHERE tenant_id=? AND session_id=? ORDER BY seq",
            (TENANT, SESSION),
        )
        exported_records = [line.value["storage_envelope"] for line in document.lines if line.kind is ExportLineKind.RECORD]
        assert [
            (
                envelope["seq"],
                envelope["record_digest"],
                envelope["chain_digest"],
                envelope["writer_epoch"],
                envelope["created_ms"],
            )
            for envelope in exported_records
        ] == stored_records

        exported_inputs = [line.value for line in document.lines if line.kind is ExportLineKind.INBOX]
        stored_inputs = [store.read_input(SESSION, input_id) for input_id in ("input-a", "input-b")]
        assert [line["input_seq"] for line in exported_inputs] == [1, 2]
        assert [line["consumed_seq"] for line in exported_inputs] == [3, None]
        assert [line["input"] for line in exported_inputs] == [stored.item.to_dict() for stored in stored_inputs]
        assert all(line["input"]["admission_proof"] is not None for line in exported_inputs)
        assert all(line["input"]["payload"]["content"] == content_ref("hello") for line in exported_inputs)
        applied_record = next(
            line.value["record"]
            for line in document.lines
            if line.kind is ExportLineKind.RECORD and line.value["record"]["kind"] == "input_applied"
        )
        assert applied_record["payload"]["input_proof"] is not None

        exported_checkpoints = [line.value for line in document.lines if line.kind is ExportLineKind.CHECKPOINT]
        assert [(line["seq"], line["chain_digest"]) for line in exported_checkpoints] == [
            (envelope["seq"], envelope["chain_digest"]) for envelope in exported_records
        ]

        wire = canonical_json_bytes(document.to_dict())
        assert b'"line":"content"' not in wire
        assert b'"decoded_value"' not in wire
        assert b"plaintext" not in wire


def test_empty_inbox_hash_only_export_is_explicitly_unsigned(tmp_path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "hash-only.sqlite")) as store:
        store.install_schema()
        _create(store)

        document = export_session(store, SESSION)
        document.verify(_head(document, "hash-only"))

        manifest = document.lines[-1].value
        assert (manifest["record_lines"], manifest["inbox_lines"], manifest["checkpoint_lines"]) == (1, 0, 0)


def test_hash_only_export_verifies_consumed_input_without_proofs(tmp_path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "hash-only-consumed.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("input"))
        _append_turn(store)
        _consume_input(store, "input")

        document = export_session(store, SESSION)
        head = _head(document, "hash-only")
        document.verify(head, expected_head=(head.final_seq, head.chain_digest))

        inbox = next(line.value for line in document.lines if line.kind is ExportLineKind.INBOX)
        applied = next(
            line.value["record"]
            for line in document.lines
            if line.kind is ExportLineKind.RECORD and line.value["record"]["kind"] == "input_applied"
        )
        assert inbox["consumed_seq"] == 3
        assert inbox["input"]["admission_proof"] is None
        assert applied["payload"]["input_proof"] is None
        assert not any(line.kind is ExportLineKind.CHECKPOINT for line in document.lines)
        assert b'"line":"content"' not in canonical_json_bytes(document.to_dict())
        assert b'"decoded_value"' not in canonical_json_bytes(document.to_dict())


def test_reopened_signed_store_without_verifier_cannot_make_unsigned_export(tmp_path) -> None:
    path = tmp_path / "reopened-signed.sqlite"
    signer = Signer()
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("input"))
        _append_turn(store)
        _consume_input(store, "input")

    with (
        SQLiteStore.standalone(str(path)) as reopened,
        pytest.raises(LedgerIntegrityError, match="signer is required"),
    ):
        export_session(reopened, SESSION)

    with SQLiteStore.standalone(str(path)) as reopened:
        document = export_session(reopened, SESSION, signer=signer)
        head = _head(document, signer.key_id)
        document.verify(head, signer, expected_head=(head.final_seq, head.chain_digest))
        assert sum(line.kind is ExportLineKind.CHECKPOINT for line in document.lines) == 3


def test_forgotten_signed_session_exports_retained_skeleton(tmp_path) -> None:
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "forgotten-export.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("input"))
        _append_turn(store)
        _consume_input(store, "input")
        content = SQLiteContentStore(store, LocalKeyProvider(MASTER_KEY))
        deletion = SessionDeletionStore(store, content, signer=signer)

        deletion.forget_session(TENANT, SESSION, actor=_deletion_principal(signer.key_id))

        document = export_session(store.for_tenant(TENANT), SESSION, signer=signer)
        head = _head(document, signer.key_id)
        document.verify(head, signer, expected_head=(head.final_seq, head.chain_digest))
        assert document.header.content_mode is ExportContentMode.SKELETON
        assert not any(line.kind is ExportLineKind.CONTENT for line in document.lines)
        inbox = next(line.value for line in document.lines if line.kind is ExportLineKind.INBOX)
        assert inbox["consumed_seq"] == 3
        assert inbox["input"]["admission_proof"] is not None
        assert sum(line.kind is ExportLineKind.CHECKPOINT for line in document.lines) == 3
        assert store._one(
            "SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        ) == ("forgotten",)


def test_export_rejects_unstable_or_incomplete_forget_state(tmp_path) -> None:
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "unstable-forget.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)

        for deletion_state in ("forgetting", "purging"):
            store._rows(
                "UPDATE sk_session SET deletion_state=%s WHERE tenant_id=%s AND session_id=%s",
                (deletion_state, TENANT, SESSION),
            )
            with pytest.raises(SessionNotLive):
                export_session(store, SESSION, signer=signer)

        store._rows(
            "UPDATE sk_session SET deletion_state='forgotten' WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        )
        with pytest.raises(InvalidExport, match="complete tombstone"):
            export_session(store, SESSION, signer=signer)


def test_export_rejects_wrong_tenant_session_and_expected_head(tmp_path) -> None:
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "binding.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        document = export_session(store, SESSION, signer=signer)
        head = _head(document, signer.key_id)

        with pytest.raises(Conflict):
            export_session(store.for_tenant("other-tenant"), SESSION, signer=signer)
        with pytest.raises(Conflict):
            export_session(store, "missing-session", signer=signer)
        with pytest.raises(ContentBindingMismatch, match="expected head"):
            export_session(
                store,
                SESSION,
                signer=signer,
                expected_head=(head.final_seq + 1, head.chain_digest),
            )


def test_export_rejects_content_hydration_mode(tmp_path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "include-content.sqlite")) as store:
        store.install_schema()
        _create(store)

        with pytest.raises(ExportModeMismatch, match="does not hydrate content"):
            export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)
