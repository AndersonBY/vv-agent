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
    ExportDocument,
    ExportLineKind,
    ForgottenSessionReuse,
    InvalidExport,
    InvalidTombstone,
    Principal,
    SessionHead,
)
from vv_agent.kernel._sql import LedgerIntegrityError, SessionNotLive, stored_input_proof
from vv_agent.kernel._store import Conflict
from vv_agent.session.content_runtime import SessionContentWriter
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
    def __init__(self, key_id: str = "export-builder-ledger-key") -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = key_id

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


class Verifier:
    def __init__(self, *signers: Signer) -> None:
        self._signers = {signer.key_id: signer for signer in signers}

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        signer = self._signers.get(key_id)
        return signer is not None and signer.verify(key_id, message, signature)


class RotatingSigner:
    def __init__(self, active: Signer, *historical: Signer) -> None:
        self.active = active
        self.verifier = Verifier(active, *historical)

    def bind_key(self):
        return self.active.bind_key()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        return self.verifier.verify(key_id, message, signature)


def _create(store: SQLiteStore, session_id: str = SESSION, *, parent: str | None = None) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                principal(),
                content_ref("workspace" if session_id == SESSION else f"workspace-{session_id}"),
                parent_session_id=parent,
                parent_operation_id=f"operation/{session_id}" if parent is not None else None,
                attributes=content_ref("attributes" if session_id == SESSION else f"attributes-{session_id}"),
            ),
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


def _deletion_principal(attributes: ContentRef) -> Principal:
    return Principal(
        TENANT,
        "export-deleter",
        "service",
        attributes,
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
    signer = Signer("export-builder-checkpoint-key")
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
        assert {line["key_id"] for line in exported_checkpoints} == {signer.key_id}
        assert signer.key_id != content_ref("hello")["key_id"]
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
        store.configure_input_admission(allow_unsigned=True)
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
        pytest.raises(LedgerIntegrityError, match="verifier is required"),
    ):
        export_session(reopened, SESSION)

    with SQLiteStore.standalone(str(path)) as reopened:
        document = export_session(reopened, SESSION, signer=signer)
        head = _head(document, signer.key_id)
        document.verify(head, signer, expected_head=(head.final_seq, head.chain_digest))
        assert sum(line.kind is ExportLineKind.CHECKPOINT for line in document.lines) == 3


def test_key_rotation_keeps_historical_checkpoints_and_input_proofs_verifiable(tmp_path) -> None:
    old_signer = Signer("ledger-old")
    new_signer = Signer("ledger-new")
    verifier = Verifier(old_signer, new_signer)
    with SQLiteStore.standalone(str(tmp_path / "rotated.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(old_signer)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("old-input"))
        lease = store.acquire(SESSION, owner="rotation-test", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                SESSION,
                lease=lease,
                expected_seq=1,
                commit_id="old-turn",
                records=(record("turn_started", sid=SESSION),),
            )
        assert store.release(lease)

        store.configure_input_admission(RotatingSigner(new_signer, old_signer))
        lease = store.acquire(SESSION, owner="rotation-test", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                SESSION,
                lease=lease,
                expected_seq=2,
                commit_id="new-usage",
                records=(record("usage_observed", sid=SESSION, meter_id="rotation", observation=1),),
            )

        document = export_session(store, SESSION, signer=verifier)
        head = _head(document, new_signer.key_id)
        document.verify(head, verifier, expected_head=(head.final_seq, head.chain_digest))
        checkpoints = [line.value for line in document.lines if line.kind is ExportLineKind.CHECKPOINT]
        inbox = next(line.value for line in document.lines if line.kind is ExportLineKind.INBOX)
        assert [checkpoint["key_id"] for checkpoint in checkpoints] == ["ledger-old", "ledger-old", "ledger-new"]
        assert inbox["input"]["admission_proof"]["admission_key_id"] == old_signer.key_id


def test_forgotten_signed_session_exports_retained_skeleton(tmp_path) -> None:
    checkpoint_signer = Signer("export-builder-checkpoint-key")
    tombstone_signer = Signer("export-builder-tombstone-key")
    verifier = Verifier(checkpoint_signer, tombstone_signer)
    with SQLiteStore.standalone(str(tmp_path / "forgotten-export.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(checkpoint_signer)
        _create(store)
        with store.atomic() as tx:
            tx.push(SESSION, item("input"))
        _append_turn(store)
        _consume_input(store, "input")
        content = SQLiteContentStore(store, LocalKeyProvider(MASTER_KEY))
        with store.for_tenant(TENANT).atomic():
            actor_ref = content.put(TENANT, SESSION, b"provider actor", ContentLabels())
        deletion = SessionDeletionStore(store, content, signer=tombstone_signer)

        deletion.forget_session(TENANT, SESSION, actor=_deletion_principal(actor_ref))

        document = export_session(store.for_tenant(TENANT), SESSION, signer=verifier)
        head = _head(document, checkpoint_signer.key_id)
        document.verify(head, verifier, expected_head=(head.final_seq, head.chain_digest))
        assert document.header.content_mode is ExportContentMode.SKELETON
        assert not any(line.kind is ExportLineKind.CONTENT for line in document.lines)
        inbox = next(line.value for line in document.lines if line.kind is ExportLineKind.INBOX)
        assert inbox["consumed_seq"] == 3
        assert inbox["input"]["admission_proof"] is not None
        checkpoints = [line.value for line in document.lines if line.kind is ExportLineKind.CHECKPOINT]
        assert len(checkpoints) == 3
        assert {checkpoint["key_id"] for checkpoint in checkpoints} == {checkpoint_signer.key_id}
        tombstone = store._tombstone(SESSION)
        assert tombstone is not None
        assert tombstone.key_id == tombstone_signer.key_id
        assert tombstone.key_id != checkpoint_signer.key_id
        assert tombstone.actor.attributes.key_id == actor_ref.key_id
        assert actor_ref.key_id not in {checkpoint_signer.key_id, tombstone_signer.key_id}
        assert store._one(
            "SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        ) == ("forgotten",)

        with pytest.raises(InvalidTombstone, match="tombstone signature is invalid"):
            export_session(store.for_tenant(TENANT), SESSION, signer=checkpoint_signer)
        with pytest.raises(LedgerIntegrityError, match="checkpoint signature is invalid"):
            export_session(store.for_tenant(TENANT), SESSION, signer=tombstone_signer)


def test_forget_cascade_exports_descendant_tombstones_with_independent_signers(tmp_path) -> None:
    child = "export-builder-child"
    checkpoint_signer = Signer("export-builder-checkpoint-key")
    tombstone_signer = Signer("export-builder-tombstone-key")
    verifier = Verifier(checkpoint_signer, tombstone_signer)
    with SQLiteStore.standalone(str(tmp_path / "forgotten-cascade.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(checkpoint_signer)
        _create(store, SESSION)
        _create(store, child, parent=SESSION)
        content = SQLiteContentStore(store, LocalKeyProvider(MASTER_KEY))
        with store.for_tenant(TENANT).atomic():
            actor_ref = content.put(TENANT, SESSION, b"provider actor", ContentLabels())
        deletion = SessionDeletionStore(store, content, signer=tombstone_signer)

        receipt = deletion.forget_session(TENANT, SESSION, actor=_deletion_principal(actor_ref))

        assert set(receipt.session_ids) == {SESSION, child}
        exported_tombstones = {}
        for session_id in (SESSION, child):
            document = export_session(store.for_tenant(TENANT), session_id, signer=verifier)
            head = _head(document, checkpoint_signer.key_id)
            document.verify(head, verifier, expected_head=(head.final_seq, head.chain_digest))
            tombstone = store._tombstone(session_id)
            assert tombstone is not None
            assert tombstone.session_id == session_id
            assert tombstone.scope_root_session_id == SESSION
            assert tombstone.key_id == tombstone_signer.key_id
            assert tombstone.actor.attributes.key_id == actor_ref.key_id
            assert len({tombstone.key_id, head.key_id, actor_ref.key_id}) == 3
            exported_tombstones[session_id] = (document, head, tombstone)
        child_document, child_head, child_tombstone = exported_tombstones[child]
        child_tombstone.verify(
            child_head,
            verifier,
            scope_root_session_id=SESSION,
        )
        with pytest.raises(ForgottenSessionReuse):
            child_tombstone.verify(
                child_head,
                verifier,
                scope_root_session_id=child_document.header.session_id,
            )
        with pytest.raises(Conflict):
            deletion.verify_tombstones(TENANT, "other")


def test_forgotten_hash_only_session_exports_unsigned_tombstone(tmp_path) -> None:
    child = "export-builder-hash-child"
    with SQLiteStore.standalone(str(tmp_path / "forgotten-hash-only.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        _create(store)
        _create(store, child, parent=SESSION)
        content = SQLiteContentStore(store, LocalKeyProvider(MASTER_KEY))
        with store.for_tenant(TENANT).atomic():
            actor_ref = content.put(TENANT, SESSION, b"provider actor", ContentLabels())
        deletion = SessionDeletionStore(store, content, allow_unsigned=True)

        receipt = deletion.forget_session(TENANT, SESSION, actor=_deletion_principal(actor_ref))

        assert set(receipt.session_ids) == {SESSION, child}
        for session_id in (SESSION, child):
            document = export_session(store.for_tenant(TENANT), session_id)
            head = _head(document, "hash-only")
            document.verify(head, expected_head=(head.final_seq, head.chain_digest))
            assert not any(line.kind is ExportLineKind.CONTENT for line in document.lines)
            assert not any(line.kind is ExportLineKind.CHECKPOINT for line in document.lines)
            tombstone = store._tombstone(session_id)
            assert tombstone is not None
            assert tombstone.key_id == "hash-only/sha256"
            assert tombstone.key_id != head.key_id
            assert tombstone.scope_root_session_id == SESSION
            assert tombstone.actor.attributes.key_id == actor_ref.key_id
            assert actor_ref.key_id != tombstone.key_id


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
        with pytest.raises(ContentBindingMismatch, match="expected_head"):
            export_session(
                store,
                SESSION,
                signer=signer,
                expected_head=(head.final_seq + 1, head.chain_digest),
            )


def test_export_includes_content_only_when_explicitly_requested(tmp_path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "include-content-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "include-content.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        with store.atomic() as tx:
            writer = SessionContentWriter(tx=tx, tenant_id=TENANT, session_id=SESSION, key_provider=provider)

            def spec(content) -> SessionSpec:
                principal_value, _ = content.service_principal("export-host")
                return SessionSpec(
                    SESSION,
                    principal_value,
                    content.put_json("/workspace").to_dict(),
                    attributes=content.put_json({"purpose": "include-content"}).to_dict(),
                )

            writer.create_session(spec, consumers=())

        skeleton = export_session(store, SESSION)
        assert not any(line.kind is ExportLineKind.CONTENT for line in skeleton.lines)

        document = export_session(
            store,
            SESSION,
            content_mode=ExportContentMode.INCLUDE_CONTENT,
            content_store=SQLiteContentStore(store, provider),
        )
        content_lines = [line.value for line in document.lines if line.kind is ExportLineKind.CONTENT]
        assert {str(line["decoded_value"]) for line in content_lines} == {
            "{'role': 'export-host'}",
            "/workspace",
            "{'purpose': 'include-content'}",
        }
        document.verify(_head(document, "hash-only"))
        without_content = [line.to_dict() for line in document.lines if line.kind is not ExportLineKind.CONTENT]
        with pytest.raises(InvalidExport, match="does not cover every referenced"):
            ExportDocument.from_dict(document.header.to_dict(), without_content).verify(_head(document, "hash-only"))
