"""Versioned skeleton export and tombstone authenticity lifecycle tests."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from typing import TypedDict

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from support.contract import fixture_root
from support.s1_fixture_values import append_fixture_records

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentLifecycleError,
    ExportContentMode,
    ExportDocument,
    ExportHeader,
    ForgetTombstone,
    InvalidExport,
    InvalidTombstone,
    SessionHead,
    TombstoneState,
)
from vv_agent.session.sqlite import SQLiteStore

from .helpers import (
    create_fixture_session,
    fixture_key_provider,
    item,
    push_fixture_input,
    record,
    wrap_fixture_store,
)

TENANT = "default"
SESSION = "export-session"
KEY_ID = "ledger-key-fixed"
FIXED_PRIVATE_BYTES = bytes.fromhex("a6d5c55c9f6216db66b7e0df14d51a4d220ba7074bdd09a8461f429b055bc84c")

type JsonObject = dict[str, object]


class PrincipalFixture(TypedDict, total=False):
    subject_id: str


class RecordPayloadFixture(TypedDict, total=False):
    principal: PrincipalFixture


class RecordFixture(TypedDict, total=False):
    payload: RecordPayloadFixture


class StorageEnvelopeFixture(TypedDict, total=False):
    tenant_id: str
    record_digest: str
    chain_digest: str


class AdmissionProofFixture(TypedDict):
    admission_key_id: str
    admission_signature: str
    admission_signed_ms: int


class InputFixture(TypedDict, total=False):
    input_id: str
    admission_proof: AdmissionProofFixture | None


class ExportHeaderFixture(TypedDict):
    tenant_id: str
    session_id: str
    head_seq: int
    chain_digest: str


class ExportLineFixture(TypedDict, total=False):
    line: str
    seq: int
    chain_digest: str
    key_id: str
    signature: str
    signed_ms: int
    input_seq: int
    received_ms: int
    input: InputFixture
    storage_envelope: StorageEnvelopeFixture
    record: RecordFixture
    checkpoint_lines: int


class FixedSigner:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.from_private_bytes(FIXED_PRIVATE_BYTES)
        self.key_id = KEY_ID

    def sign(self, message: bytes) -> bytes:
        return self.key.sign(message)

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode()
    return len(encoded).to_bytes(4, "big") + encoded


def _initial_chain(tenant_id: str, session_id: str) -> str:
    return sha256(b"vv-agent/S1" + _length_prefixed(tenant_id) + _length_prefixed(session_id)).hexdigest()


def _extend_chain(previous: str, seq: int, record_digest: str) -> str:
    return sha256(bytes.fromhex(previous) + seq.to_bytes(8, "big") + bytes.fromhex(record_digest)).hexdigest()


def _checkpoint_message(
    tenant_id: str,
    session_id: str,
    seq: int,
    chain_digest: str,
    key_id: str,
    signed_ms: int,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/checkpoint",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "seq": seq,
            "chain_digest": chain_digest,
            "key_id": key_id,
            "signed_ms": signed_ms,
        }
    )


def _input_message(
    tenant_id: str,
    session_id: str,
    input_id: str,
    input_seq: int,
    received_ms: int,
    body_digest: str,
    key_id: str,
    signed_ms: int,
) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/input",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "input_id": input_id,
            "input_seq": input_seq,
            "received_ms": received_ms,
            "body_digest": body_digest,
            "key_id": key_id,
            "signed_ms": signed_ms,
        }
    )


def _tombstone_message(value: ForgetTombstone) -> bytes:
    return canonical_json_bytes(
        {
            "domain": "vv-agent/S1/purge",
            "tenant_id": value.tenant_id,
            "session_id": value.session_id,
            "scope_root_session_id": value.scope_root_session_id,
            "final_seq": value.final_seq,
            "chain_digest": value.chain_digest,
            "deleted_ms": value.deleted_ms,
            "actor": value.actor.to_dict(),
        }
    )


def _lifecycle() -> dict:
    return json.loads((fixture_root() / "session_storage_lifecycle.json").read_text())


def _signed_fixture(signer: FixedSigner) -> tuple[ExportHeaderFixture, list[ExportLineFixture], SessionHead]:
    lifecycle = _lifecycle()
    case = deepcopy(lifecycle["export_cases"][0])
    header: ExportHeaderFixture = case["header"]
    lines: list[ExportLineFixture] = case["lines"]
    checkpoint = next(line for line in lines if line["line"] == "checkpoint")
    checkpoint["key_id"] = KEY_ID
    checkpoint["signature"] = base64.b64encode(
        signer.sign(
            _checkpoint_message(
                header["tenant_id"],
                header["session_id"],
                checkpoint["seq"],
                checkpoint["chain_digest"],
                checkpoint["key_id"],
                checkpoint["signed_ms"],
            )
        )
    ).decode("ascii")

    inbox = next(line for line in lines if line["line"] == "inbox")
    body = deepcopy(inbox["input"])
    body["admission_proof"] = None
    body_digest = sha256(canonical_json_bytes(body)).hexdigest()
    signed = signer.sign(
        _input_message(
            header["tenant_id"],
            header["session_id"],
            body["input_id"],
            inbox["input_seq"],
            inbox["received_ms"],
            body_digest,
            KEY_ID,
            inbox["received_ms"],
        )
    )
    body["admission_proof"] = {
        "admission_key_id": KEY_ID,
        "admission_signature": base64.b64encode(signed).decode("ascii"),
        "admission_signed_ms": inbox["received_ms"],
    }
    inbox["input"] = body
    head = SessionHead(
        tenant_id=header["tenant_id"],
        session_id=header["session_id"],
        final_seq=header["head_seq"],
        chain_digest=header["chain_digest"],
        key_id=KEY_ID,
    )
    return header, lines, head


def _record_line(stored, chain_digest: str) -> dict:
    return {
        "line": "record",
        "storage_envelope": {
            "schema_version": "S1",
            "tenant_id": TENANT,
            "session_id": stored.record.session_id,
            "seq": stored.seq,
            "record_digest": stored.record.digest,
            "chain_digest": chain_digest,
            "writer_epoch": stored.writer_epoch,
            "created_ms": stored.created_ms,
        },
        "record": stored.record.to_dict(),
    }


def _store_export(tmp_path) -> tuple[ExportDocument, SessionHead, FixedSigner]:
    signer = FixedSigner()
    provider = fixture_key_provider()
    try:
        with SQLiteStore.standalone(str(tmp_path / "export.sqlite")) as raw_store:
            raw_store.install_schema()
            raw_store.configure_input_admission(signer)
            store = wrap_fixture_store(raw_store, provider)
            create_fixture_session(store, SESSION, subject_id="fixture/export-host")
            push_fixture_input(store, SESSION, item("input"))
            append_fixture_records(store, SESSION, (record("turn_started", sid=SESSION),), commit_id="turn")

            page = store.read(SESSION)
            assert [stored.seq for stored in page.records] == [1, 2]
            previous = _initial_chain(TENANT, SESSION)
            record_lines: list[dict[str, object]] = []
            for stored in page.records:
                previous = _extend_chain(previous, stored.seq, stored.record.digest)
                record_lines.append(_record_line(stored, previous))

            stored_input = store.read_input(SESSION, "input")
            assert stored_input is not None
            inbox_lines: list[dict[str, object]] = [
                {
                    "line": "inbox",
                    "input_seq": stored_input.input_seq,
                    "received_ms": stored_input.received_ms,
                    "consumed_seq": stored_input.consumed_seq,
                    "input": stored_input.item.to_dict(),
                }
            ]
            checkpoint_lines: list[dict[str, object]] = [
                {
                    "line": "checkpoint",
                    "seq": seq,
                    "chain_digest": chain_digest,
                    "key_id": key_id,
                    "signature": base64.b64encode(bytes(signature)).decode("ascii"),
                    "signed_ms": signed_ms,
                }
                for seq, chain_digest, key_id, signature, signed_ms in store._rows(
                    "SELECT seq,chain_digest,key_id,signature,signed_ms FROM sk_checkpoint "
                    "WHERE tenant_id=? AND session_id=? ORDER BY seq",
                    (TENANT, SESSION),
                )
            ]
            lines = [
                *record_lines,
                *inbox_lines,
                *checkpoint_lines,
                {
                    "line": "manifest",
                    "record_lines": len(record_lines),
                    "inbox_lines": len(inbox_lines),
                    "checkpoint_lines": len(checkpoint_lines),
                    "head_seq": page.head_seq,
                    "chain_digest": previous,
                },
            ]
            document = ExportDocument.from_dict(
                ExportHeader(
                    tenant_id=TENANT,
                    session_id=SESSION,
                    head_seq=page.head_seq,
                    chain_digest=previous,
                    content_mode=ExportContentMode.SKELETON,
                ).to_dict(),
                lines,
            )
            return document, SessionHead(TENANT, SESSION, page.head_seq, previous, KEY_ID), signer
    finally:
        provider.close()


def test_versioned_skeleton_export_verifies_with_real_ed25519_proofs(tmp_path) -> None:
    document, head, signer = _store_export(tmp_path)

    document.verify(head, signer, expected_head=(head.final_seq, head.chain_digest))


def test_signed_skeleton_does_not_need_a_content_key() -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)

    ExportDocument.from_dict(header, lines).verify(head, signer)


def test_hash_only_mode_accepts_unsigned_history_but_signer_rejects_it() -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)
    unsigned = deepcopy(lines)
    unsigned = [line for line in unsigned if line["line"] != "checkpoint"]
    unsigned[-1]["checkpoint_lines"] = 0
    document = ExportDocument.from_dict(header, unsigned)

    document.verify(head)
    with pytest.raises(InvalidExport):
        document.verify(head, signer)


@pytest.mark.parametrize("mutation", ["body", "digest", "chain"])
def test_signed_export_rejects_record_tampering(mutation: str) -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)
    record_line = next(line for line in lines if line["line"] == "record")
    if mutation == "body":
        record_line["record"]["payload"]["principal"]["subject_id"] = "tampered"
        with pytest.raises(InvalidExport):
            ExportDocument.from_dict(header, lines)
        return
    if mutation == "digest":
        record_line["storage_envelope"]["record_digest"] = "0" * 64
        with pytest.raises(InvalidExport):
            ExportDocument.from_dict(header, lines)
        return
    record_line["storage_envelope"]["chain_digest"] = "0" * 64
    with pytest.raises(InvalidExport):
        ExportDocument.from_dict(header, lines).verify(head, signer)


def test_export_rejects_reordered_and_deleted_records(tmp_path) -> None:
    document, head, signer = _store_export(tmp_path)
    value = document.to_dict()
    records = [line for line in value["lines"] if line["line"] == "record"]
    reordered = deepcopy(value)
    reordered["lines"][0], reordered["lines"][1] = reordered["lines"][1], reordered["lines"][0]
    with pytest.raises(ContentLifecycleError):
        ExportDocument.from_dict(reordered["header"], reordered["lines"]).verify(head, signer)

    deleted = deepcopy(value)
    first_record = next(index for index, line in enumerate(deleted["lines"]) if line["line"] == "record")
    deleted["lines"].pop(first_record)
    first_checkpoint = next(index for index, line in enumerate(deleted["lines"]) if line["line"] == "checkpoint")
    deleted["lines"].pop(first_checkpoint)
    manifest = deleted["lines"][-1]
    manifest["record_lines"] -= 1
    manifest["checkpoint_lines"] -= 1
    with pytest.raises(InvalidExport):
        ExportDocument.from_dict(deleted["header"], deleted["lines"]).verify(head, signer)
    assert len(records) == 2


def test_signed_export_rejects_missing_checkpoint_and_aa_signature(tmp_path) -> None:
    document, head, signer = _store_export(tmp_path)
    missing = document.to_dict()
    checkpoint = next(index for index, line in enumerate(missing["lines"]) if line["line"] == "checkpoint")
    missing["lines"].pop(checkpoint)
    missing["lines"][-1]["checkpoint_lines"] -= 1
    with pytest.raises(InvalidExport):
        ExportDocument.from_dict(missing["header"], missing["lines"]).verify(head, signer)

    invalid = document.to_dict()
    checkpoint_line = next(line for line in invalid["lines"] if line["line"] == "checkpoint")
    checkpoint_line["signature"] = "AA=="
    with pytest.raises(InvalidExport):
        ExportDocument.from_dict(invalid["header"], invalid["lines"]).verify(head, signer)


def test_signed_export_rejects_invalid_input_proof() -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)
    inbox = next(line for line in lines if line["line"] == "inbox")
    proof = inbox["input"]["admission_proof"]
    assert proof is not None
    proof["admission_signature"] = "AA=="

    with pytest.raises(InvalidExport):
        ExportDocument.from_dict(header, lines).verify(head, signer)


@pytest.mark.parametrize("field", ["tenant_id", "session_id"])
def test_export_rejects_wrong_header_binding(field: str) -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)
    if field == "tenant_id":
        header["tenant_id"] = "other"
    else:
        header["session_id"] = "other"

    with pytest.raises(ContentBindingMismatch):
        ExportDocument.from_dict(header, lines).verify(head, signer)


def test_export_rejects_wrong_record_tenant_binding() -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)
    envelope = next(line for line in lines if line["line"] == "record")["storage_envelope"]
    envelope["tenant_id"] = "other"

    with pytest.raises(ContentBindingMismatch):
        ExportDocument.from_dict(header, lines).verify(head, signer)


def test_export_rejects_wrong_content_reference_key() -> None:
    case = deepcopy(_lifecycle()["export_cases"][1])
    content_line = next(line for line in case["lines"] if line["line"] == "content")
    content_line["key_id"] = "key/wrong"

    with pytest.raises(ContentBindingMismatch):
        ExportDocument.from_dict(case["header"], case["lines"]).verify(
            SessionHead(
                tenant_id=case["header"]["tenant_id"],
                session_id=case["header"]["session_id"],
                final_seq=case["header"]["head_seq"],
                chain_digest=case["header"]["chain_digest"],
                key_id=KEY_ID,
            )
        )


def test_export_rejects_wrong_expected_head() -> None:
    signer = FixedSigner()
    header, lines, head = _signed_fixture(signer)

    with pytest.raises(ContentBindingMismatch):
        ExportDocument.from_dict(header, lines).verify(head, signer, expected_head=(head.final_seq + 1, head.chain_digest))


def _signed_tombstone() -> tuple[ForgetTombstone, SessionHead, FixedSigner]:
    signer = FixedSigner()
    value = deepcopy(_lifecycle()["forget_tombstone"])
    value["key_id"] = KEY_ID
    value["actor"]["attributes"]["key_id"] = KEY_ID
    value["signature"] = "AA=="
    unsigned = ForgetTombstone.from_dict(value)
    tombstone = replace(unsigned, signature=base64.b64encode(signer.sign(_tombstone_message(unsigned))).decode("ascii"))
    head = SessionHead(
        tenant_id=tombstone.tenant_id,
        session_id=tombstone.session_id,
        final_seq=tombstone.final_seq,
        chain_digest=tombstone.chain_digest,
        key_id=KEY_ID,
    )
    return tombstone, head, signer


def test_tombstone_verifies_real_ed25519_signature_and_rejects_aa() -> None:
    tombstone, head, signer = _signed_tombstone()
    tombstone.verify(head, signer)
    assert tombstone.state is TombstoneState.COMPLETE

    with pytest.raises(InvalidTombstone):
        replace(tombstone, signature="AA==").verify(head, signer)


def test_tombstone_without_signer_is_explicit_hash_only_mode() -> None:
    tombstone, head, _signer = _signed_tombstone()

    tombstone.verify(head)
