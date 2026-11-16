"""Versioned skeleton exports from the SQL session ledger."""

from __future__ import annotations

import base64
from typing import Any

from vv_agent.kernel._content import (
    ExportContentMode,
    ExportDocument,
    ExportHeader,
    ExportLine,
    ExportLineKind,
    ExportModeMismatch,
    InvalidExport,
    SessionHead,
    TombstoneState,
    _declared_content_slots,
)
from vv_agent.kernel._sql import LedgerIntegrityError, SessionNotLive, _require_expected_head
from vv_agent.session.content_store import ContentExportRef


def _content_export_refs(records: tuple[Any, ...], inputs: tuple[Any, ...]) -> tuple[ContentExportRef, ...]:
    refs: list[ContentExportRef] = []
    for stored in records:
        for declared in _declared_content_slots(stored.record.kind, stored.record._payload):
            refs.append(
                ContentExportRef(
                    record_seq=stored.seq,
                    input_id=None,
                    slot=declared.slot,
                    ref=declared.ref,
                    nested=declared.nested,
                )
            )
    for stored in inputs:
        item = stored.item
        for declared in _declared_content_slots(
            item.kind,
            item.payload,
            input_content=True,
            actor_attributes=item.actor["attributes"] if item.actor is not None else None,
        ):
            refs.append(
                ContentExportRef(
                    record_seq=None,
                    input_id=item.input_id,
                    slot=declared.slot,
                    ref=declared.ref,
                )
            )
    return tuple(refs)


def export_session(
    store: Any,
    session_id: str,
    *,
    content_mode: ExportContentMode = ExportContentMode.SKELETON,
    signer: Any | None = None,
    expected_head: SessionHead | tuple[int, str] | None = None,
    content_store: Any | None = None,
) -> ExportDocument:
    """Build and verify one S1 export from a tenant-bound SQL store.

    The configured store signer is used when ``signer`` is omitted. Hash-only
    stores produce an unsigned export whose integrity depends on canonical
    bytes and hash chains, not signatures. Content is hydrated only when
    ``content_mode`` is explicitly ``include-content``; callers may provide a
    bound durable content store for that mode.
    """

    try:
        mode = ExportContentMode(content_mode)
    except (TypeError, ValueError) as exc:
        raise ExportModeMismatch("content_mode is closed") from exc
    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise TypeError("export_session requires a tenant-bound SQL session store")
    required = (
        "atomic",
        "configure_input_admission",
        "_input_admission_policy",
        "_lock",
        "_one",
        "_rows",
        "_stored",
        "_stored_input",
        "_tombstone",
        "_verified_records",
        "_chain_anchor",
        "_verify_tombstone_object",
    )
    if any(not callable(getattr(store, name, None)) for name in required):
        raise TypeError("export_session requires a tenant-bound SQL session store")

    with store.atomic():
        row = store._lock(session_id)
        head_seq, inbox_seq = row[:2]
        _require_expected_head(
            expected_head,
            tenant_id=tenant_id,
            session_id=session_id,
            head_seq=head_seq,
            chain_digest=store._chain_anchor(session_id, head_seq),
        )
        deletion_state = row[7]
        if deletion_state not in {"live", "forgotten"}:
            raise SessionNotLive("session is not stable for export")
        tombstone = store._tombstone(session_id)
        if deletion_state == "forgotten":
            if tombstone is None or tombstone.state is not TombstoneState.COMPLETE:
                raise InvalidExport("forgotten export requires a complete tombstone")
        elif tombstone is not None:
            raise InvalidExport("live export must not have a deletion tombstone")

        configured, allow_unsigned, configured_signer = store._input_admission_policy()
        checkpoint_evidence, admission_evidence = store._one(
            "SELECT EXISTS(SELECT 1 FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s),"
            "EXISTS(SELECT 1 FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND "
            "(admission_key_id IS NOT NULL OR admission_signature IS NOT NULL OR admission_signed_ms IS NOT NULL))",
            (tenant_id, session_id, tenant_id, session_id),
        )
        stored_proof = bool(checkpoint_evidence or admission_evidence)
        if configured and allow_unsigned and stored_proof:
            raise InvalidExport("signed ledger evidence cannot be exported in explicit hash-only mode")
        if configured and allow_unsigned:
            if signer is not None:
                raise InvalidExport("explicit hash-only export cannot use a ledger verifier")
            signed = False
            verifier = None
        else:
            signed = bool(configured or signer is not None or stored_proof)
            verifier = signer if signer is not None else configured_signer if signed else None
            if signed and verifier is None:
                raise LedgerIntegrityError("ledger verifier is required to verify signed export proofs")

        stored_rows, chains = store._verified_records(session_id, head_seq, verifier=verifier, signed=signed)
        if len(stored_rows) != head_seq or len(chains) != head_seq or head_seq < 1:
            raise InvalidExport("export requires a complete non-empty session ledger")

        inbox_rows = store._rows(
            "SELECT body,input_seq,received_ms,consumed_seq,digest,admission_key_id,"
            "admission_signature,admission_signed_ms FROM sk_inbox "
            "WHERE tenant_id=%s AND session_id=%s ORDER BY input_seq",
            (tenant_id, session_id),
        )
        inboxes = tuple(store._stored_input(session_id, row, verifier=verifier, signed=signed) for row in inbox_rows)
        if len(inboxes) != inbox_seq:
            raise InvalidExport("export inbox snapshot does not match the captured head")

        checkpoints = (
            store._rows(
                "SELECT seq,chain_digest,key_id,signature,signed_ms FROM sk_checkpoint "
                "WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
                (tenant_id, session_id),
            )
            if signed
            else []
        )
        if signed and len(checkpoints) != head_seq:
            raise InvalidExport(f"signed export checkpoint is missing at seq {len(checkpoints) + 1}")

        stored_records = tuple(store._stored(session_id, row) for row in stored_rows)
        chain_digest = chains[-1]
        record_lines = tuple(
            ExportLine(
                ExportLineKind.RECORD,
                {
                    "line": "record",
                    "storage_envelope": {
                        "schema_version": "S1",
                        "tenant_id": tenant_id,
                        "session_id": session_id,
                        "seq": stored.seq,
                        "record_digest": stored.record.digest,
                        "chain_digest": stored_chain,
                        "writer_epoch": stored.writer_epoch,
                        "created_ms": stored.created_ms,
                    },
                    "record": stored.record.to_dict(),
                },
            )
            for stored, stored_chain in zip(stored_records, chains, strict=True)
        )
        inbox_lines = tuple(
            ExportLine(
                ExportLineKind.INBOX,
                {
                    "line": "inbox",
                    "input_seq": stored_input.input_seq,
                    "received_ms": stored_input.received_ms,
                    "consumed_seq": stored_input.consumed_seq,
                    "input": stored_input.item.to_dict(),
                },
            )
            for stored_input in inboxes
        )
        content_lines: tuple[ExportLine, ...] = ()
        if mode is ExportContentMode.INCLUDE_CONTENT:
            refs = _content_export_refs(stored_records, inboxes)
            if refs:
                if content_store is None:
                    from .content_runtime import durable_content_store

                    content_store = durable_content_store(store)
                export_content = getattr(content_store, "export_content_lines", None)
                if not callable(export_content):
                    raise TypeError("include-content export requires a durable content store")
                content_connection = getattr(getattr(content_store, "_backend", None), "connection", None)
                store_connection = getattr(store, "connection", None)
                if content_connection is not None and store_connection is not None and content_connection is not store_connection:
                    raise TypeError("include-content content store must share the export SQL transaction")
                content_lines = tuple(export_content(tenant_id, session_id, refs))
        checkpoint_lines = tuple(
            ExportLine(
                ExportLineKind.CHECKPOINT,
                {
                    "line": "checkpoint",
                    "seq": seq,
                    "chain_digest": stored_chain,
                    "key_id": key_id,
                    "signature": base64.b64encode(bytes(signature)).decode("ascii"),
                    "signed_ms": signed_ms,
                },
            )
            for seq, stored_chain, key_id, signature, signed_ms in checkpoints
        )
        manifest = ExportLine(
            ExportLineKind.MANIFEST,
            {
                "line": "manifest",
                "record_lines": len(record_lines),
                "inbox_lines": len(inbox_lines),
                "checkpoint_lines": len(checkpoint_lines),
                "head_seq": head_seq,
                "chain_digest": chain_digest,
            },
        )
        document = ExportDocument(
            ExportHeader(
                tenant_id=tenant_id,
                session_id=session_id,
                head_seq=head_seq,
                chain_digest=chain_digest,
                content_mode=mode,
            ),
            (*record_lines, *inbox_lines, *content_lines, *checkpoint_lines, manifest),
        )

        record_key_id = checkpoints[-1][2] if checkpoints else getattr(verifier, "key_id", "hash-only")
        if not isinstance(record_key_id, str) or not record_key_id:
            record_key_id = "hash-only"
        record_head = SessionHead(tenant_id, session_id, head_seq, chain_digest, record_key_id)
        document.verify(record_head, verifier, expected_head=expected_head)
        if tombstone is not None:
            tombstone_head = SessionHead(tenant_id, session_id, head_seq, chain_digest, tombstone.key_id)
            tombstone.verify(tombstone_head, verifier)
            store._verify_tombstone_object(tombstone, verifier, allow_unsigned=not signed)
        return document


__all__ = ["export_session"]
