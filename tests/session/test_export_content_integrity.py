from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentIntegrityError,
    ContentUnavailable,
    ExportContentMode,
    ExportDocument,
    ExportLineKind,
    InvalidExport,
    KeyUnavailable,
    SessionHead,
)
from vv_agent.kernel._records import digest
from vv_agent.session.content_runtime import SessionContentWriter, durable_content_store
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.export import export_session
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import record

MASTER_KEY = b"x" * 32
SESSION = "export-content-integrity"


def _create_nested_session(store, provider: LocalKeyProvider) -> dict[str, object]:
    store.configure_input_admission(allow_unsigned=True)
    driver = SessionDriver(store=store, content_key_provider=provider)
    attributes = {
        "seed": {"messages": [{"role": "user", "content": "nested-export-marker"}]},
        "child_admission": {
            "definition": {
                "task": {
                    "initial_messages": [{"role": "user", "content": "child-initial-marker"}],
                    "metadata": {"vv_session": {"input_messages": [{"role": "user", "content": "child-input-marker"}]}},
                }
            }
        },
    }
    driver.create(SESSION, "/workspace", attributes)
    return attributes


def _head(document: ExportDocument) -> SessionHead:
    return SessionHead(
        document.header.tenant_id,
        document.header.session_id,
        document.header.head_seq,
        document.header.chain_digest,
        "hash-only",
    )


def _verify_lines(document: ExportDocument, lines: list[dict[str, object]]) -> None:
    ExportDocument.from_dict(document.header.to_dict(), lines).verify(_head(document))


def _content_by_slot(document: ExportDocument) -> dict[str, dict[str, object]]:
    return {line.value["slot"]: line.value for line in document.lines if line.kind is ExportLineKind.CONTENT}


@pytest.mark.persistent_store
def test_include_content_exports_nested_message_slots_on_each_sql_backend(store, tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "nested-export-keys.sqlite3")
    attributes = _create_nested_session(store, provider)
    delegate = durable_content_store(store)

    class TransactionCheckingContent:
        def export_content_lines(self, *args):
            if hasattr(store, "_depth"):
                assert store.connection.in_transaction
                assert store._depth == 1
            else:
                from psycopg.pq import TransactionStatus

                assert store.connection.info.transaction_status == TransactionStatus.INTRANS
            return delegate.export_content_lines(*args)

    document = export_session(
        store,
        SESSION,
        content_mode=ExportContentMode.INCLUDE_CONTENT,
        content_store=TransactionCheckingContent(),
    )

    content = _content_by_slot(document)
    nested_slots = {
        "session.attributes.seed.messages[0]",
        "session.attributes.child_admission.definition.task.initial_messages[0]",
        "session.attributes.child_admission.definition.task.metadata.vv_session.input_messages[0]",
    }
    assert nested_slots <= set(content)
    assert content["session.attributes.seed.messages[0]"]["decoded_value"] == attributes["seed"]["messages"][0]
    outer = content["session.attributes"]["decoded_value"]
    seed_ref = outer["seed"]["messages"][0]
    assert seed_ref["content_id"] == content["session.attributes.seed.messages[0]"]["content_id"]
    document.verify(_head(document))


def test_include_content_exports_request_message_slots_from_a_real_record(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "request-export-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "request-export.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        driver = SessionDriver(store=store, content_key_provider=provider)
        driver.create(SESSION, "/workspace", {})
        lease = store.acquire(SESSION, owner="request-export", ttl_ms=10_000)
        assert lease is not None

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            messages = [
                {"role": "user", "content": "multi-turn-one"},
                {"role": "assistant", "content": "multi-turn-two"},
            ]
            definition_ref = writer.put_json({})
            budget_ref = writer.put_json({})
            budget_admission_ref = writer.put_json({})
            message_refs = [writer.put_json(message) for message in messages]
            request_ref = writer.put_json({"messages": [ref.to_dict() for ref in message_refs]})
            turn_started = record(
                "turn_started",
                sid=SESSION,
                tid="request-turn",
                definition=definition_ref.to_dict(),
                definition_digest=digest(definition_ref.to_dict()),
                budget=budget_ref.to_dict(),
            )
            planned = record(
                "op_planned",
                sid=SESSION,
                tid="request-turn",
                oid="request-operation",
                request=request_ref.to_dict(),
                request_digest=digest(request_ref.to_dict()),
                budget_admission=budget_admission_ref.to_dict(),
            )
            receipt = tx.append(
                SESSION,
                lease=lease,
                expected_seq=1,
                commit_id="request-messages",
                records=(turn_started, planned),
            )
            sequences = dict(receipt.record_sequences)
            turn_seq = sequences[turn_started.record_id]
            writer.link_record(turn_seq, definition_ref, "definition")
            writer.link_record(turn_seq, budget_ref, "budget")
            record_seq = sequences[planned.record_id]
            writer.link_record(record_seq, request_ref, "request")
            writer.link_record(record_seq, budget_admission_ref, "budget_admission")
            for index, ref in enumerate(message_refs):
                writer.link_record(record_seq, ref, f"request.messages[{index}]")
        assert store.release(lease)

        document = export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)

        content = _content_by_slot(document)
        assert content["request.messages[0]"]["decoded_value"] == messages[0]
        assert content["request.messages[1]"]["decoded_value"] == messages[1]
        request_value = content["request"]["decoded_value"]
        assert [message["content_id"] for message in request_value["messages"]] == [
            message_refs[0].content_id,
            message_refs[1].content_id,
        ]
        document.verify(_head(document))


def test_export_verify_rejects_missing_or_tampered_nested_content(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "nested-tamper-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "nested-tamper.sqlite")) as store:
        store.install_schema()
        _create_nested_session(store, provider)
        document = export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)
        exported = document.to_dict()["lines"]
        nested_slot = "session.attributes.seed.messages[0]"

        missing = [deepcopy(line) for line in exported if line["line"] != "content" or line["slot"] != nested_slot]
        with pytest.raises(InvalidExport, match="does not cover every referenced"):
            _verify_lines(document, missing)

        wrong_key = deepcopy(exported)
        next(line for line in wrong_key if line["line"] == "content" and line["slot"] == nested_slot)["key_id"] = "wrong-key"
        with pytest.raises(ContentBindingMismatch, match="referenced content key"):
            _verify_lines(document, wrong_key)

        wrong_size = deepcopy(exported)
        next(line for line in wrong_size if line["line"] == "content" and line["slot"] == nested_slot)["decoded_value"] = {
            "role": "user",
            "content": "different-size-marker",
        }
        with pytest.raises(ContentIntegrityError, match="size"):
            _verify_lines(document, wrong_size)

        noncanonical = deepcopy(exported)
        next(line for line in noncanonical if line["line"] == "content")["decoded_value"] = [float("nan")]
        with pytest.raises(InvalidExport, match="canonical JSON"):
            ExportDocument.from_dict(document.header.to_dict(), noncanonical)


def test_include_content_rejects_nested_ref_without_its_owner_slot_link(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "nested-link-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "nested-link.sqlite")) as store:
        store.install_schema()
        _create_nested_session(store, provider)
        store._rows(
            "UPDATE sk_content_link SET slot=%s WHERE tenant_id=%s AND session_id=%s AND record_seq=1 AND slot=%s",
            (
                "session.attributes.seed.messages[9]",
                "default",
                SESSION,
                "session.attributes.seed.messages[0]",
            ),
        )

        with pytest.raises(ContentUnavailable, match="content link is unavailable"):
            export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)


def test_include_content_hydrates_inside_the_ledger_transaction_and_fails_closed(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "snapshot-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "snapshot.sqlite")) as store:
        store.install_schema()
        _create_nested_session(store, provider)

        class TransactionCheckingContent:
            def __init__(self) -> None:
                self.called = False
                self._store = SQLiteContentStore(store, provider)

            def export_content_lines(self, *args):
                self.called = True
                assert store.connection.in_transaction
                assert store._depth == 1
                return self._store.export_content_lines(*args)

        content_store = TransactionCheckingContent()
        export_session(
            store,
            SESSION,
            content_mode=ExportContentMode.INCLUDE_CONTENT,
            content_store=content_store,
        )
        assert content_store.called

        store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=1 "
            "AND slot='session.attributes.seed.messages[0]'",
            ("default", SESSION),
        )
        with pytest.raises(ContentUnavailable, match="content link is unavailable"):
            export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)


def test_include_content_with_unavailable_key_returns_no_document(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "missing-key-state.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "missing-key.sqlite")) as store:
        store.install_schema()
        _create_nested_session(store, provider)
        key_id = store._one(
            "SELECT key_id FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            ("default", SESSION),
        )[0]
        assert provider.destroy("default", key_id).status.value == "destroyed"

        with pytest.raises(KeyUnavailable):
            export_session(store, SESSION, content_mode=ExportContentMode.INCLUDE_CONTENT)


def test_skeleton_export_does_not_touch_the_content_store(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "skeleton-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "skeleton.sqlite")) as store:
        store.install_schema()
        _create_nested_session(store, provider)

        class ExplodingContentStore:
            def export_content_lines(self, *_args):
                raise AssertionError("skeleton export touched the content store")

        document = export_session(store, SESSION, content_store=ExplodingContentStore())
        assert not any(line.kind is ExportLineKind.CONTENT for line in document.lines)
