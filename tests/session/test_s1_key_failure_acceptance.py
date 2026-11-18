"""S1-A5 key and signer outage acceptance through real storage paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from support.s1_fixture_values import append_fixture_records, create_fixture_session, push_fixture_input, wrap_fixture_store

from vv_agent.kernel._content import (
    ContentRef,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    Principal,
    TombstoneState,
)
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._sql import SessionNotLive
from vv_agent.session.content_runtime import INPUT_CONTENT, SessionContentWriter, durable_content_store
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore

from .helpers import record

TENANT = "default"
MASTER_KEY = b"a" * 32
SESSION_TABLES = (
    "sk_session",
    "sk_session_key",
    "sk_content",
    "sk_content_link",
    "sk_record",
    "sk_inbox",
    "sk_commit",
    "sk_checkpoint",
    "sk_tombstone",
)


class _Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "s1-a5-ledger-key"
        self.fail = False

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                if signer.fail:
                    raise RuntimeError("signer unavailable")
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


def _counts(store: Any, session_id: str) -> dict[str, int]:
    return {
        table: store._one(
            f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
            (TENANT, session_id),
        )[0]
        for table in SESSION_TABLES
    }


def _head_and_inbox(store: Any, session_id: str) -> tuple[tuple[Any, ...], str]:
    head = store._one(
        "SELECT head_seq,inbox_seq,active_turn_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (TENANT, session_id),
    )
    return head, store._chain_anchor(session_id, head[0])


def _assert_plaintext_absent(store: Any, marker: str) -> None:
    marker_bytes = marker.encode()
    for table in SESSION_TABLES:
        for row in store._rows(f"SELECT * FROM {table} WHERE tenant_id=%s", (TENANT,)):
            for value in row:
                if isinstance(value, bytes | bytearray | memoryview):
                    assert marker_bytes not in bytes(value)
                elif isinstance(value, str):
                    assert marker not in value


def _provider(tmp_path: Path, name: str) -> LocalKeyProvider:
    return LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"{name}.sqlite3")


@pytest.mark.parametrize("operation", ["create", "push", "append"])
def test_signer_failure_rolls_back_writes_without_partial_admission(
    store,
    tmp_path: Path,
    operation: str,
) -> None:
    signer = _Signer()
    store.configure_input_admission(signer)
    key_provider = _provider(tmp_path, operation)
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = f"s1-a5-{operation}"
    marker = f"s1-a5-{operation}-plaintext"
    try:
        if operation == "create":
            signer.fail = True
            with pytest.raises(RuntimeError, match="signer unavailable"):
                create_fixture_session(
                    wrapped,
                    session_id,
                    workspace=marker,
                    attributes={"seed": {"messages": [], "shared_state": {}}, "marker": marker},
                )

            assert set(_counts(store, session_id).values()) == {0}
            _assert_plaintext_absent(store, marker)
            return

        create_fixture_session(wrapped, session_id)
        before_counts = _counts(store, session_id)
        before_head = _head_and_inbox(store, session_id)

        signer.fail = True
        with pytest.raises(RuntimeError, match="signer unavailable"):
            if operation == "push":
                push_fixture_input(
                    wrapped,
                    session_id,
                    InboxItem("s1-a5-input", "user", {"content": marker}),
                )
            else:
                append_fixture_records(
                    wrapped,
                    session_id,
                    (
                        record(
                            "turn_started",
                            sid=session_id,
                            tid=f"{session_id}/turn",
                            handler_version=marker,
                        ),
                    ),
                    commit_id="s1-a5-append",
                )

        signer.fail = False
        assert _counts(store, session_id) == before_counts
        assert _head_and_inbox(store, session_id) == before_head
        assert store.read_input(session_id, "s1-a5-input") is None
        assert store.read(session_id).head_seq == before_head[0][0]
        _assert_plaintext_absent(store, marker)
    finally:
        key_provider.close()


@pytest.mark.parametrize("operation", ["put", "get"])
def test_content_key_unavailable_has_no_plaintext_or_dangling_commit(
    store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    key_provider = _provider(tmp_path, f"content-{operation}")
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = f"s1-a5-content-{operation}"
    marker = f"s1-a5-content-{operation}-plaintext"
    try:
        create_fixture_session(wrapped, session_id)
        ref: ContentRef | None = None
        if operation == "get":
            push_fixture_input(
                wrapped,
                session_id,
                InboxItem("s1-a5-content-input", "user", {"content": marker}),
            )
            stored = store.read_input(session_id, "s1-a5-content-input")
            assert stored is not None
            ref = ContentRef.from_dict(stored.item.payload["content"])

        before_counts = _counts(store, session_id)
        before_head = _head_and_inbox(store, session_id)

        def unavailable(_tenant_id: str, _session_id: str, _wrapped_key: Any) -> bytes:
            raise KeyUnavailable("temporary key-provider outage")

        monkeypatch.setattr(key_provider, "unwrap", unavailable)
        with pytest.raises(KeyUnavailable, match="temporary key-provider outage"), wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
            if operation == "put":
                writer.put_json({"secret": marker})
            else:
                assert ref is not None
                verified = writer.assert_input("s1-a5-content-input", ref, INPUT_CONTENT)
                writer.get_json(verified)

        assert _counts(store, session_id) == before_counts
        assert _head_and_inbox(store, session_id) == before_head
        _assert_plaintext_absent(store, marker)

        monkeypatch.undo()
        if operation == "get":
            assert ref is not None
            with wrapped.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
                verified = writer.assert_input("s1-a5-content-input", ref, INPUT_CONTENT)
                assert writer.get_json(verified) == marker
    finally:
        key_provider.close()


def test_destroy_failure_stays_fenced_and_retry_erases_content(
    store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = _Signer()
    store.configure_input_admission(signer)
    key_provider = _provider(tmp_path, "destroy")
    wrapped = wrap_fixture_store(store, key_provider)
    session_id = "s1-a5-destroy"
    marker = "s1-a5-destroy-plaintext"
    original_destroy = key_provider.destroy
    try:
        create_fixture_session(wrapped, session_id)
        push_fixture_input(
            wrapped,
            session_id,
            InboxItem("s1-a5-destroy-input", "user", {"content": marker}),
        )
        stored = store.read_input(session_id, "s1-a5-destroy-input")
        assert stored is not None
        ref = ContentRef.from_dict(stored.item.payload["content"])
        created = store.read(session_id, limit=1).records[0].record
        actor = Principal.from_dict(created._payload["principal"])
        adapter = durable_content_store(wrapped)
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        monkeypatch.setattr(
            key_provider,
            "destroy",
            lambda _tenant_id, _key_id: KeyDestroyOutcome(
                False,
                KeyDestroyStatus.KEY_UNAVAILABLE,
                None,
                "temporary key-provider outage",
            ),
        )
        with pytest.raises(KeyUnavailable, match="temporary key-provider outage"):
            deletion.forget_session(TENANT, session_id, actor=actor)

        state, tombstone_state = store._one(
            "SELECT s.deletion_state,t.state FROM sk_session s "
            "JOIN sk_tombstone t ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id "
            "WHERE s.tenant_id=%s AND s.session_id=%s",
            (TENANT, session_id),
        )
        assert (state, tombstone_state) == ("forgetting", "pending")
        with pytest.raises(SessionNotLive):
            store.read(session_id)
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, session_id, ref)
        _assert_plaintext_absent(store, marker)

        monkeypatch.setattr(key_provider, "destroy", original_destroy)
        receipt = deletion.forget_session(TENANT, session_id, actor=actor)
        assert receipt.session_ids == (session_id,)
        skeleton = deletion.read_skeleton(TENANT, session_id)
        assert skeleton.tombstone is not None
        assert skeleton.tombstone.state is TombstoneState.COMPLETE
        assert _counts(store, session_id)["sk_content"] == 0
        assert _counts(store, session_id)["sk_content_link"] == 0
        assert _counts(store, session_id)["sk_session_key"] == 0
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, session_id, ref)
        _assert_plaintext_absent(store, marker)
    finally:
        key_provider.close()
