"""Stage 2 content read-frame behavior on real SQLite and PostgreSQL stores."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.sim.content import SimContentKeyProvider
from vv_agent.kernel import _content_sql
from vv_agent.kernel._boundary_hydration import (
    SessionHydration,
    hydrate_session_input,
    hydrate_session_records,
)
from vv_agent.kernel._content import (
    Classification,
    ContentLabels,
    ContentRef,
    ContentStoreError,
    ContentUnavailable,
    KeyDestroyStatus,
    Principal,
    WrappedDataKey,
)
from vv_agent.kernel._content import KeyUnavailable as BackendKeyUnavailable
from vv_agent.kernel._content_reader import DurableContentReader, input_owner, record_owner
from vv_agent.kernel._content_runtime import (
    _ValidatedContentFrame,
    _ValidatedDurableContentStore,
    wrap_session_content_store,
)
from vv_agent.kernel._content_sql import CoreSQLContentStore, _ContentReadFrame
from vv_agent.kernel._hydration import IntegrityFailure, KeyUnavailable, MissingContentLink, TypedSchemaFailure
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    bind_content_runtime,
    durable_content_reader,
    durable_content_store,
)
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.records import InboxItem, digest
from vv_agent.session.sqlite import SQLiteStore

from .helpers import append_fixture_records, create_fixture_session, push_fixture_input
from .helpers import record as fixture_record

MASTER_KEY = b"f" * 32
SESSION = "frame"
PROVIDER_RESULT_PAYLOAD = {
    "operation_id": "provider-operation",
    "attempt": 1,
    "request_digest": "a" * 64,
    "provider_binding": "provider",
    "result": {"items": [1]},
    "usage": {"tokens": 1},
    "evidence": {"source": "test"},
}


def test_sim_key_provider_registers_only_authenticated_bindings_for_destroy() -> None:
    owner = SimContentKeyProvider(17)
    wrapped = owner.generate_data_key("tenant-a", "session-a")

    restarted = SimContentKeyProvider(17)
    assert restarted.unwrap("tenant-a", "session-a", wrapped)
    outcome = restarted.destroy("tenant-a", wrapped.key_id)
    assert outcome.accepted
    with pytest.raises(BackendKeyUnavailable):
        restarted.unwrap("tenant-a", "session-a", wrapped)


def test_sim_key_provider_rejects_unauthenticated_restore_without_registering_destroy_binding() -> None:
    provider = SimContentKeyProvider(23)
    malformed = WrappedDataKey("tenant-a", "session-a", "forged", b"not-the-authenticated-envelope")

    with pytest.raises(BackendKeyUnavailable, match="envelope"):
        provider.restore_data_key(malformed)
    assert provider.destroy("tenant-a", malformed.key_id).status is KeyDestroyStatus.KEY_UNAVAILABLE


def test_sim_key_provider_rejects_binding_replacement_and_wrong_scope_without_registration() -> None:
    provider = SimContentKeyProvider(29)
    wrapped = provider.generate_data_key("tenant-a", "session-a")
    conflict = WrappedDataKey(
        "tenant-b",
        "session-a",
        wrapped.key_id,
        provider._envelope("tenant-b", "session-a", wrapped.key_id),
    )

    with pytest.raises(BackendKeyUnavailable, match="conflicts"):
        provider.restore_data_key(conflict)
    with pytest.raises(BackendKeyUnavailable, match="binding"):
        SimContentKeyProvider(29).unwrap("tenant-b", "session-a", wrapped)

    assert provider.destroy("tenant-a", wrapped.key_id).accepted
    assert provider.destroy("tenant-b", wrapped.key_id).status is KeyDestroyStatus.BINDING_MISMATCH


class CountingProvider(LocalKeyProvider):
    def __init__(self, state_path: Path) -> None:
        self.unwrap_calls = 0
        self.restore_calls = 0
        self.snapshot_depth = 0
        super().__init__(MASTER_KEY, state_path=state_path)

    def unwrap(self, tenant_id, session_id, wrapped_key):
        assert self.snapshot_depth == 0, "KMS unwrap must run after the SQL snapshot exits"
        self.unwrap_calls += 1
        return super().unwrap(tenant_id, session_id, wrapped_key)

    def restore_data_key(self, wrapped_key):
        self.restore_calls += 1
        return super().restore_data_key(wrapped_key)


class CountingCipher:
    def __init__(self) -> None:
        self._inner = AESGCMCipher()
        self.decrypt_calls = 0

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return self._inner.encrypt(key, nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        self.decrypt_calls += 1
        return self._inner.decrypt(key, nonce, ciphertext, aad)

    def random_bytes(self, n: int) -> bytes:
        return self._inner.random_bytes(n)


@pytest.fixture
def content_setup(store, tmp_path: Path):
    provider = CountingProvider(tmp_path / "frame-keys.sqlite")
    cipher = CountingCipher()
    bind_content_runtime(store, key_provider=provider, cipher=cipher)
    wrapped = wrap_session_content_store(store)
    create_fixture_session(
        wrapped,
        SESSION,
        attributes={"seed": {"messages": [], "shared_state": {}, "unseen_input_protection": True}},
    )
    reader = durable_content_reader(wrapped, tenant_id=store.tenant_id, session_id=SESSION)
    yield store, wrapped, reader, provider, cipher
    provider.close()


def _link_values(store, values: dict[str, Any]) -> dict[str, ContentRef]:
    refs: dict[str, ContentRef] = {}
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=SESSION)
        for slot, value in values.items():
            ref = writer.put_json(value)
            writer.link_record(1, ref, slot)
            refs[slot] = ref
    return refs


def _delete_session_from_other_connection(
    *,
    path: Path,
    tenant_id: str,
    provider: CountingProvider,
    cipher: CountingCipher,
    ref: ContentRef,
    operation: str,
) -> None:
    with SQLiteStore.standalone(str(path)) as deleting_store:
        bind_content_runtime(deleting_store, key_provider=provider, cipher=cipher)
        durable = durable_content_store(deleting_store)
        assert isinstance(durable, _ValidatedDurableContentStore)
        deletion = SessionDeletionStore(deleting_store, durable._durable, allow_unsigned=True)
        actor = Principal(tenant_id, "fixture/deleter", "user", ref)
        if operation == "forget":
            deletion.forget_session(tenant_id, SESSION, actor=actor)
        else:
            deletion.purge_session(tenant_id, SESSION, actor=actor)


def _change_session_key_state_from_other_connection(
    *,
    path: Path,
    tenant_id: str,
    key_id: str,
) -> None:
    with SQLiteStore.standalone(str(path)) as mutating_store, mutating_store.atomic():
        mutating_store._rows(
            "UPDATE sk_session_key SET state='destroying' WHERE tenant_id=%s AND session_id=%s AND key_id=%s AND state='active'",
            (tenant_id, SESSION, key_id),
        )


def _change_content_identity_from_other_connection(
    *,
    path: Path,
    tenant_id: str,
    content_id: str,
) -> None:
    with SQLiteStore.standalone(str(path)) as mutating_store, mutating_store.atomic():
        row = mutating_store._one(
            "SELECT nonce,size FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (tenant_id, SESSION, content_id),
        )
        assert row is not None
        nonce = bytes(row[0])
        mutating_store._rows(
            "UPDATE sk_content SET nonce=%s,mac=%s,size=%s,ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (nonce[:-1] + bytes([nonce[-1] ^ 1]), "0" * 64, row[1] + 1, b"tampered-ciphertext", tenant_id, SESSION, content_id),
        )


def _track_snapshots(reader: DurableContentReader, provider: CountingProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(reader._store, _ValidatedDurableContentStore)
    assert isinstance(reader._store._durable, CoreSQLContentStore)
    backend = reader._store._durable._backend
    original = getattr(backend, "read_snapshot", None)
    assert callable(original)

    @contextmanager
    def tracked() -> Iterator[None]:
        provider.snapshot_depth += 1
        try:
            with original():
                yield
        finally:
            provider.snapshot_depth -= 1

    monkeypatch.setattr(backend, "read_snapshot", tracked)


def test_frame_deduplicates_key_metadata_aead_and_returns_mutable_isolation(
    content_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    refs = _link_values(
        store,
        {
            "request.messages[0]": {"items": [1], "body": "shared"},
            "request.messages[1]": {"items": [1], "body": "shared"},
            "request.messages[2]": {"items": [2], "body": "distinct"},
        },
    )
    assert refs["request.messages[0]"] == refs["request.messages[1]"]
    ref = refs["request.messages[0]"]
    _track_snapshots(reader, provider, monkeypatch)
    provider.unwrap_calls = provider.restore_calls = cipher.decrypt_calls = 0
    decoded_content_ids: list[str] = []
    key_decode_calls = 0
    original_content_decoder = _content_sql._content_from_row
    original_key_decoder = _content_sql._key_from_row

    def tracked_content_decoder(row):
        decoded_content_ids.append(row[0])
        return original_content_decoder(row)

    def tracked_key_decoder(tenant_id, session_id, row):
        nonlocal key_decode_calls
        key_decode_calls += 1
        return original_key_decoder(tenant_id, session_id, row)

    monkeypatch.setattr(_content_sql, "_content_from_row", tracked_content_decoder)
    monkeypatch.setattr(_content_sql, "_key_from_row", tracked_key_decoder)

    captured_frame: _ContentReadFrame | None = None
    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        assert isinstance(frame_reader._store, _ValidatedContentFrame)
        assert isinstance(frame_reader._store._frame, _ContentReadFrame)
        captured_frame = frame_reader._store._frame
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        first = hydration.read(record_owner(1), "request.messages[0]", ref)
        second = hydration.read(record_owner(1), "request.messages[1]", ref)
        third = hydration.read(record_owner(1), "request.messages[2]", refs["request.messages[2]"])
        first["items"].append(2)
        assert second == {"items": [1], "body": "shared"}
        assert third == {"items": [2], "body": "distinct"}

    assert provider.unwrap_calls == 1
    assert provider.restore_calls == 0
    assert cipher.decrypt_calls == 2
    assert decoded_content_ids == [ref.content_id, refs["request.messages[2]"].content_id]
    assert key_decode_calls == 1
    assert captured_frame is not None
    assert captured_frame._key_row is None
    assert captured_frame._session_key is None
    assert captured_frame._content == {}
    assert captured_frame._parsed_content == {}
    assert captured_frame._data_key is None
    assert captured_frame._plaintext == {}
    assert captured_frame._links == {}
    assert captured_frame._live_row is None


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("mac", IntegrityFailure),
        ("size", IntegrityFailure),
        ("labels", IntegrityFailure),
        ("key_id", MissingContentLink),
    ],
)
def test_frame_rechecks_cached_content_identity(
    content_setup,
    mutation: str,
    expected_error: type[Exception],
) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "warm"}})["request.messages[0]"]
    if mutation == "mac":
        wrong = ContentRef(ref.content_id, "0" * 64, ref.size, ref.key_id, ref.labels)
    elif mutation == "size":
        wrong = ContentRef(ref.content_id, ref.mac, ref.size + 1, ref.key_id, ref.labels)
    elif mutation == "labels":
        wrong = ContentRef(
            ref.content_id,
            ref.mac,
            ref.size,
            ref.key_id,
            ContentLabels(classification=Classification.PUBLIC),
        )
    else:
        wrong = ContentRef(ref.content_id, ref.mac, ref.size, "other-key", ref.labels)

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        assert hydration.read(record_owner(1), "request.messages[0]", ref) == {"body": "warm"}
        with pytest.raises(expected_error):
            hydration.read(record_owner(1), "request.messages[0]", wrong)


def test_frame_enforces_owner_slot_and_tenant_bindings(content_setup) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "owned"}})["request.messages[0]"]
    push_fixture_input(store, SESSION, InboxItem("input", "user", {"content": "input-body"}))
    stored_input = store.read_input(SESSION, "input")
    assert stored_input is not None
    input_ref = ContentRef.from_dict(stored_input.item.payload["content"])

    with reader._read_frame(record_seqs=(1,), input_ids=("input",)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        assert hydration.read(record_owner(1), "request.messages[0]", ref) == {"body": "owned"}
        assert hydration.read(input_owner("input"), "input.content", input_ref) == "input-body"
        with pytest.raises(MissingContentLink):
            hydration.read(record_owner(1), "request.messages[1]", ref)
        with pytest.raises(MissingContentLink):
            hydration.read(record_owner(2), "request.messages[0]", ref)
        with pytest.raises(MissingContentLink):
            hydration.read(input_owner("input"), "input.content", ref)
        with pytest.raises(ValueError, match="stable declared slot"):
            hydration.read(record_owner(1), "input.content", ref)
        with pytest.raises(MissingContentLink):
            frame_reader.read("other-tenant", SESSION, record_owner(1), "request.messages[0]", ref)


def test_frame_hydrates_nested_refs_inside_the_same_owner(content_setup) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=SESSION)
        nested_ref = writer.put_json({"role": "user", "content": "nested"})
        outer_ref = writer.put_json({"messages": [nested_ref.to_dict()]})
        writer.link_record(1, outer_ref, "request")
        writer.link_record(1, nested_ref, "request.messages[0]")

    payload = {"request": outer_ref.to_dict()}
    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        assert hydration.read(
            record_owner(1),
            "request",
            outer_ref,
            record_kind="op_planned",
            record_payload=payload,
        ) == {"messages": [{"role": "user", "content": "nested"}]}

    with store.atomic():
        store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=1 AND slot='request.messages[0]'",
            (store.tenant_id, SESSION),
        )
    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        with pytest.raises(MissingContentLink):
            hydration.read(
                record_owner(1),
                "request",
                outer_ref,
                record_kind="op_planned",
                record_payload=payload,
            )


@pytest.mark.parametrize("outer_value", [None, []], ids=["null", "array"])
def test_frame_rejects_non_object_nested_outer_as_typed_failure(content_setup, outer_value: Any) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request": outer_value})["request"]
    payload = {"request": ref.to_dict()}

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        with pytest.raises(TypedSchemaFailure, match="request must be a JSON object"):
            hydration.read(
                record_owner(1),
                "request",
                ref,
                record_kind="op_planned",
                record_payload=payload,
            )


def test_frame_rejects_malformed_nested_ref_as_typed_failure(content_setup) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=SESSION)
        nested_ref = writer.put_json({"role": "user"})
        malformed = nested_ref.to_dict() | {"content_id": ""}
        outer_ref = writer.put_json({"messages": [malformed]})
        writer.link_record(1, outer_ref, "request")
    payload = {"request": outer_ref.to_dict()}

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        with pytest.raises(TypedSchemaFailure, match="nested content inventory is invalid"):
            hydration.read(
                record_owner(1),
                "request",
                outer_ref,
                record_kind="op_planned",
                record_payload=payload,
            )


def test_frame_preserves_opaque_json_without_nested_inventory(content_setup) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"tool": [1, 2, 3]})["tool"]
    payload = {"tool": ref.to_dict()}

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        assert hydration.read(
            record_owner(1),
            "tool",
            ref,
            record_kind="op_planned",
            record_payload=payload,
        ) == [1, 2, 3]


@pytest.mark.parametrize("mutation", ["ciphertext", "labels", "nonce", "mac", "size"])
def test_frame_rejects_target_tamper_without_rejecting_unaccessed_rows(content_setup, mutation: str) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    refs = _link_values(
        store,
        {
            "request.messages[0]": {"body": "good"},
            "request.messages[1]": {"body": "bad"},
        },
    )
    bad = refs["request.messages[1]"]
    with store.atomic():
        if mutation == "ciphertext":
            row = store._one(
                "SELECT ciphertext FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (store.tenant_id, SESSION, bad.content_id),
            )
            ciphertext = bytes(row[0])
            store._rows(
                "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]), store.tenant_id, SESSION, bad.content_id),
            )
        elif mutation == "labels":
            store._rows(
                "UPDATE sk_content SET labels=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (b"{}", store.tenant_id, SESSION, bad.content_id),
            )
        elif mutation == "nonce":
            row = store._one(
                "SELECT nonce FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (store.tenant_id, SESSION, bad.content_id),
            )
            nonce = bytes(row[0])
            store._rows(
                "UPDATE sk_content SET nonce=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (nonce[:-1] + bytes([nonce[-1] ^ 1]), store.tenant_id, SESSION, bad.content_id),
            )
        elif mutation == "mac":
            store._rows(
                "UPDATE sk_content SET mac=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                ("0" * 64, store.tenant_id, SESSION, bad.content_id),
            )
        else:
            store._rows(
                "UPDATE sk_content SET size=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (bad.size + 1, store.tenant_id, SESSION, bad.content_id),
            )

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        assert hydration.read(record_owner(1), "request.messages[0]", refs["request.messages[0]"]) == {"body": "good"}
        with pytest.raises(IntegrityFailure):
            hydration.read(record_owner(1), "request.messages[1]", bad)


def test_frame_rejects_wrong_key_and_detects_key_destroy_between_independent_calls(content_setup) -> None:
    store, _wrapped, reader, provider, _cipher = content_setup
    refs = _link_values(
        store,
        {
            "request.messages[0]": {"body": "key-bound"},
            "request.messages[1]": {"body": "wrong-key"},
        },
    )
    ref = refs["request.messages[0]"]
    other = refs["request.messages[1]"]

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        assert json.loads(frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)) == {
            "body": "key-bound"
        }
    assert provider.destroy(store.tenant_id, ref.key_id).accepted
    with reader._read_frame(record_seqs=(1,)) as frame_reader, pytest.raises(KeyUnavailable):
        frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)

    with store.atomic():
        store._rows(
            "UPDATE sk_content SET key_id=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("wrong-key", store.tenant_id, SESSION, other.content_id),
        )
    with reader._read_frame(record_seqs=(1,)) as frame_reader, pytest.raises(MissingContentLink):
        frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[1]", other)


@pytest.mark.sqlite_file
@pytest.mark.parametrize("deletion_operation", ("forget", "purge"))
def test_frame_fails_closed_when_another_connection_deletes_cached_session(
    content_setup,
    tmp_path: Path,
    deletion_operation: str,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "forget-me"}})["request.messages[0]"]

    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        assert isinstance(frame_reader._store, _ValidatedContentFrame)
        frame = frame_reader._store._frame
        assert (
            frame.assert_link(
                store.tenant_id,
                SESSION,
                ref,
                "request.messages[0]",
                record_seq=1,
            )
            == ref
        )
        assert json.loads(frame.get(store.tenant_id, SESSION, ref)) == {"body": "forget-me"}
        with pytest.raises(MissingContentLink):
            frame_reader.read(store.tenant_id, SESSION, record_owner(2), "request.messages[0]", ref)
        with pytest.raises(MissingContentLink):
            frame_reader.read("other-tenant", SESSION, record_owner(1), "request.messages[0]", ref)
        assert frame.get(store.tenant_id, SESSION, ref) == b'{"body":"forget-me"}'

        _delete_session_from_other_connection(
            path=tmp_path / "session.sqlite",
            tenant_id=store.tenant_id,
            provider=provider,
            cipher=cipher,
            ref=ref,
            operation=deletion_operation,
        )

        with pytest.raises(ContentUnavailable):
            frame.assert_link(
                store.tenant_id,
                SESSION,
                ref,
                "request.messages[0]",
                record_seq=1,
            )
        with pytest.raises(ContentUnavailable):
            frame.get(store.tenant_id, SESSION, ref)
        with pytest.raises(MissingContentLink):
            frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)

    with reader._read_frame(record_seqs=(1,)) as frame_reader, pytest.raises(MissingContentLink):
        frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)


@pytest.mark.sqlite_file
def test_get_rechecks_lifecycle_after_decrypt_before_caching_plaintext(
    content_setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "raced"}})["request.messages[0]"]
    original_decrypt = cipher.decrypt
    phases: list[str] = []

    def decrypt_then_delete(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        assert provider.snapshot_depth == 0, "SQL snapshots must not cover decryption"
        phases.append("decrypt")
        plaintext = original_decrypt(key, nonce, ciphertext, aad)
        _delete_session_from_other_connection(
            path=tmp_path / "session.sqlite",
            tenant_id=store.tenant_id,
            provider=provider,
            cipher=cipher,
            ref=ref,
            operation="forget",
        )
        phases.append("deleted")
        return plaintext

    monkeypatch.setattr(cipher, "decrypt", decrypt_then_delete)
    _track_snapshots(reader, provider, monkeypatch)
    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        assert isinstance(frame_reader._store, _ValidatedContentFrame)
        frame = frame_reader._store._frame
        original_lifecycle = frame._require_current_lifecycle

        def tracked_lifecycle() -> None:
            phases.append(f"lifecycle:{provider.snapshot_depth}")
            original_lifecycle()

        monkeypatch.setattr(frame, "_require_current_lifecycle", tracked_lifecycle)
        with pytest.raises(ContentUnavailable):
            frame.get(store.tenant_id, SESSION, ref)
        assert phases == ["lifecycle:1", "decrypt", "deleted", "lifecycle:1"]
        assert frame._plaintext == {}
        assert frame._data_key is None
        with pytest.raises(ContentUnavailable):
            frame.get(store.tenant_id, SESSION, ref)


@pytest.mark.sqlite_file
def test_direct_get_reads_live_content_before_deletion(content_setup) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "direct-live"}})["request.messages[0]"]
    durable = reader._store._durable
    assert isinstance(durable, CoreSQLContentStore)

    assert durable.get(store.tenant_id, SESSION, ref) == b'{"body":"direct-live"}'


@pytest.mark.sqlite_file
@pytest.mark.parametrize("deletion_operation", ("forget", "purge"))
def test_direct_get_rejects_session_deleted_during_decrypt(
    content_setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deletion_operation: str,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "direct-race"}})["request.messages[0]"]
    durable = reader._store._durable
    assert isinstance(durable, CoreSQLContentStore)
    original_decrypt = cipher.decrypt

    def decrypt_then_delete(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        assert provider.snapshot_depth == 0, "SQL snapshots must not cover decryption"
        plaintext = original_decrypt(key, nonce, ciphertext, aad)
        _delete_session_from_other_connection(
            path=tmp_path / "session.sqlite",
            tenant_id=store.tenant_id,
            provider=provider,
            cipher=cipher,
            ref=ref,
            operation=deletion_operation,
        )
        return plaintext

    monkeypatch.setattr(cipher, "decrypt", decrypt_then_delete)
    _track_snapshots(reader, provider, monkeypatch)
    try:
        plaintext = durable.get(store.tenant_id, SESSION, ref)
    except ContentUnavailable:
        pass
    else:
        pytest.fail(f"direct get returned plaintext after {deletion_operation} during decrypt: {plaintext!r}")

    assert not hasattr(durable, "_plaintext")
    assert not hasattr(durable, "_data_key")
    with pytest.raises(ContentUnavailable):
        durable.get(store.tenant_id, SESSION, ref)


@pytest.mark.sqlite_file
def test_direct_get_rejects_key_destroy_during_decrypt(
    content_setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "direct-key"}})["request.messages[0]"]
    durable = reader._store._durable
    assert isinstance(durable, CoreSQLContentStore)
    original_decrypt = cipher.decrypt

    def decrypt_then_destroy_key(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        assert provider.snapshot_depth == 0, "SQL snapshots must not cover decryption"
        plaintext = original_decrypt(key, nonce, ciphertext, aad)
        assert provider.destroy(store.tenant_id, ref.key_id).accepted
        _change_session_key_state_from_other_connection(
            path=tmp_path / "session.sqlite",
            tenant_id=store.tenant_id,
            key_id=ref.key_id,
        )
        return plaintext

    monkeypatch.setattr(cipher, "decrypt", decrypt_then_destroy_key)
    _track_snapshots(reader, provider, monkeypatch)
    with pytest.raises(BackendKeyUnavailable):
        durable.get(store.tenant_id, SESSION, ref)


@pytest.mark.sqlite_file
def test_direct_get_rejects_content_change_during_decrypt(
    content_setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "direct-content"}})["request.messages[0]"]
    durable = reader._store._durable
    assert isinstance(durable, CoreSQLContentStore)
    original_decrypt = cipher.decrypt

    def decrypt_then_change_content(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        assert provider.snapshot_depth == 0, "SQL snapshots must not cover decryption"
        plaintext = original_decrypt(key, nonce, ciphertext, aad)
        _change_content_identity_from_other_connection(
            path=tmp_path / "session.sqlite",
            tenant_id=store.tenant_id,
            content_id=ref.content_id,
        )
        return plaintext

    monkeypatch.setattr(cipher, "decrypt", decrypt_then_change_content)
    _track_snapshots(reader, provider, monkeypatch)
    with pytest.raises(ContentStoreError):
        durable.get(store.tenant_id, SESSION, ref)


def test_frame_and_direct_read_agree_empty_ciphertext_is_integrity_failure(content_setup, store_backend: str) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "empty-cipher"}})["request.messages[0]"]
    with store.atomic():
        store._rows(
            "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (b"", store.tenant_id, SESSION, ref.content_id),
        )

    with pytest.raises(IntegrityFailure):
        reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)
    with reader._read_frame(record_seqs=(1,)) as frame_reader:
        hydration = SessionHydration(frame_reader, tenant_id=store.tenant_id, session_id=SESSION)
        with pytest.raises(IntegrityFailure):
            hydration.read(record_owner(1), "request.messages[0]", ref)


@pytest.mark.parametrize(
    ("kind", "payload", "include_actor_attributes", "decrypts"),
    [
        ("user", {"content": {"items": [1]}}, False, 1),
        ("user", {"content": {"items": [1]}}, True, 2),
        ("provider_result", PROVIDER_RESULT_PAYLOAD, False, 3),
        ("provider_result", PROVIDER_RESULT_PAYLOAD, True, 4),
    ],
    ids=["user-content", "user-actor", "provider-result", "provider-result-actor"],
)
def test_input_hydration_uses_one_unwrap_per_local_frame(
    content_setup,
    kind: str,
    payload: dict[str, Any],
    include_actor_attributes: bool,
    decrypts: int,
) -> None:
    store, _wrapped, reader, provider, cipher = content_setup
    push_fixture_input(store, SESSION, InboxItem(f"{kind}-input", kind, payload))
    stored = store.read_input(SESSION, f"{kind}-input")
    assert stored is not None
    provider.unwrap_calls = provider.restore_calls = cipher.decrypt_calls = 0

    hydrated = hydrate_session_input(
        stored.item,
        SessionHydration(reader, tenant_id=store.tenant_id, session_id=SESSION),
        include_actor_attributes=include_actor_attributes,
    )

    assert hydrated.payload == payload
    if include_actor_attributes:
        assert hydrated.actor is not None
        assert hydrated.actor["attributes"] == {"role": f"fixture/{kind}"}
    else:
        assert hydrated.actor == stored.item.actor
    assert provider.unwrap_calls == 1
    assert provider.restore_calls == 0
    assert cipher.decrypt_calls == decrypts


def test_input_frame_mutable_values_are_isolated_and_destroy_is_seen_next_call(content_setup) -> None:
    store, _wrapped, reader, provider, _cipher = content_setup
    push_fixture_input(
        store,
        SESSION,
        InboxItem("provider-mutable", "provider_result", PROVIDER_RESULT_PAYLOAD),
    )
    stored = store.read_input(SESSION, "provider-mutable")
    assert stored is not None
    hydration = SessionHydration(reader, tenant_id=store.tenant_id, session_id=SESSION)

    first = hydrate_session_input(stored.item, hydration, include_actor_attributes=True)
    first.payload["result"]["items"].append(2)
    second = hydrate_session_input(stored.item, hydration, include_actor_attributes=True)
    assert second.payload["result"] == {"items": [1]}

    ref = ContentRef.from_dict(stored.item.payload["result"])
    assert provider.destroy(store.tenant_id, ref.key_id).accepted
    with pytest.raises(KeyUnavailable):
        hydrate_session_input(stored.item, hydration, include_actor_attributes=True)


def test_record_batch_does_not_prefetch_unused_input_owner_links(content_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    item = InboxItem("batched-input", "user", {"content": "batched"})
    push_fixture_input(store, SESSION, item)
    stored_input = store.read_input(SESSION, item.input_id)
    assert stored_input is not None
    input_record = fixture_record(
        "input_applied",
        sid=SESSION,
        input=stored_input.item.to_dict(),
        input_digest=digest(stored_input.item.to_dict()),
    )
    append_fixture_records(store, SESSION, (input_record,), consume_input_ids=(item.input_id,), commit_id="batch-input")
    records = store.read(SESSION).records

    assert isinstance(reader._store, _ValidatedDurableContentStore)
    assert isinstance(reader._store._durable, CoreSQLContentStore)
    backend = reader._store._durable._backend
    original_fetchall = backend.fetchall
    statements: list[str] = []

    def track_fetchall(statement: str, parameters=()):
        statements.append(statement)
        return original_fetchall(statement, parameters)

    monkeypatch.setattr(backend, "fetchall", track_fetchall)
    hydrated = hydrate_session_records(
        records,
        SessionHydration(reader, tenant_id=store.tenant_id, session_id=SESSION),
    )

    assert len(hydrated) == len(records)
    assert any("record_seq IN" in statement for statement in statements)
    assert not any("input_id IN" in statement for statement in statements)


def test_frame_leaves_caller_transaction_open(content_setup, store_backend: str) -> None:
    store, _wrapped, reader, _provider, _cipher = content_setup
    ref = _link_values(store, {"request.messages[0]": {"body": "caller-owned"}})["request.messages[0]"]

    if store_backend == "sqlite":
        store.connection.execute("BEGIN")
        try:
            with reader._read_frame(record_seqs=(1,)) as frame_reader:
                frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)
            assert store.connection.in_transaction
        finally:
            store.connection.execute("ROLLBACK")
        assert not store.connection.in_transaction
        return

    from psycopg.pq import TransactionStatus

    store.connection.execute("BEGIN")
    try:
        with reader._read_frame(record_seqs=(1,)) as frame_reader:
            frame_reader.read(store.tenant_id, SESSION, record_owner(1), "request.messages[0]", ref)
        assert store.connection.info.transaction_status == TransactionStatus.INTRANS
    finally:
        store.connection.rollback()
    assert store.connection.info.transaction_status == TransactionStatus.IDLE
