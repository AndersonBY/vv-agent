"""Durable encrypted-content adapter tests against real SQL backends."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any
from uuid import uuid4

import pytest

from vv_agent.kernel import _content as content
from vv_agent.kernel import _content_store as content_store
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._postgres import PostgresContentStore
from vv_agent.kernel._records import InboxItem, SessionSpec
from vv_agent.kernel._sqlite import SQLiteContentStore
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore

MASTER_KEY = b"k" * 32
TENANT = "tenant-a"
SESSION = "session-a"


def _content_ref(label: str, *, key_id: str = "key/contract") -> dict[str, Any]:
    payload = label.encode()
    return {
        "content_id": f"content/contract/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": key_id,
        "labels": {"source": [], "classification": "internal"},
    }


def _principal(tenant_id: str, subject_id: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "subject_kind": "user",
        "attributes": _content_ref(f"principal/{subject_id}"),
    }


def _create_session(store: Any, *, tenant_id: str = TENANT, session_id: str = SESSION) -> None:
    scoped = store.for_tenant(tenant_id)
    with scoped.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                _principal(tenant_id, f"user/{session_id}"),
                _content_ref(f"workspace/{tenant_id}/{session_id}"),
                attributes=_content_ref(f"attributes/{tenant_id}/{session_id}"),
            ),
            consumers=(),
        )


def _push_input(store: Any, input_id: str, *, tenant_id: str = TENANT, session_id: str = SESSION) -> None:
    scoped = store.for_tenant(tenant_id)
    with scoped.atomic() as tx:
        tx.push(
            session_id,
            InboxItem(
                input_id,
                "user",
                {"content": _content_ref(f"input/{input_id}")},
                actor=_principal(tenant_id, f"user/{input_id}"),
            ),
        )


@contextmanager
def _open_sqlite(path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        yield store


def _count(store: Any, table: str, tenant_id: str = TENANT, session_id: str = SESSION) -> int:
    assert table in {"sk_session_key", "sk_content", "sk_content_link"}
    return store._one(
        f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
        (tenant_id, session_id),
    )[0]


def test_sqlite_file_round_trip_survives_reopen_and_rejects_other_owner(tmp_path: Path) -> None:
    path = tmp_path / "content.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    plaintext = b"secret user content\x00with bytes"
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, plaintext, content.ContentLabels())
        row = store._one(
            "SELECT nonce,ciphertext,state,labels FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (TENANT, SESSION, ref.content_id),
        )
        assert row[2] == "staged"
        assert plaintext not in row[0]
        assert plaintext not in row[1]
        assert plaintext not in row[3]

    recovered_provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as recovered:
        adapter = SQLiteContentStore(recovered, recovered_provider)
        assert adapter.get(TENANT, SESSION, ref) == plaintext
        with pytest.raises(content_store.ContentUnavailable):
            adapter.get("tenant-b", SESSION, ref)
        with pytest.raises(content_store.ContentUnavailable):
            adapter.get(TENANT, "session-b", ref)


def test_duplicate_put_and_owner_links_are_scoped_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "links.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        _create_session(store, session_id="session-b")
        _push_input(store, "input-1")
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            assert isinstance(adapter, content_store.DurableContentStore)
            first = adapter.put(TENANT, SESSION, b"same plaintext", content.ContentLabels())
            repeated = adapter.put(TENANT, SESSION, b"same plaintext", content.ContentLabels())
            assert repeated == first
            adapter.link_record(TENANT, SESSION, 1, first, "workspace")
            adapter.link_input(TENANT, SESSION, "input-1", first, "primary")
            adapter.link_record(TENANT, SESSION, 1, first, "workspace")

        links = adapter.read_links(TENANT, SESSION, record_seq=1)
        assert [(link.slot, link.ref) for link in links] == [("workspace", first)]
        assert adapter.assert_link(TENANT, SESSION, first, "workspace", record_seq=1) == first
        input_links = adapter.read_links(TENANT, SESSION, input_id="input-1")
        assert [(link.slot, link.ref) for link in input_links] == [("primary", first)]
        assert store._one(
            "SELECT state FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (TENANT, SESSION, first.content_id),
        ) == ("live",)

        with store.atomic(), pytest.raises(content.ContentBindingMismatch):
            adapter.link_record(TENANT, SESSION, 1, replace(first, key_id="key/other"), "other")
        with store.atomic(), pytest.raises(content_store.ContentIntegrityError):
            adapter.link_record(TENANT, SESSION, 1, replace(first, mac="0" * 64), "other")
        with store.atomic(), pytest.raises(content_store.ContentUnavailable):
            adapter.link_record(TENANT, "session-b", 1, first, "workspace")
        with pytest.raises(content_store.ContentUnavailable):
            adapter.assert_link(TENANT, "session-b", first, "workspace", record_seq=1)


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "labels", "stored_size", "key_id", "ref_mac"])
def test_sqlite_get_rejects_tampered_content_and_reference(tmp_path: Path, field: str) -> None:
    path = tmp_path / f"tamper-{field}.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, b"authenticated content", content.ContentLabels())
        if field == "ciphertext":
            value = store._one(
                "SELECT nonce,ciphertext FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (TENANT, SESSION, ref.content_id),
            )
            nonce, ciphertext = value
            store._rows(
                "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (bytes([ciphertext[0] ^ 1]) + ciphertext[1:], TENANT, SESSION, ref.content_id),
            )
            assert len(nonce) == 12
        elif field == "nonce":
            nonce = store._one(
                "SELECT nonce FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (TENANT, SESSION, ref.content_id),
            )[0]
            store._rows(
                "UPDATE sk_content SET nonce=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (bytes([nonce[0] ^ 1]) + nonce[1:], TENANT, SESSION, ref.content_id),
            )
        elif field == "labels":
            changed = content.ContentLabels(classification=content.Classification.CONFIDENTIAL)
            store._rows(
                "UPDATE sk_content SET labels=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (canonical_json_bytes(changed.to_dict()), TENANT, SESSION, ref.content_id),
            )
        elif field == "stored_size":
            store._rows(
                "UPDATE sk_content SET size=size+1 WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (TENANT, SESSION, ref.content_id),
            )
        elif field == "key_id":
            store._rows(
                "UPDATE sk_content SET key_id='key/tampered' WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (TENANT, SESSION, ref.content_id),
            )
        else:
            ref = replace(ref, mac="0" * 64)

        expected = content.ContentBindingMismatch if field == "key_id" else content_store.ContentIntegrityError
        with pytest.raises(expected):
            adapter.get(TENANT, SESSION, ref)


def test_owner_slot_conflict_requires_different_content(tmp_path: Path) -> None:
    path = tmp_path / "slot-conflict.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            first = adapter.put(TENANT, SESSION, b"first", content.ContentLabels())
            second = adapter.put(TENANT, SESSION, b"second", content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, first, "body")
            with pytest.raises(content_store.ContentLinkConflict):
                adapter.link_record(TENANT, SESSION, 1, second, "body")


def test_forget_destroys_key_before_deleting_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "forget.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, b"forget me", content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, ref, "body")
        wrapped_key = store._one(
            "SELECT key_id FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        )[0]
        calls: list[str] = []
        original_destroy = provider.destroy

        def checked_destroy(tenant_id: str, key_id: str) -> content.KeyDestroyOutcome:
            calls.append(key_id)
            assert _count(store, "sk_content") == 1
            assert _count(store, "sk_content_link") == 1
            assert _count(store, "sk_session_key") == 1
            return original_destroy(tenant_id, key_id)

        monkeypatch.setattr(provider, "destroy", checked_destroy)
        with store.atomic():
            adapter.forget_session(TENANT, SESSION)

        assert calls == [wrapped_key]
        assert _count(store, "sk_content") == 0
        assert _count(store, "sk_content_link") == 0
        assert _count(store, "sk_session_key") == 0
        assert original_destroy(TENANT, wrapped_key).status is content.KeyDestroyStatus.ALREADY_DESTROYED


def test_unconfirmed_destroy_preserves_all_content_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "destroy-pending.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, b"keep me", content.ContentLabels())

        def pending_destroy(tenant_id: str, key_id: str) -> content.KeyDestroyOutcome:
            return content.KeyDestroyOutcome(True, content.KeyDestroyStatus.DESTROYING, None)

        monkeypatch.setattr(provider, "destroy", pending_destroy)
        with store.atomic(), pytest.raises(content_store.KeyUnavailable):
            adapter.forget_session(TENANT, SESSION)

        assert _count(store, "sk_content") == 1
        assert _count(store, "sk_session_key") == 1
        assert adapter.get(TENANT, SESSION, ref) == b"keep me"


def test_key_creation_is_saved_and_already_destroyed_key_can_finish_forget(tmp_path: Path) -> None:
    path = tmp_path / "key-lifecycle.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        with store.atomic() as tx:
            adapter = SQLiteContentStore(tx, provider)
            wrapped = adapter.create_session_key(TENANT, SESSION)
            assert adapter.create_session_key(TENANT, SESSION) == wrapped
            ref = adapter.put(TENANT, SESSION, b"key lifecycle", content.ContentLabels())
        key_row = store._one(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        )
        assert key_row[0] == wrapped.key_id
        assert key_row[1] == provider.provider_id
        assert key_row[2] == wrapped.wrapped_key
        assert key_row[3] == "active"
        assert provider.unwrap(TENANT, SESSION, wrapped) not in key_row[2]

        assert provider.destroy(TENANT, wrapped.key_id).status is content.KeyDestroyStatus.DESTROYED
        with pytest.raises(content_store.KeyUnavailable):
            adapter.get(TENANT, SESSION, ref)
        with store.atomic():
            adapter.forget_session(TENANT, SESSION)
        assert _count(store, "sk_content") == 0
        assert _count(store, "sk_session_key") == 0


def test_rollback_leaves_no_content_or_session_key(tmp_path: Path) -> None:
    path = tmp_path / "rollback.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        adapter = SQLiteContentStore(store, provider)
        with pytest.raises(RuntimeError, match="rollback"), store.atomic():
            adapter.put(TENANT, SESSION, b"rolled back", content.ContentLabels())
            raise RuntimeError("rollback")

        assert _count(store, "sk_content") == 0
        assert _count(store, "sk_session_key") == 0


def test_adapter_requires_callers_transaction_and_rejects_non_active_key_state(tmp_path: Path) -> None:
    path = tmp_path / "states.sqlite"
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    with _open_sqlite(path) as store:
        _create_session(store)
        adapter = SQLiteContentStore(store, provider)
        with pytest.raises(RuntimeError, match="outer session transaction"):
            adapter.put(TENANT, SESSION, b"no implicit transaction", content.ContentLabels())
        assert _count(store, "sk_content") == 0
        assert _count(store, "sk_session_key") == 0

        with store.atomic():
            ref = adapter.put(TENANT, SESSION, b"state-bound", content.ContentLabels())
        key_row = store._one(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            (TENANT, SESSION),
        )
        key = content_store._key_from_row(TENANT, SESSION, key_row)
        with store.atomic():
            adapter.put_session_key(replace(key, state=content_store.SessionKeyState.DESTROYING))
        with pytest.raises(content_store.KeyUnavailable):
            adapter.get(TENANT, SESSION, ref)
        with pytest.raises(content_store.KeyUnavailable):
            content_store._key_from_row(TENANT, SESSION, (*key_row[:3], "destroyed"))
        with pytest.raises(content_store.KeyUnavailable):
            content_store._key_from_row(TENANT, SESSION, (*key_row[:3], "unknown"))


@pytest.fixture
def disposable_postgres_dsn() -> Iterator[str]:
    psycopg = pytest.importorskip("psycopg", reason="PostgreSQL tests need the postgres extra")
    configured = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    admin_dsn = None
    for candidate in ([configured] if configured else []) + ["dbname=postgres"]:
        try:
            with psycopg.connect(candidate, autocommit=True, connect_timeout=3):
                admin_dsn = candidate
                break
        except psycopg.OperationalError:
            continue
    if admin_dsn is None:
        pytest.skip("No PostgreSQL via VV_AGENT_TEST_POSTGRES_DSN or local unix socket")

    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = f"vv_content_test_{uuid4().hex}"
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(admin_dsn, dbname=name)
        finally:
            deadline = monotonic() + 10
            while True:
                try:
                    admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
                    break
                except psycopg.errors.ObjectInUse:
                    if monotonic() >= deadline:
                        raise
                    Event().wait(0.05)


@pytest.mark.postgres
def test_postgres_round_trip_links_and_recovery(disposable_postgres_dsn: str) -> None:
    provider = content_store.LocalKeyProvider(MASTER_KEY)
    plaintext = b"postgres durable content"
    with PostgresStore.standalone(disposable_postgres_dsn) as store:
        store.install_schema()
        _create_session(store)
        with store.atomic() as tx:
            adapter = PostgresContentStore(tx, provider)
            ref = adapter.put(TENANT, SESSION, plaintext, content.ContentLabels())
            adapter.link_record(TENANT, SESSION, 1, ref, "workspace")

    with PostgresStore.standalone(disposable_postgres_dsn) as recovered:
        adapter = PostgresContentStore(recovered, content_store.LocalKeyProvider(MASTER_KEY))
        assert adapter.get(TENANT, SESSION, ref) == plaintext
        assert adapter.assert_link(TENANT, SESSION, ref, "workspace", record_seq=1) == ref
