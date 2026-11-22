"""S1-A3/A11 deletion and recovery acceptance across durable boundaries."""

from __future__ import annotations

import base64
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from support.s1_fixture_values import (
    create_fixture_session,
    fixture_child_spec,
    push_fixture_input,
    wrap_fixture_store,
)

from tests.sim.world import Choices
from vv_agent.kernel._content import (
    ContentLabels,
    ContentRef,
    ContentSessionWriter,
    ContentUnavailable,
    ExportContentMode,
    ExportLineKind,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    Principal,
    TombstoneState,
    _tombstone_message,
)
from vv_agent.kernel._records import InboxItem, SessionSpec
from vv_agent.kernel._sql import SessionNotLive, SQLStore
from vv_agent.kernel._store import SESSION_PURGE_DELETE_ORDER, Conflict
from vv_agent.session import kernel
from vv_agent.session.children import ChildSession, ChildTool
from vv_agent.session.content_runtime import (
    SESSION_WORKSPACE,
    SessionContentWriter,
    bind_content_runtime,
    content_runtime_binding_for,
    durable_content_store,
)
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.deletion import SessionDeletionStore
from vv_agent.session.export import export_session
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost

from .conftest import open_store
from .test_children import parent_runtime

TENANT = "default"
MASTER_KEY = b"r" * 32


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "deletion-recovery-acceptance-key"
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


def _actor(store: Any, session_id: str) -> Principal:
    created = store.read(session_id, limit=1).records[0].record
    return Principal.from_dict(created.payload["principal"])


def _create_child(store: Any, parent_session_id: str, session_id: str) -> None:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)

        def spec(content: ContentSessionWriter) -> SessionSpec:
            principal, _attributes = content.service_principal(f"fixture/{session_id}")
            return SessionSpec(
                session_id,
                principal,
                content.put_json(f"/tmp/{session_id}").to_dict(),
                parent_session_id=parent_session_id,
                parent_operation_id=f"operation/{session_id}",
                attributes=content.put_json({"seed": {"messages": [], "shared_state": {}}}).to_dict(),
            )

        writer.create_session(spec, consumers=("events",))


def _counts(store: Any, session_id: str) -> dict[str, int]:
    tables = ("sk_session_key", "sk_content", "sk_content_link")
    return {
        table: store._one(
            f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
            (TENANT, session_id),
        )[0]
        for table in tables
    }


def _session_state(store: Any, session_id: str) -> tuple[str, str | None]:
    rows = store._rows(
        "SELECT s.deletion_state,t.state FROM sk_session s "
        "LEFT JOIN sk_tombstone t ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id "
        "WHERE s.tenant_id=%s AND s.session_id=%s",
        (TENANT, session_id),
    )
    if rows:
        return rows[0]
    tombstone = store._rows(
        "SELECT state FROM sk_tombstone WHERE tenant_id=%s AND session_id=%s",
        (TENANT, session_id),
    )
    if tombstone:
        return "purged", tombstone[0][0]
    raise AssertionError("session and tombstone are both missing")


def _assert_tombstone_signatures(signer: Signer, tombstones: tuple[Any, ...]) -> None:
    for tombstone in tombstones:
        assert tombstone.state is TombstoneState.COMPLETE
        assert signer.verify(
            tombstone.key_id,
            _tombstone_message(tombstone),
            base64.b64decode(tombstone.signature),
        )


def _sqlite_backup(store: SQLiteStore, backup: Path) -> None:
    backup.unlink(missing_ok=True)
    with sqlite3.connect(backup) as target:
        store.connection.backup(target)


@contextmanager
def _signed_heartbeat_store(database: Any, signer: Signer):
    with open_store(database) as heartbeat:
        heartbeat.configure_input_admission(signer)
        yield heartbeat


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    ("operation", "pending_state", "completed_state"),
    [("forget_session", "forgetting", "forgotten"), ("purge_session", "purging", "purged")],
)
def test_deletion_requires_a_committed_fence_before_destroying_key(
    store,
    database,
    operation: str,
    pending_state: str,
    completed_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    signer = Signer()
    try:
        store.configure_input_admission(signer)
        wrapped = wrap_fixture_store(store, provider)
        create_fixture_session(wrapped, "session")
        actor = _actor(store, "session")
        workspace = ContentRef.from_dict(store.read("session", limit=1).records[0].record.payload["workspace"])
        adapter = durable_content_store(wrapped)
        deletion = SessionDeletionStore(store, adapter)
        delete = deletion.forget_session if operation == "forget_session" else deletion.purge_session

        with open_store(database, allow_unsigned=False) as observer:
            destroy_states: list[tuple[str, str | None]] = []
            original_destroy = provider.destroy

            def observed_destroy(tenant_id: str, key_id: str) -> KeyDestroyOutcome:
                destroy_states.append(_session_state(observer, "session"))
                return original_destroy(tenant_id, key_id)

            monkeypatch.setattr(provider, "destroy", observed_destroy)
            store.connection.execute("BEGIN")
            try:
                with pytest.raises(RuntimeError, match="requires an idle connection"):
                    delete(TENANT, "session", actor=actor)
            finally:
                store.connection.execute("ROLLBACK")

            assert destroy_states == []
            assert _session_state(observer, "session") == ("live", None)
            assert adapter.get(TENANT, "session", workspace) == b'"/tmp"'

            assert delete(TENANT, "session", actor=actor).session_ids == ("session",)
            assert destroy_states == [(pending_state, "pending")]
            assert _session_state(observer, "session") == (completed_state, "complete")
    finally:
        provider.close()


@pytest.mark.persistent_store
@pytest.mark.parametrize("deletion_state", ["forgetting", "purged"])
def test_fenced_session_creation_rejects_before_content_factory(
    store,
    deletion_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    signer = Signer()
    try:
        store.configure_input_admission(signer)
        wrapped = wrap_fixture_store(store, provider)
        allocations: list[tuple[str, str]] = []
        factory_calls: list[str] = []
        generate = provider.generate_data_key

        def observed_generate(tenant_id: str, session_id: str):
            allocations.append((tenant_id, session_id))
            return generate(tenant_id, session_id)

        def spec(content: ContentSessionWriter) -> SessionSpec:
            factory_calls.append("session")
            principal, _ = content.service_principal("fixture/host")
            return SessionSpec(
                "session",
                principal,
                content.put_json("/tmp").to_dict(),
                attributes=content.put_json({}).to_dict(),
            )

        monkeypatch.setattr(provider, "generate_data_key", observed_generate)
        with wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session")
            assert not writer.create_session(spec, consumers=("events",)).replayed
        assert factory_calls == ["session"]
        assert allocations == [(TENANT, "session")]

        actor = _actor(store, "session")
        if deletion_state == "forgetting":
            store._scoped(TENANT).begin_forget(
                "session", actor, bound_signer=signer.bind_key(), verifier=signer, allow_unsigned=False
            )
            assert _session_state(store, "session") == ("forgetting", "pending")
        else:
            deletion = SessionDeletionStore(store, durable_content_store(wrapped))
            deletion.purge_session(TENANT, "session", actor=actor)
            assert _session_state(store, "session") == ("purged", "complete")

        with pytest.raises(SessionNotLive, match="identity cannot be reused"), wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session")
            writer.create_session(spec, consumers=("events",))
        assert factory_calls == ["session"]
        assert allocations == [(TENANT, "session")]
    finally:
        provider.close()


@pytest.mark.sqlite_file
def test_sqlite_deletion_order_restore_convergence_and_identity_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "deletion-state.sqlite"
    backup = tmp_path / "forgotten-backup.sqlite"
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    signer = Signer()
    actor: Principal | None = None
    root_head: tuple[int, str] | None = None
    child_head: tuple[int, str] | None = None
    try:
        with SQLiteStore.standalone(str(path)) as store:
            store.install_schema()
            store.configure_input_admission(signer)
            wrapped = wrap_fixture_store(store, provider)
            create_fixture_session(wrapped, "root")
            _create_child(wrapped, "root", "child")
            actor = _actor(store, "root")
            root_head = (
                store._one("SELECT head_seq FROM sk_session WHERE tenant_id=%s AND session_id='root'", (TENANT,))[0],
                store._chain_anchor("root", store._one("SELECT head_seq FROM sk_session WHERE session_id='root'")[0]),
            )
            child_head = (
                store._one("SELECT head_seq FROM sk_session WHERE tenant_id=%s AND session_id='child'", (TENANT,))[0],
                store._chain_anchor("child", store._one("SELECT head_seq FROM sk_session WHERE session_id='child'")[0]),
            )
            adapter = durable_content_store(wrapped)
            with pytest.raises(ValueError, match="cannot override a signed admission policy"):
                SessionDeletionStore(store, adapter, allow_unsigned=True).forget_session(
                    TENANT,
                    "root",
                    actor=actor,
                )
            assert _session_state(store, "root") == ("live", None)
            deletion = SessionDeletionStore(store, adapter)
            observer = sqlite3.connect(path, isolation_level=None)
            observations: list[tuple[str, tuple[str, str | None]]] = []
            original_destroy = provider.destroy

            def observed_destroy(tenant_id: str, key_id: str) -> KeyDestroyOutcome:
                row = observer.execute(
                    "SELECT s.deletion_state,t.state FROM sk_session s JOIN sk_tombstone t "
                    "ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id "
                    "WHERE s.tenant_id=? AND s.session_id='root'",
                    (TENANT,),
                ).fetchone()
                observations.append(("destroy", row))
                return original_destroy(tenant_id, key_id)

            monkeypatch.setattr(provider, "destroy", observed_destroy)
            first = deletion.forget_session(TENANT, "root", actor=actor)
            second = deletion.forget_session(TENANT, "root", actor=actor)
            monkeypatch.setattr(provider, "destroy", original_destroy)

            assert observations == [
                ("destroy", ("forgetting", "pending")),
                ("destroy", ("forgetting", "pending")),
            ]
            assert first == second
            assert _session_state(store, "root") == ("forgotten", "complete")
            assert _session_state(store, "child") == ("forgotten", "complete")
            forgotten_export = export_session(store._scoped(TENANT), "root")
            assert forgotten_export.header.head_seq == root_head[0]
            assert forgotten_export.header.chain_digest == root_head[1]
            tombstones = deletion.verify_tombstones(TENANT, "root")
            _assert_tombstone_signatures(signer, tombstones)
            assert {(item.session_id, item.final_seq, item.chain_digest) for item in tombstones} == {
                ("root", root_head[0], root_head[1]),
                ("child", child_head[0], child_head[1]),
            }

            _sqlite_backup(store, backup)

            original_begin_purge = SQLStore.begin_purge
            purge_observations: list[tuple[str, tuple[str, str | None]]] = []

            def observed_begin_purge(self: SQLStore, *args: Any, **kwargs: Any):
                purge_observations.append(("before", _session_state(store, "root")))
                scope = original_begin_purge(self, *args, **kwargs)
                purge_observations.append(("after", _session_state(store, "root")))
                return scope

            monkeypatch.setattr(SQLStore, "begin_purge", observed_begin_purge)
            purged = deletion.purge_session(TENANT, "root", actor=actor)
            repeated = deletion.purge_session(TENANT, "root", actor=actor)
            monkeypatch.setattr(SQLStore, "begin_purge", original_begin_purge)
            assert purge_observations == [
                ("before", ("forgotten", "complete")),
                ("after", ("purging", "pending")),
                ("before", ("purged", "complete")),
                ("after", ("purged", "complete")),
            ]
            assert purged == repeated
            assert store._rows("SELECT session_id FROM sk_session WHERE tenant_id=%s ORDER BY session_id", (TENANT,)) == []
            _assert_tombstone_signatures(signer, deletion.verify_tombstones(TENANT, "root"))
            with pytest.raises(SessionNotLive, match="identity cannot be reused"):
                create_fixture_session(wrapped, "root")
            assert all(count == 0 for count in _counts(store, "root").values())
            observer.close()

        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        shutil.copy2(backup, path)

        with SQLiteStore.standalone(str(path)) as restored:
            restored.configure_input_admission(signer)
            restored_wrapped = wrap_fixture_store(restored, provider)
            restored_adapter = durable_content_store(restored_wrapped)
            restored_deletion = SessionDeletionStore(restored, restored_adapter)
            assert actor is not None
            assert _session_state(restored, "root") == ("forgotten", "complete")
            skeleton = restored_deletion.read_skeleton(TENANT, "root")
            assert skeleton.tombstone is not None
            assert skeleton.tombstone.state is TombstoneState.COMPLETE
            restored_export = export_session(restored._scoped(TENANT), "root")
            assert restored_export.header.chain_digest == root_head[1]
            with pytest.raises(ContentUnavailable), restored_wrapped.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="root")
                writer.put_bytes(b"late publication after backup restore", labels=ContentLabels())
            restored_deletion.purge_session(TENANT, "root", actor=actor)
            assert _assert_tombstone_signatures(signer, restored_deletion.verify_tombstones(TENANT, "root")) is None
    finally:
        provider.close()


@pytest.mark.persistent_store
def test_staged_late_usage_export_and_readback_stay_fenced_at_every_boundary(
    store,
    database,
    store_backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"keys-{store_backend}.sqlite3")
    signer = Signer()
    try:
        store.configure_input_admission(signer)
        wrapped = wrap_fixture_store(store, provider)
        create_fixture_session(wrapped, "session")
        actor = _actor(store, "session")
        workspace_ref = ContentRef.from_dict(store.read("session", limit=1).records[0].record.payload["workspace"])
        with wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session")
            staged = writer.put_bytes(b"callback staged marker", labels=ContentLabels())

        callback = InboxItem(
            "late-usage",
            "provider_result",
            {
                "operation_id": "operation/late",
                "attempt": 1,
                "request_digest": "a" * 64,
                "provider_binding": "provider/late",
                "result": {"text": "late result"},
                "usage": {"total_tokens": 17},
                "evidence": ["accepted/late"],
            },
        )
        first = push_fixture_input(wrapped, "session", callback)
        replay = push_fixture_input(wrapped, "session", callback)
        assert replay.replayed and replay.input_seq == first.input_seq
        with pytest.raises(Conflict):
            push_fixture_input(
                wrapped,
                "session",
                replace(callback, payload=callback.payload | {"usage": {"total_tokens": 18}}),
            )

        adapter = durable_content_store(wrapped)
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        baseline_counts = _counts(store, "session")
        head = store._one(
            "SELECT head_seq FROM sk_session WHERE tenant_id=%s AND session_id='session'",
            (TENANT,),
        )[0]
        chain = store._chain_anchor("session", head)
        with open_store(database) as observer:
            if store_backend == "postgres":
                assert observer._one("SELECT pg_backend_pid()")[0] != store._one("SELECT pg_backend_pid()")[0]

            destroy_state: list[tuple[str, str | None]] = []

            def unavailable_destroy(_tenant_id: str, _key_id: str) -> KeyDestroyOutcome:
                destroy_state.append(_session_state(observer, "session"))
                return KeyDestroyOutcome(False, KeyDestroyStatus.KEY_UNAVAILABLE, None, "provider unavailable")

            with monkeypatch.context() as scoped:
                scoped.setattr(provider, "destroy", unavailable_destroy)
                with pytest.raises(KeyUnavailable, match="provider unavailable"):
                    deletion.forget_session(TENANT, "session", actor=actor)

            assert destroy_state == [("forgetting", "pending")]
            forgetting_skeleton = deletion.read_skeleton(TENANT, "session")
            assert forgetting_skeleton.tombstone is not None
            assert forgetting_skeleton.tombstone.state is TombstoneState.PENDING
            with pytest.raises(ContentUnavailable):
                adapter.get(TENANT, "session", staged)
            with pytest.raises(ContentUnavailable), wrapped.atomic() as tx:
                SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session").put_bytes(
                    b"late put while forgetting",
                    labels=ContentLabels(),
                )
            with pytest.raises(ContentUnavailable), wrapped.atomic() as tx:
                SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session").link_record(
                    1,
                    staged,
                    SESSION_WORKSPACE,
                )
            with pytest.raises((ContentUnavailable, SessionNotLive)):
                push_fixture_input(wrapped, "session", replace(callback, input_id="late-usage-forgetting"))
            with pytest.raises(SessionNotLive, match="stable for export"):
                export_session(store._scoped(TENANT), "session")
            assert _counts(store, "session") == baseline_counts

            forgotten = deletion.forget_session(TENANT, "session", actor=actor)
            assert forgotten.session_ids == ("session",)
            assert _session_state(observer, "session") == ("forgotten", "complete")
            assert all(count == 0 for count in _counts(store, "session").values())
            with pytest.raises(ContentUnavailable):
                adapter.get(TENANT, "session", staged)
            with pytest.raises(ContentUnavailable):
                adapter.get(TENANT, "session", workspace_ref)
            with pytest.raises((ContentUnavailable, SessionNotLive)):
                push_fixture_input(wrapped, "session", replace(callback, input_id="late-usage-forgotten"))
            skeleton_export = export_session(store._scoped(TENANT), "session")
            inbox_lines = [line for line in skeleton_export.lines if line.kind is ExportLineKind.INBOX]
            assert len(inbox_lines) == 1
            assert inbox_lines[0].value["input"]["input_id"] == "late-usage"
            ContentRef.from_dict(inbox_lines[0].value["input"]["payload"]["usage"])
            with pytest.raises(ContentUnavailable):
                export_session(
                    store._scoped(TENANT),
                    "session",
                    content_mode=ExportContentMode.INCLUDE_CONTENT,
                    content_store=adapter,
                )

            original_begin_purge = SQLStore.begin_purge
            purge_states: list[tuple[str, tuple[str, str | None]]] = []

            def observed_begin_purge(self: SQLStore, *args: Any, **kwargs: Any):
                purge_states.append(("before", _session_state(observer, "session")))
                scope = original_begin_purge(self, *args, **kwargs)
                purge_states.append(("after", _session_state(observer, "session")))
                return scope

            def cleanup_timeout(_tenant_id: str, _session_id: str) -> None:
                raise KeyUnavailable("content cleanup timed out")

            with monkeypatch.context() as scoped:
                scoped.setattr(SQLStore, "begin_purge", observed_begin_purge)
                scoped.setattr(adapter, "forget_session", cleanup_timeout)
                with pytest.raises(KeyUnavailable, match="cleanup timed out"):
                    deletion.purge_session(TENANT, "session", actor=actor)

            assert purge_states == [
                ("before", ("forgotten", "complete")),
                ("after", ("purging", "pending")),
            ]
            assert _session_state(observer, "session") == ("purging", "pending")
            with pytest.raises(SessionNotLive):
                deletion.read_skeleton(TENANT, "session")
            with pytest.raises(SessionNotLive, match="stable for export"):
                export_session(store._scoped(TENANT), "session")
            with pytest.raises((ContentUnavailable, SessionNotLive)):
                push_fixture_input(wrapped, "session", replace(callback, input_id="late-usage-purging"))
            assert all(count == 0 for count in _counts(store, "session").values())

            purged = deletion.purge_session(TENANT, "session", actor=actor)
            assert purged.session_ids == ("session",)
            repeated = deletion.purge_session(TENANT, "session", actor=actor)
            assert repeated == purged
            tombstones = deletion.verify_tombstones(TENANT, "session")
            assert [(item.final_seq, item.chain_digest, item.state) for item in tombstones] == [
                (head, chain, TombstoneState.COMPLETE)
            ]
            _assert_tombstone_signatures(signer, tombstones)
            with pytest.raises((Conflict, SessionNotLive)):
                export_session(store._scoped(TENANT), "session")
            with pytest.raises(SessionNotLive, match="identity cannot be reused"):
                create_fixture_session(wrapped, "session")
            assert store._rows("SELECT session_id FROM sk_session WHERE tenant_id=%s", (TENANT,)) == []
    finally:
        provider.close()


@pytest.mark.postgres
@pytest.mark.parametrize("seed", [9003, 9004])
def test_create_rechecks_deletion_committed_before_insert(
    seed: int,
    store,
    database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The insert must recheck a tombstone committed after the admission reads."""
    choices = Choices(seed)
    operation = choices.choose("delete_before_create_insert", ("purge", "forget"))
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys-create-purge-race.sqlite3")
    signer = Signer()
    try:
        store.configure_input_admission(signer)
        wrapped = wrap_fixture_store(store, provider)
        create_fixture_session(wrapped, "session")
        payload = store.read("session", limit=1).records[0].record.payload
        replay_spec = SessionSpec("session", payload["principal"], payload["workspace"], attributes=payload["attributes"])
        actor = Principal.from_dict(payload["principal"])
        create_backend_pid = store._one("SELECT pg_backend_pid()")[0]

        original_rows = store._rows
        deleted = None

        def delete_before_insert(query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
            nonlocal deleted
            if " ".join(query.split()).startswith("INSERT INTO sk_session(") and params[:2] == (TENANT, "session"):
                assert deleted is None
                # The create transaction stays open across the other connection's commit.
                with open_store(database, allow_unsigned=False) as delete_store:
                    assert delete_store._one("SELECT pg_backend_pid()")[0] != create_backend_pid
                    delete_store._rows("SET lock_timeout = '5s'")
                    delete_store.configure_input_admission(signer)
                    delete_wrapped = wrap_fixture_store(delete_store, provider)
                    deletion = SessionDeletionStore(delete_store, durable_content_store(delete_wrapped), signer=signer)
                    delete = deletion.purge_session if operation == "purge" else deletion.forget_session
                    delete(TENANT, "session", actor=actor)
                    deleted = deletion.verify_tombstones(TENANT, "session")
                    _assert_tombstone_signatures(signer, deleted)
            return original_rows(query, params)

        monkeypatch.setattr(store, "_rows", delete_before_insert)
        with pytest.raises(SessionNotLive, match="identity cannot be reused"), store.atomic() as tx:
            tx.create(replay_spec, consumers=("events",))

        assert deleted is not None and len(deleted) == 1
        assert store._tombstone("session") == deleted[0]
        if operation == "purge":
            for table in SESSION_PURGE_DELETE_ORDER:
                assert store._one(
                    f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id='session'",
                    (TENANT,),
                ) == (0,)
        else:
            assert _session_state(store, "session") == ("forgotten", TombstoneState.COMPLETE.value)
            assert all(count == 0 for count in _counts(store, "session").values())
    finally:
        provider.close()


@pytest.mark.postgres
@pytest.mark.parametrize("isolation", ["REPEATABLE_READ", "SERIALIZABLE"])
def test_postgres_create_rejects_transaction_snapshot_before_writing(
    isolation: str,
    store,
    database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from psycopg import IsolationLevel

    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys-purge-isolation.sqlite3")
    signer = Signer()
    try:
        store.configure_input_admission(signer)
        wrapped = wrap_fixture_store(store, provider)
        create_fixture_session(wrapped, "session")
        payload = store.read("session", limit=1).records[0].record.payload
        replay_spec = SessionSpec("session", payload["principal"], payload["workspace"], attributes=payload["attributes"])
        actor = Principal.from_dict(payload["principal"])

        original_rows = store._rows

        def check_no_insert(query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
            assert not " ".join(query.split()).startswith("INSERT INTO sk_session(")
            return original_rows(query, params)

        def reject_factory(_content: ContentSessionWriter) -> SessionSpec:
            pytest.fail("a stale SQL snapshot cannot be used to prepare session content")

        monkeypatch.setattr(store, "_rows", check_no_insert)
        store.connection.isolation_level = IsolationLevel[isolation]
        with store.atomic() as tx:
            assert store._one("SELECT deletion_state FROM sk_session WHERE session_id='session'") == ("live",)
            with open_store(database, allow_unsigned=False) as purge_store:
                purge_store._rows("SET lock_timeout = '5s'")
                purge_store.configure_input_admission(signer)
                purge_wrapped = wrap_fixture_store(purge_store, provider)
                deletion = SessionDeletionStore(purge_store, durable_content_store(purge_wrapped), signer=signer)
                assert deletion.purge_session(TENANT, "session", actor=actor).session_ids == ("session",)
            assert store._tombstone("session") is None
            with pytest.raises(Conflict, match="PostgreSQL session creation requires READ COMMITTED isolation"):
                tx.create(replay_spec, consumers=("events",))
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id="session")
            with pytest.raises(Conflict, match="PostgreSQL session creation requires READ COMMITTED isolation"):
                writer.create_session(reject_factory, consumers=("events",))
            assert store._one("SELECT 1") == (1,)

        retained = store._tombstone("session")
        assert retained is not None
        _assert_tombstone_signatures(signer, (retained,))
        assert not store.list_sessions()
    finally:
        provider.close()


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        ("callback", (RuntimeError, "callback failed")),
        ("key", (TimeoutError, "key allocation timed out")),
        ("upload", (TimeoutError, "content upload timed out")),
        ("sign", (RuntimeError, "signer unavailable")),
        ("lease", (LeaseLost, "")),
    ),
)
def test_postgres_child_admission_failure_matrix_rolls_back_content_and_recovers_once(
    failure: str,
    expected: tuple[type[BaseException], str],
    store,
    database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"keys-{failure}.sqlite3")
    signer = Signer()
    binding_store = store
    bind_content_runtime(binding_store, key_provider=provider, cipher=AESGCMCipher())
    from vv_agent.kernel._content_runtime import wrap_session_content_store as wrap_bound_store

    wrapped = wrap_bound_store(binding_store)
    store.configure_input_admission(signer)
    create_fixture_session(wrapped, "s")
    push_fixture_input(wrapped, "s", InboxItem("initial", "user", {"content": "go"}))
    store._rows("CREATE TABLE callback_effect (attempt integer PRIMARY KEY, marker text NOT NULL)")
    binding = content_runtime_binding_for(wrapped)
    assert binding is not None
    cipher = binding.cipher
    original_generate = provider.generate_data_key
    original_encrypt = cipher.encrypt
    admission_calls = 0
    staged_ref: ContentRef | None = None
    retry_ref: ContentRef | None = None
    callback_returned = False
    lease_cut: list[dict[str, Any]] = []
    lease_clock = [store._now()]

    if failure == "lease":
        # Public drive has no post-callback/pre-append hook. Use the same
        # committing latch seam as test_lease_latch_order while keeping the real
        # admission and final append transaction open.
        monkeypatch.setattr(PostgresStore, "_now", lambda self: lease_clock[0])

    original_committing = kernel._Scope.committing

    @contextmanager
    def expire_at_final_commit(scope):
        with original_committing(scope) as lease:
            if failure == "lease" and callback_returned and admission_calls == 1 and not lease_cut:
                assert staged_ref is not None
                lease_cut.append(
                    {
                        "parent_head": store.read("s", limit=1).head_seq,
                        "parent_records": tuple((row.seq, row.record.encode()) for row in store.read("s", limit=1000).records),
                        "callback_effect": store._rows("SELECT attempt,marker FROM callback_effect"),
                        "staged_content": store._one(
                            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id='s' AND content_id=%s",
                            (TENANT, staged_ref.content_id),
                        )[0],
                        "child_sessions": store._one(
                            "SELECT COUNT(*) FROM sk_session WHERE tenant_id=%s AND session_id='child'",
                            (TENANT,),
                        )[0],
                        "child_inbox": store._one(
                            "SELECT COUNT(*) FROM sk_inbox WHERE tenant_id=%s AND session_id='child' AND input_id='start'",
                            (TENANT,),
                        )[0],
                        "child_content": store._one(
                            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id='child'",
                            (TENANT,),
                        )[0],
                        "child_links": store._one(
                            "SELECT COUNT(*) FROM sk_content_link WHERE tenant_id=%s AND session_id='child'",
                            (TENANT,),
                        )[0],
                    }
                )
                lease_clock[0] = lease.expires_at_ms + 1
            yield lease

    if failure == "lease":
        monkeypatch.setattr(kernel._Scope, "committing", expire_at_final_commit)

    def admit(plan):
        nonlocal admission_calls, staged_ref, retry_ref, callback_returned
        admission_calls += 1
        with wrapped.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=plan.session_id)
            ref = writer.put_json({"attempt": admission_calls, "marker": f"callback-{failure}"})
            if admission_calls == 1:
                staged_ref = ref
            else:
                retry_ref = ref
            store._rows(
                "INSERT INTO callback_effect(attempt,marker) VALUES (%s,%s)",
                (admission_calls, failure),
            )

        child = ChildSession(fixture_child_spec(wrapped, plan.session_id, "child"), "work")
        if admission_calls > 1:
            return child
        if failure == "callback":
            raise RuntimeError("callback failed after staging content")
        if failure == "key":
            monkeypatch.setattr(
                provider,
                "generate_data_key",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("key allocation timed out")),
            )
        elif failure == "upload":
            monkeypatch.setattr(
                cipher,
                "encrypt",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("content upload timed out")),
            )
        elif failure == "sign":
            signer.fail = True
        callback_returned = True
        return child

    try:
        rt = parent_runtime(wrapped, database)
        rt.ttl_ms = 30_000
        rt.heartbeat_store = lambda: _signed_heartbeat_store(database, signer)
        rt.children["child_task"] = ChildTool(admit)
        raised: BaseException | None = None
        try:
            drive(wrapped, "s", runtime=rt)
        except BaseException as exc:
            raised = exc

        if raised is not None:
            assert isinstance(raised, expected[0]), raised
            if expected[1]:
                assert expected[1] in str(raised), raised

        signer.fail = False
        monkeypatch.setattr(provider, "generate_data_key", original_generate)
        monkeypatch.setattr(cipher, "encrypt", original_encrypt)
        assert admission_calls == 1
        assert store._rows("SELECT * FROM callback_effect") == []
        assert store.list_sessions() == ("s",)
        assert staged_ref is not None
        assert store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id='s' AND content_id=%s",
            (TENANT, staged_ref.content_id),
        ) == (0,)

        if failure == "lease":
            assert callback_returned and len(lease_cut) == 1
            cut = lease_cut[0]
            assert cut["callback_effect"] == [(1, "lease")]
            assert cut["staged_content"] == cut["child_sessions"] == cut["child_inbox"] == 1
            assert cut["child_content"] > 0 and cut["child_links"] > 0
            assert store.read("s", limit=1).head_seq == cut["parent_head"]
            assert tuple((row.seq, row.record.encode()) for row in store.read("s", limit=1000).records) == cut["parent_records"]
            assert store._one(
                "SELECT COUNT(*) FROM sk_session WHERE tenant_id=%s AND session_id='child'",
                (TENANT,),
            ) == (0,)
            assert store._one(
                "SELECT COUNT(*) FROM sk_inbox WHERE tenant_id=%s AND session_id='child'",
                (TENANT,),
            ) == (0,)
            assert _counts(store, "child") == {"sk_session_key": 0, "sk_content": 0, "sk_content_link": 0}

        if not store.peek_inbox("s"):
            push_fixture_input(
                wrapped,
                "s",
                InboxItem("retry", "follow_up", {"content": "retry"}, generation=2),
            )

        retry = parent_runtime(wrapped, database)
        retry.heartbeat_store = lambda: _signed_heartbeat_store(database, signer)
        retry.children["child_task"] = ChildTool(admit)
        drive(wrapped, "s", runtime=retry)
        assert admission_calls == 2
        assert store._rows("SELECT attempt,marker FROM callback_effect") == [(2, failure)]
        assert store.list_sessions() == ("child", "s")
        if failure == "lease":
            assert retry_ref is not None
            assert store._one(
                "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id='s' AND content_id=%s",
                (TENANT, retry_ref.content_id),
            ) == (1,)
            assert store._one(
                "SELECT COUNT(*) FROM sk_session WHERE tenant_id=%s AND session_id='child'",
                (TENANT,),
            ) == (1,)
            assert store._one(
                "SELECT COUNT(*) FROM sk_inbox WHERE tenant_id=%s AND session_id='child' AND input_id='start'",
                (TENANT,),
            ) == (1,)
            assert len([row for row in read_state(wrapped, "s")[1] if row.record.kind == "op_parked"]) == 1
        assert read_state(wrapped, "s")[0].phase == "parked"
        assert store._one(
            "SELECT COUNT(*) FROM sk_inbox WHERE tenant_id=%s AND session_id='child' AND input_id='start'",
            (TENANT,),
        ) == (1,)

        unchanged_head = store.read("s", limit=1).head_seq
        final_rt = parent_runtime(wrapped, database, steps=[])
        final_rt.heartbeat_store = lambda: _signed_heartbeat_store(database, signer)
        drive(wrapped, "s", runtime=final_rt)
        assert admission_calls == 2
        assert store.read("s", limit=1).head_seq == unchanged_head
        assert store.list_sessions() == ("child", "s")
    finally:
        monkeypatch.setattr(provider, "generate_data_key", original_generate)
        monkeypatch.setattr(cipher, "encrypt", original_encrypt)
        provider.close()
