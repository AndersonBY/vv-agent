"""End-to-end S1 session forget and purge lifecycle tests."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from support.s1_fixture_values import create_fixture_session, push_fixture_input, wrap_fixture_store

from vv_agent.kernel._content import (
    Classification,
    ContentLabels,
    ContentRef,
    ContentSessionWriter,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    Principal,
    TombstoneState,
    _tombstone_message,
)
from vv_agent.kernel._postgres import SCHEMA_TABLES as POSTGRES_SCHEMA_TABLES
from vv_agent.kernel._records import InboxItem, SessionSpec
from vv_agent.kernel._sql import LedgerIntegrityError, SessionNotLive, SQLStore
from vv_agent.kernel._sqlite import SCHEMA_TABLES as SQLITE_SCHEMA_TABLES
from vv_agent.kernel._store import Conflict, SessionStore
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.deletion import (
    PURGE_DELETE_TABLES,
    SESSION_DELETION_TABLES,
    DeletionAuthenticity,
    DeletionSignerUnavailable,
    SessionDeletionStore,
)
from vv_agent.session.memory import save_projection, session_memory
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

from .helpers import acquire_fixture_lease
from .helpers import append_fixture_records as _append_fixture_records
from .helpers import record as _record

TENANT = "tenant-a"
MASTER_KEY = b"d" * 32


@pytest.fixture
def key_provider(tmp_path: Path) -> Iterator[LocalKeyProvider]:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "content-keys.sqlite3")
    try:
        yield provider
    finally:
        provider.close()


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "ledger-signing-key"

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


def _principal(
    store: SQLiteStore,
    session_id: str,
    tenant_id: str = TENANT,
    subject_id: str = "service/deleter",
) -> Principal:
    with store.for_tenant(TENANT).atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
        attributes = writer.put_json({"role": subject_id})
    return Principal(tenant_id, subject_id, "service", attributes)


@contextmanager
def _sqlite(
    path: Path,
    provider: LocalKeyProvider,
    *,
    signed: bool = True,
) -> Iterator[tuple[SQLiteStore, Signer | None]]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        signer = Signer() if signed else None
        if signer is not None:
            store.configure_input_admission(signer)
        wrap_fixture_store(store, provider)
        yield store, signer


def _fixture_store(store: SQLiteStore, provider: LocalKeyProvider) -> SessionStore:
    return wrap_fixture_store(store.for_tenant(TENANT), provider)


def _create_session(
    store: SQLiteStore,
    provider: LocalKeyProvider,
    session_id: str,
    *,
    parent: str | None = None,
) -> None:
    wrapped = _fixture_store(store, provider)
    if parent is None:
        create_fixture_session(wrapped, session_id, subject_id=f"fixture/{session_id}")
        return
    with wrapped.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)

        def spec(content: ContentSessionWriter) -> SessionSpec:
            actor, _ = content.service_principal(f"fixture/{session_id}")
            return SessionSpec(
                session_id,
                actor,
                content.put_json(f"/tmp/{session_id}").to_dict(),
                parent_session_id=parent,
                parent_operation_id=f"op/{session_id}",
                attributes=content.put_json({"seed": {"messages": [], "shared_state": {}}}).to_dict(),
            )

        writer.create_session(spec, consumers=())


def _publish_content(
    store: SQLiteStore,
    provider: LocalKeyProvider,
    session_id: str,
) -> tuple[SQLiteContentStore, ContentRef]:
    adapter = SQLiteContentStore(store, provider)
    with store.for_tenant(TENANT).atomic() as tx:
        ref = adapter.put(
            TENANT,
            session_id,
            f"secret:{session_id}".encode(),
            ContentLabels(classification=Classification.CONFIDENTIAL),
        )
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=session_id)
        writer.link_record(1, ref, "session.attributes.seed.messages[0]")
    return adapter, ref


def _pending_input(store: SQLiteStore, provider: LocalKeyProvider, session_id: str) -> None:
    push_fixture_input(
        _fixture_store(store, provider),
        session_id,
        InboxItem("pending-input", "control", {"action": "suspend"}),
    )


def _append_unknown_effect(store: SQLiteStore, provider: LocalKeyProvider, session_id: str) -> None:
    turn_id = f"{session_id}/turn"
    operation_id = f"{session_id}/operation"
    records = (
        _record("turn_started", sid=session_id, tid=turn_id),
        _record("op_planned", sid=session_id, tid=turn_id, oid=operation_id),
        _record("op_started", sid=session_id, tid=turn_id, oid=operation_id),
        _record("op_unknown", sid=session_id, tid=turn_id, oid=operation_id),
    )
    _append_fixture_records(_fixture_store(store, provider), session_id, records, commit_id="unknown-effect")


def _count(store: SQLiteStore, table: str, session_id: str) -> int:
    return store._one(
        f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
        (TENANT, session_id),
    )[0]


def _tombstone(store: SQLiteStore, session_id: str):
    return store._scoped(TENANT)._tombstone(session_id)


@pytest.mark.parametrize(
    ("backend", "schema_tables"),
    (("sqlite", SQLITE_SCHEMA_TABLES), ("postgres", POSTGRES_SCHEMA_TABLES)),
)
def test_deletion_inventory_is_closed_over_backend_schema(backend: str, schema_tables: frozenset[str]) -> None:
    # sk_settings is the database singleton (schema major and write level), not tenant session data.
    expected_session_tables = schema_tables - {"sk_settings"}
    assert set(SESSION_DELETION_TABLES) == expected_session_tables
    assert "sk_settings" not in SESSION_DELETION_TABLES
    assert SESSION_DELETION_TABLES[-1] == "sk_tombstone"
    assert set(PURGE_DELETE_TABLES) == expected_session_tables - {"sk_tombstone"}


def test_forget_and_purge_leave_no_session_memory_projection_plaintext(
    tmp_path: Path,
    key_provider: LocalKeyProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "SESSION_MEMORY_PROJECTION_SECRET"
    workspace = tmp_path / "workspace"
    task = SimpleNamespace(
        metadata={"session_id": "session", "session_memory_storage_dir": ".memory/session"},
        task_id="task",
        use_workspace=True,
        model="scripted",
    )
    memory = session_memory(task, workspace)
    assert memory.storage_path() is None

    receipt = object()
    projection_driver = SimpleNamespace(
        state=SimpleNamespace(
            boundaries={("turn", "session_memory_saved", None): receipt},
            turns={"turn": SimpleNamespace(start=SimpleNamespace(task=task))},
            active_turn_id="turn",
        ),
        runtime=SimpleNamespace(config=SimpleNamespace(workspace=str(workspace))),
        hydrated_definition=lambda _start: SimpleNamespace(task=task),
    )
    monkeypatch.setattr(
        "vv_agent.session.memory.hydrate_boundary_data",
        lambda _driver, _receipt: {"state": {"entries": [{"content": marker}]}},
    )
    save_projection(cast(Any, projection_driver))

    def projection_contains_plaintext() -> bool:
        return any(marker in path.read_text(encoding="utf-8") for path in workspace.rglob("*") if path.is_file())

    assert projection_contains_plaintext() is False
    with _sqlite(tmp_path / "memory-projection.sqlite", key_provider) as (store, signer):
        _create_session(store, key_provider, "session")
        actor = _principal(store, "session")
        deletion = SessionDeletionStore(store, SQLiteContentStore(store, key_provider), signer=signer)
        workspace.mkdir(parents=True, exist_ok=True)
        host_file = workspace / "host-owned.txt"
        host_file.write_text("caller-owned workspace content", encoding="utf-8")

        deletion.forget_session(TENANT, "session", actor=actor)
        assert projection_contains_plaintext() is False
        deletion.purge_session(TENANT, "session", actor=actor)
        assert projection_contains_plaintext() is False
        assert host_file.read_text(encoding="utf-8") == "caller-owned workspace content"


def test_forget_fences_active_and_parked_sessions_and_retries_destroy(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    for phase in ("active", "parked"):
        path = tmp_path / f"forget-{phase}.sqlite"
        with _sqlite(path, key_provider) as (store, signer):
            session_id = f"session-{phase}"
            _create_session(store, key_provider, session_id)
            _pending_input(store, key_provider, session_id)
            adapter, ref = _publish_content(store, key_provider, session_id)
            actor = _principal(store, session_id)
            store._rows(
                "UPDATE sk_session SET phase=%s WHERE tenant_id=%s AND session_id=%s",
                (phase, TENANT, session_id),
            )
            lease = acquire_fixture_lease(
                store.for_tenant(TENANT),
                session_id,
                owner=f"worker-{phase}",
                ttl_ms=60_000,
            )
            assert lease is not None

            original_destroy = key_provider.destroy
            monkeypatch.setattr(
                key_provider,
                "destroy",
                lambda _tenant, _key: KeyDestroyOutcome(
                    False,
                    KeyDestroyStatus.KEY_UNAVAILABLE,
                    None,
                    "provider unavailable",
                ),
            )
            deletion = SessionDeletionStore(store, adapter, signer=signer)
            with pytest.raises(KeyUnavailable):
                deletion.forget_session(TENANT, session_id, actor=actor)

            state, epoch, tombstone_state = store._one(
                "SELECT s.deletion_state,s.lease_epoch,t.state FROM sk_session s "
                "JOIN sk_tombstone t ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id "
                "WHERE s.tenant_id=%s AND s.session_id=%s",
                (TENANT, session_id),
            )
            assert (state, tombstone_state) == ("forgetting", "pending")
            assert epoch > lease.epoch
            assert store._one(
                "SELECT lease_owner FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (TENANT, session_id),
            ) == (None,)

            with pytest.raises(SessionNotLive):
                store.for_tenant(TENANT).read(session_id)
            with pytest.raises(SessionNotLive):
                store.for_tenant(TENANT).read_input(session_id, "pending-input")
            with pytest.raises(SessionNotLive):
                store.for_tenant(TENANT).acquire(session_id, owner="late-worker", ttl_ms=1000)
            with pytest.raises(SessionNotLive), store.for_tenant(TENANT).atomic() as tx:
                tx.push(
                    session_id,
                    InboxItem("late-input", "control", {"action": "cancel"}, actor=actor.to_dict()),
                )
            with pytest.raises(SessionNotLive), store.for_tenant(TENANT).atomic() as tx:
                tx.create(
                    SessionSpec(
                        f"child-{session_id}",
                        actor.to_dict(),
                        actor.attributes.to_dict(),
                        parent_session_id=session_id,
                        parent_operation_id="op/child",
                        attributes=actor.attributes.to_dict(),
                    ),
                    consumers=(),
                )
            with pytest.raises(ContentUnavailable):
                adapter.get(TENANT, session_id, ref)
            with pytest.raises(ContentUnavailable), store.for_tenant(TENANT).atomic():
                adapter.put(TENANT, session_id, b"late publication", ContentLabels())

            monkeypatch.setattr(key_provider, "destroy", original_destroy)
            receipt = deletion.forget_session(TENANT, session_id, actor=actor)
            assert receipt.authenticity is DeletionAuthenticity.SIGNED
            skeleton = deletion.read_skeleton(TENANT, session_id)
            assert skeleton.records and skeleton.inputs
            assert skeleton.tombstone is not None and skeleton.tombstone.state is TombstoneState.COMPLETE
            assert skeleton.chain_digest == skeleton.tombstone.chain_digest
            assert _count(store, "sk_content", session_id) == 0
            assert _count(store, "sk_content_link", session_id) == 0
            assert _count(store, "sk_session_key", session_id) == 0
            assert _count(store, "sk_record", session_id) == 1
            with pytest.raises(ContentUnavailable):
                adapter.get(TENANT, session_id, ref)


def test_forget_crash_after_content_erase_resumes_pending_tombstone(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "forget-crash.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "session")
        adapter, _ref_value = _publish_content(store, key_provider, "session")
        actor = _principal(store, "session")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        original_complete = SQLStore.complete_forget

        def crash(_store: SQLStore, _session_id: str) -> None:
            raise RuntimeError("crash before tombstone completion")

        monkeypatch.setattr(SQLStore, "complete_forget", crash)
        with pytest.raises(RuntimeError, match="crash"):
            deletion.forget_session(TENANT, "session", actor=actor)
        assert store._one(
            "SELECT s.deletion_state,t.state FROM sk_session s JOIN sk_tombstone t "
            "ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id WHERE s.session_id='session'"
        ) == ("forgetting", "pending")
        assert _count(store, "sk_content", "session") == 0
        assert _count(store, "sk_session_key", "session") == 0

        monkeypatch.setattr(SQLStore, "complete_forget", original_complete)
        deletion.forget_session(TENANT, "session", actor=actor)
        deletion.forget_session(TENANT, "session", actor=actor)
        assert store._one("SELECT deletion_state FROM sk_session WHERE session_id='session'") == ("forgotten",)
        assert store._one("SELECT state FROM sk_tombstone WHERE session_id='session'") == ("complete",)


def test_unknown_effect_is_fenced_but_retained_for_skeleton(tmp_path, key_provider: LocalKeyProvider) -> None:
    path = tmp_path / "unknown-effect.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "unknown")
        _append_unknown_effect(store, key_provider, "unknown")
        adapter, _ref_value = _publish_content(store, key_provider, "unknown")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        receipt = deletion.forget_session(TENANT, "unknown", actor=_principal(store, "unknown"))
        assert receipt.remote_effect_state == "unknown"
        with pytest.raises(SessionNotLive):
            store.for_tenant(TENANT).read("unknown")
        skeleton = deletion.read_skeleton(TENANT, "unknown")
        assert [item.record.kind for item in skeleton.records] == [
            "session_created",
            "turn_started",
            "op_planned",
            "op_started",
            "op_unknown",
        ]


def test_forget_cascades_tombstones_and_content_over_the_entire_scope(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    with _sqlite(tmp_path / "forget-scope.sqlite", key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "grandchild", parent="child")
        adapter, child_ref = _publish_content(store, key_provider, "child")
        adapter, grandchild_ref = _publish_content(store, key_provider, "grandchild")
        actor = _principal(store, "root")
        actor_ref = actor.attributes
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        receipt = deletion.forget_session(TENANT, "root", actor=actor)

        assert set(receipt.session_ids) == {"root", "child", "grandchild"}
        states = dict(
            store._rows(
                "SELECT session_id,deletion_state FROM sk_session WHERE tenant_id=%s ORDER BY session_id",
                (TENANT,),
            )
        )
        assert states == {"child": "forgotten", "grandchild": "forgotten", "root": "forgotten"}
        tombstones = deletion.verify_tombstones(TENANT, "root")
        assert {item.session_id for item in tombstones} == {"root", "child", "grandchild"}
        assert {item.scope_root_session_id for item in tombstones} == {"root"}
        assert {item.actor.attributes.key_id for item in tombstones} == {actor_ref.key_id}
        assert actor_ref.key_id != signer.key_id
        assert _count(store, "sk_content", "child") == 0
        assert _count(store, "sk_content", "grandchild") == 0
        assert _count(store, "sk_session_key", "child") == 0
        assert _count(store, "sk_session_key", "grandchild") == 0
        assert _count(store, "sk_record", "root") == 1
        assert _count(store, "sk_record", "child") == 1
        assert _count(store, "sk_record", "grandchild") == 1
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, "child", child_ref)
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, "grandchild", grandchild_ref)

        store._rows(
            "UPDATE sk_tombstone SET scope_root_session_id='other' WHERE tenant_id=%s AND session_id='child'",
            (TENANT,),
        )
        with pytest.raises(LedgerIntegrityError, match="signature"):
            deletion.verify_tombstones(TENANT, "other")


def test_purge_cascades_child_first_and_retains_only_verifiable_tombstones(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "purge.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "grandchild", parent="child")
        _pending_input(store, key_provider, "child")
        adapter, ref = _publish_content(store, key_provider, "grandchild")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        actor = _principal(store, "root")
        receipt = deletion.purge_session(TENANT, "root", actor=actor)
        assert set(receipt.session_ids) == {"root", "child", "grandchild"}
        tombstones = deletion.verify_tombstones(TENANT, "root")
        assert {item.session_id for item in tombstones} == {"root", "child", "grandchild"}
        assert all(item.state is TombstoneState.COMPLETE for item in tombstones)
        for table in PURGE_DELETE_TABLES:
            assert store._one(f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s", (TENANT,))[0] == 0
        assert (
            store._one(
                "SELECT COUNT(*) FROM sk_tombstone WHERE tenant_id=%s AND state='complete'",
                (TENANT,),
            )[0]
            == 3
        )
        assert not hasattr(adapter, "_session_keys")
        with pytest.raises(ContentUnavailable):
            adapter.get(TENANT, "grandchild", ref)

        repeated = deletion.purge_session(TENANT, "root", actor=actor)
        assert set(repeated.session_ids) == {"root", "child", "grandchild"}
        signature = store._one("SELECT signature FROM sk_tombstone WHERE session_id='child'")[0]
        store._rows(
            "UPDATE sk_tombstone SET signature=%s WHERE tenant_id=%s AND session_id='child'",
            (bytes([signature[0] ^ 1]) + signature[1:], TENANT),
        )
        with pytest.raises(LedgerIntegrityError, match="signature"):
            deletion.verify_tombstones(TENANT, "root")


def test_missing_signer_fails_before_fence_and_unsigned_mode_is_explicitly_weak(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "unsigned.sqlite"
    with _sqlite(path, key_provider, signed=False) as (setup, _signer):
        setup.configure_input_admission(allow_unsigned=True)
        _create_session(setup, key_provider, "session")
        _publish_content(setup, key_provider, "session")
        actor = _principal(setup, "session")

    with SQLiteStore.standalone(str(path)) as store:
        adapter = SQLiteContentStore(store, key_provider)
        with pytest.raises(DeletionSignerUnavailable):
            SessionDeletionStore(store, adapter).forget_session(TENANT, "session", actor=actor)
        assert store._one("SELECT deletion_state FROM sk_session WHERE session_id='session'") == ("live",)

    with SQLiteStore.standalone(str(path)) as store:
        store.configure_input_admission(allow_unsigned=True)
        adapter = SQLiteContentStore(store, key_provider)
        deletion = SessionDeletionStore(store, adapter, allow_unsigned=True)
        receipt = deletion.forget_session(TENANT, "session", actor=actor)
        assert receipt.authenticity is DeletionAuthenticity.HASH_ONLY
        tombstone = deletion.verify_tombstones(TENANT, "session")[0]
        assert tombstone.key_id == "hash-only/sha256"
        signature = store._one("SELECT signature FROM sk_tombstone WHERE session_id='session'")[0]
        store._rows(
            "UPDATE sk_tombstone SET signature=%s WHERE tenant_id=%s AND session_id='session'",
            (bytes([signature[0] ^ 1]) + signature[1:], TENANT),
        )
        with pytest.raises(LedgerIntegrityError, match="hash-only tombstone"):
            deletion.verify_tombstones(TENANT, "session")


def test_wrong_tenant_cannot_fence_or_delete_a_session(tmp_path, key_provider: LocalKeyProvider) -> None:
    path = tmp_path / "tenant.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "session")
        adapter, _ref_value = _publish_content(store, key_provider, "session")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        with pytest.raises(PermissionError, match="another tenant"):
            deletion.forget_session(TENANT, "session", actor=_principal(store, "session", "tenant-b"))
        with pytest.raises(Conflict, match="not found in tenant"):
            deletion.forget_session("tenant-b", "session", actor=_principal(store, "session", "tenant-b"))
        assert store._one(
            "SELECT deletion_state FROM sk_session WHERE tenant_id=%s AND session_id='session'",
            (TENANT,),
        ) == ("live",)
        assert store._one(
            "SELECT COUNT(*) FROM sk_tombstone WHERE tenant_id=%s AND session_id='session'",
            (TENANT,),
        ) == (0,)


def test_child_forget_then_parent_forget_cascades_three_levels(tmp_path, key_provider: LocalKeyProvider) -> None:
    path = tmp_path / "child-before-parent-forget.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "grandchild", parent="child")
        retention = _principal(store, "root", subject_id="service/retention")
        compliance = _principal(store, "root", subject_id="service/compliance")
        adapter, _ref_value = _publish_content(store, key_provider, "grandchild")
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        child_receipt = deletion.forget_session(TENANT, "child", actor=retention)
        parent_receipt = deletion.forget_session(TENANT, "root", actor=compliance)

        assert set(child_receipt.session_ids) == {"child", "grandchild"}
        assert set(parent_receipt.session_ids) == {"root", "child", "grandchild"}
        states = dict(store._rows("SELECT session_id,deletion_state FROM sk_session WHERE tenant_id=%s", (TENANT,)))
        assert states == {"root": "forgotten", "child": "forgotten", "grandchild": "forgotten"}
        child_tombstone = _tombstone(store, "child")
        grandchild_tombstone = _tombstone(store, "grandchild")
        root_tombstone = _tombstone(store, "root")
        assert child_tombstone is not None and grandchild_tombstone is not None and root_tombstone is not None
        assert child_tombstone.scope_root_session_id == "root"
        assert grandchild_tombstone.scope_root_session_id == "root"
        assert child_tombstone.actor == compliance
        assert grandchild_tombstone.actor == compliance
        assert root_tombstone.actor == compliance
        assert signer.verify(child_tombstone.key_id, _tombstone_message(child_tombstone), b64decode(child_tombstone.signature))


def test_retention_forget_then_compliance_admin_purge_is_accepted(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "different-actors.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        retention = _principal(store, "root", subject_id="service/retention")
        compliance = _principal(store, "root", subject_id="service/compliance")
        adapter, _ref_value = _publish_content(store, key_provider, "child")
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        forgotten = deletion.forget_session(TENANT, "root", actor=retention)
        purged = deletion.purge_session(TENANT, "root", actor=compliance)

        assert set(forgotten.session_ids) == {"root", "child"}
        assert set(purged.session_ids) == {"root", "child"}
        tombstones = deletion.verify_tombstones(TENANT, "root")
        assert {item.session_id for item in tombstones} == {"root", "child"}
        assert {item.actor.subject_id for item in tombstones} == {compliance.subject_id}
        assert all(item.state is TombstoneState.COMPLETE for item in tombstones)


def test_child_forget_then_parent_purge_preserves_descendant_tombstone_evidence(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "child-before-parent-purge.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "grandchild", parent="child")
        retention = _principal(store, "root", subject_id="service/retention")
        compliance = _principal(store, "root", subject_id="service/compliance")
        adapter, _ref_value = _publish_content(store, key_provider, "grandchild")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        child_forget = deletion.forget_session(TENANT, "child", actor=retention)
        child_tombstone = _tombstone(store, "child")
        assert child_tombstone is not None

        receipt = deletion.purge_session(TENANT, "root", actor=compliance)

        assert set(child_forget.session_ids) == {"child", "grandchild"}
        assert set(receipt.session_ids) == {"root", "child", "grandchild"}
        retained = {item.session_id: item for item in deletion.verify_tombstones(TENANT, "root")}
        assert set(retained) == {"root", "child", "grandchild"}
        assert retained["child"].scope_root_session_id == "root"
        assert retained["child"].actor == compliance


def test_repeated_forget_reuses_complete_tombstones_without_destroying_again(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "repeat-forget.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "session")
        actor = _principal(store, "session")
        adapter, _ref_value = _publish_content(store, key_provider, "session")
        calls: list[str] = []
        original_destroy = key_provider.destroy

        def counted_destroy(tenant_id: str, key_id: str):
            calls.append(key_id)
            return original_destroy(tenant_id, key_id)

        monkeypatch.setattr(key_provider, "destroy", counted_destroy)
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        first = deletion.forget_session(TENANT, "session", actor=actor)
        first_tombstone = _tombstone(store, "session")
        assert first_tombstone is not None
        second = deletion.forget_session(TENANT, "session", actor=actor)
        second_tombstone = _tombstone(store, "session")

        assert first == second
        assert calls
        assert len(calls) == 1
        assert first_tombstone.signature == second_tombstone.signature


def test_external_locator_cleanup_failure_stays_pending_and_retry_converges(
    tmp_path,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "external-cleanup.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        adapter, ref = _publish_content(store, key_provider, "child")
        store._rows(
            "UPDATE sk_content SET ciphertext=NULL,object_locator=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (b"remote://content/child", TENANT, "child", ref.content_id),
        )
        calls: list[bytes] = []

        def cleaner(_tenant_id: str, _session_id: str, locator: bytes) -> None:
            calls.append(locator)
            if len(calls) == 1:
                raise RuntimeError("remote deletion outcome unknown")

        adapter.set_object_locator_cleaner(cleaner)
        deletion = SessionDeletionStore(store, adapter, signer=signer)

        retention = _principal(store, "root", subject_id="service/retention")
        with pytest.raises(RuntimeError, match="outcome unknown"):
            deletion.forget_session(TENANT, "root", actor=retention)
        pending = store._rows(
            "SELECT s.deletion_state,t.state FROM sk_session s JOIN sk_tombstone t "
            "ON t.tenant_id=s.tenant_id AND t.session_id=s.session_id WHERE s.tenant_id=%s",
            (TENANT,),
        )
        assert set(pending) == {("forgetting", "pending")}

        receipt = deletion.forget_session(TENANT, "root", actor=retention)

        assert set(receipt.session_ids) == {"root", "child"}
        assert calls == [b"remote://content/child", b"remote://content/child"]
        assert all(item.state is TombstoneState.COMPLETE for item in deletion.verify_tombstones(TENANT, "root"))


def test_incomplete_locked_scope_is_rejected_before_content_erasure(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "incomplete-scope.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "grandchild", parent="child")
        adapter, _ref_value = _publish_content(store, key_provider, "grandchild")
        original_begin = SQLStore.begin_forget

        def incomplete_scope(store: SQLStore, *args, **kwargs):
            scope = original_begin(store, *args, **kwargs)
            return tuple(session_id for session_id in scope if session_id != "grandchild")

        session_ids = ("root", "child", "grandchild")
        content_tables = ("sk_session_key", "sk_content", "sk_content_link")
        actor = _principal(store, "root")
        before_counts = {
            (session_id, table): _count(store, table, session_id) for session_id in session_ids for table in content_tables
        }
        assert before_counts[("grandchild", "sk_content")] > 1

        monkeypatch.setattr(SQLStore, "begin_forget", incomplete_scope)
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        with pytest.raises(LedgerIntegrityError, match="root descendant inventory"):
            deletion.forget_session(TENANT, "root", actor=actor)
        after_counts = {
            (session_id, table): _count(store, table, session_id) for session_id in session_ids for table in content_tables
        }
        assert after_counts == before_counts


def test_signed_tombstone_with_cross_scope_root_is_rejected(tmp_path, key_provider: LocalKeyProvider) -> None:
    path = tmp_path / "forged-scope.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        adapter, _ref_value = _publish_content(store, key_provider, "child")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        retention = _principal(store, "root", subject_id="service/retention")
        deletion.forget_session(TENANT, "child", actor=retention)
        tombstone = _tombstone(store, "child")
        assert tombstone is not None
        forged_unsigned = replace(tombstone, scope_root_session_id="unrelated")
        forged = replace(
            forged_unsigned,
            signature=b64encode(signer.bind_key().sign(_tombstone_message(forged_unsigned))).decode("ascii"),
        )
        store._rows(
            "UPDATE sk_tombstone SET scope_root_session_id=%s,signature=%s WHERE tenant_id=%s AND session_id=%s",
            (forged.scope_root_session_id, b64decode(forged.signature), TENANT, "child"),
        )

        scoped = deletion._scoped(TENANT)
        authority = deletion._authority(scoped)
        with pytest.raises(LedgerIntegrityError, match="scope root"):
            deletion._verify_scope_evidence(
                scoped,
                TENANT,
                "child",
                ("child",),
                require_present=True,
                expected_states={"forgotten"},
                require_complete=True,
                authority=authority,
            )


def test_signed_tombstone_with_wrong_head_is_rejected(tmp_path, key_provider: LocalKeyProvider) -> None:
    path = tmp_path / "forged-head.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        assert signer is not None
        _create_session(store, key_provider, "session")
        adapter, _ref_value = _publish_content(store, key_provider, "session")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        deletion_actor = _principal(store, "session")
        compliance = _principal(store, "session", subject_id="service/compliance")
        deletion.forget_session(TENANT, "session", actor=deletion_actor)
        tombstone = _tombstone(store, "session")
        assert tombstone is not None
        forged_unsigned = replace(tombstone, final_seq=tombstone.final_seq + 1)
        forged = replace(
            forged_unsigned,
            signature=b64encode(signer.bind_key().sign(_tombstone_message(forged_unsigned))).decode("ascii"),
        )
        store._rows(
            "UPDATE sk_tombstone SET final_seq=%s,signature=%s WHERE tenant_id=%s AND session_id=%s",
            (forged.final_seq, b64decode(forged.signature), TENANT, "session"),
        )

        with pytest.raises(LedgerIntegrityError, match="final head"):
            deletion.purge_session(TENANT, "session", actor=compliance)


def test_scope_validation_uses_one_root_bounded_query(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "bounded-query.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        _create_session(store, key_provider, "child", parent="root")
        _create_session(store, key_provider, "unrelated")
        adapter, _ref_value = _publish_content(store, key_provider, "child")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        calls: list[str] = []
        original_rows = store._rows

        def counted_rows(query: str, params: tuple = ()):
            calls.append(query)
            return original_rows(query, params)

        monkeypatch.setattr(store, "_rows", counted_rows)
        members = deletion._read_scope(
            store,
            TENANT,
            "root",
            ("root", "child"),
            require_present=True,
        )

        assert {member.session_id for member in members} == {"root", "child"}
        assert len(calls) == 1
        assert "WITH RECURSIVE deletion_scope" in calls[0]


def test_deletion_table_counts_are_batched_across_scope_members(
    tmp_path,
    monkeypatch,
    key_provider: LocalKeyProvider,
) -> None:
    path = tmp_path / "batched-counts.sqlite"
    with _sqlite(path, key_provider) as (store, signer):
        _create_session(store, key_provider, "root")
        adapter, _ref_value = _publish_content(store, key_provider, "root")
        deletion = SessionDeletionStore(store, adapter, signer=signer)
        calls: list[str] = []
        original_rows = store._rows

        def counted_rows(query: str, params: tuple = ()):
            calls.append(query)
            return original_rows(query, params)

        monkeypatch.setattr(store, "_rows", counted_rows)
        counts = deletion._count_rows_for_sessions(
            store,
            TENANT,
            PURGE_DELETE_TABLES,
            tuple(f"session-{index}" for index in range(20)),
        )

        assert len(calls) == 1
        assert counts == dict.fromkeys(PURGE_DELETE_TABLES, 0)
