import base64
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from vv_agent.kernel._child_store import ChildSession, create_child
from vv_agent.kernel._content import (
    Classification,
    ContentCleanupUnavailable,
    ContentIntegrityError,
    ContentLabels,
    ContentRef,
    ContentUnavailable,
    ExportLineKind,
    KeyDestroyStatus,
    KeyUnavailable,
    WrappedDataKey,
)
from vv_agent.session.app_server import _KernelThreadStore
from vv_agent.session.content_runtime import (
    SESSION_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentRuntimeUnavailable,
    SessionContentWriter,
    _DeferredContentBackend,
)
from vv_agent.session.content_store import ContentExportRef, LocalKeyProvider, _content_aad
from vv_agent.session.postgres import PostgresContentStore, PostgresStore
from vv_agent.session.records import InboxItem, Record, SessionSpec
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import principal

MASTER_KEY = b"r" * 32


def _unwrap_in_new_process(
    state_path: Path,
    wrapped: WrappedDataKey,
    *,
    tenant_id: str = "default",
    session_id: str = "session-a",
) -> subprocess.CompletedProcess[str]:
    script = """
import base64
import sys

from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.kernel._content import WrappedDataKey

provider = LocalKeyProvider(base64.b64decode(sys.argv[1]), state_path=sys.argv[2])
wrapped = WrappedDataKey(sys.argv[3], sys.argv[4], sys.argv[5], base64.b64decode(sys.argv[6]))
provider.unwrap(sys.argv[3], sys.argv[4], wrapped)
"""
    return subprocess.run(
        (
            sys.executable,
            "-c",
            script,
            base64.b64encode(MASTER_KEY).decode("ascii"),
            str(state_path),
            tenant_id,
            session_id,
            wrapped.key_id,
            base64.b64encode(wrapped.wrapped_key).decode("ascii"),
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def _session_spec(writer: SessionContentWriter, session_id: str = "session-a") -> SessionSpec:
    principal, _ = writer.service_principal("test-host")
    return SessionSpec(
        session_id,
        principal,
        writer.put_json("/workspace").to_dict(),
        attributes=writer.put_json({"purpose": "content-runtime-test"}).to_dict(),
    )


def _counts(store: SQLiteStore) -> tuple[int, int, int, int]:
    return tuple(
        store._rows(f"SELECT COUNT(*) FROM {table}")[0][0]
        for table in ("sk_session", "sk_session_key", "sk_content", "sk_content_link")
    )


def _create_empty_session(
    store: SQLiteStore,
    provider: LocalKeyProvider,
    *,
    tenant_id: str = "default",
    session_id: str = "session-a",
) -> None:
    with store.for_tenant(tenant_id).atomic() as tx:
        writer = SessionContentWriter(tx=tx, tenant_id=tenant_id, session_id=session_id, key_provider=provider)
        writer.create_session(lambda content: _session_spec(content, session_id), consumers=())


def test_deferred_content_backend_rejects_unrecognized_admission_sql() -> None:
    backend = SimpleNamespace(lock_clause="")
    deferred = _DeferredContentBackend(backend, tenant_id="default", session_id="session-a")

    with pytest.raises(RuntimeError, match="unexpected content SQL before session admission"):
        deferred.fetchone("SELECT 1 FROM sk_content WHERE tenant_id={p}", ("default",))
    with pytest.raises(RuntimeError, match="unexpected content SQL before session admission"):
        deferred.run("DELETE FROM sk_content WHERE tenant_id={p}", ("default",))


def test_sqlite_ingress_persists_key_content_links_and_reopens(tmp_path: Path) -> None:
    path = str(tmp_path / "content.sqlite")
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(path) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=provider)
        driver.create("session-a", "/workspace", {"purpose": "content-runtime-test"})
        driver.push("session-a", InboxItem("input-a", "user", {"content": "trusted input"}))

        stored = store.read_input("session-a", "input-a")
        assert stored is not None
        item = stored.item
        ref = ContentRef.from_dict(item.payload["content"])
        assert ref.content_id
        assert not ref.content_id.startswith("content/local/")
        assert not ref.key_id.startswith("key/local/")
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            assert content.get("default", "session-a", ref) == b'"trusted input"'
            links = content.read_links("default", "session-a", input_id="input-a")
        assert {link.slot for link in links} == {"input.content", "input.actor.attributes"}
        assert _counts(store) == (1, 1, 4, 5)
        assert b"trusted input" not in store._rows("SELECT body FROM sk_inbox")[0][0]
        assert all(
            b"trusted input" not in bytes(ciphertext) for (ciphertext,) in store._rows("SELECT ciphertext FROM sk_content")
        )
        driver.close()

    recovered_provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(path) as recovered:
        recovered_input = recovered.read_input("session-a", "input-a")
        assert recovered_input is not None
        recovered_item = recovered_input.item
        with recovered.atomic() as tx:
            content = SQLiteContentStore(tx, recovered_provider)
            assert (
                content.get("default", "session-a", ContentRef.from_dict(recovered_item.payload["content"])) == b'"trusted input"'
            )


def test_local_key_destroy_is_durable_across_provider_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "local-key-state.sqlite3"
    provider = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    with SQLiteStore.standalone(str(tmp_path / "key-lifecycle.sqlite")) as store:
        store.install_schema()
        _create_empty_session(store, provider)
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put("default", "session-a", b"durable key", ContentLabels())
        row = store._one(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            ("default", "session-a"),
        )

    wrapped = WrappedDataKey("default", "session-a", row[0], bytes(row[2]))
    recovered = LocalKeyProvider(MASTER_KEY, state_path=state_path)
    assert recovered.unwrap("default", "session-a", wrapped) != b""
    assert _unwrap_in_new_process(state_path, wrapped).returncode == 0
    assert provider.destroy("default", wrapped.key_id).status is KeyDestroyStatus.DESTROYED
    assert recovered.destroy("default", wrapped.key_id).status is KeyDestroyStatus.ALREADY_DESTROYED
    with pytest.raises(KeyUnavailable, match="unavailable"):
        recovered.unwrap("default", "session-a", wrapped)
    rejected = _unwrap_in_new_process(state_path, wrapped)
    assert rejected.returncode != 0
    assert "unavailable" in rejected.stderr

    with (
        SQLiteStore.standalone(str(tmp_path / "key-lifecycle.sqlite")) as reopened,
        reopened.atomic() as tx,
        pytest.raises(KeyUnavailable),
    ):
        SQLiteContentStore(tx, recovered).get("default", "session-a", ref)


def test_sql_content_export_lines_hydrate_only_explicit_references(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "export-key-state.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "export-content.sqlite")) as store:
        store.install_schema()
        _create_empty_session(store, provider)
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put("default", "session-a", b'{"text":"export me"}', ContentLabels())
            tx.push(
                "session-a",
                InboxItem("export-input", "user", {"content": ref.to_dict()}, actor=principal("export-input")),
            )
            content.link_input("default", "session-a", "export-input", ref, "input.content")
            request = ContentExportRef(None, "export-input", "input.content", ref)
            lines = content.export_content_lines("default", "session-a", (request,))

        assert len(lines) == 1
        assert lines[0].kind is ExportLineKind.CONTENT
        assert lines[0].value == {
            "line": "content",
            "content_id": ref.content_id,
            "key_id": ref.key_id,
            "slot": "input.content",
            "decoded_value": {"text": "export me"},
        }

        store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "session-a", ref.content_id),
        )
        store._rows(
            "DELETE FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "session-a", ref.content_id),
        )
        with store.atomic() as tx, pytest.raises(ContentUnavailable, match="content is unavailable"):
            SQLiteContentStore(tx, provider).export_content_lines(
                "default",
                "session-a",
                (request,),
            )

        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            key_bound_ref = content.put("default", "session-a", b'{"text":"key bound"}', ContentLabels())
            content.link_input("default", "session-a", "export-input", key_bound_ref, "input.selection")
            provider.destroy("default", key_bound_ref.key_id)
            with pytest.raises(KeyUnavailable):
                content.export_content_lines(
                    "default",
                    "session-a",
                    (ContentExportRef(None, "export-input", "input.selection", key_bound_ref),),
                )


def test_sql_content_export_requires_the_persisted_owner_and_slot(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "export-binding-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "export-binding.sqlite")) as store:
        store.install_schema()
        _create_empty_session(store, provider)
        _create_empty_session(store, provider, session_id="session-b")
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put("default", "session-a", b'{"owner":"input"}', ContentLabels())
            tx.push(
                "session-a",
                InboxItem("owner-input", "user", {"content": ref.to_dict()}, actor=principal("owner-input")),
            )
            tx.push(
                "session-a",
                InboxItem("other-input", "user", {"content": ref.to_dict()}, actor=principal("other-input")),
            )
            content.link_input("default", "session-a", "owner-input", ref, "input.content")

            with pytest.raises(ContentUnavailable, match="content link is unavailable"):
                content.export_content_lines(
                    "default",
                    "session-a",
                    (ContentExportRef(None, "other-input", "input.content", ref),),
                )
            with pytest.raises(ContentUnavailable, match="content link is unavailable"):
                content.export_content_lines(
                    "default",
                    "session-a",
                    (ContentExportRef(None, "owner-input", "input.selection", ref),),
                )
            with pytest.raises(ContentUnavailable, match="content is unavailable"):
                content.export_content_lines(
                    "default",
                    "session-b",
                    (ContentExportRef(None, "owner-input", "input.content", ref),),
                )
            with pytest.raises(ContentUnavailable, match="content session is unavailable"):
                content.export_content_lines(
                    "tenant-b",
                    "session-a",
                    (ContentExportRef(None, "owner-input", "input.content", ref),),
                )


def test_content_mac_and_aad_bind_tenant_and_session(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "aad-key-state.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "aad.sqlite")) as store:
        store.install_schema()
        for tenant_id, session_id in (
            ("default", "session-a"),
            ("default", "session-b"),
            ("tenant-b", "session-a"),
        ):
            _create_empty_session(store, provider, tenant_id=tenant_id, session_id=session_id)

        refs: dict[tuple[str, str], ContentRef] = {}
        for tenant_id, session_id in (
            ("default", "session-a"),
            ("default", "session-b"),
            ("tenant-b", "session-a"),
        ):
            with store.for_tenant(tenant_id).atomic():
                refs[(tenant_id, session_id)] = SQLiteContentStore(store, provider).put(
                    tenant_id,
                    session_id,
                    b"same plaintext",
                    ContentLabels(classification=Classification.CONFIDENTIAL),
                )

        assert refs[("default", "session-a")].mac != refs[("default", "session-b")].mac
        assert refs[("default", "session-a")].mac != refs[("tenant-b", "session-a")].mac
        aad = json.loads(
            _content_aad(
                "default",
                "session-a",
                refs[("default", "session-a")].content_id,
                refs[("default", "session-a")].key_id,
                refs[("default", "session-a")].mac,
                refs[("default", "session-a")].size,
                refs[("default", "session-a")].labels,
            )
        )
        assert aad["mac"] == refs[("default", "session-a")].mac

        source = store._one(
            "SELECT nonce,ciphertext FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (
                "default",
                "session-a",
                refs[("default", "session-a")].content_id,
            ),
        )
        for tenant_id, session_id in (("default", "session-b"), ("tenant-b", "session-a")):
            store._rows(
                "UPDATE sk_content SET nonce=%s,ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (source[0], source[1], tenant_id, session_id, refs[(tenant_id, session_id)].content_id),
            )
            with store.for_tenant(tenant_id).atomic(), pytest.raises(ContentIntegrityError, match="AEAD"):
                SQLiteContentStore(store, provider).get(tenant_id, session_id, refs[(tenant_id, session_id)])


def test_external_content_cleanup_requires_capability_and_remains_retryable(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "object-key-state.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "object-cleanup.sqlite")) as store:
        store.install_schema()
        _create_empty_session(store, provider)
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put("default", "session-a", b"remote object", ContentLabels())
        store._rows(
            "UPDATE sk_content SET ciphertext=NULL,object_locator=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (b"remote://content/object-1", "default", "session-a", ref.content_id),
        )
        content_count = store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
            ("default", "session-a"),
        )[0]

        with store.atomic() as tx, pytest.raises(ContentCleanupUnavailable):
            SQLiteContentStore(tx, provider).forget_session("default", "session-a")
        assert store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
            ("default", "session-a"),
        ) == (content_count,)
        assert store._one(
            "SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            ("default", "session-a"),
        ) == (1,)

        def unknown(*_args: object) -> None:
            raise RuntimeError("remote deletion outcome unknown")

        with store.atomic() as tx, pytest.raises(RuntimeError, match="outcome unknown"):
            unknown_cleaner = SQLiteContentStore(tx, provider)
            unknown_cleaner.set_object_locator_cleaner(unknown)
            unknown_cleaner.forget_session("default", "session-a")
        assert store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s",
            ("default", "session-a"),
        ) == (content_count,)

        deleted: list[bytes] = []
        cleaner = SQLiteContentStore(store, provider)
        cleaner.set_object_locator_cleaner(lambda _tenant, _session, locator: deleted.append(locator))
        with store.atomic():
            cleaner.forget_session("default", "session-a")
        assert deleted == [b"remote://content/object-1"]
        with store.atomic():
            cleaner.forget_session("default", "session-a")
        assert deleted == [b"remote://content/object-1"]
        for table in ("sk_session_key", "sk_content", "sk_content_link"):
            assert store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s AND session_id=%s",
                ("default", "session-a"),
            ) == (0,)


def test_garbage_collection_cleans_external_objects_before_removing_pending_rows(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "gc-object-keys.sqlite3")
    with SQLiteStore.standalone(str(tmp_path / "gc-object.sqlite")) as store:
        store.install_schema()
        _create_empty_session(store, provider)
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            ref = content.put("default", "session-a", b"staged external object", ContentLabels())
        store._rows(
            "UPDATE sk_content SET ciphertext=NULL,object_locator=%s,pin_until_ms=%s "
            "WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            (b"object://staged/one", store._now() - 1, "default", "session-a", ref.content_id),
        )

        with store.atomic() as tx:
            assert SQLiteContentStore(tx, provider).collect_garbage("default", "session-a") == 0
        assert store._one(
            "SELECT state FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "session-a", ref.content_id),
        ) == ("gc_pending",)

        def unknown(*_args: object) -> None:
            raise RuntimeError("remote deletion outcome unknown")

        with store.atomic() as tx:
            retry = SQLiteContentStore(tx, provider)
            retry.set_object_locator_cleaner(unknown)
            assert retry.collect_garbage("default", "session-a") == 0
        assert store._one(
            "SELECT state FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "session-a", ref.content_id),
        ) == ("gc_pending",)

        deleted: list[bytes] = []
        with store.atomic() as tx:
            cleaner = SQLiteContentStore(tx, provider)
            cleaner.set_object_locator_cleaner(lambda _tenant, _session, locator: deleted.append(locator))
            assert cleaner.collect_garbage("default", "session-a") == 1
        assert deleted == [b"object://staged/one"]
        assert store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "session-a", ref.content_id),
        ) == (0,)


@pytest.mark.persistent_store
def test_garbage_collection_external_locator_retry_on_each_sql_backend(store, tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "persistent-gc-object-keys.sqlite3")
    driver = SessionDriver(store=store, content_key_provider=provider)
    driver.create("session-a", "/workspace", {})
    adapter = PostgresContentStore(store, provider) if isinstance(store, PostgresStore) else SQLiteContentStore(store, provider)
    with store.atomic():
        ref = adapter.put("default", "session-a", b"staged remote object", ContentLabels())
    store._rows(
        "UPDATE sk_content SET ciphertext=NULL,object_locator=%s,pin_until_ms=%s "
        "WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
        (b"object://persistent/staged", store._now() - 1, "default", "session-a", ref.content_id),
    )

    with store.atomic():
        assert adapter.collect_garbage("default", "session-a") == 0
    assert store._one(
        "SELECT state FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
        ("default", "session-a", ref.content_id),
    ) == ("gc_pending",)

    deleted: list[bytes] = []
    adapter.set_object_locator_cleaner(lambda _tenant, _session, locator: deleted.append(locator))
    with store.atomic():
        assert adapter.collect_garbage("default", "session-a") == 1
    assert deleted == [b"object://persistent/staged"]


def test_content_rejects_cross_tenant_and_cross_session_reads(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "isolation.sqlite")) as store:
        store.install_schema()
        with store.atomic() as tx:
            writer = SessionContentWriter(
                tx=tx,
                tenant_id="default",
                session_id="session-a",
                key_provider=provider,
            )
            writer.create_session(lambda content: _session_spec(content), consumers=())
            ref = writer.put_json("tenant-local")

        for tenant_id, session_id in (("tenant-b", "session-a"), ("default", "session-b")):
            with pytest.raises(ContentUnavailable), store.atomic() as tx:
                SQLiteContentStore(tx, provider).get(tenant_id, session_id, ref)


def test_create_and_push_rollback_leave_no_session_key_content_or_link(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "rollback.sqlite")) as store:
        store.install_schema()
        with pytest.raises(RuntimeError, match="rollback"), store.atomic() as tx:
            writer = SessionContentWriter(
                tx=tx,
                tenant_id="default",
                session_id="session-a",
                key_provider=provider,
            )
            writer.create_session(lambda content: _session_spec(content), consumers=())
            ref = writer.put_json("discard me")
            tx.push(
                "session-a",
                InboxItem(
                    "input-a",
                    "user",
                    {"content": ref.to_dict()},
                    actor=writer.service_principal("test-host")[0],
                ),
            )
            writer.link_input_content("input-a", ref)
            raise RuntimeError("rollback")

        assert _counts(store) == (0, 0, 0, 0)


def test_writer_without_bound_key_provider_fails_closed(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "missing-provider.sqlite")) as store:
        store.install_schema()
        with pytest.raises(ContentRuntimeUnavailable, match="key provider"), store.atomic() as tx:
            SessionContentWriter.bind(tx, tenant_id="default", session_id="session-a")


def test_app_server_thread_creation_publishes_real_creation_content(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "app-server.sqlite")) as store:
        store.install_schema()
        kernel = SessionDriver(store=store, content_key_provider=provider)
        threads = _KernelThreadStore(kernel)
        result = threads.create_thread_receipt(
            input_id="thread-a",
            agent_key="default",
            cwd="/workspace",
            metadata={"source": "host"},
        )
        assert result["accepted"] is True
        creation = store.read("thread-a", limit=1).records[0]
        attributes = kernel.read_record_content(
            "thread-a",
            creation.seq,
            creation.record.payload["attributes"],
            slot=SESSION_ATTRIBUTES,
        )
        workspace = kernel.read_record_content(
            "thread-a",
            creation.seq,
            creation.record.payload["workspace"],
            slot=SESSION_WORKSPACE,
        )
        assert workspace == "/workspace"
        assert attributes["app_server"]["metadata"] == {"source": "host"}


def test_app_server_projects_wire_input_from_the_real_input_link(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "app-server-input.sqlite")) as store:
        store.install_schema()
        kernel = SessionDriver(store=store, content_key_provider=provider)
        threads = _KernelThreadStore(kernel)
        threads.create_thread_receipt(input_id="thread-a", agent_key="default")
        kernel.push(
            "thread-a",
            InboxItem(
                "turn-a",
                "user",
                {
                    "content": {
                        "text": "hello",
                        "messages": [],
                        "app_server": {"input": [{"type": "text", "text": "hello"}]},
                    }
                },
            ),
            actor_subject_id="app-server",
        )
        snapshot = threads.read_thread("thread-a")
        assert snapshot.turns[0].input == [{"type": "text", "text": "hello"}]


def test_child_admission_copies_parent_content_into_child_key_and_links(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "child.sqlite")) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=provider)
        driver.create("parent", "/workspace", {"seed": {}})
        driver.push("parent", InboxItem("parent-input", "user", {"content": "parent"}))
        parent_creation = store.read("parent", limit=1).records[0].record
        child = ChildSession(
            SessionSpec(
                "child",
                parent_creation.payload["principal"],
                parent_creation.payload["workspace"],
                attributes={"child_admission": {"mode": "configured"}},
            ),
            "child task",
        )
        plan = cast(
            Record,
            SimpleNamespace(
                session_id="parent",
                turn_id="parent/turn/start",
                operation_id="operation-a",
                attempt=1,
            ),
        )
        with driver.store.atomic() as tx:
            handle = create_child(tx, child, plan, generation=1)

        assert handle["session_id"] == "child"
        with store.atomic() as tx:
            content = SQLiteContentStore(tx, provider)
            child_links = content.read_links("default", "child", input_id="start")
            child_workspace = content.get(
                "default",
                "child",
                ContentRef.from_dict(store.read("child", limit=1).records[0].record.payload["workspace"]),
            )
        assert {link.slot for link in child_links} == {"input.content", "input.actor.attributes"}
        assert child_workspace == b'"/workspace"'
        assert (
            store._rows("SELECT COUNT(*) FROM sk_session_key WHERE tenant_id=%s AND session_id=%s", ("default", "child"))[0][0]
            == 1
        )
        assert store._rows(
            "SELECT parent_session_id,parent_operation_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            ("default", "child"),
        ) == [("parent", "operation-a")]
        with store.atomic() as tx:
            child_writer = SessionContentWriter(tx=tx, tenant_id="default", session_id="child", key_provider=provider)
            parent_writer = SessionContentWriter(tx=tx, tenant_id="default", session_id="parent", key_provider=provider)
            child_result = child_writer.put_json({"text": "child result"})
            parent_result = parent_writer.copy_json_from(child_writer, child_result)
            assert parent_result.key_id != child_result.key_id
            assert parent_result.content_id != child_result.content_id
            parent_writer.link_child_result("parent-input", parent_result)
            assert parent_writer.get_json(parent_result) == {"text": "child result"}

        cross_tenant = ChildSession(
            SessionSpec(
                "foreign-child",
                parent_creation.payload["principal"] | {"tenant_id": "tenant-b"},
                parent_creation.payload["workspace"],
                attributes={},
            ),
            "foreign task",
        )
        with pytest.raises(ValueError, match="transaction tenant"), driver.store.atomic() as tx:
            create_child(tx, cross_tenant, plan, generation=1)


def test_child_admission_rejects_transaction_without_content_capability(tmp_path: Path) -> None:
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(tmp_path / "child-capability.sqlite")) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=provider)
        driver.create("parent", "/workspace", {"seed": {}})
        parent_creation = store.read("parent", limit=1).records[0].record
        child = ChildSession(
            SessionSpec(
                "child",
                parent_creation.payload["principal"],
                parent_creation.payload["workspace"],
                attributes={},
            ),
            "child task",
        )
        plan = cast(
            Record,
            SimpleNamespace(
                session_id="parent",
                turn_id="parent/turn/start",
                operation_id="operation-a",
                attempt=1,
            ),
        )

        with pytest.raises(ValueError, match="transaction capability"), store.atomic() as tx:
            create_child(tx, child, plan, generation=1)
