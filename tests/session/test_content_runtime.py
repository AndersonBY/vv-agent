from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from vv_agent.kernel._child_store import ChildSession, create_child
from vv_agent.kernel._content import ContentRef, ContentUnavailable
from vv_agent.session.app_server import _KernelThreadStore
from vv_agent.session.content_runtime import (
    SESSION_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentRuntimeUnavailable,
    SessionContentWriter,
    _DeferredContentBackend,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import InboxItem, Record, SessionSpec
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"r" * 32


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
