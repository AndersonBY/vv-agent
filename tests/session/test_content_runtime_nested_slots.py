from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.kernel._child_store import ChildSession, create_child
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import fixture_child_spec, record

MASTER_KEY = b"n" * 32
PARENT = "nested-parent"
CHILD = "nested-child"
SESSION = "nested-provider-slots"
PROVIDER_SLOTS = (
    "provider.result",
    "provider.usage",
    "provider.evidence",
    "provider.handle.job_id",
    "provider.handle.evidence",
    "provider.handle.query_ref",
    "provider.handle.cancel_ref",
)


def _links(store: SQLiteStore, session_id: str) -> dict[str, object]:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=session_id)
        return {link.slot: link.ref for link in writer.durable.read_links("default", session_id, record_seq=1)}


def test_creation_links_seed_and_child_admission_nested_slots(tmp_path: Path) -> None:
    seed_message = {"role": "user", "content": "seed"}
    initial_message = {"role": "user", "content": "child-initial"}
    input_message = {"role": "user", "content": "child-input"}
    parent_attributes = {
        "seed": {"messages": [seed_message]},
    }
    child_attributes = {
        "child_admission": {
            "definition": {
                "task": {
                    "initial_messages": [initial_message],
                    "metadata": {"vv_session": {"input_messages": [input_message]}},
                }
            }
        },
    }

    with SQLiteStore.standalone(str(tmp_path / "nested-slots.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "nested-slots-keys.sqlite3"),
        )
        driver.create(PARENT, "/workspace", parent_attributes)
        child = ChildSession(
            replace(fixture_child_spec(store, PARENT, CHILD), attributes=child_attributes),
            "child content",
        )
        operation_id = "operation/nested-child"
        plan = record("op_planned", sid=PARENT, tid=f"{PARENT}/turn/start", oid=operation_id)
        with driver.store.atomic() as tx:
            create_child(tx, child, plan, generation=1)

        child_created = store.read(CHILD, limit=1).records[0].record
        assert child_created.payload["parent_session_id"] == PARENT
        assert child_created.payload["parent_operation_id"] == operation_id

        assert {
            "session.attributes.child_admission.definition.task.initial_messages[0]",
            "session.attributes.child_admission.definition.task.metadata.vv_session.input_messages[0]",
        } <= set(_links(store, CHILD))
        assert "session.attributes.seed.messages[0]" in _links(store, PARENT)
        assert not any("child_admission" in slot for slot in _links(store, PARENT))


def test_record_links_keep_provider_slots(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "provider-slots.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "provider-slots-keys.sqlite3"),
        )
        driver.create(SESSION, "/workspace", {})

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            for slot in PROVIDER_SLOTS:
                writer.link_record(1, writer.put_json({"slot": slot}), slot)

        assert set(PROVIDER_SLOTS) <= set(_links(store, SESSION))


@pytest.mark.parametrize(
    "slot",
    (
        "session.attributes.child_admission.definition.task.unexpected[0]",
        "session.attributes.child_admission.definition.task.initial_messages[*]",
        "session.attributes.child_admission.definition.task.**",
        "session.attributes.child_admission.definition.task.initial_messages[0].content",
    ),
)
def test_writer_rejects_undeclared_or_recursive_slots_before_sql_link(tmp_path: Path, slot: str) -> None:
    with SQLiteStore.standalone(str(tmp_path / "rejected-slot.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "rejected-slot-keys.sqlite3"),
        )
        driver.create(SESSION, "/workspace", {})

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.link_record(1, writer.put_json({"slot": slot}), slot)

        assert slot not in _links(store, SESSION)
