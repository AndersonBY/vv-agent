from pathlib import Path

import pytest

from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"n" * 32
SESSION = "nested-slots"
PROVIDER_SLOTS = (
    "provider.result",
    "provider.usage",
    "provider.evidence",
    "provider.handle.job_id",
    "provider.handle.evidence",
    "provider.handle.query_ref",
    "provider.handle.cancel_ref",
)


def _links(store: SQLiteStore) -> dict[str, object]:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
        return {link.slot: link.ref for link in writer.durable.read_links("default", SESSION, record_seq=1)}


def test_creation_links_seed_and_child_admission_nested_slots(tmp_path: Path) -> None:
    seed_message = {"role": "user", "content": "seed"}
    initial_message = {"role": "user", "content": "child-initial"}
    input_message = {"role": "user", "content": "child-input"}
    attributes = {
        "seed": {"messages": [seed_message]},
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
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        driver.create(SESSION, "/workspace", attributes)

        assert {
            "session.attributes.seed.messages[0]",
            "session.attributes.child_admission.definition.task.initial_messages[0]",
            "session.attributes.child_admission.definition.task.metadata.vv_session.input_messages[0]",
        } <= set(_links(store))


def test_record_links_keep_provider_slots(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "provider-slots.sqlite")) as store:
        store.install_schema()
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        driver.create(SESSION, "/workspace", {})

        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            for slot in PROVIDER_SLOTS:
                writer.link_record(1, writer.put_json({"slot": slot}), slot)

        assert set(PROVIDER_SLOTS) <= set(_links(store))


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
        driver = SessionDriver(store=store, content_key_provider=LocalKeyProvider(MASTER_KEY))
        driver.create(SESSION, "/workspace", {})

        with pytest.raises(ValueError, match="stable declared slot"), store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id="default", session_id=SESSION)
            writer.link_record(1, writer.put_json({"slot": slot}), slot)

        assert slot not in _links(store)
