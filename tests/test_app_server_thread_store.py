from __future__ import annotations

from pathlib import Path

from vv_agent.app_server import ThreadItem
from vv_agent.app_server.thread_store import ThreadStore


def test_create_thread_returns_stable_id_and_stores_metadata() -> None:
    store = ThreadStore()

    thread = store.create_thread(agent_key="default", cwd="/tmp/project", metadata={"source": "test"})

    assert thread.thread_id == "thread_1"
    assert thread.agent_key == "default"
    assert thread.cwd == "/tmp/project"
    assert thread.metadata == {"source": "test"}


def test_create_turn_links_to_thread() -> None:
    store = ThreadStore()
    thread = store.create_thread(agent_key="default")

    turn = store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "hello"}], run_id="run_1")

    assert turn.turn_id == "turn_1"
    assert turn.thread_id == thread.thread_id
    assert turn.run_id == "run_1"
    assert turn.input == [{"type": "text", "text": "hello"}]


def test_append_item_and_read_thread_preserves_creation_order() -> None:
    store = ThreadStore()
    thread = store.create_thread(agent_key="default")
    first_turn = store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "first"}])
    second_turn = store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "second"}])

    first_item = ThreadItem(
        item_id="item_1",
        thread_id=thread.thread_id,
        turn_id=first_turn.turn_id,
        item_type="agentMessage",
        status="completed",
        payload={"text": "first"},
        created_at=1,
        updated_at=1,
    )
    second_item = ThreadItem(
        item_id="item_2",
        thread_id=thread.thread_id,
        turn_id=second_turn.turn_id,
        item_type="agentMessage",
        status="completed",
        payload={"text": "second"},
        created_at=2,
        updated_at=2,
    )

    store.append_item(first_item)
    store.append_item(second_item)
    snapshot = store.read_thread(thread.thread_id)

    assert [turn.turn_id for turn in snapshot.turns] == ["turn_1", "turn_2"]
    assert [item.item_id for item in snapshot.items] == ["item_1", "item_2"]
    assert snapshot.items[1].payload == {"text": "second"}


def test_archive_hides_thread_from_active_list() -> None:
    store = ThreadStore()
    thread = store.create_thread(agent_key="default")

    assert [record.thread_id for record in store.list_threads()] == [thread.thread_id]
    store.archive_thread(thread.thread_id)

    assert store.list_threads() == []
    assert [record.thread_id for record in store.list_threads(include_archived=True)] == [thread.thread_id]


def test_store_persists_thread_turn_and_item_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "app_server.sqlite"
    store = ThreadStore(db_path)
    thread = store.create_thread(agent_key="default", metadata={"source": "persist"})
    turn = store.create_turn(thread_id=thread.thread_id, input=[{"type": "text", "text": "hello"}], run_id="run_1")
    store.append_item(
        ThreadItem(
            item_id="item_1",
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            item_type="agentMessage",
            status="completed",
            payload={"text": "hello"},
            created_at=1,
            updated_at=2,
        )
    )

    reopened = ThreadStore(db_path)
    snapshot = reopened.read_thread(thread.thread_id)

    assert snapshot.thread.metadata == {"source": "persist"}
    assert snapshot.turns[0].run_id == "run_1"
    assert snapshot.items[0].payload == {"text": "hello"}
