from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

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


def test_store_recovers_running_thread_state_on_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "app_server.sqlite"
    store = ThreadStore(db_path)
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[])

    store.set_active_turn(thread.thread_id, turn.turn_id, "running")
    running = store.read_thread(thread.thread_id).thread
    assert running.status == "running"
    assert running.active_turn_id == turn.turn_id

    recovered = ThreadStore(db_path).read_thread(thread.thread_id).thread
    assert recovered.status == "idle"
    assert recovered.active_turn_id is None


def test_duplicate_item_id_fails_without_reordering_replay() -> None:
    store = ThreadStore()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[])
    item = ThreadItem(
        item_id="item_1",
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        item_type="agentMessage",
        status="completed",
        payload={"text": "original"},
        created_at=1,
        updated_at=1,
    )
    store.append_item(item)

    with pytest.raises(sqlite3.IntegrityError):
        store.append_item(
            ThreadItem(
                item_id="item_1",
                thread_id=thread.thread_id,
                turn_id=turn.turn_id,
                item_type="agentMessage",
                status="completed",
                payload={"text": "replacement"},
                created_at=2,
                updated_at=2,
            )
        )

    snapshot = store.read_thread(thread.thread_id)
    assert [item.item_id for item in snapshot.items] == ["item_1"]
    assert snapshot.items[0].payload == {"text": "original"}


def test_duplicate_run_event_replays_identical_item_once(tmp_path: Path) -> None:
    db_path = tmp_path / "thread-store.sqlite"
    first_store = ThreadStore(db_path)
    thread = first_store.create_thread(agent_key="default")
    turn = first_store.create_turn(thread_id=thread.thread_id, input=[])
    item = ThreadItem(
        item_id="item_evt_1",
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        item_type="agentMessage",
        status="completed",
        payload={"text": "durable"},
        created_at=1,
        updated_at=1,
    )

    assert first_store.append_item(item, run_event_id="evt_1") is True
    assert ThreadStore(db_path).append_item(item, run_event_id="evt_1") is False

    snapshot = first_store.read_thread(thread.thread_id)
    assert snapshot.items == [item]


def test_duplicate_run_event_rejects_conflicting_projection() -> None:
    store = ThreadStore()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[])
    original = ThreadItem(
        item_id="item_evt_1",
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        item_type="agentMessage",
        status="completed",
        payload={"text": "original"},
        created_at=1,
        updated_at=1,
    )
    store.append_item(original, run_event_id="evt_1")

    with pytest.raises(sqlite3.IntegrityError, match="conflicting App Server item projection"):
        store.append_item(
            ThreadItem(
                item_id=original.item_id,
                thread_id=original.thread_id,
                turn_id=original.turn_id,
                item_type=original.item_type,
                status=original.status,
                payload={"text": "replacement"},
                created_at=original.created_at,
                updated_at=original.updated_at,
            ),
            run_event_id="evt_1",
        )

    assert store.read_thread(thread.thread_id).items == [original]


def test_opening_legacy_database_adds_thread_lifecycle_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL UNIQUE,
            agent_key TEXT NOT NULL,
            cwd TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            archived_at REAL,
            metadata_json TEXT NOT NULL
        );
        INSERT INTO threads (
            thread_id, agent_key, cwd, created_at, updated_at, archived_at, metadata_json
        ) VALUES ('thread_1', 'default', NULL, 1.0, 1.0, NULL, '{}');
        """
    )
    connection.commit()
    connection.close()

    thread = ThreadStore(db_path).read_thread("thread_1").thread

    assert thread.status == "idle"
    assert thread.active_turn_id is None
