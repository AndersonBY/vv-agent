from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vv_agent import (
    MemorySession,
    MemorySessionStore,
    Message,
    RedisSession,
    RedisSessionStore,
    SQLiteSession,
    SQLiteSessionStore,
    session_store_conformance,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
FIXTURE_PATH = FIXTURE_DIR / "session_items_v1.jsonl"
RUNNER_SESSION_FIXTURE_PATH = FIXTURE_DIR / "runner_session_messages_v1.jsonl"


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, list[str]] = {}

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.values.get(key, [])
        normalized_end = None if end == -1 else end + 1
        return values[start:normalized_end]

    def rpush(self, key: str, *items: str) -> None:
        self.values.setdefault(key, []).extend(items)

    def rpop(self, key: str) -> str | None:
        values = self.values.get(key, [])
        if not values:
            return None
        return values.pop()

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def test_memory_session_stores_pops_and_clears_items() -> None:
    session = MemorySession("thread-1")
    session.add_items([Message(role="user", content="one"), Message(role="assistant", content="two")])

    assert [item.content for item in session.get_items(limit=1)] == ["two"]
    popped = session.pop_item()
    assert popped is not None
    assert popped.content == "two"
    assert [item.content for item in session.get_items()] == ["one"]

    session.clear_session()

    assert session.get_items() == []

    session.add_items([Message(role="user", content="alias")])
    session.clear()
    assert session.get_items() == []


def test_sqlite_session_persists_items(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite3"
    SQLiteSession("thread-1", db_path=db_path).add_items(
        [Message(role="user", content="one"), Message(role="assistant", content="two")]
    )

    restored = SQLiteSession("thread-1", db_path=db_path)

    assert [item.content for item in restored.get_items()] == ["one", "two"]
    popped = restored.pop_item()
    assert popped is not None
    assert popped.content == "two"
    assert [item.content for item in SQLiteSession("thread-1", db_path=db_path).get_items()] == ["one"]


def test_redis_session_uses_redis_list_storage() -> None:
    client = FakeRedis()
    session = RedisSession("thread-1", redis_client=client, key_prefix="vv-test")
    session.add_items([Message(role="user", content="one"), Message(role="assistant", content="two")])

    restored = RedisSession("thread-1", redis_client=client, key_prefix="vv-test")

    assert [item.content for item in restored.get_items()] == ["one", "two"]
    assert [item.content for item in restored.get_items(limit=1)] == ["two"]
    popped = restored.pop_item()
    assert popped is not None
    assert popped.content == "two"
    assert [item.content for item in session.get_items()] == ["one"]

    restored.clear_session()
    assert session.get_items() == []


def test_session_items_fixture_is_canonical_message_wire() -> None:
    fixture = FIXTURE_PATH.read_text(encoding="utf-8")
    messages = [Message.from_dict(json.loads(line)) for line in fixture.splitlines()]
    serialized = "".join(
        f"{json.dumps(message.to_dict(), ensure_ascii=False, separators=(',', ':'))}\n" for message in messages
    )

    assert serialized == fixture
    assert hashlib.sha256(fixture.encode()).hexdigest() == (
        "8985926cd4f3b7befbb0643e1429936256cbb20d0cce06440dfa68e64969599d"
    )


def test_runner_session_fixture_hash_is_shared_with_rust() -> None:
    fixture = RUNNER_SESSION_FIXTURE_PATH.read_bytes()

    assert hashlib.sha256(fixture).hexdigest() == (
        "74c7406ea33f3d676c340a9975c90220e7830a85fb0efa11a743acec726f16dd"
    )


def test_memory_session_store_conformance() -> None:
    session_store_conformance(MemorySessionStore())


def test_sqlite_session_store_conformance(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "session-store.sqlite3")
    try:
        session_store_conformance(store)
    finally:
        store.close()


def test_redis_session_store_conformance_with_injected_client() -> None:
    session_store_conformance(RedisSessionStore(redis_client=FakeRedis(), key_prefix="vv-test-store"))
