from __future__ import annotations

from vv_agent import MemorySession, Message, RedisSession, SQLiteSession


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
