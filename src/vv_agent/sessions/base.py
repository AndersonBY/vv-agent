from __future__ import annotations

from typing import Protocol

from vv_agent.types import Message


class Session(Protocol):
    session_id: str

    def get_items(self, limit: int | None = None) -> list[Message]:
        ...

    def add_items(self, items: list[Message]) -> None:
        ...

    def pop_item(self) -> Message | None:
        ...

    def clear_session(self) -> None:
        ...
