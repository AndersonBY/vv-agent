from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vv_agent.events import RunEvent, event_from_dict


@dataclass(frozen=True, slots=True)
class RunEventReplayQuery:
    run_id: str
    include_children: bool = True

    @classmethod
    def run(cls, run_id: str, *, include_children: bool = True) -> RunEventReplayQuery:
        return cls(run_id=run_id, include_children=include_children)


class EventStoreError(RuntimeError):
    def __init__(self, message: str, *, code: str, line_number: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.line_number = line_number

    @classmethod
    def corrupt_line(cls, line_number: int) -> EventStoreError:
        return cls(
            f"event store corrupt line {line_number}",
            code="event_store_corrupt_line",
            line_number=line_number,
        )


class RunEventStore(Protocol):
    def append(self, event: RunEvent) -> None:
        raise NotImplementedError

    def replay(
        self,
        query: RunEventReplayQuery | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[RunEvent]:
        raise NotImplementedError


class JsonlRunEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: RunEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"{json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)}\n")

    def replay(
        self,
        query: RunEventReplayQuery | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[RunEvent]:
        return self._replay(_resolve_replay_query(query, run_id=run_id))

    def _replay(self, query: RunEventReplayQuery) -> Iterator[RunEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("Run event line must contain a JSON object")
                    event = event_from_dict(payload)
                except (KeyError, TypeError, ValueError) as error:
                    raise EventStoreError.corrupt_line(line_number) from error

                if event.run_id == query.run_id or (
                    query.include_children and event.parent_run_id == query.run_id
                ):
                    yield event


def _resolve_replay_query(
    query: RunEventReplayQuery | None,
    *,
    run_id: str | None,
) -> RunEventReplayQuery:
    if query is not None and run_id is not None:
        raise TypeError("replay() accepts either query or run_id, not both")
    if query is not None:
        return query
    if run_id is not None:
        return RunEventReplayQuery(run_id=run_id)
    raise TypeError("replay() requires a query or run_id")
