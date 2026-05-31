from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from vv_agent.events import RunEvent, event_from_dict


class RunEventStore(Protocol):
    def append(self, event: RunEvent) -> None:
        raise NotImplementedError

    def replay(self, *, run_id: str) -> Iterator[RunEvent]:
        raise NotImplementedError


class JsonlRunEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: RunEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True))
            file.write("\n")

    def replay(self, *, run_id: str) -> Iterator[RunEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                payload = json.loads(line)
                if payload.get("run_id") == run_id:
                    yield event_from_dict(payload)
