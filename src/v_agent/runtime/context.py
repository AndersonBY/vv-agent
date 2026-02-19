from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from v_agent.runtime.cancellation import CancellationToken
    from v_agent.runtime.state import StateStore

StreamCallback = Callable[[str], None]


@dataclass(slots=True)
class ExecutionContext:
    cancellation_token: CancellationToken | None = None
    stream_callback: StreamCallback | None = None
    state_store: StateStore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.check()
