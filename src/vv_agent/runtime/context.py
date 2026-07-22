from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vv_agent.events import RunEvent

if TYPE_CHECKING:
    from vv_agent.runtime.cancellation import CancellationToken
    from vv_agent.runtime.state import CheckpointStore

RunEventHandler = Callable[[RunEvent], None]


@dataclass(slots=True)
class ExecutionContext:
    cancellation_token: CancellationToken | None = None
    event_handler: RunEventHandler | None = None
    checkpoint_store: CheckpointStore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _pending_tool_approval: Any | None = field(default=None, repr=False)
    _approved_tool_approval: Any | None = field(default=None, repr=False)

    def check_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.check()
