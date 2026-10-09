from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vv_agent.events import RunEvent
from vv_agent.runtime.context import ExecutionContext


def model_call_context(
    *,
    event_handler: Callable[[RunEvent], None] | None = None,
    metadata: dict[str, Any] | None = None,
    forward_model_events: bool = False,
) -> ExecutionContext:
    return ExecutionContext(event_handler=event_handler, metadata=dict(metadata or {}))
