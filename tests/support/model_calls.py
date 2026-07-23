from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vv_agent.events import RunEvent
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.model_calls import ModelCallCoordinator


def model_call_context(
    *,
    event_handler: Callable[[RunEvent], None] | None = None,
    metadata: dict[str, Any] | None = None,
    forward_model_events: bool = False,
) -> ExecutionContext:
    effective_metadata = dict(metadata or {})
    context = ExecutionContext(event_handler=event_handler, metadata=effective_metadata)
    run_id = str(effective_metadata.get("_vv_agent_run_id") or "run_test")
    trace_id = str(effective_metadata.get("_vv_agent_trace_id") or run_id)
    context.model_call_coordinator = ModelCallCoordinator(
        ledger=context.model_call_ledger,
        run_id=run_id,
        trace_id=trace_id,
        agent_name=_optional_string(effective_metadata.get("_vv_agent_agent_name")),
        session_id=_optional_string(effective_metadata.get("_vv_agent_session_id")),
        parent_run_id=_optional_string(effective_metadata.get("_vv_agent_parent_run_id")),
        event_sink=event_handler if forward_model_events else None,
    )
    return context


def _optional_string(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None
