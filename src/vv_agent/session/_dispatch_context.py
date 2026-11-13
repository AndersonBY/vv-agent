"""SDK context construction on either side of the core compiled call facts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from vv_agent.kernel._call_context import CompiledCallFacts, approved_tool_call
from vv_agent.kernel._tool_policy import apply_tool_context_policy
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools.base import ToolContext
from vv_agent.types import AgentTask

from .records import Record
from .store import SessionStore

if TYPE_CHECKING:
    from .runtime import Runtime


@dataclass
class _PreparedContext:
    metadata: dict[str, Any]
    ctx: ExecutionContext
    workspace: Path


def begin_context(
    runtime: Runtime,
    task: AgentTask,
    plan: Record,
    token: CancellationToken,
    *,
    approved: bool,
    shared_state: dict[str, Any] | None,
) -> _PreparedContext:
    metadata = dict(task.metadata)
    if plan._payload["op_kind"] != "model":
        apply_tool_context_policy(
            metadata, agent_policy=runtime.agent.tool_policy, run_policy=runtime.config.tool_policy, shared_state=shared_state
        )
    ctx = ExecutionContext(cancellation_token=token, metadata=metadata)
    if approved:
        ctx._approved_tool_approval = SimpleNamespace(call=approved_tool_call(plan))
    workspace = Path(runtime.config.workspace or ".")
    return _PreparedContext(metadata, ctx, workspace)


def finish_context(
    runtime: Runtime,
    prepared: _PreparedContext,
    task: AgentTask,
    plan: Record,
    facts: CompiledCallFacts,
    *,
    shared_state: dict[str, Any] | None,
    store: SessionStore | None,
    session_id: str | None,
) -> ToolContext:
    return ToolContext(
        workspace=prepared.workspace,
        shared_state=shared_state if shared_state is not None else dict(task.initial_shared_state),
        run_context=runtime.run_context(task.task_id),
        cycle_index=facts.cycle_index,
        workspace_backend=runtime.workspace_backend,
        task_id=facts.task_id,
        ctx=prepared.ctx,
        task_metadata=prepared.metadata,
        sub_task_manager=runtime.child_tasks(store, session_id)
        if store and session_id and plan._payload["request"].get("name") == "sub_task_status"
        else None,
        idempotency_key=facts.idempotency_key,
        metadata=facts.metadata(session_id),
    )


def context(
    runtime: Runtime,
    task: AgentTask,
    plan: Record,
    token: CancellationToken,
    *,
    approved: bool,
    shared_state: dict[str, Any] | None,
    store: SessionStore | None,
    session_id: str | None,
) -> ToolContext:
    prepared = begin_context(runtime, task, plan, token, approved=approved, shared_state=shared_state)
    facts = CompiledCallFacts(task, plan)
    return finish_context(runtime, prepared, task, plan, facts, shared_state=shared_state, store=store, session_id=session_id)
