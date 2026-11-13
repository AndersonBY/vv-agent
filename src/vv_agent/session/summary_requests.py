"""Tool intents and child dependencies reuse the normal model operation ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.kernel._summary_requests import compaction_request as compaction_request
from vv_agent.kernel._summary_requests import pending_compaction_requests as pending_compaction_requests
from vv_agent.kernel._summary_requests import summary_pending as summary_pending
from vv_agent.types import Message, ToolExecutionResult, ToolResultStatus

from .children import SummaryChildTool
from .compaction import manager_for, tokens
from .context import project_context
from .providers import Definitive
from .records import InboxItem, Record, digest
from .reducer import _fold_factory_for

if TYPE_CHECKING:
    from .kernel import _Driver


def source_at(driver: _Driver, version: str) -> list[Message]:
    prefix = tuple(row for row in driver.records if row.seq <= int(version))
    logical = [row.record for row in prefix]
    inputs = [InboxItem(**r._payload["input"]) for r in logical if r.kind == "input_applied"]
    factory = _fold_factory_for(driver.store)
    state = factory().extend(logical, consumed_inputs=inputs)
    return project_context(prefix, state, _fold_factory=factory)


def child_summary(driver: _Driver, plan: Record, binding: SummaryChildTool) -> tuple[bool, str | None]:
    oid = f"{plan.operation_id}/summary"
    op = driver.state.operations.get(oid)
    manager = manager_for(driver)
    # This summary is read-only input for a child, never a parent replacement.
    manager.protected_start = None
    if op is None and not binding.requires_summary(plan):
        return True, None
    if op is None:
        # Exclude the current incomplete tool batch; include all completed parent history.
        parent = driver.state.operations[plan._payload["dependencies"][0]]
        receipt = parent.attempts[parent.selected_attempt or 1].result
        version = str(next(row.seq for row in driver.records if row.record == receipt) - 1)
        source = source_at(driver, version)
        summary_plan = manager.plan_full_summary(source)
        if summary_plan is not None and tokens(manager, [Message("user", summary_plan.prompt)]) < manager.model_context_window:
            _, resolved = driver.runtime.model_route("compaction")
            task = driver.task()
            request = {
                "model": resolved.model_id,
                "messages": [Message("user", summary_plan.prompt).to_dict()],
                "tools": [],
                "metadata": {
                    "purpose": "compaction",
                    "mode": "child_summary",
                    "source_digest": digest([m.to_dict() for m in source]),
                    "source_context_version": version,
                    "child_operation_id": plan.operation_id,
                    "vv_session": {
                        "cycle_index": driver.state.operations[plan._payload["dependencies"][0]]
                        .attempts[1]
                        .plan._payload["request"]["metadata"]["vv_session"]["cycle_index"]
                    },
                },
                "prompt_bundle": None,
                "model_settings": task.model_settings.to_dict() if task.model_settings else None,
            }
            driver.commit([driver.plan(oid, request, "model", purpose="compaction")], guarded=True)
            return False, None
    elif op.selected_attempt is not None:
        attempt = op.attempts[op.selected_attempt]
        meta = attempt.plan._payload["request"]["metadata"]
        source = source_at(driver, meta["source_context_version"])
        if digest([m.to_dict() for m in source]) != meta["source_digest"]:
            raise ValueError("child summary source mismatch")
        summary_plan = manager.plan_full_summary(source)
        if (
            summary_plan is not None
            and attempt.result is not None
            and not attempt.result._payload["result"].get("error_code")
            and attempt.context == "normal"
        ):
            replacement, accepted = manager.accept_summary(
                summary_plan, attempt.result._payload["result"]["content"], notify=False, require_reduction=False
            )
            if accepted:
                return True, next(m.content for m in replacement if m.name == "memory_summary")
    elif summary_pending(driver, plan.operation_id or ""):
        return False, None
    result = ToolExecutionResult(
        plan._payload["request"]["id"],
        "A fresh parent summary is unavailable; no child was admitted.",
        status_code=ToolResultStatus.ERROR,
        error_code="child_summary_unavailable",
    )
    driver.commit(driver.completed(plan, Definitive(result.to_dict())), guarded=True)
    return False, None
