"""Tool intents and child dependencies reuse the normal model operation ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.types import Message, ToolExecutionResult, ToolResultStatus

from .children import SummaryChildTool
from .compaction import manager_for, tokens
from .context import project_context
from .providers import Definitive
from .records import InboxItem, Record, digest
from .reducer import fold

if TYPE_CHECKING:
    from .kernel import _Driver


def compaction_request(item: InboxItem) -> dict | None:
    content = item.payload.get("content")
    if item.kind != "user" or not isinstance(content, dict):
        return None
    metadata = content.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("request_compaction") is not True:
        return None
    request = content.get("app_server_compaction")
    return request if isinstance(request, dict) else None


def pending_compaction_requests(driver: _Driver) -> list[str]:
    consumed = {
        oid
        for (tid, stage, _), record in driver.state.boundaries.items()
        if tid == driver.state.active_turn_id and stage == "memory_completed"
        for oid in record._payload["data"]["event"].get("metadata", {}).get("requested_by", [])
    }
    controls = [
        key
        for key, r in driver.state.applied_inputs.items()
        if (
            r.turn_id == driver.state.active_turn_id
            or (
                driver.state.active_turn_id is not None
                and key in driver.state.turns[driver.state.active_turn_id].start._payload["input_ids"]
            )
        )
        and r._payload["disposition"] in {"applied", "queued"}
        and compaction_request(InboxItem(**r._payload["input"])) is not None
        and key not in consumed
    ]
    return controls + [
        oid
        for oid, op in driver.state.operations.items()
        if op.turn_id == driver.state.active_turn_id
        and op.kind != "model"
        and op.selected_attempt is not None
        and op.attempts[op.selected_attempt].context == "normal"
        and (receipt := op.attempts[op.selected_attempt].result) is not None
        and receipt._payload["result"].get("status_code") == "SUCCESS"
        and receipt._payload["result"].get("metadata", {}).get("request_compaction") is True
        and oid not in consumed
    ]


def summary_pending(driver: _Driver, oid: str) -> bool:
    op = driver.state.operations.get(f"{oid}/summary")
    if op is None or op.selected_attempt is not None:
        return False
    attempt = op.attempts[max(op.attempts)]
    return attempt.state in {"planned", "started", "parked"} or bool(
        attempt.unknown and attempt.unknown._payload["retry"] == "retry"
    )


def source_at(driver: _Driver, version: str) -> list[Message]:
    prefix = tuple(row for row in driver.records if row.seq <= int(version))
    logical = [row.record for row in prefix]
    inputs = [InboxItem(**r._payload["input"]) for r in logical if r.kind == "input_applied"]
    return project_context(prefix, fold(logical, consumed_inputs=inputs))


def child_summary(driver: _Driver, plan: Record, binding: SummaryChildTool) -> tuple[bool, str | None]:
    oid = f"{plan.operation_id}/summary"
    op = driver.state.operations.get(oid)
    manager = manager_for(driver)
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
        if summary_plan is not None and attempt.result is not None and attempt.context == "normal":
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
