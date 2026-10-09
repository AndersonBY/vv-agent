"""Log orchestration only; MemoryManager owns pruning, prompts and summary acceptance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from vv_agent.llm.errors import MAX_PTL_RETRIES
from vv_agent.memory.manager import MemoryManager, SummaryPlan
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.types import Message
from vv_agent.workspace.local import LocalWorkspaceBackend

from .context import message_ids, project_context
from .memory import extract_memory, finish_compact, start_compact
from .records import InboxItem, Record, digest
from .reducer import fold

if TYPE_CHECKING:
    from .kernel import _Driver


def manager_for(driver: _Driver) -> MemoryManager:
    task = driver.task()
    definition = driver.state.turns[task.task_id].start._payload["definition"]
    return MemoryManager(
        **(definition["memory_settings"] | {"artifact_scope": f"session/{driver.sid}/{task.task_id}"}),
        workspace_backend=(
            driver.runtime.memory_manager.workspace_backend
            or driver.runtime.config.workspace_backend
            or (LocalWorkspaceBackend(Path(driver.runtime.config.workspace or ".")) if task.use_workspace else None)
        ),
        recovery_tool_available=any(t["function"]["name"] == "read_file" for t in definition["tools"]),
        session_memory=None,
        summary_callback=None,
    )


def tokens(manager: MemoryManager, messages: list[Message]) -> int:
    return count_messages_tokens([m.to_openai_message() for m in messages], model=manager.model)


def replacement_record(
    driver: _Driver,
    manager: MemoryManager,
    source: list[Message],
    replacement: list[Message],
    *,
    mode: str,
    summary: Record | None = None,
    plan: SummaryPlan | None = None,
) -> Record:
    ids = message_ids(source)
    if plan is None:
        removed = {i for i, (a, b) in enumerate(zip(source, replacement, strict=True)) if a != b}
    else:
        prefix = {id(m) for m in [*plan.previous, *plan.prefix]}
        removed = {i for i, m in enumerate(source) if id(m) in prefix}
    return driver.record(
        "context_compacted",
        {
            "source_digest": digest([m.to_dict() for m in source]),
            "prefix_ids": [value for i, value in enumerate(ids) if i in removed],
            "tail_ids": [value for i, value in enumerate(ids) if i not in removed],
            "mode": mode,
            "summary_operation_id": summary.operation_id if summary else None,
            "replacement": [m.to_dict() for m in replacement],
            "evidence_manifest": manager.compaction_evidence(replacement),
        },
    )


def finish_summary(driver: _Driver) -> bool:
    """Accept a saved receipt before consuming inputs that could change its source view."""
    tid = driver.state.active_turn_id
    accepted = {r._payload["summary_operation_id"] for r in driver.state.compactions}
    for oid, op in driver.state.operations.items():
        if op.turn_id != tid or op.kind != "model" or op.selected_attempt is None or oid in accepted:
            continue
        a = op.attempts[op.selected_attempt]
        if a.plan._payload["purpose"] != "compaction" or a.result is None or a.context != "normal":
            continue
        # The source is an immutable log prefix, not the present transcript after late inputs.
        prefix = tuple(r for r in driver.records if r.seq <= int(a.plan._payload["context_version"]))
        logical = [r.record for r in prefix]
        consumed = [InboxItem(**r._payload["input"]) for r in logical if r.kind == "input_applied"]
        source = project_context(prefix, fold(logical, consumed_inputs=consumed))
        meta = a.plan._payload["request"]["metadata"]
        if digest([m.to_dict() for m in driver.transcript()]) != meta["source_digest"]:
            continue
        manager = manager_for(driver)
        plan = manager.plan_summary(source, keep_recent=meta["keep_recent"])
        if plan is None:
            continue
        replacement, accepted_result = manager.accept_summary(plan, a.result._payload["result"]["content"], notify=False)
        if accepted_result:
            record = replacement_record(driver, manager, source, replacement, mode=meta["mode"], summary=a.result, plan=plan)
            driver.commit([record], guarded=True)
            return True
    return False


def compact_context(driver: _Driver) -> bool:
    """One prune pass, then one ordinary logged summary operation for the exact source."""
    manager = manager_for(driver)
    source = driver.transcript()
    primary = [
        op
        for op in driver.state.operations.values()
        if op.turn_id == driver.state.active_turn_id
        and op.kind == "model"
        and op.attempts[1].plan._payload["purpose"] == "primary"
    ]
    failures = 0
    for op in reversed(primary):
        result = op.attempts[op.selected_attempt].result if op.selected_attempt else None
        if result is None or result._payload["result"].get("error_code") != "prompt_too_long":
            break
        failures += 1
    cycle = 1 + sum(
        not (
            op.selected_attempt
            and (receipt := op.attempts[op.selected_attempt].result)
            and receipt._payload["result"].get("error_code") == "prompt_too_long"
        )
        for op in primary
    )
    last_compaction = next(
        (
            r.record
            for r in reversed(driver.records)
            if r.record.kind == "context_compacted" and r.record.turn_id == driver.state.active_turn_id
        ),
        None,
    )
    if (
        last_compaction
        and last_compaction._payload["replacement"] == [m.to_dict() for m in source]
        and last_compaction._payload["mode"] != "micro"
        and finish_compact(driver, manager, source, mode=last_compaction._payload["mode"], changed=True)
    ):
        return True
    if failures:
        selected = primary[-1].selected_attempt
        assert selected is not None
        last_result = primary[-1].attempts[selected].result
        last_seq = next(r.seq for r in driver.records if r.record == last_result)
        if any(r.seq > last_seq and r.record.kind == "context_compacted" for r in driver.records):
            return False
    if failures > MAX_PTL_RETRIES:
        driver.close("failed", "CompactionExhaustedError")
        return True
    mode = "emergency" if failures > 1 else "summary"
    last_change = next(
        (
            r.record
            for r in reversed(driver.records)
            if r.record.kind in {"context_compacted", "turn_started", "input_applied"}
            or (
                r.record.kind == "op_completed"
                and r.record.operation_id
                and driver.state.operations[r.record.operation_id].attempts[1].plan._payload["purpose"] != "compaction"
            )
        ),
        None,
    )
    if last_change and last_change.kind == "context_compacted" and last_change._payload["mode"] != "micro":
        return False
    try:
        manager.compaction_evidence(source)
    except (TypeError, ValueError):
        # The full pipeline rejects malformed blocks before any artifact writes.
        return False
    if not failures:
        current_tokens = tokens(manager, source)
        if extract_memory(driver, source, current_tokens, cycle):
            return True
        prune = manager.plan_microcompaction(source, cycle_index=cycle, current_tokens=current_tokens)
        if ((prune and prune.candidates) or current_tokens > manager.autocompact_threshold) and start_compact(
            driver,
            manager,
            source,
            cycle=cycle,
            trigger="full_threshold" if current_tokens > manager.autocompact_threshold else "micro_threshold",
            prune=prune,
        ):
            return True
        if prune and prune.candidates and not (last_change and last_change.kind == "context_compacted"):
            result = manager.apply_microcompaction(source, plan=prune)
            if result.archived_count:
                record = replacement_record(driver, manager, source, result.messages, mode="micro")
                record = replace(
                    record,
                    payload=record._payload
                    | {
                        "micro_usage": {
                            "archived_count": result.archived_count,
                            "reclaimed_tokens": result.reclaimed_tokens,
                            "artifact_failure_count": result.artifact_failure_count,
                        }
                    },
                )
                driver.runtime.hook("after_microcompact_artifacts", record)
                driver.commit([record], guarded=True)
                return True
        if current_tokens <= manager.autocompact_threshold:
            return finish_compact(driver, manager, source, mode="none", changed=False)
    elif start_compact(driver, manager, source, cycle=cycle, trigger="prompt_too_long"):
        return True
    source_digest = digest([m.to_dict() for m in source])
    if not failures and any(
        op.turn_id == driver.state.active_turn_id
        and op.attempts[1].plan._payload["purpose"] == "compaction"
        and op.attempts[1].plan._payload["request"]["metadata"]["source_digest"] == source_digest
        and op.attempts[1].plan._payload["request"]["metadata"]["mode"] == mode
        for op in driver.state.operations.values()
    ):
        return finish_compact(driver, manager, source, mode="none", changed=False)
    plan = manager.plan_summary(source, drop_ratio=min(0.2 * failures, 0.95) if failures > 1 else 0)
    if plan is None:
        return finish_compact(driver, manager, source, mode="none", changed=False)
    # Equal source and tail targets reuse even rejected receipts across PTL retries.
    oid = f"{driver.state.active_turn_id}/model/compaction/{source_digest}/{mode}/{plan.keep_recent}"
    if oid in driver.state.operations:
        return finish_compact(driver, manager, source, mode="none", changed=False)
    task = driver.task()
    request = {
        "model": task.model,
        "messages": [Message("user", plan.prompt).to_dict()],
        "tools": [],
        "metadata": {
            "purpose": "compaction",
            "source_digest": source_digest,
            "keep_recent": plan.keep_recent,
            "mode": mode,
            "vv_session": {"cycle_index": cycle},
        },
        "prompt_bundle": None,
        "model_settings": task.model_settings.to_dict() if task.model_settings else None,
    }
    if tokens(manager, [Message("user", plan.prompt)]) >= manager.model_context_window:
        return False
    driver.commit([driver.plan(oid, request, "model", purpose="compaction")], guarded=True)
    return True
