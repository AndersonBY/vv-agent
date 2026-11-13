"""Logged compaction decisions, suspended at concrete host and ledger operations."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from vv_agent.kernel import _summary_compaction
from vv_agent.kernel._compaction_threshold import compute_compaction_threshold
from vv_agent.kernel._context import message_ids
from vv_agent.kernel._context_messages import UnseenInputCapacityError, retained_start
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._microcompaction import MicrocompactPlan, estimate_tool_artifact_path, plan_memory_microcompact
from vv_agent.kernel._microcompaction_apply import apply_microcompaction
from vv_agent.kernel._records import Record, digest
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import StoredRecord
from vv_agent.kernel._summary_compaction import SummaryPlan


class CompactionView(Protocol):
    state: ExecutionState

    @property
    def records(self) -> Sequence[StoredRecord]: ...


@dataclass(frozen=True, slots=True)
class CompactionMemory:
    settings: dict[str, Any]
    protected_start: int | None
    recovery_tool_available: bool

    @property
    def effective_context_window(self) -> int:
        return max(self.settings["model_context_window"] - self.settings["reserved_output_tokens"], 0)

    @property
    def autocompact_threshold(self) -> int:
        return compute_compaction_threshold(
            configured_threshold=self.settings["compact_threshold"],
            model_context_window=self.settings["model_context_window"],
            reserved_output_tokens=self.settings["reserved_output_tokens"],
            autocompact_buffer_tokens=self.settings["autocompact_buffer_tokens"],
        )


class CompactionRequest(Enum):
    INBOX = "inbox"
    MEMORY = "memory"
    TRANSCRIPT = "transcript"
    RESTORE_SESSION_MEMORY = "restore_session_memory"
    SAVE_MEMORY_PROJECTION = "save_memory_projection"


@dataclass(frozen=True, slots=True)
class CountContext:
    messages: list[Message]
    model: str


@dataclass(frozen=True, slots=True)
class ModelRoute:
    purpose: str


@dataclass(frozen=True, slots=True)
class CreateRecord:
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CreateBoundary:
    stage: str
    identity: str
    data: dict[str, Any]
    source_operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class PlanModel:
    oid: str
    request: dict[str, Any]
    purpose: str


@dataclass(frozen=True, slots=True)
class CommitRecords:
    records: list[Record]


@dataclass(frozen=True, slots=True)
class CloseCompaction:
    reason: str


@dataclass(frozen=True, slots=True)
class MicroArtifacts:
    record: Record


@dataclass(frozen=True, slots=True)
class BeforeCompaction:
    key: str
    source: list[Message]
    cycle: int
    trigger: str
    prune: MicrocompactPlan | None


@dataclass(frozen=True, slots=True)
class AfterCompaction:
    key: str
    start: dict[str, Any]
    source: list[Message]
    mode: str
    changed: bool
    archived_count: int
    reclaimed_tokens: int
    artifact_failure_count: int


@dataclass(frozen=True, slots=True)
class SelectExtractionMessages:
    source: list[Message]
    current_tokens: int


@dataclass(frozen=True, slots=True)
class ExtractionPrompt:
    messages: list[Message]


@dataclass(frozen=True, slots=True)
class ApplyExtractionReceipt:
    source: list[Message]
    current_tokens: int
    cycle: int
    content: str


@dataclass(frozen=True, slots=True)
class SessionMemoryCompacted:
    current_tokens: int | None


def memory_settings(definition: dict[str, Any], selection: Record | None) -> dict[str, Any]:
    """Resolve policy from durable bindings; shared by execution and context reads."""
    settings: dict[str, Any] = dict(definition["memory_settings"])
    if selection is not None:
        effective = selection._payload["input"]["payload"]["selection"]
        binding = selection._payload["model_binding"]
        settings["model"] = effective["model"]
        settings["model_max_output_tokens"] = binding["max_output_tokens"]
        reserve = effective.get("model_settings", {}).get("max_tokens")
        if reserve is not None:
            settings["reserved_output_tokens"] = reserve
            settings["reserved_output_source"] = "model_settings"
        elif settings["reserved_output_source"] != "task_metadata":
            cap = binding["max_output_tokens"]
            settings["reserved_output_tokens"] = min(16000, cap) if cap is not None else 16000
            settings["reserved_output_source"] = (
                "framework_fallback_capped_by_model_capability" if cap is not None and cap < 16000 else "framework_fallback"
            )
        settings["model_context_window"] = binding["context_length"] or (
            (settings["compact_threshold"] if settings["compact_threshold"] > 0 else 250000)
            + settings["reserved_output_tokens"]
            + settings["autocompact_buffer_tokens"]
        )
    return settings


def replacement_record(
    driver: CompactionView,
    manager: CompactionMemory,
    source: list[Message],
    replacement: list[Message],
    *,
    mode: str,
    summary: Record | None = None,
    plan: SummaryPlan | None = None,
) -> Generator[object, Any, Record]:
    retained_start(source, replacement, manager.protected_start)
    ids = message_ids(source)
    if plan is None:
        removed = {i for i, (a, b) in enumerate(zip(source, replacement, strict=True)) if a != b}
    else:
        prefix = {id(m) for m in [*plan.previous, *plan.prefix]}
        removed = {i for i, m in enumerate(source) if id(m) in prefix}
    return (
        yield CreateRecord(
            "context_compacted",
            {
                "source_digest": digest([m.to_dict() for m in source]),
                "prefix_ids": [value for i, value in enumerate(ids) if i in removed],
                "tail_ids": [value for i, value in enumerate(ids) if i not in removed],
                "mode": mode,
                "summary_operation_id": summary.operation_id if summary else None,
                "replacement": [m.to_dict() for m in replacement],
                "evidence_manifest": _summary_compaction.compaction_evidence(
                    replacement, protected_start=manager.protected_start
                ),
            },
        )
    )


def finish_summary(driver: CompactionView) -> Generator[object, Any, bool]:
    """Accept a saved receipt before consuming inputs that could change its source view."""
    tid = driver.state.active_turn_id
    generation = driver.state.turns[tid].start._payload["generation"] if tid else None
    if any(
        row.item.kind == "control"
        and row.item.payload["action"] in {"cancel", "abort", "close", "suspend"}
        and (row.item.target_turn_id in {None, tid})
        and (row.item.generation in {None, generation})
        for row in (yield CompactionRequest.INBOX)
    ):
        return False
    accepted = {r._payload["summary_operation_id"] for r in driver.state.compactions}
    for oid, op in driver.state.operations.items():
        if op.turn_id != tid or op.kind != "model" or op.selected_attempt is None or (oid in accepted):
            continue
        a = op.attempts[op.selected_attempt]
        if (
            a.plan._payload["purpose"] != "compaction"
            or a.result is None
            or a.result._payload["result"].get("error_code")
            or (a.context != "normal")
        ):
            continue
        meta = a.plan._payload["request"]["metadata"]
        if meta.get("mode") == "child_summary":
            continue
        source = yield CompactionRequest.TRANSCRIPT
        if digest([m.to_dict() for m in source]) != meta["source_digest"]:
            continue
        manager = yield CompactionRequest.MEMORY
        plan = _summary_compaction.plan_summary(
            source,
            keep_recent=meta["keep_recent"],
            drop_ratio=0,
            protected_start=manager.protected_start,
            language=manager.settings["language"],
            event_limit=manager.settings["summary_event_limit"],
        )
        if plan is None:
            continue
        accepted_result = yield from _summary_compaction.accept_summary(
            plan,
            a.result._payload["result"]["content"],
            protected_start=manager.protected_start,
            recovery_tool_available=manager.recovery_tool_available,
            effective_context_window=manager.effective_context_window,
            require_reduction=True,
            model=manager.settings["model"],
        )
        replacement = accepted_result.messages
        if accepted_result.accepted:
            record = yield from replacement_record(
                driver, manager, source, replacement, mode=meta["mode"], summary=a.result, plan=plan
            )
            yield CommitRecords([record])
            return True
    return False


def compact_context(
    driver: CompactionView, *, max_ptl_retries: int, requested_by: list[str] | None = None
) -> Generator[object, Any, bool]:
    """One prune pass, then one ordinary logged summary operation for the exact source."""
    manager = yield CompactionRequest.MEMORY
    source = yield CompactionRequest.TRANSCRIPT
    if manager.protected_start is not None:
        required = [m for m in source[: manager.protected_start] if m.role == "system"] + source[manager.protected_start :]
        if (yield CountContext(required, manager.settings["model"])) > manager.effective_context_window:
            yield CloseCompaction(UnseenInputCapacityError.code)
            return True
    primary = [
        op
        for op in driver.state.operations.values()
        if op.turn_id == driver.state.active_turn_id
        and op.kind == "model"
        and (op.attempts[1].plan._payload["purpose"] == "primary")
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
            and (receipt._payload["result"].get("error_code") == "prompt_too_long")
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
        and (last_compaction._payload["mode"] != "micro")
        and (yield from finish_compact(driver, manager, source, mode=last_compaction._payload["mode"], changed=True))
    ):
        return True
    if failures:
        selected = primary[-1].selected_attempt
        assert selected is not None
        last_result = primary[-1].attempts[selected].result
        last_seq = next(r.seq for r in driver.records if r.record == last_result)
        if any(r.seq > last_seq and r.record.kind == "context_compacted" for r in driver.records):
            return False
    if failures > max_ptl_retries:
        yield CloseCompaction("CompactionExhaustedError")
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
                and (driver.state.operations[r.record.operation_id].attempts[1].plan._payload["purpose"] != "compaction")
            )
        ),
        None,
    )
    if (
        not requested_by
        and last_change
        and (last_change.kind == "context_compacted")
        and (last_change._payload["mode"] != "micro")
    ):
        return False
    try:
        _summary_compaction.compaction_evidence(source, protected_start=manager.protected_start)
    except (TypeError, ValueError):
        return False
    if not failures:
        current_tokens = yield CountContext(source, manager.settings["model"])
        if not requested_by and (yield from extract_memory(driver, source, current_tokens, cycle)):
            return True
        prune = yield from plan_memory_microcompact(
            source,
            cycle_index=cycle,
            current_tokens=current_tokens,
            settings=manager.settings,
            protected_start=manager.protected_start,
            recovery_tool_available=manager.recovery_tool_available,
            artifact_path_estimate_for=estimate_tool_artifact_path,
            model=manager.settings["model"],
        )
        if (requested_by or (prune and prune.candidates) or current_tokens > manager.autocompact_threshold) and (
            yield from start_compact(
                driver,
                manager,
                source,
                cycle=cycle,
                trigger="full_threshold" if requested_by or current_tokens > manager.autocompact_threshold else "micro_threshold",
                prune=prune,
                requested_by=requested_by,
            )
        ):
            return True
        if not requested_by and prune and prune.candidates and (not (last_change and last_change.kind == "context_compacted")):
            result = yield from apply_microcompaction(
                source,
                plan=prune,
                protected_start=manager.protected_start,
                model=manager.settings["model"],
            )
            if result.archived_count:
                record = yield from replacement_record(driver, manager, source, result.messages, mode="micro")
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
                yield MicroArtifacts(record)
                yield CommitRecords([record])
                return True
        if not requested_by and current_tokens <= manager.autocompact_threshold:
            return (yield from finish_compact(driver, manager, source, mode="none", changed=False))
    elif (yield from start_compact(driver, manager, source, cycle=cycle, trigger="prompt_too_long")):
        return True
    source_digest = digest([m.to_dict() for m in source])
    if not failures and any(
        op.turn_id == driver.state.active_turn_id
        and op.attempts[1].plan._payload["purpose"] == "compaction"
        and (op.attempts[1].plan._payload["request"]["metadata"]["source_digest"] == source_digest)
        and (op.attempts[1].plan._payload["request"]["metadata"]["mode"] == mode)
        for op in driver.state.operations.values()
    ):
        return (yield from finish_compact(driver, manager, source, mode="none", changed=False))
    plan = _summary_compaction.plan_summary(
        source,
        keep_recent=manager.settings["keep_recent_messages"],
        drop_ratio=min(0.2 * failures, 0.95) if failures > 1 else 0,
        protected_start=manager.protected_start,
        language=manager.settings["language"],
        event_limit=manager.settings["summary_event_limit"],
    )
    if plan is None and failures and (manager.protected_start is not None):
        yield CloseCompaction(UnseenInputCapacityError.code)
        return True
    if plan is None or (requested_by and (not plan.previous) and (not any(m.role in {"assistant", "tool"} for m in plan.prefix))):
        return (yield from finish_compact(driver, manager, source, mode="none", changed=False))
    oid = f"{driver.state.active_turn_id}/model/compaction/{source_digest}/{mode}/{plan.keep_recent}"
    if oid in driver.state.operations:
        return (yield from finish_compact(driver, manager, source, mode="none", changed=False))
    assert driver.state.active_turn_id is not None
    task = driver.state.turns[driver.state.active_turn_id].start._task()
    model_id = yield ModelRoute("compaction")
    request = {
        "model": model_id,
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
    if (yield CountContext([Message("user", plan.prompt)], manager.settings["model"])) >= manager.settings[
        "model_context_window"
    ]:
        return (yield from finish_compact(driver, manager, source, mode="none", changed=False))
    record = yield PlanModel(oid, request, "compaction")
    yield CommitRecords([record])
    return True


def _boundary(driver: CompactionView, stage: str, identity: str) -> Record | None:
    return driver.state.boundaries.get((driver.state.active_turn_id or "", stage, identity))


def extract_memory(
    driver: CompactionView, source: list[Message], current_tokens: int, cycle: int
) -> Generator[object, Any, bool]:
    assert driver.state.active_turn_id is not None
    task = driver.state.turns[driver.state.active_turn_id].start._task()
    if not task.metadata.get("session_memory_enabled"):
        return False
    task = driver.state.turns[driver.state.active_turn_id].start.task()
    yield CompactionRequest.RESTORE_SESSION_MEMORY
    oid = f"{task.task_id}/model/session_memory/{digest([m.to_dict() for m in source])}"
    saved = _boundary(driver, "session_memory_saved", oid)
    if saved is not None:
        yield CompactionRequest.SAVE_MEMORY_PROJECTION
        return False
    op = driver.state.operations.get(oid)
    if op is not None:
        if op.selected_attempt is None:
            return False
        receipt = op.attempts[op.selected_attempt].result
        assert receipt is not None
        state = yield ApplyExtractionReceipt(source, current_tokens, cycle, receipt._payload["result"].get("content", ""))
        record = yield CreateBoundary("session_memory_saved", oid, {"state": state}, source_operation_id=oid)
        yield CommitRecords([record])
        yield CompactionRequest.SAVE_MEMORY_PROJECTION
        return True
    new = yield SelectExtractionMessages(source, current_tokens)
    if not new:
        return False
    model_id = yield ModelRoute("session_memory")
    prompt = yield ExtractionPrompt(new)
    request = {
        "model": model_id,
        "messages": [Message("user", prompt).to_dict()],
        "tools": [],
        "prompt_bundle": None,
        "model_settings": task.model_settings.to_dict() if task.model_settings else None,
        "metadata": {"purpose": "session_memory", "vv_session": {"cycle_index": cycle}},
    }
    record = yield PlanModel(oid, request, "session_memory")
    yield CommitRecords([record])
    return True


def pending_memory(driver: CompactionView):
    tid = driver.state.active_turn_id
    for (turn, stage, key), r in reversed(list(driver.state.boundaries.items())):
        if turn == tid and stage == "memory_started" and _boundary(driver, "memory_completed", key) is None:
            return key, r._payload["data"]["event"]
    return None


def start_compact(
    driver: CompactionView, manager, source, *, cycle: int, trigger: str, prune=None, requested_by=None
) -> Generator[object, Any, bool]:
    if pending_memory(driver):
        return False
    key = (
        digest([driver.state.active_turn_id, [m.to_dict() for m in source], trigger, requested_by])
        if requested_by
        else digest([driver.state.active_turn_id, [m.to_dict() for m in source], trigger])
    )
    if _boundary(driver, "memory_started", key):
        return False
    payload = yield BeforeCompaction(key, source, cycle, trigger, prune)
    metadata = payload["metadata"]
    if requested_by:
        metadata = metadata | {"requested_by": requested_by, "source_digest": digest([m.to_dict() for m in source])}
    payload = payload | {"metadata": metadata}
    record = yield CreateBoundary("memory_started", key, {"event": payload})
    yield CommitRecords([record])
    return True


def finish_compact(
    driver: CompactionView,
    manager,
    source,
    *,
    mode: str,
    changed: bool,
    archived_count=0,
    reclaimed_tokens=0,
    artifact_failure_count=0,
) -> Generator[object, Any, bool]:
    pending = pending_memory(driver)
    if pending is None:
        return False
    key, start = pending
    start_id = _boundary(driver, "memory_started", key)
    start_seq = next(row.seq for row in driver.records if row.record == start_id)
    micro = [
        row.record._payload["micro_usage"]
        for row in driver.records
        if row.seq > start_seq
        and row.record.turn_id == driver.state.active_turn_id
        and row.record.kind == "context_compacted"
        and "micro_usage" in row.record._payload
    ]
    if micro:
        archived_count += sum(v["archived_count"] for v in micro)
        reclaimed_tokens += sum(v["reclaimed_tokens"] for v in micro)
        artifact_failure_count += sum(v["artifact_failure_count"] for v in micro)
        if not changed:
            changed, mode = True, "micro"
    payload = yield AfterCompaction(key, start, source, mode, changed, archived_count, reclaimed_tokens, artifact_failure_count)
    metadata = payload["metadata"]
    if "requested_by" in start.get("metadata", {}):
        metadata = metadata | {"requested_by": start["metadata"]["requested_by"]}
        metadata["compaction_error"] = None
        if not changed and any(
            op.kind == "model"
            and op.attempts[1].plan._payload["purpose"] == "compaction"
            and op.attempts[1].plan._payload["request"]["metadata"].get("source_digest") == start["metadata"].get("source_digest")
            for op in driver.state.operations.values()
        ):
            metadata["compaction_error"] = "summary_unavailable"
    record = yield CreateBoundary("memory_completed", key, {"event": payload | {"metadata": metadata}})
    records = [record]
    if changed and mode in {"summary", "emergency"} and _task(driver).metadata.get("session_memory_enabled"):
        state = yield SessionMemoryCompacted(payload.get("summary_tokens"))
        record = yield CreateBoundary("session_memory_saved", f"compact/{key}", {"state": state})
        records.append(record)
    yield CommitRecords(records)
    yield CompactionRequest.SAVE_MEMORY_PROJECTION
    return True


def _task(driver: CompactionView):
    assert driver.state.active_turn_id is not None
    return driver.state.turns[driver.state.active_turn_id].start._task()
