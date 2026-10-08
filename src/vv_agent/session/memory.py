"""Memory callbacks and disk projections follow committed lifecycle receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from vv_agent.events import MemoryCompactCompleted, MemoryCompactMode, MemoryCompactStarted, MemoryCompactTrigger
from vv_agent.memory.provider import call_after_memory_providers, call_before_memory_providers
from vv_agent.memory.session_memory import SessionMemory, SessionMemoryConfig, SessionMemoryState
from vv_agent.types import Message

from .records import copy_json, digest

if TYPE_CHECKING:
    from .kernel import _Driver


def session_memory(task, workspace=None) -> SessionMemory:
    meta = task.metadata
    memory = SessionMemory(
        SessionMemoryConfig(
            min_tokens_before_extraction=max(1, int(meta.get("session_memory_min_tokens", 10000))),
            max_tokens=max(1, int(meta.get("session_memory_max_tokens", 40000))),
            min_text_messages=max(1, int(meta.get("session_memory_min_text_messages", 5))),
            storage_dir=str(meta.get("session_memory_storage_dir", ".memory/session")),
            extraction_callback=lambda *_: None,
            extraction_backend=meta.get("session_memory_extraction_backend") or meta.get("memory_summary_backend"),
            extraction_model=meta.get("session_memory_extraction_model") or meta.get("memory_summary_model") or task.model,
            token_model=task.model,
        ),
        workspace=workspace,
        storage_scope=meta.get("session_id") or task.task_id,
    )
    memory.state = SessionMemoryState.from_dict(meta.get("_vv_agent_session_memory_initial_state", {}))
    return memory


def restore_memory(driver: _Driver) -> SessionMemory:
    memory = session_memory(driver.task())
    for (tid, stage, _), r in driver.state.boundaries.items():
        if tid == driver.state.active_turn_id and stage == "session_memory_saved":
            memory.state = SessionMemoryState.from_dict(copy_json(r._payload["data"]["state"]))
    return memory


def save_projection(driver: _Driver) -> None:
    # Files are a replayable projection of the log; atomic replace never exposes partial JSON.
    latest = {}
    for (tid, stage, _), r in driver.state.boundaries.items():
        if stage != "session_memory_saved":
            continue
        task = driver.state.turns[tid].start._task()
        memory = session_memory(task, Path(driver.runtime.config.workspace or ".") if task.use_workspace else None)
        path = memory._storage_path()
        if path is not None:
            latest[path] = r._payload["data"]["state"]
    for path, state in latest.items():
        data = json.dumps(state, ensure_ascii=False, indent=2)
        if path.exists() and path.read_text(encoding="utf-8") == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            staging = Path(f.name)
            try:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                staging.unlink(missing_ok=True)
                raise
        try:
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def extract_memory(driver: _Driver, source: list[Message], current_tokens: int, cycle: int) -> bool:
    task = driver.task()
    if not task.metadata.get("session_memory_enabled"):
        return False
    memory = restore_memory(driver)
    oid = f"{task.task_id}/model/session_memory/{digest([m.to_dict() for m in source])}"
    saved = driver.boundary("session_memory_saved", oid)
    if saved is not None:
        save_projection(driver)
        return False
    op = driver.state.operations.get(oid)
    if op is not None:
        if op.selected_attempt is None:
            return False
        receipt = op.attempts[op.selected_attempt].result
        assert receipt is not None
        memory.config.extraction_callback = lambda *_: receipt._payload["result"].get("content", "")
        memory.extract(source, current_cycle=cycle, current_tokens=current_tokens)
        driver.commit(
            [driver.boundary_record("session_memory_saved", oid, {"state": memory.state.to_dict()}, source_operation_id=oid)],
            guarded=True,
        )
        save_projection(driver)
        return True
    count = sum(not memory._should_skip_message(m) and bool(m.content) for m in source)
    if not memory.should_extract(current_tokens, count):
        return False
    start = memory.state.last_extracted_message_index + 1 if 0 <= memory.state.last_extracted_message_index < len(source) else 0
    new = [m for m in source[start:] if not memory._should_skip_message(m)]
    if not new:
        return False
    request = {
        "model": memory.config.extraction_model,
        "messages": [Message("user", memory._build_extraction_prompt(new)).to_dict()],
        "tools": [],
        "prompt_bundle": None,
        "model_settings": task.model_settings.to_dict() if task.model_settings else None,
        "metadata": {"purpose": "session_memory", "cycle_index": cycle},
    }
    driver.commit([driver.plan(oid, request, "model", purpose="session_memory")], guarded=True)
    return True


def pending_memory(driver: _Driver):
    tid = driver.state.active_turn_id
    for (turn, stage, key), r in reversed(list(driver.state.boundaries.items())):
        if turn == tid and stage == "memory_started" and driver.boundary("memory_completed", key) is None:
            return key, r._payload["data"]["event"]
    return None


def start_compact(driver: _Driver, manager, source, *, cycle: int, trigger: str, prune=None) -> bool:
    if pending_memory(driver):
        return False
    key = digest([driver.state.active_turn_id, [m.to_dict() for m in source], trigger])
    if driver.boundary("memory_started", key):
        return False
    event = MemoryCompactStarted(
        run_id=driver.state.active_turn_id or "",
        trace_id=driver.state.active_turn_id or "",
        session_id=driver.sid,
        agent_name=driver.runtime.agent.name,
        cycle_index=cycle,
        event_id=f"sk/memory/{key}/started",
        created_at=driver.records[-1].created_ms / 1000,
        message_count=len(source),
        estimated_tokens=manager._calculate_effective_length(source, total_tokens=None, recent_tool_call_ids=None),
        trigger=cast(MemoryCompactTrigger, trigger),
        configured_threshold=manager.compact_threshold,
        effective_threshold=manager.autocompact_threshold,
        microcompact_threshold=manager.microcompact_trigger_threshold,
        microcompact_target=manager.microcompact_target_threshold,
        candidate_count=prune.candidate_count if prune else 0,
        estimated_reclaimable_tokens=prune.estimated_reclaimable_tokens if prune else 0,
        model_context_window=manager.model_context_window,
        model_max_output_tokens=manager.model_max_output_tokens,
        reserved_output_tokens=manager.reserved_output_tokens,
        reserved_output_source=manager.reserved_output_source,
        autocompact_buffer_tokens=manager.autocompact_buffer_tokens,
    )
    from vv_agent.events import event_from_dict

    provider_event = event_from_dict(event.to_dict() | {"metadata": {"messages": source}})
    metadata = call_before_memory_providers(driver.runtime.config.memory_providers, cast(MemoryCompactStarted, provider_event))
    payload = event.to_dict() | {"metadata": metadata}
    driver.commit([driver.boundary_record("memory_started", key, {"event": payload})], guarded=True)
    return True


def finish_compact(
    driver: _Driver, manager, source, *, mode: str, changed: bool, archived_count=0, reclaimed_tokens=0, artifact_failure_count=0
) -> bool:
    pending = pending_memory(driver)
    if pending is None:
        return False
    key, start = pending
    start_id = driver.boundary("memory_started", key)
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
    event = MemoryCompactCompleted(
        run_id=driver.state.active_turn_id or "",
        trace_id=driver.state.active_turn_id or "",
        session_id=driver.sid,
        agent_name=driver.runtime.agent.name,
        cycle_index=start["cycle_index"],
        event_id=f"sk/memory/{key}/completed",
        created_at=driver.records[-1].created_ms / 1000,
        before_count=start["message_count"],
        after_count=len(source),
        summary_tokens=manager._calculate_effective_length(source, total_tokens=None, recent_tool_call_ids=None),
        mode=cast(MemoryCompactMode, mode) if changed else "none",
        changed=changed,
        archived_count=archived_count,
        reclaimed_tokens=reclaimed_tokens,
        artifact_failure_count=artifact_failure_count,
    )
    metadata = call_after_memory_providers(driver.runtime.config.memory_providers, event)
    records = [driver.boundary_record("memory_completed", key, {"event": event.to_dict() | {"metadata": metadata}})]
    if changed and mode in {"summary", "emergency"} and driver.task().metadata.get("session_memory_enabled"):
        memory = restore_memory(driver)
        memory.on_compaction(current_tokens=event.summary_tokens)
        records.append(driver.boundary_record("session_memory_saved", f"compact/{key}", {"state": memory.state.to_dict()}))
    driver.commit(records, guarded=True)
    save_projection(driver)
    return True
