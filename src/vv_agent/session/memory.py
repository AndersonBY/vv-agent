"""Memory callbacks and disk projections follow committed lifecycle receipts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vv_agent.events import MemoryCompactCompleted, MemoryCompactMode, MemoryCompactStarted, MemoryCompactTrigger
from vv_agent.kernel._boundary_hydration import hydrate_boundary_data
from vv_agent.memory.provider import call_after_memory_providers, call_before_memory_providers
from vv_agent.memory.session_memory import SessionMemory, SessionMemoryConfig, SessionMemoryState

from .records import copy_json

if TYPE_CHECKING:
    from .kernel import _Driver


def session_memory(task, workspace=None) -> SessionMemory:
    """Build record-backed memory without a plaintext workspace projection."""

    del workspace
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
        workspace=None,
        storage_scope=meta.get("session_id") or task.task_id,
    )
    memory.state = SessionMemoryState.from_dict(meta.get("vv_session", {}).get("memory_initial_state", {}))
    return memory


def restore_memory(driver: _Driver) -> SessionMemory:
    memory = session_memory(driver.task())
    for (tid, stage, _), r in driver.state.boundaries.items():
        if tid == driver.state.active_turn_id and stage == "session_memory_saved":
            memory.state = SessionMemoryState.from_dict(copy_json(hydrate_boundary_data(driver, r)["state"]))
    return memory


def save_projection(driver: _Driver) -> None:
    """Keep the committed encrypted boundary as the only Session Memory copy."""

    del driver


def before_compaction_event(driver, manager, source, *, key, cycle, trigger, prune):
    event = MemoryCompactStarted(
        run_id=driver.state.active_turn_id or "",
        trace_id=driver.state.active_turn_id or "",
        session_id=driver.sid,
        agent_name=driver.runtime.agent.name,
        cycle_index=cycle,
        event_id=f"sk/memory/{key}/started",
        created_at=driver.records[-1].created_ms / 1000,
        message_count=len(source),
        estimated_tokens=manager.effective_tokens(source, total_tokens=None, recent_tool_call_ids=None),
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

    object.__setattr__(event, "version", "v6")
    provider_event = event_from_dict(event.to_dict() | {"metadata": {"messages": source}})
    metadata = call_before_memory_providers(driver.runtime.config.memory_providers, cast(MemoryCompactStarted, provider_event))
    object.__setattr__(event, "version", "v6")
    return event.to_dict() | {"metadata": metadata}


def after_compaction_event(
    driver, manager, source, *, key, start, mode, changed, archived_count, reclaimed_tokens, artifact_failure_count
):
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
        summary_tokens=manager.effective_tokens(source, total_tokens=None, recent_tool_call_ids=None),
        mode=cast(MemoryCompactMode, mode) if changed else "none",
        changed=changed,
        archived_count=archived_count,
        reclaimed_tokens=reclaimed_tokens,
        artifact_failure_count=artifact_failure_count,
    )
    object.__setattr__(event, "version", "v6")
    metadata = call_after_memory_providers(driver.runtime.config.memory_providers, event)
    return event.to_dict() | {"metadata": metadata}


def select_extraction_messages(memory, source, current_tokens):
    count = sum(not memory.skip_extraction_message(m) and bool(m.content) for m in source)
    if not memory.should_extract(current_tokens, count):
        return None
    start = memory.state.last_extracted_message_index + 1 if 0 <= memory.state.last_extracted_message_index < len(source) else 0
    new = [m for m in source[start:] if not memory.skip_extraction_message(m)]
    if not new:
        return None
    return new


def apply_extraction_receipt(memory, source, current_tokens, cycle, content):
    memory.config.extraction_callback = lambda *_: content
    memory.extract(source, current_cycle=cycle, current_tokens=current_tokens)
    return memory.state.to_dict()


def session_memory_compacted(driver, current_tokens):
    memory = restore_memory(driver)
    memory.on_compaction(current_tokens=current_tokens)
    return memory.state.to_dict()
