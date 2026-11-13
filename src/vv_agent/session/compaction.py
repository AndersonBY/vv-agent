"""Synchronous SDK entrypoints and host observations for the core compaction phase."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _compaction_phase as phase
from vv_agent.kernel._compaction_phase import memory_settings as memory_settings
from vv_agent.kernel._driver_requests import CommitRecords, CreateBoundary, PlanModel
from vv_agent.kernel._microcompaction_apply import ArchiveToolMessage, ToolMarkerInput
from vv_agent.kernel._token_count import TokenCount, _StopIterationTransport
from vv_agent.llm.errors import MAX_PTL_RETRIES
from vv_agent.memory.manager import MemoryManager, SummaryPlan
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.types import Message
from vv_agent.workspace.artifacts import _session_artifact_scope

from . import memory as host_memory
from .context import protected_start
from .records import Record

if TYPE_CHECKING:
    from .kernel import _Driver


def manager_for(driver: _Driver) -> MemoryManager:
    assert driver.state.active_turn_id is not None
    start = driver.state.turns[driver.state.active_turn_id].start
    task = start._task()
    definition = start._payload["definition"]
    fork = driver.records[0].record._payload["attributes"].get("app_server_fork")
    baseline = driver.records[0].record._payload["attributes"].get("app_server_context") if fork else None
    settings = memory_settings(baseline or definition, driver.state.model_selection)
    settings["artifact_scope"] = _session_artifact_scope(driver.sid, task.task_id)
    return MemoryManager(
        **settings,
        protected_start=protected_start(driver.records, driver.state),
        workspace_backend=(
            driver.runtime.memory_manager.workspace_backend
            or driver.runtime.config.workspace_backend
            or (driver.runtime.workspace_backend if task.use_workspace else None)
        ),
        recovery_tool_available=any(t["function"]["name"] == "read_file" for t in definition["tools"]),
        session_memory=None,
        summary_callback=None,
    )


def tokens(manager: MemoryManager, messages: list[Message]) -> int:
    return count_messages_tokens([m.to_openai_message() for m in messages], model=manager.model)


def _settings(manager: MemoryManager) -> phase.CompactionMemory:
    return phase.CompactionMemory(
        manager._definition_settings(),
        manager.protected_start,
        manager.recovery_tool_available,
    )


def _drain[T](driver: _Driver, steps: Generator[object, Any, T], manager: MemoryManager | None = None) -> T:
    memory = None
    response = None
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value
            except _StopIterationTransport as transport:
                raise transport.original from transport.original.__cause__
            response = None
            if request is phase.CompactionRequest.INBOX:
                response = driver.store.peek_inbox(driver.sid, through_input_seq=driver.watermark)
            elif request is phase.CompactionRequest.MEMORY:
                manager = manager_for(driver)
                response = _settings(manager)
            elif request is phase.CompactionRequest.TRANSCRIPT:
                response = driver.transcript()
            elif request is phase.CompactionRequest.RESTORE_SESSION_MEMORY:
                memory = host_memory.restore_memory(driver)
            elif request is phase.CompactionRequest.SAVE_MEMORY_PROJECTION:
                host_memory.save_projection(driver)
            elif isinstance(request, phase.CreateRecord):
                response = driver.record(request.kind, request.payload)
            elif isinstance(request, CreateBoundary):
                response = driver.boundary_record(
                    request.stage, request.identity, request.data, source_operation_id=request.source_operation_id
                )
            elif isinstance(request, PlanModel):
                response = driver.plan(request.oid, request.request, "model", purpose=request.purpose)
            elif isinstance(request, CommitRecords):
                driver.commit(request.records, guarded=True)
            elif isinstance(request, phase.CloseCompaction):
                driver.close("failed", request.reason)
            elif isinstance(request, phase.ModelRoute):
                _, resolved = driver.runtime.model_route(request.purpose)
                response = resolved.model_id
            elif isinstance(request, phase.MicroArtifacts):
                driver.runtime.hook("after_microcompact_artifacts", request.record)
            elif isinstance(request, phase.SessionMemoryCompacted):
                response = host_memory.session_memory_compacted(driver, request.current_tokens)
            elif isinstance(request, phase.SelectExtractionMessages):
                assert memory is not None
                response = host_memory.select_extraction_messages(memory, request.source, request.current_tokens)
            elif isinstance(request, phase.ExtractionPrompt):
                assert memory is not None
                response = memory.extraction_prompt(request.messages)
            elif isinstance(request, phase.ApplyExtractionReceipt):
                assert memory is not None
                response = host_memory.apply_extraction_receipt(
                    memory, request.source, request.current_tokens, request.cycle, request.content
                )
            else:
                assert manager is not None
                if isinstance(request, phase.CountContext):
                    response = tokens(manager, request.messages)
                elif isinstance(request, TokenCount):
                    response = (
                        manager._estimate_message_tokens(request.messages)
                        if isinstance(request.messages, Message)
                        else manager._calculate_message_length(request.messages)
                    )
                elif isinstance(request, ArchiveToolMessage):
                    response = manager._archive_tool_message(request.message, request.candidate)
                elif isinstance(request, ToolMarkerInput):
                    response = (
                        manager._compaction_excerpt(request.message),
                        manager.tool_result_excerpt_head,
                        manager.tool_result_excerpt_tail,
                    )
                elif isinstance(request, phase.BeforeCompaction):
                    response = host_memory.before_compaction_event(
                        driver,
                        manager,
                        request.source,
                        key=request.key,
                        cycle=request.cycle,
                        trigger=request.trigger,
                        prune=request.prune,
                    )
                elif isinstance(request, phase.AfterCompaction):
                    response = host_memory.after_compaction_event(
                        driver,
                        manager,
                        request.source,
                        key=request.key,
                        start=request.start,
                        mode=request.mode,
                        changed=request.changed,
                        archived_count=request.archived_count,
                        reclaimed_tokens=request.reclaimed_tokens,
                        artifact_failure_count=request.artifact_failure_count,
                    )
                else:
                    raise TypeError(f"unsupported compaction observation: {type(request).__name__}")
    finally:
        steps.close()


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
    return _drain(
        driver,
        phase.replacement_record(driver, _settings(manager), source, replacement, mode=mode, summary=summary, plan=plan),
        manager,
    )


def finish_summary(driver: _Driver) -> bool:
    return _drain(driver, phase.finish_summary(driver))


def compact_context(driver: _Driver, *, requested_by: list[str] | None = None) -> bool:
    return _drain(driver, phase.compact_context(driver, max_ptl_retries=MAX_PTL_RETRIES, requested_by=requested_by))
