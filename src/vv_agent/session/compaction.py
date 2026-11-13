"""Synchronous SDK entrypoints and host observations for the core compaction phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.kernel import _compaction_phase as phase
from vv_agent.kernel._compaction_phase import memory_settings as memory_settings
from vv_agent.llm.errors import MAX_PTL_RETRIES
from vv_agent.memory.manager import MemoryManager, SummaryPlan
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.types import Message
from vv_agent.workspace.artifacts import _session_artifact_scope

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
    from ._recovery_phase import run

    return run(
        driver,
        phase.replacement_record(driver, _settings(manager), source, replacement, mode=mode, summary=summary, plan=plan),
        manager=manager,
    )


def finish_summary(driver: _Driver) -> bool:
    from ._recovery_phase import run

    return run(driver, phase.finish_summary(driver))


def compact_context(driver: _Driver, *, requested_by: list[str] | None = None) -> bool:
    from ._recovery_phase import run

    return run(driver, phase.compact_context(driver, max_ptl_retries=MAX_PTL_RETRIES, requested_by=requested_by))
