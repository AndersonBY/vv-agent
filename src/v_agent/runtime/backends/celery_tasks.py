"""Worker-side Celery task for executing a single agent cycle.

This module provides ``run_single_cycle`` which:
1. Rebuilds an AgentRuntime from a ``RuntimeRecipe``
2. Loads the previous checkpoint from the shared StateStore
3. Executes exactly one cycle via the runtime's cycle executor
4. Saves the updated checkpoint (or returns the terminal result)
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from v_agent.config import build_openai_llm_from_local_settings
from v_agent.runtime.backends.celery import RuntimeRecipe
from v_agent.runtime.engine import AgentRuntime
from v_agent.runtime.hooks import RuntimeHook
from v_agent.runtime.state import Checkpoint
from v_agent.runtime.stores.sqlite import SqliteStateStore
from v_agent.tools import build_default_registry
from v_agent.types import AgentResult, AgentStatus, AgentTask
from v_agent.workspace import LocalWorkspaceBackend

logger = logging.getLogger(__name__)


def _build_state_store(recipe: RuntimeRecipe) -> Any:
    """Build a StateStore from the recipe's workspace.

    Uses SQLite by default.  If the workspace contains a
    ``state_store_url`` metadata key starting with ``redis://``,
    a RedisStateStore is used instead.
    """
    workspace = Path(recipe.workspace).resolve()
    db_path = workspace / ".v-agent-state" / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteStateStore(db_path=db_path)


def _load_hooks(class_paths: list[str]) -> list[RuntimeHook]:
    """Import hook classes by dotted path and instantiate them."""
    hooks: list[RuntimeHook] = []
    for path in class_paths:
        module_path, _, class_name = path.rpartition(".")
        if not module_path:
            logger.warning("Invalid hook class path: %s", path)
            continue
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            hooks.append(cls())
        except Exception:
            logger.warning("Failed to load hook %s", path, exc_info=True)
    return hooks


def _rebuild_runtime(recipe: RuntimeRecipe) -> AgentRuntime:
    """Reconstruct an AgentRuntime from a RuntimeRecipe on the worker."""
    workspace = Path(recipe.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm, _resolved = build_openai_llm_from_local_settings(
        recipe.settings_file,
        backend=recipe.backend,
        model=recipe.model,
        timeout_seconds=recipe.timeout_seconds,
    )
    hooks = _load_hooks(recipe.hook_class_paths)

    return AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        log_preview_chars=recipe.log_preview_chars,
        settings_file=recipe.settings_file,
        default_backend=recipe.backend,
        hooks=hooks,
    )


def run_single_cycle(
    *,
    task_dict: dict[str, Any],
    recipe_dict: dict[str, Any],
    cycle_index: int,
) -> dict[str, Any]:
    """Execute a single agent cycle on a Celery worker.

    Returns a dict with:
    - ``finished``: bool — whether the agent reached a terminal state
    - ``result``: dict — serialised AgentResult (only when finished)
    """
    recipe = RuntimeRecipe.from_dict(recipe_dict)
    task = AgentTask.from_dict(task_dict)

    # Load checkpoint saved by the scheduler (or previous cycle).
    store = _build_state_store(recipe)
    checkpoint = store.load_checkpoint(task.task_id)
    if checkpoint is None:
        return {
            "finished": True,
            "result": AgentResult(
                status=AgentStatus.FAILED,
                messages=[],
                cycles=[],
                error=f"No checkpoint found for task {task.task_id}",
                shared_state={},
            ).to_dict(),
        }

    runtime = _rebuild_runtime(recipe)

    # Build the cycle executor closure on the worker side.
    workspace_path = Path(recipe.workspace).resolve()
    memory_manager = runtime._build_memory_manager(
        task=task, workspace_path=workspace_path,
    )
    cycle_executor = runtime._build_cycle_executor(
        task=task,
        workspace_path=workspace_path,
        workspace_backend=runtime._workspace_backend or LocalWorkspaceBackend(workspace_path),
        memory_manager=memory_manager,
        before_cycle_messages=None,
        interruption_messages=None,
    )

    messages = checkpoint.messages
    cycles = checkpoint.cycles
    shared_state = checkpoint.shared_state

    # Execute exactly one cycle.
    result = cycle_executor(
        cycle_index, messages, cycles, shared_state, None,
    )

    if result is not None:
        # Terminal state — clean up checkpoint and return result.
        store.delete_checkpoint(task.task_id)
        return {"finished": True, "result": result.to_dict()}

    # Non-terminal — save updated checkpoint for the next cycle.
    store.save_checkpoint(
        Checkpoint(
            task_id=task.task_id,
            cycle_index=cycle_index,
            status=AgentStatus.RUNNING,
            messages=messages,
            cycles=cycles,
            shared_state=shared_state,
        )
    )
    return {"finished": False}
