from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vv_agent.types import SubTaskOutcome, SubTaskRequest, ToolExecutionResult

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.workspace.base import WorkspaceBackend

ToolHandler = Callable[["ToolContext", dict[str, Any]], ToolExecutionResult]
SubTaskRunner = Callable[[SubTaskRequest], SubTaskOutcome]


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    shared_state: dict[str, Any]
    cycle_index: int
    workspace_backend: WorkspaceBackend
    sub_task_runner: SubTaskRunner | None = None
    ctx: ExecutionContext | None = None
    task_metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_workspace_path(self, raw_path: str) -> Path:
        base = self.workspace.resolve()
        target = (base / raw_path).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return target


@dataclass(slots=True)
class ToolSpec:
    name: str
    handler: ToolHandler
