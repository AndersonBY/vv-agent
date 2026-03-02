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


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    shared_state: dict[str, Any]
    cycle_index: int
    workspace_backend: WorkspaceBackend
    sub_task_runner: SubTaskRunner | None = None
    ctx: ExecutionContext | None = None
    task_metadata: dict[str, Any] = field(default_factory=dict)

    def allow_outside_workspace_paths(self) -> bool:
        sources: list[dict[str, Any]] = []
        if isinstance(self.task_metadata, dict):
            sources.append(self.task_metadata)
        runtime_metadata = getattr(self.ctx, "metadata", None)
        if isinstance(runtime_metadata, dict):
            sources.append(runtime_metadata)

        for source in sources:
            for key in (
                "allow_outside_workspace_paths",
                "allow_outside_workspace",
                "workspace_allow_outside_main",
                "workspace_allow_outside",
            ):
                parsed = _parse_bool(source.get(key))
                if parsed is not None:
                    return parsed
        return False

    def resolve_workspace_path(self, raw_path: str) -> Path:
        base = self.workspace.resolve()
        candidate = Path(raw_path).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        if (
            not self.allow_outside_workspace_paths()
            and target != base
            and base not in target.parents
        ):
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return target


@dataclass(slots=True)
class ToolSpec:
    name: str
    handler: ToolHandler
