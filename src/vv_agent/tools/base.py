from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vv_agent.tools.metadata import ToolMetadata, normalize_tool_metadata
from vv_agent.types import SubTaskOutcome, SubTaskRequest, ToolExecutionResult

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.runtime.sub_task_manager import SubTaskManager
    from vv_agent.workspace.base import WorkspaceBackend

ToolHandler = Callable[["ToolContext", dict[str, Any]], ToolExecutionResult]
SubTaskRunner = Callable[[SubTaskRequest], SubTaskOutcome]


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    shared_state: dict[str, Any]
    cycle_index: int
    workspace_backend: WorkspaceBackend
    task_id: str = ""
    sub_task_runner: SubTaskRunner | None = None
    sub_task_manager: SubTaskManager | None = None
    ctx: ExecutionContext | None = None
    task_metadata: dict[str, Any] = field(default_factory=dict)
    run_context: Any | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    session: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        runtime_metadata = getattr(self.ctx, "metadata", None)
        if isinstance(runtime_metadata, dict):
            return str(runtime_metadata.get("_vv_agent_run_id") or self.task_id)
        return self.task_id

    @property
    def agent_name(self) -> str:
        runtime_metadata = getattr(self.ctx, "metadata", None)
        if isinstance(runtime_metadata, dict):
            return str(runtime_metadata.get("_vv_agent_agent_name") or self.metadata.get("agent_name") or "")
        return str(self.metadata.get("agent_name") or "")

    @property
    def raw_arguments(self) -> dict[str, Any]:
        return dict(self.arguments)

    @property
    def app_state(self) -> Any | None:
        return getattr(self.run_context, "context", None)

    def allow_outside_workspace_paths(self) -> bool:
        sources: list[dict[str, Any]] = []
        if isinstance(self.task_metadata, dict):
            sources.append(self.task_metadata)
        runtime_metadata = getattr(self.ctx, "metadata", None)
        if isinstance(runtime_metadata, dict):
            sources.append(runtime_metadata)

        for source in sources:
            if "allow_outside_workspace_paths" not in source:
                continue
            value = source["allow_outside_workspace_paths"]
            if not isinstance(value, bool):
                raise ValueError("allow_outside_workspace_paths must be a boolean")
            return value
        return False

    def resolve_workspace_path(self, raw_path: str) -> Path:
        base = self.workspace.resolve()
        candidate = Path(raw_path).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        if not self.allow_outside_workspace_paths() and target != base and base not in target.parents:
            raise ValueError(f"Path escapes workspace: {raw_path}")
        return target


def is_tool_call_preapproved(context: ToolContext, *, tool_call_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
    execution_context = context.ctx
    approval = getattr(execution_context, "_approved_tool_approval", None)
    call = getattr(approval, "call", None)
    return bool(
        call is not None
        and getattr(call, "id", None) == tool_call_id
        and getattr(call, "name", None) == tool_name
        and getattr(call, "arguments", None) == arguments
    )


@dataclass(slots=True)
class ToolSpec:
    name: str
    handler: ToolHandler
    tool_metadata: ToolMetadata | None = None

    def __post_init__(self) -> None:
        self.tool_metadata = normalize_tool_metadata(self.tool_metadata)
