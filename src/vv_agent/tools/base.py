from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vv_agent.deferred import ToolCallOutcome
from vv_agent.tools.metadata import ToolMetadata, normalize_tool_metadata
from vv_agent.types import (
    SubTaskOutcome,
    SubTaskRequest,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.runtime.sub_task_manager import SubTaskManager
    from vv_agent.workspace.base import WorkspaceBackend

ToolHandler = Callable[["ToolContext", dict[str, Any]], ToolExecutionResult | ToolCallOutcome]
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

    def defer(self) -> ToolCallOutcome:
        """Create the opaque deferred handle before a provider side effect.

        Handle identity is allocated from the active checkpoint controller and
        tool-operation plan.  A non-checkpointed invocation fails closed with a
        normal ``Completed(ERROR)`` result; it never manufactures an in-memory
        handle that could later be mistaken for durable state.
        """

        from vv_agent.deferred import DeferredToolHandle

        runtime = self.ctx
        metadata: dict[str, Any] = {}
        if isinstance(self.metadata, dict):
            metadata.update(self.metadata)
        if runtime is not None and isinstance(runtime.metadata, dict):
            metadata.update(runtime.metadata)
        controller = metadata.get("_vv_agent_checkpoint_controller")
        plan = metadata.get("_vv_agent_checkpoint_plan")
        checkpoint_key = getattr(controller, "checkpoint_key", None)
        operation_id = getattr(plan, "operation_id", None) or metadata.get("_vv_agent_operation_id")
        request_digest = getattr(plan, "request_digest", None) or metadata.get("_vv_agent_request_digest")
        attempt = getattr(plan, "attempt", None) or metadata.get("_vv_agent_operation_attempt") or 1
        if (
            not isinstance(checkpoint_key, str)
            or not checkpoint_key.strip()
            or not isinstance(operation_id, str)
            or not operation_id.strip()
            or not isinstance(request_digest, str)
        ):
            return ToolCallOutcome.Completed(
                ToolExecutionResult(
                    tool_call_id=self.tool_call_id,
                    content="Deferred execution requires a durable checkpoint.",
                    status_code=ToolResultStatus.ERROR,
                    directive=ToolDirective.CONTINUE,
                    error_code="deferred_requires_checkpoint",
                )
            )
        try:
            handle = DeferredToolHandle(
                checkpoint_key=checkpoint_key,
                operation_id=operation_id,
                attempt=attempt,
                request_digest=request_digest,
            )
        except Exception:
            return ToolCallOutcome.Completed(
                ToolExecutionResult(
                    tool_call_id=self.tool_call_id,
                    content="Deferred execution requires a durable checkpoint.",
                    status_code=ToolResultStatus.ERROR,
                    directive=ToolDirective.CONTINUE,
                    error_code="deferred_requires_checkpoint",
                )
            )
        # Keep the opaque handle available to the provider adapter without
        # exposing it in the model-visible result metadata.
        self.metadata["_vv_agent_deferred_handle"] = handle
        return ToolCallOutcome.Deferred(handle)

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
