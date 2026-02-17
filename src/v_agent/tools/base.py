from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v_agent.types import ToolExecutionResult

ToolHandler = Callable[["ToolContext", dict[str, Any]], ToolExecutionResult]


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    shared_state: dict[str, Any]
    cycle_index: int

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
