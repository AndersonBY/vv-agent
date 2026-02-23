#!/usr/bin/env python3
"""Hook composition example: combine TimingHook + SafetyHook + AuditHook."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from vv_agent.runtime import (
    AfterLLMEvent,
    AfterToolCallEvent,
    BaseRuntimeHook,
    BeforeLLMEvent,
    BeforeLLMPatch,
    BeforeToolCallEvent,
)
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from vv_agent.types import ToolExecutionResult, ToolResultStatus


# --- Hook 1: TimingHook ---
class TimingHook(BaseRuntimeHook):
    """Track per-cycle LLM latency."""

    def __init__(self) -> None:
        self._cycle_start: float = 0.0

    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch | None:
        self._cycle_start = time.monotonic()
        return None

    def after_llm(self, event: AfterLLMEvent) -> None:
        elapsed = time.monotonic() - self._cycle_start
        usage = normalize_token_usage(event.response.raw.get("usage"))
        print(
            f"[TimingHook] cycle={event.cycle_index} "
            f"latency={elapsed:.2f}s tokens={usage.total_tokens}",
            flush=True,
        )
        return None


# --- Hook 2: SafetyHook ---
class SafetyHook(BaseRuntimeHook):
    """Block tool calls that target sensitive paths."""

    BLOCKED_PATTERNS = (".env", "credentials", "secret")

    def before_tool_call(self, event: BeforeToolCallEvent) -> ToolExecutionResult | None:
        args = event.call.arguments or {}
        path = str(args.get("path", args.get("file_path", "")))
        if any(pat in path.lower() for pat in self.BLOCKED_PATTERNS):
            print(f"[SafetyHook] BLOCKED tool={event.call.name} path={path}", flush=True)
            return ToolExecutionResult(
                tool_call_id=event.call.id,
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code="blocked_by_safety_hook",
                content=json.dumps({"ok": False, "error": f"Access to '{path}' is blocked by safety policy."}),
            )
        return None


# --- Hook 3: AuditHook ---
class AuditHook(BaseRuntimeHook):
    """Log every tool call and result for audit trail."""

    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []

    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        entry = {
            "cycle": event.cycle_index,
            "tool": event.call.name,
            "status": event.result.status,
        }
        self.tool_calls.append(entry)
        print(f"[AuditHook] {entry}", flush=True)
        return None


def runtime_log(event: str, payload: dict[str, Any]) -> None:
    verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not verbose:
        return
    if event in {
        "run_started",
        "cycle_started",
        "cycle_llm_response",
        "tool_result",
        "run_completed",
        "run_wait_user",
        "run_max_cycles",
        "cycle_failed",
    }:
        print(f"[{event}] {payload}", flush=True)


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.5")

    workspace.mkdir(parents=True, exist_ok=True)

    # Hooks execute in list order: TimingHook → SafetyHook → AuditHook.
    # before_* hooks run first-to-last; after_* hooks also run first-to-last.
    audit = AuditHook()
    hooks = [TimingHook(), SafetyHook(), audit]

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            runtime_hooks=hooks,
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description="你是文件处理 Agent. 读取 workspace 下的文件并输出摘要.",
            model=model,
            backend=backend,
            max_cycles=12,
            enable_todo_management=True,
        ),
    )

    try:
        run = client.run(
            prompt=(
                "请读取 workspace 下所有文件并输出摘要. "
                "注意: 不要尝试读取 .env 或 credentials 相关文件. "
                "完成后调用 `task_finish` 输出结论。"
            ),
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
        print(f"\n[AuditHook] Total tool calls recorded: {len(audit.tool_calls)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()