#!/usr/bin/env python3
"""Runtime hook example: inject context and guard tool calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from v_agent.constants import WRITE_FILE_TOOL_NAME
from v_agent.runtime import BaseRuntimeHook, BeforeLLMEvent, BeforeLLMPatch, BeforeToolCallEvent
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.types import Message, ToolExecutionResult, ToolResultStatus


class GuardAndHintHook(BaseRuntimeHook):
    def __init__(self) -> None:
        self.verbose = os.getenv("V_AGENT_EXAMPLE_VERBOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

    def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch:
        if self.verbose:
            print(
                f"[hook.before_llm] cycle={event.cycle_index} messages={len(event.messages)}",
                flush=True,
            )
        patched_messages = list(event.messages)
        patched_messages.append(
            Message(
                role="user",
                content="系统补充要求: 任何输出都要简洁, 并在结尾附上下一步建议.",
            )
        )
        return BeforeLLMPatch(messages=patched_messages)

    def before_tool_call(self, event: BeforeToolCallEvent) -> ToolExecutionResult | None:
        if self.verbose:
            print(
                f"[hook.before_tool_call] cycle={event.cycle_index} tool={event.call.name}",
                flush=True,
            )
        if event.call.name != WRITE_FILE_TOOL_NAME:
            return None
        path = str(event.call.arguments.get("path", ""))
        if PurePosixPath(path).name != ".env":
            return None
        if self.verbose:
            print(f"[hook.blocked] refuse path={path}", flush=True)
        return ToolExecutionResult(
            tool_call_id=event.call.id,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="blocked_sensitive_path",
            content='{"ok":false,"error":"Refuse writing .env from runtime hook"}',
        )


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

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=settings_file,
            default_backend=backend,
            workspace=workspace,
            runtime_hooks=[GuardAndHintHook()],
            log_handler=runtime_log,
        ),
        agent=AgentDefinition(
            description="你是一个注重安全和可执行性的开发 Agent。",
            model=model,
            backend=backend,
            max_cycles=20,
            enable_todo_management=True,
            use_workspace=True,
        ),
    )

    try:
        run = client.run(
            prompt="请尝试把 `TEST=1` 写到 .env, 然后再把最终结论写入 artifacts/hook_result.md.",
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
