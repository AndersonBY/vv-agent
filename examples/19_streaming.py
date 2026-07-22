#!/usr/bin/env python3
"""Streaming: 实时接收 LLM 输出 token, 适合 UI 逐字显示."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from vv_agent.config import build_vv_llm_from_local_settings
from vv_agent.events import AssistantDeltaEvent, DiagnosticEvent, RunEvent
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentTask

# 收集所有 token 用于统计
collected_tokens: list[str] = []


def build_event_handler(*, verbose: bool) -> Callable[[RunEvent], None]:
    def event_handler(event: RunEvent) -> None:
        if isinstance(event, AssistantDeltaEvent):
            collected_tokens.append(event.delta)
            print(event.delta, end="", flush=True)
            return
        name = event.code if isinstance(event, DiagnosticEvent) else event.type
        if verbose and name in {"cycle_started", "run_completed"}:
            payload = event.details if isinstance(event, DiagnosticEvent) else event.to_dict()
            print(f"\n[{name}] {payload}", flush=True)

    return event_handler


def main() -> None:
    settings_file = Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3")
    workspace = Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    verbose = os.getenv("VV_AGENT_EXAMPLE_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"}

    workspace.mkdir(parents=True, exist_ok=True)

    llm, resolved = build_vv_llm_from_local_settings(settings_file, backend=backend, model=model)
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=workspace,
        event_handler=build_event_handler(verbose=verbose),
    )

    system_prompt = build_system_prompt(
        "You are a helpful agent. Answer concisely.",
        language="zh-CN",
        allow_interruption=True,
        use_workspace=True,
    )

    task = AgentTask(
        task_id=f"stream_demo_{uuid.uuid4().hex[:8]}",
        model=resolved.model_id,
        system_prompt=system_prompt,
        user_prompt=os.getenv("VV_AGENT_EXAMPLE_PROMPT", "用三句话介绍 Python 语言"),
        max_cycles=5,
    )

    print("[demo] 流式输出开始:\n")
    try:
        result = runtime.run(task)
        print(f"\n\n[demo] 状态: {result.status.value}")
        print(f"[demo] 共收到 {len(collected_tokens)} 个 token 片段")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
