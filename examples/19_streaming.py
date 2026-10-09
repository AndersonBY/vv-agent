#!/usr/bin/env python3
"""Streaming: 实时接收 LLM 输出 token, 适合 UI 逐字显示."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, VvLlmModelProvider
from vv_agent.events import AssistantDeltaEvent, DiagnosticEvent, RunEvent

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

    agent = Agent("stream-demo", "Answer concisely.", model=model)
    config = RunConfig(
        model_provider=VvLlmModelProvider(settings_file=settings_file, default_backend=backend),
        workspace=workspace,
        stream=build_event_handler(verbose=verbose),
        max_cycles=5,
    )

    print("[demo] 流式输出开始:\n")
    try:
        result = Runner.run_sync(agent, os.getenv("VV_AGENT_EXAMPLE_PROMPT", "用三句话介绍 Python 语言"), run_config=config)
        print(f"\n\n[demo] 状态: {result.status.value}")
        print(f"[demo] 共收到 {len(collected_tokens)} 个 token 片段")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
